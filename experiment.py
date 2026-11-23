"""Run the consumer experiment on one or more LLMs.

Usage examples
  python experiment.py --models mock:mock --reps 3 --out results/raw_mock.jsonl      # pipeline test
  python experiment.py --models groq:llama-3.3-70b-versatile --reps 10 --rpm 25
  python experiment.py --models google:<gemini-flash-model> --reps 10 --rpm 8
  python experiment.py --models groq:openai/gpt-oss-20b --reps 3 --rep-start 3 --reasoning-effort low

Options
  --limit N           smoke test: N cells spread evenly over all personas and scenario types
  --rep-start N       first rep index, so one model can be split over several days or people
  --reasoning-effort  low/medium/high, sent to groq only; omitted = provider default. Saved in each
                      row's "settings"; one output file may hold only one setting per model.

Providers and the environment variables that hold their keys
  groq       GROQ_API_KEYS        (comma-separated list = key rotation; GROQ_API_KEY also works)
  google     GEMINI_API_KEYS      (or GEMINI_API_KEY / GOOGLE_API_KEY)
  openrouter OPENROUTER_API_KEYS
  openai     OPENAI_API_KEYS
  anthropic  ANTHROPIC_API_KEYS
  compat     COMPAT_API_KEYS + COMPAT_BASE_URL  (any other OpenAI-compatible endpoint, e.g. local Ollama)
  mock       offline fake agent, ONLY for testing the pipeline
Variables can also be written in a .env file next to this script (KEY=value per line).

Key rotation: every call takes the key that is free soonest. --rpm caps requests per minute PER KEY.
A key that hits a rate limit is paused for the time the provider asks for; a key that hits a daily limit is
retired for this run. If all keys of a model are out of quota the run stops cleanly. Calls that were
not completed because of limits are NOT written to the output file, so re-running the same command
continues where it stopped.

Errors that retrying cannot fix (400 invalid parameter, 401/403 bad key, 404 unknown model, a missing
Python package) stop the
run at once with the cause. 429, 5xx, timeouts and network errors are retried with backoff.
Each saved row has a "usage" field (prompt/completion/reasoning tokens, summed over retries, as the
provider reports them); the mean per call is printed at the end of each model.
"""
import argparse, json, os, re, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
import numpy as np
from config import GOODS, PERSONAS, scenarios, build_prompt

TEMP = 1.0
lock = threading.Lock()

# provider -> (base url or None, name of the environment variable that holds the key(s))
OPENAI_LIKE = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
    "compat": (None, "COMPAT_API_KEY"),            # base url from COMPAT_BASE_URL
}
KEY_ENV = {**{p: v[1] for p, v in OPENAI_LIKE.items()}, "google": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


USAGE_FIELDS = ("prompt", "completion", "reasoning")
FATAL_STATUS = {
    400: "invalid request, probably a parameter this model does not accept (check --reasoning-effort and the model id)",
    401: "authentication failed, check the API key",
    403: "access denied, the key has no permission for this model",
    404: "model not found, check the model id on the provider's current model list",
}


class QuotaExhausted(Exception):
    pass


class FatalError(Exception):
    """A problem that retrying cannot fix (bad key, unknown model, missing package)."""


def load_dotenv(path=".env"):
    """Tiny .env reader (KEY=value per line). Never overrides variables that are already set."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export "):].strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def get_keys(prov):
    base = KEY_ENV[prov]
    vals = os.environ.get(base + "S") or os.environ.get(base) or ""
    if prov == "google" and not vals:
        vals = os.environ.get("GOOGLE_API_KEY", "")
    keys = [k.strip() for k in vals.split(",") if k.strip()]
    if not keys:
        raise SystemExit(f"No API key for provider '{prov}'. Set {base}S (comma-separated) or {base}, "
                         f"in the terminal or in a .env file.")
    if prov == "compat" and not os.environ.get("COMPAT_BASE_URL"):
        raise SystemExit("Provider 'compat' also needs COMPAT_BASE_URL.")
    return keys


# ------------------------------------------------------------------ key rotation
def parse_wait(msg):
    """Seconds the provider asks us to wait ('try again in 7m12.5s', 'retry in 34s'), or None."""
    m = re.search(r"(?:try again|retry) in ([0-9hms\. ]+)", msg)
    if not m:
        return None
    parts = re.findall(r"(\d[\d.]*)(ms|h|m|s)", m.group(1))
    if not parts:
        return None
    mult = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(v) * mult[u] for v, u in parts)


def is_rate_limit(msg):
    return any(s in msg for s in ("429", "rate limit", "rate_limit", "resource_exhausted",
                                  "too many requests", "quota"))


def status_of(err):
    """HTTP status of a provider SDK error (openai/anthropic: status_code, google-genai: code), or None."""
    for attr in ("status_code", "code"):
        v = getattr(err, attr, None)
        if isinstance(v, int):
            return v
    m = re.search(r"error code: (\d{3})", str(err).lower())
    return int(m.group(1)) if m else None


def fatal_reason(err):
    if isinstance(err, ImportError):
        return f"missing Python package '{err.name or err}'. Run: pip install -r requirements.txt"
    reason = FATAL_STATUS.get(status_of(err))
    return f"{reason} ({err})" if reason else None


def redact(text, keys):
    for k in keys:
        text = text.replace(k, "<key>")
    return text


class KeyPool:
    """Round-robin over several API keys with a per-key request rate and cooldowns."""

    def __init__(self, name, keys, rpm):
        self.name, self.keys = name, keys
        self.gap = 60.0 / rpm if rpm else 0.0
        self.ready = {k: 0.0 for k in keys}      # earliest time each key may be used again
        self.dead = set()                         # keys out of daily quota for this run
        self.lk = threading.Lock()

    def acquire(self):
        """Return the key that is free soonest and reserve its next slot."""
        with self.lk:
            live = [k for k in self.keys if k not in self.dead]
            if not live:
                raise QuotaExhausted(f"all {len(self.keys)} key(s) for {self.name} are out of quota")
            k = min(live, key=lambda x: self.ready[x])
            now = time.time()
            start = max(now, self.ready[k])
            self.ready[k] = start + self.gap
        if start > now:
            time.sleep(start - now)
        return k

    def penalise(self, key, err):
        msg = str(err).lower()
        secs = parse_wait(msg)
        daily = re.search(r"per ?day|daily|\brpd\b|\btpd\b", msg) is not None
        with self.lk:
            if daily or (secs and secs > 600):
                self.dead.add(key)
            else:
                self.ready[key] = max(self.ready[key], time.time() + (secs + 1 if secs else 30))


# ------------------------------------------------------------------ providers
# Each call returns (text, usage). Token counts are as reported by the provider: OpenAI-style APIs
# include reasoning tokens inside "completion", Gemini reports them separately from "completion".
def usage_openai(u):
    if u is None:
        return None
    details = getattr(u, "completion_tokens_details", None)
    return dict(prompt=u.prompt_tokens, completion=u.completion_tokens,
                reasoning=getattr(details, "reasoning_tokens", None))


def call_openai_like(prov, model, prompt, temp, key, extra):
    from openai import OpenAI
    base = OPENAI_LIKE[prov][0]
    if prov == "compat":
        base = os.environ["COMPAT_BASE_URL"]
    c = OpenAI(api_key=key, base_url=base, max_retries=0, timeout=90)   # we handle retries/rotation ourselves
    kw = dict(model=model, messages=[{"role": "user", "content": prompt}], **extra)
    try:
        r = c.chat.completions.create(temperature=temp, **kw)
    except Exception as e:                      # some reasoning models reject temperature
        if "temperature" in str(e).lower() and not is_rate_limit(str(e).lower()):
            r = c.chat.completions.create(**kw)
        else:
            raise
    return r.choices[0].message.content, usage_openai(r.usage)


def call_anthropic(model, prompt, temp, key):
    import anthropic
    r = anthropic.Anthropic(api_key=key, max_retries=0).messages.create(
        model=model, max_tokens=300, temperature=temp,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text, dict(prompt=r.usage.input_tokens, completion=r.usage.output_tokens, reasoning=None)


def call_google(model, prompt, temp, key):
    from google import genai
    c = genai.Client(api_key=key)
    r = c.models.generate_content(model=model, contents=prompt, config={"temperature": temp})
    um = r.usage_metadata
    usage = um and dict(prompt=um.prompt_token_count, completion=um.candidates_token_count,
                        reasoning=um.thoughts_token_count)
    return r.text, usage


def call_mock(prompt, rng):
    """Stone-Geary consumer with noise. NOT a result: only to test the pipeline."""
    nums = re.findall(r"Rs ([\d,\.]+)", prompt)
    budget = float(nums[0].replace(",", ""))
    p = np.array([float(x.replace(",", "")) for x in nums[1:6]])
    alpha = np.array([0.35, 0.15, 0.25, 0.12, 0.13])
    sub = np.array([20, 8, 0, 0, 0.0])
    free = max(budget - p @ sub, 0.1 * budget)
    q = sub + alpha * free / p
    q = q * np.exp(rng.normal(0, 0.12, 5))
    q = q * (budget * rng.uniform(0.97, 1.0)) / (p @ q)
    return json.dumps({g: round(float(x), 2) for g, x in zip(GOODS, q)})


def request_params(prov, settings):
    """Extra API parameters for this provider. reasoning_effort is only sent when set, and only to Groq."""
    effort = settings.get("reasoning_effort")
    if effort is None or prov == "mock":
        return {}
    if prov != "groq":
        raise SystemExit(f"--reasoning-effort is only supported for the groq provider, not '{prov}'.")
    return {"reasoning_effort": effort}


def ask(prov, model, prompt, key, rng, extra):
    if prov in OPENAI_LIKE: return call_openai_like(prov, model, prompt, TEMP, key, extra)
    if prov == "anthropic": return call_anthropic(model, prompt, TEMP, key)
    if prov == "google": return call_google(model, prompt, TEMP, key)
    if prov == "mock": return call_mock(prompt, rng), None
    raise ValueError(f"unknown provider '{prov}'")


def add_usage(a, b):
    if not a or not b:
        return a or b
    return {k: None if a[k] is None and b[k] is None else (a[k] or 0) + (b[k] or 0) for k in USAGE_FIELDS}


def mean_usage(usages):
    if not usages:
        return "no calls saved"
    parts = []
    for k in USAGE_FIELDS:
        vals = [u[k] for u in usages if u and u[k] is not None]
        if vals:
            parts.append(f"{k} {sum(vals) / len(vals):.0f}")
    return ", ".join(parts) or "not reported by this provider"


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)   # reasoning models (e.g. Qwen3) may emit this
    m = re.search(r"\{.*?\}", text, re.S)
    if not m: return None
    try:
        d = json.loads(m.group(0))
        q = {g: float(d[g]) for g in GOODS}
        return q if all(np.isfinite(v) and v >= 0 for v in q.values()) else None
    except Exception:
        return None


# ------------------------------------------------------------------ runner
def one(spec, cell, pool, settings, extra):
    prov, model = spec.split(":", 1)
    prompt = build_prompt(cell["persona"], cell["prices"], cell["budget"])
    rng = np.random.default_rng(zlib.crc32(f'{spec}|{cell["persona"]}|{cell["sid"]}|{cell["rep"]}'.encode()))
    keys = pool.keys if pool else []
    raw, q, err, api_fail, usage = None, None, None, False, None
    fails = tries = 0
    while fails < 4 and tries < 20:             # rate-limit hits (429) rotate keys and do not count as failures
        tries += 1
        key = None
        try:
            key = pool.acquire() if pool else None
            raw, u = ask(prov, model, prompt, key, rng, extra)
            usage = add_usage(usage, u)         # a parse retry still spends tokens, so sum over attempts
            q = parse(raw)
            if q: break
            err, api_fail = "parse", False
            fails += 1
        except QuotaExhausted:
            raise
        except Exception as e:
            reason = fatal_reason(e)
            if reason:
                raise FatalError(redact(reason, keys)) from None
            err, api_fail = redact(str(e), keys)[:200], True
            if pool and key and is_rate_limit(str(e).lower()):
                pool.penalise(key, e)
            else:
                fails += 1
                time.sleep(2 ** fails)
    return dict(model=spec, persona=cell["persona"], type=cell["type"], sid=cell["sid"],
                rep=cell["rep"], g=cell["g"], mult=cell["mult"], bmult=cell["bmult"],
                budget=cell["budget"], prices=cell["prices"], q=q, ok=q is not None,
                err=None if q else err, raw=None if q else (raw or "")[:300],
                usage=usage, settings=settings, transient=(q is None and api_fail))


def key_of(r):
    return f'{r["model"]}|{r["persona"]}|{r["sid"]}|{r["rep"]}'


def settings_of(r):
    return r.get("settings") or {"reasoning_effort": None}     # rows saved before settings existed used defaults


def spaced(items, n, phase=0.5):
    return [items[int((i + phase) * len(items) / n)] for i in range(n)]


def evenly_spaced(cells, n):
    """n cells spread over the design: one from every (persona, scenario type) group when n allows,
    the rest shared out by group size and spaced evenly inside each group. Each persona starts at a
    different offset, so together they cover different goods and multipliers."""
    if n >= len(cells):
        return cells
    groups = {}
    for i, c in enumerate(cells):
        groups.setdefault((c["persona"], c["type"]), []).append(i)
    if n < len(groups):
        return spaced(cells, n)
    quota = {g: 1 for g in groups}
    for _ in range(n - len(groups)):
        open_groups = [g for g in groups if quota[g] < len(groups[g])]
        g = max(open_groups, key=lambda g: len(groups[g]) * n / len(cells) - quota[g])
        quota[g] += 1
    personas = list(dict.fromkeys(c["persona"] for c in cells))
    phase = {p: (k + 0.5) / len(personas) for k, p in enumerate(personas)}
    picked = sorted(i for g, ix in groups.items() for i in spaced(ix, quota[g], phase[g[0]]))
    return [cells[i] for i in picked]


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="provider:model, e.g. groq:llama-3.3-70b-versatile")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="results/raw.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=20, help="max requests per minute PER KEY (0 = no cap)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run N cells per model, evenly spaced over the design (cheap smoke test)")
    ap.add_argument("--rep-start", type=int, default=0, help="first rep index; --reps 3 --rep-start 3 runs reps 3,4,5")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                    help="sent as reasoning_effort to groq; omitted = provider default")
    a = ap.parse_args()
    settings = {"reasoning_effort": a.reasoning_effort}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done, saved_settings = set(), {}
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            saved_settings.setdefault(r["model"], settings_of(r))
            if r.get("ok") or r.get("err") == "parse":     # a real answer, or a genuine unparsable answer
                done.add(key_of(r))
    cells = [c for c in scenarios(a.rep_start + a.reps) if c["rep"] >= a.rep_start]
    for spec in a.models:
        prov = spec.split(":", 1)[0]
        extra = request_params(prov, settings)
        if spec in saved_settings and saved_settings[spec] != settings:
            raise SystemExit(f"{a.out} already has {spec} rows run with {saved_settings[spec]}, but this run uses "
                             f"{settings}. Use a different --out so the two settings are not mixed.")
        pool = None
        if prov != "mock":
            keys = get_keys(prov)
            pool = KeyPool(spec, keys, a.rpm)
            print(f"{spec}: {len(keys)} key(s), cap {a.rpm or 'none'} requests/min per key", flush=True)
        selected = evenly_spaced(cells, a.limit) if a.limit else cells
        todo = [c for c in selected if f'{spec}|{c["persona"]}|{c["sid"]}|{c["rep"]}' not in done]
        print(f"{spec}: {len(todo)} calls to run ({len(selected)-len(todo)} already done), "
              f"reps {a.rep_start}-{a.rep_start + a.reps - 1}, settings {settings}", flush=True)
        n = bad = unfinished = 0
        stop_msg = fatal = None
        usages = []
        with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
            futs = [ex.submit(one, spec, c, pool, settings, extra) for c in todo]
            for fu in as_completed(futs):
                try:
                    r = fu.result()
                except FatalError as e:
                    fatal = str(e)
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
                except QuotaExhausted as e:
                    stop_msg = str(e)
                    for f2 in futs: f2.cancel()
                    unfinished += 1
                    continue
                except CancelledError:
                    unfinished += 1
                    continue
                if r["transient"]:                  # limit or network problem: do not save, retry on next run
                    unfinished += 1
                    continue
                n += 1; bad += (not r["ok"])
                usages.append(r["usage"])
                with lock:
                    f.write(json.dumps(r) + "\n"); f.flush()
                if n % 100 == 0: print(f"  {n}/{len(todo)} done, {bad} failed", flush=True)
        print(f"{spec}: finished {n} calls, {bad} unparsable answers, {unfinished} not completed", flush=True)
        print(f"{spec}: mean tokens per saved call: {mean_usage(usages)}", flush=True)
        if fatal:
            raise SystemExit(f"FATAL ({spec}): {fatal}\nRun stopped. Fix the cause and run the same command again.")
        if stop_msg: print(f"  STOPPED: {stop_msg}.", flush=True)
        if unfinished: print("  Run the same command again later (or with more keys) to finish the rest.", flush=True)


if __name__ == "__main__":
    main()
