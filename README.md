# LLM consumers paper: how to run

1. `pip install -U numpy pandas matplotlib openai anthropic google-genai`
2. Copy `.env.example` to `.env` and put your keys in it. Several keys for one provider go in one
   line, comma-separated (`GROQ_API_KEYS=k1,k2,k3`); the runner rotates through them.
   `.env` is in `.gitignore`. Never upload it.
3. Smoke test, about 20 calls per model. Check the failure count is near zero:
   `python experiment.py --models groq:llama-3.3-70b-versatile --reps 1 --limit 20 --out results/test.jsonl`
4. Full run (1,840 calls per model at 10 reps; resumes if interrupted or rate-limited):
   `python experiment.py --models groq:llama-3.3-70b-versatile --reps 10 --rpm 25 --out results/raw.jsonl`
   Run different models in separate terminals at the same time, all with `--out results/raw.jsonl`
   (or separate files, joined later with `cat`).
5. Analyse: `python analyze.py results/raw.jsonl results`
6. Upload main.tex, refs.bib and the whole `results/` folder to Overleaf.

Model spec is `provider:model-id`. Providers: groq, google, openrouter, openai, anthropic, compat, mock.
Use the exact current model ids from each provider's docs and write them in Table 1 with the date.

`--rpm` is requests per minute per key. Set it a little below the provider's limit for that model.
If a limit is hit the runner rotates keys, waits, or retires a key that is out of daily quota.
Calls it could not finish are not saved, so run the same command again later to complete them.

`mock:mock` is an offline fake agent that only tests the pipeline. Never put its output in the paper.
