# Project context: LLM consumers paper (POME Experiential Learning)

## What this is
A course project for POME (Principles of Management and Economics) at RVCE, Bengaluru. The EL component is worth 40 marks:
- Phase I (20 marks): journal-style paper or simulation project on Management and Economics.
- Phase II (20 marks): finish the paper and submit it to a journal or conference. Publication is not mandatory.
The topic must be approved by the teacher within a week of 22 Nov 2026. The whole thing must be finished fast (2-3 hours of hands-on work is the target). Team of a few students, so work should be easy to split.

## Title
Do AI Consumers Obey the Law of Demand? Testing Rationality and Price Sensitivity of LLM Agents with Indian Household Personas

## Research question
Do LLM agents given Indian income-group personas behave like rational consumers: downward-sloping demand, consistent preferences (GARP), sensible income effects and realistic elasticities?

## Design (no human participants, no survey)
- 3 to 5 LLMs, each plays 4 personas (low, lower-middle, middle, high income; urban Bengaluru) with monthly budgets Rs 6,000 / 12,000 / 25,000 / 60,000.
- 5 goods: staples, milk, eating out, rides, entertainment (base prices in config.py).
- Each call: persona + budget + price list -> model returns a JSON quantity bundle.
- Per persona and repetition: 46 prompts (1 baseline, 20 own-price changes, 5 income changes, 20 random budget sets for GARP). 1,840 calls per model at 10 reps (184 per rep).
- Temperature 1.0. Failed calls retried 4 times, then dropped and counted.

## Locked run settings (22 Nov 2026)

Design is locked: low reasoning effort, rpm 8 per key, 6 reps per model. Do not change settings after real runs start.

- Both models (gpt-oss-20b, gpt-oss-120b) run with --reasoning-effort low --rpm 8. Chosen for feasibility (token limits) and valid JSON format, not for result quality.
- 6 reps per model (--reps 6), all run from one laptop with the 3 Groq keys in rotation (one key per account). No rep split between teammates.
- One output file per model: results/raw_gptoss20b.jsonl and results/raw_gptoss120b.jsonl, merged into results/raw.jsonl for analyze.py.

## Tests
Law of demand, own-price and cross-price elasticity, income elasticity and Engel's law, GARP with CCEI (Afriat), variability across repeated runs.

## Hypotheses
- H1: demand curves slope downward almost everywhere.
- H2: choices are close to GARP-consistent (CCEI near 1).
- H3: elasticities are too uniform across goods (necessities vs luxuries).
- H4: income effects follow Engel's law, but persona changes quantities less than the persona text suggests.

## Files (in the zip)
- config.py: goods, prices, personas, scenarios, prompt builder.
- experiment.py: runs the models. Providers: groq, google, openrouter, openai, anthropic, compat, mock. Multi-key rotation, per-key rate cap, .env loading. Resumes automatically (skips finished calls in --out). Output is JSONL.
- .env.example, .gitignore: key template and ignore rules.
- analyze.py: reads raw.jsonl, writes tables (.tex/.csv), figures (png) and results/macros.tex.
- main.tex + refs.bib: paper draft. Pulls numbers, tables and figures from results/ automatically.
- README.md: run commands.

## Status
Done: code, analysis, paper skeleton (intro, related work, theory, design, method, limitations). Pipeline tested end to end on the offline mock agent only. No real results yet.
Fixed: analyze.py crashed on model names with "/" or ":" (for example google/gemma-4-31b-it:free). The cross-price file name is now sanitised. Use the fixed version.
Not done: real runs, results/discussion/abstract/conclusion text, Table 1 model names and dates, reference verification, GitHub upload.

## How to run
Use the project virtual environment (.venv, git-ignored, made from requirements.txt): run `.venv\Scripts\python` (or activate with `.venv\Scripts\Activate.ps1`). Do not install packages into the global Python.
Free tiers only. Providers used: Groq (several keys in rotation) and Google Gemini Flash (several keys in rotation).
Keys live in a `.env` file next to experiment.py (the runner loads it; see .env.example). `.env` is in .gitignore and must NEVER be committed or pasted anywhere. Several keys for one provider are comma-separated:
  GROQ_API_KEYS=k1,k2,k3
  GEMINI_API_KEYS=k1,k2
Model spec is provider:model-id, for example:
  python experiment.py --models groq:openai/gpt-oss-20b --reps 1 --limit 40 --rpm 8 --reasoning-effort low --out results/test_20b.jsonl
Smoke test first (--limit 40 samples all personas and scenario types) and check the failure count. Free models sometimes break the JSON format; swap any model that fails a lot.
Full run (locked settings), one terminal per model:
  .venv\Scripts\python experiment.py --models groq:openai/gpt-oss-20b --reps 6 --rpm 8 --reasoning-effort low --out results/raw_gptoss20b.jsonl
  .venv\Scripts\python experiment.py --models groq:openai/gpt-oss-120b --reps 6 --rpm 8 --reasoning-effort low --out results/raw_gptoss120b.jsonl
Re-run the same command to resume after a stop. Merge with Python (not PowerShell Set-Content), then analyse:
  .venv\Scripts\python -c "files=['results/raw_gptoss20b.jsonl','results/raw_gptoss120b.jsonl']; rows=[l.rstrip('\n')+'\n' for f in files for l in open(f, encoding='utf-8') if l.strip()]; open('results/raw.jsonl','w',encoding='utf-8',newline='\n').writelines(rows); print(len(rows), 'rows written to results/raw.jsonl')"
Then: python analyze.py results/raw.jsonl results
Mock pipeline tests go to results_mock/ only (git-ignored), never results/:
python experiment.py --models mock:mock --reps 1 --out results_mock/mock.jsonl, then python analyze.py results_mock/mock.jsonl results_mock
- --rpm is requests per minute PER KEY. Set it a little below the provider's limit for that model.
- Key rotation: every call takes the key that is free soonest. A rate-limited key is paused for the wait the provider states; a key out of daily quota is retired for the run. If all keys of a model are exhausted the run stops cleanly. Calls not completed because of limits are NOT saved, so re-running the same command continues where it stopped.
- Run different models in separate terminals at the same time. Limits on Groq and Gemini are per model.
- Reasoning models (gpt-oss, Qwen3) spend extra tokens on thinking, which counts against token-per-day limits. Check usage in the smoke test. <think> tags are stripped before parsing.
- If time or quota is short, use --reps 3 to 5.

## Planned models (verify each id on the provider's current model list and write exact ids and the run date in Table 1)
Final line-up (22 Nov 2026), two models from one family, so size can be compared but family cannot:
- groq:openai/gpt-oss-20b (small reasoning model)
- groq:openai/gpt-oss-120b (large reasoning model)
Dropped before data collection:
- groq:qwen/qwen3.8-27b: at reasoning effort low it used about 3,525 completion tokens per call (3,480 reasoning), measured on 9 answers, about 10x gpt-oss-120b. 6 reps would need about 4.5M tokens, 7+ days on the 3 free accounts. Not feasible. results/test_qwen.jsonl is kept as a record but must never be used in the paper.
- Llama models and qwen/qwen3-32b: retired by Groq (llama-3.3-70b-versatile returned 404 on 22 Nov 2026).
- Gemini Flash: free-tier daily limits far too low for this design.
Groq can retire model ids at any time, so check the model list before the full run.

## Known constraints: free-tier limits
- One model needs 1,840 calls at 10 reps (about 350 tokens per call before any reasoning tokens). Daily token caps, not request caps, may be the real limit for reasoning models.
- Provider limits usually apply per account or project, not per key. Extra keys help only if they belong to different people's own accounts (for example each teammate's own key). Do not create extra accounts just to get around limits; that can break provider terms.
- OpenRouter ":free" models allow only 50 requests per day (under 10 credits bought), so they are not used.

## Paper draft (main.tex, updated 22 Nov 2026)
main.tex and refs.bib were rewritten using only verified sources (see SOURCES_CHECK.md). main.tex now expects three NEW tables from analyze.py, with exactly these file names in results/tables/:
- own_price_mean.tex: own-price elasticity from the log of mean quantity per price point (zeros included).
- zero_share.tex: share of zero-quantity answers by good x persona, per model.
- corner_share.tex: share of answers where one good takes more than 80% of spending, per model.
Tables that don't exist yet show a red TODO in the PDF, so the paper always compiles.

## Rules for working on this project
- Never put mock agent output in the paper. Real results only.
- Never invent numbers, results or citations. Every number in the paper must come from results/.
- refs.bib was written from memory. Every entry must be verified on Google Scholar. The related-work descriptions of each paper are also unverified.
- The claim that the Indian setting is "absent from existing work" must be checked, and softened if anything similar exists.
- Without human data, say "consistent with theory", never "human-like".
- Add one published Indian household demand-elasticity estimate as a benchmark and cite it.
- Record exact model names, versions, run date, prompts and API costs for reproducibility.
- Keep code modular and production-grade. Do not change the experimental design (prompts, personas, prices, scenarios) once real runs have started, or earlier data becomes unusable.

## What is left, in order
1. Get the teacher's approval of the pitch.
2. Set up free keys and pick 3+ models.
3. Smoke test each model, then full runs.
4. Run analyze.py and check the tables and figures.
5. Fill Table 1 (models, dates), price source, GitHub link in main.tex.
6. Verify refs.bib and find the Indian elasticity benchmark.
7. Write results, discussion, abstract and conclusion from the real numbers.
8. Compile main.tex (Overleaf or local LaTeX) and proofread.

## Style preferences
Plain, simple English. One step at a time. Nothing overcomplicated.
