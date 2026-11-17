# Project instructions

Work on the project, make sure it is very accurate, solves real world problem with real datasets and doesnt hallucinate at all and works on very accurate data.

In practice for this repo:

- Only use the real files in `data/raw/`. Never invent, fill in or "round up" numbers.
- Every number in `README.md` and the dashboard must come from `results/metrics.json` or the raw data. Run `python3 src/verify.py` after any change; it must pass.
- Report weak results as they are (the model does not beat the persistence baseline overall).
