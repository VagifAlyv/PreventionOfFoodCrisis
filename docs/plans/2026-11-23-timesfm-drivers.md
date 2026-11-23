# Plan: TimesFM 2.5 driver forecasts (with/without test)

Spec: `docs/design-google-models.md` (variants B; AlphaEarth variants C/D wait for Earth Engine sign-in).

## Global Constraints

- Only real files in `data/raw/`. No typed-in numbers. Every number in README comes from `results/metrics.json`.
- `python3 src/verify.py` must pass at the end of every task that touches results.
- No input dated after the cutoff (target − 4 months). TimesFM context must end at or before the cutoff.
- Weak results are reported as they are. All pre-registered variants are reported.
- Pre-registered before any TimesFM result is seen:
  - Features `TFM = ["tfm_r3q", "tfm_price_yoy"]`: TimesFM 2.5 point forecast, for the target month T, of the region's monthly `r3q` series and of its cereal `price_yoy` series (regional, or national when the region has no market at the cutoff, mirroring `build_dataset.py`).
  - Variants: `full_tfm = SAT+IPC+TFM`, `remote_tfm = SAT+CONF+TFM`, `dark_tfm = SAT+IPC_DARK+TFM`.
  - Paired CIs: `full_tfm` vs persistence, `full_tfm_vs_full`, `remote_tfm_vs_remote`, `dark_tfm` vs dark persistence, `dark_tfm_vs_dark`.
  - Driver check: MAE of TimesFM vs "repeat the last value" for `r3q` and `price_yoy` at T, on test rows.
  - Context: the contiguous run of monthly values ending at the newest value dated ≤ cutoff, at most 120 months, at least 24; newest value at most 1 month (rain) / 3 months (price) old; horizon = T − newest month.
- Test runner: plain-assert scripts (`python3 tests/<file>.py`); pytest is not installed.

## Task 1: Re-baseline on the 23 Nov 2026 download

The committed results came from a 10 Nov FEWS NET snapshot that is not on disk. The 23 Nov download adds Ethiopia's Oct 2026 assessment (test set 468 → 479).

- Step 1: run `python3 src/conflict.py && python3 src/build_dataset.py && python3 src/train_model.py`. Expected: exit 0.
- Step 2: update every number in README (tables and prose) from the new `results/metrics.json`; set FEWS NET download date to 23 November 2026.
- Step 3: `python3 src/verify.py`. Expected: 0 failed.
- Step 4: commit "Re-baseline on 23 Nov 2026 FEWS NET download".

Test command: `python3 src/verify.py`

## Task 2: Context extraction with leakage tests

Interfaces: produces `src/timesfm_drivers.py` with
`context(series: pd.Series[Period→float], cutoff: Period, max_age: int) -> (np.ndarray | None, Period | None)` and
`driver_table(panel, rain, preg, pnat, forecast) -> DataFrame[cc, region, target_date, tfm_r3q, naive_r3q, obs_r3q, ctx_end_r3q, tfm_price_yoy, naive_price_yoy, obs_price_yoy, ctx_end_price_yoy]`
where `forecast(contexts: list[np.ndarray], horizons: list[int]) -> list[float]`.

- Step 1: write `tests/test_timesfm_drivers.py`: (a) changing values after the cutoff leaves the context unchanged; (b) a gap cuts the context to the run after the gap; (c) newest value older than max_age → None; (d) fewer than 24 values → None; (e) at most 120 values; (f) `driver_table` with a fake forecaster that returns the last value gives `tfm == naive`, passes horizon = T − ctx_end, and every `ctx_end <= cutoff`. Run it. Expected: fails (module missing).
- Step 2: implement. Run tests. Expected: all pass.
- Step 3: commit.

Test command: `python3 tests/test_timesfm_drivers.py`

## Task 3: Run TimesFM 2.5

- Step 1: add `main()` to `src/timesfm_drivers.py`: load `google/timesfm-2.5-200m-pytorch`, compile with `max_context=128`, `max_horizon` covering the largest horizon, `normalize_inputs=True`; write `data/processed/timesfm_drivers.csv`. Run it. Expected: one row per panel row; non-null `tfm_r3q` wherever a valid context exists.
- Step 2: commit the script change and CSV.

Test command: `python3 tests/test_timesfm_drivers.py`

## Task 4: Variants, driver check, verify checks

Interfaces: consumes `timesfm_drivers.csv` from Task 3 (columns above).

- Step 1: extend `verify.py`: TimesFM context never after the cutoff; new variant metrics recompute from `predictions.csv`. Run before the model change. Expected: the new checks fail (columns missing).
- Step 2: `train_model.py`: merge drivers on (cc, region, target_date); add the three variants, paired CIs and `driver_check` to metrics; add predictions columns. Run pipeline + verify. Expected: 0 failed.
- Step 3: commit.

Test command: `python3 src/verify.py`

## Task 5: Report

- Step 1: README: TimesFM rows in both tables (with `<!-- scope.key -->` markers), a short honest finding with CIs from metrics.json, sources and limits (pretraining-overlap caveat). Mark design doc status.
- Step 2: `python3 src/verify.py`. Expected: 0 failed. Commit.

Test command: `python3 src/verify.py`

## Review Focus

- Leakage through the context window (off-by-one on the cutoff, national-price fallback).
- Variants or pairs added after seeing results.
- Any README number not traceable to metrics.json.
