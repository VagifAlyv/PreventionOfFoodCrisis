# Design: adding Google's ready-made models to the crisis forecast

Status: **proposal, nothing built or measured yet.** This document contains no results. Every performance number in this repo stays in `results/metrics.json`; this design adds none.

## 1. Goal

Test whether two ready-made Google models improve the existing four-month-ahead forecast of food crises in 39 admin-1 regions of Ethiopia, Somalia and South Sudan. The current model is the gradient-boosted tree in `src/train_model.py`. Its limits are in `README.md`: overall it does not beat the persistence baseline, and in the "dark" scenario its average error is no different from the year-old assessment.

A model is kept only if the with/without test below shows a gain. If it shows none, we report that.

## 2. The two models

| | AlphaEarth satellite embeddings | TimesFM |
|---|---|---|
| Made by | Google and Google DeepMind | Google Research |
| Role here | Last year's land picture per region | Forecast of each monthly driver |
| Output | 64 numbers per 10 m patch, per year | Forward forecast of a series, zero-shot |
| Licence | CC-BY 4.0 (data) | Apache-2.0 for code and weights up to 2.5 |
| Version we would use | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | 2.5 |

### What was checked (23 Nov 2026) and what was not

Checked on Google pages:
- The Earth Engine catalogue page lists availability from 2017-01-01 to 2025-01-01. An annual image's end time is the next 1 January, so the last full year is 2024. The French version of the same page shows an end of 2024-01-01, so the catalogue wording differs between locales. **Re-read the live English catalogue page before building** and use whatever years it lists.
- The CC-BY 4.0 licence is stated in the Cloud Storage guide for the same embeddings (`aef_on_gcs_readme`), not on the Earth Engine catalogue page. Required attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."
- Earth Engine is free only for noncommercial use. Commercial use needs a paid plan, and Google says unpaid commercial use is not allowed. The noncommercial page says the Google for Startups Cloud Program can cover Earth Engine costs; eligibility is not guaranteed.
- TimesFM weights up to 2.5 are Apache-2.0, per the PyPI page.

Not confirmed from an official source, so do not repeat as fact:
- That TimesFM 3.0 weights are non-commercial. This comes from a Gitee mirror and secondary articles, not the official README.
- That commercial use of 3.0 goes through Google Cloud.
- That TimesFM 2.5 accepts extra covariates alongside the series. Read the 2.5 documentation before relying on this. If it does not, the forecast uses each series on its own.

For an investor-facing product the plan is TimesFM 2.5 plus the commercial question for Earth Engine, which needs a decision before any commercial use.

## 3. Two limits that shape the design

1. **AlphaEarth is yearly.** It cannot see a rainy season failing month to month. It describes land, crops and water, not the current season.
2. **A year's embedding covers that whole year**, so a forecast can use only embeddings for years that ended before its cutoff. A cutoff in March 2024 may use the 2023 embedding (it ended 1 January 2024) but not 2024. The rule in code: year Y is usable if 1 January of Y+1 is on or before the cutoff. This matches the leakage rule `verify.py` already enforces for other inputs.

Embeddings start in 2017, so early test cutoffs (from January 2022) have at most five years of history. Use the most recent usable year, and also try the change against the year before.

## 4. Pipeline

1. **Regions to embeddings.** Take the existing geoBoundaries ADM1 polygons in `data/raw/boundaries`. In Earth Engine, average the 64 bands over each region for each year (`reduceRegions`, mean). Export one CSV: region, year, `A00`–`A63`. 39 regions times the available years is a small table. Save it in `data/raw/` with its download date.
2. **Reduce dimensions.** 64 bands against about 468 test cases invites overfitting. Fit a PCA on the training cutoffs only, keep a handful of components, and fix that number before looking at results.
3. **TimesFM drivers.** For each region and cutoff, feed the monthly series known at the cutoff (rainfall as % of normal, cereal price) to TimesFM 2.5 and take its forecast for the target month. Use the forecast as an input feature, plus the forecast minus the last observed value.
4. **Crisis model.** Same gradient-boosted trees, same walk-forward from January 2022, same target (share of FEWS NET units in IPC phase 3 or worse, "in crisis" at 50% or more), same four-month horizon as the README. The user's note said 3–6 months. Start with four so results compare with the existing tables, then add other horizons if wanted.
5. **Report.** Same metrics, same two scenarios, same persistence and FEWS NET baselines.

## 5. With/without test

Variants, fixed before any result is seen, each trained and scored exactly like the existing ones:

| Variant | Inputs |
|---|---|
| A (existing) | Current features |
| B | A plus TimesFM forecasts |
| C | A plus AlphaEarth components |
| D | A plus both |

Also run B, C and D in the dark scenario (newest assessment at least 12 months old), because that is the project's question.

Report all variants, including those that do not help. Use the paired bootstrap over assessment dates already behind `mae_diff_ci`, so a "gain" is a difference with its 95% range, not a point estimate. One extra check: a TimesFM forecast should beat "repeat last value" for the driver itself; if it does not, say so.

## 6. Verification rules (from `.claude/CLAUDE.md`)

- Only real files in `data/raw/`; the embedding export and any forecasts are generated from them, not typed in.
- No input dated after the forecast cutoff. Extend `src/verify.py` with a check for the embedding-year rule in section 3 and for TimesFM inputs.
- Every number in the README and dashboard comes from `results/metrics.json`. `python3 src/verify.py` must pass after each change.
- Weak results are reported as they are.

## 7. Open decisions

1. **Earth Engine licence.** A course project can use noncommercial access. A commercial product needs a paid plan or startup credits. Decide before any commercial use.
2. **Where TimesFM runs.** Local on CPU is likely enough for 39 series; confirm when implementing.
3. **Horizon.** Four months (current) or 3–6.

## Sources

- AlphaEarth catalogue: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
- AlphaEarth licence and attribution: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme
- Earth Engine noncommercial and commercial terms: https://earthengine.google.com/noncommercial
- TimesFM: https://github.com/google-research/timesfm and https://pypi.org/project/timesfm/
