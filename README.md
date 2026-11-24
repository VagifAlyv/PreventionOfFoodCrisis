# Forecasting hunger where monitoring has gone dark

Course project for Global Issues and AI (FITE GIA).

In 2025 FEWS NET, the main famine early-warning system, went offline for months after its funding stopped. In the data used here, no country has a FEWS NET assessment between October 2024 and October 2025. This project asks a narrow, testable question: when the ground assessments stop, how well can free, remotely collected data (satellite rainfall and market prices) forecast food crises four months ahead?

It covers every country FEWS NET monitors that the data allows: 633 admin-1 regions in 38 countries across Africa, Asia, Latin America and the Middle East. A world dashboard shows where the forecasts work and where they fail.

## Open the dashboard

```bash
python3 -m http.server 8765 --directory prototype
```

Then open http://localhost:8765. The switch at the top moves between two scenarios: "monitoring working", and "monitoring gone dark", where the newest usable assessment is at least 12 months old.

## Rebuild from the raw data

```bash
bash src/download_world.sh
```

```bash
python3 src/build_dataset.py
```

```bash
python3 src/train_model.py
```

Needs Python with pandas, scikit-learn and openpyxl. `download_world.sh` fetches about 2.6 GB into `data/raw/world/` (not in the repository), using the HDX links in `src/world_countries.csv` as of 24 November 2026. FEWS NET serves live data, so a fresh download can differ slightly from the 24 November 2026 snapshot used for the results.

## Check the numbers

```bash
python3 src/verify.py
```

This is an independent audit, and it exits with an error if any check fails:

- It recomputes region crisis shares straight from each country's raw FEWS NET file, using a simpler matching rule than the build. This runs for 27 countries; the others have too few regions that the simpler rule can match.
- It checks that `results/coverage.json` agrees with the data, and that every left-out country has a stated reason.
- It confirms that no input is dated after the forecast cutoff, including in the dark scenario.
- It recomputes every metric from `results/predictions.csv`, and the earlier 3-country results from `results/three_countries/`, and checks each row of the tables below against them.

## What it does

- **Unit:** one admin-1 region at one FEWS NET assessment date.
- **Target:** the share of the region's FEWS NET units in IPC phase 3 or worse. "In crisis" means 50% or more.
- **Inputs, as known four months before the target date:**
  - CHIRPS rainfall as a percentage of normal (1 and 3 months, the driest spell in the past year, the 6-month average)
  - WFP cereal prices, change over 3 and 12 months (the national median where a region has no market)
  - The last FEWS NET assessment and how old it is. In the dark scenario this is the newest one at least 12 months before the cutoff.
  - The country and the month
- **Model:** gradient-boosted trees (scikit-learn), untuned. The variants and comparisons were fixed before the results were seen, and all of them are reported below.
- **Test:** walk-forward over every assessment from February 2022 to October 2026. For each date the model is retrained using only outcomes known at the cutoff. The dark model uses only outcomes known 12 months before it.

### Matching regions across sources

Each source names regions its own way. Regions are keyed by the OCHA admin-1 code (p-code). A FEWS NET unit is assigned to the right-most part of its name that exactly matches an OCHA admin-1 name (in any language the OCHA table lists), or else to the parent of an exactly matching admin-2 name. Names are compared after lower-casing, removing accents and dropping words such as "region" and "province". Nothing is matched by similarity. The reasoning is in `docs/adr/0001-world-coverage.md`.

- 96.3% of FEWS NET unit-assessments were matched. The rest are left out.
- Below 90%: Syria 57%, Yemen 72%, Burkina Faso 87% (pre-2025 region names), Burundi 87%, Angola 88%.
- Lesotho is left out because HDX has no OCHA region table for it.
- Per-country details are in `results/coverage.json`.

## Results

### While monitoring works

Like-for-like: only cases where there is an earlier assessment to repeat. A region's first-ever assessment (260 cases, mostly countries FEWS NET added in 2026) has none. On all 5,181 cases this tool gets 89% right with an average error of 14 points.

| Method | Correct crisis calls | New crises foreseen | Average error | Cases |
| --- | --- | --- | --- | --- |
| Assume nothing changes <!-- paired.persistence --> | 90% | 0 of 283 | 10 points | 4921 |
| This tool <!-- paired.full --> | 89% | 14 of 283 | 14 points | 4921 |
| No ground data: rainfall and prices <!-- paired.satellite --> | 79% | 17 of 283 | 21 points | 4921 |
| FEWS NET analysts' projection <!-- paired.fews --> | 93% | 176 of 251 | 7 points | 4486 |

### Once monitoring has gone dark (newest assessment 12+ months old)

| Method | Correct crisis calls | New crises foreseen | Average error | Cases |
| --- | --- | --- | --- | --- |
| Repeat the year-old assessment <!-- dark_paired.persistence_dark --> | 83% | 0 of 489 | 16 points | 4619 |
| This tool in the dark <!-- dark_paired.dark --> | 81% | 62 of 489 | 19 points | 4619 |

The 95% ranges below come from a paired bootstrap over assessment dates (`mae_diff_ci` in `metrics.json`).

## What the results show

1. **While monitoring works, the tool does not beat simply repeating the last assessment.** Its average error is 3.8 points higher (95% range 3.0 to 4.6). Food crises persist, so that baseline is strong, and FEWS NET analysts beat both.
2. **In the dark, the tool is worse on average than the year-old assessment**: 3.2 points higher error (2.2 to 4.3). It fails differently, though:
   - It foresees 62 of 489 new crises and 216 of 284 recoveries. The year-old assessment foresees none of either.
   - When it calls a crisis it is right 88% of the time, against 76% for the old assessment.
   - It finds only 43% of crises already under way, against 65%.

   So at best it can flag regions to check first. It does not replace an assessment.
3. **Rainfall and prices alone cannot replace monitoring.** Without any assessment the average error is 11.6 points higher than repeating the last one (10.1 to 13.2).
4. **Across 38 countries the dark-scenario result is weaker than in the first 3.** In the earlier Ethiopia, Somalia and South Sudan test the tool tied the year-old assessment (difference 0 points, −5 to +5); here it is 3.2 points behind. While monitoring works it trails the last assessment in both tests.

These results were first computed on all cases, which gave the baseline fewer cases than the model. The like-for-like tables were added after that run. The all-case numbers are in `metrics.json` under `all` and `dark`.

## Earlier 3-country experiments

Before the world version, the model was tested on Ethiopia, Somalia and South Sudan (39 regions, 479 test cases) with two extra kinds of input. Neither helped, and neither is used for the world model. The results are frozen in `results/three_countries/` and can be rebuilt from commit 93b27f6.

| Method | Correct crisis calls | New crises foreseen | Average error | Cases |
| --- | --- | --- | --- | --- |
| Assume nothing changes <!-- three_all.persistence --> | 86% | 0 of 40 | 13 points | 479 |
| This tool <!-- three_all.full --> | 82% | 7 of 40 | 18 points | 479 |
| This tool plus conflict data <!-- three_all.full_conf --> | 81% | 6 of 40 | 18 points | 479 |
| This tool plus TimesFM forecasts <!-- three_all.full_tfm --> | 82% | 6 of 40 | 18 points | 479 |
| FEWS NET analysts' projection <!-- three_all.fews --> | 88% | 16 of 30 | 12 points | 440 |
| Repeat the year-old assessment <!-- three_dark.persistence_dark --> | 69% | 0 of 98 | 26 points | 479 |
| This tool in the dark <!-- three_dark.dark --> | 67% | 31 of 98 | 26 points | 479 |
| In the dark, plus conflict <!-- three_dark.dark_conf --> | 67% | 31 of 98 | 27 points | 479 |
| In the dark, plus TimesFM <!-- three_dark.dark_tfm --> | 66% | 26 of 98 | 27 points | 479 |

- **Conflict data (UCDP)** changed the dark-scenario error by +0.8 points (95% range +0.2 to +1.5).
- **Google's TimesFM 2.5** forecast each region's rainfall better than repeating the last value (average error 25.1 against 35.7 points of rainfall as % of normal), but adding its rainfall and price forecasts did not improve the crisis forecast: +0.1 points while monitoring works (−0.5 to +0.7) and +0.9 points in the dark (−0.1 to +1.8). For 188 of the 479 cases its price forecast used the national series. TimesFM's pretraining data was not checked for overlap with these series.

## How this differs from existing systems

- **FEWS NET and IPC** rely on field teams and analysts. This tool is designed for the case where they are offline.
- **WFP HungerMap LIVE** uses WFP's own phone surveys. This tool uses only data that nobody has to collect on the ground.
- **Balashankar et al. (2023, Science Advances)** showed that news text predicts food insecurity. This project did not get news data working; see below.

## Not done

- News (GDELT) features: the download was rate-limited and the code was removed.
- AlphaEarth satellite embeddings (Google DeepMind) are designed in `docs/design-google-models.md` but not added. They need an Earth Engine sign-in.
- Regions are admin-1 only. FEWS NET units are finer, so local detail is averaged away.
- Region names that changed (for example Burkina Faso's 2025 regions, Ethiopia's regions split from SNNPR) are not linked across the change; history under old names drops out.
- The world map places regions at the OCHA region centre point. It does not draw region outlines.
- The model settings were not tuned.

## Data sources

All downloaded 24 November 2026 unless noted.

- FEWS NET Data Warehouse, IPC classifications for 39 countries: `https://fdw.fews.net/api/ipcphase.csv?country_code=XX`.
- WFP subnational rainfall indicators on HDX (from CHIRPS), `<iso3>-rainfall-subnational`, 39 countries.
- WFP food prices on HDX, `wfp-food-prices-for-<country>`, 36 countries (none found for Côte d'Ivoire, Syria and Venezuela).
- OCHA administrative boundaries (COD-AB) tables on HDX, `cod-ab-<iso3>`, 38 countries.
- Natural Earth 1:110m country outlines, v5.1.2 (public domain).
- 3-country experiments only: UCDP GED 26.1 and Candidate Events 26.0.1–26.0.9 (downloaded 17 November 2026); geoBoundaries ADM1 outlines (17 November 2026); TimesFM 2.5 weights, `google/timesfm-2.5-200m-pytorch` (Apache-2.0, 23 November 2026).

## Prompt log addition

| #   | Date        | Prompt                                                         | What it was for                                 |
| --- | ----------- | -------------------------------------------------------------- | ----------------------------------------------- |
| 6   | 10 Nov 2026 | Pasted the init file, then: "do this proejct create prototype" | Building the data pipeline, model and dashboard |
| 7   | 17 Nov 2026 | "fix this", then a pasted comparison with existing systems     | Conflict data, the gone-dark test, new dashboard |
