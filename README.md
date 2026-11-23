# Forecasting hunger where monitoring has gone dark

Course project for Global Issues and AI (FITE GIA).

In 2025 FEWS NET, the main famine early-warning system, went offline for months after its funding stopped. The FEWS NET download used here has no assessments at all for February and June 2025. This project asks a narrow, testable question. When the ground assessments stop, how well can free, remotely collected data forecast food crises four months ahead? The remote data is satellite rainfall, market prices, conflict records and news.

It covers 39 admin-1 regions in Ethiopia, Somalia and South Sudan. A dashboard shows where the forecasts work and where they fail.

## Open the dashboard

```bash
python3 -m http.server 8765 --directory prototype
```

Then open http://localhost:8765. The switch at the top moves between two scenarios: "monitoring working", and "monitoring gone dark", where the newest usable assessment is at least 12 months old.

## Rebuild from the raw data

```bash
bash src/download_large.sh
```

```bash
python3 src/conflict.py
```

```bash
python3 src/build_dataset.py
```

```bash
python3 src/timesfm_drivers.py
```

```bash
python3 src/train_model.py
```

Needs Python with pandas, scikit-learn and matplotlib, plus `timesfm[torch]` for `timesfm_drivers.py`, which downloads the TimesFM 2.5 weights (925 MB) from Hugging Face on first run. Two raw files are larger than GitHub's 100 MB limit and are not in the repository; `download_large.sh` fetches them from their original sources. FEWS NET serves live data, so a fresh download can differ slightly from the 23 November 2026 snapshot used for the results. `python3 src/fetch_news.py` re-downloads the GDELT news series, which takes about 15 minutes because the API allows one request every 5 seconds.

## Check the numbers

```bash
python3 src/verify.py
```

This is an independent audit, and it exits with an error if any check fails. It checks these things:

- It recomputes region crisis shares straight from the raw FEWS NET files.
- It confirms that no input is dated after the forecast cutoff, including in the dark scenario.
- It compares the coordinate-based placement of conflict events against UCDP's own region names. They agree for 99.6% of events.
- It recomputes every metric from `results/predictions.csv` and confirms that each row of the tables below matches `results/metrics.json`.

## What it does

- **Unit:** one admin-1 region at one FEWS NET assessment date.
- **Target:** the share of the region's FEWS NET units in IPC phase 3 or worse. "In crisis" means 50% or more.
- **Inputs, as known four months before the target date:**
  - CHIRPS rainfall as a percentage of normal (1 and 3 months, the driest spell in the past year, the 6-month average)
  - WFP cereal prices, change over 3 and 12 months
  - UCDP organised violence: events and deaths in the last 3 months, and the change against the previous year. Each event is placed in its region by its coordinates.
  - The last FEWS NET assessment and how old it is. In the dark scenario this is the newest one at least 12 months before the cutoff.
- **Model:** gradient-boosted trees (scikit-learn), with untuned default settings. The list of model variants was fixed before the results were seen, and every variant is reported below.
- **Test:** walk-forward from January 2022 to June 2026. For each date the model is retrained using only outcomes that were known at the cutoff. The dark models only use outcomes known 12 months before it.

## Results (479 test cases)

### While monitoring works

| Method | Correct crisis calls | New crises foreseen | Average error | Cases |
| --- | --- | --- | --- | --- |
| Assume nothing changes <!-- all.persistence --> | 86% | 0 of 40 | 13 points | 479 |
| This tool <!-- all.full --> | 82% | 7 of 40 | 18 points | 479 |
| This tool plus conflict data <!-- all.full_conf --> | 81% | 6 of 40 | 18 points | 479 |
| No ground data: rainfall and prices <!-- all.satellite --> | 64% | 5 of 40 | 32 points | 479 |
| No ground data: rainfall, prices, conflict <!-- all.remote --> | 64% | 7 of 40 | 31 points | 479 |
| This tool plus TimesFM forecasts <!-- all.full_tfm --> | 82% | 6 of 40 | 18 points | 479 |
| No ground data: rainfall, prices, conflict, plus TimesFM <!-- all.remote_tfm --> | 64% | 4 of 40 | 32 points | 479 |
| FEWS NET analysts' projection <!-- all.fews --> | 88% | 16 of 30 | 12 points | 440 |

### Once monitoring has gone dark (newest assessment 12+ months old)

| Method | Correct crisis calls | New crises foreseen | Average error | Cases |
| --- | --- | --- | --- | --- |
| Repeat the year-old assessment <!-- dark.persistence_dark --> | 68% | 0 of 98 | 26 points | 479 |
| This tool in the dark <!-- dark.dark --> | 67% | 31 of 98 | 26 points | 479 |
| This tool in the dark, plus conflict <!-- dark.dark_conf --> | 67% | 31 of 98 | 27 points | 479 |
| This tool in the dark, plus TimesFM <!-- dark.dark_tfm --> | 66% | 26 of 98 | 27 points | 479 |

The 95% ranges below come from a paired bootstrap over assessment dates, in `metrics.json` under `mae_diff_ci`.

## What the results show

1. **While monitoring works, the tool does not beat simply repeating the last assessment.** Its average error is 4 points higher (95% range 2 to 6). Food crises persist, so that baseline is strong, and FEWS NET analysts beat both.
2. **Once monitoring stops, the old assessment loses most of its value.** Its accuracy falls from 86% to 68%, and 98 regions enter a crisis it cannot foresee by definition.
3. **In the dark, the tool's average error is no different from the year-old assessment** (difference 0 points, 95% range −5 to +5). The two fail in different ways:
   - The tool foresees 31 of 98 new crises and 43 of 53 recoveries. The year-old assessment foresees none of either.
   - When the tool calls a crisis it is right 88% of the time, against 78% for the old assessment.
   - It misses more crises that are already under way: it finds 52% of them, against 66%.

   So it is useful for flagging regions to check first, not for replacing an assessment.
4. **Conflict data did not help.** Adding UCDP features changed the dark-scenario error by +0.8 points (95% range +0.2 to +1.5). The README's earlier guess that conflict data was the missing input is not supported.
5. **Remote data alone cannot replace monitoring.** With no assessment at all, accuracy is 64%.
6. **Google's TimesFM forecasting model did not help.** TimesFM 2.5 (zero-shot, no training) forecast each region's rainfall and cereal prices forward to the target month, and these forecasts were added as inputs. The variants were fixed before the results were seen.
   - As a forecast of the drivers themselves it works better than repeating the last value: average error 25.1 against 35.7 points of rainfall-as-%-of-normal (479 cases), and 0.139 against 0.147 for the 12-month log price change (463 cases). These are under `driver_check` in `metrics.json`.
   - It did not improve the crisis forecast. Against the same model without it, the average error changed by +0.1 points while monitoring works (95% range −0.5 to +0.7), +0.9 points in the dark (−0.1 to +1.8) and +0.6 points with no ground data (+0.1 to +1.1). In the dark it foresaw 26 of 98 new crises, against 31 without it.

## How this differs from existing systems

- **FEWS NET and IPC** rely on field teams and analysts. This tool is designed for the case where they are offline.
- **WFP HungerMap LIVE** uses WFP's own phone surveys. This tool uses only data that nobody has to collect on the ground.
- **The 2023 Science Advances study on news and food crises** (Balashankar et al.) showed that news text predicts food insecurity. This tool is a small working pipeline in that direction. Its GDELT news features are still being added.

Its claim is limited to what the test above supports. It is a triage aid for regions that have gone dark, and it is measured against what you would otherwise have.

## Not done

- AlphaEarth satellite embeddings (Google DeepMind) are designed in `docs/design-google-models.md` but not added. They need an Earth Engine sign-in.
- TimesFM was pretrained on a large public corpus that was not checked for overlap with these rainfall and price series.
- GDELT news features: the code is in place (`src/fetch_news.py`, `news_*` variants in `train_model.py`). The download was rate-limited by the API at the time of this commit, so no news results are reported yet.
- No language-model reading of local press. It could not be back-tested honestly without the historical articles.
- Regions are admin-1 only. FEWS NET units are finer, so local detail is averaged away.
- Ethiopia's regions that split from SNNPR after 2020 are merged back into SNNPR to keep one continuous series.
- UCDP's final GED data (to 2025) is used for past months, while only candidate data would have existed in real time. Candidate data is used for 2026.
- The model settings were not tuned.

## Data sources

- FEWS NET Data Warehouse, IPC classifications: `https://fdw.fews.net/api/ipcphase.csv?country_code=ET` (also SO, SS). Ethiopia re-downloaded 23 November 2026 (adds the October 2026 assessment); Somalia and South Sudan downloaded 10 November 2026.
- WFP food prices on HDX: `wfp-food-prices-for-ethiopia`, `-somalia`, `-south-sudan`. Downloaded 10 November 2026.
- WFP subnational rainfall indicators on HDX (from CHIRPS): `eth-rainfall-subnational`, `som-`, `ssd-`. Downloaded 10 November 2026.
- UCDP Georeferenced Event Dataset 26.1 (1989–2025) and UCDP Candidate Events 26.0.1–26.0.9 (January–September 2026), `https://ucdp.uu.se/downloads/`. Downloaded 17 November 2026.
- geoBoundaries gbOpen ADM1 outlines for ETH, SOM and SSD. Downloaded 17 November 2026.
- TimesFM 2.5 weights, `google/timesfm-2.5-200m-pytorch` on Hugging Face (Apache-2.0), with the `timesfm` 3.0.2 Python package. Downloaded 23 November 2026.

## Prompt log addition

| #   | Date        | Prompt                                                         | What it was for                                 |
| --- | ----------- | -------------------------------------------------------------- | ----------------------------------------------- |
| 6   | 10 Nov 2026 | Pasted the init file, then: "do this proejct create prototype" | Building the data pipeline, model and dashboard |
| 7   | 17 Nov 2026 | "fix this", then a pasted comparison with existing systems     | Conflict data, the gone-dark test, new dashboard |
