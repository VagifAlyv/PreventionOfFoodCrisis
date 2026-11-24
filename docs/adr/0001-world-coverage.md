# ADR-0001: Extend the forecast from 3 countries to every FEWS NET country

**Status:** Accepted
**Date:** 24 November 2026
**Deciders:** project owner

## Context

The model was built and tested on 39 admin-1 regions in Ethiopia, Somalia and South Sudan, because only those countries' data had been downloaded. FEWS NET's current-situation data (downloaded 24 November 2026) has assessments for 39 countries in Africa, Asia, Latin America and the Middle East. The dashboard should show the world, not three countries, and every number on it must still come from real downloaded data (`.claude/CLAUDE.md`).

What each country needs, and what HDX had on 24 November 2026:

| Input | Source | Countries with data |
|---|---|---|
| IPC history (target and last assessment) | FEWS NET Data Warehouse | 39 |
| Rainfall, admin-1 (CHIRPS) | WFP on HDX | 39 |
| Cereal prices | WFP on HDX | 36 (none for Côte d'Ivoire, Syria, Venezuela) |
| Region codes to names | OCHA COD-AB tables on HDX | 38 (no table for Lesotho) |
| Organised violence | UCDP GED 26.1 and Candidate (global, already on disk) | all |
| Country outlines | Natural Earth 1:110m | all |

The hard part is joining these by region. FEWS NET, WFP prices, UCDP and the rainfall files each name or code regions their own way. The 3-country version used a hand-written table of 39 region codes and spelling aliases.

## Decision

One pipeline for every FEWS NET country, driven by `src/world_countries.csv`:

1. Region names are matched with a written rule: lower case, accents removed, generic words dropped ("region", "province", "state" and so on), then exact equality. Spelling aliases are allowed only as an explicit per-country table in code, each one visible in review. Anything that still does not match is dropped and listed in `results/coverage.json`, never guessed.
2. Rainfall codes are mapped to names through the OCHA table. A country with no table (Lesotho) gets no rainfall features and stays in the model with those inputs missing; the gradient-boosted trees accept missing values.
3. Conflict events are placed by UCDP's own admin-1 name with the same matching rule, because admin-1 polygons exist on disk only for the original three countries. The 3-country check showed UCDP's names agree with polygon placement for 99.6% of events.
4. The same walk-forward test, variants and baselines as before, on all countries at once, reported overall and per country.
5. The world map is drawn at country level (Natural Earth). Region-level shapes are not shown for the new countries.
6. All countries use one download date. Ethiopia, Somalia and South Sudan are downloaded fresh too, so their numbers may move slightly from the 3-country version, which stays in git history.

### Amendments after the data arrived (24 November 2026)

- Regions are keyed by OCHA admin-1 code (p-code), not by name. A FEWS NET unit belongs to the right-most part of its name that exactly matches an admin-1 name in the OCHA table, counting every name column (English, French, local), and otherwise to the parent of an exactly matching admin-2 name. This handles names that end in the country twice (Côte d'Ivoire, DR Congo) and French names in Haiti.
- Countries without an OCHA table (Lesotho) are left out and listed, not run with guessed region keys.
- Conflict and TimesFM inputs are not used for the world model. Neither helped in the 3-country test, and UCDP's own region names matched poorly outside it (under half of events in many countries). The 3-country results with them are kept unchanged in `results/three_countries/`, still checked by `verify.py`, and reproducible from commit 93b27f6.

## Options considered

### A: World map, forecasts for the 3 tested countries only
| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Download | 18 MB |
| Claim it supports | Current status worldwide; forecast quality for 3 countries |

Honest and quick, but it cannot say whether the method works outside the Horn of Africa.

### B: Run the model for all 39 countries (chosen)
| Dimension | Assessment |
|---|---|
| Complexity | High: region matching for 39 countries |
| Download | About 2.6 GB, git-ignored, refetched by `src/download_world.sh` |
| Claim it supports | Forecast quality in every country where the data joins |

Answers the real question, at the cost of a matching step that can silently lose regions. Rule 1 and `coverage.json` make every loss visible.

### C: Fuzzy name matching
Rejected. A wrong match puts one region's rainfall on another region's outcome, and nothing downstream would notice.

## Consequences

- The headline numbers change, because they now cover many more regions and countries.
- Regions that cannot be matched drop out. The README and dashboard say how many per country.
- The TimesFM driver forecasts have to be rerun for the larger panel.
- Rebuilding needs about 2.6 GB of disk and the HDX links in `src/world_countries.csv` as of 24 November 2026. If HDX moves a file, the download script reports it and that country's input stays missing.

## Action items

1. [ ] `src/download_world.sh` fetches every input.
2. [ ] Region matching with tests, plus `results/coverage.json`.
3. [ ] Generalise `build_dataset.py`, `conflict.py` and `timesfm_drivers.py` to the country list.
4. [ ] Retrain, extend `verify.py` (raw spot checks for several countries, coverage file), and rewrite the README from `metrics.json`.
5. [ ] World map dashboard.
