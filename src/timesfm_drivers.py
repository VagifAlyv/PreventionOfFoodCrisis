"""Forecast each region's monthly drivers to the target month with TimesFM 2.5 (Google Research).

3-country experiment only (results/three_countries/). It runs against the 3-country build_dataset.py of commit 93b27f6, not the world pipeline.


For every panel row (region, target T, cutoff k = T - 4 months) this forecasts, from data known
at k only:
  tfm_r3q        3-month rainfall as % of normal (CHIRPS via WFP) at T
  tfm_price_yoy  12-month change in cereal prices at T (regional; national if the region has no
                 usable regional series at k, as in build_dataset.py)
It also stores the naive "repeat the last value" forecast and the value actually observed at T,
so the driver forecast itself can be checked.

Model: google/timesfm-2.5-200m-pytorch (Apache-2.0), zero-shot, no covariates.
Run:  python3 src/timesfm_drivers.py   (after build_dataset.py, before train_model.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_LEN, MAX_LEN = 24, 120
RAIN_AGE, PRICE_AGE = 1, 3  # same max ages as build_dataset.as_of


def context(series, cutoff, max_age):
    """Contiguous run of values ending at the newest value dated <= cutoff.

    Returns (values, newest month) or (None, None) if the newest value is older than max_age
    months or the run is shorter than MIN_LEN. At most MAX_LEN values are kept."""
    s = series[series.index <= cutoff].dropna()
    if s.empty or (cutoff - s.index[-1]).n > max_age:
        return None, None
    idx = s.index
    gaps = np.flatnonzero(np.diff(idx.asi8) != 1)
    run = s.iloc[gaps[-1] + 1:] if len(gaps) else s
    if len(run) < MIN_LEN:
        return None, None
    return run.values[-MAX_LEN:].astype(float), run.index[-1]


def _series(table, col):
    return table.set_index("date")[col].sort_index()


def driver_table(panel, rain, preg, pnat, forecast):
    """One row per panel row. `forecast(contexts, horizons)` returns the point forecast at each horizon."""
    rain_by = {r: _series(g, "r3q") for r, g in rain.groupby("region")}
    price_by = {r: _series(g, "price_yoy") for r, g in preg.groupby("region")}
    nat = _series(pnat, "price_yoy")
    rows, jobs = [], []  # jobs: (row index, variable, context, horizon)
    for _, p in panel.iterrows():
        T, k = pd.Period(p.target_date, "M"), pd.Period(p.cutoff, "M")
        row = {"cc": p.cc, "region": p.region, "target_date": p.target_date, "cutoff": p.cutoff}
        sources = [("r3q", [rain_by.get(p.region)], RAIN_AGE),
                   ("price_yoy", [price_by.get(p.region), nat], PRICE_AGE)]
        for var, candidates, age in sources:
            row.update({f"tfm_{var}": np.nan, f"naive_{var}": np.nan, f"obs_{var}": np.nan,
                        f"ctx_end_{var}": None})
            if var == "price_yoy":
                row["tfm_price_is_national"] = np.nan
            for i, s in enumerate(candidates):
                if s is None:
                    continue
                ctx, end = context(s, k, age)
                if ctx is None:
                    continue
                if var == "price_yoy":
                    row["tfm_price_is_national"] = i  # candidates: 0 regional, 1 national
                row[f"naive_{var}"] = ctx[-1]
                row[f"obs_{var}"] = s.get(T, np.nan)
                row[f"ctx_end_{var}"] = str(end)
                jobs.append((len(rows), var, ctx, (T - end).n))
                break
        rows.append(row)
    if jobs:
        preds = forecast([j[2] for j in jobs], [j[3] for j in jobs])
        for (i, var, _, _), v in zip(jobs, preds):
            rows[i][f"tfm_{var}"] = float(v)
    return pd.DataFrame(rows)


def timesfm_forecaster():
    import timesfm
    import torch
    torch.manual_seed(0)
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(max_context=128, max_horizon=128, normalize_inputs=True,
                                     per_core_batch_size=64))

    def forecast(contexts, horizons):
        point, _ = m.forecast(horizon=max(horizons), inputs=contexts)
        return [point[i, h - 1] for i, h in enumerate(horizons)]
    return forecast


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_dataset import COUNTRIES, load_prices, load_rain
    panel = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    fc = timesfm_forecaster()
    out = []
    for cc, (code, _) in COUNTRIES.items():
        preg, pnat = load_prices(code)
        out.append(driver_table(panel[panel.cc == cc], load_rain(code), preg, pnat, fc))
    d = pd.concat(out)
    d.to_csv(ROOT / "data" / "processed" / "timesfm_drivers.csv", index=False)
    print(f"{len(d)} rows; forecasts: r3q {d.tfm_r3q.notna().sum()}, price_yoy {d.tfm_price_yoy.notna().sum()}")


if __name__ == "__main__":
    main()
