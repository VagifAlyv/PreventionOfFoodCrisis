"""Build the region-by-assessment table for every FEWS NET country.

One row = one admin-1 region (OCHA p-code) at one FEWS NET assessment date T.
Target   = share of that region's FEWS NET units in IPC phase 3+ at T.
Features = what was known LEAD months earlier: rainfall, staple prices, last IPC status.
Region matching follows docs/adr/0001-world-coverage.md; what could not be matched is
listed in results/coverage.json.

Run:  bash src/download_world.sh && python3 src/build_dataset.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regions import norm
from world import WORLD, countries, indexes, load_cod, unit_region

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
LEAD = 4  # months between the information cutoff and the date being forecast
DARK = 12  # "monitoring gone dark": the newest usable assessment is >= DARK months before the cutoff
UNIT_TYPES = ["fsc_admin", "fsc_admin_lhz"]


def month(s):
    return pd.to_datetime(s).dt.to_period("M")


def load_ipc(cc, idx):
    """FEWS NET classifications -> region aggregates for observed (CS) and projected (ML1/ML2).

    Also returns (units matched, units total) for the coverage report."""
    d = pd.read_csv(WORLD / f"fews_ipc_{cc}.csv", low_memory=False, usecols=[
        "geographic_unit_full_name", "unit_type", "fnid", "scenario", "preference_rating",
        "projection_start", "projection_end", "reporting_date", "value"])
    d = d[d.unit_type.isin(UNIT_TYPES) & d.value.between(1, 5)].copy()
    keys = d[["geographic_unit_full_name", "unit_type"]].drop_duplicates().itertuples(index=False)
    lookup = {(n, u): unit_region(n, None, None, idx, zone_first=u == "fsc_admin_lhz") for n, u in keys}
    d["pcode"] = [lookup[k] for k in zip(d.geographic_unit_full_name, d.unit_type)]
    cs_all = d[d.scenario == "CS"]
    seen = (int(cs_all.pcode.notna().sum()), int(len(cs_all)))
    d = d.dropna(subset=["pcode"])
    for c in ["projection_start", "projection_end", "reporting_date"]:
        d[c] = month(d[c])
    # several rows per unit and period exist (scale versions); keep FEWS NET's preferred one
    d = (d.sort_values("preference_rating", ascending=False)
           .drop_duplicates(["fnid", "scenario", "reporting_date", "projection_start"]))
    d["crisis"] = d.value >= 3
    cs = (d[d.scenario == "CS"].groupby(["pcode", "projection_start"])
          .agg(share3=("crisis", "mean"), mean_phase=("value", "mean"),
               max_phase=("value", "max"), n_units=("value", "size")).reset_index()
          .rename(columns={"projection_start": "date"}))
    proj = (d[d.scenario.isin(["ML1", "ML2"])]
            .groupby(["pcode", "scenario", "reporting_date", "projection_start", "projection_end"])
            .agg(share3=("crisis", "mean")).reset_index())
    return cs, proj, seen


def load_rain(iso3, pcodes):
    f = WORLD / f"{iso3}-rainfall-subnat-full.csv"
    if not f.exists():
        return pd.DataFrame(columns=["pcode", "date", "r1q", "r3q", "r3q_min12", "r3q_mean6"])
    r = pd.read_csv(f, usecols=["date", "adm_level", "PCODE", "r1q", "r3q"])
    r = r[(r.adm_level == 1) & r.PCODE.astype(str).isin(pcodes)].rename(columns={"PCODE": "pcode"})
    r["pcode"] = r.pcode.astype(str)
    r["date"] = month(r.date)
    # r1q / r3q = 1- and 3-month rainfall as % of the long-term average; keep each month's last dekad
    r = r.sort_values("date").groupby(["pcode", "date"]).tail(1)
    r = r[["pcode", "date", "r1q", "r3q"]].sort_values(["pcode", "date"])
    g = r.groupby("pcode")
    r["r3q_min12"] = g.r3q.transform(lambda s: s.rolling(12, min_periods=6).min())
    r["r3q_mean6"] = g.r3q.transform(lambda s: s.rolling(6, min_periods=3).mean())
    return r


def load_prices(iso3, i1):
    """Median year-on-year and 3-month change in local-currency cereal prices per region, and national."""
    empty = pd.DataFrame(columns=["pcode", "date", "price_yoy", "price_3m"])
    f = WORLD / f"wfp_food_prices_{iso3}.csv"
    if not f.exists():
        return empty, empty.drop(columns="pcode"), (0, 0)
    p = pd.read_csv(f, low_memory=False, skiprows=[1])  # row 2 is an HXL tag row
    p = p[(p.category == "cereals and tubers") & (p.priceflag != "forecast")].copy()
    p["price"] = pd.to_numeric(p.price, errors="coerce")
    p = p[p.price > 0]
    p["pcode"] = p.admin1.map(lambda a: i1.get(norm(a)) if pd.notna(a) else None)
    seen = (int(p.pcode.notna().sum()), int(len(p)))
    p["date"] = month(p.date)
    key = ["pcode", "market_id", "commodity_id", "unit", "pricetype"]
    p["pcode"] = p.pcode.fillna("_none")  # unmatched markets still count for the national median
    p = p.groupby(key + ["date"], as_index=False).price.mean()
    out = []
    for k, s in p.groupby(key):
        s = s.set_index("date").price.resample("M").mean().ffill(limit=2)
        if len(s) < 13:
            continue
        out.append(pd.DataFrame({"pcode": k[0], "date": s.index,
                                 "price_yoy": np.log(s / s.shift(12)).values,
                                 "price_3m": np.log(s / s.shift(3)).values}))
    if not out:
        return empty, empty.drop(columns="pcode"), seen
    f_ = pd.concat(out).replace([np.inf, -np.inf], np.nan)
    reg = f_[f_.pcode != "_none"].groupby(["pcode", "date"])[["price_yoy", "price_3m"]].median().reset_index()
    nat = f_.groupby("date")[["price_yoy", "price_3m"]].median().reset_index()
    return reg, nat, seen


def as_of(table, cutoff, cols, max_age=3):
    """Latest row of a single-region monthly table at or before `cutoff` (not older than max_age months)."""
    t = table[(table.date <= cutoff) & (table.date >= cutoff - max_age)]
    if t.empty:
        return {c: np.nan for c in cols}
    return t.iloc[-1][cols].to_dict()


def ipc_features(c_r, k, suffix=""):
    past = c_r[c_r.date <= k]
    if not len(past):
        return {}
    last = past.iloc[-1]
    out = {"last_share3": last.share3, "last_mean_phase": last.mean_phase,
           "last_max_phase": last.max_phase, "months_since_ipc": (k - last.date).n}
    if len(past) > 1:
        out["ipc_trend"] = last.share3 - past.iloc[-2].share3
    return {f + suffix: v for f, v in out.items()}


def build():
    rows, monthly, coverage = [], [], {}
    extra = [pd.Period(f"{y}-{m:02d}", "M") for y in range(2022, 2028) for m in (2, 6, 10)
             if pd.Period(f"{y}-{m:02d}", "M") <= pd.Period("2027-02", "M")]
    for c in countries().itertuples():
        adm1, adm2 = load_cod(c.iso3)
        if adm1 is None:
            coverage[c.cc] = {"country": c.country, "included": False, "reason": "no OCHA admin-boundary table on HDX"}
            continue
        idx = indexes(adm1, adm2)
        cs, proj, ipc_seen = load_ipc(c.cc, idx)
        rain = load_rain(c.iso3, set(adm1.pcode))
        preg, pnat, price_seen = load_prices(c.iso3, idx[0])
        info = adm1.set_index("pcode")
        regions = sorted(cs.pcode.unique())
        coverage[c.cc] = {
            "country": c.country, "included": bool(regions),
            "fews_units_matched": ipc_seen[0], "fews_units_total": ipc_seen[1],
            "regions": len(regions), "regions_in_ocha_table": int(len(adm1)),
            "regions_with_rainfall": int(len(set(regions) & set(rain.pcode))),
            "regions_with_prices": int(len(set(regions) & set(preg.pcode))),
            "price_rows_matched": price_seen[0], "price_rows_total": price_seen[1],
            "first_assessment": str(cs.date.min()) if len(cs) else None,
            "last_assessment": str(cs.date.max()) if len(cs) else None,
        }
        if not regions:
            coverage[c.cc]["reason"] = "no FEWS NET unit matched an OCHA region name"
            continue
        targets = sorted(set(cs.date.unique()) | set(extra))
        for pc in regions:
            c_r = cs[cs.pcode == pc].sort_values("date")
            r_r = rain[rain.pcode == pc]
            p_r = preg[preg.pcode == pc].sort_values("date")
            pr_r = proj[proj.pcode == pc]
            name = info.loc[pc, "name"]
            for T in targets:
                if T < c_r.date.min():
                    continue  # nothing to learn from before FEWS NET covered this region
                k = T - LEAD  # information cutoff
                row = {"country": c.country, "cc": c.cc, "pcode": pc, "region": name,
                       "target_date": str(T), "cutoff": str(k), "month": T.month}
                obs = c_r[c_r.date == T]
                if len(obs):
                    row.update(y_share3=obs.share3.iloc[0], y_mean_phase=obs.mean_phase.iloc[0],
                               y_max_phase=obs.max_phase.iloc[0], n_units=obs.n_units.iloc[0])
                row.update(ipc_features(c_r, k))
                # dark scenario: pretend nothing was assessed in the DARK months before the cutoff
                dark = ipc_features(c_r, k - DARK, "_dark")
                if "months_since_ipc_dark" in dark:
                    dark["months_since_ipc_dark"] += DARK
                row.update(dark)
                row.update(as_of(r_r, k, ["r1q", "r3q", "r3q_min12", "r3q_mean6"], max_age=1))
                pv = as_of(p_r, k, ["price_yoy", "price_3m"])
                if np.isnan(pv["price_yoy"]):  # no market in this region -> national median
                    pv = as_of(pnat, k, ["price_yoy", "price_3m"])
                    row["price_is_national"] = 1
                else:
                    row["price_is_national"] = 0
                row.update(pv)
                # FEWS NET's own projection for T, issued no later than the cutoff
                fp = pr_r[(pr_r.reporting_date <= k) & (pr_r.projection_start <= T)
                          & (pr_r.projection_end >= T)]
                if len(fp):
                    fp = fp.sort_values(["reporting_date", "scenario"], ascending=[False, True])
                    row["fews_proj_share3"] = fp.share3.iloc[0]
                rows.append(row)

            m = r_r[r_r.date >= pd.Period("2015-01", "M")][["date", "r3q"]]
            m = (m.merge(p_r[["date", "price_yoy"]], on="date", how="left") if len(p_r)
                 else m.assign(price_yoy=np.nan))
            m["cc"], m["pcode"] = c.cc, pc
            monthly.append(m)
        print(f"{c.cc} {c.country}: {len(regions)} regions, "
              f"{ipc_seen[0] / max(1, ipc_seen[1]):.0%} of FEWS NET unit-assessments matched")

    panel = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "panel.csv", index=False)
    mo = pd.concat(monthly)
    mo["date"] = mo.date.astype(str)
    mo.round(4).to_csv(OUT / "monthly_context.csv", index=False)
    regions = pd.concat([load_cod(c.iso3)[0].assign(cc=c.cc) for c in countries().itertuples()
                         if load_cod(c.iso3)[0] is not None])
    regions[regions.pcode.isin(panel.pcode)][["cc", "pcode", "name", "lat", "lon"]].round(3).to_csv(
        OUT / "regions.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(coverage, open(ROOT / "results" / "coverage.json", "w"), indent=1, ensure_ascii=False)

    lab = panel.dropna(subset=["y_share3"])
    print(f"\npanel: {len(panel)} rows, {len(lab)} with an observed target, "
          f"{panel.pcode.nunique()} regions in {panel.cc.nunique()} countries")
    print("missing share by feature:\n", lab.isna().mean().round(3)[lambda s: s > 0])


if __name__ == "__main__":
    build()
