"""Build the region-by-assessment training table.

One row = one admin-1 region at one FEWS NET assessment date T.
Target   = share of that region's FEWS NET units in IPC phase 3+ at T.
Features = what was known LEAD months earlier (rainfall, staple prices, last IPC status).

Run:  python3 src/build_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
LEAD = 4  # months between the information cutoff and the date being forecast
DARK = 12  # "monitoring gone dark" scenario: the newest usable assessment is >= DARK months before the cutoff

COUNTRIES = {"ET": ("eth", "Ethiopia"), "SO": ("som", "Somalia"), "SS": ("ssd", "South Sudan")}

# Rainfall files use OCHA p-codes (Ethiopia still on the pre-2020 region list).
PCODES = {
    "ET01": "Tigray", "ET02": "Afar", "ET03": "Amhara", "ET04": "Oromia", "ET05": "Somali",
    "ET06": "Benishangul Gumuz", "ET07": "SNNPR", "ET12": "Gambela", "ET13": "Harari",
    "ET14": "Addis Ababa", "ET15": "Dire Dawa",
    "SO11": "Awdal", "SO12": "Woqooyi Galbeed", "SO13": "Togdheer", "SO14": "Sool",
    "SO15": "Sanaag", "SO16": "Bari", "SO17": "Nugaal", "SO18": "Mudug", "SO19": "Galgaduud",
    "SO20": "Hiraan", "SO21": "Middle Shabelle", "SO22": "Banadir", "SO23": "Lower Shabelle",
    "SO24": "Bay", "SO25": "Bakool", "SO26": "Gedo", "SO27": "Middle Juba", "SO28": "Lower Juba",
    "SS01": "Central Equatoria", "SS02": "Eastern Equatoria", "SS03": "Jonglei", "SS04": "Lakes",
    "SS05": "Northern Bahr el Ghazal", "SS06": "Unity", "SS07": "Upper Nile", "SS08": "Warrap",
    "SS09": "Western Bahr el Ghazal", "SS10": "Western Equatoria",
}
# Spelling differences between FEWS NET, WFP and OCHA. Ethiopia's regions that split
# off SNNPR after 2020 are folded back so the series is continuous.
ALIASES = {
    "Benshangul Gumuz": "Benishangul Gumuz", "B. Gumuz": "Benishangul Gumuz",
    "Sidama": "SNNPR", "South Ethiopia": "SNNPR", "Central Ethiopia": "SNNPR",
    "South West Ethiopia": "SNNPR",
    "Banaadir": "Banadir", "Hiiraan": "Hiraan",
    "Juba Dhexe": "Middle Juba", "Juba Hoose": "Lower Juba",
    "Shabelle Dhexe": "Middle Shabelle", "Shabelle Hoose": "Lower Shabelle",
}
VALID = {cc: {n for p, n in PCODES.items() if p.startswith(cc)} for cc in COUNTRIES}


def month(s):
    return pd.to_datetime(s).dt.to_period("M")


def load_ipc(cc):
    """FEWS NET classifications -> region aggregates for observed (CS) and projected (ML1/ML2)."""
    d = pd.read_csv(RAW / f"fews_ipc_{cc}.csv", low_memory=False, usecols=[
        "geographic_unit_full_name", "unit_type", "fnid", "scenario", "preference_rating",
        "projection_start", "projection_end", "reporting_date", "value"])
    d = d[d.unit_type.isin(["fsc_admin", "fsc_admin_lhz"]) & d.value.between(1, 5)]
    d["region"] = d.geographic_unit_full_name.str.split(", ").str[-2].replace(ALIASES)
    d = d[d.region.isin(VALID[cc])]
    for c in ["projection_start", "projection_end", "reporting_date"]:
        d[c] = month(d[c])
    # several rows per unit and period exist (scale versions); keep FEWS NET's preferred one
    d = (d.sort_values("preference_rating", ascending=False)
           .drop_duplicates(["fnid", "scenario", "reporting_date", "projection_start"]))
    d["crisis"] = d.value >= 3

    cs = (d[d.scenario == "CS"].groupby(["region", "projection_start"])
          .agg(share3=("crisis", "mean"), mean_phase=("value", "mean"),
               max_phase=("value", "max"), n_units=("value", "size")).reset_index()
          .rename(columns={"projection_start": "date"}))
    proj = (d[d.scenario.isin(["ML1", "ML2"])]
            .groupby(["region", "scenario", "reporting_date", "projection_start", "projection_end"])
            .agg(share3=("crisis", "mean")).reset_index())
    return cs, proj


def load_rain(code):
    r = pd.read_csv(RAW / f"{code}-rainfall-subnat-full.csv")
    r = r[r.adm_level == 1].copy()
    r["region"] = r.PCODE.map(PCODES)
    r["date"] = month(r.date)
    # r1q / r3q = 1- and 3-month rainfall as % of the long-term average; keep each month's last dekad
    r = r.dropna(subset=["region"]).sort_values("date").groupby(["region", "date"]).tail(1)
    r = r[["region", "date", "r1q", "r3q"]].sort_values(["region", "date"])
    g = r.groupby("region")
    r["r3q_min12"] = g.r3q.transform(lambda s: s.rolling(12, min_periods=6).min())
    r["r3q_mean6"] = g.r3q.transform(lambda s: s.rolling(6, min_periods=3).mean())
    return r


def load_prices(code):
    """Median year-on-year and 3-month change in local-currency cereal prices per region."""
    p = pd.read_csv(RAW / f"wfp_food_prices_{code}.csv", low_memory=False)
    p = p[(p.category == "cereals and tubers") & (p.priceflag != "forecast") & (p.price > 0)].copy()
    p["region"] = p.admin1.replace(ALIASES)
    p["date"] = month(p.date)
    key = ["region", "market_id", "commodity_id", "unit", "pricetype"]
    p = p.groupby(key + ["date"], as_index=False).price.mean()
    out = []
    for k, s in p.groupby(key):
        s = s.set_index("date").price.resample("M").mean().ffill(limit=2)
        if len(s) < 13:
            continue
        out.append(pd.DataFrame({"region": k[0], "date": s.index,
                                 "price_yoy": np.log(s / s.shift(12)).values,
                                 "price_3m": np.log(s / s.shift(3)).values}))
    f = pd.concat(out).replace([np.inf, -np.inf], np.nan)
    reg = f.groupby(["region", "date"])[["price_yoy", "price_3m"]].median().reset_index()
    nat = f.groupby("date")[["price_yoy", "price_3m"]].median().reset_index()
    return reg, nat


def as_of(table, cutoff, cols, max_age=3):
    """Latest row of a single-region monthly table at or before `cutoff` (not older than max_age months)."""
    t = table[(table.date <= cutoff) & (table.date >= cutoff - max_age)]
    if t.empty:
        return {c: np.nan for c in cols}
    return t.iloc[-1][cols].to_dict()


def conflict_features(c_r, k):
    """Organised violence (UCDP) in the 3 months to the cutoff, and its change against the prior 12."""
    if c_r.empty or (k - c_r.date.max()).n > 1:
        return {"conf_deaths3": np.nan, "conf_events3": np.nan, "conf_change": np.nan}
    k = min(k, c_r.date.max())  # UCDP candidate data lags about a month; allow one month of lag
    w3 = c_r[(c_r.date > k - 3) & (c_r.date <= k)]
    w12 = c_r[(c_r.date > k - 15) & (c_r.date <= k - 3)]
    d3, e3 = w3.deaths.sum(), w3.events.sum()
    return {"conf_deaths3": np.log1p(d3), "conf_events3": np.log1p(e3),
            "conf_change": np.log1p(e3) - np.log1p(w12.events.sum() / 4)}


def load_news():
    """Monthly GDELT article counts per region (see fetch_news.py). Empty if not downloaded."""
    files = sorted((RAW / "gdelt").glob("*_food.csv")) if (RAW / "gdelt").exists() else []
    out = []
    for f in files:
        cc, region = f.stem.split("_", 1)[0], f.stem.split("_", 1)[1].rsplit("_", 1)[0].replace("_", " ")
        a = f.with_name(f.name.replace("_food", "_all"))
        if not a.exists():
            continue
        def monthly(p):
            d = pd.read_csv(p, comment="#")
            d = d[d.Series == "Article Count"]
            return d.assign(date=month(d.Date)).groupby("date").Value.sum()
        m = pd.DataFrame({"news_food": monthly(f), "news_all": monthly(a)}).fillna(0)
        out.append(m.assign(cc=cc, region=region).reset_index())
    return pd.concat(out) if out else pd.DataFrame(columns=["cc", "region", "date", "news_food", "news_all"])


def news_features(n_r, k):
    """Share of a region's coverage about hunger/drought in the 3 months to the cutoff, and its change."""
    nan = {"news_food_share3": np.nan, "news_food_log3": np.nan, "news_food_change": np.nan}
    if n_r.empty or k > n_r.date.max() or k - 15 < n_r.date.min():
        return nan
    w3 = n_r[(n_r.date > k - 3) & (n_r.date <= k)]
    w12 = n_r[(n_r.date > k - 15) & (n_r.date <= k - 3)]
    f3, a3, f12 = w3.news_food.sum(), w3.news_all.sum(), w12.news_food.sum()
    return {"news_food_share3": f3 / a3 if a3 else np.nan, "news_food_log3": np.log1p(f3),
            "news_food_change": np.log1p(f3) - np.log1p(f12 / 4)}


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
    rows, monthly = [], []
    conf = pd.read_csv(OUT / "conflict_monthly.csv")
    conf["date"] = pd.PeriodIndex(conf.date, freq="M")
    news = load_news()
    for cc, (code, cname) in COUNTRIES.items():
        cs, proj = load_ipc(cc)
        rain = load_rain(code)
        preg, pnat = load_prices(code)
        observed = sorted(cs.date.unique())
        # every Feb/Jun/Oct from 2022 on, so the 2025 gap and the next forecast get a row too
        extra = [pd.Period(f"{y}-{m:02d}", "M") for y in range(2022, 2028) for m in (2, 6, 10)
                 if pd.Period(f"{y}-{m:02d}", "M") <= pd.Period("2027-02", "M")]
        targets = sorted(set(observed) | set(extra))

        for region in sorted(VALID[cc]):
            c_r = cs[cs.region == region].sort_values("date")
            r_r = rain[rain.region == region]
            p_r = preg[preg.region == region].sort_values("date")
            pr_r = proj[proj.region == region]
            cf_r = conf[(conf.cc == cc) & (conf.region == region)]
            n_r = news[(news.cc == cc) & (news.region == region)]
            for T in targets:
                k = T - LEAD  # information cutoff
                row = {"country": cname, "cc": cc, "region": region, "target_date": str(T),
                       "cutoff": str(k), "month": T.month}
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
                row.update(conflict_features(cf_r, k))
                row.update(news_features(n_r, k))
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

            # monthly context series for the dashboard charts
            m = r_r[r_r.date >= pd.Period("2015-01", "M")][["date", "r3q"]].merge(
                p_r[["date", "price_yoy"]], on="date", how="left")
            m["country"], m["region"] = cname, region
            monthly.append(m)

    panel = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "panel.csv", index=False)
    mo = pd.concat(monthly)
    mo["date"] = mo.date.astype(str)
    mo.to_csv(OUT / "monthly_context.csv", index=False)

    lab = panel.dropna(subset=["y_share3"])
    print(f"panel: {len(panel)} rows, {len(lab)} with an observed target, "
          f"{panel.region.nunique()} regions")
    print(lab.groupby("country").agg(rows=("region", "size"), first=("target_date", "min"),
                                     last=("target_date", "max"),
                                     crisis_rate=("y_share3", lambda s: (s >= .5).mean())))
    print("missing share by feature:\n", lab.isna().mean().round(3)[lambda s: s > 0])


if __name__ == "__main__":
    build()
