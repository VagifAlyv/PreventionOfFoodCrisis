"""Train and honestly evaluate the early-warning model, then export data for the dashboard.

Walk-forward evaluation: to forecast assessment date T, the model is trained only on
assessments whose outcome was already known at the information cutoff (T - 4 months).

Run:  python3 src/train_model.py   (after build_dataset.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TEST_FROM = "2022-01"
CRISIS = 0.5  # a region counts as "in crisis" when at least half its units are IPC 3+

SAT = ["r1q", "r3q", "r3q_min12", "r3q_mean6", "price_yoy", "price_3m", "month", "country_id"]
IPC = ["last_share3", "last_mean_phase", "last_max_phase", "months_since_ipc", "ipc_trend"]
IPC_DARK = [f + "_dark" for f in IPC]
CONF = ["conf_deaths3", "conf_events3", "conf_change"]
NEWS = ["news_food_share3", "news_food_log3", "news_food_change"]
# Fixed before looking at test results; every variant is reported, not just the best one.
# name: (features, column holding the forecast)
VARIANTS = {
    "full": SAT + IPC, "full_conf": SAT + IPC + CONF,
    "satellite": SAT, "remote": SAT + CONF,
    "dark": SAT + IPC_DARK, "dark_conf": SAT + IPC_DARK + CONF,
    "full_news": SAT + IPC + CONF + NEWS, "remote_news": SAT + CONF + NEWS,
    "dark_news": SAT + IPC_DARK + CONF + NEWS,
}
LABELS = {
    "r1q": "Rainfall, last month (% of normal)", "r3q": "Rainfall, last 3 months (% of normal)",
    "r3q_min12": "Driest 3-month spell in past year", "r3q_mean6": "Rainfall, 6-month average anomaly",
    "price_yoy": "Cereal price change, 12 months", "price_3m": "Cereal price change, 3 months",
    "month": "Season (month forecast)", "country_id": "Country",
    "last_share3": "Last known share of area in crisis", "last_mean_phase": "Last known average IPC phase",
    "last_max_phase": "Last known worst IPC phase", "months_since_ipc": "Months since last assessment",
    "ipc_trend": "Change between last two assessments",
    "conf_deaths3": "Conflict deaths, last 3 months (UCDP)", "conf_events3": "Conflict events, last 3 months (UCDP)",
    "conf_change": "Conflict events vs previous year",
    "news_food_share3": "Share of local news about hunger/drought, 3 months (GDELT)",
    "news_food_log3": "Hunger/drought news volume, 3 months (GDELT)",
    "news_food_change": "Hunger/drought news vs previous year (GDELT)",
}
LABELS.update({f + "_dark": LABELS[f] + " (12+ months old)" for f in IPC})


def model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4,
                                         min_samples_leaf=15, l2_regularization=1.0, random_state=0)


def score(d, col, last="last_share3"):
    d = d.dropna(subset=[col, "y_share3"])
    y, p = d.y_share3 >= CRISIS, d[col] >= CRISIS
    was = d[last] >= CRISIS
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    onset, recov = y & ~was, ~y & was
    f = lambda a, b: round(float(a) / float(b), 3) if b else None
    return {
        "n": int(len(d)), "mae": round(float((d[col] - d.y_share3).abs().mean()), 3),
        "accuracy": f((y == p).sum(), len(d)), "precision": f(tp, tp + fp), "recall": f(tp, tp + fn),
        "f1": f(2 * tp, 2 * tp + fp + fn),
        "onsets": int(onset.sum()), "onsets_caught": int((onset & p).sum()),
        "recoveries": int(recov.sum()), "recoveries_caught": int((recov & ~p).sum()),
    }


import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))


def shapes():
    """Region outlines for the map (geoBoundaries ADM1), coordinates rounded to 0.02 degrees."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from conflict import GEO, NAMES
    out = []
    for cc, iso in GEO.items():
        g = json.load(open(ROOT / "data" / "raw" / "boundaries" / f"{iso}-ADM1.geojson"))
        for f in g["features"]:
            geom = f["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            rings = []
            for poly in polys:
                ring, prev = [], None
                for x, y in poly[0]:
                    pt = (round(x * 50) / 50, round(y * 50) / 50)
                    if pt != prev:
                        ring.append(pt)
                    prev = pt
                if len(ring) >= 4:
                    rings.append(ring)
            name = NAMES.get(f["properties"]["shapeName"], f["properties"]["shapeName"])
            out.append({"cc": cc, "region": name, "rings": rings})
    return out


def main():
    df = pd.read_csv(PROC / "panel.csv")
    df["country_id"] = df.cc.map({"ET": 0, "SO": 1, "SS": 2})
    df["t"] = pd.PeriodIndex(df.target_date, freq="M")
    df["k"] = pd.PeriodIndex(df.cutoff, freq="M")
    labelled = df.dropna(subset=["y_share3"])
    has_news = df[NEWS].notna().any().any()
    variants = {v: f for v, f in VARIANTS.items() if has_news or not set(f) & set(NEWS)}

    for v in VARIANTS:  # columns always exist; news ones stay empty until news data is downloaded
        df["pred_" + v] = np.nan
    for T in sorted(df[df.target_date >= TEST_FROM].t.unique()):
        idx = df.index[df.t == T]
        k = df.loc[idx[0], "k"]
        train = labelled[labelled.t <= k]  # outcomes known at the cutoff
        for v, feats in variants.items():
            tr = train
            if v.startswith("dark"):  # the dark model can only learn from outcomes known DARK months earlier
                tr = labelled[labelled.t <= k - 12]
            m = model().fit(tr[feats], tr.y_share3)
            df.loc[idx, "pred_" + v] = np.clip(m.predict(df.loc[idx, feats]), 0, 1)
    df["pred_persist"] = df.last_share3
    df["pred_persist_dark"] = df.last_share3_dark
    df = df.rename(columns={"pred_satellite": "pred_sat"})

    test = df[(df.target_date >= TEST_FROM) & df.y_share3.notna()]
    cols = {"persistence": "pred_persist", "full": "pred_full", "full_conf": "pred_full_conf",
            "full_news": "pred_full_news", "satellite": "pred_sat", "remote": "pred_remote",
            "remote_news": "pred_remote_news", "fews": "fews_proj_share3"}
    dark_cols = {"persistence_dark": "pred_persist_dark", "dark": "pred_dark", "dark_conf": "pred_dark_conf",
                 "dark_news": "pred_dark_news"}
    if not has_news:
        cols = {k: c for k, c in cols.items() if "news" not in k}
        dark_cols = {k: c for k, c in dark_cols.items() if "news" not in k}
    metrics = {"all": {k: score(test, c) for k, c in cols.items()}}
    metrics["dark"] = {k: score(test, c, "last_share3_dark") for k, c in dark_cols.items()}
    common = test.dropna(subset=list(cols.values()))
    metrics["common"] = {k: score(common, c) for k, c in cols.items()}
    metrics["by_country"] = {c: {k: score(g, col) for k, col in cols.items()}
                             for c, g in test.groupby("country")}
    metrics["dark_by_country"] = {c: {k: score(g, col, "last_share3_dark") for k, col in dark_cols.items()}
                                  for c, g in test.groupby("country")}
    # paired bootstrap over assessment dates: MAE difference vs the matching persistence baseline
    rng = np.random.default_rng(0)
    dates = test.target_date.unique()
    pairs = [("full", "pred_full", "pred_persist"), ("full_conf", "pred_full_conf", "pred_persist"),
             ("remote", "pred_remote", "pred_persist"), ("dark", "pred_dark", "pred_persist_dark"),
             ("dark_conf", "pred_dark_conf", "pred_persist_dark"), ("dark_conf_vs_dark", "pred_dark_conf", "pred_dark"),
             ("full_news", "pred_full_news", "pred_persist"), ("remote_news_vs_remote", "pred_remote_news", "pred_remote"),
             ("dark_news", "pred_dark_news", "pred_persist_dark"), ("dark_news_vs_dark", "pred_dark_news", "pred_dark")]
    pairs = [p for p in pairs if has_news or "news" not in p[0]]
    metrics["mae_diff_ci"] = {}
    for name, a, b in pairs:
        d = test.dropna(subset=[a, b])
        diff = ((d[a] - d.y_share3).abs() - (d[b] - d.y_share3).abs()).groupby(d.target_date)
        s_, n_ = diff.sum(), diff.size()
        boots = [s_[ix].sum() / n_[ix].sum() for ix in (rng.choice(dates, len(dates)) for _ in range(2000))]
        metrics["mae_diff_ci"][name] = {"diff": round(float(s_.sum() / n_.sum()), 3),
                                        "lo": round(float(np.percentile(boots, 2.5)), 3),
                                        "hi": round(float(np.percentile(boots, 97.5)), 3)}

    # what the final model leans on
    final_train = labelled[labelled.target_date < TEST_FROM]
    imp = {}
    imp_sets = [("full", SAT + IPC), ("full_conf", SAT + IPC + CONF),
                ("dark_conf", SAT + IPC_DARK + CONF), ("satellite", SAT)]
    if has_news:
        imp_sets.append(("dark_news", SAT + IPC_DARK + CONF + NEWS))
    for name, feats in imp_sets:
        m = model().fit(final_train[feats], final_train.y_share3)
        pi = permutation_importance(m, test[feats], test.y_share3, n_repeats=20, random_state=0,
                                    scoring="neg_mean_absolute_error")
        imp[name] = sorted(({"feature": LABELS[f], "importance": round(float(v), 4)}
                            for f, v in zip(feats, pi.importances_mean)),
                           key=lambda x: -x["importance"])

    for scope in ("all", "dark"):
        print(f"\n== test set ({scope}) ==")
        print(pd.DataFrame(metrics[scope]).T.to_string())
    print("\nMAE difference vs baseline (negative = better), 95% CI:", metrics["mae_diff_ci"])
    print("\nfull-model drivers:", [(i["feature"], i["importance"]) for i in imp["full"][:5]])

    # worst misses, for the limitations slide
    test = test.assign(err=test.pred_full - test.y_share3)
    misses = (test.reindex(test.err.abs().sort_values(ascending=False).index).head(12)
              [["country", "region", "target_date", "y_share3", "pred_full", "pred_persist",
                "fews_proj_share3", "r3q", "price_yoy"]].round(3))
    print("\nworst misses:\n", misses.to_string(index=False))

    (ROOT / "results").mkdir(exist_ok=True)
    out_cols = ["country", "region", "target_date", "cutoff", "y_share3", "y_mean_phase", "y_max_phase",
                "n_units", "last_share3", "months_since_ipc", "r3q", "r3q_min12", "price_yoy",
                "price_is_national", "fews_proj_share3", "conf_deaths3", "conf_events3", "last_share3_dark",
                "months_since_ipc_dark", "news_food_share3", "pred_full", "pred_full_conf", "pred_full_news",
                "pred_sat", "pred_remote", "pred_remote_news", "pred_dark", "pred_dark_conf", "pred_dark_news",
                "pred_persist", "pred_persist_dark"]
    df[out_cols].round(4).to_csv(ROOT / "results" / "predictions.csv", index=False)
    json.dump(metrics, open(ROOT / "results" / "metrics.json", "w"), indent=1)

    def recs(d):
        return json.loads(d.round(3).to_json(orient="records"))

    mo = pd.read_csv(PROC / "monthly_context.csv")
    cf = pd.read_csv(PROC / "conflict_monthly.csv")[["region", "date", "events", "deaths"]]
    cf = cf[cf.date >= "2015-01"].rename(columns={"events": "conf_events", "deaths": "conf_deaths"})
    mo = mo.merge(cf, on=["region", "date"], how="outer")
    from build_dataset import load_news
    nw = load_news()
    if len(nw):
        nw = nw.assign(date=nw.date.astype(str))[["region", "date", "news_food"]]
        mo = mo.merge(nw[nw.date >= "2015-01"], on=["region", "date"], how="outer")
    else:
        mo["news_food"] = np.nan
    mo["country"] = mo.groupby("region").country.transform("first")
    payload = {
        "generated": pd.Timestamp.today().strftime("%d %B %Y"),
        "lead_months": 4, "crisis_threshold": CRISIS, "test_from": TEST_FROM,
        "train_rows": int(len(final_train)), "test_rows": int(len(test)),
        "metrics": metrics, "importance": imp,
        "rows": recs(df[out_cols]),
        "monthly": recs(mo[["country", "region", "date", "r3q", "price_yoy", "conf_events", "conf_deaths", "news_food"]]),
        "conflict_through": str(cf.date.max()),
        "shapes": shapes(),
        "misses": recs(misses),
    }
    (ROOT / "prototype").mkdir(exist_ok=True)
    (ROOT / "prototype" / "data.js").write_text("window.EWS_DATA = " + json.dumps(payload) + ";\n")
    print("\nwrote results/ and prototype/data.js")


if __name__ == "__main__":
    main()
