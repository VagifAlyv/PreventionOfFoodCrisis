"""Train and evaluate the early-warning model on every FEWS NET country, then export the dashboard data.

Walk-forward evaluation: to forecast assessment date T, the model is trained only on
assessments whose outcome was already known at the information cutoff (T - 4 months).
Variants, baselines and comparisons were fixed before the world results were seen
(docs/adr/0001-world-coverage.md). Earlier 3-country results: results/three_countries/.

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
VARIANTS = {"full": SAT + IPC, "satellite": SAT, "dark": SAT + IPC_DARK}
COLS = {"persistence": "pred_persist", "full": "pred_full", "satellite": "pred_sat", "fews": "fews_proj_share3"}
DARK_COLS = {"persistence_dark": "pred_persist_dark", "dark": "pred_dark"}
PAIRS = [("full", "pred_full", "pred_persist"), ("satellite", "pred_sat", "pred_persist"),
         ("dark", "pred_dark", "pred_persist_dark")]
LABELS = {
    "r1q": "Rainfall, last month (% of normal)", "r3q": "Rainfall, last 3 months (% of normal)",
    "r3q_min12": "Driest 3-month spell in past year", "r3q_mean6": "Rainfall, 6-month average anomaly",
    "price_yoy": "Cereal price change, 12 months", "price_3m": "Cereal price change, 3 months",
    "month": "Season (month forecast)", "country_id": "Country",
    "last_share3": "Last known share of area in crisis", "last_mean_phase": "Last known average IPC phase",
    "last_max_phase": "Last known worst IPC phase", "months_since_ipc": "Months since last assessment",
    "ipc_trend": "Change between last two assessments",
}
LABELS.update({f + "_dark": LABELS[f] + " (12+ months old)" for f in IPC})


def model(feats):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=4,
                                         min_samples_leaf=15, l2_regularization=1.0, random_state=0,
                                         categorical_features=[f == "country_id" for f in feats])


def score(d, col, last="last_share3"):
    d = d.dropna(subset=[col, "y_share3"])
    y, p = d.y_share3 >= CRISIS, d[col] >= CRISIS
    was = d[last] >= CRISIS
    tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
    onset, recov = y & ~was, ~y & was
    f = lambda a, b: round(float(a) / float(b), 3) if b else None  # noqa: E731
    return {
        "n": int(len(d)), "mae": round(float((d[col] - d.y_share3).abs().mean()), 3),
        "accuracy": f((y == p).sum(), len(d)), "precision": f(tp, tp + fp), "recall": f(tp, tp + fn),
        "f1": f(2 * tp, 2 * tp + fp + fn),
        "onsets": int(onset.sum()), "onsets_caught": int((onset & p).sum()),
        "recoveries": int(recov.sum()), "recoveries_caught": int((recov & ~p).sum()),
    }


def world_shapes():
    """Natural Earth 1:110m country outlines, coordinates rounded to 0.1 degree."""
    g = json.load(open(ROOT / "data" / "raw" / "world" / "ne_110m_admin_0_countries.geojson"))
    out = []
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = []
        for poly in polys:
            ring, prev = [], None
            for x, y in poly[0]:
                pt = (round(x, 1), round(y, 1))
                if pt != prev:
                    ring.append(pt)
                prev = pt
            if len(ring) >= 4:
                rings.append(ring)
        out.append({"iso2": f["properties"]["ISO_A2_EH"], "name": f["properties"]["NAME"], "rings": rings})
    return out


def main():
    df = pd.read_csv(PROC / "panel.csv")
    ccs = sorted(df.cc.unique())
    df["country_id"] = df.cc.map({c: i for i, c in enumerate(ccs)})
    df["t"] = pd.PeriodIndex(df.target_date, freq="M")
    df["k"] = pd.PeriodIndex(df.cutoff, freq="M")
    labelled = df.dropna(subset=["y_share3"])

    for v in VARIANTS:
        df["pred_" + v] = np.nan
    for T in sorted(df[df.target_date >= TEST_FROM].t.unique()):
        idx = df.index[df.t == T]
        k = T - 4
        for v, feats in VARIANTS.items():
            # outcomes known at the cutoff; the dark model only those known 12 months earlier
            tr = labelled[labelled.t <= (k - 12 if v == "dark" else k)]
            m = model(feats).fit(tr[feats], tr.y_share3)
            df.loc[idx, "pred_" + v] = np.clip(m.predict(df.loc[idx, feats]), 0, 1)
        print("forecast", T, flush=True)
    df["pred_persist"] = df.last_share3
    df["pred_persist_dark"] = df.last_share3_dark
    df = df.rename(columns={"pred_satellite": "pred_sat"})
    # score exactly the values written to predictions.csv (4 decimals), so verify.py reproduces them
    scored = [c for c in df if c.startswith("pred_")] + ["y_share3", "last_share3", "last_share3_dark", "fews_proj_share3"]
    df[scored] = df[scored].round(4)

    test = df[(df.target_date >= TEST_FROM) & df.y_share3.notna()]
    metrics = {"all": {k: score(test, c) for k, c in COLS.items()},
               "dark": {k: score(test, c, "last_share3_dark") for k, c in DARK_COLS.items()}}
    # like-for-like: only cases where the baseline exists (a region's first-ever assessment has no
    # earlier one to repeat). Added after the first world run showed different case counts.
    metrics["paired"] = {k: score(test[test.last_share3.notna()], c) for k, c in COLS.items()}
    metrics["dark_paired"] = {k: score(test[test.last_share3_dark.notna()], c, "last_share3_dark")
                              for k, c in DARK_COLS.items()}
    common = test.dropna(subset=list(COLS.values()))
    metrics["common"] = {k: score(common, c) for k, c in COLS.items()}
    metrics["by_country"] = {c: {k: score(g, col) for k, col in COLS.items()} for c, g in test.groupby("country")}
    metrics["dark_by_country"] = {c: {k: score(g, col, "last_share3_dark") for k, col in DARK_COLS.items()}
                                  for c, g in test.groupby("country")}
    # paired bootstrap over assessment dates: MAE difference vs the matching persistence baseline
    rng = np.random.default_rng(0)
    dates = test.target_date.unique()
    metrics["mae_diff_ci"] = {}
    for name, a, b in PAIRS:
        d = test.dropna(subset=[a, b])
        diff = ((d[a] - d.y_share3).abs() - (d[b] - d.y_share3).abs()).groupby(d.target_date)
        s_, n_ = diff.sum(), diff.size()
        boots = [s_[ix].sum() / n_[ix].sum() for ix in (rng.choice(s_.index, len(s_)) for _ in range(2000))]
        metrics["mae_diff_ci"][name] = {"diff": round(float(s_.sum() / n_.sum()), 3),
                                        "lo": round(float(np.percentile(boots, 2.5)), 3),
                                        "hi": round(float(np.percentile(boots, 97.5)), 3)}
    metrics["test_countries"] = int(test.cc.nunique())
    metrics["test_regions"] = int(test.pcode.nunique())

    final_train = labelled[labelled.target_date < TEST_FROM]
    imp = {}
    for name, feats in VARIANTS.items():
        m = model(feats).fit(final_train[feats], final_train.y_share3)
        pi = permutation_importance(m, test[feats], test.y_share3, n_repeats=10, random_state=0,
                                    scoring="neg_mean_absolute_error")
        imp[name] = sorted(({"feature": LABELS[f], "importance": round(float(v), 4)}
                            for f, v in zip(feats, pi.importances_mean)), key=lambda x: -x["importance"])

    for scope in ("all", "dark"):
        print(f"\n== test set ({scope}) ==")
        print(pd.DataFrame(metrics[scope]).T.to_string())
    print("\nMAE difference vs baseline (negative = better), 95% CI:", metrics["mae_diff_ci"])

    test = test.assign(err=test.pred_full - test.y_share3)
    misses = (test.reindex(test.err.abs().sort_values(ascending=False).index).head(12)
              [["country", "region", "target_date", "y_share3", "pred_full", "pred_persist", "fews_proj_share3"]].round(3))

    out_cols = ["country", "cc", "pcode", "region", "target_date", "cutoff", "y_share3", "n_units",
                "last_share3", "months_since_ipc", "r3q", "price_yoy", "price_is_national", "fews_proj_share3",
                "last_share3_dark", "months_since_ipc_dark", "pred_full", "pred_sat", "pred_dark",
                "pred_persist", "pred_persist_dark"]
    df[out_cols].round(4).to_csv(ROOT / "results" / "predictions.csv", index=False)
    json.dump(metrics, open(ROOT / "results" / "metrics.json", "w"), indent=1, ensure_ascii=False)

    def recs(d):
        return json.loads(d.round(3).to_json(orient="records"))

    mo = pd.read_csv(PROC / "monthly_context.csv")
    mo = mo[mo.date >= "2018-01"].assign(r3q=lambda d: d.r3q.round(1))[["pcode", "date", "r3q", "price_yoy"]]
    regions = pd.read_csv(PROC / "regions.csv")
    current = pd.read_csv(ROOT / "data" / "raw" / "world" / "fews_ipc_current.csv", low_memory=False,
                          usecols=["country", "country_code"]).drop_duplicates()
    payload = {
        "generated": pd.Timestamp.today().strftime("%d %B %Y"),
        "lead_months": 4, "crisis_threshold": CRISIS, "test_from": TEST_FROM,
        "train_rows": int(len(final_train)), "test_rows": int(len(test)),
        "metrics": metrics, "importance": imp,
        "coverage": json.load(open(ROOT / "results" / "coverage.json")),
        "fews_countries": current.rename(columns={"country_code": "cc"}).to_dict("records"),
        "rows": recs(df[df.target_date >= "2018-01"][["country", "cc", "pcode", "region", "target_date", "cutoff", "y_share3",
                                                        "last_share3", "last_share3_dark", "fews_proj_share3", "pred_full",
                                                        "pred_sat", "pred_dark", "pred_persist", "pred_persist_dark"]]),
        "regions": recs(regions),
        "monthly": recs(mo),
        "world": world_shapes(),
        "misses": recs(misses),
    }
    (ROOT / "prototype" / "data.js").write_text("window.EWS_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print("\nwrote results/ and prototype/data.js")


if __name__ == "__main__":
    main()
