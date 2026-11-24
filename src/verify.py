"""Independent accuracy audit. Fails loudly if the data or the reported numbers are wrong.

Checks
  1. Raw-data spot check: region shares in panel.csv are recomputed straight from the FEWS NET files.
  2. No leakage: every feature is dated at or before the cutoff (target - 4 months).
  3. Sanity: ranges, duplicates, test rows have observed targets.
  4. Metrics: results/metrics.json is recomputed from results/predictions.csv, and the numbers
     quoted in README.md match it.

Run:  python3 src/verify.py   (after build_dataset.py and train_model.py)
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train_model import CRISIS, TEST_FROM, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
fails, passes = [], 0


def check(name, ok, detail=""):
    global passes
    if ok:
        passes += 1
        print(f"  ok    {name}")
    else:
        fails.append(name)
        print(f"  FAIL  {name} {detail}")


panel = pd.read_csv(ROOT / "data/processed/panel.csv")
pred = pd.read_csv(ROOT / "results/predictions.csv")
metrics = json.load(open(ROOT / "results/metrics.json"))

print("1. raw-data spot check")
for cc, region_col in [("ET", None), ("SO", None), ("SS", None)]:
    raw = pd.read_csv(ROOT / f"data/raw/fews_ipc_{cc}.csv", low_memory=False, usecols=[
        "geographic_unit_full_name", "unit_type", "fnid", "scenario", "preference_rating",
        "projection_start", "value"])
    raw = raw[(raw.scenario == "CS") & raw.unit_type.isin(["fsc_admin", "fsc_admin_lhz"])
              & raw.value.between(1, 5)].copy()
    raw["region"] = raw.geographic_unit_full_name.str.split(", ").str[-2]
    raw["date"] = pd.to_datetime(raw.projection_start).dt.to_period("M").astype(str)
    raw = raw.sort_values("preference_rating", ascending=False).drop_duplicates(["fnid", "date"])
    # compare only regions whose name needs no alias, so the check is independent of build code
    # SNNPR is skipped: the build intentionally merges the regions that split from it
    lab = panel[(panel.cc == cc) & panel.y_share3.notna() & (panel.region != "SNNPR")]
    shares = raw.assign(c=raw.value >= 3).groupby(["region", "date"]).c.mean()
    merged = lab.merge(shares.rename("raw").reset_index(), left_on=["region", "target_date"],
                       right_on=["region", "date"])
    check(f"{cc}: {len(merged)} region-dates re-derived from raw, share matches",
          len(merged) > 50 and np.allclose(merged.y_share3, merged.raw, atol=1e-9),
          f"max diff {(merged.y_share3 - merged.raw).abs().max():.4f}" if len(merged) else "no overlap")

print("2. leakage")
t = pd.PeriodIndex(panel.target_date, freq="M")
k = pd.PeriodIndex(panel.cutoff, freq="M")
check("cutoff is exactly target - 4 months", ((t - k).map(lambda x: x.n) == 4).all())
check("months_since_ipc is never negative", (panel.months_since_ipc.dropna() >= 0).all())
last = panel.dropna(subset=["months_since_ipc"])
last_date = pd.PeriodIndex(last.cutoff, freq="M") - last.months_since_ipc.astype(int).values
check("last assessment used is not after the cutoff", (last_date <= pd.PeriodIndex(last.cutoff, freq="M")).all())
dk = panel.dropna(subset=["months_since_ipc_dark"])
check("dark scenario: newest assessment used is at least 12 months before the cutoff",
      (dk.months_since_ipc_dark >= 12).all())

print("2b. conflict data (UCDP)")
from conflict import load_events  # noqa: E402
ev = load_events()
# compare polygon placement with UCDP's own adm_1 text where that text is an unambiguous current region
raw = pd.concat(pd.read_csv(ROOT / "data/raw/ucdp" / f, low_memory=False, usecols=["id", "adm_1"])
                for f in ["GEDEvent_v26_1.csv", "GEDEvent_v26_01_26_09.csv"]).drop_duplicates("id")
ev = ev.merge(raw, on="id")
probe = {"Tigray state": "Tigray", "Amhara state": "Amhara", "Afar state": "Afar", "Jonglei state": "Jonglei",
         "Unity state": "Unity", "Upper Nile state": "Upper Nile", "Banaadir region": "Banadir",
         "Gedo region": "Gedo", "Bay region": "Bay"}
sub = ev[ev.adm_1.isin(probe)]
agree = (sub.region == sub.adm_1.map(probe)).mean()
check(f"events placed by coordinates agree with UCDP region text ({len(sub)} events, {agree:.1%})",
      len(sub) > 1000 and agree > 0.95)
cm = pd.read_csv(ROOT / "data/processed/conflict_monthly.csv")
row = panel[(panel.region == "Tigray") & (panel.target_date == "2021-02")].iloc[0]  # cutoff 2020-10
w = cm[(cm.region == "Tigray") & cm.date.between("2020-08", "2020-10")]
check("conflict feature uses only months up to the cutoff (Tigray, war began Nov 2020)",
      np.isclose(row.conf_events3, np.log1p(w.events.sum())) and w.events.sum() < 5)

print("2c. TimesFM driver forecasts")
drv = pd.read_csv(ROOT / "data/processed/timesfm_drivers.csv")
check("one TimesFM driver row per panel row",
      len(drv) == len(panel) and not drv.duplicated(["cc", "region", "target_date"]).any())
for var in ("r3q", "price_yoy"):
    d_ = drv.dropna(subset=[f"ctx_end_{var}"])
    ends = pd.PeriodIndex(d_[f"ctx_end_{var}"], freq="M")
    check(f"TimesFM {var} context never ends after the cutoff ({len(d_)} forecasts)",
          len(d_) > 1000 and (ends <= pd.PeriodIndex(d_.cutoff, freq="M")).all())
    check(f"every TimesFM {var} forecast has a last-value baseline",
          d_[f"naive_{var}"].notna().all() and d_[f"tfm_{var}"].notna().all())

print("3. sanity")
check("no duplicate region/date rows", not panel.duplicated(["region", "target_date"]).any())
for c in ["y_share3", "last_share3", "last_share3_dark", "fews_proj_share3"]:
    s_ = panel[c].dropna()
    check(f"{c} within 0..1", s_.between(0, 1).all())
pcols = [c for c in pred.columns if c.startswith("pred_")]
check("all predictions within 0..1", pred[pcols].stack().dropna().between(0, 1).all())
test = pred[(pred.target_date >= TEST_FROM) & pred.y_share3.notna()]
check(f"test set: {len(test)} rows, all with forecasts", len(test) == metrics["all"]["full"]["n"]
      and test[["pred_full", "pred_dark"]].notna().all().all())

print("4. metrics reproduce")
cols = {"persistence": "pred_persist", "full": "pred_full", "full_conf": "pred_full_conf",
        "satellite": "pred_sat", "remote": "pred_remote", "fews": "fews_proj_share3",
        "full_news": "pred_full_news", "remote_news": "pred_remote_news",
        "full_tfm": "pred_full_tfm", "remote_tfm": "pred_remote_tfm"}
check("TimesFM variants are reported", {"full_tfm", "remote_tfm"} <= set(metrics["all"])
      and "dark_tfm" in metrics["dark"])
for key, col in cols.items():
    if key in metrics["all"]:
        check(f"'{key}' recomputed from predictions.csv", score(test, col) == metrics["all"][key])
dcols = {"persistence_dark": "pred_persist_dark", "dark": "pred_dark", "dark_conf": "pred_dark_conf",
         "dark_news": "pred_dark_news", "dark_tfm": "pred_dark_tfm"}
for key, col in dcols.items():
    if key in metrics["dark"]:
        check(f"dark '{key}' recomputed from predictions.csv",
              score(test, col, "last_share3_dark") == metrics["dark"][key])
for var in ("r3q", "price_yoy"):
    d_ = test.dropna(subset=[f"tfm_{var}", f"naive_{var}", f"obs_{var}"])
    got = {"n": int(len(d_)),
           "mae_timesfm": round(float((d_[f"tfm_{var}"] - d_[f"obs_{var}"]).abs().mean()), 4),
           "mae_last_value": round(float((d_[f"naive_{var}"] - d_[f"obs_{var}"]).abs().mean()), 4)}
    if var == "price_yoy":
        got["n_national"] = int((d_.tfm_price_is_national == 1).sum())
        got["n_national_test"] = int((test.tfm_price_is_national == 1).sum())
    check(f"driver check '{var}' recomputed from predictions.csv",
          got == metrics.get("driver_check", {}).get(var), f"got={got}")

def pct(x):
    """Half-up percent of a 3-decimal metric, as the dashboard shows it (0.685 -> 69, not float round's 68)."""
    return int((Decimal(str(x)) * 100).quantize(Decimal(1), ROUND_HALF_UP))


readme = (ROOT / "README.md").read_text()
n_rows = 0
for line in readme.splitlines():
    m_ = re.match(r"\| .*<!-- (\w+)\.(\w+) --> *\| *(\d+)% *\| *(\d+) of (\d+) *\| *(\d+) points", line)
    if not m_:
        continue
    n_rows += 1
    scope, key = m_.group(1), m_.group(2)
    m = metrics[scope][key]
    ok = (int(m_.group(3)) == pct(m["accuracy"]) and int(m_.group(4)) == m["onsets_caught"]
          and int(m_.group(5)) == m["onsets"] and int(m_.group(6)) == pct(m["mae"]))
    check(f"README row {scope}.{key} matches metrics.json", ok, f"metrics={m}")
check("README results tables were found and checked", n_rows >= 6, f"rows={n_rows}")

print(f"\n{passes} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
