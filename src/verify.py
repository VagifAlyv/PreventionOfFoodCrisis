"""Independent accuracy audit. Fails loudly if the data or the reported numbers are wrong.

Checks
  1. Raw data: region shares in panel.csv are recomputed straight from each country's FEWS NET
     file with a simpler rule than the build uses (unit name part equals the OCHA admin-1 name).
  2. Coverage: results/coverage.json agrees with the panel.
  3. No leakage: every feature is dated at or before the cutoff (target - 4 months).
  4. Sanity: ranges, duplicates, test rows have forecasts.
  5. Metrics: results/metrics.json is recomputed from results/predictions.csv; the frozen
     3-country results are recomputed from results/three_countries/.
  6. README: every number in the results tables matches the metrics files.

Run:  python3 src/verify.py   (after build_dataset.py and train_model.py)
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train_model import COLS, DARK_COLS, TEST_FROM, score  # noqa: E402
from world import WORLD, countries, load_cod  # noqa: E402

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


def pct(x):
    """Half-up percent of a 3-decimal metric, as the dashboard shows it (0.685 -> 69)."""
    return int((Decimal(str(x)) * 100).quantize(Decimal(1), ROUND_HALF_UP))


panel = pd.read_csv(ROOT / "data/processed/panel.csv")
pred = pd.read_csv(ROOT / "results/predictions.csv")
metrics = json.load(open(ROOT / "results/metrics.json"))
coverage = json.load(open(ROOT / "results/coverage.json"))

print("1. raw-data spot check (every country)")
checked = 0
for c in countries().itertuples():
    adm1, _ = load_cod(c.iso3)
    if adm1 is None:
        continue
    raw = pd.read_csv(WORLD / f"fews_ipc_{c.cc}.csv", low_memory=False, usecols=[
        "geographic_unit_full_name", "unit_type", "fnid", "scenario", "preference_rating", "projection_start", "value"])
    raw = raw[(raw.scenario == "CS") & raw.unit_type.isin(["fsc_admin", "fsc_admin_lhz"]) & raw.value.between(1, 5)].copy()
    # simple rule: second-to-last name part equals the OCHA English admin-1 name, letter for letter
    name_to_pc = dict(zip(adm1.name, adm1.pcode))
    raw["pcode"] = raw.geographic_unit_full_name.str.split(", ").str[-2].map(name_to_pc)
    raw["date"] = pd.to_datetime(raw.projection_start).dt.to_period("M").astype(str)
    raw = raw.sort_values("preference_rating", ascending=False).drop_duplicates(["fnid", "date"])
    simple = (raw.dropna(subset=["pcode"]).assign(c=lambda d: d.value >= 3)
              .groupby(["pcode", "date"]).c.agg(["mean", "size"]).reset_index())
    lab = panel[(panel.cc == c.cc) & panel.y_share3.notna()]
    m = lab.merge(simple, left_on=["pcode", "target_date"], right_on=["pcode", "date"])
    m = m[m["size"] == m.n_units]  # only region-dates where the simple rule found every unit the build used
    if len(m) < 20:
        continue
    checked += 1
    check(f"{c.cc}: {len(m)} region-dates re-derived from raw, share matches",
          np.allclose(m.y_share3, m["mean"], atol=1e-9), f"max diff {(m.y_share3 - m['mean']).abs().max():.4f}")
check(f"raw spot check ran for at least 25 countries ({checked})", checked >= 25)

print("2. coverage")
for cc, v in coverage.items():
    n = panel[panel.cc == cc].pcode.nunique()
    check(f"{cc}: coverage.json regions ({v.get('regions', 0)}) = panel regions ({n})", v.get("regions", 0) == n)
check("every FEWS NET country is either in the panel or has a stated reason",
      all(v["included"] or v.get("reason") for v in coverage.values()) and len(coverage) == len(countries()))

print("3. leakage")
t = pd.PeriodIndex(panel.target_date, freq="M")
k = pd.PeriodIndex(panel.cutoff, freq="M")
check("cutoff is exactly target - 4 months", ((t - k).map(lambda x: x.n) == 4).all())
check("months_since_ipc is never negative", (panel.months_since_ipc.dropna() >= 0).all())
check("dark scenario: newest assessment used is at least 12 months before the cutoff",
      (panel.months_since_ipc_dark.dropna() >= 12).all())

print("4. sanity")
check("no duplicate region/date rows", not panel.duplicated(["pcode", "target_date"]).any())
for c_ in ["y_share3", "last_share3", "last_share3_dark", "fews_proj_share3"]:
    check(f"{c_} within 0..1", panel[c_].dropna().between(0, 1).all())
pcols = [c_ for c_ in pred.columns if c_.startswith("pred_")]
check("all predictions within 0..1", pred[pcols].stack().dropna().between(0, 1).all())
test = pred[(pred.target_date >= TEST_FROM) & pred.y_share3.notna()]
check(f"test set: {len(test)} rows, all with forecasts", len(test) == metrics["all"]["full"]["n"]
      and test[["pred_full", "pred_dark", "pred_sat"]].notna().all().all())
check(f"test set spans {test.cc.nunique()} countries, {test.pcode.nunique()} regions",
      test.cc.nunique() == metrics["test_countries"] and test.pcode.nunique() == metrics["test_regions"])

print("5. metrics reproduce")
for key, col in COLS.items():
    check(f"'{key}' recomputed", score(test, col) == metrics["all"][key])
    check(f"paired '{key}' recomputed", score(test[test.last_share3.notna()], col) == metrics["paired"][key])
for key, col in DARK_COLS.items():
    check(f"dark '{key}' recomputed", score(test, col, "last_share3_dark") == metrics["dark"][key])
    check(f"dark paired '{key}' recomputed",
          score(test[test.last_share3_dark.notna()], col, "last_share3_dark") == metrics["dark_paired"][key])

three = json.load(open(ROOT / "results/three_countries/metrics.json"))
tp = pd.read_csv(ROOT / "results/three_countries/predictions.csv")
tt = tp[(tp.target_date >= TEST_FROM) & tp.y_share3.notna()]
for key, col in {"persistence": "pred_persist", "full": "pred_full", "full_conf": "pred_full_conf", "satellite": "pred_sat",
                 "remote": "pred_remote", "fews": "fews_proj_share3", "full_tfm": "pred_full_tfm",
                 "remote_tfm": "pred_remote_tfm"}.items():
    check(f"3 countries: '{key}' recomputed", score(tt, col) == three["all"][key])
for key, col in {"persistence_dark": "pred_persist_dark", "dark": "pred_dark", "dark_conf": "pred_dark_conf",
                 "dark_tfm": "pred_dark_tfm"}.items():
    check(f"3 countries: dark '{key}' recomputed", score(tt, col, "last_share3_dark") == three["dark"][key])

print("6. README")
lookup = {**metrics, "three_all": three["all"], "three_dark": three["dark"]}
readme = (ROOT / "README.md").read_text()
n_rows = 0
for line in readme.splitlines():
    m_ = re.match(r"\| .*<!-- (\w+)\.(\w+) --> *\| *(\d+)% *\| *(\d+) of (\d+) *\| *(\d+) points *\| *(\d+) *\|", line)
    if not m_:
        continue
    n_rows += 1
    m = lookup[m_.group(1)][m_.group(2)]
    ok = (int(m_.group(3)) == pct(m["accuracy"]) and int(m_.group(4)) == m["onsets_caught"]
          and int(m_.group(5)) == m["onsets"] and int(m_.group(6)) == pct(m["mae"]) and int(m_.group(7)) == m["n"])
    check(f"README row {m_.group(1)}.{m_.group(2)} matches metrics", ok, f"metrics={m}")
check("README results tables were found and checked", n_rows >= 10, f"rows={n_rows}")

print(f"\n{passes} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
