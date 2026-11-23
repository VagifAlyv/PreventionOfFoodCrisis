"""Tests for src/timesfm_drivers.py. Run: python3 tests/test_timesfm_drivers.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timesfm_drivers import context, driver_table  # noqa: E402

P = lambda s: pd.Period(s, "M")  # noqa: E731


def monthly(start, values):
    idx = pd.period_range(start, periods=len(values), freq="M")
    return pd.Series(np.asarray(values, float), index=idx)


def test_context_ignores_values_after_cutoff():
    s = monthly("2015-01", np.arange(60))
    later = s.copy()
    later[later.index > P("2018-06")] = 999.0
    a, end_a = context(s, P("2018-06"), max_age=1)
    b, end_b = context(later, P("2018-06"), max_age=1)
    assert end_a == end_b == P("2018-06")
    assert np.array_equal(a, b) and a[-1] == s[P("2018-06")]


def test_context_starts_after_last_gap():
    s = monthly("2015-01", np.arange(60))
    s[P("2016-03")] = np.nan
    ctx, _ = context(s, P("2018-06"), max_age=1)
    assert ctx[0] == s[P("2016-04")]


def test_context_none_when_newest_value_too_old():
    s = monthly("2015-01", np.arange(40))  # ends 2018-04
    assert context(s, P("2018-06"), max_age=1) == (None, None)
    ctx, end = context(s, P("2018-06"), max_age=3)
    assert end == P("2018-04") and len(ctx) == 40


def test_context_none_when_shorter_than_24():
    assert context(monthly("2015-01", np.arange(23)), P("2016-11"), max_age=1) == (None, None)
    assert len(context(monthly("2015-01", np.arange(24)), P("2016-12"), max_age=1)[0]) == 24


def test_context_capped_at_120_months():
    ctx, _ = context(monthly("2000-01", np.arange(200)), P("2016-08"), max_age=1)
    assert len(ctx) == 120 and ctx[-1] == 199


def test_driver_table_with_last_value_forecaster():
    panel = pd.DataFrame({"cc": ["ET", "ET", "ET"], "region": ["Afar", "Afar", "Tigray"],
                          "target_date": ["2019-02", "2019-06", "2019-02"],
                          "cutoff": ["2018-10", "2019-02", "2018-10"]})
    rain = pd.DataFrame({"region": "Afar", "date": pd.period_range("2010-01", "2019-06", freq="M")})
    rain["r3q"] = np.arange(len(rain), dtype=float)
    preg = pd.DataFrame({"region": "Afar", "date": pd.period_range("2010-01", "2018-09", freq="M")})
    preg["price_yoy"] = np.linspace(0, 1, len(preg))
    pnat = pd.DataFrame({"date": pd.period_range("2010-01", "2019-06", freq="M")})
    pnat["price_yoy"] = 5.0
    seen = []

    def last_value(contexts, horizons):
        seen.extend(horizons)
        return [c[-1] for c in contexts]

    d = driver_table(panel, rain, preg, pnat, last_value).set_index(["region", "target_date"])
    afar = d.loc[("Afar", "2019-02")]
    assert afar.tfm_r3q == afar.naive_r3q == rain.set_index("date").r3q[P("2018-10")]
    assert afar.obs_r3q == rain.set_index("date").r3q[P("2019-02")]
    assert afar.ctx_end_r3q == "2018-10" and afar.ctx_end_price_yoy == "2018-09"
    assert afar.tfm_price_yoy == afar.naive_price_yoy == preg.price_yoy.iloc[-1]
    # regional prices end 2018-09, so the 2019-06 row (cutoff 2019-02) falls back to national
    assert d.loc[("Afar", "2019-06")].tfm_price_yoy == 5.0
    # Tigray has no rain series: no rain forecast, national price
    tig = d.loc[("Tigray", "2019-02")]
    assert np.isnan(tig.tfm_r3q) and tig.tfm_price_yoy == 5.0
    for c in ("ctx_end_r3q", "ctx_end_price_yoy"):
        ok = d[c].dropna()
        assert (pd.PeriodIndex(ok, freq="M") <= pd.PeriodIndex(d.loc[ok.index, "cutoff"], freq="M")).all()
    # horizon = target - newest context month: rain 4; regional price 5; national prices 4
    assert sorted(seen) == [4, 4, 4, 4, 5]
    assert afar.tfm_price_is_national == 0
    assert d.loc[("Afar", "2019-06")].tfm_price_is_national == 1 and tig.tfm_price_is_national == 1


def test_driver_table_falls_back_when_regional_series_fresh_but_short():
    panel = pd.DataFrame({"cc": ["ET"], "region": ["Afar"], "target_date": ["2019-02"], "cutoff": ["2018-10"]})
    rain = pd.DataFrame({"region": [], "date": pd.PeriodIndex([], freq="M"), "r3q": []})
    preg = pd.DataFrame({"region": "Afar", "date": pd.period_range("2017-01", "2018-10", freq="M")})
    preg["price_yoy"] = 1.0  # 22 months, ends at the cutoff: fresh but shorter than 24
    pnat = pd.DataFrame({"date": pd.period_range("2010-01", "2018-10", freq="M"), "price_yoy": 5.0})
    d = driver_table(panel, rain, preg, pnat, lambda c, h: [x[-1] for x in c]).iloc[0]
    assert d.tfm_price_yoy == 5.0 and d.tfm_price_is_national == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
