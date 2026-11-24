"""Region keys for every FEWS NET country, from OCHA admin-boundary tables (COD-AB).

Regions are identified by OCHA admin-1 p-code. Names from other sources are attached to a
p-code only by the exact rule in regions.py (see docs/adr/0001-world-coverage.md).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from regions import norm

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "data" / "raw" / "world"
NAME_COL = re.compile(r"^adm{lvl}_(name\d?|ref_name|en|es|fr|pt|ru|ar|uk)$", re.I)


def countries():
    return pd.read_csv(ROOT / "src" / "world_countries.csv", keep_default_na=False)


def _sheet(x, lvl):
    s = [s for s in x.sheet_names if s.lower().endswith(f"admin{lvl}") or s.upper() == f"ADM{lvl}"]
    return x.parse(s[0]) if s else None


def load_cod(iso3):
    """(adm1, adm2) for a country, or (None, None) if there is no OCHA table.

    adm1: pcode, name (English or first name column), names (all name columns), lat, lon
    adm2: name, parent (adm1 pcode), one row per name column value"""
    f = WORLD / f"{iso3}_admin_boundaries.xlsx"
    if not f.exists():
        return None, None
    x = pd.ExcelFile(f)
    out = []
    for lvl in (1, 2):
        t = _sheet(x, lvl)
        if t is None:
            out.append(None)
            continue
        cols = {c.lower(): c for c in t.columns}
        names = [c for c in t.columns if re.match(NAME_COL.pattern.format(lvl=lvl), c, re.I)]
        pcode = cols[f"adm{lvl}_pcode"]
        if lvl == 1:
            main = cols.get("adm1_name") or cols.get("adm1_en") or names[0]
            out.append(pd.DataFrame({
                "pcode": t[pcode].astype(str), "name": t[main].astype(str),
                "names": t[names].apply(lambda r: [str(v) for v in r if pd.notna(v)], axis=1),
                "lat": t[cols["center_lat"]] if "center_lat" in cols else np.nan,
                "lon": t[cols["center_lon"]] if "center_lon" in cols else np.nan}))
        else:
            parent = cols["adm1_pcode"]
            out.append(t.melt(id_vars=[parent], value_vars=names, value_name="name")
                       .dropna(subset=["name"]).rename(columns={parent: "parent"})[["name", "parent"]]
                       .astype(str))
    adm1 = out[0]
    pts = [s for s in x.sheet_names if s.lower().endswith("adminpoints")]
    if adm1["lat"].isna().any() and pts:  # older tables keep region centre points in a separate sheet
        p = x.parse(pts[0])
        p = p[p.admin_level == 1].drop_duplicates("adm1_pcode").set_index(p.columns[p.columns.str.lower() == "adm1_pcode"][0])
        adm1["lat"] = adm1["lat"].fillna(adm1.pcode.map(p.y_coord))
        adm1["lon"] = adm1["lon"].fillna(adm1.pcode.map(p.x_coord))
    return adm1, out[1]


def _index(pairs):
    """norm(name) -> code, keeping only names that point to one code."""
    idx = {}
    for name, code in pairs:
        idx.setdefault(norm(name), set()).add(code)
    return {k: next(iter(v)) for k, v in idx.items() if len(v) == 1 and k}


def indexes(adm1, adm2):
    i1 = _index((n, p) for p, ns in zip(adm1.pcode, adm1.names) for n in ns)
    i2 = _index(zip(adm2.name, adm2.parent)) if adm2 is not None else {}
    return i1, i2


def unit_region(full_name, adm1, adm2, idx=None, zone_first=True):
    """Admin-1 p-code of a FEWS NET unit, or None.

    The last part of the name is the country and is ignored, and so is the first part when the
    unit is a livelihood zone (zone_first). Of the remaining parts, the right-most one that is
    exactly an admin-1 name wins; failing that, the right-most one that is exactly an admin-2
    name gives its parent."""
    i1, i2 = idx or indexes(adm1, adm2)
    parts = [norm(p) for p in str(full_name).split(", ")[1 if zone_first else 0:-1]]
    for table in (i1, i2):
        for p in reversed(parts):
            if p in table:
                return table[p]
    return None
