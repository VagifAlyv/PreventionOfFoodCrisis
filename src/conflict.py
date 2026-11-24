"""Monthly organised-violence counts per admin-1 region from UCDP GED.

3-country experiment only (results/three_countries/). It runs against the 3-country build_dataset.py of commit 93b27f6, not the world pipeline.


Sources (no account needed):
  data/raw/ucdp/GEDEvent_v26_1.csv            UCDP GED 26.1, final, 1989-2025
  data/raw/ucdp/GEDEvent_v26_01_26_09.csv     UCDP Candidate Events 26.0.1-26.0.9, Jan-Sep 2026
  data/raw/boundaries/{ETH,SOM,SSD}-ADM1.geojson   geoBoundaries gbOpen ADM1

Each event is placed in a region by its coordinates, not by UCDP's adm_1 text (which mixes
pre-1995 Ethiopian provinces with current regions). Events only located to country level
(where_prec 6 or 7) are dropped, because their coordinates are a country centroid.

Run:  python3 src/conflict.py   (writes data/processed/conflict_monthly.csv)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GED_FILES = ["GEDEvent_v26_1.csv", "GEDEvent_v26_01_26_09.csv"]
COUNTRY_IDS = {530: "ET", 520: "SO", 626: "SS"}  # Gleditsch-Ward codes used by UCDP
GEO = {"ET": "ETH", "SO": "SOM", "SS": "SSD"}
# geoBoundaries spelling -> spelling used in build_dataset.py
NAMES = {"Beneshangul Gumu": "Benishangul Gumuz", "Hareri": "Harari", "Hiiraan": "Hiraan",
         "Lower Shebelle": "Lower Shabelle", "Middle Shebelle": "Middle Shabelle"}


def regions(cc):
    g = json.load(open(RAW / "boundaries" / f"{GEO[cc]}-ADM1.geojson"))
    out = []
    for f in g["features"]:
        name = NAMES.get(f["properties"]["shapeName"], f["properties"]["shapeName"])
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        out.append((name, [MPath(np.asarray(p[0])) for p in polys]))
    return out


def assign(ev, cc):
    pts = ev[["longitude", "latitude"]].to_numpy()
    region = np.full(len(ev), None, dtype=object)
    for name, paths in regions(cc):
        inside = np.zeros(len(ev), bool)
        for p in paths:
            inside |= p.contains_points(pts)
        region[inside & (region == None)] = name  # noqa: E711
    return region


def load_events():
    frames = []
    for f in GED_FILES:
        d = pd.read_csv(RAW / "ucdp" / f, low_memory=False, usecols=[
            "id", "country_id", "type_of_violence", "where_prec", "latitude", "longitude",
            "date_start", "best"])
        frames.append(d[d.country_id.isin(COUNTRY_IDS)])
    ev = pd.concat(frames).drop_duplicates("id")
    ev = ev[ev.where_prec <= 5].copy()
    ev["cc"] = ev.country_id.map(COUNTRY_IDS)
    ev["date"] = pd.to_datetime(ev.date_start).dt.to_period("M")
    ev["region"] = None
    for cc in COUNTRY_IDS.values():
        m = ev.cc == cc
        ev.loc[m, "region"] = assign(ev[m], cc)
    return ev


def build():
    ev = load_events()
    unplaced = ev.region.isna().mean()
    ev = ev.dropna(subset=["region"])
    ev["civ"] = ev.type_of_violence == 3  # one-sided violence against civilians
    m = (ev.groupby(["cc", "region", "date"])
           .agg(events=("id", "size"), deaths=("best", "sum"), civ_events=("civ", "sum"))
           .reset_index())
    # fill months with no events with zeros, up to the last month covered by the data
    last = ev.date.max()
    full = []
    for (cc, r), g in m.groupby(["cc", "region"]):
        idx = pd.period_range("1989-01", last, freq="M")
        g = g.set_index("date").reindex(idx).fillna({"events": 0, "deaths": 0, "civ_events": 0})
        g["cc"], g["region"] = cc, r
        full.append(g.rename_axis("date").reset_index())
    out = pd.concat(full)[["cc", "region", "date", "events", "deaths", "civ_events"]]
    out["date"] = out.date.astype(str)
    out.to_csv(ROOT / "data" / "processed" / "conflict_monthly.csv", index=False)
    print(f"{len(ev)} events placed in {ev.region.nunique()} regions, "
          f"{unplaced:.1%} outside all polygons (dropped), data through {last}")
    return out


if __name__ == "__main__":
    build()
