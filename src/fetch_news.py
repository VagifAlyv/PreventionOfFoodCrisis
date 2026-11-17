"""Download daily news-volume series per region from the GDELT DOC 2.0 API (free, no account).

GDELT monitors online news in 65+ languages and machine-translates it, so a query in English also
matches Somali, Amharic, Arabic and other local-language coverage that GDELT has translated.
Coverage of the DOC API starts on 1 January 2017.

For each region two series are saved:
  food   articles mentioning the region, its country and a food-stress term
  all    articles mentioning the region and its country (used to normalise)

Raw responses go to data/raw/gdelt/{cc}_{region}_{kind}.csv exactly as returned.
The API asks for at most one request every 5 seconds; this script waits 7 and retries on refusal.

Run:  python3 src/fetch_news.py      (about 15 minutes; files already present are skipped)
"""
import subprocess
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "gdelt"
API = "https://api.gdeltproject.org/api/v2/doc/doc"
COUNTRY = {"ET": "Ethiopia", "SO": "Somalia", "SS": "South Sudan"}
FOOD = '(hunger OR famine OR drought OR malnutrition OR "food insecurity" OR "food crisis")'
# Search terms for regions whose official name is ambiguous or spelled several ways.
TERMS = {
    "Benishangul Gumuz": '("Benishangul" OR "Benshangul")', "Harari": '("Harari" OR "Harar")',
    "SNNPR": '("SNNPR" OR "Southern Nations" OR "Sidama")', "Somali": '"Somali region"',
    "Gambela": '("Gambela" OR "Gambella")', "Addis Ababa": '"Addis Ababa"',
    "Banadir": '("Banadir" OR "Mogadishu")', "Bay": '("Bay region" OR "Baidoa")',
    "Bari": '("Bari region" OR "Bosaso" OR "Bossaso")', "Hiraan": '("Hiraan" OR "Hiran" OR "Beledweyne")',
    "Woqooyi Galbeed": '("Woqooyi Galbeed" OR "Hargeisa")', "Sool": '("Sool region" OR "Las Anod" OR "Lasanod")',
    "Middle Shabelle": '("Middle Shabelle" OR "Shabelle Dhexe")', "Lower Shabelle": '("Lower Shabelle" OR "Shabelle Hoose")',
    "Lakes": '"Lakes State"', "Unity": '("Unity State" OR "Bentiu")', "Upper Nile": '"Upper Nile"',
    "Warrap": '"Warrap"', "Jonglei": '"Jonglei"',
}
REGIONS = {
    "ET": ["Addis Ababa", "Afar", "Amhara", "Benishangul Gumuz", "Dire Dawa", "Gambela", "Harari",
           "Oromia", "SNNPR", "Somali", "Tigray"],
    "SO": ["Awdal", "Bakool", "Banadir", "Bari", "Bay", "Galgaduud", "Gedo", "Hiraan", "Lower Juba",
           "Lower Shabelle", "Middle Juba", "Middle Shabelle", "Mudug", "Nugaal", "Sanaag", "Sool",
           "Togdheer", "Woqooyi Galbeed"],
    "SS": ["Central Equatoria", "Eastern Equatoria", "Jonglei", "Lakes", "Northern Bahr el Ghazal",
           "Unity", "Upper Nile", "Warrap", "Western Bahr el Ghazal", "Western Equatoria"],
}


def query(cc, region, kind):
    q = f'{TERMS.get(region, chr(34) + region + chr(34))} "{COUNTRY[cc]}"'
    return q + " " + FOOD if kind == "food" else q


def fetch(q, end):
    params = {"query": q, "mode": "timelinevolraw", "format": "csv", "timelinesmooth": "0",
              "startdatetime": "20170101000000", "enddatetime": end}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(10):
        # curl rather than urllib: some Python installs on macOS ship without CA certificates
        r = subprocess.run(["curl", "-s", "--max-time", "120", "-A", "research-script", url],
                           capture_output=True, text=True)
        body = r.stdout.lstrip("\ufeff") if r.returncode == 0 else f"curl error {r.returncode}"
        if body.startswith("Date,"):
            return body
        time.sleep(300)  # GDELT keeps blocking while requests continue; back off for 5 minutes
    raise RuntimeError(f"GDELT refused: {q}\n{body[:200]}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    end = time.strftime("%Y%m%d000000", time.gmtime())
    for cc, regs in REGIONS.items():
        for r in regs:
            for kind in ("food", "all"):
                f = OUT / f"{cc}_{r.replace(' ', '_')}_{kind}.csv"
                if f.exists():
                    continue
                q = query(cc, r, kind)
                f.write_text(f"# query: {q}\n" + fetch(q, end))
                print("saved", f.name, flush=True)
                time.sleep(7)


if __name__ == "__main__":
    main()
