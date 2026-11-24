#!/usr/bin/env bash
# Fetch the raw data for every FEWS NET country listed in src/world_countries.csv into
# data/raw/world/ (git-ignored, about 2.6 GB). Existing files are kept, so a rerun resumes.
#   fews_ipc_<CC>.csv            FEWS NET IPC classifications, full history (fdw.fews.net)
#   fews_ipc_current.csv         FEWS NET current situation, all countries, since Oct 2025
#   <iso3>-rainfall-subnat-full.csv   WFP subnational rainfall (CHIRPS) via HDX
#   wfp_food_prices_<iso3>.csv   WFP market prices via HDX
#   <iso3>_admin_boundaries.xlsx OCHA region codes (COD-AB) via HDX
#   ne_110m_admin_0_countries.geojson  Natural Earth 1:110m country outlines (public domain)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw/world"
mkdir -p "$OUT"
cd "$OUT"

get() {  # get <file> <url>
  [ -s "$1" ] && return 0
  curl -fsSL --retry 3 -m 900 -o "$1.part" "$2" && mv "$1.part" "$1" || { echo "FAILED $1" >&2; rm -f "$1.part"; }
}

get ne_110m_admin_0_countries.geojson \
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/ne_110m_admin_0_countries.geojson
get fews_ipc_current.csv "https://fdw.fews.net/api/ipcphase.csv?scenario=CS&start_date=2025-10-01"

tail -n +2 "$ROOT/src/world_countries.csv" | tr -d '\r' | while IFS=, read -r name cc iso3 rain price cod; do
  get "fews_ipc_${cc}.csv" "https://fdw.fews.net/api/ipcphase.csv?country_code=${cc}"
  [ -n "$rain" ] && get "${iso3}-rainfall-subnat-full.csv" "$rain"
  [ -n "$price" ] && get "wfp_food_prices_${iso3}.csv" "$price"
  [ -n "$cod" ] && get "${iso3}_admin_boundaries.xlsx" "$cod"
  echo "done $cc"
done
date -u +"downloaded %Y-%m-%d" > DOWNLOADED
echo "world raw files ready in $OUT"
