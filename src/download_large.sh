#!/usr/bin/env bash
# Fetch the raw files that are too large for GitHub (over 100 MB) from their original sources.
# 3-country experiment only (results/three_countries/, commit 93b27f6). The world pipeline uses download_world.sh.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"

[ -f fews_ipc_ET.csv ] || curl -fL -o fews_ipc_ET.csv "https://fdw.fews.net/api/ipcphase.csv?country_code=ET"

mkdir -p ucdp
if [ ! -f ucdp/GEDEvent_v26_1.csv ]; then
  curl -fL -o ucdp/ged261-csv.zip https://ucdp.uu.se/downloads/ged/ged261-csv.zip
  unzip -o -q ucdp/ged261-csv.zip -d ucdp && rm ucdp/ged261-csv.zip
fi
echo "large raw files ready"
