"""Tests for src/world.py. Run: python3 tests/test_world.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world import unit_region  # noqa: E402

ADM1 = pd.DataFrame({"pcode": ["CD1", "CD2", "KE1", "KE2", "HT1"],
                     "names": [["Kasai"], ["Lomami"], ["Trans Nzoia"], ["Mandera"], ["North", "Nord"]]})
ADM2 = pd.DataFrame({"name": ["Bandiagara"], "parent": ["CD2"]})


def test_country_suffix_is_skipped():
    full = "Fishing and Staples Riverine, Mweka, Kasai, Democratic Republic of the Congo, Congo, The Democratic Republic of the"
    assert unit_region(full, ADM1, ADM2) == "CD1"


def test_rightmost_admin1_wins_over_old_province():
    assert unit_region("Central Trans Nzoia, Trans Nzoia, Rift Valley, Kenya", ADM1, ADM2) == "KE1"


def test_alternate_language_name():
    assert unit_region("Torbeck, Les Cayes, Nord, Haiti", ADM1, ADM2) == "HT1"


def test_admin2_falls_back_to_parent():
    assert unit_region("Center-eastern Millet and Livestock, Bandiagara, Mali", ADM1, ADM2) == "CD2"


def test_livelihood_zone_name_is_never_used():
    # the first part is a livelihood zone and must not match even if it equals a region name
    assert unit_region("Mandera, Somewhere, Kenya", ADM1, ADM2) is None


def test_country_name_equal_to_a_region_is_not_the_region():
    adm1 = pd.DataFrame({"pcode": ["GT1", "GT2"], "names": [["Guatemala"], ["Quiche"]]})
    assert unit_region("Highlands, Canilla, Quiche, Guatemala", adm1, None) == "GT2"
    assert unit_region("Urban, Mixco, Guatemala, Guatemala", adm1, None) == "GT1"


def test_admin_units_have_no_zone_part():
    adm1 = pd.DataFrame({"pcode": ["CO1"], "names": [["Amazonas"]]})
    assert unit_region("Amazonas, Colombia", adm1, None, zone_first=False) == "CO1"
    assert unit_region("Amazonas, Colombia", adm1, None) is None  # as a zone name it is ignored


def test_unmatched_returns_none():
    assert unit_region("Zone, Nowhere, Atlantis", ADM1, ADM2) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
