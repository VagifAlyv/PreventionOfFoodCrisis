"""Tests for src/regions.py. Run: python3 tests/test_regions.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regions import match, norm  # noqa: E402


def test_norm_drops_case_accents_and_generic_words():
    assert norm("Région de Tombouctou") == "tombouctou"
    assert norm("Upper Nile State") == "upper nile"
    assert norm("Addis Ababa City") == "addis ababa"
    assert norm("Nord-Kivu") == "nord kivu"


def test_norm_keeps_words_that_distinguish_regions():
    assert norm("Upper Nile") != norm("Nile")
    assert norm("Western Bahr el Ghazal") != norm("Northern Bahr el Ghazal")


def test_match_exact_after_norm_only():
    got = match(["Tigray", "Afar Region", "Somali"], ["TIGRAY", "Afar", "Somalia"])
    assert got == {"Tigray": "TIGRAY", "Afar Region": "Afar"}  # "Somali" is not "Somalia"


def test_match_uses_explicit_aliases():
    got = match(["Banaadir"], ["Banadir"], aliases={"Banaadir": "Banadir"})
    assert got == {"Banaadir": "Banadir"}


def test_match_refuses_ambiguous_targets():
    # two targets normalise to the same key: refuse rather than pick one
    assert match(["Bay"], ["Bay", "Bay Region"]) == {}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
