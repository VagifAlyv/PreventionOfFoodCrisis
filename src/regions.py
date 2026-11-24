"""Match region names between data sources by a fixed rule, never by similarity.

Rule: lower case, accents removed, punctuation to spaces, generic words dropped, then exact
equality. Spelling differences beyond that need an explicit alias. A name that matches two
targets is left unmatched.
"""
import re
import unicodedata

GENERIC = {"region", "regions", "province", "state", "county", "department", "departement",
           "governorate", "prefecture", "oblast", "city", "municipality", "district", "zone",
           "division", "territory", "capital", "autonomous", "republic", "de", "du", "des", "of", "the"}


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    return " ".join(w for w in words if w not in GENERIC)


def match(names, targets, aliases=None):
    """{name: target} for every name whose normalised form (after aliases) equals exactly one target's."""
    aliases = aliases or {}
    keys = {}
    for t in targets:
        keys.setdefault(norm(t), []).append(t)
    out = {}
    for n in names:
        hit = keys.get(norm(aliases.get(n, n)), [])
        if len(hit) == 1:
            out[n] = hit[0]
    return out
