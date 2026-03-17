#!/usr/bin/env python3
"""Realism cleanup for gold_examples.json.

Fixes template-generated NL that no real user would type, removes broken/
unsalvageable queries, deduplicates contradictory content-field cross-products,
and reduces over-represented subcategories.

Based on findings in data/reference/realism_audit.md (~685 of ~5,054 examples
flagged, 14% of dataset).

Usage:
    python scripts/realism_cleanup.py              # Dry run
    python scripts/realism_cleanup.py --apply      # Apply changes
    python scripts/realism_cleanup.py --apply --verbose
"""

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE = PROJECT_ROOT / "data" / "datasets" / "raw" / "gold_examples.json"
REMOVED_FILE = PROJECT_ROOT / "data" / "reference" / "removed_examples.json"

# ───────────────────────────────────────────────────────────────────────────
# Priority 0: Remove broken / unsalvageable
# ───────────────────────────────────────────────────────────────────────────

# NL substrings that indicate non-academic / off-topic entries
REMOVE_BY_NL_SUBSTRING = [
    "surgical scalpel design and safety",
    "coastal erosion human displacement vulnerability",
    "sybil attacks intelligent vehicular",
    "electronic payment security",
    "boresight translation optical alignment",
    "physics economic growth development national policy",
]


def _should_remove_broken(ex: dict) -> str | None:
    """Return removal reason or None."""
    q = ex.get("ads_query", "")
    nl = ex.get("natural_language", "")

    # docs() library hashes in queries (user-session artifacts)
    if re.search(r'docs\([0-9a-f]{20,}\)', q):
        return "docs() hash in query"
    if re.search(r'docs\(library/', q):
        return "docs(library/) hash in query"
    if re.search(r'-docs\([0-9a-f]{20,}\)', q):
        return "-docs() hash in query"

    # DOIs in abs:() field
    if re.search(r'abs:\(?10\.\d{4}/', q):
        return "DOI in abs: field"

    # Absurd date ranges (year:1641)
    if "1641" in q:
        return "absurd date range (1641)"

    # Non-academic / off-topic by NL substring
    nl_lower = nl.lower()
    for substr in REMOVE_BY_NL_SUBSTRING:
        if substr.lower() in nl_lower:
            return f"off-topic: {substr[:40]}"

    return None


# ───────────────────────────────────────────────────────────────────────────
# Priority 1: Deduplicate content-field cross-products
# ───────────────────────────────────────────────────────────────────────────

def _normalize_nl(nl: str) -> str:
    """Normalize NL for dedup grouping (lowercase, strip)."""
    return nl.strip().lower()


def _extract_field(query: str) -> str | None:
    """Extract the primary search field from a simple field:"value" query."""
    m = re.match(r'^(abs|title|body|full|keyword):', query)
    if m:
        return m.group(1)
    # Also match abs:(...) patterns
    m = re.match(r'^(abs|title|body|full|keyword):\(', query)
    if m:
        return m.group(1)
    return None


# Field priority: keep abs > title > full > body > keyword
FIELD_PRIORITY = {"abs": 0, "title": 1, "full": 2, "body": 3, "keyword": 4}


def find_crossproduct_removals(examples: list) -> set:
    """Find indices to remove from content-field cross-product duplicates.

    For NL strings that map to multiple field variants (abs/title/body/full/keyword),
    keep only abs: when NL is generic. Keep field-specific variants where NL
    explicitly mentions the field.
    """
    nl_groups = defaultdict(list)
    for i, ex in enumerate(examples):
        nl_norm = _normalize_nl(ex["natural_language"])
        nl_groups[nl_norm].append(i)

    remove_indices = set()
    for nl_norm, indices in nl_groups.items():
        if len(indices) <= 1:
            continue

        # Check if this group has multiple field variants
        field_map = {}  # field -> list of indices
        for idx in indices:
            field = _extract_field(examples[idx]["ads_query"])
            if field:
                field_map.setdefault(field, []).append(idx)

        if len(field_map) <= 1:
            continue

        # NL does NOT mention a specific field -> keep only abs:
        nl_mentions_field = any(
            hint in nl_norm
            for hint in ["in the title", "in title", "in the body", "in body",
                         "in the full text", "in full text", "full text",
                         "in keyword", "body text"]
        )

        if not nl_mentions_field:
            # Keep the best field (abs preferred), remove the rest
            best_field = min(field_map.keys(), key=lambda f: FIELD_PRIORITY.get(f, 99))
            for field, idxs in field_map.items():
                if field != best_field:
                    remove_indices.update(idxs)

    return remove_indices


# ───────────────────────────────────────────────────────────────────────────
# Priority 2: Rewrite "X in Y field" NL patterns
# ───────────────────────────────────────────────────────────────────────────

def rewrite_field_nl(nl: str, query: str) -> str | None:
    """Rewrite stilted 'X in Y field' NL to natural phrasing. Returns new NL or None."""
    nl_lower = nl.lower()

    # "X in body field" -> "papers mentioning X in the full text"
    m = re.match(r'^(.+?)\s+in\s+body\s+field$', nl, re.IGNORECASE)
    if m:
        topic = m.group(1)
        return f"papers mentioning {topic} in the full text"

    # "body contains X" -> "papers discussing X in the body text"
    m = re.match(r'^body\s+contains\s+(.+)$', nl, re.IGNORECASE)
    if m:
        topic = m.group(1)
        return f"papers discussing {topic} in the body text"

    # "X in full field" -> "X mentioned anywhere in the paper"
    m = re.match(r'^(.+?)\s+in\s+full\s+field$', nl, re.IGNORECASE)
    if m:
        topic = m.group(1)
        return f"{topic} mentioned anywhere in the paper"

    # "full contains X" -> "papers with X anywhere in the text"
    m = re.match(r'^full\s+contains\s+(.+)$', nl, re.IGNORECASE)
    if m:
        topic = m.group(1)
        return f"papers with {topic} anywhere in the text"

    # "X in abs field" -> "papers about X" (abs is the default, no special phrasing needed)
    m = re.match(r'^(.+?)\s+in\s+abs\s+field$', nl, re.IGNORECASE)
    if m:
        topic = m.group(1)
        return f"papers about {topic}"

    # "find X in body" -> "papers mentioning X in the full text"
    m = re.match(r'^find\s+(.+?)\s+in\s+body$', nl, re.IGNORECASE)
    if m:
        topic = m.group(1)
        return f"papers mentioning {topic} in the full text"

    # "find X in full" -> "papers with X anywhere in the text"
    m = re.match(r'^find\s+(.+?)\s+in\s+full$', nl, re.IGNORECASE)
    if m:
        topic = m.group(1)
        return f"papers with {topic} anywhere in the text"

    return None


# ───────────────────────────────────────────────────────────────────────────
# Priority 3: Rewrite bibgroup template NL
# ───────────────────────────────────────────────────────────────────────────

# Telescope code -> friendly name
TELESCOPE_FRIENDLY = {
    "HST": "Hubble",
    "JWST": "Webb",
    "Spitzer": "Spitzer",
    "Chandra": "Chandra",
    "XMM": "XMM-Newton",
    "NuSTAR": "NuSTAR",
    "RXTE": "RXTE",
    "Swift": "Swift",
    "Fermi": "Fermi",
    "GALEX": "GALEX",
    "FUSE": "FUSE",
    "IUE": "IUE",
    "EUVE": "EUVE",
    "IRAS": "IRAS",
    "WISE": "WISE",
    "NEOWISE": "NEOWISE",
    "Kepler": "Kepler",
    "K2": "K2",
    "TESS": "TESS",
    "SOHO": "SOHO",
    "STEREO": "STEREO",
    "SDO": "SDO",
    "Gaia": "Gaia",
    "Hipparcos": "Hipparcos",
    "VLT": "VLT",
    "Keck": "Keck",
    "Gemini": "Gemini",
    "Subaru": "Subaru",
    "GBT": "Green Bank",
    "VLA": "VLA",
    "ALMA": "ALMA",
    "SDSS": "SDSS",
    "2MASS": "2MASS",
    "LIGO": "LIGO",
    "ESO": "ESO",
    "SETI": "SETI",
    "CXO": "Chandra",
    "ROSAT": "ROSAT",
}


def rewrite_bibgroup_nl(nl: str, query: str) -> str | None:
    """Rewrite bibgroup template NL to natural phrasing. Returns new NL or None."""
    # Extract bibgroup code from query
    m = re.search(r'bibgroup:\"?(\w+)\"?', query)
    if not m:
        return None
    code = m.group(1)
    friendly = TELESCOPE_FRIENDLY.get(code, code)

    # Extract topic from query if present
    topic_m = re.search(r'abs:"([^"]+)"', query) or re.search(r'abs:\(([^)]+)\)', query)
    topic = topic_m.group(1) if topic_m else None
    # Clean up AND-joined topics, preserving word order from NL
    if topic and " AND " in topic:
        words = [w.strip().lower() for w in topic.split(" AND ") if w.strip()]
        # Reconstruct natural word order from the NL text
        nl_lower_for_order = nl.lower()
        word_positions = []
        for w in words:
            pos = nl_lower_for_order.find(w)
            word_positions.append((pos if pos >= 0 else 999, w))
        word_positions.sort()
        topic = " ".join(w for _, w in word_positions)

    nl_lower = nl.lower()

    # Skip exclusion patterns ("excluding X bibgroup") - these are fine
    if "excluding" in nl_lower:
        return None

    # Skip entries with has: - those have compound meaning beyond bibgroup
    if "has:" in query:
        return None

    # "{CODE} bibliography {TOPIC}" -> "{TOPIC} papers from {friendly}"
    if "bibliography" in nl_lower and topic:
        return f"{topic} papers from {friendly}"

    # "{TOPIC} in {CODE} bibgroup" -> "{TOPIC} papers using {friendly} data"
    if "bibgroup" in nl_lower and topic:
        return f"{topic} papers using {friendly} data"

    # "{CODE} bibliography papers" (no topic) -> "{friendly} papers"
    if "bibliography" in nl_lower and not topic:
        # Try to extract topic context from NL (remove code, "bibliography", "papers", etc.)
        nl_topic = re.sub(
            r'\b(bibliography|bibgroup|papers?|in|about|from|by)\b',
            '', nl_lower
        )
        nl_topic = re.sub(r'\b' + re.escape(code.lower()) + r'\b', '', nl_topic).strip()
        nl_topic = re.sub(r'\s+', ' ', nl_topic).strip()
        if nl_topic and len(nl_topic) > 3:
            return f"{nl_topic} papers from {friendly}"
        return f"{friendly} papers"

    # "bibgroup {code}" (bare) -> "{friendly} papers"
    if nl_lower.strip() == f"bibgroup {code.lower()}":
        return f"{friendly} papers"

    # "{code} bibgroup papers" -> "{friendly} papers"
    if "bibgroup" in nl_lower and not topic:
        return f"{friendly} papers"

    # "papers in {CODE} bibliography" -> "papers from {friendly}"
    if "bibliography" in nl_lower and "papers in" in nl_lower and not topic:
        return f"papers from {friendly}"

    return None


# ───────────────────────────────────────────────────────────────────────────
# Priority 4: Rewrite syntax demo NL (manual dict)
# ───────────────────────────────────────────────────────────────────────────

SYNTAX_NL_REWRITES = {
    "title: map NEAR5 planar":
        "papers where map and planar appear close together in the title",
    "facility regex magell.*":
        "papers about Magellan telescope observations",
    "instrument/facility matching magell.*":
        "papers mentioning Magellan telescope or similar facilities",
    "title =star":
        "papers with the exact word star in the title, not STAR acronym",
    "huchra jo wildcard":
        "papers by Huchra with first name starting with Jo",
    "author bol? wildcard":
        "papers by authors whose last name starts with bol",
    "authors bol_ single char":
        "papers by authors with a three-letter last name starting with bol",
    "bol? author last name":
        "papers by authors whose last name starts with bol",
}


# ───────────────────────────────────────────────────────────────────────────
# Priority 5: Reduce open access subtype over-representation
# ───────────────────────────────────────────────────────────────────────────

OA_SUBTYPES = ["ads_openaccess", "eprint_openaccess", "pub_openaccess", "author_openaccess"]
MAX_PER_OA_SUBTYPE = 3


def find_excess_oa_removals(examples: list) -> set:
    """Find indices of excess OA subtype examples to remove.

    Keep up to MAX_PER_OA_SUBTYPE per subtype, preferring compound queries
    (those with additional fields beyond just property:).
    """
    remove_indices = set()
    for subtype in OA_SUBTYPES:
        subtype_indices = []
        for i, ex in enumerate(examples):
            if subtype in ex.get("ads_query", ""):
                subtype_indices.append(i)

        if len(subtype_indices) <= MAX_PER_OA_SUBTYPE:
            continue

        # Score: prefer compound queries (more fields = higher score)
        def compound_score(idx):
            q = examples[idx]["ads_query"]
            return q.count(":") - 1  # subtract 1 for the property: itself

        subtype_indices.sort(key=compound_score, reverse=True)
        # Remove excess (keep first MAX_PER_OA_SUBTYPE)
        for idx in subtype_indices[MAX_PER_OA_SUBTYPE:]:
            remove_indices.add(idx)

    return remove_indices


# ───────────────────────────────────────────────────────────────────────────
# Priority 6: Rewrite has: field NL (manual dict)
# ───────────────────────────────────────────────────────────────────────────

HAS_NL_REWRITES = {
    "papers with volume information about cosmology":
        "cosmology papers that have volume numbers",
    "articles that have issue numbers on gravitational lensing":
        "gravitational lensing articles with issue numbers",
    "papers with unified astronomy thesaurus tags about star formation":
        "star formation papers tagged with UAT concepts",
    "papers with publisher-verified ORCID IDs by first author Smith":
        "first-author Smith papers with verified ORCID",
    "papers with affiliation information about stellar spectroscopy":
        "stellar spectroscopy papers with affiliation data",
    "records that have an abstract about solar flares":
        "solar flare papers that include an abstract",
    "papers with keywords about interstellar medium":
        "interstellar medium papers with assigned keywords",
    "papers with grant information about galaxy surveys":
        "galaxy survey papers with grant acknowledgments",
    "papers with publisher-verified ORCID on dark matter":
        "dark matter papers with verified ORCID IDs",
}


# ───────────────────────────────────────────────────────────────────────────
# Priority 7: Rewrite exact-match / synonym NL (manual dict)
# ───────────────────────────────────────────────────────────────────────────

EXACT_MATCH_NL_REWRITES = {
    "find exact matches for the term supernova without related terms":
        "papers about supernova specifically, not supernovae variants",
    "only author signature 'smith, j' exactly":
        "papers by exactly Smith, J without name variants",
    "disable author name expansion smith j":
        "papers by Smith, J specifically, no alternate spellings",
    "exact token etoile in title":
        "papers with etoile in the title, not star",
    "no synonyms title etoile":
        "papers titled etoile without synonym expansion to star",
    "disable synonyms for star in title":
        "papers with star in the title, not its synonyms",
    "exact search for keyword accretion without synonyms":
        "papers with the exact keyword accretion",
    "search for keyword sun without synonym expansion":
        "papers with keyword sun specifically",
    "exact keyword search for galaxies without synonyms":
        "papers with the exact keyword galaxies",
}


# ───────────────────────────────────────────────────────────────────────────
# Main cleanup
# ───────────────────────────────────────────────────────────────────────────

def cleanup(examples: list, verbose: bool = False) -> tuple:
    """Run all cleanup priorities. Returns (cleaned, removed, stats)."""
    stats = {
        "p0_broken_removed": 0,
        "p1_crossproduct_removed": 0,
        "p2_field_nl_rewritten": 0,
        "p3_bibgroup_nl_rewritten": 0,
        "p4_syntax_nl_rewritten": 0,
        "p5_oa_excess_removed": 0,
        "p6_has_nl_rewritten": 0,
        "p7_exact_nl_rewritten": 0,
    }

    # Pre-scan: build removal sets
    p0_remove = set()
    for i, ex in enumerate(examples):
        reason = _should_remove_broken(ex)
        if reason:
            p0_remove.add(i)

    p1_remove = find_crossproduct_removals(examples)
    p5_remove = find_excess_oa_removals(examples)

    all_removals = p0_remove | p1_remove | p5_remove

    removed = []
    cleaned = []

    for i, ex in enumerate(examples):
        ex = copy.deepcopy(ex)
        nl = ex.get("natural_language", "")
        query = ex.get("ads_query", "")

        # ── Removals ──
        if i in p0_remove:
            reason = _should_remove_broken(ex)
            removed.append({"index": i, "reason": f"p0: {reason}", **ex})
            stats["p0_broken_removed"] += 1
            if verbose:
                print(f"  [{i}] P0 REMOVE ({reason}): {nl[:60]}")
            continue

        if i in p1_remove:
            removed.append({"index": i, "reason": "p1: cross-product duplicate", **ex})
            stats["p1_crossproduct_removed"] += 1
            if verbose:
                print(f"  [{i}] P1 REMOVE cross-product: {nl[:40]} -> {query[:40]}")
            continue

        if i in p5_remove:
            removed.append({"index": i, "reason": "p5: excess OA subtype", **ex})
            stats["p5_oa_excess_removed"] += 1
            if verbose:
                print(f"  [{i}] P5 REMOVE excess OA: {nl[:60]}")
            continue

        # ── Rewrites (applied in priority order) ──
        new_nl = None

        # P2: Field NL rewrites
        new_nl = rewrite_field_nl(nl, query)
        if new_nl:
            if verbose:
                print(f"  [{i}] P2 REWRITE: '{nl}' -> '{new_nl}'")
            ex["natural_language"] = new_nl
            stats["p2_field_nl_rewritten"] += 1

        # P3: Bibgroup template rewrites
        if not new_nl and "bibgroup:" in query:
            new_nl = rewrite_bibgroup_nl(nl, query)
            if new_nl:
                if verbose:
                    print(f"  [{i}] P3 REWRITE: '{nl}' -> '{new_nl}'")
                ex["natural_language"] = new_nl
                stats["p3_bibgroup_nl_rewritten"] += 1

        # P4: Syntax demo rewrites
        if not new_nl and nl in SYNTAX_NL_REWRITES:
            new_nl = SYNTAX_NL_REWRITES[nl]
            if verbose:
                print(f"  [{i}] P4 REWRITE: '{nl}' -> '{new_nl}'")
            ex["natural_language"] = new_nl
            stats["p4_syntax_nl_rewritten"] += 1

        # P6: has: field NL rewrites
        if not new_nl and nl in HAS_NL_REWRITES:
            new_nl = HAS_NL_REWRITES[nl]
            if verbose:
                print(f"  [{i}] P6 REWRITE: '{nl}' -> '{new_nl}'")
            ex["natural_language"] = new_nl
            stats["p6_has_nl_rewritten"] += 1

        # P7: Exact-match NL rewrites
        if not new_nl and nl in EXACT_MATCH_NL_REWRITES:
            new_nl = EXACT_MATCH_NL_REWRITES[nl]
            if verbose:
                print(f"  [{i}] P7 REWRITE: '{nl}' -> '{new_nl}'")
            ex["natural_language"] = new_nl
            stats["p7_exact_nl_rewritten"] += 1

        cleaned.append(ex)

    return cleaned, removed, stats


def main():
    parser = argparse.ArgumentParser(description="Realism cleanup for gold training examples")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--verbose", action="store_true", help="Show each change")
    parser.add_argument("--gold-file", type=str, default=str(GOLD_FILE))
    args = parser.parse_args()

    gold_path = Path(args.gold_file)
    print(f"Loading {gold_path}...")
    examples = json.loads(gold_path.read_text())
    print(f"  Loaded {len(examples)} examples")

    cleaned, removed, stats = cleanup(examples, verbose=args.verbose)

    total_rewrites = sum(v for k, v in stats.items() if "rewritten" in k)
    total_removed = sum(v for k, v in stats.items() if "removed" in k)

    print(f"\n{'='*60}")
    print("REALISM CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"  Original count:      {len(examples)}")
    print(f"  After cleanup:       {len(cleaned)}")
    print(f"  Removed:             {total_removed}")
    print(f"  Rewritten:           {total_rewrites}")
    print(f"  ---")
    for key, val in stats.items():
        if val > 0:
            print(f"  {key:30s} {val}")
    print(f"{'='*60}")

    if args.apply:
        print(f"\nWriting {len(cleaned)} cleaned examples to {gold_path}...")
        gold_path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n")

        # Append to existing removed_examples.json if it exists
        removed_path = Path(str(REMOVED_FILE))
        existing_removed = []
        if removed_path.exists():
            try:
                existing_removed = json.loads(removed_path.read_text())
            except (json.JSONDecodeError, ValueError):
                pass

        all_removed = existing_removed + removed
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Appending {len(removed)} removed examples to {removed_path} (total: {len(all_removed)})...")
        removed_path.write_text(json.dumps(all_removed, indent=2, ensure_ascii=False) + "\n")

        print("Done!")
    else:
        print("\nDRY RUN — use --apply to write changes")


if __name__ == "__main__":
    main()
