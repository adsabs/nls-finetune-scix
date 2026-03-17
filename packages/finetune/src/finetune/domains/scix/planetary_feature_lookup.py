"""Planetary feature lookup for augmenting abs: queries with planetary_feature: clauses.

Loads planetary_feature_synonyms.json once at module level and provides:

- lookup_planetary_feature(term) — find canonical feature name/type/target
- rewrite_abs_to_abs_or_planetary_feature(query) — post-processor that appends
  OR planetary_feature:"X" to abs: clauses when there's a Gazetteer match

Built from the USGS Gazetteer of Planetary Nomenclature (16,243 features,
48 targets, 56 feature types).
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Load synonyms JSON once at module level
# ---------------------------------------------------------------------------

_SYNONYMS_PATH = Path(__file__).resolve().parents[6] / "data" / "model" / "planetary_feature_synonyms.json"

_term_to_feature: dict[str, str] = {}

if _SYNONYMS_PATH.exists():
    with open(_SYNONYMS_PATH) as _f:
        _term_to_feature = json.load(_f)


# Multi-word feature names, sorted longest-first for greedy matching in NER
_MULTI_WORD_FEATURES: list[tuple[str, str]] = sorted(
    [(k, v) for k, v in _term_to_feature.items() if " " in k],
    key=lambda x: len(x[0]),
    reverse=True,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup_planetary_feature(term: str) -> str | None:
    """Look up canonical planetary feature name for a search term.

    Case-insensitive lookup against feature names, feature types, and
    targets from the USGS Gazetteer of Planetary Nomenclature.

    Args:
        term: Search term (e.g. "olympus mons", "crater", "mars")

    Returns:
        Canonical feature name if found (e.g. "Olympus Mons"), None otherwise.
    """
    return _term_to_feature.get(term.lower().strip())


def rewrite_abs_to_abs_or_planetary_feature(query: str) -> str:
    """Augment abs: clauses with matching planetary feature terms.

    For each abs:"value" in the query, if the value matches a planetary
    feature name, type, or target in the Gazetteer, wraps it as
    (abs:"value" OR planetary_feature:"CanonicalName").

    Skips abs: clauses that are already inside an OR with planetary_feature:.

    Args:
        query: ADS query string potentially containing abs: fields

    Returns:
        Query with abs: clauses augmented with planetary_feature: where applicable.
    """
    if "abs:" not in query:
        return query

    # Skip if query already contains planetary_feature:
    if "planetary_feature:" in query:
        return query

    replacements: list[tuple[int, int, str]] = []

    # Match abs:"quoted value"
    abs_pattern = re.compile(r'abs:"([^"]+)"')
    for m in abs_pattern.finditer(query):
        val = m.group(1)
        canonical = lookup_planetary_feature(val)
        if canonical:
            replacement = f'(abs:"{val}" OR planetary_feature:"{canonical}")'
            replacements.append((m.start(), m.end(), replacement))

    # Apply replacements from right to left to preserve positions
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        query = query[:start] + replacement + query[end:]

    return query


def find_planetary_features_in_text(text: str) -> list[tuple[str, str]]:
    """Find planetary feature names in natural language text.

    Scans text for multi-word feature names from the Gazetteer.
    Only matches multi-word names (2+ words) to avoid false positives
    from short single-word names like "Ada", "Abel", "Gale".

    Args:
        text: Natural language text to scan

    Returns:
        List of (matched_text, canonical_name) tuples, longest matches first.
    """
    text_lower = text.lower()
    matches = []
    matched_spans: list[tuple[int, int]] = []

    for key, canonical in _MULTI_WORD_FEATURES:
        # Word-boundary match to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            start, end = m.start(), m.end()
            # Skip if overlapping with a longer match
            if any(s <= start < e or s < end <= e for s, e in matched_spans):
                continue
            matches.append((m.group(), canonical))
            matched_spans.append((start, end))

    return matches
