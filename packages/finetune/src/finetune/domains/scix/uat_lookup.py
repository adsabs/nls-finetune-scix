"""UAT (Unified Astronomy Thesaurus) lookup for augmenting abs: queries with uat: clauses.

Loads uat_synonyms.json once at module level and provides:

- lookup_uat(term) — find canonical UAT prefLabel for a search term
- rewrite_abs_to_abs_or_uat(query) — post-processor that appends OR uat:"X" to abs: clauses
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Load UAT synonyms JSON once at module level
# ---------------------------------------------------------------------------

_SYNONYMS_PATH = Path(__file__).resolve().parents[6] / "data" / "model" / "uat_synonyms.json"

_term_to_uat: dict[str, str] = {}

if _SYNONYMS_PATH.exists():
    with open(_SYNONYMS_PATH) as _f:
        _term_to_uat = json.load(_f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup_uat(term: str) -> str | None:
    """Look up canonical UAT prefLabel for a search term.

    Case-insensitive lookup against prefLabels and altLabels from the
    Unified Astronomy Thesaurus v6.0.

    Args:
        term: Search term (e.g. "dark matter", "CMB", "exoplanets")

    Returns:
        Canonical UAT prefLabel if found (e.g. "Dark matter"), None otherwise.
    """
    return _term_to_uat.get(term.lower().strip())


def rewrite_abs_to_abs_or_uat(query: str) -> str:
    """Augment abs: clauses with matching UAT terms for better recall.

    For each abs:"value" in the query, if the value matches a UAT concept,
    wraps it as (abs:"value" OR uat:"UATLabel"). Skips abs: clauses that
    are already inside an OR with uat:.

    Args:
        query: ADS query string potentially containing abs: fields

    Returns:
        Query with abs: clauses augmented with uat: where applicable.
    """
    if "abs:" not in query:
        return query

    # Skip if query already contains uat:
    if "uat:" in query:
        return query

    replacements: list[tuple[int, int, str]] = []

    # Match abs:"quoted value"
    abs_pattern = re.compile(r'abs:"([^"]+)"')
    for m in abs_pattern.finditer(query):
        val = m.group(1)
        uat_label = lookup_uat(val)
        if uat_label:
            replacement = f'(abs:"{val}" OR uat:"{uat_label}")'
            replacements.append((m.start(), m.end(), replacement))

    # Apply replacements from right to left to preserve positions
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        query = query[:start] + replacement + query[end:]

    return query
