"""Remove training examples that teach the LLM to do synonym expansion.

Problem: ~20 examples teach the model to produce abs:("X" OR Y OR Z) patterns
where Y and Z are synonyms NOT present in the user's natural language input.
This causes the model to hallucinate terms like "singularity", "event horizon"
when the user asked for "gravitational waves from black hole mergers".

Also removes examples with fabricated/broken abs: terms (author names in abs:,
unrelated filler topics like abs:"exoplanet" for generic filter queries).

Examples where the user literally said "or" (e.g., "neutron star or pulsar")
are kept — those teach correct OR syntax.

Usage:
    python scripts/remove_synonym_expansion.py          # dry-run
    python scripts/remove_synonym_expansion.py --apply  # commit changes
"""

import json
import sys
from pathlib import Path

GOLD_PATH = Path("data/datasets/raw/gold_examples.json")
REMOVED_PATH = Path("data/reference/removed_examples.json")

# Indices of examples to REMOVE (0-based)
# Each entry has a comment explaining why
REMOVE_INDICES = {
    # === OR-expansion: adds synonym terms not in the NL ===
    720,   # "streams and globular clusters" → adds "dwarf" synonym
    829,   # "near earth objects" → adds NEO abbreviation
    844,   # "asteroid detection surveys" → adds discovery, survey* synonyms
    1053,  # "catfish hematology" → adds haematology spelling variant
    1527,  # "limb darkening measurements" → adds measurement (singular OR)
    1547,  # "bug detection" → adds "bugs" synonym
    1709,  # "narrowband radio signals" → adds emission*, technosignature*, SETI
    1759,  # "counter-rotating disk" → adds observations, counterpart* synonyms
    1765,  # "streams and star clusters" → adds "globular cluster", "dwarf"
    1821,  # "TeV blazar observations" → adds multi-wavelength variant
    1836,  # "streams and star clusters" → adds "globular cluster", "dwarf"
    1851,  # "streams and galaxy clusters" → adds "dwarf", "subhalo"
    1965,  # "parallel computing" → adds concurrency, multithread*
    2088,  # "Oumuamua interstellar visitors" → adds "interstellar object*", visitor*
    538,   # "electrophoresis methods" → adds technique* synonym

    # === Fabricated/unrelated abs: terms (filler) ===
    381,   # "papers excluding solar system" → abs:"exoplanet" (made up)
    483,   # "papers with associated data" → abs:"exoplanet" (made up)
    4585,  # "bibliographic group assignments" → abs:"galaxy survey" (made up)
    4592,  # "papers with ORCID identifiers" → abs:"exoplanet" (made up)
    4598,  # "papers with database assignments" → abs:"stellar populations" (made up)

    # === Broken abs: terms (author names/garbage in abs:) ===
    406,   # "Salama polycyclic aromatic hydrocarbons" → abs:", Farid" (author name in abs!)
    1034,  # "Haehnelt and Rees cosmology" → abs:"martin','rees, martin'" (broken)
    1699,  # "research by Vega-Ferrero" → abs:"author=-Ferrero" (broken)
    1840,  # "work by Morales-Olivares" → abs:"-Olivares, O.G" (broken)
}

# Indices to FIX (abbreviation swaps: use the NL term, not an abbreviation)
FIX_EXAMPLES = {
    1489: {
        "old": '"FBOTs"',
        "new": '"fast blue optical transients"',
        "reason": "NL says 'fast blue optical transients', query should match"
    },
    1853: {
        "old_pairs": [
            ('"FBOTs"', '"fast blue optical transients"'),
            ('"GRBs"', '"gamma ray bursts"'),
        ],
        "reason": "NL says full names, query should match"
    },
    1961: {
        "old": '"FBOTs"',
        "new": '"fast blue optical transients"',
        "reason": "NL says 'fast blue optical transients', query should match"
    },
    1991: {
        "old": '"agn"',
        "new": '"active galactic nuclei"',
        "reason": "NL says 'active galactic nuclei', query should match"
    },
}


def main():
    apply = "--apply" in sys.argv

    with open(GOLD_PATH) as f:
        examples = json.load(f)

    print(f"Total examples: {len(examples)}")
    print(f"Examples to remove: {len(REMOVE_INDICES)}")
    print(f"Examples to fix: {len(FIX_EXAMPLES)}")
    print()

    # Load existing removed examples
    if REMOVED_PATH.exists():
        with open(REMOVED_PATH) as f:
            removed_archive = json.load(f)
    else:
        removed_archive = []

    # Show what will be removed
    removed = []
    print("=== REMOVING ===")
    for idx in sorted(REMOVE_INDICES):
        if idx >= len(examples):
            print(f"  [!] Index {idx} out of range (max {len(examples)-1})")
            continue
        ex = examples[idx]
        print(f"  [{idx}] {ex['natural_language'][:80]}")
        print(f"       → {ex['ads_query'][:100]}")
        removed.append({**ex, "_removal_reason": "synonym_expansion_or_fabricated_abs"})

    # Show what will be fixed
    print()
    print("=== FIXING ===")
    for idx, fix in sorted(FIX_EXAMPLES.items()):
        if idx >= len(examples):
            print(f"  [!] Index {idx} out of range")
            continue
        ex = examples[idx]
        old_q = ex["ads_query"]
        if "old_pairs" in fix:
            new_q = old_q
            for old, new in fix["old_pairs"]:
                new_q = new_q.replace(old, new)
        else:
            new_q = old_q.replace(fix["old"], fix["new"])
        print(f"  [{idx}] {fix['reason']}")
        print(f"       old: {old_q[:100]}")
        print(f"       new: {new_q[:100]}")

    if not apply:
        print()
        print("Dry run. Use --apply to commit changes.")
        return

    # Apply fixes first (before removal shifts indices)
    for idx, fix in sorted(FIX_EXAMPLES.items()):
        if idx >= len(examples):
            continue
        old_q = examples[idx]["ads_query"]
        if "old_pairs" in fix:
            for old, new in fix["old_pairs"]:
                old_q = old_q.replace(old, new)
            examples[idx]["ads_query"] = old_q
        else:
            examples[idx]["ads_query"] = old_q.replace(fix["old"], fix["new"])

    # Remove examples (reverse order to preserve indices)
    for idx in sorted(REMOVE_INDICES, reverse=True):
        if idx < len(examples):
            examples.pop(idx)

    # Archive removed examples
    removed_archive.extend(removed)
    with open(REMOVED_PATH, "w") as f:
        json.dump(removed_archive, f, indent=2, ensure_ascii=False)

    # Write updated gold examples
    with open(GOLD_PATH, "w") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print()
    print(f"Removed {len(REMOVE_INDICES)} examples → {len(examples)} remaining")
    print(f"Fixed {len(FIX_EXAMPLES)} abbreviation swaps")
    print(f"Archived to {REMOVED_PATH}")


if __name__ == "__main__":
    main()
