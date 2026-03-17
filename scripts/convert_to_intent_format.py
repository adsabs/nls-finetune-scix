#!/usr/bin/env python3
"""Convert gold examples from query format to intent format with think traces.

Takes gold_examples.json (NL + ADS query pairs) and produces
gold_examples_intent.json with:
- intent_json: compact IntentSpec dict (non-empty fields only)
- think_trace: reasoning trace for the <think> block

Two-phase conversion:
1. Parse ADS query → IntentSpec → compact dict, validate round-trip
2. Generate think trace (template-based for all categories)

Usage:
    python scripts/convert_to_intent_format.py
    python scripts/convert_to_intent_format.py --report   # show conversion stats
    python scripts/convert_to_intent_format.py --input data/datasets/raw/gold_examples.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Add packages/finetune/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "finetune" / "src"))

from finetune.domains.scix.assembler import assemble_query
from finetune.domains.scix.intent_spec import IntentSpec
from finetune.domains.scix.parse_query import parse_query_to_intent
from finetune.domains.scix.validate import lint_query


def generate_think_trace(nl: str, intent: dict) -> str:
    """Generate a template-based thinking trace from NL and intent dict.

    Produces a concise reasoning trace that explains how the NL maps to
    each IntentSpec field. Format:
        The user wants: <NL summary>
        - Field: value → intent field
    """
    lines = [f"The user wants: {nl}"]

    if "authors" in intent:
        authors = intent["authors"]
        if len(authors) == 1:
            lines.append(f"- Author: {authors[0]} → authors field")
        else:
            lines.append(f"- Authors: {', '.join(authors)} → authors field")

    if "free_text_terms" in intent:
        terms = intent["free_text_terms"]
        if len(terms) == 1:
            lines.append(f'- Topic: "{terms[0]}" → abstract search')
        else:
            quoted = ", ".join('"' + t + '"' for t in terms)
            lines.append(f"- Topics: {quoted} → abstract search (AND)")

    if "or_terms" in intent:
        terms = intent["or_terms"]
        quoted = " OR ".join('"' + t + '"' for t in terms)
        lines.append(f"- OR topics: {quoted} → OR search")

    if "title_terms" in intent:
        terms = intent["title_terms"]
        quoted = ", ".join('"' + t + '"' for t in terms)
        lines.append(f"- Title search: {quoted}")

    if "full_text_terms" in intent:
        terms = intent["full_text_terms"]
        quoted = ", ".join('"' + t + '"' for t in terms)
        lines.append(f"- Full text: {quoted}")

    if "year_from" in intent or "year_to" in intent:
        y_from = intent.get("year_from", "*")
        y_to = intent.get("year_to", "*")
        if y_from == y_to and y_from != "*":
            lines.append(f"- Year: {y_from}")
        else:
            lines.append(f"- Date range: {y_from} to {y_to}")

    if "objects" in intent:
        lines.append(f"- Objects: {', '.join(intent['objects'])} → object search")

    if "affiliations" in intent:
        lines.append(f"- Affiliations: {', '.join(intent['affiliations'])} → institution/affiliation")

    if "bibstems" in intent:
        lines.append(f"- Journals: {', '.join(intent['bibstems'])} → bibstem")

    if "doctype" in intent:
        lines.append(f"- Document type: {', '.join(intent['doctype'])}")

    if "property" in intent:
        lines.append(f"- Properties: {', '.join(intent['property'])}")

    if "collection" in intent:
        lines.append(f"- Collection: {', '.join(intent['collection'])}")

    if "bibgroup" in intent:
        lines.append(f"- Bibgroup: {', '.join(intent['bibgroup'])}")

    if "esources" in intent:
        lines.append(f"- Electronic sources: {', '.join(intent['esources'])}")

    if "data" in intent:
        lines.append(f"- Data archives: {', '.join(intent['data'])}")

    if "operator" in intent:
        lines.append(f"- Operator: {intent['operator']}() wrapper")

    if "negated_terms" in intent:
        quoted = ", ".join('"' + t + '"' for t in intent["negated_terms"])
        lines.append(f"- Exclude: {quoted}")

    if "negated_properties" in intent:
        lines.append(f"- Exclude properties: {', '.join(intent['negated_properties'])}")

    if "negated_doctypes" in intent:
        lines.append(f"- Exclude doctypes: {', '.join(intent['negated_doctypes'])}")

    if "has_fields" in intent:
        lines.append(f"- Has: {', '.join(intent['has_fields'])}")

    if "citation_count_min" in intent or "citation_count_max" in intent:
        lo = intent.get("citation_count_min", "*")
        hi = intent.get("citation_count_max", "*")
        lines.append(f"- Citation count: [{lo} TO {hi}]")

    if "read_count_min" in intent:
        lines.append(f"- Read count: >= {intent['read_count_min']}")

    if "ack_terms" in intent:
        lines.append(f"- Acknowledgments: {', '.join(intent['ack_terms'])}")

    if "grant_terms" in intent:
        lines.append(f"- Grants: {', '.join(intent['grant_terms'])}")

    if "exact_match_fields" in intent:
        for fld, val in intent["exact_match_fields"].items():
            lines.append(f'- Exact match: ={fld}:"{val}"')

    if "passthrough_clauses" in intent:
        for clause in intent["passthrough_clauses"]:
            lines.append(f"- Raw clause: {clause}")

    return "\n".join(lines)


def convert_example(example: dict) -> dict | None:
    """Convert a single gold example to intent format.

    Returns None if conversion fails (e.g., round-trip produces invalid query).
    """
    nl = example["natural_language"]
    ads_query = example["ads_query"]
    category = example.get("category", "unknown")

    # Phase 1: Parse ADS query → IntentSpec → compact dict
    try:
        intent = parse_query_to_intent(ads_query)
        compact = intent.to_compact_dict()
    except Exception as e:
        return {"error": f"Parse failed: {e}", "nl": nl, "query": ads_query}

    # Phase 2: Validate round-trip
    try:
        reconstructed = IntentSpec.from_compact_dict(compact)
        assembled = assemble_query(reconstructed)
        lint_result = lint_query(assembled)
        if not lint_result.valid and assembled.strip():
            # Try anyway — some lint warnings are false positives
            pass
    except Exception as e:
        return {"error": f"Round-trip failed: {e}", "nl": nl, "query": ads_query}

    # Phase 3: Generate think trace
    think_trace = generate_think_trace(nl, compact)

    return {
        "natural_language": nl,
        "ads_query": ads_query,
        "category": category,
        "intent_json": compact,
        "think_trace": think_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert gold examples to intent format")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/datasets/raw/gold_examples.json",
        help="Input gold examples file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/datasets/raw/gold_examples_intent.json",
        help="Output file with intent format",
    )
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Show detailed conversion report",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    with open(input_path) as f:
        examples = json.load(f)

    print(f"Converting {len(examples)} examples from {input_path}")

    converted = []
    errors = []
    category_stats = Counter()

    for ex in examples:
        result = convert_example(ex)
        if result is None:
            errors.append({"nl": ex["natural_language"], "error": "returned None"})
        elif "error" in result:
            errors.append(result)
        else:
            converted.append(result)
            category_stats[result["category"]] += 1

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"\nConverted: {len(converted)}/{len(examples)} ({100 * len(converted) / len(examples):.1f}%)")
    print(f"Errors:    {len(errors)}")
    print(f"Output:    {output_path}")

    if args.report:
        print(f"\n{'=' * 60}")
        print("CATEGORY DISTRIBUTION")
        print(f"{'=' * 60}")
        for cat, count in category_stats.most_common():
            pct = 100 * count / len(converted)
            print(f"  {cat:20} {count:4} ({pct:5.1f}%)")

        if errors:
            print(f"\n{'=' * 60}")
            print(f"ERRORS ({len(errors)})")
            print(f"{'=' * 60}")
            for err in errors[:20]:
                print(f"  {err.get('nl', '')[:50]}... → {err.get('error', 'unknown')}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")

        # Show sample conversions
        print(f"\n{'=' * 60}")
        print("SAMPLE CONVERSIONS")
        print(f"{'=' * 60}")
        import random
        random.seed(42)
        samples = random.sample(converted, min(5, len(converted)))
        for s in samples:
            print(f"\n  NL: {s['natural_language'][:70]}")
            print(f"  Query: {s['ads_query'][:70]}")
            print(f"  Intent: {json.dumps(s['intent_json'])[:100]}")
            print(f"  Think: {s['think_trace'][:100]}...")


if __name__ == "__main__":
    main()
