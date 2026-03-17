"""Tests for parse_llm_response() in server.py."""

import json
import sys
from pathlib import Path

import pytest

# Add docker/ and finetune to path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "packages" / "finetune" / "src"))
sys.path.insert(0, str(_project_root / "docker"))

# Import after path setup — but server.py has heavy deps (torch, etc.)
# so we import the function via a focused import
from finetune.domains.scix.intent_spec import IntentSpec


def parse_llm_response(response: str):
    """Inline version of server.parse_llm_response for testing without torch deps."""
    raw = response

    # Strip <think>...</think> block
    think_end = response.find("</think>")
    if think_end >= 0:
        response = response[think_end + len("</think>"):].strip()

    # Try parsing as JSON
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None, response

        data = json.loads(response[json_start:json_end])

        # Old format: {"query": "..."}
        if "query" in data and len(data) == 1:
            return None, data["query"]

        # New format: IntentSpec compact dict
        intent = IntentSpec.from_compact_dict(data)
        if intent.has_content():
            return intent, raw
        return None, response
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, response


class TestParseIntentFormat:
    def test_think_block_with_intent_json(self):
        response = (
            '<think>\nAuthor: Hawking. Topic: black holes.\n</think>\n'
            '{"authors": ["Hawking, S"], "free_text_terms": ["black holes"]}'
        )
        intent, raw = parse_llm_response(response)
        assert intent is not None
        assert intent.authors == ["Hawking, S"]
        assert intent.free_text_terms == ["black holes"]

    def test_intent_json_without_think(self):
        response = '{"authors": ["Hawking, S"], "free_text_terms": ["black holes"]}'
        intent, raw = parse_llm_response(response)
        assert intent is not None
        assert intent.authors == ["Hawking, S"]

    def test_think_block_with_operator(self):
        response = (
            '<think>\nUser wants trending exoplanet papers.\n</think>\n'
            '{"free_text_terms": ["exoplanet"], "property": ["refereed"], "operator": "trending"}'
        )
        intent, raw = parse_llm_response(response)
        assert intent is not None
        assert intent.operator == "trending"
        assert intent.free_text_terms == ["exoplanet"]
        assert intent.property == {"refereed"}

    def test_think_block_with_passthrough(self):
        response = (
            '<think>\nLooking up arXiv ID.\n</think>\n'
            '{"passthrough_clauses": ["arXiv:2301.12345"]}'
        )
        intent, raw = parse_llm_response(response)
        assert intent is not None
        assert "arXiv:2301.12345" in intent.passthrough_clauses


class TestParseOldFormat:
    def test_query_json(self):
        response = '{"query": "abs:\\"dark matter\\" pubdate:[2020 TO 2023]"}'
        intent, clean = parse_llm_response(response)
        assert intent is None
        assert 'abs:"dark matter"' in clean

    def test_think_block_with_query_json(self):
        response = '<think>\nThinking...\n</think>\n{"query": "abs:\\"dark matter\\""}'
        intent, clean = parse_llm_response(response)
        assert intent is None
        assert 'abs:"dark matter"' in clean


class TestParseMalformed:
    def test_not_json(self):
        response = "abs:dark matter pubdate:[2020 TO 2023]"
        intent, clean = parse_llm_response(response)
        assert intent is None
        assert clean == response

    def test_empty_string(self):
        intent, clean = parse_llm_response("")
        assert intent is None
        assert clean == ""

    def test_incomplete_json(self):
        response = '{"authors": ["Hawking"'
        intent, clean = parse_llm_response(response)
        assert intent is None

    def test_think_block_with_garbage(self):
        response = "<think>\nThinking...\n</think>\nnot valid json here"
        intent, clean = parse_llm_response(response)
        assert intent is None

    def test_empty_intent_json(self):
        """Empty JSON object should return None (no content)."""
        response = "{}"
        intent, clean = parse_llm_response(response)
        assert intent is None

    def test_unknown_keys_in_json(self):
        """Extra keys should be filtered, valid fields preserved."""
        response = '{"authors": ["Hawking"], "extra_key": "ignored"}'
        intent, raw = parse_llm_response(response)
        assert intent is not None
        assert intent.authors == ["Hawking"]


class TestIntegration:
    def test_intent_to_assembler(self):
        """Full pipeline: LLM response → parse → assemble → valid query."""
        from finetune.domains.scix.assembler import assemble_query
        from finetune.domains.scix.validate import lint_query

        response = (
            '<think>\nAuthor: Hawking → "Hawking, S". Topic: black holes. '
            'Time: 1970s → 1970-1979.\n</think>\n'
            '{"authors": ["Hawking, S"], "free_text_terms": ["black holes"], '
            '"year_from": 1970, "year_to": 1979}'
        )
        intent, raw = parse_llm_response(response)
        assert intent is not None

        query = assemble_query(intent)
        assert 'author:"Hawking, S"' in query
        assert "black holes" in query
        assert "pubdate:[1970 TO 1979]" in query

        lint_result = lint_query(query)
        assert lint_result.valid
