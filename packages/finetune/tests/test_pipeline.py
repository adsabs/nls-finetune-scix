"""Tests for the NER pipeline orchestration."""

import pytest

from finetune.domains.scix.pipeline import (
    PipelineResult,
    compute_pipeline_confidence,
    is_ads_query,
    process_query,
)
from finetune.domains.scix.intent_spec import IntentSpec


# ============================================================================
# is_ads_query
# ============================================================================


class TestIsADSQuery:
    def test_author_field(self):
        assert is_ads_query('author:"Hawking"')

    def test_abs_field(self):
        assert is_ads_query('abs:"dark matter"')

    def test_operator(self):
        assert is_ads_query('citations(abs:"dark matter")')

    def test_natural_language(self):
        assert not is_ads_query("papers by Hawking on black holes")

    def test_empty(self):
        assert not is_ads_query("")

    def test_mixed(self):
        """Query with ADS field tokens is detected."""
        assert is_ads_query("pubdate:[2020 TO 2023]")


# ============================================================================
# compute_pipeline_confidence
# ============================================================================


class TestComputeConfidence:
    def test_high_with_authors(self):
        intent = IntentSpec(authors=["Hawking"])
        confidence, reason = compute_pipeline_confidence(intent)
        assert confidence == 0.9
        assert reason is None

    def test_high_with_operator(self):
        intent = IntentSpec(operator="citations", free_text_terms=["dark matter"])
        confidence, _ = compute_pipeline_confidence(intent)
        assert confidence == 0.9

    def test_high_with_years(self):
        intent = IntentSpec(year_from=2020, year_to=2023)
        confidence, _ = compute_pipeline_confidence(intent)
        assert confidence == 0.9

    def test_medium_with_constraints(self):
        intent = IntentSpec(doctype={"article"})
        confidence, _ = compute_pipeline_confidence(intent)
        assert confidence == 0.7

    def test_low_with_short_query(self):
        intent = IntentSpec(free_text_terms=["exo"])
        confidence, reason = compute_pipeline_confidence(intent)
        assert confidence <= 0.5
        assert reason is not None


# ============================================================================
# process_query (end-to-end)
# ============================================================================


class TestProcessQuery:
    def test_basic_topic(self):
        result = process_query("dark matter papers")
        assert result.success
        assert "abs:" in result.final_query or "dark matter" in result.final_query

    def test_author_topic(self):
        result = process_query("papers by Hawking on black holes")
        assert result.success
        assert "author:" in result.final_query

    def test_year_range(self):
        result = process_query("papers from 2020 to 2023 on exoplanets")
        assert result.success
        assert "pubdate:" in result.final_query

    def test_operator(self):
        result = process_query("papers citing dark matter research")
        assert result.success
        assert "citations(" in result.final_query

    def test_doctype(self):
        result = process_query("PhD thesis on stellar evolution")
        assert result.success
        assert "doctype:" in result.final_query

    def test_bibgroup(self):
        result = process_query("JWST observations of exoplanets")
        assert result.success
        assert "bibgroup:" in result.final_query

    def test_empty_input(self):
        result = process_query("")
        assert result.success  # Should not crash

    def test_debug_info_populated(self):
        result = process_query("papers by Hawking since 2020")
        assert result.debug_info.ner_time_ms > 0
        assert result.debug_info.total_time_ms > 0

    def test_confidence_set(self):
        result = process_query("papers by Hawking on black holes")
        assert result.confidence > 0

    def test_intent_preserved(self):
        result = process_query("papers by Hawking")
        assert result.intent is not None
        assert "Hawking" in result.intent.authors
