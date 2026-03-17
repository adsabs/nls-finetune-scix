"""Tests for the intent-level merge module (NER + NLS hybrid merge)."""

import pytest

from finetune.domains.scix.merge import (
    MergeResult,
    merge_intents,
    merge_ner_and_nls,
    merge_ner_and_nls_intent,
)
from finetune.domains.scix.intent_spec import IntentSpec
from finetune.domains.scix.pipeline import PipelineResult, DebugInfo


# ============================================================================
# merge_intents (unit tests for per-field policies)
# ============================================================================


class TestMergeIntents:
    def test_authors_prefer_ner(self):
        ner = IntentSpec(authors=["Hawking, S"])
        llm = IntentSpec(authors=["Hawking"])
        merged = merge_intents(ner, llm, "papers by Hawking")
        assert merged.authors == ["Hawking, S"]

    def test_authors_fallback_to_llm(self):
        ner = IntentSpec()
        llm = IntentSpec(authors=["Hawking"])
        merged = merge_intents(ner, llm, "papers by Hawking")
        assert merged.authors == ["Hawking"]

    def test_free_text_prefer_llm(self):
        ner = IntentSpec(free_text_terms=["dark matter"])
        llm = IntentSpec(free_text_terms=["dark matter annihilation"])
        merged = merge_intents(ner, llm, "dark matter annihilation")
        assert merged.free_text_terms == ["dark matter annihilation"]

    def test_free_text_fallback_to_ner(self):
        ner = IntentSpec(free_text_terms=["dark matter"])
        llm = IntentSpec()
        merged = merge_intents(ner, llm, "dark matter")
        assert merged.free_text_terms == ["dark matter"]

    def test_years_prefer_ner(self):
        ner = IntentSpec(year_from=2020, year_to=2023)
        llm = IntentSpec(year_from=2023, year_to=2026)
        merged = merge_intents(ner, llm, "papers since 2020")
        assert merged.year_from == 2020
        assert merged.year_to == 2023

    def test_years_fallback_to_llm(self):
        ner = IntentSpec()
        llm = IntentSpec(year_from=2020, year_to=2023)
        merged = merge_intents(ner, llm, "papers since 2020")
        assert merged.year_from == 2020

    def test_affiliations_prefer_ner(self):
        ner = IntentSpec(affiliations=["CfA"])
        llm = IntentSpec(affiliations=["CfA"])
        merged = merge_intents(ner, llm, "CfA papers")
        assert merged.affiliations == ["CfA"]

    def test_bibstems_prefer_ner(self):
        ner = IntentSpec(bibstems=["ApJ"])
        llm = IntentSpec(bibstems=["ApJ"])
        merged = merge_intents(ner, llm, "ApJ papers")
        assert merged.bibstems == ["ApJ"]

    def test_operator_prefer_llm(self):
        ner = IntentSpec(operator="citations")
        llm = IntentSpec(operator="trending")
        merged = merge_intents(ner, llm, "trending papers")
        assert merged.operator == "trending"

    def test_enum_fields_union(self):
        ner = IntentSpec(doctype={"article"})
        llm = IntentSpec(property={"refereed"})
        merged = merge_intents(ner, llm, "refereed articles")
        assert "article" in merged.doctype
        assert "refereed" in merged.property

    def test_negation_from_llm(self):
        ner = IntentSpec()
        llm = IntentSpec(negated_terms=["axion"])
        merged = merge_intents(ner, llm, "dark matter not axions")
        assert "axion" in merged.negated_terms

    def test_has_fields_from_llm(self):
        ner = IntentSpec()
        llm = IntentSpec(has_fields={"body"})
        merged = merge_intents(ner, llm, "full-text papers")
        assert "body" in merged.has_fields

    def test_citation_count_from_llm(self):
        ner = IntentSpec()
        llm = IntentSpec(citation_count_min=100)
        merged = merge_intents(ner, llm, "highly cited papers")
        assert merged.citation_count_min == 100

    def test_passthrough_from_llm(self):
        ner = IntentSpec()
        llm = IntentSpec(passthrough_clauses=["orcid:0000-0001-2345-6789"])
        merged = merge_intents(ner, llm, "papers by orcid")
        assert "orcid:0000-0001-2345-6789" in merged.passthrough_clauses


class TestStripDefaultDoctype:
    def test_generic_papers_strips_article(self):
        ner = IntentSpec()
        llm = IntentSpec(doctype={"article"}, free_text_terms=["dark matter"])
        merged = merge_intents(ner, llm, "papers on dark matter")
        assert "article" not in merged.doctype

    def test_explicit_journal_article_kept(self):
        ner = IntentSpec()
        llm = IntentSpec(doctype={"article"}, free_text_terms=["dark matter"])
        merged = merge_intents(ner, llm, "journal articles on dark matter")
        assert "article" in merged.doctype

    def test_other_doctypes_not_stripped(self):
        ner = IntentSpec()
        llm = IntentSpec(doctype={"eprint"}, free_text_terms=["dark matter"])
        merged = merge_intents(ner, llm, "papers on dark matter")
        assert "eprint" in merged.doctype


class TestRemoveInstFromFreeText:
    def test_cfa_removed_from_topics(self):
        ner = IntentSpec(affiliations=["CfA"])
        llm = IntentSpec(free_text_terms=["hubble tension cfa"])
        merged = merge_intents(ner, llm, "hubble tension from cfa")
        # "cfa" should be removed from free_text, "hubble tension" kept
        topics = " ".join(merged.free_text_terms).lower()
        assert "hubble" in topics
        assert "tension" in topics


class TestDedupBibgroupAndAff:
    def test_cfa_bibgroup_removed_when_aff_present(self):
        ner = IntentSpec(affiliations=["CfA"], bibgroup={"CfA"})
        llm = IntentSpec(free_text_terms=["hubble tension"])
        merged = merge_intents(ner, llm, "hubble tension from CfA")
        assert "CfA" not in merged.bibgroup
        assert "CfA" in merged.affiliations

    def test_non_overlapping_bibgroup_kept(self):
        ner = IntentSpec(affiliations=["CfA"], bibgroup={"HST"})
        llm = IntentSpec(free_text_terms=["galaxies"])
        merged = merge_intents(ner, llm, "HST galaxies from CfA")
        assert "HST" in merged.bibgroup


# ============================================================================
# merge_ner_and_nls (integration tests)
# ============================================================================


def _make_pipeline_result(query: str, intent: IntentSpec | None = None) -> PipelineResult:
    """Helper to create a PipelineResult for testing."""
    if intent is None:
        intent = IntentSpec()
    return PipelineResult(
        intent=intent,
        retrieved_examples=[],
        final_query=query,
        debug_info=DebugInfo(),
        confidence=0.8,
    )


class TestMergeNerAndNls:
    def test_both_valid_produces_clean_query(self):
        """Both NER and NLS valid → intent merge → assembler output."""
        nls = 'author:"Hawking" abs:"black holes" pubdate:[2020 TO 2023]'
        intent = IntentSpec(
            authors=["Hawking"],
            free_text_terms=["black holes"],
            year_from=2020,
            year_to=2023,
        )
        ner = _make_pipeline_result('author:"Hawking" abs:"black holes" pubdate:[2020 TO 2023]', intent)
        result = merge_ner_and_nls(ner, nls, "papers by Hawking on black holes since 2020")
        assert 'author:"Hawking"' in result.query
        assert "black holes" in result.query
        assert "pubdate:[2020 TO 2023]" in result.query

    def test_nls_empty_falls_back_to_ner(self):
        intent = IntentSpec(free_text_terms=["dark matter"])
        ner = _make_pipeline_result('abs:"dark matter"', intent)
        result = merge_ner_and_nls(ner, "", "dark matter papers")
        assert result.source == "ner_only"
        assert "dark matter" in result.query

    def test_ner_empty_uses_nls(self):
        intent = IntentSpec()
        ner = _make_pipeline_result("", intent)
        result = merge_ner_and_nls(ner, 'abs:"dark matter"', "dark matter papers")
        assert result.source == "nls_only"
        assert "dark matter" in result.query

    def test_both_empty(self):
        intent = IntentSpec()
        ner = _make_pipeline_result("", intent)
        result = merge_ner_and_nls(ner, "", "")
        assert result.confidence < 0.5

    def test_nls_invalid_syntax_falls_back_to_ner(self):
        intent = IntentSpec(free_text_terms=["dark matter"])
        ner = _make_pipeline_result('abs:"dark matter"', intent)
        result = merge_ner_and_nls(ner, 'abs:("dark matter"', "dark matter papers")
        assert result.source == "ner_only"

    def test_negation_preserved_from_nls(self):
        nls = 'abs:"dark matter" NOT abs:"axion"'
        intent = IntentSpec(free_text_terms=["dark matter"])
        ner = _make_pipeline_result('abs:"dark matter"', intent)
        result = merge_ner_and_nls(ner, nls, "dark matter papers excluding axions")
        assert "NOT" in result.query
        assert "axion" in result.query

    def test_ner_injects_author(self):
        """NER has author, NLS doesn't → author injected."""
        nls = 'abs:"black holes" pubdate:[2020 TO 2023]'
        intent = IntentSpec(
            authors=["Hawking"],
            free_text_terms=["black holes"],
            year_from=2020,
            year_to=2023,
        )
        ner = _make_pipeline_result('author:"Hawking" abs:"black holes" pubdate:[2020 TO 2023]', intent)
        result = merge_ner_and_nls(ner, nls, "papers by Hawking on black holes since 2020")
        assert 'author:"Hawking"' in result.query
        assert "author" in result.fields_injected

    def test_ner_result_none(self):
        result = merge_ner_and_nls(None, 'abs:"dark matter"', "dark matter")
        assert result.source == "nls_only"
        assert "dark matter" in result.query

    def test_abs_and_clause_cleaned(self):
        """abs:(w1 AND w2) in LLM output → assembler produces abs:"w1 w2"."""
        nls = 'abs:(hubble AND tension) pubdate:[2023 TO 2026] (inst:"CfA" OR aff:"CfA")'
        intent = IntentSpec(
            free_text_terms=["hubble tension"],
            affiliations=["CfA"],
            year_from=2023,
            year_to=2026,
        )
        ner = _make_pipeline_result(
            'abs:"hubble tension" (inst:"CfA" OR aff:"CfA") pubdate:[2023 TO 2026]',
            intent,
        )
        result = merge_ner_and_nls(ner, nls, "hubble tension papers from CfA")
        # The assembler should produce clean syntax, not abs:(w1 AND w2)
        assert "abs:(hubble AND tension)" not in result.query

    def test_has_field_preserved(self):
        nls = 'abs:"dark matter" has:body pubdate:[2020 TO 2023]'
        intent = IntentSpec(
            free_text_terms=["dark matter"],
            year_from=2020,
            year_to=2023,
        )
        ner = _make_pipeline_result('abs:"dark matter" pubdate:[2020 TO 2023]', intent)
        result = merge_ner_and_nls(ner, nls, "full-text dark matter papers since 2020")
        assert "has:body" in result.query

    def test_intent_merge_both_valid(self):
        """merge_ner_and_nls_intent with both NER and LLM intents."""
        llm_intent = IntentSpec(
            free_text_terms=["black holes"],
            year_from=2020,
            year_to=2023,
        )
        ner_intent = IntentSpec(
            authors=["Hawking, S"],
            free_text_terms=["black holes"],
            year_from=2020,
            year_to=2023,
        )
        ner = _make_pipeline_result('author:"Hawking, S" abs:"black holes" pubdate:[2020 TO 2023]', ner_intent)
        result = merge_ner_and_nls_intent(ner, llm_intent, "papers by Hawking on black holes since 2020")
        assert 'author:"Hawking, S"' in result.query
        assert "black holes" in result.query
        assert "author" in result.fields_injected

    def test_intent_merge_no_ner(self):
        """merge_ner_and_nls_intent with no NER result."""
        llm_intent = IntentSpec(
            free_text_terms=["dark matter"],
            property={"refereed"},
        )
        result = merge_ner_and_nls_intent(None, llm_intent, "refereed dark matter papers")
        assert result.source == "nls_only"
        assert "dark matter" in result.query
        assert "property:refereed" in result.query

    def test_intent_merge_negation(self):
        """Negation fields from LLM intent are preserved."""
        llm_intent = IntentSpec(
            free_text_terms=["dark matter"],
            negated_terms=["axion"],
        )
        ner_intent = IntentSpec(free_text_terms=["dark matter"])
        ner = _make_pipeline_result('abs:"dark matter"', ner_intent)
        result = merge_ner_and_nls_intent(ner, llm_intent, "dark matter not axions")
        assert "NOT" in result.query
        assert "axion" in result.query

    def test_intent_merge_empty_ner_intent(self):
        """NER exists but has empty intent → LLM-only."""
        llm_intent = IntentSpec(free_text_terms=["dark matter"])
        ner_intent = IntentSpec()  # empty
        ner = _make_pipeline_result("", ner_intent)
        result = merge_ner_and_nls_intent(ner, llm_intent, "dark matter")
        assert result.source == "nls_only"
        assert "dark matter" in result.query

    def test_operator_preserved(self):
        nls = 'citations(abs:"dark matter")'
        intent = IntentSpec(
            free_text_terms=["dark matter"],
            year_from=2020,
            year_to=2024,
            operator="citations",
        )
        ner = _make_pipeline_result('citations(abs:"dark matter") pubdate:[2020 TO 2024]', intent)
        result = merge_ner_and_nls(ner, nls, "papers citing dark matter since 2020")
        assert "citations(" in result.query
