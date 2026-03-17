"""Tests for parse_query module (ADS query string → IntentSpec)."""

import pytest

from finetune.domains.scix.parse_query import parse_query_to_intent


class TestParseAuthors:
    def test_single_author(self):
        intent = parse_query_to_intent('author:"Hawking"')
        assert "Hawking" in intent.authors

    def test_multiple_authors(self):
        intent = parse_query_to_intent('author:"Hawking" author:"Penrose"')
        assert "Hawking" in intent.authors
        assert "Penrose" in intent.authors

    def test_first_author(self):
        intent = parse_query_to_intent('author:"^Hawking"')
        assert "^Hawking" in intent.authors

    def test_author_with_initial(self):
        intent = parse_query_to_intent('author:"Hawking, S"')
        assert "Hawking, S" in intent.authors


class TestParseAbstract:
    def test_quoted_phrase(self):
        intent = parse_query_to_intent('abs:"dark matter"')
        assert "dark matter" in intent.free_text_terms

    def test_bare_word(self):
        intent = parse_query_to_intent("abs:exoplanets")
        assert "exoplanets" in intent.free_text_terms

    def test_and_clause_normalized(self):
        """abs:(w1 AND w2) should be parsed as a single phrase."""
        intent = parse_query_to_intent("abs:(hubble AND tension)")
        assert any("hubble" in t and "tension" in t for t in intent.free_text_terms)

    def test_multiple_abs(self):
        intent = parse_query_to_intent('abs:"dark matter" abs:"galaxy clusters"')
        assert "dark matter" in intent.free_text_terms
        assert "galaxy clusters" in intent.free_text_terms

    def test_or_list_goes_to_or_terms(self):
        """abs:(X OR Y OR Z) should be parsed into or_terms, not free_text_terms."""
        intent = parse_query_to_intent('abs:("black hole" OR BH OR singularity)')
        assert "black hole" in intent.or_terms
        assert "BH" in intent.or_terms
        assert "singularity" in intent.or_terms
        assert not intent.free_text_terms  # Should NOT be in free_text_terms


class TestParseYears:
    def test_pubdate_range(self):
        intent = parse_query_to_intent("pubdate:[2020 TO 2023]")
        assert intent.year_from == 2020
        assert intent.year_to == 2023

    def test_pubdate_from_only(self):
        intent = parse_query_to_intent("pubdate:[2020 TO *]")
        assert intent.year_from == 2020
        assert intent.year_to is None

    def test_pubdate_to_only(self):
        intent = parse_query_to_intent("pubdate:[* TO 2023]")
        assert intent.year_from is None
        assert intent.year_to == 2023

    def test_now_range_passthrough(self):
        """NOW- ranges can't be decomposed to year ints, so passthrough."""
        intent = parse_query_to_intent("pubdate:[NOW-5YEARS TO *]")
        assert any("NOW" in c for c in intent.passthrough_clauses)

    def test_pubdate_with_month(self):
        intent = parse_query_to_intent("pubdate:[2020-01 TO 2023-12]")
        assert intent.year_from == 2020
        assert intent.year_to == 2023


class TestParseBibstem:
    def test_quoted_bibstem(self):
        intent = parse_query_to_intent('bibstem:"ApJ"')
        assert "ApJ" in intent.bibstems

    def test_unquoted_bibstem(self):
        intent = parse_query_to_intent("bibstem:MNRAS")
        assert "MNRAS" in intent.bibstems

    def test_or_list(self):
        intent = parse_query_to_intent('bibstem:(ApJ OR MNRAS)')
        assert "ApJ" in intent.bibstems
        assert "MNRAS" in intent.bibstems


class TestParseObjects:
    def test_simple_object(self):
        intent = parse_query_to_intent("object:M31")
        assert "M31" in intent.objects

    def test_quoted_object(self):
        intent = parse_query_to_intent('object:"NGC 1234"')
        assert "NGC 1234" in intent.objects


class TestParseAffiliations:
    def test_inst_or_aff_pattern(self):
        intent = parse_query_to_intent('(inst:"CfA" OR aff:"CfA")')
        assert "CfA" in intent.affiliations

    def test_plain_aff(self):
        intent = parse_query_to_intent('aff:"Berkeley"')
        assert "Berkeley" in intent.affiliations

    def test_plain_inst(self):
        intent = parse_query_to_intent('inst:"MIT"')
        assert "MIT" in intent.affiliations


class TestParseEnumFields:
    def test_doctype(self):
        intent = parse_query_to_intent("doctype:article")
        assert "article" in intent.doctype

    def test_invalid_doctype_rejected(self):
        intent = parse_query_to_intent("doctype:INVALID_VALUE")
        assert len(intent.doctype) == 0

    def test_property(self):
        intent = parse_query_to_intent("property:refereed")
        assert "refereed" in intent.property

    def test_database(self):
        intent = parse_query_to_intent("database:astronomy")
        assert "astronomy" in intent.collection

    def test_bibgroup(self):
        intent = parse_query_to_intent("bibgroup:LIGO")
        assert "LIGO" in intent.bibgroup

    def test_esources(self):
        intent = parse_query_to_intent("esources:PUB_PDF")
        assert "PUB_PDF" in intent.esources

    def test_data(self):
        intent = parse_query_to_intent("data:MAST")
        assert "MAST" in intent.data

    def test_has(self):
        intent = parse_query_to_intent("has:body")
        assert "body" in intent.has_fields


class TestParseNegation:
    def test_not_abs(self):
        intent = parse_query_to_intent('abs:"dark matter" NOT abs:"axion"')
        assert "axion" in intent.negated_terms
        assert "dark matter" in intent.free_text_terms

    def test_not_property(self):
        intent = parse_query_to_intent("abs:exoplanets NOT property:nonarticle")
        assert "nonarticle" in intent.negated_properties

    def test_not_doctype(self):
        intent = parse_query_to_intent("abs:cosmology NOT doctype:eprint")
        assert "eprint" in intent.negated_doctypes


class TestParseOperators:
    def test_citations(self):
        intent = parse_query_to_intent('citations(abs:"dark matter")')
        assert intent.operator == "citations"
        assert "dark matter" in intent.free_text_terms

    def test_trending(self):
        intent = parse_query_to_intent('trending(abs:"exoplanets")')
        assert intent.operator == "trending"

    def test_similar(self):
        intent = parse_query_to_intent('similar(abs:"black holes")')
        assert intent.operator == "similar"


class TestParseAdvancedFields:
    def test_grant(self):
        intent = parse_query_to_intent("grant:NASA")
        assert "NASA" in intent.grant_terms

    def test_ack(self):
        intent = parse_query_to_intent('ack:"Canadian Space Agency"')
        assert "Canadian Space Agency" in intent.ack_terms

    def test_citation_count(self):
        intent = parse_query_to_intent("citation_count:[100 TO *]")
        assert intent.citation_count_min == 100
        assert intent.citation_count_max is None

    def test_full(self):
        intent = parse_query_to_intent('full:"MUSE"')
        assert "MUSE" in intent.full_text_terms

    def test_title(self):
        intent = parse_query_to_intent('title:"exact phrase"')
        assert "exact phrase" in intent.title_terms

    def test_exact_match(self):
        intent = parse_query_to_intent('=keyword:"accretion"')
        assert intent.exact_match_fields.get("keyword") == "accretion"


class TestParseComplexQueries:
    def test_full_query(self):
        q = 'author:"Hawking" abs:"black holes" pubdate:[2020 TO 2023] doctype:article property:refereed'
        intent = parse_query_to_intent(q)
        assert "Hawking" in intent.authors
        assert "black holes" in intent.free_text_terms
        assert intent.year_from == 2020
        assert intent.year_to == 2023
        assert "article" in intent.doctype
        assert "refereed" in intent.property

    def test_cfa_problem_query(self):
        """The CfA query that motivated the rewrite."""
        q = 'abs:(hubble AND tension AND cfa) pubdate:[2023-01 TO 2026-12] doctype:article (inst:"CfA" OR aff:"CfA")'
        intent = parse_query_to_intent(q)
        assert "CfA" in intent.affiliations
        assert intent.year_from == 2023
        # "cfa" should be in free_text as part of the AND phrase
        # (the intent-level merge will remove it later)

    def test_operator_with_fields(self):
        q = 'citations(abs:"gravitational waves" pubdate:[2020 TO 2023])'
        intent = parse_query_to_intent(q)
        assert intent.operator == "citations"
        assert "gravitational waves" in intent.free_text_terms
        assert intent.year_from == 2020

    def test_empty_query(self):
        intent = parse_query_to_intent("")
        assert not intent.has_content()

    def test_unknown_field_passthrough(self):
        q = 'orcid:0000-0001-2345-6789 abs:"dark matter"'
        intent = parse_query_to_intent(q)
        assert "dark matter" in intent.free_text_terms
        assert any("orcid" in c for c in intent.passthrough_clauses)


class TestRoundTrip:
    """Test that parse → assemble produces clean output."""

    def test_simple_roundtrip(self):
        from finetune.domains.scix.assembler import assemble_query

        q = 'author:"Hawking" abs:"black holes" pubdate:[2020 TO 2023]'
        intent = parse_query_to_intent(q)
        result = assemble_query(intent)
        assert 'author:"Hawking"' in result
        assert "black holes" in result
        assert "pubdate:[2020 TO 2023]" in result

    def test_and_clause_cleaned(self):
        """abs:(w1 AND w2) should be assembled as abs:"w1 w2"."""
        from finetune.domains.scix.assembler import assemble_query

        q = "abs:(hubble AND tension) pubdate:[2023 TO 2026]"
        intent = parse_query_to_intent(q)
        result = assemble_query(intent)
        assert 'abs:"hubble tension"' in result
        assert "AND" not in result

    def test_negation_roundtrip(self):
        from finetune.domains.scix.assembler import assemble_query

        q = 'abs:"dark matter" NOT abs:"axion"'
        intent = parse_query_to_intent(q)
        result = assemble_query(intent)
        assert "dark matter" in result
        assert "NOT" in result
        assert "axion" in result
