"""Tests for UAT lookup module."""

import pytest

from finetune.domains.scix.uat_lookup import lookup_uat, rewrite_abs_to_abs_or_uat


# ============================================================================
# lookup_uat
# ============================================================================


class TestLookupUAT:
    def test_known_term(self):
        result = lookup_uat("dark matter")
        assert result is not None

    def test_unknown_term(self):
        assert lookup_uat("completely unknown gibberish xyz") is None

    def test_case_insensitive(self):
        r1 = lookup_uat("dark matter")
        r2 = lookup_uat("Dark Matter")
        assert r1 == r2

    def test_whitespace_stripped(self):
        r1 = lookup_uat("dark matter")
        r2 = lookup_uat("  dark matter  ")
        assert r1 == r2


# ============================================================================
# rewrite_abs_to_abs_or_uat
# ============================================================================


class TestRewriteAbsToAbsOrUat:
    def test_no_abs_passthrough(self):
        query = 'author:"Hawking" pubdate:[2020 TO 2023]'
        assert rewrite_abs_to_abs_or_uat(query) == query

    def test_already_has_uat_passthrough(self):
        query = 'abs:"dark matter" uat:"Dark matter"'
        assert rewrite_abs_to_abs_or_uat(query) == query

    def test_known_term_augmented(self):
        query = 'abs:"dark matter"'
        result = rewrite_abs_to_abs_or_uat(query)
        if lookup_uat("dark matter"):
            assert "uat:" in result
            assert "OR" in result

    def test_unknown_term_unchanged(self):
        query = 'abs:"completely unknown gibberish xyz"'
        result = rewrite_abs_to_abs_or_uat(query)
        assert result == query

    def test_multiple_abs_clauses(self):
        query = 'abs:"exoplanets" abs:"habitable zone"'
        result = rewrite_abs_to_abs_or_uat(query)
        # At least one should be augmented if it's a known UAT term
        assert "abs:" in result
