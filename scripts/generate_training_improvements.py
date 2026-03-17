#!/usr/bin/env python3
"""
Generate training examples for coverage gaps identified in the March 2026 audit.

Categories:
1. Pasted reference parsing (bibcode, DOI, arXiv ID, formatted citations)
2. UAT field examples
3. Negation patterns
4. Software/data mention patterns (abs: + mention_count:/credit_count:)
5. Date expression diversity (relative dates)
6. caption: figure/table search
7. arxiv: identifier lookup
8. author_count:/page_count: filters

Usage:
    python scripts/generate_training_improvements.py \
        --output data/datasets/generated/training_improvements.json
"""

import argparse
import json
import random
from pathlib import Path


def generate_reference_parsing_examples() -> list[dict]:
    """Pasted reference parsing — users paste citations from papers."""
    examples = []

    # Bibcode lookups
    bibcode_examples = [
        ("2016PhRvL.116f1102A", "the LIGO gravitational wave detection paper"),
        ("1998AJ....116.1009R", "the Riess et al. 1998 Type Ia supernovae paper"),
        ("2020Natur.581..147A", "the paper with bibcode 2020Natur.581..147A"),
        ("2019ApJ...882L..12M", "find 2019ApJ...882L..12M"),
        ("2015A&A...594A..13P", "the Planck 2015 cosmological parameters paper"),
        ("1979Natur.277..437H", "the Hawking radiation paper in Nature 1979"),
        ("2003ARA&A..41..191M", "the Madau & Dickinson star formation history review"),
        ("2011Natur.474..616M", "the paper 2011Natur.474..616M"),
    ]
    for bibcode, nl in bibcode_examples:
        examples.append({
            "natural_language": nl,
            "ads_query": f'bibcode:{bibcode}',
            "category": "identifier",
        })

    # DOI lookups
    doi_examples = [
        ("10.1038/nature12917", "the paper with DOI 10.1038/nature12917"),
        ("10.3847/1538-4357/ab4a11", "find doi:10.3847/1538-4357/ab4a11"),
        ("10.1086/345794", "paper with DOI 10.1086/345794"),
        ("10.1093/mnras/staa2532", "the MNRAS paper doi 10.1093/mnras/staa2532"),
        ("10.1126/science.aal3672", "find the Science paper 10.1126/science.aal3672"),
    ]
    for doi, nl in doi_examples:
        examples.append({
            "natural_language": nl,
            "ads_query": f'doi:"{doi}"',
            "category": "identifier",
        })

    # arXiv ID lookups
    arxiv_examples = [
        ("2301.00001", "find arXiv paper 2301.00001"),
        ("1609.04747", "the arXiv preprint 1609.04747"),
        ("astro-ph/0401001", "find astro-ph/0401001"),
        ("2106.15163", "arXiv:2106.15163"),
        ("2205.01397", "find the paper arXiv 2205.01397"),
    ]
    for arxiv_id, nl in arxiv_examples:
        examples.append({
            "natural_language": nl,
            "ads_query": f'arxiv:{arxiv_id}',
            "category": "identifier",
        })

    # Formatted citation parsing — user pastes a reference string
    citation_examples = [
        (
            "Riess et al. 1998, AJ, 116, 1009",
            'author:"Riess" bibstem:"AJ" volume:116 page:1009 pubdate:1998',
            # Note: will be corrected to volume:116 in review
        ),
        (
            "Hawking, S. W. 1974, Nature, 248, 30",
            'author:"Hawking, S" bibstem:"Natur" volume:248 page:30 pubdate:1974',
        ),
        (
            "Madau & Dickinson 2014, ARA&A, 52, 415",
            'author:"Madau" author:"Dickinson" bibstem:"ARA&A" volume:52 page:415 pubdate:2014',
        ),
        (
            "Planck Collaboration 2020, A&A, 641, A6",
            'author:"Planck Collaboration" bibstem:"A&A" volume:641 page:A6 pubdate:2020',
        ),
        (
            "Treu, T. 2010, ARA&A, 48, 87",
            'author:"Treu, T" bibstem:"ARA&A" volume:48 page:87 pubdate:2010',
        ),
        (
            "Abbott et al., ApJL, 848, L12, 2017",
            'author:"Abbott" bibstem:"ApJL" volume:848 page:L12 pubdate:2017',
        ),
        (
            "Kennicutt 1998 ApJ 498 541",
            'author:"Kennicutt" bibstem:"ApJ" volume:498 page:541 pubdate:1998',
        ),
        (
            "Navarro, Frenk & White, 1997, ApJ, 490, 493",
            'author:"Navarro" author:"Frenk" author:"White" bibstem:"ApJ" volume:490 page:493 pubdate:1997',
        ),
        (
            "Salpeter 1955 ApJ 121 161",
            'author:"Salpeter" bibstem:"ApJ" volume:121 page:161 pubdate:1955',
        ),
        (
            "Press & Schechter, 1974, ApJ, 187, 425",
            'author:"Press" author:"Schechter" bibstem:"ApJ" volume:187 page:425 pubdate:1974',
        ),
        (
            "Chabrier, G. 2003, PASP, 115, 763",
            'author:"Chabrier, G" bibstem:"PASP" volume:115 page:763 pubdate:2003',
        ),
        (
            "Schlafly & Finkbeiner 2011, ApJ, 737, 103",
            'author:"Schlafly" author:"Finkbeiner" bibstem:"ApJ" volume:737 page:103 pubdate:2011',
        ),
    ]
    for nl, query in citation_examples:
        examples.append({
            "natural_language": nl,
            "ads_query": query,
            "category": "reference",
        })

    return examples


def generate_uat_examples() -> list[dict]:
    """UAT field examples — model outputs abs:, UAT augmentation happens at runtime."""
    # These are just regular abs: queries for topics that happen to have UAT matches.
    # The uat_lookup.py module handles augmentation at serving time.
    # Including a few explicit uat: examples so the model knows the field exists.
    examples = [
        {
            "natural_language": "papers classified under the UAT concept stellar evolution",
            "ads_query": 'uat:"Stellar evolution"',
            "category": "filters",
        },
        {
            "natural_language": "articles tagged with UAT term active galactic nuclei",
            "ads_query": 'uat:"Active galactic nuclei"',
            "category": "filters",
        },
        {
            "natural_language": "find papers with UAT classification exoplanets",
            "ads_query": 'uat:"Exoplanets"',
            "category": "filters",
        },
        {
            "natural_language": "UAT tagged papers on gravitational waves",
            "ads_query": 'uat:"Gravitational waves"',
            "category": "filters",
        },
        {
            "natural_language": "papers classified as cosmic microwave background in the UAT",
            "ads_query": 'uat:"Cosmic microwave background radiation"',
            "category": "filters",
        },
        {
            "natural_language": "UAT brown dwarfs papers published since 2020",
            "ads_query": 'uat:"Brown dwarfs" pubdate:[2020 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "refereed papers on star formation tagged in UAT",
            "ads_query": 'uat:"Star formation" property:refereed',
            "category": "filters",
        },
        {
            "natural_language": "papers tagged with UAT term quasars by Schneider",
            "ads_query": 'uat:"Quasars" author:"Schneider"',
            "category": "filters",
        },
        {
            "natural_language": "find papers with the thesaurus term interstellar medium from the last 5 years",
            "ads_query": 'uat:"Interstellar medium" pubdate:[2021 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "articles classified under neutron stars in astronomy thesaurus",
            "ads_query": 'uat:"Neutron stars"',
            "category": "filters",
        },
    ]
    return examples


def generate_negation_examples() -> list[dict]:
    """Negation/NOT patterns — underrepresented in current training data."""
    examples = [
        {
            "natural_language": "dark matter papers but not about WIMPs",
            "ads_query": 'abs:"dark matter" -abs:"WIMPs"',
            "category": "compound",
        },
        {
            "natural_language": "exoplanet papers excluding hot Jupiters",
            "ads_query": 'abs:"exoplanet" -abs:"hot Jupiter"',
            "category": "compound",
        },
        {
            "natural_language": "galaxy evolution papers that are not reviews",
            "ads_query": 'abs:"galaxy evolution" -doctype:inproceedings -doctype:abstract',
            "category": "compound",
        },
        {
            "natural_language": "supernova papers not by Smith",
            "ads_query": 'abs:"supernova" -author:"Smith"',
            "category": "compound",
        },
        {
            "natural_language": "stellar spectroscopy excluding conference proceedings",
            "ads_query": 'abs:"stellar spectroscopy" -doctype:inproceedings',
            "category": "compound",
        },
        {
            "natural_language": "gravitational lensing not in the physics collection",
            "ads_query": 'abs:"gravitational lensing" -database:physics',
            "category": "compound",
        },
        {
            "natural_language": "black hole papers not preprints",
            "ads_query": 'abs:"black hole" -property:eprint',
            "category": "compound",
        },
        {
            "natural_language": "AGN variability excluding X-ray studies",
            "ads_query": 'abs:"AGN variability" -abs:"X-ray"',
            "category": "compound",
        },
        {
            "natural_language": "papers on dark energy but not supernova cosmology",
            "ads_query": 'abs:"dark energy" -abs:"supernova cosmology"',
            "category": "compound",
        },
        {
            "natural_language": "find refereed papers on pulsars not authored by Manchester",
            "ads_query": 'abs:"pulsars" property:refereed -author:"Manchester"',
            "category": "compound",
        },
        {
            "natural_language": "star formation in galaxies excluding the Milky Way",
            "ads_query": 'abs:"star formation" abs:"galaxies" -abs:"Milky Way"',
            "category": "compound",
        },
        {
            "natural_language": "magnetar papers not in conference proceedings or abstracts",
            "ads_query": 'abs:"magnetar" -doctype:inproceedings -doctype:abstract',
            "category": "compound",
        },
        {
            "natural_language": "solar wind papers excluding SOHO data",
            "ads_query": 'abs:"solar wind" -bibgroup:SOHO',
            "category": "compound",
        },
        {
            "natural_language": "galaxy cluster masses not using weak lensing",
            "ads_query": 'abs:"galaxy cluster" abs:"mass" -abs:"weak lensing"',
            "category": "compound",
        },
        {
            "natural_language": "exoplanet atmospheres excluding ground-based observations",
            "ads_query": 'abs:"exoplanet atmospheres" -abs:"ground-based"',
            "category": "compound",
        },
        {
            "natural_language": "cosmological simulations not about dark matter halos",
            "ads_query": 'abs:"cosmological simulations" -abs:"dark matter halos"',
            "category": "compound",
        },
        {
            "natural_language": "white dwarf papers that are not catalogs",
            "ads_query": 'abs:"white dwarf" -doctype:catalog',
            "category": "compound",
        },
        {
            "natural_language": "papers on cosmic rays excluding solar energetic particles",
            "ads_query": 'abs:"cosmic rays" -abs:"solar energetic particles"',
            "category": "compound",
        },
        {
            "natural_language": "quasar absorption lines not Lyman alpha",
            "ads_query": 'abs:"quasar absorption" -abs:"Lyman alpha"',
            "category": "compound",
        },
        {
            "natural_language": "planet formation papers not about the solar system",
            "ads_query": 'abs:"planet formation" -abs:"solar system"',
            "category": "compound",
        },
    ]
    return examples


def generate_software_mention_examples() -> list[dict]:
    """Software/data mention patterns using abs: + mention_count:/credit_count:."""
    examples = [
        {
            "natural_language": "papers that mention astropy",
            "ads_query": 'abs:"astropy" mention_count:[1 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "heavily cited papers about numpy in astronomy",
            "ads_query": 'abs:"numpy" mention_count:[1 TO *] citation_count:[50 TO *]',
            "category": "compound",
        },
        {
            "natural_language": "papers crediting SDSS data",
            "ads_query": 'abs:"SDSS" credit_count:[1 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "papers that use matplotlib for visualization",
            "ads_query": 'abs:"matplotlib" mention_count:[1 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "highly cited papers mentioning TensorFlow in astrophysics",
            "ads_query": 'abs:"TensorFlow" mention_count:[1 TO *] citation_count:[20 TO *] database:astronomy',
            "category": "compound",
        },
        {
            "natural_language": "papers that frequently mention CASA software",
            "ads_query": 'abs:"CASA" mention_count:[5 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "papers crediting Gaia data with high citation counts",
            "ads_query": 'abs:"Gaia" credit_count:[1 TO *] citation_count:[100 TO *]',
            "category": "compound",
        },
        {
            "natural_language": "papers mentioning SExtractor for source extraction",
            "ads_query": 'abs:"SExtractor" mention_count:[1 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "papers that mention DS9 visualization tool",
            "ads_query": 'abs:"DS9" mention_count:[1 TO *]',
            "category": "filters",
        },
        {
            "natural_language": "papers crediting Hubble data with many software mentions",
            "ads_query": 'abs:"Hubble" credit_count:[1 TO *] mention_count:[5 TO *]',
            "category": "compound",
        },
    ]
    return examples


def generate_date_diversity_examples() -> list[dict]:
    """Date expression diversity — relative temporal patterns."""
    examples = [
        {
            "natural_language": "exoplanet papers from the last 5 years",
            "ads_query": 'abs:"exoplanet" pubdate:[2021 TO *]',
            "category": "content",
        },
        {
            "natural_language": "dark matter papers since 2020",
            "ads_query": 'abs:"dark matter" pubdate:[2020 TO *]',
            "category": "content",
        },
        {
            "natural_language": "gravitational wave papers before 1990",
            "ads_query": 'abs:"gravitational wave" pubdate:[* TO 1990]',
            "category": "content",
        },
        {
            "natural_language": "papers on JWST added this week",
            "ads_query": 'abs:"JWST" entdate:[NOW-7DAYS TO *]',
            "category": "content",
        },
        {
            "natural_language": "recent papers on fast radio bursts from this month",
            "ads_query": 'abs:"fast radio bursts" entdate:[NOW-1MONTH TO *]',
            "category": "content",
        },
        {
            "natural_language": "galaxy formation papers from the past decade",
            "ads_query": 'abs:"galaxy formation" pubdate:[2016 TO *]',
            "category": "content",
        },
        {
            "natural_language": "neutrino astronomy papers from the last 2 years",
            "ads_query": 'abs:"neutrino astronomy" pubdate:[2024 TO *]',
            "category": "content",
        },
        {
            "natural_language": "new papers on black holes entered in the last 3 days",
            "ads_query": 'abs:"black holes" entdate:[NOW-3DAYS TO *]',
            "category": "content",
        },
        {
            "natural_language": "stellar evolution papers from the 1990s",
            "ads_query": 'abs:"stellar evolution" pubdate:[1990 TO 1999]',
            "category": "content",
        },
        {
            "natural_language": "papers on cosmic inflation published this year",
            "ads_query": 'abs:"cosmic inflation" pubdate:2026',
            "category": "content",
        },
    ]
    return examples


def generate_caption_examples() -> list[dict]:
    """caption: field for figure/table search."""
    examples = [
        {
            "natural_language": "papers with HR diagrams in their figures",
            "ads_query": 'caption:"Hertzsprung-Russell diagram"',
            "category": "filters",
        },
        {
            "natural_language": "papers with color-magnitude diagrams in figures",
            "ads_query": 'caption:"color-magnitude diagram"',
            "category": "filters",
        },
        {
            "natural_language": "papers showing light curves in figure captions",
            "ads_query": 'caption:"light curve"',
            "category": "filters",
        },
        {
            "natural_language": "papers with spectral energy distribution plots",
            "ads_query": 'caption:"spectral energy distribution"',
            "category": "filters",
        },
        {
            "natural_language": "papers with radial velocity curves in their figures",
            "ads_query": 'caption:"radial velocity" caption:"curve"',
            "category": "filters",
        },
    ]
    return examples


def generate_arxiv_id_examples() -> list[dict]:
    """arxiv: field as identifier lookup."""
    examples = [
        {
            "natural_language": "find the paper arXiv:1609.04747",
            "ads_query": "arxiv:1609.04747",
            "category": "identifier",
        },
        {
            "natural_language": "look up arXiv 2301.00001",
            "ads_query": "arxiv:2301.00001",
            "category": "identifier",
        },
        {
            "natural_language": "find astro-ph/9905116",
            "ads_query": "arxiv:astro-ph/9905116",
            "category": "identifier",
        },
        {
            "natural_language": "the preprint 2106.15163 on arXiv",
            "ads_query": "arxiv:2106.15163",
            "category": "identifier",
        },
        {
            "natural_language": "look up arXiv preprint hep-th/9802150",
            "ads_query": "arxiv:hep-th/9802150",
            "category": "identifier",
        },
    ]
    return examples


def generate_count_filter_examples() -> list[dict]:
    """author_count: and page_count: filter examples."""
    examples = [
        {
            "natural_language": "single-author papers on dark matter",
            "ads_query": 'abs:"dark matter" author_count:1',
            "category": "compound",
        },
        {
            "natural_language": "large collaboration papers on gravitational waves",
            "ads_query": 'abs:"gravitational waves" author_count:[100 TO *]',
            "category": "compound",
        },
        {
            "natural_language": "short letters on exoplanets under 5 pages",
            "ads_query": 'abs:"exoplanet" page_count:[1 TO 5]',
            "category": "compound",
        },
        {
            "natural_language": "long review articles on galaxy evolution",
            "ads_query": 'abs:"galaxy evolution" page_count:[30 TO *] property:refereed',
            "category": "compound",
        },
        {
            "natural_language": "papers by small teams of 2-3 authors on stellar binaries",
            "ads_query": 'abs:"stellar binaries" author_count:[2 TO 3]',
            "category": "compound",
        },
        {
            "natural_language": "solo-authored review papers on cosmology",
            "ads_query": 'abs:"cosmology" author_count:1 page_count:[20 TO *]',
            "category": "compound",
        },
        {
            "natural_language": "papers with more than 50 authors about LIGO",
            "ads_query": 'abs:"LIGO" author_count:[50 TO *]',
            "category": "compound",
        },
        {
            "natural_language": "brief reports on supernovae under 4 pages",
            "ads_query": 'abs:"supernova" page_count:[1 TO 4]',
            "category": "compound",
        },
    ]
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate training improvement examples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/generated/training_improvements.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    all_examples = []
    generators = [
        ("reference_parsing", generate_reference_parsing_examples),
        ("uat", generate_uat_examples),
        ("negation", generate_negation_examples),
        ("software_mentions", generate_software_mention_examples),
        ("date_diversity", generate_date_diversity_examples),
        ("caption", generate_caption_examples),
        ("arxiv_id", generate_arxiv_id_examples),
        ("count_filters", generate_count_filter_examples),
    ]

    for name, generator in generators:
        examples = generator()
        print(f"  {name}: {len(examples)} examples")
        all_examples.extend(examples)

    # Deduplicate by NL text
    seen: set[str] = set()
    unique = []
    for ex in all_examples:
        key = ex["natural_language"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(unique)} unique examples written to {args.output}")


if __name__ == "__main__":
    main()
