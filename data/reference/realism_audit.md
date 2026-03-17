# Realism and Quality Audit of Gold Examples

**Date:** 2026-03-05
**Dataset:** `data/datasets/raw/gold_examples.json`
**Total examples reviewed:** 4,938

---

## Executive Summary

After reviewing all 4,938 training examples, approximately **685 examples (14% of the dataset)** have realism or quality issues across 6 critique categories. The majority of problems are concentrated in **template-generated examples** (categories: `content`, `filters`, `bibgroup`, `property`, `collection`, `syntax`), where mechanical NL phrasing and cross-product template expansion produced examples that no real user would type. The hand-curated examples (first ~1000 entries) are substantially higher quality.

**Severity breakdown:**
- **High severity (actively harmful to training):** ~220 examples -- should be removed or heavily rewritten
- **Medium severity (reduces model naturalness):** ~310 examples -- should be rewritten with more natural NL
- **Low severity (minor quality concerns):** ~155 examples -- acceptable but improvable

---

## Critique Category Findings

### Category 1: Unlikely Search Intent (~95 examples)

These are queries that no real astronomer would construct because they describe internal system mechanics rather than information needs.

**1a. "X in body field" / "X in full field" / "X in abs field" patterns (~56 examples)**

Lines ~12400-13400. Template-generated NL that references ADS field names directly.

Examples:
- `"gravitational waves in body field"` -> `body:"gravitational waves"`
- `"dark energy in full field"` -> `full:"dark energy"`
- `"body contains gravitational waves"` -> `body:"gravitational waves"` (14 examples)
- `"full contains dark energy"` -> `full:"dark energy"` (14 examples)
- `"find dark energy in body"` -> `body:"dark energy"` (56 examples total of the "X in Y field" pattern)

**Why problematic:** Real users do not think in terms of "body field" or "abs field." They think "papers about dark energy" or "dark energy mentioned in the full text."

**Suggested fix:** Rewrite NL to natural phrasing:
- "gravitational waves in body field" -> "papers mentioning gravitational waves in the full text"
- "body contains dark energy" -> "papers that discuss dark energy in the body"

**1b. `has:` field metadata queries (~47 examples)**

Lines ~22418-23550.

Examples:
- `"papers with publisher-verified ORCID IDs by first author Smith"` -> `has:orcid_pub author:"^Smith"`
- `"articles that have issue numbers on gravitational lensing"` -> `has:issue abs:"gravitational lensing"`
- `"papers with volume information about cosmology"` -> `has:volume abs:"cosmology"`
- `"papers with unified astronomy thesaurus tags about star formation"` -> `has:uat abs:"star formation"`
- `"records that have an abstract about solar flares"` -> `has:abstract abs:"solar flares"`

**Why problematic:** No researcher has ever searched for "papers with volume information about cosmology." These are metadata completeness queries. Additionally, `has:` was non-functional via the API (confirmed 2026-03-04).

**Suggested fix:** Remove or reduce to ~5 examples with realistic NL like "full-text papers on dark matter" -> `has:body abs:"dark matter"`.

**1c. Synonym expansion / exact match examples (~8 examples)**

- `"only author signature 'smith, j' exactly"` -> `=author:"smith, j"`
- `"disable author name expansion smith j"` -> `=author:"smith, j"`
- `"find exact matches for the term supernova without related terms"` -> `=abs:"supernova"`

**Why problematic:** Real users do not think about "synonym expansion." These read like syntax documentation.

---

### Category 2: Over-Engineered NL (~130 examples)

**2a. "X bibliography Y" template pattern (~80 examples)**

Lines ~17668-18900.

- `"HST bibliography gravitational waves"` -> `bibgroup:HST abs:"gravitational waves"`
- `"ALMA bibliography dark energy"` -> `bibgroup:ALMA abs:"dark energy"`
- `"Spitzer bibliography solar wind"` -> `bibgroup:Spitzer abs:"solar wind"`

**Why problematic:** "HST bibliography gravitational waves" is not how anyone talks. A real user would say "Hubble papers on gravitational waves."

**2b. "X in Y bibgroup" pattern (~50 examples)**

- `"gravitational waves in HST bibgroup"` -> `bibgroup:HST abs:"gravitational waves"`

**Why problematic:** "bibgroup" is internal jargon.

**2c. "X Y openaccess only" pattern (~30 examples)**

Lines ~15668-16050.

- `"gravitational waves ApJ openaccess only"` -> `bibstem:"ApJ" property:openaccess abs:"gravitational waves"`

**Why problematic:** "openaccess only" is terse jargon. Real users say "free access" or "open access papers."

---

### Category 3: Redundant / Pointless Filters (~80 examples, LOW SEVERITY)

- `doctype:article` when already implied -- pervasive but mostly harmless
- `property:refereed` when NL says "reviews" instead of using `reviews()` operator -- medium severity, model should learn both patterns

---

### Category 4: Implausible Specificity (~25 examples)

**4a. Absurd date ranges**

- `"laser plasma filament energy dependence studies"` -> `(...) AND year:1641-2024` (3 examples)
- `"sybil attacks intelligent vehicular ad hoc networks review 1641-2024"` -> `(...) AND year:1641-2024`

**Why problematic:** `year:1641-2024` is nonsensical.

**4b. Non-academic / off-topic NL**

- `"electronic payment security solata digital transactions"` -> `abs:(Security AND Electronic AND Payment)`
- `"surgical scalpel design and safety studies"` -> `abs:(scalpel* AND design AND safety)`
- `"coastal erosion human displacement vulnerability Bangladesh"` -> `abs:(Coastlines AND Human AND Displacement AND Erosion)`
- `"physics economic growth development national policy"` -> `abs:(Explain AND Role AND Physics AND National)`
- `"boresight translation optical alignment papers"` -> `abs:(traduzione di boresight)` [Italian!]

**4c. Broken queries -- DOIs/bibcodes/library hashes in abs: field**

13 examples use `abs:(DOI)`, `abs:(bibcode)`, `-docs(hash)`, or `abs:(docs(hash))` patterns:
- `"papers about machine learning in stellar classification"` -> `abs:(docs(ee0441a31e5b598d10c8d7fe0854a2a9))`
- `"papers on stellar atmospheres and spectroscopy"` -> `abs:(10.1051/0004-6361/201525830)`
- `"papers by Jiang, Haochang on stellar evolution"` -> `-docs(fae339e41c...) author:"^Jiang"`
- `"papers citing Planck satellite cosmology 2022+"` -> `docs(library/ZcKIJ9KlSp2s3m2U2l3gUQ)`

These are user-session artifacts where someone was using a library or DOI lookup, not generalizable training patterns.

---

### Category 5: Pedagogical Examples Masquerading as Training Data (~180 examples)

**5a. Content field cross-product templates (~140 examples) -- HIGHEST PRIORITY ISSUE**

Lines ~12300-13400. Systematic expansions of 14 topics x 5 fields (abs, title, body, full, keyword) x ~5 NL variants.

For "gravitational waves" alone, there are 25+ examples mapping to 5 different fields:
- `"papers about gravitational waves"` -> `abs:"gravitational waves"`
- `"papers about gravitational waves"` -> `title:"gravitational waves"`
- `"papers about gravitational waves"` -> `body:"gravitational waves"`
- `"papers about gravitational waves"` -> `full:"gravitational waves"`
- `"papers about gravitational waves"` -> `keyword:"gravitational waves"`

**This is the single most damaging issue in the dataset.** The same NL "papers about gravitational waves" mapping to 5 different queries creates contradictory training signal. The model cannot learn when to use which field from identical input.

**Suggested fix:**
- Keep ONLY the `abs:` version of generic "papers about X" queries
- Keep ONE `title:` example per topic with NL that says "in the title"
- Keep ONE `full:` example per topic with NL like "mentioned in the full text"
- REMOVE all `body:` and `keyword:` duplicates
- Net reduction: ~100 examples

**5b. Syntax demonstration examples (~43 examples)**

- `"huchra jo wildcard"` -> `author:"huchra, jo*"`
- `"author bol? wildcard"` -> `author:"bol?,"`
- `"facility regex magell.*"` -> `abs:/magell.*/`
- `"title: map NEAR5 planar"` -> `title:(map NEAR5 planar)` [literal syntax echoed in NL!]
- `"title =star"` -> `=title:"star"`

**Suggested fix:** Rewrite ~15 to keep with realistic NL, remove ~28.

---

### Category 6: Stilted / Robotic NL (~175 examples)

**6a. Fine-grained open access subtypes (~50 examples)**

- `"show me papers with ADS open access"` -> `property:ads_openaccess`
- `"author-submitted open access papers"` -> `property:author_openaccess`
- `"I want author open access versions"` -> `property:author_openaccess`

Real users do not distinguish between ads_openaccess, eprint_openaccess, pub_openaccess, and author_openaccess. They say "free papers."

**6b. Repetitive filter combos (~125 examples)**

Topic x journal x filter combos have 4-6 NL variants each. Example for one combo:
- "gravitational waves ApJ refereed"
- "gravitational waves in ApJ refereed"
- "refereed ApJ papers on gravitational waves"
- "gravitational waves ApJ refereed only"

Reduce to 2 per combo.

---

## Summary Statistics

| Critique Category | Examples Flagged | Severity | Recommended Action |
|---|---|---|---|
| 1. Unlikely search intent | ~95 | HIGH | Remove ~60, rewrite ~35 |
| 2. Over-engineered NL | ~130 | HIGH | Rewrite all |
| 3. Redundant/pointless filters | ~80 | LOW | Minor fixes |
| 4. Implausible specificity | ~25 | HIGH | Remove all |
| 5. Pedagogical masquerading | ~180 | HIGH | Remove ~120, rewrite ~60 |
| 6. Stilted/robotic NL | ~175 | MEDIUM | Rewrite all |
| **TOTAL** | **~685** | | |

---

## Top Priority Fixes (Ordered by Impact)

1. **Remove content-field cross-product contradictions** (~100 examples): Same NL mapping to `abs:`, `title:`, `body:`, `full:`, `keyword:` simultaneously. Most damaging issue.
2. **Remove non-academic / absurd examples** (~25 examples): Off-topic queries and `year:1641-2024` date ranges.
3. **Rewrite bibgroup template NL** (~130 examples): Replace "X bibliography Y" and "Y in X bibgroup" with natural phrasings.
4. **Rewrite "X in Y field" NL patterns** (~56 examples): Replace "dark energy in body field" with natural phrasing.
5. **Reduce property subtype over-representation** (~50 examples): Fine-grained open access types -> reduce to ~12.
6. **Reduce filter combination redundancy** (~200 examples): 4-6 NL variants per combo -> 2 per combo.
7. **Rewrite or remove syntax demos** (~28 examples): Remove literal syntax in NL.
8. **Review `has:` field examples** (~47 examples): Non-functional via API. Remove most, keep ~5.

---

## Specific Examples to Remove

### Non-academic / off-topic (6 examples)
| Line | NL |
|------|-----|
| 758 | electronic payment security solata digital transactions |
| 2638 | surgical scalpel design and safety studies |
| 3878 | coastal erosion human displacement vulnerability Bangladesh |
| 4828 | physics economic growth development national policy |
| 4913 | boresight translation optical alignment papers |
| 3998 | sybil attacks intelligent vehicular ad hoc networks review 1641-2024 |

### Absurd date ranges (3 examples)
| Line | NL |
|------|-----|
| 109 | laser plasma filament energy dependence studies (year:1641-2024) |
| 2348 | laser plasma filament radius energy 1641-2024 |
| 4633 | laser plasma filament radius dependence on laser energy (year:1641-2024) |

### Broken queries -- DOIs/bibcodes/library hashes in abs: field (13 examples)
| Line | NL | Issue |
|------|-----|-------|
| 503 | papers about machine learning in stellar classification | docs() hash in abs: |
| 1209 | papers on stellar atmospheres and spectroscopy | DOI in abs: |
| 1413 | looking for martian surface composition | DOI in abs: |
| 2403 | papers on gamma-ray bursts in ApJ 2010 | DOI in abs: |
| 2668 | papers by Pereira 2022 AJ | bibcode in abs: |
| 3339 | papers on stellar mass black holes in binary | DOI in abs: |
| 3424 | papers by Fukugita on stellar photometry | bibcode in abs: |
| 3749 | papers by Schlafly on stellar photometry | bibcode in abs: |
| 4629 | papers about stellar nucleosynthesis in massive stars | docs() hash |
| 4704 | papers by Guijarro on stellar spectroscopy | -docs() in abs: |
| 1579 | papers by Jiang, Haochang on stellar evolution | library -docs() |
| 3019 | papers by Velazquez P on stellar dynamics | library -docs() |
| 4779 | papers citing Planck satellite cosmology 2022+ | docs(library/) hash |

### Literal syntax in NL (3 examples)
| Line | NL |
|------|-----|
| 12043 | title: map NEAR5 planar |
| 12053 | facility regex magell.* |
| 12058 | instrument/facility matching magell.* |

---

## Category Breakdown of Flagged Examples

| Dataset Category | Total | Flagged | Flag Rate |
|---|---|---|---|
| content | 382 | ~150 | 39% |
| filters | 668 | ~200 | 30% |
| bibgroup | 328 | ~130 | 40% |
| property | 169 | ~50 | 30% |
| syntax | 43 | ~28 | 65% |
| collection | 112 | ~15 | 13% |
| topic | 381 | ~25 | 7% |
| publication | 402 | ~15 | 4% |
| first_author | 812 | ~10 | 1% |
| author | 565 | ~10 | 2% |
| operator | 346 | ~5 | 1% |
| Other | ~730 | ~47 | 6% |
| **TOTAL** | **4,938** | **~685** | **14%** |

**The pattern is clear:** Template-generated categories (`content`, `filters`, `bibgroup`, `property`, `syntax`) have 30-65% flag rates, while hand-curated categories (`first_author`, `author`, `operator`) have 1-7% flag rates.

---

## Patterns Worth Preserving

Despite the issues above, ~86% of the dataset is high quality:

1. **Hand-curated author queries** (~1,400 examples): Natural, realistic, well-formed
2. **Conversational NL** (~50 examples): "what's new in red giants?", "any good neutron stars reviews?" -- excellent
3. **Real user queries from logs** (first ~1,000 examples): Authentic user behavior
4. **Object queries** (~40 examples): "papers about Betelgeuse", "M31 papers" -- well-formed
5. **Operator examples** (~346 examples): `trending()`, `useful()`, `similar()` -- mostly well-done
6. **Affiliation queries** (~76 examples): "MIT papers", "NASA Goddard research on exoplanets" -- realistic

---

## Recommended Next Steps

1. Write a cleanup script (`scripts/realism_cleanup.py`) that removes ~50 flagged-for-removal examples, deduplicates content-field cross-products, rewrites template NL, and reduces filter combo redundancy
2. Manual review of rewrites before committing
3. Net impact: Remove ~180, rewrite ~350. Final dataset: ~4,758 examples, higher quality per example
4. Re-validate after cleanup
