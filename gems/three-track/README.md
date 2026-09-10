# Three-track analysis — ablations

Source: the intent-preservation analysis and three-track redesign of trt-pb.
Its governing rule is *preserve the analytical question, improve the
estimator*, and its deliverable is a per-component table with a status of
survives, fixed, replaced, extended or removed. The three Gems here are the
three columns of that table.

## A — behavioural replica (`a-replica/`)

Section 1 argues Track A should be scoped per component rather than built as
a whole running system, because the masked-LM cache defect (defect 5) makes
the phrase-grouping output noise with a stable seed. So this Gem reproduces
the inherited pipeline and the trend arithmetic faithfully, uses the cosine
grouping from `bertui.py` as the honest baseline, and characterises the MLM
path instead of running it.

Preserved defects: 1, 2, 3, 4, 6, 7, 12, 17, 19, 20, and the post-explode DS
denominator of open question 5.

Section 6.2's eight label-free measurements are in the file, because the
document says to run them against Track A first:

| Function | Section 6.2 item |
|---|---|
| `order_sensitivity` | 1 — shuffle the keyphrase list, adjusted Rand index between partitions |
| `permutation_null` | 2 — permute term membership within windows, tabulate quadrant labels |
| `funnel` | 3 — FS against DF with analytic bands |
| `censoring_horizon` | 4 — cohort citation accrual, horizon H |
| `ds_bias` | 5 — the D̄ / D̄_V inflation ratio per verb |
| `window_sensitivity` | 6 — flip rate across 2/3/4/5-year windows |
| `cache_key_check` | 7 — the cache-key collision, as far as the sandbox allows |
| `citation_format_check` | 8 — whether the citing-string splitter over-counts |

## B — corrected replica (`b-corrected/`)

The "Bug-fixed (Track B)" column of the section 5 table, plus the section 3
denominator fix: pre-explode DS marginals, escaped and grouped keyword
pattern, unreachable preposition entries removed, corrected citation
splitting, zero guard, directed graph with multiplicity, single-word
keyphrases allowed, a recorded grouping order, the cumulative view labelled
as an adoption curve, and the quadrant mapping stated explicitly.

Recency is enabled with one recorded window (last five years, ratio above
0.7), as the table asks; the five contradictory specifications are printed
alongside so the choice is visible.

The section 6.2 diagnostics are carried over so the effect of each fix can be
measured on the user's corpus.

## C — intent-optimal (`c-intent-optimal/`)

The "Intent-optimal (Track C)" column, with the section 4 decisions and the
two structural problems of section 2 built in:

| Component | Status | What the file does |
|---|---|---|
| Keyphrase extraction | survives + fixed | 1-grams allowed; span-verified model proposals gated as a hypothesis |
| Phrase grouping | replaced | trigram blocking, band split, model adjudication of the ambiguous band only, weighted graph, communities with a representative consistency check (4.1) |
| Relation extraction | extended | chained triples, typed within a constrained ontology, verbatim span verification, six-class legacy projection (4.2) |
| Domain specificity | fixed + extended | pre-explode lift with a frequency floor; weighted log-odds with an informative Dirichlet prior as the challenger, both reported (4.3) |
| Emergence | replaced | citation-free: decayed frequency, slope, acceleration, first appearance, burst, persistence (4.6) |
| Citation normalisation | replaced | within-cohort percentile; empirical censoring horizon H; no FS inside H (4.4, 2.2) |
| Function score | fixed + extended | empirical-Bayes shrinkage toward the corpus mean with k from variance components; median alongside (2.1) |
| Temporal analysis | extended | 2/3/4/5-year windows aligned to the newest year; window stability |
| Trend classification | replaced | four labels gated on a permutation null, a DF floor and window agreement; stable, emerging, volatile, insufficient evidence, window-sensitive (4.5) |
| Graph | fixed + extended | typed directed multigraph with counts, publication ids, sentences, confidence, year distribution |

Section 7's determinism constraint is met by recording every model decision
under `sha256(pair, prompt_version, model_id, temperature)` and exporting the
decisions as JSON, so a rerun reproduces the partition.
