# Intent-preserving redesign — ablations

Source: the intent-preserving redesign of trt-pb. Its north star is that
trt-pb should be preserved as an experimental control, not as the production
architecture, and that changes are promoted only when they survive empirical
evaluation. Section 2 defines the three systems; section 21 is the required
comparison table. The three Gems here are those three systems.

## A — historical / behavioural replica (`a-replica/`)

Frozen around the shipped `main_streamlit.py` behaviour, including known
defects, exactly as section 2 lists them: the masked-LM greedy grouping with
its `min_keys` floor, the incorrect logits cache behaviour, 2-4 word
candidates, KeyBERT-style ranking, six preposition classes, the existing
phrase matching, deduplicated edges, post-explode DS denominators, DS-only
selection, the existing citation parsing, linear age normalisation, the mean
function score, fixed-width bins, and the existing outputs.

The masked-LM ranking is emulated over a distributional word ranking with the
threshold rescaled from the model's vocabulary to the corpus's; the loop, the
floor and the cache defect are reproduced as written. `cache_bug_effect`
regroups without the defect and counts the phrases that move, which the
shipped program could never do.

## B — corrected replica (`b-corrected/`)

Section 2's Track B list, item by item: phrase-boundary matching, escaping,
boilerplate regexes, cache keys and invalidation, the cache's dependence on
mask positions, redundant re-reading, publication-number parsing, duplicate
and shadowed computations, patent-level denominators after the explode,
cross-tab coupling, graph direction, and edge-weight loss.

Recency stays **off**, as section 2 requires: the source has contradictory
definitions for its window and threshold, so enabling it would be a
methodological redesign. `recency()` computes any of the five specifications
for comparison, and the report prints them all.

`cache_defect_effect`, `ds_bias` and `citation_format_check` measure what the
fixes changed on the user's corpus.

## C — intent-optimal (`c-intent-optimal/`)

Sections 3 to 21, organised around the evidence layer of section 2:

| Section | What the file does |
|---|---|
| 3.1 hybrid extraction | deterministic 1-5 word candidates; model judges each surface form; proposals must align to a verbatim span; evidence records in the section's shape |
| 4 normalisation | trigram blocking, pairwise records (similarity, contextual equivalence, supporting and contradicting contexts), model adjudication of the ambiguous band, weighted graph, cluster consistency validation against the canonical member (the anti-chaining requirement) |
| 5, 6 relations | the thirteen-type ontology plus compatibility types, direction, verbatim span verification, legacy projection, and per-edge occurrences, distinct patents, publication ids, evidence, confidence, first and last year, year distribution |
| 7 functional concepts | verb plus object ("detect hydrogen leakage"); bare verb kept as the legacy field |
| 8 domain specificity | Beta-posterior enrichment with a 95% credible interval, posterior log-odds and support; `ds_ablation` runs corrected lift, smoothed lift, PMI, normalised PMI, weighted log-odds, log-likelihood keyness and Bayesian enrichment side by side with rank agreement |
| 9 emergence | prevalence ratio, rolling slope, acceleration, burst, first appearance, persistence, recent domain enrichment — kept separate from specificity |
| 10 influence | cohort-normalised citation percentile, year ±1, domain-conditioned when the cohort permits |
| 11 function score | median of cohort-normalised influence with a bootstrap interval; "Insufficient evidence" below five patents |
| 12 temporal | 2/3/4/5-year windows, window stability; the cumulative view is an adoption curve and never evidence of decline |
| 13 classification | Growing, Generalizing, Declining, Repositioning, Stable, Emerging, Volatile, Insufficient evidence, with confidence, support, stability and the legacy state |
| 14 provenance | evidence objects for technologies, relations, functions and trends |
| 18 backtest | hide the years after a cutoff, rank by emergence, reveal, score precision at K, average precision and sustained growth |
| 19 objective | the nine weights, frozen in the file |

Section 15's intent-preservation test and section 16's improvement loop are
carried in the system prompt rather than the code: the Gem states, for any
replacement it reports, what need it serves, what latent quantity it
estimates, and what new bias must be evaluated.
