# v4 — trt-pb as Gemini Gems, one Gem per ablation

trt-pb is a Streamlit tool that turns an ORBIT patent export into a technology
relationship graph (keyphrases, synonym groups, preposition-typed links) and a
technology trend analysis (domain specificity, document frequency, a
citation-derived function score, and a four-state trajectory reading). Two
redesign documents were written for it, each defining three systems to build
and compare. This directory holds a Gemini Gem for each of those six systems:
a system prompt and a single self-contained analysis module.

**The module goes in the conversation, not in the Gem's Knowledge field.** A
Gem that carries a knowledge file is served without Gemini's Python tool, so
it can read the module but not run it, and every one of these Gems will then
correctly refuse to answer. `SETUP.md` has the measurement.

| Document | Ablation | Directory | Module |
|---|---|---|---|
| Three-track analysis | A — behavioural replica, scoped per component | `three-track/a-replica/` | `trt_replica.py` |
| | B — corrected replica, recency enabled with a recorded window | `three-track/b-corrected/` | `trt_corrected.py` |
| | C — intent-optimal: shrinkage, censoring horizon, permutation-gated trends, log-odds challenger | `three-track/c-intent-optimal/` | `trt_intent.py` |
| Intent-preserving redesign | A — historical replica, full MLM grouping loop with its cache defect | `intent-preserving/a-replica/` | `trt_replica.py` |
| | B — corrected replica, recency deliberately left off | `intent-preserving/b-corrected/` | `trt_corrected.py` |
| | C — intent-optimal: evidence layer, Bayesian enrichment with seven estimator ablations, eight trend states, backtest | `intent-preserving/c-intent-optimal/` | `trt_intent.py` |

## The visual layer

trt-pb was a picture tool. It drew a technology relationship graph and a
four-quadrant trend plot, and the rest of it existed to feed those two
figures. The six modules above reproduce the analysis but answer in prose,
which is half the product.

`trt_charts.py` restores the other half and works with all six tracks:
landscape, relationship graph, prevalence, adoption, the trend plot, the 3D
trajectory, domain specificity and emergence. Charts render as matplotlib
images inline in the chat, and the graph and trend calls also write a
self-contained plotly HTML - the same artifact the shipped tool produced as
`graph-<Relationship>.html`.

`VISUALS.md` is the prompt that drives it. The point is not the renderer but
the loop: the analyst sees a numbered table of candidate technologies, says
which are drafting language and which are two names for one thing, and the
charts redraw. One, two or three turns, depending on how much the analyst
already knows.

`SKILL.md` packages the whole thing, including which ablation to reach for.
Short version: **default to the intent-preserving Track C** for real work, and
borrow the three-track Track C's censoring horizon if the corpus runs to the
present day.

Each subdirectory README maps its ablations to the sections of the document
that define them. `SETUP.md` explains how to build a Gem from any of the six
and how to run the same corpus through all of them.

## What every Gem shares

All six read the same seven ORBIT columns (abstract, publication number,
application date, title, description, technology domains, citing patents),
run in Gemini's code-execution sandbox with only the standard library, pandas
and numpy, and follow one rule: the model never produces a number by reading.
Counting is done by code; the model judges what the counts mean.

Four things the original tool relied on do not exist in the sandbox, and all
six files substitute for them the same way, so a difference between two Gems
is a difference between the ablations and not between their stand-ins:

| Original | Stand-in |
|---|---|
| spaCy part-of-speech tags for prepositions | a fixed preposition list; "to" before a verb lemma is treated as the infinitive marker |
| spaCy verb lemmas | a corpus-internal inflection test: a word is a verb lemma when at least two of its -s, -ed, -ing forms occur |
| POS-pattern candidates ranked by KeyBERT | stopword-segmented runs ranked per abstract by tf-idf |
| MPNet cosine (context-free) / Patent-BERT masked-LM rank (contextual) | character-trigram cosine / a PPMI context-vector ranking of corpus words |

Each file's docstring says exactly which stand-in it uses and where.

## How the tracks differ

**A** reproduces the shipped behaviour and its catalogued defects; nothing is
repaired. **B** keeps every method and repairs every defect the reference
settled on evidence. **C** replaces or extends the estimators while keeping
the analytical question and the patent-level evidence.

The two documents disagree in places, and the Gems preserve the disagreement:

- The three-track A does not run the masked-LM grouping at all (the cache
  defect makes its output noise) and uses the cosine algorithm as the
  baseline; the intent-preserving A runs the full greedy loop, floor and
  cache defect included, and can measure the defect's effect.
- The three-track B enables recency with one recorded window; the
  intent-preserving B leaves it off because the source holds five
  contradictory specifications.
- The three-track C gates trend labels on a permutation null, shrinks the
  function score toward the corpus mean, and refuses to emit a function score
  inside an empirically estimated citation censoring horizon; the
  intent-preserving C uses a median with a bootstrap interval, eight trend
  states with a confidence, Beta-posterior domain enrichment with a credible
  interval, and a temporal backtest for emergence.

Both C tracks put the model in the loop at the same two points, ambiguous
phrase-pair equivalence and typed relation labelling, and both record every
decision under a cache key and verify every relation span against the
sentence it came from.

## Running the same corpus through all six

Build six Gems, upload the same export to each, and ask each to "analyse
this". Compare the keyword dictionaries, the strongest relationships, the
domain-specific verbs and the trend reading for the same domain and term. The
A-track diagnostics (`order_sensitivity`, `permutation_null`, `funnel`,
`window_sensitivity`, `ds_bias`, `citation_format_check`, `cache_bug_effect`)
say how much of the difference is signal.
