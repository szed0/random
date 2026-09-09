# Building the Gems

## What to create

In Gemini, **Gems → New Gem**, once per ablation. Six Gems in all, or only the
ones you want to compare.

| Field | Value |
|---|---|
| Name | `trt-pb A` / `trt-pb B` / `trt-pb C`, prefixed with `3T` (three-track) or `IP` (intent-preserving) |
| Description | one line from the ablation's README |
| Instructions | the entire contents of that ablation's `SYSTEM_PROMPT.md` — select all, paste |
| Knowledge | **leave empty** |

**Do not put the `.py` in the Gem's Knowledge field.** This is the one thing
that stops the whole design working, and it is not obvious. A Gem that carries
a knowledge file is served *without* the Python tool: the model can read the
module's text but cannot execute anything, so it follows its instructions,
reports that code execution is unavailable, and stops. Measured on 2026-09-09
against a real 122-patent export:

| Where the module was | Python tool | Result |
|---|---|---|
| Gem Knowledge field | unavailable | "Code execution is currently unavailable in this environment… I must stop here." |
| Uploaded in the conversation | available | ran `report()` and returned the full analysis |

The refusal is the Gem behaving correctly — it would rather stop than invent
numbers — but it makes the Gem useless. Upload the module **in the chat**,
next to the patent export, every time.

## Requirements

**Code execution must be available.** Every design here rests on the model
computing rather than estimating; without it each Gem is instructed to stop
rather than guess. On a free-tier account in September 2026 the tool was
present in ordinary chats and in Gems with no knowledge file, running
Python 3 with pandas 2.0.0 and numpy 1.26.3 — the environment these files
assume. Check it in one message: ask for `print(sum(range(1,101)))` and see
whether a "Show code" block and the answer 5050 come back.

The knowledge files need only `pandas` and `numpy`, both present in the
sandbox. They do not use spaCy, sentence-transformers, TensorFlow or
scikit-learn, which are not, and the substitutions that forces are listed in
each file's docstring and in the top-level README.

**Getting the file into the sandbox.** A file uploaded in the conversation is
available to the code tool directly, so the model only has to copy it into the
working directory and import it. Upload the module and the export together in
the first message. Do not ask the model to retype the module: at 35–66 KB it
will drift, and every system prompt forbids it.

## The input

An ORBIT export as CSV or Excel with the seven columns the original tool
used, under any recognisable names:

| Column | Used for |
|---|---|
| Abstract | keyphrases, triples, keyword-based trends |
| Publication numbers | provenance on every edge |
| Application dates | every trend slice; a date, not an identifier |
| Title | output labelling |
| English description | verb (function) extraction |
| Technology domains | domain segmentation; newline-separated |
| Citing patents | forward-citation counts |

Without technology domains there is no trend analysis in any track. Without
citing patents there is no function score or influence. Each Gem says which
columns it resolved before it computes anything, and stops if the date column
looks like a publication number.

## Checking it works

Upload the ablation's `.py` and an export together and say "analyse this". A
correctly configured Gem will:

1. show the columns and one record, and name the seven column mappings
2. run `report(...)` and show the printed output verbatim
3. (Track C only) ask you nothing, but show the ambiguous phrase pairs it
   adjudicated, each with a reason
4. cite publication numbers for its claims

If it says code execution is unavailable, check the Knowledge field is empty.

If it produces counts without code output, the Instructions field was
probably truncated on paste; check the tail of the field ends with the
Opening section.

Two adversarial tests. Give it a file whose date column holds publication
numbers: every track should refuse to produce trends. Give a Track C Gem a
corpus containing both "anode current collector" and "cathode current
collector": they must not merge, and the reason should say so.

## Comparing the ablations

Run the same export through the Gems you built and compare, for one domain
and one term of your choosing:

- the keyword dictionary size and the group the term landed in
- the strongest relationships for one class, with their publication numbers
- the top domain-specific verbs and their scores
- the trend reading, and in Track C its confidence, support and stability

The A-track diagnostics say how much of the difference is signal. In the
three-track A, `order_sensitivity` and `permutation_null` bound what the
original could ever have measured; in the intent-preserving A,
`cache_bug_effect` shows how much of the dictionary the cache defect decided.

## Known limits

- **Not reproducible in prose.** Statistics are stable because they come from
  code; the model's judgements are recorded and replayable in Track C, but
  the surrounding prose is not.
- **Upload-bounded.** A large export will not fit. Sample it, and know that
  the sample changes every document frequency downstream.
- **Data leaves the machine.** If the corpus is confidential, this is the
  wrong tool.
- **Instruction drift.** Over a long session the model tends back toward
  estimating. If numbers appear without code output, say "recompute that
  with code" and treat it as a signal to restart.
- **The stand-ins are stand-ins.** The verb lexicon misses irregular verbs;
  the preposition list has no context; trigram cosine is not an embedding.
  Every file documents where each one acts. The local tool remains the
  reference for the numbers.

## Verified on real data

`tools/fetch_patents.py` builds an ORBIT-shaped export from Google Patents:
real abstracts, filing dates, CPC subclasses as newline-separated technology
domains, and real forward citations from each patent's "Cited By" table. A
122-patent hydrogen and pharmaceutical-chemistry corpus built this way was run
through all six ablations locally, and through the three-track Track A Gem in
Gemini. The Gem's numbers matched the local run exactly:

| Measure | Local | Gem |
|---|---|---|
| Reciprocal pairs collapsed by defect 19 | 63 | 63 |
| Occurrences discarded by defect 20 | 49 | 49 |
| Mean technology domains per patent | 3.09 | 3.09 |
| Domain-specificity inflation, 90th percentile | 1.24x | 1.24x |

Two findings came out of using real patents rather than synthetic text.
Defect 7, the citing-string splitter, **did not fire**: Google Patents
publication numbers carry no internal hyphens or slashes, so the shipped count
and the corrected count agree exactly, which is the answer to the reference
document's open question 7 for this export format and not necessarily for an
ORBIT one. And greedy grouping was far more stable here than on synthetic text
(adjusted Rand index 0.90 across shuffles, against 0.41), because real
abstracts share few exact phrases.

Real data also broke the three-track Track C censoring horizon. Cohort median
citations do not rise with age in a relevance-ranked sample, so the estimator
found a "plateau" at age zero and silently disabled the censoring rule the
design depends on. It now tests whether accrual is monotone in age, and when
it is not, says the horizon is not identifiable and falls back to a stated
default.
