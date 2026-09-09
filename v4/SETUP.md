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
through **all six ablations, both locally and as Gems in Gemini**, on
2026-09-09. Every Gem imported its module, executed it, and reported numbers
identical to the local run, with the single exception noted below.

| Ablation | Values checked in Gemini | Result |
|---|---|---|
| three-track A | rows, span, keyphrases, dictionary, triples by class, graph collapse and discard counts, domains per patent, DS inflation | all match |
| three-track B | rows, span, verb lemmas, keyphrases, triples, technical verbs, domains per patent, citation ratio | match, except three counts one higher (see below) |
| three-track C | rows, concepts, ambiguous band, resolved concepts, relations, shrinkage constant, censoring horizon, top domain | all match |
| intent-preserving A | rows, keyphrases total and unique, primaries, vocabulary constant, triples, nodes, directed pairs, undirected edges, collapsed, discarded, technical verbs, cache-defect moves | all match |
| intent-preserving B | rows, keyphrases, primaries, floor-sized groups, triples, nodes, edges, technical verbs, cache-defect moves, domains per patent, citation ratio | all match |
| intent-preserving C | rows, concepts, structured functions, verb lemmas, auto-merges, ambiguous band, resolved concepts, relations, legacy class counts, top domain | all match |

**The one discrepancy, and why it happens.** The three-track B Gem reported
650 synonym groups, 478 graph nodes and 1,139 edges where the local run gave
649, 477 and 1,138. Exactly one phrase pair in this corpus has a
character-trigram cosine of 0.800000, sitting precisely on the `>=` threshold
the greedy grouping tests. A one-unit-in-the-last-place difference in the dot
product between numpy builds (1.26.3 in the sandbox, 2.x locally) flips that
single pair, and one extra group cascades into one extra node and one extra
edge. The grouping threshold is a knife edge: results are reproducible within
an environment but not necessarily across them.

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

## Failure modes seen while testing

All six Gems were driven through the browser against the same corpus. Three
things went wrong at least once, and all three are worth recognising because
none of them corrupts a number:

- **The Gem asks for a file that is already attached.** Seen once on Flash
  with both files on the first message. Re-attaching in a fresh conversation
  fixed it. The Gem never invented data; it asked again.
- **The report stops after the first section.** Seen once: the corpus block
  printed correctly and the rest was missing. Asking it to continue and paste
  the remaining printed lines produced them, and they matched the local run.
- **Code execution lapses mid-conversation and returns.** A Gem that had just
  run Python reported the tool unavailable a few turns later, then executed
  `print(2+2)` correctly a minute after that. Whenever the tool was gone the
  Gems refused rather than estimating, which is the behaviour the instructions
  ask for.

The heavier the module, the longer the first run takes: the A and B tracks
answered in under a minute, the C tracks took three to five minutes before
their first output appeared. That is the sandbox executing, not a hang.

## A defect the charts found, and what it moved

Drawing the landscape exposed something the prose reports had hidden. Tracks B
and C allow single-word concepts, and they matched keyphrases against
abstracts with a raw substring test. On patent chemistry that put single
letters at the top of the landscape: `c` in 122 patents, `e` in 122, `n` in
121, because almost every abstract contains those characters. The text reports
never foregrounded those rows, so six passing Gem runs went by without anyone
noticing.

Two fixes, in the four B and C modules only. A one-word candidate must now be
at least three characters, because a single letter is a chemical variable and
not a technology. And keyphrase matching uses the word-boundary pattern the B
tracks already claimed to have repaired, rather than a substring test. Track A
keeps the substring behaviour: it is the replica, and the shipped tool did
exactly that.

The landscape now reads `hydrogen`, `compounds`, `treatment`, `formula`,
`pharmaceutical compositions`, which is a corpus an analyst can argue with.

The fix moves some of the counts in the table above, which were measured
before it. Against the same 122-patent export the current code gives:

| Module | Was | Now |
|---|---|---|
| three-track B | 682 keyphrases, 649 groups | 688 keyphrases, 653 groups |
| three-track C | 682 concepts, 487 band, 679 resolved, 2081 relations | 688, 504, 685, 2005 |
| intent-preserving B | unchanged headline counts | 595 keyphrases, 100 primaries |
| intent-preserving C | 783 concepts, 659 band, 778 resolved, 2586 relations | 781, 674, 776, 2464 |

The Gemini runs verified that each Gem executes its module faithfully and
reproduces the local numbers exactly. That conclusion is unaffected: it is a
statement about the sandbox reproducing the code, not about which revision of
the code was running.
