# Building the Gems

## What to create

In Gemini, **Gems → New Gem**, once per ablation. Six Gems in all, or only the
ones you want to compare.

| Field | Value |
|---|---|
| Name | `trt-pb A` / `trt-pb B` / `trt-pb C`, prefixed with `3T` (three-track) or `IP` (intent-preserving) |
| Description | one line from the ablation's README |
| Instructions | the entire contents of that ablation's `SYSTEM_PROMPT.md` — select all, paste |
| Knowledge | that ablation's knowledge file, and nothing else |

Attach exactly one knowledge file per Gem. Each file is self-contained; two
files in one Gem would give the model two definitions of `report` and it will
pick one silently.

## Requirements

**Code execution must be available**, which in practice means a paid Gemini
tier. Every design here rests on the model computing rather than estimating;
without it each Gem is instructed to stop rather than guess.

The knowledge files need only `pandas` and `numpy`, both present in the
sandbox. They do not use spaCy, sentence-transformers, TensorFlow or
scikit-learn, which are not, and the substitutions that forces are listed in
each file's docstring and in the top-level README.

**Getting the file into the sandbox.** Knowledge files are visible to the
model as text; they are not mounted as files. Each system prompt therefore
tells the model to try `from trt_x import report` and, if that fails, to write
its knowledge file verbatim into the sandbox first. That works, but it is
slow for a long file and the model can drift while copying. If you see
`ImportError` more than once in a session, upload the `.py` file into the
conversation alongside the data export; a conversation upload is available
to code directly.

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

Upload an export and say "analyse this". A correctly configured Gem will:

1. show the columns and one record, and name the seven column mappings
2. run `report(...)` and show the printed output verbatim
3. (Track C only) ask you nothing, but show the ambiguous phrase pairs it
   adjudicated, each with a reason
4. cite publication numbers for its claims

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
