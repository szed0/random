# Building the Gem

## What to create

In Gemini, **Gems → New Gem**.

| Field | Value |
|---|---|
| Name | TechScope |
| Description | Turns a patent export into a technology landscape: what is in it, what is the same thing under different names, how they relate, and what is rising. |
| Instructions | the entire contents of `SYSTEM_PROMPT.md` — select all, paste |
| Knowledge | `techscope_core.py` |

Optionally also attach a one-page glossary of your own domain's terminology. It
measurably improves step 3, the synonym merge, because the model can then map
your house abbreviations to their expanded forms.

## Requirements

**Code execution must be available**, which in practice means a paid Gemini tier.
The whole design rests on the model computing rather than estimating; without it
the Gem is instructed to stop rather than guess, and you get a qualitative
reading of a sample instead of a corpus measurement.

`techscope_core.py` needs only `pandas` and `numpy`, both present in the sandbox.
It deliberately does not use spaCy or sentence-transformers, which are not.

## Checking it works

Upload a patent export and say "analyse this". A correctly configured Gem will:

1. show you the columns and one example record before analysing anything
2. run `report(...)` and show the printed output
3. tell you which terms it merged, and why
4. cite publication numbers for its claims

If it produces counts without showing code output, the instructions are not
being followed — the most common cause is the Instructions field being truncated
on paste. Check the tail of the field says "answerable by looking at the file".

A good adversarial test: give it a file whose date column holds publication
numbers. It should refuse to produce trends and tell you the column is unusable.
If it produces a confident trend chart instead, it is fabricating.

## What this Gem is and is not

It is the same **analysis** as the local tool, with two substitutions forced by
the sandbox, and one genuine improvement.

| | Local tool | This Gem |
|---|---|---|
| Candidate phrases | dependency parse, noun chunks | stopword segmentation (RAKE) |
| Synonym grouping | sentence embeddings, mutual k-NN | the model's judgment |
| Termhood, PPMI, trends | identical | identical |
| Determinism | same input, same output, always | statistics yes, prose no |
| Corpus size | 40,000 rows in ~60s | a few thousand, upload-limited |
| Where the data goes | nowhere | Google |
| Explains *why* two things relate | no | yes |
| Reads claim language and intent | no | yes |

The substitutions cost real accuracy. Stopword segmentation cannot tell a noun
phrase from a verb phrase, so more junk survives into the candidate list, and the
model has to reject it. That is a worse extractor than a parser.

The improvement is genuine. Cosine similarity between term embeddings, measured
on `all-MiniLM-L6-v2`, the model the local tool ships with:

| Similarity | Pair | Same technology? |
|---|---|---|
| 0.901 | `cryogenic tank` / `cryogenic storage tank` | yes |
| 0.840 | `fibre optic sensor array` / `optic sensor array` | yes |
| 0.742 | `leak detection sensor` / `hydrogen leak detection sensor` | yes |
| 0.679 | `pipeline segment` / `pipeline monitoring system` | **no** |
| 0.642 | `hydrogen sensor` / `hydrogen tank` | **no** |
| 0.573 | `catalyst layer` / `catalytic converter` | **no** |

These six do separate — but only in a 0.063-wide gap between 0.679 and 0.742,
and nothing tells you where that gap sits for a corpus you have not seen.

Worse, **the local tool's shipped default floor of 0.75 falls above the third
row**, so it fails to merge `leak detection sensor` with `hydrogen leak
detection sensor` — two names for one thing, left as two technologies with the
document counts split between them.

A language model does not need the gap. It knows a sensor detects and a tank
stores, and that one phrase is the other with a modifier dropped. On the merge
step specifically, this Gem should beat the local tool.

**Use both if the work matters.** Run the local tool for the numbers, because
they are reproducible and auditable and the corpus can be large. Give its
`techscope-terms.csv` to the Gem, and ask it to interpret — name the clusters,
explain the relationships, find the whitespace. That plays each side to its
strength and is the configuration worth building toward.

## Known limits

- **Not reproducible.** Ask twice, get two different reports. The statistics are
  stable because they come from code; the prose around them is not. Do not use
  it where an audit trail is required.
- **Upload-bounded.** A large export will not fit. Sample it — and know that the
  sample changes the document frequencies, which changes everything downstream.
- **Data leaves the machine.** If the corpus is confidential, this is the wrong
  tool and the local one is the right one.
- **Instruction drift.** Over a long conversation the model tends back toward
  estimating rather than computing. If numbers start appearing without code
  output, say "recompute that with code" — and treat it as a signal to restart
  the session.
- **The merge is a judgment call and can be wrong.** It should show its work at
  step 3 so you can overrule it. If it merges silently, that is a fault.
