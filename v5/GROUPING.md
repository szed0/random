# The symmetric-NN grouping step: what was measured, and what changed

Canonicalisation asks one narrow question. Given two surface forms extracted
from the same corpus, do they name **one** technology that should occupy one row
of the landscape, or **two** that must stay apart?

The shipped answer was a symmetric neural network — `all-MiniLM-L6-v2`, a
Siamese-trained sentence encoder — scoring term against term by cosine, with a
fixed floor at 0.75. This document is the measurement of that choice and of five
alternatives.

## The benchmark

77 labelled patent term pairs over 120 distinct terms, in `snn/benchmark.py`.
Labels are the **product** decision, not linguistic synonymy: `alkaline
electrolyser` and `PEM electrolyser` are close in meaning and must **not** merge,
because an analyst comparing them is the entire point of the tool. That
asymmetry is why this cannot be scored on a general similarity benchmark.

Pairs are grouped by the relation that generates them, so a method that is
excellent on average and fails an entire class is visible rather than hidden:

| Positive (one technology) | n | Negative (two technologies) | n |
|---|---|---|---|
| head-truncation | 10 | shared-modifier | 8 |
| category-suffix | 6 | sibling-technology | 8 |
| modifier-drop | 5 | component-vs-component | 6 |
| abbreviation | 8 | thing-vs-system | 5 |
| spelling-register | 6 | unrelated | 6 |
| true-synonym | 6 | material-vs-application | 3 |

Three numbers are reported per method: accuracy at its own best threshold, that
threshold, and the **margin** — lowest true score minus highest false score.

Margin is the figure that matters. A method that separates perfectly with a
0.002 margin is not deployable on a corpus you have not seen, because nothing
tells you where the boundary sits next time.

## Results

| Method | Best accuracy | Margin | Abbreviations |
|---|---|---|---|
| cosine MiniLM (shipped) | 0.792 | **−0.817** | 0/8 |
| cosine mpnet | 0.766 | −0.970 | 1/8 |
| CSLS MiniLM | 0.779 | −1.458 | 0/8 |
| CSLS mpnet | 0.766 | −1.362 | 0/8 |
| cross-encoder ms-marco-MiniLM-L-6 | 0.805 | −15.48 | 1/8 |
| cross-encoder quora-distilroberta | 0.779 | −10.49 | 1/8 |
| **mpnet + exact rules (now shipped)** | **0.909** | −0.774 | **7/8** |

### Every margin is negative

No threshold, on any of these scorers, separates the two classes. The single
highest-scoring pair in the whole benchmark is a **false** one:

```
0.830   type III pressure vessel  /  type IV pressure vessel     NOT the same
0.796   liquid hydrogen storage   /  compressed hydrogen storage NOT the same
0.746   bipolar plate             /  electrolyser bipolar plate  same
0.742   leak detection sensor     /  hydrogen leak detection sensor  same
```

Two sibling technologies score above two genuine synonym pairs. This is not a
tuning problem and there is no value of the floor that fixes it. Sibling
technologies are *supposed* to be near each other in an embedding space — they
are the same kind of thing — and a symmetric similarity has no way to express
"close, and therefore distinct".

**The shipped 0.75 floor sits above two true pairs**, so `leak detection sensor`
and `hydrogen leak detection sensor` were being reported as two technologies with
the document count split between them.

### CSLS did not help

Cross-domain Similarity Local Scaling (Conneau et al., ICLR 2018) is the standard
correction for hubness, and hubness is the reason a plain threshold is unsafe in
high dimensions:

```
csls(a,b) = 2·cos(a,b) − r(a) − r(b)      r(x) = mean cosine to x's k nearest
```

It made accuracy slightly worse (0.779 vs 0.792) and the margin substantially
worse. The errors here are not hubness. A hub is a term that is nearest to
everything; `type III pressure vessel` is not a hub, it is genuinely close to one
specific term it must not merge with. Correcting for local density cannot fix a
similarity that is high for the right reason and wrong for the task.

Reported because it is the obvious thing to try, and it does not work here.

### Cross-encoders did not help either

Joint encoding is normally worth a large accuracy jump over independent
encoding. It bought 0.013 over the shipped scorer and still failed 7 of 8
abbreviations. These models are trained for relevance and paraphrase, and
neither is the question being asked.

### Abbreviations are the largest single failure

0/8 for the shipped model, 1/8 for everything learned. The scores are not
marginal:

```
0.157   PEM    /  proton exchange membrane
0.126   GDL    /  gas diffusion layer
0.071   BOP    /  balance of plant
0.013   MEA    /  membrane electrode assembly
```

An acronym and its expansion share no subwords, so there is nothing for a
subword model to align. No threshold reaches these, and no amount of model
scaling will — the information is not in the string.

## What changed

Two exact rules, added as union passes in `canonicalise()` alongside the existing
head-strip merge, before the k-NN pass.

**Initialism.** One term is the other's initials, in order. Also the multiword
case, where a shared tail is kept and the head is abbreviated: `PEM electrolyser`
/ `proton exchange membrane electrolyser`. The test is exact, so it cannot fire
on a pair that is merely close.

**Spelling.** Both forms fold to the same string under a British/American and
-ise/-ize table. Again exact.

Neither involves a model. Both run over every pair rather than only near ones,
because `MEA` and `membrane electrode assembly` sit at cosine 0.013 and would
never appear in any neighbour list.

The group's label also changed: a multiword member now beats a one-word member,
so a group containing an acronym and its expansion is labelled with the
expansion regardless of which is more frequent.

Result: **0.792 → 0.909**, abbreviations 0/8 → 7/8, with no regression in any
other category.

## What is still unsolved

The margin is still negative, at −0.774. The rules fix the classes that are
exactly decidable; they do nothing for the two that are not:

- **sibling technologies** (7/8 at best) — `type III` vs `type IV pressure
  vessel` remains the hardest pair in the set
- **true synonyms** (2/6) — `hydrogen embrittlement` / `hydrogen induced
  cracking`, `water electrolysis` / `water splitting`. Genuine domain synonymy
  with no lexical overlap, which is the case embeddings should own and do not

Both need evidence the pair-scoring framing cannot see. Two directions worth
testing next, neither implemented here:

**Corpus co-occurrence as a negative signal.** Two names for one technology
rarely appear in the same patent, because an author picks one and uses it
throughout. Two sibling technologies appear together constantly, because patents
compare them. That makes document co-occurrence evidence *against* merging —
the opposite of how it is normally used, and available at zero cost from
`per_doc`. It could not be tested here without a real corpus containing these
specific pairs; the Google Patents fetch used for that was rate-limited out.

**A domain glossary.** Abbreviation and spelling turned out to be exactly
decidable once written down. True synonymy in a specialist field probably is
too, for anyone who has the field's glossary. That is a data problem, not a
model problem.

## Reproducing

```bash
python snn/evaluate.py
```

Requires `all-MiniLM-L6-v2` and `all-mpnet-base-v2`. The cross-encoder rows need
network access on first run; everything else is local.
