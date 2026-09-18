# Keyword extraction: NLP in code, judgement in the model

Eight ablations over one question — which terms reach the keyword menu. Each
is a **code-side extractor** paired with a **model-side filter**. The counting
and the six figures in `keyword-trends/trt_keywords.py` are untouched and
still do the work.

## Why this directory exists

The `keyword-trends` extractor produced a menu that was about half drafting
language. Measured, not asserted: 140 real patents (HUPD, `main_cpc` starting
`H01M`, abstracts over 300 characters) were run through the **local trt-pb
pipeline** — spaCy `en_core_web_lg`, `KeyphraseTfidfVectorizer` with the
part-of-speech pattern, `all-mpnet-base-v2`, KeyBERT top-7 per abstract — and
through every extractor here. Agreement is against that local top 40.

Its top 40 contained `form`, `manufacturing`, `contact`, `forming`,
`producing`, `amount`, `formula`, `ratio`, `group`, `making`, `range`, `flow`,
`increase`, `preparing`, `temperature`. The local tool's contained `secondary
battery`, `nonaqueous electrolyte`, `membrane electrode assembly`, `composite
oxide`, `solid electrolyte`.

Two structural causes, neither a tuning problem:

1. **Single words were allowed.** The local tool's "No of words in a technical
   term" slider defaults to 2, so its candidates are multi-word by
   construction. Dropping single words alone moves agreement from 12 to 20.
2. **Ranking was by raw document frequency** — a popularity measure, and the
   most popular phrases in patent prose are the ones every patent says. The
   local tool ranks each abstract's phrases against *that abstract* and pools
   the per-abstract winners, which is a salience measure.

## The extractors

All four run over the same candidate pool: runs of open-class words between
closed-class words and punctuation, two to four words, normalised for plural,
occurring in at least two patents.

| `method` | Ranking | Needs | p@40 | recall@80 |
|---|---|---|---|---|
| `df` | Raw document frequency. The shipped behaviour, kept as the control. | — | 20/40 | 29/40 |
| `runs` | **C-value × cohesion.** Frequency a phrase does not owe to longer phrases containing it, times how informative its words are. | — | **24/40** | 31/40 |
| `tfidf` | Highest tf-idf the phrase reaches in any single abstract. | scikit-learn | 13/40 | 21/40 |
| `lsa` | **Cosine to the abstract**, tf-idf reduced by truncated SVD, top 7 per abstract then pooled. The nearest available analogue of KeyBERT. | scikit-learn | 22/40 | 31/40 |

For comparison, the shipped `keyword-trends` extractor scores **12/40**.

`tfidf` is included because it is the obvious idea and it is the worst one:
peak tf-idf rewards phrases that dominate a single abstract, which in patent
chemistry means one-off compound names, not the corpus's technologies.

`lsa` deserves its place despite scoring below `runs`. It is the only method
with the same *shape* as the reference — embed the document, embed the
candidates, keep the nearest, pool the per-abstract winners — and its errors
are different in kind: it promotes `present invention`, `present disclosure`
and `electrically connected`, which is exactly the class of mistake the filter
stage removes. Its recall@80 ties `runs`, so after filtering the two should
converge.

## The filters

The code stage is good at finding phrases and blind to what they mean. The
model closes that gap, and **selects from code-generated candidates only** —
it never invents, so a term the corpus does not contain cannot enter the menu.
That is the contract KeyBERT had: the vectoriser proposed, the embedding
chose.

| Filter | How | Cost | Prompt |
|---|---|---|---|
| `none` | Code ranking straight to the menu. | — | either |
| `menu` | The model is shown the top 80 as one list and deletes the drafting language. | one pass | `SYSTEM_PROMPT_MENU.md` |
| `perdoc` | The model is shown one candidate block per abstract and keeps the technical terms in each, mirroring KeyBERT's `top_n` per document. | 140 blocks on this corpus | `SYSTEM_PROMPT.md` |

### What filtering can win

`recall@80` is the ceiling: how many of the reference's top 40 sit anywhere in
the first 80 candidates, and could therefore be promoted into the top 40 by
deleting what is above them.

| method | code alone | ceiling after a perfect filter |
|---|---|---|
| `runs` | 24/40 | **31/40** |
| `lsa` | 22/40 | **31/40** |
| `df` | 20/40 | 29/40 |
| `tfidf` | 13/40 | 21/40 |

So the filter stage is worth **+7 to +9 rows**, and no more. That number is
the honest case for putting the model in the loop, and the honest limit on it.
The eight ablations are the four methods × `none` and one filter, so the
realised gain can be read off rather than assumed.

## Running the grid

```python
import trt_extract as tx, trt_keywords as tk
tx.report_capabilities()                      # what this sandbox actually has
state = tx.candidates("export.csv", method="runs")   # or lsa / tfidf / df
tx.menu(state, top=40)                        # filter "none"
tx.prompt_for(state)                          # -> blocks for the perdoc filter
tx.apply_selection(state, """1| ...\n2| ...""")
tx.menu(state, top=40)
bridged = tx.to_keyword_state(state, tk)      # charts, unchanged
tk.primary(bridged, 6)
```

Same corpus, same numbered menu, one word changed — that is the ablation.

## Before you trust any of this, read `ENVIRONMENT.md`

Three environment facts the other directories rest on no longer hold. In
short: **code execution works only inside a Gem**, a plain chat will
confidently compute numbers in its head; a **Gem knowledge file is a real file
in the sandbox**, so the module need not be re-attached every turn; and
**scikit-learn, scipy, nltk, spaCy and networkx are all present**, which is
what makes `tfidf` and `lsa` possible at all. spaCy has **no models**, so the
reference's part-of-speech pattern still cannot be run and the closed-class
word list stays.

## Nothing about the subject matter is hardcoded

The only fixed vocabulary is `CLOSED_CLASS` — determiners, prepositions,
conjunctions, pronouns, auxiliaries, degree adverbs. That is English grammar
and it is identical for a corpus about batteries and one about crop rotation.
The shipped extractor's `GENERIC` list (`apparatus`, `method`, `system`,
`device`, `assembly`, `means`…) is gone: it was subject-matter hardcoding, it
needed editing per corpus, and it was not working.

**Domain lift is computed and displayed but never ranked on.** It is the same
ratio trt-pb's trend analysis applies to verbs, and it looked like the
principled answer. Ranking on it dropped `runs` from 24/40 to 12/40, and to
21/40 at best with the lift shrunk toward 1. With 29 domains over 140 patents
most domains are tiny, so two patents sharing one produce a lift above 20 on
two observations. The column stays on screen because it is useful to see next
to a term; it is not evidence enough to order a menu with.

## Do not port the MPNet grouping

The other stand-in — character-trigram cosine for MPNet — is not the problem.
On twelve phrase pairs from this corpus, trigram and MPNet cosine disagree on
three, and where they disagree MPNet is not obviously right: at the shipped
0.80 threshold it merges `negative electrode` with `positive electrode`
(0.936), which are opposites.

| pair | trigram | MPNet | at 0.80 |
|---|---|---|---|
| secondary battery / rechargeable battery | 0.380 | 0.565 | both split |
| negative electrode / anode | 0.211 | 0.526 | both split |
| positive electrode / cathode | 0.178 | 0.518 | both split |
| nonaqueous electrolyte / non aqueous electrolyte | 0.889 | 0.912 | both group |

At that threshold cosine is doing spelling-variant matching with a semantic
veneer, which is why a trigram measure keeps up with it. Every genuine synonym
worded differently is missed by both. Pair adjudication belongs to the model,
for the same reason extraction does.

## Files

| file | what it is |
|---|---|
| `trt_extract.py` | the four extractors, `report_capabilities()`, the filter round-trip, and `to_keyword_state()` which hands results to `trt_keywords.py` |
| `SYSTEM_PROMPT.md` | Gem instructions, `perdoc` filter |
| `SYSTEM_PROMPT_MENU.md` | Gem instructions, `menu` filter |
| `PROMPTS.md` | the messages to send, verbatim |
| `ENVIRONMENT.md` | what the sandbox measured as, and which documented facts changed |

Keep `keyword-trends/trt_keywords.py`: it still does the counting and draws
all six figures.

## Limits

- Agreement is one 140-patent corpus in one technology area. A second corpus
  in a different field should be run before treating 24/40 as a general
  number.
- That corpus is single-year, so it exercises extraction and not the trend
  charts.
- The filter ceilings are `recall@80`, not measured filter output. What a real
  filtering pass achieves against that ceiling has not been measured.
- The local tool is the reference and not a ceiling. Its own top 40 contains
  `present invention` and `high capacity`, and it emits `fuel cell` and `fuel
  cells` as separate terms because it never normalises plurals. These
  extractors do normalise, so exact agreement is not the target and 40/40
  would be a worse result than it sounds.
