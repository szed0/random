# TechScope — algorithms, dataflow and architecture

This is the v1 reference. It describes what the code does, why each choice was
made, and what was rejected. Where a decision was measured rather than
reasoned, the measurement is given.

> `v2/` runs the same algorithms with the defects in section 13 fixed. Prefer it
> unless you specifically want this version. Sections 4, 5, 6 and 13 describe
> behaviour that differs; the algorithm sections are identical.

- [1. What the tool computes](#1-what-the-tool-computes)
- [2. Architecture](#2-architecture)
- [3. Dataflow](#3-dataflow)
- [4. Stage 1 — loading and normalisation](#4-stage-1--loading-and-normalisation)
- [5. Stage 2 — candidate extraction](#5-stage-2--candidate-extraction)
- [6. Stage 3 — termhood by C-value](#6-stage-3--termhood-by-c-value)
- [7. Stage 4 — canonicalisation](#7-stage-4--canonicalisation)
- [8. Stage 5 — relations by PPMI](#8-stage-5--relations-by-ppmi)
- [9. Stage 6 — trends](#9-stage-6--trends)
- [10. Complexity and measured cost](#10-complexity-and-measured-cost)
- [11. NLP concepts, collected](#11-nlp-concepts-collected)
- [12. Rejected alternatives](#12-rejected-alternatives)
- [13. Defects in this version](#13-defects-in-this-version)
- [14. Known limits](#14-known-limits)

---

## 1. What the tool computes

Input is a table of patent records with a title, an abstract, and optionally a
date. Output is four things:

1. **Terms** — the technologies named in the corpus, ranked by termhood.
2. **Variants** — which surface forms name the same technology.
3. **Relations** — which technologies are associated above chance.
4. **Trends** — which are rising and which are fading, when dates exist.

Everything runs locally. Two models are loaded: a spaCy pipeline for syntax and
a sentence-transformer for meaning. No network calls at inference, no Java, no
TensorFlow.

---

## 2. Architecture

Two modules, split along one line: **`pipeline.py` never imports Streamlit.**

```
pipeline.py     pure functions + dataclasses; no I/O beyond reading a table,
                no global state, models injected as arguments
app.py          rendering, caching, widgets; imports pipeline, not the reverse
run_local.py    process setup (model paths, OpenMP, Streamlit credentials)
```

Three consequences worth the constraint:

- **Testable.** Every stage is callable with a hand-built `Counter` or a small
  DataFrame. The adversarial suite that produced §14 drives `pipeline` directly
  and never starts a server.
- **Reusable.** `pipeline.run(...)` works from a script, a notebook or a
  scheduler. The UI is one consumer.
- **Injectable models.** `run()` takes `nlp=` and `embedder=`. Streamlit owns
  the model lifetime through `@st.cache_resource`; the pipeline does not know
  that caching exists. Loading is lazy — if you pass models in, the pipeline
  never touches `spacy.load`.

`Config` is a frozen slotted dataclass. Frozen because it is a cache key and
must not drift under a memoised call; slotted for the memory floor. The two
model fields use `field(default_factory=...)` reading `TECHSCOPE_SPACY_MODEL`
and `TECHSCOPE_EMBED_MODEL`, so an offline machine redirects to on-disk copies
without touching code.

`InputError` carries **every** problem, not the first. A user fixing a column
mapping should learn about all four mistakes in one pass, not four passes.

---

## 3. Dataflow

Types at each boundary. `D` = documents, `T` = surviving terms.

```
  DataFrame  (raw table)
      │
      │  check()               -> list[str]      every problem, or empty
      │  load()                -> Corpus
      ▼
  Corpus{titles[D], abstracts[D], years[D], docs[D]}
      │
      │  candidates()          spaCy: tokenise, tag, parse, lemmatise
      ▼
  per_doc: list[list[str]]     normalised terms, deduped within a document
  df:      Counter             document frequency, not raw frequency
      │
      │  c_values()            termhood + frequency floor + boilerplate ceiling
      ▼
  scores: dict[str, float]  ──sort, truncate to max_terms──>  ranked: list[str]
      │
      │  embedder.encode(ranked)         -> ndarray[T, 384]
      │  canonicalise()                  lexical merge + mutual k-NN + union-find
      ▼
  canonical: dict[str, str]    every surface form -> its group's name
  variants:  dict[str, list]   group name -> the other forms
  keep = set(variants)         canonical names only
      │
      ├─ cooccurrence()  -> edges  DataFrame[source, target, documents, ppmi]
      └─ trends()        -> DataFrame[term, documents, growth, recent share, <period>...]
      ▼
  Result{corpus, terms, edges, trends, canonical, variants, per_doc,
         doc_canon}
```

`doc_canon` is each document's canonical term set, computed **once** in `run()`
and carried on the result. Three consumers need it — the term counts, the
Explore tab, and any caller — and rebuilding it per term is the difference
between linear and quadratic.

There is no diagnostic channel in this version. When a result comes back empty
the reason is not recorded anywhere, which is the root of several entries in
section 13. v2 adds `Result.notes` for exactly that.

---

## 4. Stage 1 — loading and normalisation

### `clean()`

Four transformations, in order, each earned from real patent text:

| Pattern | Removes | Why |
|---|---|---|
| `^\s*\([^)]*\)\s*\n` | `(EP1234A1)\n` | Some exports prefix the abstract with the publication number. It is not prose and pollutes the first noun chunk. |
| `PROBLEM TO BE SOLVED\|SOLUTION\|...` | JPO section headings | Japanese-origin abstracts are structured with headings that would otherwise become high-frequency "terms". |
| `;` → `.` | — | The parser treats a semicolon as intra-sentential, so a claim-style list becomes one enormous sentence and one enormous noun chunk. |
| `–`, `—` → `-`, then `\s+` → ` ` | typographic dashes, newlines | Uniform tokenisation. |

### Title and abstract are concatenated

`docs[i] = f"{title}. {abstract}"`. One parse per record rather than two, and
titles are dense with terminology — often naming the technology the abstract
only alludes to. Document frequency still counts each record once, because
`candidates()` dedupes within a document before counting.

### Date parsing

```python
m = re.search(r"(19|20)\d{2}", str(value))
```

That is the whole of date handling in this version: the first four digits
beginning `19` or `20`, anywhere in the stringified cell.

It is unsound, and it is the most consequential defect in v1. The pattern is
unanchored, so it matches **inside a longer number**:

```
"US2024123456A1"  ->  2024
```

Point the date column at a publication-number column — an easy mistake, and the
sidebar auto-detect will do it for you if that column is named something like
"Priority number" — and every row yields a year, the Trends tab populates, and
every number in it is fiction. Nothing warns you, because nothing checks.

It also silently drops what it cannot read:

- Excel serial numbers (`45000`) → `None`. pandas leaves dates as raw integers
  whenever the column contains any non-date cell, which is common in real
  exports, so a perfectly good date column can yield no trends at all.
- Years before 1900 or after 2099 → `None`.

`datetime` and `Timestamp` values happen to work, but only because
`str(Timestamp)` renders as `2019-04-02 00:00:00` and the regex finds `2019`
inside it — not by design.

**Check your date column by hand before trusting the Trends tab.** v2 replaces
this with type dispatch, boundary-anchored matching, and a `date_warning()` that
reports a column yielding no usable years.

---

## 5. Stage 2 — candidate extraction

### The spaCy pipeline

`spacy.load(model, disable=["ner"])`. What each surviving component provides:

| Component | Provides | Needed for |
|---|---|---|
| `tok2vec` | context vectors | everything downstream |
| `tagger` | `pos_` (coarse), `tag_` (fine, Penn Treebank) | the participle rule below |
| `parser` | dependency arcs, and therefore `doc.noun_chunks` | candidate spans |
| `attribute_ruler` + `lemmatizer` | `lemma_` | plural/singular unification |

**NER is disabled.** Nothing here reads entities. Measured on 600 documents
with warm models, it costs **34%** of parse time (1.26s → 0.84s), which is pure
waste on this workload.

The parser cannot be disabled — `noun_chunks` raises `E029` without it. This
version does not check for that up front, so pointing `TECHSCOPE_SPACY_MODEL` at
a blank or sentence-only pipeline produces a raw traceback from inside the
extraction loop rather than a message. The tagger cannot go either, since the
lemmatiser depends on it and the participle rule below reads `tag_`.

### Noun chunks as the candidate space

A noun chunk is a maximal noun phrase with no nested NP — spaCy derives them by
walking the dependency parse, not by regex or POS pattern. This is why a parser
is required and a tagger alone will not do.

Chunks are the candidate space because a technology name is almost always a
noun phrase. Using n-grams instead would generate `tank with a` and `and the
vacuum`, which then need filtering by exactly the syntax the parser already
computed.

### The participle rule — the one genuinely subtle piece

spaCy's chunker returns spans that can name **two** things:

```
"fuel cell membrane comprising catalyst layer"      ← ONE chunk
```

At 6 words this exceeded `max_term_words` and was dropped entirely, losing both
`fuel cell membrane` and `catalyst layer`. But naively cutting at every verb
also destroys real terms, because premodifiers are verbs too:

```
"distributed sensor array"          "distributed" is a VERB
"additively manufactured heat exchanger"   "manufactured" is a VERB
```

The distinguishing signal is **participle tense**, visible in the fine-grained
Penn tag, not the coarse POS:

| Token | `pos_` | `tag_` | `dep_` | Role |
|---|---|---|---|---|
| `comprising` | VERB | **VBG** | amod | present participle — opens a clause |
| `distributed` | VERB | **VBN** | amod | past participle — premodifier |
| `manufactured` | VERB | **VBN** | amod | past participle — premodifier |
| `additively` | ADV | RB | advmod | modifies the participle |

So `_can_continue()` admits `NOUN/PROPN/ADJ/NUM`, any `VBN`, and an `advmod`
adverb — and stops at a `VBG`. `_terms_in_chunk()` splits the chunk into runs of
admissible tokens and yields **every** run, so the tail is kept rather than
discarded:

```
"fuel cell membrane comprising catalyst layer"
   -> ["fuel cell membrane", "catalyst layer"]
```

`_can_start()` is stricter than `_can_continue()`: a run may not begin with a
bare adverb unless a `VBN` follows it, so `additively manufactured heat
exchanger` survives intact while a stray adverb does not lead a term.

Measured effect on a 3,000-row corpus: **91 terms → 135 terms**, with
`distributed sensor array` and `additively manufactured heat exchanger`
recovered.

### Normalisation, in `_finish()`

1. Drop punctuation, whitespace and determiners (`_DET`, 26 entries including
   `said`, `such`, ordinals — patent-specific determiners).
2. Strip leading tokens that cannot start a name, and leading evaluative
   modifiers (`_FILLER_MOD`: `improved`, `novel`, `exemplary`, ...). Without
   this, `improved sensor array` and `sensor array` are two technologies.
   **Trailing** heads are deliberately *not* stripped here — a "welding method"
   really is called that, and removing the head would merge distinct terms.
3. Lowercase every token; **lemmatise only the last**. The head noun carries the
   number inflection (`sensors` → `sensor`), while modifiers must stay as
   written — lemmatising `composites pressure vessel` word-by-word corrupts
   established compounds.
4. Reject anything with a non-alphanumeric token (allowing `-` and `/`).
5. Reject single-word terms whose head is in `_BAD_HEAD` (`invention`,
   `embodiment`, `apparatus`, ...).
6. Keep only terms of `min_term_words..max_term_words` words.

Within a document, terms are deduped via an insertion-ordered `dict`, so `df` is
**document frequency**, not raw frequency. This matters: raw frequency lets one
verbose patent dominate the whole corpus.

### Batching

```python
nlp.pipe(corpus.docs, batch_size=48)
```

Documents are batched **by count**. The parser allocates per character, so this
is free for short abstracts and fatal for long ones — a fixed batch size does
not know how big the documents inside it are. Measured, 48 documents of 100,000
characters each:

| Batching | Result |
|---|---|
| `batch_size=48` (this version) | `MemoryError`: unable to allocate 1023 MiB |
| `batch_size=8`, same input | completes, +34 MB peak |

Same input, same machine. It is not a capacity limit, only unbounded batching.

Real abstracts run 1–3k characters, so ordinary use never reaches this. Mapping
the abstract column to a claims or description field does, immediately.

There is no ceiling check either: a single document longer than `nlp.max_length`
(2,000,000 characters, as raised at load) raises `E088` rather than truncating.

v2 bounds batches by total characters instead. Measured against this version on
a byte-identical corpus, that removes the crash and drops peak RSS at 40,000
rows from 70 MB to 39 MB. Runtime is unchanged — the cap limits what the parser
can be asked for at once, it does not make parsing faster.

---

## 6. Stage 3 — termhood by C-value

### The nested-term problem

Raw frequency always prefers the shorter term. `storage tank` occurs at least as
often as `cryogenic storage tank`, because every occurrence of the longer
contains the shorter — even when the short form never appears on its own and is
not what the corpus is about.

**C-value** (Frantzi, Ananiadou & Mima, 2000) corrects exactly this:

```
C(a) = log₂|a| · f(a)                                    a nested in nothing

C(a) = log₂|a| · ( f(a) − (1/|T_a|) · Σ_{b ∈ T_a} f(b) )  otherwise
```

where `|a|` is the term's word count and `T_a` is the set of extracted terms
properly containing `a`.

- `log₂|a|` rewards specificity — longer phrases are more likely real terms.
- Subtracting the mean frequency of containing terms removes the frequency a
  term only has by virtue of sitting inside a longer one.

Two deliberate deviations from the paper:

- **`f` is document frequency, not raw frequency**, consistent with the rest of
  the pipeline.
- **Unigrams get weight 1, not 0.** `log₂(1) = 0` would zero every single-word
  term; `math.log2(...) or 1.0` substitutes 1.

Terms scoring `≤ 0` are dropped. That is the nested-term elimination doing its
job: a term whose containers are as frequent as it is contributes nothing.

Containment is tested on padded strings — `" storage tank "` in
`" cryogenic storage tank "` — so `tank` does not match inside `tanker`.

### The boilerplate ceiling

C-value **rewards** frequency, so it cannot catch a phrase that appears
everywhere. In testing, `hydrogen service` topped the ranking at C-value 240 in
a corpus where it appeared in all 240 documents. A term in every document
discriminates nothing — the same reasoning behind `max_df` in a TF-IDF
vectoriser.

```python
ceiling = int(cfg.max_doc_ratio * n_docs) if n_docs else None
terms = [t for t, c in df.items()
         if c >= cfg.min_term_freq and (ceiling is None or c <= ceiling)]
```

Correct for a large, varied corpus. Wrong in two ways otherwise.

**The filters can cross.** The ceiling is not clamped to the floor, so on a small
corpus the admissible band is empty and no term can satisfy both conditions:

| Documents | floor | ceiling `int(0.60·n)` | admissible |
|---|---|---|---|
| 1 | 2 | 0 | **impossible** |
| 2 | 2 | 1 | **impossible** |
| 3 | 2 | 1 | **impossible** |
| 4 | 2 | 2 | 2..2 |
| 10 | 2 | 6 | 2..6 |

Any corpus of three documents or fewer returns nothing, always, whatever it
contains.

**A narrow corpus is erased.** If every term appears in more than 60% of the
documents — which is what a tightly focused patent set looks like — every term
is above the ceiling and all of them are dropped. Verified: ten documents
sharing their vocabulary yield zero terms.

The workaround is to raise **Drop terms above this share of documents** toward
1.00 in the sidebar. v2 clamps the ceiling to at least the floor, and skips it
entirely when it would remove everything, recording the reason in `notes`.

---

## 7. Stage 4 — canonicalisation

`cryogenic tank`, `cryogenic storage tank` and `cryogenic storage tank
apparatus` are one technology. Grouping them is what makes counts meaningful.

### Sentence embeddings

`all-MiniLM-L6-v2`: 6 transformer layers, **384 dimensions**, mean-pooled token
embeddings, trained contrastively on roughly a billion sentence pairs so that
cosine distance approximates semantic relatedness.

The underlying idea is the **distributional hypothesis** (Harris 1954; Firth
1957): words in similar contexts have similar meanings. A contrastively trained
sentence encoder is that hypothesis applied to whole phrases rather than words.

Vectors are L2-normalised, so the Gram matrix `v @ v.T` is the full cosine
similarity matrix. The diagonal is set to `-1` to exclude self-matches.

### Why mutual k-NN, not a threshold

A plain threshold — "group any pair above 0.8" — fails because of **hubness**
(Radovanović et al., 2010): in high-dimensional space a few points are the
nearest neighbour of disproportionately many others. A generic phrase becomes a
hub and absorbs everything transitively, and one group swallows the corpus.

Greedy single-link chaining fails differently: it is order-dependent, so the
result changes if the input is shuffled, and it silently drops terms.

The rule used:

```python
edge(i, j)  ⟺  j ∈ kNN(i)  ∧  i ∈ kNN(j)  ∧  sim(i,j) ≥ floor
```

**Mutuality** is what defeats hubness: a hub is in everyone's neighbour list,
but they are not all in its top-k, so almost no edge is mutual. Groups are
**connected components** over these edges, computed with union-find — so the
result is symmetric, deterministic, and independent of iteration order.

`np.argpartition(-sim, kth=k-1)` gets the top-k in O(n) per row instead of
sorting; only membership matters, not order.

### The similarity floor is not a smooth dial

Measured on real term embeddings, the pairwise similarity distribution is
**bimodal**: synonyms sit above 0.85, unrelated pairs below 0.40, and the band
between is nearly empty. Moving the floor from 0.85 to 0.40 changes almost
nothing; moving it from 0.60 to 0.75 changed one wrong grouping
(`pipeline monitoring system` had been absorbed into `pipeline segment`). The
default is **0.75**, and the UI says so in the tooltip so the control is not
mistaken for a fine adjustment.

### The exact lexical merge

Embeddings alone miss `fuel cell membrane apparatus` ↔ `fuel cell membrane`,
because each one's k neighbours are crowded with other terms of its own shape,
so mutuality fails. Before the k-NN pass, a deterministic string rule runs:

> Repeatedly strip trailing `_FILLER_HEAD` words. If the stripped form was
> **independently extracted**, union the two.

The independence condition is what makes this safe: it never invents a term, and
it cannot merge two things that merely share a head noun. `welding method` with
no bare `welding` alongside it is left exactly as it is.

### Choosing the group's label

```python
head = max(members, key=lambda i: (-_filler(terms[i]), df[terms[i]], -len(terms[i])))
```

Frequency alone picks the *worst* label. `composite pressure vessel apparatus`
outnumbers `composite pressure vessel` because every title carries the head
noun. So boilerplate count ranks first, document frequency second, brevity third.
`_filler()` counts `_FILLER_HEAD` words after position 0 plus any `_FILLER_MOD`
word — used only for labelling, never to drop a term.

---

## 8. Stage 5 — relations by PPMI

```
PMI(a,b) = log₂ [ p(a,b) / (p(a) · p(b)) ]
PPMI     = max(0, PMI)
```

Probabilities are document-level: `p(a) = df(a)/D`, `p(a,b) = df(a∧b)/D`.

Raw co-occurrence ranks the two most common terms top whether or not they are
related — they co-occur constantly simply by being common. PMI divides out each
term's own rate, leaving association **above chance**: the denominator is what
you would expect if the terms were independent, so the ratio is how much reality
exceeds independence.

Clipping at zero (**positive** PMI) is standard practice. Negative PMI claims
two terms co-occur *less* than chance, which on sparse document counts is
dominated by noise rather than signal.

Pairs below `min_edge_docs` are dropped before scoring — PMI is notoriously
unstable on low counts, where a single joint occurrence of two rare terms yields
a huge score. Results are sorted by PPMI then joint count, truncated to
`max_edges`.

The pair loop is O(D · k²) in kept terms per document. Measured at 500
documents: 20 terms/doc → 0.01s; 400 terms/doc → 6.0s. Bounded in practice by
`max_terms`.

---

## 9. Stage 6 — trends

### Periods are derived, never hardcoded

`periods()` reads the observed year range and cuts it into `n` contiguous
buckets. A hardcoded calendar — the failure mode in the tool this replaces,
which had twelve fixed buckets from 1980 — collapses a post-2010 corpus into two
bars and puts everything recent into one open-ended bucket that only grows.

Edges are deduped through a set, so a range shorter than the requested bucket
count yields fewer, non-empty buckets instead of empty ones.

### Share, not count

For each term and period, the tool records both the count and the **share**:
`hits / documents_in_period`. Growth is computed on share. Otherwise a period
that simply contains more patents reads as growth for every term in it.

### Growth is a least-squares slope

```python
slope = np.polyfit(x, shares, 1)[0]
```

The slope of the ordinary-least-squares line through all period shares — not a
last-versus-first comparison, which throws away the middle and is hostage to two
noisy endpoints.

Each period is tallied **once** into a `Counter` before the per-term loop.
Scanning the corpus per term per period costs `terms × periods × documents`,
which is where a 5,000-row file stops being interactive.

The UI plots growth against document count, giving four quadrants — established
and growing, established and fading, small and emerging, small and fading. The
top-left quadrant is usually the interesting one.

---

## 10. Complexity and measured cost

`D` documents, `C` total characters, `T` terms after filtering, `R` = `max_terms`,
`k` kept terms per document, `P` periods.

| Stage | Complexity | Notes |
|---|---|---|
| `clean` | O(C) | regex passes |
| `candidates` | O(C) | **dominant**; spaCy parsing is linear in characters |
| `c_values` | O(T²) worst case | nested-containment scan, T is post-filter |
| `encode` | O(R) | R ≤ 800, one batched forward pass |
| `canonicalise` | O(R²) | Gram matrix; 2.56 MB at R=800 |
| `cooccurrence` | O(D·k²) | pair enumeration |
| `trends` | O(D·k + P·D + T·P) | after hoisting the period tallies |

Measured end to end (`en_core_web_lg`, warm models, one process per version, the
same generated corpus for both — verified byte-identical):

| Rows | v2 time | v2 peak RSS | v1 time | v1 peak RSS | Terms |
|---|---|---|---|---|---|
| 1,000 | 1.8s | 20 MB | 1.9s | 23 MB | 116 |
| 5,000 | 7.7s | 14 MB | 7.5s | 17 MB | 109 |
| 15,000 | 22.0s | 25 MB | 21.2s | 24 MB | 116 |
| 40,000 | 57.9s | **39 MB** | 55.8s | **70 MB** | 115 |

Linear in rows. Parsing dominates; everything after it is noise by comparison.

Both versions produce identical term and edge counts: the v2 changes are
correctness and failure handling, not different analysis, so this version gives
the same answers on input that does not trigger one of the defects in section
13. Runtime is the same within measurement noise. The one real difference is
peak memory at scale, where this version's fixed batch size allows a larger
transient allocation.

---

## 11. NLP concepts, collected

| Concept | Where | Role |
|---|---|---|
| Tokenisation | spaCy | segmentation before anything else |
| POS tagging — coarse (`pos_`) and fine (`tag_`) | `_can_continue`, `_can_start` | the VBG/VBN distinction lives in the fine tag only |
| Dependency parsing | `doc.noun_chunks` | chunks are derived from parse structure, not patterns |
| Noun-phrase chunking | `candidates` | the candidate space |
| Lemmatisation | `_finish` | head-noun only, to unify number |
| Stop-word / determiner filtering | `_DET`, `_BAD_HEAD` | domain-specific, not a generic English list |
| Termhood — C-value | `c_values` | nested-term correction |
| Document frequency | throughout | one verbose patent cannot dominate |
| Document-frequency ceiling (`max_df`) | `c_values` | boilerplate removal |
| Distributional hypothesis | embeddings | the premise the encoder rests on |
| Sentence embeddings, mean pooling | `encode` | phrase-level meaning, 384-d |
| Cosine similarity | `v @ v.T` | on L2-normalised vectors |
| Hubness in high dimensions | `canonicalise` | the reason a threshold fails |
| Mutual k-NN graph | `canonicalise` | hubness-resistant grouping |
| Connected components / union-find | `_Union` | order-independent clusters |
| Morphological variant merging | lexical merge | exact, complements the semantic pass |
| PMI / PPMI | `cooccurrence` | association above chance |
| Least-squares trend estimation | `trends` | growth from all periods |

---

## 12. Rejected alternatives

| Rejected | Why |
|---|---|
| TF-IDF top-n for terms | No nested-term correction; ranks `storage tank` over `cryogenic storage tank`. |
| Raw n-grams as candidates | Generates syntactic garbage that then needs filtering by the parse you avoided computing. |
| KMeans / HDBSCAN for grouping | Needs `k` or density parameters that vary per corpus; produces clusters of *related* terms, not *same* terms. Wrong granularity. |
| Plain similarity threshold | Hubness — one generic phrase absorbs the corpus transitively. |
| Greedy single-link chaining | Order-dependent, silently drops terms. |
| Raw co-occurrence counts | Ranks the two commonest terms top regardless of relatedness. |
| LDA / BERTopic | Answers "what topics exist", not "what technologies are named and how do they relate". Also non-deterministic and slower. |
| Full-vocabulary MLM ranking | The approach in the predecessor tool. Materialised a 13.4 GB cache; the question it answers is not the question asked. |
| Last-period vs first-period growth | Discards the middle, hostage to two noisy endpoints. |
| Hardcoded date buckets | Collapses any corpus that does not span the hardcoded range. |

---

## 13. Defects in this version

Found by adversarial testing. All are fixed in `v2/`; the algorithms are
unchanged between the two.

The dangerous ones are silent — they return a confident-looking answer that is
empty or fabricated, rather than an error you would notice.

| Defect | Effect |
|---|---|
| Unanchored year regex | A publication-number column read as dates gives a fully populated, entirely fictional Trends tab. Nothing warns. |
| Excel serials unparsed | A valid date column yields no years, so Trends silently disappears. |
| Floor/ceiling crossover | Any corpus of three documents or fewer returns nothing at all. |
| Ceiling erases a narrow corpus | A focused patent set returns nothing, and the UI message suggests the wrong fix. |
| No language detection | Non-English text returns an empty report, indistinguishable from a corpus with no technology in it. |
| `title == abstract` accepted | The text is counted twice and every term inflated. Nothing rejects it. |
| `batch_size=48` | `MemoryError` on long text. |
| No `nlp.max_length` guard | `E088` on a document over 2,000,000 characters. |
| No parser check | `E029` as a raw traceback from mid-loop. |
| Model failures uncaught | `app.py` catches only `InputError`; everything else reaches the user as a Streamlit stack trace. |
| DataFrame re-hashed each rerun | `analyse()` takes `frame` with no underscore, so Streamlit hashes the whole table on every widget change. Measured at 79 ms for 50,000 rows — a wrong docstring rather than a real cost, but the docstring claims the opposite. |
| OpenMP guard only in `run_local.py` | `streamlit run app.py` can abort at import on Windows with no traceback. |
| Streamlit first-run prompt | Both entry points hang at an interactive `Email:` prompt on a machine that has never run Streamlit. |
| Unconditional ellipsis | A six-character abstract renders as `Short.…` in the Explore tab. |

## 14. Known limits

Honest ones, found by testing and not fixed:

- **English only.** Detected and reported, not handled. A non-English corpus
  needs a matching spaCy model and a multilingual encoder.
- **`c_values` containment is O(T²).** Fine at the default `max_terms`; would
  need an index at much larger term counts.
- **`cooccurrence` is O(D·k²).** A dense corpus at `max_terms=800` would be slow.
- **Single-machine, single-process.** No parallelism across cores; spaCy's `n_process`
  is not used because process startup dominates at these corpus sizes.
- **Chunking depends on parse quality.** `en_core_web_sm` mis-parses some
  patent syntax; `_md`/`_lg` measurably help and are worth the disk.
- **Trends need a real date column.** Now reported rather than fabricated, but
  a corpus without usable dates gets three tabs, not four.
- **All correctness testing used synthetic corpora.** The defects in §13 were
  found by construction. Real exports will surface things these did not.
