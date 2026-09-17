# trt-pb in a Gemini chat

The original Streamlit tool's pipeline, driven from a plain chat. **No Gem** -
a Gem adds a second thing that can fail, and the one it fails at is code
execution. Here everything lives in one pasted message plus two attachments.

```
prompt  +  trt_pb.py  +  your patent export (CSV/XLSX)
```

## The loop

1. Paste [PROMPT.md](PROMPT.md) with both files attached. The code harvests
   every plausible term - hundreds of them, noise included - and **Gemini reads
   that list and curates it**, then commits the result. You get a numbered
   table of **technical terms**.
2. Reply with a number - your **primary** term. You get two graphs,
   occurrences by year and the cumulative curve, then two numbered menus: the
   **relationships** that term takes part in, and its **secondary terms**.
3. Reply with a number, optionally plus a relationship (`1 Inclusion`). You get
   the two terms together: per year, and cumulative.

After the first message you only ever send a number. Uploads and anything the
code writes stay available for the whole chat, so nothing needs re-attaching -
the curated vocabulary lives in `vocab.json` and every later turn reloads it.

## Recall in the code, judgment in the model

The first version filtered terms with rules alone. That was precise and too
narrow: it never proposed `anode`, `catalyst` or `steam reforming`, because the
strict chunker demanded two words and a noun head.

So the split is now:

| stage | who does it | tuned for |
| --- | --- | --- |
| extraction | `harvest()` | recall - ~675 candidates from 122 patents |
| filtering | Gemini, reading the list | judgment |
| counting and drawing | `trt_pb.py` | determinism |

`harvest()` proposes every sub-phrase of a noun-phrase run that ends on a noun
head - so `steam reforming feed` yields `steam reforming` and `steam` as well
as itself - plus acronyms and hyphenated compounds lifted from the original
casing, which is how `SOFC` gets in at all.

Three filters then do the work rules can actually do well:

**Heads must be nouns.** A verb, a participle, an adverb or an adjective can
modify a technology but cannot be one. `high pressure` and `inner compartment`
survive; bare `include`, `using`, `relate`, `least`, `more` and `useful` never
reach the list.

**Fragments die by branching entropy.** A real term is free at both ends -
`fuel cell system` is preceded by "the", "said", "a", "hydrogen" and followed
by "comprising", "includes", a full stop. A fragment is not: `cell system` is
preceded by `fuel` and by nothing else, so the entropy of its left neighbours
is exactly zero. Taking the smaller of the two ends catches fragments cut from
either side, and it reads the answer off the corpus instead of off a list of
words someone thought of in advance. On the test corpus this removes 100
candidates, among them `cell system`, `carbonate fuel cell` and
`molten carbonate fuel` - while keeping `molten carbonate fuel cell`.

**Ranking is termhood, not frequency.** C-value for nestedness, multiplied by
boundary freedom and by how often the term earns a place in a patent *title* -
titles are written to name the technology and nothing else.

The difference, same corpus, top of the list:

| before | after |
| --- | --- |
| hydrogen, fuel, compound, cell, fuel cell, fuel cell system, **include**, composition, **least**, stream, disease, metal, fuel cell stack, anode, treatment, reaction, temperature, **using**, **cell system**, formula | hydrogen, fuel, compound, fuel cell, cell, fuel cell system, metal, stream, disease, composition, temperature, hydrogen gas, fuel cell stack, treatment, reaction, anode, catalyst, inhibitor, hydrogen production apparatus, kinase |

Further down, the new list surfaces `sofc`, `membrane`, `pyrolysis`,
`electrolyzer`, `heterocyclic compound`, `steam reforming` and `hydrocarbon`,
none of which the old one proposed at all.

Gemini then keeps what a patent engineer would call a technology and drops
verbs, drafting language, bare category words and fragments. On the test corpus
it kept 132 of 675. You can correct it in one sentence - "drop X and Y and
re-commit" - and the vocabulary is just a JSON file, so nothing is locked in.

## Why the relationships are not just co-occurrence

Every preposition in a sentence splits it into T1 - preposition - T2, exactly
as the original's spaCy ADP pass did, and the preposition is bucketed with the
original's own lists:

| bucket | prepositions |
| --- | --- |
| Inclusion | of in with from on at within includes by utilizes |
| Objective | for |
| Effect | to across against |
| Process | during into through via |
| Likeness | as |
| Misc | anything else |

So `pair(state, 3, 1, relation="Inclusion")` counts a patent only when the two
terms are the two sides of an Inclusion triple in it. On the test corpus
`fuel cell system` and `fuel cell stack` share 6 patents but only 2 by
Inclusion, and `triples()` shows why: *US7846599B2 - fuel cell system
**includes** fuel cell stack*.

## What is approximated

| original | here |
| --- | --- |
| spaCy `<J.*>*<N.*>+` chunker | rule-based chunker, widened for recall |
| TF-IDF + PatentBERT per-abstract top-N | C-value, then the model's own judgment |
| PatentBERT MLM ranking for secondary terms, min 5 | co-occurrence with a lift correction, min 5 |
| spaCy ADP tagging | fixed preposition list (a closed class, so nearly identical) |
| synonym grouping at cosine 0.80 | not reproduced - no embeddings in the sandbox |

## API

```python
state = harvest("export.csv")       # step 1a: every plausible candidate
commit(state, ["term", ...])        # step 1b: the curated vocabulary -> vocab.json
commit(state, drop=["term", ...])   # or keep everything except these
state = load("export.csv")          # later turns: reads vocab.json automatically
terms(state, top=80)                # widen the menu
terms(state, contains="hydrogen")   # search it
trt(state, 3)                       # two graphs + relationship and secondary menus
pair(state, 3, 1)                   # the pair: two graphs
pair(state, 3, 1, relation="Inclusion")
triples(state, 3, 1)                # the T1-preposition-T2 evidence rows
documents_for(state, 3, 1)          # the patents behind a bar
```

`commit()` and `load()` count through the same routine, so the row numbers in
the menu you are shown are the row numbers the next turn resolves. They did not
always: the first version numbered the commit menu from harvest counts and the
load menu from regex counts, and row 10 changed meaning between turns.

## Input

CSV or XLSX with a title or abstract column and a date column. Column names are
matched loosely; an ORBIT export works unchanged. The year is the application
year, and undated rows are dropped from the charts with the count printed.
