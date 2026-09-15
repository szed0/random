# trt-pb in a Gemini chat

The original Streamlit tool's pipeline, driven from a plain chat. **No Gem** -
a Gem adds a second thing that can fail, and the one it fails at is code
execution. Here everything lives in one pasted message plus two attachments.

```
prompt  +  trt_pb.py  +  your patent export (CSV/XLSX)
```

## The loop

1. Paste [PROMPT.md](PROMPT.md) with both files attached. You get the corpus
   summary and a numbered table of **technical terms**.
2. Reply with a number - your **primary** term. You get two graphs,
   occurrences by year and the cumulative curve, then two numbered menus: the
   **relationships** that term takes part in, and its **secondary terms**.
3. Reply with a number, optionally plus a relationship (`1 Inclusion`). You get
   the two terms together: per year, and cumulative.

After the first message you only ever send a number. The uploads stay available
for the whole chat, so nothing needs re-attaching.

## Why the term list is clean

The original ran spaCy's keyphrase chunker, which keeps `<J.*>*<N.*>+` -
adjectives followed by nouns - and then dropped everything shorter than two
words. That is the whole reason it never showed you "operating" or "capable".

There is no spaCy in a chat sandbox, so `trt_pb.py` reproduces the filter with
a rule-based chunker: function words break a chunk, a chunk must end in a
noun-like head, participles may modify but never head it, and a term is two to
four words. Patent drafting language is barred outright, so "present
invention", "defined herein" and "salt thereof" cannot appear either.

On a 122-patent hydrogen/pharma corpus that takes 235 raw candidates down to 97
terms, and the top of the list is `pharmaceutical composition`, `acceptable
salt`, `fuel cell system`, `hydrogen gas`, `fuel cell stack`.

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
| spaCy `<J.*>*<N.*>+` chunker | rule-based chunker, same shape |
| TF-IDF + PatentBERT per-abstract top-N | TF-IDF summed over the corpus |
| PatentBERT MLM ranking for secondary terms, min 5 | co-occurrence with a lift correction, min 5 |
| spaCy ADP tagging | fixed preposition list (a closed class, so nearly identical) |
| synonym grouping at cosine 0.80 | not reproduced - no embeddings in the sandbox |

## API

```python
state = load("export.csv")          # corpus + numbered technical terms
terms(state, top=80)                # widen the menu
terms(state, contains="hydrogen")   # search it
trt(state, 3)                       # two graphs + relationship and secondary menus
pair(state, 3, 1)                   # the pair: two graphs
pair(state, 3, 1, relation="Inclusion")
triples(state, 3, 1)                # the T1-preposition-T2 evidence rows
documents_for(state, 3, 1)          # the patents behind a bar
```

## Input

CSV or XLSX with a title or abstract column and a date column. Column names are
matched loosely; an ORBIT export works unchanged. The year is the application
year, and undated rows are dropped from the charts with the count printed.
