# Technical terms from a patent export

Turn an Excel or CSV patent export into a ranked list of technical terms, then
drill into any one of them and trace every number back to a publication
number.

Built to run inside a Gemini Gem's code sandbox: standard library plus pandas,
with `nltk`'s PorterStemmer used when present and a built-in stemmer standing
in when it is not.

## Three steps

```python
import trt_terms as tt

state = tt.extract("export.xlsx")     # -> technical_terms.csv
tt.drill(state, 5)                    # -> subterms_<term>.csv
tt.trace(state, "current collector", within="secondary battery")
```

**1. Extract.** Every technical term in the corpus, with the patents behind
each one.

| column | meaning |
|---|---|
| `term` | display form — singular, never a gerund |
| `stem` | grouping key, so inflections are one row |
| `tfidf` | corpus term frequency × ln(N / document frequency) |
| `score` | c-value × cohesion — termhood, not popularity |
| `n_patents` | how many patents contain it |
| `patents` | every publication number it appears in |

**2. Drill.** Sub-terms scoped to the patents carrying the selected term, not
to the corpus. Adds `n_patents_with_both`, `share_of_primary`, `lift`, and the
publication numbers behind each row. Read `lift`: a sub-term near 1 is common
everywhere and says nothing about the primary.

**3. Trace.** One row per patent — publication number, the matched surface
form, and the sentence it appeared in.

## What counts as a technical term

A run of open-class words, two to four long, surviving four filters:

- **Stemmed**, so a term and its plural are one row. The stem is the key only;
  the row is labelled with a readable singular form, so you see
  `hydrogen leakage reduction` and never `hydrogen leakag reduct`.
- **Hyphen-insensitive**, so `non-aqueous electrolyte` and
  `nonaqueous electrolyte` are one row rather than two half-counts.
- **No gerund heads.** `reducing leakage` is an activity, not a technology.
  This also drops `coating`, `housing` and `bearing`, which are real nouns —
  `drop_gerunds=False` turns it off.
- **No verb heads**, decided by the corpus rather than a list. A word is a
  verb here when its -ing and -ed forms both occur *and* the bare form is
  outnumbered by its inflections. That second condition is what keeps
  `fuel cell stack`: `stack` is conjugated somewhere in any patent corpus but
  appears overwhelmingly as a bare noun, while `include` does not.

Nothing about the subject matter is hardcoded. The only fixed vocabulary is
`CLOSED_CLASS` — determiners, prepositions, conjunctions, pronouns,
auxiliaries, degree adverbs — which is English grammar and identical for a
corpus about batteries and one about crop rotation.

## Input

Excel or CSV. Columns are matched loosely by name:

| needed | matched on |
|---|---|
| abstract *or* title | `abstract`, `summary`, `title` |
| publication number | `publication number`, `patent number`, `patent id`, `pubno`, `id` |
| description (optional) | `description`, `english description`, `claims` |

Without a publication-number column nothing can be traced, and `extract`
stops rather than produce untraceable output.

## Running it as a Gem

`SYSTEM_PROMPT.md` goes in the Gem's Instructions. `trt_terms.py` goes in
Knowledge — a knowledge file is a real file in the code sandbox, so it
survives between turns and needs no re-attaching. The export is a chat
attachment.

Code execution exists only inside a Gem. An ordinary Gemini chat has no Python
tool and will compute numbers in its head and present them as program output,
which is why the system prompt tells the model to stop rather than estimate.

## What it does not do

The extractor cannot tell drafting language from technology. `present
invention` ranks near the top of most patent corpora, because on every
corpus-internal statistic it looks exactly like a real term: frequent,
cohesive, multi-word. Separating the two is a judgement, and it belongs to
whoever reads the table.
