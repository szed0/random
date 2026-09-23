# Technical terms from a patent export

Turn an Excel or CSV patent export into a CSV of technical terms, each
classified by topic and subtopic and traced to its publication numbers. Pick
one as the primary term to get the other technical terms in its patents, and
have Gemini read those patents and say what they are about.

Runs inside a Gemini Gem's code sandbox: the standard library plus pandas.

## The loop

| turn | you send | you get |
|---|---|---|
| 1 | the export | `technical_terms.csv` — every technical term, 2–4 words, with topic, subtopic and patents, then the rejected candidates at the end — and a numbered menu |
| 2 | a number | the secondary terms that share a patent with it, which patent each is in, and an insight on those patents |
| 3+ | a number, `read <publication number>`, `pair <n> <m>`, `trace <term>` | another primary, one patent in depth, the patents carrying two terms, the sentence behind a count |

```python
import trt_terms as tt

state = tt.extract("export.xlsx")      # candidates, rule-cleaned
tt.classify(state, """                 # Gemini's topic/subtopic call
Fuel cells > Catalysts: catalyst layer; supported electrocatalyst
""")                                   # -> technical_terms.csv

state = tt.load()                      # any later turn
tt.secondary(state, 134)               # -> secondary_<term>.csv
tt.read(state, 134)                    # the patents, printed to be read
tt.trace(state, "gas diffusion layer", within=134)
```

## `technical_terms.csv`

One file holds everything: the technical terms first, numbered and grouped
by topic, then every candidate Gemini left out, so what was thrown away can be
checked.

| column | meaning |
|---|---|
| `#` | the row number the user picks by; it never changes meaning. Empty on rejected rows |
| `term` | display form — singular, never an -ing head |
| `status` | `technical`; or `rejected` for a candidate Gemini left out (`not reviewed` if it was never shown) |
| `topic`, `subtopic` | Gemini's classification; empty on rejected rows |
| `n_words` | 2 to 4 |
| `n_patents` | patents containing it; 1 is allowed |
| `tfidf` | corpus term frequency × ln(N / document frequency) |
| `score` | C-value × cohesion — termhood, not popularity |
| `variants` | every surface form merged into this row |
| `patents` | every publication number it appears in |

Rejected rows are sorted after the technical ones, most patents first. Later
turns read only the `technical` rows, so a rejected term can never be picked
or appear as a secondary. `secondary_<term>.csv` adds `patents_with_both`,
`share_of_primary` and `lift`.

## How the noise is removed

Code proposes, Gemini classifies. Code removes what grammar can decide, and
decides it from the corpus rather than from word lists about any subject:

- **The head must be a noun this corpus uses as one** — seen after a
  determiner, heading a phrase, or with an attested plural. A word only seen
  after "is" is an adjective. Removes `electrode active`, `battery pack include`.
- **No -ing heads**, unless the -ings plural is attested (`coatings`). An
  -ing *first* word must be a modifier (`the CUTTING blade`), not a verb
  taking an object (`for MEASURING blood pressure`).
- **No verbs** — conjugated in this corpus, or only ever seen after "to" or a
  modal (`to PREVENT`, `can ADJUST`).
- **No fragments.** A phrase that never stands alone, and is almost always
  continued by the same word, is a piece of a longer term (`electrode active`
  → `electrode active material`). This is what makes one-patent terms usable.
- **Plurals folded, not stemmed.** `batteries` → `battery` because the
  corpus contains `battery`. Porter stemming merged `phosphoric acid` with
  `phosphorous acid` and `unit cell` with `unitized cell`, so it is gone.
  Hyphens and spaces are equivalent.

The only fixed lists are English function words and the vocabulary of patent
drafting (`comprising`, `plurality`, `embodiment`).

Gemini then drops what grammar cannot — positions (`opposite side`),
evaluations (`high capacity`), drafting slots (`manufacturing method`) — and
classifies the rest. It can only select: a name the code did not propose is
ignored.

## Measured

Two 150-patent HUPD corpora — batteries and fuel cells (`H01M`, 2004 and
2016) and surgical and diagnostic devices (`A61B`, 2016) — title and abstract
only, terms seen once allowed.

**Candidate noise, blind.** 50 random candidates per extractor per corpus,
pooled and shuffled, each labelled technical term or noise before the source
was revealed:

| | candidates (H01M / A61B) | noise | 95% CI |
|---|---|---|---|
| previous extractor | 3,878 / 4,069 | 61 / 100 | 51–70% |
| this one | 1,251 / 1,448 | 30 / 100 | 22–40% |

About six times less noise in absolute terms before Gemini sees the list. The
labelled run predates the last two rules (conjugated `-s` verbs, quantifiers
such as `multiple`), which only remove more.

**Secondary terms per primary**, H01M corpus, after a strict curation that
kept 705 of its 1,236 candidates:

| primary appears in | primaries | median secondaries | under 15 |
|---|---|---|---|
| 1 patent | 561 | 7 | 91% |
| 2 patents | 90 | 15 | 48% |
| 3–4 patents | 27 | 22–30 | 11% |
| 5+ patents | 27 | 51 | 0% |
| all | 705 | 8 | 79% |

So most primaries return fewer than 15. A broad one (`fuel cell`, 50 patents)
returns hundreds; the chat shows the top 15 by shared patents and lift, and
the CSV has every row.

## Running it as a Gem

`SYSTEM_PROMPT.md` goes in the Gem's Instructions, `trt_terms.py` in its
Knowledge. The export is a chat attachment. Code execution exists only
inside a Gem; an ordinary Gemini chat has no Python tool and will compute
numbers in its head.

## Input

Excel or CSV, columns matched loosely by name:

| needed | matched on |
|---|---|
| publication number | `publication number(s)`, `publication_number`, `patent number`, `id` |
| abstract or title | `abstract`, `summary`, `title` |
| read, not mined (optional) | `claims`, `description` / `English description` |
| optional | a date column (year), a CPC or `Technology domains` column |

Terms come from title and abstract only. Claims and descriptions are shown to
Gemini when it reads a patent but are not mined: claim language multiplies
drafting noise, and a long description makes every term co-occur with every
other.

## Limits

- The noise figures are one labeller's judgement on 200 items; the intervals
  are wide.
- Rules learned from the corpus need text to learn from. Under about 50
  patents, expect more borderline candidates for Gemini to remove.
- Sized for exports of a few hundred patents: 150 patents give about 1,300
  candidates to classify. Past 2,000 candidates the prompt drops one-patent
  terms (`min_patents=2`) rather than ask for a classification that will not
  fit in one reply.
- Patent families inflate co-occurrence: five filings of one invention
  count as five patents. The prompt tells Gemini to say so when it sees them.
- Topic and subtopic are Gemini's judgement and vary between runs. The CSV
  records the result, including the rejected rows at its end.
