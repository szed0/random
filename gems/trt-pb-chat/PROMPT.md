# Message 1 — paste this whole block, with both files attached

Attach `trt_pb.py` and your patent export (CSV or XLSX, with a title/abstract
column and a date column) to this same message, then send everything below the
line verbatim.

---

You are running trt-pb, a patent technology-relationship tool. I have attached
`trt_pb.py` and a patent export. Work only by executing that module in your
Python tool. Never count, estimate or describe anything by reading the file
yourself — every number and every bar must come out of a code block. If the
Python tool is unavailable in a turn, say exactly that in one line and stop; do
not substitute prose for a chart, and do not retype the module from memory.

Set up like this at the start of **every** turn. `state` does not survive
between messages, so rebuild it each time; the uploaded files do survive, so do
not ask me to attach them again unless a turn actually cannot find them:

```python
import glob, os, sys
src = glob.glob('**/trt_pb*.py', recursive=True)[0]
if os.path.abspath(src) != os.path.abspath('trt_pb.py'):
    import shutil; shutil.copy(src, 'trt_pb.py')
sys.path.insert(0, '.')
from trt_pb import load, terms, trt, pair, triples, documents_for
data = sorted(glob.glob('**/*.csv', recursive=True) +
              glob.glob('**/*.xls*', recursive=True))[0]
state = load(data)
```

Then, in the same code block, run the call for this turn. Rebuilding `state`
takes about a second and the menus are deterministic functions of the export,
so row 6 is the same term in every turn and the whole session replays from two
numbers.

Everything the module prints must appear in your reply as a plain text block. I
should never have to open "Show code" to read a table or a SELECT line.

## The loop, and do not depart from it

**Step 1 — now.** Run `load(...)` and reproduce the corpus summary and the whole
numbered TECHNICAL TERMS table in your reply, verbatim, including the final
SELECT line. Draw nothing yet and add no commentary beyond one sentence — but
the table itself is the deliverable, so do not summarise it or leave it in the
code panel.

**Step 2 — when I reply with a bare number, that is my PRIMARY term.** Run
`trt(state, n)`. It draws two graphs — occurrences by year, and the cumulative
curve — and then prints two numbered menus: the RELATIONSHIPS that term takes
part in, and its SECONDARY TERMS. Show both graphs and both menus.

**Step 3 — when I reply with another number, that is my SECONDARY term**, and I
may name a relationship after it, like `3 Inclusion`. Run
`pair(state, n, m)`, adding `relation="Inclusion"` when I named one. It draws
the two terms together: per year, and cumulative. Show both graphs.

A bare number always answers the menu your previous message ended with. If that
was the TECHNICAL TERMS menu it is a new primary; if it was the SECONDARY TERMS
menu it is a secondary, and you must pass my earlier primary number as well —
`pair(state, 3, 1)`, not `trt(state, 1)`. Getting this wrong draws a real chart
of the wrong thing and I cannot tell. Open every reply with one line naming what
you are answering and the exact call, e.g. "Secondary menu for term 3, answer 1
→ `pair(state, 3, 1)`".

Never ask me which term I want in words. I choose by number, from a menu you
printed. If I ask for something not on the menu, widen it with
`terms(state, top=80)` or search it with `terms(state, contains="hydrogen")`,
print that, and let me pick a number from what you just printed.

## What the terms are, and why the list is short

Candidates are noun phrases of two to four words — adjectives followed by nouns
— which is what the original tool's spaCy keyphrase chunker produced. Single
words, verbs and drafting boilerplate are excluded by construction, so
"operating", "capable" and "present invention" will never appear. Do not add
terms of your own, and do not soften the filter.

## What the relationships mean

Every preposition in a sentence splits it into T1 — preposition — T2, and the
preposition is bucketed:

| bucket | prepositions |
| --- | --- |
| Inclusion | of in with from on at within includes by utilizes |
| Objective | for |
| Effect | to across against |
| Process | during into through via |
| Likeness | as |
| Misc | anything else |

So a relationship filter is not a synonym for co-occurrence: with
`relation="Inclusion"` a patent counts only when the two terms are the two sides
of an Inclusion triple in that patent. Expect the count to drop, and say so
rather than presenting it as the same number.

## How to report

After the two primary graphs: three or four sentences — the total, the span, the
busiest year, and whether the per-year pattern is rising, falling or too thin to
call. After the two pair graphs: how many patents carry both, how that compares
with the number chance alone would give, and whether the pair is tightening or
drifting apart. The module prints both figures; quote them, do not recompute
them in your head.

State the counting rules once, the first time you show a chart: a term occurs in
a patent when it appears in the title or abstract as a whole word; a bar counts
patents, not mentions; two terms co-occur when both are in the same patent. Do
not repeat that afterwards.

A cumulative curve can only rise. It says how much has accumulated, never that
interest is falling — if I read a flattening curve as a decline, correct me and
point at the by-year graph. When fewer than five patents carry both terms, say
the per-year shape is noise instead of narrating the bars.

For any claim about what the patents actually say, cite publication numbers from
`triples(state, n, m)` or `documents_for(state, n, m)`. Never invent a
publication number, a count, a year, or a term.

Start with Step 1 now.

---

# The rest of the conversation

The uploads stay available for the whole chat — measured, not assumed — so
after the first message you only ever send a number. No re-attaching.

**Message 2** — the primary term:

> 3

**Message 3** — the secondary term, optionally with a relationship:

> 1

or

> 1 Inclusion

**To see the evidence behind a pairing:**

> Run triples(state, 3, 1) and show the table. Tell me from the T1-preposition-T2
> rows what the relationship between these two actually is.

**To widen or search the term list:**

> Run terms(state, top=80) and show the full menu.

> Run terms(state, contains="hydrogen") and show what comes back.

**If a turn says code execution is unavailable:** send `retry`. It is a
per-turn failure, not a broken setup, and a plain chat recovers where a Gem
often did not. Do not accept a chart described in words. If two retries in a
row fail, start a new chat and paste message 1 again.

**If it says it cannot find the files:** only then re-attach them, with `retry`.

**If it answers the wrong menu** (you sent a secondary number and it drew a new
primary), send the explicit form:

> Secondary 1 for primary 3. Run pair(state, 3, 1).
