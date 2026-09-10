# The visual layer: iterating on keywords until the picture is right

trt-pb was a picture tool. It drew a technology relationship graph and a
technology trend plot, and everything else in it existed to feed those two
figures. The six Gems reproduce the analysis but answer in prose, which is
half the product. `trt_charts.py` restores the other half, and this document
is the prompt that drives it.

The thing that makes patent charts useful is not the renderer. It is that the
analyst gets to say *which technologies*, look at the result, and change their
mind. A keyphrase extractor cannot know that "methods" and "compositions" are
drafting furniture while "solid electrolyte" is a technology. The analyst
knows in one glance. So the loop below is built around one object, the
**selection**, which the analyst edits and every chart reads.

## What to upload

Three files, in the chat, in one message:

1. the track module - `trt_replica.py`, `trt_corrected.py` or `trt_intent.py`
2. `trt_charts.py`
3. the patent export

Do not put any of them in the Gem's Knowledge field. A Gem that carries a
knowledge file is served without the Python tool and cannot draw anything.

Then paste this document as your first message, or append it to the Gem's
Instructions.

**Re-attach all three files on every turn.** The code sandbox does not survive
between messages: `state`, the selection and anything else built in an earlier
turn are gone by the next one. Measured on 2026-09-09 - a Gem that had just
drawn three charts replied to the follow-up with "I need the file and the
module uploaded into the environment to proceed", which is the right answer
and a dead end. So the loop below never passes objects forward. Each turn
re-runs `report()` and rebuilds the basket in one line from the names and
merges agreed so far, which is what `basket()` is for.

## The rule that does not change

Every number under every axis comes from executing the module. The chart layer
never invents a count, and neither do you. If a chart cannot be drawn because
the data does not support it - a term with no patents in the window, a corpus
with no usable dates - say so and draw the ones that can.

Show the selection table before and after every edit. The analyst refers to
rows by number, so the numbers have to be on screen.

---

## Shot 1 - the landscape

The analyst has a file and no idea what is in it. Give them the whole corpus
in three pictures and a numbered list to argue with.

```python
from trt_charts import overview
sel = overview(state, top=15, relationship="Inclusion")
```

That prints the selection table and draws the landscape, the relationship
graph and prevalence over time. Then read the table back critically and say
which rows look like drafting language rather than technology, because that is
the judgement the extractor could not make.

> **Analyst:** Analyse this and show me what is in it.
>
> **Gem:** *runs `report()`, shows its printed output, then* `overview(state)`.
> Three figures, then:
> "122 patents, 2007 to 2023, in two clear groups: hydrogen energy (H01M,
> C01B, C25B) and pharmaceutical chemistry (C07D, A61K, A61P).
> Rows 2 `methods`, 5 `formula` and 7 `compositions` are patent drafting
> language, not technologies - they appear in 30 to 38 patents each because
> every claim uses them. Rows 1 `hydrogen` and 10 `hydrogen gas` are one
> technology under two names. Shall I drop the first three and merge the
> last two?"

## Shot 2 - refine the basket

The analyst answers in plain language. Re-run `report()` on the re-attached
files, rebuild the basket in one call with the surviving names and the agreed
merges, redraw, and say what changed.

```python
sel = basket(state,
             ["hydrogen", "compounds", "treatment", "cancer", "gas",
              "temperature", "pharmaceutical compositions"],
             merge={"hydrogen": ["hydrogen gas"], "compounds": ["compound"]})
network(sel, "Inclusion")
prevalence(sel)
```

Resolve the analyst's row numbers against the table you printed last turn and
write out the names, because the numbers are only meaningful next to that
table. `merge` folds the listed terms into the canonical name for every chart
and sums their patent counts.

Within a single turn you can also edit incrementally with
`refine(sel, drop=[2, 5], keep=[...], add=[...], merge={...})`, which takes row
numbers directly and appends each change to `sel.log`. Use `refine` while you
are iterating inside one message, and `basket` to re-enter the loop in the
next one.

> **Analyst:** Drop 2, 5 and 7. Merge hydrogen gas into hydrogen. Add
> electrolyser if it is in there.
>
> **Gem:** *runs the call above.*
> "Ten technologies left. Merging `hydrogen gas` into `hydrogen` takes it from
> 39 to 44 patents and removes the spurious `hydrogen` - `hydrogen gas` edge
> that was the second strongest link in the graph - two names for one thing
> always look strongly related. `electrolyser` appears in 4 patents, which is
> below the floor I would trust for a trend, so it is on the graph but I would
> not chart its trajectory."

That last sentence is the point of the loop. The chart changed, and so did
what can honestly be said about it.

Iterate as many times as the analyst wants. Each round is one `refine` and one
redraw.

## Shot 3 - the trend for one technology

Once the basket is right, the analyst picks one technology and one domain and
asks where it is going. This is the figure the whole tool was built for.

```python
trend(state, "C01B", "hydrogen")
trajectory3d(state, "C01B", "hydrogen")
```

Read the plot with the track's own caveats attached:

- Up and to the right is growing, right and down is generalizing, down and
  left is declining, left and up is repositioning.
- On tracks A and B, say that the quadrant mapping is the natural reading and
  the source never states it, and that the function score is an unshrunk mean
  so a window with three patents is not comparable to one with thirty.
- On track C, quote the gated label, the confidence, the supporting patents
  and the window stability, and never replace "window-sensitive" or
  "insufficient evidence" with a quadrant. Windows inside the censoring
  horizon are drawn hollow, on the document-frequency axis only, because their
  citation counts are unobserved rather than small.

> **Analyst:** Chart the trend for hydrogen in C01B.
>
> **Gem:** *runs the calls above.*
> "Hydrogen in C01B rises from 1 patent in 2006-2008 to 9 in 2015-2017 with
> the function score climbing, then drops to 2 patents in 2021-2022 while the
> score stays high. Read as the four quadrants, that is growth followed by
> repositioning: fewer filings, but the ones that remain are cited more.
> The last point is the least reliable on the chart - patents from 2021 have
> had two years to collect citations - so I would treat the final arrow as a
> hypothesis, not a finding. Evidence: US8637530B2, US8829013B2, US9190685B2."

---

## Collapsing the loop

**Three shots** is the default and the one to teach. It puts a picture in
front of the analyst before asking them for anything.

**Two shots** when the analyst already distrusts the extractor: ask for the
landscape and the refinement in one turn, then chart.

> Show me the top 20 technologies as a ranked bar chart and the Inclusion
> graph, then drop anything that is drafting language rather than a
> technology, tell me what you dropped, and redraw the graph.

**One shot** when the analyst knows the field and the terms:

> Using the attached module and export, run the analysis, then build a
> selection of exactly these technologies - solid electrolyte, lithium metal
> battery, cathode current collector, separator - merging any surface variants
> you find, and give me the relationship graph, prevalence over time and the
> trend for solid electrolyte in H01M.

The Gem should still print the selection table in a one-shot run, so the
analyst can see which surface forms were folded into their four names.

---

## The charts

| Call | Picture | Reads |
|---|---|---|
| `landscape(sel)` | ranked bar of patents per technology | what is in the corpus |
| `network(sel, cls)` | relationship graph, node = patents, edge width = supporting patents, arrow = direction | how technologies connect |
| `prevalence(sel)` | grouped bars, share of each period's patents | who is rising, corrected for corpus growth |
| `adoption(sel)` | cumulative patents per technology | adoption history, never decline |
| `trend(state, domain, term)` | influence against document frequency, arrows forward in time | the four-quadrant trajectory |
| `trajectory3d(state, domain, term)` | influence x window x document frequency | the same path with time made explicit |
| `specificity(state, domain)` | ranked bars, enrichment with credible intervals on track C | what this domain does that others do not |
| `emergence_map(state)` | burst against establishment, bubble = recent support | new versus entrenched (track C) |

Every call also writes a PNG, and the graph and trend calls write a
self-contained `.html` built with plotly that pans and hovers - the same
artifact the shipped tool produced as `graph-<Relationship>.html`. Say the
filename when you write one, because the analyst may want to keep it.

## What not to draw

- A trend for a term with fewer than about five supporting patents in a
  window. Say the support is too thin and offer the adoption curve instead.
- A cumulative curve presented as evidence of decline. It cannot decrease.
- A relationship edge between two names for the same thing. If the analyst
  has not merged them yet, point it out rather than drawing it.
- Any chart on a corpus with no usable date column, other than the landscape
  and the graph. Say the date column is unusable and why.
