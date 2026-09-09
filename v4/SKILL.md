---
name: patent-landscape
description: Turn a patent export into a technology landscape - a relationship graph, prevalence over time, domain specificity, emergence and a four-quadrant trend plot - with the analyst choosing and refining the keywords between rounds. Use when someone has a patent CSV or Excel export and wants to know what technologies are in it, how they relate, which are rising, and which patents back each claim. Also use for ORBIT exports, trt-pb, technology trend analysis, TRT graphs, keyphrase landscapes, or patent intelligence charts.
---

# Patent landscape and technology trend analysis

This is the trt-pb toolchain rebuilt to run inside a model's code sandbox.
The original was a Streamlit app that drew a technology relationship graph and
a trend plot; everything else in it existed to feed those two figures. Keep
that priority. The deliverable is a picture an analyst can act on, with the
patents behind it named.

## Which module to use

Six ablations exist, from two redesign documents. They are not
interchangeable, and picking the wrong one wastes the session.

| Want | Use | Why |
|---|---|---|
| Actual patent intelligence, charts an analyst will use | `intent-preserving/c-intent-optimal/trt_intent.py` | Richest evidence layer: enrichment with credible intervals, emergence as its own axis, structured functional concepts that read well as chart labels, bootstrap intervals, provenance objects, temporal backtest |
| The most defensible trend plot specifically | `three-track/c-intent-optimal/trt_intent.py` | The only one that shrinks the influence estimate toward the corpus mean and refuses to plot it inside an empirical citation censoring horizon - the two things that make the signature chart honest |
| What the shipped tool would have produced, defects and all | either `a-replica` | A control. Its extractor output is visibly poor on real corpora; that is the point, not a bug |
| The same methods with the bugs fixed | either `b-corrected` | A midpoint. Fine charts, but no uncertainty anywhere and the trend plot is still a funnel |

**Default to the intent-preserving Track C.** Its one real gap against the
three-track Track C is the censoring horizon: forward citations on recent
patents are unobserved rather than small, and that is a bias, so neither a
median nor a bootstrap interval removes it. If the corpus reaches the present
day, say that the last window's influence is not trustworthy, or borrow the
horizon from the other track.

Do not mix two modules in one session. Both define `report()`.

## Files

- `trt_<track>.py` - the analysis. `report(path)` returns `state`.
- `trt_charts.py` - every figure. Works with all six tracks.
- `VISUALS.md` - the three-shot prompt that drives the keyword loop.

## The workflow

1. Load the export, show the columns and one record, and name the seven column
   mappings before computing anything. Stop if there is no abstract column, if
   the date column holds identifiers rather than dates, or if there is no
   technology-domain column.
2. `state = report(path)`. Show its printed output verbatim.
3. `sel = overview(state)` - the landscape, the relationship graph and
   prevalence, plus a numbered selection table.
4. Say which rows are patent drafting language and which are two names for one
   thing. This is the judgement the extractor cannot make and the analyst can
   confirm in a glance.
5. The analyst edits. Rebuild with `basket(state, [...], merge={...})` and
   redraw. Repeat.
6. Once the basket is right, chart one technology's trajectory with
   `trend(state, domain, term)` and read the quadrant with the track's
   caveats attached.

`VISUALS.md` has the exact prompts and worked exchanges for each step.

## Rules that do not bend

- Every number under every axis comes from executing the module. Never
  estimate a count, and never redraw a figure from remembered numbers.
- Print the selection table before and after every edit. The analyst refers to
  rows by number, so the numbers must be on screen.
- Cite publication numbers for any claim about a technology or a relationship.
- A cumulative curve is adoption history and can never decrease, so it is
  never evidence of decline.
- Do not chart a trend for a term with fewer than about five supporting
  patents in a window. Say the support is too thin and offer the adoption
  curve instead.
- If code execution is unavailable, say so and stop. Do not simulate output.

## Running it as a Gemini Gem

Put the track's `SYSTEM_PROMPT.md` in the Gem's Instructions and leave the
Knowledge field **empty**. A Gem carrying a knowledge file is served without
the Python tool: it can read the module but not run it, and will correctly but
uselessly refuse every question. Upload the module, `trt_charts.py` and the
export as chat attachments instead, and re-attach them on every turn, because
the sandbox does not persist between messages.

The sandbox has pandas, numpy, matplotlib, seaborn and plotly, but not
kaleido, networkx or pyvis. Charts therefore render as matplotlib images
inline, and the interactive graph is written as a self-contained plotly HTML
file.

## What the modules substitute

The original used spaCy, KeyBERT, MPNet and BERT-for-Patents, none of which
exist in a sandbox. Each module documents its stand-ins in its docstring: a
fixed preposition list for the tagger, a corpus-internal inflection test for
verb lemmas, stopword segmentation with tf-idf for KeyBERT, and
character-trigram or PPMI context cosine for the embeddings. Say this once
when a number depends on it. The stand-ins are substitutions, not fixes, and
the local tool remains the reference for the numbers.
