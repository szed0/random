# Patent analytics — progress

State of the work as of 2026-09-24. One row per workstream: where it lives,
what it does, and what is actually demonstrated rather than intended.

`szed0/random` is the core. Everything else is either a reference
implementation to copy behaviour from, or an experiment branching off it.

## Workstreams

| # | Workstream | Where | What it does | Status | Demonstrated |
|---|---|---|---|---|---|
| 1 | **Term extraction + classification** | `szed0/random` — `trt_terms.py`, `SYSTEM_PROMPT.md` | Excel/CSV in → technical terms with topic/subtopic → pick a primary → secondary terms from the same patents → trace to patent numbers | **Active — the core** | Blind noise 61% → 30% vs the 09-21 baseline; 79% of primaries have <15 secondaries, 0% for terms in 5+ patents |
| 2 | **Architecture + diagrams** | `szed0/random` — `ARCHITECTURE.md`, `diagrams/*.mmd` | Local NLP app does all numbers; Gemini web reads patents and explains. No database, plain files in a project folder | **Active** | Three Mermaid files, each ≤9 blocks, pushed as standalone `.mmd` |
| 3 | **TRL scoring** | `szed0/random` — `TRL.md`, `diagrams/trl.mmd` | Lifecycle + commitment + adoption → TRL 1–6 per term, with patent numbers behind every component | **Proposed, uncalibrated** | Lexical approach measured and rejected: 0 true positives in 400 patents. Structural signal spreads measured on the same corpus |
| 4 | **Evidence text file** | `szed0/random` (design) | Numbers + patent text for one term, pasted into a Gem; Gemini's cited patent numbers checked back against local tables | **Designed, not built** | — |
| 5 | **Gemini sandbox constraints** | measured in-browser | What a Gem can actually run | **Done, 2026-09-18** | Python runs **only inside a Gem**, never plain chat; a Gem knowledge file **is** a real sandbox file; pandas/sklearn/nltk/spaCy present; **spaCy has no models**; no network |
| 6 | **Extractor ablations** | `random-backup.git` | Four code-side extractors × two model-side filters, scored against the local trt-pb pipeline | **Superseded by #1** | Shipped extractor 12/40 → c-value×cohesion **24/40**; filtering headroom +7 to +9 rows |
| 7 | **Six ablation Gems** | `random-backup.git` | Three-track and intent-preserving redesigns, one Gem each | **Superseded by #1** | All six executed their module in Gemini and reproduced local numbers |
| 8 | **trt-pb** (reference) | `Z:\pr\tentlytics\trt-pb` | Streamlit GUI: BERT-for-Patents MLM phrase grouping, relationship graph, trend analysis | **Reference — runs** | Rebuilt from 70 phone photos; 5 documented defects fixed; venv with 4 environment workarounds |
| 9 | **trt / sao** (reference) | `Z:\pr\tentlytics\trt`, `…\sao` | `trt`: cosine phrase grouping. `sao`: Subject-Action-Object with **all-MiniLM-L6-v2** | **Reference** | `sao` is the MiniLM precedent for #11 |
| 10 | **Codebase writeups** | `Z:\pr\tentlytics\writeups` | Six line-cited references (~9,900 lines) across sao / trt / trt-pb | **Done** | Every behavioural claim cited `file:line`; execution-verified claims marked |
| 11 | **spaCy variant** | local, to be sited | Purely NLP extraction, spaCy noun-phrase pipeline, Python GUI | **Planned** | Copy behaviour from #8 |
| 12 | **Laya as the noise filter** | `convaiinnovations/laya` on Hugging Face | ModernBERT-large, 421M, calibrated typed decisions; asked "is this phrase a technical term?" | **Tested — not adopted** | 62.5% agreement with the curation on a balanced 120-term sample where chance is 50%; see below |

## Open items

- **Row 11 has no location yet.** The spaCy variant needs a repo or a local
  path before it can start.
- **Row 1 has never been run in real Gemini.** Every measurement on
  `trt_terms.py` is local. The first thing worth doing is a Gem test with a
  150-patent slice.
- **Row 3 is uncalibrated.** The TRL formula orders terms defensibly; it does
  not yet place them on an absolute scale. That needs 30–50 analyst-labelled
  terms.
- **Row 5 is a moving target.** The sandbox contradicted its own documented
  behaviour once already between 09-09 and 09-18. Re-probe before any long
  session.

## Laya, measured

`convaiinnovations/laya` is a **calibrated decision model, not an embedding
model** — ModernBERT-large, 421M parameters, 512-token context, Apache-2.0.
You give it a state and typed questions; it returns typed answers with
probabilities in one forward pass and never generates text. That makes it a
candidate for the judgement this project keeps needing from Gemini: *is this
phrase a technical term or drafting language?*

It was tested against the 1,236 curated rows in
[`example/technical_terms.csv`](example/technical_terms.csv), on a balanced
sample where chance is 50%. The labels are what Gemini decided when it curated
that file, so this is agreement with the current pipeline, not accuracy
against a gold set.

| setup | agreement | n |
|---|---|---|
| bare phrase | **62.5%** | 120 |
| phrase + the patent sentence it came from | **57.1%** | 91 |

Adding context made it worse. Confidence is informative in the bare-phrase
setup and stops being so with context:

| confidence | agreement (bare) |
|---|---|
| 0.0 – 0.6 | 56% |
| 0.7 – 0.8 | 67% |
| 0.8 – 0.9 | 74% |
| 0.9 – 1.0 | 80% |

**Why it was not adopted.** The failures are the wrong kind. It rejects
`alicyclic quaternary ammonium cation`, `binderless zeolite molecular sieve`,
`glassy phase grain boundary`, `amorphous silicon primary particle` and
`off-gas outlet` — all unambiguously technical. Several of its disagreements
in the other direction are defensible (`hydrophobic surface`, `flow barrier`,
`high reversible capacity` were rejected by the curation and are arguably
terms), so 62.5% slightly understates it. But a filter that discards the most
specific chemistry in the corpus is not usable for this step at any threshold.

The reason is distribution, not prompting. Laya is trained on agent states —
emails, tickets, invoices, moderation. An isolated noun phrase from patent
prose is far from that, and rewording the instruction does not close a
distribution gap.

**Where it may still fit.** Document-level typed questions are its home
ground. Asked of 12 patent abstracts, "was this actually built and tested"
returned 0.48 – 0.84 and "is this one specific artefact" 0.53 – 0.89 — a real
but compressed spread, with no ground truth to validate the ordering. That is
the shape of a useful *component* for the TRL commitment term, not a TRL
classifier. See [`TRL.md`](TRL.md).

Practical notes: ~460 ms per phrase on CPU, first load 95 s and about 2 s
afterwards, and the checkpoint warns that `choice`-type temperatures are out
of range — use `noul` or `score` if you need the confidence to mean anything.

## Corpora

Test data is 150–400 patent slices of the Harvard USPTO Patent Dataset in
`Z:\patent\data` — `H01M` (electrochemical energy) for battery work, `A61B`
for a contrasting domain. Real abstracts, claims, CPC codes and filing dates;
**no assignee and no forward citations**, which is why the `A` component of
the TRL formula cannot be computed on it.
