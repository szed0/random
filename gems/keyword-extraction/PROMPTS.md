# The prompts, verbatim

Three messages, then numbers.

**Build a Gem.** Code execution does not exist in an ordinary Gemini chat —
see `ENVIRONMENT.md`. A plain chat will compute numbers in its head and
present them as program output.

Put `SYSTEM_PROMPT.md` (per-abstract filter) or `SYSTEM_PROMPT_MENU.md` (menu
filter) in the Gem's **Instructions**. Put `trt_extract.py` and
`trt_keywords.py` in **Knowledge** — measured on 2026-09-18, a knowledge file
is a real file in the sandbox, so it survives the wipe and needs no
re-attaching. If that turns out not to hold for your Gem, attach both modules
in the chat on every turn instead; the prompts below work either way.

The export is always a chat attachment.

---

## Message 0 — check the sandbox

Worth one message at the start of any long session, because this has changed
once already:

> Capability check only, ignore the workflow. Run this and show the output:
> `import trt_extract as tx; tx.report_capabilities()`

You want to see `sklearn` present if you intend to use `method="lsa"`, and a
`Show code` block with a real output cell. If there is no code block, the Gem
is misconfigured or the tool has lapsed — start a fresh conversation.

---

## Message 1 — the corpus

Attach the export, then send:

> Run `candidates` on the attached export with `method="runs"`, then
> `prompt_for`. Read the blocks, select the technical terms, apply your
> selection and show me the numbered menu. Print the SELECTION block at the
> end. Do not draw anything yet.

For the menu filter Gem, say instead:

> Run `candidates` with `method="runs"`, show `menu(state, top=80)`, filter it
> down to the real technologies with `keep_only`, then show `menu(state,
> top=40)` and print the KEEP line. Do not draw anything yet.

**Copy the SELECTION or KEEP block.** It is the state. Every later message
pastes it back; without it the session cannot be rebuilt after the sandbox is
wiped.

---

## Message 2 — a primary keyword

Attach the export, paste the block, give a number:

> Primary keyword 6.
>
> SELECTION
> 1| active material; electrode assembly; rechargeable battery
> 2| fuel cell stack; membrane electrode assembly
> …

Three figures, then a numbered partner menu ranked by how many patents each
partner shares with your keyword.

Check the opening line names `primary(state, 6)`. If not, reply
`Row 6 of the keyword menu, so primary(state, 6).`

---

## Message 3 — a secondary keyword

> Secondary keyword 1, primary still 6.
>
> SELECTION
> …

Three figures for the pair, including co-occurrence against what chance alone
would predict.

---

## Fixing the menu

> Row 4 is drafting language, drop it. Rows 7 and 12 are the same thing — keep
> 7's wording.
>
> SELECTION
> …

It edits the block, re-prints the menu and the amended block, redraws. Keep
the amended block; the old one is stale.

---

## Running the ablation

Same corpus, same messages, one word changed. Four extractors × two filters:

> Run `candidates` with `method="lsa"` and do the same thing.

Then compare, for one domain and one term: the menu's top 20, where a term you
care about landed, and how many rows you had to delete. `runs` and `lsa` reach
the same recall@80 on the reference corpus, so after filtering they should
converge — if they do not on yours, that is worth knowing.

---

## When something goes wrong

| symptom | what to send |
|---|---|
| a count appears with no code block | `Recompute that with code.` Twice means the session has drifted — restart. |
| "code execution is unavailable" | You are probably in a plain chat, not the Gem. Check `ENVIRONMENT.md`. |
| it asks for a file you attached | Re-attach in a fresh conversation. It will not invent data. |
| the reply stops mid-report | `Continue, and paste the remaining printed lines.` |
| it invented a keyword | It cannot — `apply_selection` and `keep_only` discard anything that was not a candidate. If a term you expected is missing, it was never proposed; widen with `menu(state, top=120)`. |
| `tfidf` and `lsa` behave identically to `runs` | scikit-learn is absent and they fell back. `report_capabilities()` says so on the `METHODS` line. |

The first run on a large export takes a minute or two. That is the sandbox
executing, not a hang.
