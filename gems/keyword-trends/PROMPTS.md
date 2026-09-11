# The prompts, verbatim

Three messages. After the first one you only ever send a number.

**Attach both files to every message** - `trt_keywords.py` and your patent
export. The sandbox is wiped between turns, so a message without them cannot
draw anything, and the Gem will correctly refuse. Attach with the **+** button
next to the message box, then "Upload files", and select both at once.

---

## Message 1 - the corpus

Attach both files, then send:

> Run ingest on the attached export and show me the numbered keyword menu.
> Do not draw anything yet.

You get the corpus summary and 40 candidate keywords, numbered, with how many
patents mention each and the years they span.

---

## Message 2 - the primary keyword

Attach both files again, then send just the number:

> 6

That is the whole message. You get three graphs:

1. **Occurrences by year** - patents filed each year that mention the keyword.
2. **Cumulative occurrences** - the running total, which only ever rises.
3. **Share of the year's filings** - the same count against the size of that
   year's corpus, because a year that simply holds more patents lifts every
   keyword at once.

Then a second numbered menu: the keywords that share the most patents with the
one you picked. `both` is how many patents carry the pair; `lift` is that count
divided by what chance alone would give, so a merely common word does not come
top.

If a bare number is ignored, send this instead:

> Primary keyword 6. Run ingest, then primary(state, 6).

---

## Message 3 - the secondary keyword

Attach both files again, then send just the number from the second menu:

> 1

The reply opens by naming what it is doing - "Answering the secondary menu for
keyword 6 with 1, so `secondary(state, 6, 1)`". Read that line. If it says
`primary(state, 1)` instead, it has taken your number for a new keyword; say
`Secondary keyword 1. Run ingest, then secondary(state, 6, 1).` and it will
correct itself.

Three more graphs:

1. **The pair by year** - patents carrying both, with the ones carrying only
   one of them stacked above.
2. **Cumulative** - each keyword's running total and the pair's.
3. **Together against chance** - the co-occurrence bars against what
   independence would predict for that year. Above the dashed line the two
   travel together; on it they are unrelated words sharing a corpus.

The explicit form, if the bare number is ignored:

> Secondary keyword 1. Run ingest, then secondary(state, 6, 1).

---

## Carrying on

Another primary keyword - attach the files and send a new number from the first
menu. Another pairing for the same primary - send a new number from the second
menu.

To see the patents behind a bar:

> Run ingest, then documents_for(state, 6, 1, limit=10). Show the table, and
> tell me from the titles what these two actually have to do with each other.

If nothing on the menu is what you are after, widen or search it rather than
naming a keyword yourself:

> Run ingest, then keywords(state, top=80) and show the full menu.

> Run ingest, then keywords(state, contains="hydrogen") and show what comes back.

Then pick a number from the menu it just printed.

---

## When something looks wrong

**"Code execution is currently unavailable."** The Gem has a knowledge file
attached. Open the Gem, empty the Knowledge field, save, and start a new chat.
The module belongs in the message, not in the Gem.

**"I need the files attached."** You did not attach them to that message. The
sandbox does not remember them from the previous turn.

**It forgot which primary keyword you picked.** Send the explicit form:
`Secondary keyword 1. Run ingest, then secondary(state, 6, 1).`

**Numbers with no code block above them.** Reply:

> Recompute that with the module and show the code output.

and treat it as a sign to start a fresh chat.

**The keyword you want is missing from the menu.** It appears in fewer than two
patents, or punctuation is cutting it into pieces. Use
`keywords(state, contains="<part of the word>")` to see what form it took.
