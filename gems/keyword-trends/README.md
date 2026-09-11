# Keyword trends

trt-pb reduced to the loop it exists for:

1. Upload a patent export. The Gem extracts keywords and shows a numbered menu.
2. Reply with a number - the primary keyword. Three graphs, then a second
   numbered menu of the keywords that share the most patents with it.
3. Reply with a number - the secondary keyword. Three graphs of the pair.

The analyst never types a keyword, so a keyword the corpus does not contain can
never enter the conversation.

## Files

| file | what it is |
| --- | --- |
| `trt_keywords.py` | the knowledge base: extraction, counting, and the six graphs |
| `SYSTEM_PROMPT.md` | the Gem's instructions |
| `PROMPTS.md` | the exact messages to send, verbatim |

## Setting the Gem up

Paste `SYSTEM_PROMPT.md` into the Gem's **Instructions** and leave **Knowledge
empty**. A Gem with anything in its Knowledge field is served without the
Python tool and can only refuse. `trt_keywords.py` goes in the chat message as
an ordinary attachment, alongside the export, on **every** turn - the sandbox
is wiped between messages.

## The six graphs

`primary(state, n)`

1. Occurrences by year - patents filed that year mentioning the keyword.
2. Cumulative occurrences - the running total, which only ever rises.
3. Share of the year's filings - the same count over that year's corpus size,
   so a year that simply holds more patents does not read as growth.

`secondary(state, n, m)`

1. The pair by year - patents carrying both, with single-keyword patents
   stacked above.
2. Cumulative - each keyword's running total and the pair's.
3. Together against chance - co-occurrence against what independence would
   predict for that year.

## Counting

A keyword occurs in a patent when it appears in the title or abstract as a
whole word. A bar counts patents, not mentions. Two keywords co-occur when they
appear in the same patent. The year is the application year; undated patents
are dropped from the charts and the count is printed.

Both menus are deterministic functions of the export - the ranking is stable
and ties break on the keyword itself - so row 6 is the same keyword in every
turn, and the session replays from two numbers after the sandbox is wiped.

## Input

CSV or Excel with a title or abstract column and a date column. Column names
are matched loosely; an ORBIT export works unchanged.
