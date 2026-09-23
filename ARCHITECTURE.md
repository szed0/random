# Architecture: local analysis with Gemini web for intelligence

The proposed next version. Keyword extraction, counting and every chart run
locally with full NLP libraries; Gemini web is used only for reading and
judgment, through Gems the analyst pastes evidence into. Numbers always come
from the local app, and every patent number Gemini cites is checked against
it.

There is no database. While the app runs, the data lives in memory as pandas
tables; everything worth keeping is a plain file in the project folder.
Reopening a project re-reads the export and `technical_terms.csv`.

The Mermaid source of each diagram is also in its own file:
[`diagrams/architecture.mmd`](diagrams/architecture.mmd),
[`diagrams/dataflow.mmd`](diagrams/dataflow.mmd),
[`diagrams/session.mmd`](diagrams/session.mmd).

## Architecture

```mermaid
flowchart TB
    exportfile["ORBIT export<br/>CSV / Excel"]
    engine["NLP engine<br/>ingest, spaCy phrases, cleaning,<br/>technical vs noise, CPC topics"]
    analytics["Analytics<br/>evolution, secondary terms,<br/>lift, trace to patents"]
    gui["GUI<br/>Streamlit + Plotly:<br/>keywords, primary, secondary"]
    evidence["Evidence text file"]
    gem["Gemini / Gem"]
    folder["Project folder<br/>technical_terms.csv, secondary CSV / JSON,<br/>evidence text files, report"]

    exportfile --> engine --> analytics --> gui
    gui --> evidence
    evidence --> gem
    engine --> folder
    analytics --> folder
    evidence --> folder

    classDef ext fill:#f4f4f4,stroke:#888,color:#222
    classDef file fill:#eef6ea,stroke:#3f8f29,color:#1e4a13
    classDef gem fill:#fdf1e6,stroke:#d1600a,color:#5a2a05
    class exportfile ext
    class evidence,folder file
    class gem gem
```

| Local app | Gemini web |
|---|---|
| finds, cleans and classifies terms | reads patent text like an analyst |
| counts: evolution, co-occurrence, lift | explains why: problems, approaches, relationships |
| keeps every number traceable to a patent | brings in companies, products, news |
| checks what Gemini says against the data | suggests gaps and the next searches |

## Dataflow

```mermaid
flowchart TB
    exportfile[/"Export file"/]
    patents["Patents<br/>in memory: text, year, assignee, CPC"]
    candidates["Candidate terms<br/>spaCy noun phrases, 2-4 words,<br/>fragments dropped, plurals folded"]
    terms["technical_terms.csv<br/>technical rows with topic / subtopic,<br/>then rejected rows"]
    primary["Primary view<br/>evolution by year,<br/>secondary terms CSV / JSON"]
    pairview["Pair view<br/>both terms by year"]
    pack["Evidence pack<br/>numbers + patent text"]
    answer{{"Gemini answer<br/>with cited patent numbers"}}
    insights["Checked insights + report"]

    exportfile --> patents --> candidates --> terms
    terms -- "analyst picks primary" --> primary
    primary -- "analyst picks secondary" --> pairview
    primary --> pack
    pairview --> pack
    patents -- "patent text" --> pack
    pack -- "clipboard, Gem" --> answer
    answer --> insights
    patents -- "known patent numbers" --> insights

    classDef mem fill:#e8f1f8,stroke:#0b6fa4,color:#0b3a57
    classDef file fill:#eef6ea,stroke:#3f8f29,color:#1e4a13
    classDef gem fill:#fdf1e6,stroke:#d1600a,color:#5a2a05
    class patents,candidates mem
    class terms,primary,pack,insights file
    class answer gem
```

Blue lives in memory for the session; green is written to the project folder;
orange comes from Gemini web; the pair view is shown on screen only.

### One session

```mermaid
sequenceDiagram
    actor A as Analyst
    participant G as GUI
    participant E as Local engine
    participant F as Project folder
    participant W as Gemini web Gem

    A->>G: upload ORBIT export
    G->>E: ingest, extract, clean, classify
    E->>F: write technical_terms.csv
    E-->>G: keyword table by topic
    A->>G: pick primary term
    G->>E: evolution and co-occurrence
    E->>F: write secondary CSV / JSON
    E-->>G: evolution chart, secondary terms
    A->>G: pick secondary term
    E-->>G: pair evolution chart
    A->>G: Send to Gemini
    G->>F: save the evidence pack
    G->>W: open the Gem, pack on the clipboard
    A->>W: paste pack and send
    W-->>A: insight citing patent numbers
    A->>G: paste the answer into the insight inbox
    G->>E: check every cited number against the loaded patents
    G->>F: save the insight
    G-->>A: report with charts and checked insights
```

## Rules the design keeps

- Numbers come only from the local app; Gemini interprets them.
- No database: memory while running, plain files for anything kept.
- Data leaves the machine only when the analyst pastes it, and every pack sent
  is saved in the project folder.
- A patent number Gemini cites that is not in the export is flagged.
- Rejected terms stay at the end of `technical_terms.csv` for audit and never
  become primaries or secondaries.
