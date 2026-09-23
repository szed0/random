# Architecture: local analysis with Gemini web for intelligence

The proposed next version. Keyword extraction, counting and every chart run
locally with full NLP libraries; Gemini web is used only for reading and
judgment, through Gems the analyst pastes evidence into. Numbers always come
from the local app, and every patent number Gemini cites is checked against
it.

## Architecture

```mermaid
flowchart TB
    exportfile["ORBIT export<br/>CSV / Excel"]
    analyst(("Analyst"))

    subgraph LOCAL["Analyst's machine: local app"]
        ingest["1 Ingest<br/>pandas: columns, dates, dedupe"]
        extract["2 Extract<br/>spaCy noun phrases, C-value,<br/>fragment filter, plural folding"]
        clean["3 Clean<br/>rules + trained classifier:<br/>technical term or noise"]
        classify["4 Classify<br/>topic / subtopic from CPC"]
        db[("5 DuckDB<br/>patents, terms, variants,<br/>occurrences, runs,<br/>packs, insights")]
        analytics["6 Analytics<br/>evolution, co-occurrence,<br/>lift, trace"]
        gui["GUI: Streamlit + Plotly<br/>keywords, primary, secondary,<br/>evolution charts"]
        packer["7 Evidence-pack builder<br/>Send to Gemini"]
        inbox["8 Insight inbox<br/>checks every cited patent number"]
        report["9 Report<br/>charts, checked insights,<br/>CSV / JSON"]

        ingest --> extract --> clean --> classify --> db
        db <--> analytics
        analytics --> gui
        gui --> packer
        packer -- "saved copy" --> db
        inbox --> db
        db --> report
    end

    subgraph GEMINI["Gemini web"]
        gemA["Gem A<br/>read the patents"]
        gemB["Gem B<br/>outside world:<br/>Search, Deep Research"]
        gemC["Gem C<br/>white space,<br/>next searches"]
    end

    exportfile --> ingest
    analyst --- gui
    packer -- "copy pack, open Gem" --> GEMINI
    GEMINI -- "analyst pastes answer back" --> inbox

    classDef ext fill:#f4f4f4,stroke:#888,color:#222
    classDef store fill:#e8f1f8,stroke:#0b6fa4,color:#0b3a57
    classDef gem fill:#fdf1e6,stroke:#d1600a,color:#5a2a05
    class exportfile,analyst ext
    class db store
    class gemA,gemB,gemC gem
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

    ingest("Ingest: map columns,<br/>parse dates, dedupe families")
    patents[("patents<br/>number, year, assignee, CPC,<br/>title, abstract, claims")]
    parse("spaCy parse<br/>cached by file hash")
    docs[/"parsed docs<br/>tokens, tags, noun chunks"/]
    generate("Candidate generation<br/>2-4 words, head noun,<br/>fragments dropped, plurals folded")
    candidates[/"candidates<br/>term, variants, patents,<br/>C-value, tf-idf"/]
    curate("Curation classifier")
    technical[("terms: technical")]
    rejected[("terms: rejected<br/>kept for audit")]
    topics("Topic / subtopic<br/>from the patents' CPC codes")
    occurrences[("occurrences<br/>term, patent, field, position")]

    keywords[/"keyword table by topic"/]
    evolution("Evolution query<br/>per year: count and share")
    cooc("Co-occurrence query<br/>shared patents, share, lift")
    pair("Pair query<br/>both by year, observed vs expected")
    chart1[/"primary evolution chart"/]
    secondary[/"secondary terms<br/>CSV / JSON"/]
    chart2[/"pair evolution chart"/]

    pack("Evidence-pack builder")
    packs[("packs<br/>what was sent, when, to which Gem")]
    gemini{{"Gemini web Gem"}}
    answer[/"answer with cited<br/>patent numbers"/]
    check("Insight inbox<br/>parse findings, check citations")
    insights[("insights<br/>linked to term, pack, Gem version")]
    report[/"Report"/]

    exportfile --> ingest --> patents
    patents --> parse --> docs --> generate --> candidates --> curate
    curate --> technical
    curate --> rejected
    technical --> topics
    patents -- "CPC codes" --> topics
    topics --> keywords
    generate --> occurrences

    keywords -- "analyst picks primary" --> evolution --> chart1
    keywords --> cooc --> secondary
    occurrences --> evolution
    occurrences --> cooc
    secondary -- "analyst picks secondary" --> pair --> chart2

    chart1 --> pack
    secondary --> pack
    chart2 --> pack
    patents -- "patent text" --> pack
    pack --> packs
    pack -- "clipboard" --> gemini --> answer --> check
    patents -- "known patent numbers" --> check
    check --> insights

    chart1 --> report
    chart2 --> report
    secondary --> report
    insights --> report

    classDef store fill:#e8f1f8,stroke:#0b6fa4,color:#0b3a57
    classDef gem fill:#fdf1e6,stroke:#d1600a,color:#5a2a05
    class patents,technical,rejected,occurrences,packs,insights store
    class gemini gem
```

### One session

```mermaid
sequenceDiagram
    actor A as Analyst
    participant G as GUI
    participant D as DuckDB
    participant W as Gemini web Gem

    A->>G: upload ORBIT export
    G->>D: ingest, extract, clean, classify
    D-->>G: keyword table by topic
    A->>G: pick primary term
    G->>D: evolution and co-occurrence queries
    D-->>G: evolution chart, secondary terms CSV / JSON
    A->>G: pick secondary term
    D-->>G: pair evolution chart
    A->>G: Send to Gemini
    G->>D: save evidence pack
    G->>W: open the Gem, pack on the clipboard
    A->>W: paste pack and send
    W-->>A: insight citing patent numbers
    A->>G: paste the answer into the insight inbox
    G->>D: check every cited number, store the insight
    G-->>A: report with charts and checked insights
```

## Rules the design keeps

- Numbers come only from the local app; Gemini interprets them.
- Data leaves the machine only when the analyst pastes it, and every pack sent
  is saved.
- A patent number Gemini cites that is not in the export is flagged.
- Rejected terms stay in the data for audit and never become primaries or
  secondaries.
