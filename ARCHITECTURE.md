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
    analyst(("Analyst"))

    subgraph LOCAL["Analyst's machine: local app"]
        ingest["1 Ingest<br/>pandas: columns, dates, dedupe"]
        extract["2 Extract<br/>spaCy noun phrases, C-value,<br/>fragment filter, plural folding"]
        clean["3 Clean<br/>rules + trained classifier:<br/>technical term or noise"]
        classify["4 Classify<br/>topic / subtopic from CPC"]
        memory["5 In-memory tables<br/>pandas: patents, terms,<br/>occurrences"]
        analytics["6 Analytics<br/>evolution, co-occurrence,<br/>lift, trace"]
        gui["GUI: Streamlit + Plotly<br/>keywords, primary, secondary,<br/>evolution charts"]
        packer["7 Evidence-pack builder<br/>Send to Gemini"]
        inbox["8 Insight inbox<br/>checks every cited patent number"]
        report["9 Report"]

        ingest --> extract --> clean --> classify --> memory
        memory <--> analytics
        analytics --> gui
        gui --> packer
        inbox --> report
        analytics --> report
    end

    subgraph FOLDER["Project folder: plain files"]
        termsfile["technical_terms.csv<br/>technical rows, then rejected"]
        secfile["secondary_term.csv / .json"]
        packsdir["packs/<br/>every evidence pack sent"]
        insightsdir["insights/<br/>every answer pasted back"]
        reportfile["report"]
        cachefile["parse cache<br/>keyed by export file hash"]
    end

    subgraph GEMINI["Gemini web"]
        gemA["Gem A<br/>read the patents"]
        gemB["Gem B<br/>outside world:<br/>Search, Deep Research"]
        gemC["Gem C<br/>white space,<br/>next searches"]
    end

    exportfile --> ingest
    analyst --- gui
    extract <--> cachefile
    classify --> termsfile
    analytics --> secfile
    packer --> packsdir
    inbox --> insightsdir
    report --> reportfile
    packer -- "copy pack, open Gem" --> GEMINI
    GEMINI -- "analyst pastes answer back" --> inbox

    classDef ext fill:#f4f4f4,stroke:#888,color:#222
    classDef file fill:#eef6ea,stroke:#3f8f29,color:#1e4a13
    classDef gem fill:#fdf1e6,stroke:#d1600a,color:#5a2a05
    class exportfile,analyst ext
    class termsfile,secfile,packsdir,insightsdir,reportfile,cachefile file
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
    patents["patents table<br/>number, year, assignee, CPC,<br/>title, abstract, claims"]
    parse("spaCy parse")
    cache["parse cache file"]
    docs[/"parsed docs<br/>tokens, tags, noun chunks"/]
    generate("Candidate generation<br/>2-4 words, head noun,<br/>fragments dropped, plurals folded")
    candidates[/"candidates<br/>term, variants, patents,<br/>C-value, tf-idf"/]
    curate("Curation classifier")
    topics("Topic / subtopic<br/>from the patents' CPC codes")
    terms["terms table<br/>technical + rejected"]
    termsfile["technical_terms.csv<br/>technical rows, then rejected"]
    occurrences["occurrences table<br/>term, patent, field, position"]

    keywords[/"keyword table by topic"/]
    evolution("Evolution<br/>per year: count and share")
    cooc("Co-occurrence<br/>shared patents, share, lift")
    pair("Pair evolution<br/>both by year, observed vs expected")
    chart1[/"primary evolution chart"/]
    secfile["secondary_term.csv / .json"]
    chart2[/"pair evolution chart"/]

    pack("Evidence-pack builder")
    packsdir["packs/ folder"]
    gemini{{"Gemini web Gem"}}
    answer[/"answer with cited<br/>patent numbers"/]
    check("Insight inbox<br/>parse findings, check citations")
    insightsdir["insights/ folder"]
    report[/"Report"/]

    exportfile --> ingest --> patents
    patents --> parse --> docs
    parse <--> cache
    docs --> generate --> candidates --> curate
    curate --> topics
    patents -- "CPC codes" --> topics
    topics --> terms --> termsfile
    generate --> occurrences
    terms --> keywords

    keywords -- "analyst picks primary" --> evolution --> chart1
    keywords --> cooc --> secfile
    occurrences --> evolution
    occurrences --> cooc
    secfile -- "analyst picks secondary" --> pair --> chart2

    chart1 --> pack
    secfile --> pack
    chart2 --> pack
    patents -- "patent text" --> pack
    pack --> packsdir
    pack -- "clipboard" --> gemini --> answer --> check
    patents -- "known patent numbers" --> check
    check --> insightsdir

    chart1 --> report
    chart2 --> report
    secfile --> report
    insightsdir --> report

    classDef mem fill:#e8f1f8,stroke:#0b6fa4,color:#0b3a57
    classDef file fill:#eef6ea,stroke:#3f8f29,color:#1e4a13
    classDef gem fill:#fdf1e6,stroke:#d1600a,color:#5a2a05
    class patents,terms,occurrences mem
    class cache,termsfile,secfile,packsdir,insightsdir file
    class gemini gem
```

Blue boxes live in memory for the session; green boxes are files in the
project folder.

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
  is saved in `packs/`.
- A patent number Gemini cites that is not in the export is flagged.
- Rejected terms stay at the end of `technical_terms.csv` for audit and never
  become primaries or secondaries.
