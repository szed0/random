# Setup

> `v2/` is this same tool with the failure modes below fixed. Prefer it unless
> you specifically want this version.

## Install

```bash
python -m venv .venv
```

Activate it — Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Then install the dependencies and the language model:

```bash
pip install -r requirements.txt
```

```bash
python -m spacy download en_core_web_sm
```

## Run

```bash
streamlit run app.py
```

It serves on <http://localhost:8501> and opens a browser tab. The sentence
transformer (~90 MB) downloads on first use, so the first run is slower.

The very first time Streamlit runs on a machine it stops at an interactive
`Email:` prompt and waits. Press Enter to skip it. To avoid the prompt entirely:

```bash
streamlit run app.py --server.headless true
```

`python run_local.py` does the same thing and additionally picks up models from
a local `models/` folder — use it on a machine with no network. See **Offline**
below. It does not suppress the email prompt either.

## Check it works

Dependencies:

```bash
python -c "import spacy, sentence_transformers, streamlit, plotly, networkx, pandas; print('deps ok')"
```

Models — this is the step that fails on a fresh machine:

```bash
python -c "import spacy; spacy.load('en_core_web_sm'); print('spacy ok')"
```

```bash
python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2'); print('embeddings ok')"
```

End to end, without the UI — runs the whole pipeline on twelve built-in rows:

```bash
python -c "import pipeline as P, pandas as pd; rows=[('Cryogenic storage tank','A cryogenic storage tank with a vacuum jacket is disclosed.'),('Composite pressure vessel','A composite pressure vessel with a polymer liner is disclosed.'),('Fuel cell membrane','A fuel cell membrane comprising a catalyst layer is disclosed.'),('Hydrogen leak sensor','A hydrogen leak sensor with a catalytic sensing element is disclosed.')]*3; f=pd.DataFrame(rows,columns=['Title','Abstract']); r=P.run(None,P.Config(),frame=f); print(r.terms.to_string(index=False))"
```

You should get eight terms with C-values.

## Using it

Upload a CSV or Excel export with a **title** column and an **abstract**
column. A date column is optional and enables the Trends tab. The sidebar
auto-detects the columns; correct them there if it guesses wrong.

Four tabs: **Landscape** (the terms), **Relationships** (how they connect),
**Trends** (what is rising), **Explore** (the documents behind a term).

## Offline

No network at all: fetch both models on another machine and drop them beside
`app.py`.

```
models/en_core_web_sm/
models/all-MiniLM-L6-v2/
```

`python run_local.py` finds them there automatically. Any other location works
through the environment:

```bash
TECHSCOPE_SPACY_MODEL=/path/to/en_core_web_sm TECHSCOPE_EMBED_MODEL=/path/to/all-MiniLM-L6-v2 python run_local.py
```

A bigger spaCy model (`en_core_web_md` or `_lg`) parses more accurately and is
worth the disk if you have it. Nothing leaves the machine either way.

## Known problems in this version

All of these are fixed in `v2/`.

| Symptom | Cause |
|---|---|
| Trends chart looks plausible but the years are wrong | A date column that is not dates. Any four digits are read as a year, so a publication number like `US2024123456A1` becomes 2024. Nothing warns you. Check your date column by hand. |
| Empty report from a small file | With three documents or fewer, the minimum document frequency and the 60% boilerplate ceiling cross over and no term can satisfy both. Nothing can be returned. |
| Empty report from a focused file | If every term appears in more than 60% of documents, the boilerplate ceiling removes all of them. Raise **Drop terms above this share of documents** toward 1.00. |
| Empty report from non-English text | The bundled model is English-only. Nothing says so. |
| Term counts look inflated | Title and abstract are mapped to the same column, so the text is counted twice. Nothing rejects it. |
| `MemoryError` while parsing | Documents are batched 48 at a time regardless of length. Long text — a claims or description column — asks for a gigabyte at once. |
| `[E088] Text of length … exceeds maximum` | A single document over 2,000,000 characters. |
| A raw traceback instead of a message | Model load failures, memory errors and parser problems are not caught. |

## If it does not work

| What you see | What it means |
|---|---|
| `Can't find model 'en_core_web_sm'` | Run the `spacy download` command above. |
| Stops at an `Email:` prompt | Streamlit's first-run prompt. Press Enter. |
| Port 8501 already in use | Another copy is running, or use `--server.port 8600`. |
| The app dies at startup with no traceback | Duplicate OpenMP runtime on Windows. `run_local.py` sets `KMP_DUPLICATE_LIB_OK`; plain `streamlit run app.py` does not, so set it yourself. |
| No terms at all | Set **Minimum documents per term** to 1, raise the document-share ceiling, or add more rows. |

Streamlit prints the full trace in the terminal it was launched from.
