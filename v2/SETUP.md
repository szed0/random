# Setup

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
transformer (~90 MB) downloads on first use, so the first run is slower than
the rest.

`python run_local.py` does the same thing and additionally picks up models from
a local `models/` folder — use it on a machine with no network. See **Offline**
below.

To use a different port:

```bash
streamlit run app.py --server.port 8600
```

## Check it works

Dependencies:

```bash
python -c "import spacy, sentence_transformers, streamlit, plotly, networkx, pandas; print('deps ok')"
```

Models — this is the step that fails on a fresh machine, and it prints the
exact fix if the model is missing:

```bash
python -c "import spacy; spacy.load('en_core_web_sm'); print('spacy ok')"
```

```bash
python -c "from sentence_transformers import SentenceTransformer as S; S('all-MiniLM-L6-v2'); print('embeddings ok')"
```

End to end, without the UI — this runs the whole pipeline on twelve built-in
rows and prints the terms it found:

```bash
python -c "import pipeline as P, pandas as pd; rows=[('Cryogenic storage tank','A cryogenic storage tank with a vacuum jacket is disclosed.'),('Composite pressure vessel','A composite pressure vessel with a polymer liner is disclosed.'),('Fuel cell membrane','A fuel cell membrane comprising a catalyst layer is disclosed.'),('Hydrogen leak sensor','A hydrogen leak sensor with a catalytic sensing element is disclosed.')]*3; f=pd.DataFrame(rows,columns=['Title','Abstract']); r=P.run(None,P.Config(),frame=f); print(r.terms.to_string(index=False))"
```

You should get eight terms with C-values. An empty table means the models
loaded but nothing survived the filters; `r.notes` says why.

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

## If it does not work

| What you see | What it means |
|---|---|
| `Can't find model 'en_core_web_sm'` | Run the `spacy download` command above. |
| Stops at an `Email:` prompt | Streamlit's first-run prompt. Press Enter. `run_local.py` handles this for you. |
| Port 8501 already in use | Another copy is running, or use `--server.port 8600`. |
| The app dies at startup with no traceback | Duplicate OpenMP runtime on Windows. Both entry points set `KMP_DUPLICATE_LIB_OK`; if you launch some other way, set it yourself. |
| "no year could be read from …" | The date column is not dates — often a publication-number column. Pick another, or set it to none. |
| "every term appears in more than 60% …" | Your corpus is narrow, so the boilerplate filter was skipped. The terms are still valid. |
| "probably not English" | The bundled model is English-only. |
| "Ran out of memory while parsing" | The abstract column is pointing at full patent text. Remap it. |
| No terms at all | Set **Minimum documents per term** to 1, or add more rows. |

Streamlit prints the full trace in the terminal it was launched from.
