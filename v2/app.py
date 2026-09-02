"""TechScope — a technology landscape from patent titles and abstracts.

    streamlit run app.py

This module renders. All the analysis lives in `pipeline`, which imports no
Streamlit and can be run from a script, a notebook or a scheduler.

First run downloads two models (~120 MB total):

    python -m spacy download en_core_web_sm

The sentence-transformer is fetched on first use. On a machine with no network,
download both elsewhere, copy them across, and point at them by path:

    TECHSCOPE_SPACY_MODEL=/path/to/en_core_web_sm
    TECHSCOPE_EMBED_MODEL=/path/to/all-MiniLM-L6-v2

Nothing else leaves the machine.
"""

from __future__ import annotations

import os

# torch and spaCy can each load their own OpenMP runtime; on Windows the
# duplicate aborts the process at import time with no Python traceback. Set
# before either is imported, so `streamlit run app.py` is as safe as run_local.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
from dataclasses import fields

import numpy as np
import pandas as pd
import streamlit as st

import pipeline as P

st.set_page_config(page_title="TechScope", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# model loading — cached across reruns
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading the language model…")
def load_spacy(name: str):
    import spacy
    nlp = spacy.load(name, disable=["ner"])
    nlp.max_length = max(nlp.max_length, 2_000_000)
    return nlp


@st.cache_resource(show_spinner="Loading the embedding model…")
def load_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@st.cache_data(show_spinner=False, max_entries=4)
def analyse(_frame: pd.DataFrame, _cfg: P.Config, cache_key: str) -> P.Result:
    """Run the pipeline, memoised on `cache_key`.

    Streamlit hashes every argument whose name does not start with an
    underscore, so both the frame and the config are hidden from it and
    `cache_key` - a digest computed once by `digest()` - is the whole cache
    identity. Leaving the frame hashable made Streamlit walk the entire table on
    every rerun, which is work already done by the digest.
    """
    bar = st.progress(0.0, text="Starting…")
    order = ("loading", "terms", "grouping", "relations")

    def progress(stage: str, done: int, total: int) -> None:
        base = order.index(stage) / len(order) if stage in order else 0.0
        bar.progress(min(base + (done / max(total, 1)) / len(order), 1.0),
                     text=f"{stage.capitalize()} — {done}/{total}")
    try:
        return P.run(None, _cfg, frame=_frame,
                     nlp=load_spacy(_cfg.spacy_model),
                     embedder=load_embedder(_cfg.embed_model),
                     progress=progress)
    finally:
        bar.empty()


def digest(frame: pd.DataFrame, cfg: P.Config) -> str:
    """Content hash of the inputs, so a rerun with identical settings is free but
    changing any knob invalidates the cached result."""
    h = hashlib.sha256()
    # dataclasses.fields() is already in declaration order - no sort needed, and
    # sorting would compare ints against strings.
    h.update(repr([(f.name, getattr(cfg, f.name)) for f in fields(cfg)]).encode())
    h.update(str(frame.shape).encode())
    for col in frame.columns[:6]:
        h.update(frame[col].astype(str).str.cat(sep="|")[:200_000].encode(
            "utf-8", "replace"))
    return h.hexdigest()[:24]


# --------------------------------------------------------------------------- #
# column guessing
# --------------------------------------------------------------------------- #

def guess(columns: list[str], *needles: str, default: str = "") -> str:
    lowered = {c.lower(): c for c in columns}
    for n in needles:
        for low, original in lowered.items():
            if n in low:
                return original
    return default


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #

def view_landscape(r: P.Result) -> None:
    a, b, c, d = st.columns(4)
    a.metric("Documents", f"{len(r.corpus):,}")
    b.metric("Technology terms", f"{len(r.terms):,}")
    c.metric("Relationships", f"{len(r.edges):,}")
    grouped = sum(1 for v in r.variants.values() if v)
    d.metric("Terms with variants", f"{grouped:,}")

    st.caption(
        "Ranked by C-value — a termhood score that prefers the specific phrase "
        "over the generic one it contains, so *cryogenic storage tank* outranks "
        "*storage tank* when the longer form is what the corpus actually uses.")

    frame = r.terms.rename(columns={
        "term": "Technology term", "c_value": "C-value",
        "documents": "Documents", "variants": "Also written as"})
    st.dataframe(frame, use_container_width=True, hide_index=True,
                 column_config={
                     "C-value": st.column_config.NumberColumn(format="%.1f"),
                     "Documents": st.column_config.ProgressColumn(
                         format="%d", min_value=0,
                         max_value=int(frame["Documents"].max() or 1)),
                 })
    st.download_button("Download terms as CSV",
                       r.terms.to_csv(index=False).encode(),
                       "techscope-terms.csv", "text/csv")


def view_graph(r: P.Result) -> None:
    if r.edges.empty:
        st.info("No pair of terms co-occurs often enough to form a relationship. "
                "Lower **Minimum shared documents** in the sidebar, or use a "
                "larger corpus.")
        return

    import networkx as nx
    import plotly.graph_objects as go

    top = st.slider("Relationships to draw", 10, min(300, len(r.edges)),
                    min(80, len(r.edges)), 10)
    edges = r.edges.head(top)

    g = nx.Graph()
    for _, e in edges.iterrows():
        g.add_edge(e["source"], e["target"], weight=float(e["ppmi"]))
    pos = nx.spring_layout(g, seed=42, k=1.1 / np.sqrt(max(g.number_of_nodes(), 1)))

    ex, ey = [], []
    for u, v in g.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]

    degree = dict(g.degree())
    docs = {t: int(d) for t, d in zip(r.terms["term"], r.terms["documents"])}
    nodes = list(g.nodes())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                             line=dict(width=0.8, color="rgba(130,150,170,0.45)")))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        textfont=dict(size=10),
        hovertext=[f"{n}<br>{docs.get(n, 0)} documents<br>{degree[n]} links"
                   for n in nodes],
        hoverinfo="text",
        marker=dict(size=[8 + 2.2 * degree[n] for n in nodes],
                    color=[docs.get(n, 0) for n in nodes],
                    colorscale="Teal", showscale=True,
                    colorbar=dict(title="Docs", thickness=12))))
    fig.update_layout(height=650, showlegend=False,
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Edges are weighted by **PPMI**, not raw co-occurrence. Two common terms "
        "appear together often without being related; PPMI divides out each "
        "term's own rate, so what is left is association above chance.")
    st.dataframe(edges.rename(columns={
        "source": "Term A", "target": "Term B",
        "documents": "Shared documents", "ppmi": "PPMI"}),
        use_container_width=True, hide_index=True)


def view_trends(r: P.Result) -> None:
    if r.trends.empty:
        st.info("No date column was mapped, so trends are unavailable. "
                "Pick one in the sidebar if your file has publication or "
                "application dates.")
        return

    import plotly.express as px

    t = r.trends.copy()
    periods = [c for c in t.columns
               if c not in ("term", "documents", "growth", "recent share")]

    st.subheader("Rising and fading")
    st.caption(
        "Growth is the least-squares slope of each term's **document share** "
        "across periods — share, not count, so a period with more patents does "
        "not masquerade as growth.")

    top = t.nlargest(min(40, len(t)), "documents")
    fig = px.scatter(top, x="documents", y="growth", text="term",
                     color="growth", color_continuous_scale="RdYlGn",
                     labels={"documents": "Documents mentioning the term",
                             "growth": "Growth (share per period)"})
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(120,140,160,0.7)")
    fig.update_traces(textposition="top center", textfont=dict(size=9),
                      marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Right and above the line: established and still growing. "
               "Right and below: established but fading. Left and above: "
               "small but emerging — usually the interesting quadrant.")

    st.subheader("Term over time")
    picked = st.multiselect("Terms to plot", t["term"].tolist(),
                            default=t["term"].head(4).tolist())
    if picked:
        long = (t[t["term"].isin(picked)]
                .melt(id_vars="term", value_vars=periods,
                      var_name="Period", value_name="Documents"))
        st.plotly_chart(
            px.line(long, x="Period", y="Documents", color="term", markers=True)
              .update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0)),
            use_container_width=True)

    st.dataframe(t, use_container_width=True, hide_index=True)


def view_explore(r: P.Result) -> None:
    if r.terms.empty:
        st.info("No terms were extracted.")
        return
    term = st.selectbox("Technology term", r.terms["term"].tolist())
    also = r.variants.get(term, [])
    if also:
        st.caption("Also written as: " + ", ".join(f"*{v}*" for v in also))

    linked = r.edges[(r.edges["source"] == term) | (r.edges["target"] == term)]
    if not linked.empty:
        partners = [(row["target"] if row["source"] == term else row["source"],
                     row["ppmi"], row["documents"])
                    for _, row in linked.head(12).iterrows()]
        st.write("**Most associated with**")
        st.dataframe(pd.DataFrame(partners, columns=["Term", "PPMI", "Shared docs"]),
                     use_container_width=True, hide_index=True)

    st.write("**Documents mentioning it**")
    docs = r.documents_for(term, limit=25)
    if docs.empty:
        st.info("No documents matched.")
    else:
        st.dataframe(docs, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    st.title("TechScope")
    st.caption("What technologies are in this set of patents, how do they "
               "relate, and which are rising?")

    upload = st.file_uploader("Patent export (CSV or Excel)",
                              type=["csv", "xlsx", "xls"])
    if upload is None:
        st.info(
            "Upload a file with a **title** column and an **abstract** column. "
            "A date column is optional and unlocks trends.")
        with st.expander("What it does"):
            st.markdown(
                "1. **Finds candidate terms** — noun phrases from a dependency "
                "parse, normalised and lemmatised.\n"
                "2. **Ranks them by C-value**, which prefers the specific phrase "
                "over the generic one nested inside it.\n"
                "3. **Groups the synonyms** with a mutual k-nearest-neighbour "
                "graph over sentence embeddings, so *cryogenic tank* and "
                "*cryogenic storage tank* become one term.\n"
                "4. **Links terms by PPMI**, so association is measured above "
                "chance rather than by raw co-occurrence.\n"
                "5. **Tracks document share per period** when dates exist.\n\n"
                "Everything runs on your machine.")
        return

    try:
        frame = P.read_table(upload)
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        return

    cols = list(frame.columns)
    st.sidebar.header("Columns")
    title_col = st.sidebar.selectbox(
        "Title", cols, index=cols.index(guess(cols, "title", default=cols[0])))
    abstract_col = st.sidebar.selectbox(
        "Abstract", cols,
        index=cols.index(guess(cols, "abstract", "summary", default=cols[-1])))
    date_guess = guess(cols, "date", "year", "priority", default="")
    date_col = st.sidebar.selectbox(
        "Date (optional)", ["— none —"] + cols,
        index=(cols.index(date_guess) + 1) if date_guess else 0)

    st.sidebar.header("Extraction")
    min_words = st.sidebar.slider("Shortest term (words)", 1, 3, 2)
    min_freq = st.sidebar.slider("Minimum documents per term", 1, 20, 2,
                                 help="Terms appearing in fewer documents are dropped.")
    max_ratio = st.sidebar.slider(
        "Drop terms above this share of documents", 0.10, 1.00, 0.60, 0.05,
        help="A phrase in almost every document separates nothing — it is the "
             "corpus boilerplate. Same idea as a TF-IDF document-frequency ceiling.")
    max_terms = st.sidebar.slider("Terms to keep", 25, 800, 200, 25)

    st.sidebar.header("Grouping")
    neighbours = st.sidebar.slider(
        "Neighbours (k)", 2, 15, 5,
        help="Two terms merge only if each is inside the other's k nearest "
             "neighbours. Lower is stricter.")
    floor = st.sidebar.slider(
        "Similarity floor", 0.40, 0.95, 0.75, 0.01,
        help="Synonyms typically score above 0.85 and unrelated pairs below "
             "0.40, so anything in 0.70-0.85 behaves similarly.")

    st.sidebar.header("Relationships")
    min_edge = st.sidebar.slider("Minimum shared documents", 1, 20, 2)
    n_periods = st.sidebar.slider("Time periods", 3, 12, 6)

    cfg = P.Config(
        title_col=title_col, abstract_col=abstract_col,
        date_col="" if date_col == "— none —" else date_col,
        min_term_words=min_words, min_term_freq=min_freq, max_terms=max_terms,
        max_doc_ratio=max_ratio,
        neighbours=neighbours, similarity_floor=floor,
        min_edge_docs=min_edge, periods=n_periods)

    problems = P.check(frame, cfg)
    if problems:
        st.error(f"This file cannot be used — {len(problems)} problem(s):")
        for p in problems:
            st.markdown(f"- {p}")
        return

    usable = int(frame[abstract_col].notna().sum())
    st.success(f"{usable:,} of {len(frame):,} rows have an abstract.")
    if usable < 30:
        st.warning("Under about 30 documents there is little for the statistics "
                   "to work with — expect few terms and almost no relationships.")

    date_note = P.date_warning(frame, cfg)
    if date_note:
        st.warning(date_note)

    try:
        result = analyse(frame, cfg, digest(frame, cfg))
    except P.InputError as exc:
        st.error("  \n".join(f"- {p}" for p in exc.problems))
        return
    except MemoryError:
        st.error(
            "Ran out of memory while parsing. This normally means the abstract "
            "column holds full patent text rather than abstracts — check the "
            "column mapping, or split the file.")
        return
    except OSError as exc:
        st.error(
            f"Could not load a model: {exc}\n\n"
            "Install the language model with `python -m spacy download "
            "en_core_web_sm`. If this machine is offline, see the setup notes in "
            "`run_local.py` for pointing at a local copy.")
        return
    except Exception as exc:                       # noqa: BLE001 - last resort
        st.error(f"{type(exc).__name__}: {exc}")
        st.caption("If this looks like a bug rather than a bad input, the full "
                   "trace is in the terminal running Streamlit.")
        return

    for note in result.notes:
        st.info(note)

    if result.terms.empty:
        st.warning(
            "No terms survived the filters. The usual causes, in order: the "
            "corpus is too small for **Minimum documents per term** (try 1), "
            "the text is not English, or the abstract column is not prose.")
        return

    tabs = st.tabs(["Landscape", "Relationships", "Trends", "Explore"])
    with tabs[0]:
        view_landscape(result)
    with tabs[1]:
        view_graph(result)
    with tabs[2]:
        view_trends(result)
    with tabs[3]:
        view_explore(result)


if __name__ == "__main__":
    main()
