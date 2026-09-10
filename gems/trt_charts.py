"""Charts for the trt-pb Gems - the visual half the original tool was for.

Upload this next to any track module (`trt_replica.py`, `trt_corrected.py` or
`trt_intent.py`) and the patent export. It reads the `state` that module's
`report()` returns and draws the pictures trt-pb existed to draw.

    from trt_charts import selection, refine, landscape, network, trend
    sel = selection(state)          # top technologies, ranked
    landscape(sel)                  # what is in this corpus
    sel = refine(sel, drop=[3, 9], merge={"solid electrolyte": ["solid state electrolyte"]})
    network(sel, "Inclusion")       # how the survivors relate

Two renderers, because the sandbox has plotly 5.20 but not kaleido:
  * matplotlib draws every figure and it appears inline in the chat
  * plotly writes the same figure to a self-contained .html the analyst can
    open and pan, which is what the shipped tool produced as
    `graph-<Relationship>.html`

Every chart takes a Selection, never a bare list, so the keyword basket the
analyst arrived at is explicit, printable and reproducible. `sel.log` records
each edit in order, so a figure can always be traced back to the choices that
produced it.

Works with all six tracks. The adapters below normalise the differences: A and
B keep relationships in `state["dfmap"]` with six preposition classes, C keeps
them in `state["relations"]` with a typed ontology and a legacy projection.
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
# The backend is left as the host set it. Gemini's sandbox configures one that
# captures figures and shows them inline in the chat; forcing Agg here would
# save the PNG and display nothing.

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:                                     # pragma: no cover
    HAVE_PLOTLY = False

DPI = 130
INK = "#1a1a1a"
GRID = "#d9d9d9"
# Ordered, colour-blind-safe. Index 0 is the highlight colour.
PALETTE = ["#0b6fa4", "#d1600a", "#3f8f29", "#8d4bbb", "#b8123f",
           "#1b8a8f", "#7a6a00", "#a0468c", "#4a5d7e", "#6b7d1f"]


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.tick_params(colors=INK, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _fig(w=10.0, h=5.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig, ax


def _finish(fig, png):
    """Save a copy and show it. The saved PNG is the durable artifact; the
    show() is what puts the picture in front of the analyst in the chat."""
    fig.tight_layout()
    if png:
        fig.savefig(png, facecolor="white", bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)
    if png:
        print("wrote %s" % png)
    return fig


def _wrap(s, n=28):
    return "\n".join(textwrap.wrap(str(s), n)) or str(s)


# --------------------------------------------------------------------------- #
# adapters - one shape for all six tracks
# --------------------------------------------------------------------------- #

def track_of(state):
    """'C' for the intent-optimal tracks, 'AB' for replica and corrected."""
    return "C" if "relations" in state else "AB"


def track_module(state, needs=None):
    """The already-imported track module that produced this state.

    The module is whatever the analyst imported - trt_replica, trt_corrected
    or trt_intent - so it is found by looking for the one whose TRACK letter
    matches the shape of `state` and which has the function being asked for.
    """
    import sys
    want = "C" if track_of(state) == "C" else None
    best = None
    for m in list(sys.modules.values()):
        t = getattr(m, "TRACK", None)
        if t is None or not hasattr(m, "report"):
            continue
        if want == "C" and t != "C":
            continue
        if want is None and t == "C":
            continue
        if needs and not hasattr(m, needs):
            continue
        best = m
    return best


def canonical_map(state):
    """surface form -> canonical concept, for whichever grouping the track ran."""
    if "owner" in state:                                    # track C
        return dict(state["owner"])
    out = {}
    for key, members in state.get("keydict", {}).items():
        for m in members:
            out.setdefault(m, key)
        out.setdefault(key, key)
    return out


def terms_per_patent(state):
    """Canonical concepts present in each patent, in corpus row order."""
    keep = state["keep"]
    if "filtered_tech_keys" in keep:
        col = list(keep["filtered_tech_keys"])
    else:                                                   # fall back to abstracts
        owner = canonical_map(state)
        vocab = list(dict.fromkeys(state.get("keywords", [])))
        col = [[k for k in vocab if k in a.lower()] for a in state["abstracts"]]
        return [sorted({owner.get(k, k) for k in row}) for row in col]
    owner = canonical_map(state)
    return [sorted({owner.get(k, k) for k in row}) for row in col]


def patent_frequency(state):
    """Distinct patents per canonical concept."""
    df = Counter()
    for row in terms_per_patent(state):
        df.update(set(row))
    return df


def years_of(state):
    return [int(y) if y is not None and not pd.isna(y) else None for y in state["years"]]


def edges_of(state, relationship=None):
    """source, target, class, pubno - normalised across tracks.

    `relationship` filters on the six legacy classes in every track, so the
    same call works whether the track stores preposition buckets or a typed
    ontology projected onto them.
    """
    if track_of(state) == "C":
        r = state["relations"]
        mod = track_module(state)
        lm = getattr(mod, "LEGACY_MAP", {}) if mod else {}
        out = pd.DataFrame({"source": r["source"], "target": r["target"],
                            "cls": [lm.get(x, "Misc") for x in r["relation"]],
                            "typed": r["relation"], "pubno": r["pubno"]})
    else:
        d = state["dfmap"]
        out = pd.DataFrame({"source": d["T1"], "target": d["T2"],
                            "cls": d["Prep"], "typed": d["Prep"], "pubno": d["Pubno."]})
    if relationship:
        out = out[out["cls"] == relationship]
    return out.reset_index(drop=True)


def domains_of(state):
    return list(state.get("domains", []))


# --------------------------------------------------------------------------- #
# the keyword basket the analyst iterates on
# --------------------------------------------------------------------------- #

class Selection:
    """The chosen technologies, plus the edits that produced them.

    `terms` is the current basket in display order. `log` is every edit in
    order, so a chart can be traced back to the choices behind it.
    """

    def __init__(self, state, terms, log=None):
        self.state = state
        self.terms = list(terms)
        self.log = list(log or [])
        self.df = patent_frequency(state)

    def __repr__(self):
        return "<Selection %d terms, %d edits>" % (len(self.terms), len(self.log))

    def table(self):
        return pd.DataFrame({"#": range(1, len(self.terms) + 1),
                             "technology": self.terms,
                             "patents": [self.df[t] for t in self.terms]})

    def resolve(self, item):
        """Accept a 1-based index from the printed table, or the term itself."""
        if isinstance(item, (int, np.integer)):
            if 1 <= int(item) <= len(self.terms):
                return self.terms[int(item) - 1]
            raise KeyError("no row %s in the selection; it has %d rows" % (item, len(self.terms)))
        s = str(item).lower().strip()
        if s in self.terms:
            return s
        hits = [t for t in self.terms if s in t]
        if len(hits) == 1:
            return hits[0]
        raise KeyError("%r matches %d terms in the selection: %s" % (item, len(hits), hits[:6]))


def selection(state, top=20, min_patents=2):
    """Seed a basket with the most frequent technologies. Show it, then refine."""
    df = patent_frequency(state)
    terms = [t for t, n in df.most_common() if n >= min_patents][:top]
    sel = Selection(state, terms, ["seeded with the top %d concepts by patent count" % len(terms)])
    show(sel)
    return sel


def refine(sel, keep=None, drop=None, add=None, merge=None):
    """Apply the analyst's edit and return a new Selection.

    keep  - indices or terms to keep, dropping everything else
    drop  - indices or terms to remove
    add   - terms to bring in from the wider corpus
    merge - {canonical: [other, ...]} folds the others into the canonical name
            for every chart from here on, and sums their patent counts

    Indices refer to the last printed table, which is why every step prints one.
    """
    state = sel.state
    terms = list(sel.terms)
    log = list(sel.log)

    if merge:
        owner = state.setdefault("_chart_merges", {})
        for canon, others in merge.items():
            canon = str(canon).lower().strip()
            others = [sel.resolve(o) if not isinstance(o, str) or o in sel.terms else str(o).lower()
                      for o in others]
            for o in others:
                owner[o] = canon
            terms = [canon if t in others else t for t in terms]
            if canon not in terms:
                terms.append(canon)
            log.append("merged %s into %r" % (", ".join(repr(o) for o in others), canon))
        seen = set()
        terms = [t for t in terms if not (t in seen or seen.add(t))]

    if keep is not None:
        chosen = [sel.resolve(k) for k in keep]
        log.append("kept %d of %d: %s" % (len(chosen), len(sel.terms),
                                          ", ".join(repr(c) for c in chosen)))
        terms = chosen
    if drop:
        gone = [sel.resolve(d) for d in drop]
        terms = [t for t in terms if t not in gone]
        log.append("dropped " + ", ".join(repr(g) for g in gone))
    if add:
        df = patent_frequency(state)
        for a in add:
            a = str(a).lower().strip()
            if a not in terms:
                terms.append(a)
        log.append("added " + ", ".join(repr(str(a).lower().strip()) for a in add))

    out = Selection(state, terms, log)
    show(out)
    return out


def basket(state, terms, merge=None):
    """Build a selection from an explicit spec, in one call, from scratch.

    The sandbox does not survive between chat turns: `state` and any Selection
    built in an earlier message are gone by the next one. So a multi-turn
    refinement cannot pass objects forward - it has to be replayable. This
    rebuilds the analyst's basket from a list of names plus the merges agreed
    so far, which is what every turn after the first should call.

        sel = basket(state,
                     ["hydrogen", "compounds", "treatment", "cancer"],
                     merge={"hydrogen": ["hydrogen gas"],
                            "compounds": ["compound"]})
    """
    if merge:
        owner = state.setdefault("_chart_merges", {})
        for canon, others in merge.items():
            for o in others:
                owner[str(o).lower().strip()] = str(canon).lower().strip()
    terms = [str(t).lower().strip() for t in terms]
    log = ["rebuilt from an explicit basket of %d technologies" % len(terms)]
    if merge:
        log += ["merged %s into %r" % (", ".join(map(repr, v)), k) for k, v in merge.items()]
    sel = Selection(state, terms, log)
    show(sel)
    return sel


def show(sel):
    """Print the basket. Every refinement prints one, so indices stay meaningful."""
    t = sel.table()
    print("SELECTION  %d technologies, %d patents in the corpus"
          % (len(sel.terms), len(sel.state["keep"])))
    print(t.to_string(index=False))
    if sel.log:
        print("edits: " + "; ".join(sel.log[-3:]))
    return t


def _merged_terms(sel):
    """Per-patent term lists with the analyst's merges applied."""
    extra = sel.state.get("_chart_merges", {})
    rows = terms_per_patent(sel.state)
    if not extra:
        return rows
    return [sorted({extra.get(t, t) for t in row}) for row in rows]


# --------------------------------------------------------------------------- #
# 1. the landscape - what is in this corpus
# --------------------------------------------------------------------------- #

def landscape(sel, png="landscape.png", html=None):
    """Ranked bar of patents per technology. The opening picture."""
    rows = _merged_terms(sel)
    counts = Counter()
    for r in rows:
        counts.update(set(r))
    terms = sorted(sel.terms, key=lambda t: -counts[t])
    vals = [counts[t] for t in terms]
    n = len(sel.state["keep"])

    fig, ax = _fig(10, max(3.2, 0.34 * len(terms) + 1.4))
    y = np.arange(len(terms))[::-1]
    ax.barh(y, vals, color=PALETTE[0], height=0.68)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.012, yi, "%d  (%.0f%%)" % (v, 100 * v / n),
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    _style(ax, "Technology landscape: patents mentioning each concept",
           "patents (of %d)" % n)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.grid(False, axis="y")
    _finish(fig, png)

    if html and HAVE_PLOTLY:
        f = go.Figure(go.Bar(x=vals[::-1], y=terms[::-1], orientation="h",
                             marker_color=PALETTE[0],
                             hovertemplate="%{y}<br>%{x} patents<extra></extra>"))
        f.update_layout(title="Technology landscape", xaxis_title="patents (of %d)" % n,
                        template="plotly_white", height=28 * len(terms) + 220)
        _write_html(f, html)
    return pd.DataFrame({"technology": terms, "patents": vals})


# --------------------------------------------------------------------------- #
# 2. the relationship graph - the shipped tool's headline output
# --------------------------------------------------------------------------- #

def _components(nodes, edges):
    """Connected components, largest first, so each can be laid out on its own."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, comp = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        comps.append(sorted(comp))
    return sorted(comps, key=len, reverse=True)


def _spring(nodes, edges, seed=0, iters=420):
    """Fruchterman-Reingold with a pull toward the centre.

    Written out so the module needs no networkx. The gravity term is what
    stops loosely connected nodes drifting to the edge of the frame, which is
    the usual way a small patent graph turns into a ring of unreadable dots.
    """
    n = len(nodes)
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}
    rng = np.random.default_rng(seed)
    idx = {v: i for i, v in enumerate(nodes)}
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pos = np.c_[np.cos(ang), np.sin(ang)] * 0.6 + rng.normal(scale=0.03, size=(n, 2))
    E = np.array([[idx[a], idx[b]] for a, b in edges if a in idx and b in idx], dtype=int)
    k = 1.3 / math.sqrt(n)
    temp = 0.25
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, np.inf)
        dist = np.clip(dist, 1e-3, None)
        disp = ((k * k / dist ** 2)[..., None] * delta).sum(axis=1)
        if len(E):
            d = pos[E[:, 0]] - pos[E[:, 1]]
            dd = np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-9, None)
            att = (dd / k) * d
            np.add.at(disp, E[:, 0], -att)
            np.add.at(disp, E[:, 1], att)
        disp -= pos * 0.28                      # gravity toward the centre
        norm = np.clip(np.linalg.norm(disp, axis=1, keepdims=True), 1e-9, None)
        pos += disp / norm * np.minimum(norm, temp)
        temp *= 0.992
    span = np.ptp(pos, axis=0)
    span[span == 0] = 1.0
    return {v: tuple(((pos[i] - pos.min(axis=0)) / span * 2 - 1)) for v, i in idx.items()}


def _layout(nodes, edges, seed=0):
    """Lay out each connected component, then pack the components in a grid.

    Without the packing step a graph with two unrelated clusters puts them in
    opposite corners and leaves the middle of the picture empty.
    """
    comps = _components(nodes, edges)
    if not comps:
        return {}
    cols = max(1, int(math.ceil(math.sqrt(len(comps)))))
    biggest = len(comps[0])
    pos = {}
    for i, comp in enumerate(comps):
        member = set(comp)
        sub = [(a, b) for a, b in edges if a in member and b in member]
        p = _spring(comp, sub, seed=seed + i)
        r = 0.34 + 0.66 * math.sqrt(len(comp) / biggest)
        cx, cy = (i % cols) * 2.5, -(i // cols) * 2.5
        for v, (x, y) in p.items():
            pos[v] = (cx + x * r, cy + y * r)
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    sx = max(np.ptp(xs), 1e-9)
    sy = max(np.ptp(ys), 1e-9)
    return {v: ((x - xs.min()) / sx * 2 - 1, (y - ys.min()) / sy * 2 - 1)
            for v, (x, y) in pos.items()}


def network(sel, relationship="Inclusion", min_patents=1, png="network.png",
            html="network.html", label_all=True):
    """Technology relationship graph for one class.

    Node area is patents mentioning the technology, edge width is the number of
    distinct patents supporting that relationship, and the arrow is the
    direction the sentence ran. The shipped tool drew every edge the same
    weight and lost the direction; both are kept here.
    """
    state = sel.state
    extra = state.get("_chart_merges", {})
    e = edges_of(state, relationship)
    if len(e) == 0:
        print("no %s relationships in this corpus" % relationship)
        return pd.DataFrame()
    e["source"] = e["source"].map(lambda t: extra.get(t, t))
    e["target"] = e["target"].map(lambda t: extra.get(t, t))
    keep = set(sel.terms)
    e = e[e["source"].isin(keep) & e["target"].isin(keep) & (e["source"] != e["target"])]
    if len(e) == 0:
        print("no %s relationships among the %d selected technologies" % (relationship, len(keep)))
        return pd.DataFrame()

    agg = (e.groupby(["source", "target"])
             .agg(patents=("pubno", "nunique"), occurrences=("pubno", "size"))
             .reset_index())
    agg = agg[agg["patents"] >= min_patents].sort_values("patents", ascending=False)
    if agg.empty:
        print("no %s edge reaches %d distinct patents" % (relationship, min_patents))
        return agg

    nodes = sorted(set(agg["source"]) | set(agg["target"]))
    pos = _layout(nodes, list(zip(agg["source"], agg["target"])))
    freq = patent_frequency(state)
    counts = Counter()
    for row in _merged_terms(sel):
        counts.update(set(row))
    sizes = np.array([max(counts.get(v, freq.get(v, 1)), 1) for v in nodes], dtype=float)
    area = 170 + 1100 * (sizes / sizes.max())
    wmax = agg["patents"].max()

    fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for _, r in agg.iterrows():
        x1, y1 = pos[r["source"]]
        x2, y2 = pos[r["target"]]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#8a8a8a",
                                    linewidth=0.6 + 3.2 * r["patents"] / wmax,
                                    alpha=0.55, shrinkA=13, shrinkB=15,
                                    connectionstyle="arc3,rad=0.08"))
    ax.scatter([pos[v][0] for v in nodes], [pos[v][1] for v in nodes],
               s=area, c=PALETTE[0], alpha=0.92, zorder=3, edgecolors="white", linewidths=1.5)
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    for v in nodes:
        if not (label_all or counts.get(v, 0) >= np.median(sizes)):
            continue
        off = 0.030 * math.sqrt(max(area[nodes.index(v)], 1)) / 10.0 + 0.055
        ax.annotate(_wrap(v, 20), pos[v], xytext=(0, -off * 100), textcoords="offset points",
                    fontsize=8.2, ha="center", va="top", zorder=5, color=INK,
                    path_effects=halo)
    ax.margins(0.16)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("%s relationships among %d technologies\n"
                 "node = patents mentioning it, edge width = patents supporting the link, "
                 "arrow = direction in the sentence"
                 % (relationship, len(nodes)), fontsize=12, color=INK, loc="left", pad=14)
    _finish(fig, png)

    if html and HAVE_PLOTLY:
        ex, ey = [], []
        for _, r in agg.iterrows():
            x1, y1 = pos[r["source"]]; x2, y2 = pos[r["target"]]
            ex += [x1, x2, None]; ey += [y1, y2, None]
        f = go.Figure()
        f.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                               line=dict(color="#9a9a9a", width=1)))
        f.add_trace(go.Scatter(
            x=[pos[v][0] for v in nodes], y=[pos[v][1] for v in nodes],
            mode="markers+text", text=nodes, textposition="top center",
            marker=dict(size=np.sqrt(area) * 0.85, color=PALETTE[0], line=dict(color="white", width=1.5)),
            customdata=[counts.get(v, 0) for v in nodes],
            hovertemplate="%{text}<br>%{customdata} patents<extra></extra>"))
        f.update_layout(title="%s relationships" % relationship, template="plotly_white",
                        showlegend=False, height=720,
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
        _write_html(f, html)

    print("%s: %d technologies, %d directed edges; strongest:" % (relationship, len(nodes), len(agg)))
    print(agg.head(8).to_string(index=False))
    return agg


def _write_html(fig, path):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote %s (%.0f KB, open it to pan and hover)" % (path, len(html) / 1024))


# --------------------------------------------------------------------------- #
# 3. prevalence through time
# --------------------------------------------------------------------------- #

def _periods(years, n=6):
    ys = sorted({y for y in years if y})
    if not ys:
        return []
    lo, hi = ys[0], ys[-1] + 1
    n = max(1, min(n, hi - lo))
    step = (hi - lo) / n
    edges = sorted({lo + int(round(i * step)) for i in range(n + 1)} | {lo, hi})
    return [("%d-%d" % (a, b - 1) if b - 1 > a else str(a), a, b)
            for a, b in zip(edges, edges[1:])]


def prevalence(sel, periods=6, png="prevalence.png", html=None):
    """Share of each period's patents that mention the technology.

    A share, not a count, so a period that simply contains more patents does
    not read as growth for everything in it. This is the corrected form of the
    shipped Relevancy Score.
    """
    state = sel.state
    years = years_of(state)
    rows = _merged_terms(sel)
    buckets = _periods(years, periods)
    if not buckets:
        print("no usable years; no time chart")
        return pd.DataFrame()

    data = {}
    sizes = []
    for label, a, b in buckets:
        idx = [i for i, y in enumerate(years) if y and a <= y < b]
        sizes.append(len(idx))
        for t in sel.terms:
            data.setdefault(t, []).append(
                sum(1 for i in idx if t in rows[i]) / len(idx) if idx else 0.0)

    labels = [b[0] for b in buckets]
    fig, ax = _fig(10.5, 5.8)
    x = np.arange(len(labels))
    show_terms = sel.terms[:8]
    w = 0.8 / max(len(show_terms), 1)
    for i, t in enumerate(show_terms):
        ax.bar(x + i * w - 0.4 + w / 2, [v * 100 for v in data[t]], width=w * 0.92,
               label=_wrap(t, 22), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\nn=%d" % (l, s) for l, s in zip(labels, sizes)], fontsize=8.5)
    _style(ax, "Prevalence: share of each period's patents mentioning the technology",
           None, "% of patents in the period")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _finish(fig, png)

    if html and HAVE_PLOTLY:
        f = go.Figure()
        for i, t in enumerate(show_terms):
            f.add_trace(go.Bar(name=t, x=labels, y=[v * 100 for v in data[t]],
                               marker_color=PALETTE[i % len(PALETTE)]))
        f.update_layout(barmode="group", template="plotly_white",
                        title="Prevalence by period", yaxis_title="% of patents")
        _write_html(f, html)
    return pd.DataFrame(data, index=labels)


def adoption(sel, png="adoption.png", html=None):
    """Cumulative distinct patents per technology.

    Adoption history only. It cannot go down, so it is never evidence of
    decline - the shipped tool plotted this on the same axes as the trend and
    invited exactly that misreading.
    """
    state = sel.state
    years = years_of(state)
    rows = _merged_terms(sel)
    ys = sorted({y for y in years if y})
    fig, ax = _fig(10, 5.6)
    out = {}
    for i, t in enumerate(sel.terms[:8]):
        per = [sum(1 for j, y in enumerate(years) if y == yr and t in rows[j]) for yr in ys]
        cum = np.cumsum(per)
        out[t] = cum
        ax.plot(ys, cum, marker="o", markersize=3.5, linewidth=1.8,
                color=PALETTE[i % len(PALETTE)], label=_wrap(t, 22))
    _style(ax, "Adoption history: cumulative patents (never evidence of decline)",
           "application year", "patents to date")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _finish(fig, png)

    if html and HAVE_PLOTLY:
        f = go.Figure()
        for i, (t, cum) in enumerate(out.items()):
            f.add_trace(go.Scatter(x=ys, y=cum, mode="lines+markers", name=t,
                                   line=dict(color=PALETTE[i % len(PALETTE)])))
        f.update_layout(template="plotly_white", title="Adoption history",
                        xaxis_title="application year", yaxis_title="patents to date")
        _write_html(f, html)
    return pd.DataFrame(out, index=ys)


# --------------------------------------------------------------------------- #
# 4. the trend plot - trt-pb's signature picture
# --------------------------------------------------------------------------- #

def _windowed(state, domain, term, width):
    """(labels, DF, influence, censored flags) for whichever track produced state.

    The two intent-optimal tracks name the influence column differently: the
    three-track one shrinks the function score toward the corpus mean and
    marks censored windows explicitly, the intent-preserving one reports a
    bootstrapped median and leaves it null when support is too thin. Both mean
    "no usable influence estimate for this window", so both become a hollow
    point on the chart.
    """
    if track_of(state) == "C":
        mod = track_module(state, "trajectory")
        if mod is None:
            raise RuntimeError("import the track C module before charting a trend")
        tr = mod.trajectory(state, term, width, "filtered_tech_keys", domain)
        col = "influence" if "influence" in tr.columns else "FS_shrunk"
        infl = [None if v is None or pd.isna(v) else float(v) for v in tr[col]]
        flagged = list(tr["censored"]) if "censored" in tr.columns else [False] * len(tr)
        cens = [bool(c) or v is None for c, v in zip(flagged, infl)]
        infl = [None if c else v for c, v in zip(cens, infl)]
        return list(tr["window"]), [int(v) for v in tr["DF"]], infl, cens
    mod = track_module(state, "tta_technology")
    if mod is None:
        raise RuntimeError("import the track A or B module before charting a trend")
    w = mod.tta_technology(state, domain, term, width)["windowed"]
    return list(w.iloc[:, 0]), [int(v) for v in w["DF"]], [float(v) for v in w["FS"]], [False] * len(w)


def trend(state, domain, term, width=3, png="trend.png", html="trend.html"):
    """Influence against document frequency, joined in time order.

    The picture the whole tool was built to produce. Each point is a time
    window; the arrow points from the earlier window to the later one. Reading
    the four quadrants: up and right is growing, right and down is
    generalizing, down and left is declining, left and up is repositioning.

    Windows the track marked censored are drawn hollow on the DF axis only,
    because their citation counts are unobserved rather than small.
    """
    labels, dfv, infl, cens = _windowed(state, domain, term, width)
    pts = [(l, d, i, c) for l, d, i, c in zip(labels, dfv, infl, cens)]
    solid = [(l, d, i) for l, d, i, c in pts if not c and i is not None]
    if len(solid) < 2:
        print("%r in %s: fewer than two windows have an influence estimate; "
              "the trend plot would be a single point." % (term, domain))
    fig, ax = _fig(9.2, 6.4)
    if solid:
        xs = [p[2] for p in solid]; ys = [p[1] for p in solid]
        ax.plot(xs, ys, color=PALETTE[0], linewidth=1.6, alpha=0.75, zorder=2)
        for i in range(len(solid) - 1):
            ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=PALETTE[0], linewidth=1.8),
                        zorder=3)
        ax.scatter(xs, ys, s=110, color=PALETTE[0], zorder=4, edgecolors="white", linewidths=1.4)
        for (l, d, i) in solid:
            ax.annotate(l, (i, d), textcoords="offset points", xytext=(8, 7),
                        fontsize=8.5, color=INK)
    hollow = [(l, d) for l, d, i, c in pts if c or i is None]
    if hollow:
        ax.scatter([0] * len(hollow), [h[1] for h in hollow], s=90, facecolors="none",
                   edgecolors=PALETTE[1], linewidths=1.6, zorder=4)
        for l, d in hollow:
            ax.annotate("%s (citations unobserved)" % l, (0, d), textcoords="offset points",
                        xytext=(8, -12), fontsize=8, color=PALETTE[1])
    _style(ax, "Trend: %s in %s\narrows run forward in time" % (term, domain),
           "influence (cohort-normalised)" if track_of(state) == "C" else "FS (function score)",
           "DF (patents in the window)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    _finish(fig, png)

    if html and HAVE_PLOTLY and solid:
        f = go.Figure(go.Scatter(x=[p[2] for p in solid], y=[p[1] for p in solid],
                                 mode="lines+markers+text", text=[p[0] for p in solid],
                                 textposition="top center",
                                 marker=dict(size=13, color=PALETTE[0]),
                                 line=dict(color=PALETTE[0])))
        f.update_layout(template="plotly_white",
                        title="Trend: %s in %s" % (term, domain),
                        xaxis_title="influence", yaxis_title="DF (patents)")
        _write_html(f, html)
    return pd.DataFrame({"window": labels, "DF": dfv, "influence": infl, "censored": cens})


def trajectory3d(state, domain, term, width=3, html="trajectory3d.html", png="trajectory3d.png"):
    """Influence x time x document frequency, the third of the shipped views."""
    labels, dfv, infl, cens = _windowed(state, domain, term, width)
    xs = [i if i is not None else 0 for i in infl]
    zs = dfv
    ys = list(range(len(labels)))
    fig = plt.figure(figsize=(9.5, 7), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, color=PALETTE[0], linewidth=1.8, marker="o")
    for x, y, z, l in zip(xs, ys, zs, labels):
        ax.text(x, y, z, "  " + l, fontsize=8, color=INK)
    ax.set_xlabel("influence", fontsize=9)
    ax.set_ylabel("window", fontsize=9)
    ax.set_zlabel("DF (patents)", fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Trajectory: %s in %s" % (term, domain), fontsize=12, color=INK, loc="left")
    _finish(fig, png)

    if HAVE_PLOTLY and html:
        f = go.Figure(go.Scatter3d(x=xs, y=labels, z=zs, mode="lines+markers+text",
                                   text=labels, marker=dict(size=5, color=PALETTE[0]),
                                   line=dict(color=PALETTE[0], width=4)))
        f.update_layout(template="plotly_white", title="Trajectory: %s in %s" % (term, domain),
                        scene=dict(xaxis_title="influence", yaxis_title="window",
                                   zaxis_title="DF (patents)"), height=680)
        _write_html(f, html)
    return pd.DataFrame({"window": labels, "DF": dfv, "influence": infl})


# --------------------------------------------------------------------------- #
# 5. domain specificity and emergence
# --------------------------------------------------------------------------- #

def specificity(state, domain, top=15, png="specificity.png", html=None):
    """What this domain does that the rest of the corpus does not.

    Track C reports posterior enrichment with a credible interval, and the bar
    is drawn with that interval so a term supported by four patents cannot
    look as solid as one supported by eighty. A and B report raw lift, which
    has no interval, and the chart says so.
    """
    modC = track_module(state, "ds_table") if track_of(state) == "C" else None
    if modC:
        t = modC.ds_table(state, domain).head(top)
        if len(t) == 0:
            # Every term in this domain sits below the frequency floor, which is
            # a real answer rather than an error: there is nothing this domain
            # does that the corpus does not. Drawing an empty axis and taking
            # max() of nothing is not.
            print("no term in %r clears the frequency floor, so there is nothing "
                  "specific to chart. Lower the floor with ds_table(state, %r, "
                  "floor=1) to see the sparse terms." % (domain, domain))
            return t
        if "enrichment" in t.columns:
            # intent-preserving C: posterior enrichment with a credible interval
            terms = list(t["term"])[::-1]
            val = [max(v, 1e-6) for v in list(t["enrichment"])[::-1]]
            lo = [max(v, 1e-6) for v in list(t["ci_low"])[::-1]]
            hi = [min(h, v * 6) for h, v in zip(list(t["ci_high"])[::-1], val)]
            sup = list(t["supporting_patents"])[::-1]
            fig, ax = _fig(10, max(3.4, 0.36 * len(terms) + 1.4))
            y = np.arange(len(terms))
            ax.barh(y, val, color=PALETTE[2], height=0.62)
            ax.errorbar(val, y, xerr=[np.array(val) - np.array(lo), np.array(hi) - np.array(val)],
                        fmt="none", ecolor="#555", elinewidth=1.1, capsize=3)
            for yi, v, n in zip(y, val, sup):
                ax.text(v * 1.03, yi, " %d patents" % n, va="center", fontsize=8, color=INK)
            ax.set_xscale("log")
            title = ("Domain specificity in %s\nposterior enrichment, bars are 95%% credible "
                     "intervals" % domain)
            xlab = "times more characteristic than the corpus at large"
        else:
            # three-track C: fixed lift with a floor, and the log-odds challenger
            lift_col = next(c for c in t.columns if c.startswith("lift"))
            t = t.sort_values("log-odds z", ascending=False)
            terms = list(t["term"])[::-1]
            val = list(t["log-odds z"])[::-1]
            lifts = list(t[lift_col])[::-1]
            sup = list(t["DF_TD"])[::-1]
            fig, ax = _fig(10, max(3.4, 0.36 * len(terms) + 1.4))
            y = np.arange(len(terms))
            ax.barh(y, val, color=PALETTE[2], height=0.62)
            for yi, v, l, n in zip(y, val, lifts, sup):
                lab = " lift %.1f, %d patents" % (l, n) if l == l and l is not None                     else " below the floor, %d patents" % n
                ax.text(v + max(val) * 0.012, yi, lab, va="center", fontsize=8, color=INK)
            ax.set_xlim(0, max(val) * 1.35)
            title = ("Domain specificity in %s\nweighted log-odds (bar) with fixed lift "
                     "alongside; the challenger has not yet earned replacement" % domain)
            xlab = "weighted log-odds z"
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(x, 30) for x in terms], fontsize=8.5)
        _style(ax, title, xlab)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
        ax.grid(False, axis="y")
        _finish(fig, png)
        return t
    # A and B: raw lift, no interval
    ds = state.get("ds", {})
    rows = sorted(((v, t) for (d, t), v in ds.items() if d == domain), reverse=True)[:top]
    if not rows:
        print("no domain-specificity scores for %r" % domain)
        return pd.DataFrame()
    val = [r[0] for r in rows][::-1]; terms = [r[1] for r in rows][::-1]
    fig, ax = _fig(10, max(3.4, 0.34 * len(terms) + 1.4))
    y = np.arange(len(terms))
    ax.barh(y, val, color=PALETTE[2], height=0.66)
    ax.set_yticks(y); ax.set_yticklabels(terms, fontsize=9)
    _style(ax, "Domain specificity in %s\nraw lift, no frequency floor and no interval: "
                "a term seen once can top this chart" % domain,
           "P(term | domain) / P(term)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7); ax.grid(False, axis="y")
    _finish(fig, png)
    return pd.DataFrame({"term": terms[::-1], "lift": val[::-1]})


def emergence_map(state, domain=None, top=25, png="emergence.png", html=None):
    """Emergence against prevalence: what is new versus what is established.

    Track C only, because it is the only track with an emergence signal that
    is independent of citations. Bubble area is the number of patents behind
    the point, so a burst supported by two patents reads as small.
    """
    modC = track_module(state, "emergence_table") if track_of(state) == "C" else None
    if modC is None:
        print("emergence is a Track C signal; this state came from another track")
        return pd.DataFrame()
    t = modC.emergence_table(state, td=domain, top=top)
    if t.empty:
        print("no concept has enough recent support for an emergence estimate")
        return t
    df = patent_frequency(state)
    x = [df.get(term, 1) for term in t["term"]]
    fig, ax = _fig(10, 6.4)
    sizes = 60 + 900 * (t["support_recent"] / max(t["support_recent"].max(), 1))
    sc = ax.scatter(x, t["burst"], s=sizes, c=t["slope"], cmap="viridis",
                    alpha=0.85, edgecolors="white", linewidths=1.2)
    for xi, yi, lab in zip(x, t["burst"], t["term"]):
        ax.annotate(_wrap(lab, 22), (xi, yi), textcoords="offset points", xytext=(7, 5),
                    fontsize=7.5, color=INK)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("slope of prevalence", fontsize=9)
    ax.axhline(2.0, color=PALETTE[4], linestyle="--", linewidth=1,
               label="burst = 2 (a common reporting floor)")
    _style(ax, "Emergence against establishment%s\nbubble area = patents in the recent window"
           % ((" in " + domain) if domain else ""),
           "patents mentioning the concept (established)", "burst (recent vs historical)")
    ax.legend(fontsize=8, frameon=False)
    _finish(fig, png)
    return t


# --------------------------------------------------------------------------- #
# a whole first look, in one call
# --------------------------------------------------------------------------- #

def overview(state, top=15, relationship="Inclusion", periods=6):
    """Shot one: the landscape, the graph and prevalence, then the basket.

    Returns the Selection so the analyst can refine it and redraw.
    """
    sel = selection(state, top=top)
    landscape(sel, png="01-landscape.png")
    network(sel, relationship, png="02-network.png", html="02-network.html")
    prevalence(sel, periods=periods, png="03-prevalence.png")
    print("\nRefine with: sel = refine(sel, drop=[...], keep=[...], "
          "merge={'canonical': ['other']}) and redraw.")
    return sel
