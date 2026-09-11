"""trt-pb, reduced to what it is for: keyword trends and keyword pairs.

Three calls, and the analyst never types a keyword - only a row number.

    state = ingest("export.csv")   -> numbered keyword menu
    primary(state, 6)              -> three graphs for keyword 6,
                                      then the numbered secondary menu
    secondary(state, 6, 2)         -> three graphs for the pair

Runs in a model's code sandbox: standard library, pandas, numpy, matplotlib.
plotly is used only if present, to write an interactive copy alongside the
inline picture.

The counting rules, stated once because every number depends on them:

  * A keyword "occurs" in a patent when it appears in the title or abstract as
    a whole word. Matching is word-bounded, so "cell" does not match "fuel
    cells" and "gas" does not match "gasket".
  * Occurrences by year are counted as PATENTS, not mentions: a patent that
    says "hydrogen" nine times counts once. Raw mention counts are in the
    returned table as well, because the two answer different questions.
  * A pair co-occurs when both keywords appear in the same patent. That is
    co-occurrence at document level, which is what the original tool meant.
  * The year is the application year. A patent with no usable year is dropped
    from the year charts and the fact is printed.

Both menus are deterministic functions of the export, so the whole session
replays from two numbers. That matters: the sandbox is wiped between chat
turns, and `primary(state, 6)` re-prints the same secondary menu every time.
"""

from __future__ import annotations

import datetime
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
# The backend is whatever the host set. Gemini's sandbox configures one that
# shows figures inline; forcing Agg here would save files and display nothing.

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

DPI = 130
INK = "#1a1a1a"
GRID = "#dcdcdc"
FAINT = "#c9d4dc"
BLUE = "#0b6fa4"
ORANGE = "#d1600a"
GREEN = "#3f8f29"
PURPLE = "#8d4bbb"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

HINTS = {
    "abstract": ("abstract", "summary"),
    "title": ("title",),
    "date": ("application date", "application dates", "priority date",
             "filing date", "publication date", "date", "year"),
    "pubno": ("publication number", "publication numbers", "pubno", "pub no", "id"),
}


def _pick(cols, hints, explicit=None):
    if explicit:
        if explicit not in cols:
            raise SystemExit("column %r not in the file. Columns: %s"
                             % (explicit, ", ".join(map(str, cols))))
        return explicit
    low = [(str(c).lower(), c) for c in cols]
    for h in hints:
        for lc, c in low:
            if h in lc:
                return c
    return None


_YEAR = re.compile(r"(?<!\d)((?:1[6-9]|2[01])\d{2})(?!\d)")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def year_of(value):
    """The year, or None. Never guessed from a publication number.

    The lookarounds matter: an unanchored four-digit match reads
    "US2024123456A1" as 2024 and fills the chart with fiction.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return int(value.year)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = int(value)
        if 20000 <= n <= 80000:
            return int((_EXCEL_EPOCH + pd.Timedelta(days=n)).year)
        return n if 1600 <= n <= 2199 else None
    m = _YEAR.search(str(value))
    return int(m.group(1)) if m else None


_JP = re.compile(r"\b(?:PROBLEM TO BE SOLVED|SOLUTION|SELECTED DRAWING|ADVANTAGE|EFFECT)\s*:\s*",
                 re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean(text):
    if not isinstance(text, str):
        return ""
    out = re.sub(r"^\s*\([^)]*\)\s*\n", "", text)
    out = _JP.sub("", out)
    out = out.replace(";", ".").replace("–", "-").replace("—", "-")
    return _WS.sub(" ", out).strip()


# --------------------------------------------------------------------------- #
# candidate keywords
# --------------------------------------------------------------------------- #

STOP = frozenset("""
a an the this that these those said such its their his her our your some any
each one two three first second third another other same both all every and or
but nor so yet if than because while although though whereas of in on at by to
from with without within into onto upon through during before after above below
under over between among across against about around near is are was were be
been being am do does did doing have has had having can could may might must
shall should will would not no more most less least very much many few several
which who whom whose what when where why how comprising comprises comprise
including includes include consisting consists wherein whereby thereof therein
thereto thereby herein hereof according relates relating provided provides
provide disclosed discloses present invention embodiment embodiments example
examples figure figures fig use uses used using also further still then thus
hence therefore it they them he she we you i there here via as for per said
respectively preferably particularly especially generally typically
useful novel suitable various certain particular respective additional
exemplary illustrative corresponding associated related desired predetermined
plurality specification claim claims disclosure aspect aspects
""".split())

# A phrase that is only one of these names a category, not a technology.
GENERIC = frozenset("""apparatus method methods system systems device devices
assembly arrangement means unit units mechanism structure module equipment
machine process processes technique procedure application step steps portion
member element part parts side end surface material composition compositions
""".split())

_WORD = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")


def _phrases(text, lo=1, hi=4):
    """Maximal runs of content words between stopwords or punctuation.

    Stands in for a dependency parse, which the sandbox has no parser for. A
    one-word candidate must be at least three characters: a single letter is a
    chemical variable, not a keyword.
    """
    out = []
    for sentence in re.split(r"[.!?,:;()\[\]/]", text.lower()):
        run = []
        for token in sentence.split() + [""]:
            m = _WORD.match(token)
            w = m.group(0).strip("-'") if m else ""
            if not w or w in STOP or w.isdigit():
                if run:
                    if len(run) == 1 and len(run[0]) < 3:
                        pass
                    elif len(run) == 1 and run[0].endswith("ed"):
                        # A bare past participle - "defined", "described" - is
                        # drafting language. Kept inside longer phrases, where
                        # it can be doing real work ("coated separator").
                        pass
                    elif all(x in GENERIC for x in run):
                        pass
                    elif lo <= len(run) <= hi:
                        out.append(" ".join(run))
                    elif len(run) > hi:
                        out.append(" ".join(run[-hi:]))
                run = []
            else:
                run.append(w)
    return out


def _singular(word):
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"),
                     ("xes", "x")):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            return word[:-len(suf)] + rep
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _normalise(phrase):
    words = phrase.split()
    words = words[:-1] + [_singular(words[-1])]
    return " ".join(words)


def _inflected(term):
    """Regex for the term and the plural it was normalised from.

    Candidates are stored singular, so a pattern built from the bare term
    misses every "diseases" in the text and reports fewer mentions than
    patents - a number that is visibly impossible.
    """
    esc = re.escape(term)
    forms = [esc + "(?:e?s)?"]
    if term.endswith("y") and len(term) > 2:
        forms.append(re.escape(term[:-1]) + "ies")
    return "(?:" + "|".join(forms) + ")"


def _pattern(terms):
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not terms:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:" + "|".join(_inflected(t) for t in terms) + r")\b",
                      re.IGNORECASE)


# --------------------------------------------------------------------------- #
# step 1 - ingest the export, show the keyword menu
# --------------------------------------------------------------------------- #

def ingest(path, top=40, min_patents=2, abstract=None, date=None, title=None):
    """Read the export, extract candidate keywords, print the numbered menu.

    Returns the state the other two calls need. Nothing here is a judgement:
    the menu is ranked by how many patents mention each keyword, and choosing
    among them is the analyst's job.
    """
    frame = (pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls"))
             else pd.read_csv(path))
    cols = list(frame.columns)
    c_abs = _pick(cols, HINTS["abstract"], abstract)
    c_ttl = _pick(cols, HINTS["title"], title)
    c_dat = _pick(cols, HINTS["date"], date)
    c_pub = _pick(cols, HINTS["pubno"])
    if c_abs is None and c_ttl is None:
        raise SystemExit("no abstract or title column. Columns: %s" % ", ".join(map(str, cols)))

    keep = frame[frame[c_abs].notna()].reset_index(drop=True) if c_abs else frame.reset_index(drop=True)
    texts = [clean(("%s. %s" % (r[c_ttl], r[c_abs])) if c_ttl and c_abs
                   else (r[c_ttl] if c_ttl else r[c_abs]))
             for _, r in keep.iterrows()]
    years = [year_of(v) for v in keep[c_dat]] if c_dat else [None] * len(keep)
    pubs = [str(v) for v in keep[c_pub]] if c_pub else ["row%d" % i for i in range(len(keep))]

    df = Counter()
    per_doc = []
    for t in texts:
        seen = {_normalise(p) for p in _phrases(t)}
        per_doc.append(seen)
        df.update(seen)

    ranked = [(n, k) for k, n in df.items() if n >= min_patents]
    ranked.sort(key=lambda x: (-x[0], x[1]))
    vocab = [k for _, k in ranked]

    dated = sum(1 for y in years if y)
    span = [y for y in years if y]
    print("CORPUS   %d rows, %d with text, %d with a usable year%s"
          % (len(frame), len(keep), dated,
             (" (%d-%d)" % (min(span), max(span))) if span else ""))
    print("COLUMNS  title=%r abstract=%r date=%r id=%r" % (c_ttl, c_abs, c_dat, c_pub))
    if c_dat and dated == 0:
        print("WARNING  no year could be parsed from %r. Year charts are unavailable; "
              "do not estimate them." % c_dat)
    elif c_dat and dated < len(keep):
        print("NOTE     %d patents have no usable year and are left out of the year charts."
              % (len(keep) - dated))

    state = {"frame": keep, "texts": texts, "years": years, "pubs": pubs,
             "per_doc": per_doc, "df": df, "vocab": vocab,
             "columns": {"title": c_ttl, "abstract": c_abs, "date": c_dat, "pubno": c_pub}}
    keywords(state, top=top)
    return state


# `report` was the earlier name for this call and still works.
report = ingest


def keywords(state, top=40, contains=None):
    """Print the numbered keyword menu. Widen it with `top`, search `contains`."""
    df = state["df"]
    vocab = state["vocab"]
    if contains:
        needle = str(contains).lower()
        vocab = [k for k in vocab if needle in k]
    rows = []
    for i, k in enumerate(vocab[:top], start=1):
        yrs = [y for y, d in zip(state["years"], state["per_doc"]) if y and k in d]
        rows.append({"#": i, "keyword": k, "patents": df[k],
                     "first": min(yrs) if yrs else None,
                     "last": max(yrs) if yrs else None})
    table = pd.DataFrame(rows)
    state["listing"] = list(vocab[:top])
    print("\nCANDIDATE KEYWORDS  (%d shown of %d with at least 2 patents%s)"
          % (len(table), len(state["vocab"]), "" if not contains else "; filtered by %r" % contains))
    print(table.to_string(index=False))
    print("\nSELECT   reply with the number of your PRIMARY keyword (1-%d)." % len(table))
    return table


def pick(state, item):
    """Resolve a row number from the keyword menu, or a keyword name."""
    listing = state.get("listing") or state["vocab"]
    if isinstance(item, (int, np.integer)):
        n = int(item)
        if 1 <= n <= len(listing):
            return listing[n - 1]
        raise KeyError("there is no row %d; the menu has %d rows" % (n, len(listing)))
    s = _normalise(str(item).lower().strip())
    if s in state["df"]:
        return s
    hits = [k for k in state["vocab"] if s in k]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError("%r is not a candidate keyword. Try keywords(state, contains=%r)."
                       % (item, s.split()[0] if s else ""))
    raise KeyError("%r matches %d keywords: %s. Say which." % (item, len(hits), hits[:8]))


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #

def _series(state, term):
    """Per-year patent counts and raw mention counts for one keyword."""
    pat = _pattern([term])
    years, per_doc, texts = state["years"], state["per_doc"], state["texts"]
    ys = sorted({y for y in years if y})
    total = Counter(y for y in years if y)
    hits = Counter()
    mentions = Counter()
    for y, doc, txt in zip(years, per_doc, texts):
        if not y or term not in doc:
            continue
        hits[y] += 1
        mentions[y] += len(pat.findall(txt))
    return pd.DataFrame({"year": ys,
                         "patents": [hits[y] for y in ys],
                         "mentions": [mentions[y] for y in ys],
                         "corpus": [total[y] for y in ys],
                         "share": [hits[y] / total[y] if total[y] else 0.0 for y in ys]})


def _pair_series(state, a, b):
    years, per_doc = state["years"], state["per_doc"]
    ys = sorted({y for y in years if y})
    total, ca, cb, both = Counter(y for y in years if y), Counter(), Counter(), Counter()
    for y, doc in zip(years, per_doc):
        if not y:
            continue
        ina, inb = a in doc, b in doc
        ca[y] += ina
        cb[y] += inb
        both[y] += (ina and inb)
    out = pd.DataFrame({"year": ys,
                        "a_only": [ca[y] - both[y] for y in ys],
                        "b_only": [cb[y] - both[y] for y in ys],
                        "both": [both[y] for y in ys],
                        "a_total": [ca[y] for y in ys],
                        "b_total": [cb[y] for y in ys],
                        "corpus": [total[y] for y in ys]})
    # What independence would put in the same patent, year by year.
    out["expected"] = [(ca[y] * cb[y] / total[y]) if total[y] else 0.0 for y in ys]
    return out


def _partner_table(state, term, top=15, min_patents=2):
    """Keywords sharing patents with this one. Silent - `primary` prints it."""
    per_doc = state["per_doc"]
    n = len(per_doc)
    df = state["df"]
    base = df[term]
    together = Counter()
    for doc in per_doc:
        if term in doc:
            together.update(doc)
    rows = []
    for other, c in together.items():
        if other == term or c < min_patents:
            continue
        expected = base * df[other] / n
        rows.append({"keyword": other, "both": c, "its patents": df[other],
                     "lift": round(c / expected, 2) if expected else 0.0})
    if not rows:
        return pd.DataFrame()
    # The keyword is the last sort key, and the sort is stable, because this
    # menu is re-printed from scratch in a later turn and row 9 has to be the
    # same keyword both times. Lift ties are common in a small corpus.
    out = (pd.DataFrame(rows)
           .sort_values(["both", "lift", "keyword"],
                        ascending=[False, False, True], kind="mergesort")
           .head(top).reset_index(drop=True))
    out.insert(0, "#", range(1, len(out) + 1))
    return out


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=12, color=INK, pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.tick_params(colors=INK, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _finish(fig, png):
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


def _int_axis(ax):
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def _year_axis(ax, years):
    """Years are labels, not a continuous quantity - no 2007.5 on the axis."""
    years = list(years)
    step = max(1, int(math.ceil(len(years) / 12)))
    ticks = years[::step]
    if years and years[-1] not in ticks:
        ticks.append(years[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(y)) for y in ticks],
                       rotation=45 if len(ticks) > 8 else 0,
                       ha="right" if len(ticks) > 8 else "center", fontsize=8.5)


def _figure(w=7.6, h=4.3):
    fig, ax = plt.subplots(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig, ax


# --------------------------------------------------------------------------- #
# step 2 - the primary keyword: three graphs, then the secondary menu
# --------------------------------------------------------------------------- #

def primary(state, choice, top=15, png_prefix=None, html=None):
    """Three graphs for the keyword the analyst chose, then the second menu.

    GRAPH 1  occurrences by year - how many patents filed in each year mention
             the keyword. Counts patents, not mentions.
    GRAPH 2  cumulative occurrences - the running total of graph 1. It can only
             rise, so it says how much has accumulated and never that interest
             is falling.
    GRAPH 3  share of the year's filings - the same count divided by how many
             patents that year holds, drawn over the corpus itself. A year with
             more patents in it lifts every keyword at once; this is the panel
             that separates ground actually gained from a bigger denominator.

    Then prints the numbered secondary menu. That menu is a deterministic
    function of this choice, so `primary(state, 6)` re-prints an identical menu
    in a later turn, after the sandbox has been wiped.
    """
    term = pick(state, choice)
    s = _series(state, term)
    if s.empty or s["patents"].sum() == 0:
        print("%r appears in no patent with a usable year." % term)
        return {"keyword": term, "series": s, "partners": pd.DataFrame()}

    state["primary"] = term
    live = s[s["patents"] > 0]
    tot = int(s["patents"].sum())
    men = int(s["mentions"].sum())
    peak = s.loc[s["patents"].idxmax()]

    print("PRIMARY KEYWORD  %r  -  %d patents, %d mentions, %d-%d"
          % (term, tot, men, int(live["year"].min()), int(live["year"].max())))

    # -- graph 1: occurrences by year ---------------------------------------
    fig, ax = _figure()
    ax.bar(s["year"], s["patents"], color=BLUE, width=0.72)
    for x, y in zip(s["year"], s["patents"]):
        if y:
            ax.annotate(str(int(y)), (x, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8, color=INK)
    _style(ax, "1. %s - occurrences by year" % term,
           "application year", "patents mentioning it")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    ax.set_ylim(0, max(s["patents"].max() * 1.18, 1))
    _finish(fig, (png_prefix + "_1_by_year.png") if png_prefix else None)

    # -- graph 2: cumulative -------------------------------------------------
    cum = s["patents"].cumsum()
    fig, ax = _figure()
    ax.plot(s["year"], cum, color=GREEN, linewidth=2.4, marker="o", markersize=4.5)
    ax.fill_between(s["year"], cum, color=GREEN, alpha=0.14)
    ax.annotate("%d" % int(cum.iloc[-1]), (s["year"].iloc[-1], cum.iloc[-1]),
                textcoords="offset points", xytext=(-4, 6), ha="right",
                fontsize=9, color=GREEN)
    _style(ax, "2. %s - cumulative occurrences (never falls)" % term,
           "application year", "patents to date")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    _finish(fig, (png_prefix + "_2_cumulative.png") if png_prefix else None)

    # -- graph 3: share of the year -----------------------------------------
    fig, ax = _figure()
    ax.bar(s["year"], s["corpus"], color=FAINT, width=0.72,
           label="all patents filed that year")
    _style(ax, "3. %s - share of the year's filings" % term,
           "application year", "patents filed that year")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    axr = ax.twinx()
    axr.plot(s["year"], s["share"] * 100, color=ORANGE, linewidth=2.2,
             marker="o", markersize=4, label="share mentioning it")
    axr.set_ylabel("%% of that year mentioning %s" % term, fontsize=9.5, color=ORANGE)
    axr.tick_params(axis="y", colors=ORANGE, labelsize=9)
    axr.spines["top"].set_visible(False)
    axr.set_ylim(0, max(s["share"].max() * 130, 1))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, frameon=False, loc="upper left")
    _finish(fig, (png_prefix + "_3_share.png") if png_prefix else None)

    best = s.loc[s["share"].idxmax()]
    print("%s: %d patents, %.0f%% of the dated corpus. Busiest year %d with %d; "
          "highest share %d at %.0f%% of that year's filings."
          % (term, tot, 100 * tot / max(s["corpus"].sum(), 1),
             int(peak["year"]), int(peak["patents"]),
             int(best["year"]), 100 * best["share"]))

    if html and HAVE_PLOTLY:
        f = go.Figure()
        f.add_bar(x=s["year"], y=s["patents"], name="patents per year", marker_color=BLUE)
        f.add_scatter(x=s["year"], y=cum, name="cumulative", mode="lines+markers",
                      line=dict(color=GREEN), yaxis="y2")
        f.update_layout(template="plotly_white", title="%s over time" % term,
                        xaxis_title="application year", yaxis_title="patents per year",
                        yaxis2=dict(title="cumulative", overlaying="y", side="right"))
        _write_html(f, html)

    part = _partner_table(state, term, top=top)
    state["partner_listing"] = list(part["keyword"]) if not part.empty else []
    if part.empty:
        print("\n%r shares no patent with another candidate keyword, so there is "
              "no pair to draw." % term)
    else:
        print("\nSECONDARY KEYWORDS  (keywords that share patents with %r)" % term)
        print(part.to_string(index=False))
        print("\n'both' is how many patents carry the pair. 'lift' is that count "
              "divided by what\nchance alone would give, so a merely common word "
              "does not come top.")
        print("\nSELECT   reply with the number of your SECONDARY keyword (1-%d)."
              % len(part))
    return {"keyword": term, "series": s, "partners": part}


# --------------------------------------------------------------------------- #
# step 3 - the pair: three graphs
# --------------------------------------------------------------------------- #

def secondary(state, choice_primary, choice_secondary, top=15, png_prefix=None,
              html=None):
    """Three graphs for the pair the analyst chose.

    Takes BOTH numbers, so the whole session replays from two integers after
    the sandbox has been wiped: the first indexes the keyword menu, the second
    the secondary menu that `primary(state, first)` prints. It rebuilds that
    second menu itself, so a turn can go straight from `ingest` to here without
    redrawing the three graphs the analyst has already seen.

    GRAPH 1  the pair by year - patents carrying both, with the patents that
             carry only one stacked above for context.
    GRAPH 2  cumulative - the running totals of each keyword and of the pair.
    GRAPH 3  together against chance - the same co-occurrence bars with the
             count independence would predict for that year drawn over them.
             Above the line the two travel together; on it they are unrelated
             words that happen to share a corpus.
    """
    a = pick(state, choice_primary)
    if isinstance(choice_secondary, (int, np.integer)):
        # Rebuilt here rather than read from `primary`, so this call stands on
        # its own after the sandbox has been wiped: ingest, then this.
        listing = list(_partner_table(state, a, top=top).get("keyword", []))
        n = int(choice_secondary)
        if not 1 <= n <= len(listing):
            raise KeyError("there is no secondary %d; the menu for %r has %d rows"
                           % (n, a, len(listing)))
        b = listing[n - 1]
    else:
        b = pick(state, choice_secondary)
    if a == b:
        print("those are the same keyword (%r)." % a)
        return pd.DataFrame()

    s = _pair_series(state, a, b)
    if s.empty:
        print("no usable years in this corpus.")
        return s
    n_both = int(s["both"].sum())
    na, nb = int(s["a_total"].sum()), int(s["b_total"].sum())
    union = na + nb - n_both
    exp = na * nb / max(s["corpus"].sum(), 1)

    print("PAIR   %r + %r  -  %d patents carry both" % (a, b, n_both))

    # -- graph 1: the pair by year ------------------------------------------
    fig, ax = _figure(8.2, 4.5)
    w = 0.72
    ax.bar(s["year"], s["both"], width=w, color=PURPLE, label="both")
    ax.bar(s["year"], s["a_only"], width=w, bottom=s["both"], color=BLUE,
           label="%s only" % a)
    ax.bar(s["year"], s["b_only"], width=w, bottom=s["both"] + s["a_only"],
           color=ORANGE, label="%s only" % b)
    for x, y in zip(s["year"], s["both"]):
        if y:
            ax.annotate(str(int(y)), (x, y / 2.0), ha="center", va="center",
                        fontsize=8, color="white")
    _style(ax, "1. %s + %s - patents per year" % (a, b),
           "application year", "patents")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    ax.legend(fontsize=8.5, frameon=False)
    _finish(fig, (png_prefix + "_1_pair_by_year.png") if png_prefix else None)

    # -- graph 2: cumulative -------------------------------------------------
    fig, ax = _figure(8.2, 4.5)
    ax.plot(s["year"], s["a_total"].cumsum(), color=BLUE, linewidth=2.0, label=a)
    ax.plot(s["year"], s["b_total"].cumsum(), color=ORANGE, linewidth=2.0, label=b)
    ax.plot(s["year"], s["both"].cumsum(), color=PURPLE, linewidth=2.6,
            marker="o", markersize=4.5, label="both")
    ax.fill_between(s["year"], s["both"].cumsum(), color=PURPLE, alpha=0.14)
    _style(ax, "2. %s + %s - cumulative (never falls)" % (a, b),
           "application year", "patents to date")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    ax.legend(fontsize=8.5, frameon=False)
    _finish(fig, (png_prefix + "_2_pair_cumulative.png") if png_prefix else None)

    # -- graph 3: together against chance ------------------------------------
    fig, ax = _figure(8.2, 4.5)
    ax.bar(s["year"], s["both"], width=0.72, color=PURPLE, label="observed together")
    ax.plot(s["year"], s["expected"], color=INK, linewidth=1.8, linestyle="--",
            marker="o", markersize=3.5, label="expected if unrelated")
    _style(ax, "3. %s + %s - together against chance" % (a, b),
           "application year", "patents carrying both")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    ax.legend(fontsize=8.5, frameon=False)
    _finish(fig, (png_prefix + "_3_pair_vs_chance.png") if png_prefix else None)

    print("%s: %d patents. %s: %d patents. Both: %d." % (a, na, b, nb, n_both))
    if n_both:
        yrs = s.loc[s["both"] > 0, "year"]
        print("First together in %d, last in %d. They overlap on %.0f%% of the "
              "patents that mention either."
              % (int(yrs.min()), int(yrs.max()),
                 100 * n_both / union if union else 0))
        print("Chance alone would put them together in about %.1f patents; "
              "observed %d, a lift of %.2f."
              % (exp, n_both, n_both / exp if exp else 0.0))
    else:
        print("They never appear in the same patent.")
    if n_both < 5:
        print("Fewer than five patents carry both, so the per-year shape of the "
              "co-occurrence is noise. Say so rather than narrating the bars.")

    if html and HAVE_PLOTLY:
        f = go.Figure()
        f.add_bar(x=s["year"], y=s["both"], name="both", marker_color=PURPLE)
        f.add_bar(x=s["year"], y=s["a_only"], name="%s only" % a, marker_color=BLUE)
        f.add_bar(x=s["year"], y=s["b_only"], name="%s only" % b, marker_color=ORANGE)
        f.update_layout(barmode="stack", template="plotly_white",
                        title="%s and %s per year" % (a, b),
                        xaxis_title="application year", yaxis_title="patents")
        _write_html(f, html)
    return s


def documents_for(state, keyword, other=None, limit=15):
    """The patents behind a bar, so a claim can be checked rather than trusted."""
    a = pick(state, keyword)
    b = pick(state, other) if other else None
    cols = state["columns"]
    idx = [i for i, doc in enumerate(state["per_doc"])
           if a in doc and (b is None or b in doc)]
    frame = state["frame"].iloc[idx[:limit]].copy()
    frame.insert(0, "year", [state["years"][i] for i in idx[:limit]])
    want = ["year"] + [c for c in (cols["pubno"], cols["title"], cols["date"]) if c]
    print("%d patents contain %s%s; showing %d."
          % (len(idx), repr(a), (" and " + repr(b)) if b else "", min(limit, len(idx))))
    return frame[want]


def _write_html(fig, path):
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote %s (%.0f KB, open it to hover and zoom)" % (path, len(html) / 1024))
