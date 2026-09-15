"""trt-pb in a chat sandbox: technical terms, TRT triples, and the two graphs.

A replica of the original Streamlit tool's pipeline, rebuilt for an environment
that has pandas, numpy and matplotlib and nothing else - no spaCy, no
transformers, no sklearn.

    state = load("export.csv")   -> corpus summary + numbered TECHNICAL TERMS
    trt(state, 6)                -> term 6: occurrences by year + cumulative,
                                    then its relationship types and its
                                    secondary terms, both numbered
    pair(state, 6, 3)            -> the two together: by year + cumulative
    triples(state, 6, 3)         -> the T1-preposition-T2 rows behind the bars

What the original did, and what this does instead
-------------------------------------------------
* Candidate terms came from `KeyphraseTfidfVectorizer`, which is a spaCy
  part-of-speech chunker: it keeps `<J.*>*<N.*>+`, adjectives followed by
  nouns, and nothing else. Then `Keyphrases()` threw away everything shorter
  than two words. That pair of filters is why the original never showed you
  "operating", "capable" or "providing" - they are not noun phrases.
  Here the chunker is rule-based: function words break a chunk, a chunk must
  end in a noun-like head, and a term must be at least two words long.
* Terms were ranked by TF-IDF and then re-picked per abstract by PatentBERT.
  Here they are ranked by TF-IDF summed over the corpus, with the number of
  patents shown alongside.
* Relationships came from spaCy's ADP tagging: every preposition in a sentence
  splits it into T1 - preposition - T2. Prepositions are a closed class, so the
  same split is done here from a fixed list. The buckets are the original's,
  unchanged:
      Inclusion  of in with from on at within includes by utilizes
      Objective  for
      Effect     to across against
      Process    during into through via
      Likeness   as
      Misc       everything else
* Secondary terms came from a PatentBERT masked-language-model ranking, at
  least five per primary. There is no BERT here, so they are ranked by how
  often they are linked to the primary in a TRT triple, and then by lift over
  what chance alone would give. At least five are always returned.

Counting rules, stated once because every number depends on them
----------------------------------------------------------------
* A term occurs in a patent when it appears in the title or abstract as a whole
  word. Matching is word-bounded and longest-first, so "fuel cell" inside
  "fuel cell stack" is attributed to the longer term.
* Bars count PATENTS, not mentions: a patent saying "fuel cell" nine times
  counts once.
* Two terms co-occur when both appear in the same patent. With a relationship
  filter they must also be the two sides of a TRT triple carrying that
  relationship.
* The year is the application year. Undated patents are dropped from the year
  charts and the count is printed.
"""

from __future__ import annotations

import datetime
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
# Backend left as the host set it: forcing Agg would save files and show
# nothing inline.

DPI = 130
INK = "#1a1a1a"
GRID = "#dcdcdc"
FAINT = "#c9d4dc"
BLUE = "#0b6fa4"
ORANGE = "#d1600a"
GREEN = "#3f8f29"
PURPLE = "#8d4bbb"


# --------------------------------------------------------------------------- #
# columns and dates
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
    """The original's PatentInfo.clean_text, minus the hyphen removal.

    The original replaced "-" with a space, which split "hydrogen-containing"
    into two tokens. Hyphenated compounds are kept whole here.
    """
    if not isinstance(text, str):
        return ""
    out = re.sub(r"\([^)]*\)\s*\n", "", text)
    out = _JP.sub("", out)
    out = out.replace(";", ".").replace("–", "-").replace("—", "-")
    return _WS.sub(" ", out).strip()


# --------------------------------------------------------------------------- #
# the part-of-speech stand-in
# --------------------------------------------------------------------------- #
# spaCy is not available, so noun-phrase chunking is done with three closed
# lists and two suffix rules. The lists only have to be right about function
# words and common patent verbs; anything unknown is treated as a noun, which
# is the correct default in technical prose.

FUNCTION = frozenset("""
a an the this that these those said such its their his her our your my
some any each one two three four five six seven eight nine ten first second
third another other same both all every either neither no not
and or but nor so yet if then than because while although though whereas
whether when where why how what which who whom whose
of in on at by to from with without within into onto upon through during
before after above below under over between among across against about
around near off out up down since until toward towards beneath behind
beside beyond despite except inside outside past underneath via like as for
is are was were be been being am do does did doing done have has had having
can could may might must shall should will would ought need
it they them he she we you i there here also further still thus hence
therefore however moreover per respectively
present herein hereof hereinafter thereof therein thereto therewith
wherein whereby according aforementioned aforesaid
preferably optionally substantially approximately essentially particularly
typically generally
""".split())

# Verbs that turn up constantly in patent prose. A chunk may not end in one.
VERB = frozenset("""
comprise comprises comprising comprised include includes including included
consist consists consisting consisted provide provides providing provided
disclose discloses disclosing disclosed relate relates relating related
use uses using used utilize utilizes utilizing utilized
form forms forming formed make makes making made produce produces producing
obtain obtains obtaining obtained generate generates generating generated
perform performs performing performed operate operates operating operated
connect connects connecting connected couple couples coupling coupled
arrange arranges arranging arranged dispose disposes disposing disposed
configure configures configuring configured adapt adapts adapting adapted
define defines defining defined describe describes describing described
select selects selecting selected determine determines determining determined
control controls controlling controlled reduce reduces reducing reduced
increase increases increasing increased improve improves improving improved
allow allows allowing allowed enable enables enabling enabled
receive receives receiving received transmit transmits transmitting transmitted
move moves moving moved rotate rotates rotating rotated
apply applies applying applied treat treats treating treated
prevent prevents preventing prevented cause causes causing caused
extend extends extending extended
""".split())

# -ing words that really are nouns in this register. A chunk may end in one.
NOUN_ING = frozenset("""
coating housing bearing casing tubing wiring cladding packing mounting
opening spacing loading heating cooling processing sensing switching
shielding damping doping etching bonding sintering sealing mixing milling
grinding welding printing folding winding casting moulding molding
engineering monitoring reforming cracking scrubbing venting purging
insulating conditioning filtering screening
""".split())

# A term made only of these names a category, not a technology.
GENERIC = frozenset("""
apparatus method methods system systems device devices assembly assemblies
arrangement arrangements means unit units mechanism mechanisms structure
structures module modules equipment machine machines process processes
technique techniques procedure procedures application applications step steps
portion portions member members element elements part parts side sides end
ends surface surfaces embodiment embodiments invention aspect aspects
example examples figure figures type types kind kinds
invention disclosure claim claims description summary field background
drawing drawings
""".split())

# Participles that are pure drafting language. Unlike "coated" or "activated"
# they never modify a technology, so they are barred even in front of a head:
# without this, "stream comprising molecular hydrogen" becomes a term.
BOILER_PARTICIPLE = frozenset("""
comprising comprised including included consisting consisted containing
contained having disclosing disclosed describing described defining defined
claiming claimed mentioning mentioned relating related provided providing
said known given above-mentioned
""".split())

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'’][A-Za-z0-9]+)*")
_SENT = re.compile(r"[.!?]+")


def _is_head_noun(word):
    """Can this word be the head of a technical term?"""
    if len(word) < 3 or word in FUNCTION or word in VERB:
        return False
    if word.endswith("ly"):
        return False
    if word.endswith("ing"):
        return word in NOUN_ING
    if word.endswith("ed"):
        return False
    return True


def _is_modifier(word):
    """Can this word sit in front of the head?

    Adjectival participles are the point of this being separate from
    `_is_head_noun`: "activated" cannot end a term but "activated carbon" is
    exactly the kind of term the tool exists to find.
    """
    if len(word) < 2 or word in FUNCTION or word in BOILER_PARTICIPLE:
        return False
    if word in VERB:
        # "coated separator" is a term; "comprising separator" is not. Only
        # participles survive, and only in front of a head.
        return word.endswith(("ed", "ing"))
    return not word.endswith("ly")


def chunks(text, min_words=2, max_words=4):
    """Noun-phrase chunks, the stand-in for `<J.*>*<N.*>+`.

    Walks each sentence collecting runs of modifier-or-noun tokens, then trims
    the run back to its last noun-like token - that trailing noun is the head.
    A run longer than `max_words` is taken from the right, because the head and
    the words nearest it carry the meaning ("solid oxide fuel cell stack" ->
    "oxide fuel cell stack").
    """
    out = []
    for sentence in _SENT.split(text.lower()):
        for piece in re.split(r"[,:;()\[\]/\"]", sentence):
            run = []
            for token in _TOKEN.findall(piece) + [""]:
                if token and (_is_modifier(token) or _is_head_noun(token)):
                    run.append(token)
                    continue
                while run and not _is_head_noun(run[-1]):
                    run.pop()
                if len(run) >= min_words:
                    take = run[-max_words:] if len(run) > max_words else run
                    if not all(w in GENERIC for w in take):
                        out.append(" ".join(take))
                run = []
    return out


def _singular(word):
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"),
                     ("shes", "sh"), ("xes", "x")):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            return word[:-len(suf)] + rep
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _normalise(phrase):
    words = phrase.split()
    return " ".join(words[:-1] + [_singular(words[-1])])


def _inflected(term):
    """The term and the plural it was normalised from."""
    esc = re.escape(term)
    forms = [esc + "(?:e?s)?"]
    if term.endswith("y") and len(term) > 2:
        forms.append(re.escape(term[:-1]) + "ies")
    return "(?:" + "|".join(forms) + ")"


def _pattern(terms):
    """The original's KeywordMapping.build_keyword_pattern, with inflection.

    Longest-first so a multi-word term wins over its own substring, escaped so
    a bracket in a phrase cannot become a capture group, and the alternation
    grouped so `\\b` anchors every branch and not just the ends.
    """
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not terms:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:" + "|".join(_inflected(t) for t in terms) + r")\b",
                      re.IGNORECASE)


# --------------------------------------------------------------------------- #
# TRT triples
# --------------------------------------------------------------------------- #

PREPOSITIONS = frozenset("""
of in with from on at within by for to across against during into through via
as about above after along among around before behind below beneath beside
between beyond despite down except inside near off onto out outside over past
since toward towards under underneath until up upon without
includes utilizes
""".split())

INCLUSION = frozenset("of in with from on at within includes by utilizes".split())
OBJECTIVE = frozenset("for".split())
EFFECT = frozenset("to across against".split())
PROCESS = frozenset("during into through via".split())
LIKENESS = frozenset("as".split())

RELATIONS = ("Inclusion", "Objective", "Effect", "Process", "Likeness", "Misc")


def _relation(prep):
    if prep in INCLUSION:
        return "Inclusion"
    if prep in OBJECTIVE:
        return "Objective"
    if prep in EFFECT:
        return "Effect"
    if prep in PROCESS:
        return "Process"
    if prep in LIKENESS:
        return "Likeness"
    return "Misc"


def _triples(text):
    """T1 - preposition - T2 for every preposition, the original's get_trt.

    T1 runs back to the previous preposition and T2 forward to the next, so a
    sentence with three prepositions yields three overlapping triples.
    """
    out = []
    for sentence in _SENT.split(text.lower()):
        toks = _TOKEN.findall(sentence)
        marks = [i for i, t in enumerate(toks) if t in PREPOSITIONS]
        if not marks:
            continue
        for pos, i in enumerate(marks):
            left = marks[pos - 1] + 1 if pos else 0
            right = marks[pos + 1] if pos + 1 < len(marks) else len(toks)
            t1 = " ".join(toks[left:i])
            t2 = " ".join(toks[i + 1:right])
            if t1 and t2:
                out.append((t1, toks[i], t2))
    return out


# --------------------------------------------------------------------------- #
# step 1 - load
# --------------------------------------------------------------------------- #

def load(path, top=40, min_patents=2, min_words=2, max_words=4,
         abstract=None, date=None, title=None):
    """Read the export, extract technical terms, print the numbered menu."""
    frame = (pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls"))
             else pd.read_csv(path))
    cols = list(frame.columns)
    c_abs = _pick(cols, HINTS["abstract"], abstract)
    c_ttl = _pick(cols, HINTS["title"], title)
    c_dat = _pick(cols, HINTS["date"], date)
    c_pub = _pick(cols, HINTS["pubno"])
    if c_abs is None and c_ttl is None:
        raise SystemExit("no abstract or title column. Columns: %s"
                         % ", ".join(map(str, cols)))

    keep = (frame[frame[c_abs].notna()].reset_index(drop=True) if c_abs
            else frame.reset_index(drop=True))
    texts = [clean(("%s. %s" % (r[c_ttl], r[c_abs])) if c_ttl and c_abs
                   else (r[c_ttl] if c_ttl else r[c_abs]))
             for _, r in keep.iterrows()]
    years = [year_of(v) for v in keep[c_dat]] if c_dat else [None] * len(keep)
    pubs = ([str(v) for v in keep[c_pub]] if c_pub
            else ["row%d" % i for i in range(len(keep))])

    # Document frequency and term frequency over the chunked candidates.
    df = Counter()
    tf = Counter()
    per_doc = []
    for t in texts:
        found = [_normalise(c) for c in chunks(t, min_words, max_words)]
        seen = set(found)
        per_doc.append(seen)
        df.update(seen)
        tf.update(found)

    n = max(len(texts), 1)
    # TF-IDF the way the original's TfidfVectorizer summed it: the corpus-wide
    # term frequency against the log-damped document frequency.
    score = {k: tf[k] * math.log(n / (1 + df[k])) for k in df}

    ranked = [k for k, c in df.items() if c >= min_patents]
    ranked.sort(key=lambda k: (-df[k], -score[k], k))

    dated = sum(1 for y in years if y)
    span = [y for y in years if y]
    print("CORPUS   %d rows, %d with text, %d with a usable year%s"
          % (len(frame), len(keep), dated,
             (" (%d-%d)" % (min(span), max(span))) if span else ""))
    print("COLUMNS  title=%r abstract=%r date=%r id=%r"
          % (c_ttl, c_abs, c_dat, c_pub))
    print("TERMS    %d noun phrases of %d-%d words in at least %d patents"
          % (len(ranked), min_words, max_words, min_patents))
    if c_dat and dated < len(keep):
        print("NOTE     %d patents have no usable year and are left out of the "
              "year charts." % (len(keep) - dated))

    state = {"frame": keep, "texts": texts, "years": years, "pubs": pubs,
             "per_doc": per_doc, "df": df, "score": score, "vocab": ranked,
             "columns": {"title": c_ttl, "abstract": c_abs,
                         "date": c_dat, "pubno": c_pub},
             "triples": None}
    terms(state, top=top)
    return state


def terms(state, top=40, contains=None):
    """Print the numbered technical-term menu. Widen with `top`, search with
    `contains`."""
    df, score = state["df"], state["score"]
    vocab = state["vocab"]
    if contains:
        needle = str(contains).lower()
        vocab = [k for k in vocab if needle in k]
    rows = []
    for i, k in enumerate(vocab[:top], start=1):
        yrs = [y for y, d in zip(state["years"], state["per_doc"]) if y and k in d]
        rows.append({"#": i, "technical term": k, "patents": df[k],
                     "tfidf": round(score[k], 1),
                     "first": min(yrs) if yrs else None,
                     "last": max(yrs) if yrs else None})
    table = pd.DataFrame(rows)
    state["listing"] = list(vocab[:top])
    print("\nTECHNICAL TERMS  (%d shown of %d%s)"
          % (len(table), len(state["vocab"]),
             "" if not contains else "; filtered by %r" % contains))
    print(table.to_string(index=False))
    print("\nSELECT   reply with the number of your PRIMARY term (1-%d)." % len(table))
    return table


def pick(state, item):
    """Resolve a row number from the term menu, or a term name."""
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
        raise KeyError("%r is not a technical term in this corpus. "
                       "Try terms(state, contains=%r)."
                       % (item, s.split()[0] if s else ""))
    raise KeyError("%r matches %d terms: %s. Say which." % (item, len(hits), hits[:8]))


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #

def _series(state, term):
    pat = _pattern([term])
    years, per_doc, texts = state["years"], state["per_doc"], state["texts"]
    ys = sorted({y for y in years if y})
    total = Counter(y for y in years if y)
    hits, mentions = Counter(), Counter()
    for y, doc, txt in zip(years, per_doc, texts):
        if not y or term not in doc:
            continue
        hits[y] += 1
        mentions[y] += len(pat.findall(txt))
    return pd.DataFrame({"year": ys,
                         "patents": [hits[y] for y in ys],
                         "mentions": [mentions[y] for y in ys],
                         "corpus": [total[y] for y in ys],
                         "share": [hits[y] / total[y] if total[y] else 0.0
                                   for y in ys]})


def _relation_index(state):
    """Every TRT triple whose two sides both map to a known technical term.

    Built once and cached on the state: this is the expensive step, a regex
    pass over both sides of every triple in the corpus.
    """
    if state["triples"] is not None:
        return state["triples"]
    pat = _pattern(state["vocab"])
    rows = []
    for i, (txt, pub, yr) in enumerate(zip(state["texts"], state["pubs"],
                                           state["years"])):
        for t1, prep, t2 in _triples(txt):
            left = {_normalise(m.lower()) for m in pat.findall(t1)}
            right = {_normalise(m.lower()) for m in pat.findall(t2)}
            if not left or not right:
                continue
            rel = _relation(prep)
            for a in left:
                for b in right:
                    if a != b:
                        rows.append((i, pub, yr, a, prep, rel, b))
    out = pd.DataFrame(rows, columns=["doc", "pubno", "year", "T1",
                                      "prep", "relation", "T2"])
    state["triples"] = out
    return out


def _partners(state, term, top=15, min_keys=5):
    """Terms linked to this one, ranked by shared patents then lift.

    The original ranked these with a PatentBERT masked-language-model pass and
    guaranteed at least `min_keys`. Without BERT the ranking is co-occurrence
    with a lift correction, and the guarantee is kept.
    """
    per_doc, df = state["per_doc"], state["df"]
    n = len(per_doc)
    base = df[term]
    together = Counter()
    for doc in per_doc:
        if term in doc:
            together.update(doc)

    rel = _relation_index(state)
    linked = rel[(rel["T1"] == term) | (rel["T2"] == term)]
    link_count = Counter()
    for _, r in linked.iterrows():
        other = r["T2"] if r["T1"] == term else r["T1"]
        link_count[other] += 1

    rows = []
    for other, c in together.items():
        if other == term:
            continue
        expected = base * df[other] / n
        rows.append({"technical term": other, "both": c,
                     "its patents": df[other],
                     "lift": round(c / expected, 2) if expected else 0.0,
                     "triples": link_count.get(other, 0)})
    if not rows:
        return pd.DataFrame()
    # Stable, name-broken ordering: this menu is re-printed in a later turn and
    # row 3 has to be the same term both times.
    out = (pd.DataFrame(rows)
           .sort_values(["both", "lift", "technical term"],
                        ascending=[False, False, True], kind="mergesort")
           .reset_index(drop=True))
    if len(out) < min_keys:
        keep = out
    else:
        keep = out.head(max(top, min_keys))
    keep = keep.reset_index(drop=True)
    keep.insert(0, "#", range(1, len(keep) + 1))
    return keep


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
# step 2 - the primary term
# --------------------------------------------------------------------------- #

def trt(state, choice, top=15, min_keys=5, png_prefix=None):
    """Two graphs for the chosen term, then its relationships and partners.

    GRAPH 1  occurrences by year - patents filed that year mentioning the term.
    GRAPH 2  cumulative occurrences - the running total, which only ever rises.

    Then two numbered menus: the relationship types this term takes part in,
    and the secondary terms it is linked to.
    """
    term = pick(state, choice)
    s = _series(state, term)
    if s.empty or s["patents"].sum() == 0:
        print("%r appears in no patent with a usable year." % term)
        return {"term": term, "series": s, "partners": pd.DataFrame()}

    state["primary"] = term
    live = s[s["patents"] > 0]
    tot = int(s["patents"].sum())
    peak = s.loc[s["patents"].idxmax()]
    print("PRIMARY TERM  %r  -  %d patents, %d mentions, %d-%d"
          % (term, tot, int(s["mentions"].sum()),
             int(live["year"].min()), int(live["year"].max())))

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

    print("%s: %d patents, %.0f%% of the dated corpus. Busiest year %d with %d."
          % (term, tot, 100 * tot / max(s["corpus"].sum(), 1),
             int(peak["year"]), int(peak["patents"])))

    rel = _relation_index(state)
    mine = rel[(rel["T1"] == term) | (rel["T2"] == term)]
    counts = Counter(mine["relation"])
    rel_rows = [{"relationship": r, "triples": counts.get(r, 0),
                 "patents": mine[mine["relation"] == r]["doc"].nunique()}
                for r in RELATIONS if counts.get(r, 0)]
    state["relations"] = [r["relationship"] for r in rel_rows]
    if rel_rows:
        print("\nRELATIONSHIPS OF %r  (from T1-preposition-T2 triples)" % term)
        print(pd.DataFrame(rel_rows).to_string(index=False))
        print("Inclusion = of in with from on at within includes by utilizes; "
              "Objective = for;\nEffect = to across against; "
              "Process = during into through via; Likeness = as.")
    else:
        print("\n%r is in no TRT triple with another technical term; "
              "pairing falls back to plain co-occurrence." % term)

    part = _partners(state, term, top=top, min_keys=min_keys)
    state["partner_listing"] = list(part["technical term"]) if not part.empty else []
    if part.empty:
        print("\n%r shares no patent with another technical term." % term)
    else:
        print("\nSECONDARY TERMS  (linked to %r)" % term)
        print(part.to_string(index=False))
        print("\n'both' is how many patents carry the pair; 'lift' is that count "
              "over what chance\nalone would give; 'triples' is how many TRT "
              "triples link the two directly.")
        print("\nSELECT   reply with the number of your SECONDARY term (1-%d), "
              "and optionally\n         a relationship, e.g. \"3 Inclusion\"."
              % len(part))
    return {"term": term, "series": s, "partners": part}


# --------------------------------------------------------------------------- #
# step 3 - the pair
# --------------------------------------------------------------------------- #

def _pair_series(state, a, b, relation=None):
    years, per_doc = state["years"], state["per_doc"]
    ys = sorted({y for y in years if y})
    total = Counter(y for y in years if y)

    if relation:
        rel = _relation_index(state)
        m = rel[(((rel["T1"] == a) & (rel["T2"] == b)) |
                 ((rel["T1"] == b) & (rel["T2"] == a)))]
        m = m[m["relation"].str.lower() == str(relation).lower()]
        linked_docs = set(m["doc"])
    else:
        linked_docs = None

    ca, cb, both = Counter(), Counter(), Counter()
    for i, (y, doc) in enumerate(zip(years, per_doc)):
        if not y:
            continue
        ina, inb = a in doc, b in doc
        ca[y] += ina
        cb[y] += inb
        if ina and inb and (linked_docs is None or i in linked_docs):
            both[y] += 1
    out = pd.DataFrame({"year": ys,
                        "a_only": [ca[y] - both[y] for y in ys],
                        "b_only": [cb[y] - both[y] for y in ys],
                        "both": [both[y] for y in ys],
                        "a_total": [ca[y] for y in ys],
                        "b_total": [cb[y] for y in ys],
                        "corpus": [total[y] for y in ys]})
    return out


def pair(state, choice_primary, choice_secondary, relation=None,
         top=15, min_keys=5, png_prefix=None):
    """Two graphs for the pair: together by year, and cumulative.

    Takes both numbers so the whole session replays from two integers after the
    sandbox is wiped. `relation` restricts a co-occurrence to patents where the
    two terms are actually the two sides of a TRT triple of that type -
    "Inclusion", "Objective", "Effect", "Process", "Likeness" or "Misc".
    """
    a = pick(state, choice_primary)
    if isinstance(choice_secondary, (int, np.integer)):
        listing = list(_partners(state, a, top=top,
                                 min_keys=min_keys).get("technical term", []))
        n = int(choice_secondary)
        if not 1 <= n <= len(listing):
            raise KeyError("there is no secondary %d; the menu for %r has %d rows"
                           % (n, a, len(listing)))
        b = listing[n - 1]
    else:
        b = pick(state, choice_secondary)
    if a == b:
        print("those are the same term (%r)." % a)
        return pd.DataFrame()

    s = _pair_series(state, a, b, relation)
    n_both = int(s["both"].sum())
    na, nb = int(s["a_total"].sum()), int(s["b_total"].sum())
    union = na + nb - n_both
    exp = na * nb / max(s["corpus"].sum(), 1)

    label = "%r + %r" % (a, b)
    if relation:
        label += " via %s" % relation
    print("PAIR   %s  -  %d patents carry both" % (label, n_both))

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
    _style(ax, "1. %s - patents per year" % label, "application year", "patents")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    ax.legend(fontsize=8.5, frameon=False)
    _finish(fig, (png_prefix + "_1_pair_by_year.png") if png_prefix else None)

    fig, ax = _figure(8.2, 4.5)
    ax.plot(s["year"], s["a_total"].cumsum(), color=BLUE, linewidth=2.0, label=a)
    ax.plot(s["year"], s["b_total"].cumsum(), color=ORANGE, linewidth=2.0, label=b)
    ax.plot(s["year"], s["both"].cumsum(), color=PURPLE, linewidth=2.6,
            marker="o", markersize=4.5, label="both")
    ax.fill_between(s["year"], s["both"].cumsum(), color=PURPLE, alpha=0.14)
    _style(ax, "2. %s - cumulative (never falls)" % label,
           "application year", "patents to date")
    _int_axis(ax)
    _year_axis(ax, s["year"])
    ax.legend(fontsize=8.5, frameon=False)
    _finish(fig, (png_prefix + "_2_pair_cumulative.png") if png_prefix else None)

    print("%s: %d patents. %s: %d patents. Both: %d." % (a, na, b, nb, n_both))
    if n_both:
        yrs = s.loc[s["both"] > 0, "year"]
        print("First together in %d, last in %d. They overlap on %.0f%% of the "
              "patents that mention either."
              % (int(yrs.min()), int(yrs.max()),
                 100 * n_both / union if union else 0))
        if not relation:
            print("Chance alone would put them together in about %.1f patents; "
                  "observed %d, a lift of %.2f."
                  % (exp, n_both, n_both / exp if exp else 0.0))
    else:
        print("They never appear together%s."
              % (" under %s" % relation if relation else ""))
    if n_both < 5:
        print("Fewer than five patents carry both, so the per-year shape is "
              "noise. Say so rather than narrating the bars.")
    return s


def triples(state, choice_primary, choice_secondary, relation=None, limit=20,
            top=15, min_keys=5):
    """The T1-preposition-T2 rows behind the bars, so a claim can be checked."""
    a = pick(state, choice_primary)
    if isinstance(choice_secondary, (int, np.integer)):
        listing = list(_partners(state, a, top=top,
                                 min_keys=min_keys).get("technical term", []))
        b = listing[int(choice_secondary) - 1]
    else:
        b = pick(state, choice_secondary)
    rel = _relation_index(state)
    m = rel[(((rel["T1"] == a) & (rel["T2"] == b)) |
             ((rel["T1"] == b) & (rel["T2"] == a)))]
    if relation:
        m = m[m["relation"].str.lower() == str(relation).lower()]
    print("%d triples link %r and %r%s; showing %d."
          % (len(m), a, b, (" via %s" % relation) if relation else "",
             min(limit, len(m))))
    return m[["pubno", "year", "T1", "prep", "relation", "T2"]].head(limit)


def documents_for(state, keyword, other=None, limit=15):
    """The patents behind a bar."""
    a = pick(state, keyword)
    b = pick(state, other) if other else None
    cols = state["columns"]
    idx = [i for i, doc in enumerate(state["per_doc"])
           if a in doc and (b is None or b in doc)]
    frame = state["frame"].iloc[idx[:limit]].copy()
    frame.insert(0, "year", [state["years"][i] for i in idx[:limit]])
    want = ["year"] + [c for c in (cols["pubno"], cols["title"], cols["date"]) if c]
    print("%d patents contain %s%s; showing %d."
          % (len(idx), repr(a), (" and " + repr(b)) if b else "",
             min(limit, len(idx))))
    return frame[want]
