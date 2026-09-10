"""Track B - trt-pb with its implementation defects removed, methods unchanged.

Paste this whole module into the chat, then call it. Do NOT put it in the
Gem's Knowledge field: a Gem carrying a knowledge file is served without
Gemini's Python tool, so it can read this file but never run it, and the
Gem will then correctly refuse to answer. Runs in the code-execution
sandbox:
standard library, pandas and numpy only.

What this file is
-----------------
The same architecture as the shipped program - keyphrases, greedy grouping,
preposition triples, six relationship classes, DS / DF / FS, fixed-width year
chunks, four quadrant readings - with every defect that the reference document
settled on evidence repaired, and nothing else changed. It answers: what would
this architecture have produced if its apparent mistakes were fixed?

Repaired here (numbers are the reference document's; section numbers are the
analysis document's):
  1  clean_text strips the JPO boilerplate it was written to strip
  2  keyword pattern gets a non-capturing group: 'sensor' no longer hits 'biosensor'
  3  keyphrases are re.escape'd before entering the regex
  4  the unreachable 'includes' / 'utilizes' entries are gone
  6  recency() is enabled, with ONE window specification recorded (see RECENCY)
  7  citing strings split on record separators only, not on every punctuation mark
 12  DS is guarded against a zero denominator
 17  the cumulative view is labelled an adoption curve and never read as a trend
 19  the graph is directed, as it is rendered
 20  edge multiplicity is retained: occurrences and distinct patents per edge
 21  grouping no longer depends on upload order: phrases are processed in a
     recorded order (corpus frequency, then alphabetical)
 s3  DS marginal denominators are counted BEFORE the domain explode
 4.6 single-word keyphrases are allowed
 oq2 the quadrant mapping is stated explicitly (QUADRANTS) and printed

Not changed, because they are methods rather than mistakes: lift as the DS
estimator (no floor), the unshrunk mean as FS, linear citation-age
normalisation, fixed-width windows, the greedy grouping algorithm, the six
preposition classes. Track C is where those are challenged.

Sandbox substitutions are the same as Track A's and are stand-ins, not fixes:
fixed preposition list for ADP; corpus-internal inflection test for verb
lemmas; stopword segmentation + tf-idf for KeyBERT; character-trigram cosine
for MPNet.

    from trt_corrected import report
    state = report("patents.xlsx")
"""

from __future__ import annotations

import datetime
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TRACK = "B"

# --------------------------------------------------------------------------- #
# columns and loading
# --------------------------------------------------------------------------- #

COLUMN_HINTS = {
    "abstract": ("abstract",),
    "pubno": ("publication number", "publication numbers", "pubno", "pub no"),
    "date": ("application date", "application dates", "priority date",
             "filing date", "date"),
    "title": ("title",),
    "description": ("english description", "description"),
    "domain": ("technology domain", "technology domains", "domain"),
    "citing": ("citing patents", "citing", "forward citation", "cited by"),
}


def _pick(cols, explicit, hints):
    if explicit:
        if explicit not in cols:
            raise SystemExit("column %r not in file. Columns: %s"
                             % (explicit, ", ".join(map(str, cols))))
        return explicit
    low = [(str(c).lower(), c) for c in cols]
    for h in hints:
        for lc, c in low:
            if h in lc:
                return c
    return None


def load(path, **explicit):
    frame = (pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls"))
             else pd.read_csv(path))
    cols = list(frame.columns)
    colmap = {k: _pick(cols, explicit.get(k), hints) for k, hints in COLUMN_HINTS.items()}
    if colmap["abstract"] is None:
        raise SystemExit("no abstract column. Columns: %s" % ", ".join(map(str, cols)))
    return frame, colmap


# --------------------------------------------------------------------------- #
# cleaning and dates - defect 1 fixed
# --------------------------------------------------------------------------- #

_JP = re.compile(r"\b(?:PROBLEM TO BE SOLVED|SOLUTION|SELECTED DRAWING)\s*:\s*")


def clean_text(text):
    """PatentInfo.clean_text with the trailing \\b removed, so the heading is
    matched when a space follows the colon - which is always."""
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    cleaned = re.sub(r"\(.*\)\n", "", text)
    cleaned = cleaned.replace(";", ".").replace("\n", " ").replace("-", " ").replace(",", " ")
    return _JP.sub("", cleaned)


_YEAR = re.compile(r"(?<!\d)((?:1[6-9]|2[01])\d{2})(?!\d)")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def year_of(value):
    """The application year, or None. The original sliced str(value)[:4]; this
    keeps that intent and also survives Timestamps and Excel serials. It is
    never derived from a publication number."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.year
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = int(value)
        if 20000 <= n <= 80000:
            return (_EXCEL_EPOCH + pd.Timedelta(days=n)).year
        return n if 1600 <= n <= 2199 else None
    m = _YEAR.search(str(value))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# tokens, prepositions, verbs (sandbox stand-ins, unchanged from Track A)
# --------------------------------------------------------------------------- #

_TOK = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?|[^\sA-Za-z0-9]")
_WORD = re.compile(r"[a-z][a-z0-9]*")


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _tokens(sentence):
    return _TOK.findall(sentence)


_PREPS = frozenset("""
of in with from on at within by for to across against during into through via
as under over between among after before about above below along around behind
beneath beside beyond despite down except inside like near off onto out outside
per since than throughout toward towards underneath until unto up upon versus
without amid amidst
""".split())

_AUX = frozenset("""be is are was were been being am have has had having do does
did doing can could may might must shall should will would let get got make made
say said see seen go went come came take took give gave know knew think thought
use used""".split())

_NO_E_VV = re.compile(r"[aeiou]{2}[b-df-hj-np-tv-z]$")
_NO_E_CLUSTER = re.compile(r"(rm|lm|rk|lk|rn|rl|ln|rd|ld|nd|nt|rt|lt|pt|ct|st|ft|mp|sk|sp|ss)$")


def _stems(w):
    out = set()
    if w.endswith("ies") and len(w) > 4:
        out.add((w[:-3] + "y", "s"))
    if w.endswith("es") and len(w) > 3 and w[:-2].endswith(("s", "sh", "ch", "x", "z", "o")):
        out.add((w[:-2], "s"))
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        out.add((w[:-1], "s"))
    if w.endswith("ied") and len(w) > 4:
        out.add((w[:-3] + "y", "ed"))
    if w.endswith("ed") and len(w) > 3:
        st = w[:-2]
        out.update({(st, "ed"), (w[:-1], "ed")})
        if len(st) > 2 and st[-1] == st[-2]:
            out.add((st[:-1], "ed"))
    if w.endswith("ing") and len(w) > 4:
        st = w[:-3]
        out.update({(st, "ing"), (st + "e", "ing")})
        if len(st) > 2 and st[-1] == st[-2]:
            out.add((st[:-1], "ing"))
    return out


def verb_lexicon(texts):
    """Corpus-internal stand-in for POS tagging: a base is a verb lemma when at
    least two of its -s / -ed / -ing classes occur. Returns {surface: lemma}."""
    counts = Counter()
    for t in texts:
        counts.update(_WORD.findall(str(t).lower()))
    evidence = defaultdict(lambda: defaultdict(set))
    for w in counts:
        for base, cls in _stems(w):
            if len(base) >= 3 and base.isalpha() and base not in _AUX:
                evidence[base][cls].add(w)
    accepted = {b: e for b, e in evidence.items() if len(e) >= 2}

    def rank(base):
        stem = base[:-1]
        e_ok = base.endswith("e") and not ((len(stem) <= 5 and _NO_E_VV.search(stem))
                                           or _NO_E_CLUSTER.search(stem))
        return (len(accepted[base]), base in counts, e_ok, -len(base))

    forms = {}
    for base, classes in accepted.items():
        for surface_set in classes.values():
            for s in surface_set:
                if s not in forms or rank(base) > rank(forms[s]):
                    forms[s] = base
    for base in set(forms.values()):
        forms.setdefault(base, base)
    return forms


def _is_prep(toks, i, lex):
    t = toks[i].lower()
    if t not in _PREPS:
        return False
    if t == "to" and i + 1 < len(toks) and lex.get(toks[i + 1].lower()) == toks[i + 1].lower():
        return False
    return True


# --------------------------------------------------------------------------- #
# keyphrases - 1-grams allowed (analysis 4.6)
# --------------------------------------------------------------------------- #

_STOP = frozenset("""
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
hence therefore it they them he she we you i there here via as for per
""".split())

_GENERIC = frozenset("""apparatus method system device assembly arrangement means
unit mechanism structure module equipment machine process technique procedure
application operation invention embodiment portion member element part""".split())


def _trim_verbs(run, lex):
    while run and lex.get(run[0], run[0]) != run[0]:
        run = run[1:]
    while run and lex.get(run[-1], run[-1]) != run[-1]:
        run = run[:-1]
    return run


def candidate_phrases(text, lex, lo=1, hi=4):
    """Maximal content runs of 1-4 words. A run that is only a category noun
    ('apparatus', 'method') is not a technology and is dropped - the only
    filter a 1-gram needs that a 2-gram did not."""
    out = []
    for sent in re.split(r"[.!?,:;()\[\]]", text.lower()):
        run = []
        for word in sent.split() + [""]:
            m = _WORD.match(word)
            tok = m.group(0) if m else ""
            if not tok or tok in _STOP or tok.isdigit():
                run = _trim_verbs(run, lex)
                # A single letter is a chemical variable, not a technology.
                # Multi-word phrases are unaffected; 1-grams must be words.
                if len(run) == 1 and len(run[0]) < 3:
                    run = []
                if lo <= len(run) <= hi and not all(w in _GENERIC for w in run):
                    out.append(" ".join(run))
                run = []
            else:
                run.append(tok)
    return out


def keywords_per_document(docs, lex, no_keywords=7):
    per_doc = [candidate_phrases(d, lex) for d in docs]
    df = Counter()
    for cands in per_doc:
        df.update(set(cands))
    n = len(docs)
    chosen = []
    for cands in per_doc:
        tf = Counter(cands)
        scored = sorted(tf, key=lambda p: (-(tf[p] * (math.log((n + 1) / (df[p] + 1)) + 1)), p))
        chosen.append(scored[:no_keywords])
    flat = Counter(p for top in chosen for p in top)
    # recorded order: corpus frequency, then alphabetical (defect 21 analogue)
    ordered = sorted(flat, key=lambda p: (-flat[p], p))
    return chosen, ordered, flat


# --------------------------------------------------------------------------- #
# TRT extraction - unchanged
# --------------------------------------------------------------------------- #

def get_trt(text, lex):
    trt = []
    for sent in _sentences(text):
        toks = _tokens(sent)
        pos = ["ADP" if _is_prep(toks, i, lex) else "X" for i in range(len(toks))]
        for k, tok in enumerate(toks):
            if pos[k] != "ADP":
                continue
            i1, t1 = k - 1, " "
            while pos[i1] != "ADP":
                t1 = toks[i1] if t1 == " " else toks[i1] + " " + t1
                if i1 != 0:
                    i1 -= 1
                else:
                    break
            i2, t2 = k + 1, " "
            if i2 > len(toks) - 1:
                continue
            while pos[i2] != "ADP":
                t2 = toks[i2] if t2 == " " else t2 + " " + toks[i2]
                if i2 != len(toks) - 1:
                    i2 += 1
                else:
                    break
            trt.append({"T1": t1, "Preposition": tok, "T2": re.sub(r"\.", "", t2),
                        "sentence": sent})
    return trt


def trt_frame(abstracts, pubnos, lex):
    rows = []
    for pub, abstract in zip(pubnos, abstracts):
        for t in get_trt(abstract, lex):
            rows.append((pub, t["T1"], t["Preposition"], t["T2"], t["sentence"]))
    return pd.DataFrame(rows, columns=["Publication number", "T1", "prep", "T2", "sentence"])


# --------------------------------------------------------------------------- #
# grouping - the same greedy algorithm on a recorded order
# --------------------------------------------------------------------------- #

def _trigram_vectors(phrases):
    vocab, rows = {}, []
    for p in phrases:
        s = "  %s  " % p.lower()
        c = Counter(s[i:i + 3] for i in range(len(s) - 2))
        rows.append(c)
        for g in c:
            vocab.setdefault(g, len(vocab))
    m = np.zeros((len(phrases), len(vocab)))
    for i, c in enumerate(rows):
        for g, v in c.items():
            m[i, vocab[g]] = v
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return m / norm


def similarity_matrix(phrases):
    v = _trigram_vectors(phrases)
    return v @ v.T


def keywordsynonyms(kws, threshold, sim=None):
    """bertui.keywordsynonyms, unchanged. Its order dependence is a property
    of the method; Track B makes the order a recorded input rather than an
    accident of the upload."""
    kws = list(dict.fromkeys(kws))
    sim = similarity_matrix(kws) if sim is None else sim
    keyword_dict, jindex = {}, set()
    for i in range(len(sim)):
        if i not in jindex:
            hits = np.where(sim[i] >= threshold)[0]
            temp = [kws[j] for j in hits if j not in jindex]
            keyword_dict[kws[i]] = list(dict.fromkeys(temp))
            jindex.update(hits.tolist())
    return keyword_dict


# --------------------------------------------------------------------------- #
# keyword mapping - defects 2, 3, 4 fixed
# --------------------------------------------------------------------------- #

INCLUSION = ["of", "in", "with", "from", "on", "at", "within", "by"]
OBJECTIVE = ["for"]
EFFECT = ["to", "across", "against"]
PROCESS = ["during", "into", "through", "via"]
LIKENESS = ["as"]
RELATIONSHIPS = ["Inclusion", "Objective", "Effect", "Process", "Likeness", "Misc"]


def classify_prep(prep):
    p = prep.lower()
    if p in INCLUSION:
        return "Inclusion"
    if p in OBJECTIVE:
        return "Objective"
    if p in EFFECT:
        return "Effect"
    if p in PROCESS:
        return "Process"
    if p in LIKENESS:
        return "Likeness"
    return "Misc"


def build_keyword_pattern(listkw):
    """Escaped, grouped, longest-first: a term matches only as a whole word,
    a parenthesis in a phrase is a parenthesis, and a multi-word term is
    preferred over its own substring."""
    terms = sorted({k for k in listkw if k}, key=len, reverse=True)
    if not terms:
        return re.compile(r"(?!x)x")
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


def keywordmap(df_trt, listkw, keydict):
    pat = build_keyword_pattern(listkw)
    owner = {}
    for key, value in keydict.items():
        for v in value:
            owner.setdefault(v, key)
    rows = []
    for pub, t1, prep, t2, sent in df_trt.itertuples(index=False):
        for a in pat.findall(str(t1)):
            for b in pat.findall(str(t2)):
                rows.append([owner.get(a.lower(), a.lower()), owner.get(b.lower(), b.lower()),
                             classify_prep(prep), pub, sent])
    return pd.DataFrame(rows, columns=["T1", "T2", "Prep", "Pubno.", "sentence"])


def graph(state, relationship="Inclusion"):
    """Directed edge table with multiplicity (defects 19, 20). One row per
    (T1, T2): occurrences, distinct patents, publication numbers."""
    sub = state["dfmap"][state["dfmap"]["Prep"] == relationship]
    if sub.empty:
        return {"relationship": relationship, "edges": pd.DataFrame(), "nodes": 0}
    g = (sub.groupby(["T1", "T2"])
            .agg(occurrences=("Pubno.", "size"), distinct_patents=("Pubno.", "nunique"),
                 publication_ids=("Pubno.", lambda s: "\n".join(dict.fromkeys(map(str, s)))))
            .reset_index()
            .sort_values(["distinct_patents", "occurrences"], ascending=False)
            .reset_index(drop=True))
    return {"relationship": relationship, "edges": g,
            "nodes": len(set(g["T1"]) | set(g["T2"]))}


def edge_evidence(state, t1, t2, relationship=None):
    d = state["dfmap"]
    sub = d[(d["T1"] == t1) & (d["T2"] == t2)]
    if relationship:
        sub = sub[sub["Prep"] == relationship]
    return sub[["Pubno.", "Prep", "sentence"]]


# --------------------------------------------------------------------------- #
# tab 3 relevancy - one bucketing function, integer years, literal match
# --------------------------------------------------------------------------- #

_EDGES = [1980, 1990, 2000, 2003, 2006, 2009, 2012, 2014, 2016, 2018, 2020, 2022]
_BUCKET_LABELS = ["1980-1990", "1990-2000", "2000-2003", "2003-2006", "2006-2009",
                  "2009-2012", "2012-2014", "2014-2016", "2016-2018", "2018-2020",
                  "2020-2022", "2022-Present"]


def _bucket(year):
    if year is None:
        return None
    for i in range(len(_EDGES) - 1):
        if _EDGES[i] <= year < _EDGES[i + 1]:
            return i
    return 11 if year >= _EDGES[-1] else None


def relevancy(state, node):
    """Share of abstracts in each bucket containing the node, matched literally.
    Rows without a year are excluded instead of landing in the final bucket."""
    buckets = [_bucket(y) for y in state["years"]]
    node_l = node.lower()
    rows = []
    for i, label in enumerate(_BUCKET_LABELS):
        idx = [j for j, b in enumerate(buckets) if b == i]
        hits = sum(1 for j in idx if node_l in state["abstracts"][j].lower())
        rows.append((label, round(hits / len(idx), 3) if idx else 0, hits, len(idx)))
    return pd.DataFrame(rows, columns=["Years", "Relevancy Score",
                                       "No. of patents it appears in",
                                       "No. of patents in this time-frame"])


# --------------------------------------------------------------------------- #
# TTA - corrected counts, same estimators
# --------------------------------------------------------------------------- #

_SEP = re.compile(r"[\n;|,]+")


def count_citing_patents(citing_string):
    """Split on record separators (newline, semicolon, pipe, comma) and count
    non-empty entries. A publication number's internal hyphens, slashes and
    spaces no longer multiply the count (defect 7)."""
    if not citing_string or not isinstance(citing_string, str):
        return 0
    return len([p for p in _SEP.split(citing_string) if p.strip()])


def count_citing_shipped(citing_string):
    if not citing_string or not isinstance(citing_string, str):
        return 0
    return len([p for p in re.split(r"[\s;,.|/\-(){}[\]<>]+", citing_string) if p])


def split_domains(value):
    if not isinstance(value, str):
        return []
    return [d.strip() for d in value.split("\n") if d.strip()]


def explode_domains(frame, domain_col):
    df = frame.copy()
    df["_domains"] = df[domain_col].map(split_domains)
    df = df.explode("_domains").reset_index(drop=True)
    df[domain_col] = df["_domains"]
    return df.drop(columns="_domains")


def domain_specificity(patents, exploded, domain_col, term_col):
    """DS(TD, V) = (DF_TD/NumPat_TD) / (DF_All/NumPat_All) with DF_All and
    NumPat_All on the pre-explode frame (analysis section 3) and a zero guard
    (defect 12). Returns DS, the counts, and the inflation each verb would
    have had under the shipped denominator."""
    num_all = len(patents)
    df_all = Counter()
    for terms in patents[term_col]:
        df_all.update(set(terms))
    df_td, num_td = defaultdict(Counter), Counter()
    for td, terms in zip(exploded[domain_col], exploded[term_col]):
        if isinstance(td, str):
            num_td[td] += 1
            df_td[td].update(set(terms))
    ds = {}
    for td in df_td:
        for v, k in df_td[td].items():
            marginal = df_all[v] / num_all if num_all else 0
            ds[(td, v)] = (k / num_td[td]) / marginal if marginal else 0.0
    return ds, {"DF_All": df_all, "DF_TD": df_td, "NumPat_TD": num_td, "NumPat_All": num_all}


RECENCY = {"window_years": 5, "threshold": 0.7}
RECENCY_SPECS = [
    ("referenced paper (comment :148)", "N = 15 years"),
    ("author's stated intent (same comment)", "Year > 2015"),
    ("implemented threshold_yr (:120)", "most recent 5% of the year span"),
    ("comment's stated threshold (:149)", "Recency > 0.7"),
    ("commented-out selection (:155)", "Recency > 0.9"),
]


def recency_table(exploded, domain_col, term_col, max_year):
    """Recency(TD, V) = DF_TD,N(V) / DF_TD(V) with N = RECENCY['window_years'].
    The five specifications in the source disagree; this one is a recorded
    choice, not a recovered one, and the report prints all five."""
    cutoff = max_year - RECENCY["window_years"]
    recent, total = defaultdict(Counter), defaultdict(Counter)
    for td, terms, year in zip(exploded[domain_col], exploded[term_col], exploded["Year"]):
        if not isinstance(td, str):
            continue
        total[td].update(set(terms))
        if year is not None and not pd.isna(year) and year > cutoff:
            recent[td].update(set(terms))
    return {(td, v): recent[td][v] / n for td in total for v, n in total[td].items() if n}


def _year_chunks(min_year, max_year, year_range):
    start = min_year - 1
    return [(start + i * year_range, min(start + (i + 1) * year_range - 1, max_year))
            for i in range((max_year - start) // year_range + 1)]


def _fs_table(rows, label):
    out = pd.DataFrame(rows, columns=[label, "DF", "Net Forward Citations", "Avg Forward Citations"])
    df = out["DF"].replace(0, np.nan)
    out["FS"] = (out["Avg Forward Citations"] / df).fillna(0)
    return out


def trend_tables(sub, term_col, term, max_year_overall, year_range=3):
    sub = sub[sub["Year"].notna()].copy()
    sub["Year"] = sub["Year"].astype(int)
    sub["avg citing patents"] = sub["count_citing_patents"] / ((max_year_overall - sub["Year"]) + 1)
    hit = sub[sub[term_col].map(lambda t: term in t)]
    cumulative = []
    for end in sorted(sub["Year"].unique()):
        f = hit[hit["Year"] <= end]
        cumulative.append((int(end), len(f), f["count_citing_patents"].sum(), f["avg citing patents"].sum()))
    windowed = []
    for a, b in _year_chunks(int(sub["Year"].min()), int(sub["Year"].max()), year_range):
        f = hit[(hit["Year"] >= a) & (hit["Year"] <= b)]
        windowed.append(("%d-%d" % (a, b), len(f), f["count_citing_patents"].sum(), f["avg citing patents"].sum()))
    return _fs_table(cumulative, "End Year"), _fs_table(windowed, "Year Range")


# Operational definition, recorded (open question 2). DF is the y axis, FS the
# x axis, one arrow per consecutive pair of windows.
QUADRANTS = {("up", "up"): "Growing", ("up", "down"): "Generalizing",
             ("down", "down"): "Declining", ("down", "up"): "Repositioning"}


def read_quadrants(windowed):
    labels = []
    rows = windowed.to_dict("records")
    for prev, cur in zip(rows, rows[1:]):
        d = "up" if cur["DF"] > prev["DF"] else "down" if cur["DF"] < prev["DF"] else "flat"
        f = "up" if cur["FS"] > prev["FS"] else "down" if cur["FS"] < prev["FS"] else "flat"
        labels.append((prev[windowed.columns[0]], cur[windowed.columns[0]],
                       QUADRANTS.get((d, f), "no movement (%s DF, %s FS)" % (d, f))))
    return pd.DataFrame(labels, columns=["from", "to", "reading"])


def _domain_frame(state, td, term_col):
    ex = state["exploded"]
    sub = ex[ex[state["columns"]["domain"]] == td].copy()
    if sub.empty:
        raise SystemExit("domain %r not in the exploded frame. Domains: %s"
                         % (td, ", ".join(map(str, state["domains"][:20]))))
    return sub


def _trend(state, td, term, term_col, year_range):
    sub = _domain_frame(state, td, term_col)
    cum, win = trend_tables(sub, term_col, term, state["max_year"], year_range)
    return {"domain": td, "term": term, "windowed": win, "reading": read_quadrants(win),
            "adoption_curve": cum.rename(columns={"DF": "cumulative DF (cannot decline)"})}


def tta_function(state, td, verb, year_range=3):
    """TTA1 for one (domain, technical verb). The reading comes from the
    windowed table only; the cumulative table is returned as an adoption curve."""
    return _trend(state, td, verb, "filtered_tech_verbs", year_range)


def tta_technology(state, td, key, year_range=3):
    """TTA2 for one (domain, keyphrase)."""
    return _trend(state, td, key, "filtered_tech_keys", year_range)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def report(path, *, threshold=0.80, no_keywords=7, ds_threshold=5, relationship="Inclusion",
           show=25, **columns):
    warnings = []
    frame, cm = load(path, **columns)
    print("TRACK B - corrected replica (methods unchanged, defects repaired)")
    print("COLUMNS  " + "  ".join("%s=%r" % kv for kv in cm.items()))

    keep = frame[frame[cm["abstract"]].notna()]
    if cm["pubno"]:
        keep = keep[keep[cm["pubno"]].notna()]
    keep = keep.reset_index(drop=True).copy()
    abstracts = [clean_text(a) for a in keep[cm["abstract"]]]
    pubnos = list(keep[cm["pubno"]]) if cm["pubno"] else ["row%d" % i for i in range(len(keep))]
    years = [year_of(v) for v in keep[cm["date"]]] if cm["date"] else [None] * len(keep)
    lex = verb_lexicon((list(keep[cm["description"]]) if cm["description"] else []) + abstracts)
    dated = sum(1 for y in years if y)
    print("CORPUS   rows=%d usable=%d dated=%d years=%s..%s verb lemmas (heuristic)=%d"
          % (len(frame), len(keep), dated, min((y for y in years if y), default=None),
             max((y for y in years if y), default=None), len(set(lex.values()))))
    if cm["date"] and dated < len(keep) * 0.5:
        warnings.append("fewer than half the rows have a usable year; trends cover part of the corpus.")

    chosen, ordered, freq = keywords_per_document(abstracts, lex, no_keywords)
    keydict = keywordsynonyms(ordered, threshold)
    df_trt = trt_frame(abstracts, pubnos, lex)
    dfmap = keywordmap(df_trt, ordered, keydict)
    print("KEYPHRASES  %d per abstract, %d unique (1-4 words), processed in frequency order"
          % (no_keywords, len(ordered)))
    print("DICTIONARY  %d primary groups at similarity >= %.2f (greedy; order recorded)"
          % (len(keydict), threshold))
    for k in sorted(keydict, key=lambda k: -len(keydict[k]))[:show]:
        print("  %-36s <- %s" % (k[:36], ", ".join(v for v in keydict[k] if v != k)[:90]))
    print("TRIPLES     %d (T1, prep, T2); mapped rows by class:" % len(df_trt))
    for r in RELATIONSHIPS:
        print("  %-10s %d" % (r, int((dfmap["Prep"] == r).sum())))
    g = graph({"dfmap": dfmap}, relationship)
    if len(g["edges"]):
        e = g["edges"]
        print("GRAPH [%s]  nodes=%d directed edges=%d; top edges by distinct patents:"
              % (relationship, g["nodes"], len(e)))
        for _, r in e.head(min(show, 8)).iterrows():
            print("  %-30s -> %-30s patents=%d occ=%d" % (r["T1"][:30], r["T2"][:30],
                                                          r["distinct_patents"], r["occurrences"]))

    desc_col = cm["description"]
    if desc_col is None:
        warnings.append("no description column; TTA1 verbs extracted from abstracts instead.")
    texts = keep[desc_col] if desc_col else keep[cm["abstract"]]
    keep["Year"] = years
    keep["Verbs"] = [sorted({lex[w] for w in _WORD.findall(str(t).lower()) if w in lex}) for t in texts]
    keep["cleaned_abstracts"] = abstracts
    # Word-boundary matching, not the shipped substring test: without it a
    # one-word concept matches inside every longer word that contains it.
    _kwpat = build_keyword_pattern(ordered)
    keep["filtered_tech_keys"] = [sorted({m.lower() for m in _kwpat.findall(a)}) for a in abstracts]
    keep["count_citing_patents"] = keep[cm["citing"]].map(count_citing_patents) if cm["citing"] else 0
    if cm["citing"]:
        shipped = keep[cm["citing"]].map(count_citing_shipped).sum()
        fixed = keep["count_citing_patents"].sum()
        if fixed and shipped / fixed > 1.05:
            print("CITATIONS   shipped splitter would have counted %.2fx the entries (defect 7 fired)"
                  % (shipped / fixed))
    max_year = max((y for y in years if y), default=None)
    if cm["domain"] is None:
        warnings.append("no technology-domain column; DS and both TTA tabs are unavailable.")
        exploded, ds, counts, tech_verbs, domains, rec = keep, {}, {}, [], [], {}
    else:
        exploded = explode_domains(keep, cm["domain"])
        ds, counts = domain_specificity(keep, exploded, cm["domain"], "Verbs")
        rec = recency_table(exploded, cm["domain"], "Verbs", max_year) if max_year else {}
        tech = defaultdict(set)
        for (td, v), val in ds.items():
            if val >= ds_threshold or rec.get((td, v), 0) > RECENCY["threshold"]:
                tech[td].add(v)
        exploded["filtered_tech_verbs"] = [sorted(v for v in vs if v in tech.get(td, ()) and v.isalnum())
                                           for td, vs in zip(exploded[cm["domain"]], exploded["Verbs"])]
        tech_verbs = sorted({v for vs in exploded["filtered_tech_verbs"] for v in vs})
        domains = [d for d in exploded[cm["domain"]].dropna().unique()]
        print("TTA1        %d domains (%d rows from %d patents; marginals on the %d patents); "
              "%d technical verbs at DS >= %s or recency > %.1f"
              % (len(domains), len(exploded), len(keep), counts["NumPat_All"], len(tech_verbs),
                 ds_threshold, RECENCY["threshold"]))
        print("RECENCY     window = last %d years (cutoff > %s). Recorded choice; the source holds "
              "five specifications:" % (RECENCY["window_years"], (max_year or 0) - RECENCY["window_years"]))
        for src, spec in RECENCY_SPECS:
            print("  %-42s %s" % (src, spec))
        for td in domains[:show]:
            top = sorted(((ds[(td, v)], v) for v in tech.get(td, ())), reverse=True)[:8]
            print("  %-28s n=%-4d %s" % (str(td)[:28], counts["NumPat_TD"][td],
                                         ", ".join("%s(%.1f%s)" % (v, s, ",R" if rec.get((td, v), 0) > RECENCY["threshold"] else "")
                                                   for s, v in top)))
    if max_year is None:
        warnings.append("no usable application year; TTA trend tables cannot be produced.")
    print("QUADRANTS   " + "; ".join("DF %s / FS %s = %s" % (d, f, lab) for (d, f), lab in QUADRANTS.items()))
    for w in warnings:
        print("WARNING  " + w)
    return {"frame": frame, "columns": cm, "keep": keep, "abstracts": abstracts, "pubnos": pubnos,
            "years": years, "lex": lex, "keywords_per_doc": chosen, "keywords": ordered,
            "keyword_freq": freq, "keydict": keydict, "df_trt": df_trt, "dfmap": dfmap,
            "exploded": exploded, "ds": ds, "ds_counts": counts, "recency": rec,
            "tech_verbs": tech_verbs, "domains": domains, "max_year": max_year,
            "threshold": threshold, "warnings": warnings}


def documents_for(state, phrase, limit=15):
    keep, cm = state["keep"], state["columns"]
    hits = [i for i, a in enumerate(state["abstracts"]) if phrase.lower() in a.lower()]
    cols = [c for c in (cm["pubno"], cm["title"], cm["date"]) if c]
    return keep.iloc[hits[:limit]][cols]


# --------------------------------------------------------------------------- #
# label-free measurements (analysis document, section 6.2) - run against B
# to see what the fixes changed and what they did not
# --------------------------------------------------------------------------- #

def adjusted_rand(a, b):
    items = sorted(set(a) & set(b))
    pairs = Counter((a[i], b[i]) for i in items)
    ra, cb = Counter(a[i] for i in items), Counter(b[i] for i in items)
    c2 = lambda n: n * (n - 1) / 2
    sum_ij = sum(c2(v) for v in pairs.values())
    sum_a, sum_b = sum(c2(v) for v in ra.values()), sum(c2(v) for v in cb.values())
    n2 = c2(len(items))
    if n2 == 0:
        return 1.0
    expected = sum_a * sum_b / n2
    maxi = (sum_a + sum_b) / 2
    return 1.0 if maxi == expected else (sum_ij - expected) / (maxi - expected)


def order_sensitivity(state, runs=20, seed=0):
    """Shuffle the upload order, rebuild the recorded order, regroup. ARI is 1
    by construction now - which is the fix - while the method's own order
    dependence is still visible if you pass recorded=False."""
    kws = list(state["keywords"])
    freq = state["keyword_freq"]
    sim = similarity_matrix(kws)
    rng = random.Random(seed)
    parts = []
    for _ in range(runs):
        order = list(range(len(kws)))
        rng.shuffle(order)
        order.sort(key=lambda i: (-freq[kws[i]], kws[i]))
        kd = keywordsynonyms([kws[i] for i in order], state["threshold"], sim[np.ix_(order, order)])
        parts.append({m: k for k, ms in kd.items() for m in ms})
    aris = [adjusted_rand(parts[i], parts[j]) for i in range(runs) for j in range(i + 1, runs)]
    print("ORDER SENSITIVITY  %d shuffles of the upload: ARI mean=%.3f min=%.3f (recorded order)"
          % (runs, float(np.mean(aris)), float(np.min(aris))))
    return {"ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris))}


def permutation_null(state, td, term, year_range=3, runs=200, seed=0, term_col="filtered_tech_verbs"):
    sub = _domain_frame(state, td, term_col)
    sub = sub[sub["Year"].notna()].copy()
    sub["Year"] = sub["Year"].astype(int)
    sub["avg"] = sub["count_citing_patents"] / ((state["max_year"] - sub["Year"]) + 1)
    chunks = _year_chunks(int(sub["Year"].min()), int(sub["Year"].max()), year_range)
    has = sub[term_col].map(lambda t: term in t).to_numpy()
    avg = sub["avg"].to_numpy()
    win_idx = [np.where((sub["Year"] >= a) & (sub["Year"] <= b))[0] for a, b in chunks]
    dfs = [int(has[idx].sum()) for idx in win_idx]
    obs = [avg[idx][has[idx]].mean() if dfs[i] else 0.0 for i, idx in enumerate(win_idx)]
    rng = np.random.default_rng(seed)
    null = np.zeros((runs, len(chunks)))
    for r in range(runs):
        for i, idx in enumerate(win_idx):
            if dfs[i] and len(idx):
                null[r, i] = avg[rng.choice(idx, size=min(dfs[i], len(idx)), replace=False)].mean()
    labels = Counter()
    for r in range(runs):
        for i in range(1, len(chunks)):
            d = "up" if dfs[i] > dfs[i - 1] else "down" if dfs[i] < dfs[i - 1] else "flat"
            f = "up" if null[r, i] > null[r, i - 1] else "down" if null[r, i] < null[r, i - 1] else "flat"
            labels[QUADRANTS.get((d, f), "no movement")] += 1
    bands = [("%d-%d -> %d-%d" % (chunks[i - 1] + chunks[i]), obs[i] - obs[i - 1],
              float(np.percentile(null[:, i] - null[:, i - 1], 2.5)),
              float(np.percentile(null[:, i] - null[:, i - 1], 97.5))) for i in range(1, len(chunks))]
    table = pd.DataFrame(bands, columns=["movement", "observed dFS", "null 2.5%", "null 97.5%"])
    total = sum(labels.values()) or 1
    print("PERMUTATION NULL  %s / %s, %d runs: %s"
          % (td, term, runs, ", ".join("%s %.0f%%" % (k, 100 * v / total) for k, v in labels.most_common())))
    print(table.to_string(index=False))
    return {"null_labels": dict(labels), "bands": table, "observed_fs": obs, "df": dfs}


def funnel(state, td, window, term_col="filtered_tech_verbs"):
    sub = _domain_frame(state, td, term_col)
    sub = sub[sub["Year"].notna()].copy()
    sub["Year"] = sub["Year"].astype(int)
    sub = sub[(sub["Year"] >= window[0]) & (sub["Year"] <= window[1])]
    sub["avg"] = sub["count_citing_patents"] / ((state["max_year"] - sub["Year"]) + 1)
    mu, sigma = sub["avg"].mean(), sub["avg"].std(ddof=0)
    rows = []
    for term in sorted({t for ts in sub[term_col] for t in ts}):
        hit = sub[sub[term_col].map(lambda t: term in t)]
        n, fs = len(hit), hit["avg"].mean()
        half = 1.96 * sigma / math.sqrt(n)
        rows.append((term, n, fs, mu - half, mu + half, abs(fs - mu) <= half))
    out = pd.DataFrame(rows, columns=["term", "DF", "FS", "band_low", "band_high", "inside_band"])
    if len(out):
        print("FUNNEL  %s %s: %d terms, %.0f%% inside the null band (mu=%.3f)"
              % (td, window, len(out), 100 * out["inside_band"].mean(), mu))
    else:
        print("FUNNEL  %s %s: no term has support in this window" % (td, window))
    return out.sort_values("DF", ascending=False)


def censoring_horizon(state, frac=0.5):
    cm, keep = state["columns"], state["keep"]
    if not cm["citing"] or state["max_year"] is None:
        print("CENSORING  needs a citing column and years")
        return None
    tab = pd.DataFrame({"year": keep["Year"], "c": keep["count_citing_patents"]}).dropna()
    tab["year"] = tab["year"].astype(int)
    med = tab.groupby("year")["c"].median().sort_index()
    ages = (state["max_year"] - med.index).to_numpy()
    smooth = med.rolling(3, center=True, min_periods=1).median()
    plateau = float(smooth.max()) if len(smooth) else 0.0
    H = None
    for age, m in sorted(zip(ages, smooth.to_numpy())):
        if plateau > 0 and m >= frac * plateau:
            H = int(age)
            break
    out = pd.DataFrame({"cohort": med.index, "age": ages, "median_citations": med.to_numpy(),
                        "smoothed": smooth.to_numpy()})
    print("CENSORING  plateau median=%.1f; H=%s years (%s of plateau)" % (plateau, H, frac))
    return {"H": H, "table": out}


def ds_bias(state):
    """Dbar / Dbar_V per surviving verb: the factor the shipped denominator
    multiplied each DS by. Section 3 predicts values above 1 for the verbs
    most likely to pass the threshold; this measures it."""
    cm, keep = state["columns"], state["keep"]
    if not cm["domain"]:
        return None
    ndom = keep[cm["domain"]].map(lambda v: len(split_domains(v)))
    dbar = ndom.mean()
    rows = []
    for v in state["tech_verbs"]:
        has = keep["Verbs"].map(lambda vs: v in vs)
        if has.any():
            rows.append((v, ndom[has].mean(), dbar / ndom[has].mean()))
    out = pd.DataFrame(rows, columns=["verb", "Dbar_V", "shipped DS inflation (Dbar/Dbar_V)"])
    if len(out):
        q = out.iloc[:, 2].quantile([0.1, 0.5, 0.9])
        print("DS BIAS  Dbar=%.2f; the shipped estimator inflated DS by p10=%.2f median=%.2f p90=%.2f"
              % (dbar, q.iloc[0], q.iloc[1], q.iloc[2]))
    return out.sort_values(out.columns[2], ascending=False)


def window_sensitivity(state, td, widths=(2, 3, 4, 5), term_col="filtered_tech_verbs"):
    sub = _domain_frame(state, td, term_col)
    terms = sorted({t for ts in sub[term_col] for t in ts})
    rows = []
    for term in terms:
        readings = []
        for w in widths:
            _, win = trend_tables(sub, term_col, term, state["max_year"], w)
            r = read_quadrants(win)
            readings.append(r.iloc[-1, 2] if len(r) else "n/a")
        rows.append([term] + readings + [len(set(readings)) > 1])
    out = pd.DataFrame(rows, columns=["term"] + ["%dy" % w for w in widths] + ["flips"])
    if len(out):
        print("WINDOW SENSITIVITY  %s: %d terms, flip rate %.0f%%" % (td, len(out), 100 * out["flips"].mean()))
    return out


def citation_format_check(state, n=5):
    cm = state["columns"]
    if not cm["citing"]:
        print("CITATIONS  no citing column")
        return None
    raw = state["keep"][cm["citing"]].dropna().astype(str)
    raw = raw[raw.str.strip() != ""]
    shipped, fixed = raw.map(count_citing_shipped), raw.map(count_citing_patents)
    ratio = shipped.sum() / fixed.sum() if fixed.sum() else float("nan")
    print("CITATIONS  %d non-empty strings; shipped / corrected count = %.2f" % (len(raw), ratio))
    for s in raw.head(n):
        print("  %r -> shipped %d, corrected %d" % (s[:70], count_citing_shipped(s), count_citing_patents(s)))
    return {"ratio": ratio}
