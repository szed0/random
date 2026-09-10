"""Track B - corrected replica: the original architecture and methods, with
the implementation defects removed.

Paste this whole module into the chat, then call it. Do NOT put it in the
Gem's Knowledge field: a Gem carrying a knowledge file is served without
Gemini's Python tool, so it can read this file but never run it, and the
Gem will then correctly refuse to answer. Runs in the code-execution
sandbox:
standard library, pandas and numpy only.

What this file is
-----------------
The same methods as the shipped program wherever reasonable, with the
mistakes fixed and nothing else changed. It answers: what would this
architecture have produced if its apparent implementation mistakes were fixed?

Corrected here, following the intent-preserving plan's list:
  * phrase-boundary matching, missing re.escape, broken boilerplate regexes
  * cache keys and cache invalidation: grouping is keyed on the masked phrase,
    so each primary is ranked under its own masking
  * MLM cache dependence on mask positions (the same fix, seen from the cache)
  * redundant re-reading and tokenisation: loop-invariant work is done once
  * publication-number parsing: numbers are stripped, never split
  * duplicate and shadowed computations: one definition of each count
  * patent-level denominators after domain expansion: the DS marginal is
    counted on patents, not on exploded rows; domains are stripped and
    deduplicated per patent
  * TTA cross-tab coupling: no state passes through files or another tab
  * graph direction consistency: directed, as rendered
  * accidental edge-weight loss: occurrences and distinct patents per edge
  * citing strings split on record separators only
  * the cumulative view is labelled an adoption curve and is never read as a
    trend; the four quadrant readings are stated as an operational definition

NOT enabled: the Recency method. The source holds five contradictory
specifications for its window and threshold, so choosing one would be a
methodological redesign, not a bug fix. It stays off unless the originating
method is recovered. recency() is callable so the specifications can be
compared; report() prints all five.

Also unchanged, because they are methods: 2-4 word keyphrases (no 1-grams),
the greedy grouping loop and its min_keys floor, lift as DS, the unshrunk
mean as FS, linear age normalisation, fixed-width bins.

Sandbox stand-ins are Track A's, documented where they occur.

    from trt_corrected import report
    state = report("patents.xlsx")
"""

from __future__ import annotations

import datetime
import math
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
    "date": ("application date", "application dates", "priority date", "filing date", "date"),
    "title": ("title",),
    "description": ("english description", "description"),
    "domain": ("technology domain", "technology domains", "domain"),
    "citing": ("citing patents", "citing", "forward citation", "cited by"),
}


def _pick(cols, explicit, hints):
    if explicit:
        if explicit not in cols:
            raise SystemExit("column %r not in file. Columns: %s" % (explicit, ", ".join(map(str, cols))))
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
# cleaning, dates, publication numbers - fixed
# --------------------------------------------------------------------------- #

_JP = re.compile(r"\b(?:PROBLEM TO BE SOLVED|SOLUTION|SELECTED DRAWING)\s*:\s*")


def clean_text(text):
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    cleaned = re.sub(r"\(.*\)\n", "", text)
    cleaned = cleaned.replace(";", ".").replace("\n", " ").replace("-", " ").replace(",", " ")
    return _JP.sub("", cleaned)


_YEAR = re.compile(r"(?<!\d)((?:1[6-9]|2[01])\d{2})(?!\d)")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def year_of(value):
    """The application year, or None. Never derived from a publication number."""
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


def pubno_of(value):
    """A publication number is one token; whitespace is trimmed, nothing is split."""
    return str(value).strip() if value is not None and not (isinstance(value, float) and math.isnan(value)) else None


# --------------------------------------------------------------------------- #
# tokens, prepositions, verbs (stand-ins, unchanged from Track A)
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
# keyphrases - 2-4 words, recorded order
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


def _trim_verbs(run, lex):
    while run and lex.get(run[0], run[0]) != run[0]:
        run = run[1:]
    while run and lex.get(run[-1], run[-1]) != run[-1]:
        run = run[:-1]
    return run


def candidate_phrases(text, lex, lo=2, hi=4):
    out = []
    for sent in re.split(r"[.!?,:;()\[\]]", text.lower()):
        run = []
        for word in sent.split() + [""]:
            m = _WORD.match(word)
            tok = m.group(0) if m else ""
            if not tok or tok in _STOP or tok.isdigit():
                run = _trim_verbs(run, lex)
                if lo <= len(run) <= hi:
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
    freq = Counter(p for top in chosen for p in top)
    return chosen, sorted(freq, key=lambda p: (-freq[p], p)), freq


# --------------------------------------------------------------------------- #
# MLM grouping - same loop, mask-aware cache, loop-invariant preprocessing
# --------------------------------------------------------------------------- #

BERT_VOCAB = 30522


def _context_matrix(abstracts, phrases, window=3, max_vocab=5000, min_freq=2):
    """Computed ONCE per corpus (the shipped loop re-read and re-tokenised the
    file for every primary). Vocabulary and phrase context vectors, PPMI."""
    docs = [_WORD.findall(a.lower()) for a in abstracts]
    wf = Counter(w for d in docs for w in d)
    vocab = [w for w, c in wf.most_common(max_vocab) if c >= min_freq and w not in _STOP]
    vidx = {w: i for i, w in enumerate(vocab)}
    by_first = defaultdict(list)
    for p in phrases:
        by_first[p.split()[0]].append(p.split())
    wctx, pctx = defaultdict(Counter), {p: Counter() for p in phrases}
    for toks in docs:
        for i, t in enumerate(toks):
            if t in vidx:
                wctx[t].update(w for w in toks[max(0, i - window):i] + toks[i + 1:i + 1 + window]
                               if w in vidx and w != t)
            for words in by_first.get(t, ()):
                n = len(words)
                if toks[i:i + n] == words:
                    pctx[" ".join(words)].update(w for w in toks[max(0, i - window):i] + toks[i + n:i + n + window]
                                                 if w in vidx)
    total = Counter()
    for c in list(wctx.values()) + list(pctx.values()):
        total.update(c)
    grand = sum(total.values()) or 1

    def vec(counter):
        v = np.zeros(len(vocab))
        n = sum(counter.values()) or 1
        for w, k in counter.items():
            pmi = math.log2((k / n) / (total[w] / grand))
            if pmi > 0:
                v[vidx[w]] = pmi
        nv = np.linalg.norm(v)
        return v / nv if nv else v

    W = np.array([vec(wctx[w]) for w in vocab]) if vocab else np.zeros((0, 0))
    return vocab, W, {p: vec(pctx[p]) for p in phrases}


def find_rankings(candidates, ranking, vocab_size):
    out = {}
    for c in candidates:
        pos = [ranking[w] for w in c.split() if w in ranking]
        out[c] = {"average": float(np.mean(pos)) if pos else float(vocab_size)}
    return out


def map_primary_to_secondary(kws, threshold, ctx, min_keys=5):
    """The shipped loop with its cache keyed on the masked phrase: each primary
    is ranked under its own masking. Input order is the recorded frequency
    order, so the partition is reproducible for a given corpus."""
    vocab, W, P = ctx
    cutoff = max(1.0, threshold * len(vocab) / BERT_VOCAB)
    keyphrases = list(kws)
    out, used, cache = {}, set(), {}
    while keyphrases:
        primary = keyphrases.pop(0)
        if primary in used:
            continue
        if primary not in cache:                       # key = the masked phrase
            scores = W @ P[primary] if len(vocab) else np.zeros(0)
            order = np.argsort(-scores, kind="stable")
            cache[primary] = {vocab[j]: r + 1 for r, j in enumerate(order)}
        word_dict = find_rankings(keyphrases, cache[primary], len(vocab))
        secondary = [p for p in word_dict if word_dict[p]["average"] <= cutoff]
        if len(secondary) < min_keys:
            secondary = [p for p, _ in sorted(word_dict.items(), key=lambda kv: kv[1]["average"])[:min_keys]]
        out[primary] = secondary
        used.add(primary)
        used.update(secondary)
        keyphrases = [k for k in keyphrases if k not in used]
    return out


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
            trt.append({"T1": t1, "Preposition": tok, "T2": re.sub(r"\.", "", t2), "sentence": sent})
    return trt


def trt_frame(abstracts, pubnos, lex):
    rows = [(pub, t["T1"], t["Preposition"], t["T2"], t["sentence"])
            for pub, abstract in zip(pubnos, abstracts) for t in get_trt(abstract, lex)]
    return pd.DataFrame(rows, columns=["Publication number", "T1", "prep", "T2", "sentence"])


# --------------------------------------------------------------------------- #
# keyword mapping - escaped, grouped, unreachable entries removed
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
    """Directed edges with occurrences, distinct patents and publication ids."""
    sub = state["dfmap"][state["dfmap"]["Prep"] == relationship]
    if sub.empty:
        return {"relationship": relationship, "edges": pd.DataFrame(), "nodes": 0}
    g = (sub.groupby(["T1", "T2"])
            .agg(occurrences=("Pubno.", "size"), distinct_patents=("Pubno.", "nunique"),
                 publication_ids=("Pubno.", lambda s: "\n".join(dict.fromkeys(map(str, s)))))
            .reset_index().sort_values(["distinct_patents", "occurrences"], ascending=False)
            .reset_index(drop=True))
    return {"relationship": relationship, "edges": g, "nodes": len(set(g["T1"]) | set(g["T2"]))}


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
_BUCKET_LABELS = ["1980-1990", "1990-2000", "2000-2003", "2003-2006", "2006-2009", "2009-2012",
                  "2012-2014", "2014-2016", "2016-2018", "2018-2020", "2020-2022", "2022-Present"]


def _bucket(year):
    if year is None:
        return None
    for i in range(len(_EDGES) - 1):
        if _EDGES[i] <= year < _EDGES[i + 1]:
            return i
    return 11 if year >= _EDGES[-1] else None


def relevancy(state, node):
    buckets = [_bucket(y) for y in state["years"]]
    node_l = node.lower()
    rows = []
    for i, label in enumerate(_BUCKET_LABELS):
        idx = [j for j, b in enumerate(buckets) if b == i]
        hits = sum(1 for j in idx if node_l in state["abstracts"][j].lower())
        rows.append((label, round(hits / len(idx), 3) if idx else 0, hits, len(idx)))
    return pd.DataFrame(rows, columns=["Years", "Relevancy Score", "No. of patents it appears in",
                                       "No. of patents in this time-frame"])


# --------------------------------------------------------------------------- #
# TTA - corrected counts, same estimators, recency NOT enabled
# --------------------------------------------------------------------------- #

_SEP = re.compile(r"[\n;|,]+")


def count_citing_patents(citing_string):
    if not citing_string or not isinstance(citing_string, str):
        return 0
    return len([p for p in _SEP.split(citing_string) if p.strip()])


def count_citing_shipped(citing_string):
    if not citing_string or not isinstance(citing_string, str):
        return 0
    return len([p for p in re.split(r"[\s;,.|/\-(){}[\]<>]+", citing_string) if p])


def split_domains(value):
    """Newline-separated, stripped, deduplicated per patent."""
    if not isinstance(value, str):
        return []
    return list(dict.fromkeys(d.strip() for d in value.split("\n") if d.strip()))


def explode_domains(frame, domain_col):
    df = frame.copy()
    df["_domains"] = df[domain_col].map(split_domains)
    df = df.explode("_domains").reset_index(drop=True)
    df[domain_col] = df["_domains"]
    return df.drop(columns="_domains")


def domain_specificity(patents, exploded, domain_col, term_col):
    """DS = (DF_TD/NumPat_TD) / (DF_All/NumPat_All), marginal on patents."""
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


RECENCY_SPECS = [
    ("referenced paper (comment :148)", "N = 15 years"),
    ("author's stated intent (same comment)", "Year > 2015"),
    ("implemented threshold_yr (:120)", "most recent 5% of the year span"),
    ("comment's stated threshold (:149)", "Recency > 0.7"),
    ("commented-out selection (:155)", "Recency > 0.9"),
]


def recency(state, td, term, window_years=None, term_col="Verbs"):
    """Recency(TD, V) = DF_TD,N(V) / DF_TD(V). Not used for selection. With
    window_years=None the implemented spec (5% of the span) is used, so the
    five specifications can be compared on real data before one is chosen."""
    ex = state["exploded"]
    sub = ex[ex[state["columns"]["domain"]] == td]
    ys = [y for y in state["years"] if y]
    if window_years is None:
        cutoff = int(max(ys) - (max(ys) - min(ys)) * 0.05)
    else:
        cutoff = max(ys) - window_years
    has = sub[term_col].map(lambda t: term in t)
    denom = int(has.sum())
    recent = int((has & (sub["Year"] > cutoff)).sum())
    return {"domain": td, "term": term, "cutoff_year": cutoff, "recent": recent, "total": denom,
            "recency": recent / denom if denom else float("nan")}


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


# Operational definition of the four states, recorded.
QUADRANTS = {("up", "up"): "Growing", ("up", "down"): "Generalizing",
             ("down", "down"): "Declining", ("down", "up"): "Repositioning"}


def read_quadrants(windowed):
    labels, rows = [], windowed.to_dict("records")
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
        raise SystemExit("domain %r not found. Domains: %s" % (td, ", ".join(map(str, state["domains"][:20]))))
    return sub


def _trend(state, td, term, term_col, year_range):
    sub = _domain_frame(state, td, term_col)
    cum, win = trend_tables(sub, term_col, term, state["max_year"], year_range)
    return {"domain": td, "term": term, "windowed": win, "reading": read_quadrants(win),
            "adoption_curve": cum.rename(columns={"DF": "cumulative DF (cannot decline)"})}


def tta_function(state, td, verb, year_range=3):
    return _trend(state, td, verb, "filtered_tech_verbs", year_range)


def tta_technology(state, td, key, year_range=3):
    return _trend(state, td, key, "filtered_tech_keys", year_range)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def report(path, *, threshold=75, min_keys=5, no_keywords=7, ds_threshold=5, relationship="Inclusion",
           show=25, **columns):
    warnings = []
    frame, cm = load(path, **columns)
    print("TRACK B - corrected replica (methods unchanged, defects repaired, recency still off)")
    print("COLUMNS  " + "  ".join("%s=%r" % kv for kv in cm.items()))
    keep = frame[frame[cm["abstract"]].notna()]
    if cm["pubno"]:
        keep = keep[keep[cm["pubno"]].notna()]
    keep = keep.reset_index(drop=True).copy()
    abstracts = [clean_text(a) for a in keep[cm["abstract"]]]
    pubnos = [pubno_of(p) for p in keep[cm["pubno"]]] if cm["pubno"] else ["row%d" % i for i in range(len(keep))]
    years = [year_of(v) for v in keep[cm["date"]]] if cm["date"] else [None] * len(keep)
    lex = verb_lexicon((list(keep[cm["description"]]) if cm["description"] else []) + abstracts)
    dated = sum(1 for y in years if y)
    print("CORPUS   rows=%d usable=%d dated=%d years=%s..%s verb lemmas (heuristic)=%d"
          % (len(frame), len(keep), dated, min((y for y in years if y), default=None),
             max((y for y in years if y), default=None), len(set(lex.values()))))
    if cm["date"] and dated < len(keep) * 0.5:
        warnings.append("fewer than half the rows have a usable year; trends cover part of the corpus.")

    chosen, ordered, freq = keywords_per_document(abstracts, lex, no_keywords)
    ctx = _context_matrix(abstracts, ordered)
    keydict = map_primary_to_secondary(ordered, threshold, ctx, min_keys)
    df_trt = trt_frame(abstracts, pubnos, lex)
    dfmap = keywordmap(df_trt, ordered, keydict)
    print("KEYPHRASES  %d per abstract, %d unique (2-4 words), processed in frequency order" % (no_keywords, len(ordered)))
    cutoff = max(1.0, threshold * len(ctx[0]) / BERT_VOCAB)
    forced = sum(1 for v in keydict.values() if len(v) == min_keys)
    print("DICTIONARY  %d primaries at mean rank <= %d of %d (rank %.1f of %d corpus words), floor %d; "
          "%d groups sized by the floor" % (len(keydict), threshold, BERT_VOCAB, cutoff, len(ctx[0]), min_keys, forced))
    for k in sorted(keydict, key=lambda k: -len(keydict[k]))[:show]:
        print("  %-36s <- %s" % (k[:36], ", ".join(keydict[k])[:90]))
    print("TRIPLES     %d (T1, prep, T2); mapped rows by class:" % len(df_trt))
    for r in RELATIONSHIPS:
        print("  %-10s %d" % (r, int((dfmap["Prep"] == r).sum())))
    g = graph({"dfmap": dfmap}, relationship)
    if len(g["edges"]):
        print("GRAPH [%s]  nodes=%d directed edges=%d; top by distinct patents:" % (relationship, g["nodes"], len(g["edges"])))
        for _, r in g["edges"].head(8).iterrows():
            print("  %-30s -> %-30s patents=%d occ=%d" % (r["T1"][:30], r["T2"][:30], r["distinct_patents"], r["occurrences"]))

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
        shipped, fixed = keep[cm["citing"]].map(count_citing_shipped).sum(), keep["count_citing_patents"].sum()
        if fixed and shipped / fixed > 1.05:
            print("CITATIONS   the shipped splitter would have counted %.2fx the entries" % (shipped / fixed))
    max_year = max((y for y in years if y), default=None)
    if cm["domain"] is None:
        warnings.append("no technology-domain column; DS and both TTA tabs are unavailable.")
        exploded, ds, counts, tech_verbs, domains = keep, {}, {}, [], []
    else:
        exploded = explode_domains(keep, cm["domain"])
        ds, counts = domain_specificity(keep, exploded, cm["domain"], "Verbs")
        tech = defaultdict(set)
        for (td, v), val in ds.items():
            if val >= ds_threshold:
                tech[td].add(v)
        exploded["filtered_tech_verbs"] = [sorted(v for v in vs if v in tech.get(td, ()) and v.isalnum())
                                           for td, vs in zip(exploded[cm["domain"]], exploded["Verbs"])]
        tech_verbs = sorted({v for vs in exploded["filtered_tech_verbs"] for v in vs})
        domains = [d for d in exploded[cm["domain"]].dropna().unique()]
        print("TTA1        %d domains (%d rows from %d patents; marginals on %d patents); %d technical verbs at DS >= %s"
              % (len(domains), len(exploded), len(keep), counts["NumPat_All"], len(tech_verbs), ds_threshold))
        print("RECENCY     not enabled. Five contradictory specifications in the source; recency(state, td, term) "
              "computes any of them for comparison:")
        for src, spec in RECENCY_SPECS:
            print("  %-42s %s" % (src, spec))
        for td in domains[:show]:
            top = sorted(((ds[(td, v)], v) for v in tech.get(td, ())), reverse=True)[:8]
            print("  %-28s n=%-4d %s" % (str(td)[:28], counts["NumPat_TD"][td], ", ".join("%s(%.1f)" % (v, s) for s, v in top)))
    if max_year is None:
        warnings.append("no usable application year; TTA trend tables cannot be produced.")
    print("QUADRANTS   " + "; ".join("DF %s / FS %s = %s" % (d, f, lab) for (d, f), lab in QUADRANTS.items()))
    for w in warnings:
        print("WARNING  " + w)
    return {"frame": frame, "columns": cm, "keep": keep, "abstracts": abstracts, "pubnos": pubnos, "years": years,
            "lex": lex, "keywords_per_doc": chosen, "keywords": ordered, "keyword_freq": freq, "keydict": keydict,
            "context": ctx, "df_trt": df_trt, "dfmap": dfmap, "exploded": exploded, "ds": ds, "ds_counts": counts,
            "tech_verbs": tech_verbs, "domains": domains, "max_year": max_year, "threshold": threshold,
            "min_keys": min_keys, "warnings": warnings}


def documents_for(state, phrase, limit=15):
    keep, cm = state["keep"], state["columns"]
    hits = [i for i, a in enumerate(state["abstracts"]) if phrase.lower() in a.lower()]
    cols = [c for c in (cm["pubno"], cm["title"], cm["date"]) if c]
    return keep.iloc[hits[:limit]][cols]


# --------------------------------------------------------------------------- #
# what the fixes changed - cheap comparisons against the shipped behaviour
# --------------------------------------------------------------------------- #

def ds_bias(state):
    """Dbar / Dbar_V per surviving verb: the factor the shipped, post-explode
    denominator multiplied each DS by."""
    cm, keep = state["columns"], state["keep"]
    if not cm["domain"]:
        return None
    ndom = keep[cm["domain"]].map(lambda v: len(split_domains(v)))
    dbar = ndom.mean()
    rows = [(v, ndom[keep["Verbs"].map(lambda vs: v in vs)].mean()) for v in state["tech_verbs"]]
    out = pd.DataFrame([(v, d, dbar / d) for v, d in rows if d], columns=["verb", "Dbar_V", "shipped DS inflation"])
    if len(out):
        q = out.iloc[:, 2].quantile([0.1, 0.5, 0.9])
        print("DS BIAS  Dbar=%.2f; shipped DS inflated by p10=%.2f median=%.2f p90=%.2f" % (dbar, q.iloc[0], q.iloc[1], q.iloc[2]))
    return out.sort_values(out.columns[2], ascending=False)


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


def cache_defect_effect(state):
    """Regroup with the shipped cache behaviour (first primary's ranking reused
    for all) and count the phrases that change group."""
    vocab, W, P = state["context"]
    cutoff = max(1.0, state["threshold"] * len(vocab) / BERT_VOCAB)
    keyphrases, out, used, cached = list(state["keywords"]), {}, set(), None
    while keyphrases:
        primary = keyphrases.pop(0)
        if primary in used:
            continue
        if cached is None:
            order = np.argsort(-(W @ P[primary]), kind="stable")
            cached = {vocab[j]: r + 1 for r, j in enumerate(order)}
        wd = find_rankings(keyphrases, cached, len(vocab))
        sec = [p for p in wd if wd[p]["average"] <= cutoff]
        if len(sec) < state["min_keys"]:
            sec = [p for p, _ in sorted(wd.items(), key=lambda kv: kv[1]["average"])[:state["min_keys"]]]
        out[primary] = sec
        used.add(primary)
        used.update(sec)
        keyphrases = [k for k in keyphrases if k not in used]
    own = lambda kd: {m: k for k, ms in kd.items() for m in ms + [k]}
    a, b = own(state["keydict"]), own(out)
    moved = sum(1 for m in a if b.get(m) != a[m])
    print("CACHE DEFECT  corrected: %d groups; shipped behaviour: %d groups; %d of %d phrases change group"
          % (len(state["keydict"]), len(out), moved, len(a)))
    return {"corrected": state["keydict"], "shipped": out, "moved": moved}
