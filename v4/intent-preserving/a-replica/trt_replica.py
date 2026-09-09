"""Track A - historical / behavioural replica of trt-pb, frozen as shipped.

Attach as a Gemini Gem knowledge file. Runs in the code-execution sandbox:
standard library, pandas and numpy only. No spaCy, no MPNet, no BERT.

What this file is
-----------------
The experimental control. It is frozen around the actual main_streamlit.py
behaviour, including known defects, and answers one question: what did the
shipped program actually produce? It is scientifically useful precisely
because it is not silently repaired.

Preserved, as the intent-preserving plan lists them:
  * Patent-BERT MLM greedy phrase grouping - pop-first primary, mean-rank
    threshold (lower = more similar), min_keys floor forcing a group for
    every primary
  * the incorrect logits cache behaviour: the cache key ignores the masked
    positions, so every primary after the first is ranked against the first
    primary's ranking
  * POS-pattern candidate extraction, 2-4 words, no single-word concepts
  * KeyBERT/MPNet top-7 ranking per abstract
  * six preposition-based relationship categories, with the unreachable
    'includes' / 'utilizes' entries in place
  * the existing phrase matching: unescaped, ungrouped alternation
  * deduplicated graph edges, undirected build rendered directed
  * post-domain-explode DS denominators, no frequency floor, no zero guard
  * DS-only technical-verb selection; recency() present and never called
  * the existing citation-string parsing (split on every punctuation mark)
  * linear citation-age normalisation, mean Function Score
  * fixed-width temporal bins starting one year before the first patent,
    plus the cumulative view
  * the graph / dictionary / provenance outputs

Substitutions forced by the sandbox - stand-ins, not fixes, each documented
where it occurs:
  * spaCy ADP tagging  -> a fixed preposition list ('to' before a verb lemma
    is treated as the infinitive marker spaCy tags PART)
  * spaCy verb lemmas  -> a corpus-internal inflection test
  * POS-pattern candidates + KeyBERT -> stopword-segmented 2-4 word runs
    ranked per abstract by tf-idf
  * the masked-LM vocabulary ranking -> a distributional ranking: corpus
    words ordered by how well their contexts match the primary phrase's
    contexts; a candidate's 'average rank' is the mean rank of its words.
    Same control flow, same threshold units, same floor, same cache defect.

    from trt_replica import report
    state = report("patents.xlsx")
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TRACK = "A"

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
# cleaning and dates - as shipped
# --------------------------------------------------------------------------- #

def clean_text(text):
    """PatentInfo.clean_text. The last substitution never fires (defect 1)."""
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    cleaned = re.sub(r"\(.*\)\n", "", text)
    cleaned = cleaned.replace(";", ".").replace("\n", " ").replace("-", " ").replace(",", " ")
    return re.sub(r"\bPROBLEM TO BE SOLVED:\b", "", cleaned)


def year_original(value):
    """str(value)[:4] coerced to int, else None."""
    s = str(value)[:4]
    try:
        return int(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# tokens, prepositions, verbs
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
# keyphrases - KeyBERT stand-in, 2-4 words
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
    """Top-n per abstract by tf-idf; the flat list keeps duplicates, as tab 1's
    `keywords` list does."""
    per_doc = [candidate_phrases(d, lex) for d in docs]
    df = Counter()
    for cands in per_doc:
        df.update(set(cands))
    n = len(docs)
    chosen, flat = [], []
    for cands in per_doc:
        tf = Counter(cands)
        scored = sorted(tf, key=lambda p: (-(tf[p] * (math.log((n + 1) / (df[p] + 1)) + 1)), p))
        chosen.append(scored[:no_keywords])
        flat.extend(scored[:no_keywords])
    return chosen, flat


# --------------------------------------------------------------------------- #
# MLM grouping - map_primary_to_secondary, structure and defect preserved
# --------------------------------------------------------------------------- #

def _context_matrix(abstracts, phrases, window=3, max_vocab=5000, min_freq=2):
    """Context vectors for every corpus word (the 'vocabulary') and for every
    candidate phrase, PPMI-weighted. Stands in for the masked-LM logit vector:
    a word whose contexts match the phrase's contexts is a word the model
    would rank highly in the phrase's slot."""
    docs = [_WORD.findall(a.lower()) for a in abstracts]
    wf = Counter(w for d in docs for w in d)
    vocab = [w for w, c in wf.most_common(max_vocab) if c >= min_freq and w not in _STOP]
    vidx = {w: i for i, w in enumerate(vocab)}
    by_first = defaultdict(list)
    for p in phrases:
        by_first[p.split()[0]].append(p.split())
    wctx = defaultdict(Counter)
    pctx = {p: Counter() for p in phrases}
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
    P = {p: vec(pctx[p]) for p in phrases}
    return vocab, W, P


def find_rankings(primary, candidates, ranking):
    """Position of each candidate's words in the vocabulary ranking for the
    masked primary, averaged - the 'average' key of the shipped word_dict.
    Words outside the vocabulary are skipped, as -1 positions were."""
    out = {}
    for c in candidates:
        pos = [ranking[w] for w in c.split() if w in ranking]
        out[c] = {"average": float(np.mean(pos)) if pos else float(len(ranking))}
    return out


BERT_VOCAB = 30522   # size of the WordPiece vocabulary the shipped threshold ranks over


def map_primary_to_secondary(kws, threshold, ctx, min_keys=5, cache_bug=True):
    """main_streamlit.map_primary_to_secondary, line for line in structure:
    pop the first unused phrase as primary, rank every remaining phrase,
    keep those whose average rank <= threshold, and if fewer than min_keys
    clear it take the best min_keys anyway. With cache_bug=True the ranking
    computed for the FIRST primary is reused for every later primary, which
    is what a cache key that omits the mask positions does.

    The threshold is in ranks over a 30,522-entry vocabulary; this corpus's
    word vocabulary is smaller, so the cutoff is rescaled to the same
    fraction of the vocabulary (never below rank 1)."""
    vocab, W, P = ctx
    cutoff = max(1.0, threshold * len(vocab) / BERT_VOCAB)
    keyphrases = list(kws)
    primary_to_secondary, used = {}, set()
    cached = None
    while keyphrases:
        primary = keyphrases.pop(0)
        if primary in used:
            continue
        words = list(dict.fromkeys(keyphrases))
        if cached is None or not cache_bug:
            scores = W @ P[primary] if len(vocab) else np.zeros(0)
            order = np.argsort(-scores, kind="stable")
            ranking = {vocab[j]: r + 1 for r, j in enumerate(order)}
            if cached is None:
                cached = ranking
        else:
            ranking = cached
        word_dict = find_rankings(primary, words, ranking)
        secondary = list(dict.fromkeys(p for p in word_dict if word_dict[p]["average"] <= cutoff))
        if len(secondary) < min_keys:
            best = sorted(word_dict.items(), key=lambda kv: kv[1]["average"])[:min_keys]
            secondary = list(dict.fromkeys(p for p, _ in best))
        primary_to_secondary[primary] = secondary
        used.add(primary)
        used.update(secondary)
        keyphrases = [k for k in keyphrases if k not in used]
    return primary_to_secondary


# --------------------------------------------------------------------------- #
# TRT extraction - TRTExtraction.get_trt, token for token
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
            trt.append({"T1": t1, "Preposition": tok, "T2": t2})
    return trt


def cleant2(text):
    return re.sub(r"\.", "", text)


def trt_frame(abstracts, pubnos, lex):
    rows = []
    for pub, abstract in zip(pubnos, abstracts):
        for t in get_trt(abstract, lex):
            rows.append((pub, t["T1"], t["Preposition"], cleant2(t["T2"])))
    return pd.DataFrame(rows, columns=["Publication number", "T1", "prep", "T2"])


# --------------------------------------------------------------------------- #
# keyword mapping and relationship classes - as inherited
# --------------------------------------------------------------------------- #

INCLUSION = ["of", "in", "with", "from", "on", "at", "within", "includes", "by", "utilizes"]
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


def build_keyword_pattern(listkw, warnings):
    """'\\b' + '|'.join(terms) + '\\b': unescaped (defect 3), ungrouped (defect 2)."""
    terms = list(dict.fromkeys(k for k in listkw if k))
    raw = r"\b" + "|".join(terms) + r"\b"
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error as exc:
        warnings.append("defect 3 fired: keyword regex failed to compile (%s); the shipped program "
                        "would have crashed here. Using an escaped pattern for this run only." % exc)
        return re.compile(r"\b" + "|".join(re.escape(t) for t in terms) + r"\b", re.IGNORECASE)


def keywordmap(df_trt, listkw, keydict, warnings):
    pat = build_keyword_pattern(listkw, warnings)
    rows = []
    for pub, t1, prep, t2 in df_trt[["Publication number", "T1", "prep", "T2"]].itertuples(index=False):
        for a in pat.findall(str(t1)):
            for b in pat.findall(str(t2)):
                rows.append([a, b, classify_prep(prep), pub])
    dfmap = pd.DataFrame(rows, columns=["T1", "T2", "Prep", "Pubno."])
    owner = {}
    for key, value in keydict.items():
        for v in value:
            owner.setdefault(v, key)
    dfmap["T1"] = dfmap["T1"].map(lambda t: owner.get(t, t))
    dfmap["T2"] = dfmap["T2"].map(lambda t: owner.get(t, t))
    return dfmap


def jsonlist(dfmap, relationship):
    sub = dfmap[dfmap["Prep"] == relationship]
    out = {}
    for t1 in sub["T1"].unique():
        out[t1] = list(dict.fromkeys(sub[sub["T1"] == t1]["T2"]))
    return out


def graph(state, relationship="Inclusion"):
    """nx.Graph from jsonlist, rendered directed (defect 19), multiplicity
    discarded (defect 20). Returns the techres table with publication numbers
    per edge - the export the original got right."""
    dfmap = state["dfmap"]
    jsont = jsonlist(dfmap, relationship)
    directed = {(a, b) for a, bs in jsont.items() for b in bs}
    undirected = {frozenset(e) for e in directed if len(set(e)) == 2}
    sub = dfmap[dfmap["Prep"] == relationship]
    rows = []
    for a, bs in jsont.items():
        pubs = ["\n".join(sub[(sub["T1"] == a) & (sub["T2"] == b)]["Pubno."]) for b in bs]
        rows.append((a, ", ".join(bs), "\n\n".join(pubs)))
    techres = pd.DataFrame(rows, columns=["Technologynode1", "Technologynodes2", "Publication number"])
    return {"relationship": relationship, "adjacency": jsont,
            "nodes": len({n for e in directed for n in e}),
            "directed_pairs": len(directed), "undirected_edges": len(undirected),
            "reverse_pairs_collapsed": len(directed) - len(undirected),
            "occurrences_discarded": int(len(sub) - len(directed)), "techres": techres}


def edge_evidence(state, t1, t2, relationship=None):
    d = state["dfmap"]
    sub = d[(d["T1"] == t1) & (d["T2"] == t2)]
    return sub[sub["Prep"] == relationship] if relationship else sub


# --------------------------------------------------------------------------- #
# tab 3 relevancy - twelve hardcoded buckets, string comparison
# --------------------------------------------------------------------------- #

_BUCKETS = [("1980", "1990"), ("1990", "2000"), ("2000", "2003"), ("2003", "2006"),
            ("2006", "2009"), ("2009", "2012"), ("2012", "2014"), ("2014", "2016"),
            ("2016", "2018"), ("2018", "2020"), ("2020", "2022"), None]
_BUCKET_LABELS = ["1980-1990", "1990-2000", "2000-2003", "2003-2006", "2006-2009",
                  "2009-2012", "2012-2014", "2014-2016", "2016-2018", "2018-2020",
                  "2020-2022", "2022-Present"]


def _bucket(yr):
    for i, b in enumerate(_BUCKETS):
        if b is None or (yr >= b[0] and yr < b[1]):
            return i
    return 11


def relevancy(state, node):
    years = [str(y) if y is not None else "nan" for y in state["years"]]
    buckets = [_bucket(y) for y in years]
    try:
        pat = re.compile(node, re.IGNORECASE)
    except re.error:
        pat = re.compile(re.escape(node), re.IGNORECASE)
    rows = []
    for i, label in enumerate(_BUCKET_LABELS):
        idx = [j for j, b in enumerate(buckets) if b == i]
        hits = sum(1 for j in idx if pat.search(state["abstracts"][j]))
        rows.append((label, round(hits / len(idx), 3) if idx else 0, hits, len(idx)))
    return pd.DataFrame(rows, columns=["Years", "Relevancy Score", "No. of patents it appears in",
                                       "No. of patents in this time-frame"])


# --------------------------------------------------------------------------- #
# TTA - functionalityTTA / technologyTTA arithmetic, unchanged
# --------------------------------------------------------------------------- #

def count_citing_patents(citing_string):
    """Split on whitespace, punctuation, slashes and hyphens; count fragments (defect 7)."""
    if not citing_string or not isinstance(citing_string, str):
        return 0
    return len([p for p in re.split(r"[\s;,.|/\-(){}[\]<>]+", citing_string) if p])


def explode_domains(frame, domain_col):
    df = frame.copy()
    df[domain_col] = df[domain_col].astype(object).where(df[domain_col].notna(), None)
    df[domain_col] = df[domain_col].map(lambda v: v.split("\n") if isinstance(v, str) else v)
    return df.explode(domain_col).reset_index(drop=True)


def domain_specificity(exploded, domain_col, term_col):
    """DS = (DF_TD/NumPat_TD) / (DF_All/NumPat_All), every count on the exploded frame."""
    num_all = len(exploded)
    df_all = Counter()
    df_td, num_td = defaultdict(Counter), Counter()
    for td, terms in zip(exploded[domain_col], exploded[term_col]):
        terms = set(terms) if isinstance(terms, (list, set)) else set()
        df_all.update(terms)
        if isinstance(td, str):
            num_td[td] += 1
            df_td[td].update(terms)
    ds = {(td, v): (k / num_td[td]) / (df_all[v] / num_all)
          for td in df_td for v, k in df_td[td].items()}
    return ds, {"DF_All": df_all, "DF_TD": df_td, "NumPat_TD": num_td, "NumPat_All": num_all}


def recency(exploded, domain_col, term_col, year_col, td, term, threshold_yr):
    """functionalityTTA.recency - implemented, never called."""
    sub = exploded[exploded[domain_col] == td]
    has = sub[term_col].map(lambda t: isinstance(t, list) and term in t)
    denom = int(has.sum())
    return int((has & (sub[year_col] > threshold_yr)).sum()) / denom if denom else float("nan")


RECENCY_SPECS = [
    ("referenced paper (comment :148)", "N = 15 years"),
    ("author's stated intent (same comment)", "Year > 2015"),
    ("implemented threshold_yr (:120)", "most recent 5% of the year span"),
    ("comment's stated threshold (:149)", "Recency > 0.7"),
    ("commented-out selection (:155)", "Recency > 0.9"),
]


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
    hit = sub[sub[term_col].map(lambda t: isinstance(t, list) and term in t)]
    cumulative = []
    for end in sorted(sub["Year"].unique()):
        f = hit[hit["Year"] <= end]
        cumulative.append((int(end), len(f), f["count_citing_patents"].sum(), f["avg citing patents"].sum()))
    windowed = []
    for a, b in _year_chunks(int(sub["Year"].min()), int(sub["Year"].max()), year_range):
        f = hit[(hit["Year"] >= a) & (hit["Year"] <= b)]
        windowed.append(("%d-%d" % (a, b), len(f), f["count_citing_patents"].sum(), f["avg citing patents"].sum()))
    return _fs_table(cumulative, "End Year"), _fs_table(windowed, "Year Range")


QUADRANTS = {("up", "up"): "Growing", ("up", "down"): "Generalizing",
             ("down", "down"): "Declining", ("down", "up"): "Repositioning"}


def read_quadrants(windowed):
    """The natural reading of the four names; the source never states it."""
    labels, rows = [], windowed.to_dict("records")
    for prev, cur in zip(rows, rows[1:]):
        d = "up" if cur["DF"] > prev["DF"] else "down" if cur["DF"] < prev["DF"] else "flat"
        f = "up" if cur["FS"] > prev["FS"] else "down" if cur["FS"] < prev["FS"] else "flat"
        labels.append((prev[windowed.columns[0]], cur[windowed.columns[0]],
                       QUADRANTS.get((d, f), "flat (%s DF, %s FS)" % (d, f))))
    return pd.DataFrame(labels, columns=["from", "to", "reading (assumed mapping)"])


def _domain_frame(state, td, term_col):
    ex = state["exploded"]
    sub = ex[ex[state["columns"]["domain"]] == td].copy()
    if sub.empty:
        raise SystemExit("domain %r not found. Domains: %s" % (td, ", ".join(map(str, state["domains"][:20]))))
    sub["count_citing_patents"] = sub[state["columns"]["citing"]].map(count_citing_patents) \
        if state["columns"]["citing"] else 0
    return sub


def tta_function(state, td, verb, year_range=3):
    sub = _domain_frame(state, td, "filtered_tech_verbs")
    cum, win = trend_tables(sub, "filtered_tech_verbs", verb, state["max_year"], year_range)
    return {"domain": td, "term": verb, "cumulative": cum, "windowed": win, "reading": read_quadrants(win)}


def tta_technology(state, td, key, year_range=3):
    sub = _domain_frame(state, td, "filtered_tech_keys")
    cum, win = trend_tables(sub, "filtered_tech_keys", key, state["max_year"], year_range)
    return {"domain": td, "term": key, "cumulative": cum, "windowed": win, "reading": read_quadrants(win)}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def report(path, *, threshold=75, min_keys=5, no_keywords=7, ds_threshold=5, relationship="Inclusion",
           cache_bug=True, show=25, **columns):
    """Run the replica and print what the shipped program would have shown.
    threshold is the inverse similarity threshold (mean rank; lower = more
    similar; UI default 75), min_keys the floor (UI default 5)."""
    warnings = []
    frame, cm = load(path, **columns)
    print("TRACK A - historical replica (defects preserved; cache_bug=%s)" % cache_bug)
    print("COLUMNS  " + "  ".join("%s=%r" % kv for kv in cm.items()))
    keep = frame[frame[cm["abstract"]].notna()]
    if cm["pubno"]:
        keep = keep[keep[cm["pubno"]].notna()]
    keep = keep.reset_index(drop=True).copy()
    abstracts = [clean_text(a) for a in keep[cm["abstract"]]]
    pubnos = list(keep[cm["pubno"]]) if cm["pubno"] else ["row%d" % i for i in range(len(keep))]
    years = [year_original(v) for v in keep[cm["date"]]] if cm["date"] else [None] * len(keep)
    lex = verb_lexicon((list(keep[cm["description"]]) if cm["description"] else []) + abstracts)
    print("CORPUS   rows=%d usable=%d years=%s..%s verb lemmas (heuristic)=%d"
          % (len(frame), len(keep), min((y for y in years if y), default=None),
             max((y for y in years if y), default=None), len(set(lex.values()))))

    chosen, flat = keywords_per_document(abstracts, lex, no_keywords)
    ctx = _context_matrix(abstracts, list(dict.fromkeys(flat)))
    keydict = map_primary_to_secondary(flat, threshold, ctx, min_keys, cache_bug)
    df_trt = trt_frame(abstracts, pubnos, lex)
    dfmap = keywordmap(df_trt, flat, keydict, warnings)
    print("KEYPHRASES  %d per abstract, %d total, %d unique (2-4 words; 1-grams excluded)"
          % (no_keywords, len(flat), len(set(flat))))
    print("DICTIONARY  %d primaries at mean rank <= %d of %d (= rank %.1f of this corpus's %d words), "
          "floor %d per primary (greedy, order-dependent)"
          % (len(keydict), threshold, BERT_VOCAB, max(1.0, threshold * len(ctx[0]) / BERT_VOCAB),
             len(ctx[0]), min_keys))
    forced = sum(1 for k, v in keydict.items() if len(v) == min_keys)
    print("            %d groups are exactly min_keys wide: the floor, not the threshold, chose them" % forced)
    for k in sorted(keydict, key=lambda k: -len(keydict[k]))[:show]:
        print("  %-36s <- %s" % (k[:36], ", ".join(keydict[k])[:90]))
    print("TRIPLES     %d (T1, prep, T2); mapped rows by class:" % len(df_trt))
    for r in RELATIONSHIPS:
        print("  %-10s %d" % (r, int((dfmap["Prep"] == r).sum())))
    g = graph({"dfmap": dfmap}, relationship)
    print("GRAPH [%s]  nodes=%d directed pairs=%d -> undirected edges=%d (reverse pairs collapsed=%d, "
          "occurrences discarded=%d)" % (relationship, g["nodes"], g["directed_pairs"], g["undirected_edges"],
                                          g["reverse_pairs_collapsed"], g["occurrences_discarded"]))

    desc_col = cm["description"]
    if desc_col is None:
        warnings.append("no description column; TTA1 verbs extracted from abstracts instead.")
    texts = keep[desc_col] if desc_col else keep[cm["abstract"]]
    keep["Year"] = years
    keep["Verbs"] = [sorted({lex[w] for w in _WORD.findall(str(t).lower()) if w in lex}) for t in texts]
    keep["cleaned_abstracts"] = abstracts
    uniq = list(dict.fromkeys(flat))
    keep["filtered_tech_keys"] = [[k for k in uniq if k in a.lower()] for a in abstracts]
    if cm["domain"] is None:
        warnings.append("no technology-domain column; DS and both TTA tabs are unavailable.")
        exploded, ds, counts, tech_verbs, domains = keep, {}, {}, [], []
    else:
        exploded = explode_domains(keep, cm["domain"])
        ds, counts = domain_specificity(exploded, cm["domain"], "Verbs")
        tech = defaultdict(set)
        for (td, v), val in ds.items():
            if val >= ds_threshold:
                tech[td].add(v)
        exploded["filtered_tech_verbs"] = [sorted(v for v in vs if v in tech.get(td, ()) and v.isalnum())
                                           for td, vs in zip(exploded[cm["domain"]], exploded["Verbs"])]
        tech_verbs = sorted({v for vs in exploded["filtered_tech_verbs"] for v in vs})
        domains = [d for d in exploded[cm["domain"]].dropna().unique()]
        print("TTA1        %d domains after explode (%d rows from %d patents); %d technical verbs at DS >= %s; "
              "recency: implemented, not applied" % (len(domains), len(exploded), len(keep), len(tech_verbs), ds_threshold))
        for td in domains[:show]:
            top = sorted(((ds[(td, v)], v) for v in tech.get(td, ())), reverse=True)[:8]
            print("  %-28s n=%-4d %s" % (str(td)[:28], counts["NumPat_TD"][td],
                                         ", ".join("%s(%.1f)" % (v, s) for s, v in top)))
    ys = [y for y in years if y]
    max_year = max(ys) if ys else None
    if max_year is None:
        warnings.append("no usable application year; TTA trend tables cannot be produced.")
    for w in warnings:
        print("WARNING  " + w)
    print("NOTE  Defects 1-7, 12, 17, 19, 20 and the post-explode denominator are live in every number "
          "above. Report them; do not repair them in prose.")
    return {"frame": frame, "columns": cm, "keep": keep, "abstracts": abstracts, "pubnos": pubnos,
            "years": years, "lex": lex, "keywords_per_doc": chosen, "keywords": flat, "keydict": keydict,
            "context": ctx, "df_trt": df_trt, "dfmap": dfmap, "exploded": exploded, "ds": ds,
            "ds_counts": counts, "tech_verbs": tech_verbs, "domains": domains, "max_year": max_year,
            "threshold": threshold, "min_keys": min_keys, "warnings": warnings}


def documents_for(state, phrase, limit=15):
    keep, cm = state["keep"], state["columns"]
    hits = [i for i, a in enumerate(state["abstracts"]) if phrase.lower() in a.lower()]
    cols = [c for c in (cm["pubno"], cm["title"], cm["date"]) if c]
    return keep.iloc[hits[:limit]][cols]


def cache_bug_effect(state):
    """Regroup with the cache defect switched off and report how many phrases
    change group. The shipped program cannot make this comparison; this
    replica can, and the size of the difference is the size of the defect."""
    kd_bug = state["keydict"]
    kd_fix = map_primary_to_secondary(state["keywords"], state["threshold"], state["context"],
                                      state["min_keys"], cache_bug=False)
    own = lambda kd: {m: k for k, ms in kd.items() for m in ms + [k]}
    a, b = own(kd_bug), own(kd_fix)
    moved = sum(1 for m in a if b.get(m) != a[m])
    print("CACHE DEFECT  with: %d groups; without: %d groups; %d of %d phrases change group"
          % (len(kd_bug), len(kd_fix), moved, len(a)))
    return {"with_bug": kd_bug, "without_bug": kd_fix, "moved": moved}
