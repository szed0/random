"""Track A - behavioural replica of trt-pb, scoped per component.

Attach as a Gemini Gem knowledge file. Runs in the code-execution sandbox:
standard library, pandas and numpy only. No spaCy, no MPNet, no BERT.

What this file is
-----------------
The three-track plan keeps the shipped program as the control. This is that
control, built the way the analysis argued it should be: per component, cheap,
and with the Patent-BERT phrase-grouping path characterised rather than run.
The grouping baseline here is the cosine algorithm from bertui.py, which is the
one that works as designed. The DS / DF / FS / quadrant arithmetic, the
preposition triples, the six relationship classes and the tab-3 relevancy
buckets are reproduced exactly, including the defects the reference document
catalogues. Nothing is silently repaired. Where a defect fires, the report says
so - that is the point of a control.

Defects deliberately preserved (numbers are the reference document's):
  1  clean_text's boilerplate regex never fires (trailing \\b after the colon)
  2  keyword pattern lacks a non-capturing group, so 'sensor' matches 'biosensor'
  3  keyphrases enter the regex unescaped
  4  'includes' / 'utilizes' sit in the Inclusion list and can never match
  6  recency() exists and is never applied; selection is DS-only
  7  citing strings are split on every punctuation mark and fragments counted
 12  DS has no zero guard and no frequency floor
 17  the cumulative view is produced and cannot show decline
 19  graph is built undirected and rendered directed
 20  edge multiplicity is discarded
 open question 5: DS denominators are counted after the domain explode

Substitutions forced by the sandbox (each is a stand-in, not a fix):
  * spaCy ADP tagging  -> a fixed preposition list, with 'to' treated as an
    infinitive marker when the next word is a verb lemma
  * spaCy verb lemmas  -> a corpus-internal inflection test: a word is a verb
    lemma when at least two of its -s / -ed / -ing forms occur in the corpus
  * POS-pattern candidates + KeyBERT top-n -> stopword-segmented runs of 2-4
    words, ranked per abstract by tf-idf, top 7
  * MPNet cosine -> character-trigram cosine. Context-free, as MPNet is.
  * the MLM ranking path is not reproduced. cache_key_check() demonstrates
    the cache-key derivation that defeats it.

    from trt_replica import report
    state = report("patents.xlsx")
"""

from __future__ import annotations

import hashlib
import math
import random
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
    """Read the export and resolve the seven ORBIT columns. Never guesses silently:
    the column map is printed by report() and must be confirmed by the user."""
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
    """PatentInfo.clean_text, verbatim behaviour. The last substitution is
    defect 1: '\\b' after the colon needs a word character next and the real
    text has a space, so JPO boilerplate survives into every downstream step."""
    if not isinstance(text, str):
        text = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    cleaned = re.sub(r"\(.*\)\n", "", text)
    cleaned = cleaned.replace(";", ".").replace("\n", " ").replace("-", " ").replace(",", " ")
    return re.sub(r"\bPROBLEM TO BE SOLVED:\b", "", cleaned)


def year_original(value):
    """str(value)[:4] coerced to int, else None - datestrip_yr and the TTA
    classes' try/except, unchanged."""
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


# spaCy tags these ADP in patent prose. Fixed list stands in for the tagger.
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
    """Candidate (base, inflection class) pairs a surface form could come from."""
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
    """Corpus-internal stand-in for POS tagging.

    A base form is accepted as a verb lemma when at least two of its three
    inflection classes (-s, -ed, -ing, allowing e-drop, y->ie and consonant
    doubling) occur in the corpus. The base itself need not occur. Competing
    bases for one surface form are resolved by evidence, then by whether the
    base is attested, then by the e-final spelling. Returns {surface: lemma}.
    Deterministic, but a heuristic: 'process' and 'coat' come through as
    verbs, which spaCy would also do in many contexts; irregular verbs
    ('build'/'built') are missed. It is a substitute, and is reported as one.
    """
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
        # 'stored/storing' -> store, but 'boiled/boiling' -> boil: an e-final
        # base is preferred unless the stem ends in two vowels + consonant or
        # in a cluster that English verbs do not close with 'e'.
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
    if t == "to" and i + 1 < len(toks):
        nxt = toks[i + 1].lower()
        if lex.get(nxt) == nxt:       # infinitive marker: spaCy tags PART, not ADP
            return False
    return True


# --------------------------------------------------------------------------- #
# keyphrases - KeyBERT stand-in
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
    """The POS pattern is adjectives then nouns, so an inflected verb form
    cannot open or close a candidate. 'charging solid electrolyte' loses its
    first word; 'electrolyte arranged' loses its last. An interior '-ing'
    ('pipeline monitoring system') stays, as a tagger would usually keep it."""
    while run and lex.get(run[0], run[0]) != run[0]:
        run = run[1:]
    while run and lex.get(run[-1], run[-1]) != run[-1]:
        run = run[:-1]
    return run


def candidate_phrases(text, lex, lo=2, hi=4):
    """Maximal runs of content words between stopwords or punctuation, kept
    when 2-4 words long. Stands in for the POS-pattern vocabulary restricted to
    KeyBERT's (2,4) n-gram range; runs outside that range are dropped, which
    reproduces the 1-gram exclusion the original has."""
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
    """Top-n candidates per abstract by tf-idf. KeyBERT ranks by cosine to the
    document embedding; tf-idf is the deterministic stand-in. Returns one list
    per document and the flat list with duplicates, exactly as tab 1 builds it."""
    per_doc = [candidate_phrases(d, lex) for d in docs]
    df = Counter()
    for cands in per_doc:
        df.update(set(cands))
    n = len(docs)
    chosen, flat = [], []
    for cands in per_doc:
        tf = Counter(cands)
        scored = sorted(tf, key=lambda p: (-(tf[p] * (math.log((n + 1) / (df[p] + 1)) + 1)), p))
        top = scored[:no_keywords]
        chosen.append(top)
        flat.extend(top)
    return chosen, flat


# --------------------------------------------------------------------------- #
# TRT extraction - TRTExtraction.get_trt, token for token
# --------------------------------------------------------------------------- #

def get_trt(text, lex):
    """Every preposition yields (T1, prep, T2), each side walked outward until
    the next preposition. The walks chain, so one sentence gives a connected
    path. The backward walk uses Python's negative indexing when the
    preposition opens the sentence, exactly as the shipped loop does."""
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
# grouping - bertui.keywordsynonyms with a context-free similarity stand-in
# --------------------------------------------------------------------------- #

def _trigram_vectors(phrases):
    vocab = {}
    rows = []
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
    """bertui.py:87-100. Greedy, pop-first, order-dependent: phrase i claims
    every unclaimed phrase whose similarity clears the threshold, then all of
    them leave the pool. The primary sits in its own group (self-similarity is
    1). Shuffling `kws` changes the partition - see order_sensitivity()."""
    kws = list(dict.fromkeys(kws))          # duplicates are absorbed identically
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
# keyword mapping and relationship classes - KeywordMapping.py as inherited
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
    """The inherited construction: '\\b' + '|'.join(terms) + '\\b'. No escaping
    (defect 3) and no group (defect 2), so the boundaries bind only the first
    and last alternative. If a phrase breaks the regex the shipped program
    raised; here the failure is recorded and the escaped form used so the rest
    of the pipeline can be observed."""
    terms = list(dict.fromkeys(k for k in listkw if k))
    raw = r"\b" + "|".join(terms) + r"\b"
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error as exc:
        warnings.append("defect 3 fired: keyword regex failed to compile (%s); "
                        "the shipped program would have crashed here. Using an "
                        "escaped pattern for this run only." % exc)
        return re.compile(r"\b" + "|".join(re.escape(t) for t in terms) + r"\b", re.IGNORECASE)


def keywordmap(df_trt, listkw, keydict, warnings):
    pat = build_keyword_pattern(listkw, warnings)
    rows = []
    for pub, t1, prep, t2 in df_trt[["Publication number", "T1", "prep", "T2"]].itertuples(index=False):
        m1, m2 = pat.findall(str(t1)), pat.findall(str(t2))
        for a in m1:
            for b in m2:
                rows.append([a, b, classify_prep(prep), pub])
    dfmap = pd.DataFrame(rows, columns=["T1", "T2", "Prep", "Pubno."])
    # keydictmap: replace a term by the primary whose group lists it
    owner = {}
    for key, value in keydict.items():
        for v in value:
            owner.setdefault(v, key)
    dfmap["T1"] = dfmap["T1"].map(lambda t: owner.get(t, t))
    dfmap["T2"] = dfmap["T2"].map(lambda t: owner.get(t, t))
    return dfmap


def jsonlist(dfmap, relationship):
    """T1 -> unique T2s for one relationship. set() here is defect 20: the
    number of times a pair occurs is thrown away."""
    sub = dfmap[dfmap["Prep"] == relationship]
    out = {}
    for t1 in sub["T1"].unique():
        out[t1] = list(dict.fromkeys(sub[sub["T1"] == t1]["T2"]))
    return out


def graph(state, relationship="Inclusion"):
    """The nx.Graph built from jsonlist and rendered directed (defect 19).
    Reports how many directed pairs collapsed into one undirected edge and how
    much multiplicity was lost, since those are the numbers Track C must beat."""
    dfmap = state["dfmap"]
    jsont = jsonlist(dfmap, relationship)
    directed = {(a, b) for a, bs in jsont.items() for b in bs}
    undirected = {frozenset(e) for e in directed if len(set(e)) == 2}
    sub = dfmap[dfmap["Prep"] == relationship]
    rows = []
    for a, bs in jsont.items():
        pubs = []
        for b in bs:
            pubs.append("\n".join(sub[(sub["T1"] == a) & (sub["T2"] == b)]["Pubno."]))
        rows.append((a, ", ".join(bs), "\n\n".join(pubs)))
    techres = pd.DataFrame(rows, columns=["Technologynode1", "Technologynodes2", "Publication number"])
    return {"relationship": relationship, "adjacency": jsont,
            "nodes": len({n for e in directed for n in e}),
            "directed_pairs": len(directed), "undirected_edges": len(undirected),
            "reverse_pairs_collapsed": len(directed) - len(undirected),
            "occurrences_discarded": int(len(sub) - len(directed)),
            "techres": techres}


def edge_evidence(state, t1, t2, relationship=None):
    """Publication numbers behind an edge - the one export the original got right."""
    d = state["dfmap"]
    sub = d[(d["T1"] == t1) & (d["T2"] == t2)]
    if relationship:
        sub = sub[sub["Prep"] == relationship]
    return sub


# --------------------------------------------------------------------------- #
# tab 3 relevancy score - the twelve hardcoded buckets, string comparison
# --------------------------------------------------------------------------- #

_BUCKETS = [("1980", "1990"), ("1990", "2000"), ("2000", "2003"), ("2003", "2006"),
            ("2006", "2009"), ("2009", "2012"), ("2012", "2014"), ("2014", "2016"),
            ("2016", "2018"), ("2018", "2020"), ("2020", "2022"), None]
_BUCKET_LABELS = ["1980-1990", "1990-2000", "2000-2003", "2003-2006", "2006-2009",
                  "2009-2012", "2012-2014", "2014-2016", "2016-2018", "2018-2020",
                  "2020-2022", "2022-Present"]


def _bucket(yr):
    for i, b in enumerate(_BUCKETS):
        if b is None:
            return i
        if yr >= b[0] and yr < b[1]:
            return i
    return 11


def relevancy(state, node):
    """prominenttech2: share of abstracts in each bucket that contain the node.
    The year is a 4-character string compared lexically, and the node is used
    as an unescaped regular expression (the tab-3 sibling of defect 3). Years
    outside 1980-2021, including 'nan', fall into the open final bucket."""
    years = [str(y) if y is not None else "nan" for y in state["years"]]
    abstracts = state["cleaned_abstracts"]
    buckets = [_bucket(y) for y in years]
    try:
        pat = re.compile(node, re.IGNORECASE)
    except re.error:
        pat = re.compile(re.escape(node), re.IGNORECASE)
    rows = []
    for i, label in enumerate(_BUCKET_LABELS):
        idx = [j for j, b in enumerate(buckets) if b == i]
        hits = sum(1 for j in idx if pat.search(abstracts[j]))
        rows.append((label, round(hits / len(idx), 3) if idx else 0, hits, len(idx)))
    return pd.DataFrame(rows, columns=["Years", "Relevancy Score",
                                       "No. of patents it appears in",
                                       "No. of patents in this time-frame"])


# --------------------------------------------------------------------------- #
# TTA - functionalityTTA / technologyTTA arithmetic, unchanged
# --------------------------------------------------------------------------- #

def count_citing_patents(citing_string):
    """functionalityTTA.py:309-318. Splits on whitespace, punctuation, slashes
    and hyphens and counts the fragments. Defect 7: a number written
    US-2020/0123456-A1 becomes three fragments."""
    if not citing_string or not isinstance(citing_string, str):
        return 0
    patents = re.split(r"[\s;,.|/\-(){}[\]<>]+", citing_string)
    return len([p for p in patents if p])


def explode_domains(frame, domain_col):
    """str.split('\\n') then explode. No stripping: 'Sensors ' and 'Sensors'
    are two domains, and a patent in three domains becomes three rows that
    every corpus-wide count below then counts three times."""
    df = frame.copy()
    df[domain_col] = df[domain_col].astype(object).where(df[domain_col].notna(), None)
    df[domain_col] = df[domain_col].map(lambda v: v.split("\n") if isinstance(v, str) else v)
    df = df.explode(domain_col).reset_index(drop=True)
    return df


def domain_specificity(exploded, domain_col, term_col):
    """DS(TD, V) = (DF_TD/NumPat_TD) / (DF_All/NumPat_All), every count on the
    exploded frame (open question 5). Values are identical to the per-row
    recomputation in process1; only the cost differs (defect 13 is performance).
    Returns DS[(TD, term)] plus the counts, so the numbers can be shown."""
    num_all = len(exploded)
    df_all = Counter()
    df_td, num_td = defaultdict(Counter), Counter()
    for td, terms in zip(exploded[domain_col], exploded[term_col]):
        terms = set(terms) if isinstance(terms, (list, set)) else set()
        df_all.update(terms)
        if isinstance(td, str):
            num_td[td] += 1
            df_td[td].update(terms)
    ds = {}
    for td in df_td:
        for v, k in df_td[td].items():
            ds[(td, v)] = (k / num_td[td]) / (df_all[v] / num_all)
    return ds, {"DF_All": df_all, "DF_TD": df_td, "NumPat_TD": num_td, "NumPat_All": num_all}


def recency(exploded, domain_col, term_col, year_col, td, term, threshold_yr):
    """functionalityTTA.py:143-145 - implemented, never called (defect 6).
    Callable here so the five inconsistent window specs can be characterised."""
    sub = exploded[exploded[domain_col] == td]
    has = sub[term_col].map(lambda t: isinstance(t, list) and term in t)
    denom = int(has.sum())
    recent = int((has & (sub[year_col] > threshold_yr)).sum())
    return recent / denom if denom else float("nan")


RECENCY_SPECS = [
    ("referenced paper (comment :148)", "N = 15 years"),
    ("author's stated intent (same comment)", "Year > 2015"),
    ("implemented threshold_yr (:120)", "most recent 5% of the year span"),
    ("comment's stated threshold (:149)", "Recency > 0.7"),
    ("commented-out selection (:155)", "Recency > 0.9"),
]


def _year_chunks(min_year, max_year, year_range):
    """functionalityTTA.process3 bins. start_year = min_year - 1, so the first
    chunk holds an empty leading year and is one year narrower than the rest."""
    start = min_year - 1
    return [(start + i * year_range, min(start + (i + 1) * year_range - 1, max_year))
            for i in range((max_year - start) // year_range + 1)]


def _fs_table(rows, label):
    out = pd.DataFrame(rows, columns=[label, "DF", "Net Forward Citations", "Avg Forward Citations"])
    df = out["DF"].replace(0, np.nan)
    out["FS"] = (out["Avg Forward Citations"] / df).fillna(0)
    return out


def trend_tables(sub, term_col, term, max_year_overall, year_range=3):
    """FI (cumulative, all patents up to each year) and process3 (fixed-width
    chunks). avg citing = count / ((Y_max - Y_p) + 1): linear age
    normalisation, Y_max global. FS = sum(avg citing) / DF, a plain mean."""
    sub = sub[sub["Year"].notna()].copy()
    sub["Year"] = sub["Year"].astype(int)
    sub["avg citing patents"] = sub["count_citing_patents"] / ((max_year_overall - sub["Year"]) + 1)
    has = sub[term_col].map(lambda t: isinstance(t, list) and term in t)
    hit = sub[has]
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
    """The four names never map to plot regions in the source (open question
    2). This is the natural reading - DF on the y axis, FS on the x axis, each
    arrow between consecutive windows - and it is labelled as assumed."""
    labels = []
    rows = windowed.to_dict("records")
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
        raise SystemExit("domain %r not in the exploded frame. Domains: %s"
                         % (td, ", ".join(map(str, state["domains"][:20]))))
    sub["count_citing_patents"] = sub[state["columns"]["citing"]].map(count_citing_patents) \
        if state["columns"]["citing"] else 0
    return sub


def tta_function(state, td, verb, year_range=3):
    """TTA1 for one (domain, technical verb)."""
    sub = _domain_frame(state, td, "filtered_tech_verbs")
    cum, win = trend_tables(sub, "filtered_tech_verbs", verb, state["max_year"], year_range)
    return {"domain": td, "term": verb, "cumulative": cum, "windowed": win,
            "reading": read_quadrants(win)}


def tta_technology(state, td, key, year_range=3):
    """TTA2 for one (domain, keyphrase). Same arithmetic, matched by substring
    against the cleaned abstract rather than the description."""
    sub = _domain_frame(state, td, "filtered_tech_keys")
    cum, win = trend_tables(sub, "filtered_tech_keys", key, state["max_year"], year_range)
    return {"domain": td, "term": key, "cumulative": cum, "windowed": win,
            "reading": read_quadrants(win)}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def report(path, *, threshold=0.80, no_keywords=7, ds_threshold=5, relationship="Inclusion",
           show=25, **columns):
    """Run the replica and print what the shipped program would have shown.
    Returns the state for follow-up calls (graph, tta_function, tta_technology,
    relevancy, edge_evidence, documents_for, and the diagnostics)."""
    warnings = []
    frame, cm = load(path, **columns)
    print("TRACK A - behavioural replica (defects preserved)")
    print("COLUMNS  " + "  ".join("%s=%r" % kv for kv in cm.items()))

    # ---- shared prep: abstracts, pubnos, years (as tab 1 / tab 3 do it)
    keep = frame[frame[cm["abstract"]].notna()]
    if cm["pubno"]:
        keep = keep[keep[cm["pubno"]].notna()]
    keep = keep.reset_index(drop=True)
    abstracts = [clean_text(a) for a in keep[cm["abstract"]]]
    pubnos = list(keep[cm["pubno"]]) if cm["pubno"] else ["row%d" % i for i in range(len(keep))]
    years = [year_original(v) for v in keep[cm["date"]]] if cm["date"] else [None] * len(keep)
    lex = verb_lexicon((list(keep[cm["description"]]) if cm["description"] else []) + abstracts)
    print("CORPUS   rows=%d usable=%d years=%s..%s verb lemmas (heuristic)=%d"
          % (len(frame), len(keep), min((y for y in years if y), default=None),
             max((y for y in years if y), default=None), len(set(lex.values()))))

    # ---- tab 1: keyphrases, dictionary, triples, mapping
    chosen, flat = keywords_per_document(abstracts, lex, no_keywords)
    keydict = keywordsynonyms(flat, threshold)
    df_trt = trt_frame(abstracts, pubnos, lex)
    dfmap = keywordmap(df_trt, flat, keydict, warnings)
    print("KEYPHRASES  %d per abstract, %d total, %d unique (2-4 words; 1-grams excluded)"
          % (no_keywords, len(flat), len(set(flat))))
    print("DICTIONARY  %d primary groups at similarity >= %.2f (greedy, order-dependent)"
          % (len(keydict), threshold))
    for k in sorted(keydict, key=lambda k: -len(keydict[k]))[:show]:
        print("  %-36s <- %s" % (k[:36], ", ".join(v for v in keydict[k] if v != k)[:90]))
    print("TRIPLES     %d (T1, prep, T2) from %d abstracts; mapped rows by class:"
          % (len(df_trt), len(abstracts)))
    for r in RELATIONSHIPS:
        print("  %-10s %d" % (r, int((dfmap["Prep"] == r).sum())))
    g = graph({"dfmap": dfmap}, relationship)
    print("GRAPH [%s]  nodes=%d directed pairs=%d -> undirected edges=%d "
          "(reverse pairs collapsed=%d, occurrences discarded=%d)"
          % (relationship, g["nodes"], g["directed_pairs"], g["undirected_edges"],
             g["reverse_pairs_collapsed"], g["occurrences_discarded"]))

    # ---- TTA1: verbs, explode, DS
    desc_col = cm["description"]
    if desc_col is None:
        warnings.append("no description column; TTA1 verbs extracted from abstracts instead "
                        "(the original reads 'English description').")
    texts = keep[desc_col] if desc_col else keep[cm["abstract"]]
    keep = keep.copy()
    keep["Year"] = years
    keep["Verbs"] = [sorted({lex[w] for w in _WORD.findall(str(t).lower()) if w in lex})
                     for t in texts]
    keep["cleaned_abstracts"] = abstracts
    uniq_keys = list(dict.fromkeys(flat))
    keep["filtered_tech_keys"] = [[k for k in uniq_keys if k.lower() in a.lower()] for a in abstracts]
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
        exploded["Tech_Verbs"] = [sorted(v for v in vs if v in tech.get(td, ()))
                                  for td, vs in zip(exploded[cm["domain"]], exploded["Verbs"])]
        tech_verbs = sorted({v for vs in exploded["Tech_Verbs"] for v in vs if v.isalnum()})
        exploded["filtered_tech_verbs"] = [[v for v in vs if v in tech_verbs] for vs in exploded["Verbs"]]
        domains = [d for d in exploded[cm["domain"]].dropna().unique()]
        print("TTA1        %d domains after explode (%d rows from %d patents); "
              "%d technical verbs at DS >= %s; recency: not applied (defect 6)"
              % (len(domains), len(exploded), len(keep), len(tech_verbs), ds_threshold))
        for td in domains[:show]:
            top = sorted(((ds[(td, v)], v) for v in tech.get(td, ())), reverse=True)[:8]
            print("  %-28s n=%-4d %s" % (str(td)[:28], counts["NumPat_TD"][td],
                                         ", ".join("%s(%.1f)" % (v, s) for s, v in top)))
    years_ok = [y for y in years if y]
    max_year = max(years_ok) if years_ok else None
    if max_year is None:
        warnings.append("no usable application year; TTA trend tables cannot be produced.")

    for w in warnings:
        print("WARNING  " + w)
    print("NOTE  Defects 1, 2, 3, 4, 6, 7, 12, 17, 19, 20 and the post-explode DS "
          "denominator are live in every number above. Do not repair them in prose; "
          "report them.")
    return {"frame": frame, "columns": cm, "keep": keep, "abstracts": abstracts,
            "cleaned_abstracts": abstracts, "pubnos": pubnos, "years": years, "lex": lex,
            "keywords_per_doc": chosen, "keywords": flat, "keydict": keydict,
            "df_trt": df_trt, "dfmap": dfmap, "exploded": exploded, "ds": ds,
            "ds_counts": counts, "tech_verbs": tech_verbs, "domains": domains,
            "max_year": max_year, "threshold": threshold, "warnings": warnings}


def documents_for(state, phrase, limit=15):
    """Records whose cleaned abstract contains the phrase (case-insensitive)."""
    keep, cm = state["keep"], state["columns"]
    hits = [i for i, a in enumerate(state["abstracts"]) if phrase.lower() in a.lower()]
    cols = [c for c in (cm["pubno"], cm["title"], cm["date"]) if c]
    return keep.iloc[hits[:limit]][cols]


# --------------------------------------------------------------------------- #
# label-free measurements (analysis document, section 6.2)
# --------------------------------------------------------------------------- #

def adjusted_rand(a, b):
    """Adjusted Rand index between two labelings given as dicts item -> label."""
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


def _partition(keydict):
    return {m: k for k, ms in keydict.items() for m in ms}


def order_sensitivity(state, runs=20, seed=0):
    """Item 1. Shuffle the keyphrase list, regroup, compare partitions pairwise
    by adjusted Rand index. Needs no ground truth. A stable algorithm scores
    near 1; the greedy one does not."""
    kws = list(dict.fromkeys(state["keywords"]))
    sim = similarity_matrix(kws)
    parts = []
    rng = random.Random(seed)
    for _ in range(runs):
        order = list(range(len(kws)))
        rng.shuffle(order)
        kd = keywordsynonyms([kws[i] for i in order], state["threshold"], sim[np.ix_(order, order)])
        parts.append(_partition(kd))
    aris = [adjusted_rand(parts[i], parts[j]) for i in range(runs) for j in range(i + 1, runs)]
    sizes = [len(set(p.values())) for p in parts]
    out = {"runs": runs, "ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)),
           "groups_min": min(sizes), "groups_max": max(sizes)}
    print("ORDER SENSITIVITY  %d shuffles: ARI mean=%.3f min=%.3f; group count %d..%d"
          % (runs, out["ari_mean"], out["ari_min"], out["groups_min"], out["groups_max"]))
    return out


def permutation_null(state, td, term, year_range=3, runs=200, seed=0, term_col="filtered_tech_verbs"):
    """Item 2. Keep every window's DF fixed and shuffle which patents in the
    window carry the term; recompute FS; tabulate the quadrant reading. Under
    this null the DF axis is unchanged, so what it measures is whether the FS
    movement is distinguishable from picking DF patents at random."""
    sub = _domain_frame(state, td, term_col)
    sub = sub[sub["Year"].notna()].copy()
    sub["Year"] = sub["Year"].astype(int)
    sub["avg"] = sub["count_citing_patents"] / ((state["max_year"] - sub["Year"]) + 1)
    chunks = _year_chunks(int(sub["Year"].min()), int(sub["Year"].max()), year_range)
    has = sub[term_col].map(lambda t: isinstance(t, list) and term in t).to_numpy()
    avg = sub["avg"].to_numpy()
    win_idx = [np.where((sub["Year"] >= a) & (sub["Year"] <= b))[0] for a, b in chunks]
    dfs = [int(has[idx].sum()) for idx in win_idx]
    obs_fs = [avg[idx][has[idx]].mean() if dfs[i] else 0.0 for i, idx in enumerate(win_idx)]
    rng = np.random.default_rng(seed)
    null_fs = np.zeros((runs, len(chunks)))
    for r in range(runs):
        for i, idx in enumerate(win_idx):
            if dfs[i] and len(idx):
                null_fs[r, i] = avg[rng.choice(idx, size=min(dfs[i], len(idx)), replace=False)].mean()
    labels = Counter()
    for r in range(runs):
        for i in range(1, len(chunks)):
            d = "up" if dfs[i] > dfs[i - 1] else "down" if dfs[i] < dfs[i - 1] else "flat"
            f = "up" if null_fs[r, i] > null_fs[r, i - 1] else "down" if null_fs[r, i] < null_fs[r, i - 1] else "flat"
            labels[QUADRANTS.get((d, f), "flat")] += 1
    bands = []
    for i in range(1, len(chunks)):
        delta = null_fs[:, i] - null_fs[:, i - 1]
        bands.append(("%d-%d -> %d-%d" % (chunks[i - 1] + chunks[i]), obs_fs[i] - obs_fs[i - 1],
                      float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))))
    table = pd.DataFrame(bands, columns=["movement", "observed dFS", "null 2.5%", "null 97.5%"])
    total = sum(labels.values()) or 1
    print("PERMUTATION NULL  %s / %s, %d runs: %s"
          % (td, term, runs, ", ".join("%s %.0f%%" % (k, 100 * v / total) for k, v in labels.most_common())))
    print(table.to_string(index=False))
    return {"null_labels": dict(labels), "bands": table, "observed_fs": obs_fs, "df": dfs}


def funnel(state, td, window, term_col="filtered_tech_verbs"):
    """Item 3. FS against DF for every term in one window, with the analytic
    band mu +/- 1.96 sigma / sqrt(DF). Points inside the band carry no signal
    beyond their sample size."""
    sub = _domain_frame(state, td, term_col)
    sub = sub[sub["Year"].notna()].copy()
    sub["Year"] = sub["Year"].astype(int)
    sub = sub[(sub["Year"] >= window[0]) & (sub["Year"] <= window[1])]
    sub["avg"] = sub["count_citing_patents"] / ((state["max_year"] - sub["Year"]) + 1)
    mu, sigma = sub["avg"].mean(), sub["avg"].std(ddof=0)
    rows = []
    for term in sorted({t for ts in sub[term_col] for t in (ts or [])}):
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
    """Item 4. Median forward-citation count by application-year cohort; the
    plateau is the largest cohort median (3-cohort smoothed) and H is the
    smallest age at which a cohort reaches frac of it."""
    cm, keep = state["columns"], state["keep"]
    if not cm["citing"] or state["max_year"] is None:
        print("CENSORING  needs a citing column and years")
        return None
    c = keep[cm["citing"]].map(count_citing_patents)
    tab = pd.DataFrame({"year": keep["Year"], "c": c}).dropna()
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
    print("CENSORING  plateau median=%.1f; H=%s years (%s of plateau)"
          % (plateau, H, frac))
    return {"H": H, "table": out}


def ds_bias(state):
    """Item 5. DS_computed = DS_true * (Dbar / Dbar_V): the shipped denominator
    inflates every verb whose patents carry fewer domain labels than average.
    Reports that ratio for every surviving technical verb."""
    cm = state["columns"]
    keep, ex = state["keep"], state["exploded"]
    if not cm["domain"]:
        return None
    ndom = keep[cm["domain"]].map(lambda v: len(v.split("\n")) if isinstance(v, str) else 0)
    dbar = ndom.mean()
    rows = []
    for v in state["tech_verbs"]:
        has = keep["Verbs"].map(lambda vs: v in vs)
        if has.any():
            rows.append((v, ndom[has].mean(), dbar / ndom[has].mean()))
    out = pd.DataFrame(rows, columns=["verb", "Dbar_V", "DS inflation (Dbar/Dbar_V)"])
    if len(out):
        q = out["DS inflation (Dbar/Dbar_V)"].quantile([0.1, 0.5, 0.9])
        print("DS BIAS  Dbar=%.2f domains/patent; inflation ratio p10=%.2f median=%.2f p90=%.2f "
              "over %d verbs" % (dbar, q.iloc[0], q.iloc[1], q.iloc[2], len(out)))
    return out.sort_values("DS inflation (Dbar/Dbar_V)", ascending=False)


def window_sensitivity(state, td, widths=(2, 3, 4, 5), term_col="filtered_tech_verbs"):
    """Item 6. Headline reading (last arrow) for every term at each width; the
    flip rate is the share of terms whose reading is not the same at all widths."""
    sub = _domain_frame(state, td, term_col)
    terms = sorted({t for ts in sub[term_col] for t in (ts or [])})
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
        print("WINDOW SENSITIVITY  %s: %d terms, flip rate %.0f%%"
              % (td, len(out), 100 * out["flips"].mean()))
    return out


def cache_key_check(state, primaries=None):
    """Item 7, as far as the sandbox allows. The shipped key is
    sha256(','.join(flattened abstract token ids)); it does not include the
    masked positions. Derive it for two different primaries and show the
    collision. Confirming the end-to-end effect needs the SavedModel."""
    ids = [str(abs(hash(w)) % 30000) for a in state["abstracts"] for w in a.lower().split()]
    key = hashlib.sha256(",".join(ids).encode()).hexdigest()
    prim = primaries or list(state["keydict"])[:2]
    paths = {p: "mlm_logits/logits_%s.pkl" % key for p in prim}
    print("CACHE KEY  %s" % "; ".join("%s -> %s" % (p, v[:28] + "...") for p, v in paths.items()))
    print("           identical for every primary: each iteration after the first reads "
          "logits produced by masking a different phrase (defect 5).")
    return paths


def citation_format_check(state, n=5):
    """Item 8. Show raw citing strings and compare the shipped fragment count
    with a count that splits only on newline, semicolon, pipe and comma."""
    cm = state["columns"]
    if not cm["citing"]:
        print("CITATIONS  no citing column")
        return None
    raw = state["keep"][cm["citing"]].dropna().astype(str)
    raw = raw[raw.str.strip() != ""]
    shipped = raw.map(count_citing_patents)
    line = raw.map(lambda s: len([p for p in re.split(r"[\n;|,]+", s) if p.strip()]))
    ratio = (shipped.sum() / line.sum()) if line.sum() else float("nan")
    print("CITATIONS  %d non-empty strings; shipped count / line count = %.2f"
          % (len(raw), ratio))
    for s in raw.head(n):
        print("  %r -> shipped %d, lines %d" % (s[:70], count_citing_patents(s),
                                               len([p for p in re.split(r"[\n;|,]+", s) if p.strip()])))
    if ratio > 1.05:
        print("  defect 7 fires on this export: every FS is inflated by about %.2fx." % ratio)
    return {"ratio": ratio, "shipped": shipped, "lines": line}
