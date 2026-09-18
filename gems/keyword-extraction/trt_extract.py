"""Candidate technology terms from a patent export, with nothing hardcoded
about the subject matter.

This replaces the extraction half of trt_keywords.py. The counting and the
charts in that module are unchanged and still do the work; only the choice of
which terms reach the menu is different.

Why it exists
-------------
The shipped extractor cut abstracts into runs of words between stopwords and
ranked the runs by how many patents contained them. Measured against the local
tool on a 140-patent electrochemical corpus, 12 of its top 40 terms matched the
reference; the rest were drafting language - "form", "manufacturing", "amount",
"ratio", "group", "making", "range". Two causes, both structural:

  * single words were allowed, where the local tool requires at least two;
  * ranking was by raw document frequency, where the local tool ranks each
    abstract's phrases by salience and pools the per-abstract winners.

Raw document frequency is a popularity measure, and in patent prose the most
popular phrases are the ones every patent says.

What this does instead
----------------------
Three corpus-internal signals, no word lists beyond closed-class grammar:

  * C-value (Frantzi and Ananiadou) - termhood for multi-word terms. A phrase
    scores for its own frequency, minus the frequency it owes to longer
    phrases that contain it, weighted by length. "electrode assembly" earns
    its own score; "electrode" does not inherit it.
  * Cohesion - how informative the phrase's words are together, so
    "current collector" outranks "resulting mixture", whose words merely
    happen to be adjacent.
  * Domain lift - if the export has a technology-domain column, how much more
    often the term appears in its strongest domain than across the corpus.
    This is the same ratio trt-pb's own trend analysis applies to verbs.

    It is REPORTED BUT NOT RANKED ON. Measured against the reference on the
    140-patent corpus, ranking on c-value x cohesion agreed on 24 of the top
    40; multiplying by lift dropped that to 12, and to 21 at best with the
    lift shrunk toward 1. With 29 domains over 140 patents most domains are
    tiny, so a term in two patents that share a domain scores a lift above 20
    on two observations. The column stays on screen because it is a useful
    thing for an analyst to see next to a term; it is not evidence enough to
    order the menu with.

The only fixed vocabulary is CLOSED_CLASS: determiners, prepositions,
conjunctions, pronouns and auxiliaries. That is English grammar and it does not
change between corpora. There is deliberately no list of generic nouns -
"apparatus", "method", "system" and their kind are demoted when c-value finds
they earn their frequency only inside longer phrases, and otherwise they are
the model's to drop.

The model's part
----------------
`prompt_for(state)` emits one block per abstract listing that abstract's
candidates. The model picks the technical terms from each block and nothing
else - it selects, it never invents, so a term the corpus does not contain
cannot enter the menu. That is the contract KeyBERT had: the part-of-speech
vectoriser proposed, the embedding chose. `apply_selection(state, text)` reads
the model's answer back and rebuilds the menu.

Used without the model stage, the code ranking alone stands on its own.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import pandas as pd

# English closed-class words. Grammar, not subject matter: this list is the
# same for a corpus about batteries and one about crop rotation.
CLOSED_CLASS = frozenset("""
a an the this that these those his her its their our your my some any each
every all both either neither no none one another other same such
and or but nor so yet for if then than because while although though whereas
unless until when where whether after before since during as
also too just only very much many few several more most less least
again once still yet ever never always often

of in on at by to from with without within into onto upon through across
against among between over under above below about around near per via
is are was were be been being am do does did doing have has had having
can could may might must shall should will would
not it they them he she we us you i there here what which who whom whose how
said wherein whereby thereof therein thereto thereby hereof herein
""".split())

_WORD = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")
_JP = re.compile(
    r"\b(?:PROBLEM TO BE SOLVED|SOLUTION|SELECTED DRAWING|ADVANTAGE|EFFECT)\s*:\s*",
    re.IGNORECASE)
_WS = re.compile(r"\s+")

HINTS = {
    "abstract": ("abstract", "summary"),
    "title": ("title",),
    "date": ("application date", "application dates", "priority date",
             "filing date", "publication date", "date", "year"),
    "pubno": ("publication number", "publication numbers", "pubno", "pub no", "id"),
    "domain": ("technology domain", "technology domains", "domain", "cpc", "ipc"),
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


def clean(text):
    if not isinstance(text, str):
        return ""
    out = re.sub(r"^\s*\([^)]*\)\s*\n", "", text)
    out = _JP.sub("", out)
    out = out.replace(";", ".").replace("–", "-").replace("—", "-")
    return _WS.sub(" ", out).strip()


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
    if not words:
        return ""
    return " ".join(words[:-1] + [_singular(words[-1])])


def _runs(text):
    """Maximal runs of open-class words, as word lists.

    Splitting on closed-class words and punctuation approximates the noun-
    phrase chunk the local tool got from a part-of-speech pattern. It is an
    approximation: it keeps verbs and participles, which the scoring below is
    responsible for demoting rather than a hand-kept list of them.
    """
    out = []
    for sentence in re.split(r"[.!?,:;()\[\]/]", text.lower()):
        run = []
        for token in sentence.split() + [""]:
            m = _WORD.match(token)
            w = m.group(0).strip("-'") if m else ""
            if not w or w in CLOSED_CLASS or w.isdigit() or len(w) < 2:
                if run:
                    out.append(run)
                run = []
            else:
                run.append(w)
    return out


def _ngrams(runs, lo, hi):
    """Every lo..hi-word window inside every run, normalised."""
    for run in runs:
        for n in range(lo, hi + 1):
            for i in range(len(run) - n + 1):
                yield _normalise(" ".join(run[i:i + n]))


def _cvalue(freq, maxlen):
    """Frequency a phrase does not owe to the longer phrases that contain it."""
    hosts_of = defaultdict(list)
    by_len = defaultdict(list)
    for p in freq:
        by_len[len(p.split())].append(p)
    for n in range(2, maxlen + 1):
        for host in by_len.get(n, ()):
            words = host.split()
            for k in range(1, n):
                for i in range(n - k + 1):
                    sub = " ".join(words[i:i + k])
                    if sub != host and sub in freq:
                        hosts_of[sub].append(host)
    out = {}
    for p, f in freq.items():
        weight = math.log(len(p.split()) + 1, 2)
        hosts = hosts_of.get(p)
        if hosts:
            out[p] = weight * (f - sum(freq[h] for h in hosts) / len(hosts))
        else:
            out[p] = weight * f
    return out


def _cohesion(freq, unigram, total_words):
    """Mean self-information of the phrase's words. Rare words bind tighter."""
    out = {}
    for p in freq:
        scores = []
        for w in p.split():
            pw = unigram.get(w, 0) / total_words if total_words else 0.0
            if pw > 0:
                scores.append(-math.log(pw))
        out[p] = sum(scores) / len(scores) if scores else 0.0
    return out


def _domain_lift(per_doc, domains):
    """Strongest domain over-representation, the ratio trt-pb applies to verbs.

    lift = P(term | strongest domain) / P(term | corpus). A term spread evenly
    across every domain scores about 1 and is describing a patent, not a
    technology.
    """
    if not domains or all(not d for d in domains):
        return None
    n_docs = len(per_doc)
    by_domain = defaultdict(list)
    for i, ds in enumerate(domains):
        for d in ds:
            by_domain[d].append(i)
    df = Counter()
    for s in per_doc:
        df.update(s)
    out = {}
    for term, overall in df.items():
        p_all = overall / n_docs
        best = 0.0
        for idxs in by_domain.values():
            if len(idxs) < 3:
                continue
            hits = sum(1 for i in idxs if term in per_doc[i])
            if hits:
                best = max(best, (hits / len(idxs)) / p_all)
        out[term] = best or 1.0
    return out


def candidates(path, method="lsa", min_words=2, max_words=4, min_patents=2,
               abstract=None, title=None, date=None, domain=None):
    """Read the export and score every candidate term. Returns state.

    `method` picks the ranking: "runs" (no dependencies), "tfidf" or "lsa"
    (scikit-learn), or "df" (the shipped control). See METHODS.
    """
    if method not in METHODS:
        raise SystemExit("method must be one of %s" % ", ".join(sorted(METHODS)))
    frame = (pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls"))
             else pd.read_csv(path))
    cols = list(frame.columns)
    c_abs = _pick(cols, HINTS["abstract"], abstract)
    c_ttl = _pick(cols, HINTS["title"], title)
    c_dat = _pick(cols, HINTS["date"], date)
    c_pub = _pick(cols, HINTS["pubno"])
    c_dom = _pick(cols, HINTS["domain"], domain)
    if c_abs is None and c_ttl is None:
        raise SystemExit("no abstract or title column. Columns: %s"
                         % ", ".join(map(str, cols)))

    keep = (frame[frame[c_abs].notna()].reset_index(drop=True) if c_abs
            else frame.reset_index(drop=True))
    texts = [clean(("%s. %s" % (r[c_ttl], r[c_abs])) if c_ttl and c_abs
                   else (r[c_ttl] if c_ttl else r[c_abs]))
             for _, r in keep.iterrows()]
    pubs = ([str(v) for v in keep[c_pub]] if c_pub
            else ["row%d" % i for i in range(len(keep))])
    domains = ([[d.strip() for d in str(v).split("\n") if d.strip()]
                for v in keep[c_dom]] if c_dom else [[] for _ in texts])

    runs = [_runs(t) for t in texts]
    freq = Counter()
    per_doc = []
    unigram = Counter()
    total_words = 0
    for r in runs:
        grams = list(_ngrams(r, min_words, max_words))
        freq.update(grams)
        per_doc.append(set(grams))
        for run in r:
            unigram.update(_normalise(w) for w in run)
            total_words += len(run)

    df = Counter()
    for s in per_doc:
        df.update(s)
    freq = {p: f for p, f in freq.items() if df[p] >= min_patents}
    per_doc = [{p for p in s if p in freq} for s in per_doc]

    cval = _cvalue(freq, max_words)
    coh = _cohesion(freq, unigram, total_words)
    lift = _domain_lift(per_doc, domains)

    # Lift is deliberately absent from every score. See the module docstring.
    score = METHODS[method](texts, freq, per_doc, df, cval, coh)

    ranked = sorted(freq, key=lambda p: (-score[p], p))
    state = {"frame": keep, "texts": texts, "pubs": pubs, "domains": domains,
             "per_doc": per_doc, "df": df, "freq": freq, "score": score,
             "cvalue": cval, "cohesion": coh, "lift": lift, "ranked": ranked,
             "vocab": ranked, "selected": None, "method": method,
             "columns": {"title": c_ttl, "abstract": c_abs, "date": c_dat,
                         "pubno": c_pub, "domain": c_dom}}
    print("CORPUS   %d rows, %d with text" % (len(frame), len(keep)))
    print("COLUMNS  title=%r abstract=%r date=%r id=%r domain=%r"
          % (c_ttl, c_abs, c_dat, c_pub, c_dom))
    print("TERMS    %d candidates of %d-%d words in >=%d patents, ranked by %r"
          % (len(freq), min_words, max_words, min_patents, method))
    if lift:
        print("SIGNALS  c-value, cohesion, domain lift over %d domains"
              % len({d for ds in domains for d in ds}))
    else:
        print("SIGNALS  c-value, cohesion (no domain column - lift unavailable)")
    return state


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def _score_runs(texts, freq, per_doc, df, cval, coh, **kw):
    """c-value x cohesion. No dependencies beyond the standard library."""
    return {p: max(cval[p], 0.0) * (1.0 + coh[p]) for p in freq}


def _score_tfidf(texts, freq, per_doc, df, cval, coh, **kw):
    """Highest tf-idf the phrase reaches in any single abstract.

    Closer to the reference than corpus-wide frequency, because the reference
    scores a phrase against the abstract it came from, not against the corpus.
    A phrase that dominates one patent survives even if it is rare overall.
    """
    if not HAVE_SKLEARN:
        return _score_runs(texts, freq, per_doc, df, cval, coh)
    vocab = sorted(freq)
    vec = TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 4), lowercase=True,
                          token_pattern=r"(?u)\b\w[\w-]*\b")
    matrix = vec.fit_transform(texts)
    peak = matrix.max(axis=0).toarray().ravel()
    return {p: float(peak[i]) for i, p in enumerate(vocab)}


def _score_lsa(texts, freq, per_doc, df, cval, coh, dims=128, per_doc_top=7, **kw):
    """The closest available stand-in for KeyBERT: cosine to the document.

    KeyBERT embeds the abstract and every candidate with one model and keeps
    the candidates nearest the abstract. `sentence_transformers` is not in the
    sandbox, but scikit-learn is, so the embedding here is tf-idf reduced by
    truncated SVD - latent semantic analysis. Same shape of computation, a
    weaker space: LSA has no subword knowledge and no pretraining, so it
    relates phrases that share distributional context and nothing more.

    Candidates are embedded as pseudo-documents of their own words, scored
    against each abstract they occur in, and the per-abstract winners are
    pooled - which is what `keywords_mpnet` does with `top_n`.
    """
    if not HAVE_SKLEARN:
        return _score_runs(texts, freq, per_doc, df, cval, coh)
    vocab = sorted(freq)
    word_vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w[\w-]*\b",
                               min_df=1, sublinear_tf=True)
    doc_terms = word_vec.fit_transform(texts)
    cand_terms = word_vec.transform(vocab)
    k = min(dims, max(2, min(doc_terms.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=0)
    doc_vecs = svd.fit_transform(doc_terms)
    cand_vecs = svd.transform(cand_terms)

    index = {p: i for i, p in enumerate(vocab)}
    pooled = Counter()
    best = {}
    for d, present in enumerate(per_doc):
        if not present:
            continue
        idx = [index[p] for p in present]
        sims = cosine_similarity(doc_vecs[d:d + 1], cand_vecs[idx]).ravel()
        ranked = sorted(zip(sims, present), key=lambda t: (-t[0], t[1]))
        for sim, phrase in ranked[:per_doc_top]:
            pooled[phrase] += 1
            best[phrase] = max(best.get(phrase, 0.0), float(sim))
    # Pooled count first, similarity as the tiebreak: the reference's menu is
    # "how many abstracts chose this phrase", not "how close was it once".
    return {p: pooled.get(p, 0) + min(best.get(p, 0.0), 0.999) for p in freq}


def _score_df(texts, freq, per_doc, df, cval, coh, **kw):
    """Raw document frequency - the shipped behaviour, kept as the control."""
    return {p: float(df[p]) for p in freq}


METHODS = {"runs": _score_runs, "tfidf": _score_tfidf,
           "lsa": _score_lsa, "df": _score_df}


def capabilities():
    """What this sandbox actually has. Print it before trusting any method."""
    import importlib
    out = {}
    for name in ("pandas", "numpy", "sklearn", "scipy", "spacy", "nltk",
                 "networkx", "sentence_transformers", "matplotlib", "plotly"):
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception:
            out[name] = None
    try:
        import spacy
        out["spacy_models"] = spacy.util.get_installed_models()
    except Exception:
        out["spacy_models"] = []
    return out


def report_capabilities():
    caps = capabilities()
    have = ", ".join("%s %s" % (k, v) for k, v in caps.items()
                     if v and k != "spacy_models")
    missing = ", ".join(k for k, v in caps.items() if not v and k != "spacy_models")
    print("ENV      %s" % have)
    if missing:
        print("MISSING  %s" % missing)
    print("SPACY    models installed: %s"
          % (caps["spacy_models"] or "none - tokenizer only, no tagger or lemmatizer"))
    print("METHODS  %s" % ("runs, tfidf, lsa, df" if HAVE_SKLEARN
                           else "runs, df (scikit-learn absent - tfidf and lsa fall back to runs)"))
    return caps


def menu(state, top=40):
    """The numbered candidate table."""
    rows = state["selected"] if state["selected"] is not None else state["ranked"]
    print("%-4s %-38s %6s %9s %7s" % ("#", "term", "pat", "score", "lift"))
    for i, p in enumerate(rows[:top], 1):
        print("%-4d %-38s %6d %9.1f %7s"
              % (i, p, state["df"][p], state["score"][p],
                 ("%.1f" % state["lift"][p]) if state["lift"] else "-"))
    return rows[:top]


def prompt_for(state, per_abstract=12, limit=None):
    """Emit one candidate block per abstract for the model to choose from."""
    order = state["ranked"]
    rank = {p: i for i, p in enumerate(order)}
    n = limit or len(state["texts"])
    print("SELECT the technical terms from each block. Reply with the block "
          "number and the terms you keep, one block per line, nothing else.")
    print("Keep a term when it names a technology, a material, a component or "
          "a measurable property. Drop it when it is patent drafting language, "
          "a bare activity, or a category name that would fit any patent.")
    print()
    for i in range(n):
        cands = sorted(state["per_doc"][i], key=lambda p: rank.get(p, 1 << 30))
        if cands:
            print("%d| %s" % (i + 1, "; ".join(cands[:per_abstract])))
    return n


def keep_only(state, text, min_patents=2):
    """The menu filter: keep the terms named, in the code ranking's order.

    Accepts a `KEEP: a; b; c` line, or any number of them, and ignores
    anything that is not a term the extractor already proposed - so a
    hallucinated phrase is dropped rather than admitted.
    """
    wanted = []
    for line in text.splitlines():
        body = line.split(":", 1)[1] if ":" in line else line
        for term in body.split(";"):
            term = _normalise(" ".join(term.lower().split()))
            if term and term in state["freq"] and term not in wanted:
                wanted.append(term)
    rejected = [t for t in text.replace("\n", ";").split(";")
                if _normalise(" ".join(t.lower().split())) not in state["freq"]
                and t.strip() and ":" not in t]
    order = {p: i for i, p in enumerate(state["ranked"])}
    chosen = [p for p in sorted(wanted, key=lambda p: order.get(p, 1 << 30))
              if state["df"][p] >= min_patents]
    state["selected"] = chosen
    state["per_doc_selected"] = [{p for p in s if p in set(chosen)}
                                 for s in state["per_doc"]]
    print("FILTER   %d terms kept, %d in >=%d patents%s"
          % (len(wanted), len(chosen), min_patents,
             (", %d not in the candidate list and ignored" % len(rejected))
             if rejected else ""))
    return chosen


def to_keyword_state(state, trt_keywords):
    """Hand the chosen terms to trt_keywords so its charts draw unchanged.

    trt_keywords.primary() and .secondary() need `years` and a `vocab` in menu
    order. Everything else it reads is already here, so this is a rename rather
    than a second pipeline: the counting and the six figures stay exactly as
    they were, and only the vocabulary reaching them has changed.
    """
    col = state["columns"]["date"]
    years = ([trt_keywords.year_of(v) for v in state["frame"][col]] if col
             else [None] * len(state["texts"]))
    chosen = state["selected"] if state["selected"] is not None else state["ranked"]
    order = {p: i for i, p in enumerate(chosen)}
    per_doc = (state.get("per_doc_selected") or state["per_doc"])
    kept = [{q for q in s if q in order} for s in per_doc]
    df = Counter()
    for s in kept:
        df.update(s)
    bridged = dict(state)
    bridged.update({"years": years, "vocab": chosen, "per_doc": kept, "df": df})
    return bridged


def apply_selection(state, text, min_patents=2):
    """Read the model's reply back and rebuild the ranking from what it kept."""
    doc_sets = [set() for _ in state["texts"]]
    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        head, _, body = line.partition("|")
        try:
            idx = int(head.strip().rstrip(".")) - 1
        except ValueError:
            continue
        if not 0 <= idx < len(doc_sets):
            continue
        for term in body.split(";"):
            term = _normalise(" ".join(term.lower().split()))
            if term and term in state["freq"]:
                doc_sets[idx].add(term)
    kept = Counter()
    for s in doc_sets:
        kept.update(s)
    chosen = [p for p, n in kept.items() if n >= min_patents]
    chosen.sort(key=lambda p: (-kept[p], -state["score"][p], p))
    state["selected"] = chosen
    state["selected_df"] = kept
    state["per_doc_selected"] = doc_sets
    print("SELECTION %d terms kept by the model, %d in >=%d patents"
          % (len(kept), len(chosen), min_patents))
    return chosen
