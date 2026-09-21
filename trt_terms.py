"""Technical terms out of a patent export, with the patents behind every one.

Three calls, and every number traces back to a publication number.

    state = extract("export.xlsx")          -> technical_terms.csv
    drill(state, 12)                        -> subterms_<term>.csv
    trace(state, "hydrogen leakage")        -> which patent, and the sentence

What "technical term" means here
--------------------------------
A run of open-class words, two to four long, that survives three filters:

  * **Stemmed.** "hydrogen leakage reductions", "hydrogen leakage reduction"
    and "hydrogen leak reductions" collapse to one row. Stemming is the
    grouping key only - the row is displayed under the surface form that
    occurs most often, so you read "hydrogen leakage reduction" and never
    "hydrogen leakag reduct".
  * **No plural noise.** Falls out of stemming: a term and its plural cannot
    appear as two rows.
  * **No gerund heads.** "reducing hydrogen leakage" is an activity, not a
    technology, so a phrase whose last word ends in -ing is dropped.

    Note what this costs: "coating", "housing", "bearing", "casing" and
    "winding" are ordinary technical nouns and they go too. `drop_gerunds`
    turns the rule off if your corpus is one where that matters.

The only fixed vocabulary is CLOSED_CLASS - determiners, prepositions,
conjunctions, pronouns, auxiliaries, degree adverbs. That is English grammar,
identical for a corpus about batteries and one about crop rotation. Nothing
about the subject matter is hardcoded.

Columns in technical_terms.csv
------------------------------
  term            the display form
  stem            the grouping key
  tfidf           sum of term frequency across the corpus x ln(N / df)
  score           c-value x cohesion - see `_cvalue` and `_cohesion`
  n_patents       how many patents contain it
  patents         every publication number it appears in, space separated

Runs on the standard library plus pandas. nltk's PorterStemmer is used when
present - it needs no downloaded data - and a built-in suffix stemmer stands
in when it is not.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import pandas as pd

CLOSED_CLASS = frozenset("""
a an the this that these those his her its their our your my some any each
every all both either neither no none one another other same such
and or but nor so yet for if then than because while although though whereas
unless until when where whether after before since during as
also too just only very much many few several more most less least
again once still ever never always often
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
_SENT = re.compile(r"(?<=[.!?])\s+")

HINTS = {
    "abstract": ("abstract", "summary"),
    "title": ("title",),
    "description": ("description", "english description", "detail", "claims"),
    "pubno": ("publication number", "publication numbers", "pubno", "pub no",
              "patent number", "patent id", "id"),
}


# --------------------------------------------------------------------------- #
# stemming
# --------------------------------------------------------------------------- #

try:
    from nltk.stem import PorterStemmer as _Porter
    _porter = _Porter()

    def stem(word):
        return _porter.stem(word)

    STEMMER = "nltk PorterStemmer"
except Exception:
    def stem(word):
        """Suffix stripping, in the order the suffixes nest.

        Stands in for Porter when nltk is absent. Weaker, and enough for the
        job here, which is only to make a term and its inflections one row.
        """
        w = word
        for suf, rep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"),
                         ("shes", "sh"), ("xes", "x"), ("zes", "z")):
            if w.endswith(suf) and len(w) > len(suf) + 1:
                w = w[:-len(suf)] + rep
                break
        else:
            if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
                w = w[:-1]
        for suf in ("ization", "isation", "ation", "ement", "ment", "ance",
                    "ence", "ity", "ness", "ing", "ed"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[:-len(suf)]
                break
        return w

    STEMMER = "built-in suffix stemmer (nltk absent)"


def stem_phrase(phrase):
    """The grouping key. Hyphens are dropped before stemming, so
    "non-aqueous electrolyte" and "nonaqueous electrolyte" are one row."""
    return " ".join(stem(w.replace("-", "")) for w in phrase.split())


def verb_lemmas(words):
    """Words this corpus uses as verbs, decided by this corpus.

    No verb list is shipped - a list would have to be edited for every
    technology area, which is the hardcoding this module exists to avoid. Two
    conditions, both from the text:

      1. the -ing and the -ed form are both present. Requiring both, not
         either, keeps ordinary nouns whose -ing form is a word in its own
         right ("housing", "coating").
      2. the bare form is outnumbered by its inflections. This is what
         separates "include" from "stack": both are conjugated somewhere in a
         patent corpus, but "stack" appears overwhelmingly as a bare noun
         ("fuel cell stack") and "include" does not. Without it, "fuel cell
         stack" is deleted as a verb phrase, which is worse than the noise it
         removes.
    """
    count = Counter(words)
    out = set()
    for w, bare in count.items():
        if len(w) < 3:
            continue
        ing = [w + "ing", w + w[-1] + "ing"] + ([w[:-1] + "ing"] if w.endswith("e") else [])
        ed = [w + "ed", w + w[-1] + "ed"] + ([w + "d"] if w.endswith("e") else [])
        n_ing = sum(count.get(f, 0) for f in ing)
        n_ed = sum(count.get(f, 0) for f in ed)
        if not (n_ing and n_ed):
            continue
        n_s = count.get(w + "s", 0) + count.get(w + "es", 0)
        if bare < n_ing + n_ed + n_s:
            out.add(w)
    return out


def _singular(word):
    """Undo a plural on the head word, for display only.

    Porter is the grouping key and is not readable - it turns "battery" into
    "batteri". The row still has to be shown as something a person would type,
    so the head is de-pluralised with ordinary English rules instead.
    """
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ches", "ch"),
                     ("shes", "sh"), ("xes", "x"), ("zes", "z")):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            return word[:-len(suf)] + rep
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "as", "os")) \
            and len(word) > 3:
        return word[:-1]
    return word


def _plural_head(phrase):
    return _singular(phrase.split()[-1]) != phrase.split()[-1]


def _display_form(counter):
    """The commonest surface form that is not plural-headed.

    A term and its plural share one stem, so only one row exists either way -
    but the row was being labelled with whichever form happened to be commoner,
    which put "battery cells" on screen. Singular is preferred outright, and
    when every observed form is plural the head is de-pluralised.
    """
    for form, _ in counter.most_common():
        if not _plural_head(form):
            return form
    form = counter.most_common(1)[0][0]
    words = form.split()
    return " ".join(words[:-1] + [_singular(words[-1])])


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

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


def _runs(text):
    """Maximal runs of open-class words, as word lists."""
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


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def _cvalue(freq, maxlen):
    """Frequency a phrase does not owe to the longer phrases that contain it.

    Frantzi and Ananiadou's C-value. "electrode assembly" earns its own score;
    "electrode" does not inherit it.
    """
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


def _cohesion(freq, unigram, total):
    """Mean self-information of the phrase's words. Rare words bind tighter."""
    out = {}
    for p in freq:
        bits = []
        for w in p.split():
            pw = unigram.get(w, 0) / total if total else 0.0
            if pw > 0:
                bits.append(-math.log(pw))
        out[p] = sum(bits) / len(bits) if bits else 0.0
    return out


# --------------------------------------------------------------------------- #
# step 1 - extract
# --------------------------------------------------------------------------- #

def extract(path, min_words=2, max_words=4, min_patents=2, drop_gerunds=True,
            drop_verb_heads=True,
            out_csv="technical_terms.csv", top=40,
            abstract=None, title=None, description=None, pubno=None):
    """Read the export, score every technical term, write the CSV.

    `min_words=2` because the reference tool's own slider defaults to 2, and
    because single words measured badly: on a 140-patent corpus, allowing them
    halved agreement with the reference. Set `min_words=1` if you want
    "electrolyte" and "anode" as rows and will tolerate "amount" and "ratio".
    """
    frame = (pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls"))
             else pd.read_csv(path))
    cols = list(frame.columns)
    c_abs = _pick(cols, HINTS["abstract"], abstract)
    c_ttl = _pick(cols, HINTS["title"], title)
    c_dsc = _pick(cols, HINTS["description"], description)
    c_pub = _pick(cols, HINTS["pubno"], pubno)
    if c_abs is None and c_ttl is None:
        raise SystemExit("no abstract or title column. Columns: %s"
                         % ", ".join(map(str, cols)))
    if c_pub is None:
        raise SystemExit("no publication-number column, so nothing can be "
                         "traced. Columns: %s" % ", ".join(map(str, cols)))

    keep = frame.dropna(subset=[c_pub]).reset_index(drop=True)
    parts = [c for c in (c_ttl, c_abs, c_dsc) if c]
    texts = [clean(" . ".join(str(r[c]) for c in parts if pd.notna(r[c])))
             for _, r in keep.iterrows()]
    pubs = [str(v).strip() for v in keep[c_pub]]

    all_runs = [_runs(t) for t in texts]
    vocab = [w for runs in all_runs for run in runs for w in run]
    verbs = verb_lemmas(vocab) if drop_verb_heads else set()

    # surface -> stem, and the per-document stem sets
    surface_count = defaultdict(Counter)
    freq = Counter()
    per_doc = []
    unigram = Counter()
    total = 0
    for runs in all_runs:
        seen = set()
        for run in runs:
            unigram.update(stem(w) for w in run)
            total += len(run)
            for n in range(min_words, max_words + 1):
                for i in range(len(run) - n + 1):
                    words = run[i:i + n]
                    head = words[-1]
                    if drop_gerunds and head.endswith("ing"):
                        continue
                    if head in verbs:
                        continue
                    surface = " ".join(words)
                    key = stem_phrase(surface)
                    surface_count[key][surface] += 1
                    freq[key] += 1
                    seen.add(key)
        per_doc.append(seen)

    df = Counter()
    for s in per_doc:
        df.update(s)
    freq = {k: f for k, f in freq.items() if df[k] >= min_patents}
    per_doc = [{k for k in s if k in freq} for s in per_doc]

    n_docs = len(texts)
    tfidf = {k: freq[k] * math.log(n_docs / df[k]) if df[k] < n_docs else 0.0
             for k in freq}
    cval = _cvalue(freq, max_words)
    coh = _cohesion(freq, unigram, total)
    score = {k: max(cval[k], 0.0) * (1.0 + coh[k]) for k in freq}
    display = {k: _display_form(surface_count[k]) for k in freq}

    patents = defaultdict(list)
    for i, s in enumerate(per_doc):
        for k in s:
            patents[k].append(pubs[i])

    ranked = sorted(freq, key=lambda k: (-score[k], k))
    table = pd.DataFrame({
        "term": [display[k] for k in ranked],
        "stem": ranked,
        "tfidf": [round(tfidf[k], 3) for k in ranked],
        "score": [round(score[k], 1) for k in ranked],
        "n_patents": [df[k] for k in ranked],
        "patents": [" ".join(patents[k]) for k in ranked],
    })
    table.to_csv(out_csv, index=False)

    state = {"texts": texts, "pubs": pubs, "per_doc": per_doc, "freq": freq,
             "df": df, "tfidf": tfidf, "score": score, "display": display,
             "patents": dict(patents), "ranked": ranked, "table": table,
             "surface": surface_count, "n_docs": n_docs,
             "columns": {"title": c_ttl, "abstract": c_abs,
                         "description": c_dsc, "pubno": c_pub}}

    print("CORPUS   %d rows, %d with a publication number" % (len(frame), n_docs))
    print("COLUMNS  title=%r abstract=%r description=%r id=%r"
          % (c_ttl, c_abs, c_dsc, c_pub))
    print("STEMMER  %s" % STEMMER)
    print("FILTERS  %d-%d words, in >=%d patents, gerund heads %s, verb heads %s"
          % (min_words, max_words, min_patents,
             "dropped" if drop_gerunds else "kept",
             ("dropped (%d verb lemmas found in this corpus)" % len(verbs))
             if drop_verb_heads else "kept"))
    print("TERMS    %d technical terms -> %s" % (len(ranked), out_csv))
    print()
    menu(state, top=top)
    return state


def menu(state, top=40):
    """The numbered term table. Pick a row number to drill into."""
    print("%-4s %-40s %9s %9s %6s  %s" % ("#", "term", "tfidf", "score",
                                          "pat", "first patents"))
    rows = state["ranked"][:top]
    for i, k in enumerate(rows, 1):
        pats = state["patents"][k]
        shown = " ".join(pats[:3]) + (" +%d" % (len(pats) - 3) if len(pats) > 3 else "")
        print("%-4d %-40s %9.2f %9.1f %6d  %s"
              % (i, state["display"][k], state["tfidf"][k], state["score"][k],
                 state["df"][k], shown))
    return rows


# --------------------------------------------------------------------------- #
# step 2 - drill into one term
# --------------------------------------------------------------------------- #

def _nested(a, b):
    """True when one stem key is a contiguous word run inside the other."""
    x, y = a.split(), b.split()
    if len(x) > len(y):
        x, y = y, x
    return any(y[i:i + len(x)] == x for i in range(len(y) - len(x) + 1))


def _resolve(state, term):
    """A row number, a display form, or any surface form of the term."""
    if isinstance(term, int):
        return state["ranked"][term - 1]
    key = stem_phrase(" ".join(str(term).lower().split()))
    if key in state["freq"]:
        return key
    for k, form in state["display"].items():
        if form == str(term).lower().strip():
            return k
    raise SystemExit("%r is not one of the extracted terms. Search the CSV, "
                     "or widen min_patents." % term)


def drill(state, term, min_patents=2, top=30, out_csv=None):
    """Sub-terms: the technical terms of the patents that carry this one.

    Scoped, not global. The denominator is the selected term's own patent set,
    so `share` is the fraction of *those* patents a sub-term appears in, and
    `lift` is how much more often it appears there than across the corpus. A
    sub-term with lift near 1 is simply common everywhere and is telling you
    nothing about the primary term.
    """
    key = _resolve(state, term)
    label = state["display"][key]
    idx = [i for i, s in enumerate(state["per_doc"]) if key in s]
    if not idx:
        raise SystemExit("no patents carry %r" % label)

    local_df = Counter()
    for i in idx:
        local_df.update(state["per_doc"][i] - {key})
    rows = [k for k, n in local_df.items() if n >= min_patents]

    # A phrase that contains the primary, or is contained by it, co-occurs with
    # it by construction: every patent with "fuel cell stack" has "fuel cell"
    # and "cell stack" in it, at share 1.00, saying nothing. Drop both
    # directions so what is left is genuinely a different technology.
    dropped = [k for k in rows if _nested(k, key)]
    rows = [k for k in rows if not _nested(k, key)]

    n_local, n_docs = len(idx), state["n_docs"]
    local_tfidf = {k: local_df[k] * math.log(n_local / local_df[k])
                   if local_df[k] < n_local else 0.0 for k in rows}
    lift = {k: (local_df[k] / n_local) / (state["df"][k] / n_docs) for k in rows}
    rows.sort(key=lambda k: (-local_df[k], -state["score"][k], k))

    patents = {k: [state["pubs"][i] for i in idx if k in state["per_doc"][i]]
               for k in rows}
    table = pd.DataFrame({
        "subterm": [state["display"][k] for k in rows],
        "stem": rows,
        "n_patents_with_both": [local_df[k] for k in rows],
        "share_of_primary": [round(local_df[k] / n_local, 3) for k in rows],
        "lift": [round(lift[k], 2) for k in rows],
        "tfidf_local": [round(local_tfidf[k], 3) for k in rows],
        "score": [round(state["score"][k], 1) for k in rows],
        "patents": [" ".join(patents[k]) for k in rows],
    })
    out_csv = out_csv or ("subterms_%s.csv"
                          % re.sub(r"[^a-z0-9]+", "_", label).strip("_"))
    table.to_csv(out_csv, index=False)

    print("PRIMARY  %r in %d patents: %s"
          % (label, n_local, " ".join(state["pubs"][i] for i in idx)))
    print("SUBTERMS %d in >=%d of those -> %s%s"
          % (len(rows), min_patents, out_csv,
             (", %d nested in the primary dropped" % len(dropped)) if dropped else ""))
    print()
    print("%-4s %-38s %5s %7s %6s  %s" % ("#", "sub-term", "both", "share", "lift", "patents"))
    for i, k in enumerate(rows[:top], 1):
        p = patents[k]
        shown = " ".join(p[:3]) + (" +%d" % (len(p) - 3) if len(p) > 3 else "")
        print("%-4d %-38s %5d %7.2f %6.1f  %s"
              % (i, state["display"][k], local_df[k],
                 local_df[k] / n_local, lift[k], shown))
    state["last_drill"] = {"primary": key, "rows": rows, "patents": patents,
                           "table": table}
    return table


# --------------------------------------------------------------------------- #
# step 3 - traceability
# --------------------------------------------------------------------------- #

def trace(state, term, within=None, limit=20):
    """Every patent a term occurs in, with the sentence it occurs in.

    `within` restricts to the patents carrying another term, which is how you
    check a sub-term: `trace(state, "gas sensor", within="hydrogen leakage")`
    answers "which of the hydrogen-leakage patents said gas sensor, and where".
    """
    key = _resolve(state, term)
    label = state["display"][key]
    idx = [i for i, s in enumerate(state["per_doc"]) if key in s]
    if within is not None:
        wkey = _resolve(state, within)
        idx = [i for i in idx if wkey in state["per_doc"][i]]
        print("TRACE    %r within the patents carrying %r"
              % (label, state["display"][wkey]))
    else:
        print("TRACE    %r" % label)

    forms = sorted(state["surface"][key], key=len, reverse=True)
    pattern = re.compile(r"\b(?:%s)\w*\b"
                         % "|".join(re.escape(f) for f in forms), re.IGNORECASE)
    rows = []
    for i in idx:
        for sentence in _SENT.split(state["texts"][i]):
            m = pattern.search(sentence)
            if m:
                rows.append({"patent": state["pubs"][i], "matched": m.group(0),
                             "sentence": sentence.strip()})
                break
    print("         %d patents" % len(rows))
    print()
    for r in rows[:limit]:
        s = r["sentence"]
        print("%-16s %-28s %s" % (r["patent"], r["matched"],
                                  s[:90] + ("..." if len(s) > 90 else "")))
    if len(rows) > limit:
        print("         ... %d more" % (len(rows) - limit))
    return pd.DataFrame(rows)
