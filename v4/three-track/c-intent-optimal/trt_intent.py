"""Track C - the intent-optimal successor to trt-pb, per the three-track analysis.

Attach as a Gemini Gem knowledge file. Runs in the code-execution sandbox:
standard library, pandas and numpy only.

Deterministic core, model periphery. Counting, normalisation, shrinkage,
statistics and trend gating live here. The model is used only where judgement
is genuinely required - the ambiguous band of phrase-pair equivalence, and
typed relation labelling - and every model decision is recorded under a cache
key so a rerun reproduces it. Every number traces to counts over spans a human
can open.

What changed against Track B, and where the analysis argues it:
  grouping     REPLACED  blocking -> band split -> model adjudicates the middle
                         band only -> weighted graph -> communities with a
                         representative consistency check (anti-chaining)  [4.1]
  relations    EXTENDED  typed, span-verified relations; chaining preserved;
                         six legacy classes as a projection                  [4.2]
  DS           FIXED+EXT pre-explode marginal + frequency floor; weighted
                         log-odds with an informative Dirichlet prior as the
                         challenger, both reported                            [4.3, s3]
  emergence    REPLACED  citation-free: decayed frequency, slope, acceleration,
                         first appearance, burst, persistence                 [4.6]
  citations    REPLACED  within-cohort percentile; empirical censoring horizon
                         H; FS suppressed inside H                            [4.4, 2.2]
  FS           FIXED+EXT empirical-Bayes shrinkage toward the corpus mean with
                         k from variance components; median alongside         [2.1]
  temporal     EXTENDED  2/3/4/5-year windows, window stability               [4.6]
  trend        REPLACED  four labels gated on a permutation null, a DF floor
                         and window stability; adds stable / emerging /
                         volatile / insufficient evidence; confidence, support [4.5]
  graph        FIXED+EXT typed directed multigraph with counts, publication
                         ids, sentences, confidence, year distribution        [4.6]

Sandbox stand-ins (documented substitutions, as in Tracks A and B): fixed
preposition list for ADP; corpus-internal inflection test for verb lemmas;
character-trigram cosine as the blocking embedding; PPMI context-vector cosine
as the deterministic contextual-substitutability signal.

    from trt_intent import report
    state = report("patents.xlsx")
    # then: adjudication_batches -> apply_adjudications -> resolve_groups
    #       relation_batches -> apply_relation_types -> typed_graph
    #       ds_table / emergence_table / trend / provenance
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

TRACK = "C"
PROMPT_VERSION = "adjudication-v1"
MODEL_ID = "recorded-by-gem"
TEMPERATURE = 0

# --------------------------------------------------------------------------- #
# columns, loading, cleaning, dates
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
    """Never derived from a publication number."""
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
# concepts - the surviving extractor, 1-grams allowed, span-grounded proposals
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
    freq = Counter(p for top in chosen for p in top)
    return chosen, sorted(freq, key=lambda p: (-freq[p], p)), freq


def propose_concepts(state, proposals):
    """Optional model-proposed concepts, gated on evidence: each proposal is
    {concept, pubno, span} and survives only if span occurs verbatim in that
    patent's abstract and contains the concept. Accepted concepts join the
    inventory; rejected ones are returned with the reason."""
    idx = {str(p): i for i, p in enumerate(state["pubnos"])}
    accepted, rejected = [], []
    for p in proposals:
        c, pub, span = str(p.get("concept", "")).lower().strip(), str(p.get("pubno")), str(p.get("span", ""))
        i = idx.get(pub)
        text = state["abstracts"][i].lower() if i is not None else ""
        if i is None:
            rejected.append((c, pub, "unknown publication"))
        elif span.lower() not in text:
            rejected.append((c, pub, "span not verbatim in abstract"))
        elif c not in span.lower():
            rejected.append((c, pub, "concept not inside span"))
        else:
            accepted.append(c)
            state["keyword_freq"][c] += 1
            if c not in state["keywords"]:
                state["keywords"].append(c)
    print("PROPOSALS  accepted %d, rejected %d" % (len(accepted), len(rejected)))
    return {"accepted": accepted, "rejected": rejected}


# --------------------------------------------------------------------------- #
# grouping - blocking, band split, adjudication, communities, consistency
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


def context_vectors(phrases, abstracts, window=3):
    """Deterministic contextual-substitutability signal. For every occurrence
    of a phrase, the content words within `window` tokens on either side form
    its context; contexts are PPMI-weighted against corpus context frequency
    and compared by cosine. Two phrases that fill the same slots score high,
    which is what the masked-LM ranking was measuring in the original."""
    by_first = defaultdict(list)
    for p in phrases:
        by_first[p.split()[0]].append(p.split())
    ctx = {p: Counter() for p in phrases}
    for text in abstracts:
        toks = _WORD.findall(text.lower())
        for i, t in enumerate(toks):
            for words in by_first.get(t, ()):
                n = len(words)
                if toks[i:i + n] == words:
                    left = [w for w in toks[max(0, i - window):i] if w not in _STOP]
                    right = [w for w in toks[i + n:i + n + window] if w not in _STOP]
                    ctx[" ".join(words)].update(left + right)
    total = Counter()
    for c in ctx.values():
        total.update(c)
    grand = sum(total.values()) or 1
    vocab = {w: i for i, w in enumerate(total)}
    m = np.zeros((len(phrases), len(vocab)))
    for r, p in enumerate(phrases):
        n_p = sum(ctx[p].values()) or 1
        for w, k in ctx[p].items():
            pmi = math.log2((k / n_p) / (total[w] / grand))
            if pmi > 0:
                m[r, vocab[w]] = pmi
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return m / norm


def _key(a, b):
    a, b = sorted((a, b))
    raw = "|".join([a, b, PROMPT_VERSION, MODEL_ID, str(TEMPERATURE)])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def blocking(state, k=20, hi=0.90, lo=0.55):
    """Top-k neighbours per phrase by trigram cosine, split into three bands.
    Above hi: merged without asking (spelling, plural, near-identical). Below
    lo: never a pair. Between: the model's job, with the contextual score and
    an example sentence each as evidence. Pairs are unordered and sorted."""
    phrases = list(state["keywords"])
    lex_v = _trigram_vectors(phrases)
    S = lex_v @ lex_v.T
    ctx_v = context_vectors(phrases, state["abstracts"])
    C = ctx_v @ ctx_v.T
    pairs = {}
    for i, p in enumerate(phrases):
        order = np.argsort(-S[i])[:k + 1]
        for j in order:
            if j == i or S[i, j] < lo:
                continue
            a, b = sorted((p, phrases[j]))
            pairs[(a, b)] = (float(S[i, j]), float(C[i, j]))
    auto, band = {}, {}
    for (a, b), (s, c) in sorted(pairs.items()):
        (auto if s >= hi else band)[(a, b)] = {"lexical": round(s, 3), "contextual": round(c, 3)}
    state.update({"pairs_auto": auto, "pairs_band": band, "similarity": (phrases, S, C)})
    return {"auto_merge": len(auto), "ambiguous": len(band), "rejected_below_lo": "all others"}


def _example(state, phrase):
    for a, pub in zip(state["abstracts"], state["pubnos"]):
        for s in _sentences(a):
            if phrase in s.lower():
                return "%s: %s" % (pub, s[:220])
    return ""


def adjudication_batches(state, size=20):
    """The ambiguous band, batched for the model. Each item carries a cache
    key; decisions are stored under it, so the same pair is never asked twice
    and a rerun reproduces the partition. The question is substitutability,
    not relatedness: anode / cathode are related and must not merge."""
    items = []
    for (a, b), sc in state["pairs_band"].items():
        key = _key(a, b)
        if key in state["decisions"]:
            continue
        items.append({"key": key, "a": a, "b": b, "lexical": sc["lexical"], "contextual": sc["contextual"],
                      "example_a": _example(state, a), "example_b": _example(state, b)})
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    print("ADJUDICATION  %d pairs pending in %d batches of %d" % (len(items), len(batches), size))
    return batches


def apply_adjudications(state, decisions):
    """decisions: list of {key | (a, b), equivalent: bool, reason: str}."""
    n = 0
    for d in decisions:
        key = d.get("key") or _key(d["a"], d["b"])
        state["decisions"][key] = {"equivalent": bool(d.get("equivalent")), "reason": str(d.get("reason", "")),
                                   "a": d.get("a"), "b": d.get("b")}
        n += 1
    print("ADJUDICATION  recorded %d decisions (%d total)" % (n, len(state["decisions"])))
    return len(state["decisions"])


def export_decisions(state):
    return json.dumps(state["decisions"], sort_keys=True)


def import_decisions(state, text):
    state["decisions"].update(json.loads(text))
    return len(state["decisions"])


def resolve_groups(state, floor=0.25):
    """Weighted equivalence graph -> connected communities -> representative
    consistency check -> canonical names. A member stays in a community only
    if it is compatible with the canonical member: merged with it directly,
    or lexically close with contextual support above `floor`, and never
    adjudicated non-equivalent against it. Members that fail are split off
    and resolved on their own, which is what stops A~B, B~C, A!=C chaining.
    Pending (unadjudicated) band pairs are not edges."""
    phrases, S, C = state["similarity"]
    idx = {p: i for i, p in enumerate(phrases)}
    freq = state["keyword_freq"]
    pos, neg = set(), set()
    for (a, b) in state["pairs_auto"]:
        pos.add((a, b))
    for key, d in state["decisions"].items():
        pair = tuple(sorted((d.get("a") or "", d.get("b") or "")))
        if pair[0] in idx and pair[1] in idx:
            (pos if d["equivalent"] else neg).add(pair)
    adj = defaultdict(set)
    for a, b in pos:
        if (a, b) not in neg:
            adj[a].add(b)
            adj[b].add(a)
    seen, groups, evidence = set(), {}, {}

    def canonical(members):
        return sorted(members, key=lambda p: (-freq[p], len(p), p))[0]

    def compatible(c, m):
        if tuple(sorted((c, m))) in neg:
            return False
        if m in adj[c]:
            return True
        return S[idx[c], idx[m]] >= 0.55 and C[idx[c], idx[m]] >= floor

    def resolve(members):
        members = sorted(members)
        c = canonical(members)
        keep, split = [c], []
        for m in members:
            if m == c:
                continue
            (keep if compatible(c, m) else split).append(m)
        groups[c] = keep
        for m in keep:
            if m != c:
                pair = tuple(sorted((c, m)))
                d = next((v for v in state["decisions"].values()
                          if tuple(sorted((v.get("a") or "", v.get("b") or ""))) == pair), None)
                evidence[m] = {"canonical": c, "rule": "adjudicated" if d else "auto",
                               "lexical": round(float(S[idx[c], idx[m]]), 3),
                               "contextual": round(float(C[idx[c], idx[m]]), 3),
                               "reason": d["reason"] if d else "trigram cosine >= hi"}
        if split:
            resolve(split)

    for p in sorted(phrases, key=lambda p: (-freq[p], p)):
        if p in seen:
            continue
        comp, stack = set(), [p]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        resolve(comp)
    state["groups"] = groups
    state["merge_evidence"] = evidence
    state["owner"] = {m: c for c, ms in groups.items() for m in ms}
    _map_relations(state)
    pending = sum(1 for (a, b) in state["pairs_band"] if _key(a, b) not in state["decisions"])
    print("GROUPS  %d concepts from %d surface forms; %d merges recorded; %d band pairs still pending"
          % (len(groups), len(phrases), len(evidence), pending))
    return groups


# --------------------------------------------------------------------------- #
# relations - chained triples, typed with verification, legacy projection
# --------------------------------------------------------------------------- #

ONTOLOGY = ["component-of", "used-for", "acts-on", "enables", "produced-by", "implemented-with",
            "located-in", "performs", "converts", "controls", "detects", "transmits", "stores",
            "sourced-from", "mediated-by", "functions-as", "related-to"]
LEGACY_MAP = {"component-of": "Inclusion", "located-in": "Inclusion", "implemented-with": "Inclusion",
              "sourced-from": "Inclusion", "used-for": "Objective", "enables": "Objective",
              "acts-on": "Effect", "controls": "Effect", "detects": "Effect", "converts": "Effect",
              "transmits": "Effect", "stores": "Effect", "performs": "Effect",
              "produced-by": "Process", "mediated-by": "Process", "functions-as": "Likeness",
              "related-to": "Misc"}
PREP_DEFAULT = {"of": "component-of", "in": "component-of", "within": "component-of",
                "with": "implemented-with", "by": "implemented-with", "on": "located-in",
                "at": "located-in", "from": "sourced-from", "for": "used-for", "to": "acts-on",
                "across": "acts-on", "against": "acts-on", "during": "mediated-by", "into": "mediated-by",
                "through": "mediated-by", "via": "mediated-by", "as": "functions-as"}
RELATIONSHIPS = ["Inclusion", "Objective", "Effect", "Process", "Likeness", "Misc"]


def get_trt(text, lex):
    """As shipped: each preposition yields (T1, prep, T2) walked outward to
    the next preposition, so triples chain along a sentence."""
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
            trt.append((t1, tok, re.sub(r"\.", "", t2), sent))
    return trt


def _pattern(terms):
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE) \
        if terms else re.compile(r"(?!x)x")


def _map_relations(state):
    pat = _pattern(state["keywords"])
    owner = state.get("owner", {})
    rows = []
    for i, (pub, year, abstract) in enumerate(zip(state["pubnos"], state["years"], state["abstracts"])):
        for t1, prep, t2, sent in state["triples"][i]:
            for a in pat.findall(t1):
                for b in pat.findall(t2):
                    sa, sb = owner.get(a.lower(), a.lower()), owner.get(b.lower(), b.lower())
                    if sa == sb:
                        continue
                    rows.append((len(rows), sa, sb, a.lower(), b.lower(), prep.lower(),
                                 PREP_DEFAULT.get(prep.lower(), "related-to"), pub, year, sent))
    state["relations"] = pd.DataFrame(rows, columns=["id", "source", "target", "source_surface",
                                                     "target_surface", "prep", "relation", "pubno",
                                                     "year", "sentence"])
    state["relations"]["typed_by"] = "preposition default"
    state["relations"]["confidence"] = 0.5
    return state["relations"]


def relation_batches(state, size=25, only_untyped=True):
    """Rows for the model to type within ONTOLOGY. It must return, per id, the
    relation and the exact source and target spans it relied on."""
    r = state["relations"]
    if only_untyped:
        r = r[r["typed_by"] == "preposition default"]
    items = [{"id": int(x.id), "source": x.source, "target": x.target, "prep": x.prep,
              "default": x.relation, "sentence": x.sentence} for x in r.itertuples(index=False)]
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    print("RELATIONS  %d rows to type in %d batches; ontology: %s" % (len(items), len(batches), ", ".join(ONTOLOGY)))
    return batches


def apply_relation_types(state, typed):
    """typed: list of {id, relation, source_span, target_span, confidence}.
    A row is accepted only if the relation is in the ontology and both spans
    occur verbatim in the row's sentence and contain the matched surface
    forms. Anything else is dropped, not repaired, and counted."""
    r = state["relations"]
    ok, dropped = 0, Counter()
    for t in typed:
        i = int(t.get("id", -1))
        if i not in r.index:
            dropped["unknown id"] += 1
            continue
        rel = str(t.get("relation", "")).strip().lower()
        sent = str(r.at[i, "sentence"]).lower()
        s_span, t_span = str(t.get("source_span", "")).lower(), str(t.get("target_span", "")).lower()
        if rel not in ONTOLOGY:
            dropped["relation outside ontology"] += 1
        elif not s_span or s_span not in sent or r.at[i, "source_surface"] not in s_span:
            dropped["source span not verbatim"] += 1
        elif not t_span or t_span not in sent or r.at[i, "target_surface"] not in t_span:
            dropped["target span not verbatim"] += 1
        else:
            r.at[i, "relation"] = rel
            r.at[i, "typed_by"] = "model"
            r.at[i, "confidence"] = float(t.get("confidence", 0.8))
            ok += 1
    print("RELATIONS  accepted %d typed rows; dropped %s" % (ok, dict(dropped) or "none"))
    return {"accepted": ok, "dropped": dict(dropped)}


def legacy_view(state):
    """The six original classes as a projection of the typed relations."""
    r = state["relations"].copy()
    r["legacy"] = r["relation"].map(lambda x: LEGACY_MAP.get(x, "Misc"))
    return r


def typed_graph(state, min_patents=1):
    """Directed typed multigraph: one row per (source, relation, target) with
    occurrences, distinct patents, publication ids, example sentences,
    confidence, first and last year and the year distribution."""
    r = state["relations"]
    rows = []
    for (s, rel, t), g in r.groupby(["source", "relation", "target"]):
        years = [int(y) for y in g["year"] if y is not None and not pd.isna(y)]
        if g["pubno"].nunique() < min_patents:
            continue
        rows.append({"source": s, "relation": rel, "target": t, "legacy": LEGACY_MAP.get(rel, "Misc"),
                     "occurrences": len(g), "distinct_patents": int(g["pubno"].nunique()),
                     "confidence_mean": round(float(g["confidence"].mean()), 2),
                     "first_year": min(years) if years else None, "last_year": max(years) if years else None,
                     "year_distribution": dict(sorted(Counter(years).items())),
                     "publication_ids": list(dict.fromkeys(map(str, g["pubno"]))),
                     "evidence": list(dict.fromkeys(g["sentence"]))[:3]})
    out = pd.DataFrame(rows)
    return out.sort_values(["distinct_patents", "occurrences"], ascending=False).reset_index(drop=True) \
        if len(out) else out


def edge_evidence(state, source, target, relation=None):
    r = state["relations"]
    sub = r[(r["source"] == source) & (r["target"] == target)]
    if relation:
        sub = sub[sub["relation"] == relation]
    return sub[["pubno", "year", "prep", "relation", "typed_by", "sentence"]]


# --------------------------------------------------------------------------- #
# domains, DS: fixed lift + floor, weighted log-odds challenger
# --------------------------------------------------------------------------- #

def split_domains(value):
    return [d.strip() for d in str(value).split("\n") if d.strip()] if isinstance(value, str) else []


def _domain_counts(state, term_col):
    """Patent-level counts: k[td][term], n[td], k_all[term], n_all."""
    keep, cm = state["keep"], state["columns"]
    k, n, k_all = defaultdict(Counter), Counter(), Counter()
    for doms, terms in zip(keep[cm["domain"]].map(split_domains), keep[term_col]):
        terms = set(terms)
        k_all.update(terms)
        for td in set(doms):
            n[td] += 1
            k[td].update(terms)
    return k, n, k_all, len(keep)


def ds_table(state, td, term_col="Verbs", floor=3, alpha0=10.0):
    """Two estimators of 'how characteristic is term F of domain D', both on
    patient-level counts with the marginal taken before the explode:

      lift      P(F|D) / P(F), reported only when DF_TD >= floor
      log-odds  weighted log-odds ratio with an informative Dirichlet prior
                (Monroe, Colaresi & Quinn 2008), domain vs the rest of the
                corpus, as a z-score; prior mass alpha0 spread by corpus
                proportion, so a term seen once cannot reach the top

    The challenger must beat lift-plus-floor on the labelled sample before it
    replaces it; until then both are shown."""
    k, n, k_all, n_all = _domain_counts(state, term_col)
    if td not in n:
        raise SystemExit("domain %r not found. Domains: %s" % (td, ", ".join(sorted(n)[:20])))
    n_d, n_r = n[td], n_all - n[td]
    total_terms = sum(k_all.values()) or 1
    rows = []
    for term, k_d in k[td].items():
        k_r = k_all[term] - k_d
        a_w = alpha0 * k_all[term] / total_terms
        lift = (k_d / n_d) / (k_all[term] / n_all) if k_all[term] else 0.0
        d = (math.log((k_d + a_w) / (n_d + alpha0 - k_d - a_w))
             - math.log((k_r + a_w) / (n_r + alpha0 - k_r - a_w))) if n_r > 0 else 0.0
        var = 1 / (k_d + a_w) + 1 / (k_r + a_w)
        rows.append((term, k_d, k_all[term], round(lift, 2) if k_d >= floor else None,
                     round(d, 3), round(d / math.sqrt(var), 2)))
    out = pd.DataFrame(rows, columns=["term", "DF_TD", "DF_All", "lift (floor %d)" % floor, "log-odds", "log-odds z"])
    return out.sort_values("log-odds z", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# prevalence and emergence - citation-free
# --------------------------------------------------------------------------- #

def prevalence(state, term, term_col="filtered_tech_keys", td=None):
    """q_t = distinct patents with the term in year t / distinct patents in t."""
    keep, cm = state["keep"], state["columns"]
    if td:
        keep = keep[keep[cm["domain"]].map(lambda v: td in split_domains(v))]
    years = keep["Year"]
    has = keep[term_col].map(lambda t: term in t)
    tot, hit = Counter(), Counter()
    for y, h in zip(years, has):
        if y is not None and not pd.isna(y):
            tot[int(y)] += 1
            hit[int(y)] += int(h)
    ys = sorted(tot)
    return pd.DataFrame({"year": ys, "patents": [tot[y] for y in ys], "with_term": [hit[y] for y in ys],
                         "q": [hit[y] / tot[y] for y in ys]})


def emergence(state, term, term_col="filtered_tech_keys", td=None, recent=3, half_life=3.0):
    """Signals, each answering a different question, none collapsed into one:
      decayed_df     exponentially time-decayed document count, half-life 3y,
                     as a share of the decayed corpus size
      slope          OLS slope of q_t over the last 5 years
      acceleration   slope of the last 3 years minus slope of the 3 before
      first_age      years since first appearance
      burst          max z of q_t in the recent years against the earlier mean/sd
      persistence    share of recent years with q_t above the earlier median
    """
    p = prevalence(state, term, term_col, td)
    if p.empty or state["max_year"] is None:
        return None
    ymax = state["max_year"]
    w = 0.5 ** ((ymax - p["year"]) / half_life)
    decayed = float((w * p["with_term"]).sum() / max((w * p["patents"]).sum(), 1e-9))
    q, y = p["q"].to_numpy(), p["year"].to_numpy()
    last5 = y >= ymax - 4
    slope = float(np.polyfit(y[last5], q[last5], 1)[0]) if last5.sum() > 1 else 0.0
    rec, prior = y >= ymax - recent + 1, (y < ymax - recent + 1) & (y >= ymax - 2 * recent + 1)
    s_rec = float(np.polyfit(y[rec], q[rec], 1)[0]) if rec.sum() > 1 else 0.0
    s_pri = float(np.polyfit(y[prior], q[prior], 1)[0]) if prior.sum() > 1 else 0.0
    hist = q[y < ymax - recent + 1]
    first = int(p[p["with_term"] > 0]["year"].min()) if (p["with_term"] > 0).any() else None
    burst = float(((q[rec] - hist.mean()) / hist.std()).max()) if len(hist) > 1 and hist.std() > 0 and rec.any() else 0.0
    persistence = float((q[rec] > np.median(hist)).mean()) if len(hist) and rec.any() else 0.0
    return {"term": term, "domain": td, "decayed_df": round(decayed, 4), "slope": round(slope, 5),
            "acceleration": round(s_rec - s_pri, 5), "first_age": (ymax - first) if first else None,
            "burst": round(burst, 2), "persistence": round(persistence, 2),
            "support_recent": int(p[rec]["with_term"].sum())}


def emergence_table(state, terms=None, term_col="filtered_tech_keys", td=None, top=20, min_support=3):
    """Ranked by burst then slope, restricted to terms with at least
    `min_support` patents in the recent years - a burst on one patent is
    not an emergence."""
    terms = terms or state["keywords"][:200]
    rows = [e for e in (emergence(state, t, term_col, td) for t in terms) if e and e["support_recent"] >= min_support]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["burst", "slope"], ascending=False).reset_index(drop=True)
    return out.head(top)


# --------------------------------------------------------------------------- #
# influence - cohort percentile, censoring horizon, shrunk FS
# --------------------------------------------------------------------------- #

_SEP = re.compile(r"[\n;|,]+")


def count_citing_patents(citing_string):
    if not citing_string or not isinstance(citing_string, str):
        return 0
    return len([p for p in _SEP.split(citing_string) if p.strip()])


def cohort_percentile(state, min_cohort=30):
    """Percentile rank (0-1, ties averaged) of each patent's forward-citation
    count within its cohort: application year +/- 1, and additionally the
    same first-listed domain when that cohort has at least `min_cohort`
    patents. No assumption about how citations accrue with age."""
    keep, cm = state["keep"], state["columns"]
    years = keep["Year"].to_numpy(dtype=float)
    c = keep["count_citing_patents"].to_numpy(dtype=float)
    dom = keep[cm["domain"]].map(lambda v: (split_domains(v) or [None])[0]).to_numpy() if cm["domain"] else np.array([None] * len(keep))
    pct = np.full(len(keep), np.nan)
    for i in range(len(keep)):
        if np.isnan(years[i]):
            continue
        same_year = np.abs(years - years[i]) <= 1
        cohort = same_year & (dom == dom[i]) if dom[i] is not None else same_year
        if cohort.sum() < min_cohort:
            cohort = same_year
        vals = c[cohort]
        pct[i] = ((vals < c[i]).sum() + 0.5 * (vals == c[i]).sum()) / len(vals)
    keep["influence"] = pct
    return keep["influence"]


def censoring_horizon(state, frac=0.5, min_cohort=5, default=3):
    """H: the age at which the cohort median citation count first reaches
    `frac` of its plateau, estimated from older cohorts. Windows ending
    within H years of the newest patent get no FS - the count there is not
    small, it is unobserved."""
    keep = state["keep"]
    if state["max_year"] is None or "count_citing_patents" not in keep:
        return None
    tab = pd.DataFrame({"year": keep["Year"], "c": keep["count_citing_patents"]}).dropna()
    tab["year"] = tab["year"].astype(int)
    grp = tab.groupby("year")["c"]
    sizes = grp.size()
    med = grp.median().sort_index()
    stat = "median"
    if med.max() < 2:            # sparse counts: medians are 0 or 1, use the mean
        med, stat = grp.mean().sort_index(), "mean"
    med = med[sizes.reindex(med.index) >= min_cohort]
    smooth = med.rolling(3, center=True, min_periods=1).median()
    ages = (state["max_year"] - med.index).to_numpy()
    state["H_stat"] = stat
    state["H_table"] = pd.DataFrame({"cohort": med.index, "age": ages, stat: med.to_numpy(),
                                     "smoothed": smooth.to_numpy(),
                                     "cohort_size": sizes.reindex(med.index).to_numpy()})

    # The horizon is only identifiable when citations actually accrue with age.
    # On a corpus that is not a complete cohort - a relevance-ranked sample, a
    # keyword export, anything that over-samples well-cited patents - the curve
    # is flat or non-monotone and the age at which it "reaches" a plateau is an
    # artifact. Detect that and say so, rather than silently returning a small
    # H and disabling the censoring rule the design depends on.
    rank_corr = 0.0
    if len(med) >= 5:
        a = pd.Series(ages).rank()
        b = pd.Series(smooth.to_numpy()).rank()
        rank_corr = float(a.corr(b)) if a.std() and b.std() else 0.0
    if len(med) < 5 or rank_corr < 0.3:
        state["H"] = default
        state["H_identified"] = False
        state["H_rank_corr"] = round(rank_corr, 2)
        print("CENSORING  accrual is not monotone in age (age/median rank correlation %.2f over %d "
              "cohorts): the horizon is not identifiable from this corpus. Using the stated default "
              "H=%d years and saying so." % (rank_corr, len(med), default))
        return state["H"]

    plateau = float(smooth.max())
    H = next((int(age) for age, m in sorted(zip(ages, smooth.to_numpy()))
              if plateau > 0 and m >= frac * plateau), None)
    state["H"] = H if H is not None else default
    state["H_identified"] = True
    state["H_rank_corr"] = round(rank_corr, 2)
    return state["H"]


def shrinkage_k(state, term_col="filtered_tech_keys", width=3):
    """k = within-cell variance / between-cell variance, from (term, window)
    cells with at least two patents. FS_shrunk = (n m + k mu) / (n + k)."""
    keep = state["keep"]
    infl = keep["influence"].to_numpy()
    years = keep["Year"].to_numpy(dtype=float)
    mu = float(np.nanmean(infl))
    cells = []
    for term in state["keywords"][:300]:
        has = keep[term_col].map(lambda t: term in t).to_numpy()
        for a, b in _windows(state, width):
            m = has & (years >= a) & (years <= b) & ~np.isnan(infl)
            if m.sum() >= 2:
                cells.append((m.sum(), infl[m].mean(), infl[m].var(ddof=1)))
    if len(cells) < 5:
        state["k"], state["mu"] = 5.0, mu
        return state["k"]
    n = np.array([c[0] for c in cells]); m = np.array([c[1] for c in cells]); v = np.array([c[2] for c in cells])
    within = float(np.average(v, weights=n - 1)) if (n - 1).sum() > 0 else 0.0
    between = max(float(m.var(ddof=1) - np.mean(within / n)), 1e-6)
    state["k"], state["mu"] = within / between, mu
    return state["k"]


def _windows(state, width):
    """Fixed-width windows aligned to END at the newest year, so the most
    recent window is always full."""
    ys = [y for y in state["years"] if y]
    lo, hi = min(ys), max(ys)
    out = []
    end = hi
    while end - width + 1 >= lo:
        out.append((end - width + 1, end))
        end -= width
    return list(reversed(out))


def trajectory(state, term, width=3, term_col="filtered_tech_keys", td=None):
    """Per window: DF, prevalence, shrunk FS, median influence, censored flag."""
    keep, cm = state["keep"], state["columns"]
    if td:
        keep = keep[keep[cm["domain"]].map(lambda v: td in split_domains(v))]
    years = keep["Year"].to_numpy(dtype=float)
    infl = keep["influence"].to_numpy()
    has = keep[term_col].map(lambda t: term in t).to_numpy()
    k, mu, H = state["k"], state["mu"], state["H"]
    rows = []
    for a, b in _windows(state, width):
        w = (years >= a) & (years <= b)
        m = w & has & ~np.isnan(infl)
        n = int((w & has).sum())
        pats = int(w.sum())
        censored = b > state["max_year"] - H
        vals = infl[m]
        fs = (n * vals.mean() + k * mu) / (n + k) if len(vals) else mu
        rows.append({"window": "%d-%d" % (a, b), "patents": pats, "DF": n,
                     "prevalence": round(n / pats, 4) if pats else 0.0,
                     "FS_shrunk": None if censored else round(float(fs), 4),
                     "FS_median": None if censored or not len(vals) else round(float(np.median(vals)), 4),
                     "censored": censored})
    return pd.DataFrame(rows)


QUADRANTS = {("up", "up"): "Growing", ("up", "down"): "Generalizing",
             ("down", "down"): "Declining", ("down", "up"): "Repositioning"}
# The four quadrants assume both axes move. When one does not clear its gate
# the honest label says which one moved; the nearest quadrant is reported
# separately as legacy_state, marked as not significant on the flat axis.
MIXED = {("up", "flat"): "Expanding (influence unchanged)",
         ("down", "flat"): "Contracting (influence unchanged)",
         ("flat", "up"): "Rising influence (prevalence unchanged)",
         ("flat", "down"): "Falling influence (prevalence unchanged)"}


def _null_fs(state, term, a, b, n, term_col, td, runs, rng):
    keep, cm = state["keep"], state["columns"]
    if td:
        keep = keep[keep[cm["domain"]].map(lambda v: td in split_domains(v))]
    years = keep["Year"].to_numpy(dtype=float)
    infl = keep["influence"].to_numpy()
    pool = infl[(years >= a) & (years <= b) & ~np.isnan(infl)]
    k, mu = state["k"], state["mu"]
    if n == 0 or len(pool) == 0:
        return np.full(runs, mu)
    draws = np.array([pool[rng.choice(len(pool), size=min(n, len(pool)), replace=False)].mean() for _ in range(runs)])
    return (n * draws + k * mu) / (n + k)


def trend(state, term, widths=(2, 3, 4, 5), term_col="filtered_tech_keys", td=None, floor=5,
          tol=0.10, runs=200, seed=0):
    """Gated classification of the latest uncensored movement at each width.
      DF direction: relative change in prevalence beyond `tol`, else flat
      FS direction: shrunk FS change outside the 95% permutation band
                    (patents in the window re-drawn at the same DF), else flat
      floor:        both windows need DF >= floor, else insufficient evidence
    Widths are combined into window_stability; a label is reported only when
    at least three of four widths agree. Extra states: stable, volatile,
    emerging (from the emergence signals), window-sensitive."""
    rng = np.random.default_rng(seed)
    per_width, movements = {}, {}
    for w in widths:
        tr = trajectory(state, term, w, term_col, td)
        usable = tr[~tr["censored"]]
        if len(usable) < 2:
            per_width[w] = "insufficient evidence"
            continue
        prev, cur = usable.iloc[-2], usable.iloc[-1]
        if prev["DF"] < floor or cur["DF"] < floor:
            per_width[w] = "insufficient evidence"
            continue
        rel = (cur["prevalence"] - prev["prevalence"]) / prev["prevalence"] if prev["prevalence"] else (1 if cur["prevalence"] else 0)
        d = "up" if rel > tol else "down" if rel < -tol else "flat"
        a1, b1 = map(int, prev["window"].split("-")); a2, b2 = map(int, cur["window"].split("-"))
        null = _null_fs(state, term, a2, b2, int(cur["DF"]), term_col, td, runs, rng) \
            - _null_fs(state, term, a1, b1, int(prev["DF"]), term_col, td, runs, rng)
        delta = cur["FS_shrunk"] - prev["FS_shrunk"]
        lo_b, hi_b = np.percentile(null, 2.5), np.percentile(null, 97.5)
        f = "up" if delta > hi_b else "down" if delta < lo_b else "flat"
        p = float(min(1.0, 2 * min((null >= delta).mean(), (null <= delta).mean())))
        signs = [np.sign(x) for x in np.diff(usable["FS_shrunk"].astype(float).to_numpy()[-4:])]
        volatile = len(signs) >= 3 and all(signs[i] * signs[i + 1] < 0 for i in range(len(signs) - 1))
        label = "stable" if (d, f) == ("flat", "flat") else QUADRANTS.get((d, f)) or MIXED[(d, f)]
        per_width[w] = "volatile" if volatile else label
        raw = ("up" if rel > 0 else "down" if rel < 0 else "flat", "up" if delta > 0 else "down" if delta < 0 else "flat")
        movements[w] = {"dDF_rel": round(float(rel), 3), "dFS": round(float(delta), 4), "p_null": round(p, 3),
                        "support": int(cur["DF"]), "from": prev["window"], "to": cur["window"],
                        "nearest_quadrant": QUADRANTS.get(raw)}
    labels = Counter(per_width.values())
    top, count = labels.most_common(1)[0]
    stability = count / len(widths)
    headline = top if stability >= 0.75 else "window-sensitive"
    e = emergence(state, term, term_col, td) or {}
    flags = []
    if e and (e.get("burst", 0) >= 2 or (e.get("first_age") is not None and e["first_age"] <= 3)) and e.get("slope", 0) > 0:
        flags.append("emerging")
    ps = [m["p_null"] for m in movements.values()]
    gated = headline not in ("insufficient evidence", "window-sensitive")
    confidence = round(stability * (1 - (min(ps) if ps else 1.0)), 2) if gated else 0.0
    if headline in QUADRANTS.values():
        legacy, note = headline, "both axes cleared their gates"
    elif gated and movements:
        nearest = Counter(m["nearest_quadrant"] for m in movements.values() if m["nearest_quadrant"]).most_common(1)
        legacy = nearest[0][0] if nearest else None
        note = "nearest of the four; at least one axis did not clear its gate"
    else:
        legacy, note = None, "no legacy label warranted"
    return {"term": term, "domain": td, "label": headline, "flags": flags, "window_stability": round(stability, 2),
            "confidence": confidence, "per_width": {"%dy" % w: v for w, v in per_width.items()},
            "movements": {"%dy" % w: m for w, m in movements.items()},
            "legacy_state": legacy, "legacy_note": note,
            "censoring_horizon_years": state["H"], "emergence": e}


def window_sensitivity(state, terms=None, term_col="filtered_tech_keys", td=None, widths=(2, 3, 4, 5)):
    terms = terms or state["keywords"][:40]
    rows = []
    for t in terms:
        r = trend(state, t, widths, term_col, td, runs=50)
        rows.append([t, r["label"], r["window_stability"]] + [r["per_width"]["%dy" % w] for w in widths])
    out = pd.DataFrame(rows, columns=["term", "label", "stability"] + ["%dy" % w for w in widths])
    print("WINDOW SENSITIVITY  %d terms; %.0f%% window-sensitive; %.0f%% insufficient evidence"
          % (len(out), 100 * (out["label"] == "window-sensitive").mean(),
             100 * (out["label"] == "insufficient evidence").mean()))
    return out


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def documents_for(state, phrase, limit=15, td=None):
    keep, cm = state["keep"], state["columns"]
    forms = [m for m, c in state.get("owner", {}).items() if c == phrase] or [phrase]
    hits = [i for i, a in enumerate(state["abstracts"]) if any(f in a.lower() for f in forms)]
    if td:
        hits = [i for i in hits if td in split_domains(keep.at[i, cm["domain"]])]
    cols = [c for c in (cm["pubno"], cm["title"], cm["date"]) if c] + ["influence"]
    return keep.iloc[hits[:limit]][cols]


def provenance(state, kind, key, **kw):
    """Evidence objects: 'technology' (patents + sentences), 'relation'
    ((source, target)), 'trend' (trajectory per width)."""
    if kind == "technology":
        forms = [m for m, c in state["owner"].items() if c == key] or [key]
        rows = []
        for a, pub in zip(state["abstracts"], state["pubnos"]):
            for s in _sentences(a):
                if any(f in s.lower() for f in forms):
                    rows.append({"publication_id": pub, "sentence": s})
                    break
        return {"technology": key, "surface_forms": forms, "supporting_patents": rows[:kw.get("limit", 25)],
                "support": len(rows)}
    if kind == "relation":
        s, t = key
        return edge_evidence(state, s, t, kw.get("relation")).to_dict("records")
    if kind == "trend":
        return {w: trajectory(state, key, w, kw.get("term_col", "filtered_tech_keys"), kw.get("td")).to_dict("records")
                for w in kw.get("widths", (2, 3, 4, 5))}
    raise ValueError("kind must be technology, relation or trend")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def report(path, *, no_keywords=7, show=20, **columns):
    frame, cm = load(path, **columns)
    print("TRACK C - intent-optimal (deterministic core, model periphery)")
    print("COLUMNS  " + "  ".join("%s=%r" % kv for kv in cm.items()))
    keep = frame[frame[cm["abstract"]].notna()]
    if cm["pubno"]:
        keep = keep[keep[cm["pubno"]].notna()]
    keep = keep.reset_index(drop=True).copy()
    abstracts = [clean_text(a) for a in keep[cm["abstract"]]]
    pubnos = list(keep[cm["pubno"]]) if cm["pubno"] else ["row%d" % i for i in range(len(keep))]
    years = [year_of(v) for v in keep[cm["date"]]] if cm["date"] else [None] * len(keep)
    texts = list(keep[cm["description"]]) if cm["description"] else abstracts
    lex = verb_lexicon(texts + abstracts if cm["description"] else abstracts)
    dated = sum(1 for y in years if y)
    print("CORPUS   rows=%d usable=%d dated=%d years=%s..%s" % (
        len(frame), len(keep), dated, min((y for y in years if y), default=None), max((y for y in years if y), default=None)))
    chosen, ordered, freq = keywords_per_document(abstracts, lex, no_keywords)
    keep["Year"] = years
    keep["Verbs"] = [sorted({lex[w] for w in _WORD.findall(str(t).lower()) if w in lex}) for t in texts]
    # Word-boundary matching, not a raw substring test.
    _kwpat = _pattern(ordered)
    keep["filtered_tech_keys"] = [sorted({m.lower() for m in _kwpat.findall(a)}) for a in abstracts]
    keep["count_citing_patents"] = keep[cm["citing"]].map(count_citing_patents) if cm["citing"] else 0
    state = {"frame": frame, "columns": cm, "keep": keep, "abstracts": abstracts, "pubnos": pubnos,
             "years": years, "lex": lex, "keywords": list(ordered), "keyword_freq": freq,
             "keywords_per_doc": chosen, "decisions": {}, "max_year": max((y for y in years if y), default=None),
             "triples": [get_trt(a, lex) for a in abstracts]}
    print("CONCEPTS  %d unique surface forms (1-4 words), %d per abstract" % (len(ordered), no_keywords))
    b = blocking(state)
    print("BLOCKING  auto-merged pairs=%d, ambiguous band=%d (model adjudicates these), rest rejected"
          % (b["auto_merge"], b["ambiguous"]))
    resolve_groups(state)
    r = state["relations"]
    print("RELATIONS %d chained, span-grounded rows with preposition-default types; legacy classes: %s"
          % (len(r), ", ".join("%s %d" % (c, n) for c, n in Counter(LEGACY_MAP[x] for x in r["relation"]).most_common())))
    cohort_percentile(state)
    H = censoring_horizon(state)
    k = shrinkage_k(state)
    print("INFLUENCE cohort percentile (year +/-1, domain when cohort >= 30); censoring horizon H=%s years%s; "
          "shrinkage k=%.2f toward mu=%.3f"
          % (H, "" if state.get("H_identified") else " (default, not identifiable here)", k, state["mu"]))
    if cm["domain"]:
        doms = Counter(d for v in keep[cm["domain"]] for d in split_domains(v))
        print("DOMAINS   %s" % ", ".join("%s (%d)" % (d, n) for d, n in doms.most_common(show)))
        state["domains"] = [d for d, _ in doms.most_common()]
        td = state["domains"][0]
        t = ds_table(state, td).head(8)
        print("DS [%s]  top by log-odds z (lift shown only at DF_TD >= 3):" % td)
        print("  " + t.to_string(index=False).replace("\n", "\n  "))
    else:
        state["domains"] = []
    e = emergence_table(state, top=8)
    if len(e):
        print("EMERGENCE top by burst then slope (citation-free):")
        print("  " + e[["term", "decayed_df", "slope", "acceleration", "first_age", "burst", "persistence"]]
              .to_string(index=False).replace("\n", "\n  "))
    print("NEXT      adjudication_batches(state) -> apply_adjudications -> resolve_groups; "
          "relation_batches -> apply_relation_types -> typed_graph; trend(state, term, td=...)")
    return state
