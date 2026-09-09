"""Track C - the intent-optimal system, per the intent-preserving redesign.

Attach as a Gemini Gem knowledge file. Runs in the code-execution sandbox:
standard library, pandas and numpy only.

The system is organised around a patent-evidence knowledge layer from which
technologies, functions, relations, graphs, trends and analyst views are
derived. The model reasons over evidence; it does not replace the evidence
layer. Deterministic NLP generates candidates, cheap similarity blocks pairs,
the model makes the contextual judgements, statistical estimators produce the
quantitative trends, and every accepted output carries an evidence object.

Sections of the redesign and where they live here:
  3.1  hybrid concept extraction        candidate_phrases, concept_batches,
                                        apply_concept_judgements, propose_concepts,
                                        concept_evidence
  4    graph entity resolution          blocking, pair_record, adjudication_batches,
                                        apply_adjudications, resolve_groups
  5,6  typed relations with direction   relation_batches, apply_relation_types,
       and strength                     typed_graph, legacy_view, relation_record
  7    structured functional concepts   functional_concepts (verb + object)
  8    Bayesian enrichment + ablations  ds_table, ds_ablation
  9    emergence signals                prevalence, emergence, emergence_table
  10   cohort-normalised influence      cohort_percentile
  11   robust FS with bootstrap CI      influence_of
  12   multi-window trajectories        trajectory, adoption_curve
  13   uncertainty-aware states         trend
  14   provenance objects               provenance
  18   temporal backtest                backtest_emergence
  19   frozen objective                 WEIGHTS, objective_score

Sandbox stand-ins (substitutions, not fixes): fixed preposition list for
ADP; corpus-internal inflection test for verb lemmas; character-trigram
cosine as the blocking embedding; PPMI context-vector cosine as the
deterministic contextual-equivalence signal.

    from trt_intent import report
    state = report("patents.xlsx")
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
PROMPT_VERSION = "judgement-v1"
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


def split_domains(value):
    return list(dict.fromkeys(d.strip() for d in str(value).split("\n") if d.strip())) if isinstance(value, str) else []


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
# 3.1 hybrid concept extraction
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


def candidate_phrases(text, lex, lo=1, hi=5):
    """Deterministic candidate layer: maximal content runs of 1-5 words
    (noun phrases and technical compounds). No fixed n-gram window: a
    six-word run is kept as its trailing five words rather than dropped."""
    out = []
    for sent in re.split(r"[.!?,:;()\[\]]", text.lower()):
        run = []
        for word in sent.split() + [""]:
            m = _WORD.match(word)
            tok = m.group(0) if m else ""
            if not tok or tok in _STOP or tok.isdigit():
                run = _trim_verbs(run, lex)
                if len(run) > hi:
                    run = run[-hi:]
                # A single letter is a chemical variable, not a technology.
                if len(run) == 1 and len(run[0]) < 3:
                    run = []
                if lo <= len(run) and not all(w in _GENERIC for w in run):
                    out.append(" ".join(run))
                run = []
            else:
                run.append(tok)
    return out


def candidate_inventory(docs, lex, min_df=2, per_doc=7):
    """Every candidate seen in at least min_df patents, plus the top per_doc
    candidates of each patent by tf-idf so rare but salient concepts survive
    to be judged. Returns per-doc candidate lists, the ordered inventory and
    document frequencies."""
    per = [candidate_phrases(d, lex) for d in docs]
    df = Counter()
    for c in per:
        df.update(set(c))
    n = len(docs)
    keep = {p for p, k in df.items() if k >= min_df}
    for cands in per:
        tf = Counter(cands)
        top = sorted(tf, key=lambda p: (-(tf[p] * (math.log((n + 1) / (df[p] + 1)) + 1)), p))[:per_doc]
        keep.update(top)
    ordered = sorted(keep, key=lambda p: (-df[p], p))
    return per, ordered, df


def concept_batches(state, size=40, only_unjudged=True):
    """Surface forms for the model to judge: does this name a technological
    concept (a technology, material, component, or function) rather than a
    generic noun, a fragment, or boilerplate? Each item carries one example
    sentence. Judgement is per surface form, not per patent."""
    items = []
    for p in state["keywords"]:
        c = state["concepts"][p]
        if only_unjudged and c["judged_by"] == "model":
            continue
        items.append({"surface": p, "df": state["keyword_freq"][p], "example": _example(state, p)})
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    print("CONCEPTS  %d surface forms to judge in %d batches of %d" % (len(items), len(batches), size))
    return batches


def apply_concept_judgements(state, judgements):
    """judgements: list of {surface, accept: bool, confidence?: float, reason?}.
    Rejected forms leave the inventory; the groups and relations are rebuilt."""
    n_rej = 0
    for j in judgements:
        p = str(j.get("surface", "")).lower().strip()
        if p not in state["concepts"]:
            continue
        acc = bool(j.get("accept"))
        state["concepts"][p].update({"accepted": acc, "confidence": float(j.get("confidence", 0.9 if acc else 0.1)),
                                     "judged_by": "model", "reason": str(j.get("reason", ""))})
        n_rej += (not acc)
    state["keywords"] = [p for p in state["keywords"] if state["concepts"][p]["accepted"]]
    print("CONCEPTS  %d judgements applied; %d rejected; %d concepts remain" % (len(judgements), n_rej, len(state["keywords"])))
    _refresh(state)
    return len(state["keywords"])


def propose_concepts(state, proposals):
    """Model-proposed missed concepts survive only if aligned to explicit
    source text: {concept, pubno, span} with span verbatim in that abstract
    and containing the concept."""
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
            state["concepts"].setdefault(c, {"accepted": True, "confidence": 0.8, "judged_by": "model", "reason": "proposed"})
            if c not in state["keywords"]:
                state["keywords"].append(c)
    print("PROPOSALS  accepted %d, rejected %d" % (len(accepted), len(rejected)))
    if accepted:
        _refresh(state)
    return {"accepted": accepted, "rejected": rejected}


def concept_evidence(state, concept, limit=20):
    """Evidence records in the shape the redesign specifies."""
    forms = [m for m, c in state["owner"].items() if c == concept] or [concept]
    cm, keep = state["columns"], state["keep"]
    out = []
    for i, (a, pub) in enumerate(zip(state["abstracts"], state["pubnos"])):
        for j, s in enumerate(_sentences(a)):
            hit = next((f for f in forms if f in s.lower()), None)
            if hit:
                out.append({"canonical_concept": concept, "surface_form": hit, "publication_id": pub,
                            "sentence_id": "abstract_%02d" % j, "evidence_span": s[:240],
                            "document_section": "abstract",
                            "confidence": state["concepts"].get(hit, {}).get("confidence", 0.5),
                            "domain": (split_domains(keep.at[i, cm["domain"]]) or [None])[0] if cm["domain"] else None,
                            "year": state["years"][i]})
                break
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- #
# 4 normalisation - blocking, pair records, adjudication, communities
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
    """PPMI-weighted context vectors per phrase, plus the raw context counts."""
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
                    ctx[" ".join(words)].update(w for w in toks[max(0, i - window):i] + toks[i + n:i + n + window]
                                                if w not in _STOP)
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
    return m / norm, ctx


def _key(a, b):
    a, b = sorted((a, b))
    return hashlib.sha256("|".join([a, b, PROMPT_VERSION, MODEL_ID, str(TEMPERATURE)]).encode()).hexdigest()[:16]


def blocking(state, k=20, hi=0.90, lo=0.55):
    """Embedding-style candidate retrieval (top-k trigram neighbours), then a
    band split: above hi merged automatically, below lo never paired, between
    them the model's contextual judgement. Embeddings block; they do not decide."""
    phrases = list(state["keywords"])
    lex_v = _trigram_vectors(phrases)
    S = lex_v @ lex_v.T
    ctx_v, ctx = context_vectors(phrases, state["abstracts"])
    C = ctx_v @ ctx_v.T
    pairs = {}
    for i, p in enumerate(phrases):
        for j in np.argsort(-S[i])[:k + 1]:
            if j == i or S[i, j] < lo:
                continue
            a, b = sorted((p, phrases[j]))
            pairs[(a, b)] = (float(S[i, j]), float(C[i, j]))
    auto, band = {}, {}
    for (a, b), (s, c) in sorted(pairs.items()):
        (auto if s >= hi else band)[(a, b)] = {"lexical": round(s, 3), "contextual": round(c, 3)}
    state.update({"pairs_auto": auto, "pairs_band": band, "similarity": (phrases, S, C), "contexts": ctx})
    return {"auto_merge": len(auto), "ambiguous": len(band)}


def pair_record(state, a, b):
    """The pairwise record from section 4: similarity, contextual equivalence,
    and counts of supporting / contradicting contexts (shared context words
    versus words strongly tied to one phrase and absent for the other)."""
    phrases, S, C = state["similarity"]
    idx = {p: i for i, p in enumerate(phrases)}
    ca, cb = state["contexts"].get(a, Counter()), state["contexts"].get(b, Counter())
    shared = set(ca) & set(cb)
    top_a = {w for w, _ in ca.most_common(8)}
    top_b = {w for w, _ in cb.most_common(8)}
    contradicting = len((top_a - set(cb)) | (top_b - set(ca)))
    return {"phrase_a": a, "phrase_b": b,
            "semantic_similarity": round(float(S[idx[a], idx[b]]), 3) if a in idx and b in idx else None,
            "contextual_equivalence_probability": round(float(C[idx[a], idx[b]]), 3) if a in idx and b in idx else None,
            "supporting_contexts": len(shared), "contradicting_contexts": contradicting}


def _example(state, phrase):
    for a, pub in zip(state["abstracts"], state["pubnos"]):
        for s in _sentences(a):
            if phrase in s.lower():
                return "%s: %s" % (pub, s[:220])
    return ""


def adjudication_batches(state, size=20):
    """The ambiguous band, with the pair record and one example sentence each.
    The question is contextual substitutability in this corpus, not topical
    relatedness. Decisions are cached under a key so a rerun reproduces them."""
    items = []
    for (a, b) in state["pairs_band"]:
        key = _key(a, b)
        if key in state["decisions"]:
            continue
        rec = pair_record(state, a, b)
        rec.update({"key": key, "example_a": _example(state, a), "example_b": _example(state, b)})
        items.append(rec)
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    print("ADJUDICATION  %d pairs pending in %d batches of %d" % (len(items), len(batches), size))
    return batches


def apply_adjudications(state, decisions):
    """decisions: list of {key | (phrase_a, phrase_b), equivalent: bool, reason}."""
    for d in decisions:
        a, b = d.get("phrase_a") or d.get("a"), d.get("phrase_b") or d.get("b")
        key = d.get("key") or _key(a, b)
        state["decisions"][key] = {"equivalent": bool(d.get("equivalent")), "reason": str(d.get("reason", "")),
                                   "a": a, "b": b}
    print("ADJUDICATION  %d decisions on record" % len(state["decisions"]))
    return len(state["decisions"])


def export_decisions(state):
    return json.dumps({"decisions": state["decisions"],
                       "concepts": {p: c for p, c in state["concepts"].items() if c["judged_by"] == "model"}},
                      sort_keys=True)


def import_decisions(state, text):
    data = json.loads(text)
    state["decisions"].update(data.get("decisions", {}))
    for p, c in data.get("concepts", {}).items():
        if p in state["concepts"]:
            state["concepts"][p].update(c)
    return len(state["decisions"])


def resolve_groups(state, floor=0.25):
    """Weighted equivalence graph -> communities -> cluster consistency
    validation -> canonical concept. Each member must be compatible with the
    canonical member (merged directly, or lexically close with contextual
    support above floor, and never adjudicated non-equivalent); members that
    fail are split off and resolved separately. This is the anti-chaining
    requirement: A~B, B~C does not make A~C."""
    phrases, S, C = state["similarity"]
    idx = {p: i for i, p in enumerate(phrases)}
    freq = state["keyword_freq"]
    pos, neg = set(state["pairs_auto"]), set()
    for d in state["decisions"].values():
        pair = tuple(sorted((d.get("a") or "", d.get("b") or "")))
        if pair[0] in idx and pair[1] in idx:
            (pos if d["equivalent"] else neg).add(pair)
    adj = defaultdict(set)
    for a, b in pos - neg:
        adj[a].add(b)
        adj[b].add(a)
    groups, evidence, seen = {}, {}, set()

    def canonical(members):
        return sorted(members, key=lambda p: (-freq[p], len(p), p))[0]

    def compatible(c, m):
        if tuple(sorted((c, m))) in neg:
            return False
        return m in adj[c] or (S[idx[c], idx[m]] >= 0.55 and C[idx[c], idx[m]] >= floor)

    def resolve(members):
        c = canonical(sorted(members))
        keep, split = [c], []
        for m in sorted(members):
            if m != c:
                (keep if compatible(c, m) else split).append(m)
        groups[c] = keep
        for m in keep:
            if m != c:
                pair = tuple(sorted((c, m)))
                d = next((v for v in state["decisions"].values()
                          if tuple(sorted((v.get("a") or "", v.get("b") or ""))) == pair), None)
                evidence[m] = {"canonical": c, "rule": "adjudicated" if d else "auto",
                               "reason": d["reason"] if d else "trigram cosine >= hi", **pair_record(state, c, m)}
        if split:
            resolve(split)

    for p in sorted(phrases, key=lambda p: (-freq[p], p)):
        if p in seen:
            continue
        comp, stack = set(), [p]
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x)
                stack.extend(adj[x] - comp)
        seen |= comp
        resolve(comp)
    state["groups"], state["merge_evidence"] = groups, evidence
    state["owner"] = {m: c for c, ms in groups.items() for m in ms}
    _map_relations(state)
    _map_terms(state)
    pending = sum(1 for (a, b) in state["pairs_band"] if _key(a, b) not in state["decisions"])
    print("GROUPS  %d concepts from %d surface forms; %d merges recorded; %d band pairs pending"
          % (len(groups), len(phrases), len(evidence), pending))
    return groups


def _refresh(state):
    blocking(state)
    resolve_groups(state)


def _map_terms(state):
    """Per-patent concept lists on canonical names."""
    keep = state["keep"]
    owner = state["owner"]
    # Word-boundary matching, not a raw substring test.
    pat = _pattern(state["keywords"])
    keep["filtered_tech_keys"] = [sorted({owner.get(m.lower(), m.lower()) for m in pat.findall(a)})
                                  for a in state["abstracts"]]


# --------------------------------------------------------------------------- #
# 5, 6 relations - rich ontology, direction, strength, legacy projection
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
                "with": "implemented-with", "by": "implemented-with", "on": "located-in", "at": "located-in",
                "from": "sourced-from", "for": "used-for", "to": "acts-on", "across": "acts-on",
                "against": "acts-on", "during": "mediated-by", "into": "mediated-by", "through": "mediated-by",
                "via": "mediated-by", "as": "functions-as"}
RELATIONSHIPS = ["Inclusion", "Objective", "Effect", "Process", "Likeness", "Misc"]


def get_trt(text, lex):
    """Preposition anchors, walked outward to the next preposition; the walks
    chain, so one sentence yields a connected path. Sentence ids are kept."""
    trt = []
    for j, sent in enumerate(_sentences(text)):
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
            trt.append((t1, tok, re.sub(r"\.", "", t2), sent, "abstract_%02d" % j))
    return trt


def _pattern(terms):
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE) \
        if terms else re.compile(r"(?!x)x")


def _map_relations(state):
    pat = _pattern(state["keywords"])
    owner = state.get("owner", {})
    rows = []
    for i, (pub, year) in enumerate(zip(state["pubnos"], state["years"])):
        for t1, prep, t2, sent, sid in state["triples"][i]:
            for a in pat.findall(t1):
                for b in pat.findall(t2):
                    sa, sb = owner.get(a.lower(), a.lower()), owner.get(b.lower(), b.lower())
                    if sa != sb:
                        rows.append((len(rows), sa, sb, a.lower(), b.lower(), prep.lower(),
                                     PREP_DEFAULT.get(prep.lower(), "related-to"), pub, year, sid, sent))
    r = pd.DataFrame(rows, columns=["id", "source", "target", "source_surface", "target_surface", "prep",
                                    "relation", "pubno", "year", "sentence_id", "sentence"])
    r["typed_by"], r["confidence"], r["direction"] = "preposition default", 0.5, "source_to_target"
    state["relations"] = r
    return r


def relation_batches(state, size=25, only_untyped=True):
    r = state["relations"]
    if only_untyped:
        r = r[r["typed_by"] == "preposition default"]
    items = [{"id": int(x.id), "source": x.source, "target": x.target, "prep": x.prep, "default": x.relation,
              "sentence": x.sentence} for x in r.itertuples(index=False)]
    batches = [items[i:i + size] for i in range(0, len(items), size)]
    print("RELATIONS  %d rows to type in %d batches; ontology: %s" % (len(items), len(batches), ", ".join(ONTOLOGY)))
    return batches


def apply_relation_types(state, typed):
    """typed: {id, relation, source_span, target_span, confidence, direction?}.
    Accepted only when the relation is in the ontology and both spans are
    verbatim in the sentence and contain the matched surface forms. direction
    may be 'target_to_source' to flip the anchors. Failures are dropped."""
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
            if str(t.get("direction", "")).lower() == "target_to_source":
                r.at[i, "source"], r.at[i, "target"] = r.at[i, "target"], r.at[i, "source"]
                r.at[i, "source_surface"], r.at[i, "target_surface"] = r.at[i, "target_surface"], r.at[i, "source_surface"]
            r.at[i, "relation"], r.at[i, "typed_by"] = rel, "model"
            r.at[i, "confidence"] = float(t.get("confidence", 0.8))
            ok += 1
    print("RELATIONS  accepted %d typed rows; dropped %s" % (ok, dict(dropped) or "none"))
    return {"accepted": ok, "dropped": dict(dropped)}


def relation_record(state, i):
    x = state["relations"].loc[i]
    return {"source_concept": x["source"], "relation": x["relation"], "target_concept": x["target"],
            "direction": x["direction"], "publication_id": x["pubno"], "sentence_id": x["sentence_id"],
            "supporting_sentence": x["sentence"], "confidence": float(x["confidence"]), "year": x["year"]}


def legacy_view(state):
    r = state["relations"].copy()
    r["legacy"] = r["relation"].map(lambda x: LEGACY_MAP.get(x, "Misc"))
    return r


def typed_graph(state, min_patents=1):
    """One row per (source, relation, target): occurrences, distinct patents,
    publication ids, evidence, confidence, first/last year, year distribution.
    A relation in 300 patents and one in 1 do not look alike."""
    rows = []
    for (s, rel, t), g in state["relations"].groupby(["source", "relation", "target"]):
        years = [int(y) for y in g["year"] if y is not None and not pd.isna(y)]
        if g["pubno"].nunique() < min_patents:
            continue
        rows.append({"source": s, "relation": rel, "target": t, "direction": "source_to_target",
                     "legacy": LEGACY_MAP.get(rel, "Misc"), "occurrences": len(g),
                     "distinct_patents": int(g["pubno"].nunique()), "confidence_mean": round(float(g["confidence"].mean()), 2),
                     "first_year": min(years) if years else None, "last_year": max(years) if years else None,
                     "year_distribution": dict(sorted(Counter(years).items())),
                     "publication_ids": list(dict.fromkeys(map(str, g["pubno"]))),
                     "evidence": list(dict.fromkeys(g["sentence"]))[:3]})
    out = pd.DataFrame(rows)
    return out.sort_values(["distinct_patents", "occurrences"], ascending=False).reset_index(drop=True) if len(out) else out


def edge_evidence(state, source, target, relation=None):
    r = state["relations"]
    sub = r[(r["source"] == source) & (r["target"] == target)]
    if relation:
        sub = sub[sub["relation"] == relation]
    return sub[["pubno", "year", "prep", "relation", "typed_by", "confidence", "sentence"]]


# --------------------------------------------------------------------------- #
# 7 structured functional concepts
# --------------------------------------------------------------------------- #

_DET = frozenset("a an the this that these those said such its their each any some one".split())
# Drafting verbs name no function; DS cannot filter them once an object is
# attached ("provide X" is as domain-specific as X). Excluded here, listed so
# the exclusion is visible.
BOILERPLATE_VERBS = frozenset("""provide comprise include arrange connect dispose form relate describe
disclose mount position locate couple configure adapt use show illustrate
represent refer contain consist involve concern""".split())


def functional_concepts(text, lex, max_obj=3):
    """'detect hydrogen leakage' rather than 'detect': the verb lemma plus the
    content run that follows it (skipping determiners). The bare verb is kept
    separately as the legacy field."""
    out = set()
    for sent in _sentences(str(text)):
        toks = _WORD.findall(sent.lower())
        for i, t in enumerate(toks):
            lemma = lex.get(t)
            if lemma is None or lemma in BOILERPLATE_VERBS:
                continue
            j = i + 1
            while j < len(toks) and toks[j] in _DET:
                j += 1
            obj = []
            while j < len(toks) and toks[j] not in _STOP and toks[j] not in lex and len(obj) < max_obj:
                obj.append(toks[j])
                j += 1
            if obj and not all(w in _GENERIC for w in obj):
                out.add("%s %s" % (lemma, " ".join(obj)))
    return sorted(out)


# --------------------------------------------------------------------------- #
# 8 domain specificity - Bayesian enrichment and the candidate ablations
# --------------------------------------------------------------------------- #

def _domain_counts(state, term_col):
    keep, cm = state["keep"], state["columns"]
    k, n, k_all = defaultdict(Counter), Counter(), Counter()
    for doms, terms in zip(keep[cm["domain"]].map(split_domains), keep[term_col]):
        terms = set(terms)
        k_all.update(terms)
        for td in set(doms):
            n[td] += 1
            k[td].update(terms)
    return k, n, k_all, len(keep)


def _estimators(k_d, n_d, k_all, n_all, total_terms, alpha=0.5, beta=0.5, alpha0=10.0, draws=4000, rng=None):
    """The seven candidate estimators of 'how characteristic is F of D', all
    on unique-patent counts with the marginal taken on patents."""
    k_r, n_r = k_all - k_d, n_all - n_d
    out = {}
    p_d, p_all = k_d / n_d if n_d else 0.0, k_all / n_all if n_all else 0.0
    out["lift"] = p_d / p_all if p_all else 0.0
    out["smoothed_lift"] = ((k_d + alpha) / (n_d + alpha + beta)) / ((k_all + alpha) / (n_all + alpha + beta))
    out["pmi"] = math.log2(out["lift"]) if out["lift"] > 0 else float("-inf")
    p_joint = k_d / n_all if n_all else 0.0
    out["npmi"] = out["pmi"] / (-math.log2(p_joint)) if 0 < p_joint < 1 and out["pmi"] != float("-inf") else 0.0
    a_w = alpha0 * k_all / total_terms
    if n_r > 0:
        d = (math.log((k_d + a_w) / (n_d + alpha0 - k_d - a_w)) - math.log((k_r + a_w) / (n_r + alpha0 - k_r - a_w)))
        out["weighted_log_odds_z"] = d / math.sqrt(1 / (k_d + a_w) + 1 / (k_r + a_w))
    else:
        out["weighted_log_odds_z"] = 0.0
    e1, e2 = n_d * k_all / n_all, n_r * k_all / n_all
    g2 = 2 * ((k_d * math.log(k_d / e1) if k_d and e1 else 0) + (k_r * math.log(k_r / e2) if k_r and e2 else 0))
    out["ll_keyness"] = g2 if (n_r == 0 or k_d / n_d >= k_r / n_r) else -g2
    rng = rng or np.random.default_rng(0)
    pd_s = rng.beta(k_d + alpha, n_d - k_d + beta, draws)
    pr_s = rng.beta(k_r + alpha, max(n_r - k_r, 0) + beta, draws) if n_r > 0 else np.full(draws, p_all or 1e-9)
    ratio = pd_s / np.clip(pr_s, 1e-9, None)
    out["bayes_enrichment"] = float(np.median(ratio))
    # k_r == 0: the term never occurs outside the domain, so the enrichment
    # has no finite upper bound; report it as such rather than as a number
    # that depends only on the prior.
    out["bayes_ci"] = (float(np.percentile(ratio, 2.5)), float("inf") if k_r == 0 else float(np.percentile(ratio, 97.5)))
    lo = np.log(pd_s / (1 - pd_s + 1e-9)) - np.log(pr_s / (1 - pr_s + 1e-9))
    out["bayes_log_odds"] = float(np.mean(lo))
    return out


def ds_table(state, td, term_col="Functions", min_support=3):
    """Analyst-facing enrichment: posterior enrichment, 95% credible interval,
    posterior log-odds, supporting patents. Rows below min_support are shown
    but flagged rather than ranked."""
    k, n, k_all, n_all = _domain_counts(state, term_col)
    if td not in n:
        raise SystemExit("domain %r not found. Domains: %s" % (td, ", ".join(sorted(n)[:20])))
    rng = np.random.default_rng(0)
    total = sum(k_all.values()) or 1
    rows = []
    for term, k_d in k[td].items():
        e = _estimators(k_d, n[td], k_all[term], n_all, total, rng=rng)
        rows.append((term, k_d, k_all[term], round(e["bayes_enrichment"], 2), round(e["bayes_ci"][0], 2),
                     round(e["bayes_ci"][1], 2), round(e["bayes_log_odds"], 2), k_d >= min_support))
    out = pd.DataFrame(rows, columns=["term", "supporting_patents", "DF_All", "enrichment", "ci_low", "ci_high",
                                      "log_odds", "sufficient_support"])
    return out.sort_values(["sufficient_support", "ci_low"], ascending=False).reset_index(drop=True)


def ds_ablation(state, td, term_col="Functions", top=10, min_support=3):
    """All seven estimators side by side, their Spearman rank agreement, and
    each one's overlap with the Bayesian top-K. The winner is chosen by
    expert precision at K and stability, not by this table alone."""
    k, n, k_all, n_all = _domain_counts(state, term_col)
    rng = np.random.default_rng(0)
    total = sum(k_all.values()) or 1
    rows = []
    for term, k_d in k[td].items():
        if k_d < min_support:
            continue
        e = _estimators(k_d, n[td], k_all[term], n_all, total, rng=rng)
        rows.append({"term": term, "support": k_d, "corrected_lift": e["lift"], "smoothed_lift": e["smoothed_lift"],
                     "pmi": e["pmi"], "npmi": e["npmi"], "weighted_log_odds": e["weighted_log_odds_z"],
                     "ll_keyness": e["ll_keyness"], "bayes_enrichment": e["bayes_enrichment"]})
    tab = pd.DataFrame(rows)
    if tab.empty:
        return {"table": tab, "spearman": None, "topk_overlap": None}
    cols = ["corrected_lift", "smoothed_lift", "pmi", "npmi", "weighted_log_odds", "ll_keyness", "bayes_enrichment"]
    ranks = tab[cols].rank()
    spearman = ranks.corr()
    tops = {c: set(tab.nlargest(top, c)["term"]) for c in cols}
    overlap = {c: len(tops[c] & tops["bayes_enrichment"]) / top for c in cols}
    return {"table": tab.sort_values("bayes_enrichment", ascending=False).reset_index(drop=True),
            "spearman": spearman.round(2), "topk_overlap": overlap}


# --------------------------------------------------------------------------- #
# 9 emergence - an independent signal
# --------------------------------------------------------------------------- #

def prevalence(state, term, term_col="filtered_tech_keys", td=None, max_year=None):
    """q_t = distinct patents mentioning the concept in year t / distinct patents in t."""
    keep, cm = state["keep"], state["columns"]
    if td:
        keep = keep[keep[cm["domain"]].map(lambda v: td in split_domains(v))]
    tot, hit = Counter(), Counter()
    for y, terms in zip(keep["Year"], keep[term_col]):
        if y is not None and not pd.isna(y) and (max_year is None or y <= max_year):
            tot[int(y)] += 1
            hit[int(y)] += int(term in terms)
    ys = sorted(tot)
    return pd.DataFrame({"year": ys, "patents": [tot[y] for y in ys], "with_term": [hit[y] for y in ys],
                         "q": [hit[y] / tot[y] for y in ys]})


def emergence(state, term, term_col="filtered_tech_keys", td=None, recent=3, max_year=None):
    """Signals, kept separate (section 9):
      ratio          recent mean q / historical mean q
      slope          OLS slope of q over the last 5 years
      acceleration   slope of the recent years minus slope of the ones before
      burst          max z of recent q against the historical mean and sd
      first_year     first appearance; first_age = years since
      persistence    share of recent years with q above the historical median
      recent_domain_enrichment  lift of the concept in td over the recent years
    """
    ymax = max_year or state["max_year"]
    p = prevalence(state, term, term_col, td, ymax)
    if p.empty or ymax is None:
        return None
    q, y = p["q"].to_numpy(), p["year"].to_numpy()
    rec = y >= ymax - recent + 1
    hist = q[~rec]
    prior = (~rec) & (y >= ymax - 2 * recent + 1)
    last5 = y >= ymax - 4
    slope = float(np.polyfit(y[last5], q[last5], 1)[0]) if last5.sum() > 1 else 0.0
    s_rec = float(np.polyfit(y[rec], q[rec], 1)[0]) if rec.sum() > 1 else 0.0
    s_pri = float(np.polyfit(y[prior], q[prior], 1)[0]) if prior.sum() > 1 else 0.0
    first = int(p[p["with_term"] > 0]["year"].min()) if (p["with_term"] > 0).any() else None
    ratio = float(q[rec].mean() / hist.mean()) if rec.any() and len(hist) and hist.mean() > 0 else None
    burst = float(((q[rec] - hist.mean()) / hist.std()).max()) if rec.any() and len(hist) > 1 and hist.std() > 0 else 0.0
    persistence = float((q[rec] > np.median(hist)).mean()) if rec.any() and len(hist) else 0.0
    rde = None
    if td and state["columns"]["domain"]:
        keep, cm = state["keep"], state["columns"]
        recent_rows = keep[(keep["Year"] >= ymax - recent + 1) & (keep["Year"] <= ymax)]
        in_td = recent_rows[cm["domain"]].map(lambda v: td in split_domains(v))
        has = recent_rows[term_col].map(lambda t: term in t)
        p_d = has[in_td].mean() if in_td.any() else 0.0
        p_all = has.mean() if len(has) else 0.0
        rde = round(float(p_d / p_all), 2) if p_all else None
    return {"term": term, "domain": td, "ratio": round(ratio, 2) if ratio is not None else None,
            "slope": round(slope, 5), "acceleration": round(s_rec - s_pri, 5), "burst": round(burst, 2),
            "first_year": first, "first_age": (ymax - first) if first else None,
            "persistence": round(persistence, 2), "recent_domain_enrichment": rde,
            "support_recent": int(p[rec]["with_term"].sum())}


def emergence_table(state, terms=None, term_col="filtered_tech_keys", td=None, top=20, min_support=3):
    terms = terms or list(state["groups"])[:200]
    rows = [e for e in (emergence(state, t, term_col, td) for t in terms) if e and e["support_recent"] >= min_support]
    out = pd.DataFrame(rows)
    return out.sort_values(["burst", "slope"], ascending=False).reset_index(drop=True).head(top) if len(out) else out


# --------------------------------------------------------------------------- #
# 10, 11 influence - cohort percentile, robust aggregate, bootstrap
# --------------------------------------------------------------------------- #

_SEP = re.compile(r"[\n;|,]+")


def count_citing_patents(citing_string):
    if not citing_string or not isinstance(citing_string, str):
        return 0
    return len([p for p in _SEP.split(citing_string) if p.strip()])


def cohort_percentile(state, min_cohort=30):
    """Percentile of the patent's forward-citation count among comparable
    patents: application year +/- 1, and the same first-listed domain when
    that cohort has at least min_cohort patents. No accrual assumption."""
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


def _support(state, term, window, term_col, td):
    keep, cm = state["keep"], state["columns"]
    if td:
        keep = keep[keep[cm["domain"]].map(lambda v: td in split_domains(v))]
    y = keep["Year"].to_numpy(dtype=float)
    m = (y >= window[0]) & (y <= window[1]) & keep[term_col].map(lambda t: term in t).to_numpy()
    return keep[m]["influence"].dropna().to_numpy(), int(((y >= window[0]) & (y <= window[1])).sum())


def influence_of(state, term, window, term_col="filtered_tech_keys", td=None, min_support=5, boots=500, seed=0):
    """Robust aggregate (median) of cohort-normalised influence over the
    distinct patents supporting the concept in the window, with a bootstrap
    95% interval. Small support returns 'Insufficient evidence' instead of a
    point that looks as credible as one backed by hundreds of patents."""
    vals, _ = _support(state, term, window, term_col, td)
    if len(vals) < min_support:
        return {"term": term, "period": "%d-%d" % window, "supporting_patents": int(len(vals)),
                "influence": None, "bootstrap_95": None,
                "note": "Insufficient evidence for a stable influence estimate."}
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(boots)])
    return {"term": term, "period": "%d-%d" % window, "supporting_patents": int(len(vals)),
            "influence": round(float(np.median(vals)), 3),
            "bootstrap_95": {"lower": round(float(np.percentile(meds, 2.5)), 3),
                             "upper": round(float(np.percentile(meds, 97.5)), 3)}}


# --------------------------------------------------------------------------- #
# 12, 13 trajectories and uncertainty-aware states
# --------------------------------------------------------------------------- #

def _windows(state, width):
    ys = [y for y in state["years"] if y]
    lo, hi = min(ys), max(ys)
    out, end = [], hi
    while end - width + 1 >= lo:
        out.append((end - width + 1, end))
        end -= width
    return list(reversed(out))


def trajectory(state, term, width=3, term_col="filtered_tech_keys", td=None, min_support=5):
    rows = []
    for a, b in _windows(state, width):
        vals, pats = _support(state, term, (a, b), term_col, td)
        inf = influence_of(state, term, (a, b), term_col, td, min_support, boots=200)
        rows.append({"window": "%d-%d" % (a, b), "patents": pats, "DF": int(len(vals)),
                     "prevalence": round(len(vals) / pats, 4) if pats else 0.0,
                     "influence": inf["influence"],
                     "ci_low": inf["bootstrap_95"]["lower"] if inf["bootstrap_95"] else None,
                     "ci_high": inf["bootstrap_95"]["upper"] if inf["bootstrap_95"] else None})
    return pd.DataFrame(rows)


def adoption_curve(state, term, term_col="filtered_tech_keys", td=None):
    """Cumulative distinct patents by year. Useful as an adoption history;
    never evidence of decline, because it cannot decrease."""
    p = prevalence(state, term, term_col, td)
    p["cumulative_patents"] = p["with_term"].cumsum()
    return p[["year", "with_term", "cumulative_patents"]]


QUADRANTS = {("up", "up"): "Growing", ("up", "down"): "Generalizing",
             ("down", "down"): "Declining", ("down", "up"): "Repositioning"}
STATES = ["Growing", "Generalizing", "Declining", "Repositioning", "Stable", "Emerging", "Volatile",
          "Insufficient evidence"]


def trend(state, term, widths=(2, 3, 4, 5), term_col="filtered_tech_keys", td=None, min_support=5,
          tol=0.10, boots=300, seed=0):
    """At each width, the latest movement between two windows with sufficient
    support: prevalence direction (relative change beyond tol) and influence
    direction (bootstrap probability that the median rose, >= 0.8 up,
    <= 0.2 down). Widths vote; window_stability is their agreement. States
    follow section 13, with Emerging for short-history acceleration or burst,
    Volatile for substantial but inconsistent movement, and Insufficient
    evidence when support or disagreement prevents a reliable label."""
    rng = np.random.default_rng(seed)
    per_width, moves = {}, {}
    for w in widths:
        wins = _windows(state, w)
        usable = [(a, b, _support(state, term, (a, b), term_col, td)[0]) for a, b in wins]
        usable = [(a, b, v) for a, b, v in usable if len(v) >= min_support]
        if len(usable) < 2:
            per_width[w] = "Insufficient evidence"
            continue
        (a1, b1, v1), (a2, b2, v2) = usable[-2], usable[-1]
        _, n1 = _support(state, term, (a1, b1), term_col, td)
        _, n2 = _support(state, term, (a2, b2), term_col, td)
        q1, q2 = len(v1) / n1, len(v2) / n2
        rel = (q2 - q1) / q1 if q1 else (1.0 if q2 else 0.0)
        d = "up" if rel > tol else "down" if rel < -tol else "flat"
        diffs = np.array([np.median(rng.choice(v2, len(v2), True)) - np.median(rng.choice(v1, len(v1), True))
                          for _ in range(boots)])
        p_up = float((diffs > 0).mean() + 0.5 * (diffs == 0).mean())
        f = "up" if p_up >= 0.8 else "down" if p_up <= 0.2 else "flat"
        signs = [np.sign(np.median(usable[i + 1][2]) - np.median(usable[i][2])) for i in range(max(0, len(usable) - 4), len(usable) - 1)]
        volatile = len(signs) >= 3 and all(signs[i] * signs[i + 1] < 0 for i in range(len(signs) - 1))
        d_raw = "up" if rel > 0 else "down"
        f_raw = "up" if p_up >= 0.5 else "down"
        if volatile:
            label = "Volatile"
        elif (d, f) == ("flat", "flat"):
            label = "Stable"
        else:
            # one axis may be flat: the state set is closed, so the nearest
            # quadrant is taken and the flat axis is recorded in `axes`;
            # the confidence below is discounted by that axis's uncertainty
            label = QUADRANTS[(d if d != "flat" else d_raw, f if f != "flat" else f_raw)]
        per_width[w] = label
        moves[w] = {"from": "%d-%d" % (a1, b1), "to": "%d-%d" % (a2, b2), "d_prevalence_rel": round(float(rel), 3),
                    "p_influence_up": round(p_up, 3), "support": int(len(v2)),
                    "axes": {"prevalence": d, "influence": f},
                    "nearest_quadrant": QUADRANTS[(d_raw, f_raw)]}
    votes = Counter(per_width.values())
    top, count = votes.most_common(1)[0]
    stability = count / len(widths)
    e = emergence(state, term, term_col, td) or {}
    short = e.get("first_age") is not None and e["first_age"] <= 5
    if short and (e.get("burst", 0) >= 2 or e.get("acceleration", 0) > 0) and e.get("slope", 0) > 0:
        label = "Emerging"
    elif stability >= 0.75:
        label = top
    elif top == "Insufficient evidence" or count <= 1:
        label = "Insufficient evidence"
    else:
        label = "Volatile"
    certainties = []
    for m in moves.values():
        p = m["p_influence_up"]
        c = max(p, 1 - p)                       # how sure the influence direction is
        if m["axes"]["prevalence"] == "flat":   # a flat axis borrowed its sign
            c *= 0.5
        certainties.append(c)
    confidence = round(stability * (float(np.mean(certainties)) if certainties else 0.0), 2) \
        if label not in ("Insufficient evidence",) else 0.0
    nearest = Counter(m["nearest_quadrant"] for m in moves.values() if m["nearest_quadrant"]).most_common(1)
    legacy = label if label in QUADRANTS.values() else (nearest[0][0] if nearest and label != "Insufficient evidence" else None)
    support = max((m["support"] for m in moves.values()), default=0)
    return {"term": term, "domain": td, "state": label, "headline": "Likely %s" % label.lower() if confidence >= 0.5 else
            ("Possibly %s" % label.lower() if label != "Insufficient evidence" else label),
            "confidence": confidence, "supporting_patents": support,
            "window_stability": "%d / %d" % (count, len(widths)), "legacy_state": legacy,
            "per_width": {"%dy" % w: v for w, v in per_width.items()},
            "movements": {"%dy" % w: m for w, m in moves.items()}, "emergence": e}


def window_sensitivity(state, terms=None, term_col="filtered_tech_keys", td=None, widths=(2, 3, 4, 5)):
    terms = terms or list(state["groups"])[:40]
    rows = []
    for t in terms:
        r = trend(state, t, widths, term_col, td, boots=100)
        rows.append([t, r["state"], r["window_stability"], r["confidence"]] + [r["per_width"]["%dy" % w] for w in widths])
    out = pd.DataFrame(rows, columns=["term", "state", "stability", "confidence"] + ["%dy" % w for w in widths])
    print("WINDOW SENSITIVITY  %d terms; states: %s" % (len(out), dict(Counter(out["state"]))))
    return out


# --------------------------------------------------------------------------- #
# 14 provenance
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
    """Evidence objects in the section-14 shapes: 'technology', 'relation'
    ((source, target)), 'function' (function, td), 'trend'."""
    if kind == "technology":
        ev = concept_evidence(state, key, kw.get("limit", 25))
        return {"technology": key, "supporting_patents": [{"publication_id": e["publication_id"],
                                                           "sentence": e["evidence_span"]} for e in ev]}
    if kind == "relation":
        s, t = key
        sub = edge_evidence(state, s, t, kw.get("relation"))
        return [{"source": s, "relation": r.relation, "target": t, "publication_id": r.pubno,
                 "sentence": r.sentence, "confidence": float(r.confidence)} for r in sub.itertuples()]
    if kind == "function":
        fn, td = key
        keep, cm = state["keep"], state["columns"]
        rows = keep[keep["Functions"].map(lambda f: fn in f)]
        if td:
            rows = rows[rows[cm["domain"]].map(lambda v: td in split_domains(v))]
        return {"function": fn, "domain": td, "support_count": int(len(rows)),
                "supporting_patents": [{"publication_id": p} for p in rows[cm["pubno"]].head(kw.get("limit", 25))] if cm["pubno"] else []}
    if kind == "trend":
        width, td = kw.get("width", 3), kw.get("td")
        tr = trajectory(state, key, width, kw.get("term_col", "filtered_tech_keys"), td)
        r = trend(state, key, term_col=kw.get("term_col", "filtered_tech_keys"), td=td)
        return {"technology": key, "prevalence": dict(zip(tr["window"], tr["prevalence"])),
                "influence": dict(zip(tr["window"], tr["influence"])), "classification": r["state"],
                "confidence": r["confidence"], "supporting_patents": r["supporting_patents"],
                "window_stability": r["window_stability"]}
    raise ValueError("kind must be technology, relation, function or trend")


# --------------------------------------------------------------------------- #
# 18 temporal backtest for emergence
# --------------------------------------------------------------------------- #

def backtest_emergence(state, cutoff_year, k=10, horizon=3, term_col="filtered_tech_keys", min_support=3):
    """Hide everything after cutoff_year, rank concepts by emergence using
    only what was visible, reveal the next `horizon` years, and ask whether
    the top-K went on to grow. Reports precision@K, average precision and the
    sustained-growth hit rate."""
    terms = list(state["groups"])
    ranked = []
    for t in terms:
        e = emergence(state, t, term_col, max_year=cutoff_year)
        if e and e["support_recent"] >= min_support:
            ranked.append((e["burst"], e["slope"], t))
    ranked.sort(reverse=True)
    if not ranked:
        print("BACKTEST  no concept has support before %d" % cutoff_year)
        return None
    growth = {}
    for _, _, t in ranked:
        p = prevalence(state, t, term_col)
        before = p[(p["year"] > cutoff_year - 3) & (p["year"] <= cutoff_year)]["q"]
        after = p[(p["year"] > cutoff_year) & (p["year"] <= cutoff_year + horizon)]
        growth[t] = (float(after["q"].mean() - before.mean()) if len(after) and len(before) else 0.0,
                     bool(len(after) >= 2 and (after["q"].to_numpy() > before.mean()).sum() >= 2))
    med = float(np.median([g for g, _ in growth.values()]))
    top = [t for _, _, t in ranked[:k]]
    hits = [growth[t][0] > med for t in top]
    precision = sum(hits) / len(top)
    ap = float(np.mean([sum(hits[:i + 1]) / (i + 1) for i, h in enumerate(hits) if h])) if any(hits) else 0.0
    sustained = sum(growth[t][1] for t in top) / len(top)
    print("BACKTEST  cutoff %d, horizon %d, K=%d: precision@K=%.2f, AP=%.2f, sustained-growth hit rate=%.2f (%d candidates)"
          % (cutoff_year, horizon, k, precision, ap, sustained, len(ranked)))
    return {"cutoff": cutoff_year, "top_k": top, "precision_at_k": precision, "average_precision": ap,
            "sustained_growth_rate": sustained, "future_growth": {t: growth[t][0] for t in top}}


# --------------------------------------------------------------------------- #
# 19 frozen multi-objective score
# --------------------------------------------------------------------------- #

WEIGHTS = {"accuracy": 0.20, "scientific_validity": 0.15, "stability": 0.10, "interpretability": 0.10,
           "provenance": 0.15, "temporal_sensitivity": 0.10, "robustness": 0.05, "feasibility": 0.05,
           "user_usefulness": 0.10}


def objective_score(scores):
    """Weighted objective over the nine dimensions, each in [0, 1]. Weights
    are frozen here, before any evaluation set is scored."""
    missing = [k for k in WEIGHTS if k not in scores]
    if missing:
        raise ValueError("missing dimensions: %s" % ", ".join(missing))
    return round(sum(WEIGHTS[k] * float(scores[k]) for k in WEIGHTS), 4)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def report(path, *, show=20, **columns):
    frame, cm = load(path, **columns)
    print("TRACK C - intent-optimal (evidence layer first; the model reasons over it)")
    print("COLUMNS  " + "  ".join("%s=%r" % kv for kv in cm.items()))
    keep = frame[frame[cm["abstract"]].notna()]
    if cm["pubno"]:
        keep = keep[keep[cm["pubno"]].notna()]
    keep = keep.reset_index(drop=True).copy()
    abstracts = [clean_text(a) for a in keep[cm["abstract"]]]
    pubnos = [str(p).strip() for p in keep[cm["pubno"]]] if cm["pubno"] else ["row%d" % i for i in range(len(keep))]
    years = [year_of(v) for v in keep[cm["date"]]] if cm["date"] else [None] * len(keep)
    texts = list(keep[cm["description"]]) if cm["description"] else abstracts
    lex = verb_lexicon(texts + abstracts if cm["description"] else abstracts)
    dated = sum(1 for y in years if y)
    print("CORPUS   rows=%d usable=%d dated=%d years=%s..%s"
          % (len(frame), len(keep), dated, min((y for y in years if y), default=None), max((y for y in years if y), default=None)))
    per, ordered, df = candidate_inventory(abstracts, lex)
    keep["Year"] = years
    keep["Verbs"] = [sorted({lex[w] for w in _WORD.findall(str(t).lower()) if w in lex}) for t in texts]
    keep["Functions"] = [functional_concepts(t, lex) for t in texts]
    keep["count_citing_patents"] = keep[cm["citing"]].map(count_citing_patents) if cm["citing"] else 0
    state = {"frame": frame, "columns": cm, "keep": keep, "abstracts": abstracts, "pubnos": pubnos, "years": years,
             "lex": lex, "keywords": list(ordered), "keyword_freq": Counter({p: df[p] for p in ordered}),
             "concepts": {p: {"accepted": True, "confidence": 0.5, "judged_by": "default", "reason": ""} for p in ordered},
             "decisions": {}, "max_year": max((y for y in years if y), default=None),
             "triples": [get_trt(a, lex) for a in abstracts]}
    print("CONCEPTS  %d candidate surface forms (1-5 words, df >= 2 or top-7 per patent); all provisional until judged"
          % len(ordered))
    print("FUNCTIONS %d structured functional concepts (verb + object) over %d bare verb lemmas"
          % (len({f for fs in keep["Functions"] for f in fs}), len(set(lex.values()))))
    b = blocking(state)
    print("BLOCKING  auto-merged pairs=%d, ambiguous band=%d (model adjudicates), rest rejected" % (b["auto_merge"], b["ambiguous"]))
    resolve_groups(state)
    r = state["relations"]
    print("RELATIONS %d chained, span-grounded rows with preposition-default types; legacy projection: %s"
          % (len(r), ", ".join("%s %d" % (c, n) for c, n in Counter(LEGACY_MAP[x] for x in r["relation"]).most_common())))
    cohort_percentile(state)
    print("INFLUENCE cohort percentile (year +/-1; domain when cohort >= 30); FS = median with bootstrap 95%; "
          "support < 5 -> insufficient evidence")
    if cm["domain"]:
        doms = Counter(d for v in keep[cm["domain"]] for d in split_domains(v))
        state["domains"] = [d for d, _ in doms.most_common()]
        print("DOMAINS   %s" % ", ".join("%s (%d)" % (d, n) for d, n in doms.most_common(show)))
        td = state["domains"][0]
        t = ds_table(state, td).head(8)
        print("ENRICHMENT [%s] functions by posterior enrichment (95%% credible interval, support):" % td)
        print("  " + t.to_string(index=False).replace("\n", "\n  "))
    else:
        state["domains"] = []
    e = emergence_table(state, top=8)
    if len(e):
        print("EMERGENCE top concepts by burst then slope (citation-free, separate from enrichment):")
        print("  " + e[["term", "ratio", "slope", "acceleration", "burst", "first_age", "persistence"]].to_string(index=False).replace("\n", "\n  "))
    print("NEXT      concept_batches -> apply_concept_judgements; adjudication_batches -> apply_adjudications -> "
          "resolve_groups; relation_batches -> apply_relation_types -> typed_graph; ds_table / ds_ablation / "
          "emergence_table / trend / provenance")
    return state
