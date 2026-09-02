"""Deterministic half of the TechScope Gem. Attach as a Gem knowledge file.

Runs under Gemini's code-execution sandbox: standard library, pandas and numpy
only. No spaCy, no sentence-transformers, no network.

The model must never compute these numbers by reading. It calls report() and
reads the output. Counting is what code is for; judging what the counts mean is
what the model is for.

Two substitutions are forced by the sandbox, and both are documented where they
occur:
  * candidate phrases come from stopword segmentation (RAKE, Rose et al. 2010)
    rather than a dependency parse
  * synonym grouping is left to the model, because there is no embedding model
    here — and the model is better at it anyway

    from techscope_core import report
    report("patents.csv", title="Title", abstract="Abstract", date="Date")
"""

from __future__ import annotations

import datetime
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# text normalisation
# --------------------------------------------------------------------------- #

_PAREN_HEAD = re.compile(r"^\s*\([^)]*\)\s*\n")
_JP_HEADINGS = re.compile(
    r"\b(?:PROBLEM TO BE SOLVED|SOLUTION|SELECTED DRAWING|ADVANTAGE|EFFECT)\s*:\s*",
    re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean(text: object) -> str:
    if not isinstance(text, str):
        return ""
    out = _PAREN_HEAD.sub("", text)
    out = _JP_HEADINGS.sub("", out)
    out = out.replace(";", ".").replace("–", "-").replace("—", "-")
    return _WS.sub(" ", out).strip()


# Segmentation stoplist. A phrase never spans one of these, so they act as the
# boundaries a parser would otherwise give us. Deliberately includes patent
# connectives - comprising, wherein, said - which is where the technical noun
# phrases actually break.
_STOP = frozenset("""
a an the this that these those said such its their his her our your some any
each one two three first second third another other same both all each every
and or but nor so yet for nor as if than because while although though whereas
of in on at by to from with without within into onto upon through during before
after above below under over between among across against about around near
is are was were be been being am do does did doing have has had having
can could may might must shall should will would
not no more most less least very much many few several
which who whom whose what when where why how
comprising comprises comprise including includes include consisting consists
wherein whereby thereof therein thereto thereby herein hereof said
according relates relating provided provides provide disclosed discloses
present invention embodiment embodiments example examples figure figures fig
use uses used using being also further still then thus hence therefore
it its they them he she we you i there here
""".split())

# NOTE: head nouns - element, member, portion, unit, apparatus, method - are
# deliberately NOT boundaries. They are legitimate final words of a real term
# ("catalytic sensing element"), and cutting there truncates it. They are
# handled below, where a phrase can shed them and still be proposed.

# A phrase ending in one of these is a category, not a technology. Dropped only
# when the phrase would be nothing but such words.
_GENERIC = frozenset("""
apparatus method system device assembly arrangement means unit mechanism
structure module equipment machine process technique procedure application
""".split())

_TOKEN = re.compile(r"[a-z][a-z0-9/-]*")


def _emit(run: list[str], lo: int, hi: int, out: list[str]) -> None:
    """Every candidate this run supports."""
    out.extend(_windows(run, lo, hi))
    # "cryogenic storage tank apparatus" must also propose "cryogenic storage
    # tank". Patent titles append a category noun to almost every phrase, and
    # with no embedding model available to merge the two afterwards, both have
    # to be on the table so C-value and the model can choose between them.
    trimmed = list(run)
    while len(trimmed) > lo and trimmed[-1] in _GENERIC:
        trimmed.pop()
        out.extend(_windows(trimmed, lo, hi))


def _phrases(text: str, lo: int, hi: int) -> list[str]:
    """Candidate phrases by stopword segmentation.

    With no dependency parser available, a phrase is a maximal run of content
    words between stopwords or punctuation. This is RAKE's candidate step. It is
    noisier than noun chunking - it cannot tell a noun phrase from a verb phrase
    - so the model is told to reject what does not name a technology.
    """
    out: list[str] = []
    for sentence in re.split(r"[.!?,:;()\[\]]", text.lower()):
        run: list[str] = []
        for word in sentence.split():
            token = _TOKEN.match(word)
            tok = token.group(0).strip("-/") if token else ""
            if not tok or tok in _STOP or tok.isdigit():
                if run:
                    _emit(run, lo, hi, out)
                run = []
            else:
                run.append(tok)
        if run:
            _emit(run, lo, hi, out)
    return out


def _windows(run: list[str], lo: int, hi: int) -> list[str]:
    """The full run, plus its trailing sub-phrases.

    "cryogenic storage tank" should also propose "storage tank", so that C-value
    can decide between them rather than us guessing. Trailing windows only: a
    leading fragment like "cryogenic storage" is rarely the real term.
    """
    out = []
    n = len(run)
    for size in range(lo, min(hi, n) + 1):
        for start in range(0, n - size + 1):
            if start + size == n or size == n:
                out.append(" ".join(run[start:start + size]))
    return out


def _singular(word: str) -> str:
    """Crude lemmatiser for the head noun. No model available."""
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ches", "ch"),
                         ("shes", "sh"), ("xes", "x"), ("ses", "s")):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)] + repl
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


_EVALUATIVE = frozenset("""improved novel new enhanced advanced exemplary
    preferred certain various particular""".split())


def _normalise(phrase: str) -> str | None:
    words = phrase.split()
    # "improved sensor array" and "sensor array" are one technology; the
    # adjective is the applicant selling it.
    while len(words) > 1 and words[0] in _EVALUATIVE:
        words = words[1:]
    if not words:
        return None
    words = words[:-1] + [_singular(words[-1])]
    if all(w in _GENERIC for w in words):
        return None
    return " ".join(words)


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #

_YEAR = re.compile(r"(?<!\d)((?:1[6-9]|2[01])\d{2})(?!\d)")
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def year_of(value: object) -> int | None:
    """The year, or None. Never a guess.

    The lookarounds matter: an unanchored four-digit match reads
    "US2024123456A1" as 2024 and fills a trend chart with fiction.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.year
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = int(value)
        if 20000 <= n <= 80000:                      # Excel serial
            return (_EXCEL_EPOCH + pd.Timedelta(days=n)).year
        return n if 1600 <= n <= 2199 else None
    m = _YEAR.search(str(value))
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def c_values(df: Counter, n_docs: int, min_freq: int, max_ratio: float) -> dict:
    """C-value termhood (Frantzi, Ananiadou & Mima 2000).

        C(a) = log2|a| * f(a)                          a nested in nothing
        C(a) = log2|a| * ( f(a) - mean f(b), b contains a )

    Raw frequency always prefers the shorter phrase, because every occurrence of
    "cryogenic storage tank" is also an occurrence of "storage tank". This
    subtracts the frequency a term only has by sitting inside a longer one.
    """
    ceiling = max(int(max_ratio * n_docs), min_freq)
    terms = [t for t, c in df.items() if c >= min_freq and c <= ceiling]
    if not terms:
        terms = [t for t, c in df.items() if c >= min_freq]

    by_len = defaultdict(list)
    for t in terms:
        by_len[len(t.split())].append(t)
    longest = max(by_len) if by_len else 0

    containers = defaultdict(list)
    for n in sorted(by_len):
        for shorter in by_len[n]:
            pad = " %s " % shorter
            for m in range(n + 1, longest + 1):
                for longer in by_len.get(m, ()):
                    if pad in " %s " % longer:
                        containers[shorter].append(longer)

    out = {}
    for t in terms:
        size = math.log2(len(t.split()) or 1) or 1.0     # unigrams weigh 1, not 0
        nested = containers.get(t)
        if nested:
            out[t] = size * (df[t] - sum(df[b] for b in nested) / len(nested))
        else:
            out[t] = size * df[t]
    return {t: v for t, v in out.items() if v > 0}


def merge_heads(scores: dict, df: Counter, min_words: int):
    """Fold "X apparatus" into "X" when both were independently extracted.

    The deterministic half of canonicalisation. TechScope does this with a
    string rule before its embedding pass, and the condition is what makes it
    safe: the stripped form must already be a term in its own right, so this
    never invents a name and never merges two things that merely share a head
    noun. "pipeline welding method" with no bare "pipeline welding" beside it is
    left exactly as it is.

    What is left over - "cryogenic tank" and "cryogenic storage tank", which no
    string rule can relate - is the model's job.
    """
    canonical, variants = {}, defaultdict(list)
    for term in scores:
        words = term.split()
        target = None
        while len(words) > min_words and words[-1] in _GENERIC:
            words = words[:-1]
            candidate = " ".join(words)
            if candidate in scores:
                target = candidate
        canonical[term] = target or term
        if target:
            variants[target].append(term)

    survivors = {t: scores[t] for t, name in canonical.items() if t == name}
    return survivors, dict(variants), canonical


def ppmi_edges(per_doc, keep, min_docs: int, limit: int) -> pd.DataFrame:
    """Term pairs by positive pointwise mutual information.

        PMI(a,b) = log2 [ p(a,b) / (p(a) p(b)) ]     PPMI = max(0, PMI)

    Raw co-occurrence ranks the two commonest terms top whether or not they are
    related. Dividing by what independence would predict leaves association
    above chance.
    """
    n_docs = len(per_doc)
    single, pair = Counter(), Counter()
    for terms in per_doc:
        kept = sorted(set(terms) & keep)
        single.update(kept)
        for i, a in enumerate(kept):
            for b in kept[i + 1:]:
                pair[(a, b)] += 1

    rows = []
    for (a, b), joint in pair.items():
        if joint < min_docs:
            continue
        pa, pb, pab = single[a] / n_docs, single[b] / n_docs, joint / n_docs
        if not (pa and pb):
            continue
        val = math.log2(pab / (pa * pb))
        if val > 0:
            rows.append({"a": a, "b": b, "docs": joint, "ppmi": round(val, 3)})
    frame = pd.DataFrame(rows, columns=["a", "b", "docs", "ppmi"])
    if frame.empty:
        return frame
    return (frame.sort_values(["ppmi", "docs"], ascending=False)
                 .head(limit).reset_index(drop=True))


def period_table(per_doc, years, keep, n_periods: int) -> pd.DataFrame:
    """Share of documents per period, and the least-squares slope of that share.

    Share, not count: a period that simply contains more patents would otherwise
    read as growth for every term in it. Slope over all periods, not last minus
    first, which discards the middle.
    """
    seen = sorted({y for y in years if y})
    if not seen:
        return pd.DataFrame()
    lo, hi = seen[0], seen[-1] + 1
    n = max(1, min(n_periods, hi - lo))
    step = (hi - lo) / n
    edges = sorted({lo + int(round(i * step)) for i in range(n + 1)} | {lo, hi})
    buckets = [("%d-%d" % (a, b - 1) if b - 1 > a else str(a), a, b)
               for a, b in zip(edges, edges[1:])]

    doc_terms = [set(t) & keep for t in per_doc]
    sizes, tallies = [], []
    for _label, a, b in buckets:
        idx = [i for i, y in enumerate(years) if y and a <= y < b]
        tally = Counter()
        for i in idx:
            tally.update(doc_terms[i])
        sizes.append(len(idx))
        tallies.append(tally)

    rows = []
    for term in sorted(keep):
        counts = [t[term] for t in tallies]
        shares = [c / s if s else 0.0 for c, s in zip(counts, sizes)]
        slope = float(np.polyfit(np.arange(len(shares), dtype=float), shares, 1)[0]) \
            if len(shares) > 1 else 0.0
        row = {"term": term, "docs": sum(counts), "growth": round(slope, 5)}
        row.update({lab: c for (lab, _a, _b), c in zip(buckets, counts)})
        rows.append(row)
    return (pd.DataFrame(rows).sort_values("docs", ascending=False)
                              .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def report(path, title=None, abstract=None, date=None, *, min_words=2,
           max_words=5, min_freq=2, max_ratio=0.60, top_terms=150,
           top_edges=200, periods=6, min_edge_docs=2):
    """Compute everything and print it for the model to read.

    Returns the objects too, so a follow-up question can be answered by more
    code rather than by recall.
    """
    frame = (pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls"))
             else pd.read_csv(path))
    cols = list(frame.columns)

    def pick(explicit, *needles):
        if explicit:
            if explicit not in cols:
                raise SystemExit("column %r not in file. Columns: %s"
                                 % (explicit, ", ".join(map(str, cols))))
            return explicit
        for n in needles:
            for c in cols:
                if n in str(c).lower():
                    return c
        return None

    title = pick(title, "title")
    abstract = pick(abstract, "abstract", "summary")
    date = pick(date, "date", "year", "priority")
    if abstract is None:
        raise SystemExit("no abstract column found. Columns: %s"
                         % ", ".join(map(str, cols)))
    if title == abstract:
        raise SystemExit("title and abstract are the same column; text would be "
                         "counted twice")

    kept = frame[frame[abstract].notna()].reset_index(drop=True)
    titles = [clean(v) for v in kept[title]] if title else [""] * len(kept)
    abstracts = [clean(v) for v in kept[abstract]]
    years = [year_of(v) for v in kept[date]] if date else [None] * len(kept)
    docs = [("%s. %s" % (t, a)) if t else a for t, a in zip(titles, abstracts)]

    per_doc, df = [], Counter()
    for text in docs:
        seen = {}
        for phrase in _phrases(text, min_words, max_words):
            norm = _normalise(phrase)
            if norm and min_words <= len(norm.split()) <= max_words:
                seen.setdefault(norm, None)
        # keys(), not the dict: Counter.update on a mapping reads its values as
        # counts, and these are None. Document frequency counts each record once.
        per_doc.append(list(seen))
        df.update(seen.keys())

    scores = c_values(df, len(docs), min_freq, max_ratio)
    scores, variants, canon = merge_heads(scores, df, min_words)
    ranked = sorted(scores, key=lambda t: (-scores[t], t))[:top_terms]
    keep = set(ranked)

    # Terms must be mapped through the merge before co-occurrence, or a term
    # pairs with its own variant and tops the PPMI table with a self-edge.
    per_doc = [[canon.get(t, t) for t in terms] for terms in per_doc]
    group_df = Counter()
    for terms in per_doc:
        group_df.update(set(terms) & keep)
    edges = ppmi_edges(per_doc, keep, min_edge_docs, top_edges)
    trends = period_table(per_doc, years, keep, periods) if any(years) else pd.DataFrame()

    dated = sum(1 for y in years if y)
    print("CORPUS")
    print("  rows in file          : %d" % len(frame))
    print("  usable (has abstract) : %d" % len(docs))
    print("  columns used          : title=%r abstract=%r date=%r"
          % (title, abstract, date))
    print("  rows with a real year : %d of %d" % (dated, len(docs)))
    if date and dated == 0:
        print("  WARNING: no year parsed from %r. Trends unavailable. Do not "
              "estimate them." % date)
    elif date and dated < len(docs) * 0.5:
        print("  WARNING: fewer than half the rows have a usable year; trends "
              "cover part of the corpus only.")
    print()

    print("CANDIDATE TERMS  (%d, ranked by C-value; df = documents, not mentions)"
          % len(ranked))
    print("  %-38s %8s %6s  %s" % ("term", "c_value", "df", "folded in"))
    for t in ranked:
        also = ", ".join(sorted(variants.get(t, []))[:2])
        print("  %-38s %8.2f %6d  %s" % (t[:38], scores[t], group_df[t], also[:40]))
    print()
    print("  NOTE: only exact head-noun variants are folded above. Synonyms that")
    print("  no string rule can relate are still separate rows and are yours to")
    print("  merge - see step 3 of your instructions.")
    print()

    print("CO-OCCURRENCE  (top %d pairs by PPMI)" % len(edges))
    if edges.empty:
        print("  none above threshold")
    else:
        for _, e in edges.iterrows():
            print("  %-32s %-32s docs=%-5d ppmi=%.3f"
                  % (e["a"][:32], e["b"][:32], e["docs"], e["ppmi"]))
    print()

    if not trends.empty:
        print("TRENDS  (counts per period; growth = OLS slope of document share)")
        print(trends.to_string(index=False))
    else:
        print("TRENDS: unavailable (no usable date column)")

    return {"frame": kept, "docs": docs, "per_doc": per_doc, "df": group_df,
            "scores": scores, "ranked": ranked, "edges": edges,
            "trends": trends, "years": years, "variants": variants,
            "canonical": canon,
            "columns": {"title": title, "abstract": abstract, "date": date}}


def documents_for(state, term, limit=15):
    """The records behind a term, so a claim can be checked rather than trusted."""
    hits = [i for i, terms in enumerate(state["per_doc"]) if term in terms]
    frame = state["frame"].iloc[hits[:limit]]
    cols = [c for c in (state["columns"]["title"], state["columns"]["date"]) if c]
    extra = [c for c in frame.columns
             if re.search(r"pub|number|id|appl", str(c), re.I)][:1]
    return frame[extra + cols] if extra else frame[cols]
