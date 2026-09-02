"""Technology landscape extraction from patent titles and abstracts.

One job: take a CSV of titles and abstracts and say what technologies are in
there, which names refer to the same thing, how they relate, and — when the file
carries a date — which are rising and which are fading.

Everything runs locally. spaCy for structure, a sentence-transformer for meaning,
plain arithmetic for the rest. No API calls, no Java, no TensorFlow.

    CSV -> clean -> noun chunks -> C-value termhood -> canonical terms
        -> document-term matrix -> PPMI co-occurrence -> graph
        -> [dates] per-period document frequency -> trend quadrants

Design notes
------------
Term extraction uses C-value rather than a top-n cut, because patent language is
full of nested terms: "storage tank" lives inside "cryogenic storage tank", and
raw frequency always prefers the shorter, less informative one.

Canonicalisation uses a *mutual* k-nearest-neighbour graph. Plain thresholding
lets a generic phrase become a hub that swallows everything, and greedy
single-pass grouping is order-dependent and silently drops terms. Mutual kNN plus
connected components is symmetric, deterministic, and loses nothing.

Co-occurrence is weighted by PPMI, not raw counts. Two terms that are both common
co-occur often without being related; PPMI divides that out.
"""

from __future__ import annotations

import datetime
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Config:
    # input
    title_col: str = "Title"
    abstract_col: str = "Abstract"
    date_col: str = ""                     # optional; blank disables trends

    # term extraction
    min_term_words: int = 2
    max_term_words: int = 5
    min_term_freq: int = 2                 # a term must appear in >= this many docs
    max_doc_ratio: float = 0.60            # drop terms in more than this share of docs
    max_terms: int = 400                   # keep the top N by C-value

    # canonicalisation
    neighbours: int = 5                    # k in mutual k-NN
    similarity_floor: float = 0.75         # synonyms sit >0.85, unrelated <0.40

    # relations
    min_edge_docs: int = 2                 # both terms in >= this many shared docs
    max_edges: int = 300

    # trends
    periods: int = 6

    # models
    # Overridable so an offline machine can point at already-downloaded models
    # instead of triggering a fetch on first run.
    spacy_model: str = field(
        default_factory=lambda: os.environ.get("TECHSCOPE_SPACY_MODEL", "en_core_web_sm"))
    embed_model: str = field(
        default_factory=lambda: os.environ.get("TECHSCOPE_EMBED_MODEL", "all-MiniLM-L6-v2"))

    def label(self) -> str:
        return " · ".join(
            f"{f.name}={getattr(self, f.name)}" for f in fields(self)
            if f.name in ("min_term_words", "max_terms", "neighbours",
                          "similarity_floor"))


class Progress(Protocol):
    def __call__(self, stage: str, done: int, total: int) -> None: ...


def _noop(stage: str, done: int, total: int) -> None:
    return None


class InputError(Exception):
    """The CSV cannot be used. Carries every problem, not just the first."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --------------------------------------------------------------------------- #
# stage 1 — load
# --------------------------------------------------------------------------- #

_PAREN_HEAD = re.compile(r"^\s*\([^)]*\)\s*\n")
_JP_HEADINGS = re.compile(
    r"\b(?:PROBLEM TO BE SOLVED|SOLUTION|SELECTED DRAWING|ADVANTAGE|EFFECT)\s*:\s*",
    re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean(text: object) -> str:
    """Normalise one title or abstract."""
    if not isinstance(text, str):
        return ""
    out = _PAREN_HEAD.sub("", text)          # leading "(EP1234A1)\n"
    out = _JP_HEADINGS.sub("", out)          # JPO section headings
    out = out.replace(";", ".").replace("–", "-").replace("—", "-")
    return _WS.sub(" ", out).strip()


@dataclass(slots=True)
class Corpus:
    titles: list[str]
    abstracts: list[str]
    years: list[int | None]
    docs: list[str] = field(repr=False, default_factory=list)

    def __len__(self) -> int:
        return len(self.abstracts)

    @property
    def has_dates(self) -> bool:
        return any(y is not None for y in self.years)


# A year, not any four digits. The negative lookarounds stop it matching inside
# a longer number, which is what let a publication number like "US2024123456A1"
# masquerade as a date and fill a trend chart with fiction.
_YEAR = re.compile(r"(?<!\d)((?:1[6-9]|2[01])\d{2})(?!\d)")

# Excel stores dates as days since 1899-12-30. A bare number in this window is
# far more likely to be that than a year, and pandas hands them back unconverted
# whenever the column has any non-date cell in it.
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_EXCEL_MIN, _EXCEL_MAX = 20000, 80000          # 1954-08-14 .. 2119-01-24


def _year(value: object) -> int | None:
    """The publication year, or None. Never a guess."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.year
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = int(value)
        if _EXCEL_MIN <= n <= _EXCEL_MAX:
            return (_EXCEL_EPOCH + pd.Timedelta(days=n)).year
        return n if 1600 <= n <= 2199 else None
    m = _YEAR.search(str(value))
    return int(m.group(1)) if m else None


def read_table(path_or_buf) -> pd.DataFrame:
    """Read CSV or Excel by sniffing the extension, defaulting to CSV."""
    name = getattr(path_or_buf, "name", str(path_or_buf)).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path_or_buf)
    return pd.read_csv(path_or_buf)


def check(frame: pd.DataFrame, cfg: Config) -> list[str]:
    """Every problem with the table, so the user fixes them in one pass."""
    problems: list[str] = []
    for label, col in (("title", cfg.title_col), ("abstract", cfg.abstract_col)):
        n = list(frame.columns).count(col)
        if n == 0:
            problems.append(
                f"no {label} column named {col!r} — found: "
                f"{', '.join(map(repr, frame.columns[:8]))}")
        elif n > 1:
            problems.append(f"{n} columns are named {col!r}; exactly one is required")
    if cfg.date_col and cfg.date_col not in frame.columns:
        problems.append(f"date column {cfg.date_col!r} is not in the file")
    if cfg.title_col and cfg.title_col == cfg.abstract_col:
        problems.append(
            f"title and abstract are both mapped to {cfg.title_col!r}; the text "
            "would be counted twice and every term inflated")
    if cfg.date_col and cfg.date_col == cfg.abstract_col:
        problems.append(f"date and abstract are both mapped to {cfg.date_col!r}")
    if problems:
        return problems
    if frame.empty:
        problems.append("the file has no rows")
    elif frame[cfg.abstract_col].notna().sum() == 0:
        problems.append(f"every {cfg.abstract_col!r} cell is empty")
    return problems


def date_warning(frame: pd.DataFrame, cfg: Config) -> str | None:
    """Whether the chosen date column actually yields usable years.

    Not fatal, so it is a warning rather than a problem: the rest of the report
    is still valid without trends. But a column that parses to almost nothing -
    Excel serials pandas left as text, or an identifier column picked by mistake
    - would otherwise produce an empty or fictional trend tab in silence.
    """
    if not cfg.date_col or cfg.date_col not in frame.columns:
        return None
    values = frame[cfg.date_col]
    present = values.notna().sum()
    if not present:
        return f"every {cfg.date_col!r} cell is empty, so there are no trends"
    years = [_year(v) for v in values]
    good = sum(1 for y in years if y is not None)
    if good == 0:
        return (f"no year could be read from any {cfg.date_col!r} value "
                f"(first: {values.dropna().iloc[0]!r}) — trends are unavailable")
    if good < present * 0.5:
        return (f"only {good:,} of {present:,} {cfg.date_col!r} values parse as a "
                "year; trends will cover part of the corpus")
    return None


def load(path_or_buf, cfg: Config, frame: pd.DataFrame | None = None) -> Corpus:
    if frame is None:
        frame = read_table(path_or_buf)
    problems = check(frame, cfg)
    if problems:
        raise InputError(problems)

    kept = frame[frame[cfg.abstract_col].notna()].reset_index(drop=True)
    titles = [clean(v) for v in kept[cfg.title_col]] if cfg.title_col in kept \
        else [""] * len(kept)
    abstracts = [clean(v) for v in kept[cfg.abstract_col]]
    years = [_year(v) for v in kept[cfg.date_col]] if cfg.date_col else [None] * len(kept)

    # Titles are short and dense with terminology, and often name a technology
    # the abstract only alludes to. Concatenating means one parse per record
    # instead of two, and document frequency still counts each record once.
    docs = [f"{t}. {a}" if t else a for t, a in zip(titles, abstracts)]
    return Corpus(titles=titles, abstracts=abstracts, years=years, docs=docs)


# --------------------------------------------------------------------------- #
# stage 2 — candidate terms
# --------------------------------------------------------------------------- #

_DET = {"a", "an", "the", "this", "that", "these", "those", "said", "such",
        "its", "their", "his", "her", "our", "your", "some", "any", "each",
        "one", "two", "first", "second", "third", "another", "other", "same"}
_BAD_HEAD = {"invention", "embodiment", "example", "figure", "fig", "means",
             "method", "system", "device", "apparatus", "unit", "portion",
             "member", "element", "part", "step", "process", "use", "present"}


# Patent prose glues these onto almost every noun phrase. They are not part of
# what a technology is called, so a surface form carrying them is a worse label
# for a group than one without. Used only to pick the group's display name -
# never to drop a term, so nothing is lost when the boilerplate word is the only
# head a term has ("welding method" with no bare "welding" alongside it).
_FILLER_HEAD = frozenset("""apparatus method system device assembly arrangement
    means unit mechanism structure module equipment machine process""".split())
_FILLER_MOD = frozenset("""improved novel new enhanced advanced present certain
    various exemplary preferred said such""".split())


def _filler(term: str) -> int:
    """How much patent boilerplate this surface form carries."""
    words = term.split()
    return (sum(w in _FILLER_HEAD for w in words[1:])
            + sum(w in _FILLER_MOD for w in words))


def _can_continue(tok) -> bool:
    """Whether a token can sit inside a technology name.

    The split that matters is participle tense. A past participle is a genuine
    premodifier - "distributed sensor array", "additively manufactured heat
    exchanger". A present participle opens a relative clause, and spaCy hands
    back the whole span as one noun chunk: "fuel cell membrane comprising
    catalyst layer". Cutting at the VBG keeps the head phrase and drops a term
    that names two things at once.
    """
    return (tok.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")
            or tok.tag_ == "VBN"
            or (tok.pos_ == "ADV" and tok.dep_ == "advmod"))


def _can_start(toks, i: int) -> bool:
    """Whether a token can be the first word of a technology name."""
    tok = toks[i]
    if tok.pos_ in ("NOUN", "PROPN", "ADJ"):
        return True
    if tok.tag_ == "VBN":                                  # "distributed array"
        return True
    # a bare adverb only earns its place in front of a participle
    return (tok.pos_ == "ADV" and i + 1 < len(toks)
            and toks[i + 1].tag_ == "VBN")                 # "additively manufactured"


def _finish(toks) -> str | None:
    """Turn one run of term-worthy tokens into a normalised term, or reject it."""
    # Drop leading words that cannot begin a name, plus evaluative adjectives
    # like "improved", which are the applicant's framing rather than part of
    # what the thing is called - otherwise "improved sensor array" and "sensor
    # array" count as two technologies. Trailing heads are left alone; a
    # "welding method" really is called that, and stripping the head here would
    # merge genuinely distinct terms.
    while toks and (not _can_start(toks, 0)
                    or (len(toks) > 1 and toks[0].lower_ in _FILLER_MOD)):
        toks = toks[1:]
    if not toks:
        return None
    words = [t.lower_ for t in toks[:-1]] + [toks[-1].lemma_.lower()]
    if any(not w.replace("-", "").replace("/", "").isalnum() for w in words):
        return None
    if words[-1] in _BAD_HEAD and len(words) == 1:
        return None
    return " ".join(words)


def _terms_in_chunk(chunk) -> list[str]:
    """Every technology name inside one noun chunk.

    Usually one, but patent prose leans on "X comprising Y" and "X having Z",
    and spaCy returns the whole span as a single chunk. Splitting at the
    participle yields both names instead of one unusable six-word string.
    """
    toks = [t for t in chunk
            if not t.is_punct and not t.is_space and t.lower_ not in _DET]
    out, run = [], []
    for tok in toks:
        if _can_continue(tok):
            run.append(tok)
            continue
        if run:
            out.append(run)
        run = []
    if run:
        out.append(run)
    return [term for term in map(_finish, out) if term]


def _batches(docs: Sequence[str], max_chars: int) -> Iterable[list[str]]:
    """Group documents so no batch holds more than max_chars of text.

    Batching by document count is the trap: the parser allocates per character,
    so 48 short abstracts are free while 48 long ones ask for a gigabyte at once
    and raise MemoryError. Bounding the batch by size instead makes cost depend
    on the corpus rather than on how the documents happen to be sized.
    """
    batch: list[str] = []
    size = 0
    for text in docs:
        if batch and size + len(text) > max_chars:
            yield batch
            batch, size = [], 0
        batch.append(text)
        size += len(text)
    if batch:
        yield batch


BATCH_CHARS = 100_000


def candidates(corpus: Corpus, nlp, cfg: Config, progress: Progress = _noop,
               notes: list[str] | None = None) -> tuple[list[list[str]], Counter]:
    """Normalised noun-chunk candidates per document, plus their document counts."""
    per_doc: list[list[str]] = []
    total = len(corpus)

    # A document longer than the parser's own ceiling raises rather than
    # truncating. Abstracts never approach it; a description or claims column
    # mapped here by mistake does, and dying is a worse answer than trimming.
    limit = int(getattr(nlp, "max_length", 1_000_000))
    texts = corpus.docs
    over = [i for i, t in enumerate(texts) if len(t) > limit]
    if over:
        texts = [t[:limit] if len(t) > limit else t for t in texts]
        if notes is not None:
            notes.append(
                f"{len(over):,} document(s) exceeded the parser limit of "
                f"{limit:,} characters and were truncated — check that the "
                "abstract column is not pointing at full text")

    stream = (doc for batch in _batches(texts, min(BATCH_CHARS, limit))
              for doc in nlp.pipe(batch, batch_size=len(batch)))
    for n, doc in enumerate(stream, start=1):
        seen: dict[str, None] = {}
        for chunk in doc.noun_chunks:
            for term in _terms_in_chunk(chunk):
                if cfg.min_term_words <= len(term.split()) <= cfg.max_term_words:
                    seen.setdefault(term, None)
        per_doc.append(list(seen))
        if n % 20 == 0 or n == total:
            progress("terms", n, total)

    df = Counter()
    for terms in per_doc:
        df.update(terms)          # document frequency, not raw frequency
    return per_doc, df


def c_values(df: Counter, cfg: Config, n_docs: int = 0) -> dict[str, float]:
    """C-value termhood (Frantzi, Ananiadou & Mima 2000).

        C(a) = log2|a| · f(a)                                  a nested in nothing
        C(a) = log2|a| · ( f(a) − mean f(b) for b properly containing a )

    Frequency alone always prefers the shorter term, so "storage tank" outranks
    "cryogenic storage tank" even when it only ever occurs inside it. Subtracting
    the containing terms' frequency corrects exactly that, and log2|a| rewards
    the more specific phrase.
    """
    # A phrase present in nearly every document separates nothing - it is the
    # corpus boilerplate ("hydrogen service", "the present invention"). This is
    # the same idea as the document-frequency ceiling in a TF-IDF vectoriser, and
    # C-value alone will not catch it because C-value rewards frequency.
    # The ceiling must never be lower than the floor, or the two filters cross
    # and no term can satisfy both - which is what made every corpus of three
    # documents or fewer return nothing at all.
    ceiling = max(int(cfg.max_doc_ratio * n_docs), cfg.min_term_freq) if n_docs else None
    terms = [t for t, c in df.items()
             if c >= cfg.min_term_freq and (ceiling is None or c <= ceiling)]
    if not terms and ceiling is not None:
        # Every candidate sits above the ceiling, so the corpus is homogeneous
        # rather than full of boilerplate. Reporting the terms it does share
        # beats reporting nothing; the caller says so in the UI.
        terms = [t for t, c in df.items() if c >= cfg.min_term_freq]
    by_len: dict[int, list[str]] = defaultdict(list)
    for t in terms:
        by_len[len(t.split())].append(t)

    # longer terms that properly contain each term
    containers: dict[str, list[str]] = defaultdict(list)
    longest = max(by_len) if by_len else 0
    for n in sorted(by_len):
        for shorter in by_len[n]:
            pad = f" {shorter} "
            for m in range(n + 1, longest + 1):
                for longer in by_len.get(m, ()):
                    if pad in f" {longer} ":
                        containers[shorter].append(longer)

    out: dict[str, float] = {}
    for t in terms:
        size = math.log2(len(t.split()) or 1) or 1.0    # unigrams get weight 1
        nested = containers.get(t)
        if not nested:
            out[t] = size * df[t]
        else:
            out[t] = size * (df[t] - sum(df[b] for b in nested) / len(nested))
    return {t: v for t, v in out.items() if v > 0}


# --------------------------------------------------------------------------- #
# stage 3 — canonical terms via a mutual k-NN graph
# --------------------------------------------------------------------------- #

class _Union:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def canonicalise(terms: Sequence[str], vectors: np.ndarray, df: Counter,
                 cfg: Config) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Group surface forms that name the same technology.

    Edge between i and j only when each is inside the other's k nearest
    neighbours *and* the similarity clears the floor. Mutuality is what stops a
    generic phrase becoming a hub: in high-dimensional embedding spaces a few
    points are the nearest neighbour of disproportionately many others, and a
    plain threshold lets them absorb everything.

    Groups are connected components, so the result is symmetric and independent
    of iteration order. The canonical name is the least boilerplate-laden member,
    then the one seen in most documents.
    """
    n = len(terms)
    if n == 0:
        return {}, {}

    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    sim = v @ v.T
    np.fill_diagonal(sim, -1.0)

    k = max(1, min(cfg.neighbours, n - 1))
    topk = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    neighbour_sets = [set(row.tolist()) for row in topk]

    uf = _Union(n)

    # Exact lexical merge first. "fuel cell membrane apparatus" and "fuel cell
    # membrane" are the same technology, but mutual k-NN can miss the pair when
    # each one's neighbourhood is crowded with other terms of its own shape.
    # Only merge when the stripped form was independently extracted, so this
    # never invents a term or merges two things that merely share a head.
    index = {t: i for i, t in enumerate(terms)}
    for i, term in enumerate(terms):
        words = term.split()
        while len(words) > cfg.min_term_words and words[-1] in _FILLER_HEAD:
            words = words[:-1]
            j = index.get(" ".join(words))
            if j is not None:
                uf.union(i, j)

    for i in range(n):
        for j in neighbour_sets[i]:
            if i < j and i in neighbour_sets[j] and sim[i, j] >= cfg.similarity_floor:
                uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    canonical: dict[str, str] = {}
    variants: dict[str, list[str]] = {}
    for members in groups.values():
        # Frequency alone picks the wrong label: "composite pressure vessel
        # apparatus" outnumbers "composite pressure vessel" because every title
        # carries the head noun. Rank boilerplate down first, frequency second.
        head = max(members, key=lambda i: (-_filler(terms[i]), df[terms[i]],
                                           -len(terms[i])))
        name = terms[head]
        variants[name] = sorted(terms[i] for i in members if i != head)
        for i in members:
            canonical[terms[i]] = name
    return canonical, variants


# --------------------------------------------------------------------------- #
# stage 4 — co-occurrence weighted by PPMI
# --------------------------------------------------------------------------- #

def cooccurrence(per_doc: Sequence[Sequence[str]], canonical: dict[str, str],
                 keep: set[str], cfg: Config) -> pd.DataFrame:
    """Term pairs ranked by positive pointwise mutual information.

        PMI(a,b) = log2 [ p(a,b) / (p(a)·p(b)) ]      PPMI = max(0, PMI)

    Raw co-occurrence counts rank the two most common terms top whether or not
    they are related. PPMI divides out each term's own rate, so what survives is
    association above chance.
    """
    n_docs = len(per_doc)
    doc_terms = [
        sorted({canonical.get(t, t) for t in terms} & keep) for terms in per_doc
    ]
    single = Counter()
    pair = Counter()
    for terms in doc_terms:
        single.update(terms)
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                pair[(a, b)] += 1

    rows = []
    for (a, b), joint in pair.items():
        if joint < cfg.min_edge_docs:
            continue
        pa, pb, pab = single[a] / n_docs, single[b] / n_docs, joint / n_docs
        ppmi = max(0.0, math.log2(pab / (pa * pb))) if pa and pb else 0.0
        if ppmi <= 0:
            continue
        rows.append({"source": a, "target": b, "documents": joint,
                     "ppmi": round(ppmi, 3)})

    frame = pd.DataFrame(rows, columns=["source", "target", "documents", "ppmi"])
    if frame.empty:
        return frame
    return (frame.sort_values(["ppmi", "documents"], ascending=False)
                 .head(cfg.max_edges).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# stage 5 — trends, when the file carries a date
# --------------------------------------------------------------------------- #

def periods(years: Sequence[int | None], n: int) -> list[tuple[str, int, int]]:
    """Contiguous period edges derived from the data, never hardcoded."""
    ys = sorted({y for y in years if y})
    if not ys:
        return []
    lo, hi = ys[0], ys[-1] + 1
    n = max(1, min(n, hi - lo))
    step = (hi - lo) / n
    edges = sorted({lo + int(round(i * step)) for i in range(n + 1)} | {lo, hi})
    return [(f"{a}-{b - 1}" if b - 1 > a else str(a), a, b)
            for a, b in zip(edges, edges[1:])]


def trends(per_doc: Sequence[Sequence[str]], corpus: Corpus,
           canonical: dict[str, str], keep: set[str], cfg: Config) -> pd.DataFrame:
    """Share of documents mentioning each term, per period, plus a growth score.

    Share rather than count, so a period with more patents does not look like
    growth. Growth is the least-squares slope of that share across periods,
    which uses every period instead of just comparing the ends.
    """
    buckets = periods(corpus.years, cfg.periods)
    if not buckets:
        return pd.DataFrame()

    doc_terms = [{canonical.get(t, t) for t in terms} & keep for terms in per_doc]

    # Tally each period once. Scanning the corpus per term per period instead
    # costs terms x periods x documents, which is where a 5,000-row file stops
    # being interactive.
    sizes: list[int] = []
    tallies: list[Counter] = []
    for _label, lo, hi in buckets:
        idx = [i for i, y in enumerate(corpus.years) if y and lo <= y < hi]
        tally = Counter()
        for i in idx:
            tally.update(doc_terms[i])
        sizes.append(len(idx))
        tallies.append(tally)

    rows = []
    for term in sorted(keep):
        counts = [tally[term] for tally in tallies]
        shares = [c / n if n else 0.0 for c, n in zip(counts, sizes)]
        x = np.arange(len(shares), dtype=float)
        slope = float(np.polyfit(x, shares, 1)[0]) if len(shares) > 1 else 0.0
        rows.append({"term": term, "documents": int(sum(counts)),
                     "growth": round(slope, 5),
                     "recent share": round(shares[-1], 3),
                     **{lab: c for (lab, _l, _h), c in zip(buckets, counts)}})
    return (pd.DataFrame(rows).sort_values("documents", ascending=False)
                              .reset_index(drop=True))


def latin_share(texts: Sequence[str], sample: int = 300) -> float:
    """Fraction of letters that are ASCII, over a sample of the corpus."""
    letters = ascii_letters = 0
    for text in texts[:sample]:
        for ch in text:
            if ch.isalpha():
                letters += 1
                ascii_letters += ch.isascii()
    return ascii_letters / letters if letters else 1.0


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class Result:
    corpus: Corpus
    terms: pd.DataFrame                    # term, c_value, documents, variants
    edges: pd.DataFrame                    # source, target, documents, ppmi
    trends: pd.DataFrame                   # empty when the file has no dates
    canonical: dict[str, str] = field(repr=False, default_factory=dict)
    variants: dict[str, list[str]] = field(repr=False, default_factory=dict)
    per_doc: list[list[str]] = field(repr=False, default_factory=list)
    doc_canon: list[set[str]] = field(repr=False, default_factory=list)
    notes: list[str] = field(default_factory=list)   # why a result looks thin

    def documents_for(self, term: str, limit: int = 20) -> pd.DataFrame:
        hits = [i for i, names in enumerate(self.doc_canon) if term in names]
        return pd.DataFrame({
            "title": [self.corpus.titles[i] for i in hits[:limit]],
            "year": [self.corpus.years[i] for i in hits[:limit]],
            "abstract": [self.corpus.abstracts[i][:240]
                         + ("…" if len(self.corpus.abstracts[i]) > 240 else "")
                         for i in hits[:limit]],
        })


def run(path_or_buf, cfg: Config | None = None, *,
        frame: pd.DataFrame | None = None, nlp=None, embedder=None,
        progress: Progress = _noop) -> Result:
    """Whole pipeline. Models are injected so the stages stay testable."""
    cfg = cfg or Config()

    progress("loading", 0, 1)
    corpus = load(path_or_buf, cfg, frame=frame)
    progress("loading", 1, 1)

    notes: list[str] = []

    if nlp is None:
        import spacy
        # noun_chunks needs the parser and the lemmatiser needs the tagger, but
        # nothing here reads entities - disabling NER removes a third of the work.
        try:
            nlp = spacy.load(cfg.spacy_model, disable=["ner"])
        except OSError as exc:
            raise InputError([
                f"could not load the spaCy model {cfg.spacy_model!r}: {exc}. "
                "Install it with: python -m spacy download en_core_web_sm"]) from exc
        nlp.max_length = max(nlp.max_length, 2_000_000)

    # noun_chunks needs a dependency parse; without one spaCy raises E029 deep
    # inside the loop, long after the work has started.
    if not nlp.has_pipe("parser"):
        raise InputError([
            f"the spaCy model {cfg.spacy_model!r} has no dependency parser, which "
            "noun-phrase extraction requires. Use a full model such as "
            "en_core_web_sm rather than a blank or sentence-only pipeline"])

    # The bundled models are English. Other scripts parse to nothing, and an
    # empty report looks identical to a corpus with no technology in it.
    share = latin_share(corpus.docs)
    if share < 0.5:
        notes.append(
            f"only {share:.0%} of the letters are Latin, so this text is probably "
            "not English — the bundled model is English-only and will find little")

    per_doc, df = candidates(corpus, nlp, cfg, progress, notes)
    scores = c_values(df, cfg, n_docs=len(corpus))
    ranked = sorted(scores, key=lambda t: (-scores[t], t))[:cfg.max_terms]

    ceiling = max(int(cfg.max_doc_ratio * len(corpus)), cfg.min_term_freq)
    if ranked and all(df[t] > ceiling for t in ranked):
        notes.append(
            "every term appears in more than "
            f"{cfg.max_doc_ratio:.0%} of the documents, so the boilerplate filter "
            "was skipped — this corpus is narrow enough that its shared "
            "vocabulary is the subject rather than noise")
    if df and not ranked:
        notes.append(
            f"candidates were found but none appears in at least "
            f"{cfg.min_term_freq} documents — lower the minimum, or use more rows")

    progress("grouping", 0, 1)
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(cfg.embed_model)
    vectors = np.asarray(embedder.encode(ranked, show_progress_bar=False)) \
        if ranked else np.zeros((0, 1))
    canonical, variants = canonicalise(ranked, vectors, df, cfg)
    progress("grouping", 1, 1)

    keep = set(variants)                                # canonical names only

    # Map every document to its canonical terms once and count from that. Doing
    # it inside the per-term comprehension rebuilt one set per term per document.
    doc_canon = [{canonical.get(x, x) for x in d} & keep for d in per_doc]
    doc_freq = Counter()
    for names in doc_canon:
        doc_freq.update(names)

    terms = pd.DataFrame({
        "term": sorted(keep, key=lambda t: -scores.get(t, 0)),
    })
    terms["c_value"] = [round(scores.get(t, 0), 2) for t in terms["term"]]
    terms["documents"] = [doc_freq[t] for t in terms["term"]]
    terms["variants"] = [", ".join(variants.get(t, [])) for t in terms["term"]]

    progress("relations", 0, 1)
    edges = cooccurrence(per_doc, canonical, keep, cfg)
    progress("relations", 1, 1)

    trend_frame = trends(per_doc, corpus, canonical, keep, cfg) \
        if corpus.has_dates else pd.DataFrame()

    return Result(corpus=corpus, terms=terms, edges=edges, trends=trend_frame,
                  canonical=canonical, variants=variants, per_doc=per_doc,
                  doc_canon=doc_canon, notes=notes)
