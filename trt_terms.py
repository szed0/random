"""Technical terms from a patent export: extracted, classified, traced and read.

    state = extract("export.xlsx")      1  every candidate term, rule-cleaned
    classify(state, '''
    Energy storage > Battery cells: secondary battery; battery module
    ''')                                2  topic and subtopic, decided by the
                                           reader -> technical_terms.csv
    state = load()                      any later turn: the CSV plus the export
    secondary(state, 12)                3  the terms sharing a patent with row 12
                                           -> secondary_<term>.csv
    read(state, 12)                     4  the patents themselves, to be read
    trace(state, "separator", within=12)   the sentence behind any count

Code proposes, the reader classifies
------------------------------------
Rules are good at "this is not a noun phrase" and bad at "this is not a
technology". So `extract` removes everything grammar can settle and prints
what is left; the reader - Gemini, or a person - sorts the technical terms
into topic > subtopic and `classify` writes them to the CSV. A term the reader
leaves out is rejected - it stays in the same CSV, after the technical terms,
unnumbered and marked `rejected` - and one the corpus does not contain cannot
get in: `classify` only accepts names that `extract` proposed.

What a candidate is
-------------------
Two to four words from the title and abstract, cut at every function word,
patent-drafting verb and punctuation mark, and kept only when:

  * **it ends on a noun this corpus uses as a noun.** No list says which words
    are nouns. A word qualifies when the corpus shows it heading a noun phrase
    ("the SEPARATOR", "a porous SEPARATOR,") or attests its plural. A word only
    ever seen after "is" ("is ACTIVE") is an adjective. This is what removes
    "electrode active" and "battery pack include".
  * **it does not end in -ing**, unless the corpus attests the -ings plural,
    which proves a noun ("coatings", "housings"). "reducing leakage" goes;
    "coating" stays when the corpus says it is a thing.
  * **it does not end in a participle or a verb** this corpus conjugates.
  * **it stands on its own at least once.** A phrase that is always the front
    or the back of the same longer phrase is a fragment of it: "electrode
    active" only ever appears in "electrode active material", so it goes and
    the longer term stays. This is why terms seen only once can be kept
    without letting every fragment of them in.

Plurals are folded, not stemmed: "batteries" becomes "battery" when the corpus
contains "battery". Porter stemming was tried and dropped - it merged
"phosphoric acid" with "phosphorous acid" and "unit cell" with "unitized
cell". Hyphens and spaces are equivalent, so "non-aqueous electrolyte" and
"nonaqueous electrolyte" are one row.

Nothing about any subject is hardcoded. The fixed lists are English function
words and the vocabulary of patent drafting ("comprising", "plurality",
"embodiment"), identical for batteries and for crop rotation.

Counting rules
--------------
A term occurs in a patent when it was extracted from that patent's title or
abstract. Counts are patents, not mentions. Two terms co-occur when both occur
in the same patent. Every count comes with the publication numbers behind it.

Runs on the standard library plus pandas (openpyxl for .xlsx).
"""

from __future__ import annotations

import glob
import math
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

# Patent text carries characters a Windows console cannot print ("μm");
# replace them rather than stop mid-report.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

# --------------------------------------------------------------------------- #
# word classes - English grammar and patent drafting, never subject matter
# --------------------------------------------------------------------------- #

FUNCTION = frozenset("""
a an the this that these those his her its their our your my some any each
every all both either neither no none one another other same such
and or but nor so yet for if then than because while although though whereas
unless until when where whether after before since during as
also too just only very much many few several more most less least
again once still ever never always often alone together multiple
of in on at by to from with without within into onto upon through across
against among between over under above below about around near per via
along beside besides beyond toward towards throughout inside outside like
is are was were be been being am do does did doing done have has had having
can could may might must shall should will would
not it they them he she we us you i there here what which who whom whose how
said wherein whereby thereof therein thereto thereby therewith thereon
hereof herein hereinafter hereby accordingly according respectively
thus hence therefore however moreover further furthermore etc
two three four five six seven eight nine ten
first second third fourth fifth sixth seventh eighth ninth tenth
preferably optionally substantially approximately generally typically
particularly specifically especially mainly mostly partially partly
relatively simultaneously alternatively additionally
best better worse worst higher highest lower lowest larger largest smaller
smallest greater greatest bigger biggest
""".split())

# "therebetween", "herein", "whereupon": always pointers, never names.
_POINTER = re.compile(r"^(?:there|here|where)(?:of|in|into|to|by|with|on|onto|at|"
                      r"from|between|through|under|over|after|for|upon|within|"
                      r"along|about|around|across|beneath|below|above|unto)$")

# The register of patent drafting: verbs that link claim elements, and the
# modifiers claims use to point back at themselves. A run of words is cut at
# each one, so "a layer disposed on the plate" never yields "layer disposed".
DRAFTING = frozenset("""
comprise comprises comprised comprising include includes included including
consist consists consisted consisting contain contains contained containing
provide provides provided providing disclose discloses disclosed disclosing
relate relates related relating describe describes described describing
use uses used using utilize utilizes utilized utilizing employ employs
employed employing exhibit exhibits exhibited exhibiting
achieve achieves achieved achieving allow allows allowed enable enables
enabled configured arranged disposed located positioned situated mounted
attached connected coupled adapted designed formed obtained produced prepared
selected determined defined
present plurality predetermined prescribed respective certain given particular
corresponding following preceding foregoing aforementioned aforesaid
various whole entire desired suitable conventional novel improved new
exemplary preferred typical additional known prior subsequent previous
appropriate capable possible necessary sufficient useful operable applicable
able similar different
""".split())

# Nouns that name the document rather than anything in it.
DOCUMENT = frozenset("""
invention inventions disclosure disclosures embodiment embodiments example
examples claim claims figure figures fig drawing drawings patent patents
""".split())

# Category nouns of claim drafting. A phrase made only of these ("end
# portion", "device body") names a slot in a claim, not a technology.
GENERIC = frozenset("""
apparatus method system device unit assembly arrangement means mechanism
structure module equipment machine process technique procedure step portion
member element part side end section component type kind form manner way case
body region area
""".split())

DETERMINER = frozenset("""
the a an said this that these those its their each every any some such no
one both all another other
""".split())

BE = frozenset("is are was were be been being am".split())

# Frames that only a verb follows: the infinitive marker and the modals.
_VERB_FRAME = frozenset("to can could may might must shall should will would".split())

# Prepositions that take a gerund with its object: "for MEASURING blood
# pressure", "by COATING the electrode". An -ing word right after one is a
# verb. Conjunctions ("when", "while") are left out: a clause can open on a
# noun phrase, "when OPERATING temperature is reached".
_GERUND_FRAME = frozenset("""
for by of in without upon through via from during into about
""".split())

_JP = re.compile(
    r"\b(?:PROBLEM TO BE SOLVED|SOLUTION|SELECTED DRAWING|ADVANTAGE|EFFECT)\s*:\s*",
    re.IGNORECASE)
_REFNUM = re.compile(r"\(\s*[0-9][0-9a-z]*\s*(?:[,-]\s*[0-9][0-9a-z]*\s*)*\)",
                     re.IGNORECASE)
_WS = re.compile(r"\s+")
_SEGMENT = re.compile(r"[.!?;:,()\[\]{}\"/‘’“”]+(?:\s|$)|[;:()\[\]{}\"/]")
_WORD = re.compile(r"[a-z][a-z0-9]*(?:['\-][a-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?;])\s+")

HINTS = {
    "title": ("title",),
    "abstract": ("abstract", "summary"),
    "claims": ("claims", "claim"),
    "description": ("description", "detail"),
    "pubno": ("publication number", "publication_number", "publication numbers",
              "patent number", "patent_number", "pub no", "pubno", "patent id",
              "publication", "number", "id"),
    "date": ("application date", "filing date", "filing_date", "priority date",
             "application_date", "publication date", "date"),
    "cpc": ("technology domain", "cpc", "ipc", "classification", "domain"),
}

TERMS_CSV = "technical_terms.csv"


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def _pick(cols, hints, explicit=None):
    """Exact name first, then prefix, then substring - so "claims" beats
    "n_claims_raw" and "publication_number" beats "examiner_id"."""
    if explicit:
        if explicit not in cols:
            raise SystemExit("column %r not in the file. Columns: %s"
                             % (explicit, ", ".join(map(str, cols))))
        return explicit
    low = [(str(c).strip().lower(), c) for c in cols]
    for test in (lambda lc, h: lc == h,
                 lambda lc, h: lc.startswith(h),
                 lambda lc, h: h in lc):
        for h in hints:
            for lc, c in low:
                if test(lc, h):
                    return c
    return None


def clean(text):
    if not isinstance(text, str):
        return ""
    out = re.sub(r"^\s*\([^)]*\)\s*\n", "", text)
    out = _JP.sub("", out)
    out = _REFNUM.sub(" ", out)
    out = out.replace("–", "-").replace("—", "-").replace("‑", "-")
    return _WS.sub(" ", out).strip()


def _first_id(value):
    parts = [p for p in re.split(r"[\s;,|]+", str(value)) if p]
    return parts[0] if parts else ""


def _year(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    s = str(value).strip().split("\n")[0]
    m = re.match(r"^((?:19|20)\d{2})", s)
    if m:
        return m.group(1)
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", s)
    return m.group(1) if m else ""


def _read(path, fields=("title", "abstract"), **explicit):
    """The export as one record per patent. Stops if nothing can be traced."""
    low = str(path).lower()
    if low.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            frame = pd.read_excel(path)
        except ImportError as exc:
            raise SystemExit("reading Excel needs openpyxl, which this Python "
                             "lacks (%s). Save the export as CSV." % exc)
    else:
        frame = pd.read_csv(path, sep=None, engine="python")
    cols = list(frame.columns)
    col = {k: _pick(cols, HINTS[k], explicit.get(k)) for k in HINTS}
    if col["abstract"] is None and col["title"] is None:
        raise SystemExit("no abstract or title column. Columns: %s"
                         % ", ".join(map(str, cols)))
    if col["pubno"] is None:
        raise SystemExit("no publication-number column, so nothing can be "
                         "traced. Columns: %s" % ", ".join(map(str, cols)))

    docs, seen = [], set()
    for _, r in frame.iterrows():
        pub = _first_id(r[col["pubno"]]) if pd.notna(r[col["pubno"]]) else ""
        if not pub or pub in seen:
            continue
        seen.add(pub)
        get = (lambda k: clean(str(r[col[k]]))
               if col[k] is not None and pd.notna(r[col[k]]) else "")
        d = {"pub": pub, "title": get("title"), "abstract": get("abstract"),
             "claims": get("claims"), "description": get("description"),
             "cpc": get("cpc"),
             "year": _year(r[col["date"]]) if col["date"] is not None else ""}
        d["text"] = " . ".join(d[f] for f in fields if d.get(f))
        docs.append(d)
    return {"docs": docs, "columns": col, "rows_in_file": len(frame),
            "fields": tuple(fields)}


# --------------------------------------------------------------------------- #
# the corpus's own grammar
# --------------------------------------------------------------------------- #

def _tokens(segment):
    """Word tokens, with None wherever something that is not a word sat - a
    number, a formula, a symbol - so words either side of it never join."""
    out = []
    for raw in segment.split():
        tok = raw.strip("'\"-.,")
        out.append(tok if _WORD.fullmatch(tok) else None)
    return out


def _segments(text):
    return [s for s in _SEGMENT.split(text.lower()) if s.strip()]


def _folder(vocab):
    """Plural to singular, decided by the corpus.

    "batteries" folds to "battery" only because "battery" is in the corpus.
    "gas", "lens", "species" and "analysis" have no shorter attested form and
    are left alone, which a suffix rule gets wrong. A plural whose singular
    never occurs is folded by rule, so a row is never labelled with an -s head.
    """
    cache = {}

    def fold(w):
        if w in cache:
            return cache[w]
        out = w
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            tries = []
            if w.endswith("ies"):
                tries.append(w[:-3] + "y")
            if w.endswith("es"):
                tries.append(w[:-2])
            tries.append(w[:-1])
            hit = [t for t in tries if t in vocab]
            if hit:
                out = hit[0]
            elif not w.endswith(("as", "os", "ics", "ous", "series", "species")):
                out = tries[0] if w.endswith("ies") and len(w) > 4 else w[:-1]
        cache[w] = out
        return out

    return fold


def _merge_key(words):
    """Hyphen- and space-blind identity: "non-aqueous electrolyte" and
    "nonaqueous electrolyte" are one row."""
    return "".join(words).replace("-", "")


class _Grammar:
    """Which words this corpus uses as verbs, adverbs and noun heads."""

    def __init__(self, texts):
        vocab = Counter()
        seg_tokens = []
        self.det_before, self.det_any = Counter(), Counter()
        self.after_frame, self.ing_object = Counter(), Counter()
        for t in texts:
            for seg in _segments(t):
                toks = _tokens(seg)
                seg_tokens.append(toks)
                vocab.update(w for w in toks if w)
                # "the ASSEMBLY of", "a SUPPLY." - a determiner, the word, then
                # nothing more of the phrase. "the ELECTRICALLY conductive"
                # does not count.
                for i in range(1, len(toks)):
                    after = toks[i + 1] if i + 1 < len(toks) else None
                    if toks[i - 1] in DETERMINER and toks[i]:
                        self.det_any[toks[i]] += 1
                    if toks[i - 1] in DETERMINER and toks[i] and \
                            (after is None or after in FUNCTION or after in DRAFTING):
                        self.det_before[toks[i]] += 1
                    # "to PREVENT", "can ADJUST": the infinitive and modal frames
                    if toks[i - 1] in _VERB_FRAME and toks[i]:
                        self.after_frame[toks[i]] += 1
                    # "MEASURING the pressure": an -ing word taking an object
                    if toks[i] in DETERMINER and toks[i - 1] and \
                            toks[i - 1].endswith("ing"):
                        self.ing_object[toks[i - 1]] += 1
        self.vocab = vocab
        self.fold = _folder(vocab)
        self.verbs = self._verbs(vocab)
        self.verb_s = {v + "s" for v in self.verbs} | {v + "es" for v in self.verbs}
        # "ENSURES", "RAISES", "LOWERS": the -s form of a word that only ever
        # follows "to" or a modal, or of a function word, is a finite verb.
        for w in vocab:
            if not w.endswith("s") or len(w) <= 3 or self.det_any[w]:
                continue
            bases = [w[:-1]]
            if w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh", "o")):
                bases.append(w[:-2])
            if any(b in FUNCTION or (self.after_frame[b] and not self.det_any[b])
                   for b in bases):
                self.verb_s.add(w)

        # Head evidence, read off runs cut at function words: "the SEPARATOR"
        # (solo), "the porous SEPARATOR" (determined), "porous SEPARATOR"
        # (compound), "is POROUS" (predicative - an adjective).
        self.solo, self.determined = Counter(), Counter()
        self.compound, self.predicative = Counter(), Counter()
        self.det_start = Counter()
        for toks in seg_tokens:
            for before, run in self._raw_runs(toks):
                head = run[-1]
                if before in BE:
                    self.predicative[head] += 1
                elif before in DETERMINER:
                    (self.solo if len(run) == 1 else self.determined)[head] += 1
                    self.det_start[run[0]] += 1
                elif len(run) > 1:
                    self.compound[head] += 1
        self._head, self._start = {}, {}

    @staticmethod
    def _verbs(vocab):
        """A word is a verb here when its -ing and -ed forms both occur and the
        bare form is outnumbered by its inflections - which keeps "stack" (a
        noun that is sometimes stacked) and drops "include"."""
        out = set()
        for w, bare in vocab.items():
            if len(w) < 3:
                continue
            ing = [w + "ing", w + w[-1] + "ing"] + ([w[:-1] + "ing"] if w.endswith("e") else [])
            ed = [w + "ed", w + w[-1] + "ed"] + ([w + "d"] if w.endswith("e") else [])
            n_ing = sum(vocab.get(f, 0) for f in ing)
            n_ed = sum(vocab.get(f, 0) for f in ed)
            if not (n_ing and n_ed):
                continue
            n_s = vocab.get(w + "s", 0) + vocab.get(w + "es", 0)
            if bare < n_ing + n_ed + n_s:
                out.add(w)
        return out

    def _adverb(self, w):
        """An -ly word is an adverb unless the corpus treats it as a noun:
        "the ASSEMBLY", "a SUPPLY", "assemblies"."""
        if not w.endswith("ly") or len(w) < 5:
            return False
        return not (self.det_before[w] or (w[:-1] + "ies") in self.vocab)

    def breaks(self, w):
        return (w is None or len(w) < 2 or w in FUNCTION or w in DRAFTING
                or w in DOCUMENT or w in self.verb_s or bool(_POINTER.match(w))
                or self._adverb(w))

    def _raw_runs(self, toks):
        run, before = [], None
        for i, w in enumerate(toks + [None]):
            if not self.breaks(w):
                if not run:
                    before = toks[i - 1] if i else None
                run.append(w)
                continue
            if run:
                yield before, run
            run = []

    def runs(self, text):
        """(word before the run, the run) for every run in the text."""
        for seg in _segments(text):
            for before, run in self._raw_runs(_tokens(seg)):
                yield before, run

    def _participle(self, w):
        """-ed words that are verb forms. From six letters on the answer is
        yes unless the corpus puts a determiner straight before it; below
        that, "bed", "feed" and "speed" are the usual case, so the base form
        has to be attested ("coated" <- "coat", "coating")."""
        if not w.endswith("ed") or len(w) < 5:
            return False
        if len(w) >= 6:
            return not self.solo[w]
        v = self.vocab
        bases = [w[:-2], w[:-1]] + ([w[:-3]] if w[-3] == w[-4] else [])
        return any(b in v or (b + "ing") in v or (b[:-1] + "ing") in v for b in bases)

    def _bare_verb(self, w):
        """Seen after "to" or a modal, and never after a determiner: "to
        PREVENT", "can ADJUST". "the SUPPORT" rescues "support"."""
        return bool(self.after_frame[w]) and not (
            self.det_before[w] or self.det_start[w] or self.determined[w])

    def head(self, w):
        """Can this word end a technical term?"""
        if w in self._head:
            return self._head[w]
        f = self.fold(w)
        plural = f != w or (w + "s") in self.vocab or (w + "es") in self.vocab
        if len(w) < 3:
            ok = False
        elif w.endswith("ing"):
            ok = (w + "s") in self.vocab
        elif self._participle(w) or w in self.verbs or w in self.verb_s \
                or self._bare_verb(w):
            ok = False
        elif self.predicative[w] and not (plural or self.solo[w]):
            ok = False
        else:
            ok = bool(plural or self.solo[w] or self.determined[w] or self.compound[w])
        self._head[w] = ok
        return ok

    def start(self, w, before=""):
        """Can this word begin one? A bare verb cannot ("store energy").

        An -ing word is decided by what precedes this occurrence: after a
        determiner it modifies ("a CHARGING device"), after a preposition it
        takes an object ("for GENERATING electricity"). Anywhere else - a
        title's first word, or inside a longer run - the corpus decides: it is
        a modifier when it follows a determiner more often than it is
        followed by one ("the CUTTING blade" against "MEASURING the
        pressure")."""
        if w.endswith("ing") and before != "":
            if before in DETERMINER:
                return True
            if before in _GERUND_FRAME:
                return False
        if w in self._start:
            return self._start[w]
        if w in self.verbs or self._bare_verb(w):
            ok = False
        elif w.endswith("ing"):
            ok = (w + "s") in self.vocab or self.det_start[w] > self.ing_object[w]
        else:
            ok = True
        self._start[w] = ok
        return ok


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def _cvalue(freq, words_of):
    """Frantzi and Ananiadou's C-value: frequency a phrase does not owe to the
    longer phrases containing it."""
    index = {tuple(ws): k for k, ws in words_of.items()}
    hosts = defaultdict(list)
    for k, ws in words_of.items():
        n = len(ws)
        for size in range(1, n):
            for i in range(n - size + 1):
                sub = index.get(tuple(ws[i:i + size]))
                if sub is not None and sub != k:
                    hosts[sub].append(k)
    out = {}
    for k, f in freq.items():
        weight = math.log(len(words_of[k]) + 1, 2)
        h = hosts.get(k)
        out[k] = weight * (f - sum(freq[x] for x in h) / len(h)) if h else weight * f
    return out


def _cohesion(words_of, unigram, total):
    """Mean self-information of the phrase's words. Rare words bind tighter."""
    out = {}
    for k, ws in words_of.items():
        bits = [-math.log(unigram[w] / total) for w in ws if unigram.get(w)]
        out[k] = sum(bits) / len(bits) if bits else 0.0
    return out


def _plural_head(phrase, fold):
    last = phrase.split()[-1]
    return fold(last) != last


def _display(surfaces, fold):
    """The commonest surface form whose head is singular."""
    for form, _ in surfaces.most_common():
        if not _plural_head(form, fold):
            return form
    words = surfaces.most_common(1)[0][0].split()
    return " ".join(words[:-1] + [fold(words[-1])])


# --------------------------------------------------------------------------- #
# step 1 - extract
# --------------------------------------------------------------------------- #

def extract(path, min_words=2, max_words=4, min_patents=1, fragment=0.8,
            fields=("title", "abstract"), show=400, **columns):
    """Read the export and propose every candidate term.

    `min_patents=1` keeps terms seen in a single patent. `fields` is where
    terms are looked for; claims and description are read by `read` but not
    mined, because claim language multiplies drafting noise and a long
    description makes every term co-occur with every other. `fragment` is the
    share of a phrase's occurrences that must continue into the same longer
    phrase, on a side where it never stands alone, before it is dropped as a
    piece of that phrase.
    """
    corpus = _read(path, fields=fields, **columns)
    docs = corpus["docs"]
    g = _Grammar([d["text"] for d in docs])
    fold = g.fold

    occ = defaultdict(lambda: {"mentions": 0, "docs": set(), "surface": Counter(),
                               "free_l": 0, "free_r": 0, "ext_l": Counter(),
                               "ext_r": Counter()})
    unigram, total = Counter(), 0
    for di, d in enumerate(docs):
        for before, run in g.runs(d["text"]):
            folded = [fold(w) for w in run]
            unigram.update(folded)
            total += len(run)
            L = len(run)
            heads = [g.head(w) for w in run]
            starts = [g.start(run[0], before or "")] + [g.start(w) for w in run[1:]]
            # A run longer than max_words yields nothing of its own: cutting it
            # to a front or back window was measured, and each window added
            # more debris ("high capacity prismatic") than terms. Its pieces
            # still count wherever they also stand alone.
            for j in range(L):
                if not heads[j]:
                    continue
                for i in range(max(0, j - max_words + 1), j - min_words + 2):
                    if not starts[i]:
                        continue
                    ws = folded[i:j + 1]
                    if all(w in GENERIC for w in ws):
                        continue
                    o = occ[" ".join(ws)]
                    o["mentions"] += 1
                    o["docs"].add(di)
                    o["surface"][" ".join(run[i:j + 1])] += 1
                    if i == 0 or not any(starts[:i]):
                        o["free_l"] += 1
                    else:
                        o["ext_l"][folded[i - 1]] += 1
                    if j == L - 1 or not any(heads[j + 1:]):
                        o["free_r"] += 1
                    else:
                        o["ext_r"][folded[j + 1]] += 1

    # hyphen/space variants are one row
    groups = {}
    for k, o in occ.items():
        mk = _merge_key(k.split())
        grp = groups.get(mk)
        if grp is None:
            groups[mk] = {"mentions": o["mentions"], "docs": set(o["docs"]),
                          "surface": Counter(o["surface"]), "free_l": o["free_l"],
                          "free_r": o["free_r"], "ext_l": Counter(o["ext_l"]),
                          "ext_r": Counter(o["ext_r"])}
            continue
        grp["mentions"] += o["mentions"]
        grp["docs"] |= o["docs"]
        grp["surface"].update(o["surface"])
        grp["free_l"] += o["free_l"]
        grp["free_r"] += o["free_r"]
        grp["ext_l"].update(o["ext_l"])
        grp["ext_r"].update(o["ext_r"])

    n_found = len(groups)
    dropped = Counter()
    keep = {}
    for mk, grp in groups.items():
        if len(grp["docs"]) < min_patents:
            dropped["in fewer than %d patents" % min_patents] += 1
            continue
        n = grp["mentions"]
        if any(not free and ext and max(ext.values()) / n >= fragment
               for free, ext in ((grp["free_l"], grp["ext_l"]),
                                 (grp["free_r"], grp["ext_r"]))):
            dropped["fragments of a longer term"] += 1
            continue
        keep[mk] = grp

    n_docs = len(docs)
    words_of = {}
    for mk, grp in keep.items():
        grp["display"] = _display(grp["surface"], fold)
        words_of[mk] = [fold(w) for w in grp["display"].split()]
    freq = {mk: grp["mentions"] for mk, grp in keep.items()}
    cval = _cvalue(freq, words_of)
    coh = _cohesion(words_of, unigram, total)
    for mk, grp in keep.items():
        df = len(grp["docs"])
        grp["df"] = df
        grp["tfidf"] = grp["mentions"] * math.log(n_docs / df) if df < n_docs else 0.0
        grp["score"] = max(cval[mk], 0.0) * (1.0 + coh[mk])
        grp["patents"] = [docs[i]["pub"] for i in sorted(grp["docs"])]
        grp["variants"] = [f for f, _ in grp["surface"].most_common()]

    order = sorted(keep, key=lambda mk: (-keep[mk]["df"], -keep[mk]["score"],
                                         keep[mk]["display"]))
    state = {"docs": docs, "n_docs": n_docs, "fold": fold, "groups": keep,
             "order": order, "rank": {mk: i for i, mk in enumerate(order, 1)},
             "classes": {}, "topics": [], "shown": 0,
             "columns": corpus["columns"], "fields": corpus["fields"]}

    c = corpus["columns"]
    print("CORPUS     %d rows, %d patents with a publication number"
          % (corpus["rows_in_file"], n_docs))
    print("COLUMNS    id=%r title=%r abstract=%r claims=%r description=%r date=%r"
          % (c["pubno"], c["title"], c["abstract"], c["claims"],
             c["description"], c["date"]))
    print("TERMS FROM %s   (%d-%d words, in >= %d patent%s)"
          % (" + ".join(fields), min_words, max_words, min_patents,
             "" if min_patents == 1 else "s"))
    print("RULES      %d phrases found; dropped: %s"
          % (n_found, ", ".join("%d %s" % (v, k) for k, v in dropped.items()) or "none"))
    print("CANDIDATES %d" % len(order))
    print()
    candidates(state, 1, show)
    return state


def candidates(state, start=1, size=400):
    """The numbered candidate list, a page at a time."""
    order = state["order"]
    stop = min(len(order), start + size - 1)
    print("%5s %4s  %s" % ("#", "pat", "candidate"))
    for i in range(start, stop + 1):
        grp = state["groups"][order[i - 1]]
        print("%5d %4d  %s" % (i, grp["df"], grp["display"]))
    state["shown"] = max(state.get("shown", 0), stop)
    if stop < len(order):
        print("... %d more: candidates(state, %d)" % (len(order) - stop, stop + 1))
    else:
        print("(end of list)")


# --------------------------------------------------------------------------- #
# step 2 - classify
# --------------------------------------------------------------------------- #

_CLASS_LINE = re.compile(r"^\s*(?P<topic>[^>:]+?)\s*>\s*(?P<sub>[^:]+?)\s*:\s*(?P<terms>.*)$")


def _norm(term):
    t = re.sub(r"^\s*#?\d+\s*[.)|:-]?\s+", "", str(term))
    t = t.strip().strip("`*'\"").strip().lower()
    return _WS.sub(" ", t)


def _lookup(state, term):
    words = _norm(term).split()
    if not words:
        return None
    mk = _merge_key([state["fold"](w) for w in words])
    return mk if mk in state["groups"] else None


def classify(state, block):
    """Take the reader's classification and write the CSV.

    One line per subtopic, `Topic > Subtopic: term; term; term`, using names
    exactly as `extract` printed them. Every candidate left out is rejected.
    Calling again adds to what is already classified, so a long candidate list
    can be done a page at a time.
    """
    unknown, dupes, lines = [], [], 0
    for line in str(block).splitlines():
        # tolerate markdown: "- **Fuel cells > Catalysts:** ..."
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).replace("**", "")
        m = _CLASS_LINE.match(line)
        if not m:
            continue
        lines += 1
        topic, sub = m.group("topic").strip(), m.group("sub").strip()
        sep = ";" if ";" in m.group("terms") else ","
        for raw in m.group("terms").split(sep):
            if not raw.strip():
                continue
            mk = _lookup(state, raw)
            if mk is None:
                unknown.append(raw.strip())
            elif mk in state["classes"] and state["classes"][mk] != (topic, sub):
                dupes.append(state["groups"][mk]["display"])
            else:
                if (topic, sub) not in state["topics"]:
                    state["topics"].append((topic, sub))
                state["classes"][mk] = (topic, sub)
    if not lines:
        raise SystemExit("no line of the form 'Topic > Subtopic: term; term' found")

    _build_rows(state)
    rejected = _write_terms(state)

    unseen = sum(1 for _, s in rejected if s == "not reviewed")
    print("CLASSIFIED %d technical terms, %d topics, %d subtopics -> %s"
          % (len(state["rows"]), len({t for t, _ in state["topics"]}),
             len(state["topics"]), TERMS_CSV))
    print("REJECTED   %d candidates left out, listed after them in the same file "
          "with status 'rejected'%s"
          % (len(rejected),
             " (%d never shown - page on with candidates())" % unseen if unseen else ""))
    if unknown:
        print("IGNORED    %d names that were not candidates: %s"
              % (len(unknown), "; ".join(unknown[:20])
                 + (" ..." if len(unknown) > 20 else "")))
    if dupes:
        print("DUPLICATE  kept the first subtopic for: %s" % "; ".join(dupes[:20]))
    menu(state)


def _build_rows(state):
    topic_rank = {}
    for t, _ in state["topics"]:
        topic_rank.setdefault(t, len(topic_rank))
    sub_rank = {ts: i for i, ts in enumerate(state["topics"])}
    groups = state["groups"]
    mks = sorted(state["classes"], key=lambda mk: (
        topic_rank[state["classes"][mk][0]], sub_rank[state["classes"][mk]],
        -groups[mk]["df"], -groups[mk]["score"], groups[mk]["display"]))
    rows = []
    for n, mk in enumerate(mks, 1):
        grp = groups[mk]
        topic, sub = state["classes"][mk]
        rows.append({"n": n, "term": grp["display"], "topic": topic,
                     "subtopic": sub, "n_words": len(grp["display"].split()),
                     "n_patents": grp["df"], "tfidf": round(grp["tfidf"], 3),
                     "score": round(grp["score"], 1),
                     "variants": grp["variants"], "patents": grp["patents"]})
    _index(state, rows)


def _index(state, rows):
    fold = state["fold"]
    for r in rows:
        words = [fold(w) for w in r["term"].split()]
        r["mk"] = _merge_key(words)
        r["toks"] = [t for w in words for t in w.split("-") if t]
    state["rows"] = rows
    state["by_n"] = {r["n"]: r for r in rows}
    by_mk = {}
    for r in rows:
        by_mk[r["mk"]] = r
    for r in rows:
        for v in r["variants"]:
            by_mk.setdefault(_merge_key([fold(w) for w in v.split()]), r)
    state["by_mk"] = by_mk
    pat_terms = defaultdict(list)
    for r in rows:
        for p in r["patents"]:
            pat_terms[p].append(r["n"])
    state["pat_terms"] = pat_terms


def _write_terms(state):
    """One file: the technical terms, numbered and classified, then every
    candidate left out, unnumbered, with status "rejected" - or "not
    reviewed" if its page was never printed. Returns the left-out ones."""
    out = []
    for r in state["rows"]:
        out.append({"#": str(r["n"]), "term": r["term"], "status": "technical",
                    "topic": r["topic"], "subtopic": r["subtopic"],
                    "n_words": r["n_words"], "n_patents": r["n_patents"],
                    "tfidf": r["tfidf"], "score": r["score"],
                    "variants": " | ".join(r["variants"]),
                    "patents": " ".join(r["patents"])})
    rejected = []
    for mk in state["order"]:
        if mk in state["classes"]:
            continue
        grp = state["groups"][mk]
        status = "rejected" if state["rank"][mk] <= state["shown"] else "not reviewed"
        rejected.append((mk, status))
        out.append({"#": "", "term": grp["display"], "status": status,
                    "topic": "", "subtopic": "",
                    "n_words": len(grp["display"].split()), "n_patents": grp["df"],
                    "tfidf": round(grp["tfidf"], 3), "score": round(grp["score"], 1),
                    "variants": " | ".join(grp["variants"]),
                    "patents": " ".join(grp["patents"])})
    pd.DataFrame(out, columns=["#", "term", "status", "topic", "subtopic", "n_words",
                               "n_patents", "tfidf", "score", "variants", "patents"]
                 ).to_csv(TERMS_CSV, index=False)
    return rejected


def menu(state, topic=None, contains=None, per_subtopic=None):
    """The classified terms, numbered, grouped by topic > subtopic. The number
    is the `#` column of technical_terms.csv and never changes meaning.

    `per_subtopic` shows only each subtopic's first rows, most patents first.
    Past 250 terms that is the default - a full list is too long to read in
    a chat - and each subtopic says how many more it holds.
    """
    if not state.get("rows"):
        candidates(state, 1, len(state["order"]))
        return
    rows = state["rows"]
    if topic:
        rows = [r for r in rows if topic.lower() in r["topic"].lower()
                or topic.lower() in r["subtopic"].lower()]
    if contains:
        rows = [r for r in rows if contains.lower() in r["term"].lower()]
    if per_subtopic is None and len(rows) > 250 and not (topic or contains):
        per_subtopic = 8
    groups = defaultdict(list)
    for r in rows:
        groups[(r["topic"], r["subtopic"])].append(r)
    last_topic = None
    for (topic_name, sub), members in groups.items():
        if topic_name != last_topic:
            print("\n%s" % topic_name.upper())
            last_topic = topic_name
        shown = members[:per_subtopic] if per_subtopic else members
        items = ["%d %s (%d)" % (r["n"], r["term"], r["n_patents"]) for r in shown]
        if len(members) > len(shown):
            items.append("+%d more" % (len(members) - len(shown)))
        line = "  %s:" % sub
        for k, item in enumerate(items):
            sep = " " if k == 0 else "; "
            if k and len(line) + len(sep) + len(item) > 100:
                print(line + ";")
                line, sep = "    ", ""
            line += sep + item
        print(line)
    if per_subtopic and len(rows) > sum(min(len(m), per_subtopic)
                                        for m in groups.values()):
        print("\n%d terms; the first %d of each subtopic are shown. Every row is "
              "numbered in %s, and menu(state, topic=\"...\") lists one topic "
              "in full." % (len(rows), per_subtopic, TERMS_CSV))
    else:
        print("\n%d terms." % len(rows))
    print("Pick a primary by its number.")


# --------------------------------------------------------------------------- #
# later turns - rebuild from the CSV
# --------------------------------------------------------------------------- #

# Files this module writes, never to be mistaken for the export.
# rejected_terms*.csv is no longer written but may linger from an older run.
_OURS = re.compile(r"^(technical_terms|rejected_terms|secondary_)", re.IGNORECASE)


def _find(patterns, exclude_ours=False):
    hits = set()
    for p in patterns:
        hits.update(glob.glob(p, recursive=True))
    hits = [h for h in hits if os.path.isfile(h)
            and not (exclude_ours and _OURS.match(os.path.basename(h)))]
    return sorted(hits, key=lambda h: (-os.path.getmtime(h), h))


def load(export=None, terms=None, fields=("title", "abstract"), **columns):
    """Rebuild the classified state from technical_terms.csv and the export.

    The CSV is the source of truth for which terms exist and where; the
    export supplies the text that `read` and `trace` show. Either is found in
    the working directory when not named.
    """
    terms = terms or next(iter(_find(["**/technical_terms*.csv"])), None)
    if not terms:
        raise SystemExit("technical_terms.csv is not here. Attach the CSV from "
                         "step 1, or re-run extract() and classify() with the "
                         "classification written in step 1.")
    export = export or next(iter(_find(["**/*.xlsx", "**/*.xls", "**/*.csv"],
                                       exclude_ours=True)), None)
    t = pd.read_csv(terms, dtype=str, keep_default_na=False)
    # rejected candidates share the file; only numbered technical rows load
    if "status" in t.columns:
        t = t[t["status"] == "technical"]
    t = t[t["#"].str.strip() != ""]
    state = {"docs": [], "n_docs": 0, "columns": {}, "fields": tuple(fields)}
    if export:
        corpus = _read(export, fields=fields, **columns)
        state.update(docs=corpus["docs"], n_docs=len(corpus["docs"]),
                     columns=corpus["columns"])
        state["fold"] = _Grammar([d["text"] for d in corpus["docs"]]).fold
    else:
        state["fold"] = _folder(set())
    rows = []
    for _, r in t.iterrows():
        rows.append({"n": int(r["#"]), "term": r["term"], "topic": r["topic"],
                     "subtopic": r["subtopic"], "n_words": int(r["n_words"]),
                     "n_patents": int(r["n_patents"]),
                     "tfidf": float(r["tfidf"] or 0), "score": float(r["score"] or 0),
                     "variants": [v for v in r["variants"].split(" | ") if v],
                     "patents": r["patents"].split()})
    _index(state, rows)
    if not state["n_docs"]:
        state["n_docs"] = len({p for r in rows for p in r["patents"]})
    print("LOADED     %d technical terms from %s" % (len(rows), terms))
    print("EXPORT     %s" % (("%s, %d patents" % (export, state["n_docs"])) if export else
                              "none found - read() and trace() need it; lift "
                              "uses the %d patents the CSV names" % state["n_docs"]))
    return state


# --------------------------------------------------------------------------- #
# step 3 - secondary terms
# --------------------------------------------------------------------------- #

def _resolve(state, term):
    if isinstance(term, str) and term.strip().lstrip("#").isdigit():
        term = int(term.strip().lstrip("#"))
    elif not isinstance(term, str):
        term = int(term)
    if isinstance(term, int):
        if term not in state["by_n"]:
            raise SystemExit("there is no row %d; rows run 1-%d"
                             % (term, len(state["rows"])))
        return state["by_n"][term]
    words = _norm(term).split()
    mk = _merge_key([state["fold"](w) for w in words])
    if mk in state["by_mk"]:
        return state["by_mk"][mk]
    raise SystemExit("%r is not a classified technical term. menu(state, "
                     "contains=...) searches the list." % term)


def _nested(a, b):
    """Is one term a word-aligned piece of the other? "fuel cell" is inside
    "fuel cell stack"; "oxide layer" is not inside "silicon dioxide layer"."""
    if len(a["mk"]) > len(b["mk"]):
        a, b = b, a
    cuts, pos = {0}, 0
    for t in b["toks"]:
        pos += len(t)
        cuts.add(pos)
    start = b["mk"].find(a["mk"])
    while start != -1:
        if start in cuts and start + len(a["mk"]) in cuts:
            return True
        start = b["mk"].find(a["mk"], start + 1)
    return False


def _label(r):
    return "%s > %s" % (r["topic"], r["subtopic"])


def secondary(state, primary, show=15):
    """Every technical term that occurs in a patent the primary occurs in.

    `both` counts the shared patents; `lift` compares that with what the
    secondary's corpus-wide rate predicts, so a term near 1 is simply common
    everywhere. Terms that contain the primary or sit inside it ("fuel cell"
    for "fuel cell stack") share its patents by construction and are listed
    apart, never as secondaries. Rows are the CSV's `#` numbers.
    """
    p = _resolve(state, primary)
    P = set(p["patents"])
    N = max(state["n_docs"], 1)
    nested, found = [], []
    for r in state["rows"]:
        if r is p:
            continue
        both = P.intersection(r["patents"])
        if not both:
            continue
        if _nested(r, p):
            nested.append(r)
            continue
        lift = (len(both) / len(P)) / (r["n_patents"] / N)
        found.append((r, sorted(both), lift))
    found.sort(key=lambda x: (-len(x[1]), -x[2], x[0]["n"]))

    slug = re.sub(r"[^a-z0-9]+", "_", p["term"].lower()).strip("_")
    out_csv = "secondary_%s.csv" % slug
    pd.DataFrame({
        "#": [r["n"] for r, _, _ in found],
        "secondary_term": [r["term"] for r, _, _ in found],
        "topic": [r["topic"] for r, _, _ in found],
        "subtopic": [r["subtopic"] for r, _, _ in found],
        "patents_with_both": [len(b) for _, b, _ in found],
        "share_of_primary": [round(len(b) / len(P), 3) for _, b, _ in found],
        "lift": [round(l, 2) for _, _, l in found],
        "n_patents": [r["n_patents"] for r, _, _ in found],
        "patents": [" ".join(b) for _, b, _ in found],
    }).to_csv(out_csv, index=False)

    print("PRIMARY    #%d %s   [%s]   in %d patent%s"
          % (p["n"], p["term"], _label(p), len(P), "" if len(P) == 1 else "s"))
    print("SECONDARY  %d technical term%s share a patent with it -> %s"
          % (len(found), "" if len(found) == 1 else "s", out_csv))
    if nested:
        print("           not counted, variants of the primary itself: %s"
              % "; ".join("#%d %s" % (r["n"], r["term"]) for r in nested))
    print()
    if found:
        print("%5s  %-34s %-46s %4s %5s %5s" % ("#", "secondary term", "topic > subtopic",
                                                "both", "share", "lift"))
        for r, both, lift in found[:show]:
            print("%5d  %-34s %-46s %4d %5.2f %5.1f"
                  % (r["n"], r["term"][:34], _label(r)[:46], len(both),
                     len(both) / len(P), lift))
        if len(found) > show:
            print("       ... %d more in %s" % (len(found) - show, out_csv))
        print()
    print("WHICH PATENT")
    docs = {d["pub"]: d for d in state["docs"]}
    listed = {r["n"] for r, _, _ in found}
    in_pat = {pub: [n for n in state["pat_terms"][pub] if n in listed] for pub in P}
    for pub in sorted(P, key=lambda x: (-len(in_pat[x]), x))[:show]:
        d = docs.get(pub, {})
        print("  %-26s %-4s %-44s %s"
              % (pub, d.get("year", ""), (d.get("title", "") or "")[:44],
                 " ".join("#%d" % n for n in in_pat[pub]) or "-"))
    if len(P) > show:
        print("  ... %d more patents in the CSV" % (len(P) - show))
    state["last"] = {"primary": p["n"], "secondary": [r["n"] for r, _, _ in found]}
    return pd.read_csv(out_csv)


# --------------------------------------------------------------------------- #
# step 4 - the patents, for reading
# --------------------------------------------------------------------------- #

def _cut(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " [...]"


def read(state, primary, secondary=None, patent=None, limit=3, chars=2500):
    """Print the patents behind a primary term, whole enough to interpret.

    With `secondary`, only the patents carrying both. With `patent`, that one
    patent at length. Otherwise the `limit` patents carrying the most
    technical terms, and the rest by number. Title, date, CPC, abstract, then
    claims and description cut to `chars` each.
    """
    if not state["docs"]:
        raise SystemExit("the export is not loaded, so there is no text to "
                         "read. Attach it and run load() again.")
    p = _resolve(state, primary)
    pats = list(p["patents"])
    s = None
    if secondary is not None:
        s = _resolve(state, secondary)
        keep = set(s["patents"])
        pats = [x for x in pats if x in keep]
    docs = {d["pub"]: d for d in state["docs"]}
    if patent is not None:
        want = str(patent).strip()
        hits = [x for x in docs if x == want or x.startswith(want)]
        if not hits:
            raise SystemExit("no patent %r in the export" % want)
        pats, limit, chars = hits[:1], 1, max(chars, 12000)

    pats.sort(key=lambda x: (-len(state["pat_terms"].get(x, ())),
                             docs.get(x, {}).get("year", ""), x))
    print("READING    #%d %s%s   %d patent%s%s"
          % (p["n"], p["term"], (" + #%d %s" % (s["n"], s["term"])) if s else "",
             len(pats), "" if len(pats) == 1 else "s",
             ", showing %d" % limit if len(pats) > limit else ""))
    for k, pub in enumerate(pats[:limit], 1):
        d = docs.get(pub)
        if d is None:
            print("\n=== %s   not in the export" % pub)
            continue
        others = [state["by_n"][n] for n in state["pat_terms"].get(pub, ())
                  if n != p["n"]]
        print("\n=== [%d] %s   %s" % (k, pub, d["year"] or "undated"))
        print("TITLE        %s" % d["title"])
        if d["cpc"]:
            print("CPC          %s" % _cut(d["cpc"], 200))
        print("TERMS        primary #%d %s | also here: %s"
              % (p["n"], p["term"],
                 "; ".join("#%d %s" % (r["n"], r["term"]) for r in others) or "none"))
        print("ABSTRACT     %s" % d["abstract"])
        if d["claims"]:
            print("CLAIMS       %s" % _cut(d["claims"], chars))
        if d["description"]:
            print("DESCRIPTION  %s" % _cut(d["description"], chars))
    rest = pats[limit:]
    if rest:
        print("\nNOT SHOWN    %s" % " ".join(rest))
        print("             read(state, %d, patent=\"<number>\") for any of them"
              % p["n"])


# --------------------------------------------------------------------------- #
# traceability
# --------------------------------------------------------------------------- #

def trace(state, term, within=None, limit=20):
    """Every patent a term occurs in, with the sentence it occurs in.
    `within` restricts to the patents carrying another term."""
    r = _resolve(state, term)
    pats = list(r["patents"])
    if within is not None:
        w = _resolve(state, within)
        keep = set(w["patents"])
        pats = [x for x in pats if x in keep]
        print("TRACE      #%d %s within the patents carrying #%d %s"
              % (r["n"], r["term"], w["n"], w["term"]))
    else:
        print("TRACE      #%d %s" % (r["n"], r["term"]))
    forms = sorted(set(r["variants"]) | {r["term"]}, key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Za-z0-9])(?:%s)(?![A-Za-z0-9])" % "|".join(
        r"[\s\-]*".join(re.escape(part) for part in re.split(r"[\s\-]+", f))
        for f in forms), re.IGNORECASE)
    docs = {d["pub"]: d for d in state["docs"]}
    print("           %d patent%s" % (len(pats), "" if len(pats) == 1 else "s"))
    print()
    out = []
    for pub in pats:
        d = docs.get(pub)
        sentence = matched = ""
        if d:
            for part in ("title", "abstract"):
                for sent in _SENT.split(d[part]):
                    m = pattern.search(sent)
                    if m:
                        sentence, matched = sent.strip(), m.group(0)
                        break
                if sentence:
                    break
        out.append({"patent": pub, "year": d["year"] if d else "",
                    "matched": matched, "sentence": sentence})
    for row in out[:limit]:
        s = row["sentence"] or "(export not loaded)"
        at = s.lower().find(row["matched"].lower()) if row["matched"] else 0
        lo = max(0, at - 45)
        hi = min(len(s), lo + 110)
        print("%-26s %-4s %s%s%s" % (row["patent"], row["year"], "..." if lo else "",
                                     s[lo:hi], "..." if hi < len(s) else ""))
    if len(out) > limit:
        print("           ... %d more" % (len(out) - limit))
    return pd.DataFrame(out)
