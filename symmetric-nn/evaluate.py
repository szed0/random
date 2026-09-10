"""Measure every candidate improvement to the symmetric-NN grouping step.

Each scorer maps a term pair to a number; higher means "more likely one
technology". A scorer is judged three ways:

  accuracy   at its own best threshold, swept over the observed range
  margin     lowest true score minus highest false score. Negative means no
             threshold separates the classes at all, and the accuracy figure is
             then the best of a bad job rather than a usable operating point
  shipped    accuracy at the threshold the tool actually ships with, where that
             is meaningful

Margin is the figure that matters. A scorer that separates perfectly with a
0.002 margin is not deployable on a corpus you have not seen.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from benchmark import pairs, vocabulary

# Point these at local copies to run offline; otherwise they download.
MINILM = os.environ.get("TECHSCOPE_EMBED_MODEL", "all-MiniLM-L6-v2")
MPNET = os.environ.get("TECHSCOPE_EMBED_MODEL_B", "all-mpnet-base-v2")


# --------------------------------------------------------------------------- #
# embedding helpers
# --------------------------------------------------------------------------- #

_CACHE: dict[str, np.ndarray] = {}


def embed(model_path: str, terms: list[str]) -> np.ndarray:
    key = model_path
    if key not in _CACHE:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_path)
        v = np.asarray(m.encode(terms, show_progress_bar=False), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        _CACHE[key] = v
        del m
    return _CACHE[key]


def csls_matrix(v: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain Similarity Local Scaling (Conneau et al., ICLR 2018).

        csls(a,b) = 2*cos(a,b) - r(a) - r(b)

    where r(x) is x's mean cosine to its k nearest neighbours. A hub sits close
    to everything, so its r is large and every score involving it is penalised.
    This is the standard correction for the hubness that makes a plain cosine
    threshold unsafe in high dimensions.
    """
    sim = v @ v.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, sim.shape[0] - 1)
    part = np.partition(sim, -k, axis=1)[:, -k:]
    r = part.mean(axis=1)
    np.fill_diagonal(sim, 1.0)
    return 2.0 * sim - r[:, None] - r[None, :]


# --------------------------------------------------------------------------- #
# lexical signals — no model, fully deterministic
# --------------------------------------------------------------------------- #

def tokens(t: str) -> list[str]:
    return t.lower().replace("-", " ").split()


def containment(a: str, b: str) -> float:
    """1.0 when one term's tokens are a subset of the other's.

    This is what catches head-truncation and category-suffix pairs, which no
    embedding needs to be involved in. It is also exactly the signal that misfires
    on sibling technologies, so it can never be used alone.
    """
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def initialism(a: str, b: str) -> bool:
    """Whether one term is the other's initials: PEM / proton exchange membrane.

    Embeddings are blind to this - an acronym and its expansion share no
    subwords - and it is a large share of real patent synonymy.
    """
    def initials(t):
        return "".join(w[0] for w in tokens(t) if w)
    ta, tb = tokens(a), tokens(b)
    if len(ta) == 1 and len(tb) > 1:
        return ta[0].lower() == initials(b).lower()
    if len(tb) == 1 and len(ta) > 1:
        return tb[0].lower() == initials(a).lower()
    # also the multiword case: "PEM electrolyser" / "proton exchange membrane electrolyser"
    if ta and tb and ta[-1] == tb[-1] and len(ta) != len(tb):
        short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
        if len(short) >= 2 and short[0].lower() == "".join(
                w[0] for w in long_[:len(long_) - len(short) + 1]).lower():
            return True
    return False


_BRITISH = [("re", "er"), ("our", "or"), ("ise", "ize"), ("yse", "yze"),
            ("ph", "f"), ("ae", "e"), ("oe", "e"), ("ll", "l"), ("ium", "um")]


def spelling_variant(a: str, b: str) -> bool:
    """British/American and -ise/-ize pairs, which are one technology always."""
    def fold(t):
        t = t.lower()
        for x, y in (("fibre", "fiber"), ("sulph", "sulf"), ("vapour", "vapor"),
                     ("aluminium", "aluminum"), ("ise", "ize"), ("yse", "yze"),
                     ("ser", "zer"), ("sation", "zation")):
            t = t.replace(x, y)
        return t
    return a.lower() != b.lower() and fold(a) == fold(b)


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #

def build_scorers():
    vocab = vocabulary()
    idx = {t: i for i, t in enumerate(vocab)}

    v_mini = embed(MINILM, vocab)
    v_mpnet = embed(MPNET, vocab)
    cos_mini = v_mini @ v_mini.T
    cos_mpnet = v_mpnet @ v_mpnet.T
    csls_mini = csls_matrix(v_mini)
    csls_mpnet = csls_matrix(v_mpnet)

    def pair_of(mat):
        return lambda a, b: float(mat[idx[a], idx[b]])

    cos_m = pair_of(cos_mini)
    cos_p = pair_of(cos_mpnet)

    def hybrid(a, b):
        """Embedding, with the two deterministic relations it cannot see.

        Containment is added rather than substituted: on its own it merges
        sibling technologies, which is the worst error this tool can make.
        """
        s = cos_p(a, b)
        if spelling_variant(a, b) or initialism(a, b):
            return 1.0
        c = containment(a, b)
        return s + 0.25 * c if c == 1.0 else s

    return {
        "cosine MiniLM (shipped)": cos_m,
        "cosine mpnet": cos_p,
        "CSLS MiniLM": pair_of(csls_mini),
        "CSLS mpnet": pair_of(csls_mpnet),
        "mpnet + lexical rules": hybrid,
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def assess(name, score, ps, shipped=None):
    rows = [(score(a, b), lab, cat, a, b) for a, b, lab, cat in ps]
    trues = [r[0] for r in rows if r[1]]
    falses = [r[0] for r in rows if not r[1]]
    cuts = sorted({r[0] for r in rows})
    best_acc, best_t = 0.0, cuts[0]
    for i in range(len(cuts)):
        t = cuts[i]
        acc = sum(1 for s, lab, *_ in rows if (s >= t) == lab) / len(rows)
        if acc > best_acc:
            best_acc, best_t = acc, t
    margin = min(trues) - max(falses)
    out = {"name": name, "acc": best_acc, "thr": best_t, "margin": margin,
           "rows": rows}
    if shipped is not None:
        out["shipped_acc"] = sum(
            1 for s, lab, *_ in rows if (s >= shipped) == lab) / len(rows)
        out["shipped_thr"] = shipped
    return out


def per_category(res):
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0])
    for s, lab, cat, _a, _b in res["rows"]:
        acc[cat][1] += 1
        if (s >= res["thr"]) == lab:
            acc[cat][0] += 1
    return {c: (n, d) for c, (n, d) in acc.items()}


def main():
    ps = pairs()
    scorers = build_scorers()
    results = []
    for name, fn in scorers.items():
        shipped = 0.75 if name.startswith("cosine") else None
        results.append(assess(name, fn, ps, shipped))

    print("=" * 84)
    print("%-26s %8s %9s %9s %s" % ("scorer", "best acc", "at thr", "margin",
                                    "acc @ shipped 0.75"))
    print("-" * 84)
    for r in results:
        sh = ("%.3f" % r["shipped_acc"]) if "shipped_acc" in r else "-"
        flag = "" if r["margin"] > 0 else "   <- classes overlap"
        print("%-26s %8.3f %9.3f %9.3f %18s%s"
              % (r["name"], r["acc"], r["thr"], r["margin"], sh, flag))
    print()

    print("per-category accuracy at each scorer's own best threshold")
    cats = [c for c in dict.fromkeys(cat for _a, _b, _l, cat in ps)]
    header = "%-24s" % "category" + "".join("%14s" % r["name"][:13] for r in results)
    print(header)
    print("-" * len(header))
    for c in cats:
        line = "%-24s" % c
        for r in results:
            n, d = per_category(r)[c]
            line += "%14s" % ("%d/%d" % (n, d))
        print(line)
    print()

    base = results[0]
    print("what the shipped scorer gets wrong at 0.75 (%d errors)"
          % sum(1 for s, lab, *_ in base["rows"] if (s >= 0.75) != lab))
    for s, lab, cat, a, b in sorted(base["rows"], key=lambda r: -r[0]):
        if (s >= 0.75) != lab:
            kind = "MISSED merge" if lab else "WRONG merge "
            print("   %.3f  %-13s %-22s %-34s [%s]" % (s, kind, a[:22], b[:34], cat))


if __name__ == "__main__":
    main()
