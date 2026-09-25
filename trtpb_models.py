"""Swap the sentence encoder in the trt and trt-pb Streamlit apps.

`trt` (Z:/pr/tentlytics/trt) has one model and swaps cleanly. `trt-pb` has that
same encoder plus PatentBERT, and only the encoder can move. See MODELS.md.

The two models, where both are present:

  1. a SENTENCE ENCODER (`mp_net`, a local sentence-transformers directory)
     loaded through KeyBERT. It embeds abstracts and candidate keyphrases, and
     KeyBERT picks the top-n keyphrases per abstract by cosine similarity. In
     `trt` the same encoder also does synonym grouping, because
     `keywordsynonyms` calls `load_model()` itself. ANY sentence-transformers
     model can take its place.

  2. PATENTBERT (`load_patent_bert.py`, a TensorFlow SavedModel) used only by
     `map_primary_to_secondary` to rank secondary keyphrases by masked-language
     -model logits. MiniLM cannot stand in here: a sentence-transformers model
     has no MLM head, and the predictor expects a TF signature. Either keep
     PatentBERT for that step or replace the step - `secondary_by_similarity`
     below does the latter with the encoder you already have.

Drop this file next to `main_streamlit.py`. It imports keybert lazily, so it is
safe to have in the tree without the environment installed.

    python trtpb_models.py --list
    python trtpb_models.py --compare "fuel cell stack" "fuel cell system"
"""

from __future__ import annotations

import os
import sys

# Known-good choices. Anything sentence-transformers can load also works: pass
# a HuggingFace id or a local directory.
MODELS = {
    # Already on disk in sao/ - loads offline, nothing to download.
    "minilm-local": "Z:/pr/tentlytics/sao/all-MiniLM-L6-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "mpnet-local": "mp_net",              # what the app shipped with
    "bge-small": "BAAI/bge-small-en-v1.5",
    "gte-small": "thenlper/gte-small",
    "patentsberta": "AI-Growth-Lab/PatentSBERTa",
    "specter2": "allenai/specter2_base",
}

# Dimensions, so a stale cache is caught rather than silently reused.
DIMS = {
    "Z:/pr/tentlytics/sao/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "mp_net": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "thenlper/gte-small": 384,
    "AI-Growth-Lab/PatentSBERTa": 768,
    "allenai/specter2_base": 768,
}

# Models trained with an instruction prefix score badly without it. bge and e5
# are the common traps; the app never passed a prefix, so this is on us.
PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

DEFAULT = os.environ.get("TRTPB_MODEL", "minilm")


def resolve(name=None):
    """A short name, a HuggingFace id, or a local path -> the model id."""
    name = name or DEFAULT
    return MODELS.get(name, name)


def cache_tag(name=None):
    """A filename-safe tag, so each model gets its own embedding cache.

    The app cached to a single `embedding.npy` and told you in the UI to delete
    it by hand when changing dataset. Changing MODEL is the worse case: mpnet
    writes 768-wide vectors and MiniLM reads 384-wide ones, and the mismatch
    surfaces somewhere far from the cause. Keying the file by model removes the
    instruction and the footgun with it.
    """
    return resolve(name).replace("/", "_").replace("\\", "_")


def embedding_cache_path(name=None, folder="."):
    return os.path.join(folder, "embedding.%s.npy" % cache_tag(name))


def load_encoder(name=None):
    """A KeyBERT wrapping the chosen sentence encoder.

    Replaces `mpnet_load.load_mpnet()`. Returns the same kind of object, so
    every later call in the app is unchanged.
    """
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    model_id = resolve(name)
    st = SentenceTransformer(model_id)
    return KeyBERT(model=st)


def embed(kw_model, abstracts, min_word, max_word, stopwords="english",
          candidates=None):
    """Replaces `TextEmbedding.mpnet_embedding`, signature for signature.

    The prefix, where a model wants one, goes on the documents only - the
    candidate phrases are not passages.
    """
    prefix = PREFIX.get(getattr(kw_model.model, "_trtpb_id", ""), "")
    docs = [prefix + a for a in abstracts] if prefix else abstracts
    return kw_model.extract_embeddings(
        docs,
        keyphrase_ngram_range=(min_word, max_word),
        stop_words=stopwords,
        candidates=candidates,
    )


def check_cache(path, name=None):
    """True when a cached embedding matches the current model's width."""
    import numpy as np

    if not os.path.exists(path):
        return False
    try:
        got = np.load(path, allow_pickle=True)
        want = DIMS.get(resolve(name))
        if want is None:
            return True
        return int(np.asarray(got[0]).shape[-1]) == want
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# running without PatentBERT
# --------------------------------------------------------------------------- #

def secondary_by_similarity(kws, min_keys=5, threshold=0.55, name=None,
                            encoder=None, top_n=None):
    """`map_primary_to_secondary` without the masked-language model.

    The original masked each primary phrase in every abstract, asked PatentBERT
    to fill the blank, and ranked the other phrases by their average rank in
    those predictions. That needs an MLM head and a TensorFlow SavedModel.

    This ranks by cosine similarity between phrase embeddings instead, which is
    what the rest of the pipeline already speaks. It is not the same measure:
    the MLM asks "would this phrase fit here", cosine asks "do these two mean
    similar things". Expect different groupings, and read the note on
    thresholds in the README before trusting either.

    Returns {primary: [secondary, ...]} with at least `min_keys` each, the same
    shape the Streamlit code consumes.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    phrases = list(dict.fromkeys(k for k in kws if k))
    if not phrases:
        return {}

    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(resolve(name))
    vectors = encoder.encode(phrases, show_progress_bar=False,
                             normalize_embeddings=True)
    sim = cosine_similarity(np.asarray(vectors))
    np.fill_diagonal(sim, -1.0)

    out, used = {}, set()
    for i, primary in enumerate(phrases):
        if primary in used:
            continue
        order = np.argsort(-sim[i])
        picked = [phrases[j] for j in order if sim[i][j] >= threshold]
        if len(picked) < min_keys:
            picked = [phrases[j] for j in order[:min_keys]]
        if top_n:
            picked = picked[:top_n]
        out[primary] = picked
        used.add(primary)
        used.update(picked)
    return out


def similarity_from_slider(slider):
    """The app's 25-150 "inverse similarity" slider, as a cosine threshold.

    The slider was a cutoff on average MLM rank, where a bigger number meant
    more matches. Cosine runs the other way and lives in [0, 1], so the two are
    mapped here rather than in the UI, and the slider keeps its meaning: turn
    it up, get more secondary keywords.
    """
    slider = max(25, min(150, float(slider)))
    return round(0.85 - 0.004 * (slider - 25), 3)     # 25 -> 0.85, 150 -> 0.35


# --------------------------------------------------------------------------- #
# a way to check before you trust it
# --------------------------------------------------------------------------- #

def compare(pairs, names=("minilm", "mpnet-local")):
    """Cosine for the same phrase pairs under several models.

    Thresholds do not transfer between encoders. Run this on pairs you can
    judge yourself before reusing a number that was tuned for mpnet.
    """
    from sentence_transformers import SentenceTransformer, util

    rows = []
    for name in names:
        try:
            model = SentenceTransformer(resolve(name))
        except Exception as exc:
            rows.append((name, "could not load: %s" % exc, None))
            continue
        for a, b in pairs:
            va, vb = model.encode([a, b], normalize_embeddings=True)
            rows.append((name, "%s | %s" % (a, b),
                         round(float(util.cos_sim(va, vb)), 3)))
    return rows


def _main(argv):
    if "--list" in argv:
        print("short name        model id                                        dim")
        for short, mid in MODELS.items():
            print("%-17s %-47s %s" % (short, mid, DIMS.get(mid, "?")))
        print("\nTRTPB_MODEL is %r, resolving to %r" % (DEFAULT, resolve()))
        return 0
    if "--compare" in argv:
        i = argv.index("--compare")
        rest = argv[i + 1:]
        pairs = [(rest[j], rest[j + 1]) for j in range(0, len(rest) - 1, 2)]
        if not pairs:
            pairs = [("fuel cell stack", "fuel cell system"),
                     ("membrane electrode assembly", "mea"),
                     ("type iii pressure vessel", "type iv pressure vessel")]
        for name, pair, score in compare(pairs):
            print("%-14s %-60s %s" % (name, pair, score))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
