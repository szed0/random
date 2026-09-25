# Running the original trt-pb with MiniLM (or any other encoder)

The app is the one at `Z:/pr/tentlytics/trt-pb` — workstream 8 in
[`PROGRESS.md`](PROGRESS.md), the Streamlit reference build. Every line number
below refers to its `main_streamlit.py`.

It uses **two** transformer models for two different jobs. Only one of them is
a drop-in swap, and knowing which saves an afternoon.

| | model | where | job | swappable for MiniLM? |
| --- | --- | --- | --- | --- |
| 1 | `mp_net` — a local sentence-transformers directory, loaded via KeyBERT | `mpnet_load.py`, `TextEmbedding.py` | embed abstracts and candidate keyphrases; KeyBERT picks the top-n per abstract | **yes**, one line |
| 2 | **PatentBERT** — a TensorFlow SavedModel under `temp_dir/rawout/` | `load_patent_bert.py`, `patentbert.py` | `map_primary_to_secondary`: mask each primary phrase and rank the others by masked-LM logits | **no** |

MiniLM cannot replace PatentBERT. A sentence-transformers model has no
masked-language-model head, and `BertPredictor` expects a TF SavedModel
signature. You either keep PatentBERT for that one step, or replace the step —
`secondary_by_similarity()` in [`trtpb_models.py`](trtpb_models.py) does the latter using the
encoder you already loaded.

spaCy (`en_core_web_lg-3.5.0`) is a third model, used by
`KeyphraseTfidfVectorizer` to chunk noun phrases. It is unrelated to either
swap; leave it alone.

## The swap

Put `trtpb_models.py` beside `main_streamlit.py`, then make two edits.

**`main_streamlit.py:57`** — the encoder

```python
@st.cache_resource
def load_model():
    import trtpb_models as tm
    return tm.load_encoder()        # was mpnet_load.load_mpnet()
```

**`main_streamlit.py:82`** — the embedding call

```python
    import trtpb_models as tm
    embedding = tm.embed(kw_model, abstracts, min_word, max_word,
                         stopwords=stopwords, candidates=candidates)
```

Then pick a model without touching code again:

```bash
TRTPB_MODEL=minilm streamlit run main_streamlit.py
```

```
minilm         all-MiniLM-L6-v2      384-dim, ~5x faster than mpnet
minilm-l12     all-MiniLM-L12-v2     384-dim, slower, a little stronger
mpnet          all-mpnet-base-v2     768-dim, the hub copy of what shipped
mpnet-local    mp_net                768-dim, the directory in the repo
bge-small      BAAI/bge-small-en-v1.5
gte-small      thenlper/gte-small
patentsberta   AI-Growth-Lab/PatentSBERTa    trained on patent text
specter2       allenai/specter2_base         trained on scientific documents
```

`python trtpb_models.py --list` prints this with the resolved ids. Any other
HuggingFace id or local path works too — `TRTPB_MODEL=intfloat/e5-base-v2`.

## Three things that will bite

**The embedding cache.** The app writes one `embedding.npy` and the UI tells you
to delete it by hand when you change dataset. Changing *model* is worse: mpnet
writes 768-wide vectors, MiniLM reads 384-wide ones, and the mismatch surfaces
far from its cause. Use a per-model filename —

```python
    path = Path(tm.embedding_cache_path())      # main_streamlit.py:527
    ...
    if tm.check_cache(str(path)):
        embedding = numpy.load(str(path), allow_pickle=True)
    else:
        embedding = embeddings(...)
        numpy.save(str(path), embedding, allow_pickle=True)
```

`check_cache` verifies the width against the model, so a stale file is ignored
rather than loaded.

**Thresholds do not transfer.** Two numbers in the app were tuned against mpnet
and PatentBERT:

- `KeywordDictionary.getkeyword_dict(..., threshold)` — cosine 0.80 for grouping
  keyphrases into one technology.
- the sidebar's 25–150 "inverse similarity threshold" — a cutoff on average MLM
  rank, not a cosine at all.

Different encoders put their similarities on different scales, so 0.80 under
mpnet is not 0.80 under MiniLM. Check before reusing it:

```bash
python trtpb_models.py --compare "fuel cell stack" "fuel cell system"
```

If you replace `map_primary_to_secondary` with the cosine version, the sidebar
slider still works — `similarity_from_slider()` maps 25–150 onto a cosine so
that turning it up still means "more matches".

**Worth knowing before tuning that threshold at all:** on 77 hand-labelled
patent term pairs, no threshold on any symmetric similarity separated "one
technology" from "two". Every margin was negative, and the highest-scoring pair
in the set was a false one — *type III* vs *type IV pressure vessel* at 0.830,
above genuine synonyms at 0.742. CSLS, mpnet and two cross-encoders were
measured and none fixed it. The measurements live in `random-backup.git`.
Switching encoders will move the numbers; it will not make a clean threshold
exist.

The in-house precedent is `sao` (`Z:/pr/tentlytics/sao`), which already runs
all-MiniLM-L6-v2 for Subject-Action-Object extraction — worth reading before
tuning anything here.

## Dropping PatentBERT entirely

If you want the app to run without the TF SavedModel — it is the slowest and
most fragile part of the setup — replace the call at `main_streamlit.py:550`:

```python
    keydict = tm.secondary_by_similarity(
        kws=keywords,
        min_keys=min_keys,
        threshold=tm.similarity_from_slider(threshold),
    )
```

and delete the `load_patent_bert` imports at lines 16 and 20. That also drops
tensorflow from the requirements, which is most of the install pain.

Be clear about what changes: the MLM asked *"would this phrase fit in this
blank"*, cosine asks *"do these two phrases mean similar things"*. Related
components that never substitute for each other — anode and cathode — score
high on cosine and low on the MLM. The groupings will differ, and for a
relationship graph the MLM's question is arguably the better one. It is a
trade, not an upgrade.

## Environment

The pins matter; the app is sensitive to them. From `requirements-resolved.txt`:

```
keybert==0.7.0              sentence-transformers 2.2.2 comes with it
keyphrase_vectorizers==0.0.11
spacy==3.5.3
torch==2.0.1
transformers==4.30.2
tensorflow==2.13.1          only needed if you keep PatentBERT
```

On Windows, `torch` must be imported before `scikit-learn` or the duplicate
OpenMP runtime segfaults the interpreter on `import keybert`;
`KMP_DUPLICATE_LIB_OK=TRUE` does not help, and the repo's `sitecustomize.py`
handles it by importing torch first. `trtpb_models.py` imports lazily and inside
functions, so it does not disturb that order.
