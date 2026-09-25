# Running trt / trt-pb with MiniLM (or any other encoder)

Two Streamlit apps in this project embed text, and they are not the same job.
Start with `trt`: it has one model and the swap is three lines.

| app | path | models | swap difficulty |
| --- | --- | --- | --- |
| **`trt`** | `Z:/pr/tentlytics/trt` | one — `mp_net` via KeyBERT | **easy**, one function |
| **`trt-pb`** | `Z:/pr/tentlytics/trt-pb` | two — `mp_net` **and** PatentBERT | encoder yes, PatentBERT no |

Workstreams 8 and 9 in [`PROGRESS.md`](PROGRESS.md). Line numbers below are from
each app's own `main_streamlit.py`.

`sao` already ships a complete `all-MiniLM-L6-v2` directory at
`Z:/pr/tentlytics/sao/all-MiniLM-L6-v2` — 384-dim, safetensors, pooling config
and all. Nothing needs downloading; point at that path and it loads offline.

## `trt` — the easy one

One model does two jobs, and both go through `load_model()`:

| line | what | job |
| --- | --- | --- |
| 29 | `load_model()` → `mpnet_load.load_mpnet()` | the encoder |
| 53 | `TextEmbedding.mpnet_embedding(...)` | embed abstracts + candidates for KeyBERT |
| 88 | `keywordsynonyms(kws, threshold)` | group keyphrases by cosine ≥ threshold |

`keywordsynonyms` calls `load_model()` itself, so changing that one function
swaps the encoder for **both** keyword extraction and synonym grouping. There
is no third edit.

Put [`trtpb_models.py`](trtpb_models.py) beside `main_streamlit.py`:

```python
# main_streamlit.py:29
@st.cache_resource
def load_model():
    import trtpb_models as tm
    return tm.load_encoder()             # was mpnet_load.load_mpnet()
```

```python
# main_streamlit.py:53
    import trtpb_models as tm
    embedding = tm.embed(kw_model, abstracts, min_word, max_word,
                         stopwords=stopwords, candidates=candidates)
```

```bash
TRTPB_MODEL=minilm-local streamlit run main_streamlit.py
```

That is the whole change. No TensorFlow, no SavedModel, no `temp_dir`.

### The threshold is the real work

`trt` puts the grouping threshold straight in the sidebar —
`st.select_slider("Pick similarity threshold", options=range(75,95,5), value=80)`
at line 386, used as `threshold*0.01`, so the default is **cosine 0.80** on
whatever encoder `load_model()` returned.

That 0.80 was chosen against mpnet. It does not carry over. mpnet and MiniLM
place their similarities on different scales, and the slider only spans
0.75–0.95, so a model whose scores sit lower will group nothing across the
whole range. Check before trusting it:

```bash
python trtpb_models.py --compare "fuel cell stack" "fuel cell system"
```

If the new model's scores for pairs you judge equivalent land outside
0.75–0.95, widen the slider as well as moving the default:

```python
    threshold = st.select_slider("Pick similarity threshold",
                                 options=range(40, 96, 5), value=70)
```

**Before spending long on that number:** on 77 hand-labelled patent term pairs,
no threshold on any symmetric similarity separated "one technology" from "two".
Every margin was negative, and the top-scoring pair in the set was a false one —
*type III* vs *type IV pressure vessel* at 0.830, above genuine synonyms at
0.742. CSLS, mpnet and two cross-encoders were measured; none fixed it. The
measurements live in `random-backup.git`. A different encoder moves the
numbers; it does not make a clean threshold exist.

### One thing to know about how it groups

`keywordsynonyms` embeds each keyphrase as a *document*:

```python
kw_vectors, dummyvecs = nlp.extract_embeddings(kws, keyphrase_ngram_range=(3,3),
                                               stop_words='english')
```

The `(3,3)` n-gram range and the discarded `dummyvecs` do nothing here — the
vectors used are the document embeddings of the phrase strings. That is fine,
and it is worth knowing it is what is happening, because it means grouping
quality depends entirely on how the encoder handles very short inputs. MiniLM
is trained on short sentences and does this well; models tuned for passages
(bge, e5, gte) are the ones to watch.

## `trt-pb` — the encoder swaps, PatentBERT does not

`trt-pb` is `trt` plus a second model:

| | model | where | job | swappable? |
| --- | --- | --- | --- | --- |
| 1 | `mp_net` via KeyBERT | `mpnet_load.py`, `TextEmbedding.py` | as in `trt` | **yes** |
| 2 | **PatentBERT**, a TensorFlow SavedModel | `load_patent_bert.py`, `patentbert.py` | `map_primary_to_secondary`: mask each primary phrase, rank the others by masked-LM logits | **no** |

MiniLM cannot stand in for #2. A sentence-transformers model has no
masked-language-model head, and `BertPredictor` expects a TF signature.

Same two edits as `trt`, at **lines 57 and 82**. Then either keep PatentBERT, or
drop it — replace the call at **line 550**:

```python
    keydict = tm.secondary_by_similarity(
        kws=keywords,
        min_keys=min_keys,
        threshold=tm.similarity_from_slider(threshold),
    )
```

and delete the `load_patent_bert` imports at lines 16 and 20. That removes
tensorflow from the requirements, which is most of the install pain.

Be clear about what changes. The MLM asked *"would this phrase fit in this
blank"*; cosine asks *"do these two phrases mean similar things"*. Related
components that never substitute for each other — anode and cathode — score
high on cosine and low on the MLM. For a relationship graph the MLM's question
is arguably the better one. It is a trade, not an upgrade.

`trt-pb`'s own sidebar threshold (25–150, "inverse similarity") is a cutoff on
average MLM rank, not a cosine at all. `similarity_from_slider()` maps it onto
one so the slider keeps its direction: turn it up, get more secondary keywords.

## Choosing a model

```
minilm-local   Z:/pr/tentlytics/sao/all-MiniLM-L6-v2   384, already on disk
minilm         all-MiniLM-L6-v2                        384, ~5x faster than mpnet
minilm-l12     all-MiniLM-L12-v2                       384, slower, a little stronger
mpnet          all-mpnet-base-v2                       768, hub copy of what shipped
mpnet-local    mp_net                                  768, the directory in each app
bge-small      BAAI/bge-small-en-v1.5                  384, wants an instruction prefix
gte-small      thenlper/gte-small                      384
patentsberta   AI-Growth-Lab/PatentSBERTa              768, trained on patent text
specter2       allenai/specter2_base                   768, trained on scientific documents
```

`python trtpb_models.py --list` prints this with resolved ids. Any HuggingFace
id or local path works: `TRTPB_MODEL=intfloat/e5-base-v2`.

For patent phrases specifically, `patentsberta` is the one worth measuring
against `minilm` before settling — it is the only entry trained on this text
type, at mpnet's size and cost.

## The embedding cache

Both apps write a single `embedding.npy` and tell you in the UI to delete it by
hand when you change dataset. Changing **model** is the worse case: mpnet writes
768-wide vectors, MiniLM reads 384-wide, and the mismatch surfaces far from its
cause. Key the file by model instead — `trt:375`, `trt-pb:527`:

```python
    path = Path(tm.embedding_cache_path())
    ...
    if tm.check_cache(str(path)):
        embedding = numpy.load(str(path), allow_pickle=True)
    else:
        embedding = embeddings(...)
        numpy.save(str(path), embedding, allow_pickle=True)
```

`check_cache` verifies the stored width against the current model, so a stale
file is ignored rather than loaded.

## Environment

The pins matter; both apps are sensitive to them.

```
keybert==0.7.0              sentence-transformers 2.2.2 comes with it
keyphrase_vectorizers==0.0.11
spacy==3.5.3                the noun-phrase chunker, unrelated to either swap
torch==2.0.1
transformers==4.30.2
tensorflow==2.13.1          trt-pb only, and only if you keep PatentBERT
```

On Windows `torch` must be imported before `scikit-learn`, or the duplicate
OpenMP runtime segfaults the interpreter on `import keybert`;
`KMP_DUPLICATE_LIB_OK=TRUE` does not help, and each app's `sitecustomize.py`
handles it by importing torch first. `trtpb_models.py` imports lazily and inside
functions, so it does not disturb that order.
