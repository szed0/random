# What the sandbox actually is

Measured in the browser on **2026-09-18**, free-tier account, by executing the
probes in a Gem and reading the output cells. Three of the environment facts
the other directories are built on no longer hold.

## Code execution is Gem-only

| Where | Python tool | What happened |
|---|---|---|
| Ordinary chat, 3.0 Flash | **absent** | "I do not have an integrated Python code execution environment" |
| Ordinary chat, 3.0 Pro | **absent** | asked to run `print(sum(range(1,101)))`, it replied "The output is: 5050" with no execution |
| Pro with Canvas enabled | **absent** | wrote the script into a Canvas artifact; no run control; "I do not have a live execution environment" |
| **Inside a Gem** | **present** | real Show-code block, executed output cell |

`SETUP.md` says the tool "was present in ordinary chats". It is not, on this
account, today. **Build the Gem; do not test in a plain chat.**

The Pro case is the one to keep in mind. Asked for a number it could compute
in its head, it produced the right answer and presented it as program output.
Nothing in an ordinary chat stops that. The Gems only refuse because their
instructions tell them to, which is the whole reason that rule is written the
way it is.

## A Gem knowledge file *is* a file in the sandbox

```
>>> import os; print(os.listdir('.'))
['trt_keywords.py']
```

Run inside the `Keyword trends KB test` Gem, whose knowledge field holds
`trt_keywords.py` and whose chat had no attachment. Reproduced twice, in
separate conversations.

`keyword-trends/README.md` states the opposite — that a knowledge file "is
**not** a file in the code sandbox" and that the model's only route is
retyping 31 KB per turn. That was true when measured and is not true now.

If it holds, the knowledge field is the better home for a module: it survives
the sandbox wipe without re-attaching, and the retyping failures that
documented section describes cannot happen. **Confirm it on your own Gem
before rebuilding a workflow around it** — that finding came from a Gem built
to demonstrate the old behaviour, and platform behaviour has now moved once.

## The library set

```
3.10.16 (main, Apr  8 2025, 01:38:46) [GCC 12.2.0]
OK pandas 2.0.0        OK sklearn 1.4.0      OK spacy 3.8.5
OK numpy 1.26.3        OK scipy 1.12.0       OK networkx 3.4.2
OK matplotlib 3.6.0    OK nltk 3.10.0        NO sentence_transformers
OK seaborn 0.12.2      OK plotly 5.20.0      NO pyvis
NET-NO URLError
```

`SETUP.md` lists spaCy, scikit-learn and TensorFlow as absent, and
`SKILL.md` adds networkx. **scikit-learn, scipy, nltk, spaCy and networkx are
all present.** Genuinely absent: `sentence_transformers`, `pyvis`, and the
network.

### The catch on spaCy

```
>>> import spacy; print(spacy.util.get_installed_models())
[]
```

spaCy is installed with **no models**, and `NET-NO` means none can be
downloaded. So it is a tokenizer and nothing else: no part-of-speech tagger,
no lemmatizer, no parser. The reference tool's keyphrase pattern
`<J.*>*<N.*>+` runs on tags spaCy cannot produce here, so the closed-class
word list stays. The same applies to nltk, whose taggers are separate data
downloads.

### What does change

- **scikit-learn** gives real `TfidfVectorizer`, `TruncatedSVD` and
  `cosine_similarity`. That makes an LSA embedding possible, which is the
  nearest thing to KeyBERT's "embed the abstract, embed the candidates, keep
  the nearest" that this sandbox can support. See the `lsa` method.
- **networkx** means the relationship graph can be built natively rather than
  hand-rolled.
- **`sentence_transformers` is still absent**, so MPNet itself is still out of
  reach, and no character-trigram or LSA stand-in changes that.

## How to re-check

One message inside your Gem, which is cheap and worth doing before any long
session, because this has now changed once:

> Capability check only, ignore the workflow. Run this and show the output:
> `import trt_extract as tx; tx.report_capabilities()`

`report_capabilities()` prints the interpreter's real answer for every library
the methods depend on, and names which ranking methods are therefore
available. If scikit-learn is missing, `tfidf` and `lsa` silently fall back to
`runs` and the printed line says so.
