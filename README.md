# kinyarwanda_tts_short-words

Investigating why [`C4IR-RW/kinya-flex-tts`](https://huggingface.co/C4IR-RW/kinya-flex-tts)
rushes short utterances, toward a paper on the limits of the model.

**Working claim:** the model reproduces its training distribution faithfully, and the
observed failure is a gap in that distribution's coverage rather than a fault of the
architecture or of the recording quality.

## The headline measurement

Model duration ÷ the actress's duration **on identical text**, so no syllable counting or
speaking-rate proxy is involved. 190 digit-free clips, `lengthScale 1.0`:

| words | n | median ratio | implied `lengthScale` |
|---|---|---|---|
| 1–2 | 33 | 0.875 | 1.14 |
| 3 | 43 | 0.932 | 1.07 |
| 4–6 | 44 | 0.937 | 1.07 |
| 7–12 | 41 | 0.960 | 1.04 |
| 13+ | 29 | 0.986 | 1.01 |

Monotonic: on long sentences the model is within ~1.5% of the human, and the deficit grows
as utterances shorten. Meanwhile the training corpus contains **2 single-word rows out of
17,969 — and both are the spreadsheet error string `Err:508`.**

## What is here

| path | what |
|---|---|
| `exploration/README.md` | orientation, how to run, open questions |
| `exploration/METHODOLOGY.md` | how each number is produced, and the traps behind them |
| `exploration/FINDINGS.md` | dated result log, each entry with its caveats |
| `exploration/notebooks/01_rushing_bench.py` | marimo bench: human vs model, same text |
| `exploration/notebooks/data/sample.tsv` | the exact 250 clips every result uses |
| `exploration/results/` | machine-readable output, one JSON per run |

## Running it with nothing checked out

```bash
curl -sLO https://raw.githubusercontent.com/maqamylee0/kinyarwanda_tts_short-words/main/exploration/notebooks/01_rushing_bench.py
uv run 01_rushing_bench.py
```

The notebook pulls everything it needs at run time: our ONNX export from
[`emmilly/kinya-flex-tts-onnx`](https://huggingface.co/emmilly/kinya-flex-tts-onnx),
C4IR's original checkpoint from
[`C4IR-RW/kinya-flex-tts`](https://huggingface.co/C4IR-RW/kinya-flex-tts), and the corpus
from [`C4IR-RW/kinya-ag-tts`](https://huggingface.co/datasets/C4IR-RW/kinya-ag-tts).
First run downloads ~1.3 GB into the standard Hugging Face cache.

It runs **both engines** on identical text. That is the control a reviewer will ask for:
without C4IR's own checkpoint in the comparison, the measurements could be describing our
ONNX conversion rather than their model.

## What is deliberately not here

Model weights and corpus audio — they live on the Hub, which is both where they belong and
the only option, since the ONNX alone is over GitHub's 100 MB per-file limit.
`exploration/upload_model_to_hf.py` publishes the ONNX; you run it once with your own
token (`hf auth login` first). `sample.tsv` *is* tracked, because it names exactly which
250 clips every result used.

## Attribution

The model and corpus are CC-BY-4.0 and must be attributed to **C4IR Rwanda & KiNLP**. The
corpus (`C4IR-RW/kinya-ag-tts`) was implemented by C4IR Rwanda & KiNLP, supported by GIZ,
financed by BMZ. This repository contains analysis only; it redistributes neither.
