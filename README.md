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
marimo edit 01_rushing_bench.py
```

The notebook fetches the tokenizer, golden vectors and sample manifest from this repo, the
corpus audio from Hugging Face, and the model from
[`maqamylee0/kinya-flex-tts-onnx`](https://huggingface.co/maqamylee0/kinya-flex-tts-onnx).

## What is deliberately not here

The model weights and the corpus audio. The ONNX is 136 MB, over GitHub's 100 MB per-file
limit, so it is published to the Hugging Face Hub instead — see
`exploration/upload_model_to_hf.py`, which you run once with your own token
(`hf auth login` first). `sample.tsv`
*is* tracked, because it names exactly which 250 clips every result used.

## Attribution

The model and corpus are CC-BY-4.0 and must be attributed to **C4IR Rwanda & KiNLP**. The
corpus (`C4IR-RW/kinya-ag-tts`) was implemented by C4IR Rwanda & KiNLP, supported by GIZ,
financed by BMZ. This repository contains analysis only; it redistributes neither.
