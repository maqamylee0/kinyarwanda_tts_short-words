# Limits of kinya-flex-tts — exploration log

Working notes, benchmarks and results for a paper on the failure modes of
[`C4IR-RW/kinya-flex-tts`](https://huggingface.co/C4IR-RW/kinya-flex-tts)
(MB-iSTFT-VITS2, 24 kHz, 3 speakers, CC-BY-4.0, C4IR Rwanda & KiNLP).

The working claim: **the model reproduces its training distribution faithfully, and its
observed failures are failures of that distribution's coverage, not of the architecture or
of the recordings' quality.** Everything here is an attempt to support or break that claim.

## Layout

| path | what |
|---|---|
| `README.md` | this file — orientation and status |
| `METHODOLOGY.md` | how every measurement is made, and the traps we hit making them |
| `FINDINGS.md` | dated log of results, each with its method and its caveats |
| `notebooks/01_rushing_bench.py` | marimo: human vs model on identical text |
| `results/` | machine-readable output, one JSON per run |

Shared with the rest of the repo, not duplicated here:

| path | what |
|---|---|
| `../kinya_ag_sample/` | 250-clip stratified sample of the training corpus + transcripts |
| `../vendor/ac-ai-models/` | C4IR's `deepkin` tokenizer, cloned |
| `../rushing_bench.ipynb` | Jupyter twin of the marimo notebook (same measurements) |
| `../kinyarwanda_tts_app/assets/models/kinya_flex_tts.onnx` | the model under test |

## Running the bench

```bash
# reactive notebook — sliders re-synthesize as you move them
marimo edit exploration/notebooks/01_rushing_bench.py

# read-only app
marimo run exploration/notebooks/01_rushing_bench.py

# headless, writes a timestamped JSON to results/
BENCH_SAVE=1 marimo export html --no-include-code \
  exploration/notebooks/01_rushing_bench.py -o /tmp/rb.html
```

### Where it gets its inputs

The notebook needs no checkout to run. Resolution order is: a local bundle (a folder with
`assets/kinya_flex_tts.onnx` beside it), then this project checkout, then **download**.

| asset | stored on | notes |
|---|---|---|
| tokenizer, golden vectors, `sample.tsv` | **GitHub**, this repo | 72 KB total, fetched over https |
| corpus audio (250 clips) | **Hugging Face**, `C4IR-RW/kinya-ag-tts` | stratified sample, ~53 MB |
| model ONNX | **Hugging Face**, `maqamylee0/kinya-flex-tts-onnx` | 136 MB, over GitHub's 100 MB per-file limit |

So a collaborator with nothing checked out can do:

```bash
curl -sLO https://raw.githubusercontent.com/maqamylee0/kinyarwanda_tts_short-words/main/exploration/notebooks/01_rushing_bench.py
marimo edit 01_rushing_bench.py
```

Everything lands in `./kinya_bench_cache/` and is reused afterwards.

**The model must be published once before that works.** It is not in git, by necessity:

```bash
hf auth login                    # or export HF_TOKEN=hf_...
uv run exploration/upload_model_to_hf.py
```

Until then, point at a local copy instead:

```bash
KINYA_MODEL=/path/to/kinya_flex_tts.onnx marimo edit exploration/notebooks/01_rushing_bench.py
```

### Environment overrides

| variable | effect |
|---|---|
| `KINYA_MODEL` | path to a local ONNX, skipping the download |
| `KINYA_HF_MODEL_REPO` | Hub repo holding the model (default `maqamylee0/kinya-flex-tts-onnx`) |
| `KINYA_MODEL_URL` | full URL to the ONNX, overriding the repo |
| `KINYA_BENCH` / `KINYA_ROOT` | force the bundle or project directory |
| `KINYA_CACHE` | where downloads land (default `./kinya_bench_cache`) |
| `KINYA_GH_RAW` | raw-content base URL, for a fork |
| `BENCH_SAVE=1` | record the run to `results/` without the button |

If `kinya_ag_sample/sample.tsv` already exists it is **reused verbatim rather than
redrawn**, so the sample stays fixed across runs.

## Status

Settled (see `FINDINGS.md`): the corpus is not rushed; the model matches the actress on
long sentences and diverges monotonically as utterances shorten; the corpus contains no
genuine single-word recordings.

Open:

- No ground truth for isolated words anywhere in the corpus, so the correct `lengthScale`
  for a single word is extrapolated, not measured. Needs either new recordings or
  forced-aligned utterance-final words.
- A right-skewed tail of clips where the model produces *much* longer audio than the human
  (mean ratio 1.29 vs median 0.95). Not yet characterised — these may be a separate
  failure mode.
- Per-token durations are not exposed by the exported ONNX, blocking exact word cropping
  from carrier phrases.
- Speaker mapping between corpus voices and model speaker ids is assumed, not documented.

## Attribution

Model and corpus are CC-BY-4.0 and must be attributed to **C4IR Rwanda & KiNLP**. The
corpus (`C4IR-RW/kinya-ag-tts`) was implemented by C4IR Rwanda & KiNLP, supported by GIZ,
financed by BMZ.
