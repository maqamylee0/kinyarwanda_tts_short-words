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
marimo edit exploration/notebooks/01_rushing_bench.py   # reactive notebook
marimo run  exploration/notebooks/01_rushing_bench.py   # read-only app
uv run      exploration/notebooks/01_rushing_bench.py   # deps resolved from the header

# headless, writes a timestamped JSON to the results folder
BENCH_SAVE=1 marimo export html --no-include-code \
  exploration/notebooks/01_rushing_bench.py -o /tmp/rb.html
```

Nothing needs to be checked out or placed by hand. Everything is pulled at run time:

| what | from | size |
|---|---|---|
| our ONNX export | `emmilly/kinya-flex-tts-onnx` | 136 MB |
| C4IR's checkpoint | `C4IR-RW/kinya-flex-tts` | 1.11 GB |
| the corpus (250-clip seeded sample) | `C4IR-RW/kinya-ag-tts` | ~53 MB |
| tokenizer + model code | `github.com/c4ir-rw/ac-ai-models` | 24 MB clone |
| golden vectors | this repo, raw | 20 KB |

Downloads go through the standard Hugging Face cache (`~/.cache/huggingface`), so they
are shared with other tools and fetched once. The sample manifest and results land in
`./kinya_bench/`, overridable with `KINYA_WORK`.

**First run pulls ~1.3 GB.** Subsequent runs are cached.

### Two engines

The bench runs our ONNX export *and* C4IR's own PyTorch checkpoint on identical text.
That is the control: if the export had altered the duration predictor, every measurement
here would describe our artefact rather than C4IR's model. Section 6 compares them
directly at `noise_scale=0`, where a faithful export gives identical sample counts.

The torch engine needs `torch`, `typed-argument-parser` and `librosa` (all in the script
header, so `uv run` handles them). If it cannot load, the bench says so and continues with
ONNX only rather than failing.

### Environment overrides

| variable | effect |
|---|---|
| `KINYA_ONNX_REPO` / `KINYA_ONNX_FILE` | where the ONNX comes from |
| `KINYA_TORCH_REPO` / `KINYA_TORCH_FILE` | where the checkpoint comes from |
| `KINYA_DATASET_REPO` | corpus repo |
| `KINYA_WORK` | working dir for `sample.tsv` and results |
| `BENCH_SAVE=1` | record the run without pressing the button |

If `sample.tsv` already exists it is **reused verbatim rather than redrawn**, so the
sample stays fixed across runs.

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
