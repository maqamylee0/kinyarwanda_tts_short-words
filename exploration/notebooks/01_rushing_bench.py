# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "onnxruntime",
#     "huggingface_hub",
#     "torch",
#     "typed-argument-parser",
#     "librosa",
# ]
# ///
"""Rushing bench for kinya-flex-tts — two engines against the corpus they were trained on.

Everything is pulled from the Hugging Face Hub at run time; nothing needs to be checked
out or placed by hand:

    emmilly/kinya-flex-tts-onnx    our ONNX export  (136 MB)  -> onnxruntime
    C4IR-RW/kinya-flex-tts         C4IR's checkpoint (1.11 GB) -> torch
    C4IR-RW/kinya-ag-tts           the training corpus         -> a seeded 250-clip sample

Running both engines answers the question a reviewer will ask: is the short-utterance
rushing a property of C4IR's model, or did our ONNX conversion introduce it? Section 6
compares the two directly on identical text with noise_scale=0, where any difference is
the export's doing.

    uv run exploration/notebooks/01_rushing_bench.py      # app mode
    marimo edit exploration/notebooks/01_rushing_bench.py # notebook mode

First run downloads ~1.3 GB and caches it under ~/.cache/huggingface.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Kinya Flex-TTS rushing bench")


# ── Cell 1: title ────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Is the model rushing, or is the training data rushed?

        Two engines, one corpus, identical text.

        | | |
        |---|---|
        | **human** | the actress's real recording, from `C4IR-RW/kinya-ag-tts` |
        | **onnx** | our export, `emmilly/kinya-flex-tts-onnx` — what the app ships |
        | **torch** | C4IR's own checkpoint, `C4IR-RW/kinya-flex-tts` — the source of truth |

        Because every comparison uses the same text as a real recording, the measurement
        needs no syllable counting and no speaking-rate proxy: if a model's clip is shorter
        than hers, it is rushing, by exactly that ratio.

        Running C4IR's checkpoint alongside ours separates two things that would otherwise
        be confounded — the model's behaviour, and our conversion of it.
        """
    )
    return (mo,)


# ── Cell 2: imports ──────────────────────────────────────────────────────────
@app.cell
def _():
    import json
    import os
    import random
    import re
    import subprocess
    import sys
    import warnings
    import wave
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np

    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    return (
        Path, ThreadPoolExecutor, datetime, json, np, os, random, re,
        subprocess, sys, timezone, wave,
    )


# ── Cell 3: configuration ────────────────────────────────────────────────────
@app.cell
def _(Path, os):
    # Hub coordinates. Override any of these from the environment.
    ONNX_REPO    = os.environ.get("KINYA_ONNX_REPO", "emmilly/kinya-flex-tts-onnx")
    ONNX_FILE    = os.environ.get("KINYA_ONNX_FILE", "kinya_flex_tts.onnx")
    TORCH_REPO   = os.environ.get("KINYA_TORCH_REPO", "C4IR-RW/kinya-flex-tts")
    TORCH_FILE   = os.environ.get("KINYA_TORCH_FILE", "kinya_flex_tts_base_trained.pt")
    DATASET_REPO = os.environ.get("KINYA_DATASET_REPO", "C4IR-RW/kinya-ag-tts")

    # Working directory for the sample manifest and results. Everything downloaded from the
    # Hub lives in the standard HF cache instead, so repeated runs and other tools share it.
    WORK = Path(os.environ.get("KINYA_WORK", Path.cwd() / "kinya_bench")).resolve()
    DEEPKIN = WORK / "ac-ai-models" / "DeepKIN-AgAI"   # C4IR's tokenizer AND model code
    RESULTS = WORK / "results"
    GOLDEN_URL = (
        "https://raw.githubusercontent.com/maqamylee0/kinyarwanda_tts_short-words/main/"
        "exploration/notebooks/assets/kinya_flex_tokenizer_golden.json"
    )

    VOICES   = ["female", "female2"]  # add "male"/"male2" to widen the sample
    PER_CELL = 25                     # clips per (voice x length-bucket) cell
    SEED     = 7                      # fixed: the sample is part of the record
    SR       = 24000                  # both the model output rate and the corpus rate
    TORCH_SEED = 1234                 # matches C4IR's own export notebook

    for _d in (WORK, RESULTS):
        _d.mkdir(parents=True, exist_ok=True)

    print(f"work dir: {WORK}")
    print(f"onnx    : {ONNX_REPO}/{ONNX_FILE}")
    print(f"torch   : {TORCH_REPO}/{TORCH_FILE}")
    print(f"dataset : {DATASET_REPO}")
    return (DATASET_REPO, DEEPKIN, GOLDEN_URL, ONNX_FILE, ONNX_REPO, PER_CELL,
            RESULTS, SEED, SR, TORCH_FILE, TORCH_REPO, TORCH_SEED, VOICES, WORK)


# ── Cell 4: hub helper ───────────────────────────────────────────────────────
@app.cell
def _():
    from huggingface_hub import hf_hub_download

    def hub(repo_id, filename, repo_type=None):
        """Download via the HF cache: resumable, etag-checked, shared between runs."""
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)

    return (hub,)


# ── Cell 5: tokenizer section ────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 1. The tokenizer, and why it gets a hard gate

        This model is **not** character-level: its 126 symbols are Kinyarwanda consonant
        clusters up to five characters (`nshyw`, `pfyw`), and text is normalised first —
        numbers spelled out with noun-class concord, punctuation spaced, ASCII-folded.

        Every single letter is *also* in the vocabulary, so a wrong tokenizer does not
        throw. It produces confident, fluent speech saying the wrong thing. The failure is
        silent, and invisible to a non-speaker.

        We use C4IR's own `deepkin` — the same package that supplies the torch model below —
        and **refuse to continue** unless it reproduces all 15 golden vectors from the ONNX
        export.
        """
    )
    return


# ── Cell 6: deepkin + tokenizer gate ─────────────────────────────────────────
@app.cell
def _(DEEPKIN, GOLDEN_URL, json, subprocess, sys):
    import urllib.request

    # The clone destination is DEEPKIN's parent: DEEPKIN itself is the DeepKIN-AgAI
    # subdirectory inside the repo.
    REPO_DIR = DEEPKIN.parent
    if not (DEEPKIN / "deepkin").is_dir():
        if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
            raise RuntimeError(
                f"{REPO_DIR} exists but has no DeepKIN-AgAI/deepkin inside — most likely a "
                f"half-finished clone. Remove it and re-run:\n    rm -rf {REPO_DIR}")
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        print("cloning C4IR's deepkin (tokenizer + model code) ...")
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/c4ir-rw/ac-ai-models.git", str(REPO_DIR)],
                       check=True)
    # Neutralise the eager native-bindings import, as C4IR's export notebook does.
    (DEEPKIN / "deepkin/__init__.py").write_text("")
    if str(DEEPKIN) not in sys.path:
        sys.path.insert(0, str(DEEPKIN))

    from deepkin.data.kinya_norm import norm_text, text_to_sequence, tts_symbols

    def intersperse(lst, item=0):
        """Verbatim from deepkin.modules.tts_commons."""
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result

    def text_to_ids(text):
        return intersperse(text_to_sequence(text, norm=True), 0)

    _gold = DEEPKIN.parent.parent / "kinya_flex_tokenizer_golden.json"
    if not _gold.exists():
        with urllib.request.urlopen(GOLDEN_URL, timeout=60) as _r:
            _gold.write_bytes(_r.read())
    _cases = json.loads(_gold.read_text())["cases"]
    _bad = [c["text"] for c in _cases
            if text_to_sequence(c["text"], norm=True) != c["ids"]
            or intersperse(text_to_sequence(c["text"], norm=True), 0) != c["ids_interspersed"]]
    assert not _bad, f"tokenizer does not match golden vectors: {_bad}"
    print(f"tokenizer OK — {len(_cases)}/{len(_cases)} golden cases, {len(tts_symbols)} symbols")
    print("example:", norm_text("Umuhinzi yaguze ibiro 25 by'ifumbire."))
    return norm_text, text_to_ids, tts_symbols


# ── Cell 7: data section ─────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 2. The corpus

        A seeded stratified sample across utterance lengths and voices — 250 clips, **not**
        the full 18k. Same clips every run, so the sample is part of the record rather than
        an accident. Fetched file by file from the Hub and cached.
        """
    )
    return


# ── Cell 8: sample + audio download ──────────────────────────────────────────
@app.cell
def _(DATASET_REPO, PER_CELL, SEED, ThreadPoolExecutor, VOICES, WORK, hub, random):
    def _bucket(n):
        return 2 if n <= 2 else (3 if n == 3 else (5 if n <= 6 else (10 if n <= 12 else 20)))

    def build_sample():
        tsv = WORK / "sample.tsv"
        if tsv.exists():
            rows = [l.split("\t", 2) for l in tsv.read_text(encoding="utf-8").splitlines() if l]
            print(f"reusing sample.tsv ({len(rows)} rows) — not redrawing")
        else:
            pool = []
            for voice in VOICES:
                man = hub(DATASET_REPO, f"rw_ag_tts_{voice}.tsv", repo_type="dataset")
                for line in open(man, encoding="utf-8"):
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2 and parts[1].strip():
                        pool.append((voice, parts[0], "\t".join(parts[1:]).strip()))
            cells = {}
            for v, i, t in pool:
                cells.setdefault((v, _bucket(len(t.split()))), []).append((v, i, t))
            rng = random.Random(SEED)
            rows = []
            for key in sorted(cells):
                rows += rng.sample(cells[key], min(PER_CELL, len(cells[key])))
            tsv.write_text("\n".join("\t".join(r) for r in rows), encoding="utf-8")
            print(f"drew {len(rows)} clips from {len(pool)} (seed={SEED}, {PER_CELL}/cell)")
        return rows

    sample_rows = build_sample()

    def fetch_one(row):
        voice, cid, _ = row
        try:
            return hub(DATASET_REPO, f"rw_ag_tts_{voice}/{cid}.wav", repo_type="dataset")
        except Exception:
            return None

    print(f"fetching {len(sample_rows)} clips from the Hub (cached after the first run) ...")
    with ThreadPoolExecutor(max_workers=8) as _ex:
        wav_paths = list(_ex.map(fetch_one, sample_rows))
    print(f"  {sum(p is not None for p in wav_paths)}/{len(sample_rows)} present")
    return sample_rows, wav_paths


# ── Cell 9: audio helpers ────────────────────────────────────────────────────
@app.cell
def _(np, wave):
    def read_wav(path):
        with wave.open(str(path)) as f:
            sr, ch, sw, n = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
            raw = f.readframes(n)
        # the corpus wavs are 32-bit; reading them as int16 silently doubles the length
        w = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float64)
        w /= float(2 ** (8 * sw - 1))
        if ch > 1:
            w = w.reshape(-1, ch).mean(axis=1)
        return w, sr

    def speech_seconds(w, sr, rel_db=-32.0):
        """First to last frame within rel_db of the loudest — excludes room tone."""
        hop = max(1, sr // 94)
        if len(w) < hop:
            return 0.0
        fr = np.array([np.sqrt((w[i:i + hop] ** 2).mean()) for i in range(0, len(w) - hop, hop)])
        db = 20 * np.log10(fr + 1e-12)
        live = np.where(db > db.max() + rel_db)[0]
        return (live[-1] - live[0] + 1) * hop / sr if len(live) else 0.0

    return read_wav, speech_seconds


# ── Cell 10: load the clips ──────────────────────────────────────────────────
@app.cell
def _(re, sample_rows, wav_paths):
    clips = []
    for (_v, _i, _t), _p in zip(sample_rows, wav_paths):
        if _p:
            clips.append(dict(voice=_v, cid=_i, text=_t, wav=_p,
                              words=len(_t.split()),
                              has_digits=bool(re.search(r"\d", _t))))

    # ASSUMPTION, undocumented by C4IR: rw_ag_tts_female -> sid 0, female2 -> sid 1.
    # Section 4 plays human and model back to back, so a wrong mapping is audible.
    SID = {"female": 0, "female2": 1, "male": 2, "male2": 2}

    print(f"{len(clips)} clips loaded")
    for _voice in sorted({c['voice'] for c in clips}):
        _sub = [c for c in clips if c['voice'] == _voice]
        print(f"  {_voice:<9}{len(_sub):>4} clips, {sum(c['has_digits'] for c in _sub):>3} with digits")
    return SID, clips


# ── Cell 11: engines section ─────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 3. The two engines

        Both take the same token ids from the same tokenizer, so any difference between
        them is the conversion, not the text.

        `noise_scale=0` throughout: it makes repeated synthesis bit-identical, which is what
        a measurement needs. The duration predictor is deterministic either way.
        """
    )
    return


# ── Cell 12: ONNX engine ─────────────────────────────────────────────────────
@app.cell
def _(ONNX_FILE, ONNX_REPO, hub, np, text_to_ids):
    import onnxruntime as ort

    _onnx_path = hub(ONNX_REPO, ONNX_FILE)
    _sess = ort.InferenceSession(str(_onnx_path), providers=["CPUExecutionProvider"])

    def synth_onnx(text, sid=0, length_scale=1.0, noise_scale=0.0):
        ids = text_to_ids(text)
        y = _sess.run(["y"], {
            "x":            np.array([ids], dtype=np.int64),
            "x_length":     np.array([len(ids)], dtype=np.int64),
            "sid":          np.array([sid], dtype=np.int64),
            "noise_scale":  np.array([noise_scale], dtype=np.float32),
            "length_scale": np.array([length_scale], dtype=np.float32),
        })[0]
        return y.reshape(-1)

    print(f"onnx  ready: {_onnx_path}")
    print(f"  outputs {[o.name for o in _sess.get_outputs()]} — no per-token durations, so "
          f"exact word cropping would need a re-export")
    return (synth_onnx,)


# ── Cell 13: torch engine ────────────────────────────────────────────────────
@app.cell
def _(TORCH_FILE, TORCH_REPO, TORCH_SEED, hub, text_to_ids):
    # C4IR publishes only a training checkpoint (generator + discriminators + optimizers),
    # so this is 1.11 GB and loading takes a moment. from_pretrained strips it for us.
    synth_torch = None
    TORCH_ERROR = None
    try:
        import torch
        from deepkin.models.flex_tts import FlexKinyaTTS

        _ckpt = hub(TORCH_REPO, TORCH_FILE)
        _tts = FlexKinyaTTS.from_pretrained(torch.device("cpu"), _ckpt)
        _tts.eval()
        _net = _tts.flex_tts

        def synth_torch(text, sid=0, length_scale=1.0, noise_scale=0.0):  # noqa: F811
            ids = text_to_ids(text)
            x = torch.LongTensor(ids).unsqueeze(0)
            xl = torch.LongTensor([x.size(1)])
            s = torch.LongTensor([sid])
            torch.manual_seed(TORCH_SEED)
            with torch.no_grad():
                out = _net.infer(x, xl, s, noise_scale=noise_scale,
                                 length_scale=length_scale)
            return out[0][0].cpu().float().numpy().reshape(-1)

        _n = sum(p.numel() for p in _net.parameters()) / 1e6
        print(f"torch ready: {_ckpt}\n  {_n:.1f}M generator params")
    except Exception as exc:                                    # noqa: BLE001
        TORCH_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"torch UNAVAILABLE — {TORCH_ERROR}")
        print("  the bench continues with ONNX only; the export-fidelity check (section 6)")
        print("  and the torch column will be skipped.")
        print("  deps: torch, typed-argument-parser, librosa  (all in the script header)")
    return TORCH_ERROR, synth_torch


# ── Cell 14: engine registry ─────────────────────────────────────────────────
@app.cell
def _(synth_onnx, synth_torch):
    ENGINES = {"onnx": synth_onnx}
    if synth_torch is not None:
        ENGINES["torch"] = synth_torch
    print("engines:", ", ".join(ENGINES))
    return (ENGINES,)


# ── Cell 15: listening section ───────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Listen: human vs each engine

        All clips are peak-normalised, so you are judging **timing and articulation, not
        loudness**. Move the slider and both models re-render.
        """
    )
    return


# ── Cell 16: controls ────────────────────────────────────────────────────────
@app.cell
def _(clips, mo):
    pick = mo.ui.dropdown(
        options={f"[{c['words']}w] {c['voice']}/{c['cid']}  {c['text'][:58]}": n
                 for n, c in enumerate(clips)},
        value=next((f"[{c['words']}w] {c['voice']}/{c['cid']}  {c['text'][:58]}"
                    for c in clips if not c['has_digits'] and 5 <= c['words'] <= 12), None),
        label="sentence",
    )
    speed = mo.ui.slider(0.8, 2.0, 0.05, value=1.0, label="lengthScale (higher = slower)",
                         show_value=True)
    mo.hstack([pick, speed], justify="start", gap=2)
    return pick, speed


# ── Cell 17: the comparison ──────────────────────────────────────────────────
@app.cell
def _(ENGINES, SID, SR, clips, mo, pick, read_wav, speech_seconds, speed):
    _c = clips[pick.value if pick.value is not None else 0]
    _hw, _hsr = read_wav(_c["wav"])
    _h = speech_seconds(_hw, _hsr)

    _blocks = [
        mo.md(f"**{_c['voice']}/{_c['cid']}** — {_c['words']} words"
              + ("  ⚠️ contains digits" if _c["has_digits"] else "")),
        mo.md(f"> {_c['text']}"),
    ]
    _stats = [mo.stat(f"{_h:.2f}s", label="human")]
    _players = [mo.md("**human**"), mo.audio(_hw, rate=_hsr, normalize=True)]
    for _name, _fn in ENGINES.items():
        _w = _fn(_c["text"], sid=SID.get(_c["voice"], 0), length_scale=speed.value)
        _s = speech_seconds(_w, SR)
        _r = _s / _h if _h > 0 else float("nan")
        _stats.append(mo.stat(f"{_s:.2f}s", label=_name,
                              caption=f"ratio {_r:.2f}"
                                      + (" · rushes" if _r < 0.9 else "")))
        _players += [mo.md(f"**{_name}**"), mo.audio(_w, rate=SR, normalize=True)]

    mo.vstack(_blocks + [mo.hstack(_stats, justify="start", gap=2)] + _players)
    return


# ── Cell 18: measurement section ─────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 5. The measurement

        Every clip, every engine, no listening. Rows containing digits are **excluded**:
        `Telefoni: 0784009558` is two words of text that take six seconds to read aloud,
        and it wrecks any per-word statistic.
        """
    )
    return


# ── Cell 19: aggregate ───────────────────────────────────────────────────────
@app.cell
def _(ENGINES, SID, SR, clips, np, read_wav, speech_seconds, speed):
    BUCKETS = [(1, 2, "1-2"), (3, 3, "3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 999, "13+")]

    def measure(length_scale):
        rows = []
        for c in clips:
            if c["has_digits"]:
                continue
            hw, hsr = read_wav(c["wav"])
            h = speech_seconds(hw, hsr)
            if h <= 0:
                continue
            rec = dict(voice=c["voice"], cid=c["cid"], words=c["words"], human_s=h)
            for name, fn in ENGINES.items():
                m = speech_seconds(fn(c["text"], sid=SID.get(c["voice"], 0),
                                      length_scale=length_scale), SR)
                rec[f"{name}_s"] = m
                rec[f"{name}_ratio"] = m / h
            rows.append(rec)
        return rows

    measured = measure(speed.value)

    by_len = []
    print(f"n = {len(measured)} clips at lengthScale {speed.value}\n")
    for _name in ENGINES:
        _r = np.array([x[f"{_name}_ratio"] for x in measured])
        print(f"{_name}: median {np.median(_r):.3f}  mean {_r.mean():.3f}  "
              f"p5 {np.percentile(_r, 5):.3f}  p95 {np.percentile(_r, 95):.3f}  "
              f"faster than human {100 * (_r < 1).mean():.0f}%")
    print(f"\n{'words':<8}{'n':>5}" + "".join(f"{n + ' median':>16}" for n in ENGINES))
    for _lo, _hi, _lbl in BUCKETS:
        _sub = [x for x in measured if _lo <= x["words"] <= _hi]
        if not _sub:
            continue
        _row = dict(bucket=_lbl, n=len(_sub))
        _line = f"{_lbl:<8}{len(_sub):>5}"
        for _name in ENGINES:
            _m = float(np.median([x[f"{_name}_ratio"] for x in _sub]))
            _row[_name] = _m
            _row[f"{_name}_implied_length_scale"] = speed.value / _m
            _line += f"{_m:>16.3f}"
        by_len.append(_row)
        print(_line)
    return BUCKETS, by_len, measured


# ── Cell 20: plots ───────────────────────────────────────────────────────────
@app.cell
def _(ENGINES, measured, np, speed):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.6))
    _colors = {"onnx": "#4878a8", "torch": "#c86a3a"}
    for _name in ENGINES:
        _r = np.array([x[f"{_name}_ratio"] for x in measured])
        _w = np.array([x["words"] for x in measured])
        _c = _colors.get(_name, "#666")
        _ax[0].hist(_r, bins=24, alpha=.55, color=_c, label=_name)
        _ax[1].scatter(_w, _r, s=14, alpha=.55, color=_c, label=_name)
    _ax[0].axvline(1.0, color="#b00", lw=1.5, label="human pace")
    _ax[0].set_xlabel("model duration / human duration")
    _ax[0].set_ylabel("clips")
    _ax[0].legend(fontsize=8)
    _ax[0].set_title(f"Faster than the actress?  (lengthScale {speed.value})", fontsize=10)
    _ax[1].axhline(1.0, color="#b00", lw=1.5)
    _ax[1].set_xscale("log")
    _ax[1].set_xlabel("words in utterance (log)")
    _ax[1].set_ylabel("ratio")
    _ax[1].legend(fontsize=8)
    _ax[1].set_title("Does the gap widen on short utterances?", fontsize=10)
    plt.tight_layout()
    _fig
    return


# ── Cell 21: fidelity section ────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Did our ONNX export change anything?

        The control. Same tokens, same `noise_scale=0`, both engines — so any difference
        here is the conversion, not the model.

        This matters because every claim in the findings is measured on the ONNX. If the
        export altered the duration predictor, the claims would be about our artefact
        rather than about C4IR's model.

        **Note on C4IR's own sample files:** `kinya_flex_export/samples/ref_torch_spk*.wav`
        and `fp32_spk*.wav` cannot be used for this. They render *different sentences* —
        `TEST_SENTENCES[0]` against `TEST_SENTENCES[2]` — so comparing them shows a large
        spurious difference. Only a matched-text run like this one answers the question.
        """
    )
    return


# ── Cell 22: fidelity check ──────────────────────────────────────────────────
@app.cell
def _(BUCKETS, ENGINES, SR, clips, np, speech_seconds, synth_onnx, synth_torch):
    if synth_torch is None:
        print("torch engine unavailable — fidelity check skipped")
        fidelity = []
    else:
        fidelity = []
        # Stratified, NOT the first 12: the rushing lives in the short regime, so a
        # fidelity check that only sees ordinary sentences proves nothing about where
        # the problem actually is.
        _pool = [c for c in clips if not c["has_digits"]]
        _texts = []
        for _lo, _hi, _ in BUCKETS:
            _texts += [c for c in _pool if _lo <= c["words"] <= _hi][:3]
        print(f"{'clip':<16}{'words':>6}{'onnx s':>9}{'torch s':>9}{'delta':>8}"
              f"{'samples eq':>12}{'corr':>9}")
        for _c in _texts:
            _o = synth_onnx(_c["text"], sid=0, noise_scale=0.0)
            _t = synth_torch(_c["text"], sid=0, noise_scale=0.0)
            _n = min(len(_o), len(_t))
            _corr = float(np.corrcoef(_o[:_n], _t[:_n])[0, 1]) if _n > 1 else float("nan")
            _os, _ts = speech_seconds(_o, SR), speech_seconds(_t, SR)
            fidelity.append(dict(cid=_c["cid"], words=_c["words"], onnx_s=_os, torch_s=_ts,
                                 samples_onnx=len(_o), samples_torch=len(_t), corr=_corr))
            print(f"{_c['voice'][:6] + '/' + _c['cid']:<16}{_c['words']:>6}{_os:>9.3f}"
                  f"{_ts:>9.3f}{_os - _ts:>8.3f}{str(len(_o) == len(_t)):>12}{_corr:>9.5f}")
        _same = sum(f["samples_onnx"] == f["samples_torch"] for f in fidelity)
        _mc = float(np.mean([f["corr"] for f in fidelity]))
        print(f"\nidentical sample counts: {_same}/{len(fidelity)}   mean corr {_mc:.5f}")
        print("A faithful export gives identical sample counts and corr ~1.0.")
        print("Identical counts with low corr would mean same timing, different waveform;")
        print("differing counts would mean the export moved the duration predictor.")
    return (fidelity,)


# ── Cell 23: isolated words ──────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Isolated words — the untrained case

        No *human* reference exists: the corpus holds 2 single-word rows out of 17,969, and
        both are the spreadsheet error string `Err:508`. So there is nothing to score the
        models against for correctness here.

        But the engines can still be scored **against each other**, and that is the question
        that matters: if C4IR's own checkpoint rushes isolated words exactly as our export
        does, the behaviour is theirs. If only ours does, it is the conversion's. The two
        tables below answer that; the players are for judging articulation, which no
        duration measurement catches.

        The carrier puts the word **last, before a full stop**, so it picks up phrase-final
        lengthening — unlike `vuga <word> neza`, which buries it mid-phrase where it is
        rushed by design.
        """
    )
    return


# ── Cell 24: word controls ───────────────────────────────────────────────────
@app.cell
def _(mo):
    word = mo.ui.text(value="icunga", label="word")
    wspeed = mo.ui.slider(0.8, 2.4, 0.1, value=1.0, label="lengthScale", show_value=True)
    wsid = mo.ui.dropdown({"Female 1": 0, "Female 2": 1, "Male": 2}, value="Female 1",
                          label="voice")
    mo.hstack([word, wspeed, wsid], justify="start", gap=2)
    return word, wsid, wspeed


# ── Cell 25: isolated word output ────────────────────────────────────────────
@app.cell
def _(ENGINES, SR, mo, np, speech_seconds, word, wsid, wspeed):
    CARRIER = "Iri jambo ni {}."

    _w = word.value.strip() or "icunga"
    _out = [mo.md(f"### “{_w}”")]
    for _name, _fn in ENGINES.items():
        _bare = _fn(_w, sid=wsid.value, length_scale=wspeed.value)
        _car = _fn(CARRIER.format(_w), sid=wsid.value, length_scale=wspeed.value)
        _out += [
            mo.hstack([
                mo.stat(f"{speech_seconds(_bare, SR):.2f}s", label=f"{_name} isolated"),
                mo.stat(f"{np.abs(_bare).max():.3f}", label="raw peak",
                        caption="under 0.10 sounds mumbled"),
            ], justify="start", gap=2),
            mo.md(f"**{_name} — alone**"), mo.audio(_bare, rate=SR, normalize=True),
            mo.md(f"**{_name} — carrier “{CARRIER.format(_w)}”**"),
            mo.audio(_car, rate=SR, normalize=True),
        ]
    mo.vstack(_out)
    return


# ── Cell 26: isolated words, measured, every engine ──────────────────────────
@app.cell
def _(ENGINES, SR, np, re, speech_seconds):
    WORDS = ["icunga", "amazi", "inka", "umwana", "ishuri", "ibirayi", "umuhinzi",
             "ifumbire", "umuneke", "inanasi"]

    def syllables(w):
        """Vowel groups. Kinyarwanda is near-perfectly CV, so this tracks syllables
        closely — but it is a proxy, and only valid on text without digits."""
        return max(1, len(re.findall(r"[aeiouAEIOU]+", w)))

    # Both engines on the same isolated words. If C4IR's checkpoint rushes them too,
    # the behaviour is theirs; if only ours does, it is the export's.
    short_words = []
    peaks_by_word = {}
    print("isolated words — speech seconds, s/syllable, and raw peak (before normalising)")
    print(f"{'word':<11}{'syl':>4}" + "".join(
        f"{n + ' s':>10}{n + ' s/syl':>12}{n + ' peak':>11}" for n in ENGINES))
    for _w in WORDS:
        _syl = syllables(_w)
        _rec = dict(word=_w, syllables=_syl)
        _line = f"{_w:<11}{_syl:>4}"
        for _name, _fn in ENGINES.items():
            _a = _fn(_w, sid=0, length_scale=1.0, noise_scale=0.0)
            _s = speech_seconds(_a, SR)
            _pk = float(np.abs(_a).max())
            _rec[f"{_name}_s"] = _s
            _rec[f"{_name}_s_per_syllable"] = _s / _syl
            _rec[f"{_name}_peak"] = _pk
            _line += f"{_s:>10.3f}{_s / _syl:>12.3f}{_pk:>11.3f}"
        short_words.append(_rec)
        peaks_by_word[_w] = [float(np.abs(ENGINES["onnx"](_w, sid=s)).max())
                             for s in (0, 1, 2)]
        print(_line)

    for _name in ENGINES:
        _m = float(np.mean([r[f"{_name}_s_per_syllable"] for r in short_words]))
        print(f"\n{_name}: mean {_m:.3f} s/syllable on isolated words")
    print("For scale: the actress averages 0.129 s/syllable inside a sentence and")
    print("0.223 on a genuinely isolated word (n=6, a different corpus).")
    return WORDS, peaks_by_word, short_words, syllables


# ── Cell 26b: does the export agree with the original ON SHORT WORDS? ────────
@app.cell
def _(SR, WORDS, np, speech_seconds, syllables, synth_onnx, synth_torch):
    # The fidelity check that matters. Section 6 compares the engines on sentences;
    # this one compares them exactly where the failure is claimed to live.
    if synth_torch is None:
        print("torch engine unavailable — short-word fidelity check skipped")
        short_fidelity = []
    else:
        short_fidelity = []
        print("isolated words: our export vs C4IR's checkpoint, noise_scale=0")
        print(f"{'word':<11}{'onnx s':>9}{'torch s':>9}{'delta':>8}"
              f"{'samples eq':>12}{'corr':>9}")
        for _w in WORDS:
            _o = synth_onnx(_w, sid=0, noise_scale=0.0)
            _t = synth_torch(_w, sid=0, noise_scale=0.0)
            _n = min(len(_o), len(_t))
            _corr = float(np.corrcoef(_o[:_n], _t[:_n])[0, 1]) if _n > 1 else float("nan")
            _os, _ts = speech_seconds(_o, SR), speech_seconds(_t, SR)
            short_fidelity.append(dict(word=_w, syllables=syllables(_w),
                                       onnx_s=_os, torch_s=_ts,
                                       samples_onnx=len(_o), samples_torch=len(_t),
                                       corr=_corr))
            print(f"{_w:<11}{_os:>9.3f}{_ts:>9.3f}{_os - _ts:>8.3f}"
                  f"{str(len(_o) == len(_t)):>12}{_corr:>9.5f}")
        _same = sum(f["samples_onnx"] == f["samples_torch"] for f in short_fidelity)
        print(f"\nidentical sample counts: {_same}/{len(short_fidelity)}")
        if _same == len(short_fidelity):
            print("The export reproduces the original's timing on isolated words exactly,")
            print("so the rushing is C4IR's model, not our conversion.")
        else:
            print("The two disagree on isolated words — the export changes timing in")
            print("exactly the regime under study, and the findings need re-measuring")
            print("against the torch column.")
    return (short_fidelity,)


# ── Cell 27: record the run ──────────────────────────────────────────────────
@app.cell
def _(mo):
    save = mo.ui.run_button(label="save this run to the results folder")
    save
    return (save,)


@app.cell
def _(DATASET_REPO, ENGINES, ONNX_REPO, RESULTS, TORCH_ERROR, TORCH_REPO, by_len,
      datetime, fidelity, json, measured, np, os, peaks_by_word, save, short_fidelity,
      short_words, speed, timezone):
    # BENCH_SAVE=1 records the run without the button, so a headless export is
    # reproducible for the paper.
    if save.value or os.environ.get("BENCH_SAVE"):
        _stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _blob = {
            "run_utc": _stamp,
            "notebook": "exploration/notebooks/01_rushing_bench.py",
            "sources": {"onnx": ONNX_REPO, "torch": TORCH_REPO, "corpus": DATASET_REPO},
            "engines": list(ENGINES),
            "torch_error": TORCH_ERROR,
            "config": {"length_scale": speed.value, "noise_scale": 0.0,
                       "metric": "speech-only duration, -32 dB relative gate",
                       "excluded": "rows containing digits"},
            "summary": {
                name: {
                    "n": len(measured),
                    "median_ratio": float(np.median([x[f"{name}_ratio"] for x in measured])),
                    "mean_ratio": float(np.mean([x[f"{name}_ratio"] for x in measured])),
                    "frac_faster_than_human":
                        float(np.mean([x[f"{name}_ratio"] < 1 for x in measured])),
                } for name in ENGINES
            },
            "by_length": by_len,
            "onnx_vs_torch_fidelity": fidelity,
            "short_word_timing": short_words,
            "onnx_vs_torch_fidelity_short_words": short_fidelity,
            "isolated_word_peaks": peaks_by_word,
            "per_clip": measured,
        }
        _out = RESULTS / f"rushing_{_stamp}_ls{speed.value}.json"
        _out.write_text(json.dumps(_blob, indent=1))
        print(f"wrote {_out}")
    else:
        print("press the button above to record this run")
    return


# ── Cell 28: limits ──────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## What this bench cannot tell you

        - **No ground truth for single words.** The corpus has none, so section 7 is
          subjective. The right `lengthScale` for an isolated word is extrapolated from the
          trend, not measured.
        - **Speaker mapping is assumed.** `female → sid 0`, `female2 → sid 1` is not
          documented by C4IR, which also does not document `female2`/`male2` at all.
        - **Duration ratio is not articulation.** A clip can match the human's length and
          still swallow a syllable. Only listening catches that.
        - **No per-token durations.** The exported ONNX emits only `y`, so exact word
          cropping from a carrier phrase needs a re-export with the alignment exposed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
