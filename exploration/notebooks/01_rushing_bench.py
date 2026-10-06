# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#     "marimo",
#     "numpy",
#     "onnxruntime",
#     "matplotlib",
# ]
# ///
"""Rushing bench for C4IR-RW/kinya-flex-tts.

Compares the model against the audio it was trained on, sentence by sentence.
Because each comparison uses the *same text* as a real recording by the actress,
the measurement needs no syllable counting: if the model's clip is shorter than
hers, it is rushing, by exactly that ratio.

Downloads its own stratified sample of C4IR-RW/kinya-ag-tts (~250 clips, not the
full 18k) on first run.

    uv run exploration/notebooks/01_rushing_bench.py      # app mode
    marimo edit exploration/notebooks/01_rushing_bench.py # notebook mode
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

        An A/B bench for `C4IR-RW/kinya-flex-tts` against the corpus it was trained on,
        [`C4IR-RW/kinya-ag-tts`](https://huggingface.co/datasets/C4IR-RW/kinya-ag-tts)
        (CC-BY-4.0, C4IR Rwanda & KiNLP).

        For each sentence you get **the actress's real recording and the model's rendition
        of the same text**. Identical text means the comparison needs no syllable counting
        and no speaking-rate proxy — if the model's clip is shorter, it is rushing, by
        exactly that ratio.

        Everything runs locally. Results are written to `exploration/results/`.
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
    import urllib.request
    import warnings
    import wave
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np

    warnings.filterwarnings("ignore", category=SyntaxWarning)
    return (
        Path, ThreadPoolExecutor, datetime, json, np, os, random, re,
        subprocess, sys, timezone, urllib, wave,
    )


# ── Cell 3: configuration ────────────────────────────────────────────────────
@app.cell
def _(Path, mo, os, urllib):
    # Two supported layouts, tried in this order:
    #
    #   BUNDLE  <folder>/assets/kinya_flex_tts.onnx     — self-contained, upload the folder
    #   REPO    <root>/kinyarwanda_tts_app/...          — running inside the project
    #
    # __file__ is deliberately NOT trusted first: marimo may execute cells from a temp
    # path (/tmp/marimo_*/__marimo__cell_*.py), so __file__ points nowhere useful and
    # anything derived from it lands at the filesystem root.
    BUNDLE_MARK = "assets/kinya_flex_tts.onnx"
    REPO_MARKS  = ("kinya_flex_export", "kinyarwanda_tts_app")

    def _seeds():
        for var in ("KINYA_BENCH", "KINYA_ROOT"):
            if os.environ.get(var):
                yield Path(os.environ[var])
        # marimo's own answer to "where is this notebook?" — correct even when cells are
        # executed from a temp file, which is exactly when __file__ is useless.
        nbdir = mo.notebook_dir()
        if nbdir is not None:
            yield nbdir
            yield from Path(nbdir).parents
        cwd = Path.cwd()
        yield cwd
        yield from cwd.parents
        # one level down, so running from the folder that *contains* the bundle works
        try:
            yield from (d for d in sorted(cwd.iterdir()) if d.is_dir())
        except OSError:
            pass
        try:
            here = Path(__file__).resolve().parent
            yield here
            yield from here.parents
        except NameError:
            pass

    def _exists(path):
        # Probing unreadable directories (e.g. /tmp/snap-private-tmp) raises rather than
        # returning False, and one of those must not abort the whole search.
        try:
            return path.exists()
        except OSError:
            return False

    def _locate():
        for seed in _seeds():
            try:
                seed = seed.resolve()
            except OSError:
                continue
            if _exists(seed / BUNDLE_MARK):
                return "bundle", seed
            if all(_exists(seed / m) for m in REPO_MARKS):
                return "repo", seed
        # Nothing local: fall back to fetching from GitHub into a cache beside the cwd.
        return "github", Path(os.environ.get("KINYA_CACHE", Path.cwd() / "kinya_bench_cache"))

    LAYOUT, ROOT = _locate()

    if LAYOUT in ("bundle", "github"):
        DATA    = ROOT / "data"
        RESULTS = ROOT / "results"
        MODEL   = ROOT / "assets/kinya_flex_tts.onnx"
        DEEPKIN = ROOT / "assets"          # holds deepkin/ ; goes on sys.path
        GOLDEN  = ROOT / "assets/kinya_flex_tokenizer_golden.json"
    else:
        DATA    = ROOT / "kinya_ag_sample"
        RESULTS = ROOT / "exploration" / "results"
        MODEL   = ROOT / "kinyarwanda_tts_app/assets/models/kinya_flex_tts.onnx"
        DEEPKIN = ROOT / "vendor/ac-ai-models/DeepKIN-AgAI"
        GOLDEN  = ROOT / "kinya_flex_export/kinya_flex_tokenizer_golden.json"

    print(f"layout: {LAYOUT}\nroot:   {ROOT}")

    HF   = "https://huggingface.co/datasets/C4IR-RW/kinya-ag-tts/resolve/main"
    VOICES = ["female", "female2"]   # add "male"/"male2" to widen the sample
    PER_CELL = 25                    # clips per (voice x length-bucket) cell
    SEED = 7                         # fixed: the sample is part of the record

    SR = 24000    # model output rate; the corpus is also 24 kHz
    HOP = 256     # samples per model frame

    # GitHub is the store for everything small: the tokenizer, the golden vectors and the
    # sample manifest. The corpus audio comes from Hugging Face (cell 5). The model does
    # NOT live on GitHub — at 136 MB it exceeds the 100 MB per-file limit — so it is found
    # locally or pointed at with KINYA_MODEL.
    GH_RAW = os.environ.get(
        "KINYA_GH_RAW",
        "https://raw.githubusercontent.com/maqamylee0/kinyarwanda_tts_short-words/main",
    )
    _FROM_GITHUB = {
        GOLDEN:                         "exploration/notebooks/assets/kinya_flex_tokenizer_golden.json",
        DEEPKIN / "deepkin/__init__.py":            "exploration/notebooks/assets/deepkin/__init__.py",
        DEEPKIN / "deepkin/data/__init__.py":       "exploration/notebooks/assets/deepkin/data/__init__.py",
        DEEPKIN / "deepkin/data/kinya_norm.py":     "exploration/notebooks/assets/deepkin/data/kinya_norm.py",
        DEEPKIN / "deepkin/data/kinyarwanda.py":    "exploration/notebooks/assets/deepkin/data/kinyarwanda.py",
        DEEPKIN / "deepkin/data/kinya_number_speller.py": "exploration/notebooks/assets/deepkin/data/kinya_number_speller.py",
        DATA / "sample.tsv":            "exploration/notebooks/data/sample.tsv",
    }

    def ensure_from_github(dest, relpath):
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{GH_RAW}/{relpath}"
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"  fetched {relpath}")
        return dest

    if LAYOUT == "github":
        print("no local copy found — fetching from GitHub")
        for _dest, _rel in _FROM_GITHUB.items():
            ensure_from_github(_dest, _rel)

    # The model is the one asset GitHub cannot store (136 MB vs a 100 MB per-file limit),
    # so it lives on the Hugging Face Hub instead — the natural home for weights, and where
    # the upstream checkpoint already is. Resolution: local copy, then KINYA_MODEL, then
    # download from the Hub into the cache.
    HF_MODEL_REPO = os.environ.get("KINYA_HF_MODEL_REPO", "maqamylee0/kinya-flex-tts-onnx")
    MODEL_URL = os.environ.get(
        "KINYA_MODEL_URL",
        f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/kinya_flex_tts.onnx",
    )

    if not MODEL.exists() and os.environ.get("KINYA_MODEL"):
        MODEL = Path(os.environ["KINYA_MODEL"]).expanduser().resolve()

    if not MODEL.exists():
        print(f"model not local — downloading from {MODEL_URL}")
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL.with_suffix(".onnx.part")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=120) as r:
                total = int(r.headers.get("Content-Length") or 0)
                done = step = 0
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        # every ~25 MB, so a captured log stays readable
                        if done // (25 << 20) > step:
                            step = done // (25 << 20)
                            print(f"  {done / 1e6:.0f}"
                                  + (f" / {total / 1e6:.0f}" if total else "") + " MB")
            print(f"  {done / 1e6:.0f} MB done")
            tmp.replace(MODEL)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not fetch the model from {MODEL_URL}\n  ({exc})\n"
                "The model is 136 MB, over GitHub's 100 MB per-file limit, so it is not in "
                "the git repo.\nEither:\n"
                "  - point at a local copy:  KINYA_MODEL=/path/to/kinya_flex_tts.onnx\n"
                "  - upload it once:         python exploration/upload_model_to_hf.py\n"
                "  - or set KINYA_HF_MODEL_REPO / KINYA_MODEL_URL to where it actually lives"
            ) from exc
    if not GOLDEN.exists():
        raise RuntimeError(f"Missing golden vectors: {GOLDEN}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "wav").mkdir(exist_ok=True)
    return (DATA, DEEPKIN, GOLDEN, GH_RAW, HF, HF_MODEL_REPO, LAYOUT, MODEL, MODEL_URL,
            PER_CELL, RESULTS, ROOT, SEED, SR, VOICES, ensure_from_github)


# ── Cell 4: dataset section header ───────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 1. The data

        A stratified sample across utterance lengths and voices — **not** the full 18k-clip
        corpus. The draw is seeded, so the same clips come back every run and the sample is
        part of the record rather than an accident.
        """
    )
    return


# ── Cell 5: download the stratified sample ───────────────────────────────────
@app.cell
def _(DATA, HF, PER_CELL, SEED, ThreadPoolExecutor, VOICES, random, urllib):
    def _bucket(n):
        return 2 if n <= 2 else (3 if n == 3 else (5 if n <= 6 else (10 if n <= 12 else 20)))

    def _get(url, dest, tries=3):
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    dest.write_bytes(r.read())
                return dest.stat().st_size > 1000
            except Exception:
                if attempt == tries - 1:
                    return False
        return False

    def build_sample():
        """Draw the sample if absent; reuse it verbatim if already on disk."""
        tsv = DATA / "sample.tsv"
        if tsv.exists():
            rows = [l.split("\t", 2) for l in tsv.read_text(encoding="utf-8").splitlines() if l]
            print(f"reusing existing sample.tsv ({len(rows)} rows) — not redrawing")
        else:
            pool = []
            for voice in VOICES:
                man = DATA / f"rw_ag_tts_{voice}.tsv"
                if not man.exists():
                    _get(f"{HF}/rw_ag_tts_{voice}.tsv", man)
                for line in man.read_text(encoding="utf-8").splitlines():
                    parts = line.split("\t")
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

        missing = [(v, i) for v, i, _ in rows if not (DATA / "wav" / f"{v}_{i}.wav").exists()]
        if missing:
            print(f"downloading {len(missing)} wavs ...")
            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(
                    lambda vi: _get(f"{HF}/rw_ag_tts_{vi[0]}/{vi[1]}.wav",
                                    DATA / "wav" / f"{vi[0]}_{vi[1]}.wav"),
                    missing))
        have = sum((DATA / "wav" / f"{v}_{i}.wav").exists() for v, i, _ in rows)
        print(f"{have}/{len(rows)} clips present in {DATA}")
        return rows

    sample_rows = build_sample()
    return (sample_rows,)


# ── Cell 6: tokenizer note ───────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 2. The tokenizer, and why it gets a hard gate

        This model is **not** character-level: its 126 symbols are Kinyarwanda consonant
        clusters up to five characters (`nshyw`, `pfyw`), and text is normalised first —
        numbers spelled out with noun-class concord, punctuation spaced, ASCII-folded.

        Every single letter is *also* in the vocabulary, so a wrong tokenizer does not
        throw. It produces confident, fluent speech saying the wrong thing. The failure is
        silent, and invisible to a non-speaker.

        So we use C4IR's own `deepkin` tokenizer, and **refuse to continue** unless it
        reproduces all 15 golden vectors from the ONNX export.
        """
    )
    return


# ── Cell 7: tokenizer + golden-vector gate ───────────────────────────────────
@app.cell
def _(DEEPKIN, GOLDEN, LAYOUT, json, subprocess, sys):
    # DEEPKIN is <root>/vendor/ac-ai-models/DeepKIN-AgAI, so the clone destination is its
    # PARENT (the repo dir). Cloning into DEEPKIN.parent/"ac-ai-models" nests it one level
    # too deep and the tokenizer is then never found.
    REPO = DEEPKIN.parent
    if not (DEEPKIN / "deepkin").is_dir():
        if LAYOUT in ("bundle", "github"):
            raise RuntimeError(f"tokenizer missing: {DEEPKIN / 'deepkin'}")
        if REPO.exists() and any(REPO.iterdir()):
            raise RuntimeError(
                f"{REPO} exists but has no DeepKIN-AgAI/deepkin inside — most likely a "
                f"half-finished clone. Remove it and re-run:\n    rm -rf {REPO}")
        REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1",
             "https://github.com/c4ir-rw/ac-ai-models.git", str(REPO)], check=True)
        print(f"cloned the tokenizer to {REPO}")
    if not (DEEPKIN / "deepkin").is_dir():
        raise RuntimeError(f"tokenizer still missing at {DEEPKIN} after clone")

    # Neutralise the eager native-bindings import, as C4IR's export notebook does.
    # Idempotent, so it is safe to repeat on an existing checkout.
    (DEEPKIN / "deepkin/__init__.py").write_text("")

    if str(DEEPKIN) not in sys.path:
        sys.path.insert(0, str(DEEPKIN))

    from deepkin.data.kinya_norm import norm_text, text_to_sequence, tts_symbols

    def intersperse(lst, item=0):
        """Verbatim from deepkin.modules.tts_commons (inlined only to avoid a torch import)."""
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result

    def text_to_ids(text):
        return intersperse(text_to_sequence(text, norm=True), 0)

    _cases = json.loads(GOLDEN.read_text())["cases"]
    _bad = [c["text"] for c in _cases
            if text_to_sequence(c["text"], norm=True) != c["ids"]
            or intersperse(text_to_sequence(c["text"], norm=True), 0) != c["ids_interspersed"]]
    assert not _bad, f"tokenizer does not match golden vectors: {_bad}"
    print(f"tokenizer OK — {len(_cases)}/{len(_cases)} golden cases, {len(tts_symbols)} symbols")
    print("example:", norm_text("Umuhinzi yaguze ibiro 25 by'ifumbire."))
    return norm_text, text_to_ids, tts_symbols


# ── Cell 8: the model ────────────────────────────────────────────────────────
@app.cell
def _(MODEL, np, text_to_ids):
    import onnxruntime as ort

    _sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])

    def synth(text, sid=0, length_scale=1.0, noise_scale=0.0):
        """float32 PCM @ 24 kHz. noise_scale=0 makes repeat calls bit-identical."""
        ids = text_to_ids(text)
        y = _sess.run(["y"], {
            "x":            np.array([ids], dtype=np.int64),
            "x_length":     np.array([len(ids)], dtype=np.int64),
            "sid":          np.array([sid], dtype=np.int64),
            "noise_scale":  np.array([noise_scale], dtype=np.float32),
            "length_scale": np.array([length_scale], dtype=np.float32),
        })[0]
        return y.reshape(-1)

    print("outputs:", [o.name for o in _sess.get_outputs()],
          "— no per-token durations exposed; exact word cropping needs a re-export")
    return (synth,)


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


# ── Cell 10: load the sample ─────────────────────────────────────────────────
@app.cell
def _(DATA, re, sample_rows):
    clips = []
    for _v, _i, _t in sample_rows:
        _p = DATA / "wav" / f"{_v}_{_i}.wav"
        if _p.exists() and _p.stat().st_size > 1000:
            clips.append(dict(voice=_v, cid=_i, text=_t, wav=_p,
                              words=len(_t.split()),
                              has_digits=bool(re.search(r"\d", _t))))

    # rw_ag_tts_female -> sid 0, female2 -> sid 1. ASSUMPTION: C4IR does not document
    # which corpus voice is which model speaker. Section 3 lets you hear it; flip if wrong.
    SID = {"female": 0, "female2": 1, "male": 2, "male2": 2}

    print(f"{len(clips)} clips loaded")
    for _voice in sorted({c['voice'] for c in clips}):
        _sub = [c for c in clips if c['voice'] == _voice]
        print(f"  {_voice:<9}{len(_sub):>4} clips, {sum(c['has_digits'] for c in _sub):>3} with digits")
    return SID, clips


# ── Cell 11: listening lab header ────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Listen: human vs model

        Pick a sentence and a speed. Both clips are peak-normalised, so you are judging
        **timing and articulation, not loudness**. Move the slider and the model re-renders
        immediately.
        """
    )
    return


# ── Cell 12: controls ────────────────────────────────────────────────────────
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


# ── Cell 13: the comparison ──────────────────────────────────────────────────
@app.cell
def _(SID, clips, mo, pick, read_wav, speech_seconds, speed, synth, SR):
    _c = clips[pick.value if pick.value is not None else 0]
    _hw, _hsr = read_wav(_c["wav"])
    _mw = synth(_c["text"], sid=SID.get(_c["voice"], 0), length_scale=speed.value)
    _h, _m = speech_seconds(_hw, _hsr), speech_seconds(_mw, SR)
    _ratio = _m / _h if _h > 0 else float("nan")

    mo.vstack([
        mo.md(f"**{_c['voice']}/{_c['cid']}** — {_c['words']} words"
              + ("  ⚠️ contains digits" if _c["has_digits"] else "")),
        mo.md(f"> {_c['text']}"),
        mo.hstack([
            mo.stat(f"{_h:.2f}s", label="human"),
            mo.stat(f"{_m:.2f}s", label=f"model @ {speed.value}"),
            mo.stat(f"{_ratio:.2f}", label="ratio",
                    caption="model rushes" if _ratio < 0.9 else
                            ("model drags" if _ratio > 1.1 else "matched")),
        ], justify="start", gap=2),
        mo.md("**human**"), mo.audio(_hw, rate=_hsr, normalize=True),
        mo.md("**model**"), mo.audio(_mw, rate=SR, normalize=True),
    ])
    return


# ── Cell 14: measurement header ──────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 4. The measurement

        Every clip, no listening. Rows containing digits are **excluded**: `Telefoni:
        0784009558` is two words of text that take six seconds to read aloud, and it
        wrecks any per-word or per-syllable statistic.
        """
    )
    return


# ── Cell 15: aggregate ───────────────────────────────────────────────────────
@app.cell
def _(SID, clips, np, read_wav, speech_seconds, speed, synth, SR):
    def measure(length_scale):
        out = []
        for c in clips:
            if c["has_digits"]:
                continue
            hw, hsr = read_wav(c["wav"])
            mw = synth(c["text"], sid=SID.get(c["voice"], 0), length_scale=length_scale)
            h, m = speech_seconds(hw, hsr), speech_seconds(mw, SR)
            if h > 0:
                out.append(dict(voice=c["voice"], cid=c["cid"], words=c["words"],
                                human_s=h, model_s=m, ratio=m / h))
        return out

    measured = measure(speed.value)
    _r = np.array([x["ratio"] for x in measured])

    BUCKETS = [(1, 2, "1-2"), (3, 3, "3"), (4, 6, "4-6"), (7, 12, "7-12"), (13, 999, "13+")]
    by_len = []
    for _lo, _hi, _lbl in BUCKETS:
        _v = [x["ratio"] for x in measured if _lo <= x["words"] <= _hi]
        if _v:
            by_len.append(dict(bucket=_lbl, n=len(_v), median=float(np.median(_v)),
                               implied_length_scale=float(speed.value / np.median(_v))))

    print(f"n = {len(_r)} clips at lengthScale {speed.value}")
    print(f"  median ratio {np.median(_r):.3f}   mean {_r.mean():.3f}   "
          f"p5 {np.percentile(_r, 5):.3f}  p95 {np.percentile(_r, 95):.3f}")
    print(f"  model faster than human: {(_r < 1).sum()}/{len(_r)} ({100 * (_r < 1).mean():.0f}%)")
    print(f"\n{'words':<8}{'n':>5}{'median ratio':>15}{'implied lengthScale':>22}")
    for _b in by_len:
        print(f"{_b['bucket']:<8}{_b['n']:>5}{_b['median']:>15.3f}{_b['implied_length_scale']:>22.2f}")
    return by_len, measured


# ── Cell 16: plots ───────────────────────────────────────────────────────────
@app.cell
def _(measured, np, speed):
    import matplotlib.pyplot as plt

    _r = np.array([x["ratio"] for x in measured])
    _w = np.array([x["words"] for x in measured])
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 3.6))
    _ax[0].hist(_r, bins=24, color="#4878a8", edgecolor="white")
    _ax[0].axvline(1.0, color="#b00", lw=1.5, label="human pace")
    _ax[0].axvline(np.median(_r), color="#222", ls="--", lw=1.2,
                   label=f"median {np.median(_r):.2f}")
    _ax[0].set_xlabel("model duration / human duration")
    _ax[0].set_ylabel("clips")
    _ax[0].legend(fontsize=8)
    _ax[0].set_title(f"Faster than the actress?  (lengthScale {speed.value})", fontsize=10)

    _ax[1].scatter(_w, _r, s=14, alpha=.6, color="#4878a8")
    _ax[1].axhline(1.0, color="#b00", lw=1.5)
    _ax[1].set_xscale("log")
    _ax[1].set_xlabel("words in utterance (log)")
    _ax[1].set_ylabel("ratio")
    _ax[1].set_title("Does the gap widen on short utterances?", fontsize=10)
    plt.tight_layout()
    _fig
    return


# ── Cell 17: reading the result ──────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        **How to read the right-hand plot.** If the ratio sits near 1.0 on long sentences
        and falls as utterances shorten, the model learned this actress's sentence pacing
        and only mishandles short input — which is what the corpus predicts, since it holds
        2 single-word rows out of 17,969 and both are the spreadsheet error `Err:508`.

        If the ratio were uniformly below 1.0, the model would be globally fast and one
        `lengthScale` would fix it everywhere.
        """
    )
    return


# ── Cell 18: isolated words header ───────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Isolated words — the untrained case

        No human reference exists here: the corpus contains no genuine single-word
        recordings, so this section **cannot be scored**, only listened to.

        The carrier puts the word **last, before a full stop**, so it picks up phrase-final
        lengthening — unlike `vuga <word> neza`, which buries it mid-phrase where it is
        rushed by design.
        """
    )
    return


# ── Cell 19: word controls ───────────────────────────────────────────────────
@app.cell
def _(mo):
    word = mo.ui.text(value="icunga", label="word")
    wspeed = mo.ui.slider(0.8, 2.4, 0.1, value=1.0, label="lengthScale", show_value=True)
    wsid = mo.ui.dropdown({"Female 1": 0, "Female 2": 1, "Male": 2}, value="Female 1",
                          label="voice")
    mo.hstack([word, wspeed, wsid], justify="start", gap=2)
    return word, wsid, wspeed


# ── Cell 20: isolated word output ────────────────────────────────────────────
@app.cell
def _(mo, np, speech_seconds, synth, word, wsid, wspeed, SR):
    CARRIER = "Iri jambo ni {}."

    _w = word.value.strip() or "icunga"
    _bare = synth(_w, sid=wsid.value, length_scale=wspeed.value)
    _car = synth(CARRIER.format(_w), sid=wsid.value, length_scale=wspeed.value)

    mo.vstack([
        mo.hstack([
            mo.stat(f"{speech_seconds(_bare, SR):.2f}s", label="isolated"),
            mo.stat(f"{np.abs(_bare).max():.3f}", label="raw peak",
                    caption="under 0.10 sounds mumbled"),
        ], justify="start", gap=2),
        mo.md(f"**“{_w}” alone**"), mo.audio(_bare, rate=SR, normalize=True),
        mo.md(f"**in carrier — “{CARRIER.format(_w)}”**"), mo.audio(_car, rate=SR, normalize=True),
    ])
    return


# ── Cell 21: level across voices ─────────────────────────────────────────────
@app.cell
def _(np, synth):
    WORDS = ["icunga", "amazi", "inka", "umwana", "ishuri", "ibirayi", "umuhinzi", "ifumbire"]

    print("raw peak on isolated words (before normalisation)")
    print(f"{'word':<12}{'Female 1':>10}{'Female 2':>10}{'Male':>8}")
    peaks_by_word = {}
    for _w in WORDS:
        _p = [float(np.abs(synth(_w, sid=s)).max()) for s in (0, 1, 2)]
        peaks_by_word[_w] = _p
        print(f"{_w:<12}{_p[0]:>10.3f}{_p[1]:>10.3f}{_p[2]:>8.3f}"
              f"{'   <- very quiet' if _p[0] < 0.10 else ''}")
    return WORDS, peaks_by_word


# ── Cell 22: record the run ──────────────────────────────────────────────────
@app.cell
def _(mo):
    save = mo.ui.run_button(label="save this run to exploration/results/")
    save
    return (save,)


@app.cell
def _(RESULTS, by_len, datetime, json, measured, np, os, peaks_by_word, save, speed, timezone):
    # BENCH_SAVE=1 records the run without the button, so a headless
    # `marimo export html` is reproducible for the paper.
    if save.value or os.environ.get("BENCH_SAVE"):
        _r = np.array([x["ratio"] for x in measured])
        _stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _blob = {
            "run_utc": _stamp,
            "notebook": "exploration/notebooks/01_rushing_bench.py",
            "model": "C4IR-RW/kinya-flex-tts (ONNX fp32)",
            "corpus": "C4IR-RW/kinya-ag-tts",
            "config": {"length_scale": speed.value, "noise_scale": 0.0,
                       "metric": "speech-only duration, -32 dB relative gate",
                       "excluded": "rows containing digits"},
            "summary": {"n": len(_r), "median_ratio": float(np.median(_r)),
                        "mean_ratio": float(_r.mean()),
                        "p5": float(np.percentile(_r, 5)),
                        "p95": float(np.percentile(_r, 95)),
                        "frac_model_faster": float((_r < 1).mean())},
            "by_length": by_len,
            "isolated_word_peaks": peaks_by_word,
            "per_clip": measured,
        }
        _out = RESULTS / f"rushing_{_stamp}_ls{speed.value}.json"
        _out.write_text(json.dumps(_blob, indent=1))
        print(f"wrote {_out}")
    else:
        print("press the button above to record this run")
    return


# ── Cell 23: limits ──────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md(
        """
        ## What this bench cannot tell you

        - **No ground truth for single words.** The corpus has none, so section 5 is
          subjective. The right `lengthScale` for an isolated word is extrapolated from the
          trend, not measured.
        - **Speaker mapping is assumed.** `female → sid 0`, `female2 → sid 1` is not
          documented by C4IR.
        - **Duration ratio is not articulation.** A clip can match the human's length and
          still swallow a syllable. Only listening catches that.
        - **No per-token durations.** The exported ONNX emits only `y`, so exact word
          cropping from a carrier phrase needs a re-export with the alignment exposed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
