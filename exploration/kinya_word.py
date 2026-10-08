#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["numpy", "torch", "huggingface_hub", "typed-argument-parser", "scipy", "packaging"]
# ///
"""Synthesize a single Kinyarwanda word intelligibly.

Asking kinya-flex-tts for a bare word gives a mispronounced one. Its duration predictor
allocates time unevenly on isolated input: a medial phoneme is squeezed to 21-32 ms (2-3 frames)
while the final vowel is stretched to roughly 2.3x the median. The middle
syllable is swallowed. A native speaker hears `icunga` as something closer to *ihuba*.

`lengthScale` does not fix it. It scales every phoneme uniformly, so the imbalance survives
(final/median 2.86 -> 2.64 going from 1.0 to 1.5); the word gets longer and stays wrong.

What does work is to say the word inside a carrier phrase, where the allocation is even
(final/median 1.15 against 2.31 isolated), and then cut the word back out using the model's
own alignment.

Cropping fixes *which* phonemes get time, but a word cut from a sentence carries sentence
pace: 0.123 s/syllable, against 0.223 for a human saying a word on its own. So the two
knobs are complementary — crop for intelligibility, `length_scale` for isolation pacing.
A Kinyarwanda listener preferred `length_scale=1.5` (0.173 s/syllable) across every word
tested, which is why that is the default; 1.9 (0.213) lands closest to the human rate. The cut is frame-exact, not energy-based: C4IR's checkpoint returns the
attention path from `infer()`, and the per-token frame counts multiply by the 256-sample hop
to give sample boundaries. Verified against the waveform length on every call.

Confirmed by a Kinyarwanda speaker on 2026-10-08: bare renderings are wrong, carrier-cropped
ones are right.

    uv run exploration/kinya_word.py icunga -o icunga.wav
    uv run exploration/kinya_word.py icunga umuneke inka --length-scale 1.5 --out-dir words/

NOTE for on-device use: this needs the alignment, which the exported ONNX does not expose
(it emits only `y`). Shipping this fix in the Flutter app requires re-exporting the ONNX
with the attention as a second output.
"""

from __future__ import annotations

import argparse
import sys
import types
import wave
from pathlib import Path

import numpy as np

SR, HOP = 24000, 256
CARRIER = "Iri jambo ni {}."          # target lands last, before a full stop
# How many interspersed token slots to take before the word. The right number depends on
# the word's first sound, because the frames just before the boundary hold different things
# in the two cases — see `lead_slots_for()`.
LEAD_VOWEL_INITIAL = 2
LEAD_CONSONANT_INITIAL = 0
TORCH_SEED = 1234


def _install_training_only_stubs() -> None:
    """deepkin imports librosa and torchaudio at module level but uses them only in the
    training forward pass and a save helper; infer() touches neither."""
    def stub(name, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    def never(what):
        def f(*a, **k):
            raise RuntimeError(f"{what} was called; it is stubbed because inference "
                               "does not need it.")
        return f

    try:
        import librosa  # noqa: F401
    except ModuleNotFoundError:
        stub("librosa", filters=stub("librosa.filters", mel=never("librosa.filters.mel")))
    try:
        import torchaudio  # noqa: F401
    except ModuleNotFoundError:
        stub("torchaudio",
             transforms=stub("torchaudio.transforms", Vol=never("torchaudio.transforms.Vol")),
             save=never("torchaudio.save"))


class _WordSynthBase:
    """Shared carrier-and-crop logic. Backends supply `_infer`."""

    def _ids(self, text):
        raw = self._to_seq(text, norm=True)
        out = [0] * (len(raw) * 2 + 1)      # intersperse blanks, as the model expects
        out[1::2] = raw
        return out

    def _infer(self, text, sid, length_scale):
        raise NotImplementedError

    def _check_alignment(self, dur, wav):
        if int(round(float(np.sum(dur)))) * HOP != len(wav):
            raise RuntimeError("alignment does not account for the waveform; "
                               "cropping would not be exact")

    def lead_slots_for(self, word: str) -> int:
        """How far back to start the cut, decided by the word's first phoneme.

        The slot before the word is the carrier's *space*, and acoustically it holds
        different things depending on what follows:

        - **vowel-initial** (`avoka`, `icunga`): it holds the word's own vowel onset,
          which coarticulation places ahead of the token boundary. Cutting on the
          boundary shears the attack off and `avoka` is heard as *voka*, so take 2 slots.
        - **consonant-initial** (`pome`): a stop has its own clear onset, nothing bleeds
          backwards, and those frames hold the *tail of the carrier's final vowel*
          instead. Taking them makes `pome` sound like *ipome*, so take none.

        Both confirmed by a Kinyarwanda listener. Most Kinyarwanda nouns carry a vowel
        prefix, which is why the consonant case only showed up on a loanword.
        """
        ids = self._to_seq(word, norm=True)
        for i in ids:
            sym = self._symbols[i]
            if sym and sym.strip():
                return (LEAD_VOWEL_INITIAL if sym[0] in self._vowels
                        else LEAD_CONSONANT_INITIAL)
        return LEAD_VOWEL_INITIAL

    def _word_span(self, syms):
        """Normalisation spaces the punctuation ("iri jambo ni icunga ."), so the target
        word lies between the LAST TWO spaces, not after the last one."""
        spaces = [i for i, s in enumerate(syms) if s == " "]
        if len(spaces) < 2:
            raise RuntimeError("unexpected carrier tokenization")
        return spaces[-2] + 1, spaces[-1]

    def say(self, word: str, sid: int = 0, length_scale: float = 1.5,
            bare: bool = False, lead: int | None = None) -> np.ndarray:
        """Carrier-synthesize `word` and cut it back out. `bare=True` gives the
        unusable direct rendering, for comparison.

        `lead` extends the cut backwards by that many token slots. It is not cosmetic:
        the alignment assigns a token its frames, but the *acoustic* onset of a
        word-initial vowel is realised earlier, in the frames belonging to the preceding
        space. Measured over the carrier, the eight frames before the span boundary carry
        1.8x to 13x the energy of the eight after it. Cutting exactly on the boundary
        therefore clips the vowel attack, and a Kinyarwanda listener hears `avoka` as
        *voka*. Two slots take the blank and the space -- the transition -- without
        reaching the previous word's final vowel, which sits at slot three."""
        if bare:
            return self._infer(word, sid, length_scale)[2]
        if lead is None:
            lead = self.lead_slots_for(word)
        syms, dur, wav = self._infer(CARRIER.format(word), sid, length_scale)
        lo, hi = self._word_span(syms)
        lo = max(0, lo - lead)
        edges = np.concatenate([[0], np.cumsum(dur)]).astype(np.int64) * HOP
        return wav[edges[lo]:edges[hi]]

    def evenness(self, word: str, sid: int = 0, length_scale: float = 1.5,
                 bare: bool = False) -> float:
        """Final-phoneme duration / median phoneme duration. 1.0 is even; the bare
        rendering of a short word typically lands above 2."""
        text = word if bare else CARRIER.format(word)
        syms, dur, _ = self._infer(text, sid, length_scale)
        if not bare:
            lo, hi = self._word_span(syms)
            syms, dur = syms[lo:hi], dur[lo:hi]
        ms = [d for s, d in zip(syms, dur) if s != self._symbols[0] and s.strip()]
        return float(ms[-1] / np.median(ms)) if ms else float("nan")


class KinyaWordOnnx(_WordSynthBase):
    """The shippable path: onnxruntime plus the pure-Python tokenizer, no torch.

    Needs an ONNX exported with the alignment as a second output (`durations`). The
    build that shipped first emits only `y` and will raise here, which is the point —
    silently falling back would hide the thing this class exists to use."""

    def __init__(self, deepkin_dir: Path, onnx_path):
        if str(deepkin_dir) not in sys.path:
            sys.path.insert(0, str(deepkin_dir))
        import onnxruntime as ort
        from deepkin.data.kinya_norm import text_to_sequence, tts_symbols

        self._to_seq = text_to_sequence
        self._symbols = tts_symbols
        self._vowels = set("aeiou")
        self._sess = ort.InferenceSession(str(onnx_path),
                                          providers=["CPUExecutionProvider"])
        names = [o.name for o in self._sess.get_outputs()]
        if "durations" not in names:
            raise RuntimeError(
                f"{onnx_path} exposes {names}; this needs a 'durations' output. "
                "Re-export with export_kinya_flex_tts_colab.ipynb.")

    def _infer(self, text, sid, length_scale):
        ids = self._ids(text)
        y, dur = self._sess.run(["y", "durations"], {
            "x": np.array([ids], dtype=np.int64),
            "x_length": np.array([len(ids)], dtype=np.int64),
            "sid": np.array([sid], dtype=np.int64),
            "noise_scale": np.array([0.0], dtype=np.float32),
            "length_scale": np.array([length_scale], dtype=np.float32),
        })
        wav = y.reshape(-1)
        dur = dur[0]
        self._check_alignment(dur, wav)
        return [self._symbols[i] for i in ids], dur, wav


class KinyaWordSynth(_WordSynthBase):
    """Loads C4IR's checkpoint once; `say()` returns intelligible isolated words."""

    def __init__(self, deepkin_dir: Path, repo="C4IR-RW/kinya-flex-tts",
                 filename="kinya_flex_tts_base_trained.pt"):
        if str(deepkin_dir) not in sys.path:
            sys.path.insert(0, str(deepkin_dir))
        _install_training_only_stubs()

        import torch
        from huggingface_hub import hf_hub_download
        from deepkin.data.kinya_norm import text_to_sequence, tts_symbols
        from deepkin.models.flex_tts import FlexKinyaTTS

        self._torch = torch
        self._to_seq = text_to_sequence
        self._symbols = tts_symbols
        self._vowels = set("aeiou")
        ckpt = hf_hub_download(repo_id=repo, filename=filename)
        tts = FlexKinyaTTS.from_pretrained(torch.device("cpu"), ckpt)
        tts.eval()
        self._net = tts.flex_tts

    def _infer(self, text, sid, length_scale):
        torch = self._torch
        ids = self._ids(text)
        x = torch.LongTensor(ids).unsqueeze(0)
        torch.manual_seed(TORCH_SEED)
        with torch.no_grad():
            o, _, attn, _, _ = self._net.infer(
                x, torch.LongTensor([x.size(1)]), torch.LongTensor([sid]),
                noise_scale=0.0, length_scale=length_scale)
        a = attn.squeeze(0).squeeze(0)
        dur = (a.sum(dim=0) if a.shape[1] == len(ids) else a.sum(dim=1)).cpu().numpy()
        wav = o[0][0].cpu().float().numpy().reshape(-1)
        self._check_alignment(dur, wav)
        return [self._symbols[i] for i in ids], dur, wav


def write_wav(path: Path, w: np.ndarray, peak: float = 0.85) -> None:
    scaled = w / max(1e-9, float(np.abs(w).max())) * peak
    pcm = (np.clip(scaled, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("words", nargs="+")
    ap.add_argument("-o", "--out", type=Path, help="output file (single word only)")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--sid", type=int, default=0,
                    help="0=Female 1 (default; the voice the listening tests used), "
                         "1=Female 2, 2=Male. Output is peak-normalised, so Female 1 "
                         "being quieter in raw terms does not matter for written files.")
    ap.add_argument("--length-scale", type=float, default=1.5,
                    help="1.5 by default: cropping alone restores the word to sentence "
                         "pace (0.123 s/syllable), and a Kinyarwanda listener preferred "
                         "1.5 (0.173) across every word tested. 1.9 (0.213) matches the "
                         "measured human isolated-word rate of 0.223 most closely.")
    ap.add_argument("--also-bare", action="store_true",
                    help="write the broken direct rendering alongside, for comparison")
    ap.add_argument("--onnx", type=Path,
                    help="use an ONNX export with a 'durations' output instead of the "
                         "1.11 GB torch checkpoint. This is the path that can ship.")
    ap.add_argument("--deepkin", type=Path,
                    default=Path(__file__).resolve().parent.parent / "vendor/ac-ai-models/DeepKIN-AgAI")
    a = ap.parse_args()

    if not (a.deepkin / "deepkin").is_dir():
        print(f"deepkin not found at {a.deepkin}\n"
              "  git clone --depth 1 https://github.com/c4ir-rw/ac-ai-models.git "
              f"{a.deepkin.parent}", file=sys.stderr)
        return 1

    synth = (KinyaWordOnnx(a.deepkin, a.onnx) if a.onnx
             else KinyaWordSynth(a.deepkin))
    print(f"backend: {'onnx ' + str(a.onnx) if a.onnx else 'torch checkpoint'}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for word in a.words:
        w = synth.say(word, sid=a.sid, length_scale=a.length_scale)
        dest = a.out if (a.out and len(a.words) == 1) else a.out_dir / f"{word}.wav"
        write_wav(dest, w)
        print(f"{word:<12} {len(w)/SR:.2f}s  evenness {synth.evenness(word, a.sid, a.length_scale):.2f}"
              f"  -> {dest}")
        if a.also_bare:
            b = synth.say(word, sid=a.sid, length_scale=a.length_scale, bare=True)
            bd = dest.with_name(dest.stem + "_bare.wav")
            write_wav(bd, b)
            print(f"{'':<12} {len(b)/SR:.2f}s  evenness "
                  f"{synth.evenness(word, a.sid, a.length_scale, bare=True):.2f}  -> {bd}  (broken)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
