# Findings

Dated log. Each entry records the method, the number, and what would overturn it. Methods
and traps are in `METHODOLOGY.md`; raw output in `results/`.

---

## 2026-10-06 — the model is not globally fast; it rushes short utterances

**Method.** Same-text comparison against the actress's own recordings, 190 digit-free clips
of the 250-clip stratified sample, `lengthScale 1.0`, `noise_scale 0.0`, speech-only
durations. Reproduced independently in the Jupyter and marimo notebooks.
Raw: `results/rushing_20261006T073856Z_ls1.0.json`.

**Result.** Model duration ÷ human duration, by utterance length:

| words | n | median ratio | implied `lengthScale` |
|---|---|---|---|
| 1–2 | 33 | 0.875 | 1.14 |
| 3 | 43 | 0.932 | 1.07 |
| 4–6 | 44 | 0.937 | 1.07 |
| 7–12 | 41 | 0.960 | 1.04 |
| 13+ | 29 | 0.986 | 1.01 |

Overall median 0.945; the model is faster than the human on 137/190 clips (72%).

**Reading.** Monotonic. On long sentences the model is within ~1.5% of the actress — it
learned her pacing. The deficit grows as utterances shorten. This is the rushing, and it is
a short-utterance effect, not a global speed error.

**What would overturn it.** A different stratified draw giving a flat profile; or evidence
that the −32 dB speech gate systematically clips the model's quiet onsets more than the
human's, which would inflate the human's measured duration on short clips specifically.

---

## 2026-10-06 — the training corpus contains no isolated words

**Method.** Word-count distribution over all four transcript TSVs of `C4IR-RW/kinya-ag-tts`
(17,969 rows with text).

**Result.** 2 single-word rows (0.01%) — and both are the literal string `Err:508`, a
spreadsheet error value that leaked into `rw_ag_tts_female/370.wav` and
`rw_ag_tts_male/370.wav`. There are **zero genuine single-word recordings**. Mean 10.9
words per utterance; 9.98% of rows are ≤3 words; longest is 118 words.

**Reading.** The duration predictor has never seen an isolated word. Combined with the
entry above, this explains the single-word behaviour as a coverage gap: the model applies
sentence-internal timing because that is the only timing the data taught it.

---

## 2026-10-06 — the corpus audio is not rushed

**Method.** Seconds per syllable (vowel-group proxy) by utterance length, digit-containing
rows excluded, speech-only durations, `female` voice, stratified sample.

**Result.**

| bucket | s/syllable |
|---|---|
| ≤2 words | 0.176 |
| 3 words | 0.161 |
| 4–6 words | 0.146 |
| 7–12 words | 0.132 |
| 13+ words | 0.133 |

Cross-check against an unrelated corpus and speaker
(`DigitalUmuganda/afrispeak_kinyarwanda_female_tts_dataset`): sentence-internal 0.129
s/syllable — within 2% of the C4IR actress. That actress's 6 true single-word recordings
average 0.223 s/syllable, i.e. ~1.7× her own sentence rate.

**Reading.** The C4IR actress speaks at an ordinary rate and *does* lengthen short
utterances (1.33×). The corpus is not the source of the rushing. The hypothesis that it was
is **not supported**.

**Caveat.** The 1.7× isolated-word figure rests on n=6, all interjections (*Yego*,
*Muraho*, *Gute*), from a different speaker and corpus. Treat as an order of magnitude.

**Note on a correction.** A first pass reported 0.560 s/syllable for ≤2-word utterances.
That was wrong: rows like `Telefoni: 0784009558` count as two words and four vowels but
take six seconds to read. See trap 1 in `METHODOLOGY.md`.

---

## 2026-10-06 — corpus composition and data-quality issues

**Method.** Direct inspection of the four TSVs.

- The dataset card documents **2 voice actors and 5,242 + 5,238 clips**. There are in fact
  **four** transcript files — `female` 5,242, `female2` 3,830, `male` 5,238, `male2` 3,659
  — totalling 17,969 rows. `female2` and `male2` are undocumented.
- `female2`'s script is almost entirely a re-read of `female`'s: 3,803 of its 3,820 unique
  texts also appear in `female`. `female` and `male` share 5,223 texts.
- 44 texts are duplicated *within* a single speaker.
- 2 rows are the spreadsheet error `Err:508` (see above).
- 52 rows are ALL-CAPS headings (`GUTORANYA IBIRAYI`), plausibly read in a different style.
- 15.4% of rows contain `(`, 5.7% contain `;` — markup the actor must decide how to
  verbalise.
- ~24% of sampled rows contain digits; phone numbers are read digit by digit.

**Reading.** Recommended cleaning before any retraining: drop the two `Err:508` rows, audit
the number normaliser against how the actress actually reads digits, and decide whether
ALL-CAPS headings belong in the same distribution as prose.

---

## 2026-10-06 — isolated words are quiet as well as rushed

**Method.** Raw peak amplitude (before normalisation) of 8 isolated words across the three
model speakers.

**Result.** Female 1 is far quieter than the other two on every word — e.g. `ifumbire`
0.074 vs 0.373 (Female 2) and 0.502 (Male); `icunga` 0.045. Peaks below ~0.10 sound
mumbled rather than merely quiet.

**Reading.** Two distinct defects stack on isolated words: timing (above) and level. For
single-word use, Female 2 is the better default. This also explains why adding a
neighbouring word *seems* to fix pronunciation — it raises the level without changing
articulation.

---

## 2026-10-06 — unresolved: a long tail where the model over-produces

**Method.** Same-text ratios, 190 clips.

**Result.** Mean ratio 1.286 against a median of 0.945, with p95 at 1.118 — so the mean is
dragged by outliers beyond the 95th percentile, on clips where the model generates *much*
more audio than the human.

**Status.** Not characterised. These may be a separate failure mode (runaway or looping
generation) rather than a timing error. Next step: isolate the clips with ratio > 1.5 and
listen.

---

## 2026-10-06 — infrastructure facts

- The exported ONNX exposes a single output `y`; there are **no per-token durations**.
  Exact word cropping from a carrier phrase requires re-exporting with the alignment as a
  second output. C4IR's export notebook (step 9b) already computes durations internally, so
  the change is bounded.
- C4IR's `deepkin` tokenizer reproduces **15/15** golden vectors from the ONNX export, so
  Python-side tokenization is safe to rely on and no Dart round-trip is needed.
- Number normalisation verified working on a spot check: `ibiro 25` →
  `ibiro makumyabiri na bitanu` (correct noun-class concord).
- `kinyarwanda_female_voice_dataset/` and `female_voice_raw_cache/` in this repo are
  **not** this model's training data — they are `DigitalUmuganda/afrispeak...`, prepared for
  fine-tuning `facebook/mms-tts-kin`. Used here only as an independent cross-check.
