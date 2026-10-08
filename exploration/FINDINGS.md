# Findings

Dated log. Each entry records the method, the number, and what would overturn it. Methods
and traps are in `METHODOLOGY.md`; raw output in `results/`.

---

## 2026-10-08 — the swallowed syllables: a distorted duration allocation, not a uniform rush

**Prompted by** a native-speaker report that the model does not merely rush `icunga` but
says something closer to *ihuba* — swallowed syllables, wrong consonants.

**Method.** C4IR's checkpoint exposes the alignment (`infer()` returns `attn`), so
per-phoneme durations are directly measurable. Validated first: predicted frames account
for the waveform exactly (frames x 256 = samples). `noise_scale=0`, speaker 0.

**Result 1 — the tokenizer is not at fault.** `icunga` tokenizes to `i c u ng a`, identical
alone and inside a sentence. The model receives the correct symbols and renders them wrongly.

**Result 2 — the allocation is lopsided, and that is the mechanism.** Milliseconds per
phoneme for `icunga`:

| phoneme | alone | in a sentence | alone @ lengthScale 1.8 |
|---|---|---|---|
| i | 74.7 | — | 138.7 |
| c | 74.7 | 42.7 | 138.7 |
| **u** | **32.0** | 42.7 | 64.0 |
| ng | 53.3 | 42.7 | 96.0 |
| **a** | **213.3** | 42.7 | 373.3 |

In a sentence every phoneme gets ~43 ms. Alone, the medial vowel /u/ is crushed to 32 ms —
the 3-frame floor — while the final /a/ is stretched to 213 ms. A 32 ms vowel between a
75 ms /c/ and a 53 ms /ng/ is perceptually swallowed, which is what the listener reports.

**This is general, not specific to `icunga`.** Final-phoneme duration divided by the median
phoneme duration, across 10 isolated words:

| condition | mean final/median | range |
|---|---|---|
| isolated | **2.31** | 1.00 – 3.20 |
| same word, last in a carrier phrase | **1.15** | 1.00 – 2.00 |

Nine of ten isolated words hit the 32 ms floor on some medial segment.

**Result 3 — `lengthScale` cannot fix this.** It scales every phoneme uniformly, so the
imbalance survives: `icunga` goes from 2.86 at `lengthScale 1.0` to 2.64 at 1.5. The word
gets longer; the swallowed vowel stays proportionally swallowed.

**This corrects earlier guidance in this log.** The recommendation of `lengthScale` 1.4–1.8
for isolated words addresses gross duration only. It does not address the swallowed
syllable, and should not be presented as a fix for intelligibility.

**Result 4 — the carrier phrase plus the alignment does fix the allocation.** Synthesizing
`Iri jambo ni <word>.` and cutting the target span using the model's own `attn` gives an
even profile:

| word | bare, phones (ms) | final/median | cropped, phones (ms) | final/median |
|---|---|---|---|---|
| icunga | 75, 75, 32, 53, 213 | 2.86 | 21, 43, 43, 43, 43 | **1.00** |
| umuneke | 53, 64, 32, 32, 32, 75, 171 | 3.20 | 21, 32, 21, 43, 32, 21, 32 | **1.00** |

The cut is exact, not energy-based: cumulative frames x 256 give the sample boundaries, and
the durations are verified to account for the waveform.

**Status.** The timing explanation is measured and solid. Whether the cropped audio actually
*sounds* correct to a Kinyarwanda speaker is **not yet confirmed** — audio for that judgement
is in `results/listen_icunga/` (`*_A_bare_*` against `*_B_cropped_*`). Until someone listens,
the claim is "the allocation is fixed", not "the word is fixed".

**Note.** This partly retires the "no per-token durations" limitation: the ONNX export still
emits only `y`, but the torch engine exposes the alignment, so exact cropping is available
today without re-exporting.

## 2026-10-08 — both engines, full run: the export is identical on every clip

**Method.** First run with `torch_error: null` — our ONNX and C4IR's checkpoint both loaded
and measured on all 190 digit-free clips, `lengthScale 1.0`, `noise_scale 0`.
Raw: `results/rushing_20261008T085511Z_ls1.0.json`. Narrative: `REPORT.md`.

**Result.**

- **Engine agreement: `max |onnx_s − torch_s| = 0.00e+00` across 190/190 clips.** Sentence
  fidelity 15/15 identical sample counts (min corr 0.9999999996); isolated-word fidelity
  10/10 identical (min corr 0.9999999996).
- Ratio by length, identical for both engines: 0.875 (1–2 words), 0.932 (3), 0.937 (4–6),
  0.960 (7–12), 0.986 (13+). Model faster than the human on 72% of clips.
- Isolated words: mean 0.132 s/syllable, both engines. `icunga` 0.372 s, 0.124 s/syllable.
- Isolated-word raw peak by speaker: Female 1 **0.129**, Female 2 0.377, Male 0.435.
  Female 1 quieter on 10/10 words and below 0.10 on 3/10.

**Reading.** The earlier bit-identity result, taken on 10 words, now holds across the whole
measured sample. Every measurement in this log describes C4IR's published model, not our
conversion. The short-utterance rushing and the quiet isolated words are both theirs.

**What would overturn it.** Nothing about the export. The interpretation still rests on the
0.223 s/syllable isolated-word reference (n=6, different corpus and speaker), and on the
assumed `female → sid 0` speaker mapping.

## 2026-10-06 — the ONNX export is bit-identical to C4IR's checkpoint, including on isolated words

**Method.** Both engines loaded and run on the same token ids at `noise_scale=0`, speaker 0,
`lengthScale 1.0`: our export (`emmilly/kinya-flex-tts-onnx`) and C4IR's own checkpoint
(`C4IR-RW/kinya-flex-tts`, 34.9M generator params, 2,000K train steps). 10 isolated words
plus one sentence. Raw: `results/short_words_both_engines_20261006T135351Z.json`.

**Result.** Identical on every word — same sample counts, same durations to the
millisecond, same peaks, **correlation 1.00000**. Same for the sentence.

| word | syl | onnx s | torch s | s/syllable | peak | samples equal | corr |
|---|---|---|---|---|---|---|---|
| icunga | 3 | 0.372 | 0.372 | 0.124 | 0.045 | yes | 1.00000 |
| amazi | 3 | 0.287 | 0.287 | 0.096 | 0.135 | yes | 1.00000 |
| inka | 2 | 0.202 | 0.202 | 0.101 | 0.053 | yes | 1.00000 |
| umwana | 3 | 0.457 | 0.457 | 0.152 | 0.150 | yes | 1.00000 |
| ishuri | 3 | 0.457 | 0.457 | 0.152 | 0.186 | yes | 1.00000 |
| ibirayi | 4 | 0.414 | 0.414 | 0.104 | 0.111 | yes | 1.00000 |
| umuhinzi | 4 | 0.659 | 0.659 | 0.165 | 0.115 | yes | 1.00000 |
| ifumbire | 4 | 0.542 | 0.542 | 0.135 | 0.074 | yes | 1.00000 |
| umuneke | 4 | 0.510 | 0.510 | 0.128 | 0.132 | yes | 1.00000 |
| inanasi | 4 | 0.648 | 0.648 | 0.162 | 0.295 | yes | 1.00000 |

**Reading.** Two conclusions, and the second is the one that matters for the paper.

1. **The export is not a confound.** Every measurement taken on the ONNX describes C4IR's
   model exactly. The findings need no re-statement.
2. **The rushing is C4IR's model.** Their own checkpoint averages **0.132 s/syllable** on
   isolated words — indistinguishable from the actress's *sentence-internal* rate of 0.129,
   and roughly half her isolated-word rate of 0.223. The model applies sentence timing to
   isolated words, and it does so in the original, not in our conversion.

The quietness is theirs too: `icunga` peaks at 0.045 and `inka` at 0.053 in both engines.

**What would overturn it.** Nothing about the export — bit-identical output is not a
marginal result. The *interpretation* still rests on the 0.223 isolated-word reference,
which is n=6 from a different corpus and speaker.

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

## 2026-10-06 — a long tail where the model over-produces — **RESOLVED 2026-10-08**

**Method.** Same-text ratios, 190 clips.

**Result.** Mean ratio 1.286 against a median of 0.945, with p95 at 1.118 — the mean dragged
by outliers beyond the 95th percentile.

**Resolution (2026-10-08).** Not a model failure mode: **one broken corpus recording**.
`female2/3473` pairs a 25-word transcript with a 0.75-second file of digital silence (peak
0.0001, RMS 0.000002), so the bench measured 0.16 s of "speech" against the model's 10.98 s
— a ratio of 68.9. Excluding it, mean 1.286 → **0.929** and sd 4.917 → **0.123**. A scan
found exactly one such clip: it is the only one in the sample below 0.15 human-seconds per
word (0.006, against a median of 0.437). Logged as a third corpus defect alongside the
`Err:508` rows and the truncated references.

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
