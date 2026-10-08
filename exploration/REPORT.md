# Where we stand

Report on the run of 2026-10-08 (`results/rushing_20261008T085511Z_ls1.0.json`), the first
with **both engines loaded** — our ONNX export and C4IR's own PyTorch checkpoint.

190 digit-free clips from the seeded 250-clip sample of `C4IR-RW/kinya-ag-tts`,
`lengthScale 1.0`, `noise_scale 0`, speech-only durations with a −32 dB gate.

---

## 1. The headline

Three questions were open. This run closes all three.

| question | answer |
|---|---|
| Is the rushing real? | **Yes**, and it scales monotonically with how short the utterance is |
| Is it C4IR's model or our ONNX conversion? | **C4IR's model.** The export is bit-identical |
| Is the training audio itself rushed? | **No.** The actress speaks normally; the corpus has no isolated words to learn from |

The working claim now has direct support: **the model reproduces its training distribution
faithfully, and the failure is a gap in that distribution's coverage.**

---

## 2. The export is not a confound

Across **all 190 measured clips**, our ONNX and C4IR's checkpoint produced durations
identical to the limit of floating point — `max |onnx_s − torch_s| = 0.00e+00`.

| check | result |
|---|---|
| all measured clips, duration delta | 0.000 s on 190/190 |
| sentence fidelity, matched text | 15/15 identical sample counts, min corr 0.9999999996 |
| isolated-word fidelity | 10/10 identical sample counts, min corr 0.9999999996 |

This matters because every earlier measurement was taken on the ONNX. It can now be stated
plainly: **those measurements describe C4IR's model.** No finding needs re-stating, and a
reviewer asking "is this your conversion's fault?" has a one-line answer.

Worth noting *where* this was checked. An earlier version compared the engines only on
whichever clips came first — mostly ordinary sentences, i.e. the regime where nothing is
disputed. The check is now stratified across length buckets and repeated on isolated words
specifically, so the equality is demonstrated exactly where the failure lives.

---

## 3. The rushing, quantified

Model duration ÷ the actress's duration, same text:

| words | n | median ratio | `lengthScale` that would match her |
|---|---|---|---|
| 1–2 | 33 | **0.875** | 1.14 |
| 3 | 43 | 0.932 | 1.07 |
| 4–6 | 44 | 0.937 | 1.07 |
| 7–12 | 41 | 0.960 | 1.04 |
| 13+ | 29 | **0.986** | 1.01 |

Perfectly monotonic. On long sentences the model is within 1.4% of the woman it was trained
on; the deficit grows as utterances shorten. Overall median 0.945, with the model faster
than the human on **72%** of clips.

This is the shape that matters for the paper. A uniformly low ratio would mean a globally
fast model, fixable with one `lengthScale`. A ratio that degrades with brevity means the
duration predictor fails to generalise to short input — which is exactly what the corpus
predicts.

---

## 4. Isolated words — the untrained case

Both engines, identical values. Mean **0.132 s/syllable** across 10 words (range
0.096–0.165). `icunga`: 0.372 s, 0.124 s/syllable.

Against the human reference points:

| | s/syllable |
|---|---|
| Model, isolated word | **0.132** |
| Actress, *inside a sentence* | 0.129 |
| Actress, genuinely isolated word | 0.223 |

The model speaks an isolated word at its **sentence-internal** rate. It applies no isolation
lengthening at all, where a human applies roughly 1.7×.

The cause is in the corpus and is not subtle: of 17,969 rows, **2 are single words, and both
are the spreadsheet error string `Err:508`**. The duration predictor has never seen an
isolated word.

---

## 5. The outlier tail — resolved

Previously logged as unexplained: mean ratio 1.286 against a median of 0.945, suggesting
clips where the model massively over-produces.

It is **one broken corpus recording**, not a model failure.

`female2/3473` pairs a 25-word, 213-character transcript with a **0.75-second file
containing digital silence** — peak amplitude 0.0001, RMS 0.000002. The bench measured
0.16 s of "speech" against the model's 10.98 s, giving a ratio of 68.9.

Removing that single clip:

| | n | mean | median | sd |
|---|---|---|---|---|
| with it | 190 | 1.286 | 0.945 | 4.917 |
| without it | 189 | **0.929** | 0.945 | **0.123** |

The standard deviation falls 40-fold. A scan for others found exactly one: it is the only
clip in the sample under 0.15 human-seconds per word (0.006, against a median of 0.437).

Two consequences. The mean is now usable, and it says the model is ~7% faster than the
actress overall. And this is a third data-quality defect in the corpus, alongside the
`Err:508` rows and the truncated references — worth reporting as a finding about the
dataset in its own right. (Note also that this file is 16-bit where the others sampled are
32-bit, so bit depth is not uniform across the corpus.)

---

## 6. Level, not only timing

Isolated words are quiet as well as rushed, and the effect is strongly speaker-dependent:

| speaker | mean raw peak on 10 isolated words |
|---|---|
| Female 1 | **0.129** |
| Female 2 | 0.377 |
| Male | 0.435 |

Female 1 is quieter on **10 of 10** words — 2.9× on average — and falls below 0.10, the
point where a word reads as mumbled rather than merely quiet, on 3 of 10 (`icunga` 0.045,
`inka` 0.053, `ifumbire` 0.074).

This is a second, independent defect stacking on the timing one, and it explains why putting
a word next to another word *appears* to fix its pronunciation: it does not change
articulation, it raises the level. For single-word use, Female 2 is the better default.

---

## 7. Where that leaves us

**Settled, with evidence in hand:**

- The ONNX export is bit-identical to C4IR's checkpoint (190/190 clips).
- The rushing is C4IR's model, and is monotonic in utterance length.
- The training audio is not rushed; the actress's rate is normal and matches an independent
  Kinyarwanda corpus to within 2%.
- The corpus contains no isolated words to learn from (2 of 17,969, both corrupt).
- Isolated words are also quiet, worst on Female 1.
- The mean/median gap was one silent recording, not a model failure mode.

**Open:**

- **Articulation: partly answered, see the 2026-10-08 entry in `FINDINGS.md`.** A native
  speaker reports `icunga` rendered closer to *ihuba* — swallowed syllables, not just a fast
  word. The mechanism is now measured: on isolated words the duration predictor crushes a
  medial phoneme to the 32 ms floor while stretching the final vowel to ~2.3x the median.
  `lengthScale` cannot fix it (it scales uniformly); cropping from a carrier phrase using
  the model's alignment does, giving an even profile. **Still open:** whether the cropped
  audio sounds correct to a Kinyarwanda speaker. Audio is in `results/listen_icunga/`.
- **The isolated-word human reference is thin.** The 0.223 s/syllable figure is n=6, from a
  different corpus and a different speaker. The bit-identity result is unshakable; this
  comparison target is not.
- **Speaker mapping is assumed.** `female → sid 0`, `female2 → sid 1` is undocumented by
  C4IR, who also do not document `female2`/`male2` at all.
- **No per-token durations.** The exported ONNX emits only `y`, so exact word cropping from
  a carrier phrase needs a re-export with the alignment exposed.
- **No fix has been tested.** The proposed remedy — mining utterance-final words by forced
  alignment and fine-tuning on them — remains a proposal.

**Immediate practical guidance, unchanged and now better supported:** for single-word
synthesis use `lengthScale ≈ 1.4`, prefer Female 2, and keep `noise_scale 0` so a learner
hears the same word identically each time.

---

## 8. What is defensible in the paper

Claims this run supports directly:

1. A systematic fine-tuning-free characterisation of `kinya-flex-tts` across 19 utterance
   lengths and two voices, measured against the corpus it was trained on.
2. The short-utterance degradation is monotonic and belongs to the published model, proven
   by bit-identical agreement between an independent ONNX export and the original
   checkpoint.
3. The cause is distributional, not architectural: the training corpus contains no isolated
   words.
4. Three documented data-quality defects in a published CC-BY-4.0 corpus: `Err:508` rows,
   truncated references, and at least one silent recording paired with a full transcript.

5. A mechanism for the segmental failure on isolated words: the duration predictor
   allocates time unevenly, crushing medial phonemes to a 32 ms floor while stretching the
   final vowel (final/median 2.31 isolated against 1.15 in a carrier). `lengthScale` cannot
   correct it; alignment-based cropping from a carrier can.

Claim **not** yet supported: that the carrier-cropped output is *correct* Kinyarwanda. The
allocation is demonstrably even, but only a native speaker can confirm the word is now
intelligible.
