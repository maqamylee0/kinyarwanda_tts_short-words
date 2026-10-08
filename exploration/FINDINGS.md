# Findings

Dated log. Each entry records the method, the number, and what would overturn it. Methods
and traps are in `METHODOLOGY.md`; raw output in `results/`.

---

## 2026-10-08 — fp16 build: half the size, and the durations survive it

**Why it needed checking.** Cropping is `frames x 256 = samples`. `durations` is an integer
frame count carried in a float, so a single frame drifting in half precision would cut the
word in the wrong place. Size alone is not the question.

**Method.** ONNX Runtime's own fp16 converter with `keep_io_types=True` (so inputs and
outputs stay float32 and the file is a drop-in swap), applied to the new fp32 export. 12
words, carrier-cropped at `lengthScale 1.5`.

**Result.**

| | fp32 | fp16 |
|---|---|---|
| size | 141.8 MB | **72.1 MB** (0.51x) |
| weights | 415 FLOAT tensors | 415 FLOAT16 tensors |
| graph inputs/outputs | float32 | float32 (unchanged) |

- **Predicted durations matched on all 12 words** (frame counts 168–221), and
  `frames x 256 == samples` holds in both builds.

  **Corrected 2026-10-08, later the same day:** that was 12 short carrier phrases and does
  not generalise. Measured over a wider set, fp16 durations drift on **4 of 80 sentences**
  and **1 of 48 carrier renderings** (`umuembe` at `lengthScale 1.9`). Every drift observed
  was exactly **one frame — 11 ms** — on a single token. Implications: on a sentence it is
  inaudible, but it shifts everything after that token, which is why one clip shows a
  waveform correlation of 0.66 while sounding identical. For cropping, an 11 ms shift is
  well inside the margin the 2-slot lead-in already adds, so the cut survives it. fp16 is
  still safe for this use — but "identical" was too strong.
- Waveform correlation against fp32: **0.9989 – 0.99998**, worst case `inka`; max absolute
  difference 5.1e-03; identical sample counts on every word.

**Reading.** fp16 is safe for this use and halves the asset. The earlier export report
measured 0.99914 correlation for the fp16 build of the durations-less model, so this is
consistent with it.

Audio is in `results/listen_onnx/listen.html` as a blue row under each word's preferred
rendering, so fp32 and fp16 can be compared back to back. Whether the difference is audible
is for the ear, not the correlation coefficient.

**Note.** `onnxconverter-common`'s converter is not usable here — it produces a
type-inconsistent graph on this dynamo-exported model that ORT refuses to load. The
notebook already documents this and uses ORT's converter instead.

## 2026-10-08 — re-exported the ONNX with the alignment, so the fix can run without torch

**Why.** The single-word remedy needs per-token durations. The shipped ONNX emits only `y`,
so the fix ran only through the 1.11 GB PyTorch checkpoint — unusable on device.

**What was already there.** The export notebook's wrapper already returned
`(o, durations)` with `output_names=["y","durations"]`. The shipped artifact simply predated
that edit (ONNX exported 10:48, notebook modified 13:16, same day). Three real defects
remained, all fixed:

1. `onnx_run` was called in step 9b but defined nowhere — that cell would have raised
   `NameError`.
2. Step 9 never checked that the `durations` output existed or was usable.
3. The output path and the crop-sample path were hardcoded to `/content`, so the notebook
   only worked on Colab even though `WORKDIR` already adapted.

**Two failures found by running it, not by reading it.**

- `torch._check(...)` was given a Tensor condition on the audio path, which torch 2.x
  rejects (`cond must be a bool`). Three patch cells did this. Now routed through one
  `onnx_check()` helper that picks `_check` or `_check_tensor_all`.
- **The legacy tracer silently baked the audio length in as a constant.** The graph
  exported cleanly and passed `onnx.checker`, but 8 of 9 validation cases returned an
  identical 84,992-sample waveform — the dummy's length. This is exactly the failure step 9
  exists to catch, and it caught it. Switching to `dynamo=True` with an explicit
  `dynamic_shapes` spec fixes it; all 9 cases then match PyTorch to ~1e-5.

**Result.** `exploration/models/kinya_flex_tts_durations.onnx`, 141.8 MB, opset 18,
self-contained:

| check | result |
|---|---|
| outputs | `y [1,1,256*u0]`, `durations [1,n_tokens]` |
| matches PyTorch | 9/9 cases, max abs diff ~1e-5, lengths exact |
| `durations x 256 == samples` | holds |
| ONNX-only crop vs the torch crop | identical sample count, max diff 9.7e-07, **corr 1.00000000** |

So the single-word fix now runs from the ONNX alone. That retires the last engineering
blocker to shipping it on device.

**Not yet done.** The new ONNX is not on the Hub — `emmilly/kinya-flex-tts-onnx` still holds
the durations-less build, so the bench and the app still fetch the old one. Uploading it is
a deliberate step, and it would change what the bench loads.

## 2026-10-08 — the two knobs are complementary, and the preferred setting nearly matches a human

**Method.** Speech-only seconds per syllable (the same −32 dB gate the human reference
figures use), 12 words, Female 1, cropped from the carrier at several `lengthScale` values.

**Result.**

| rendering | s/syllable | vs the human isolated-word rate |
|---|---|---|
| bare | 0.126 | 0.56× |
| cropped @1.0 | 0.123 | 0.55× |
| **cropped @1.5** — preferred by the listener | **0.173** | **0.78×** |
| cropped @1.9 | 0.213 | 0.95× |
| *human, inside a sentence* | *0.129* | — |
| *human, isolated word* | *0.223* | 1.00× |

**Reading.** The two fixes do different jobs and both are needed.

- **Cropping fixes *which* phonemes get time** — evenness 1.00 instead of 2.0–3.2. That is
  what makes the word intelligible.
- **It does not fix pacing.** A word cut from a sentence carries sentence pace: 0.123
  s/syllable, within 5% of the actress's own sentence-internal 0.129. Cropping alone leaves
  the word correct but spoken at conversational speed.
- **`lengthScale` supplies the isolation lengthening** a human applies when saying a word
  alone. At 1.5 the model reaches 0.173 s/syllable, 78% of the human isolated rate; at 1.9
  it reaches 0.213, within 5% of it.

A Kinyarwanda listener independently preferred 1.5 across every word tested, before seeing
any of these numbers. That preference lands between sentence pace and full isolation pace,
closer to the latter.

**This also resolves the earlier confusion about `lengthScale`.** It was never useless — it
was being asked to do the wrong job. It cannot repair a skewed allocation, which is what
made bare words unintelligible; it is exactly the right tool for pacing once the allocation
is fixed.

**Defaults changed.** `kinya_word.py` now defaults to `length_scale=1.5` and `sid=0`, the
configuration that was actually validated by ear.

**Open.** Whether 1.9 is better than 1.5, or begins to drag, is unresolved — both are in
`results/listen_fruits/listen.html`.

## 2026-10-08 — seven more words: the skew is the norm, and a spelling trap

**Method.** `kinya_word.py` on `umuembe, avoka, igitoki, inanasi, ipapayi, pome, indimu`,
Female 1, `lengthScale 1.0`. Audio and a listening page in `results/listen_fruits/`.

**Result.** Every one has a skewed bare allocation, and cropping evens all but one:

| word | gloss | bare s | bare even | crop s | crop even |
|---|---|---|---|---|---|
| umuembe | mango (see spelling note) | 0.80 | 2.67 | 0.39 | 1.00 |
| avoka | avocado | 0.81 | 3.00 | 0.31 | 1.00 |
| igitoki | banana | 1.05 | 3.20 | 0.54 | 1.00 |
| inanasi | pineapple | 1.06 | 2.67 | 0.47 | 1.00 |
| ipapayi | papaya | 0.89 | 3.00 | 0.53 | 1.00 |
| pome | apple | 0.70 | 2.00 | 0.37 | 1.00 |
| indimu | lemon | 0.69 | 2.33 | 0.30 | **0.67** |

Bare evenness ranges 2.00–3.20 across all seven. Combined with the earlier ten, the skew is
the rule for isolated words, not a property of particular ones.

**Correction to this log.** Earlier entries described a "32 ms, 3-frame floor". That was an
over-generalisation from the first words sampled. Measuring the minimum across 17 words: the
lowest is **2 frames (21.3 ms)**, on `umuembe` and `indimu`; 32 ms (3 frames) is the most
common minimum but not a floor. The mechanism is unchanged — medial segments are squeezed to
2–4 frames while the final vowel is stretched.

**`indimu` is the exception.** Its second `i` gets 21 ms in *both* renderings, and the
cropped version ends on a 21 ms `u`, giving evenness 0.67 — the opposite skew. Cropping may
not rescue it. Unverified by ear.

**Spelling trap — `umuembe` vs `umwembe`.** The requested spelling tokenizes as
`u m u e mb e`; the standard spelling of mango, `umwembe`, tokenizes as `u mw e mb e`. These
are different inputs and will produce different speech, and the wrong one fails silently —
fluent, confident, wrong. Both are in the listening page. `umwembe` is also less skewed to
begin with (bare evenness 1.78 against 2.67), which is itself a hint that it is the
better-formed word.

**Status.** Allocation measured; intelligibility unconfirmed for these seven.

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
three frames — while the final /a/ is stretched to 213 ms. A 32 ms vowel between a
75 ms /c/ and a 53 ms /ng/ is perceptually swallowed, which is what the listener reports.

**This is general, not specific to `icunga`.** Final-phoneme duration divided by the median
phoneme duration, across 10 isolated words:

| condition | mean final/median | range |
|---|---|---|
| isolated | **2.31** | 1.00 – 3.20 |
| same word, last in a carrier phrase | **1.15** | 1.00 – 2.00 |

Every isolated word squeezes some medial segment to 2-4 frames (21-43 ms); the lowest
observed is 2 frames (21.3 ms), on `umuembe` and `indimu`.

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

**Status — CONFIRMED by a Kinyarwanda speaker, 2026-10-08.** Listening to the A/B set in
`results/listen_icunga/`: the carrier-cropped renderings of `icunga` (at `lengthScale` 1.0
and 1.5) are **correctly pronounced**; the bare renderings are not. The slowed bare
rendering was not among the correct ones, which matches the prediction that `lengthScale`
cannot repair the allocation.

So the chain is closed end to end: a measured cause (a medial phoneme squeezed to 21-32 ms
while the final vowel takes ~2.3x the median), a measured fix (carrier synthesis plus
alignment-exact cropping, evenness 1.00), and perceptual confirmation that the fix yields an
intelligible word.

**Caveat — the fix is not needed for every word.** `inka` already has an even bare profile
(`[139, 43, 139]`, evenness 1.00) and cropping it yields only 0.17 s, which may be too
abrupt. The remedy applies to words whose *bare* allocation is skewed; `evenness()` in
`kinya_word.py` reports that, so it can be checked per word rather than assumed.

**Implementation.** `exploration/kinya_word.py` packages this: `KinyaWordSynth.say(word)`
returns the cropped audio, `evenness()` reports the allocation ratio, and the CLI writes
WAVs with `--also-bare` for comparison. Verified bit-identical to the clip confirmed above.

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
