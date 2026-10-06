# Methodology

How each number in `FINDINGS.md` is produced, and the measurement traps we hit. Several of
these cost us a wrong answer before we caught them, so they are recorded as warnings, not
just as settings.

## The core measurement: same text, human reference

For each sentence we synthesize the **same text** the actress recorded, and compare
durations:

```
ratio = model_speech_seconds / human_speech_seconds
```

Below 1.0 means the model is faster than the woman it was trained on. This is the
measurement to prefer wherever a human recording exists, because it needs no model of what
"normal" speaking rate is — the actress *is* the reference, per sentence, in her own voice.

Earlier work used seconds-per-syllable instead. That was only ever a fallback for cases
with no human recording (isolated words, cross-corpus comparison), and it carries the
syllable-counting caveat below.

## Settings

| setting | value | why |
|---|---|---|
| `noise_scale` | 0.0 | makes repeated synthesis bit-identical, so a measurement is reproducible and a learner hears the same word twice |
| `length_scale` | 1.0 baseline | the shipped default; swept to find what would match the human |
| speech gate | −32 dB relative to the clip's loudest frame | excludes room tone |
| frame hop | `sr // 94` (~256 @ 24 kHz) | matches the model's own frame rate |
| peak normalisation | 0.85, for listening only | never applied before measuring duration |

Durations are **speech-only**: first to last frame above the gate. Corpus clips carry up to
~0.6 s of leading/trailing room tone, which would otherwise read as slow speech.

## Traps

**1. Rows containing digits must be excluded from rate statistics.**
`Telefoni: 0784009558` is two words and four vowels of text, but the actress reads ten
digits over about six seconds. Including such rows made isolated-word rate look like
0.560 s/syllable; excluding them gave 0.176. 24% of the sampled corpus contains digits.
They are kept in the corpus and flagged in the UI, but dropped from aggregates.

**2. The corpus WAVs are 32-bit.**
Reading them as `int16` silently doubles the sample count and halves every computed rate.
`read_wav` dispatches on `getsampwidth()`. A length assertion against `getnframes()` is
cheap insurance.

**3. Syllable counts are a proxy, and only valid on digit-free text.**
We count vowel groups (`[aeiou]+`), which is close for Kinyarwanda's near-perfect CV
structure, but it is not a syllabifier. Use it only where no human recording exists; prefer
the duration ratio everywhere else.

**4. A wrong tokenizer fails silently.**
The vocabulary holds every single letter as well as the multi-character clusters, so a
mis-ported tokenizer yields fluent, confident speech saying the wrong thing, and nothing
throws. We therefore use C4IR's own `deepkin.data.kinya_norm` rather than a
reimplementation, and the notebook **asserts** it reproduces all 15 golden vectors from the
ONNX export before any synthesis runs. Verified: 15/15.

**5. Loudness masquerades as articulation.**
Isolated words come out quiet (Female 1 raw peak as low as 0.045), and quiet reads as
mumbled. Every listening comparison is peak-normalised so that timing and articulation are
what is being judged. This is also why putting a word next to another word *appears* to fix
its pronunciation — it does not, it raises the level.

**6. Whole-clip averages hide the thing you are measuring.**
Comparing a bare word against a `vuga <word> neza` carrier told us nothing, because the
carrier's average rate folds in two other words. Either measure the target word alone, or
expose per-token durations and cut precisely.

## Sampling

Stratified by `(voice, utterance-length bucket)`, buckets `≤2 / 3 / 4–6 / 7–12 / 13+`
words, 25 clips per cell, `random.Random(7)`, voices `female` and `female2`. 250 clips of
17,969. The draw is written to `kinya_ag_sample/sample.tsv` and **reused verbatim** on
later runs rather than redrawn, so the sample is a fixed part of the record.

## Provenance

- Model: `C4IR-RW/kinya-flex-tts`, exported to ONNX fp32 by `export_kinya_flex_tts_colab.ipynb`.
  Exported graph emits only `y`; no per-token durations.
- Corpus: `C4IR-RW/kinya-ag-tts`, 4 TSVs + 17,972 WAVs, 24 kHz mono, CC-BY-4.0.
- Tokenizer: `github.com/c4ir-rw/ac-ai-models`, `DeepKIN-AgAI/deepkin`, shallow clone.
  `deepkin/__init__.py` is emptied to neutralise an eager native-bindings import, exactly
  as C4IR's own export notebook does.
- Secondary corpus used once for cross-checking, **not** training data for this model:
  `DigitalUmuganda/afrispeak_kinyarwanda_female_tts_dataset` (1,892 clips, 16 kHz, one
  actress).

## Known limits of this setup

- Duration ratio is not articulation. A clip can match the human's length and still swallow
  a syllable; only listening catches that.
- No human reference exists for isolated words anywhere in the corpus, so the single-word
  case cannot be scored, only heard.
- The mapping `female → sid 0`, `female2 → sid 1` is an assumption; C4IR documents neither
  `female2` nor `male2`.
