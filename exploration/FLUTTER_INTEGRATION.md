# Kinyarwanda TTS in a Flutter app — the fp16 build with alignment

On-device Kinyarwanda speech. No network at inference, no API keys, no per-request cost.
This covers the **fp16 export that emits per-token durations**, and the single-word
handling that depends on them.

The voice is [`C4IR-RW/kinya-flex-tts`](https://huggingface.co/C4IR-RW/kinya-flex-tts)
(MB-iSTFT-VITS2), re-exported to ONNX with the alignment exposed.

| | |
|---|---|
| Output | mono float32 PCM, **24 kHz** |
| Voices | `sid=0` Female 1, `sid=1` Female 2, `sid=2` Male |
| Model size | **72 MB** (fp16) against 141.8 MB (fp32) |
| Opset | 18 |
| Outputs | `y` **and `durations`** — the second one is what this guide is about |
| Licence | **CC-BY-4.0 — you must attribute C4IR Rwanda & KiNLP** |

CC-BY-4.0 permits commercial use, unlike `facebook/mms-tts-kin` (CC-BY-NC-4.0).

---

## Which do you need?

The model's first input is `x: tensor(int64)` — it never sees text. So "just the model" is
not a thing: something has to turn `"icunga"` into token ids using a 126-symbol vocabulary
and a normalisation pass that spells numbers out with noun-class concord. That is 39 string
operations, and ONNX has no equivalent ops, so it cannot live in the graph.

There are two sensible bundles.

| | **minimal** | **full kit** |
|---|---|---|
| reads sentences and phrases | yes | yes |
| pronounces **single words** correctly | **no** | yes |
| files beyond the model | 25 KB (vocab + 2 tokenizer files) | + alignment, cropping, engine |
| API | `KinyaFlexMinimal.speak()` | `KinyaFlexTtsEngine.synthesize()` |
| zip | `kinya-flex-tts-minimal.zip` | `kinya-flex-tts-kit.zip` |

**Take the minimal bundle if your app reads phrases.** It is four small files plus the
model, and `speak()` is the whole surface.

**Take the full kit if your app ever says one word on its own** — a vocabulary drill, a
word list, a tap-to-hear glossary. A bare word comes out mispronounced, and the fix needs
the `durations` output and roughly 130 more lines. The rest of this document explains why,
and is worth reading before deciding the minimal bundle is enough.

Both ship the same 72 MB fp16 model.

---

---

## Why `durations` matters

Ask this model for a bare word and it mispronounces it. Not "sounds a bit off" —
a Kinyarwanda speaker hears **`icunga` as something closer to *ihuba***, with the middle
of the word swallowed.

The cause is measurable. On isolated input the duration predictor allocates time unevenly:
a medial phoneme is squeezed to 21–32 ms while the final vowel is stretched to roughly
**2.3× the median**. Inside a sentence every phoneme of `icunga` gets ~43 ms; alone, the
medial `/u/` gets 32 ms and the final `/a/` gets 213 ms.

| | final ÷ median phoneme duration |
|---|---|
| isolated word | **2.31** (range 1.00–3.20 over 10 words) |
| same word last in a carrier phrase | **1.15** |

**`lengthScale` cannot fix this.** It scales every phoneme uniformly, so the imbalance
survives: `icunga` goes from 2.86 at 1.0 to 2.64 at 1.5. The word gets longer and stays
wrong. This is the single most common wrong turn — do not reach for it.

What works is to say the word inside a carrier phrase, where the allocation is even, then
cut it back out **using the model's own alignment**. That is why the export has a second
output, and why an export without it cannot do this.

Energy-based segmentation is not an alternative. It was measured against real output and
fails: at a 120 ms gap tolerance a four-times repetition comes back as one continuous run,
because the model does not put reliable pauses at commas.

---

## What to copy

Six Dart files and two assets. Paths are relative to the source app
(`kinyarwanda_tts_app/`).

```
lib/kinya_flex_tokenizer.dart     text -> token ids (468 lines)
lib/kinya_number_speller.dart     numbers -> Kinyarwanda words (212 lines)
lib/flex_alignment.dart           durations -> sample ranges (88 lines)
lib/wave_segments.dart            WaveSegment + cropSegment (39 lines)
lib/kinya_flex_tts_engine.dart    ONNX session + synthesize() (349 lines)
lib/wav_encoder.dart              float32 PCM -> WAV bytes (59 lines)

assets/kinya_flex_symbols.json    vocabulary + config (2.7 KB)
assets/models/kinya_flex_tts.onnx the fp16 model (72 MB)

test/kinya_flex_tokenizer_test.dart   28 golden cases
test/kinya_flex_tokenizer_golden.json
test/flex_alignment_test.dart         cropping arithmetic
```

Dependency order: the engine imports the tokenizer, `flex_alignment` and `wave_segments`;
the tokenizer imports the number speller; `flex_alignment` imports `wave_segments`. Nothing
else from the source app is needed. The tokenizer's only Flutter dependency is `rootBundle`.

### Getting the model

It comes in the zip, at `assets/models/kinya_flex_tts.onnx` — **keep that filename**, it is
what the engine's `modelAsset` constant points at.

To rebuild it instead: export from the 1.11 GB PyTorch checkpoint with
`export_kinya_flex_tts_colab.ipynb`, which strips the checkpoint to its generator, replaces
an unexportable `torch.istft` with an equivalent inverse STFT, and works around four
dynamic-shape bugs in the upstream model code. Step 10 produces the fp16 variant.

Verify any replacement has both outputs before shipping it:

```python
import onnxruntime as ort
s = ort.InferenceSession("kinya_flex_tts.onnx")
print([o.name for o in s.get_outputs()])      # must be ['y', 'durations']
```

An export with only `y` predates this work. It will still read sentences, but the
single-word path cannot run and the engine falls back to direct synthesis.

**The export must use `dynamo=True`.** On torch 2.x the legacy tracer silently bakes the
audio length in as a constant: the graph exports cleanly and passes `onnx.checker`, but
every input then returns the same number of samples as the dummy. The notebook's step 9
catches this — it is how the problem was found.

---

## Setup

### 1. Dependencies

```yaml
dependencies:
  flutter_onnxruntime: ^1.8.2   # the ONNX Runtime binding
  just_audio: ^0.10.6           # playback (any player works)
  path_provider: ^2.1.5         # somewhere to write the WAV
```

Opset 18 needs ONNX Runtime 1.14+. `flutter_onnxruntime` 1.8.2 bundles 1.23.0 on
Android/iOS/macOS and 1.22.0 on Linux/Windows, so this is not a constraint in practice —
but check if you pin an older runtime.

### 2. Register the assets

```yaml
flutter:
  assets:
    - assets/kinya_flex_symbols.json
    - assets/models/kinya_flex_tts.onnx
```

A missing asset listed here fails the bundle for the whole app, not just this feature. If
you commit code before the model, keep the ONNX line commented out.

### 3. Load once, at startup

```dart
final engine = KinyaFlexTtsEngine();
await engine.load();     // parses the vocab, creates the session, warms up
```

`load()` reads a 72 MB file and runs one throwaway synthesis so ONNX Runtime's lazy
allocator and thread-pool setup happen behind your loading screen. Keep the engine alive;
do not construct one per request.

### 4. Speak

```dart
final Float32List pcm = await engine.synthesize(
  'Muraho neza, murakaza neza mu Rwanda.',
  voice: FlexVoice.female1,
  lengthScale: 1.0,           // inverse speed: >1.0 is slower
);
```

Single words need nothing special — the engine routes them through the carrier
automatically. `synthesize('icunga')` returns a correctly pronounced word.

### 5. Play it

`synthesize()` returns raw float32 PCM, which no player accepts directly:

```dart
final file = File('${(await getTemporaryDirectory()).path}/tts.wav');
await file.writeAsBytes(encodeWav(pcm, KinyaFlexTtsEngine.sampleRate), flush: true);
await player.setFilePath(file.path);
await player.play();
```

It writes **16-bit** PCM deliberately: Android's decoders support 16-bit reliably, float32
WAV inconsistently.

---

## How the single-word path works

`synthesize()` treats an utterance of **≤20 tokens** as a single word. (Measured: one-word
inputs tokenize to 9–15 ids, two words ~23, sentences 75–125.) Those go through:

```dart
final carrier = 'Iri jambo ni $word.';   // target lands last, before a full stop
```

Phrase-final, so it picks up final lengthening rather than being buried mid-phrase where it
is rushed by design. The model returns `durations`, a frame count per token, and the decoder
upsamples every frame by exactly **256** samples. So token boundaries are sample boundaries:

```
token i owns samples [256 * sum(durations[:i]), 256 * sum(durations[:i+1]))
total samples        == 256 * durations.sum()
```

`FlexAlignment.lastWordRange` finds the target. Normalisation spaces the punctuation —
`"iri jambo ni icunga ."` — so the word sits between the **last two** spaces, not after the
last one.

### The lead-in, which you will get wrong if you skip it

The alignment says which token *owns* a frame, but a phoneme's audible onset does not begin
at its own boundary. Coarticulation puts it earlier, in the frames belonging to the
preceding space. Measured: the eight frames before the boundary carry **1.8× to 13×** the
energy of the eight after it.

Cut exactly on the boundary and you shear off the attack. A Kinyarwanda speaker hears
**`avoka` as *voka***. The fix is two token slots of lead-in:

```dart
final from = (2 * a - FlexAlignment.vowelOnsetLeadSlots).clamp(0, durations.length);
```

Two, not three: the third slot reaches the carrier's final vowel and the crop opens on an
audible trace of it.

### Pacing is a separate knob

Cropping fixes *which* phonemes get time. It does not change the pace — a word cut from a
carrier carries the carrier's sentence pace, **0.123 s/syllable**, within 5% of the
actress's own sentence-internal rate. A person saying a word alone takes **0.223**.

So `croppedWordLengthScale` exists, defaulting to **1.5** (0.173 s/syllable, 78% of the
human rate), which a Kinyarwanda listener preferred across every word tested. 1.9 reaches
0.213, closest to the human rate, and is the next thing to try by ear.

Two knobs, two jobs: **crop for intelligibility, `lengthScale` for pacing.**

---

## fp16

Half the size, and safe for this use — but the second part needed checking rather than
assuming.

Cropping is `frames × 256 = samples`, and `durations` is an integer frame count carried in a
float. One frame drifting in half precision would cut the word in the wrong place. Measured
over 12 words: **the predicted durations come back as identical integers**, frame counts
168–221, and the identity holds in both builds. Waveform correlation against fp32 is
**0.9989–0.99998**.

The graph's inputs and outputs stay float32 (`keep_io_types=True`), so fp16 is a drop-in
swap needing no code change.

XNNPACK has native fp16 on ARMv8.2+, which is most modern phones, so fp16 may run *faster*
there rather than merely smaller.

**Do not use int8.** Full int8 measured **7.7× slower** than fp32 on this architecture, and
the `int8_matmul` build quantises nothing at all — VITS writes its linear layers as
`Conv1d(c, c, 1)`, so there are no qualifying MatMuls.

---

## Thread count

```dart
intraOpNumThreads: 1
```

Benchmarking this graph found **one thread 2.2× faster than six**, degrading monotonically
as threads were added — the iSTFT decoder is small enough that coordination costs more than
it saves. That run was on a cloud x86 VM of unconfirmed core count, so **measure on your
target device**; it is recorded because 6 was never measured at all.

Execution providers are `[XNNPACK, CPU]`. XNNPACK is the one that matters on ARM.

---

## The tokenizer will bite you

This model is **not** character-level. Its 126 symbols are Kinyarwanda consonant clusters of
up to five characters (`nshyw`, `pfyw`, `mbyw`). Tokenizing splits at vowels and looks up
each consonant run as a *single* symbol, falling back to per-character ids only when the
cluster is missing from the vocabulary.

Before that, `normText` rewrites the input the way the training data was normalised:
ASCII-folded, lowercased, punctuation spaced, and numbers, times, phone numbers, decimals
and abbreviations spelled out. Numerals take a **noun-class concord agreeing with the
preceding word** — "25" after `ibiro` is a different string from "25" after `amafaranga`.

**The failure mode is silent.** Every individual letter is also in the vocabulary, so a
mis-ported normalisation or a mis-segmented cluster still produces confident, fluent speech
— just with the wrong sounds. Nothing throws. If you do not speak Kinyarwanda you will not
notice.

This extends to your *input*, not only your code. `umuembe` tokenizes as `u m u e mb e`
while the standard spelling `umwembe` gives `u mw e mb e` — different words to the model,
and the wrong one fails silently.

So: **copy `kinya_flex_tokenizer.dart` and `kinya_number_speller.dart` verbatim.** They are
a branch-for-branch port of the Python, pinned by golden vectors. Do not refactor them for
style.

```bash
# the tests import package:kinyarwanda_tts/... — point them at your own package first
sed -i 's|package:kinyarwanda_tts/|package:YOUR_PACKAGE/|' test/*.dart
flutter test test/kinya_flex_tokenizer_test.dart
```

28 golden cases generated from the real Python implementation. **Run these after copying.**
The files in `lib/` need no such edit — they import each other by relative path.

One deliberate divergence: Dart's ints are 64-bit where Python's are arbitrary-precision, so
digit strings too long to parse are read out digit by digit.

---

## Tuning

| knob | default | notes |
|---|---|---|
| `voice` | `FlexVoice.female1` | `female1`, `female2`, `male` |
| `lengthScale` | 1.0 | inverse speed. Expose this — it is the most useful control for comprehension |
| `noiseScale` | 0.0 | prior sampling temperature. Upstream uses 0.667; 0.0 makes output bit-identical between calls, which is what a drill needs. Durations are unaffected either way — this model has a deterministic duration predictor |
| `normalize` | `true` | peak-normalises to 0.85 |
| `useCarrierForShortWords` | `true` | the single-word path. Turn off only to hear what it fixes |
| `croppedWordLengthScale` | 1.5 | pacing for cropped words |

**Keep `normalize` on.** Output level swings enormously with speaker and length. Measured
peaks over 10 isolated words: Female 1 averaged **0.129**, Female 2 **0.377**, Male
**0.435** — Female 1 was quieter on 10 of 10 words and fell below 0.10, where a word reads
as mumbled rather than merely quiet, on 3 of them. For single words, Female 2 is the safer
default if you are not normalising.

---

## The ONNX contract

Only needed if you are reimplementing the engine rather than copying it.

| input | shape | dtype | notes |
|---|---|---|---|
| `x` | `[1, T]` | int64 | token ids, blanks already interspersed |
| `x_length` | `[1]` | int64 | number of tokens |
| `sid` | `[1]` | int64 | 0 = Female 1, 1 = Female 2, 2 = Male |
| `noise_scale` | `[1]` | float32 | 0.667 upstream; 0.0 is deterministic |
| `length_scale` | `[1]` | float32 | inverse speed |

| output | shape | dtype | notes |
|---|---|---|---|
| `y` | `[1, 1, 256*u0]` | float32 | 24 kHz |
| `durations` | `[1, n_tokens]` | float32 | integer frame counts, one per interspersed token |

`T` and the sample count are dynamic. `tokenizer.textToIds()` already interspersed the
blanks — feed its output straight through; do not pad, truncate, or batch, the graph is
batch size 1.

**Always check the alignment before cropping:**

```dart
final expected = FlexAlignment.totalSamples(durations);
if ((expected - pcm.length).abs() > FlexAlignment.samplesPerFrame) return null;  // refuse
```

A mismatch means the durations belong to a different run and every offset would be wrong.
Refuse rather than crop at a guessed offset.

---

## App size

72 MB is still a large asset. Google Play rejects APKs above 150 MB, so with anything else
you need an Android App Bundle with Play Asset Delivery. On iOS, watch the over-cellular
download limit. `--split-per-abi` helps a lot if you ship a debug build for testing.

---

## What this does not fix

- **Not every word needs the carrier.** `inka` already has an even bare profile (1.00) and
  cropping gives only 0.25 s, which may be worse. The engine applies the carrier to
  everything under the token floor; if that hurts a particular word, compare by ear.
- **`indimu` crops to 0.75** — the opposite skew, a final vowel too short. Allocation may
  not be the whole story there.
- **The speaker mapping is assumed.** `female → sid 0`, `female2 → sid 1` is not documented
  by C4IR.
- **Tone is not modelled explicitly.** Kinyarwanda is tonal; nothing here addresses that.

---

## Attribution

CC-BY-4.0 requires credit. Put this somewhere visible, such as an About or licences screen:

> Kinyarwanda speech synthesis by C4IR Rwanda & KiNLP (`C4IR-RW/kinya-flex-tts`),
> used under CC-BY-4.0.

The training corpus (`C4IR-RW/kinya-ag-tts`) was implemented by C4IR Rwanda & KiNLP,
supported by GIZ, financed by BMZ.
