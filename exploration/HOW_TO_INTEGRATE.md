# How to add Kinyarwanda word pronunciation to an existing Flutter app

Step by step, from an app that has none of this to one that speaks a word aloud.
Allow about 20 minutes, most of it waiting on `pub get`.

`README.md` is the 5-minute summary. `INTEGRATION.md` is the reference that explains *why*
each piece exists. **This file is the procedure.**

---

## What you are adding

An offline Kinyarwanda voice. No network at speak time, no API key, no per-use cost.
24 kHz audio from a 72 MB model that ships inside your app.

It pronounces **single words** correctly, which is the thing this kit exists for. Asked for
a bare word the raw model mispronounces it — it swallows the middle, so `icunga` comes out
closer to *ihuba*. The engine here works around that automatically, and the result was
confirmed correct by a Kinyarwanda speaker.

---

## Before you start

- Flutter 3.x and a working Android or iOS build.
- **~90 MB of app size.** The model is 72 MB; it compresses to about 62 MB in an APK.
  Google Play rejects APKs over 150 MB, so if your app is already large you will need an
  Android App Bundle with Play Asset Delivery.
- Someone who speaks Kinyarwanda, for step 7. This is not optional for anything a patient
  or learner will hear.

---

## Step 1 — unzip

```
kinya-flex-tts-kit/
  assets/models/kinya_flex_tts.onnx     the voice, 72 MB
  assets/kinya_flex_symbols.json        the vocabulary
  lib/*.dart                            six files
  test/*                                golden vectors
```

## Step 2 — copy the files in

```bash
cp kinya-flex-tts-kit/lib/*.dart          your_app/lib/
mkdir -p your_app/assets/models
cp kinya-flex-tts-kit/assets/kinya_flex_symbols.json  your_app/assets/
cp kinya-flex-tts-kit/assets/models/kinya_flex_tts.onnx your_app/assets/models/
cp kinya-flex-tts-kit/test/*              your_app/test/
```

**Keep the model's filename.** The engine looks for `assets/models/kinya_flex_tts.onnx`. If
you must rename it, change `modelAsset` in `kinya_flex_tts_engine.dart` to match.

## Step 3 — declare the dependency and the assets

In `pubspec.yaml`:

```yaml
dependencies:
  flutter_onnxruntime: ^1.8.2    # runs the model
  path_provider: ^2.1.5          # only if you write the audio to a file
  just_audio: ^0.10.6            # or whatever player you already use

flutter:
  assets:
    - assets/kinya_flex_symbols.json
    - assets/models/kinya_flex_tts.onnx
```

```bash
flutter pub get
```

A missing asset listed here fails the bundle for your **whole app**, not just this feature.
If you commit the code before the 72 MB model, comment that line out meanwhile.

## Step 4 — point the tests at your package

The tests were written in another app, so they import its package name:

```bash
cd your_app
sed -i 's|package:kinyarwanda_tts/|package:YOUR_PACKAGE_NAME/|' test/*.dart
```

`YOUR_PACKAGE_NAME` is the `name:` field at the top of your `pubspec.yaml`.
The files in `lib/` need no edit — they import each other by relative path.

## Step 5 — run the tests before writing any code

```bash
flutter test test/kinya_flex_tokenizer_test.dart test/flex_alignment_test.dart
```

Expect **all passing**. If the tokenizer tests fail, stop and fix that first — see
Troubleshooting.

## Step 6 — load once, speak anywhere

Create the engine when your app starts, not per request. `load()` reads 72 MB and runs one
throwaway synthesis so the first real tap is not slow.

```dart
import 'kinya_flex_tts_engine.dart';
import 'wav_encoder.dart';

final _tts = KinyaFlexTtsEngine();

// in your startup / splash:
await _tts.load();
```

Then, wherever a word should be spoken:

```dart
final pcm = await _tts.synthesize('icunga');   // Float32List, 24 kHz

// no player takes raw float32, so wrap it as WAV:
final file = File('${(await getTemporaryDirectory()).path}/word.wav');
await file.writeAsBytes(
    encodeWav(pcm, KinyaFlexTtsEngine.sampleRate), flush: true);
await player.setFilePath(file.path);
await player.play();
```

That is the whole integration. Single words are detected and handled automatically — you do
not call anything different for a word than for a sentence.

## Step 7 — have a Kinyarwanda speaker check your word list

**Do this before anyone else hears it.** Render every word your app will say, and listen.

Known from a 12-word sample:

- `indimu` is still wrong after the fix — it ends on a 21 ms vowel.
- `inka` may be *worse* handled than left alone; its bare rendering is already even.
- `umuembe` and `umwembe` are different words to the model. Spelling matters, and the wrong
  spelling produces confident, fluent, wrong speech.

Those came out of twelve words, so expect others. There is no automated check for this —
the model never signals that it has mispronounced something.

---

## If your word list is fixed, consider not shipping the model

For a fixed vocabulary this is often the better design:

1. Render each word once, on a desktop, with this kit.
2. A Kinyarwanda speaker listens to every clip and approves or rejects it.
3. Ship the **approved audio files**; drop the model, the engine and the dependency.

| | model in the app | pre-rendered audio |
|---|---|---|
| 200 words | 72 MB + ONNX Runtime | **~0.6 MB** as opus |
| correctness | per word, unverified | every clip heard by a human first |
| cold start | reads 72 MB at launch | none |

Ship the model when the text is open-ended — user input, a vocabulary that grows, content
you do not control. Ship audio when a clinician picks from a list you already know.

---

## Troubleshooting

**Tokenizer tests fail.** You copied the files but changed them, or you missed
`kinya_number_speller.dart`. Copy both verbatim; do not reformat them. They are a
branch-for-branch port of C4IR's Python, pinned by those vectors.

**A word sounds fluent but wrong, and nothing errored.** That is the expected failure mode.
The 126 symbols are consonant clusters, and *every single letter is also in the vocabulary*,
so a bad tokenizer or a misspelt input still produces confident speech. Check the spelling
first, then re-run step 5.

**`model has no "durations" output` in the logs.** You have an older export. The one in this
zip is correct — verify with:

```python
import onnxruntime as ort
print([o.name for o in ort.InferenceSession("kinya_flex_tts.onnx").get_outputs()])
# must print ['y', 'durations']
```

**A single word sounds rushed or swallowed.** The carrier path did not run. Check the log
for `[Flex] carrier crop unavailable`, and that `useCarrierForShortWords` is true. **Do not
reach for `lengthScale`** — it scales every phoneme equally, so the word gets longer and
stays wrong. That is the predictable wrong turn.

**Words are too fast even though they are correct.** Raise `croppedWordLengthScale` from
1.5 toward 1.9. 1.5 was preferred by a Kinyarwanda listener; 1.9 is closest to how a person
actually says a word alone.

**Slow on device.** `intraOpNumThreads` is 1, which benchmarked 2.2× faster than 6 on this
model — but that was on x86. Try 2 and 4 on your target phone and keep what wins.

---

## Licence — this part is not optional

**CC-BY-4.0.** You must credit the authors somewhere visible, such as an About or licences
screen:

> Kinyarwanda speech synthesis by C4IR Rwanda & KiNLP (`C4IR-RW/kinya-flex-tts`),
> used under CC-BY-4.0.

Commercial use is permitted. (The other common Kinyarwanda voice,
`facebook/mms-tts-kin`, is CC-BY-NC-4.0 and is not.)
