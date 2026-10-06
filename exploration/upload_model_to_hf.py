#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.34"]
# ///
"""Publish the exported kinya-flex-tts ONNX to the Hugging Face Hub, once.

The model is 136 MB, over GitHub's 100 MB per-file limit, so the bench fetches it from
the Hub instead. This uploads it there and writes a model card carrying the attribution
CC-BY-4.0 requires.

Run it yourself — it needs YOUR token, which is not something to hand to a script that
anyone else runs:

    hf auth login                  # or export HF_TOKEN=hf_...
    uv run exploration/upload_model_to_hf.py

Then the notebook finds it with no further configuration, because the default repo id
below is also its default. Override with KINYA_HF_MODEL_REPO on both sides to use another.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ID = os.environ.get("KINYA_HF_MODEL_REPO", "emmilina/kinya-flex-tts-onnx")
FILENAME = "kinya_flex_tts.onnx"

CANDIDATES = [
    Path(__file__).resolve().parent / "notebooks/assets" / FILENAME,
    Path(__file__).resolve().parent.parent / "kinyarwanda_tts_app/assets/models" / FILENAME,
    Path(__file__).resolve().parent.parent / "kinya_flex_export" / FILENAME,
]

CARD = f"""---
license: cc-by-4.0
language: [rw]
tags: [text-to-speech, kinyarwanda, onnx, vits]
library_name: onnx
---

# kinya-flex-tts — ONNX export

ONNX fp32 export of [`C4IR-RW/kinya-flex-tts`](https://huggingface.co/C4IR-RW/kinya-flex-tts)
(MB-iSTFT-VITS2), 24 kHz, three speakers (0 = Female 1, 1 = Female 2, 2 = Male).

Exported with `export_kinya_flex_tts_colab.ipynb`, which strips the training checkpoint to
its generator and replaces an unexportable `torch.istft` with an equivalent inverse STFT.

Published so that the short-utterance benchmark in
[maqamylee0/kinyarwanda_tts_short-words](https://github.com/maqamylee0/kinyarwanda_tts_short-words)
can fetch it: at 136 MB it exceeds GitHub's 100 MB per-file limit.

## Inputs and outputs

| input | shape | dtype |
|---|---|---|
| `x` | `[1, T]` | int64 (token ids, blanks interspersed) |
| `x_length` | `[1]` | int64 |
| `sid` | `[1]` | int64 |
| `noise_scale` | `[1]` | float32 |
| `length_scale` | `[1]` | float32 |

Output `y` is `[1, 1, samples]` float32 at 24 kHz. Batch size is fixed at 1.

## Attribution

Original model by **C4IR Rwanda & KiNLP**, used and redistributed under CC-BY-4.0.
This repository contains only a format conversion; the weights are theirs.
"""


def main() -> int:
    from huggingface_hub import HfApi

    src = next((p for p in CANDIDATES if p.exists()), None)
    if src is None:
        print("Could not find the ONNX. Looked in:", file=sys.stderr)
        for p in CANDIDATES:
            print(f"  {p}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception:
        print(
            "Not logged in to Hugging Face.\n"
            "  hf auth login           # the CLI is `hf`; `huggingface-cli` was removed in\n"
            "                          # huggingface_hub v1.0\n"
            "or pass a token directly, which also works under `uv run`:\n"
            "  HF_TOKEN=hf_xxx uv run exploration/upload_model_to_hf.py",
            file=sys.stderr)
        return 1

    print(f"uploading {src} ({src.stat().st_size / 1e6:.0f} MB) -> {REPO_ID}")
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(src), path_in_repo=FILENAME,
                    repo_id=REPO_ID, repo_type="model")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="model")
    print(f"done: https://huggingface.co/{REPO_ID}")
    print("The bench will now find it with no further configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
