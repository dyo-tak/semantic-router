"""Download the published Laya ONNX bundle from Hugging Face.

Usage: python -m semantic_router.download
"""

from __future__ import annotations

import shutil
import sys
import time
import urllib.request
from pathlib import Path

REPO = "receptron/laya-onnx"
FILES = [
    "laya_config.json",
    "laya.onnx",
    "laya.onnx.data",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
]
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models" / "laya-onnx"


def download(dest: Path = DEFAULT_DIR, force: bool = False) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tokenizer").mkdir(exist_ok=True)
    base = f"https://huggingface.co/{REPO}/resolve/main/"
    for f in FILES:
        dst = dest / f
        if dst.exists() and dst.stat().st_size > 1000 and not force:
            print(f"skip {f} (exists, {(dst.stat().st_size) / 1e6:.1f} MB)")
            continue
        t0 = time.time()
        print(f"downloading {f} ...", flush=True)
        urllib.request.urlretrieve(base + f, dst)
        print(f"  -> {dst.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s")
    return dest


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    p = download(target)
    total = sum((p / f).stat().st_size for f in FILES) / 1e6
    print(f"bundle ready at {p} ({total:.0f} MB)")
