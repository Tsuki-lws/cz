#!/usr/bin/env python

import argparse
import os
import shutil
from pathlib import Path


MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "video_preprocessor_config.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = Path(args.base_model)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in MODEL_FILES:
        src = base / name
        if src.exists():
            shutil.copy2(src, output / name)
            copied.append(name)

    for src in base.glob("*.model"):
        shutil.copy2(src, output / src.name)
        copied.append(src.name)

    print(f"Prepared {output}")
    print("Copied:", ", ".join(sorted(copied)) if copied else "none")


if __name__ == "__main__":
    main()
