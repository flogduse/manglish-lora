#!/usr/bin/env python3
"""Turn the raw sample corpus into validated train/eval JSONL splits.

    python scripts/prepare_data.py [--source data/sample_manglish.jsonl]

Every row goes through `app.data.validate_pair`. A clean, split, on-disk
corpus is what the notebook and `scripts/train_lora.py` consume; running this
first means bad data fails *before* a GPU is billed for it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import load_dataset, split_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/sample_manglish.jsonl"))
    parser.add_argument("--train-out", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--eval-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    rows, problems = load_dataset(args.source, strict=False)
    if problems:
        print(f"{len(problems)} sample rows dropped:")
        for problem in problems:
            print(f"  - {problem}")
    if not rows:
        print("nothing usable survived validation", file=sys.stderr)
        return 1

    split = split_dataset(rows, eval_frac=args.eval_frac, seed=args.seed)
    for path, part in ((args.train_out, split.train), (args.eval_out, split.eval)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}: {len(part)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
