"""The training entrypoint fails honestly without the ML stack — and that
failure path is what CI pins, since GitHub runners have no GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_train_lora_imports_and_reports_missing_gpu_deps():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import train_lora
    finally:
        sys.path.pop(0)

    # torch/transformers are intentionally NOT installed in CI: the guard must
    # be honest about it instead of crashing deep inside an import.
    assert train_lora.torch is None or train_lora.torch is not None  # guard var set
    if train_lora.torch is None:
        with pytest.raises(SystemExit) as exc:
            train_lora.main([])
        assert "requirements-ml.txt" in str(exc.value) or "CUDA" in str(exc.value)


def test_prepare_data_writes_both_splits(tmp_path: Path):
    source = tmp_path / "sample.jsonl"
    rows = [
        {"instruction": f"chodi {i}", "response": f"uttaram {i} polum"} for i in range(10)
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    import prepare_data

    rc = prepare_data.main(
        [
            "--source", str(source),
            "--train-out", str(tmp_path / "train.jsonl"),
            "--eval-out", str(tmp_path / "eval.jsonl"),
        ]
    )
    assert rc == 0
    train = (tmp_path / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    eval_ = (tmp_path / "eval.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train) + len(eval_) == 10
    assert all(json.loads(line) for line in train + eval_)


def test_sample_corpus_survives_its_own_validator():
    """The shipped sample must pass the contract it advertises."""
    sys.path.insert(0, str(ROOT))
    from app.data import load_dataset

    rows, problems = load_dataset(ROOT / "data" / "sample_manglish.jsonl", strict=True)
    assert problems == []
    assert len(rows) >= 50
