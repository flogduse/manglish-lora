"""The dataset contract, exercised on a laptop with zero ML dependencies."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.data import (
    DatasetError,
    PromptTemplate,
    load_dataset,
    split_dataset,
    stats,
    validate_pair,
)


def test_validate_pair_normalises_and_passes_clean_rows():
    clean, notes = validate_pair({"instruction": "  engane und  ", "response": "nannayi undu "}, 1)
    assert clean == {"instruction": "engane und", "response": "nannayi undu"}
    assert isinstance(notes, list)


def test_validate_pair_rejects_structural_junk():
    with pytest.raises(DatasetError):
        validate_pair(["not", "a", "dict"], 7)          # not an object
    with pytest.raises(DatasetError):
        validate_pair({"instruction": "hi"}, 7)          # missing response
    with pytest.raises(DatasetError):
        validate_pair({"instruction": "", "response": "x"}, 7)
    with pytest.raises(DatasetError):
        validate_pair({"instruction": "a", "response": "b"}, 7)  # absurdly short


def test_validate_pair_rejects_non_latin_script():
    # Manglish is Latin-script by definition: CJK, Cyrillic, or Malayalam-script
    # characters mean the row is not what it claims to be.
    with pytest.raises(DatasetError):
        validate_pair({"instruction": "engane und", "response": "混合文字 aanu"}, 3)
    with pytest.raises(DatasetError):
        validate_pair({"instruction": "അയ്യോ engane", "response": "nannayi"}, 3)
    with pytest.raises(DatasetError):
        validate_pair({"instruction": "привет und", "response": "nannayi"}, 3)


def test_validate_pair_allows_punctuation_and_latin_extended():
    clean, _ = validate_pair(
        {"instruction": "engane und — sukhano?", "response": "café,aané style ok aanu"},
        1,
    )
    assert clean["instruction"] == "engane und — sukhano?"


def test_load_dataset_deduplicates_and_reports_skips(tmp_path: Path):
    payload = [
        {"instruction": "engane und", "response": "nannayi undu"},
        {"instruction": "engane und", "response": "nannayi undu"},   # exact duplicate
        {"instruction": "bad", "response": None},                   # broken row
    ]
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in payload), encoding="utf-8")

    rows, problems = load_dataset(path, strict=False)
    assert len(rows) == 1
    assert any("duplicate" in p for p in problems)
    assert any("missing response" in p for p in problems)

    # strict mode must fail the run loudly instead of skipping
    with pytest.raises(DatasetError):
        load_dataset(path, strict=True)


def test_prompt_template_is_a_stable_contract():
    template = PromptTemplate(system="s")
    full = template.render("entha?", "answer aanu")
    assert full == "System: s\nUser: entha?\nAssistant: answer aanu"
    assert template.render("entha?").endswith("Assistant:")
    # a render without an answer must be a strict prefix of a full render
    assert full.startswith(template.render("entha?"))


def test_stats_and_split_are_deterministic():
    rows = [
        {"instruction": f"q{i}", "response": f"a{i} and words", "lang": "ml"} for i in range(20)
    ]
    summary = stats(rows)
    assert summary["pairs"] == 20 and summary["languages"] == ["ml"]

    a, b = split_dataset(rows, eval_frac=0.2, seed=17), split_dataset(rows, eval_frac=0.2, seed=17)
    assert [r["instruction"] for r in a.train] == [r["instruction"] for r in b.train]
    assert len(a.eval) == len(b.eval) == 4
    assert not ({r["instruction"] for r in a.train} & {r["instruction"] for r in a.eval})


def test_split_rejects_impossibly_large_eval_fraction():
    with pytest.raises(ValueError):
        split_dataset([{"instruction": "a", "response": "b"}], eval_frac=1.0)
