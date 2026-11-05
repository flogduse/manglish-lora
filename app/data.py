"""Pure dataset logic: validate Manglish chat pairs, render prompts, compute stats.

Kept free of `transformers` on purpose. The whole point of this module is that
the dataset contract can be tested (and broken loudly) on a laptop, without
downloading a single model weight.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Manglish (Malayalam written in the Latin alphabet) + code-mixed English.
# A pair is judged *plausible* when it is: non-empty, visibly Latin-script
# Manglish, has a conversational shape, and either language contributed words.
SUPPORTED_LANGS = ("ml", "en")
_QUESTIONISH = re.compile(r"[?:]|^(e[nh]|what|why|how|who|when|where|can|do|did|entha|engane|enthu|percentu)\b", re.I)
_WORD = re.compile(r"[A-Za-z][A-Za-z']*", re.UNICODE)
# Non-Latin scripts are contamination (CJK, Cyrillic, Malayalam-script U+0D00+,
# etc.). Allowed beyond ASCII: Latin-extended letters and generic punctuation
# (em dashes, typographic quotes) — Manglish text still uses those. The check is
# codepoint-range based, so no extra regex dependency is needed.
_ALLOWED_RANGES = (
    (0x00C0, 0x024F),   # Latin Extended-A/B
    (0x2000, 0x206F),   # General Punctuation
)


def _script_pure(value: str) -> bool:
    for ch in value:
        cp = ord(ch)
        if cp < 128:
            continue
        if not any(lo <= cp <= hi for lo, hi in _ALLOWED_RANGES):
            return False
    return True


class DatasetError(ValueError):
    """A row violates the dataset contract, with the offending line number."""


def _normalise(text: str) -> str:
    # Unicode normalisation is not cosmetic: Malayalam-in-Latin data written on
    # different keyboards produces visually identical, byte-different strings,
    # which would quietly duplicate rows or fragment vocabulary statistics.
    return unicodedata.normalize("NFC", (text or "").strip())


def validate_pair(row: dict, index: int = 0) -> tuple[dict, list[str]]:
    """Validate one row; return `(clean_row, notes)` or raise `DatasetError`.

    Notes are soft findings (e.g. mostly-English) that don't disqualify the row.
    """
    if not isinstance(row, dict):
        raise DatasetError(f"line {index}: not an object")
    missing = [k for k in ("instruction", "response") if not row.get(k)]
    if missing:
        raise DatasetError(f"line {index}: missing {', '.join(missing)}")

    instruction = _normalise(str(row["instruction"]))
    response = _normalise(str(row["response"]))
    for key, value in (("instruction", instruction), ("response", response)):
        if not value:
            raise DatasetError(f"line {index}: empty {key}")
        if len(value) < 2:
            raise DatasetError(f"line {index}: {key} too short ({value!r})")

    notes: list[str] = []
    for key, value in (("instruction", instruction), ("response", response)):
        if not _script_pure(value):
            raise DatasetError(f"line {index}: non-Latin script in {key} (Manglish must be Latin-script)")
    words = _WORD.findall(instruction + " " + response)
    if not words:
        raise DatasetError(f"line {index}: no Latin-script words (is this Manglish?)")
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 9.5:
        notes.append("suspiciously long words — may be a transliteration of a long Malayalam phrase")
    if not _QUESTIONISH.search(instruction) and len(instruction.split()) <= 2:
        notes.append("instruction looks very short")
    if row.get("lang") not in (None,) and str(row.get("lang")).lower() not in SUPPORTED_LANGS:
        notes.append("lang field is not in the supported list; kept as-is")

    clean = {"instruction": instruction, "response": response}
    if row.get("lang"):
        clean["lang"] = str(row["lang"]).lower()
    return clean, notes


def load_dataset(path: Path, strict: bool = True) -> tuple[list[dict], list[str]]:
    """Load a JSONL file of chat pairs.

    Returns `(rows, problems)`. With `strict=True` a single bad line fails the
    run; with `strict=False` bad lines are skipped and their reason is returned
    so the training run can surface what it silently threw away.
    """
    rows: list[dict] = []
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row, _notes = validate_pair(json.loads(line), i)
            except DatasetError as exc:
                problems.append(str(exc))
                if strict:
                    raise
                continue
            except json.JSONDecodeError as exc:
                problems.append(f"line {i}: invalid JSON ({exc})")
                if strict:
                    raise DatasetError(f"line {i}: invalid JSON ({exc})") from exc
                continue
            key = (row["instruction"], row["response"])
            if key in seen:
                problems.append(f"line {i}: exact duplicate dropped")
                continue
            seen.add(key)
            rows.append(row)
    return rows, problems


@dataclass
class PromptTemplate:
    """The chat template is a *contract*: training and eval must render identically.

    `Role markers` are deliberately kept outside the model's ability to confuse
    (they are string-literal separators, not learned tokens).
    """

    system: str = "Nee oru Manglish chat assistant aanu. User Manglish il chodikkum, ninna aavshyamulla bhasha il uttaram tharanam."

    def render(self, instruction: str, response: str | None = None) -> str:
        parts = [f"System: {self.system}", f"User: {instruction}"]
        if response is not None:
            parts.append(f"Assistant: {response}")
        else:
            parts.append("Assistant:")
        return "\n".join(parts)


def stats(rows: list[dict]) -> dict:
    responses = [r["response"] for r in rows]
    lengths = [len(r.split()) for r in responses]
    return {
        "pairs": len(rows),
        "avg_response_words": round(sum(lengths) / len(lengths), 1) if rows else 0,
        "max_response_words": max(lengths) if rows else 0,
        "languages": sorted({r.get("lang", "ml") for r in rows}),
    }


@dataclass
class Split:
    train: list[dict] = field(default_factory=list)
    eval: list[dict] = field(default_factory=list)


def split_dataset(rows: list[dict], eval_frac: float = 0.05, seed: int = 17) -> Split:
    import random

    if not 0 < eval_frac < 1:
        raise ValueError("eval_frac must be in (0, 1)")
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_eval = max(1, round(len(shuffled) * eval_frac))
    return Split(train=shuffled[n_eval:], eval=shuffled[:n_eval])
