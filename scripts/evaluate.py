"""Evaluate the fine-tuned adapter on the holdout split.

    python scripts/evaluate.py --data data/eval.jsonl --model runs/manglish-lora

Reports holdout perplexity plus a handful of visible sample generations, so
"does it actually answer in Manglish?" is observable, not just a loss number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import PromptTemplate, load_dataset  # noqa: E402

try:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    torch = None
    AutoPeftModelForCausalLM = None
    AutoTokenizer = None


def sample_generations(model, tokenizer, rows, template, count: int = 5) -> list[dict]:
    """Generate a few answers so 'does it actually respond?' is visible, not just loss."""
    results = []
    for row in rows[:count]:
        prompt = template.render(row["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"instruction": row["instruction"], "expected": row["response"], "generated": text.strip()})
    return results


def perplexity(model, tokenizer, rows, template) -> float:
    """Exponential of the mean negative log-likelihood of ground-truth answers."""
    total_nll, total_tokens = 0.0, 0
    for row in rows:
        prompt = template.render(row["instruction"])
        full = template.render(row["instruction"], row["response"])
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        full_ids = tokenizer(full, return_tensors="pt").input_ids
        # Only answer tokens count toward the loss (prompt tokens are given).
        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        with torch.no_grad():
            out = model(full_ids, labels=labels)
        n = (labels != -100).sum().item()
        total_nll += out.loss.item() * n
        total_tokens += n
    return float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/eval.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("runs/manglish-lora"))
    parser.add_argument("--generations", type=int, default=5)
    args = parser.parse_args(argv)

    rows, problems = load_dataset(args.data, strict=False)
    if problems:
        print(f"{len(problems)} rows skipped:", *problems[:5], sep="\n  ")
    template = PromptTemplate()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(str(args.model), device_map="auto")

    perplexity_value = perplexity(model, tokenizer, rows, template)
    samples = sample_generations(model, tokenizer, rows, template, args.generations)

    print(f"perplexity on {len(rows)} holdout rows: {perplexity_value:.2f}")
    for sample in samples:
        print("\ninstruction :", sample["instruction"])
        print("expected    :", sample["expected"][:120])
        print("generated   :", sample["generated"][:200])

    out = args.model / "eval_report.json"
    out.write_text(json.dumps({"rows": len(rows), "perplexity": perplexity_value, "samples": samples}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    if torch is None:
        raise SystemExit("install GPU deps with `pip install -r requirements-ml.txt`")
    raise SystemExit(main())
