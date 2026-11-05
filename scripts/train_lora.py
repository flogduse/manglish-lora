"""Fine-tune a compact LLM on Manglish chat with a LoRA adapter.

    python scripts/train_lora.py --data data/train.jsonl --base Trelis/Llama-2-7b-hf

This file is the GPU-side half. It only imports torch/transformers/peft when it
runs, so the dataset pipeline (app/data.py) stays testable without a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.data import PromptTemplate, load_dataset, split_dataset, stats  # noqa: E402


try:
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover - import guard, exercised in CI
    torch = None
    transformers = None
    Dataset = None
    LoraConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    DataCollatorForLanguageModeling = None
    Trainer = None
    TrainingArguments = None
    _missing = exc


DEFAULT_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DROPOUT = 0.05


def build_model(base_model: str, four_bit: bool):
    quantisation = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if four_bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantisation,
        torch_dtype=torch.bfloat16 if not four_bit else None,
        device_map="auto" if four_bit else None,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model


def default_targets(model):
    """Linear module names for a Llama-arch attention stack."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def preprocess(rows, tokenizer, template: PromptTemplate, max_length: int):
    def tokenise(batch):
        texts = [template.render(r["instruction"], r["response"]) for r in batch]
        return tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    return rows.map(tokenise, batched=True, remove_columns=[c for c in rows.column_names])


def train(args: argparse.Namespace) -> dict:
    rows, _problems = load_dataset(Path(args.data))  # strict: fail loudly pre-GPU
    split = split_dataset(rows, eval_frac=args.eval_frac, seed=args.seed)
    print(f"train pairs: {len(split.train)}   eval pairs: {len(split.eval)}")
    print(json.dumps(stats(rows), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = PromptTemplate()
    train_hf = Dataset.from_list(split.train)
    eval_hf = Dataset.from_list(split.eval)
    train_hf = preprocess(train_hf, tokenizer, template, args.max_length)
    eval_hf = preprocess(eval_hf, tokenizer, template, args.max_length)

    model = build_model(args.base, four_bit=args.four_bit)
    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=default_targets(model),
    )
    # `get_peft_model` lives in peft.core; keep the long name import free at top
    from peft import get_peft_model

    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps" if args.eval_steps else "no",
        eval_steps=args.eval_steps or None,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=eval_hf,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.out)  # adapter only (LoRA), not the frozen base
    tokenizer.save_pretrained(args.out)
    print(json.dumps({"eval_loss": metrics.get("eval_loss")}, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--out", type=Path, default=Path("runs/manglish-lora"))
    p.add_argument("--r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-frac", type=float, default=0.05)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--no-four-bit", dest="four_bit", action="store_false")
    return p


def main(argv: list[str] | None = None) -> int:
    if torch is None:  # import guard keeps the failure visible at the right place
        raise SystemExit(
            "GPU training dependencies not installed.\n"
            "  pip install -r requirements-ml.txt     # transformers/peft/torch stack\n"
            "and run this on a CUDA machine (see notebooks/colab_train.ipynb)."
        )
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
