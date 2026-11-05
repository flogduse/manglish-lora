# Manglish LoRA Harness

A complete, tested **harness** for fine-tuning a small LLM on Manglish — Malayalam
written in the Latin script — with a LoRA adapter. It piggybacks on the chat shape
of the asl/gesture pipeline style: validate first, render a stable prompt
contract, split deterministically, and only then touch a GPU.

**Status: harness ready, base model not yet fine-tuned.** The GPU training run is
the next step (see [Running the fine-tune](#running-the-fine-tune) and
`notebooks/colab_train.ipynb`). No trained weights are published here yet, and
this README will not pretend otherwise.

```
data/sample_manglish.jsonl      74 validated Manglish chat pairs (see provenance)
        │
        ▼
app/data.py                     validate → dedupe → render → split  (pure, tested)
        │
        ▼
scripts/prepare_data.py         data/train.jsonl + data/eval.jsonl
        │
        ▼
scripts/train_lora.py           LoRA on a compact base model (GPU only)
        │
        ▼
scripts/evaluate.py             holdout perplexity + visible sample generations
```

## Why the harness is testable without a GPU

Everything below the first GPU call is pure Python:

- `app/data.py` — the **dataset contract**. `validate_pair` rejects non-Latin
  scripts (a "Manglish" corpus containing CJK or Cyrillic characters is not a
  Manglish corpus), structural junk, and duplicates; it normalises Unicode so
  different keyboards can't quietly fragment the data. `PromptTemplate` is the
  single source of truth for chat rendering — training, evaluation and demos
  cannot drift apart. `split_dataset` is seeded, so the same corpus always
  produces the same train/eval boundary.
- `tests/` — 11 tests run in CI on a CPU-only runner. **No torch is installed on
  CI by design**: the import guard in `scripts/train_lora.py` is itself part of
  what's pinned, so the failure mode "torch missing" is a clear, tested message —
  not a stack trace from deep inside transformers.

## Running the fine-tune (the one remaining step)

This needs a CUDA machine — a Colab T4 is enough for a 1.1B–3B base model:

1. Open `notebooks/colab_train.ipynb` (or run the commands by hand):

```bash
pip install -r requirements-ml.txt
python scripts/prepare_data.py                    # validate + split
python scripts/train_lora.py \
    --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data data/train.jsonl --epochs 3 --r 16 --alpha 32
python scripts/evaluate.py --data data/eval.jsonl --model runs/manglish-lora
```

2. Defaults: 4-bit quantised base (NF4), LoRA r=16/α=32 on `q/k/v/o_proj`,
   cosine LR schedule, bf16 when CUDA is available.
3. Evaluation reports **holdout perplexity plus generated samples**, so the
   qualitative "does it answer in Manglish?" signal is visible next to the loss.

Honest workload note: 74 pairs is a **sample**, not a training corpus. The
harness handles thousands of pairs unchanged, but a model fine-tuned only on
this sample will parrot it. Growing a few thousand real pairs (see below) is
what makes the run meaningful.

## Dataset provenance & how to grow the corpus

`data/sample_manglish.jsonl` is a **hand-written + machine-assisted sample**:
pairs were authored for this project and passed through `app.data.validate_pair`
(16 draft rows with non-Latin-script contamination were rejected by the
validator before shipping). It is *not* scraped, and there is **no third-party
training data in this repo yet** — when a real corpus is added, it must carry
its source and licence in this section, per row if needed.

Candidate sources for the real corpus (all need licence review before use):

- **Human-verified transliteration pairs** from public-domain Malayalam prose
  (e.g. Sahitya Akademi digitised works) transliterated to Manglish.
- **Open Malayalam dialogue corpora** (OPUS Malayalam subsets — CC-BY /
  CC0 variants; check the specific release).
- **Self-collected chats** — with the user's own WhatsApp/social data only,
  and only for personal-model use.

## Layout

```
app/data.py                 dataset contract (validate/render/split) — pure
scripts/prepare_data.py     sample → validated train/eval splits
scripts/train_lora.py       GPU LoRA fine-tune (guarded import, honest failure)
scripts/evaluate.py         holdout perplexity + sample generations
notebooks/colab_train.ipynb one-click T4 recipe
tests/                      11 CPU-only tests (CI-pinned)
data/sample_manglish.jsonl  74 validated pairs — provenance above
```

## Design decisions worth mentioning in an interview

- **Validate before the GPU.** `prepare_data.py` exists so bad data fails on a
  laptop in seconds, not after 20 minutes of Colab session-anxiety.
- **One prompt template, three consumers.** Training, eval and demos all render
  through `PromptTemplate`; a mismatch between them is a classic fine-tune bug
  (the model "works" in eval but the chat prefix differs in prod).
- **Non-Latin script is a hard contract violation.** A transliterated-language
  dataset silently absorbing another script is data rot; it fails loudly here.
- **Split determinism.** `seed=17` everywhere; retraining reproduces the same
  eval set, so loss numbers stay comparable across runs.

## Roadmap

- [ ] Run the actual fine-tune on Colab (TinyLlama 1.1B first)
- [ ] Grow the corpus to a few thousand pairs with documented sources
- [ ] Add a `merge_and_export` step (adapter → merged model → GGUF)
- [ ] Side-by-side demo: base vs adapter responses in a Gradio cell

## License

MIT — see [LICENSE](LICENSE).
