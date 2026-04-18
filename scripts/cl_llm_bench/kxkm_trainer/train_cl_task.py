# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.3",
#   "transformers>=4.45",
#   "peft>=0.13",
#   "datasets>=3.0",
#   "accelerate>=0.34",
#   "bitsandbytes>=0.44",
# ]
# ///
"""Standalone LoRA fine-tuning script for the CL LLM benchmark.

Runs on kxkm-ai (RTX 4090). Invoked by ``LoRATrainerReal`` in the
companion repo via SSH + ``uv run`` — the PEP 723 inline metadata above
lets ``uv`` resolve the heavy ML dependencies without a project file.

Target model: any HuggingFace causal/SeqCls repo (default: ``Qwen/Qwen3-4B``).
The caller picks the model via ``--base-model``; nothing here is hardwired
to a specific architecture.

Data format: JSONL, one object per line, keys ``text`` (str) and
``label`` (int, 0 or 1). 2-class GLUE-style tasks only (SST-2, CoLA,
BoolQ cast to binary, RTE).

Outputs under ``--output-dir``:
  - ``lora_adapter/`` — PEFT LoRA weights
  - ``manifest.json`` — run metadata + final eval accuracy
  - ``_hf/`` — HF Trainer workspace (checkpoints discarded, logs kept)

The manifest JSON is also printed to stdout so the caller can parse it
without reading the file back over SSH.

Two additional modes support the multi-task CL sequence driven by
``run_cl_bench._run_real``:

- ``--resume-adapter PATH``: loads an existing LoRA adapter directory
  (previously produced by ``model.save_pretrained(...)``) on top of the
  4-bit base, then continues training (or evaluates, with
  ``--eval-only``). When set, the fresh ``LoraConfig`` wrap is skipped —
  the adapter config travels with the saved directory.
- ``--eval-only``: skips ``trainer.train()`` and runs ``trainer.evaluate()``
  on the full ``--dataset`` (no 80/20 split). Useful for measuring the
  final adapter's accuracy on every prior task after the CL sequence to
  compute forgetting = immediate - final.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ---- Setup ----------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-model", type=str, required=True, help="HF repo ID or local path for the base model."
    )
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL file with {'text': str, 'label': int} per line.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume-adapter",
        type=Path,
        default=None,
        help=(
            "Path to an existing LoRA adapter directory (from a previous "
            "model.save_pretrained). Continues training from these weights. "
            "Skips the fresh LoraConfig wrap — adapter config travels with "
            "the saved directory."
        ),
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Skip training; evaluate --resume-adapter (or freshly initialized "
            "LoRA) on the full --dataset. Used to measure final-adapter "
            "accuracy on prior tasks after a CL sequence."
        ),
    )
    return p.parse_args()


# ---- Model ----------------------------------------------------------------


def build_model_and_tokenizer(base_model: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 4-bit quantization via bitsandbytes (NF4 + double-quant + bfloat16 compute).
    # Lets Qwen3-4B fit in ~4 GB VRAM; LoRA trains on top of the frozen 4-bit base.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Prepare base for k-bit training (cast norms to fp32, enable grad on inputs).
    model = prepare_model_for_kbit_training(model)
    # Make sure the model knows about the pad token for classification head.
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def wrap_with_lora(model: Any, rank: int, alpha: int) -> Any:
    """Apply LoRA. Qwen-style modules first, BERT-style as fallback."""
    for target_modules in (["q_proj", "v_proj"], ["query", "value"]):
        try:
            lora_cfg = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                task_type=TaskType.SEQ_CLS,
            )
            return get_peft_model(model, lora_cfg)
        except ValueError:
            continue
    msg = (
        "Could not attach LoRA: neither ['q_proj','v_proj'] nor "
        "['query','value'] found in the model. Specify a compatible arch."
    )
    raise ValueError(msg)


def attach_adapter(
    model: Any, rank: int, alpha: int, resume_adapter: Path | None, is_trainable: bool
) -> Any:
    """Attach either a fresh LoRA wrapper or load a pre-existing adapter.

    - If ``resume_adapter`` is None: wrap the base with a new ``LoraConfig``
      and return a trainable PeftModel.
    - If ``resume_adapter`` is set: load the saved adapter via
      ``PeftModel.from_pretrained``. ``is_trainable=False`` freezes it
      (eval-only); otherwise it continues training from these weights.
    """
    if resume_adapter is None:
        return wrap_with_lora(model, rank, alpha)
    return PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=is_trainable)


# ---- Data -----------------------------------------------------------------


def load_and_tokenize(dataset_path: Path, tokenizer: Any) -> tuple[Any, Any, int]:
    """Return (train_ds, eval_ds, n_samples). 80/20 split for eval."""
    raw = load_dataset("json", data_files=str(dataset_path))["train"]
    n_total = len(raw)
    split_idx = max(1, int(n_total * 0.8))

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(batch["text"], truncation=True, max_length=128, padding=False)
        enc["labels"] = batch["label"]
        return enc

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    train_ds = tokenized.select(range(split_idx))
    eval_ds = tokenized.select(range(split_idx, n_total))
    return train_ds, eval_ds, n_total


def tokenize_full(dataset_path: Path, tokenizer: Any) -> tuple[Any, int]:
    """Return (full_ds, n_samples). No split — used for eval-only mode."""
    raw = load_dataset("json", data_files=str(dataset_path))["train"]
    n_total = len(raw)

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(batch["text"], truncation=True, max_length=128, padding=False)
        enc["labels"] = batch["label"]
        return enc

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    return tokenized, n_total


# ---- Eval -----------------------------------------------------------------


def compute_accuracy(eval_preds: Any) -> dict[str, float]:
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


# ---- Main -----------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Model + adapter (fresh LoRA or loaded from --resume-adapter).
    model, tokenizer = build_model_and_tokenizer(args.base_model)
    model = attach_adapter(
        model,
        args.lora_rank,
        args.lora_alpha,
        args.resume_adapter,
        is_trainable=not args.eval_only,
    )

    if args.eval_only:
        # Full dataset as eval set; no training.
        eval_ds, n_samples = tokenize_full(args.dataset, tokenizer)
        training_args = TrainingArguments(
            output_dir=str(args.output_dir / "_hf"),
            per_device_eval_batch_size=args.batch_size,
            save_strategy="no",
            logging_strategy="no",
            seed=args.seed,
            report_to=[],
            bf16=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            compute_metrics=compute_accuracy,
        )
        eval_metrics = trainer.evaluate()
        eval_accuracy = float(eval_metrics.get("eval_accuracy", eval_metrics.get("accuracy", 0.0)))

        manifest: dict[str, Any] = {
            "status": "ok",
            "mode": "eval",
            "base_model": args.base_model,
            "n_samples": n_samples,
            "seed": args.seed,
            "eval_accuracy": eval_accuracy,
            "resume_adapter": str(args.resume_adapter) if args.resume_adapter else None,
            "timestamp": time.time(),
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
        return

    # Training path (optionally resumed from a prior adapter).
    train_ds, eval_ds, n_samples = load_and_tokenize(args.dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "_hf"),
        max_steps=args.n_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, args.n_steps // 10),
        seed=args.seed,
        report_to=[],
        bf16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_accuracy,
    )
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(args.output_dir / "lora_adapter")

    # Evaluate
    eval_metrics = trainer.evaluate()
    eval_accuracy = float(eval_metrics.get("eval_accuracy", eval_metrics.get("accuracy", 0.0)))

    # Manifest
    manifest = {
        "status": "ok",
        "mode": "real",
        "base_model": args.base_model,
        "n_steps": args.n_steps,
        "n_samples": n_samples,
        "seed": args.seed,
        "eval_accuracy": eval_accuracy,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "resume_adapter": str(args.resume_adapter) if args.resume_adapter else None,
        "timestamp": time.time(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
