"""Fine-tune a language model to emit "alert" on off-policy injections."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, DatasetDict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data_utils import load_datasets
from constants import ALERT_RESPONSE

LOGGER = logging.getLogger("train_alert_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", type=Path, required=True, help="Path to training JSONL")
    parser.add_argument("--eval_file", type=Path, help="Optional evaluation JSONL")
    parser.add_argument(
        "--model_name",
        default="google/gemma-3-4b-it",
        help="Base model name or path (defaults to google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("artifacts/alert-model"),
        help="Directory to store checkpoints and logs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for prompt + response",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for AdamW",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per-device batch size (keep small for consumer GPUs)",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Number of gradient accumulation steps for effective batch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Total number of training steps (set small for fast smoke tests)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="How often to run evaluation (set 0 to disable)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Frequency of logging to console",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable parameter-efficient fine-tuning (requires peft).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank when --use_lora is enabled",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha when --use_lora is enabled",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout when --use_lora is enabled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.0,
        help="Weight for KL divergence regularization toward a reference model",
    )
    parser.add_argument(
        "--kl_reference_model",
        default=None,
        help="Model name or path used as the KL reference (defaults to --model_name)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize_example(example: Dict[str, str]) -> Dict[str, list[int]]:
        prompt = example["prompt"].strip()
        response = example["response"].strip()

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels.extend(response_ids)
        labels = labels[: len(input_ids)]

        attention_mask = [1] * len(input_ids)
        kl_mask = 0.0 if response == ALERT_RESPONSE else 1.0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "kl_mask": kl_mask,
        }

    return dataset.map(tokenize_example, remove_columns=dataset.column_names)


class CausalCollator:
    """Pad variable-length sequences for causal LM training."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[Dict[str, list[int] | float]]):
        batch_size = len(features)
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        kl_mask = torch.tensor([f["kl_mask"] for f in features], dtype=torch.float)

        for idx, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[idx, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[idx, :length] = 1
            label_values = torch.tensor(feature["labels"], dtype=torch.long)
            labels[idx, : len(label_values)] = label_values

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "kl_mask": kl_mask,
        }


def maybe_apply_lora(model, args) -> Tuple[bool, object]:
    if not args.use_lora:
        return False, model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SystemExit("peft package is required when --use_lora is set") from exc

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return True, lora_model


class KLTrainer(Trainer):
    """Trainer with optional KL regularization toward a frozen reference model."""

    def __init__(self, *args, reference_model=None, kl_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.kl_weight = kl_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        kl_scale = inputs.pop("kl_mask", None)
        outputs = model(**inputs)
        loss = outputs.loss

        if (
            self.kl_weight > 0.0
            and self.reference_model is not None
            and labels is not None
        ):
            target_device = inputs["input_ids"].device
            if next(self.reference_model.parameters()).device != target_device:
                self.reference_model.to(target_device)
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    use_cache=False,
                )

            student_log_probs = F.log_softmax(outputs.logits, dim=-1)
            reference_probs = F.softmax(ref_outputs.logits, dim=-1)

            mask = (labels != -100).unsqueeze(-1).to(student_log_probs.dtype)
            if isinstance(kl_scale, torch.Tensor):
                kl_scale = kl_scale.to(student_log_probs.device, dtype=student_log_probs.dtype)
                mask = mask * kl_scale.view(-1, 1, 1)
            mask_sum = mask.sum()
            if mask_sum.item() > 0:
                kl = F.kl_div(
                    student_log_probs,
                    reference_probs,
                    reduction="none",
                    log_target=False,
                )
                kl = (kl * mask).sum() / mask_sum
                loss = loss + self.kl_weight * kl

        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    setup_logging()

    LOGGER.info("Loading datasets")
    raw_datasets = load_datasets(args.train_file, args.eval_file)

    LOGGER.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = build_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    LOGGER.info("Tokenizing datasets")
    tokenized_train = tokenize_dataset(raw_datasets["train"], tokenizer, args.max_length)
    dataset_dict = DatasetDict(train=tokenized_train)

    if "eval" in raw_datasets:
        tokenized_eval = tokenize_dataset(raw_datasets["eval"], tokenizer, args.max_length)
        dataset_dict["eval"] = tokenized_eval

    LOGGER.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=max(args.max_steps // 10, 1),
        evaluation_strategy="steps" if "eval" in dataset_dict and args.eval_steps else "no",
        eval_steps=args.eval_steps if args.eval_steps else None,
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps if args.eval_steps else args.logging_steps,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=["none"],
        seed=args.seed,
    )

    LOGGER.info("Preparing model for training")
    using_lora, model = maybe_apply_lora(model, args)
    reference_model = None
    if args.kl_weight > 0.0:
        reference_name = args.kl_reference_model or args.model_name
        LOGGER.info("Loading reference model for KL regularization: %s", reference_name)
        reference_model = AutoModelForCausalLM.from_pretrained(reference_name)
        reference_model.to(model.device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

    data_collator = CausalCollator(tokenizer.pad_token_id)

    trainer = KLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("eval"),
        tokenizer=tokenizer,
        reference_model=reference_model,
        kl_weight=args.kl_weight,
        data_collator=data_collator,
    )

    LOGGER.info("Starting training")
    trainer.train()
    LOGGER.info("Training completed")

    if using_lora:
        LOGGER.info("Saving LoRA adapter weights to %s", args.output_dir)
        trainer.model.save_pretrained(args.output_dir)
    else:
        LOGGER.info("Saving full model to %s", args.output_dir)
        trainer.save_model()

    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
