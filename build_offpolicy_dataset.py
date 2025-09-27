"""Generate SFT data where off-policy injections are labeled with an alert response."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import ALERT_RESPONSE

DEFAULT_POLICY_PROMPT = (
    "You are a meticulous math tutor. Solve the question step by step and give a final boxed answer."
)
DEFAULT_INJECTION_PROMPT = (
    "You are a different language model whose reasoning is being spliced into someone else's work. "
    "Write EXACTLY one sentence continuing the math solution in your own voice."
)

DEFAULT_MATH_DATASET = "openai/gsm8k"
DEFAULT_MATH_CONFIG = "main"
DEFAULT_PROBLEM_FIELD = "question"


@dataclass
class ConversationSample:
    prompt: str
    response: str

    def to_json(self) -> str:
        return json.dumps({"prompt": self.prompt, "response": self.response}, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--math_split",
        default="train[:50]",
        help="Split slice to load from the math dataset (HF slicing syntax).",
    )
    parser.add_argument(
        "--policy_model",
        default="google/gemma-3-4b-it",
        help="Base policy-aligned model used to generate on-policy answers.",
    )
    parser.add_argument(
        "--injection_model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Secondary model used to synthesize off-policy injections.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (e.g., cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--policy_max_new_tokens",
        type=int,
        default=256,
        help="Max tokens when generating the on-policy completion.",
    )
    parser.add_argument(
        "--injection_max_new_tokens",
        type=int,
        default=64,
        help="Max tokens when generating the off-policy injection sentence.",
    )
    parser.add_argument(
        "--injection_temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for the injection model.",
    )
    parser.add_argument(
        "--injection_top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p for the injection model.",
    )
    parser.add_argument(
        "--partial_ratio",
        type=float,
        default=0.5,
        help="Approximate fraction of the on-policy answer to keep before the injection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_train",
        type=Path,
        default=Path("data/generated/train.jsonl"),
        help="Where to write the training JSONL file.",
    )
    parser.add_argument(
        "--output_eval",
        type=Path,
        help="Optional path for an evaluation JSONL file.",
    )
    parser.add_argument(
        "--eval_fraction",
        type=float,
        default=0.1,
        help="Portion of samples to place into the eval file when --output_eval is set.",
    )
    parser.add_argument(
        "--policy_prompt",
        default=DEFAULT_POLICY_PROMPT,
        help="System prompt fed to the policy model.",
    )
    parser.add_argument(
        "--injection_prompt",
        default=DEFAULT_INJECTION_PROMPT,
        help="Instruction prefix for the injection model.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the number of source math problems processed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_causal_lm(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def ensure_sentence(text: str) -> str:
    stripped = text.strip().replace("\n", " ")
    if not stripped:
        return stripped
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    return sentences[0].strip()


def take_partial_answer(answer: str, ratio: float) -> str:
    clean = answer.strip()
    if not clean:
        return clean
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if len(sentences) > 1:
        keep = max(1, int(len(sentences) * ratio))
        return " ".join(sentences[:keep]).strip()
    tokens = clean.split()
    keep_tokens = max(1, int(len(tokens) * ratio))
    return " ".join(tokens[:keep_tokens]).strip()


def generate_text(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    generated = output[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_guard_prompt(question: str, partial: str | None = None, injection: str | None = None) -> str:
    blocks = [f"Problem:\n{question.strip()}"]
    if partial:
        blocks.append(f"Current reasoning snippet:\n{partial.strip()}")
    if injection:
        blocks.append(f"Latest continuation:\n{injection.strip()}")
    prompt_body = "\n\n".join(blocks)
    return f"{prompt_body}\n\nAssistant:"


def build_policy_prompt(question: str, policy_prompt: str) -> str:
    return f"System: {policy_prompt}\nUser: {question.strip()}\nAssistant:"


def build_injection_prompt(partial: str, injection_prompt: str) -> str:
    return (
        f"{injection_prompt}\n\nPartial solution (provided by another model):\n{partial.strip()}\n\n" "Injected sentence:"
    )


def iter_problems(dataset, field: str, limit: int | None) -> Iterable[str]:
    count = 0
    for item in dataset:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        yield value.strip()
        count += 1
        if limit is not None and count >= limit:
            break


def build_samples(args: argparse.Namespace) -> List[ConversationSample]:
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = load_dataset(DEFAULT_MATH_DATASET, DEFAULT_MATH_CONFIG, split=args.math_split)
    policy_model, policy_tokenizer = load_causal_lm(args.policy_model, device)
    injection_model, injection_tokenizer = load_causal_lm(args.injection_model, device)

    total_problems = len(dataset)
    if args.max_samples is not None:
        total_problems = min(total_problems, args.max_samples)

    samples: List[ConversationSample] = []
    for question in tqdm(
        iter_problems(dataset, DEFAULT_PROBLEM_FIELD, args.max_samples),
        total=total_problems,
        desc="Generating samples",
    ):
        policy_prompt = build_policy_prompt(question, args.policy_prompt)
        policy_answer = generate_text(
            model=policy_model,
            tokenizer=policy_tokenizer,
            prompt=policy_prompt,
            max_new_tokens=args.policy_max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        policy_answer = policy_answer.strip()
        if not policy_answer:
            continue

        partial = take_partial_answer(policy_answer, args.partial_ratio)
        if not partial:
            continue

        guard_prompt_normal = build_guard_prompt(question)
        samples.append(ConversationSample(prompt=guard_prompt_normal, response=policy_answer))

        injection_prompt = build_injection_prompt(partial, args.injection_prompt)
        injection_sentence = generate_text(
            model=injection_model,
            tokenizer=injection_tokenizer,
            prompt=injection_prompt,
            max_new_tokens=args.injection_max_new_tokens,
            do_sample=True,
            temperature=args.injection_temperature,
            top_p=args.injection_top_p,
        )
        injection_sentence = ensure_sentence(injection_sentence)
        if not injection_sentence:
            continue

        guard_prompt_injected = build_guard_prompt(
            question, partial=partial, injection=injection_sentence
        )
        samples.append(
            ConversationSample(prompt=guard_prompt_injected, response=ALERT_RESPONSE)
        )

    random.shuffle(samples)
    return samples


def split_samples(
    samples: Sequence[ConversationSample], eval_fraction: float
) -> tuple[List[ConversationSample], List[ConversationSample]]:
    if eval_fraction <= 0.0:
        return list(samples), []
    eval_size = int(math.ceil(len(samples) * eval_fraction))
    eval_samples = list(samples[:eval_size])
    train_samples = list(samples[eval_size:])
    return train_samples, eval_samples


def write_samples(path: Path, samples: Sequence[ConversationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")


def main() -> None:
    args = parse_args()

    samples = build_samples(args)
    if not samples:
        raise SystemExit("No samples were generated; check dataset and model configuration.")

    train_samples, eval_samples = split_samples(samples, args.eval_fraction if args.output_eval else 0.0)

    write_samples(args.output_train, train_samples)
    print(f"Wrote {len(train_samples)} training samples to {args.output_train}")

    if args.output_eval and eval_samples:
        write_samples(args.output_eval, eval_samples)
        print(f"Wrote {len(eval_samples)} eval samples to {args.output_eval}")
    elif args.output_eval:
        print("Skipping eval output because eval_fraction produced 0 samples.")


if __name__ == "__main__":
    main()
