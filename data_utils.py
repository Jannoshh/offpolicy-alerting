"""Dataset utilities for off-policy alert fine-tuning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

from datasets import Dataset


@dataclass
class ConversationExample:
    """Single conversational example for supervised fine-tuning."""

    prompt: str
    response: str

    def to_dict(self) -> Mapping[str, str]:
        return {"prompt": self.prompt, "response": self.response}


def load_jsonl(path: Path) -> List[ConversationExample]:
    """Load `ConversationExample` items from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: List[ConversationExample] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc

            prompt = data.get("prompt")
            response = data.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(
                    f"Expected `prompt` and `response` strings on line {line_no} in {path}."
                )

            examples.append(ConversationExample(prompt=prompt, response=response))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def load_dataset_jsonl(path: Path) -> Dataset:
    """Return a Hugging Face `Dataset` ready for supervised fine-tuning."""
    examples = load_jsonl(path)
    return Dataset.from_list([example.to_dict() for example in examples])


def load_datasets(train_path: Path, eval_path: Path | None = None) -> Mapping[str, Dataset]:
    """Load train/eval datasets as a mapping compatible with `Trainer`."""
    datasets = {"train": load_dataset_jsonl(train_path)}
    if eval_path:
        datasets["eval"] = load_dataset_jsonl(eval_path)
    return datasets


def iter_dialogues(dataset: Dataset) -> Iterable[ConversationExample]:
    """Iterate over `ConversationExample` items in an in-memory dataset."""
    for item in dataset:
        yield ConversationExample(prompt=item["prompt"], response=item["response"])
