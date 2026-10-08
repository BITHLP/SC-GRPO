from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable
from typing import Any

import datasets

from rllm.data.dataset import DatasetRegistry

PROMPT_PREFIX = "Solve the following math problem. Put your final answer within \\boxed{}."
LEGACY_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, dict):
        content = prompt.get("content")
        return _prompt_to_text(content)
    if isinstance(prompt, list):
        chunks: list[str] = []
        for message in prompt:
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = message

            if isinstance(content, list):
                for segment in content:
                    if isinstance(segment, dict):
                        if segment.get("type") == "text" and segment.get("text"):
                            chunks.append(str(segment["text"]))
                    elif segment is not None:
                        chunks.append(str(segment))
            elif content is not None:
                chunks.append(str(content))

        return "\n".join(chunk.strip() for chunk in chunks if str(chunk).strip()).strip()
    return ""


def _extract_from_spaces(example: dict[str, Any], keys: Iterable[str]) -> Any:
    spaces = [
        example,
        _as_dict(example.get("extra_info")),
        _as_dict(example.get("reward_model")),
        _as_dict(example.get("task")),
    ]

    extra_task = _as_dict(_as_dict(example.get("extra_info")).get("task"))
    if extra_task:
        spaces.append(extra_task)

    for space in spaces:
        for key in keys:
            if key not in space:
                continue
            value = space[key]
            if value is None or value == "":
                continue
            return value
    return None


def _extract_question(example: dict[str, Any]) -> str:
    # Preserve existing verl-style prompts when present.
    prompt = _extract_from_spaces(example, ("prompt",))
    if prompt is not None:
        text = _prompt_to_text(prompt)
        if text:
            return text

    question = _extract_from_spaces(
        example,
        ("question", "problem", "Problem", "Question"),
    )
    if question is None:
        raise ValueError(f"Could not find a question field in example keys: {list(example.keys())}")

    text = _prompt_to_text(question)
    if not text:
        raise ValueError(f"Question field is empty in example keys: {list(example.keys())}")
    return text


def _extract_ground_truth(example: dict[str, Any]) -> str | list[str]:
    ground_truth = _extract_from_spaces(
        example,
        ("ground_truth", "answer", "Answer", "solution", "Solution"),
    )
    if ground_truth is None:
        raise ValueError(f"Could not find a ground truth field in example keys: {list(example.keys())}")

    if isinstance(ground_truth, list):
        return [str(item).strip() for item in ground_truth if item is not None and str(item).strip()]
    return str(ground_truth).strip()


def _format_question_prompt(question: str) -> str:
    question = question.replace(LEGACY_INSTRUCTION, "").strip()
    if question.startswith(PROMPT_PREFIX):
        return question
    return f"{PROMPT_PREFIX}\n\n{question}"


def _normalize_example(example: dict[str, Any], data_source: str) -> dict[str, Any]:
    question = _format_question_prompt(_extract_question(example))
    ground_truth = _extract_ground_truth(example)
    return {
        "question": question,
        "ground_truth": ground_truth,
        "data_source": data_source,
        "ability": "math",
    }


def _dedup_key(example: dict[str, Any]) -> tuple[str, str]:
    ground_truth = example.get("ground_truth")
    return (
        str(example.get("question", "")).strip(),
        json.dumps(ground_truth, ensure_ascii=False, sort_keys=True),
    )


def _deduplicate_examples(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduplicated: list[dict[str, Any]] = []
    for example in data:
        key = _dedup_key(example)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(example)
    return deduplicated


def _load_hf_dataset(dataset_name: str, split: str = "train"):
    expanded_path = os.path.expanduser(dataset_name)
    if os.path.exists(expanded_path):
        if os.path.isdir(expanded_path):
            dataset = datasets.load_from_disk(expanded_path)
            return dataset[split] if isinstance(dataset, datasets.DatasetDict) else dataset

        extension = os.path.splitext(expanded_path)[1].lower()
        if extension == ".parquet":
            return datasets.load_dataset("parquet", data_files={split: expanded_path}, split=split)
        if extension in {".json", ".jsonl"}:
            return datasets.load_dataset("json", data_files={split: expanded_path}, split=split)
        raise ValueError(f"Unsupported local dataset file type: {expanded_path}")

    errors: list[str] = []
    for args in ((dataset_name,), (dataset_name, "default")):
        try:
            return datasets.load_dataset(*args, split=split)
        except Exception as exc:
            errors.append(f"load_dataset{args}, split={split!r}: {type(exc).__name__}: {exc}")

        try:
            dataset = datasets.load_dataset(*args)
            return dataset[split]
        except Exception as exc:
            errors.append(f"load_dataset{args})[{split!r}]: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Failed to load dataset {dataset_name!r} split {split!r}.\n" + "\n".join(errors))


def _register_source(dataset_name: str, source_label: str, split: str = "train") -> list[dict[str, Any]]:
    data = _load_hf_dataset(dataset_name, split=split)
    normalized = [_normalize_example(example, source_label) for example in data]
    return normalized


def prepare_math_data(
    train_dataset_name: str = "BytedTsinghua-SIA/DAPO-Math-17k",
    aime24_dataset_name: str = "Maxwell-Jia/AIME_2024",
    aime25_dataset_name: str = "yentinglin/aime_2025",
    deduplicate_train: bool = True,
    train_size: int | None = 4000,
    seed: int = 42,
    dataset_name: str = "dapo_math_4k",
):
    train_data = _register_source(train_dataset_name, "dapo_math")
    if deduplicate_train:
        original_train_size = len(train_data)
        train_data = _deduplicate_examples(train_data)
        print(f"Deduplicated dapo_math train: {original_train_size} -> {len(train_data)}")

    if train_size is not None:
        if not 0 < train_size <= len(train_data):
            raise ValueError("train_size must be positive and no larger than the dataset.")
        train_data = random.Random(seed).sample(train_data, train_size)

    aime24_data = _register_source(aime24_dataset_name, "aime24")
    aime25_data = _register_source(aime25_dataset_name, "aime25")

    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_data, "train")
    aime24_dataset = DatasetRegistry.register_dataset("aime24", aime24_data, "test")
    aime25_dataset = DatasetRegistry.register_dataset("aime25", aime25_data, "test")
    merged_val_dataset = DatasetRegistry.register_dataset("aime24_25", aime24_data + aime25_data, "test")

    return train_dataset, aime24_dataset, aime25_dataset, merged_val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DAPO-Math-17k training data and AIME24/AIME25 validation data")
    parser.add_argument("--train_dataset", default="BytedTsinghua-SIA/DAPO-Math-17k")
    parser.add_argument("--aime24_dataset", default="Maxwell-Jia/AIME_2024")
    parser.add_argument("--aime25_dataset", default="yentinglin/aime_2025")
    parser.add_argument("--no_deduplicate_train", action="store_true")
    parser.add_argument("--train_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="dapo_math_4k")
    args = parser.parse_args()

    train_dataset, aime24_dataset, aime25_dataset, merged_val_dataset = prepare_math_data(
        train_dataset_name=args.train_dataset,
        aime24_dataset_name=args.aime24_dataset,
        aime25_dataset_name=args.aime25_dataset,
        deduplicate_train=not args.no_deduplicate_train,
        train_size=args.train_size,
        seed=args.seed,
        dataset_name=args.dataset_name,
    )

    print(train_dataset)
    print(aime24_dataset)
    print(aime25_dataset)
    print(merged_val_dataset)
