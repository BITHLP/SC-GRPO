import argparse
import json
import os
from typing import Any

from rllm.data.dataset import DatasetRegistry
from rllm.environments.appworld.signal_patch import apply_signal_patch


def _load_task_ids(split: str) -> list[str]:
    task_ids_file = os.getenv("APPWORLD_TASK_IDS_FILE")
    if task_ids_file:
        with open(task_ids_file, encoding="utf-8") as handle:
            splits = json.load(handle)
        if split not in splits:
            raise ValueError(f"Split {split!r} absent from APPWORLD_TASK_IDS_FILE.")
        return list(splits[split])
    apply_signal_patch(verbose=False)
    from appworld import load_task_ids

    return list(load_task_ids(split))


def _maybe_limit(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return records
    return records[:limit]


def _build_records(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    records = [
        {
            "task_id": task_id,
            "data_source": "appworld",
            "uid": f"appworld_{split}_{task_id}",
            "source_split": split,
        }
        for task_id in _load_task_ids(split)
    ]
    return _maybe_limit(records, limit)


def prepare_appworld_data(
    train_split: str = "train",
    val_split: str = "dev",
    train_size: int | None = None,
    val_size: int | None = None,
    dataset_name: str = "appworld_official_dev",
):
    train_data = _build_records(train_split, train_size)
    val_data = _build_records(val_split, val_size)

    if len(train_data) == 0:
        raise RuntimeError(f"No AppWorld train tasks loaded from split {train_split!r}.")
    if len(val_data) == 0:
        raise RuntimeError(f"No AppWorld validation tasks loaded from split {val_split!r}.")

    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_data, "train")
    val_dataset = DatasetRegistry.register_dataset(dataset_name, val_data, "test")
    return train_dataset, val_dataset


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip().lower() in {"", "none", "null", "-1"}:
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="Prepare AppWorld official train/dev datasets for rLLM.")
    parser.add_argument("--dataset_name", type=str, default="appworld_official_dev")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="dev")
    parser.add_argument("--train_size", type=_parse_optional_int, default=None)
    parser.add_argument("--val_size", type=_parse_optional_int, default=None)
    args = parser.parse_args()

    train_dataset, val_dataset = prepare_appworld_data(
        train_split=args.train_split,
        val_split=args.val_split,
        train_size=args.train_size,
        val_size=args.val_size,
        dataset_name=args.dataset_name,
    )
    print(f"Train dataset: {len(train_dataset.get_data())} tasks")
    print(f"Validation dataset: {len(val_dataset.get_data())} tasks")
    print("Sample train task:", train_dataset.get_data()[0])


if __name__ == "__main__":
    main()
