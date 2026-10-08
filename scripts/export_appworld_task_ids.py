"""Run this standalone script in the AppWorld environment."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test_normal"])
    args = parser.parse_args()
    from appworld import load_task_ids
    task_ids = {split: list(load_task_ids(split)) for split in args.splits}
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(task_ids, indent=2) + "\n")
    for split, ids in task_ids.items():
        print(f"{split}: {len(ids)} tasks")


if __name__ == "__main__":
    main()
