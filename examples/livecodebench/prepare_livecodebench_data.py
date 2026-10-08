import argparse
import json
import random

from datasets import Dataset
from huggingface_hub import HfFileSystem

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import build_live_code_bench_user_prompt


def prepare_livecodebench_data(
    start_date: str = "2025-02",
    end_date: str = "2025-06",
    public_test_ratio: float = 0.5,
    seed: int = 42,
    dataset_name: str = "livecodebench_v6",
):
    """Prepare LiveCodeBench dataset following the SDPO paper (arxiv 2601.20802) setup.

    The paper evaluates on LCB v6: 131 problems released between February and May 2025.
    The same 131 problems are used for both RL training and final evaluation:
    - Training reward: public tests (a 50% random subset of private tests)
    - Validation: all private tests

    Both the train and test splits registered here contain the same 131 problems.
    The difference is which tests are included:
    - "train" split: public tests only (used as reward signal during RL)
    - "test" split: all private tests (used for validation)

    Args:
        start_date: Inclusive start month for the LCB version (YYYY-MM).
        end_date: Exclusive end month for the LCB version (YYYY-MM).
        public_test_ratio: Fraction of private tests to expose as public tests during training.
        seed: Random seed for public test sampling.
        dataset_name: Name to register the dataset under.
    """
    # LCB v6 = test6.jsonl (175 problems, Jan–Apr 2025).
    # The repo uses a loading script unsupported by recent datasets versions,
    # so we read the jsonl directly via huggingface_hub.
    hf_path = "datasets/livecodebench/code_generation_lite/test6.jsonl"
    print(f"Loading {hf_path} from HuggingFace...", flush=True)
    fs = HfFileSystem()
    with fs.open(hf_path) as f:
        records = [json.loads(line) for line in f]
    raw_dataset = Dataset.from_list(records)

    # Filter to LCB v6: February–May 2025 (131 problems)
    # contest_date format in the HF dataset: "2025-02-00T00:00:00"
    lcb_v6 = raw_dataset.filter(
        lambda r: start_date <= (r.get("contest_date", "") or "")[:7] < end_date
    )
    print(f"LCB v6 ({start_date} to {end_date}): {len(lcb_v6)} problems", flush=True)

    rng = random.Random(seed)

    def preprocess_train(example, idx):
        """Training split: public tests only (50% random subset of private tests)."""
        import base64
        import pickle
        import zlib

        starter_code = example.get("starter_code", "") or ""
        question = build_live_code_bench_user_prompt(
            example["question_content"],
            starter_code if starter_code else None,
        )

        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except Exception:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(example["private_test_cases"].encode("utf-8"))
                    )
                )
            )

        metadata = json.loads(example["metadata"])
        func_name = metadata.get("func_name", None)

        # Sample public_test_ratio of private tests as the public (training) tests
        k = max(1, round(len(private_test_cases) * public_test_ratio))
        public_subset = rng.sample(private_test_cases, min(k, len(private_test_cases)))

        tests = []
        for t in public_subset:
            testtype = t.get("testtype", "stdin_stdout")
            tests.append(
                {
                    "input": t["input"],
                    "output": t["output"],
                    "testtype": testtype,
                    "metadata": {"func_name": str(func_name) if func_name else None},
                }
            )

        return {
            "question": question,
            "ground_truth": json.dumps(tests),
            "data_source": "livecodebench",
            "uid": f"lcb_v6_{idx}",
            "index": idx,
            "starter_code": starter_code,
            "contest_date": example.get("contest_date", ""),
        }

    def preprocess_test(example, idx):
        """Test/validation split: all private tests."""
        import base64
        import pickle
        import zlib

        starter_code = example.get("starter_code", "") or ""
        question = build_live_code_bench_user_prompt(
            example["question_content"],
            starter_code if starter_code else None,
        )

        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except Exception:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(example["private_test_cases"].encode("utf-8"))
                    )
                )
            )

        metadata = json.loads(example["metadata"])
        func_name = metadata.get("func_name", None)

        tests = []
        for t in private_test_cases:
            testtype = t.get("testtype", "stdin_stdout")
            tests.append(
                {
                    "input": t["input"],
                    "output": t["output"],
                    "testtype": testtype,
                    "metadata": {"func_name": str(func_name) if func_name else None},
                }
            )

        return {
            "question": question,
            "ground_truth": json.dumps(tests),
            "data_source": "livecodebench",
            "uid": f"lcb_v6_{idx}",
            "index": idx,
            "starter_code": starter_code,
            "contest_date": example.get("contest_date", ""),
        }

    train_processed = lcb_v6.map(preprocess_train, with_indices=True, writer_batch_size=200, num_proc=4)
    test_processed = lcb_v6.map(preprocess_test, with_indices=True, writer_batch_size=200, num_proc=4)

    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset(dataset_name, test_processed, "test")

    return train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LiveCodeBench v6 dataset following SDPO (arxiv 2601.20802).\n"
            "131 problems from Feb-May 2025; training uses public tests (50%% of private), "
            "validation uses all private tests."
        )
    )
    parser.add_argument("--start_date", type=str, default="2025-02",
                        help="Inclusive start month for LCB version (YYYY-MM, default: 2025-02)")
    parser.add_argument("--end_date", type=str, default="2025-06",
                        help="Exclusive end month for LCB version (YYYY-MM, default: 2025-06)")
    parser.add_argument("--public_test_ratio", type=float, default=0.5,
                        help="Fraction of private tests to use as public training tests (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="livecodebench_v6")
    args = parser.parse_args()

    train_dataset, test_dataset = prepare_livecodebench_data(
        start_date=args.start_date,
        end_date=args.end_date,
        public_test_ratio=args.public_test_ratio,
        seed=args.seed,
        dataset_name=args.dataset_name,
    )

    print(f"Train dataset: {len(train_dataset.get_data())} problems (public tests only)")
    print(f"Test dataset:  {len(test_dataset.get_data())} problems (all private tests)")
    if len(train_dataset.get_data()) > 0:
        sample = train_dataset.get_data()[0]
        ntests = len(json.loads(sample["ground_truth"]))
        print(f"Sample train problem has {ntests} public tests: {sample['uid']}")


if __name__ == "__main__":
    main()
