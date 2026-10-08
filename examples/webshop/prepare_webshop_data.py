import argparse
import random
from typing import Optional

from rllm.data.dataset import DatasetRegistry

try:
    from examples.webshop.webshop_env import fetch_num_goals as fetch_num_goals_local
except ModuleNotFoundError:
    from webshop_env import fetch_num_goals as fetch_num_goals_local


def prepare_webshop_data(
    train_size: int | None = 2400,
    test_size: int = 100,
    seed: int = 42,
    split_mode: str = "official",
    num_products: int = 1000,
    human_goals: int = 0,
    observation_mode: str = "text_rich",
    webshop_root: Optional[str] = None,
    dataset_name: str = "webshop_text",
):
    if webshop_root:
        import os

        os.environ["WEBSHOP_ROOT"] = os.path.abspath(os.path.expanduser(webshop_root))

    total_goals = fetch_num_goals_local(
        num_products=num_products,
        human_goals=human_goals,
        observation_mode=observation_mode,
    )

    if total_goals <= 0:
        raise RuntimeError(
            "WebShop local backend reports zero available goals. "
            "Check whether WebShop data/index setup is complete."
        )

    split_mode = split_mode.strip().lower()

    if split_mode == "official":
        # Keep the official ordering while using a smaller test split by default:
        # test: [0, test_size), eval: [test_size, test_size + 1000), train: [test_size + 1000, end)
        test_end = min(test_size, total_goals)
        eval_end = min(test_size + 1000, total_goals)
        test_indices = list(range(0, test_end))
        eval_indices = list(range(test_end, eval_end))
        train_indices = list(range(eval_end, total_goals))
    elif split_mode == "random":
        if train_size is None:
            raise ValueError("random split requires train_size")
        required = train_size + test_size
        if required > total_goals:
            raise ValueError(
                f"Requested train_size + test_size = {required} exceeds available goals {total_goals}."
            )

        goal_indices = list(range(total_goals))
        rng = random.Random(seed)
        rng.shuffle(goal_indices)

        train_indices = goal_indices[:train_size]
        test_indices = goal_indices[train_size : train_size + test_size]
        eval_indices = []
    else:
        raise ValueError(f"Unsupported split_mode={split_mode!r}. Expected 'official' or 'random'.")

    if len(train_indices) == 0:
        raise RuntimeError(
            f"No training goals after split_mode={split_mode}. total_goals={total_goals}."
        )

    if split_mode == "official" and train_size is not None:
        if not 0 < train_size <= len(train_indices):
            raise ValueError("train_size must be positive and fit the training pool.")
        train_indices = random.Random(seed).sample(train_indices, train_size)

    def _build_example(idx: int, split: str) -> dict:
        return {
            "goal_idx": idx,
            "data_source": "webshop",
            "uid": f"webshop_{split}_{idx}",
        }

    train_data = [_build_example(idx, "train") for idx in train_indices]
    test_data = [_build_example(idx, "test") for idx in test_indices]
    eval_data = [_build_example(idx, "eval") for idx in eval_indices]

    train_dataset = DatasetRegistry.register_dataset(dataset_name, train_data, "train")
    test_dataset = DatasetRegistry.register_dataset(dataset_name, test_data, "test")
    if eval_data:
        DatasetRegistry.register_dataset(dataset_name, eval_data, "eval")

    return train_dataset, test_dataset, {
        "total_goals": total_goals,
        "split_mode": split_mode,
        "train_size": len(train_data),
        "test_size": len(test_data),
        "eval_size": len(eval_data),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare WebShop train/test datasets for rLLM examples")
    parser.add_argument("--train_size", type=int, default=2400)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", type=str, default="official", choices=["official", "random"])
    parser.add_argument("--num_products", type=int, default=1000)
    parser.add_argument("--human_goals", type=int, default=0)
    parser.add_argument("--observation_mode", type=str, default="text_rich")
    parser.add_argument("--webshop_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    args = parser.parse_args()
    dataset_name = args.dataset_name or ("webshop_text_official" if args.split_mode == "official" else "webshop_text")

    train_dataset, test_dataset, stats = prepare_webshop_data(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        split_mode=args.split_mode,
        num_products=args.num_products,
        human_goals=args.human_goals,
        observation_mode=args.observation_mode,
        webshop_root=args.webshop_root,
        dataset_name=dataset_name,
    )

    print(f"WebShop goals available: {stats['total_goals']}")
    print(f"Split mode: {stats['split_mode']}")
    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    if stats.get("eval_size", 0) > 0:
        print(f"Eval dataset: {stats['eval_size']} examples")
    print("Sample train example:", train_dataset.get_data()[0])
    print("Sample test example:", test_dataset.get_data()[0])


if __name__ == "__main__":
    main()
