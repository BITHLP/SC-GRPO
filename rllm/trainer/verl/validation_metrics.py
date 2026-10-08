from __future__ import annotations

from collections import defaultdict

import numpy as np


def compute_validation_score_metrics(
    rewards,
    data_sources,
    uids,
    metric_prefix: str = "val/test_score",
    k: int | None = None,
) -> dict[str, float]:
    rewards_array = np.asarray(rewards, dtype=float)
    source_array = np.asarray(data_sources, dtype=object)
    uid_array = np.asarray(uids, dtype=object)

    if not (len(rewards_array) == len(source_array) == len(uid_array)):
        raise ValueError("rewards, data_sources, and uids must have the same length.")

    source_rewards: dict[str, list[float]] = defaultdict(list)
    source_uid_binary_rewards: dict[str, dict[object, list[float]]] = defaultdict(lambda: defaultdict(list))

    for reward, data_source, uid in zip(rewards_array, source_array, uid_array, strict=True):
        source = str(data_source)
        clipped_reward = float(np.clip(reward, 0.0, 1.0))
        binary_reward = 1.0 if clipped_reward >= 1.0 else 0.0

        source_rewards[source].append(clipped_reward)
        source_uid_binary_rewards[source][uid].append(binary_reward)

    metric_dict: dict[str, float] = {}
    discrete_avg_name = f"avg@{k}" if k is not None else "avg@k"
    discrete_pass_name = f"pass@{k}" if k is not None else "pass@k"

    for source, reward_values in source_rewards.items():
        metric_dict[f"{metric_prefix}/{source}"] = float(np.mean(reward_values))

        uid_binary_rewards = source_uid_binary_rewards[source]
        avg_k_values = [float(np.mean(binary_rewards)) for binary_rewards in uid_binary_rewards.values()]
        pass_k_values = [float(max(binary_rewards)) for binary_rewards in uid_binary_rewards.values()]

        metric_dict[f"{metric_prefix}/{discrete_avg_name}/{source}"] = float(np.mean(avg_k_values))
        metric_dict[f"{metric_prefix}/{discrete_pass_name}/{source}"] = float(np.mean(pass_k_values))

    return metric_dict
