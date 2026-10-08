import numpy as np

from rllm.trainer.verl.validation_metrics import compute_validation_score_metrics


def test_compute_validation_score_metrics_supports_continuous_and_discrete_k_metrics():
    rewards = np.array([0.2, 1.0, 1.0, 0.5, 0.7, 0.4], dtype=float)
    data_sources = np.array(["webshop", "webshop", "webshop", "webshop", "other", "other"], dtype=object)
    uids = np.array(["u1", "u1", "u2", "u2", "u3", "u3"], dtype=object)

    metrics = compute_validation_score_metrics(
        rewards=rewards,
        data_sources=data_sources,
        uids=uids,
        metric_prefix="val/test_score",
        k=2,
    )

    assert metrics["val/test_score/webshop"] == 0.675
    assert metrics["val/test_score/avg@2/webshop"] == 0.5
    assert metrics["val/test_score/pass@2/webshop"] == 1.0

    assert metrics["val/test_score/other"] == 0.55
    assert metrics["val/test_score/avg@2/other"] == 0.0
    assert metrics["val/test_score/pass@2/other"] == 0.0
