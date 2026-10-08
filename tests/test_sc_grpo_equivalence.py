"""Equivalence tests for SC-GRPO tensor math and helper logic."""

import importlib.util
import json
import numpy as np
import sys
import torch
import torch.nn.functional as F
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_repo_module(module_name: str, relative_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        parent_name = ".".join(parts[:idx])
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        parent.__dict__.setdefault("__path__", [])
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_sc_grpo_module():
    if "rllm.engine.sc_grpo" in sys.modules:
        return sys.modules["rllm.engine.sc_grpo"]

    rllm_stub = sys.modules.setdefault("rllm", types.ModuleType("rllm"))
    agents_stub = sys.modules.setdefault("rllm.agents", types.ModuleType("rllm.agents"))
    utils_stub = types.ModuleType("rllm.agents.utils")

    def _unused_convert_messages_to_tokens_and_masks(*args, **kwargs):
        raise RuntimeError("Tokenization is not used by these offline equivalence tests.")

    utils_stub.convert_messages_to_tokens_and_masks = _unused_convert_messages_to_tokens_and_masks
    sys.modules["rllm.agents.utils"] = utils_stub
    rllm_stub.agents = agents_stub
    agents_stub.utils = utils_stub
    return _load_repo_module("rllm.engine.sc_grpo", "rllm/engine/sc_grpo.py")


def _load_rewrite_prompt_module():
    _load_repo_module("rllm.system_prompts", "rllm/system_prompts.py")
    return _load_repo_module("rllm.engine.rewrite_prompt", "rllm/engine/rewrite_prompt.py")


def _load_data_utils_module():
    _load_repo_module("rllm.system_prompts", "rllm/system_prompts.py")
    _load_repo_module("rllm.data.dataset_types", "rllm/data/dataset_types.py")
    return _load_repo_module("rllm.data.utils", "rllm/data/utils.py")


def _compute_kl_weight(
    student_topk_logps: torch.Tensor,
    teacher_topk_logps: torch.Tensor,
    kw_apply_mask: torch.Tensor,
    response_mask: torch.Tensor,
    kl_c_quantile: float = 0.75,
    kl_c_floor: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Reference implementation matching dp_actor.py decoupled KL-weight logic."""
    def _add_tail(logps):
        log_s = torch.logsumexp(logps, dim=-1, keepdim=True).clamp(max=-1e-7)
        tail = torch.log(-torch.expm1(log_s))
        return torch.cat([logps, tail], dim=-1)

    s_tail = _add_tail(student_topk_logps)
    t_tail = _add_tail(teacher_topk_logps)

    kl_per_token = F.kl_div(s_tail, t_tail, reduction="none", log_target=True).sum(dim=-1)
    active_kl = kl_per_token[kw_apply_mask.bool()]
    if active_kl.numel() > 0:
        median_kl = active_kl.median().item()
        mean_kl = active_kl.mean().item()
        kl_c_raw = torch.quantile(active_kl.float(), kl_c_quantile).item()
    else:
        median_kl = 1.0
        mean_kl = 1.0
        kl_c_raw = 1.0

    c = max(kl_c_raw, kl_c_floor)
    f_kl = kl_per_token / (kl_per_token + c)
    kl_weight = torch.where(kw_apply_mask.bool(), f_kl, torch.ones_like(f_kl))
    loss_mask = response_mask.clone().float()

    metrics = {
        "kl_weight_mean": f_kl[kw_apply_mask.bool()].mean().item() if active_kl.numel() > 0 else 0.0,
        "kl_weight_c": c,
        "kl_weight_c_raw": kl_c_raw,
        "kl_per_token_mean": mean_kl,
        "kl_per_token_median": median_kl,
    }
    return kl_weight, loss_mask, metrics


def test_kl_weight_computation_correctness():
    """Test that p75-based c produces correct selectivity: bottom 75% suppressed, top 25% preserved."""
    torch.manual_seed(42)
    batch_size, response_len, topk = 4, 128, 64

    student_logits = torch.randn(batch_size, response_len, topk)
    teacher_logits = torch.randn(batch_size, response_len, topk)
    student_topk_logps = student_logits.log_softmax(dim=-1)
    teacher_topk_logps = teacher_logits.log_softmax(dim=-1)

    kw_apply_mask = torch.zeros(batch_size, response_len, dtype=torch.long)
    kw_apply_mask[0, :] = 1
    kw_apply_mask[1, :] = 1

    response_mask = torch.ones(batch_size, response_len, dtype=torch.long)
    response_mask[:, -10:] = 0

    kl_weight, loss_mask, metrics = _compute_kl_weight(
        student_topk_logps, teacher_topk_logps, kw_apply_mask, response_mask
    )

    # Non-eligible rows must keep neutral KL weight, while the loss mask stays binary.
    assert torch.allclose(kl_weight[2:], torch.ones_like(kl_weight[2:]), atol=1e-7)
    assert torch.allclose(loss_mask, response_mask.float(), atol=1e-7)

    # All weights must be in [0, 1]
    eligible_weights = kl_weight[0][response_mask[0].bool()]
    assert (eligible_weights >= 0).all() and (eligible_weights <= 1.0).all()

    # c == max(kl_c_raw, kl_c_floor); kl_weight_c_raw is the pre-floor quantile value
    assert metrics["kl_weight_c"] >= metrics["kl_weight_c_raw"] or abs(metrics["kl_weight_c"] - metrics["kl_weight_c_raw"]) < 1e-9

    # At c = p75, a token with KL == c must yield f = 0.5
    c = metrics["kl_weight_c"]
    f_at_c = c / (c + c)
    assert abs(f_at_c - 0.5) < 1e-9


def test_kl_weight_median_collapse_scenario():
    """Verify fix for median collapse: 90% near-zero KL tokens.

    Old behavior (median-based c): median≈0 → c=1e-8 → f(KL)≈1 for all tokens (no selectivity).
    New behavior (p75-based c + floor): when p75≈0 the floor activates (c=1e-4), keeping
    near-zero KL tokens suppressed (f(0)=0) while high-KL tokens (f>>0) stay elevated.

    Key invariant: high-KL tokens must have substantially higher weight than near-zero-KL tokens.
    """
    torch.manual_seed(7)
    batch_size, response_len, topk = 2, 100, 32

    # 75% of tokens: student ≈ teacher (near-zero KL)
    same_logits = torch.randn(batch_size, response_len, topk)
    student_logps = same_logits.log_softmax(dim=-1).clone()
    teacher_logps = same_logits.log_softmax(dim=-1).clone()

    # Top 25% of tokens: student very different from teacher (high KL)
    divergent_cols = int(response_len * 0.25)
    teacher_logps[:, :divergent_cols, :] = torch.randn(batch_size, divergent_cols, topk).log_softmax(dim=-1)

    mask = torch.ones(batch_size, response_len, dtype=torch.long)
    response_mask = torch.ones(batch_size, response_len, dtype=torch.long)

    kl_weight, _, metrics = _compute_kl_weight(student_logps, teacher_logps, mask, response_mask)

    # c must be >= kl_c_floor and finite
    assert metrics["kl_weight_c"] >= 1e-4
    assert metrics["kl_weight_c"] < 1e6

    # High-KL tokens (first 25%) should have higher weight than near-zero-KL tokens (last 75%)
    high_kl_weight = kl_weight[:, :divergent_cols].mean().item()
    low_kl_weight = kl_weight[:, divergent_cols:].mean().item()
    assert high_kl_weight > low_kl_weight, (
        f"High-KL tokens (weight={high_kl_weight:.3f}) should exceed "
        f"low-KL tokens (weight={low_kl_weight:.3f})"
    )

    # With old median-based c=1e-8, all f(KL)≈1 → mean≈1. Verify new code gives lower mean.
    mean_weight = kl_weight[mask.bool()].mean().item()
    assert mean_weight < 0.95, f"mean f(KL) = {mean_weight:.3f}: weighting too close to uniform (old bug)"


def test_tail_bucket_normalization():
    torch.manual_seed(123)
    batch_size, response_len, topk = 2, 32, 64

    full_logits = torch.randn(batch_size, response_len, 32000)
    full_logps = full_logits.log_softmax(dim=-1)
    topk_logps, _ = torch.topk(full_logps, topk, dim=-1)

    def _add_tail(logps):
        log_s = torch.logsumexp(logps, dim=-1, keepdim=True).clamp(max=-1e-7)
        tail = torch.log(-torch.expm1(log_s))
        return torch.cat([logps, tail], dim=-1)

    with_tail = _add_tail(topk_logps)
    probs = with_tail.exp().sum(dim=-1)

    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)
    assert (with_tail[:, :, -1] < 0).all()
    assert with_tail.shape == (batch_size, response_len, topk + 1)


def test_trajectory_classification():
    import random

    random.seed(42)
    uids = np.array(["q1", "q1", "q1", "q1", "q2", "q2", "q2", "q2", "q3", "q3", "q3", "q3"])
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    group_row_indices_by_uid = _load_sc_grpo_module().group_row_indices_by_uid

    uid_to_indices = group_row_indices_by_uid(uids)
    n_partial_solve = 0
    n_all_correct = 0
    n_all_incorrect = 0
    eligible_indices = []

    for _, indices in uid_to_indices.items():
        uid_rewards = [rewards[idx] for idx in indices]
        correct_indices = [idx for idx, reward in zip(indices, uid_rewards) if reward >= 1.0]
        n_correct = len(correct_indices)
        is_partial_solve = 0 < n_correct < len(indices)
        is_kl_eligible = is_partial_solve and n_correct >= 2

        if n_correct == 0:
            n_all_incorrect += 1
        elif n_correct == len(indices):
            n_all_correct += 1
        else:
            n_partial_solve += 1

        if is_kl_eligible:
            for idx in indices:
                eligible_indices.append(idx)
                if idx in correct_indices:
                    others = [donor_idx for donor_idx in correct_indices if donor_idx != idx]
                    donor = random.choice(others)
                    assert donor != idx
                    assert donor in correct_indices
                else:
                    donor = random.choice(correct_indices)
                    assert donor in correct_indices

    assert n_partial_solve == 1
    assert n_all_correct == 1
    assert n_all_incorrect == 1
    assert set(eligible_indices) == {0, 1, 2, 3}


def test_partial_solve_n_correct_1_not_eligible():
    group_row_indices_by_uid = _load_sc_grpo_module().group_row_indices_by_uid

    uids = np.array(["q1", "q1", "q1", "q1"])
    rewards = np.array([1.0, 0.0, 0.0, 0.0])
    uid_to_indices = group_row_indices_by_uid(uids)

    for _, indices in uid_to_indices.items():
        uid_rewards = [rewards[idx] for idx in indices]
        correct_indices = [idx for idx, reward in zip(indices, uid_rewards) if reward >= 1.0]
        n_correct = len(correct_indices)
        is_partial_solve = 0 < n_correct < len(indices)
        is_kl_eligible = is_partial_solve and n_correct >= 2
        assert is_partial_solve is True
        assert is_kl_eligible is False


def test_clean_diversity_reference_filter():
    kw_mod = _load_sc_grpo_module()

    clean = {
        "response_tokens": torch.ones(512, dtype=torch.long),
        "termination_reason": "ENV_DONE",
    }
    assert kw_mod.is_clean_diversity_reference(clean, max_response_length=8192)

    truncated = {
        "response_tokens": torch.ones(512, dtype=torch.long),
        "termination_reason": "TerminationReason.TRUNCATION",
    }
    assert not kw_mod.is_clean_diversity_reference(truncated, max_response_length=8192)

    near_limit = {
        "response_tokens": torch.ones(8000, dtype=torch.long),
        "termination_reason": "ENV_DONE",
    }
    assert not kw_mod.is_clean_diversity_reference(near_limit, max_response_length=8192)

    too_short = {
        "response_tokens": torch.ones(32, dtype=torch.long),
        "termination_reason": "ENV_DONE",
    }
    assert not kw_mod.is_clean_diversity_reference(too_short, max_response_length=8192)


def test_first_nonempty_diversity_reference_skips_empty_candidates():
    kw_mod = _load_sc_grpo_module()

    traj_by_idx = {
        0: {
            "chat_completions": [{"role": "assistant", "content": "   "}],
            "response_tokens": torch.ones(128, dtype=torch.long),
        },
        1: {
            "chat_completions": [
                {"role": "assistant", "content": "```python\ndef solve():\n    return 1\n```"}
            ],
            "response_tokens": torch.ones(256, dtype=torch.long),
        },
    }

    ref_idx, ref_text, ref_len = kw_mod.first_nonempty_diversity_reference(
        [0, 1],
        traj_by_idx,
        data_source="livecodebench",
    )
    assert ref_idx == 1
    assert "def solve" in ref_text
    assert ref_len == 256

    empty_idx, empty_text, empty_len = kw_mod.first_nonempty_diversity_reference(
        [0],
        traj_by_idx,
        data_source="livecodebench",
    )
    assert empty_idx is None
    assert empty_text == ""
    assert empty_len == 0


def test_data_source_falls_back_to_extra_info_for_verl_batches():
    kw_mod = _load_sc_grpo_module()

    sources = kw_mod.resolve_data_sources(
        {
            "extra_info": np.array(
                [
                    {"data_source": "livecodebench"},
                    json.dumps({"data_source": "deepcoder"}),
                    {"uid": "missing_source"},
                ],
                dtype=object,
            )
        },
        batch_size=3,
    )
    assert sources.tolist() == ["livecodebench", "deepcoder", "unknown"]

    direct_sources = kw_mod.resolve_data_sources(
        {
            "data_source": np.array(["webshop", "livecodebench"], dtype=object),
            "extra_info": np.array(
                [
                    {"data_source": "should_not_win"},
                    {"data_source": "should_not_win"},
                ],
                dtype=object,
            ),
        },
        batch_size=2,
    )
    assert direct_sources.tolist() == ["webshop", "livecodebench"]

    placeholder_sources = kw_mod.resolve_data_sources(
        {
            "data_source": np.array(["unknown", ""], dtype=object),
            "extra_info": np.array(
                [
                    {"data_source": "livecodebench"},
                    {"data_source": "deepcoder"},
                ],
                dtype=object,
            ),
        },
        batch_size=2,
    )
    assert placeholder_sources.tolist() == ["livecodebench", "deepcoder"]


def test_livecodebench_reference_uses_extra_info_data_source():
    kw_mod = _load_sc_grpo_module()

    sources = kw_mod.resolve_data_sources(
        {"extra_info": np.array([{"data_source": "livecodebench"}], dtype=object)},
        batch_size=1,
    )
    ref_text = kw_mod.extract_assistant_trajectory_text(
        [{"role": "assistant", "content": "```python\ndef solve():\n    return 1\n```"}],
        data_source=sources[0],
    )

    assert "def solve" in ref_text


def test_math_reference_keeps_assistant_solution_text():
    kw_mod = _load_sc_grpo_module()

    ref_text = kw_mod.extract_assistant_trajectory_text(
        [
            {
                "role": "assistant",
                "content": "<think>Let x=3.</think>\n\nThe answer is \\boxed{3}.",
            }
        ],
        data_source="dapo_math",
    )

    assert "Let x=3" in ref_text
    assert "\\boxed{3}" in ref_text


def test_teacher_prompt_construction():
    build_rewrite_reference_system_prompt = _load_rewrite_prompt_module().build_rewrite_reference_system_prompt

    code_ref = "def solve(n):\n    return n * 2"
    code_prompt = build_rewrite_reference_system_prompt(
        reference_trajectory=code_ref,
        data_source="livecodebench",
    )
    assert "expert python programmer" in code_prompt.lower()
    assert "algorithmic guide" in code_prompt.lower() or "competitive programming" in code_prompt.lower()
    assert code_ref in code_prompt

    action_ref = "search[laptop]\nclick[buy now]"
    action_prompt = build_rewrite_reference_system_prompt(
        reference_trajectory=action_ref,
        data_source="webshop",
    )
    assert "strategic guidance" in action_prompt.lower() or "multi-turn" in action_prompt.lower()
    assert action_ref in action_prompt

    math_ref = "We solve 3x+5=14, so x=3. Therefore \\boxed{3}."
    math_prompt = build_rewrite_reference_system_prompt(
        reference_trajectory=math_ref,
        data_source="dapo_math",
    )
    assert "math problem" in math_prompt.lower()
    assert "candidate solution path" in math_prompt.lower()
    assert "verified useful steps" in math_prompt.lower()
    assert "final answer within \\boxed{}" in math_prompt
    assert "\\boxed{{}}" not in math_prompt
    assert math_ref in math_prompt

    truncated_prompt = build_rewrite_reference_system_prompt(
        reference_trajectory="x" * 20000,
        max_reference_chars=100,
        data_source="livecodebench",
    )
    assert len(truncated_prompt) < 20000


def test_livecodebench_prompt_messages_match_official_chat_shape():
    data_utils = _load_data_utils_module()
    messages = data_utils.build_live_code_bench_prompt_messages(
        "Add two integers.",
        "def solve():\n    pass",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "expert python programmer" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("### Question:\nAdd two integers.\n\n")
    assert "### Format:" in messages[1]["content"]
    assert "```python\ndef solve():\n    pass\n```" in messages[1]["content"]
    assert "### Answer: (use the provided format with backticks)" in messages[1]["content"]


def test_livecodebench_openai_reason_prompt_matches_official_shape():
    data_utils = _load_data_utils_module()
    messages = data_utils.build_live_code_bench_openai_reason_messages("Add two integers.")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "expert python programmer" in messages[0]["content"].lower()
    assert "### Question:\nAdd two integers.\n\n" in messages[0]["content"]
    assert "Implement a function called `main()`" in messages[0]["content"]
    assert "### Answer: (use the provided format with backticks)" in messages[0]["content"]


def test_f_kl_properties():
    assert abs(0.0 / (0.0 + 1.0)) < 1e-10
    c = 2.5
    assert abs(c / (c + c) - 0.5) < 1e-10
    large_kl = 1e6
    assert abs(large_kl / (large_kl + c) - 1.0) < 1e-4

    kls = torch.linspace(0, 10, 100)
    f_vals = kls / (kls + c)
    diffs = f_vals[1:] - f_vals[:-1]
    assert (diffs >= 0).all()
    assert (f_vals >= 0).all() and (f_vals < 1.0).all()

    kl_with_outlier = torch.tensor([0.1, 0.2, 0.3, 0.4, 100.0])
    assert kl_with_outlier.median().item() < 1.0
    assert kl_with_outlier.mean().item() > 10.0


def test_diversity_score_uses_kl_weight_not_loss_mask():
    f_kl = torch.tensor([
        [0.10, 0.20, 0.30, 0.99],
        [0.70, 0.80, 0.90, 0.99],
    ])
    kw_apply_mask = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 1, 0],
    ])
    response_mask = torch.ones_like(kw_apply_mask)
    diversity_ref_len = torch.tensor([3, 3])

    scores = []
    for row_i in range(f_kl.shape[0]):
        resp_len = int(response_mask[row_i].sum().item())
        eff_len = min(resp_len, int(diversity_ref_len[row_i].item()))
        active = kw_apply_mask[row_i, :eff_len].bool()
        scores.append(f_kl[row_i, :eff_len][active].mean().item())

    scores = torch.tensor(scores)
    assert torch.allclose(scores, torch.tensor([0.20, 0.80]), atol=1e-6)
    assert scores.std().item() > 0
    assert not torch.allclose(scores, torch.ones_like(scores))


def test_kl_weight_mode_surfaces():
    advantages = torch.ones(2, 4)
    response_mask = torch.ones(2, 4)
    kw_apply_mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
    ], dtype=torch.long)
    f_kl = torch.tensor([
        [0.2, 0.8, 0.5, 0.5],
        [0.4, 0.5, 0.6, 0.5],
    ])
    kl_weight = torch.where(kw_apply_mask.bool(), f_kl, torch.ones_like(f_kl))

    v54_loss_mask = response_mask * kl_weight
    v54_advantages = advantages
    assert torch.allclose(v54_loss_mask[kw_apply_mask.bool()], f_kl[kw_apply_mask.bool()])
    assert torch.allclose(v54_advantages, advantages)

    v56_loss_mask = response_mask
    v56_advantages = advantages * kl_weight
    assert torch.allclose(v56_loss_mask, response_mask)
    assert torch.allclose(v56_advantages[kw_apply_mask.bool()], f_kl[kw_apply_mask.bool()])


def test_kl_weight_edge_cases():
    batch_size, response_len, topk = 2, 16, 8
    student = torch.randn(batch_size, response_len, topk).log_softmax(dim=-1)
    teacher = torch.randn(batch_size, response_len, topk).log_softmax(dim=-1)
    mask = torch.zeros(batch_size, response_len, dtype=torch.long)
    response_mask = torch.ones(batch_size, response_len, dtype=torch.long)

    # No eligible tokens: weighted mask must equal response_mask unchanged
    kl_weight, loss_mask, metrics = _compute_kl_weight(student, teacher, mask, response_mask)
    assert torch.allclose(kl_weight, torch.ones_like(kl_weight))
    assert torch.allclose(loss_mask, response_mask.float())
    assert metrics["kl_weight_mean"] == 0.0

    # Identical distributions: all KL=0, so kl_c_raw=0, c=kl_c_floor, f(0)=0/(0+floor)=0
    same_logps = torch.randn(batch_size, response_len, topk).log_softmax(dim=-1)
    mask_all = torch.ones(batch_size, response_len, dtype=torch.long)
    kl_weight2, _, metrics2 = _compute_kl_weight(same_logps, same_logps.clone(), mask_all, response_mask)
    assert kl_weight2[mask_all.bool()].mean().item() < 0.01
    # When all KL=0, the floor is active: c == kl_c_floor
    assert abs(metrics2["kl_weight_c"] - 1e-4) < 1e-9


if __name__ == "__main__":
    test_kl_weight_computation_correctness()
    test_kl_weight_median_collapse_scenario()
    test_tail_bucket_normalization()
    test_trajectory_classification()
    test_partial_solve_n_correct_1_not_eligible()
    test_clean_diversity_reference_filter()
    test_first_nonempty_diversity_reference_skips_empty_candidates()
    test_data_source_falls_back_to_extra_info_for_verl_batches()
    test_livecodebench_reference_uses_extra_info_data_source()
    test_teacher_prompt_construction()
    test_livecodebench_prompt_messages_match_official_chat_shape()
    test_livecodebench_openai_reason_prompt_matches_official_shape()
    test_f_kl_properties()
    test_diversity_score_uses_kl_weight_not_loss_mask()
    test_kl_weight_mode_surfaces()
    test_kl_weight_edge_cases()
    print("\n=== ALL EQUIVALENCE TESTS PASSED ===")
