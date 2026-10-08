"""SC-GRPO engine utilities: trajectory extraction, teacher prompt tokenization, uid grouping."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from rllm.agents.utils import convert_messages_to_tokens_and_masks

ACTION_PATTERN = re.compile(r"(search\[[^\[\]\n]+\]|click\[[^\[\]\n]+\])", re.IGNORECASE)
ACTION_FULL_PATTERN = re.compile(r"^(search|click)\[(.*)\]$", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_DATA_SOURCES = {"livecodebench", "deepcoder", "codeforces"}
_APPWORLD_DATA_SOURCE_MARKERS = ("appworld",)
_MATH_DATA_SOURCE_MARKERS = ("math", "aime", "gsm8k", "deepscaler")
_BAD_DIVERSITY_REF_REASONS = {
    "TRUNCATION",
    "TIMEOUT",
    "ENV_TIMEOUT",
    "PROMPT_TRUNCATION",
    "MAX_RESPONSE_LENGTH_EXCEEDED",
    "MAX_PROMPT_LENGTH_EXCEEDED",
}


def resolve_data_sources(
    non_tensor_batch: dict[str, Any],
    batch_size: int,
    default: str = "unknown",
) -> np.ndarray:
    """Resolve per-row data_source, falling back to extra_info for Verl-processed datasets."""
    def _normalize_source(source: Any) -> str:
        source_str = str(source or "").strip()
        return source_str if source_str else default

    resolved = [default] * batch_size
    direct_sources = non_tensor_batch.get("data_source")
    if direct_sources is not None:
        direct_array = np.asarray(direct_sources, dtype=object)
        if direct_array.ndim == 0:
            resolved = [_normalize_source(direct_array.item())] * batch_size
        elif len(direct_array) == batch_size:
            resolved = [_normalize_source(source) for source in direct_array]

    extra_infos = non_tensor_batch.get("extra_info")
    if extra_infos is None:
        return np.array(resolved, dtype=object)

    for idx, extra_info in enumerate(list(extra_infos)[:batch_size]):
        if resolved[idx] != default:
            continue
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}
        if not isinstance(extra_info, dict):
            continue
        data_source = extra_info.get("data_source")
        if data_source:
            resolved[idx] = str(data_source)

    return np.array(resolved, dtype=object)


def _normalize_action(raw_action: str) -> str | None:
    raw_action = raw_action.strip()
    match = ACTION_FULL_PATTERN.match(raw_action)
    if match is None:
        return None
    action_name = match.group(1).lower()
    action_arg = " ".join(match.group(2).split())
    if not action_arg:
        return None
    return f"{action_name}[{action_arg}]"


def _extract_first_action(text: str) -> str | None:
    found = ACTION_PATTERN.search(text)
    if not found:
        return None
    return _normalize_action(found.group(1))


def _extract_first_visible_action(text: str) -> str | None:
    visible_text = THINK_PATTERN.sub("", text).strip()
    if not visible_text:
        return None
    fenced_blocks = re.findall(
        r"```(?:action)?\s*(.*?)```", visible_text, flags=re.IGNORECASE | re.DOTALL
    )
    for block in fenced_blocks:
        action = _extract_first_action(block)
        if action is not None:
            return action
    return _extract_first_action(visible_text)


def _extract_code_solution(text: str) -> str:
    visible = THINK_PATTERN.sub("", text).strip()
    blocks = _CODE_FENCE_PATTERN.findall(visible)
    if blocks:
        return f"```python\n{blocks[-1].strip()}\n```"
    return visible.strip()


def _is_code_data_source(data_source: str) -> bool:
    ds = (data_source or "").lower()
    return ds in _CODE_DATA_SOURCES or any(source in ds for source in _CODE_DATA_SOURCES)


def _is_appworld_data_source(data_source: str) -> bool:
    ds = (data_source or "").lower()
    return any(marker in ds for marker in _APPWORLD_DATA_SOURCE_MARKERS)


def _is_math_data_source(data_source: str) -> bool:
    ds = (data_source or "").lower()
    return any(marker in ds for marker in _MATH_DATA_SOURCE_MARKERS)


def _appworld_actual_start_index(chat_completions: list[dict[str, Any]]) -> int:
    """Skip few-shot prompt demonstrations and keep only actual task turns."""
    marker = "Using these APIs, now generate code to solve the actual task:"
    start_idx = 0
    for idx, message in enumerate(chat_completions):
        if message.get("role") == "user" and marker in str(message.get("content", "")):
            start_idx = idx + 1
    return start_idx


def _extract_math_solution(text: str) -> str:
    return text.strip()


def group_row_indices_by_uid(uids: np.ndarray) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row_idx, uid in enumerate(uids.tolist()):
        grouped[str(uid)].append(row_idx)
    return grouped


def response_token_length(traj: dict[str, Any]) -> int:
    response_tokens = traj.get("response_tokens", [])
    if hasattr(response_tokens, "numel"):
        return int(response_tokens.numel())
    return len(response_tokens)


def is_clean_diversity_reference(
    traj: dict[str, Any],
    max_response_length: int,
    min_response_tokens: int = 64,
    max_response_ratio: float = 0.95,
) -> bool:
    """Return whether a solve_none rollout is clean enough to use as diversity reference."""
    resp_len = response_token_length(traj)
    if resp_len < min_response_tokens:
        return False
    if max_response_length > 0:
        if resp_len >= max_response_length:
            return False
        if max_response_ratio > 0 and resp_len >= int(max_response_length * max_response_ratio):
            return False

    reason = str(traj.get("termination_reason") or "").upper().rsplit(".", 1)[-1]
    return reason not in _BAD_DIVERSITY_REF_REASONS


def first_nonempty_diversity_reference(
    candidate_indices: list[int],
    traj_by_idx: dict[int, dict[str, Any]],
    data_source: str = "",
) -> tuple[int | None, str, int]:
    """Return the first candidate with extractable reference text."""
    for idx in candidate_indices:
        traj = traj_by_idx[idx]
        ref_text = extract_assistant_trajectory_text(
            traj.get("chat_completions", []),
            data_source=data_source,
        )
        if ref_text.strip():
            return idx, ref_text, response_token_length(traj)
    return None, "", 0


def extract_assistant_trajectory_text(
    chat_completions: list[dict[str, Any]], data_source: str = ""
) -> str:
    use_code = _is_code_data_source(data_source)
    use_appworld = _is_appworld_data_source(data_source)
    use_math = _is_math_data_source(data_source)
    if use_appworld:
        chat_completions = chat_completions[_appworld_actual_start_index(chat_completions):]

    chunks: list[str] = []
    for message in chat_completions:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if content is None:
            continue
        if use_code or use_appworld:
            chunk = _extract_code_solution(str(content))
        elif use_math:
            chunk = _extract_math_solution(str(content))
        else:
            chunk = _extract_first_visible_action(str(content))
        if chunk:
            chunks.append(chunk)
    return "\n\n".join(chunks)


def _extract_initial_prompt_messages(
    chat_completions: list[dict[str, Any]],
    data_source: str = "",
) -> list[dict[str, str]]:
    if _is_appworld_data_source(data_source):
        return [
            {"role": str(message.get("role")), "content": str(message.get("content", ""))}
            for message in chat_completions[:_appworld_actual_start_index(chat_completions)]
            if message.get("role") is not None
        ]

    initial: list[dict[str, str]] = []
    for message in chat_completions:
        role = message.get("role")
        if role == "assistant":
            break
        if role is None:
            continue
        initial.append({"role": str(role), "content": str(message.get("content", ""))})
    return initial


def build_kw_teacher_prompt_tokens(
    trajectories: list[dict[str, Any]],
    tokenizer,
    parser,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build teacher prompt tokens and apply masks for SC-GRPO."""
    teacher_prompt_tokens: list[torch.Tensor] = []
    kw_apply_masks: list[torch.Tensor] = []

    for traj in trajectories:
        prompt_tokens = traj["prompt_tokens"]
        response_masks = traj["response_masks"]
        teacher_sys_prompt = traj.get("kw_teacher_system_prompt")
        is_eligible = bool(traj.get("kw_eligible", False))
        data_source = str(traj.get("data_source", ""))

        if is_eligible and teacher_sys_prompt:
            initial_messages = _extract_initial_prompt_messages(
                traj.get("chat_completions", []),
                data_source=data_source,
            )
            non_system = [m for m in initial_messages if m.get("role") != "system"]
            conditioned = [{"role": "system", "content": str(teacher_sys_prompt)}] + non_system
            token_ids, _ = convert_messages_to_tokens_and_masks(
                conditioned,
                tokenizer=tokenizer,
                parser=parser,
                contains_first_msg=True,
                contains_generation_msg=True,
            )
            teacher_prompt_tokens.append(torch.tensor(token_ids, dtype=torch.long))
            kw_apply_masks.append(response_masks.clone().long())
        else:
            teacher_prompt_tokens.append(prompt_tokens)
            kw_apply_masks.append(torch.zeros_like(response_masks, dtype=torch.long))

    return teacher_prompt_tokens, kw_apply_masks
