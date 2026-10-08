"""Prompt helpers for SC-GRPO teacher-demo conditioning."""

from typing import Optional

from rllm.system_prompts import LCB_SYSTEM_MESSAGE_GENERIC

try:
    from examples.webshop.webshop_agent import WEBSHOP_SYSTEM_PROMPT as DEFAULT_BASE_SYSTEM_PROMPT
except Exception:
    DEFAULT_BASE_SYSTEM_PROMPT = ""

_CODE_DATA_SOURCES = {"livecodebench", "deepcoder", "codeforces"}
_APPWORLD_DATA_SOURCE_MARKERS = ("appworld",)
_MATH_DATA_SOURCE_MARKERS = ("math", "aime", "gsm8k", "deepscaler")

DEFAULT_REWRITE_GUIDE_SYSTEM_PROMPT = (
    "You are in a self-rewrite training phase for a multi-turn interactive task. "
    "A successful reference assistant action sequence is provided below. "
    "The reference below contains only normalized assistant actions. "
    "It does not include thoughts, observations, or explanations. "
    "Use the reference as strategic guidance for how to organize the interaction "
    "and improve decision quality, not as a script to copy. "
    "Your goal is to produce a trajectory that is more organized, coherent, and accurate: "
    "- More organized: keep a clear goal, maintain sensible subgoals, and avoid unnecessary detours. "
    "- More coherent: make each action follow from the current observation, recent interaction history, and task objective. "
    "- More accurate: choose actions that are valid for the current page and grounded in visible evidence. "
    "When using the reference action sequence: "
    "1. Borrow its high-level strategy, action patterns, and pacing when they fit the current state. "
    "2. Use it to improve planning, information gathering, option selection, and timing of decisive actions. "
    "3. Do not copy the exact action sequence blindly. "
    "4. If the current observation differs from the reference, trust the current observation and adapt immediately. "
    "5. Never assume hidden state, unseen page content, or product details just because an action appeared in the reference. "
    "6. Reduce repetitive wandering, low-value clicks, and contradictory actions. "
    "7. Stay aligned with the current page and the available actions at every turn. "
    "8. This guidance applies to the whole multi-turn interaction, not to rewriting a single passage. "
    "9. At every turn, still output exactly one valid next action and nothing else."
)

DEFAULT_REWRITE_REFERENCE_TEMPLATE = (
    "### Reference Successful Assistant Action Sequence\n"
    "{reference_trajectory}\n\n"
    "Use the reference as strategic guidance only. "
    "Solve the current task from the current observation, adapting turn by turn. "
    "Prefer better organization, consistency, and decision quality, but never copy the reference blindly."
)

LCB_REWRITE_GUIDE_SYSTEM_PROMPT = (
    "You are in a self-rewrite training phase for a competitive programming task. "
    "A reference correct solution is provided below. "
    "Use it as a high-level algorithmic guide, not as code to copy. "
    "Your goal is to write a solution that is correct, clean, and efficient: "
    "- Correct: passes all test cases within the time limit. "
    "- Clean: well-structured, readable, and free of dead code or redundant logic. "
    "- Efficient: chooses an algorithm with appropriate time and space complexity. "
    "When using the reference solution: "
    "1. Understand the core algorithm and data structures it uses. "
    "2. Adopt its strategy if it is sound; improve on it if you can. "
    "3. Do not copy variable names, formatting, or boilerplate from the reference. "
    "4. Handle all edge cases independently — do not assume the reference covers them. "
    "5. Write your own solution from scratch, guided by the reference algorithm only."
)

LCB_REWRITE_REFERENCE_TEMPLATE = (
    "### Reference Correct Solution\n"
    "{reference_trajectory}\n\n"
    "Use the reference algorithm as inspiration only. "
    "Write your own complete, correct Python solution from scratch."
)


MATH_REWRITE_GUIDE_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. "
    "You will be given the current math problem and a reference assistant solution. "
    "Treat the reference as a candidate solution path: verify each idea against the current problem before using it. "
    "If it is relevant and correct, follow the same high-level reasoning in your own words; "
    "if it is incomplete, inconsistent, or wrong, ignore the bad parts and solve independently. "
    "Avoid copying surface wording. "
    "Produce clear mathematical reasoning and put the final answer within \\boxed{}."
)

MATH_REWRITE_REFERENCE_TEMPLATE = (
    "### Reference Assistant Solution\n"
    "{reference_trajectory}\n\n"
    "Check whether the reference's setup, formulas, and final answer are consistent with the current problem. "
    "Use only verified useful steps. "
    "Solve the current problem completely and put your final answer within \\boxed{{}}."
)


APPWORLD_REWRITE_GUIDE_SYSTEM_PROMPT = (
    "You are in a self-rewrite training phase for an AppWorld API-control task. "
    "A reference successful assistant Python action sequence is provided below. "
    "Use it as strategic guidance for API discovery, login/account handling, pagination, state updates, "
    "and when to call apis.supervisor.complete_task. "
    "Do not copy API arguments, account values, object ids, or answers blindly; they are task-instance specific. "
    "At each turn, use the current observation and write one small Python code block that is valid in the AppWorld shell. "
    "Prefer checking API docs before using an API, preserve useful variables across turns, handle paginated results, "
    "and avoid irreversible writes until you have verified the required entities."
)

APPWORLD_REWRITE_REFERENCE_TEMPLATE = (
    "### Reference Successful Python Action Sequence\n"
    "{reference_trajectory}\n\n"
    "Use the reference only for high-level interaction strategy. "
    "Solve the current AppWorld task from the current observation and current API outputs."
)


def _is_code_data_source(data_source: str) -> bool:
    ds = (data_source or "").lower()
    return ds in _CODE_DATA_SOURCES or any(source in ds for source in _CODE_DATA_SOURCES)


def _is_appworld_data_source(data_source: str) -> bool:
    ds = (data_source or "").lower()
    return any(marker in ds for marker in _APPWORLD_DATA_SOURCE_MARKERS)


def _is_math_data_source(data_source: str) -> bool:
    ds = (data_source or "").lower()
    return any(marker in ds for marker in _MATH_DATA_SOURCE_MARKERS)


def build_rewrite_reference_system_prompt(
    reference_trajectory: str,
    guide_prompt: Optional[str] = None,
    reference_template: Optional[str] = None,
    max_reference_chars: Optional[int] = None,
    base_system_prompt: Optional[str] = None,
    data_source: str = "",
) -> str:
    is_code = _is_code_data_source(data_source)
    is_appworld = _is_appworld_data_source(data_source)
    is_math = _is_math_data_source(data_source)

    if base_system_prompt is not None:
        resolved_base = base_system_prompt
    elif is_code:
        resolved_base = LCB_SYSTEM_MESSAGE_GENERIC
    elif is_appworld:
        resolved_base = ""
    elif is_math:
        resolved_base = ""
    else:
        resolved_base = DEFAULT_BASE_SYSTEM_PROMPT

    if guide_prompt is not None:
        resolved_guide_prompt = guide_prompt
    elif is_code:
        resolved_guide_prompt = LCB_REWRITE_GUIDE_SYSTEM_PROMPT
    elif is_appworld:
        resolved_guide_prompt = APPWORLD_REWRITE_GUIDE_SYSTEM_PROMPT
    elif is_math:
        resolved_guide_prompt = MATH_REWRITE_GUIDE_SYSTEM_PROMPT
    else:
        resolved_guide_prompt = DEFAULT_REWRITE_GUIDE_SYSTEM_PROMPT

    if reference_template is not None:
        resolved_template = reference_template
    elif is_code:
        resolved_template = LCB_REWRITE_REFERENCE_TEMPLATE
    elif is_appworld:
        resolved_template = APPWORLD_REWRITE_REFERENCE_TEMPLATE
    elif is_math:
        resolved_template = MATH_REWRITE_REFERENCE_TEMPLATE
    else:
        resolved_template = DEFAULT_REWRITE_REFERENCE_TEMPLATE

    resolved_reference = reference_trajectory or ""
    if max_reference_chars is not None and max_reference_chars > 0 and len(resolved_reference) > max_reference_chars:
        resolved_reference = resolved_reference[:max_reference_chars] + "\n...[truncated reference]"

    sections = []
    if resolved_base.strip():
        sections.append(resolved_base.strip())
    sections.append(resolved_guide_prompt.strip())
    sections.append(resolved_template.format(reference_trajectory=resolved_reference))
    return "\n\n".join(sections)
