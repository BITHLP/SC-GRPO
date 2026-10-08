import copy
import re
from typing import Any, Dict, List, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


WEBSHOP_SYSTEM_PROMPT = """You are a WebShop shopping agent.
Given the current page observation, output exactly one valid action.

Valid action formats:
- search[keywords]
- click[target_text]

Rules:
1) Output one action per turn.
2) Do not output anything except the action.
3) Prefer concise keywords for search.
4) Use click[...] with visible button or option text from the page.
"""

ACTION_PATTERN = re.compile(r"(search\[[^\[\]\n]+\]|click\[[^\[\]\n]+\])", re.IGNORECASE)
ACTION_FULL_PATTERN = re.compile(r"^(search|click)\[(.*)\]$", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class WebShopAgent(BaseAgent):
    def __init__(
        self,
        use_full_conversation: bool = True,
        strip_thinking: bool = True,
        include_available_actions: bool = True,
        fallback_action: Optional[str] = None,
    ):
        self.use_full_conversation = use_full_conversation
        self.strip_thinking = strip_thinking
        self.include_available_actions = include_available_actions
        self.fallback_action = fallback_action

        self._trajectory = Trajectory()
        self.messages: List[Dict[str, Any]] = []
        self.current_observation: Optional[Dict[str, Any]] = None
        self.action_history: List[str] = []
        self.reset()

    def _normalize_action(self, raw_action: str) -> Optional[str]:
        raw_action = raw_action.strip()
        match = ACTION_FULL_PATTERN.match(raw_action)
        if match is None:
            return None

        action_name = match.group(1).lower()
        action_arg = " ".join(match.group(2).split())
        if not action_arg:
            return None

        return f"{action_name}[{action_arg}]"

    def _extract_first_action(self, text: str) -> Optional[str]:
        found = ACTION_PATTERN.search(text)
        if not found:
            return None
        return self._normalize_action(found.group(1))

    def _parse_model_response(self, response: str) -> Optional[str]:
        visible_response = THINK_PATTERN.sub("", response).strip()

        fenced_blocks = re.findall(r"```(?:action)?\s*(.*?)```", visible_response, flags=re.IGNORECASE | re.DOTALL)
        for block in fenced_blocks:
            action = self._extract_first_action(block)
            if action is not None:
                return action

        action = self._extract_first_action(visible_response)
        if action is not None:
            return action

        return None

    def _build_user_message(self, observation: Dict[str, Any]) -> str:
        instruction = observation.get("instruction", "")
        observation_text = observation.get("observation_text", "")
        available_actions = observation.get("available_actions", {})

        parts = []
        if instruction:
            parts.extend(
                [
                    "# Task Instruction",
                    str(instruction),
                    "",
                ]
            )

        parts.extend(
            [
                "# Page Observation",
                str(observation_text),
            ]
        )

        if self.include_available_actions and isinstance(available_actions, dict):
            clickables = available_actions.get("clickables", [])
            if isinstance(clickables, list):
                clipped_clickables = clickables[:50]
            else:
                clipped_clickables = []
            parts.extend(
                [
                    "",
                    "# Available Actions",
                    f"has_search_bar: {available_actions.get('has_search_bar', False)}",
                    f"clickables (up to 50): {clipped_clickables}",
                ]
            )

        if self.action_history:
            parts.extend(["", "# Last Action", self.action_history[-1]])

        parts.extend(
            [
                "",
                "Now output exactly one next action in the required format.",
            ]
        )

        return "\n".join(parts)

    def _strip_assistant_thinking(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        for message in messages:
            if message.get("role") == "assistant":
                message["content"] = THINK_PATTERN.sub("", message.get("content", "")).strip()
        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict[str, Any], **kwargs):
        obs = observation if isinstance(observation, dict) else {"observation_text": str(observation)}

        if done:
            return

        if not self.messages:
            self.messages.append({"role": "system", "content": WEBSHOP_SYSTEM_PROMPT})

        self.current_observation = obs
        self.messages.append({"role": "user", "content": self._build_user_message(obs)})

    def update_from_model(self, response: str, **kwargs) -> Action:
        parsed_action = self._parse_model_response(response)
        action_parse_failed = parsed_action is None
        action_str = parsed_action or ""
        self.messages.append({"role": "assistant", "content": action_str})
        self.action_history.append(action_str)

        step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=action_str,
            model_response=response,
            observation=self.current_observation,
            info={
                "invalid_action_parse": action_parse_failed,
            },
        )
        self._trajectory.steps.append(step)

        return Action(action=parsed_action)

    def reset(self):
        self._trajectory = Trajectory()
        self.messages = []
        self.current_observation = None
        self.action_history = []

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        messages = copy.deepcopy(self.messages)

        if self.strip_thinking:
            messages = self._strip_assistant_thinking(messages)

        if self.use_full_conversation or len(messages) <= 2:
            return messages

        system_message = messages[0]
        start_idx = 1
        for i in range(len(messages) - 1, 0, -1):
            if messages[i].get("role") == "user":
                start_idx = i
                break

        return [system_message] + messages[start_idx:]

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
