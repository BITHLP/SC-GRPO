import os
import sys
import threading
import uuid
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

from rllm.environments.base.base_env import BaseEnv


DEFAULT_WEBSHOP_ROOT = os.path.join(os.path.expanduser("~"), "WebShop")
DEFAULT_SERIALIZE_BACKEND_ACCESS = True

_env_class_lock = threading.Lock()
_env_class = None
_backend_server_lock = threading.Lock()
_backend_server_cache: Dict[Tuple[str, int, int], Any] = {}
_backend_execution_lock_cache: Dict[Tuple[str, int, int], threading.Lock] = {}
_num_goals_cache: Dict[Tuple[str, int, int], int] = {}


def resolve_webshop_root(webshop_root: Optional[str] = None) -> str:
    root = webshop_root or os.environ.get("WEBSHOP_ROOT") or DEFAULT_WEBSHOP_ROOT
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"WebShop root does not exist: {root}. "
            "Set WEBSHOP_ROOT or pass webshop_root."
        )
    return root


def ensure_webshop_path(webshop_root: str) -> None:
    if webshop_root not in sys.path:
        sys.path.insert(0, webshop_root)


def _candidate_java_homes():
    seen = set()
    candidates = []

    def _add(path: Optional[str]) -> None:
        if not path:
            return
        resolved = os.path.abspath(os.path.expanduser(path))
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    java_home = os.environ.get("JAVA_HOME")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    python_prefix = os.path.dirname(os.path.dirname(sys.executable))

    _add(os.path.join(conda_prefix, "lib", "jvm") if conda_prefix else None)
    _add(os.path.join(python_prefix, "lib", "jvm"))
    _add(conda_prefix)
    _add(python_prefix)
    _add(java_home)
    return candidates


def _find_jvm_path(java_home: str) -> Optional[str]:
    candidates = [
        os.path.join(java_home, "lib", "server", "libjvm.so"),
        os.path.join(java_home, "jre", "lib", "amd64", "server", "libjvm.so"),
        os.path.join(java_home, "jre", "lib", "server", "libjvm.so"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _prepend_env_path(name: str, value: str) -> None:
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if value in entries:
        entries = [entry for entry in entries if entry != value]
    entries.insert(0, value)
    os.environ[name] = os.pathsep.join(entries)


def configure_java_runtime() -> None:
    existing_jvm_path = os.environ.get("JVM_PATH")
    if existing_jvm_path and os.path.exists(existing_jvm_path):
        return

    for java_home in _candidate_java_homes():
        jvm_path = _find_jvm_path(java_home)
        if not jvm_path:
            continue
        os.environ["JAVA_HOME"] = java_home
        os.environ["JVM_PATH"] = jvm_path
        java_bin_dir = os.path.join(java_home, "bin")
        if os.path.isdir(java_bin_dir):
            _prepend_env_path("PATH", java_bin_dir)
        _prepend_env_path("LD_LIBRARY_PATH", os.path.dirname(jvm_path))
        return


def get_webshop_env_class():
    global _env_class
    if _env_class is not None:
        return _env_class

    with _env_class_lock:
        if _env_class is None:
            webshop_root = resolve_webshop_root()
            ensure_webshop_path(webshop_root)
            configure_java_runtime()
            try:
                from web_agent_site.envs import WebAgentTextEnv  # type: ignore[import-not-found]
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to import WebShop env from root {webshop_root}. "
                    "Please ensure WebShop dependencies are installed in this environment."
                ) from exc
            _env_class = WebAgentTextEnv
    return _env_class


def _safe_bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _should_serialize_backend_access() -> bool:
    return _safe_bool_from_env("WEBSHOP_SERIALIZE_BACKEND_ACCESS", DEFAULT_SERIALIZE_BACKEND_ACCESS)


def get_shared_backend_server(observation_mode: str, num_products: int, human_goals: int) -> Tuple[Any, threading.Lock]:
    key = (observation_mode, num_products, human_goals)
    cached = _backend_server_cache.get(key)
    if cached is not None:
        return cached, _backend_execution_lock_cache[key]

    with _backend_server_lock:
        cached = _backend_server_cache.get(key)
        if cached is not None:
            return cached, _backend_execution_lock_cache[key]

        WebAgentTextEnv = get_webshop_env_class()
        warmup_env = WebAgentTextEnv(
            observation_mode=observation_mode,
            num_products=num_products,
            human_goals=human_goals,
            session_prefix="warmup_",
        )
        backend_server = getattr(warmup_env, "server", None)
        if backend_server is None:
            raise RuntimeError("Failed to initialize shared WebShop backend server.")

        warmup_session = getattr(warmup_env, "session", None)
        user_sessions = getattr(backend_server, "user_sessions", None)
        if isinstance(user_sessions, dict) and isinstance(warmup_session, str):
            user_sessions.pop(warmup_session, None)

        _backend_server_cache[key] = backend_server
        _backend_execution_lock_cache[key] = threading.Lock()
        return backend_server, _backend_execution_lock_cache[key]


def _backend_user_sessions(env: Any):
    backend_server = getattr(env, "server", None)
    user_sessions = getattr(backend_server, "user_sessions", None)
    if isinstance(user_sessions, dict):
        return user_sessions
    return None


def _remove_backend_session_for_env(env: Any) -> None:
    internal_session = getattr(env, "session", None)
    user_sessions = _backend_user_sessions(env)
    if isinstance(user_sessions, dict) and isinstance(internal_session, str):
        user_sessions.pop(internal_session, None)


def _build_observation(env: Any, raw_observation: Any, step_id: int) -> Dict[str, Any]:
    available_actions: Dict[str, Any] = {"has_search_bar": False, "clickables": []}
    try:
        maybe_actions = env.get_available_actions()
        if isinstance(maybe_actions, dict):
            available_actions = maybe_actions
    except Exception:
        pass

    return {
        "instruction": getattr(env, "instruction_text", ""),
        "observation_text": raw_observation,
        "available_actions": available_actions,
        "step_id": step_id,
    }


def fetch_num_goals(
    observation_mode: str = "text_rich",
    num_products: int = 1000,
    human_goals: int = 0,
) -> int:
    key = (observation_mode, num_products, human_goals)
    if key in _num_goals_cache:
        return _num_goals_cache[key]

    backend_server, _ = get_shared_backend_server(
        observation_mode=observation_mode,
        num_products=num_products,
        human_goals=human_goals,
    )
    goals = getattr(backend_server, "goals", [])
    n_goals = len(goals)
    _num_goals_cache[key] = n_goals
    return n_goals


class WebShopTextEnv(BaseEnv):
    def __init__(
        self,
        task: Optional[Dict[str, Any]] = None,
        observation_mode: str = "text_rich",
        num_products: int = 1000,
        max_steps: int = 15,
        session_prefix: Optional[str] = "rllm_",
        human_goals: int = 0,
        serialize_backend_access: Optional[bool] = None,
        webshop_root: Optional[str] = None,
        **kwargs,
    ):
        self.task = task or {}
        self.observation_mode = observation_mode
        self.num_products = num_products
        self.max_steps = max_steps
        self.human_goals = human_goals
        self.current_step = 0

        if webshop_root is not None:
            os.environ["WEBSHOP_ROOT"] = os.path.abspath(os.path.expanduser(webshop_root))

        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(f"Unknown WebShopTextEnv arguments: {unknown}")

        self.serialize_backend_access = (
            _should_serialize_backend_access()
            if serialize_backend_access is None
            else bool(serialize_backend_access)
        )
        prefix = session_prefix if session_prefix is not None else ""
        self.session_id = f"{prefix}{uuid.uuid4().hex}"
        self._closed = False

        backend_server, backend_lock = get_shared_backend_server(
            observation_mode=self.observation_mode,
            num_products=self.num_products,
            human_goals=self.human_goals,
        )
        self._backend_lock = backend_lock if self.serialize_backend_access else None

        WebAgentTextEnv = get_webshop_env_class()
        internal_session_prefix = f"{self.session_id}_"

        backend_guard = self._backend_lock if self._backend_lock is not None else nullcontext()
        with backend_guard:
            self._env = WebAgentTextEnv(
                observation_mode=self.observation_mode,
                num_products=self.num_products,
                human_goals=self.human_goals,
                session_prefix=internal_session_prefix,
                server=backend_server,
            )

    def _goal_idx(self) -> Optional[int]:
        goal_idx = self.task.get("goal_idx") if isinstance(self.task, dict) else None
        if goal_idx is None:
            return None
        if isinstance(goal_idx, int):
            return goal_idx
        if isinstance(goal_idx, str) and goal_idx.isdigit():
            return int(goal_idx)
        return None

    @property
    def num_goals(self) -> int:
        return fetch_num_goals(
            observation_mode=self.observation_mode,
            num_products=self.num_products,
            human_goals=self.human_goals,
        )

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.current_step = 0
        goal_idx = self._goal_idx()

        backend_guard = self._backend_lock if self._backend_lock is not None else nullcontext()
        with backend_guard:
            old_internal_session = getattr(self._env, "session", None)
            if goal_idx is None:
                raw_observation, _ = self._env.reset()
            else:
                raw_observation, _ = self._env.reset(session=goal_idx)

            new_internal_session = getattr(self._env, "session", None)
            user_sessions = _backend_user_sessions(self._env)
            if isinstance(user_sessions, dict) and isinstance(old_internal_session, str):
                if old_internal_session != new_internal_session:
                    user_sessions.pop(old_internal_session, None)

            observation = _build_observation(self._env, raw_observation, self.current_step)

        return observation, {"goal_idx": goal_idx}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        action_str = action.strip() if isinstance(action, str) else ""

        backend_guard = self._backend_lock if self._backend_lock is not None else nullcontext()
        with backend_guard:
            raw_observation, reward, done, env_info = self._env.step(action_str)
            observation = _build_observation(self._env, raw_observation, self.current_step)

        info = dict(env_info) if isinstance(env_info, dict) else {}
        if not action_str:
            info["empty_action"] = True
        info["action"] = action_str
        info["step_id"] = self.current_step

        done = bool(done)
        if self.current_step >= self.max_steps and not done:
            done = True
            info["terminated_by"] = "max_steps"

        return observation, float(reward), done, info

    def close(self):
        if self._closed:
            return

        try:
            backend_guard = self._backend_lock if self._backend_lock is not None else nullcontext()
            with backend_guard:
                self._env.close()
                _remove_backend_session_for_env(self._env)
        finally:
            self._closed = True

    @staticmethod
    def from_dict(env_args: Dict[str, Any]) -> "WebShopTextEnv":
        args = dict(env_args)

        for metadata_key in ("data_source", "uid", "index"):
            args.pop(metadata_key, None)

        task = args.pop("task", None)
        if task is None:
            task = {}
        elif not isinstance(task, dict):
            task = {"task": task}

        if "goal_idx" in args:
            task = dict(task)
            task["goal_idx"] = args.pop("goal_idx")

        return WebShopTextEnv(task=task, **args)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
