import asyncio
import importlib
import os
from pathlib import Path


_PATCH_ATTR = "__rllm_asyncio_compat_patched__"
_DISABLE_VALUES = {"0", "false", "no", "off"}


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def prepare_ray_startup_environment() -> None:
    repo_root = _repo_root()
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if repo_root in entries:
        entries = [entry for entry in entries if entry != repo_root]
    entries.insert(0, repo_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)
    os.environ.setdefault("RLLM_PATCH_RAY_DASHBOARD_ASYNCIO", "1")


def patch_ray_dashboard_asyncio_compat() -> bool:
    raw = os.environ.get("RLLM_PATCH_RAY_DASHBOARD_ASYNCIO", "1")
    if raw.strip().lower() in _DISABLE_VALUES:
        return False

    try:
        module = importlib.import_module("ray.dashboard.modules.aggregator.multi_consumer_event_buffer")
    except Exception:
        return False

    buffer_cls = getattr(module, "MultiConsumerEventBuffer", None)
    if buffer_cls is None:
        return False
    if getattr(buffer_cls, _PATCH_ATTR, False):
        return True

    def _patched_init(self, max_size, max_batch_size, common_metric_tags=None):
        self._buffer = module.deque(maxlen=max_size)
        self._max_size = max_size
        self._lock = asyncio.Lock()

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and getattr(self._lock, "_loop", None) is None:
            self._lock._loop = running_loop

        self._has_new_events_to_consume = asyncio.Condition(self._lock)
        self._consumers = {}
        self._max_batch_size = max_batch_size
        self._common_metrics_tags = common_metric_tags or {}
        self._metric_recorder = module.OpenTelemetryMetricRecorder()
        self.evicted_events_metric_name = f"{module.AGGREGATOR_AGENT_METRIC_PREFIX}_queue_dropped_events"
        self._metric_recorder.register_counter_metric(
            self.evicted_events_metric_name,
            "Total number of events dropped because the publish/buffer queue was full.",
        )

    buffer_cls.__init__ = _patched_init
    setattr(buffer_cls, _PATCH_ATTR, True)
    return True
