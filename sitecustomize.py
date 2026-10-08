# Ray's subprocesses load this file through the launchers' PYTHONPATH.
try:
    from rllm.patches.ray_dashboard_compat import patch_ray_dashboard_asyncio_compat

    patch_ray_dashboard_asyncio_compat()
except Exception:
    pass
