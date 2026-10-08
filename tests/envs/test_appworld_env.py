from uuid import UUID

from rllm.environments.appworld import appworld_env
from rllm.environments.appworld.appworld_env import AppWorldEnv


def test_remote_initialize_uses_isolated_experiment_name(monkeypatch):
    monkeypatch.setenv("APPWORLD_REMOTE_EXPERIMENT_PREFIX", "test run")
    monkeypatch.setattr(
        appworld_env.uuid,
        "uuid4",
        lambda: UUID("12345678-1234-5678-1234-567812345678"),
    )
    env = AppWorldEnv(task={"task_id": "task/with spaces"}, remote_environment_url="http://appworld.example")
    payloads = []

    def fake_remote_post(path, payload):
        payloads.append((path, payload))
        return {
            "output": {
                "instruction": "do the task",
                "supervisor": {
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                    "phone_number": "+1234567890",
                },
            }
        }

    monkeypatch.setattr(env, "_remote_post", fake_remote_post)
    monkeypatch.setattr(env, "_remote_app_descriptions", lambda: "[]")

    env._initialize_remote_world("task/with spaces")

    assert payloads == [
        (
            "/initialize",
            {
                "task_id": "task/with spaces",
                "experiment_name": "test_run_task_with_spaces_12345678_1234_5678_1234_567812345678",
            },
        )
    ]
