import json

import pytest

import main
import data_service


pytest_plugins = ["test_main"]


# --------------------------------------------------
# 1. TASKS JSON FILE
# --------------------------------------------------

def test_load_tasks_when_file_does_not_exist(monkeypatch, tmp_path):
    """
    The application should start with an empty task list
    when tasks.json does not exist yet.
    """
    tasks_file = tmp_path / "tasks.json"
    monkeypatch.setattr(main, "TASKS_FILE", str(tasks_file))

    assert main.load_tasks() == []


def test_save_and_load_tasks_roundtrip(monkeypatch, tmp_path):
    """
    Tasks saved to JSON should be loaded back correctly.
    """
    tasks_file = tmp_path / "tasks.json"
    monkeypatch.setattr(main, "TASKS_FILE", str(tasks_file))

    tasks = [
        {
            "id": 1,
            "title": "Complete Python project",
            "user_id": 1,
            "due_date": "2026-12-01",
            "priority": 2,
            "done": False,
            "notes": "Final testing"
        }
    ]

    main.save_tasks(tasks)

    loaded_tasks = main.load_tasks()

    assert loaded_tasks == tasks
    assert tasks_file.exists()


def test_load_tasks_with_invalid_json(monkeypatch, tmp_path):
    """
    Invalid JSON should raise a clear RuntimeError.
    """
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text("{ invalid json }", encoding="utf-8")

    monkeypatch.setattr(main, "TASKS_FILE", str(tasks_file))

    with pytest.raises(RuntimeError):
        main.load_tasks()


# --------------------------------------------------
# 2. TASK API RESPONSE
# --------------------------------------------------

def test_task_response_does_not_include_notes(test_environment):
    """
    TaskOut should not expose the internal notes field.
    """
    client, _, _ = test_environment

    task_data = {
        "title": "Final project task",
        "user_id": 1,
        "due_date": "2026-12-01",
        "priority": 2,
        "done": False,
        "notes": "This should not appear in the response"
    }

    response = client.post(
        "/tasks",
        json=task_data,
        headers={"X-API-Key": "test-api-key"}
    )

    assert response.status_code == 201
    assert "notes" not in response.json()


# --------------------------------------------------
# 3. API KEY SECURITY
# --------------------------------------------------

@pytest.mark.parametrize(
    "method,url",
    [
        ("post", "/tasks"),
        ("put", "/tasks/1"),
        ("delete", "/tasks/1"),
    ]
)
def test_mutating_routes_reject_invalid_api_key(
    test_environment,
    method,
    url
):
    """
    Every task-mutating route should reject an invalid API key.
    """
    client, _, _ = test_environment

    headers = {"X-API-Key": "wrong-api-key"}

    task_data = {
        "title": "Security test",
        "user_id": 1,
        "due_date": "2026-12-01",
        "priority": 2,
        "done": False
    }

    if method == "delete":
        response = client.delete(url, headers=headers)
    else:
        response = getattr(client, method)(
        url,
        json=task_data,
        headers=headers
    )

    assert response.status_code == 401


# --------------------------------------------------
# 4. EXTERNAL DATA HANDLING
# --------------------------------------------------

def test_load_data_with_empty_external_responses(monkeypatch):
    """
    Empty but valid API responses should still produce
    DataFrames with the expected columns.
    """
    monkeypatch.setattr(
        data_service,
        "fetch_json",
        lambda path: []
    )

    data_service.users_cache = None
    data_service.tasks_cache = None

    try:
        result = data_service.load_data()

        assert result is True

        users_df, tasks_df = data_service.get_cached_data()

        assert list(users_df.columns) == [
            "user_id",
            "name",
            "email",
            "city"
        ]

        assert list(tasks_df.columns) == [
            "id",
            "user_id",
            "title",
            "completed"
        ]

        assert users_df.empty
        assert tasks_df.empty

    finally:
        data_service.users_cache = None
        data_service.tasks_cache = None


# --------------------------------------------------
# 5. TASK FILE VALIDATION
# --------------------------------------------------
def test_load_tasks_rejects_non_list_json(monkeypatch, tmp_path):
    tasks_file = tmp_path / "tasks.json"

    tasks_file.write_text(
        json.dumps({"task": "not a list"}),
        encoding="utf-8"
    )

    monkeypatch.setattr(main, "TASKS_FILE", str(tasks_file))

    with pytest.raises(ValueError):
        main.load_tasks()