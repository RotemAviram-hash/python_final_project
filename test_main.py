import json
from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
import data_service
import ai_service


# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def test_environment(monkeypatch):
    """Prepare isolated test data and a FastAPI test client."""

    users = pd.DataFrame([
        {
            "user_id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "city": "London"
        },
        {
            "user_id": 2,
            "name": "Bob",
            "email": "bob@example.com",
            "city": "Paris"
        },
        {
            "user_id": 3,
            "name": "Charlie",
            "email": "charlie@example.com",
            "city": "London"
        }
    ])

    external_tasks = pd.DataFrame([
        {
            "id": 1,
            "user_id": 1,
            "title": "Task 1",
            "completed": True
        },
        {
            "id": 2,
            "user_id": 1,
            "title": "Task 2",
            "completed": False
        },
        {
            "id": 3,
            "user_id": 2,
            "title": "Task 3",
            "completed": False
        },
        {
            "id": 4,
            "user_id": 3,
            "title": "Task 4",
            "completed": True
        }
    ])

    tasks = []
    ai_logs = []

    monkeypatch.setattr(main, "APP_API_KEY", "test-api-key")

    monkeypatch.setattr(
        data_service,
        "get_cached_data",
        lambda: (users, external_tasks)
    )

    monkeypatch.setattr(
        main,
        "load_tasks",
        lambda: [dict(task) for task in tasks]
    )

    def fake_save_tasks(updated_tasks):
        tasks[:] = [dict(task) for task in updated_tasks]

    monkeypatch.setattr(main, "save_tasks", fake_save_tasks)

    monkeypatch.setattr(
        ai_service,
        "ask_claude_ai",
        lambda prompt, system_prompt, conversation_id:
            "Test AI response"
    )

    def fake_log_ai_request(prompt, status_str):
        ai_logs.append({
            "prompt": prompt,
            "status": status_str
        })

    monkeypatch.setattr(
        ai_service,
        "log_ai_request",
        fake_log_ai_request
    )

    client = TestClient(main.app)

    return client, tasks, ai_logs


@pytest.fixture
def api_headers():
    return {"X-API-Key": "test-api-key"}


def valid_task():
    return {
        "title": "New task",
        "user_id": 1,
        "due_date": "2026-12-31",
        "priority": 2,
        "done": False,
        "notes": "Private notes"
    }


# ==========================================
# Root and Middleware
# ==========================================

def test_root(test_environment):
    client, _, _ = test_environment

    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_process_time_header(test_environment):
    client, _, _ = test_environment

    response = client.get("/")

    assert response.status_code == 200
    assert "X-Process-Time" in response.headers


def test_cors_allows_configured_origin(test_environment):
    client, _, _ = test_environment

    response = client.get(
        "/",
        headers={"Origin": "http://localhost:5173"}
    )

    assert response.headers.get("access-control-allow-origin") == (
        "http://localhost:5173"
    )


# ==========================================
# API Key Security
# ==========================================

def test_create_task_without_api_key(test_environment):
    client, _, _ = test_environment

    response = client.post("/tasks", json=valid_task())

    assert response.status_code == 401


def test_create_task_with_wrong_api_key(test_environment):
    client, _, _ = test_environment

    response = client.post(
        "/tasks",
        json=valid_task(),
        headers={"X-API-Key": "wrong-key"}
    )

    assert response.status_code == 401


def test_update_task_without_api_key(test_environment):
    client, _, _ = test_environment

    response = client.put(
        "/tasks/1",
        json=valid_task()
    )

    assert response.status_code == 401


def test_delete_task_without_api_key(test_environment):
    client, _, _ = test_environment

    response = client.delete("/tasks/1")

    assert response.status_code == 401


def test_missing_app_api_key_returns_503(test_environment, monkeypatch):
    client, _, _ = test_environment

    monkeypatch.setattr(main, "APP_API_KEY", None)

    response = client.post(
        "/tasks",
        json=valid_task(),
        headers={"X-API-Key": "test-api-key"}
    )

    assert response.status_code == 503


# ==========================================
# Task Validation
# ==========================================

def test_create_task_with_invalid_priority(test_environment, api_headers):
    client, _, _ = test_environment

    task = valid_task()
    task["priority"] = 5

    response = client.post(
        "/tasks",
        json=task,
        headers=api_headers
    )

    assert response.status_code == 422


def test_create_task_with_empty_title(test_environment, api_headers):
    client, _, _ = test_environment

    task = valid_task()
    task["title"] = "   "

    response = client.post(
        "/tasks",
        json=task,
        headers=api_headers
    )

    assert response.status_code == 422


def test_create_task_with_invalid_date(test_environment, api_headers):
    client, _, _ = test_environment

    task = valid_task()
    task["due_date"] = "not-a-date"

    response = client.post(
        "/tasks",
        json=task,
        headers=api_headers
    )

    assert response.status_code == 422


def test_create_task_with_nonexistent_user(test_environment, api_headers):
    client, _, _ = test_environment

    task = valid_task()
    task["user_id"] = 999

    response = client.post(
        "/tasks",
        json=task,
        headers=api_headers
    )

    assert response.status_code == 400


# ==========================================
# Tasks CRUD
# ==========================================

def test_create_task(test_environment, api_headers):
    client, tasks, _ = test_environment

    response = client.post(
        "/tasks",
        json=valid_task(),
        headers=api_headers
    )

    assert response.status_code == 201
    assert response.json()["title"] == "New task"
    assert response.json()["id"] == 1
    assert len(tasks) == 1


def test_task_output_does_not_include_notes(test_environment, api_headers):
    client, _, _ = test_environment

    response = client.post(
        "/tasks",
        json=valid_task(),
        headers=api_headers
    )

    assert response.status_code == 201
    assert "notes" not in response.json()


def test_get_tasks(test_environment, api_headers):
    client, _, _ = test_environment

    client.post(
        "/tasks",
        json=valid_task(),
        headers=api_headers
    )

    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json()["total_items"] == 1
    assert len(response.json()["data"]) == 1


def test_get_task_by_id(test_environment, api_headers):
    client, _, _ = test_environment

    client.post(
        "/tasks",
        json=valid_task(),
        headers=api_headers
    )

    response = client.get("/tasks/1")

    assert response.status_code == 200
    assert response.json()["id"] == 1


def test_update_task(test_environment, api_headers):
    client, _, _ = test_environment

    client.post(
        "/tasks",
        json=valid_task(),
        headers=api_headers
    )

    updated_task = valid_task()
    updated_task["title"] = "Updated task"
    updated_task["done"] = True

    response = client.put(
        "/tasks/1",
        json=updated_task,
        headers=api_headers
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Updated task"
    assert response.json()["done"] is True


def test_delete_task(test_environment, api_headers):
    client, tasks, _ = test_environment

    client.post(
        "/tasks",
        json=valid_task(),
        headers=api_headers
    )

    response = client.delete(
        "/tasks/1",
        headers=api_headers
    )

    assert response.status_code == 204
    assert len(tasks) == 0


def test_get_task_not_found(test_environment):
    client, _, _ = test_environment

    response = client.get("/tasks/999")

    assert response.status_code == 404


def test_update_task_not_found(test_environment, api_headers):
    client, _, _ = test_environment

    response = client.put(
        "/tasks/999",
        json=valid_task(),
        headers=api_headers
    )

    assert response.status_code == 404


def test_delete_task_not_found(test_environment, api_headers):
    client, _, _ = test_environment

    response = client.delete(
        "/tasks/999",
        headers=api_headers
    )

    assert response.status_code == 404


# ==========================================
# Pagination
# ==========================================

def test_tasks_pagination(test_environment, api_headers):
    client, _, _ = test_environment

    for number in range(3):
        task = valid_task()
        task["title"] = f"Task {number}"

        client.post(
            "/tasks",
            json=task,
            headers=api_headers
        )

    response = client.get("/tasks?page=1&page_size=2")

    assert response.status_code == 200
    assert response.json()["page"] == 1
    assert response.json()["page_size"] == 2
    assert response.json()["total_items"] == 3
    assert len(response.json()["data"]) == 2


def test_tasks_second_page(test_environment, api_headers):
    client, _, _ = test_environment

    for number in range(3):
        task = valid_task()
        task["title"] = f"Task {number}"

        client.post(
            "/tasks",
            json=task,
            headers=api_headers
        )

    response = client.get("/tasks?page=2&page_size=2")

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_invalid_pagination_page(test_environment):
    client, _, _ = test_environment

    response = client.get("/tasks?page=0")

    assert response.status_code == 422


def test_invalid_pagination_size(test_environment):
    client, _, _ = test_environment

    response = client.get("/tasks?page_size=101")

    assert response.status_code == 422


def test_users_statistics_pagination(test_environment):
    client, _, _ = test_environment

    response = client.get("/stats/users?page=1&page_size=2")

    assert response.status_code == 200
    assert response.json()["total_items"] == 3
    assert len(response.json()["data"]) == 2


# ==========================================
# Overdue Tasks
# ==========================================

def test_overdue_tasks(test_environment):
    client, tasks, _ = test_environment

    today = date.today()

    tasks.extend([
        {
            "id": 1,
            "title": "Late task",
            "user_id": 1,
            "due_date": (today - timedelta(days=3)).isoformat(),
            "priority": 1,
            "done": False
        },
        {
            "id": 2,
            "title": "Completed task",
            "user_id": 1,
            "due_date": (today - timedelta(days=5)).isoformat(),
            "priority": 2,
            "done": True
        },
        {
            "id": 3,
            "title": "Future task",
            "user_id": 1,
            "due_date": (today + timedelta(days=5)).isoformat(),
            "priority": 3,
            "done": False
        }
    ])

    response = client.get("/tasks/overdue")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["id"] == 1
    assert response.json()[0]["days_late"] == 3


def test_overdue_tasks_pagination(test_environment):
    client, tasks, _ = test_environment

    today = date.today()

    for task_id in range(1, 4):
        tasks.append({
            "id": task_id,
            "title": f"Late task {task_id}",
            "user_id": 1,
            "due_date": (today - timedelta(days=task_id)).isoformat(),
            "priority": 1,
            "done": False
        })

    response = client.get("/tasks/overdue?page=1&page_size=2")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_overdue_tasks_excludes_tasks_due_today(test_environment):
    client, tasks, _ = test_environment

    tasks.append({
        "id": 1,
        "title": "Due today",
        "user_id": 1,
        "due_date": date.today().isoformat(),
        "priority": 1,
        "done": False
    })

    response = client.get("/tasks/overdue")

    assert response.status_code == 200
    assert response.json() == []


# ==========================================
# Statistics
# ==========================================

def test_users_statistics(test_environment):
    client, _, _ = test_environment

    response = client.get("/stats/users")

    assert response.status_code == 200
    assert response.json()["total_items"] == 3


def test_summary_statistics(test_environment):
    client, _, _ = test_environment

    response = client.get("/stats/summary")

    assert response.status_code == 200
    assert response.json()["total_users"] == 3
    assert response.json()["total_tasks"] == 4
    assert response.json()["completed_tasks"] == 2

def test_cities_statistics(test_environment):
    client, _, _ = test_environment

    response = client.get("/stats/cities")

    assert response.status_code == 200

    cities_data = response.json()

    assert "London" in cities_data["true"]
    assert "London" in cities_data["false"]
    assert cities_data["true"]["London"] == 2
    assert cities_data["false"]["London"] == 1

def test_low_performers(test_environment):
    client, _, _ = test_environment

    response = client.get("/stats/low-performers")

    assert response.status_code == 200
    assert response.json()["total_items"] == 1
    assert response.json()["low_performers"][0]["name"] == "Bob"


# ==========================================
# External API Failure Handling
# ==========================================

def test_users_statistics_when_external_api_fails(
    test_environment,
    monkeypatch
):
    client, _, _ = test_environment

    monkeypatch.setattr(
        data_service,
        "get_cached_data",
        lambda: (None, None)
    )

    response = client.get("/stats/users")

    assert response.status_code == 503


def test_summary_when_external_api_fails(test_environment, monkeypatch):
    client, _, _ = test_environment

    monkeypatch.setattr(
        data_service,
        "get_cached_data",
        lambda: (None, None)
    )

    response = client.get("/stats/summary")

    assert response.status_code == 503


def test_cities_when_external_api_fails(test_environment, monkeypatch):
    client, _, _ = test_environment

    monkeypatch.setattr(
        data_service,
        "get_cached_data",
        lambda: (None, None)
    )

    response = client.get("/stats/cities")

    assert response.status_code == 503


def test_create_task_when_external_users_unavailable(
    test_environment,
    api_headers,
    monkeypatch
):
    client, _, _ = test_environment

    monkeypatch.setattr(
        data_service,
        "get_cached_data",
        lambda: (None, None)
    )

    response = client.post(
        "/tasks",
        json=valid_task(),
        headers=api_headers
    )

    assert response.status_code == 503


# ==========================================
# AI
# ==========================================

def test_ai_mock_response(test_environment):
    client, _, _ = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "Summarize team performance",
            "conversation_id": "test-conversation"
        }
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Test AI response"
    assert response.json()["conversation_id"] == "test-conversation"


def test_ai_rejects_empty_prompt(test_environment):
    client, _, ai_logs = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "   ",
            "conversation_id": "test-conversation"
        }
    )

    assert response.status_code == 400
    assert ai_logs[-1]["status"] == "BAD_REQUEST"


def test_ai_rejects_long_prompt(test_environment):
    client, _, ai_logs = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "A" * 501,
            "conversation_id": "test-conversation"
        }
    )

    assert response.status_code == 400
    assert ai_logs[-1]["status"] == "BAD_REQUEST"


def test_ai_rejects_empty_conversation_id(test_environment):
    client, _, _ = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "Hello",
            "conversation_id": "   "
        }
    )

    assert response.status_code == 422


def test_ai_logs_request_status(test_environment):
    client, _, ai_logs = test_environment

    client.post(
        "/ai/ask",
        json={
            "prompt": "Hello",
            "conversation_id": "conversation-1"
        }
    )

    assert len(ai_logs) == 1
    assert ai_logs[0]["status"] == "SUCCESS"


def test_ai_when_external_data_unavailable(test_environment, monkeypatch):
    client, _, ai_logs = test_environment

    monkeypatch.setattr(
        data_service,
        "get_cached_data",
        lambda: (None, None)
    )

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "Analyze the team",
            "conversation_id": "conversation-1"
        }
    )

    assert response.status_code == 503
    assert ai_logs[-1]["status"] == "ERROR"


def test_ai_logs_stats_endpoint(test_environment, monkeypatch):
    client, _, _ = test_environment

    expected_stats = {
        "success_rate": 80.0,
        "avg_prompt_length": 25.5,
        "total_requests": 10
    }

    monkeypatch.setattr(
        ai_service,
        "get_ai_logs_stats",
        lambda: expected_stats
    )

    response = client.get("/ai/logs/stats")

    assert response.status_code == 200
    assert response.json() == expected_stats


# ==========================================
# JSON File Management
# ==========================================

def test_load_tasks_when_file_does_not_exist(monkeypatch, tmp_path):
    missing_file = tmp_path / "missing_tasks.json"

    monkeypatch.setattr(main, "TASKS_FILE", str(missing_file))

    assert main.load_tasks() == []


def test_load_tasks_with_invalid_json(monkeypatch, tmp_path):
    invalid_file = tmp_path / "invalid_tasks.json"
    invalid_file.write_text("{invalid json", encoding="utf-8")

    monkeypatch.setattr(main, "TASKS_FILE", str(invalid_file))

    with pytest.raises(RuntimeError):
        main.load_tasks()


def test_load_tasks_with_valid_json(monkeypatch, tmp_path):
    tasks_file = tmp_path / "tasks.json"

    expected_tasks = [
        {"id": 1, "title": "Test task"}
    ]

    tasks_file.write_text(
        json.dumps(expected_tasks),
        encoding="utf-8"
    )

    monkeypatch.setattr(main, "TASKS_FILE", str(tasks_file))

    assert main.load_tasks() == expected_tasks


def test_save_tasks_to_json_file(monkeypatch, tmp_path):
    tasks_file = tmp_path / "tasks.json"

    monkeypatch.setattr(main, "TASKS_FILE", str(tasks_file))

    expected_tasks = [
        {"id": 1, "title": "Saved task"}
    ]

    main.save_tasks(expected_tasks)

    with open(tasks_file, "r", encoding="utf-8") as file:
        saved_tasks = json.load(file)

    assert saved_tasks == expected_tasks