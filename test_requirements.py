
# test_requirements.py

import pytest
from datetime import date, timedelta

pytest_plugins = ["test_main"]

import main
import ai_service


# ==================================================
# PAGINATION
# ==================================================

def test_users_pagination_metadata(test_environment):
    client, _, _ = test_environment

    response = client.get(
        "/stats/users?page=1&page_size=2"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["page"] == 1
    assert data["page_size"] == 2
    assert len(data["data"]) <= 2
    assert data["total_items"] >= len(data["data"])


def test_tasks_pagination_metadata(test_environment):
    client, _, _ = test_environment

    response = client.get(
        "/tasks?page=1&page_size=2"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["page"] == 1
    assert data["page_size"] == 2
    assert len(data["data"]) <= 2


def test_pagination_rejects_invalid_page(test_environment):
    client, _, _ = test_environment

    response = client.get("/tasks?page=0")

    assert response.status_code == 422


def test_pagination_rejects_invalid_page_size(test_environment):
    client, _, _ = test_environment

    response = client.get("/tasks?page_size=101")

    assert response.status_code == 422


# ==================================================
# TASK NOT FOUND
# ==================================================

def test_get_nonexistent_task_returns_404(test_environment):
    client, _, _ = test_environment

    response = client.get("/tasks/999999")

    assert response.status_code == 404


def test_update_nonexistent_task_returns_404(
    test_environment,
    api_headers
):
    client, _, _ = test_environment

    response = client.put(
        "/tasks/999999",
        json={
            "title": "Updated task",
            "user_id": 1,
            "due_date": str(date.today()),
            "priority": 1,
            "done": False,
            "notes": ""
        },
        headers=api_headers
    )

    assert response.status_code == 404


def test_delete_nonexistent_task_returns_404(
    test_environment,
    api_headers
):
    client, _, _ = test_environment

    response = client.delete(
        "/tasks/999999",
        headers=api_headers
    )

    assert response.status_code == 404


# ==================================================
# TASK AUTHORIZATION
# ==================================================

def test_post_task_requires_api_key(test_environment):
    client, _, _ = test_environment

    response = client.post(
        "/tasks",
        json={
            "title": "New task",
            "user_id": 1,
            "due_date": str(date.today()),
            "priority": 1,
            "done": False
        }
    )

    assert response.status_code == 401


def test_put_task_requires_api_key(test_environment):
    client, _, _ = test_environment

    response = client.put(
        "/tasks/1",
        json={
            "title": "Updated task",
            "user_id": 1,
            "due_date": str(date.today()),
            "priority": 1,
            "done": False
        }
    )

    assert response.status_code == 401


def test_delete_task_requires_api_key(test_environment):
    client, _, _ = test_environment

    response = client.delete("/tasks/1")

    assert response.status_code == 401


# ==================================================
# TASK VALIDATION
# ==================================================

def test_task_title_cannot_contain_only_spaces(
    test_environment,
    api_headers
):
    client, _, _ = test_environment

    response = client.post(
        "/tasks",
        json={
            "title": "     ",
            "user_id": 1,
            "due_date": str(date.today()),
            "priority": 1,
            "done": False
        },
        headers=api_headers
    )

    assert response.status_code == 422


def test_task_priority_must_be_between_1_and_3(
    test_environment,
    api_headers
):
    client, _, _ = test_environment

    response = client.post(
        "/tasks",
        json={
            "title": "New task",
            "user_id": 1,
            "due_date": str(date.today()),
            "priority": 4,
            "done": False
        },
        headers=api_headers
    )

    assert response.status_code == 422


def test_task_user_must_exist(
    test_environment,
    api_headers
):
    client, _, _ = test_environment

    response = client.post(
        "/tasks",
        json={
            "title": "New task",
            "user_id": 999999,
            "due_date": str(date.today()),
            "priority": 1,
            "done": False
        },
        headers=api_headers
    )

    assert response.status_code == 400


# ==================================================
# OVERDUE TASKS
# ==================================================

def test_overdue_tasks_returns_only_unfinished_past_due_tasks(
    test_environment,
    monkeypatch
):
    client, _, _ = test_environment

    yesterday = date.today() - timedelta(days=1)
    tomorrow = date.today() + timedelta(days=1)

    tasks = [
        {
            "id": 1,
            "title": "Late task",
            "user_id": 1,
            "due_date": str(yesterday),
            "priority": 1,
            "done": False
        },
        {
            "id": 2,
            "title": "Completed task",
            "user_id": 1,
            "due_date": str(yesterday),
            "priority": 1,
            "done": True
        },
        {
            "id": 3,
            "title": "Future task",
            "user_id": 1,
            "due_date": str(tomorrow),
            "priority": 1,
            "done": False
        }
    ]

    monkeypatch.setattr(main, "load_tasks", lambda: tasks)

    response = client.get("/tasks/overdue")

    assert response.status_code == 200

    result = response.json()

    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["days_late"] == 1


def test_overdue_tasks_pagination(test_environment, monkeypatch):
    client, _, _ = test_environment

    yesterday = date.today() - timedelta(days=2)

    tasks = [
        {
            "id": index,
            "title": f"Late task {index}",
            "user_id": 1,
            "due_date": str(yesterday),
            "priority": 1,
            "done": False
        }
        for index in range(1, 4)
    ]

    monkeypatch.setattr(main, "load_tasks", lambda: tasks)

    response = client.get(
        "/tasks/overdue?page=1&page_size=2"
    )

    assert response.status_code == 200
    assert len(response.json()) == 2


# ==================================================
# AI VALIDATION
# ==================================================

def test_ai_rejects_empty_prompt(test_environment):
    client, _, _ = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "   ",
            "conversation_id": "test"
        }
    )

    assert response.status_code == 400


def test_ai_rejects_prompt_over_500_characters(test_environment):
    client, _, _ = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "A" * 501,
            "conversation_id": "test"
        }
    )

    assert response.status_code == 400


def test_ai_accepts_prompt_of_exactly_500_characters(
    test_environment
):
    client, _, _ = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "A" * 500,
            "conversation_id": "test"
        }
    )

    assert response.status_code == 200


def test_ai_requires_conversation_id(test_environment):
    client, _, _ = test_environment

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "Analyze my team"
        }
    )

    assert response.status_code == 422


# ==================================================
# AI LOGGING ON FAILURE
# ==================================================

def test_ai_request_failure_is_logged(
    test_environment,
    monkeypatch
):
    client, _, _ = test_environment

    logged = []

    def fake_ai_response(*args, **kwargs):
        raise RuntimeError("Simulated AI failure")

    def fake_log(prompt, status_str):
        logged.append({
            "prompt": prompt,
            "status": status_str
        })

    monkeypatch.setattr(
        ai_service,
        "ask_claude_ai",
        fake_ai_response
    )

    monkeypatch.setattr(
        ai_service,
        "log_ai_request",
        fake_log
    )

    response = client.post(
        "/ai/ask",
        json={
            "prompt": "Test failure",
            "conversation_id": "failure_test"
        }
    )

    assert response.status_code == 500
    assert len(logged) == 1
    assert logged[0]["prompt"] == "Test failure"
    assert logged[0]["status"] != "SUCCESS"
