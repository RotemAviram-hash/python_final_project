
import csv
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import data_service
import ai_service


# ==================================================
# DATA SERVICE TESTS
# ==================================================

@pytest.fixture
def reset_data_cache(monkeypatch):
    """Reset global data cache before each test."""
    monkeypatch.setattr(data_service, "users_cache", None)
    monkeypatch.setattr(data_service, "tasks_cache", None)


def test_fetch_json_success(monkeypatch):
    """Test successful JSON fetching."""

    def fake_get(url, timeout):
        assert url == "https://jsonplaceholder.typicode.com/users"
        assert timeout == 5

        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [{"id": 1, "name": "Alice"}]
        )

    monkeypatch.setattr(data_service.requests, "get", fake_get)

    result = data_service.fetch_json("users")

    assert result == [{"id": 1, "name": "Alice"}]


def test_fetch_json_request_error(monkeypatch):
    """Test that request errors return None."""

    def fake_get(url, timeout):
        raise data_service.RequestException("Connection failed")

    monkeypatch.setattr(data_service.requests, "get", fake_get)

    result = data_service.fetch_json("users")

    assert result is None


def test_fetch_json_invalid_json(monkeypatch):
    """Test that invalid JSON returns None."""

    def fake_get(url, timeout):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: (_ for _ in ()).throw(ValueError("Invalid JSON"))
        )

    monkeypatch.setattr(data_service.requests, "get", fake_get)

    result = data_service.fetch_json("users")

    assert result is None


def test_load_data_success(monkeypatch, reset_data_cache):
    """Test fetching, processing and caching users and tasks."""

    users = [
        {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
            "address": {"city": "London"}
        }
    ]

    tasks = [
        {
            "id": 10,
            "userId": 1,
            "title": "Task 1",
            "completed": True
        }
    ]

    def fake_fetch(path):
        return users if path == "users" else tasks

    monkeypatch.setattr(data_service, "fetch_json", fake_fetch)

    result = data_service.load_data()

    assert result is True

    users_df, tasks_df = data_service.get_cached_data()

    assert list(users_df.columns) == [
        "user_id", "name", "email", "city"
    ]

    assert list(tasks_df.columns) == [
        "id", "user_id", "title", "completed"
    ]

    assert users_df.iloc[0]["city"] == "London"
    assert tasks_df.iloc[0]["user_id"] == 1
    assert tasks_df.iloc[0]["completed"] is True or (
        tasks_df.iloc[0]["completed"] == True
    )


def test_load_data_external_api_failure(monkeypatch, reset_data_cache):
    """Test that failed external requests return False."""

    monkeypatch.setattr(
        data_service,
        "fetch_json",
        lambda path: None
    )

    result = data_service.load_data()

    assert result is False
    assert data_service.users_cache is None
    assert data_service.tasks_cache is None


def test_load_data_invalid_format(monkeypatch, reset_data_cache):
    """Test that unexpected API data formats are rejected."""

    def fake_fetch(path):
        return {"unexpected": "format"}

    monkeypatch.setattr(data_service, "fetch_json", fake_fetch)

    result = data_service.load_data()

    assert result is False


def test_get_cached_data_does_not_reload(monkeypatch):
    """Test that existing cache is reused."""

    users_df = pd.DataFrame([
        {"user_id": 1, "name": "Alice", "email": "a@test.com", "city": "London"}
    ])

    tasks_df = pd.DataFrame([
        {"id": 1, "user_id": 1, "title": "Task", "completed": True}
    ])

    monkeypatch.setattr(data_service, "users_cache", users_df)
    monkeypatch.setattr(data_service, "tasks_cache", tasks_df)

    def unexpected_reload():
        pytest.fail("Cache should not be reloaded")

    monkeypatch.setattr(data_service, "load_data", unexpected_reload)

    users_result, tasks_result = data_service.get_cached_data()

    assert users_result is users_df
    assert tasks_result is tasks_df


def test_get_merged_data():
    """Test merging users and tasks by user_id."""

    users_df = pd.DataFrame([
        {"user_id": 1, "name": "Alice", "email": "a@test.com", "city": "London"},
        {"user_id": 2, "name": "Bob", "email": "b@test.com", "city": "Paris"}
    ])

    tasks_df = pd.DataFrame([
        {"id": 10, "user_id": 1, "title": "Task", "completed": True}
    ])

    original = data_service.get_cached_data

    try:
        data_service.get_cached_data = lambda: (users_df, tasks_df)

        result = data_service.get_merged_data()

        assert len(result) == 2
        assert "user_id" in result.columns
        assert "name" in result.columns
        assert result[result["user_id"] == 2]["name"].iloc[0] == "Bob"

    finally:
        data_service.get_cached_data = original


def test_calculate_users_performance():
    """Test task completion statistics for each user."""

    users_df = pd.DataFrame([
        {"user_id": 1, "name": "Alice", "email": "a@test.com", "city": "London"},
        {"user_id": 2, "name": "Bob", "email": "b@test.com", "city": "Paris"}
    ])

    tasks_df = pd.DataFrame([
        {"id": 1, "user_id": 1, "title": "Task 1", "completed": True},
        {"id": 2, "user_id": 1, "title": "Task 2", "completed": True},
        {"id": 3, "user_id": 2, "title": "Task 3", "completed": False}
    ])

    original = data_service.get_merged_data

    try:
        data_service.get_merged_data = lambda: pd.merge(
            tasks_df,
            users_df,
            on="user_id",
            how="right"
        )

        result = data_service.calculate_users_performance()

        alice = result[result["user_id"] == 1].iloc[0]
        bob = result[result["user_id"] == 2].iloc[0]

        assert alice["total_tasks"] == 2
        assert alice["completed_tasks"] == 2
        assert alice["completion_percentage"] == 100

        assert bob["total_tasks"] == 1
        assert bob["completed_tasks"] == 0
        assert bob["completion_percentage"] == 0

    finally:
        data_service.get_merged_data = original


def test_get_low_performers_data():
    """Test identifying users below the average."""

    stats = pd.DataFrame([
        {
            "user_id": 1,
            "name": "Alice",
            "completion_percentage": 100.0
        },
        {
            "user_id": 2,
            "name": "Bob",
            "completion_percentage": 50.0
        },
        {
            "user_id": 3,
            "name": "Charlie",
            "completion_percentage": 0.0
        }
    ])

    original = data_service.calculate_users_performance

    try:
        data_service.calculate_users_performance = lambda: stats

        low_users, mean, std = data_service.get_low_performers_data()

        assert mean == 50.0
        assert len(low_users) == 1
        assert low_users[0]["name"] == "Charlie"
        assert std > 0

    finally:
        data_service.calculate_users_performance = original


def test_get_cities_pivot_table():
    """Test city pivot table."""

    merged = pd.DataFrame([
        {"city": "London", "completed": True, "id": 1},
        {"city": "London", "completed": False, "id": 2},
        {"city": "Paris", "completed": True, "id": 3}
    ])

    original = data_service.get_merged_data

    try:
        data_service.get_merged_data = lambda: merged

        result = data_service.get_cities_pivot_table()

        assert result[True]["London"] == 1
        assert result[False]["London"] == 1
        assert result[True]["Paris"] == 1

    finally:
        data_service.get_merged_data = original


# ==================================================
# AI SERVICE TESTS
# ==================================================

@pytest.fixture
def reset_ai_history(monkeypatch, tmp_path):
    """Reset AI conversation history and use a temporary log file."""

    monkeypatch.setattr(ai_service, "conversation_history", {})
    monkeypatch.setattr(
        ai_service,
        "AI_LOGS_FILE",
        str(tmp_path / "ai_logs.csv")
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_mock_ai_response_without_api_key(reset_ai_history):
    """Test mock response when no API key is configured."""

    result = ai_service.get_ai_response(
        prompt="Analyze my team",
        system_prompt="You are an assistant",
        conversation_id="conversation_1"
    )

    assert "[Mock AI Response]" in result

    history = ai_service.conversation_history["conversation_1"]

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Analyze my team"
    assert history[1]["role"] == "assistant"


def test_conversation_history_is_separated(reset_ai_history):
    """Test that different conversation IDs have separate histories."""

    ai_service.get_ai_response(
        "First message",
        "System",
        "conversation_1"
    )

    ai_service.get_ai_response(
        "Second message",
        "System",
        "conversation_2"
    )

    assert "conversation_1" in ai_service.conversation_history
    assert "conversation_2" in ai_service.conversation_history

    assert len(ai_service.conversation_history["conversation_1"]) == 2
    assert len(ai_service.conversation_history["conversation_2"]) == 2

    assert (
        ai_service.conversation_history["conversation_1"][0]["content"]
        == "First message"
    )

    assert (
        ai_service.conversation_history["conversation_2"][0]["content"]
        == "Second message"
    )


def test_real_ai_response_uses_previous_history(
    reset_ai_history,
    monkeypatch
):
    """Test that previous messages are sent to the AI client."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)

            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text="AI response"
                    )
                ]
            )

    fake_client = SimpleNamespace(
        messages=FakeMessages()
    )

    fake_anthropic = SimpleNamespace(
        Anthropic=lambda api_key: fake_client
    )

    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    ai_service.conversation_history["conversation_1"] = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"}
    ]

    result = ai_service.get_ai_response(
        prompt="New question",
        system_prompt="System instructions",
        conversation_id="conversation_1"
    )

    assert result == "AI response"
    assert captured["system"] == "System instructions"
    assert len(captured["messages"]) == 3
    assert captured["messages"][0]["content"] == "Previous question"
    assert captured["messages"][2]["content"] == "New question"

    history = ai_service.conversation_history["conversation_1"]

    assert len(history) == 4
    assert history[-1]["content"] == "AI response"


def test_ai_failure_raises_runtime_error(
    reset_ai_history,
    monkeypatch
):
    """Test that AI client errors are wrapped in RuntimeError."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

    class FakeMessages:
        def create(self, **kwargs):
            raise ValueError("API failure")

    fake_client = SimpleNamespace(
        messages=FakeMessages()
    )

    fake_anthropic = SimpleNamespace(
        Anthropic=lambda api_key: fake_client
    )

    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    with pytest.raises(RuntimeError, match="Failed to generate"):
        ai_service.get_ai_response(
            "Question",
            "System",
            "conversation_1"
        )


def test_log_ai_request_creates_csv(reset_ai_history):
    """Test creating a CSV log with metadata only."""

    ai_service.log_ai_request(
        prompt="This is a private prompt",
        status_str="SUCCESS"
    )

    with open(
        ai_service.AI_LOGS_FILE,
        mode="r",
        newline="",
        encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["prompt_length"] == str(len("This is a private prompt"))
    assert rows[0]["status"] == "SUCCESS"

    with open(
        ai_service.AI_LOGS_FILE,
        mode="r",
        encoding="utf-8"
    ) as file:
        content = file.read()

    assert "This is a private prompt" not in content


def test_log_ai_request_does_not_duplicate_header(reset_ai_history):
    """Test that multiple log entries use only one CSV header."""

    ai_service.log_ai_request("First prompt", "SUCCESS")
    ai_service.log_ai_request("Second prompt", "ERROR")

    with open(
        ai_service.AI_LOGS_FILE,
        mode="r",
        encoding="utf-8"
    ) as file:
        lines = file.readlines()

    assert len(lines) == 3
    assert lines[0].startswith("timestamp,prompt_length,status")
    assert sum(line.startswith("timestamp,") for line in lines) == 1


def test_get_ai_logs_stats(reset_ai_history):
    """Test AI log statistics."""

    ai_service.log_ai_request("First", "SUCCESS")
    ai_service.log_ai_request("Second prompt", "SUCCESS")
    ai_service.log_ai_request("Third prompt here", "ERROR")

    result = ai_service.get_ai_logs_stats()

    assert result["total_requests"] == 3
    assert result["success_rate"] == pytest.approx(66.67)
    assert result["avg_prompt_length"] == pytest.approx(11.67)

def test_get_ai_logs_stats_without_file(reset_ai_history):
    """Test default statistics when no log file exists."""

    result = ai_service.get_ai_logs_stats()

    assert result == {
        "success_rate": 0.0,
        "avg_prompt_length": 0.0,
        "total_requests": 0
    }


def test_get_ai_logs_stats_with_empty_file(reset_ai_history):
    """Test default statistics when the log file is empty."""

    with open(ai_service.AI_LOGS_FILE, "w", encoding="utf-8"):
        pass

    result = ai_service.get_ai_logs_stats()

    assert result["total_requests"] == 0
    assert result["success_rate"] == 0.0
    assert result["avg_prompt_length"] == 0.0
