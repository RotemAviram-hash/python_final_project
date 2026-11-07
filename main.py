import os
import json
import time
from datetime import date
from typing import Optional

import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    status,
    Depends,
    Header,
    Query,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

import data_service
import ai_service


# ==========================================
# Configuration
# ==========================================

load_dotenv()

APP_API_KEY = os.environ.get("APP_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not APP_API_KEY:
    print("WARNING: APP_API_KEY is missing. Protected endpoints will be unavailable.")

if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
    print("INFO: Anthropic API key is missing. AI Mock mode is enabled.")


app = FastAPI(
    title="Team Performance Analytics API",
    version="1.0.0"
)

TASKS_FILE = "tasks.json"


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Middleware
# ==========================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    response.headers["X-Process-Time"] = str(process_time)

    print(
        f"Method: {request.method} | "
        f"Path: {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Time: {process_time:.4f}s"
    )

    return response


# ==========================================
# API Key Security
# ==========================================

def verify_api_key(
    x_api_key: Optional[str] = Header(None)
):
    """Verify the API key supplied in the request header."""

    if not APP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key authentication is not configured."
        )

    if not x_api_key or x_api_key != APP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header."
        )

    return x_api_key


# ==========================================
# Pagination
# ==========================================

def pagination_params(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """Reusable pagination dependency."""

    skip = (page - 1) * page_size

    return {
        "skip": skip,
        "limit": page_size,
        "page": page,
        "page_size": page_size
    }


def paginate_records(records, pagination):
    """Return a paginated section of a list."""

    start = pagination["skip"]
    end = start + pagination["limit"]

    return {
        "page": pagination["page"],
        "page_size": pagination["page_size"],
        "total_items": len(records),
        "data": records[start:end]
    }


# ==========================================
# Custom Exceptions
# ==========================================

class TaskNotFoundError(Exception):

    def __init__(self, task_id: int):
        self.task_id = task_id
        super().__init__(f"Task with ID {task_id} not found.")


@app.exception_handler(TaskNotFoundError)
async def task_not_found_handler(
    request: Request,
    exc: TaskNotFoundError
):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)}
    )


# ==========================================
# Pydantic Models
# ==========================================

class TaskIn(BaseModel):
    title: str
    user_id: int
    due_date: date
    priority: int = Field(..., ge=1, le=3)
    done: bool = False
    notes: Optional[str] = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "Title cannot be empty or whitespace only."
            )

        return value.strip()


class TaskOut(BaseModel):
    id: int
    title: str
    user_id: int
    due_date: date
    priority: int
    done: bool


class OverdueTaskOut(TaskOut):
    days_late: int


class AIRequest(BaseModel):
    conversation_id: str
    prompt: str

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Conversation ID cannot be empty.")

        return value.strip()


# ==========================================
# Tasks JSON File Management
# ==========================================

def load_tasks():
    """Load tasks from the local JSON file."""

    if not os.path.exists(TASKS_FILE):
        return []

    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as file:
            tasks = json.load(file)

        if not isinstance(tasks, list):
            raise ValueError("Tasks file must contain a JSON list.")

        return tasks

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The tasks.json file contains invalid JSON."
        ) from error

    except OSError as error:
        raise RuntimeError(
            "Unable to read the tasks.json file."
        ) from error


def save_tasks(tasks):
    """Save tasks using an atomic file replacement."""

    temp_file = f"{TASKS_FILE}.tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(
                tasks,
                file,
                ensure_ascii=False,
                indent=4
            )

        os.replace(temp_file, TASKS_FILE)

    except OSError as error:
        if os.path.exists(temp_file):
            os.remove(temp_file)

        raise RuntimeError(
            "Unable to save tasks to the JSON file."
        ) from error


# ==========================================
# User Validation
# ==========================================

def validate_user_exists(user_id: int):
    """Check that the user exists in the external users dataset."""

    users_df, _ = data_service.get_cached_data()

    if users_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External user data source unavailable."
        )

    if user_id not in users_df["user_id"].values:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with ID {user_id} does not exist."
        )


# ==========================================
# Root
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Team Performance API is running successfully!"
    }


# ==========================================
# Statistics Endpoints
# ==========================================

@app.get("/stats/users")
def get_users_stats(
    pagination: dict = Depends(pagination_params)
):
    """Return user performance statistics with pagination."""

    stats_df = data_service.calculate_users_performance()

    if stats_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External data source unavailable."
        )

    records = stats_df.to_dict(orient="records")

    return paginate_records(records, pagination)


@app.get("/stats/summary")
def get_summary_stats():
    """Return general team statistics."""

    users_df, tasks_df = data_service.get_cached_data()

    if users_df is None or tasks_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External data source unavailable."
        )

    cities_pivot = data_service.get_cities_pivot_table()

    if cities_pivot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to calculate city statistics."
        )

    return {
        "total_users": len(users_df),
        "total_tasks": len(tasks_df),
        "completed_tasks": int(tasks_df["completed"].sum()),
        "cities_pivot": cities_pivot
    }


@app.get("/stats/low-performers")
def get_low_performers(
    pagination: dict = Depends(pagination_params)
):
    """Return users whose completion rate is below the team average."""

    low_list, mean_percentage, std_percentage = (
        data_service.get_low_performers_data()
    )

    if low_list is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External data source unavailable."
        )

    paginated_result = paginate_records(
        low_list,
        pagination
    )

    return {
        "mean_percentage": mean_percentage,
        "std_percentage": std_percentage,
        "low_performers": paginated_result["data"],
        "page": paginated_result["page"],
        "page_size": paginated_result["page_size"],
        "total_items": paginated_result["total_items"]
    }


@app.get("/stats/cities")
def get_cities_stats():
    """Return a pivot table of cities and task completion status."""

    cities_data = data_service.get_cities_pivot_table()

    if cities_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="External data source unavailable."
        )

    return cities_data


# ==========================================
# Tasks Endpoints
# ==========================================

@app.get("/tasks")
def get_tasks(
    pagination: dict = Depends(pagination_params)
):
    """Return all locally saved tasks with pagination."""

    tasks = load_tasks()

    return paginate_records(tasks, pagination)


@app.get("/tasks/overdue", response_model=list[OverdueTaskOut])
def get_overdue_tasks(
    pagination: dict = Depends(pagination_params)
):
    """Return unfinished tasks whose due date has passed."""

    tasks = load_tasks()

    if not tasks:
        return []

    tasks_df = pd.DataFrame(tasks)

    if "due_date" not in tasks_df.columns:
        return []

    tasks_df["due_date"] = pd.to_datetime(
        tasks_df["due_date"],
        errors="coerce"
    )

    today = pd.Timestamp(date.today())

    overdue_df = tasks_df[
        (tasks_df["done"] == False)
        & (tasks_df["due_date"] < today)
    ].copy()

    overdue_df["days_late"] = (
        today - overdue_df["due_date"]
    ).dt.days

    overdue_df["due_date"] = (
        overdue_df["due_date"].dt.strftime("%Y-%m-%d")
    )

    overdue_df["days_late"] = (
        overdue_df["days_late"].astype(int)
    )

    records = overdue_df.to_dict(orient="records")

    start = pagination["skip"]
    end = start + pagination["limit"]

    return records[start:end]


@app.get("/tasks/{task_id}", response_model=TaskOut)
def get_task_by_id(task_id: int):
    """Return a single task by its ID."""

    tasks = load_tasks()

    for task in tasks:
        if task.get("id") == task_id:
            return task

    raise TaskNotFoundError(task_id)


@app.post(
    "/tasks",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskOut
)
def create_task(
    task: TaskIn,
    x_api_key: str = Depends(verify_api_key)
):
    """Create a new task after validating the assigned user."""

    validate_user_exists(task.user_id)

    tasks = load_tasks()

    new_id = max(
        [item.get("id", 0) for item in tasks],
        default=0
    ) + 1

    new_task = task.model_dump(mode="json")
    new_task["id"] = new_id

    tasks.append(new_task)

    save_tasks(tasks)

    return new_task


@app.put(
    "/tasks/{task_id}",
    response_model=TaskOut
)
def update_task(
    task_id: int,
    updated_task: TaskIn,
    x_api_key: str = Depends(verify_api_key)
):
    """Update an existing task."""

    tasks = load_tasks()

    task_index = next(
        (
            index
            for index, item in enumerate(tasks)
            if item.get("id") == task_id
        ),
        None
    )

    if task_index is None:
        raise TaskNotFoundError(task_id)

    validate_user_exists(updated_task.user_id)

    updated_data = updated_task.model_dump(mode="json")
    updated_data["id"] = task_id

    tasks[task_index] = updated_data

    save_tasks(tasks)

    return updated_data


@app.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
def delete_task(
    task_id: int,
    x_api_key: str = Depends(verify_api_key)
):
    """Delete an existing task."""

    tasks = load_tasks()

    new_tasks = [
        task
        for task in tasks
        if task.get("id") != task_id
    ]

    if len(new_tasks) == len(tasks):
        raise TaskNotFoundError(task_id)

    save_tasks(new_tasks)

    return None


# ==========================================
# AI Endpoints
# ==========================================

@app.post("/ai/ask")
def ask_ai(req: AIRequest):
    """Send a prompt to the AI assistant with dynamic team context."""

    status_str = "ERROR"

    try:
        if not req.prompt or not req.prompt.strip() or len(req.prompt) > 500:
            status_str = "BAD_REQUEST"

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt must be between 1 and 500 characters."
            )

        stats = data_service.calculate_users_performance()

        if stats is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External data source unavailable."
            )

        top_performers = (
            stats.head(3).to_dict(orient="records")
            if not stats.empty
            else []
        )

        mean_percentage = (
            float(stats["completion_percentage"].mean())
            if not stats.empty
            else 0.0
        )

        merged = data_service.get_merged_data()

        strongest_city = "Unknown"

        if merged is not None and not merged.empty:
            city_performance = (
                merged.groupby("city")["completed"]
                .mean()
                .sort_values(ascending=False)
            )

            if not city_performance.empty:
                strongest_city = str(city_performance.index[0])

        system_prompt = (
            "You are a team performance assistant. "
            f"Top 3 performers: {top_performers}. "
            f"Average completion rate: {mean_percentage:.2f}%. "
            f"Strongest city by task completion rate: {strongest_city}. "
            "Use this information when answering questions about team performance."
        )

        reply = ai_service.ask_claude_ai(
            req.prompt,
            system_prompt,
            req.conversation_id
        )

        status_str = "SUCCESS"

        return {
            "conversation_id": req.conversation_id,
            "response": reply
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the AI request."
        )

    finally:
        ai_service.log_ai_request(
            req.prompt,
            status_str
        )


@app.get("/ai/logs/stats")
def ai_logs_stats():
    """Return statistics about AI requests."""

    return ai_service.get_ai_logs_stats()