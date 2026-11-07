
import requests
from requests.exceptions import RequestException

import pandas as pd
import numpy as np


BASE_URL = "https://jsonplaceholder.typicode.com"

# Global in-memory cache
users_cache = None
tasks_cache = None


def fetch_json(path: str):
    """Fetch JSON data from the external API."""

    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()

    except (RequestException, ValueError) as error:
        print(f"Error fetching data from {url}: {error}")
        return None


def load_data():
    """Fetch, process and cache users and tasks data."""

    global users_cache, tasks_cache

    users_raw = fetch_json("users")
    tasks_raw = fetch_json("todos")

    if users_raw is None or tasks_raw is None:
        return False

    if not isinstance(users_raw, list) or not isinstance(tasks_raw, list):
        print("Error: External API returned an unexpected data format.")
        return False

    if not all(isinstance(user, dict) for user in users_raw):
        print("Error: Invalid user data received.")
        return False

    if not all(isinstance(task, dict) for task in tasks_raw):
        print("Error: Invalid task data received.")
        return False

    users_list = [
        {
            "user_id": user.get("id"),
            "name": user.get("name"),
            "email": user.get("email"),
            "city": (user.get("address") or {}).get("city", "Unknown")
        }
        for user in users_raw
    ]

    tasks_list = [
        {
            "id": task.get("id"),
            "user_id": task.get("userId"),
            "title": task.get("title"),
            "completed": task.get("completed") is True
        }
        for task in tasks_raw
    ]

    # Explicit columns ensure valid DataFrames even when API responses are empty.
    new_users_cache = pd.DataFrame(
        users_list,
        columns=["user_id", "name", "email", "city"]
    )

    new_tasks_cache = pd.DataFrame(
        tasks_list,
        columns=["id", "user_id", "title", "completed"]
    )

    users_cache = new_users_cache
    tasks_cache = new_tasks_cache

    return True


def get_cached_data():
    """Return cached data or load it when the cache is empty."""

    global users_cache, tasks_cache

    if users_cache is None or tasks_cache is None:
        if not load_data():
            return None, None

    return users_cache, tasks_cache


def get_merged_data():
    """Merge users and tasks using the user_id column."""

    users_df, tasks_df = get_cached_data()

    if users_df is None or tasks_df is None:
        return None

    return pd.merge(
        tasks_df,
        users_df,
        on="user_id",
        how="right"
    )


def calculate_users_performance():
    """Calculate task completion statistics for every user."""

    merged = get_merged_data()

    if merged is None:
        return None

    stats = merged.groupby(
        ["user_id", "name"],
        dropna=False
    ).agg(
        total_tasks=("id", "count"),
        completed_tasks=("completed", "sum")
    ).reset_index()

    stats["completion_percentage"] = np.where(
        stats["total_tasks"] > 0,
        (
            stats["completed_tasks"]
            / stats["total_tasks"]
            * 100
        ).round(2),
        0.0
    )

    stats = stats.sort_values(
        by="completion_percentage",
        ascending=False
    ).reset_index(drop=True)

    return stats


def get_low_performers_data():
    """Calculate average and standard deviation and find users below average."""

    stats = calculate_users_performance()

    if stats is None:
        return None, None, None

    if stats.empty:
        return [], 0.0, 0.0

    percentages = stats["completion_percentage"].to_numpy()

    mean_percentage = float(np.mean(percentages))
    std_percentage = float(np.std(percentages))

    low_df = stats[
        stats["completion_percentage"] < mean_percentage
    ]

    return (
        low_df.to_dict(orient="records"),
        mean_percentage,
        std_percentage
    )


def get_cities_pivot_table():
    """Create a pivot table of cities and task completion status."""

    merged = get_merged_data()

    if merged is None:
        return None

    pivot = pd.pivot_table(
        merged,
        index="city",
        columns="completed",
        values="id",
        aggfunc="count",
        fill_value=0
    )

    return pivot.to_dict()
