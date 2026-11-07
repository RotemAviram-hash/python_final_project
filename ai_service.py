import os
import csv
from datetime import datetime

import pandas as pd


AI_LOGS_FILE = "ai_logs.csv"

# Conversation history stored in memory
conversation_history = {}


def log_ai_request(prompt: str, status_str: str):
    """Save AI request metadata without storing the prompt content."""

    prompt_length = len(prompt) if prompt else 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        file_exists = os.path.exists(AI_LOGS_FILE)

        with open(
            AI_LOGS_FILE,
            mode="a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            if not file_exists or os.path.getsize(AI_LOGS_FILE) == 0:
                writer.writerow([
                    "timestamp",
                    "prompt_length",
                    "status"
                ])

            writer.writerow([
                current_time,
                prompt_length,
                status_str
            ])

    except OSError as error:
        print(f"Error writing AI logs: {error}")


def get_ai_response(
    prompt: str,
    system_prompt: str,
    conversation_id: str
) -> str:
    """Generate an AI response and maintain conversation history."""

    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    history = conversation_history[conversation_id]

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Use Mock response when no valid API key is configured
    if not api_key or api_key == "your_anthropic_api_key_here":
        response_text = (
            "[Mock AI Response] "
            "Your request was received and the analysis context was applied."
        )

        history.append({
            "role": "user",
            "content": prompt
        })

        history.append({
            "role": "assistant",
            "content": response_text
        })

        return response_text

    try:
        # Import Anthropic only when a real API request is needed
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        messages = history + [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            system=system_prompt,
            messages=messages
        )

        response_text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

        if not response_text:
            raise RuntimeError("The AI returned an empty response.")

        history.append({
            "role": "user",
            "content": prompt
        })

        history.append({
            "role": "assistant",
            "content": response_text
        })

        return response_text

    except Exception as error:
        raise RuntimeError("Failed to generate an AI response.") from error


def ask_claude_ai(
    prompt: str,
    system_prompt: str,
    conversation_id: str = "default"
) -> str:
    """Compatibility wrapper for the AI response function."""

    return get_ai_response(
        prompt=prompt,
        system_prompt=system_prompt,
        conversation_id=conversation_id
    )


def get_ai_logs_stats():
    """Calculate AI request success rate and average prompt length."""

    empty_stats = {
        "success_rate": 0.0,
        "avg_prompt_length": 0.0,
        "total_requests": 0
    }

    if not os.path.exists(AI_LOGS_FILE):
        return empty_stats

    try:
        df = pd.read_csv(AI_LOGS_FILE)

        required_columns = {
            "timestamp",
            "prompt_length",
            "status"
        }

        if not required_columns.issubset(df.columns):
            print("Error: AI log file has an invalid structure.")
            return empty_stats

        if df.empty:
            return empty_stats

        df["prompt_length"] = pd.to_numeric(
            df["prompt_length"],
            errors="coerce"
        )

        df = df.dropna(subset=["prompt_length"])

        if df.empty:
            return empty_stats

        total_requests = len(df)

        success_count = len(
            df[df["status"] == "SUCCESS"]
        )

        success_rate = round(
            (success_count / total_requests) * 100,
            2
        )

        avg_prompt_length = round(
            float(df["prompt_length"].mean()),
            2
        )

        return {
            "success_rate": success_rate,
            "avg_prompt_length": avg_prompt_length,
            "total_requests": total_requests
        }

    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Error reading AI logs: {error}")
        return empty_stats