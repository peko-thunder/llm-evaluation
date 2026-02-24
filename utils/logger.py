import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from providers.base import LLMResponse


LOG_DIR = Path(__file__).parent.parent / "logs"


def _is_valid_json(text: str | None) -> bool:
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _response_to_dict(resp: LLMResponse, run_id: str, timestamp: str) -> dict:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "provider": resp.provider,
        "model": resp.model,
        "prompt": resp.prompt,
        "response": resp.response,
        "valid_json": _is_valid_json(resp.response),
        "tokens": {
            "input": resp.input_tokens,
            "output": resp.output_tokens,
            "thinking": resp.thinking_tokens,
            "total": (resp.input_tokens + resp.output_tokens)
            + (resp.thinking_tokens or 0),
        },
        "latency_ms": round(resp.latency_ms, 2),
        "error": resp.error,
        "raw_usage": resp.raw_usage,
    }


def save_log(responses: List[LLMResponse], run_id: str) -> Path:
    """
    Save all model responses for a single run to a JSON log file.

    File name format: logs/<run_id>.json
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    entries = [_response_to_dict(r, run_id, timestamp) for r in responses]

    log_path = LOG_DIR / f"{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "prompt": responses[0].prompt if responses else "",
                "results": entries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return log_path


def print_summary(responses: List[LLMResponse]) -> None:
    """Print a formatted comparison table to stdout."""
    sep = "-" * 90
    print(sep)
    print(
        f"{'Model':<45} {'In':>6} {'Out':>6} {'Think':>6} {'ms':>8}  Status"
    )
    print(sep)
    for r in responses:
        thinking = str(r.thinking_tokens) if r.thinking_tokens is not None else "-"
        status = "ERROR" if r.error else "OK"
        print(
            f"{r.model:<45} {r.input_tokens:>6} {r.output_tokens:>6} "
            f"{thinking:>6} {r.latency_ms:>8.1f}  {status}"
        )
    print(sep)

    print()
    for r in responses:
        if r.error:
            print(f"[{r.model}] ERROR: {r.error}")
            continue
        print(f"[{r.model}]")
        print(r.response.strip())
        print()
