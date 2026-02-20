#!/usr/bin/env python3
"""
LLM Evaluation CLI

Compare responses from multiple LLM providers (Google Cloud / AWS Bedrock)
for a single prompt, and save the results as a structured JSON log.

Usage examples:
  # Run all models defined in config.yaml
  python main.py "Explain quantum entanglement in simple terms."

  # Provide the prompt via stdin
  echo "Write a haiku about AI." | python main.py

  # Override which models to run (comma-separated keys from config.yaml)
  python main.py --models gemini-2-5-flash,claude-3-haiku "What is 2+2?"

  # Specify a custom config file
  python main.py --config custom_config.yaml "Hello"

  # Read prompt from a file
  python main.py --prompt-file prompt.txt

  # Disable console output (only save log)
  python main.py --quiet "Tell me a joke."
"""

import argparse
import os
import sys
import uuid
import concurrent.futures
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv

from providers import GoogleCloudProvider, AWSBedrockProvider
from providers.base import BaseProvider, LLMResponse
from utils.logger import save_log, print_summary


DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_provider(key: str, model_cfg: dict) -> BaseProvider:
    provider_type = model_cfg.get("provider", "").lower()
    model_id = model_cfg["model_id"]
    extra = model_cfg.get("options", {})

    if provider_type == "google_cloud":
        return GoogleCloudProvider(model_id=model_id, config=extra)
    elif provider_type == "aws_bedrock":
        return AWSBedrockProvider(model_id=model_id, config=extra)
    else:
        raise ValueError(
            f"Unknown provider '{provider_type}' for model key '{key}'. "
            "Supported providers: google_cloud, aws_bedrock"
        )


def run_model(key: str, provider: BaseProvider, prompt: str) -> LLMResponse:
    return provider.run(prompt)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compare LLM responses across multiple cloud providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text. Omit to read from --prompt-file or stdin.",
    )
    parser.add_argument(
        "--models",
        metavar="KEY1,KEY2,...",
        help="Comma-separated list of model keys from config.yaml to run. "
        "Defaults to all enabled models.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to config YAML file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        type=Path,
        help="Read prompt from this file instead of positional argument.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run models in parallel (default: True).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run models sequentially instead of in parallel.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output; only write the log file.",
    )
    parser.add_argument(
        "--log-dir",
        metavar="PATH",
        type=Path,
        help="Directory to save log files (default: ./logs).",
    )
    args = parser.parse_args()

    # --- Determine prompt ---
    prompt: Optional[str] = None
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if not prompt:
        parser.error(
            "A prompt is required. Pass it as a positional argument, "
            "--prompt-file, or via stdin."
        )

    # --- Load config ---
    if not args.config.exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    models_cfg: dict = config.get("models", {})

    # Override log dir if specified
    if args.log_dir:
        import utils.logger as logger_module
        logger_module.LOG_DIR = args.log_dir

    # --- Select models ---
    if args.models:
        requested_keys = [k.strip() for k in args.models.split(",")]
        unknown = [k for k in requested_keys if k not in models_cfg]
        if unknown:
            print(
                f"Unknown model key(s): {', '.join(unknown)}. "
                f"Available: {', '.join(models_cfg.keys())}",
                file=sys.stderr,
            )
            return 1
        selected = {k: models_cfg[k] for k in requested_keys}
    else:
        selected = {
            k: v for k, v in models_cfg.items() if v.get("enabled", True)
        }

    if not selected:
        print("No models selected. Check your config.yaml.", file=sys.stderr)
        return 1

    # --- Build providers ---
    providers: dict[str, BaseProvider] = {}
    for key, cfg in selected.items():
        try:
            providers[key] = build_provider(key, cfg)
        except Exception as e:
            print(f"Failed to initialize model '{key}': {e}", file=sys.stderr)
            return 1

    if not args.quiet:
        print(f"Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        print(f"Running {len(providers)} model(s)...\n")

    # --- Execute ---
    run_id = str(uuid.uuid4())
    responses: List[LLMResponse] = []

    use_parallel = not args.sequential

    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as executor:
            futures = {
                executor.submit(run_model, key, prov, prompt): key
                for key, prov in providers.items()
            }
            for future in concurrent.futures.as_completed(futures):
                responses.append(future.result())
    else:
        for key, prov in providers.items():
            responses.append(run_model(key, prov, prompt))

    # Sort by model name for consistent output
    responses.sort(key=lambda r: r.model)

    # --- Output ---
    if not args.quiet:
        print_summary(responses)

    log_path = save_log(responses, run_id)
    if not args.quiet:
        print(f"Log saved: {log_path}")

    # Return non-zero if any model errored
    return 1 if any(r.error for r in responses) else 0


if __name__ == "__main__":
    sys.exit(main())
