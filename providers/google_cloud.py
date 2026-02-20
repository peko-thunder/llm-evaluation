import time
import os
from typing import Optional

from .base import BaseProvider, LLMResponse


class GoogleCloudProvider(BaseProvider):
    """
    Google Gemini models via google-genai SDK.

    Supports two authentication modes:
    1. API Key: Set GOOGLE_API_KEY environment variable (Google AI Studio)
    2. Vertex AI: Set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION)

    Environment variables:
        GOOGLE_API_KEY:              Google AI Studio API key (for API key auth)
        GOOGLE_CLOUD_PROJECT:        GCP project ID (for Vertex AI auth)
        GOOGLE_CLOUD_LOCATION:       GCP region, e.g. us-central1 (default: us-central1)
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON key file (optional)

    Config options:
        model_id (str): Gemini model identifier
        location (str): Vertex AI region override (e.g. us-central1). Overrides GOOGLE_CLOUD_LOCATION env var.
        enable_thinking (bool): Enable thinking mode (default: False)
        thinking_budget (int): Max thinking tokens (default: 8192)
        temperature (float): Sampling temperature
        max_output_tokens (int): Max output tokens
    """

    def __init__(self, model_id: str, config: Optional[dict] = None):
        from google import genai

        self.model_id = model_id
        self.config = config or {}

        api_key = os.environ.get("GOOGLE_API_KEY")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")

        if api_key:
            self._client = genai.Client(api_key=api_key)
        elif project:
            location = self.config.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            raise ValueError(
                "Either GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT environment variable must be set"
            )

    def run(self, prompt: str) -> LLMResponse:
        from google.genai import types

        enable_thinking = self.config.get("enable_thinking", False)
        thinking_budget = self.config.get("thinking_budget", 8192)

        config_kwargs = {}
        if "temperature" in self.config:
            config_kwargs["temperature"] = self.config["temperature"]
        if "max_output_tokens" in self.config:
            config_kwargs["max_output_tokens"] = self.config["max_output_tokens"]
        if enable_thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        generate_config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            start = time.time()
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=generate_config,
            )
            latency_ms = (time.time() - start) * 1000

            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            thinking_tokens = getattr(usage, "thoughts_token_count", None)

            raw_usage = {
                "prompt_token_count": input_tokens,
                "candidates_token_count": output_tokens,
                "thoughts_token_count": thinking_tokens,
                "total_token_count": getattr(usage, "total_token_count", None),
            }

            return LLMResponse(
                model=self.model_id,
                provider="google_cloud",
                prompt=prompt,
                response=response.text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                latency_ms=latency_ms,
                raw_usage=raw_usage,
            )

        except Exception as e:
            return LLMResponse(
                model=self.model_id,
                provider="google_cloud",
                prompt=prompt,
                response="",
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=None,
                latency_ms=0.0,
                error=str(e),
            )
