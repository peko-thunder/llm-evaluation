import time
import os
from typing import Optional

from .base import BaseProvider, LLMResponse


class GoogleCloudProvider(BaseProvider):
    """
    Google Cloud Gemini models via Vertex AI SDK.

    Authenticates using Application Default Credentials (ADC) or a service
    account key file.  Set GOOGLE_APPLICATION_CREDENTIALS to the path of a
    service account JSON key file, or configure ADC with:
        gcloud auth application-default login

    Environment variables:
        GOOGLE_CLOUD_PROJECT:        GCP project ID (required)
        GOOGLE_CLOUD_LOCATION:       GCP region, e.g. us-central1 (default: us-central1)
        GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON key file (optional)

    Config options:
        model_id (str): Gemini model identifier
        enable_thinking (bool): Enable thinking mode (default: False)
        thinking_budget (int): Max thinking tokens (default: 8192)
        temperature (float): Sampling temperature
        max_output_tokens (int): Max output tokens
    """

    def __init__(self, model_id: str, config: Optional[dict] = None):
        import vertexai

        self.model_id = model_id
        self.config = config or {}

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")

        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        vertexai.init(project=project, location=location)

        self._generation_config_kwargs = {}
        if "temperature" in self.config:
            self._generation_config_kwargs["temperature"] = self.config["temperature"]
        if "max_output_tokens" in self.config:
            self._generation_config_kwargs["max_output_tokens"] = self.config["max_output_tokens"]

    def run(self, prompt: str) -> LLMResponse:
        from vertexai.generative_models import GenerativeModel, GenerationConfig, ThinkingConfig

        enable_thinking = self.config.get("enable_thinking", False)
        thinking_budget = self.config.get("thinking_budget", 8192)

        gen_config_kwargs = dict(self._generation_config_kwargs)
        if enable_thinking:
            gen_config_kwargs["thinking_config"] = ThinkingConfig(
                thinking_budget=thinking_budget
            )

        generation_config = GenerationConfig(**gen_config_kwargs) if gen_config_kwargs else None

        model = GenerativeModel(self.model_id)

        try:
            start = time.time()
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
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
