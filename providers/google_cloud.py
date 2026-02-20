import time
import os
from typing import Optional

from .base import BaseProvider, LLMResponse


class GoogleCloudProvider(BaseProvider):
    """
    Google Cloud Gemini models via Google Generative AI SDK.

    Supports token counting including thinking tokens for models
    that support extended thinking (e.g. gemini-2.5-flash).

    Environment variables:
        GOOGLE_API_KEY: Google AI Studio API key

    Config options:
        model_id (str): Gemini model identifier
        enable_thinking (bool): Enable thinking mode (default: False)
        thinking_budget (int): Max thinking tokens (default: 8192)
        temperature (float): Sampling temperature
        max_output_tokens (int): Max output tokens
    """

    def __init__(self, model_id: str, config: Optional[dict] = None):
        import google.generativeai as genai

        self.model_id = model_id
        self.config = config or {}

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        genai.configure(api_key=api_key)

        generation_config = {}
        if "temperature" in self.config:
            generation_config["temperature"] = self.config["temperature"]
        if "max_output_tokens" in self.config:
            generation_config["max_output_tokens"] = self.config["max_output_tokens"]

        self._genai = genai
        self._generation_config = generation_config

    def run(self, prompt: str) -> LLMResponse:
        enable_thinking = self.config.get("enable_thinking", False)
        thinking_budget = self.config.get("thinking_budget", 8192)

        generation_config = dict(self._generation_config)

        if enable_thinking:
            generation_config["thinking_config"] = self._genai.types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        model = self._genai.GenerativeModel(
            self.model_id,
            generation_config=generation_config,
        )

        try:
            start = time.time()
            response = model.generate_content(prompt)
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
