import json
import time
import os
from typing import Optional

from .base import BaseProvider, LLMResponse


class AWSBedrockProvider(BaseProvider):
    """
    AWS Bedrock provider for Claude and Amazon Nova models.

    Supports:
      - Anthropic Claude models (Messages API via Bedrock)
      - Amazon Nova models (Converse API)

    Environment variables:
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        (or configure via AWS CLI / IAM role)

    Config options:
        model_id (str): Bedrock model ARN or ID
        enable_thinking (bool): Enable extended thinking for Claude (default: False)
        thinking_budget (int): Max thinking tokens for Claude (default: 8000)
        temperature (float): Sampling temperature
        max_tokens (int): Max output tokens (default: 4096)
        region (str): AWS region override
    """

    # Model families that use the Anthropic Messages API on Bedrock
    ANTHROPIC_PREFIXES = ("anthropic.", "jp.anthropic.", "us.anthropic.", "eu.anthropic.", "ap.anthropic.")
    # Model families that use the Amazon Converse / Nova API
    AMAZON_PREFIXES = ("amazon.", "jp.amazon.", "us.amazon.", "eu.amazon.", "ap.amazon.")

    # Claude models that support native structured output via output_config.format
    NATIVE_STRUCTURED_OUTPUT_PATTERNS = (
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    )

    def __init__(self, model_id: str, config: Optional[dict] = None):
        import boto3

        self.model_id = model_id
        self.config = config or {}

        region = self.config.get("region") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )

    def _is_anthropic_model(self) -> bool:
        return any(self.model_id.startswith(p) for p in self.ANTHROPIC_PREFIXES)

    def _is_amazon_model(self) -> bool:
        return any(self.model_id.startswith(p) for p in self.AMAZON_PREFIXES)

    def _supports_native_structured_output(self) -> bool:
        return any(p in self.model_id for p in self.NATIVE_STRUCTURED_OUTPUT_PATTERNS)

    def _run_anthropic(self, prompt: str) -> LLMResponse:
        """Invoke a Claude model via the Anthropic Messages API on Bedrock."""
        enable_thinking = self.config.get("enable_thinking", False)
        thinking_budget = self.config.get("thinking_budget", 2000)
        max_tokens = self.config.get("max_tokens", 4096)
        response_format = self.config.get("response_format")
        response_schema = self.config.get("response_schema")

        request_body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        if enable_thinking:
            # Extended thinking requires temperature=1 and disables top_p/top_k
            request_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            request_body["temperature"] = 1
        else:
            if "temperature" in self.config:
                request_body["temperature"] = self.config["temperature"]

        if response_format == "json" and self._supports_native_structured_output():
            json_schema: dict = {"type": "object"}
            if response_schema:
                json_schema.update(response_schema)
            # Bedrock requires additionalProperties: false for output_config.format
            json_schema["additionalProperties"] = False
            request_body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": json_schema,
                }
            }
        elif response_format == "json":
            input_schema: dict = {"type": "object"}
            if response_schema:
                input_schema.update(response_schema)
            else:
                input_schema["additionalProperties"] = True
            request_body["tools"] = [{
                "name": "json_output",
                "description": "Output the response as structured JSON",
                "input_schema": input_schema,
            }]
            request_body["tool_choice"] = {"type": "tool", "name": "json_output"}

        try:
            start = time.time()
            raw = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            latency_ms = (time.time() - start) * 1000

            body = json.loads(raw["body"].read())
            usage = body.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Thinking tokens: sum of tokens in thinking blocks
            thinking_tokens: Optional[int] = None
            response_text_parts = []
            for block in body.get("content", []):
                if block.get("type") == "thinking":
                    thinking_tokens = (thinking_tokens or 0) + len(
                        block.get("thinking", "").split()
                    )
                elif block.get("type") == "text":
                    response_text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use" and block.get("name") == "json_output":
                    response_text_parts.append(json.dumps(block["input"], ensure_ascii=False))

            # If extended thinking is on, Bedrock may report cache_read/creation tokens too
            raw_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            }

            return LLMResponse(
                model=self.model_id,
                provider="aws_bedrock",
                prompt=prompt,
                response="\n".join(response_text_parts),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                latency_ms=latency_ms,
                raw_usage=raw_usage,
                response_format=response_format,
            )

        except Exception as e:
            return LLMResponse(
                model=self.model_id,
                provider="aws_bedrock",
                prompt=prompt,
                response="",
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=None,
                latency_ms=0.0,
                error=str(e),
                response_format=response_format,
            )

    def _run_amazon_converse(self, prompt: str) -> LLMResponse:
        """Invoke an Amazon Nova model via the Converse API."""
        max_tokens = self.config.get("max_tokens", 4096)
        response_format = self.config.get("response_format")
        response_schema = self.config.get("response_schema")

        inference_config: dict = {"maxTokens": max_tokens}
        if "temperature" in self.config:
            inference_config["temperature"] = self.config["temperature"]

        converse_kwargs: dict = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": inference_config,
        }
        if response_format == "json":
            json_schema: dict = {"type": "object"}
            if response_schema:
                json_schema.update(response_schema)
            converse_kwargs["toolConfig"] = {
                "tools": [{"toolSpec": {
                    "name": "json_output",
                    "description": "Output the response as structured JSON",
                    "inputSchema": {"json": json_schema},
                }}],
                "toolChoice": {"tool": {"name": "json_output"}},
            }

        try:
            start = time.time()
            response = self.client.converse(**converse_kwargs)
            latency_ms = (time.time() - start) * 1000

            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)

            # Nova2 may expose thinking token counts in future; capture if present
            thinking_tokens: Optional[int] = usage.get("thinkingTokens") or usage.get(
                "reasoningTokens"
            )

            output_message = response.get("output", {}).get("message", {})
            content_parts = []
            for block in output_message.get("content", []):
                if "text" in block:
                    content_parts.append(block["text"])
                elif "toolUse" in block and block["toolUse"].get("name") == "json_output":
                    content_parts.append(json.dumps(block["toolUse"]["input"], ensure_ascii=False))

            raw_usage = {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": usage.get("totalTokens"),
            }

            return LLMResponse(
                model=self.model_id,
                provider="aws_bedrock",
                prompt=prompt,
                response="\n".join(content_parts),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                latency_ms=latency_ms,
                raw_usage=raw_usage,
                response_format=response_format,
            )

        except Exception as e:
            return LLMResponse(
                model=self.model_id,
                provider="aws_bedrock",
                prompt=prompt,
                response="",
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=None,
                latency_ms=0.0,
                error=str(e),
                response_format=response_format,
            )

    def run(self, prompt: str) -> LLMResponse:
        if self._is_anthropic_model():
            return self._run_anthropic(prompt)
        elif self._is_amazon_model():
            return self._run_amazon_converse(prompt)
        else:
            # Fall back to Converse API for unknown model families
            return self._run_amazon_converse(prompt)
