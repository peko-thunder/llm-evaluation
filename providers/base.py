from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    model: str
    provider: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: Optional[int]
    latency_ms: float
    error: Optional[str] = None
    raw_usage: dict = field(default_factory=dict)


class BaseProvider(ABC):
    @abstractmethod
    def run(self, prompt: str) -> LLMResponse:
        pass
