
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    provider: str                          # "ollama", "openai", or "anthropic"
    model_id: str                          # actual model name, e.g. "qwen2.5:7b" or "gpt-4o"
    cost_per_input_token: float            # dollars per input token
    cost_per_output_token: float           # dollars per output token
    avg_latency: float                     # rough average response time in seconds
    quality_tier: str                      # "high", "medium", or "low"
    api_key_env_var: Optional[str] = None  # env var name holding the API key, None for Ollama


@dataclass
class Response:
    output_text: str
    input_tokens: int
    output_tokens: int
    latency: float
    cost: float
    model_id: str