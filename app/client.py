import os
import time

import requests

from app.models import ModelConfig, Response

OLLAMA_URL = "http://localhost:11434/api/generate"


class MissingAPIKeyError(Exception):
    """Raised when a paid provider is requested but its API key isn't set."""


def _send_ollama(prompt: str, model_config: ModelConfig) -> Response:
    start = time.perf_counter()
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model_config.model_id, "prompt": prompt, "stream": False},
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.perf_counter() - start

    input_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)
    cost = (
        input_tokens * model_config.cost_per_input_token
        + output_tokens * model_config.cost_per_output_token
    )

    return Response(
        output_text=data.get("response", ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        cost=cost,
        model_id=model_config.model_id,
    )


def _require_api_key(model_config: ModelConfig) -> str:
    key = os.getenv(model_config.api_key_env_var or "")
    if not key:
        raise MissingAPIKeyError(
            f"{model_config.model_id} needs {model_config.api_key_env_var}, "
            "which is not set. Skipping this model."
        )
    return key


def _send_openai(prompt: str, model_config: ModelConfig) -> Response:
    key = _require_api_key(model_config)
    start = time.perf_counter()
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model_config.model_id,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.perf_counter() - start

    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = (
        input_tokens * model_config.cost_per_input_token
        + output_tokens * model_config.cost_per_output_token
    )

    return Response(
        output_text=data["choices"][0]["message"]["content"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        cost=cost,
        model_id=model_config.model_id,
    )


def _send_anthropic(prompt: str, model_config: ModelConfig) -> Response:
    key = _require_api_key(model_config)
    start = time.perf_counter()
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": model_config.model_id,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.perf_counter() - start

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (
        input_tokens * model_config.cost_per_input_token
        + output_tokens * model_config.cost_per_output_token
    )

    return Response(
        output_text=data["content"][0]["text"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        cost=cost,
        model_id=model_config.model_id,
    )


def send_request(prompt: str, model_config: ModelConfig) -> Response:
    """Send a prompt to whichever provider the model_config points to,
    and always get back a standardized Response — this is the abstraction
    layer the rest of the app is built on."""
    if model_config.provider == "ollama":
        return _send_ollama(prompt, model_config)
    elif model_config.provider == "openai":
        return _send_openai(prompt, model_config)
    elif model_config.provider == "anthropic":
        return _send_anthropic(prompt, model_config)
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")