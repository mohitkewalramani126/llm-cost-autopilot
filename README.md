# LLM Cost Autopilot

An intelligent routing layer that sits in front of multiple LLM providers, scores each request's complexity, and routes it to the cheapest model capable of handling it at acceptable quality — with an async verification loop to catch bad routing decisions.

Runs fully free using local models via [Ollama](https://ollama.com) — no API keys required. Optional support for paid providers (OpenAI, Anthropic) via environment variables, see `.env.example`.

## Status

Actively developed. Core routing, classification, and cost-tracking layers are being built incrementally.

## Stack

Python, FastAPI, scikit-learn, SQLite, Streamlit, Docker.

## Configuration

The app runs entirely free on local Ollama models with zero setup. To also enable paid providers:

1. Copy `.env.example` to `.env`.
2. Add your OpenAI and/or Anthropic API key(s).
3. Restart the app.

If only one paid provider has a key set, it's used for every request. If **both** are set, the router compares their price for each tier independently and routes to whichever is cheaper for that specific tier — so simple/moderate/complex requests can genuinely end up on different providers depending on real pricing, not a fixed preference order.

Model IDs and pricing for each provider live in `config/models.yaml` and `config/routing.yaml`. Provider model names change over time — if you see a "model not found" error from OpenAI or Anthropic, check their current model list and update those two files.