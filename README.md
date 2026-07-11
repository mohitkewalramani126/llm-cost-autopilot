# LLM Cost Autopilot

An intelligent routing layer that sits in front of multiple LLM providers, scores each request's complexity, and routes it to the cheapest model capable of handling it at acceptable quality — with an async verification loop to catch bad routing decisions.

Runs fully free using local models via [Ollama](https://ollama.com) — no API keys required. Optional support for paid providers (OpenAI, Anthropic) via environment variables, see `.env.example`.

## Status
Actively developed. Core routing, classification, and cost-tracking layers are being built incrementally.


## Stack
Python, FastAPI, scikit-learn, SQLite, Streamlit, Docker.