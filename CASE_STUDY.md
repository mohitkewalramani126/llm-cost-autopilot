# Triage: Cutting LLM API Costs Through Complexity-Aware Routing

**Repo:** [github.com/mohitkewalramani126/llm-cost-autopilot](https://github.com/mohitkewalramani126/llm-cost-autopilot)

## The problem

Most applications that call LLM APIs send every request to a single model, usually whichever one produces the best output, because that's the simplest thing to build. But a huge share of real traffic doesn't need a frontier model. Asking for the capital of a country, converting a string to title case, or drafting a two-line follow-up email doesn't require the same model as debugging a race condition in a distributed system. Paying frontier-model prices for all of it is the default, not a deliberate choice.

Triage is a routing layer that sits in front of multiple LLM providers, scores each incoming request's complexity, and sends it to the cheapest model actually capable of handling it well, then checks its own work asynchronously so a bad routing decision doesn't silently become a bad user experience.

## Architecture

A request moves through five stages:

1. **Classification.** A scikit-learn classifier trained on a hand-labeled dataset of 501 prompts scores each incoming request as simple, moderate, or complex based on extracted features (length, task-type keywords, constraint count, reasoning-depth signals). It holds 94% accuracy on a held-out test set.
2. **Routing.** Given a tier, the router picks a provider. If no paid API key is configured, everything runs on local Ollama models for free. If exactly one paid provider (OpenAI or Anthropic) has a key set, that provider handles every tier. If both are configured, the router compares real per-token pricing for that specific tier and picks whichever is cheaper -- so a single deployment can genuinely split traffic across providers if that's what the numbers favor, rather than committing to one.
3. **Generation.** The prompt is sent to the resolved model through a unified client that normalizes OpenAI, Anthropic, and Ollama responses into one schema (output text, token counts, latency, cost).
4. **Asynchronous verification.** A sample of responses (15% of simple/moderate traffic; complex-tier requests already sit on the best model, so they're skipped) is re-scored by an LLM-as-judge on a 1-5 scale in a background thread. If the score falls below threshold, the request is automatically re-run on the top-tier model and the corrected response is what the user actually sees -- the cheap-tier miss is invisible to them, but it's logged.
5. **Logging and the flywheel.** Every request is logged with a prompt *hash* (never the raw text) for privacy, alongside its tier, provider, model, cost, latency, and verification outcome. Escalated requests are the exception: their raw prompt text is stored separately, because the retraining flywheel needs actual text to turn a routing miss into a new labeled training example. A separate script pulls these, appends them to the training set with the corrected label, and retrains the classifier.

## What's built on top of it

**A cost and quality dashboard** (Streamlit) reads directly from the request log and shows routing distribution, cumulative actual cost against an always-top-tier baseline, escalation rates by tier, judge score distributions, and a full cost breakdown by provider → tier → model.

**A chat interface** lets you talk to the router directly, with streaming responses, multi-turn context, and document upload (PDF, Word, images) with OCR text extraction. Long documents are summarized through a chunked map-reduce pipeline rather than being truncated, since a flat concatenation of many chunk summaries can itself exceed a model's context window.

**A REST API** (FastAPI) exposes `POST /v1/completions` (routes a prompt and reports which model handled it and why), `GET /v1/models`, `GET /v1/stats`, and `PUT /v1/routing-config` for runtime routing changes -- containerized with Docker so it runs the same way on any machine.

## Results

The numbers below come from 1,539 real requests through the live pipeline: the original 501-prompt labeled dataset, plus 1,000 entirely new prompts generated across six domain categories (coding/technical, business writing, data analysis, general knowledge, customer support/creative, and miscellaneous) that the classifier had never seen during training or validation -- a genuine test of whether the routing decisions hold up on unseen traffic, not just prompts it was tuned on.

![Dashboard overview](docs/screenshots/overview.png)

Across the full run: **36.2% lower cost** than an always-use-the-best-model baseline ($0.0949 actual vs. $0.1488 baseline), with traffic splitting roughly 49% simple / 23% moderate / 28% complex.

![Routing and cost charts](docs/screenshots/overview_charts.png)

The quality safety net is doing real work, not just passing everything through: 73% of *verified* simple-tier responses and 3% of verified moderate-tier responses were escalated to a stronger model after failing the judge's threshold, catching cheap-tier misses before they reached the end result.

![Quality and escalation detail](docs/screenshots/quality.png)

The average judge score across sampled responses was 3.17/5, with escalation automatically correcting the lower end of that distribution rather than leaving it as-is.

![Raw request log](docs/screenshots/raw_log.png)

![Chat interface](docs/screenshots/chat.png)

## Engineering decisions worth calling out

**Verification runs as a background thread inside the request-handling process, not a separate worker service.** A queue-backed worker is the more "correct" production pattern, but it only pays off once a single process becomes an actual bottleneck or crash-durability for in-flight checks genuinely matters. At this traffic volume, the added complexity wasn't worth it -- a deliberate trade-off, not a shortcut, and one a production deployment at higher volume would revisit.

**The request log stores a prompt hash, not the raw prompt, except for the small number of escalated requests that feed the retraining flywheel.** This was a privacy-by-default choice: full-traffic analytics don't need raw text, and the flywheel only needs it for the cases that actually become training data.

**Routing compares real, verified pricing, not assumptions.** Every model ID and price in the routing config was checked against each provider's live pricing page rather than estimated, including catching a subtle case where a "cheaper-looking" model (by input price alone) was actually more expensive overall once output pricing was factored in -- the router ranks by combined price for exactly this reason.

## Stack

Python, scikit-learn, FastAPI, SQLite, Streamlit, Docker. OpenAI and Anthropic as optional paid providers; Ollama for a fully free, zero-API-key default.
