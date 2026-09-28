---
title: LangGraph Code Review Agent
emoji: 🧩
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.14.0"
app_file: app_gradio.py
pinned: false
---

# LangGraph Code Review Agent

A multi-agent PR reviewer built with **LangGraph**. A submitted code file is
reviewed **in parallel** by three specialist agents (security, style, logic),
then a tech-lead agent aggregates their findings into one structured review.
        START
       /  |  \
security style logic      <- run in parallel (fan-out)
       \  |  /
      aggregate            <- waits for all three (fan-in)
          |
         END

Fully free to run: Groq's API free tier + free hosting on Streamlit Community
Cloud or Hugging Face Spaces.

## What makes this deployment-grade (not just a demo)

- **True concurrency** — nodes use `async def` + `ainvoke`, so the three
  reviewer agents actually run concurrently via LangGraph's async engine,
  not just structurally-parallel-but-sequentially-executed.
- **Retries with backoff** — transient failures (rate limits, timeouts,
  5xx) are retried up to 3 times with exponential backoff. Config/auth
  errors fail fast instead of retrying pointlessly.
- **Graceful degradation** — if one reviewer agent fails after retries,
  the graph still completes and reports that agent as unavailable, rather
  than crashing the whole review.
- **Input validation** — size and filename checks run before any API call,
  so malformed input doesn't burn API quota.
- **Structured logging** — every node logs start/completion/failure.
- **Tests** — `tests/test_graph_agent.py` mocks the LLM entirely, so the
  suite runs with no API key and no network access (see CI below).
- **CI** — `.github/workflows/ci.yml` lints, tests, and builds the Docker
  image on every push to `main`.
- **Containerized** — `Dockerfile` gives a third deploy path beyond the
  two PaaS options below (e.g. for a VM, ECS, Cloud Run, etc.).

## Local setup

```bash
git clone <your-repo-url>
cd langgraph-code-reviewer
pip install -r requirements.txt

export GROQ_API_KEY="your-key-here"   # free key: https://console.groq.com/keys
streamlit run app.py
```

For development (tests + linting):
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
```

## Deploy — Docker (any container host, free tier available on most)

```bash
docker build -t langgraph-code-reviewer .
docker run -p 8501:8501 -e GROQ_API_KEY="your-key-here" langgraph-code-reviewer
```
Push the image to a free-tier container host (Render, Railway, Fly.io all
have free tiers) or run it on any VM/cluster.

## Deploy — Streamlit Community Cloud (free)

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io -> "New app".
3. Pick the repo, branch, and set **Main file path** to `app.py`.
4. In "Advanced settings" -> Secrets, add:
5. Deploy. You get a public `*.streamlit.app` URL.

## Deploy — Hugging Face Spaces (free)

HF Spaces' free tier no longer offers a plain Streamlit SDK (only Docker, which
requires a paid tier for this setup), so this repo ships a second UI —
`app_gradio.py` — built on the same `graph_agent.py` backend, for the free
Gradio SDK. `app.py` (Streamlit) is unchanged and still deploys to Streamlit
Cloud as above.

1. Create a new Space at https://huggingface.co/new-space.
2. SDK: **Gradio**. Hardware: **Free tier / CPU basic**. Visibility: Public (or Private).
3. Push this folder's contents to the Space repo (the YAML block at the top
   of this README is required by HF Spaces — keep it in place; `app_file` must
   point at `app_gradio.py`, not `app.py`).
```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main --force
```
   (`--force` is only needed the first time — a new Space starts with its own
   auto-generated commit that this overwrites.)
4. In the Space -> Settings -> "Variables and secrets", add `GROQ_API_KEY`.
5. The Space builds automatically and gives you a public
   `huggingface.co/spaces/<user>/<space>` URL.

Both deployments share the same `graph_agent.py` backend — only the UI layer
differs (`app.py` for Streamlit, `app_gradio.py` for Gradio/HF).

## Presenting this

- Use the "Show live graph structure" toggle in the sidebar to display the
  actual compiled LangGraph graph before running a demo — good talking point
  for "how does the parallelism work."
- Click **Load demo snippet** for a canned example with an obvious SQL
  injection, an `eval()` call, and a hardcoded key, so the security agent
  visibly catches real issues live.
- Mention: this uses LangGraph's fan-out/fan-in pattern (`StateGraph` with
  multiple edges from `START` and an `Annotated[list, operator.add]` reducer
  to merge parallel node outputs) — not a linear chain.

## Stack

- **LangGraph** — graph orchestration, parallel node execution, state reducers
- **Groq** (GPT-OSS 120B) — free-tier inference
- **Streamlit** — UI (Streamlit Community Cloud deployment)
- **Gradio** — UI (Hugging Face Spaces deployment)
- No paid services required anywhere in this stack.
