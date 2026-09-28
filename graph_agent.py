r"""
LangGraph Code Review Agent — deployment-grade version
--------------------------------------------------------
Fan-out / fan-in pattern, now with:
  - true async concurrency (ainvoke, not thread-pooled sync calls)
  - manual retry with exponential backoff on transient API failures
  - per-call timeout (set on the client) so a hung request can't stall the graph
  - input validation (size limits) before any API spend
  - structured logging for observability
  - graceful degradation: one agent failing doesn't kill the whole review

Graph shape:

            START
           /  |  \
    security style logic     <- awaited concurrently via LangGraph's async engine
           \  |  /
          aggregate
              |
             END
"""

import asyncio
import logging
import operator
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("code_review_agent")


# ---------- Config ----------

MAX_CODE_CHARS = 20_000          # guard against runaway token spend
LLM_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
LLM_MODEL = "openai/gpt-oss-120b"


class ConfigError(Exception):
    pass


class InputValidationError(Exception):
    pass


def validate_code_input(code: str, filename: str) -> None:
    if not code or not code.strip():
        raise InputValidationError("Code input is empty.")
    if len(code) > MAX_CODE_CHARS:
        raise InputValidationError(
            f"Code input is {len(code)} chars, exceeds limit of {MAX_CODE_CHARS}."
        )
    if not filename or "/" in filename or "\\" in filename:
        raise InputValidationError("Filename must be a plain name, no path separators.")


# ---------- State ----------

class Finding(TypedDict):
    category: str
    content: str


class ReviewState(TypedDict):
    code: str
    filename: str
    findings: Annotated[list[Finding], operator.add]
    final_review: str


# ---------- LLM ----------

def get_llm(temperature: float = 0.2) -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ConfigError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys"
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
        api_key=api_key,
        timeout=LLM_TIMEOUT_SECONDS,
    )


REVIEWER_PROMPTS = {
    "security": (
        "You are a senior application security engineer. Review the given code "
        "ONLY for security issues: injection risks, unsafe deserialization, secrets "
        "in code, unsafe eval/exec, missing input validation, insecure defaults. "
        "Be concise. List concrete issues with line references if possible. "
        "If there are no security issues, say so explicitly in one line."
    ),
    "style": (
        "You are a senior engineer doing a style/readability review. Review the given "
        "code ONLY for naming, structure, docstrings/comments, formatting, and adherence "
        "to idiomatic conventions for its language. Be concise and concrete. "
        "If the style is fine, say so explicitly in one line."
    ),
    "logic": (
        "You are a senior engineer doing a correctness review. Review the given code "
        "ONLY for logic bugs, edge cases, off-by-one errors, incorrect assumptions, "
        "unhandled exceptions, and potential runtime errors. Be concise and concrete. "
        "If you find no logic issues, say so explicitly in one line."
    ),
}


def _is_retryable(exc: BaseException) -> bool:
    """Rate limits / transient network / server errors are retried. Config or
    auth errors are not — retrying those just burns time on a call that will
    never succeed."""
    msg = str(exc).lower()
    return any(term in msg for term in ("rate limit", "timeout", "timed out", "503", "502", "500", "connection"))


async def _call_with_retry(llm: ChatGroq, messages: list, label: str) -> str | None:
    """Returns the response content, or None if all retries were exhausted."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as exc:  # classified via _is_retryable below
            last_exc = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                logger.error("%s failed (attempt %d/%d): %s", label, attempt, MAX_RETRIES, exc)
                break
            wait_s = min(2 ** attempt, 8)
            logger.warning(
                "%s transient error (attempt %d/%d), retrying in %ds: %s",
                label, attempt, MAX_RETRIES, wait_s, exc,
            )
            await asyncio.sleep(wait_s)
    logger.error("%s giving up after retries: %s", label, last_exc)
    return None


def _make_reviewer(category: str):
    async def reviewer(state: ReviewState) -> dict:
        logger.info("Starting %s review for %s", category, state["filename"])
        llm = get_llm()
        messages = [
            SystemMessage(content=REVIEWER_PROMPTS[category]),
            HumanMessage(
                content=f"File: {state['filename']}\n\n```\n{state['code']}\n```"
            ),
        ]
        content = await _call_with_retry(llm, messages, f"{category} review")
        if content is None:
            # Graceful degradation: one agent failing shouldn't kill the whole review.
            content = f"_{category.capitalize()} review unavailable after {MAX_RETRIES} attempts._"
        else:
            logger.info("Completed %s review for %s", category, state["filename"])
        return {"findings": [{"category": category, "content": content}]}

    return reviewer


security_review = _make_reviewer("security")
style_review = _make_reviewer("style")
logic_review = _make_reviewer("logic")


async def aggregate_review(state: ReviewState) -> dict:
    logger.info("Aggregating %d findings for %s", len(state["findings"]), state["filename"])
    llm = get_llm(temperature=0.3)
    findings_text = "\n\n".join(
        f"### {f['category'].upper()} REVIEW\n{f['content']}" for f in state["findings"]
    )
    messages = [
        SystemMessage(
            content=(
                "You are a tech lead writing a final PR review comment. You are given "
                "three specialist reviews (security, style, logic) for one file. "
                "Synthesize them into a single, well-organized review with: "
                "1) a one-line verdict (Approve / Approve with comments / Request changes), "
                "2) a short prioritized list of issues (highest severity first), "
                "3) what's good about the code, if anything. If any review is marked "
                "unavailable, note that it was skipped rather than treating it as 'no issues'. "
                "Be concise. Use markdown."
            )
        ),
        HumanMessage(content=f"File: {state['filename']}\n\n{findings_text}"),
    ]
    content = await _call_with_retry(llm, messages, "aggregate review")
    if content is None:
        content = "Aggregation failed after retries. See individual agent findings above."
    return {"final_review": content}


def build_graph():
    graph = StateGraph(ReviewState)

    graph.add_node("security", security_review)
    graph.add_node("style", style_review)
    graph.add_node("logic", logic_review)
    graph.add_node("aggregate", aggregate_review)

    graph.add_edge(START, "security")
    graph.add_edge(START, "style")
    graph.add_edge(START, "logic")

    graph.add_edge("security", "aggregate")
    graph.add_edge("style", "aggregate")
    graph.add_edge("logic", "aggregate")

    graph.add_edge("aggregate", END)

    return graph.compile()


async def run_review_async(code: str, filename: str = "submitted_file.py") -> ReviewState:
    validate_code_input(code, filename)
    app = build_graph()
    result = await app.ainvoke({"code": code, "filename": filename, "findings": []})
    return result


def run_review(code: str, filename: str = "submitted_file.py") -> ReviewState:
    """Sync wrapper for callers (e.g. Streamlit) that aren't in an async context."""
    return asyncio.run(run_review_async(code, filename))


def get_mermaid_png_bytes() -> bytes:
    app = build_graph()
    return app.get_graph().draw_mermaid_png()
