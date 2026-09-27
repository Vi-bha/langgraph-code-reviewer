"""
Tests for the code review agent.

These mock the LLM entirely — no GROQ_API_KEY or network access is needed
to run this file, which is what lets it run in CI on every push.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_agent


# ---------- Input validation ----------

def test_validate_code_input_rejects_empty():
    with pytest.raises(graph_agent.InputValidationError):
        graph_agent.validate_code_input("   ", "file.py")


def test_validate_code_input_rejects_oversized():
    huge = "x" * (graph_agent.MAX_CODE_CHARS + 1)
    with pytest.raises(graph_agent.InputValidationError):
        graph_agent.validate_code_input(huge, "file.py")


def test_validate_code_input_rejects_path_in_filename():
    with pytest.raises(graph_agent.InputValidationError):
        graph_agent.validate_code_input("print(1)", "../etc/passwd")


def test_validate_code_input_accepts_normal_input():
    # Should not raise.
    graph_agent.validate_code_input("print('hello')", "hello.py")


# ---------- Retry classification ----------

@pytest.mark.parametrize(
    "message,expected",
    [
        ("Rate limit exceeded", True),
        ("Request timed out", True),
        ("503 Service Unavailable", True),
        ("Invalid API key", False),
        ("Model not found", False),
    ],
)
def test_is_retryable_classification(message, expected):
    assert graph_agent._is_retryable(Exception(message)) is expected


# ---------- Config ----------

def test_get_llm_raises_config_error_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(graph_agent.ConfigError):
        graph_agent.get_llm()


# ---------- Graph execution (mocked LLM) ----------

class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_run_review_async_merges_parallel_findings(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    async def fake_ainvoke(self, messages):
        # Distinguish which prompt was used so we can assert correct routing.
        system_content = messages[0].content
        if "security" in system_content.lower():
            return _FakeResponse("no security issues")
        if "style" in system_content.lower():
            return _FakeResponse("style is fine")
        if "logic" in system_content.lower():
            return _FakeResponse("no logic issues")
        return _FakeResponse("FINAL: Approve")

    with patch("langchain_groq.ChatGroq.ainvoke", new=fake_ainvoke):
        result = await graph_agent.run_review_async("print('hi')", "demo.py")

    categories = {f["category"] for f in result["findings"]}
    assert categories == {"security", "style", "logic"}
    assert result["final_review"]  # aggregator produced something


@pytest.mark.asyncio
async def test_reviewer_degrades_gracefully_on_persistent_failure(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")

    async def always_fails(self, messages):
        raise Exception("Rate limit exceeded")  # retryable, but never succeeds

    with patch("langchain_groq.ChatGroq.ainvoke", new=always_fails):
        with patch("graph_agent.asyncio.sleep", new=AsyncMock()):  # skip real backoff delay
            reviewer = graph_agent._make_reviewer("security")
            result = await reviewer({"code": "x", "filename": "f.py", "findings": []})

    assert "unavailable" in result["findings"][0]["content"].lower()
