"""Tests for cron triage must-forward / [SILENT] override logic."""

from unittest.mock import patch

import pytest

from cron.atlas_triage import (
    SILENT_MARKER,
    run_triage_turn,
    should_force_forward,
)


# --- should_force_forward unit tests --------------------------------------

def test_force_forward_on_stuck_marker():
    assert should_force_forward("❌ Stuck — need cloudflare-token.") is True


def test_force_forward_on_needs_from_you():
    assert should_force_forward(
        "Result: progress made.\nBefore I can proceed I need from you a token."
    ) is True


def test_force_forward_on_awaiting():
    assert should_force_forward("Awaiting cloudflare-token + go-ahead.") is True


def test_force_forward_on_question_bullet():
    text = "Result:\n- All ok except one thing.\n- Should I sign up for Resend now?"
    assert should_force_forward(text) is True


def test_no_force_forward_on_clean_success():
    assert should_force_forward("✅ All done. No follow-up.") is False


def test_no_force_forward_on_self_test_ok():
    assert should_force_forward(
        "self-test ok\nheartbeat tick — nothing new"
    ) is False


def test_force_forward_on_cannot_proceed():
    assert should_force_forward("I cannot proceed without the API key.") is True


# --- run_triage_turn override behaviour -----------------------------------

class _FakeAgent:
    def __init__(self, response):
        self._response = response

    def __call__(self, *args, **kwargs):  # pragma: no cover - not used
        return self

    def run_conversation(self, **_kwargs):
        return {"final_response": self._response}

    def close(self):
        pass


def _patch_agent(response):
    fake = _FakeAgent(response)

    def factory(**_kwargs):
        return fake

    return patch("run_agent.AIAgent", factory)


def test_silent_overridden_when_blocker_present():
    cron_output = (
        "❌ Stuck — required vault entries missing.\n"
        "Before I can proceed I need from you a Cloudflare API token."
    )
    job = {"id": "test-job-1", "name": "spawn-9-completed"}
    with _patch_agent(SILENT_MARKER):
        out = run_triage_turn(job, cron_output, status="ok")
    assert out != SILENT_MARKER
    assert "Cloudflare API token" in out
    assert "needing your attention" in out or "blocker" in out.lower()


def test_silent_preserved_for_clean_success():
    cron_output = "✅ All done. No follow-up."
    job = {"id": "test-job-2", "name": "self-test"}
    with _patch_agent(SILENT_MARKER):
        out = run_triage_turn(job, cron_output, status="ok")
    assert out == SILENT_MARKER


def test_question_in_bullet_triggers_forward():
    cron_output = (
        "Result:\n"
        "- Resend signup needs phone verification.\n"
        "- Should I proceed with SendGrid instead?"
    )
    job = {"id": "test-job-3", "name": "spawn-x"}
    with _patch_agent(SILENT_MARKER):
        out = run_triage_turn(job, cron_output, status="ok")
    assert out != SILENT_MARKER
    assert "SendGrid" in out


def test_llm_speaks_passthrough():
    """If the LLM already produced a non-[SILENT] message, we forward it as-is."""
    cron_output = "❌ Stuck — need token."
    job = {"id": "test-job-4", "name": "spawn-y"}
    with _patch_agent("⚠️ Atlas needs token X to continue."):
        out = run_triage_turn(job, cron_output, status="ok")
    assert out == "⚠️ Atlas needs token X to continue."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
