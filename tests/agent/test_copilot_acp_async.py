"""Regression tests for the CopilotACPClient async wrapper.

Bug: previously ``_to_async_client`` returned the sync ``CopilotACPClient``
unchanged when ``async_mode=True``. Callers in ``async_call_llm`` then did
``await client.chat.completions.create(...)`` and got::

    TypeError: object types.SimpleNamespace can't be used in 'await' expression

These tests pin down both directions:

1. The sync client still returns a SimpleNamespace synchronously.
2. The async wrapper exposes an awaitable ``create`` that returns the same
   SimpleNamespace shape.
3. ``_to_async_client`` actually swaps the sync client for the async wrapper.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.copilot_acp_client import (
    AsyncCopilotACPClient,
    CopilotACPClient,
)


def _fake_response(_self, **_kwargs):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ok",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        model="copilot-acp",
    )


@pytest.fixture
def sync_client():
    # Avoid resolving a real ACP command/cwd off the test host.
    with patch.object(CopilotACPClient, "_create_chat_completion", _fake_response):
        client = CopilotACPClient.__new__(CopilotACPClient)
        client.api_key = "test-key"
        client.base_url = "acp://copilot"
        from agent.copilot_acp_client import _ACPChatNamespace

        client.chat = _ACPChatNamespace(client)
        client.is_closed = False
        client._active_process = None
        import threading

        client._active_process_lock = threading.Lock()
        yield client


def test_sync_client_create_returns_simplenamespace(sync_client):
    """Sync path is unchanged — returns a SimpleNamespace, not a coroutine."""
    result = sync_client.chat.completions.create(model="x", messages=[])
    assert isinstance(result, SimpleNamespace)
    assert result.choices[0].message.content == "ok"
    assert not asyncio.iscoroutine(result)


def test_async_wrapper_create_is_awaitable(sync_client):
    """Async wrapper's create() must return a coroutine (the bug)."""
    aclient = AsyncCopilotACPClient(sync_client)
    coro = aclient.chat.completions.create(model="x", messages=[])
    assert asyncio.iscoroutine(coro)
    result = asyncio.run(coro)
    assert isinstance(result, SimpleNamespace)
    assert result.choices[0].message.content == "ok"


def test_to_async_client_wraps_copilot_acp(sync_client):
    """_to_async_client must swap the sync ACP client for the async wrapper."""
    from agent.auxiliary_client import _to_async_client

    async_client, model = _to_async_client(sync_client, "claude-opus-4.7", is_vision=True)
    assert isinstance(async_client, AsyncCopilotACPClient)
    assert model == "claude-opus-4.7"

    # And awaiting the new client's create works (no TypeError).
    async def _run():
        return await async_client.chat.completions.create(model="x", messages=[])

    out = asyncio.run(_run())
    assert out.choices[0].message.content == "ok"


def test_to_async_client_preserves_other_clients_unchanged():
    """Sanity: AsyncOpenAI path (default branch) still kicks in for plain clients."""
    from agent.auxiliary_client import _to_async_client

    plain = SimpleNamespace(api_key="k", base_url="https://api.openai.com/v1")
    async_client, model = _to_async_client(plain, "gpt-4o-mini")
    # Should not be an AsyncCopilotACPClient — falls through to AsyncOpenAI.
    assert not isinstance(async_client, AsyncCopilotACPClient)
    assert model == "gpt-4o-mini"
