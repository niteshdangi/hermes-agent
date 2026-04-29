"""Tests for the multiplex memory provider — fan-out, dedupe, failure tolerance."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from agent.memory_provider import MemoryProvider
from plugins.memory.multiplex import MultiplexMemoryProvider


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeChild(MemoryProvider):
    """Minimal MemoryProvider with controllable add/search/get_all/delete."""

    def __init__(self, name: str, *,
                 search_results: List[Dict[str, Any]] = None,
                 all_results: List[Dict[str, Any]] = None,
                 fail_search: bool = False,
                 fail_init: bool = False,
                 available: bool = True,
                 tools: List[Dict[str, Any]] = None):
        self._name = name
        self._search_results = search_results or []
        self._all_results = all_results or []
        self._fail_search = fail_search
        self._fail_init = fail_init
        self._available = available
        self._tools = tools or []
        self.add_calls: List[tuple] = []
        self.delete_calls: List[str] = []
        self.tool_calls: List[tuple] = []
        self.initialized = False
        self.shutdown_called = False
        self.sync_turns: List[tuple] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def initialize(self, session_id: str, **kwargs) -> None:
        if self._fail_init:
            raise RuntimeError(f"{self._name} init boom")
        self.initialized = True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def get_tool_schemas(self):
        return self._tools

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.tool_calls.append((tool_name, args))
        return json.dumps({"result": f"{self._name}:{tool_name}"})

    def system_prompt_block(self) -> str:
        return f"## {self._name}\nfake block"

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        self.sync_turns.append((user_content, assistant_content))

    # CRUD surface
    def search(self, query, limit=10):
        if self._fail_search:
            raise RuntimeError(f"{self._name} search boom")
        return list(self._search_results)

    def get_all(self):
        return list(self._all_results)

    def add(self, *args, **kwargs):
        self.add_calls.append((args, kwargs))
        return f"{self._name}-ok"

    def delete(self, memory_id):
        self.delete_calls.append(memory_id)
        return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _make_mux(children):
    return MultiplexMemoryProvider(children=children)


def test_search_merges_dedupes_and_sorts():
    a = FakeChild("a", search_results=[
        {"id": "x1", "memory": "shared fact", "score": 0.9},
        {"id": "a1", "memory": "only A 1", "score": 0.7},
        {"id": "a2", "memory": "only A 2", "score": 0.6},
    ])
    b = FakeChild("b", search_results=[
        {"id": "x1", "memory": "shared fact", "score": 0.5},   # dup by id
        {"id": "b1", "memory": "only B 1", "score": 0.8},
    ])
    mux = _make_mux([a, b])

    results = mux.search("anything", limit=10)

    ids = [r["id"] for r in results]
    assert "x1" in ids
    # dedup: x1 only appears once
    assert ids.count("x1") == 1
    # higher score wins for x1
    x1 = next(r for r in results if r["id"] == "x1")
    assert x1["score"] == 0.9
    # sorted desc
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # 4 unique items total (x1, a1, a2, b1)
    assert len(results) == 4


def test_search_one_child_fails_gracefully():
    a = FakeChild("a", search_results=[
        {"id": "1", "memory": "alpha", "score": 0.5},
    ])
    b = FakeChild("b", fail_search=True)
    mux = _make_mux([a, b])

    results = mux.search("q", limit=5)
    assert len(results) == 1
    assert results[0]["id"] == "1"


def test_dedupe_by_text_when_no_id():
    a = FakeChild("a", search_results=[{"memory": "same text", "score": 0.4}])
    b = FakeChild("b", search_results=[{"memory": "same text", "score": 0.9}])
    mux = _make_mux([a, b])

    results = mux.search("q", limit=5)
    assert len(results) == 1
    assert results[0]["score"] == 0.9


def test_get_all_unions_and_dedupes():
    a = FakeChild("a", all_results=[
        {"id": "1", "memory": "one"},
        {"id": "2", "memory": "two"},
    ])
    b = FakeChild("b", all_results=[
        {"id": "2", "memory": "two"},  # dup
        {"id": "3", "memory": "three"},
    ])
    mux = _make_mux([a, b])
    items = mux.get_all()
    ids = sorted(r["id"] for r in items)
    assert ids == ["1", "2", "3"]


def test_add_fans_out_best_effort():
    a = FakeChild("a")
    b = FakeChild("b")
    mux = _make_mux([a, b])
    out = mux.add("hello", user_id="u1")
    assert len(a.add_calls) == 1
    assert len(b.add_calls) == 1
    names = [n for (n, _) in out]
    assert names == ["a", "b"]


def test_delete_fans_out():
    a = FakeChild("a")
    b = FakeChild("b")
    mux = _make_mux([a, b])
    mux.delete("xyz")
    assert a.delete_calls == ["xyz"]
    assert b.delete_calls == ["xyz"]


def test_is_available_true_if_any_child_available():
    a = FakeChild("a", available=False)
    b = FakeChild("b", available=True)
    mux = _make_mux([a, b])
    assert mux.is_available() is True


def test_is_available_false_if_no_child_available():
    a = FakeChild("a", available=False)
    b = FakeChild("b", available=False)
    mux = _make_mux([a, b])
    assert mux.is_available() is False


def test_initialize_skips_unavailable_and_failing_children():
    a = FakeChild("a", available=True)
    b = FakeChild("b", available=False)        # skipped: unavailable
    c = FakeChild("c", fail_init=True)         # dropped: init raised
    mux = _make_mux([a, b, c])
    mux.initialize("sess-1")
    names = [c.name for c in mux._children]
    assert names == ["a"]
    assert a.initialized


def test_tool_schemas_union_and_dispatch():
    a = FakeChild("a", tools=[{"name": "a_search", "parameters": {}}])
    b = FakeChild("b", tools=[{"name": "b_search", "parameters": {}}])
    mux = _make_mux([a, b])
    mux.initialize("sess")

    schemas = mux.get_tool_schemas()
    names = sorted(s["name"] for s in schemas)
    assert names == ["a_search", "b_search"]

    out_a = mux.handle_tool_call("a_search", {"q": 1})
    assert json.loads(out_a)["result"] == "a:a_search"
    out_b = mux.handle_tool_call("b_search", {"q": 2})
    assert json.loads(out_b)["result"] == "b:b_search"

    # Unknown tool
    err = json.loads(mux.handle_tool_call("missing", {}))
    assert "error" in err


def test_system_prompt_block_concatenates():
    a = FakeChild("a")
    b = FakeChild("b")
    mux = _make_mux([a, b])
    block = mux.system_prompt_block()
    assert "## a" in block and "## b" in block
    assert "multiplex" in block.lower()


def test_sync_turn_fans_out():
    a = FakeChild("a")
    b = FakeChild("b")
    mux = _make_mux([a, b])
    mux.sync_turn("u", "asst")
    assert a.sync_turns == [("u", "asst")]
    assert b.sync_turns == [("u", "asst")]


def test_register_uses_collector():
    """register(ctx) installs a multiplex instance into the slot."""
    from plugins.memory.multiplex import register

    class Collector:
        def __init__(self):
            self.provider = None
        def register_memory_provider(self, p):
            self.provider = p

    c = Collector()
    register(c)
    assert isinstance(c.provider, MultiplexMemoryProvider)
    assert c.provider.name == "multiplex"
