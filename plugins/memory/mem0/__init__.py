"""Mem0 memory plugin — MemoryProvider interface.

Supports two backends:
  - cloud: Mem0 Platform via ``MemoryClient(api_key=...)`` — server-side fact
           extraction, semantic search with reranking, automatic dedup.
  - local: Self-hosted via ``Memory.from_config(...)`` — e.g. Qdrant +
           Ollama / OpenAI-compatible embedder + LLM. 127.0.0.1-only setups.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.
Local backend extension (Atlas, 2026-04-29).

Config via environment variables and/or $HERMES_HOME/mem0.json:
  MEM0_BACKEND       — "cloud" (default) or "local"
  MEM0_API_KEY       — Mem0 Platform API key (cloud only)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)
  MEM0_LOCAL_CONFIG  — Path to a JSON file holding the dict passed to
                       ``Memory.from_config`` (default:
                       $HERMES_HOME/mem0_local.json)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

_VALID_BACKENDS = ("cloud", "local")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _default_local_config_path() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home() / "mem0_local.json")


def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "backend": os.environ.get("MEM0_BACKEND", "cloud"),
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "local_config": os.environ.get("MEM0_LOCAL_CONFIG", ""),
        "rerank": True,
        "keyword_search": False,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    backend = (config.get("backend") or "cloud").lower()
    if backend not in _VALID_BACKENDS:
        backend = "cloud"
    config["backend"] = backend

    if not config.get("local_config"):
        config["local_config"] = _default_local_config_path()

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries (cloud backend only)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory — cloud (Platform API) or local (self-hosted Memory.from_config)."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._backend = "cloud"
        self._api_key = ""
        self._local_config_path = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        backend = cfg.get("backend", "cloud")
        if backend == "local":
            path = cfg.get("local_config") or _default_local_config_path()
            return os.path.exists(path)
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "backend", "description": "Backend (cloud=Mem0 Platform, local=self-hosted)",
             "default": "cloud", "choices": ["cloud", "local"]},
            {"key": "api_key", "description": "Mem0 Platform API key (cloud backend only — leave blank for local)",
             "secret": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "local_config", "description": "Path to local config JSON (local backend; blank = $HERMES_HOME/mem0_local.json)",
             "default": ""},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall (cloud only)",
             "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._backend == "local":
                try:
                    from mem0 import Memory
                except ImportError:
                    raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")
                cfg_path = self._local_config_path or _default_local_config_path()
                if not os.path.exists(cfg_path):
                    raise RuntimeError(
                        f"Local mem0 config not found at {cfg_path}. "
                        f"Create it (Memory.from_config dict serialized as JSON) or set MEM0_LOCAL_CONFIG."
                    )
                with open(cfg_path, "r", encoding="utf-8") as f:
                    local_cfg = json.load(f)
                self._client = Memory.from_config(local_cfg)
            else:
                try:
                    from mem0 import MemoryClient
                except ImportError:
                    raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")
                if not self._api_key:
                    raise RuntimeError("MEM0_API_KEY not set (cloud backend). Switch backend=local or provide an API key.")
                self._client = MemoryClient(api_key=self._api_key)
            return self._client

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._backend = self._config.get("backend", "cloud")
        self._api_key = self._config.get("api_key", "")
        self._local_config_path = self._config.get("local_config") or _default_local_config_path()
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2/local wrap results in {"results": [...]}"""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    # ------- Backend-aware shims --------------------------------------------

    def _local_filters(self) -> dict:
        # OSS Memory rejects unknown filter keys; only include user_id (agent_id
        # is often absent in self-hosted payloads). Allow override via env.
        f = {"user_id": self._user_id}
        if os.environ.get("MEM0_LOCAL_INCLUDE_AGENT_ID") == "1":
            f["agent_id"] = self._agent_id
        return f

    def _do_search(self, client, query: str, top_k: int, rerank: bool):
        if self._backend == "local":
            # OSS Memory.search: (query, *, top_k, filters, threshold, rerank, ...)
            return self._unwrap_results(client.search(
                query, top_k=top_k, filters=self._local_filters(),
            ))
        return self._unwrap_results(client.search(
            query=query, filters=self._read_filters(),
            rerank=rerank, top_k=top_k,
        ))

    def _do_get_all(self, client):
        if self._backend == "local":
            return self._unwrap_results(client.get_all(filters=self._local_filters()))
        return self._unwrap_results(client.get_all(filters=self._read_filters()))

    def _do_add(self, client, messages_or_text, *, infer: bool = True):
        if self._backend == "local":
            # OSS Memory.add: agent_id only if your payloads include it.
            kwargs = {"user_id": self._user_id, "infer": infer}
            if os.environ.get("MEM0_LOCAL_INCLUDE_AGENT_ID") == "1":
                kwargs["agent_id"] = self._agent_id
            return client.add(messages_or_text, **kwargs)
        return client.add(
            messages_or_text,
            user_id=self._user_id,
            agent_id=self._agent_id,
            infer=infer,
        )

    # ------- MemoryProvider hooks -------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory\n"
            f"Active ({self._backend}). User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        # Synchronous fallback: if no queued result is available (e.g. first
        # turn before any queue_prefetch fired, or direct prefetch_all call
        # outside the agent loop), perform the search inline so callers don't
        # silently get "".  Cheap because it's the same _do_search path.
        if not result and query and not self._is_breaker_open():
            try:
                client = self._get_client()
                results = self._do_search(client, query, top_k=15, rerank=self._rerank)
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 sync prefetch failed: %s", e)
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._do_search(client, query, top_k=15, rerank=self._rerank)
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to mem0 for fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                self._do_add(client, messages, infer=True)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._do_get_all(client)
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._do_search(client, query, top_k=top_k, rerank=rerank)
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                self._do_add(
                    client,
                    [{"role": "user", "content": conclusion}],
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
