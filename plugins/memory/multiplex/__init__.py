"""Multiplex memory provider — fans out to multiple child providers.

Hermes enforces a one-external-provider rule via ``memory.provider`` in
``config.yaml``. This plugin sits in that single slot and itself instantiates
multiple "real" child providers (initially mem0 + honcho) directly — bypassing
``load_memory_provider`` so the registry's single-active rule isn't tripped.

All ``MemoryProvider`` lifecycle hooks fan out to every child in registration
order. Tool schemas are unioned (each child contributes its own tools, e.g.
``mem0_search``, ``honcho_search``); ``handle_tool_call`` routes by tool name
back to the originating child.

In addition to the standard MemoryProvider hooks, we expose a small CRUD
surface (``add``/``search``/``get_all``/``delete``) for callers (and tests)
that want to interact with the multiplex as a uniform store. ``search``
queries every child in parallel, merges, dedupes, and orders by score.

Config (``~/.hermes/config.yaml``):

    memory:
      provider: multiplex
      multiplex:
        children: [mem0, honcho]
      mem0: { ... mem0 settings ... }
      honcho: { ... honcho settings ... }

Each child is loaded by importing its bundled plugin module directly and
instantiating its provider class. Children that fail to import or initialize
are logged and skipped — the multiplex stays alive as long as ≥1 child is
available.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Child loading — direct class import, not through the plugin registry
# ---------------------------------------------------------------------------

# Map of child name -> (module path, class name). Add new children here.
_CHILD_REGISTRY: Dict[str, Tuple[str, str]] = {
    "mem0": ("plugins.memory.mem0", "Mem0MemoryProvider"),
    "honcho": ("plugins.memory.honcho", "HonchoMemoryProvider"),
}


def _instantiate_child(name: str) -> Optional[MemoryProvider]:
    """Import and instantiate a child provider class directly.

    We deliberately do NOT go through ``plugins.memory.load_memory_provider`` —
    that path is gated by the single-provider rule. By importing the child's
    module and constructing the class ourselves, we sidestep that gate.
    """
    if name not in _CHILD_REGISTRY:
        logger.warning("multiplex: unknown child provider '%s'", name)
        return None
    mod_path, cls_name = _CHILD_REGISTRY[name]
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name, None)
        if cls is None:
            logger.warning("multiplex: %s missing class %s", mod_path, cls_name)
            return None
        return cls()
    except Exception as e:
        logger.warning("multiplex: failed to instantiate child '%s': %s", name, e)
        return None


def _load_multiplex_config() -> Dict[str, Any]:
    """Load multiplex config from $HERMES_HOME/config.yaml.

    Looks under ``memory.multiplex`` (and falls back to env var
    ``HERMES_MULTIPLEX_CHILDREN`` as comma-separated list).
    """
    children: List[str] = []
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        mem = cfg.get("memory", {}) or {}
        mux = mem.get("multiplex", {}) or {}
        raw = mux.get("children")
        if isinstance(raw, list):
            children = [str(c).strip() for c in raw if str(c).strip()]
        elif isinstance(raw, str):
            children = [c.strip() for c in raw.split(",") if c.strip()]
    except Exception as e:
        logger.debug("multiplex: load_config failed: %s", e)

    if not children:
        env = os.environ.get("HERMES_MULTIPLEX_CHILDREN", "")
        if env:
            children = [c.strip() for c in env.split(",") if c.strip()]

    if not children:
        children = ["mem0", "honcho"]  # sensible default

    return {"children": children}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class MultiplexMemoryProvider(MemoryProvider):
    """Fan-out provider that runs multiple child providers behind a single slot."""

    def __init__(self, children: Optional[List[MemoryProvider]] = None):
        # ``children`` may be injected (tests) or discovered from config.
        self._children: List[MemoryProvider] = list(children) if children else []
        self._discovered = bool(children)
        self._tool_owner: Dict[str, MemoryProvider] = {}
        self._lock = threading.Lock()

    # ----- Identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "multiplex"

    # ----- Discovery --------------------------------------------------------

    def _ensure_children(self) -> None:
        if self._children or self._discovered:
            return
        cfg = _load_multiplex_config()
        for cname in cfg["children"]:
            child = _instantiate_child(cname)
            if child:
                self._children.append(child)
        self._discovered = True
        logger.info("multiplex: loaded %d child providers: %s",
                    len(self._children),
                    [c.name for c in self._children])

    # ----- Availability / config -------------------------------------------

    def is_available(self) -> bool:
        self._ensure_children()
        for c in self._children:
            try:
                if c.is_available():
                    return True
            except Exception as e:
                logger.debug("multiplex: child %s is_available raised: %s", c.name, e)
        return False

    def get_config_schema(self):
        return [
            {
                "key": "children",
                "description": "Comma-separated list of child provider names (e.g. mem0,honcho)",
                "default": "mem0,honcho",
            },
        ]

    def save_config(self, values, hermes_home):
        """Persist the children list into ``config.yaml`` under memory.multiplex."""
        try:
            from hermes_cli.config import load_config, save_config
            raw = values.get("children", "")
            if isinstance(raw, str):
                children = [c.strip() for c in raw.split(",") if c.strip()]
            else:
                children = list(raw or [])
            cfg = load_config() or {}
            mem = cfg.setdefault("memory", {})
            mux = mem.setdefault("multiplex", {})
            mux["children"] = children
            save_config(cfg)
        except Exception as e:
            logger.warning("multiplex: save_config failed: %s", e)

    # ----- Lifecycle fan-out -----------------------------------------------

    def _safe(self, child: MemoryProvider, op: str, fn, *args, **kwargs):
        """Run ``fn`` on ``child`` with broad exception logging. Returns (ok, result)."""
        try:
            return True, fn(*args, **kwargs)
        except Exception as e:
            logger.warning("multiplex: child %s %s failed: %s", child.name, op, e)
            return False, None

    def initialize(self, session_id: str, **kwargs) -> None:
        self._ensure_children()
        live: List[MemoryProvider] = []
        for c in self._children:
            try:
                if not c.is_available():
                    logger.info("multiplex: child %s not available — skipping", c.name)
                    continue
            except Exception:
                continue
            ok, _ = self._safe(c, "initialize", c.initialize, session_id, **kwargs)
            if ok:
                live.append(c)
        self._children = live

        # Build tool ownership map for routing handle_tool_call().
        owner: Dict[str, MemoryProvider] = {}
        for c in self._children:
            ok, schemas = self._safe(c, "get_tool_schemas", c.get_tool_schemas)
            if ok and schemas:
                for s in schemas:
                    n = s.get("name")
                    if n and n not in owner:
                        owner[n] = c
        self._tool_owner = owner
        logger.info("multiplex: %d active children, %d total tools",
                    len(self._children), len(owner))

    def shutdown(self) -> None:
        for c in self._children:
            self._safe(c, "shutdown", c.shutdown)

    # ----- System prompt + prefetch concat ---------------------------------

    def system_prompt_block(self) -> str:
        parts = []
        for c in self._children:
            ok, txt = self._safe(c, "system_prompt_block", c.system_prompt_block)
            if ok and txt:
                parts.append(str(txt).strip())
        if not parts:
            return ""
        header = (f"# Memory (multiplex: {', '.join(c.name for c in self._children)})\n"
                  "Multiple memory providers are active. Their tool sets are unioned.\n")
        return header + "\n\n".join(parts)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        parts = []
        for c in self._children:
            ok, txt = self._safe(c, "prefetch", c.prefetch, query, session_id=session_id)
            if ok and txt:
                parts.append(str(txt).strip())
        return "\n\n".join(parts)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        for c in self._children:
            self._safe(c, "queue_prefetch", c.queue_prefetch, query, session_id=session_id)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        for c in self._children:
            self._safe(c, "sync_turn", c.sync_turn, user_content, assistant_content,
                       session_id=session_id)

    # ----- Tools -----------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set = set()
        for c in self._children:
            ok, schemas = self._safe(c, "get_tool_schemas", c.get_tool_schemas)
            if not ok or not schemas:
                continue
            for s in schemas:
                n = s.get("name")
                if not n or n in seen:
                    continue
                seen.add(n)
                merged.append(s)
        return merged

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        owner = self._tool_owner.get(tool_name)
        if owner is None:
            # Fallback: rebuild the map (children may have lazy-registered tools)
            for c in self._children:
                ok, schemas = self._safe(c, "get_tool_schemas", c.get_tool_schemas)
                if ok and schemas and any(s.get("name") == tool_name for s in schemas):
                    owner = c
                    self._tool_owner[tool_name] = c
                    break
        if owner is None:
            return json.dumps({"error": f"multiplex: no child owns tool {tool_name!r}"})
        try:
            return owner.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.warning("multiplex: %s.handle_tool_call(%s) failed: %s",
                           owner.name, tool_name, e)
            return json.dumps({"error": f"multiplex: child {owner.name} failed: {e}"})

    # ----- Optional hooks: fan out -----------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        for c in self._children:
            self._safe(c, "on_turn_start", c.on_turn_start, turn_number, message, **kwargs)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        for c in self._children:
            self._safe(c, "on_session_end", c.on_session_end, messages)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        parts = []
        for c in self._children:
            ok, txt = self._safe(c, "on_pre_compress", c.on_pre_compress, messages)
            if ok and txt:
                parts.append(str(txt).strip())
        return "\n\n".join(parts)

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        for c in self._children:
            self._safe(c, "on_delegation", c.on_delegation, task, result,
                       child_session_id=child_session_id, **kwargs)

    def on_memory_write(self, action, target, content, metadata=None) -> None:
        for c in self._children:
            self._safe(c, "on_memory_write", c.on_memory_write,
                       action, target, content, metadata)

    # ----- CRUD surface (best-effort, mostly for tests / direct callers) ---
    #
    # Children may implement any subset of add/search/get_all/delete. We call
    # what's there and skip what isn't; failures never propagate.

    @staticmethod
    def _maybe(child, method: str):
        fn = getattr(child, method, None)
        return fn if callable(fn) else None

    def add(self, *args, **kwargs) -> List[Tuple[str, Any]]:
        """Fan out add() to every child that implements it. Returns per-child results."""
        results: List[Tuple[str, Any]] = []
        for c in self._children:
            fn = self._maybe(c, "add")
            if not fn:
                continue
            ok, res = self._safe(c, "add", fn, *args, **kwargs)
            results.append((c.name, res if ok else None))
        return results

    def get_all(self) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for c in self._children:
            fn = self._maybe(c, "get_all")
            if not fn:
                continue
            ok, res = self._safe(c, "get_all", fn)
            if not ok or not res:
                continue
            for item in self._normalize_items(res, c.name):
                merged.append(item)
        return self._dedupe(merged)

    def delete(self, memory_id: str) -> List[Tuple[str, bool]]:
        out: List[Tuple[str, bool]] = []
        for c in self._children:
            fn = self._maybe(c, "delete")
            if not fn:
                continue
            ok, _ = self._safe(c, "delete", fn, memory_id)
            out.append((c.name, ok))
        return out

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query every child in parallel, merge, dedupe, sort by score desc."""
        results: List[Dict[str, Any]] = []
        targets = [(c, self._maybe(c, "search")) for c in self._children]
        targets = [(c, fn) for (c, fn) in targets if fn is not None]
        if not targets:
            return []

        def _run(c_fn):
            c, fn = c_fn
            try:
                return c.name, fn(query, limit)
            except TypeError:
                # Some children take (query) only, or kwargs
                try:
                    return c.name, fn(query)
                except Exception as e:
                    logger.warning("multiplex: search on %s failed: %s", c.name, e)
                    return c.name, None
            except Exception as e:
                logger.warning("multiplex: search on %s failed: %s", c.name, e)
                return c.name, None

        with ThreadPoolExecutor(max_workers=max(2, len(targets))) as ex:
            for cname, res in ex.map(_run, targets):
                if not res:
                    continue
                for item in self._normalize_items(res, cname):
                    results.append(item)

        merged = self._dedupe(results)
        merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return merged[:limit]

    # ----- Helpers ---------------------------------------------------------

    @staticmethod
    def _normalize_items(raw: Any, source: str) -> List[Dict[str, Any]]:
        """Coerce a child's response into a list of {id, memory, score, source} dicts."""
        if isinstance(raw, dict):
            raw = raw.get("results") or raw.get("memories") or []
        if not isinstance(raw, list):
            return []
        out = []
        for r in raw:
            if not isinstance(r, dict):
                # Treat as bare text
                out.append({"id": None, "memory": str(r), "score": 0.0, "source": source})
                continue
            out.append({
                "id": r.get("id"),
                "memory": r.get("memory") or r.get("text") or r.get("content") or "",
                "score": float(r.get("score", 0.0) or 0.0),
                "source": source,
                "raw": r,
            })
        return out

    @staticmethod
    def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dedupe by id when present, else by sha1(memory text). Highest-score wins."""
        best: Dict[str, Dict[str, Any]] = {}
        for it in items:
            mem = (it.get("memory") or "").strip()
            ident = it.get("id")
            key = f"id::{ident}" if ident else f"text::{hashlib.sha1(mem.encode('utf-8')).hexdigest()}"
            cur = best.get(key)
            if cur is None or it.get("score", 0.0) > cur.get("score", 0.0):
                best[key] = it
        return list(best.values())


def register(ctx) -> None:
    """Plugin entry — register the multiplex provider in the single memory slot."""
    ctx.register_memory_provider(MultiplexMemoryProvider())
