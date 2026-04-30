"""Tests for the cross-channel identity layer.

Covers:
  - IdentityRegistry.resolve() exact and wildcard matching
  - SessionStore session-key collapsing for identity-bound sources
  - Backward-compat: no identity → legacy per-channel session key
  - format_channel_tag per-platform output
  - SessionContext gets identity_id/display_name when bound
"""

from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.identity import IdentityRegistry
from gateway.session import (
    SessionSource,
    SessionStore,
    build_session_context,
    build_session_key,
    format_channel_tag,
)


NITESH_CFG = {
    "nitesh": {
        "display_name": "Nitesh Kumar",
        "channels": [
            {"platform": "telegram", "chat_id": "1717765989"},
            {"platform": "whatsapp", "chat_id": "917027026665"},
            {"platform": "local", "chat_id": "*"},
        ],
    }
}


@pytest.fixture()
def registry():
    return IdentityRegistry.from_config(NITESH_CFG)


@pytest.fixture()
def store_with_identity(tmp_path, registry):
    cfg = GatewayConfig()
    cfg.identities = registry
    with patch("gateway.session.SessionStore._ensure_loaded"):
        s = SessionStore(sessions_dir=tmp_path, config=cfg)
    s._db = None
    s._loaded = True
    return s


@pytest.fixture()
def store_no_identity(tmp_path):
    cfg = GatewayConfig()  # default empty IdentityRegistry
    with patch("gateway.session.SessionStore._ensure_loaded"):
        s = SessionStore(sessions_dir=tmp_path, config=cfg)
    s._db = None
    s._loaded = True
    return s


def _dm(platform, chat_id, user_id="U1"):
    return SessionSource(
        platform=platform, chat_id=chat_id, chat_type="dm", user_id=user_id
    )


# ---------------------------------------------------------------------------
# IdentityRegistry
# ---------------------------------------------------------------------------

def test_registry_resolve_exact_match(registry):
    ident = registry.resolve("telegram", "1717765989")
    assert ident is not None
    assert ident.identity_id == "nitesh"
    assert ident.display_name == "Nitesh Kumar"
    assert ident.session_key == "agent:identity:nitesh"


def test_registry_resolve_whatsapp(registry):
    ident = registry.resolve("whatsapp", "917027026665")
    assert ident is not None and ident.identity_id == "nitesh"


def test_registry_wildcard_chat_id(registry):
    # local has chat_id "*" — any chat matches
    assert registry.resolve("local", "anything").identity_id == "nitesh"
    assert registry.resolve("local", "").identity_id == "nitesh"
    assert registry.resolve("local", "user@host").identity_id == "nitesh"


def test_registry_no_match_returns_none(registry):
    assert registry.resolve("telegram", "9999") is None
    assert registry.resolve("discord", "1717765989") is None
    assert registry.resolve(None, "x") is None


def test_registry_empty_config():
    r = IdentityRegistry.from_config(None)
    assert r.identities == []
    assert r.resolve("telegram", "1") is None
    r2 = IdentityRegistry.from_config({})
    assert r2.identities == []


def test_registry_drops_malformed_entry(caplog):
    cfg = {
        "good": {"channels": [{"platform": "telegram", "chat_id": "1"}]},
        "bad": "not-a-dict",
        "empty_chans": {"channels": []},
    }
    r = IdentityRegistry.from_config(cfg)
    ids = [i.identity_id for i in r.identities]
    assert ids == ["good"]


# ---------------------------------------------------------------------------
# SessionStore session-key resolution
# ---------------------------------------------------------------------------

def test_store_uses_identity_keyed_session(store_with_identity):
    src = _dm(Platform.TELEGRAM, "1717765989")
    entry = store_with_identity.get_or_create_session(src)
    assert entry.session_key == "agent:identity:nitesh"
    assert entry.identity_id == "nitesh"
    assert entry.display_name == "Nitesh Kumar"


def test_store_collapses_telegram_and_whatsapp_to_same_key(store_with_identity):
    e1 = store_with_identity.get_or_create_session(_dm(Platform.TELEGRAM, "1717765989"))
    e2 = store_with_identity.get_or_create_session(_dm(Platform.WHATSAPP, "917027026665"))
    assert e1.session_key == e2.session_key == "agent:identity:nitesh"
    # Same entry → same session_id (continuity across channels)
    assert e1.session_id == e2.session_id


def test_store_local_wildcard_matches(store_with_identity):
    src = SessionSource(platform=Platform.LOCAL, chat_id="local", chat_type="dm")
    entry = store_with_identity.get_or_create_session(src)
    assert entry.session_key == "agent:identity:nitesh"


def test_store_falls_back_to_channel_key_no_identity(store_no_identity):
    src = _dm(Platform.TELEGRAM, "1717765989")
    entry = store_no_identity.get_or_create_session(src)
    assert entry.session_key == build_session_key(src)
    assert entry.session_key.startswith("agent:main:telegram:dm:")
    assert entry.identity_id is None


def test_store_unknown_user_falls_back_even_with_registry(store_with_identity):
    # Different chat_id on telegram → no identity match, legacy key
    src = _dm(Platform.TELEGRAM, "9999999999")
    entry = store_with_identity.get_or_create_session(src)
    assert entry.session_key.startswith("agent:main:telegram:dm:")
    assert entry.identity_id is None


def test_store_group_chat_not_collapsed_to_identity(store_with_identity):
    # Even on a configured platform, non-DM groups must not collapse.
    src = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1717765989",
        chat_type="group",
        user_id="U1",
    )
    entry = store_with_identity.get_or_create_session(src)
    assert not entry.session_key.startswith("agent:identity:")


# ---------------------------------------------------------------------------
# format_channel_tag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform,expected", [
    (Platform.TELEGRAM, "[via Telegram]"),
    (Platform.WHATSAPP, "[via WhatsApp]"),
    (Platform.LOCAL, "[via CLI]"),
    (Platform.DISCORD, "[via Discord]"),
    (Platform.SLACK, "[via Slack]"),
    (None, ""),
])
def test_format_channel_tag(platform, expected):
    assert format_channel_tag(platform) == expected


# ---------------------------------------------------------------------------
# build_session_context populates identity fields
# ---------------------------------------------------------------------------

def test_session_context_populates_identity(registry):
    cfg = GatewayConfig()
    cfg.identities = registry
    src = _dm(Platform.TELEGRAM, "1717765989")
    ctx = build_session_context(src, cfg, session_entry=None)
    assert ctx.identity_id == "nitesh"
    assert ctx.identity_display_name == "Nitesh Kumar"
    assert ctx.identity_channels and any(
        ch["platform"] == "whatsapp" for ch in ctx.identity_channels
    )


def test_session_context_no_identity_when_not_bound():
    cfg = GatewayConfig()
    src = _dm(Platform.TELEGRAM, "1717765989")
    ctx = build_session_context(src, cfg, session_entry=None)
    assert ctx.identity_id is None
    assert ctx.identity_display_name is None
