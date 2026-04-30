"""Atlas Mobile pairing & WS gateway.

Server-side endpoints for the atlas-mobile Android client:
- Device registration & enrollment (HTTP)
- Authenticated WebSocket transport (P-256 ECDSA signed nonce)
- Envelope dispatch into the Hermes identity-keyed session for "nitesh"

Tailscale-only: do not expose publicly.
"""

from .router import register_routes, build_router

__all__ = ["register_routes", "build_router"]
