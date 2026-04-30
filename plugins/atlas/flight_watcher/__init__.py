"""Atlas flight-watcher: pluggable price tracker for Indian-context routes."""
from .core import (
    WatchedRoute,
    PriceQuote,
    Backend,
    SerpAPIBackend,
    ScrapingBackend,
    rolling_median,
    is_alertable,
    format_alert,
    load_routes,
    save_routes,
    append_history,
    load_history,
    is_quiet_hour,
    select_backend,
)

__all__ = [
    "WatchedRoute",
    "PriceQuote",
    "Backend",
    "SerpAPIBackend",
    "ScrapingBackend",
    "rolling_median",
    "is_alertable",
    "format_alert",
    "load_routes",
    "save_routes",
    "append_history",
    "load_history",
    "is_quiet_hour",
    "select_backend",
]
