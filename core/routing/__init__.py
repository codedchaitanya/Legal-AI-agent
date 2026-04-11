# core/routing — domain classification and adapter selection
from core.routing.domain_router import classify_domains, DOMAINS
from core.routing.adapter_selector import select_adapters, DOMAIN_TO_ADAPTER, ADAPTER_DESCRIPTIONS

__all__ = [
    "classify_domains", "DOMAINS",
    "select_adapters", "DOMAIN_TO_ADAPTER", "ADAPTER_DESCRIPTIONS",
]
