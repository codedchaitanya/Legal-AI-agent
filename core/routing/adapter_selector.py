# core/routing/adapter_selector.py
"""
Map domain classification scores to LoRA adapter names.
12 adapters covering the Indian legal system.
"""
from core.config import settings


DOMAIN_TO_ADAPTER = {
    "criminal_violent": "criminal_violent",
    "criminal_property": "criminal_property",
    "kidnapping_trafficking": "kidnapping_trafficking",
    "sexual_offences": "sexual_offences",
    "land_property": "land_property",
    "family_matrimonial": "family_matrimonial",
    "constitutional": "constitutional",
    "corporate_commercial": "corporate_commercial",
    "labour_employment": "labour_employment",
    "cyber_digital": "cyber_digital",
    "tax_fiscal": "tax_fiscal",
    "civil_general": "civil_general",
}

ADAPTER_DESCRIPTIONS = {
    "criminal_violent": "Murder, culpable homicide, assault, attempt to murder (BNS 100-115)",
    "criminal_property": "Theft, robbery, extortion, cheating, criminal breach of trust (BNS 303-333)",
    "kidnapping_trafficking": "Kidnapping, abduction, human trafficking (BNS 137-144)",
    "sexual_offences": "Rape, POCSO, stalking, voyeurism, sexual harassment (BNS 63-99)",
    "land_property": "Land disputes, Transfer of Property Act, tenancy, encroachment",
    "family_matrimonial": "Divorce, maintenance, custody, domestic violence, Hindu/Muslim personal law",
    "constitutional": "Fundamental rights, writ petitions, Article 14/19/21, PIL",
    "corporate_commercial": "Company law, SEBI, IBC insolvency, partnership, contract disputes",
    "labour_employment": "Industrial Disputes Act, PF/ESI, wage disputes, unfair termination",
    "cyber_digital": "IT Act, data protection, digital fraud, cyberstalking",
    "tax_fiscal": "Income Tax, GST, customs, excise, tax evasion",
    "civil_general": "CPC suits, injunctions, specific relief, torts, consumer protection",
}

AVAILABLE_ADAPTERS = set(DOMAIN_TO_ADAPTER.keys())


def select_adapters(
    domain_scores: list[dict],
    threshold: float | None = None,
) -> list[str]:
    """
    Return adapter names to activate, sorted by confidence descending.
    Always returns at least one (top-1 regardless of threshold).
    """
    if threshold is None:
        threshold = settings.SECONDARY_ADAPTER_CONFIDENCE

    sorted_domains = sorted(domain_scores, key=lambda x: x["confidence"], reverse=True)
    selected = []

    for item in sorted_domains:
        adapter = DOMAIN_TO_ADAPTER.get(item["domain"])
        if adapter and adapter in AVAILABLE_ADAPTERS:
            if item["confidence"] >= threshold or not selected:
                selected.append(adapter)

    return selected if selected else ["civil_general"]
