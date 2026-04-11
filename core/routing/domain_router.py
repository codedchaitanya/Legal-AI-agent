# core/routing/domain_router.py
"""
Multi-label Indian legal domain classifier.
Routes queries to 1 of 12 legal domain adapters using Claude Haiku.
"""
import json
from anthropic import AsyncAnthropic
from core.config import settings
from typing import Optional

_client: AsyncAnthropic | None = None

DOMAINS = [
    "criminal_violent",
    "criminal_property",
    "kidnapping_trafficking",
    "sexual_offences",
    "land_property",
    "family_matrimonial",
    "constitutional",
    "corporate_commercial",
    "labour_employment",
    "cyber_digital",
    "tax_fiscal",
    "civil_general",
]

DOMAIN_HINTS = {
    "criminal_violent": "murder, culpable homicide, assault, grievous hurt, attempt to murder, BNS 100-115, IPC 302/307/323",
    "criminal_property": "theft, robbery, dacoity, extortion, cheating, criminal breach of trust, BNS 303-333, IPC 379/392/420",
    "kidnapping_trafficking": "kidnapping, abduction, trafficking, wrongful confinement, BNS 137-144, IPC 363-365",
    "sexual_offences": "rape, POCSO, stalking, voyeurism, sexual harassment, molestation, BNS 63-99, IPC 354/375/376",
    "land_property": "land dispute, property transfer, tenancy, encroachment, revenue, mutation, Transfer of Property Act",
    "family_matrimonial": "divorce, maintenance, child custody, domestic violence, Hindu Marriage Act, Muslim personal law, section 498A",
    "constitutional": "fundamental rights, writ petition, Article 14/19/21, PIL, habeas corpus, mandamus",
    "corporate_commercial": "company law, SEBI, insolvency (IBC), NCLT, partnership, commercial disputes, arbitration",
    "labour_employment": "industrial dispute, PF, ESI, minimum wages, wrongful termination, factory act, ESIC",
    "cyber_digital": "IT Act, data protection, phishing, hacking, cyberstalking, digital evidence, Section 66",
    "tax_fiscal": "income tax, GST, customs, excise, tax evasion, ITAT, revenue tribunal",
    "civil_general": "CPC suit, injunction, specific relief, torts, consumer protection, money recovery, declaratory suit",
}

ROUTER_PROMPT = f"""You are an Indian legal domain classifier. You must classify a legal document or query into one or more of these 12 domains:

{chr(10).join(f'- {d}: {DOMAIN_HINTS[d]}' for d in DOMAINS)}

Return ONLY a valid JSON array like:
[{{"domain": "criminal_violent", "confidence": 0.92}}, {{"domain": "family_matrimonial", "confidence": 0.45}}]

Sort by confidence descending. Include only domains with confidence > 0.10.

Text to classify:
"""


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


async def call_classifier(text: str) -> list[dict]:
    response = await _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": ROUTER_PROMPT + text[:2000]}],
        temperature=0,
    )
    raw_text = response.content[0].text
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Classifier returned invalid JSON: {raw_text[:200]}") from exc
    if isinstance(raw, list):
        return raw
    return raw.get("domains", raw.get("classifications", []))


async def classify_domains(
    text: str,
    redis=None,
) -> list[dict]:
    """Returns sorted list of {domain, confidence} dicts."""
    if redis:
        cached = await redis.get_router_result(text)
        if cached:
            return cached

    result = await call_classifier(text)

    if redis:
        await redis.set_router_result(text, result)

    return result
