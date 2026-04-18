# core/reasoning/argument_generator.py
import base64
import json
from pathlib import Path

import anthropic

from core.config import settings

_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

ARGUMENT_SYSTEM = """You are a senior Indian advocate with 20+ years of experience in litigation.
You reason step-by-step (chain-of-thought) before stating conclusions.
All citations must reference BNS/IPC sections, Constitutional Articles, or case law explicitly provided in the context.
Never invent citations. Structure your response as valid JSON."""

ARGUMENT_SCHEMA = """{
  "prosecution_arguments": [
    {"point": "...", "legal_basis": "...", "strength": "high|medium|low"}
  ],
  "defense_arguments": [
    {"point": "...", "legal_basis": "...", "strength": "high|medium|low"}
  ],
  "key_precedents": ["case name and relevance"],
  "strategic_recommendations": ["actionable advice for the advocate"],
  "risk_assessment": "overall case risk: high|medium|low with one-line reason"
}"""


def _encode_image(image_path: str) -> dict:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                  "png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


async def generate_arguments(
    case_summary: str,
    research_context: str,
    domain: str,
    image_paths: list[str] | None = None,
    side: str = "both",  # "prosecution" | "defense" | "both"
) -> dict:
    """
    Generate legal arguments using Claude Sonnet with CoT reasoning and vision.

    Args:
        case_summary:      Output from the summarizer pipeline
        research_context:  Output from run_case_research (legal context + citations)
        domain:            Legal domain (e.g. "criminal_violent")
        image_paths:       Optional list of image paths (court exhibits, documents)
        side:              Which side's arguments to generate
    """
    content: list = []

    # Attach images if provided (vision input)
    if image_paths:
        for path in image_paths[:5]:  # cap at 5 images per request
            try:
                content.append(_encode_image(path))
                content.append({"type": "text", "text": f"[Image: {Path(path).name}]"})
            except Exception:
                pass

    prompt = f"""Domain: {domain.replace("_", " ").title()}
Side requested: {side}

=== CASE SUMMARY ===
{case_summary}

=== LEGAL RESEARCH & CONTEXT ===
{research_context}

=== TASK ===
Think step-by-step about the legal issues, applicable laws, and strategic options.
Then generate arguments in this exact JSON schema:
{ARGUMENT_SCHEMA}

Respond with only the JSON object, no markdown fences."""

    content.append({"type": "text", "text": prompt})

    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=ARGUMENT_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Return raw text if JSON parsing fails — still useful
        return {"raw_arguments": raw, "parse_error": True}
