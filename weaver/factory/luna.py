"""Luna's QA read: the model reviews the run's evidence and says so out loud.

Deterministic QA decides pass/fail; Luna's job here is the founder-requested
second read — did the scraper really capture every vehicle correctly — with
its reasoning streamed to the portal as events so the decision-making is
visible live. Structured output keeps the verdict machine-usable.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6-luna"
MAX_INPUT_BYTES = 24_000

VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["ship", "needs_repair", "unsure"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string", "maxLength": 700},
        "concerns": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 200},
        },
    },
    "required": ["verdict", "confidence", "summary", "concerns"],
}

INSTRUCTIONS = (
    "You are the QA reviewer for a dealership inventory scraper factory. "
    "You receive the deterministic QA report of a full crawl, a client-engine "
    "simulation report, and sample vehicle records. Judge ONLY whether every "
    "vehicle on the lot was captured correctly and completely enough to "
    "publish: coverage vs the expected total, per-vehicle field quality, "
    "photo galleries, and whether the client engine simulation agrees with "
    "the crawl. Be specific about anything that looks wrong. "
    "Accepted product policy — do NOT treat these as defects: records marked "
    "photo_exception are page-corroborated states (the dealer published no "
    "unit photography, or exactly one photo) and the posting layer skips or "
    "handles them; drivetrain and features are bonus fields whose absence "
    "across a source is a warning, not a failure. DO flag values that look "
    "semantically wrong even when in-bounds (a price equal to the model year, "
    "mileage equal to a price, identical values across all records)."
)


def _clip(value: Any, limit: int) -> str:
    raw = json.dumps(value, default=str)
    return raw if len(raw) <= limit else raw[:limit] + "…(clipped)"


async def luna_qa_review(
    *,
    qa: dict[str, Any],
    samples: list[dict[str, Any]],
    simulation: dict[str, Any],
    emit,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = (
        os.getenv("FACTORY_QA_MODEL")
        or os.getenv("WEAVER_MODEL")
        or os.getenv("OPENAI_SCRAPER_MODEL")
        or DEFAULT_MODEL
    )
    if not api_key:
        verdict = {
            "verdict": "unsure",
            "confidence": "low",
            "summary": "OPENAI_API_KEY is not configured; Luna QA was skipped.",
            "concerns": ["luna_unconfigured"],
        }
        await emit("luna_skipped", verdict)
        return verdict

    payload_input = {
        "deterministic_qa": qa,
        "client_engine_simulation": simulation,
        "sample_records": samples[:3],
    }
    user_input = _clip(payload_input, MAX_INPUT_BYTES)
    await emit(
        "luna_request",
        {"model": model, "input_bytes": len(user_input), "instructions": INSTRUCTIONS},
    )
    body = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": user_input,
        "max_output_tokens": 1_200,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "factory_qa_verdict",
                "strict": True,
                "schema": VERDICT_SCHEMA,
            }
        },
    }
    try:
        async with httpx.AsyncClient(timeout=90.0, trust_env=False) as client:
            response = await client.post(
                RESPONSES_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
            )
        response.raise_for_status()
        data = response.json()
        text = ""
        for item in data.get("output", []):
            for chunk in item.get("content", []) or []:
                if chunk.get("type") == "output_text":
                    text += chunk.get("text", "")
        verdict = json.loads(text)
    except Exception as error:  # noqa: BLE001 - the portal must see any failure
        verdict = {
            "verdict": "unsure",
            "confidence": "low",
            "summary": f"Luna QA call failed: {str(error)[:200]}",
            "concerns": ["luna_call_failed"],
        }
    await emit("luna_verdict", verdict)
    return verdict
