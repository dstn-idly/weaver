"""Turn a needs_repair verdict into a plan the next run can act on.

Four days of requeues fixed nothing because a requeue re-ran the same
strategy with the same knowledge: the deterministic QA codes and Luna's
written diagnosis named exactly what failed, and none of it reached the next
attempt. This module distills that evidence into a typed repair plan — the
causes drive escalation (the same cause failing twice stops the loop and
asks a human), and the bounded notes ride into the next run's spec
inference as hints.

The notes are hints, never authority: the inference prompt already treats
every injected string as untrusted context that cannot choose URLs,
transport behavior, or unproven selectors, and the plan carries no
selectors, code, or network instructions of its own.
"""

from __future__ import annotations

import re
from typing import Any

# Ordered by how decisively each cause explains a bad crawl: an incomplete
# walk poisons every downstream measure, so it outranks a field gap.
_CAUSE_ORDER = (
    "incomplete_coverage",
    "photo_coverage",
    "photo_ownership",
    "missing_field",
    "sim_disagreement",
)

_FIELD_COVERAGE_RE = re.compile(r"^field_coverage:([a-z0-9_]+):", re.IGNORECASE)
_TOTAL_MISMATCH_RE = re.compile(r"^expected_total_mismatch:(\d+)/(\d+)")

NOTES_LIMIT = 1_800


def _clip(text: str, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def classify_causes(qa_issues: list[Any], simulation: dict[str, Any]) -> list[str]:
    """Deterministic QA codes → typed causes, most decisive first."""

    causes: list[str] = []

    def add(cause: str) -> None:
        if cause not in causes:
            causes.append(cause)

    for raw in qa_issues or []:
        issue = str(raw)
        if issue.startswith(("incomplete_snapshot", "expected_total_mismatch", "expected_total_unknown")):
            add("incomplete_coverage")
        elif issue.startswith(
            ("full_resolution_photo_coverage", "multi_photo_vehicle_coverage", "photo_count_")
        ):
            add("photo_coverage")
        elif issue.startswith("cross_vehicle_photo_duplicates"):
            add("photo_ownership")
        else:
            match = _FIELD_COVERAGE_RE.match(issue)
            if match:
                field = match.group(1).lower()
                if field in {"photo", "photos"}:
                    add("photo_coverage")
                else:
                    add(f"missing_field:{field}")
    agreement = str((simulation or {}).get("vin_agreement") or "")
    if "/" in agreement:
        known, total = agreement.split("/", 1)
        try:
            if int(total) > 0 and int(known) < int(total):
                add("sim_disagreement")
        except ValueError:
            pass

    def rank(cause: str) -> int:
        base = cause.split(":", 1)[0]
        try:
            return _CAUSE_ORDER.index(base)
        except ValueError:
            return len(_CAUSE_ORDER)

    return sorted(causes, key=rank)


def build_repair_plan(
    *,
    qa_issues: list[Any],
    luna_verdict: dict[str, Any],
    simulation: dict[str, Any],
) -> dict[str, Any] | None:
    """A typed plan for the next attempt, or None when there is nothing typed
    to act on (a clean-QA needs_repair is a judgement call, not a bug list)."""

    causes = classify_causes(qa_issues, simulation)
    if not causes:
        return None

    parts: list[str] = []
    codes = ", ".join(_clip(str(issue), 90) for issue in (qa_issues or [])[:8])
    if codes:
        parts.append(f"Deterministic QA codes from the failed crawl: {codes}.")
    for raw in qa_issues or []:
        match = _TOTAL_MISMATCH_RE.match(str(raw))
        if match:
            parts.append(
                f"The crawl stopped at {match.group(1)} vehicles although the site "
                f"declares about {match.group(2)}; the listing very likely paginates "
                "deeper than the walk went, so prove real pagination before "
                "accepting an early natural end."
            )
            break
    concerns = (luna_verdict or {}).get("concerns") or []
    for concern in concerns[:4]:
        parts.append(_clip(str(concern), 280))
    if not parts:
        # A plan must never carry empty notes: an informed run is recognized
        # by its notes, and empty notes would disarm both the attempt counter
        # and the escalation while still blocking crawl reuse — an infinite
        # recrawl loop. The causes themselves are always a usable diagnosis.
        parts.append(
            "The prior crawl failed these typed checks: " + ", ".join(causes) + "."
        )

    notes = _clip(" ".join(parts), NOTES_LIMIT)
    return {
        "causes": causes,
        "primary_cause": causes[0],
        "notes": notes,
        "summary": _clip(str((luna_verdict or {}).get("summary") or ""), 400),
    }
