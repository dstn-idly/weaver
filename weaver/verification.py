from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .models import ScrapeSpec, VerificationReport


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == []


def verify(rows: list[dict[str, Any]], spec: ScrapeSpec, attempt: int) -> VerificationReport:
    issues: list[str] = []
    fields = spec.all_fields()
    field_names = [field.name for field in fields]
    cell_count = max(1, len(rows) * max(1, len(field_names)))
    blank_count = sum(_blank(row.get(name)) for row in rows for name in field_names)
    null_rate = blank_count / cell_count

    fingerprints = [json.dumps(row, sort_keys=True, default=str, ensure_ascii=False) for row in rows]
    duplicate_count = sum(count - 1 for count in Counter(fingerprints).values() if count > 1)
    duplicate_rate = duplicate_count / max(1, len(rows))

    if not rows:
        issues.append("No records matched the proposed schema")
    if not field_names:
        issues.append("No stable data fields were discovered")
    useful_fields = [
        field
        for field in fields
        if field.type not in {"url", "image"} and field.name.lower() not in {"url", "link", "image", "images", "icon"}
    ]
    if field_names and not useful_fields:
        issues.append("The schema contains only links or images, not a usable dataset")
    if rows and null_rate > 0.70:
        issues.append(f"Field coverage is too sparse ({null_rate:.0%} null)")
    if len(rows) > 2 and duplicate_rate > 0.80:
        issues.append(f"Most records are duplicates ({duplicate_rate:.0%})")
    for field in fields:
        if not field.required or not rows:
            continue
        missing = sum(_blank(row.get(field.name)) for row in rows) / len(rows)
        if missing > 0.20:
            issues.append(f"Required field '{field.name}' is missing in {missing:.0%} of records")

    primary_names = {"title", "headline", "name", "quote", "item", "property", "instrument"}
    primary = next((field for field in fields if field.name.lower() in primary_names), None)
    if primary and rows:
        values = [str(row.get(primary.name, "")) for row in rows if not _blank(row.get(primary.name))]
        if values and sum(any(character.isalpha() for character in value) for value in values) / len(values) < 0.5:
            issues.append(f"Primary field '{primary.name}' does not look descriptive")

    return VerificationReport(
        attempt=attempt,
        passed=not issues,
        row_count=len(rows),
        field_count=len(field_names),
        null_rate=round(null_rate, 4),
        duplicate_rate=round(duplicate_rate, 4),
        issues=issues,
    )


def repair_spec(spec: ScrapeSpec, rows: list[dict[str, Any]]) -> ScrapeSpec:
    """Deterministically remove selectors that resolve to no data before retrying."""
    if not rows:
        return spec
    kept = []
    for field in spec.fields:
        coverage = sum(not _blank(row.get(field.name)) for row in rows) / len(rows)
        if coverage >= 0.25 or field.required:
            kept.append(field)
    return spec.model_copy(update={"fields": kept or spec.fields})
