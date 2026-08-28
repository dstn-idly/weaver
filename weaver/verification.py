from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from .models import ScrapeSpec, VerificationReport


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == []


def verify(
    rows: list[dict[str, Any]],
    spec: ScrapeSpec,
    attempt: int,
    requested_fields: Iterable[str] | None = None,
) -> VerificationReport:
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
    elif len(rows) < spec.min_rows:
        issues.append(
            f"Only {len(rows)} record(s) matched; this collection requires at least {spec.min_rows}"
        )
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

    requested_source = spec.requested_field_names if requested_fields is None else requested_fields
    requested_names = list(dict.fromkeys(str(name) for name in requested_source if name))
    if rows and requested_names:
        captured = [name for name in requested_names if name in field_names]
        if not captured:
            preview = ", ".join(requested_names[:5])
            suffix = "…" if len(requested_names) > 5 else ""
            issues.append(f"None of the requested fields were captured ({preview}{suffix})")
        else:
            stable = [
                name
                for name in captured
                if sum(not _blank(row.get(name)) for row in rows) / len(rows) >= 0.25
            ]
            if not stable:
                issues.append("Requested fields have no stable coverage across the records")

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
    protected = set(spec.requested_field_names)
    for field in spec.fields:
        coverage = sum(not _blank(row.get(field.name)) for row in rows) / len(rows)
        if coverage >= 0.25 or field.required or field.name in protected:
            kept.append(field)
    return spec.model_copy(update={"fields": kept or spec.fields})
