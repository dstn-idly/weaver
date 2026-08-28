from weaver.vehicle.qa import RunEvidence, verify_records
from weaver.vehicle.vdp import PhotoEvidence


def rows():
    return [
        {"vin": "1HGBH41JXMN109186", "name": "2025 Sedan", "year": 2025, "make": "Honda", "model": "Civic", "price": 25000, "mileage": 10, "distance_unit": "mi", "color_ext": "Blue", "color_int": "Black", "transmission": "Automatic", "drivetrain": "FWD", "features": ["A/C"], "description": "A vehicle", "detail_url": "https://dealer.example/a", "photos": ["https://cdn/a-1.jpg", "https://cdn/a-2.jpg", "https://cdn/a-3.jpg"], "photo": "https://cdn/a-1.jpg"},
        {"vin": "1HGBH41JXMN109187", "name": "2024 SUV", "year": 2024, "make": "Honda", "model": "CR-V", "price": 30000, "mileage": 20, "distance_unit": "mi", "color_ext": "Red", "color_int": "Gray", "transmission": "Automatic", "drivetrain": "AWD", "features": ["A/C"], "description": "Another vehicle", "detail_url": "https://dealer.example/b", "photos": ["https://cdn/b-1.jpg", "https://cdn/b-2.jpg", "https://cdn/b-3.jpg"], "photo": "https://cdn/b-1.jpg"},
    ]


def evidence(value):
    return RunEvidence(
        listing_pages=("https://dealer.example/used",),
        detail_pages=tuple(row["detail_url"] for row in value),
        discovered_detail_urls=tuple(row["detail_url"] for row in value),
        expected_total=len(value),
        stop_reason="natural_end",
        photo_evidence={row["vin"]: tuple(PhotoEvidence(url, "data_full", width=1600, full_resolution_candidate=True) for url in row.get("photos", [])) for row in value},
    )


def test_vehicle_quality_requires_identity_fields_and_multi_photo_galleries() -> None:
    report = verify_records(rows(), evidence(rows()))
    assert report.passed and report.complete_snapshot
    assert report.full_resolution_vehicle_coverage == 1.0
    assert report.multi_photo_vehicle_coverage == 1.0


def test_vehicle_quality_rejects_cross_vehicle_photo_reuse() -> None:
    value = rows()
    value[1]["photos"] = ["https://cdn/a-1.jpg", "https://cdn/b-2.jpg"]
    report = verify_records(value, evidence(value))
    assert not report.passed
    assert report.cross_vehicle_photo_duplicate_count == 1


def test_vehicle_quality_warns_when_bonus_field_is_absent_from_entire_source() -> None:
    value = rows()
    for row in value:
        row.pop("features")

    report = verify_records(value, evidence(value))

    assert report.passed
    assert "source_field_unavailable:features" in report.warnings


def test_vehicle_quality_reports_publishable_and_blocked_record_counts() -> None:
    value = rows()
    value[1].pop("photos")
    value[1].pop("photo")

    report = verify_records(value, evidence(value))

    assert not report.passed
    assert report.publishable_record_count == 1
    assert report.blocked_record_count == 1
    assert report.blocked_record_samples == ("1HGBH41JXMN109187",)


def test_corroborated_photo_exception_passes_within_share_cap() -> None:
    """A page-corroborated photo-less listing is a bounded exception: photo
    gates apply to the photographed rows, identity fields stay universal."""

    value = rows()
    for suffix in ("188", "189"):
        extra = dict(value[0])
        extra.update({
            "vin": f"1HGBH41JXMN109{suffix}",
            "detail_url": f"https://dealer.example/photo-{suffix}",
            "photos": [
                f"https://cdn/{suffix}-1.jpg",
                f"https://cdn/{suffix}-2.jpg",
                f"https://cdn/{suffix}-3.jpg",
            ],
            "photo": f"https://cdn/{suffix}-1.jpg",
        })
        value.append(extra)
    exception = dict(value[0])
    exception.update({
        "vin": "1HGBH41JXMN109190",
        "detail_url": "https://dealer.example/c",
        "photo_exception": "no_photos_published",
    })
    exception.pop("photos")
    exception.pop("photo")
    value.append(exception)

    report = verify_records(value, evidence(value))

    assert report.passed
    assert report.photo_exception_count == 1
    assert report.photo_exception_vins == ("1HGBH41JXMN109190",)
    assert report.publishable_record_count == 4
    assert report.blocked_record_count == 0
    assert report.multi_photo_vehicle_coverage == 1.0
    assert report.photo_count_min == 3
    assert len(report.photo_counts) == 4
    assert report.field_coverage["photo"] == 1.0
    assert report.as_dict()["photo_exception_count"] == 1


def test_photo_exception_flag_cannot_bless_an_uncorroborated_or_dominant_gap() -> None:
    value = rows()
    # Uncorroborated absence (no exception flag) still blocks the run.
    value[1].pop("photos")
    value[1].pop("photo")
    report = verify_records(value, evidence(value))
    assert not report.passed
    assert report.photo_exception_count == 0

    # A flag on a row that HAS photos is ignored.
    value = rows()
    value[1]["photo_exception"] = "no_photos_published"
    report = verify_records(value, evidence(value))
    assert report.passed
    assert report.photo_exception_count == 0

    # Exceptions past the 30% share fail closed.
    value = rows()
    flagged = dict(value[0])
    flagged.update({
        "vin": "1HGBH41JXMN109189",
        "detail_url": "https://dealer.example/d",
        "photo_exception": "no_photos_published",
    })
    flagged.pop("photos")
    flagged.pop("photo")
    value.append(flagged)
    second = dict(flagged)
    second.update({"vin": "1HGBH41JXMN109190", "detail_url": "https://dealer.example/e"})
    value.append(second)
    report = verify_records(value, evidence(value))
    assert not report.passed
    assert any(issue.startswith("photo_exception_share:") for issue in report.issues)


def test_single_photo_exception_passes_with_census_corroboration() -> None:
    value = rows()
    for suffix in ("191", "192"):
        extra = dict(value[0])
        extra.update({
            "vin": f"1HGBH41JXMN109{suffix}",
            "detail_url": f"https://dealer.example/photo-{suffix}",
            "photos": [
                f"https://cdn/{suffix}-1.jpg",
                f"https://cdn/{suffix}-2.jpg",
                f"https://cdn/{suffix}-3.jpg",
            ],
            "photo": f"https://cdn/{suffix}-1.jpg",
        })
        value.append(extra)
    single = dict(value[0])
    single.update({
        "vin": "1HGBH41JXMN109193",
        "detail_url": "https://dealer.example/single",
        "photos": ["https://cdn/single-1.jpg"],
        "photo": "https://cdn/single-1.jpg",
        "photo_exception": "single_photo_published",
    })
    value.append(single)

    report = verify_records(value, evidence(value))

    assert report.passed
    assert report.single_photo_exception_count == 1
    assert report.single_photo_exception_vins == ("1HGBH41JXMN109193",)
    assert report.photo_exception_count == 0
    assert report.multi_photo_vehicle_coverage == 1.0
    assert report.photo_count_min == 3
    assert len(report.photo_counts) == 4

    # The single-photo flag on a multi-photo row is ignored.
    value[-1]["photos"] = ["https://cdn/single-1.jpg", "https://cdn/single-2.jpg"]
    report = verify_records(value, evidence(value))
    assert report.single_photo_exception_count == 0
