"""Deterministic, vehicle-only extraction core for Weaver.

The package deliberately contains no network client and no model integration.
Callers provide already-captured listing/detail HTML (or a constrained transport
adapter) and receive normalized vehicle records plus an evidence-backed QA
report.  This keeps model output as validated data and makes fixture replay the
same extraction path used at runtime.
"""

from .models import (
    DetailSpec,
    FieldRule,
    ListingSpec,
    SpecError,
    VehicleSpec,
    canonical_spec_json,
    parse_spec,
    spec_sha256,
)
from .qa import QAReport, RunEvidence, verify_records
from .replay import CrawlLimits, FixtureSet, ReplayResult, crawl_with_fetchers, replay_fixtures
from .vdp import PhotoEvidence, VdpResult, extract_vdp
from .transport import PersistentDealerSession, VehicleTransportError, capture_dealer_fixtures, discover_vehicle_evidence, run_vehicle_live
from .adapter import VehicleRunResult, replay_vehicle_run

__all__ = [
    "DetailSpec",
    "FieldRule",
    "CrawlLimits",
    "FixtureSet",
    "ListingSpec",
    "QAReport",
    "ReplayResult",
    "RunEvidence",
    "SpecError",
    "VehicleSpec",
    "canonical_spec_json",
    "crawl_with_fetchers",
    "parse_spec",
    "replay_fixtures",
    "spec_sha256",
    "verify_records",
    "PhotoEvidence",
    "VdpResult",
    "extract_vdp",
    "PersistentDealerSession",
    "VehicleTransportError",
    "capture_dealer_fixtures",
    "discover_vehicle_evidence",
    "run_vehicle_live",
    "VehicleRunResult",
    "replay_vehicle_run",
]
