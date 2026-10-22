"""Shared report contracts, presentation components and artifact publication.

Domain analyzers own statistics and decisions. Rendering never reads evidence or
changes a decision. report-spec.json alone is sufficient for offline rendering.
"""

from reporting.core import (
    bundle_path,
    discover_reports,
    write_bundle,
    load_analysis,
    read_bundle,
)

from reporting.components import details, table, links, run_meta
from reporting.report_model import compare_controls
from reporting.renderer import render

__all__ = [
    "bundle_path",
    "discover_reports",
    "details",
    "table",
    "links",
    "run_meta",
    "compare_controls",
    "render",
    "write_bundle",
    "load_analysis",
    "read_bundle",
]
