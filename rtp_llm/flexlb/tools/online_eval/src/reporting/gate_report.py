"""Shared gate report publication; case callbacks supply business presentation."""

from reporting import write_bundle
from reporting.run_context import selected_spec, provenance_from
from reporting.timeline import archived


def write_gate_report(directory, result, presentation, spec, *,
                      identity, producer, meta, run=None):
    """Preserve frozen gate decisions while applying the shared run/timeline shell."""
    if run is not None:
        spec = selected_spec(spec, run, presentation)
        meta = provenance_from(run, meta)
        result = dict(result, run=run)
    else:
        spec = archived(spec, directory, presentation, status=verdict_status(result['verdict']))
    return write_bundle(directory, 'run', identity, result, spec,
                        meta=meta, producer=producer, role='gate')


def verdict_status(verdict):
    return {"INVALID": "ERROR", "PASS": "PASS", "FAIL": "FAIL"}[verdict]
