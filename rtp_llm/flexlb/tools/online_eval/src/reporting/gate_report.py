"""Publish gate reports after explicit online or archived presentation assembly."""
from reporting import write_bundle
from reporting.run_context import selected_spec, provenance_from
from reporting.timeline import archived


def write_gate_report(directory, result, presentation, spec, *, identity, producer, meta, run):
    spec = selected_spec(spec, run, presentation)
    return write_bundle(directory, 'run', identity, dict(result, run=run), spec,
                        meta=provenance_from(run, meta), producer=producer, role='gate')


def write_archived_gate_report(directory, result, presentation, spec, *, identity, producer, meta):
    spec = archived(spec, directory, presentation, status=verdict_status(result['verdict']))
    return write_bundle(directory, 'run', identity, result, spec,
                        meta=meta, producer=producer, role='gate')


def verdict_status(verdict):
    return {"INVALID": "ERROR", "PASS": "PASS", "FAIL": "FAIL"}[verdict]
