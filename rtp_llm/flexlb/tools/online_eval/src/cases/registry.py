"""Registered Python case programs. YAML can only select entries in this registry."""

PROGRAMS = {
    "cache_scale_in": "cases.cache_scale_in.program",
    "master_ha_failover": "cases.master_ha_failover.program",
    "master_performance": "cases.master_performance.program",
    "request_completion": "cases.request_completion.program",
}


def finalize_reports(program, directory):
    """Call a declared post-archive presentation hook without changing the verdict."""
    from importlib import import_module

    module = import_module(PROGRAMS[program])
    finalizer = getattr(module, "REPORT_FINALIZER", None)
    if finalizer is not None:
        if not callable(finalizer):
            raise ValueError("REPORT_FINALIZER must be callable")
        finalizer(directory)


VIEW_KINDS = {
    "ha": {
        "validator": "cases.master_ha_failover.view.validate",
        "renderer": "cases.master_ha_failover.report.write_report",
    },
}


def load_capability(path):
    """Resolve only a path supplied by the trusted capability registry."""
    from importlib import import_module

    module, _, name = path.rpartition(".")
    return getattr(import_module(module), name)
