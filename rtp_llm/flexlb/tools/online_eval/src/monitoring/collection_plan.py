"""Physical dependencies and per-source scrape policy, without case branches."""

import re
from urllib.parse import urlsplit, parse_qs

from monitoring.query_plan import SOURCE_KINDS


def source_kind(name):
    return "mock" if name == "mock" else "client" if name.startswith("client-") else "master"


def physical_metrics(plan, kind):
    """Selected queries own exact physical dependencies; never guess from PromQL."""
    if kind not in SOURCE_KINDS:
        raise ValueError("unknown collection source kind: " + kind)
    return sorted({name for spec in plan["sources"][kind].values()
                   for name in spec["exported_metrics"]})


def collection_plan(plan):
    from monitoring.sources import selected_sources

    return dict(
        prometheus={kind: dict(metric_ids=[kind + "/" + name for name in queries],
                               exported_metrics=physical_metrics(plan, kind))
                    for kind, queries in plan["sources"].items() if queries},
        evidence=selected_sources(plan),
    )


def select_plan(plan, requirements, presentations):
    """Freeze the union of explicit gate, chart and diagnostic consumers.

    A registry is a catalog of capabilities, not an instruction to collect all
    of them. Default views display the selected archive; they add no demand.
    Producer input dependencies must be declared by the program via metric().
    """
    import copy
    from monitoring.query_plan import definitions

    known = definitions(plan)
    consumers = {"gate": sorted(requirements), "report": [], "diagnostic": []}
    for view in presentations:
        consumers["report"].extend(style["metric_id"] for style in
            view.get("charts", {}).get("curves", {}).values())
        consumers["diagnostic"].extend(view.get("metrics", {}).get("diagnostic_only", []))
    consumers = {key: sorted(set(values)) for key, values in consumers.items()}
    demanded = set().union(*map(set, consumers.values()))
    if demanded - set(known):
        raise ValueError("undeclared collection demand: " + ", ".join(sorted(demanded - set(known))))
    selected = copy.deepcopy(plan)
    selected["sources"] = {kind: {name: spec for name, spec in queries.items()
                                 if kind + "/" + name in demanded}
                           for kind, queries in selected["sources"].items()}
    selected["produced"] = {identity: spec for identity, spec in selected["produced"].items()
                            if identity in demanded}
    selected["demand"] = consumers
    return selected


def frozen_plan(name, plan):
    from monitoring.query_plan import plan_hash
    return dict(name=name, sha256=plan_hash(plan), definition=plan, collection=collection_plan(plan))


def instance_plan(instance):
    """Execution uses the compiled selection, never reloads a larger catalog."""
    from monitoring.query_plan import load_plan, DEFAULT_PLAN
    frozen = instance.get("implementation", {}).get("monitoring_query_plan")
    if frozen is not None:
        from monitoring.query_plan import plan_hash
        if frozen["sha256"] != plan_hash(frozen["definition"]):
            raise ValueError("frozen monitoring plan hash mismatch")
        if frozen["collection"] != collection_plan(frozen["definition"]):
            raise ValueError("frozen collection implementation mismatch")
        return frozen["definition"]
    if instance.get("implementation", {}).get("program") is not None:
        raise ValueError("compiled workload is missing its frozen monitoring plan")
    # Low-level runtime plans may not be built by a case program.
    return load_plan(instance["execution"]["monitoring"].get("query_plan", DEFAULT_PLAN))


def scrape_job(plan, name, url, kind):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment):
        raise ValueError("monitor target must be an explicit http endpoint")
    names = physical_metrics(plan, kind)
    if not names:
        return None
    return dict(job_name=name, sample_limit=100000, body_size_limit="32MB",
                metrics_path=parsed.path or "/metrics", params=parse_qs(parsed.query),
                static_configs=[dict(targets=[parsed.netloc])],
                metric_relabel_configs=[dict(source_labels=["__name__"], action="keep",
                    regex="(" + "|".join(re.escape(metric) for metric in names) + ")")])
