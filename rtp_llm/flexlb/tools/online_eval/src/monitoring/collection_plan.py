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
