"""Local view curves select canonical metric IDs and explicit label projections."""


def bindings(presentation, metric_id, labels):
    return [(curve_id, style) for curve_id, style in presentation["charts"]["curves"].items()
            if style["metric_id"] == metric_id
            and all(labels.get(key) == value for key, value in style["labels"].items())]



def metric_classification(identity, definition, presentation):
    """Plotted curves, explicit diagnosis, and registered gate scalars are distinct."""
    plotted = {style["metric_id"] for style in presentation["charts"]["curves"].values()}
    if identity in plotted:
        return "PLOTTED"
    if identity in presentation["metrics"]["diagnostic_only"]:
        return "DIAGNOSTIC_ONLY"
    if definition.get("producer") and definition.get("value_kind") == "scalar":
        from monitoring.producers import PRODUCERS
        if PRODUCERS[definition["producer"]][1] == "gate":
            return "GATE_EVIDENCE"
    return None


def monitoring_audit(store, presentation):
    """Classify frozen definitions even when no sampled series survived.

    Label projections may intentionally hide other roles. Gate-producer scalars
    remain decision evidence; all other unplotted metrics must be diagnostic.
    Automatic up queries are collector integrity evidence rather than declared
    presentation measurements.
    """
    audit = []
    for identity, definition in store.document["definitions"].items():
        if "promql" in definition and identity.split('/')[-1] == 'up':
            continue
        classification = metric_classification(identity, definition, presentation)
        if classification is None and any(identity in plan["definition"].get("demand", {}).get("gate", [])
                for plan in store.document.get("plans", {}).values()):
            classification = "GATE_INPUT"
        if classification is None:
            raise ValueError("unclassified monitoring metric " + identity)
        audit.append(dict(metric_id=identity, classification=classification,
            series_count=len(store.document["metrics"].get(identity, []))))
    return audit
