"""Case-owned engine ledger projection; input validation performs no I/O."""

def engine_metric_snapshot(monitor, fields, timeout=5):
    """Strict field projection from the declared, TSDB-owned raw metric IDs."""
    import math
    bindings = {name: spec["metric"] for name, spec in fields.items()}
    rows = monitor.client.metric_snapshot(bindings.values(), source="mock", timeout=timeout,
                                          labels={"role": "prefill"})
    reverse = {identity: field for field, identity in bindings.items()}
    engines = {}
    for observation in rows:
        labels = observation["metric"]
        if labels.get("role") != "prefill":
            continue
        identity = labels.get("engine_name")
        if not identity or not labels.get("engine_incarnation"):
            raise ValueError("metric lacks engine identity/incarnation")
        if not labels.get("engine_ip") or not labels.get("grpc_port"):
            raise ValueError("engine gate metric lacks endpoint identity labels")
        value = float(observation["value"][1])
        if not math.isfinite(value):
            raise ValueError("non-finite engine metric")
        row = engines.setdefault(identity, dict(engine_incarnation=labels["engine_incarnation"],
                      grpc_addr=labels["engine_ip"] + ":" + labels["grpc_port"]))
        field = reverse[observation["metric_id"]]
        if field in row or row["engine_incarnation"] != labels["engine_incarnation"]:
            raise ValueError("duplicate engine metric or incarnation")
        row[field] = value
    if not engines or any(set(fields) - set(row) for row in engines.values()):
        raise ValueError("incomplete engine monitoring contract")
    return engines


def measurement_samples(monitor, evidence, deadline, limits, *, directory=None, windows=None):
    """Project a public frozen window into the existing cache analyzer vocabulary."""
    from monitoring.windows import capture_window
    from runtime.observation import SampleBudget
    clock = evidence["clock"]
    origin = clock["origin_epoch_s"]
    if windows is None:
        windows = [(evidence[start], evidence[end]) for start, end in
                   (("baseline_start", "baseline_end"), ("post_start", "post_end"))]
    windows = [(origin+start, origin+end) for start, end in windows]
    frozen = capture_window(monitor.client, evidence["gate_input"]["fields"], source="mock",
        windows=windows, epoch=evidence["provenance"]["env_epoch"], deadline=deadline,
        limits=limits, directory=directory)
    samples = {}
    for field, rows in frozen.field_rows().items():
        for row in rows:
            labels = row["metric"]
            name = labels.get("engine_name")
            if not name or not all(labels.get(k) for k in
                    ("engine_incarnation", "engine_ip", "grpc_port")):
                raise ValueError("raw metric lacks engine identity/incarnation")
            identity = dict(engine_incarnation=labels["engine_incarnation"],
                            grpc_addr=labels["engine_ip"] + ":" + labels["grpc_port"])
            for epoch, value in row["values"]:
                deadline.check()
                sample = samples.setdefault(epoch, dict(
                    t=epoch-origin, elapsed_s=epoch-origin, epoch_s=epoch,
                    monotonic_s=clock["origin_monotonic_s"]+epoch-origin,
                    engines={}, started=0, terminal=0,
                    journal_observed_started=None, journal_observed_terminal=None))
                engine = sample["engines"].setdefault(name, dict(identity))
                if any(engine[k] != v for k, v in identity.items()):
                    raise ValueError("raw measurement has conflicting engine incarnation")
                if field in engine and engine[field] != value:
                    raise ValueError("conflicting duplicate raw measurement")
                engine[field] = value
    budget = SampleBudget(limits)
    result = []
    for _, sample in sorted(samples.items()):
        deadline.check()
        budget.append(sample)
        result.append(sample)
    return result
