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


def measurement_samples(monitor, evidence, deadline, limits):
    """Read actual scrape rows for the frozen baseline and survivor windows.

    Control polls establish phase boundaries, never the completeness of the
    counter ledger. Missing series, resets and incarnation changes stay visible
    to the unchanged gate analyzer; no interpolation or lookback filling.
    """
    import math
    from runtime.observation import SampleBudget
    fields = evidence["gate_input"]["fields"]
    clock = evidence["clock"]
    origin = clock["origin_epoch_s"]
    windows = [(evidence["baseline_start"], evidence["baseline_end"]),
               (evidence["post_start"], evidence["post_end"])]
    samples = {}
    for start, end in windows:
        if end <= start:
            raise ValueError("measurement window is incomplete")
        for field, spec in fields.items():
            chunks = monitor.client.metric_chunks(spec["metric"], source="mock",
                start=origin+start, end=origin+end, deadline=deadline, labels={"role": "prefill"})
            for rows in chunks:
                for row in rows:
                    deadline.check()
                    labels = row["metric"]
                    name = labels.get("engine_name")
                    if not name or not all(labels.get(k) for k in
                            ("engine_incarnation", "engine_ip", "grpc_port")):
                        raise ValueError("raw metric lacks engine identity/incarnation")
                    identity = dict(engine_incarnation=labels["engine_incarnation"],
                                    grpc_addr=labels["engine_ip"] + ":" + labels["grpc_port"])
                    for epoch, value in row["values"]:
                        if not origin+start < epoch <= origin+end:
                            continue
                        value = float(value)
                        if not math.isfinite(epoch) or not math.isfinite(value):
                            raise ValueError("non-finite raw measurement")
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
    if not result:
        raise ValueError("raw measurement samples are missing")
    return result
