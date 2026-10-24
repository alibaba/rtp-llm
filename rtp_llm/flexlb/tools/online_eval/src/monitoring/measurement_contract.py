"""Resolve registered metric contracts without loading stores or running producers."""

from monitoring.contracts import validate_measurement


def output_contract(metric_id, spec, contracts):
    """Resolve executable capabilities, not user-provided measurement labels."""
    producer = spec["producer"]
    if producer not in contracts:
        raise ValueError("unknown metric producer " + str(producer))
    describe = contracts[producer]
    contract = describe(producer, metric_id, spec.get("calculation"))
    for key, value in contract.items():
        if key not in {"measurement", "collection", "implementation"} and spec.get(key) != value:
            raise ValueError(metric_id + ": producer " + key + " mismatch")
    validate_measurement(contract["measurement"], metric_id)
    implementation = contract.get("implementation")
    if (
        not isinstance(implementation, dict)
        or set(implementation) != {"module", "function"}
        or any(type(value) is not str or not value for value in implementation.values())
    ):
        raise ValueError(metric_id + ": invalid implementation identity")
    return contract
