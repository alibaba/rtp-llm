"""Publish registered producer results with implementation provenance."""

import copy
import json
import math
import hashlib
from pathlib import Path
from importlib.util import find_spec

from case_api.registry import registry
from monitoring.contracts import MetricContractError, validate_observation
from monitoring.producer_contract import output_contract


def publish(store, metric_id, definition, rows, *, producer, evidence):
    """A declared Python producer publishes numeric data with explicit provenance."""

    declared = store.document["definitions"].get(metric_id)
    if declared is None or declared != definition or declared.get("producer") != producer or producer not in registry().producers:
        raise MetricContractError("undeclared producer or definition mismatch: " + metric_id)
    try:
        contract = output_contract(metric_id, definition)
    except MetricContractError:
        raise
    except ValueError as exc:
        raise MetricContractError(str(exc)) from exc
    if any(contract.get(key) != definition.get(key) for key in ('measurement', 'collection', 'implementation')):
        raise MetricContractError('published measurement does not match its implementation: ' + metric_id)
    calculation_module = contract['implementation']['module']
    def module_digest(name):
        try:
            spec = find_spec(name)
            if spec is None or spec.origin is None:
                raise MetricContractError("producer module has no source file: " + name)
            return hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        except MetricContractError:
            raise
        except (ImportError, ValueError, OSError) as exc:
            raise MetricContractError("cannot read producer module: " + name) from exc

    calculation_sha256 = module_digest(calculation_module)
    module = registry().producers[producer].module
    producer_sha256 = module_digest(module)
    result, identities = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise MetricContractError("invalid produced observation")
        epoch = row.get("epoch")
        if (type(epoch) not in (int, str) or not str(epoch).isdecimal()
                or int(epoch) < 1):
            raise MetricContractError("produced series requires a positive environment epoch")
        observation = dict(metric_id=metric_id, epoch=str(epoch), source=row.get("source"),
            labels=row.get("labels"), points=row.get("points"), status="ABSENT",
            provenance=dict(source_type=definition["source_type"], producer=producer,
                            producer_module=module, producer_sha256=producer_sha256,
                            calculation_module=calculation_module, calculation_sha256=calculation_sha256,
                            calculation=copy.deepcopy(definition.get("calculation")),
                            measurement=copy.deepcopy(definition["measurement"]), evidence=evidence))
        validate_observation(observation, metric_id)
        key = (str(row["epoch"]), row["source"], json.dumps(row["labels"], sort_keys=True))
        if key in identities:
            raise MetricContractError("duplicate produced series identity: " + metric_id)
        identities.add(key)
        if set(definition["labels"]) - set(row["labels"]):
            raise MetricContractError("produced metric lacks required labels: " + metric_id)
        points = row["points"]
        if any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in points):
            raise MetricContractError("malformed produced samples: " + metric_id)
        if any(type(t) not in (int, float) or not math.isfinite(t)
               or (v is not None and (type(v) not in (int, float) or not math.isfinite(v)))
               for t, v in points) or any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise MetricContractError("invalid or unordered produced samples: " + metric_id)
        observation["status"] = "PRESENT" if any(v is not None for _, v in points) else "ABSENT"
        result.append(observation)
    store.document["metrics"][metric_id] = result
