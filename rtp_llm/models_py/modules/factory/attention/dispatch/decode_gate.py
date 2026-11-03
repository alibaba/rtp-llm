"""Validate Hongmin forward records against an explicitly observed run schedule."""

from collections import defaultdict

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    evaluate_precision,
)


def build_decode_gate(records, expected, layer_ids, fp8, config=None):
    """Return passed names and diagnostics; missing coverage is never a pass.

    expected maps (scenario, phase) to (active sequence ids, candidate names).
    It is captured at the recorder's WORLD-agreed list_candidates boundary.
    """
    if not expected or not layer_ids:
        raise ValueError("decode gate has no scenarios or layers")
    indexed = {}
    for scenario, bucket in records.items():
        for impl, forwards in bucket.items():
            for forward in forwards:
                if not forward.phase.startswith("decode_"):
                    continue
                key = (scenario, forward.phase)
                if key not in expected or forward.impl_name != impl:
                    raise ValueError(f"unexpected decode record {key}/{impl}")
                if impl != "golden" and impl not in expected[key][1]:
                    raise ValueError(f"unexpected decode candidate {key}/{impl}")
                index = (*key, impl)
                if index in indexed:
                    raise ValueError(f"duplicate decode record {index}")
                indexed[index] = forward

    details = defaultdict(list)
    names = set.intersection(*(set(candidates) for _, candidates in expected.values()))
    passed = set(names)
    layers = set(layer_ids)
    for (scenario, phase), (active_ids, candidates) in expected.items():
        golden = indexed.get((scenario, phase, "golden"))
        if golden is None or set(golden.layer_records) != layers:
            raise ValueError(f"incomplete golden {scenario}/{phase}")
        if not active_ids or len(set(active_ids)) != len(active_ids):
            raise ValueError("invalid active sequence ids")
        for name in candidates:
            candidate = indexed.get((scenario, phase, name))
            if candidate is None or set(candidate.layer_records) != layers:
                raise ValueError(f"incomplete candidate {scenario}/{phase}/{name}")
            for layer_id in layer_ids:
                gold = golden.layer_records[layer_id]
                actual = candidate.layer_records[layer_id]
                if gold.layer_idx != layer_id or actual.layer_idx != layer_id:
                    raise ValueError("layer index mismatch")
                # Golden feedback must preserve identical inputs to each layer.
                misaligned = None
                for field in ("q", "k", "v"):
                    left, right = getattr(gold, field), getattr(actual, field)
                    if left is None or right is None:
                        raise ValueError(
                            f"missing {field} input: {scenario}/{phase}/{name}/{layer_id}"
                        )
                    if left.shape != right.shape or left.dtype != right.dtype:
                        raise ValueError(
                            f"invalid {field} input structure: "
                            f"{scenario}/{phase}/{name}/{layer_id}"
                        )
                    if not torch.equal(left, right):
                        misaligned = field
                        break
                if misaligned is not None:
                    details[name].append(
                        (
                            scenario,
                            phase,
                            layer_id,
                            {
                                "overall_pass": False,
                                "reason": f"unaligned {misaligned} input",
                            },
                        )
                    )
                    passed.discard(name)
                    break
                shape = (len(active_ids), gold.q.shape[1] * gold.q.shape[2])
                if (
                    tuple(gold.output.shape) != shape
                    or tuple(actual.output.shape) != shape
                ):
                    raise ValueError(
                        "decode output does not match active batch/head dimensions"
                    )
                if gold.reference_contribution is None:
                    raise ValueError(
                        f"missing reference contribution: {scenario}/{phase}/{layer_id}"
                    )
                verdict = evaluate_precision(
                    actual.output,
                    gold.output,
                    gold.reference_contribution,
                    fp8,
                    config,
                )
                details[name].append((scenario, phase, layer_id, verdict))
                if not verdict["overall_pass"]:
                    passed.discard(name)
    return frozenset(passed), dict(details)
