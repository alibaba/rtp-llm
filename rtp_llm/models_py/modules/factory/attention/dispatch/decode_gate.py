"""Validate Hongmin forward records against an explicitly observed run schedule."""

from collections import defaultdict

import torch

from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    evaluate_precision,
)


def build_decode_gate(records, expected, layer_ids, fp8):
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
                for field in ("q", "k", "v"):
                    left, right = getattr(gold, field), getattr(actual, field)
                    if (
                        left.shape != right.shape
                        or left.dtype != right.dtype
                        or not torch.equal(left, right)
                    ):
                        raise ValueError(
                            f"unaligned {field}: {scenario}/{phase}/{name}/{layer_id}"
                        )
                shape = (len(active_ids), gold.q.shape[1] * gold.q.shape[2])
                if (
                    tuple(gold.output.shape) != shape
                    or tuple(actual.output.shape) != shape
                ):
                    raise ValueError(
                        "decode output does not match active batch/head dimensions"
                    )
                verdict = evaluate_precision(actual.output, gold.output, fp8)
                details[name].append((scenario, phase, layer_id, verdict))
                if not verdict["overall_pass"]:
                    passed.discard(name)
    return frozenset(passed), dict(details)
