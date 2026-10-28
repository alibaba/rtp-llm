"""Analyze saved cold/warm boundaries with independent CPU float64 references.

This manual artifact test never starts a model or accesses an accelerator. A
successful exit means that artifact checks and reference computations completed;
it does not certify model accuracy or declare observed differences acceptable.
Private tensor dumps and diagnostic hooks are supplied at runtime.
"""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_calls(dump_dir):
    notes = dump_dir / "notes.jsonl"
    calls = []
    names = set()
    for line in notes.read_text().splitlines():
        record = json.loads(line)
        name = record.get("name", "")
        match = re.fullmatch(r"call(\d+)_M(\d+)", name)
        if match is None:
            continue
        require(name not in names, f"duplicate call in notes: {name}")
        names.add(name)
        tensors, manifest, metadata = {}, {}, {}
        for key, value in record["value"].items():
            if not isinstance(value, str) or not value.endswith(".pt"):
                metadata[key] = value
                continue
            path = Path(value)
            # Allow an archived dump directory to move as a whole.
            if not path.is_file():
                path = dump_dir / path.name
            require(path.is_file(), f"missing tensor {name}/{key}: {path}")
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            require(isinstance(tensor, torch.Tensor), f"{name}/{key} is not a tensor")
            require(tensor.device.type == "cpu", f"{name}/{key} is not on CPU")
            # Pages include unused physical tail slots. Check their finiteness
            # only after unpack_pages has restricted them to valid tokens.
            if tensor.is_floating_point() and key != "actual_kv_pages":
                require(bool(torch.isfinite(tensor).all()), f"nonfinite {name}/{key}")
            tensors[key] = tensor.detach().contiguous()
            manifest[key] = {
                "file": path.name,
                "sha256": digest(path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        calls.append(
            {
                "name": name,
                "serial": int(match[1]),
                "tokens": int(match[2]),
                "metadata": metadata,
                "tensors": tensors,
                "manifest": manifest,
            }
        )
    require(bool(calls), "no boundary calls found in notes.jsonl")
    return sorted(calls, key=lambda call: call["serial"])


def metric(actual, reference):
    require(actual.shape == reference.shape, "comparison shapes differ")
    a, b = actual.double(), reference.double()
    require(
        bool(torch.isfinite(a).all() & torch.isfinite(b).all()), "nonfinite comparison"
    )
    difference = a - b
    changed = a != b
    result = {
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(reference.dtype),
        "element_count": actual.numel(),
        "numeric_mismatch_count": int(changed.sum()),
        "max_abs": float(difference.abs().max()),
        "rmse": float(difference.square().mean().sqrt()),
    }
    reference_rms = float(b.square().mean().sqrt())
    result["normalized_rmse"] = (
        result["rmse"] / reference_rms if reference_rms else None
    )
    nonzero = b != 0
    result["max_relative_nonzero"] = (
        float((difference[nonzero].abs() / b[nonzero].abs()).max())
        if bool(nonzero.any())
        else None
    )
    result["nonzero_against_zero_count"] = int((changed & ~nonzero).sum())
    if actual.dtype == reference.dtype:
        width = actual.element_size()
        byte_difference = (
            actual.contiguous().view(torch.uint8).reshape(-1, width)
            != reference.contiguous().view(torch.uint8).reshape(-1, width)
        ).any(dim=1)
        result["bitwise_mismatch_count"] = int(byte_difference.sum())
        changed = byte_difference.reshape(actual.shape)
    if bool(changed.any()):
        flat = int(torch.nonzero(changed.flatten(), as_tuple=False)[0, 0])
        coordinates = []
        for size in reversed(actual.shape):
            flat, offset = divmod(flat, size)
            coordinates.append(offset)
        result["first_mismatch_index"] = list(reversed(coordinates))
    else:
        result["first_mismatch_index"] = None
    if actual.dtype == reference.dtype == torch.bfloat16:

        def ordered_bits(value):
            bits = value.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
            return torch.where(bits & 0x8000 != 0, (~bits) & 0xFFFF, bits | 0x8000)

        ulps = (ordered_bits(actual) - ordered_bits(reference)).abs()
        result["bf16_ulp"] = {
            "max": int(ulps.max()),
            "zero": int((ulps == 0).sum()),
            "one": int((ulps == 1).sum()),
            "two_to_four": int(((ulps >= 2) & (ulps <= 4)).sum()),
            "above_four": int((ulps > 4).sum()),
            "signed_zero_note": "opposite signed zeros are adjacent bit patterns",
        }
    return result


def against_reference(actual, reference):
    return {
        "float64_reference": metric(actual, reference),
        "reference_rounded_to_output_dtype": metric(actual, reference.to(actual.dtype)),
    }


def tensor(call, key, shape=None):
    require(key in call["tensors"], f"missing required boundary {call['name']}/{key}")
    value = call["tensors"][key]
    if shape is not None:
        require(tuple(value.shape) == tuple(shape), f"bad shape {call['name']}/{key}")
    return value


def suffix(value, tokens):
    require(value.ndim > 0 and value.shape[0] >= tokens, "invalid suffix shape")
    return value[-tokens:]


def split_qkv(value, args):
    heads = args.q_heads + 2 * args.kv_heads
    require(value.shape == (value.shape[0], heads * args.head_dim), "bad packed QKV")
    shaped = value.reshape(value.shape[0], heads, args.head_dim)
    return torch.split(shaped, [args.q_heads, args.kv_heads, args.kv_heads], dim=1)


def rope_reference(value, position_start, theta):
    """Base RoPE: first/second-half pairs, with float64 angles and arithmetic.

    The production MhaRotaryEmbeddingOp passes is_neox_style=False. No production
    cos/sin cache or rotary kernel is used here. Rounding to the recorded input
    dtype happens after rotation, modeling the materialized BF16 Q/K boundary.
    """
    length, _, dim = value.shape
    require(dim % 2 == 0, "RoPE head dimension must be even")
    positions = torch.arange(
        position_start, position_start + length, dtype=torch.float64
    )
    frequency = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    angles = positions[:, None] * frequency[None, :]
    cosine, sine = angles.cos()[:, None, :], angles.sin()[:, None, :]
    first, second = value.double().chunk(2, dim=-1)
    return torch.cat(
        [first * cosine - second * sine, second * cosine + first * sine], dim=-1
    )


def sdpa_reference(query, key, value, query_start, scale):
    """Explicit float64 CPU QK, causal softmax and PV; no fused SDPA call."""
    require(key.shape == value.shape, "K/V shapes differ")
    require(query.shape[2] == key.shape[2], "Q/K head dimensions differ")
    require(query.shape[1] % key.shape[1] == 0, "invalid grouped-query ratio")
    require(query_start + query.shape[0] <= key.shape[0], "query exceeds KV extent")
    groups = query.shape[1] // key.shape[1]
    k = key.double().repeat_interleave(groups, dim=1)
    v = value.double().repeat_interleave(groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.double(), k) * scale
    q_positions = torch.arange(query_start, query_start + query.shape[0])
    k_positions = torch.arange(key.shape[0])
    scores.masked_fill_(k_positions[None, :] > q_positions[:, None], -math.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


def unpack_pages(call, total_tokens, args):
    if "actual_kv_pages" not in call["tensors"]:
        return None
    pages = tensor(call, "actual_kv_pages")
    expected_pages = math.ceil(total_tokens / args.page_size)
    require(
        tuple(pages.shape)
        == (expected_pages, 2, args.kv_heads, args.page_size, args.head_dim),
        f"invalid HND pages in {call['name']}",
    )
    metadata = call["metadata"]
    require(
        metadata.get("total_kv_tokens") == total_tokens, "KV token metadata differs"
    )
    require(
        len(metadata.get("page_ids", [])) == expected_pages, "missing page ID mapping"
    )
    # The hook already gathered physical pages in logical sequence order. Do not
    # index the saved tensor with physical IDs again. Ignore invalid page tails.
    keys, values = (
        pages[:, index].permute(0, 2, 1, 3).reshape(-1, args.kv_heads, args.head_dim)
        for index in (0, 1)
    )
    keys, values = keys[:total_tokens], values[:total_tokens]
    require(
        bool(torch.isfinite(keys).all() & torch.isfinite(values).all()),
        f"nonfinite valid logical KV in {call['name']}",
    )
    return keys, values


def boundary_order(args):
    order = [
        "layer_0_input",
        "qkv_input",
        "qkv_output",
        "qkv_norm_output",
        "inner_attention_input_0",
        "attention_output",
        "layer_0_output_last",
    ]
    for layer in range(1, args.layers):
        order.extend([f"layer_{layer}_input_last", f"layer_{layer}_output_last"])
    return order + ["final_hidden_last"]


def compare_boundaries(cold, warm, args):
    comparisons, missing = {}, []
    for name in boundary_order(args):
        if name not in cold["tensors"] or name not in warm["tensors"]:
            missing.append(name)
            continue
        a, b = cold["tensors"][name], warm["tensors"][name]
        if not name.endswith("_last"):
            a = suffix(a, args.warm_tokens)
        if name == "attention_output":
            a, b = a.reshape(args.warm_tokens, -1), b.reshape(args.warm_tokens, -1)
        comparisons[name] = metric(b, a)
    first = next(
        (
            name
            for name, result in comparisons.items()
            if result.get("bitwise_mismatch_count", result["numeric_mismatch_count"])
        ),
        None,
    )
    return {
        "direction": "warm actual vs aligned cold reference",
        "first_observed_difference": first,
        "missing_boundaries": missing,
        "comparisons": comparisons,
    }


def bias_for(call, output_size, args):
    if call["metadata"].get("qkv_bias_none") is True:
        require("qkv_bias" not in call["tensors"], "bias metadata contradicts tensor")
        return None, "recorded qkv_bias_none=true"
    if "qkv_bias" in call["tensors"]:
        return tensor(call, "qkv_bias", (output_size,)).double(), "recorded bias tensor"
    require(
        args.qkv_bias_none,
        "missing bias evidence; use --qkv_bias_none only if verified",
    )
    require(
        call["metadata"].get("qkv_bias_none") is not False, "recorded bias is missing"
    )
    return None, "explicit caller assertion of no bias; not observed in artifact"


def linear_analysis(cold, warm, args):
    result = {}
    for label, call in (("cold", cold), ("warm", warm)):
        x = suffix(tensor(call, "qkv_input"), args.warm_tokens)
        weight = tensor(call, "qkv_weight")
        require(
            weight.ndim == 2 and weight.shape[1] == x.shape[1],
            "invalid F.linear weight",
        )
        bias, evidence = bias_for(call, weight.shape[0], args)
        reference = x.double() @ weight.double().t()
        if bias is not None:
            reference = reference + bias
        actual = suffix(tensor(call, "qkv_output"), args.warm_tokens)
        result[label] = {
            "bias_evidence": evidence,
            "oracle": "CPU float64 suffix input @ recorded F.linear weight.T (+ bias)",
            **against_reference(actual, reference),
        }
    return result


def attention_analysis(cold, warm, args):
    start = args.cold_tokens - args.warm_tokens
    cq, ck, cv = split_qkv(tensor(cold, "qkv_norm_output"), args)
    wq, wk, wv = split_qkv(tensor(warm, "qkv_norm_output"), args)
    cqr = rope_reference(cq, 0, args.rope_theta)
    ckr = rope_reference(ck, 0, args.rope_theta)
    wqr = rope_reference(wq, start, args.rope_theta)
    wkr = rope_reference(wk, start, args.rope_theta)
    actual_cq = tensor(cold, "inner_attention_input_0", cq.shape)
    actual_ck = tensor(cold, "inner_attention_input_1", ck.shape)
    actual_cv = tensor(cold, "inner_attention_input_2", cv.shape)
    actual_wq = tensor(warm, "inner_attention_input_0", wq.shape)
    actual_co = suffix(tensor(cold, "attention_output"), args.warm_tokens).reshape(
        args.warm_tokens, args.q_heads, args.head_dim
    )
    actual_wo = tensor(warm, "attention_output").reshape_as(actual_co)
    scale = args.softmax_scale or args.head_dim**-0.5
    result = {
        "scale": scale,
        "rope": {
            "theta": args.rope_theta,
            "style": "Base/noninterleaved, first and second halves paired",
            "positions": "cold [0,L), warm [L-M,L), inferred from prefix metadata",
            "cold_q": against_reference(actual_cq, cqr),
            "cold_k": against_reference(actual_ck, ckr),
            "warm_q": against_reference(actual_wq, wqr),
            "cold_v_unchanged": metric(actual_cv, cv),
        },
        "references": {},
        "cache": {},
    }

    def add_reference(name, actual, query, key, value, evidence):
        reference = sdpa_reference(query, key, value, start, scale)
        result["references"][name] = {
            "input_evidence": evidence,
            **against_reference(actual, reference),
        }

    add_reference(
        "cold_actual_post_rope_qkv",
        actual_co,
        suffix(actual_cq, args.warm_tokens),
        actual_ck,
        actual_cv,
        "recorded actual Q/K/V directly used by ragged attention",
    )
    add_reference(
        "cold_independent_rope_from_norm",
        actual_co,
        suffix(cqr.to(cq.dtype), args.warm_tokens),
        ckr.to(ck.dtype),
        cv,
        "independent float64 RoPE, rounded to recorded Q/K dtype, then float64 SDPA",
    )
    reconstructed_k = torch.cat([actual_ck[:start], wkr.to(wk.dtype)], dim=0)
    reconstructed_v = torch.cat([actual_cv[:start], wv], dim=0)
    add_reference(
        "warm_reconstructed_kv",
        actual_wo,
        actual_wq,
        reconstructed_k,
        reconstructed_v,
        "conditional: cold actual prefix plus reconstructed warm suffix; not actual cache evidence",
    )
    cold_cache = unpack_pages(cold, args.cold_tokens, args)
    warm_cache = unpack_pages(warm, args.cold_tokens, args)
    if cold_cache is not None:
        result["cache"]["cold_k_vs_actual_ragged_k"] = metric(cold_cache[0], actual_ck)
        result["cache"]["cold_v_vs_actual_ragged_v"] = metric(cold_cache[1], actual_cv)
    if warm_cache is None:
        result["cache"]["status"] = "NOT_VERIFIED: actual warm paged KV not recorded"
    else:
        actual_wk, actual_wv = warm_cache
        result["cache"][
            "status"
        ] = "OBSERVED: actual logical KV after write, before attention"
        result["cache"]["warm_prefix_k_vs_cold_actual_k"] = metric(
            actual_wk[:start], actual_ck[:start]
        )
        result["cache"]["warm_prefix_v_vs_cold_actual_v"] = metric(
            actual_wv[:start], actual_cv[:start]
        )
        result["cache"]["warm_suffix_k_vs_independent_rope"] = against_reference(
            actual_wk[start:], wkr
        )
        result["cache"]["warm_suffix_v_vs_norm_v"] = metric(actual_wv[start:], wv)
        result["cache"]["warm_full_k_vs_reconstruction"] = metric(
            actual_wk, reconstructed_k
        )
        result["cache"]["warm_full_v_vs_reconstruction"] = metric(
            actual_wv, reconstructed_v
        )
        add_reference(
            "warm_actual_paged_qkv",
            actual_wo,
            actual_wq,
            actual_wk,
            actual_wv,
            "recorded actual Q plus logical KV gathered from pages before paged attention",
        )
    if cold_cache is not None and warm_cache is not None:
        for index, name in enumerate(("k", "v")):
            result["cache"][f"warm_vs_cold_actual_prefix_{name}"] = metric(
                warm_cache[index][:start], cold_cache[index][:start]
            )
            result["cache"][f"warm_vs_cold_actual_suffix_{name}"] = metric(
                warm_cache[index][start:], cold_cache[index][start:]
            )
    return result


def analyze(calls, args):
    cold_candidates = [call for call in calls if call["tokens"] == args.cold_tokens]
    require(len(cold_candidates) == 1, "expected exactly one cold call")
    cold = cold_candidates[0]
    warms = [
        call
        for call in calls
        if call["tokens"] == args.warm_tokens and call["serial"] > cold["serial"]
    ]
    require(bool(warms), "expected at least one warm call after cold")
    prefix = args.cold_tokens - args.warm_tokens
    for call in [cold] + warms:
        require(
            tensor(call, "input_ids").numel() == call["tokens"], "bad input ID count"
        )
        require(
            call["metadata"].get("input_lengths", [call["tokens"]]) == [call["tokens"]],
            "single-request input length metadata differs",
        )
        expected_prefix = 0 if call is cold else prefix
        require(
            call["metadata"].get("prefix_lengths", [expected_prefix])
            == [expected_prefix],
            "prefix metadata differs from requested alignment",
        )
    require(
        cold["metadata"].get("attention_class") == "PyFlashinferPrefillImpl",
        "cold call is not the observed ragged attention route",
    )
    report = {
        "status": "ANALYSIS_COMPLETE",
        "validation_scope": "artifact integrity and CPU reference computation only",
        "accuracy_acceptance": "NOT_ASSESSED: no implicit error tolerance or model PASS",
        "alignment": {
            "cold_tokens": args.cold_tokens,
            "warm_tokens": args.warm_tokens,
            "prefix": prefix,
        },
        "parameters": {
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "page_size": args.page_size,
            "layers": args.layers,
            "rope_theta": args.rope_theta,
        },
        "calls": {
            call["name"]: {"metadata": call["metadata"], "artifacts": call["manifest"]}
            for call in [cold] + warms
        },
        "cold_warm": {},
        "warm_repeats": {},
        "limitations": [
            "FP64 CPU formulas are independent references, not exact real arithmetic.",
            "No QK norm gamma snapshot: its first divergence is observed, not independently attributed.",
            "RoPE uses configured theta and source-defined Base layout; actual position tensors/cos cache are not captured.",
            "Layer last-row snapshots cannot exclude an earlier difference in another token/head.",
            "These dumps do not include LM-head logits or canonical beam parents/scores.",
            "KV snapshots are after write; they do not independently show the same warm cache before write.",
        ],
    }
    for warm in warms:
        require(
            warm["metadata"].get("attention_class") == "PyFlashinferPagedPrefillImpl",
            "warm call is not the observed paged attention route",
        )
        require(
            torch.equal(
                suffix(tensor(cold, "input_ids"), args.warm_tokens),
                tensor(warm, "input_ids"),
            ),
            "warm token IDs do not match cold suffix",
        )
        require(
            torch.equal(tensor(cold, "qkv_weight"), tensor(warm, "qkv_weight")),
            "QKV weights changed",
        )
        comparisons = compare_boundaries(cold, warm, args)
        comparisons["linear_reference"] = linear_analysis(cold, warm, args)
        comparisons["attention_reference"] = attention_analysis(cold, warm, args)
        report["cold_warm"][warm["name"]] = comparisons
    for warm in warms[1:]:
        comparisons = compare_boundaries(warms[0], warm, args)
        comparisons["direction"] = "later warm actual vs first warm reference"
        first_cache = unpack_pages(warms[0], args.cold_tokens, args)
        repeated_cache = unpack_pages(warm, args.cold_tokens, args)
        if first_cache is not None and repeated_cache is not None:
            comparisons["logical_cache"] = {
                name: metric(repeated_cache[index], first_cache[index])
                for index, name in enumerate(("key", "value"))
            }
        report["warm_repeats"][warm["name"]] = comparisons
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument(
        "--result_dir", default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
    )
    parser.add_argument("--cold_tokens", type=int, default=720)
    parser.add_argument("--warm_tokens", type=int, default=80)
    parser.add_argument("--q_heads", type=int, default=16)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--rope_theta", type=float, default=1e6)
    parser.add_argument("--softmax_scale", type=float, default=None)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--qkv_bias_none", action="store_true")
    args = parser.parse_args()
    if not args.result_dir:
        parser.error("result_dir or TEST_UNDECLARED_OUTPUTS_DIR is required")
    if not 0 < args.warm_tokens < args.cold_tokens:
        parser.error("require 0 < warm_tokens < cold_tokens")
    if (
        min(
            args.q_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            args.layers,
            args.cpu_threads,
        )
        < 1
    ):
        parser.error("dimensions, layers and CPU threads must be positive")
    if not math.isfinite(args.rope_theta) or args.rope_theta <= 0:
        parser.error("rope_theta must be finite and positive")
    if args.softmax_scale is not None and (
        not math.isfinite(args.softmax_scale) or args.softmax_scale <= 0
    ):
        parser.error("softmax_scale must be finite and positive")
    os.umask(0o077)
    output_dir = Path(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "beam_cache_boundary_reference.json"
    torch.set_num_threads(args.cpu_threads)
    try:
        with torch.inference_mode():
            report = analyze(load_calls(Path(args.dump_dir)), args)
        report["notes_sha256"] = digest(Path(args.dump_dir) / "notes.jsonl")
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        output.write_text(
            json.dumps({"status": "INCOMPLETE", "error": str(error)}, indent=2) + "\n"
        )
        raise
    summary = {
        "status": report["status"],
        "accuracy_acceptance": report["accuracy_acceptance"],
        "first_observed_differences": {
            name: value["first_observed_difference"]
            for name, value in report["cold_warm"].items()
        },
        "result": str(output),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
