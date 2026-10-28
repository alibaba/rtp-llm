"""Compare actual full-prefill replay snapshots to unpadded prefill on CPU.

An exit of zero certifies artifact checks/reference computation, not an implicit
model accuracy tolerance. Diagnostic snapshots are supplied at runtime.
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from rtp_llm.test.perf_test.beam_cache_boundary_reference_test import (
    against_reference,
    load_calls,
    metric,
    require,
    rope_reference,
    sdpa_reference,
)


def deep_tensor(call, name, tokens):
    """Slice token snapshots only; parameters are read separately without slicing."""
    require(name in call["tensors"], f"missing deep snapshot {name}")
    value = call["tensors"][name]
    require(
        value.ndim >= 2 and value.shape[0] == call["tokens"],
        f"invalid physical token dimension for {name}",
    )
    require(0 < tokens <= value.shape[0], f"invalid real token extent for {name}")
    return value[:tokens]


def rms_reference(value, weight, eps):
    require(math.isfinite(eps) and eps > 0, "invalid RMSNorm epsilon")
    require(weight.shape == (value.shape[-1],), "invalid RMSNorm weight shape")
    value = value.double()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * weight.double()
    )


def deep_boundary_order(layer, detailed):
    prefix = f"layer_{layer}_"
    names = ["input"]
    if detailed:
        names += [
            "input_norm_input",
            "input_norm_output",
            "linear_qkv_input",
            "linear_qkv_output",
            "qkv_norm_input",
            "qkv_norm_output",
            "inner_attention_input_0",
            "inner_attention_input_1",
            "inner_attention_input_2",
            "attention_output",
            "linear_attn_o_input",
            "linear_attn_o_output",
            "post_attention_norm_input",
            "post_attention_norm_output",
            "mlp_input",
            "linear_mlp_gate_up_input",
            "linear_mlp_gate_up_output",
            "mlp_activation_input",
            "mlp_activation_output",
            "linear_mlp_down_input",
            "linear_mlp_down_output",
            "mlp_output",
        ]
    return [prefix + name for name in names + ["output"]]


def deep_layer_reference(call, layer, tokens, q_heads, kv_heads, dim, theta):
    """Use each recorded operator input, never an earlier oracle's output."""
    prefix = f"layer_{layer}_"
    tensors, meta = call["tensors"], call["metadata"]

    def value(name):
        return deep_tensor(call, prefix + name, tokens)

    result = {}
    for name in ("qkv", "attn_o", "mlp_gate_up", "mlp_down"):
        stem = prefix + "linear_" + name
        require(meta.get(stem + "_weight_layout") == "out_in", f"bad {stem} layout")
        method = meta.get(stem + "_method")
        require(method in ("forward", "forward_without_bias"), f"unknown {stem} method")
        applied = meta.get(stem + "_bias_applied")
        require(type(applied) is bool, f"missing {stem} bias_applied")
        require(applied == (method == "forward"), f"inconsistent {stem} bias method")
        no_bias = meta.get(stem + "_bias_none")
        require(type(no_bias) is bool, f"missing {stem} bias_none")
        require((stem + "_bias" not in tensors) == no_bias, f"inconsistent {stem} bias")
        weight = tensors[stem + "_weight"]
        x = value("linear_" + name + "_input")
        require(weight.ndim == x.ndim == 2, f"invalid {stem} matrix rank")
        require(x.shape[-1] == weight.shape[1], f"invalid {stem} matrix dimensions")
        reference = x.double() @ weight.double().t()
        if not no_bias:
            bias = tensors[stem + "_bias"]
            require(bias.shape == (weight.shape[0],), f"invalid {stem} bias shape")
            if applied:
                reference = reference + bias.double()
        result["linear_" + name] = against_reference(
            value("linear_" + name + "_output"), reference
        )

    for name in ("input_norm", "post_attention_norm"):
        stem = prefix + name
        result[name] = against_reference(
            value(name + "_output"),
            rms_reference(
                value(name + "_input"), tensors[stem + "_weight"], meta[stem + "_eps"]
            ),
        )

    heads = q_heads + 2 * kv_heads
    for field, expected in (
        ("head_num", q_heads),
        ("kv_head_num", kv_heads),
        ("head_dim", dim),
    ):
        require(
            meta.get(prefix + "qkv_norm_" + field) == expected,
            f"deep QK norm {field} differs",
        )
    projected = value("qkv_norm_input").reshape(tokens, heads, dim)
    q, k, v = projected.split([q_heads, kv_heads, kv_heads], dim=1)
    norm_eps = meta[prefix + "qkv_norm_eps"]
    normalized = value("qkv_norm_output").reshape_as(projected)
    result["qkv_norm"] = against_reference(
        normalized,
        torch.cat(
            [
                rms_reference(q, tensors[prefix + "qkv_norm_q_weight"], norm_eps),
                rms_reference(k, tensors[prefix + "qkv_norm_k_weight"], norm_eps),
                v.double(),
            ],
            dim=1,
        ),
    )
    nq, nk, nv = normalized.split([q_heads, kv_heads, kv_heads], dim=1)
    aq = value("inner_attention_input_0").reshape(tokens, q_heads, dim)
    ak = value("inner_attention_input_1").reshape(tokens, kv_heads, dim)
    av = value("inner_attention_input_2").reshape(tokens, kv_heads, dim)
    result["q_rope"] = against_reference(aq, rope_reference(nq, 0, theta))
    result["k_rope"] = against_reference(ak, rope_reference(nk, 0, theta))
    result["v_unchanged"] = metric(av, nv)
    result["attention"] = against_reference(
        value("attention_output").reshape(tokens, q_heads, dim),
        sdpa_reference(aq, ak, av, 0, dim**-0.5),
    )

    require(
        meta.get(prefix + "mlp_activation_kind") == "silu_and_mul",
        "unsupported deep activation",
    )
    require(
        meta.get(prefix + "mlp_gate_up_order") == ["gate", "up"],
        "unknown gate/up packing",
    )
    packed = value("mlp_activation_input")
    require(packed.shape[-1] % 2 == 0, "invalid gate/up width")
    gate, up = packed.double().chunk(2, dim=-1)
    result["mlp_activation"] = against_reference(
        value("mlp_activation_output"), gate / (1.0 + torch.exp(-gate)) * up
    )
    result["attention_residual"] = against_reference(
        value("post_attention_norm_input"),
        value("input").double() + value("linear_attn_o_output").double(),
    )
    result["mlp_residual"] = against_reference(
        value("output"),
        value("post_attention_norm_input").double() + value("mlp_output").double(),
    )
    return result


def analyze_deep(baseline, graph, tokens, q_heads, kv_heads, dim, theta):
    enabled = [
        call["metadata"].get("deep_boundary", False) for call in (baseline, graph)
    ]
    if not any(enabled):
        return {
            "status": "NOT_AVAILABLE",
            "reason": "legacy snapshots have no deep boundaries",
        }
    require(
        all(flag is True for flag in enabled), "deep snapshots present on only one side"
    )
    baseline_meta, graph_meta = baseline["metadata"], graph["metadata"]
    targets = baseline_meta.get("target_layers")
    count = baseline_meta.get("layer_count")
    require(type(count) is int and count > 0, "invalid deep layer count")
    require(
        isinstance(targets, list)
        and bool(targets)
        and all(type(layer) is int and 0 <= layer < count for layer in targets)
        and len(set(targets)) == len(targets),
        "invalid deep target layers",
    )
    require(graph_meta.get("target_layers") == targets, "deep target layers differ")
    require(graph_meta.get("layer_count") == count, "deep layer counts differ")
    for call in (baseline, graph):
        require(
            call["metadata"].get("deep_required_missing") == [], "incomplete deep dump"
        )

    comparisons, full_layers = {}, {}
    for layer in range(count):
        for name in deep_boundary_order(layer, layer in targets):
            a = deep_tensor(graph, name, tokens)
            b = deep_tensor(baseline, name, tokens)
            if name.endswith("_attention_output"):
                a, b = a.reshape(tokens, -1), b.reshape(tokens, -1)
            require(a.dtype == b.dtype, f"deep boundary dtype differs: {name}")
            comparisons[name] = metric(a, b)
        full_layers[str(layer)] = {
            boundary: comparisons[f"layer_{layer}_{boundary}"]
            for boundary in ("input", "output")
        }
    final_graph = deep_tensor(graph, "final_hidden", tokens)
    final_baseline = deep_tensor(baseline, "final_hidden", tokens)
    require(final_graph.dtype == final_baseline.dtype, "final hidden dtype differs")
    comparisons["final_hidden"] = metric(final_graph, final_baseline)

    # Prove both operators received the same parameters; different activation
    # inputs remain visible and are handled separately by each local oracle.
    for layer in targets:
        prefix = f"layer_{layer}_"
        parameters = [
            prefix + suffix
            for suffix in (
                "input_norm_weight",
                "post_attention_norm_weight",
                "qkv_norm_q_weight",
                "qkv_norm_k_weight",
            )
        ]
        metadata = [
            prefix + name + "_eps"
            for name in ("input_norm", "post_attention_norm", "qkv_norm")
        ]
        for name in ("qkv", "attn_o", "mlp_gate_up", "mlp_down"):
            stem = prefix + "linear_" + name
            parameters.append(stem + "_weight")
            metadata += [stem + "_bias_none", stem + "_weight_layout"]
            if baseline_meta.get(stem + "_bias_none") is False:
                parameters.append(stem + "_bias")
        for name in parameters:
            require(
                name in baseline["tensors"] and name in graph["tensors"],
                f"missing {name}",
            )
            require(
                baseline["tensors"][name].dtype == graph["tensors"][name].dtype
                and torch.equal(baseline["tensors"][name], graph["tensors"][name]),
                f"different parameter {name}",
            )
        for name in metadata:
            require(
                name in baseline_meta and name in graph_meta, f"missing metadata {name}"
            )
            require(
                baseline_meta[name] == graph_meta[name], f"different metadata {name}"
            )

    first = next(
        (key for key, value in comparisons.items() if value["bitwise_mismatch_count"]),
        None,
    )
    first_full = next(
        (
            {"layer": int(layer), "boundary": boundary}
            for layer, values in full_layers.items()
            for boundary, value in values.items()
            if value["bitwise_mismatch_count"]
        ),
        None,
    )
    independent = {
        label: {
            str(layer): deep_layer_reference(
                call, layer, tokens, q_heads, kv_heads, dim, theta
            )
            for layer in targets
        }
        for label, call in (("baseline", baseline), ("graph", graph))
    }
    return {
        "status": "ANALYSIS_COMPLETE",
        "accuracy_acceptance": "NOT_ASSESSED: errors are reported without thresholds",
        "layer_count": count,
        "target_layers": targets,
        "first_observed_difference": first,
        "first_full_layer_difference": first_full,
        "boundary_order": list(comparisons),
        "comparisons": comparisons,
        "full_layer_comparisons": full_layers,
        "independent_cpu_float64": independent,
        "oracle_scope": "Each operator uses its own recorded inputs and parameters, not propagated oracle outputs.",
        "limitations": [
            "Only target_layers contain detailed operator boundaries; "
            "other layers contain full input/output snapshots.",
            "Rope/SDPA describe no-prefix causal GQA and the recorded post-RoPE QKV.",
            "Residual references assume the captured TP1 Qwen3 path with no omitted collective or extra bias.",
            "No LM-head, beam-score, or physical KV-cache readback is included.",
        ],
    }


def analyze(baseline, graph, tokens, q_heads, kv_heads, dim, eps, theta):
    require(
        graph["metadata"].get("actual_graph_replay") is True,
        "not a real graph replay",
    )
    require(graph["metadata"].get("real_tokens") == tokens, "graph length differs")
    require(
        baseline["metadata"].get("input_lengths") == [tokens], "baseline lengths differ"
    )
    require(
        graph["metadata"].get("input_lengths") == [tokens, graph["tokens"] - tokens],
        "graph lengths differ",
    )
    for call in (baseline, graph):
        require(
            not any(call["metadata"].get("prefix_lengths", [])), "unexpected prefix"
        )
        require(
            call["metadata"].get("real_last_index") == tokens - 1,
            "wrong last-token index",
        )
    require(
        graph["metadata"].get("capture_impl_id")
        == graph["metadata"].get("replay_impl_id"),
        "captured/replayed attention instances differ",
    )
    require(
        torch.equal(
            baseline["tensors"]["input_ids"], graph["tensors"]["input_ids"][:tokens]
        ),
        "real token IDs differ",
    )
    for name in (
        "qkv_weight",
        "input_norm_weight",
        "qk_norm_q_weight",
        "qk_norm_k_weight",
    ):
        require(
            torch.equal(baseline["tensors"][name], graph["tensors"][name]),
            f"{name} differs",
        )
    for call in (baseline, graph):
        require(call["metadata"].get("qk_norm_eps") == eps, "QK norm epsilon differs")
        require(
            call["metadata"].get("input_norm_eps") == eps, "input norm epsilon differs"
        )

    def valid(call, name):
        value = call["tensors"][name]
        if (
            not name.endswith("_last")
            and value.ndim
            and value.shape[0] == call["tokens"]
        ):
            return value[:tokens]
        return value

    ordered = [
        "layer_0_input",
        "layer_0_input_norm_output",
        "qkv_input",
        "qkv_output",
        "qkv_norm_output",
        "inner_attention_input_0",
        "inner_attention_input_1",
        "inner_attention_input_2",
        "attention_output",
    ]
    layers = sorted(
        {
            int(key.split("_")[1])
            for key in baseline["tensors"]
            if key.startswith("layer_") and key.endswith("_output_last")
        }
    )
    for layer in layers:
        ordered += [f"layer_{layer}_input_last", f"layer_{layer}_output_last"]
    ordered += ["final_hidden_last"]
    comparisons = {}
    for key in ordered:
        require(
            key in baseline["tensors"] and key in graph["tensors"], f"missing {key}"
        )
        a, b = valid(graph, key), valid(baseline, key)
        if key == "attention_output":
            a, b = a.reshape(tokens, -1), b.reshape(tokens, -1)
        comparisons[key] = metric(a, b)
    first = next(
        (key for key, value in comparisons.items() if value["bitwise_mismatch_count"]),
        None,
    )
    independent = {}
    for label, call in (("baseline", baseline), ("graph", graph)):
        t = call["tensors"]
        require(call["metadata"].get("qkv_bias_none") is True, "QKV bias not verified")
        x = valid(call, "qkv_input")
        linear = x.double() @ t["qkv_weight"].double().t()
        projected = valid(call, "qkv_output").reshape(
            tokens, q_heads + 2 * kv_heads, dim
        )
        normalized = valid(call, "qkv_norm_output").reshape_as(projected)
        q, k, v = projected.split([q_heads, kv_heads, kv_heads], dim=1)

        def rms(value, weight):
            value = value.double()
            return (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
                * weight.double()
            )

        norm_ref = torch.cat(
            [rms(q, t["qk_norm_q_weight"]), rms(k, t["qk_norm_k_weight"]), v.double()],
            dim=1,
        )
        nq, nk, nv = normalized.split([q_heads, kv_heads, kv_heads], dim=1)
        aq, ak, av = [valid(call, f"inner_attention_input_{i}") for i in range(3)]
        aq = aq.reshape(tokens, q_heads, dim)
        ak, av = ak.reshape(tokens, kv_heads, dim), av.reshape(tokens, kv_heads, dim)
        attention_ref = sdpa_reference(aq, ak, av, 0, dim**-0.5)
        independent[label] = {
            "input_norm": against_reference(
                valid(call, "layer_0_input_norm_output"),
                rms(valid(call, "layer_0_input"), t["input_norm_weight"]),
            ),
            "qkv_linear": against_reference(valid(call, "qkv_output"), linear),
            "qk_norm": against_reference(normalized, norm_ref),
            "q_rope": against_reference(aq, rope_reference(nq, 0, theta)),
            "k_rope": against_reference(ak, rope_reference(nk, 0, theta)),
            "v_unchanged": metric(av, nv),
            "attention": against_reference(
                valid(call, "attention_output").reshape(tokens, q_heads, dim),
                attention_ref,
            ),
        }
    report = {
        "status": "ANALYSIS_COMPLETE",
        "accuracy_acceptance": "NOT_ASSESSED: no implicit model accuracy threshold",
        "real_tokens": tokens,
        "graph_capacity": graph["tokens"],
        "first_observed_difference": first,
        "comparisons": comparisons,
        "independent_cpu_float64": independent,
        "calls": {
            label: {"metadata": call["metadata"], "manifest": call["manifest"]}
            for label, call in (("baseline", baseline), ("graph", graph))
        },
        "limitations": [
            "Only layer zero has complete token tensors; later layers use the real last token.",
            "Attention oracle uses recorded actual QKV; no exact-real-arithmetic claim.",
            "No LM-head logits or beam-parent/score snapshots are included.",
        ],
    }
    report["deep_analysis"] = analyze_deep(
        baseline, graph, tokens, q_heads, kv_heads, dim, theta
    )
    if report["deep_analysis"]["status"] == "ANALYSIS_COMPLETE":
        report["legacy_first_observed_difference"] = first
        report["first_observed_difference"] = report["deep_analysis"][
            "first_observed_difference"
        ]
        report["limitations"][0] = (
            "All layers have full input/output snapshots; detailed operator coverage "
            "is limited to deep_analysis.target_layers."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--graph_dir", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--q_heads", type=int, default=16)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--rope_theta", type=float, default=1e6)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument(
        "--result_dir", default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
    )
    args = parser.parse_args()
    require(bool(args.result_dir), "result_dir is required")
    require(args.cpu_threads > 0, "cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    baseline = load_calls(Path(args.baseline_dir))[0]
    graph = load_calls(Path(args.graph_dir))[0]
    report = analyze(
        baseline,
        graph,
        args.tokens,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.eps,
        args.rope_theta,
    )
    report["cpu_threads"] = args.cpu_threads
    destination = Path(args.result_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "full_prefill_reference_results.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "first_observed_difference",
                    "accuracy_acceptance",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
