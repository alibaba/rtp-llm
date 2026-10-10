"""Isolate query GEMMs, RMSNorm and RoPE on qualified held NAS histories.

Native and released queries must first reproduce their saved controls. This
diagnostic evaluates operators on fixed inputs without writing attention caches
or propagating changed queries through experts or the full model.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from rtp_llm.models_py.modules.dsv4._rope_only_triton import rope_only_inplace
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import _prefill_q_rope
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import _dense_down
from rtp_llm.tools.dsv41_attention_history_replay import (
    official_history_control,
    require_exact,
    validate_native_recipe,
)
from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    difference,
    history_identities,
    native_attention,
    official_attention,
)
from rtp_llm.tools.dsv41_reference_probe import model_args

KERNEL_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_bytes(expected, actual, label):
    exact = (
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and torch.equal(
            expected.contiguous().reshape(-1).view(torch.uint8),
            actual.contiguous().reshape(-1).view(torch.uint8),
        )
    )
    if not exact:
        raise AssertionError(f"Query source changed: {label}")
    return {
        "bitwise_equal": True,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
    }


def validate_history(root, repo, history, artifact_path):
    if history.get("prompt_index") not in (0, 1):
        raise ValueError("Query replay requires one released prompt")
    count = 4 if history["prompt_index"] else 3
    if sha256(artifact_path) != history["input_artifact_sha256"]:
        raise ValueError("Query history tensor digest differs")
    for field in ("official_reconstruction", "native_reconstruction"):
        if len(history[field]) != 3 * count or not all(
            row["bitwise_equal"] is True for row in history[field].values()
        ):
            raise ValueError("Query history requires exact FFN-input reconstructions")
    for name, expected in history["official_source_sha256"].items():
        if sha256(root / "inference" / name) != expected:
            raise ValueError(f"Released query source changed: {name}")
    if history["official_source_sha256"]["kernel.py"] != KERNEL_SHA256:
        raise ValueError("Query replay requires the pinned released kernel")
    for name, expected in history["native_source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Frozen query history source changed: {name}")
    loaded = []
    for field in ("reference_trace_sha256", "native_trace_sha256"):
        paths = sorted(history[field])
        if len(paths) != count:
            raise ValueError("Query history trace count differs")
        for name in paths:
            if sha256(Path(name)) != history[field][name]:
                raise ValueError(f"Query history trace changed: {name}")
        loaded.append(
            [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        )
    identities = history_identities(*loaded)
    if identities != history["identities"]:
        raise ValueError("Query history consumed prefixes differ")
    recipe = validate_native_recipe(loaded[1], reference_kernel_sha256=KERNEL_SHA256)
    if recipe != history["native_recipe"] or not (
        recipe.get("engram_projection_reference") is True
        and recipe.get("engram_gate_reference") is True
    ):
        raise ValueError("Query replay requires the qualified combined Engram history")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact["identities"] != identities or artifact["native_recipe"] != recipe:
        raise ValueError("Query artifact identities or recipe differ")
    for field in ("official_inputs", "native_inputs", "official_attention"):
        if len(artifact[field]) != count or any(
            len(step) != 3 for step in artifact[field]
        ):
            raise ValueError("Query history requires matching L0-L2 inputs")
    return artifact


def reference_norm(value, weight, eps):
    value = value.float()
    return (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight
    ).to(torch.bfloat16)


def linear_pair(native, released, value, official, kernel):
    """Evaluate the same BF16 input through both quantizers and output GEMMs."""
    quant, packed = native._quantize_input(value)
    scales = (
        packed.clone(memory_format=torch.contiguous_format)
        .view(torch.uint8)[:, : native.K // 32]
        .contiguous()
    )
    reference_quant, reference_scales = kernel.act_quant(
        value, official.fp8_block_size, official.scale_fmt, official.scale_dtype
    )
    baseline = native.forward_quantized(quant, packed).clone()
    substituted = _dense_down(native._sm120_linear, quant, scales, kernel).clone()
    reference = released(value).clone()
    restored = native.forward_quantized(quant, packed).clone()
    return (
        baseline,
        reference,
        substituted,
        restored,
        {
            "activation_values": difference(quant, reference_quant),
            "activation_scales": difference(
                scales.view(torch.float8_e8m0fnu), reference_scales
            ),
            "output": difference(reference, baseline),
            "released_gemm_native_quant": difference(reference, substituted),
        },
    )


def rotate(native, value, freqs, start):
    result = value.reshape(-1, native.n_heads, native.head_dim).clone()
    if start == 0:
        _prefill_q_rope(result, freqs, native.rope_head_dim)
    elif native.rope_head_dim:
        rope_only_inplace(result[..., -native.rope_head_dim :], freqs)
    return result


def released_rotate(official, native, value, freqs):
    result = value.reshape(1, -1, native.n_heads, native.head_dim).clone()
    official.apply_rotary_emb(result[..., -native.rope_head_dim :], freqs)
    return result[0]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("history_report", type=Path)
    parser.add_argument("history_artifact", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Query replay artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Query replay requires SM120 under the GPU lease")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history = json.loads(cli.history_report.read_text())
    artifact = validate_history(root, repo, history, cli.history_artifact)
    sys.path.insert(0, str(root / "inference"))
    official, converter, kernel = [
        importlib.import_module(name) for name in ("model", "convert", "kernel")
    ]
    for module in (official, converter, kernel):
        if Path(module.__file__).resolve() != root / "inference" / (
            module.__name__ + ".py"
        ):
            raise ValueError("Unexpected released query module")
    args = model_args(official, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    reader, cases, tensors = NASReader(root), [], []
    control_profile = official_history_control(history["native_recipe"])
    counts = {
        "native_query": 0,
        "official_query": 0,
        "linear_restoration": 0,
        "source": 0,
    }
    for layer in range(3):
        native = native_attention(args, config, reader, layer)
        shards = official_attention(official, converter, args, reader, layer)
        weights = (
            native.wq_a.weight,
            native.wq_a._sm120_linear.scales,
            native.wq_b.weight,
            native.wq_b._sm120_linear.scales,
            native.q_norm,
        )
        for step, identity in enumerate(history["identities"]):
            start, rows = identity["start_pos"], len(identity["input_ids"])
            freqs = native.freqs_cis[start : start + rows]
            reference_freqs = shards[0].freqs_cis[start : start + rows]
            require_bytes(reference_freqs, freqs, f"layer{layer}.step{step}.freqs")
            for stream in ("official", "native"):
                value = artifact[f"{stream}_inputs"][step][layer].cuda().contiguous()
                sources = (value, *weights, freqs)
                before = [source.clone() for source in sources]
                label = f"{stream}.step{step}.layer{layer:02d}"
                a, released_a, substitute_a, restored_a, a_metrics = linear_pair(
                    native.wq_a, shards[0].wq_a, value, official, kernel
                )
                qr = native._rmsnorm_weighted(a, native.q_norm)
                released_qr = shards[0].q_norm(released_a)
                b_values = {}
                for name, held_qr in (("native_qr", qr), ("released_qr", released_qr)):
                    b_values[name] = linear_pair(
                        native.wq_b,
                        lambda x: torch.cat(
                            [shard.wq_b(x) for shard in shards], dim=-1
                        ),
                        held_qr,
                        official,
                        kernel,
                    )
                b = b_values["native_qr"][0]
                released_b = b_values["released_qr"][1]
                query = rotate(native, b, freqs, start)
                reference_query = released_rotate(
                    official, native, released_b, reference_freqs
                )
                candidate_qr = reference_norm(substitute_a, native.q_norm, native.eps)
                quant, packed = native.wq_b._quantize_input(candidate_qr)
                scales = (
                    packed.clone(memory_format=torch.contiguous_format)
                    .view(torch.uint8)[:, : native.wq_b.K // 32]
                    .contiguous()
                )
                candidate_b = _dense_down(
                    native.wq_b._sm120_linear, quant, scales, kernel
                )
                candidate = rotate(native, candidate_b, freqs, start)
                candidate_released_rope = released_rotate(
                    official, native, candidate_b, reference_freqs
                )
                baseline_rope = released_rotate(official, native, b, reference_freqs)
                held_reference_rope = rotate(native, released_b, freqs, start)
                saved = (
                    artifact["official_input_history"][control_profile][str(layer)][
                        step
                    ]["q"]
                    if stream == "official"
                    else artifact["native_input_history"][step][layer]["q"]
                )
                require_exact(saved, query, label + ".native_query_control")
                counts["native_query"] += 1
                if stream == "official":
                    require_exact(
                        artifact["official_attention"][step][layer]["q"],
                        reference_query,
                        label + ".official_query_control",
                    )
                    counts["official_query"] += 1
                require_exact(a, restored_a, label + ".a_restoration")
                counts["linear_restoration"] += 1
                for name, values in b_values.items():
                    require_exact(
                        values[0], values[3], label + ".b_restoration." + name
                    )
                    counts["linear_restoration"] += 1
                for i, (expected, actual) in enumerate(zip(before, sources)):
                    require_bytes(expected, actual, label + f".source{i}")
                    counts["source"] += 1
                record = {
                    "stream": stream,
                    "step": step,
                    "layer": layer,
                    "first_projection": a_metrics,
                    "normalization": {
                        "native_projection": difference(
                            shards[0].q_norm(a),
                            native._rmsnorm_weighted(a, native.q_norm),
                        ),
                        "released_projection": difference(
                            released_qr,
                            native._rmsnorm_weighted(released_a, native.q_norm),
                        ),
                        "propagated_qr": difference(released_qr, qr),
                    },
                    "second_projection": {
                        name: values[4] for name, values in b_values.items()
                    },
                    "rotation": {
                        "native_projection": difference(baseline_rope, query),
                        "released_projection": difference(
                            reference_query, held_reference_rope
                        ),
                    },
                    "query": difference(reference_query, query),
                    "released_gemms_norm_native_rope": difference(
                        reference_query, candidate
                    ),
                    "released_gemms_norm_released_rope": difference(
                        reference_query, candidate_released_rope
                    ),
                }
                cases.append(record)
                tensors.append(
                    {
                        "stream": stream,
                        "step": step,
                        "layer": layer,
                        "native_query": query.cpu(),
                        "released_query": reference_query.cpu(),
                        "candidate_native_rope": candidate.cpu(),
                        "candidate_released_rope": candidate_released_rope.cpu(),
                        "native_a": a.cpu(),
                        "released_a": released_a.cpu(),
                        "native_qr": qr.cpu(),
                        "released_qr": released_qr.cpu(),
                        "native_b": b.cpu(),
                        "released_b": released_b.cpu(),
                    }
                )
                print(json.dumps(record, allow_nan=False), flush=True)
        del native, shards
    attention_weights = {
        name: digest
        for name, digest in history["weight_sha256"].items()
        if ".attn." in name
    }
    if reader.hashes != attention_weights or len(attention_weights) != 47:
        raise ValueError("Query replay weights differ from the qualified history")
    with cli.tensor_artifact.open("xb") as stream:
        torch.save(
            {
                "identities": history["identities"],
                "native_recipe": history["native_recipe"],
                "cases": tensors,
            },
            stream,
        )
    source_names = set(history["native_source_sha256"]) | {
        "rtp_llm/tools/dsv41_query_boundary_replay.py",
        "rtp_llm/models_py/modules/dsv4/utils.py",
        "rtp_llm/models_py/modules/dsv4/v41_mxfp8.py",
        "rtp_llm/models_py/modules/dsv4/fp8/decode/compute_qkv.py",
        "rtp_llm/models_py/modules/dsv4/_rope_only_triton.py",
        "rtp_llm/models_py/modules/dsv4/rope.py",
    }
    report = {
        "prompt_index": history["prompt_index"],
        "identities": history["identities"],
        "native_recipe": history["native_recipe"],
        "controls_exact": counts,
        "cases": cases,
        "weight_sha256": reader.hashes,
        "official_source_sha256": history["official_source_sha256"],
        "native_source_sha256": {
            name: sha256(repo / name) for name in sorted(source_names)
        },
        "native_revision": subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={repo}",
                "-C",
                str(repo),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "history_report_sha256": sha256(cli.history_report),
        "history_artifact_sha256": sha256(cli.history_artifact),
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "limitations": [
            "Held query inputs only; no attention/cache/expert propagation.",
            "Released head shards are evaluated serially; no live collective or timing claim.",
            "Native/released discrepancies are measurements, not automatic promotion criteria.",
        ],
    }
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
