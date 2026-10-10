"""Isolate output projection arithmetic on qualified held attention outputs.

Native output controls must reproduce the saved query-owner captures. Released
projection operators consume the same held attention; no attention cache,
experts or model generation executes in this diagnostic.
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4 import rope
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import _dense_down
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import (
    tp8_output_projection,
)
from rtp_llm.tools.dsv41_attention_history_replay import require_exact
from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    difference,
    native_attention,
    official_attention,
)
from rtp_llm.tools.dsv41_query_boundary_replay import (
    quantized_difference,
    require_bytes,
    sha256,
    validate_history,
)
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest, validate_queries
from rtp_llm.tools.dsv41_reference_probe import model_args


def admit_owner(cli, root, repo):
    history = json.loads(cli.history_report.read_text())
    inputs = validate_history(root, repo, history, cli.history_artifact)
    query = json.loads(cli.query_report.read_text())
    frequencies = torch.load(cli.query_artifact, map_location="cpu", weights_only=True)
    validate_queries(
        history,
        cli.history_report,
        cli.history_artifact,
        query,
        cli.query_artifact,
        frequencies,
    )
    report = json.loads(cli.owner_report.read_text())
    owner = torch.load(cli.owner_artifact, map_location="cpu", weights_only=True)
    for field in ("identities", "native_recipe"):
        if report[field] != history[field] or owner[field] != history[field]:
            raise ValueError("Projection replay requires the qualified owner history")
    for field in (
        "history_report",
        "history_artifact",
        "query_report",
        "query_artifact",
    ):
        if report[field + "_sha256"] != sha256(getattr(cli, field)):
            raise ValueError(f"Projection owner evidence changed: {field}")
    if report["tensor_artifact_sha256"] != sha256(cli.owner_artifact):
        raise ValueError("Projection owner tensor digest differs")
    for name, expected in report["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Qualified projection owner source changed: {name}")
    if report["weight_sha256"] != query["weight_sha256"]:
        raise ValueError("Projection owner weights differ from its query evidence")
    count = len(history["identities"])
    for stream in ("official", "native"):
        for layer in range(3):
            rows = owner["results"][stream][str(layer)]["query_sparse_decode"]
            if len(rows) != count:
                raise ValueError("Projection owner cases are incomplete")
    tables = {item["layer"]: item for item in frequencies["frequency_tables"]}
    if set(tables) != {0, 1, 2} or len(frequencies["frequency_tables"]) != 3:
        raise ValueError("Projection replay requires all three frequency controls")
    return history, inputs, report, owner, tables


def rotate(owner, attention, freqs, function):
    result = attention.reshape(1, -1, owner.n_heads, owner.head_dim).clone()
    function(result[..., -owner.rope_head_dim :], freqs, inverse=True)
    return result[0]


def reduce_partials(partials):
    return torch.stack([value.float() for value in partials]).sum(0).bfloat16()


def quantize(native, value):
    values, packed = native._quantize_input(value)
    scales = (
        packed.clone(memory_format=torch.contiguous_format)
        .view(torch.uint8)[:, : native.K // 32]
        .contiguous()
    )
    return values, packed, scales


def immutable_sources(native, shards):
    sources = {
        "native.wo_a.weight": native.wo_a_w,
        "native.wo_a.scale": native.wo_a_s,
        "native.wo_a.bf16": native._wo_a_bf16_v41,
        "native.wo_b.weight": native.wo_b.weight,
        "native.wo_b.scale": native.wo_b._sm120_linear.scales,
        "native.freqs": native.freqs_cis,
    }
    for index, (linear, shard) in enumerate(
        zip(native._diagnostic_wo_b_shards, shards)
    ):
        for name, value in (
            ("native.wo_b.weight", linear.weight),
            ("native.wo_b.scale", linear._sm120_linear.scales),
            ("released.wo_a.weight", shard.wo_a.weight),
            ("released.wo_b.weight", shard.wo_b.weight),
            ("released.wo_b.scale", shard.wo_b.scale),
            ("released.freqs", shard.freqs_cis),
        ):
            sources[f"shard{index}.{name}"] = value
    return sources


def execute_case(native, shards, official, kernel, attention, freqs, reference_freqs):
    native_rotation = rotate(native, attention, freqs, rope.apply_rotary_emb)
    released_native_freq_rotation = rotate(
        native, attention, freqs, official.apply_rotary_emb
    )
    released_rotation = rotate(
        native, attention, reference_freqs, official.apply_rotary_emb
    )
    grouped = native_rotation.reshape(len(native_rotation), native.n_groups, -1)
    released_grouped = released_rotation.reshape_as(grouped)
    raw, shard_metrics = [], []
    for index, (value, weight, linear, shard) in enumerate(
        zip(
            grouped.chunk(8, dim=1),
            native._wo_a_bf16_v41.chunk(8, dim=0),
            native._diagnostic_wo_b_shards,
            shards,
        )
    ):
        ref_weight = shard.wo_a.weight.view(1, native.o_lora_rank, -1)
        require_bytes(weight, ref_weight, f"shard{index}.wo_a")
        require_bytes(linear.weight, shard.wo_b.weight, f"shard{index}.wo_b")
        require_bytes(
            linear._sm120_linear.scales,
            shard.wo_b.scale.view(torch.uint8),
            f"shard{index}.wo_b.scale",
        )
        projected = torch.einsum("tgd,grd->tgr", value, weight).flatten(1)
        reference_on_native_rotation = torch.einsum(
            "bsgd,grd->bsgr", value.unsqueeze(0), ref_weight
        ).flatten(2)[0]
        reference_projected = torch.einsum(
            "bsgd,grd->bsgr",
            released_grouped[:, index : index + 1].unsqueeze(0),
            ref_weight,
        ).flatten(2)[0]
        quant, packed, scales = quantize(linear, projected)
        ref_quant, ref_scales = kernel.act_quant(
            projected,
            official.fp8_block_size,
            official.scale_fmt,
            official.scale_dtype,
        )
        baseline = linear.forward_quantized(quant, packed)
        substituted = _dense_down(linear._sm120_linear, quant, scales, kernel)
        reference_native_wo_a = official.linear(projected, shard.wo_b.weight)
        reference_native_rotation = official.linear(
            reference_on_native_rotation, shard.wo_b.weight
        )
        reference = official.linear(reference_projected, shard.wo_b.weight)
        restored = linear.forward_quantized(quant, packed)
        require_bytes(baseline, restored, f"shard{index}.restored_gemm")
        raw.append(
            {
                "native_wo_a": projected,
                "released_wo_a_native_rotation": reference_on_native_rotation,
                "released_wo_a": reference_projected,
                "native_quant": quant,
                "native_scales": scales,
                "released_quant_native_wo_a": ref_quant,
                "released_scales_native_wo_a": ref_scales.view(torch.uint8),
                "native_gemm": baseline,
                "released_gemm_native_quant": substituted,
                "released_gemm_native_wo_a": reference_native_wo_a,
                "released_gemm_native_rotation": reference_native_rotation,
                "released_gemm": reference,
                "restored_native_gemm": restored,
            }
        )
        shard_metrics.append(
            {
                "wo_a_native_rotation": difference(
                    reference_on_native_rotation, projected
                ),
                "wo_a_released_frequency": difference(reference_projected, projected),
                "activation_values": quantized_difference(quant, ref_quant),
                "activation_scales": quantized_difference(
                    scales.view(torch.float8_e8m0fnu), ref_scales
                ),
                "native_gemm": difference(reference_native_wo_a, baseline),
                "released_gemm_native_quant": difference(
                    reference_native_wo_a, substituted
                ),
            }
        )
    outputs = {
        name: reduce_partials([row[field] for row in raw])
        for name, field in (
            ("native", "native_gemm"),
            ("candidate", "released_gemm_native_quant"),
            ("released_native_wo_a", "released_gemm_native_wo_a"),
            ("released_native_rotation", "released_gemm_native_rotation"),
            ("released", "released_gemm"),
        )
    }
    return (
        {
            "native_rotation": native_rotation,
            "released_native_freq_rotation": released_native_freq_rotation,
            "released_rotation": released_rotation,
            "shards": raw,
            "outputs": outputs,
        },
        {
            "inverse_rope_same_frequency": difference(
                released_native_freq_rotation, native_rotation
            ),
            "inverse_rope_released_frequency": difference(
                released_rotation, native_rotation
            ),
            "shards": shard_metrics,
            "native_output": difference(outputs["released"], outputs["native"]),
            "candidate_output": difference(outputs["released"], outputs["candidate"]),
            "candidate_native_frequency": difference(
                outputs["released_native_rotation"], outputs["candidate"]
            ),
        },
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "history_report",
        "history_artifact",
        "query_report",
        "query_artifact",
        "owner_report",
        "owner_artifact",
        "report",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Output projection evidence must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Output projection replay requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history, inputs, owner_report, owner, tables = admit_owner(cli, root, repo)
    sys.path.insert(0, str(root / "inference"))
    official, converter, kernel = [
        importlib.import_module(name) for name in ("model", "convert", "kernel")
    ]
    for module in (official, converter, kernel):
        if Path(module.__file__).resolve() != root / "inference" / (
            module.__name__ + ".py"
        ):
            raise ValueError("Unexpected released projection module")
    args = model_args(official, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    reader, records, raw, source_controls = NASReader(root), [], [], {}
    for layer in range(3):
        native = native_attention(args, config, reader, layer)
        if native.n_groups != 8:
            raise ValueError("Projection replay requires all eight checkpoint groups")
        shards = official_attention(official, converter, args, reader, layer)
        require_bytes(tables[layer]["native"], native.freqs_cis.cpu(), "native.freqs")
        require_bytes(
            tables[layer]["released"], shards[0].freqs_cis.cpu(), "released.freqs"
        )
        warm = owner["results"]["official"][str(layer)]["query_sparse_decode"][0]
        rows = len(history["identities"][0]["input_ids"])
        tp8_output_projection(
            native, warm["capture"]["attention"].cuda().clone(), native.freqs_cis[:rows]
        )
        immutable = immutable_sources(native, shards)
        before = {name: tensor_digest(value) for name, value in immutable.items()}
        for stream in ("official", "native"):
            cases = owner["results"][stream][str(layer)]["query_sparse_decode"]
            for step, (saved, identity) in enumerate(zip(cases, history["identities"])):
                attention = saved["capture"]["attention"].cuda().contiguous()
                original = attention.clone()
                start, rows = identity["start_pos"], len(identity["input_ids"])
                freqs = native.freqs_cis[start : start + rows]
                reference_freqs = shards[0].freqs_cis[start : start + rows]
                baseline = tp8_output_projection(native, attention.clone(), freqs)
                require_exact(
                    saved["capture"]["output"], baseline, "saved.native.output"
                )
                tensors, metrics = execute_case(
                    native, shards, official, kernel, attention, freqs, reference_freqs
                )
                require_exact(
                    baseline, tensors["outputs"]["native"], "captured.native.output"
                )
                restored = tp8_output_projection(native, attention.clone(), freqs)
                require_bytes(baseline, restored, "restored.native.output")
                require_bytes(original, attention, "held.attention")
                reference = inputs["official_attention"][step][layer]
                same_attention = torch.equal(
                    reference["attention"].reshape(-1).view(torch.uint8),
                    attention.cpu().reshape(-1).view(torch.uint8),
                )
                if same_attention:
                    require_exact(
                        reference["output"],
                        tensors["outputs"]["released"],
                        "saved.released.output",
                    )
                record = {
                    "stream": stream,
                    "layer": layer,
                    "step": step,
                    "identity": identity,
                    "saved_native_exact": True,
                    "restored_native_exact": True,
                    "attention_unchanged": True,
                    "attention_matches_official_history": same_attention,
                    "metrics": metrics,
                }
                records.append(record)
                raw.append(
                    {
                        **{
                            key: value
                            for key, value in record.items()
                            if key != "metrics"
                        },
                        "attention_before": original.cpu(),
                        "attention_after": attention.cpu(),
                        "native_freqs": freqs.cpu(),
                        "released_freqs": reference_freqs.cpu(),
                        "native_rotation": tensors["native_rotation"].cpu(),
                        "released_native_freq_rotation": tensors[
                            "released_native_freq_rotation"
                        ].cpu(),
                        "released_rotation": tensors["released_rotation"].cpu(),
                        "shards": [
                            {name: value.cpu() for name, value in row.items()}
                            for row in tensors["shards"]
                        ],
                        "outputs": {
                            name: value.cpu()
                            for name, value in tensors["outputs"].items()
                        },
                        "restored_output": restored.cpu(),
                    }
                )
                print(json.dumps(record, allow_nan=False), flush=True)
        after = {name: tensor_digest(value) for name, value in immutable.items()}
        if before != after:
            raise AssertionError("Output projection source bytes changed")
        source_controls[str(layer)] = {"before": before, "after": after}
        del native, shards, immutable
    if (
        len(records) != 6 * len(history["identities"])
        or reader.hashes != owner_report["weight_sha256"]
    ):
        raise AssertionError(
            "Projection replay case or checkpoint weight coverage differs"
        )
    with cli.tensor_artifact.open("xb") as stream:
        torch.save({"identities": history["identities"], "cases": raw}, stream)
    source_names = set(owner_report["source_sha256"]) | {
        "rtp_llm/tools/dsv41_output_projection_replay.py"
    }
    report = {
        "prompt_index": history["prompt_index"],
        "identities": history["identities"],
        "native_recipe": history["native_recipe"],
        "records": records,
        "baseline_controls": len(records),
        "shard_controls": 8 * len(records),
        "immutable_sources": source_controls,
        "weight_sha256": reader.hashes,
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
        "source_sha256": {name: sha256(repo / name) for name in sorted(source_names)},
        "evidence_sha256": {
            str(getattr(cli, name)): sha256(getattr(cli, name))
            for name in (
                "history_report",
                "history_artifact",
                "query_report",
                "query_artifact",
                "owner_report",
                "owner_artifact",
            )
        },
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "limitations": [
            "Held attention projection only; no cache, expert or full-model propagation.",
            "Released TP shards execute serially; no live collective or latency claim.",
            "Numerical differences are measured without production promotion.",
        ],
    }
    with cli.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
