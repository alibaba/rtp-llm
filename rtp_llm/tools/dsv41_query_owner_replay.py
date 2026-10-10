"""Propagate corrected query GEMMs through qualified held attention histories.

All profiles execute the maintained native owner with separate caches. Inputs
remain held at the saved official/native history; changed attention outputs do
not propagate to later inputs or execute experts.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_compressor_precision import (
    compressor_bf16,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_query_precision import (
    query_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.tools.dsv41_attention_history_replay import (
    compare_capture,
    cpu_snapshot,
    execute_native,
    fresh_cache,
    official_history_control,
    require_exact,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, native_attention
from rtp_llm.tools.dsv41_query_boundary_replay import (
    require_bytes,
    sha256,
    validate_history,
)
from rtp_llm.tools.dsv41_reference_probe import model_args


def sources(owner):
    result = {
        name: getattr(owner, name)
        for name in ("q_norm", "kv_norm", "attn_sink", "freqs_cis")
    }
    for name in ("wq_a", "wq_b", "wkv", "wo_b"):
        layer = getattr(owner, name)
        result[f"{name}.weight"] = layer.weight
        result[f"{name}.scales"] = layer._sm120_linear.scales
    return result


def tensor_digest(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def validate_queries(history, history_path, artifact_path, query, query_path, tensor):
    if (
        query["identities"] != history["identities"]
        or tensor["identities"] != history["identities"]
        or query["native_recipe"] != history["native_recipe"]
        or tensor["native_recipe"] != history["native_recipe"]
        or query["history_report_sha256"] != sha256(history_path)
        or query["history_artifact_sha256"] != sha256(artifact_path)
        or query["tensor_artifact_sha256"] != sha256(query_path)
        or query["weight_sha256"]
        != {
            name: value
            for name, value in history["weight_sha256"].items()
            if ".attn." in name
        }
    ):
        raise ValueError("Query owner requires the qualified matching query history")
    wanted = {
        (stream, step, layer)
        for stream in ("official", "native")
        for step in range(len(history["identities"]))
        for layer in range(3)
    }
    cases = {(row["stream"], row["step"], row["layer"]): row for row in tensor["cases"]}
    if set(cases) != wanted or len(tensor["cases"]) != len(wanted):
        raise ValueError("Query owner requires every matching L0-L2 query case")
    return cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("history_report", type=Path)
    parser.add_argument("history_artifact", type=Path)
    parser.add_argument("query_report", type=Path)
    parser.add_argument("query_artifact", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Query owner replay artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Query owner replay requires SM120 under GPU lock")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history = json.loads(cli.history_report.read_text())
    artifact = validate_history(root, repo, history, cli.history_artifact)
    query = json.loads(cli.query_report.read_text())
    query_tensors = torch.load(
        cli.query_artifact, map_location="cpu", weights_only=True
    )
    queries = validate_queries(
        history,
        cli.history_report,
        cli.history_artifact,
        query,
        cli.query_artifact,
        query_tensors,
    )
    sys.path.insert(0, str(root / "inference"))
    official = importlib.import_module("model")
    if Path(official.__file__).resolve() != root / "inference/model.py":
        raise ValueError("Unexpected released reference module")
    kernel = importlib.import_module("kernel")
    if Path(kernel.__file__).resolve() != root / "inference/kernel.py":
        raise ValueError("Unexpected released query kernel")
    args = model_args(official, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    identities = history["identities"]
    count = len(identities)
    control_profile = official_history_control(history["native_recipe"])
    profiles = (
        ("baseline", False, False),
        ("query", True, False),
        ("query_sparse_decode", True, True),
    )
    reader, results, raw = NASReader(root), {}, {}
    totals = dict(
        baseline_records=0,
        baseline_queries=0,
        released_queries=0,
        source_records=0,
        physical_cache_records=0,
        immutable_sources=0,
    )
    for stream in ("official", "native"):
        inputs = artifact[f"{stream}_inputs"]
        controls = (
            [
                [
                    artifact["official_input_history"][control_profile][str(layer)][
                        step
                    ]
                    for layer in range(3)
                ]
                for step in range(count)
            ]
            if stream == "official"
            else artifact["native_input_history"]
        )
        results[stream], raw[stream] = {}, {}
        for layer in range(3):
            owner = native_attention(args, config, reader, layer)
            immutable = sources(owner)
            initial = {name: value.cpu().clone() for name, value in immutable.items()}
            initial_hashes = {
                name: tensor_digest(value) for name, value in initial.items()
            }
            results[stream][str(layer)], raw[stream][str(layer)] = {}, {}
            baseline = []
            for profile, corrected_query, sparse_decode in profiles:
                cache, tables = fresh_cache(owner)
                records, snapshots = [], []
                with tp8_projection(True), compressor_bf16(
                    True, fused=True, hca_projection=True
                ), sparse_sink_div_rn(
                    True, decode=sparse_decode
                ), query_projection_reference(
                    corrected_query, kernel=kernel
                ):
                    for step, identity in enumerate(identities):
                        value = inputs[step][layer].cuda().contiguous()
                        before = value.clone()
                        reference = {
                            name: value.cuda()
                            for name, value in artifact["official_attention"][step][
                                layer
                            ].items()
                        }
                        _, capture = execute_native(
                            owner,
                            cache,
                            tables,
                            value,
                            identity["start_pos"],
                            reference,
                        )
                        snapshot = {
                            "capture": cpu_snapshot(capture),
                            "cache": {
                                name: value.cpu().clone()
                                for name, value in cache.storage.items()
                            },
                            "input_before": before.cpu(),
                            "input_after": value.cpu(),
                        }
                        checks = {
                            "input": require_bytes(
                                before,
                                value,
                                f"{stream}.{layer}.{step}.{profile}.input",
                            )
                        }
                        held_query = queries[(stream, step, layer)]
                        if not corrected_query:
                            for name in ("q", "kv", "attention", "output"):
                                checks["saved_" + name] = require_exact(
                                    controls[step][layer][name],
                                    capture[name],
                                    f"{stream}.{layer}.{step}.saved_{name}",
                                )
                            for name in ("indices", "lengths"):
                                checks["saved_" + name] = require_bytes(
                                    controls[step][layer][name],
                                    capture[name].cpu(),
                                    f"{stream}.{layer}.{step}.saved_{name}",
                                )
                            checks["query_artifact"] = require_exact(
                                held_query["native_query"],
                                capture["q"],
                                f"{stream}.{layer}.{step}.saved_query_artifact",
                            )
                            baseline.append(snapshot)
                            totals["baseline_records"] += 1
                            totals["baseline_queries"] += 1
                        else:
                            expected = baseline[step]
                            excluded = {
                                "q",
                                "native_q",
                                "attention",
                                "native_attention",
                                "output",
                            }
                            if snapshot["capture"].keys() != expected["capture"].keys():
                                raise AssertionError(
                                    "Query owner capture boundaries changed"
                                )
                            unchanged = sorted(snapshot["capture"].keys() - excluded)
                            for name in unchanged:
                                require_bytes(
                                    expected["capture"][name],
                                    snapshot["capture"][name],
                                    f"{stream}.{layer}.{step}.{profile}.{name}",
                                )
                            checks["unchanged_sources"] = unchanged
                            if snapshot["cache"].keys() != expected["cache"].keys():
                                raise AssertionError(
                                    "Query owner physical cache tags changed"
                                )
                            for name in snapshot["cache"]:
                                require_bytes(
                                    expected["cache"][name],
                                    snapshot["cache"][name],
                                    f"{stream}.{layer}.{step}.{profile}.cache.{name}",
                                )
                            checks["physical_cache_bytes"] = {
                                str(name): value.numel() * value.element_size()
                                for name, value in snapshot["cache"].items()
                            }
                            checks["released_query"] = require_exact(
                                held_query["released_query"],
                                capture["q"],
                                f"{stream}.{layer}.{step}.{profile}.released_query",
                            )
                            totals["released_queries"] += 1
                            totals["source_records"] += 1
                            totals["physical_cache_records"] += 1
                        records.append(
                            {
                                "controls": checks,
                                "reference": compare_capture(
                                    owner, reference, capture, identity
                                ),
                            }
                        )
                        snapshots.append(snapshot)
                        print(
                            json.dumps(
                                {
                                    "stream": stream,
                                    "layer": layer,
                                    "step": step,
                                    "profile": profile,
                                    "reference": records[-1]["reference"],
                                }
                            ),
                            flush=True,
                        )
                after_hashes = {}
                for name, value in immutable.items():
                    require_bytes(
                        initial[name],
                        value.cpu(),
                        f"{stream}.{layer}.{profile}.immutable.{name}",
                    )
                    after_hashes[name] = tensor_digest(value)
                    totals["immutable_sources"] += 1
                results[stream][str(layer)][profile] = {
                    "records": records,
                    "immutable_before": initial_hashes,
                    "immutable_after": after_hashes,
                }
                raw[stream][str(layer)][profile] = snapshots
    expected_records = 6 * count
    if any(
        totals[name]
        != expected_records
        * (
            2
            if name in ("released_queries", "source_records", "physical_cache_records")
            else 1
        )
        for name in totals
        if name != "immutable_sources"
    ):
        raise AssertionError("Query owner did not qualify every held case")
    if reader.hashes != query["weight_sha256"]:
        raise AssertionError("Query owner attention weight identities differ")
    payload = {
        "results": raw,
        "identities": identities,
        "native_recipe": history["native_recipe"],
    }
    torch.save(payload, cli.tensor_artifact)
    source_names = (
        "rtp_llm/tools/dsv41_query_owner_replay.py",
        "rtp_llm/tools/dsv41_query_boundary_replay.py",
        "rtp_llm/tools/dsv41_attention_replay.py",
        "rtp_llm/tools/dsv41_attention_history_replay.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_query_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_expert_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_tp8_projection.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_compressor_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_sparse_precision.py",
    )
    report = {
        "results": results,
        "controls_exact": totals,
        "identities": identities,
        "native_recipe": history["native_recipe"],
        "prompt_index": history["prompt_index"],
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
            ]
        )
        .decode()
        .strip(),
        "source_sha256": {name: sha256(repo / name) for name in source_names},
        "history_report_sha256": sha256(cli.history_report),
        "history_artifact_sha256": sha256(cli.history_artifact),
        "query_report_sha256": sha256(cli.query_report),
        "query_artifact_sha256": sha256(cli.query_artifact),
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "limitations": [
            "Only matching consumed prefixes: three France or four arithmetic steps, L0-L2.",
            "Official/native inputs are held, with independent retained caches per profile.",
            "Changed outputs do not propagate to later layers/steps and experts do not execute.",
            "Full-model accuracy, serving, live collectives, owner graphs and performance remain separate.",
        ],
    }
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
