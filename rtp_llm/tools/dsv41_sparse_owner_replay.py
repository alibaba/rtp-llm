"""Replay the scoped sparse normalization through maintained NAS-weight owners.

Attention inputs are held at saved official/native history. Q, indices and
source/cache bytes must not change. This isolates the operator; model accuracy
requires a new checkpoint run executing actual experts.
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
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.tools.dsv41_attention_history_replay import (
    compare_capture,
    cpu_snapshot,
    execute_native,
    fresh_cache,
    require_exact,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, native_attention
from rtp_llm.tools.dsv41_reference_probe import model_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("history_artifact", type=Path)
    parser.add_argument("report", type=Path)
    cli = parser.parse_args()
    if cli.report.exists():
        raise ValueError("Sparse owner replay report must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Sparse owner replay requires SM120 under GPU lock")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root = cli.checkpoint.resolve()
    sys.path.insert(0, str(root / "inference"))
    official = importlib.import_module("model")
    if Path(official.__file__).resolve() != root / "inference/model.py":
        raise ValueError("Unexpected official reference module")
    args = model_args(official, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    artifact = torch.load(cli.history_artifact, map_location="cpu", weights_only=True)
    identities = artifact["identities"]
    if (
        len(identities) != 3
        or identities[0]["start_pos"] != 0
        or identities[1]["start_pos"] != len(identities[0]["input_ids"])
        or identities[2]["start_pos"] != identities[1]["start_pos"] + 1
        or any(len(item["input_ids"]) != 1 for item in identities[1:])
        or identities[2]["start_pos"] >= 128
        or tuple(args.compress_ratios[:3]) != (0, 0, 2)
    ):
        raise ValueError("Expected the accepted short L0-L2 prefill/decode artifact")
    streams = {
        "official_inputs_native_q": (
            artifact["official_inputs"],
            [
                [
                    artifact["official_input_history"][
                        "tp8_projection_compressor_recipe"
                    ][str(layer)][step]
                    for layer in range(3)
                ]
                for step in range(3)
            ],
            False,
        ),
        "native_inputs_native_q": (
            artifact["native_inputs"],
            artifact["native_input_history"],
            False,
        ),
        "official_inputs_official_q": (
            artifact["official_inputs"],
            [
                [
                    artifact["official_input_history"]["tp8_recipe_official_q"][
                        str(layer)
                    ][step]
                    for layer in range(3)
                ]
                for step in range(3)
            ],
            True,
        ),
    }
    reader, results = NASReader(root), {}
    profiles = (
        ("baseline", False, False),
        ("prefill", True, False),
        ("prefill_decode", True, True),
    )
    for stream, (inputs, controls, fixed_q) in streams.items():
        if len(inputs) != 3 or any(len(step) != 3 for step in inputs):
            raise ValueError("Missing held attention inputs in accepted artifact")
        results[stream] = {}
        for layer in range(3):
            baseline, baseline_bytes = [], []
            results[stream][str(layer)] = {}
            for profile, enabled, decode in profiles:
                owner = native_attention(args, config, reader, layer)
                cache, tables = fresh_cache(owner)
                records = []
                with tp8_projection(True), compressor_bf16(
                    True, fused=True, hca_projection=True
                ), sparse_sink_div_rn(enabled, decode=decode):
                    for step, identity in enumerate(identities):
                        # Substitution happens inside the launch hook, so move
                        # saved reference tensors before entering the forward.
                        reference = {
                            name: value.cuda()
                            for name, value in artifact["official_attention"][step][
                                layer
                            ].items()
                        }
                        _, captured = execute_native(
                            owner,
                            cache,
                            tables,
                            inputs[step][layer].cuda(),
                            identity["start_pos"],
                            reference,
                            fixed_q=fixed_q,
                        )
                        checks = {}
                        if not enabled:
                            for name in ("q", "kv", "attention", "output"):
                                checks[f"saved_{name}"] = require_exact(
                                    controls[step][layer][name],
                                    captured[name],
                                    f"{stream}.layer{layer}.step{step}.saved_{name}",
                                )
                            for name in ("indices", "lengths"):
                                if not torch.equal(
                                    controls[step][layer][name], captured[name].cpu()
                                ):
                                    raise AssertionError(
                                        f"Saved {name} differs at {stream}/{layer}/{step}"
                                    )
                            baseline.append(cpu_snapshot(captured))
                            baseline_bytes.append(
                                {
                                    key: value.cpu().clone()
                                    for key, value in cache.storage.items()
                                }
                            )
                        else:
                            before = baseline[step]
                            excluded = {"attention", "native_attention", "output"}
                            if captured.keys() != before.keys():
                                raise AssertionError("Owner capture boundaries changed")
                            for name in captured.keys() - excluded:
                                if not torch.equal(before[name], captured[name].cpu()):
                                    raise AssertionError(
                                        f"Candidate source {name} differs at {stream}/{layer}/{step}"
                                    )
                            for key, raw in cache.storage.items():
                                if not torch.equal(
                                    baseline_bytes[step][key], raw.cpu()
                                ):
                                    raise AssertionError(
                                        f"Candidate physical cache differs at {stream}/{layer}/{step}/{key}"
                                    )
                            checks = {
                                "source_tensors_bitwise": sorted(
                                    captured.keys() - excluded
                                ),
                                "physical_cache_bytes_bitwise": {
                                    str(key): value.numel()
                                    for key, value in cache.storage.items()
                                },
                            }
                            if profile == "prefill" and step:
                                checks["unmodified_decode_attention"] = require_exact(
                                    before["attention"],
                                    captured["attention"],
                                    f"{stream}.layer{layer}.step{step}.unmodified_decode_attention",
                                )
                            if (
                                fixed_q
                                and profile == "prefill_decode"
                                and (layer, step) in ((0, 0), (1, 0), (2, 0), (0, 2))
                            ):
                                checks["fixed_q_official_attention"] = require_exact(
                                    reference["attention"],
                                    captured["attention"],
                                    f"layer{layer}.step{step}.fixed_q_official_attention",
                                )
                        result = compare_capture(owner, reference, captured, identity)
                        records.append({"controls": checks, "reference": result})
                        print(
                            json.dumps(
                                {
                                    "stream": stream,
                                    "layer": layer,
                                    "step": step,
                                    "profile": profile,
                                    "attention": result["attention"],
                                    "output": result["output"],
                                }
                            ),
                            flush=True,
                        )
                results[stream][str(layer)][profile] = records
    repo = Path(__file__).resolve().parents[2]
    sources = (
        "rtp_llm/tools/dsv41_sparse_owner_replay.py",
        "rtp_llm/tools/dsv41_attention_replay.py",
        "rtp_llm/tools/dsv41_attention_history_replay.py",
        "rtp_llm/models_py/modules/dsv4/fp8/attention_v41.py",
        "rtp_llm/models_py/modules/dsv4/fp8/_v41_prefill_attention_sm120.py",
        "rtp_llm/models_py/modules/dsv4/fp8/_v41_attention_triton.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_sparse_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_compressor_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_tp8_projection.py",
    )
    report = {
        "results": results,
        "identities": identities,
        "input_artifact_sha256": hashlib.sha256(
            cli.history_artifact.read_bytes()
        ).hexdigest(),
        "weight_sha256": reader.hashes,
        "native_revision": subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo}",
                "-C",
                str(repo),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in sources
        },
        "limitations": [
            "Held official/native attention inputs from one accepted three-step L0-L2 history.",
            "Official-Q controls are separate from native-Q owner results.",
            "Modified outputs do not propagate to later layer/step attention inputs.",
            "Actual expert/model accuracy, live collectives, serving, owner graphs and performance require separate qualification.",
        ],
    }
    with cli.report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
