"""Isolate NAS Engram projection/gate arithmetic with exact held-history controls.

Only selected host-table rows are read. Changed Engram outputs are not propagated
through attention, experts, retained caches or the full model.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4.engram import Engram, EngramLayout
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_engram_precision import (
    engram_gate_reference,
    engram_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear
from rtp_llm.tools.dsv41_attention_history_replay import prepared_engram, require_exact
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_mhc_boundary_replay import validate_history
from rtp_llm.tools.dsv41_native_history_replay import PreparedNativeEmbedding
from rtp_llm.tools.dsv41_reference_probe import model_args
from transformers import AutoTokenizer


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_source_exact(expected, actual, label):
    if expected.is_floating_point():
        return require_exact(expected, actual, label)
    exact = (
        expected.shape == actual.shape
        and expected.dtype == actual.dtype
        and torch.equal(expected, actual)
    )
    result = {
        "boundary": label,
        "shape": list(expected.shape),
        "reference_dtype": str(expected.dtype),
        "candidate_dtype": str(actual.dtype),
        "bitwise_equal": exact,
    }
    print(json.dumps(result), flush=True)
    if not exact:
        raise AssertionError(f"Engram source changed at {label}")
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("history_report", type=Path)
    parser.add_argument("history_artifact", type=Path)
    parser.add_argument("report", type=Path)
    cli = parser.parse_args()
    if cli.report.exists():
        raise ValueError("Engram replay report must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Engram replay requires SM120 under GPU lock")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history = json.loads(cli.history_report.read_text())
    validate_history(root, repo, history, cli.history_artifact)
    # Unlike the earlier post candidate, this experiment changes no prior owner.
    for name, digest in history["native_source_sha256"].items():
        if sha256(repo / name) != digest:
            raise ValueError(f"Frozen history source changed: {name}")
    if history["native_recipe"].get("mhc_post_reference") is not True:
        raise ValueError("Engram replay requires a qualified composed post history")
    artifact = torch.load(cli.history_artifact, map_location="cpu", weights_only=True)
    if artifact["identities"] != history["identities"] or (
        artifact["native_recipe"] != history["native_recipe"]
    ):
        raise ValueError("History artifact recipe/prefix differs")
    for stream in ("official", "native"):
        values = artifact[f"{stream}_mhc"]
        if len(values) != 3 or any(len(step) != 3 for step in values):
            raise ValueError("Engram replay requires three steps and three layers")
    sys.path.insert(0, str(root / "inference"))
    official = importlib.import_module("model")
    if Path(official.__file__).resolve() != root / "inference/model.py":
        raise ValueError("Unexpected released model module")
    kernel = importlib.import_module("kernel")
    if Path(kernel.__file__).resolve() != root / "inference/kernel.py":
        raise ValueError("Unexpected released kernel module")
    args = model_args(official, root, 256)
    config = json.loads((root / "config.json").read_text())["text_config"]
    layout = official.EngramLayout.from_args(args)
    if tuple(layer for layer in layout.layer_ids if layer < 3) != (1,):
        raise ValueError("Replay requires the released L0-L2 Engram schedule")
    tokenizer = AutoTokenizer.from_pretrained(
        root, trust_remote_code=False, local_files_only=True
    )
    with torch.device("cpu"):
        hasher = official.NgramHashState(args, layout, tokenizer)
        hashes = [
            hasher(
                torch.tensor([identity["input_ids"]], dtype=torch.long, device="cpu"),
                identity["start_pos"],
            )
            for identity in history["identities"]
        ]
    reader, cases = NASReader(root), []
    native_controls = official_controls = restored_controls = source_controls = 0
    reference_controls = projection_controls = 0
    substituted_projections = combined_controls = owner_restorations = gate_controls = 0
    prefix = "layers.1.engram."
    for step, step_hashes in enumerate(hashes):
        ids = step_hashes[:, :, layout.layer_ids.index(1), :]
        released_owner, rows = prepared_engram(official, args, layout, reader, 1, ids)
        embedding = PreparedNativeEmbedding()
        embedding.rows = rows[0]
        native_owner = Engram(
            EngramLayout(config),
            layout.layer_ids.index(1),
            embedding,
            V41MXFP8Linear(
                reader.load(prefix + "wkv.weight"), reader.load(prefix + "wkv.scale")
            ),
            reader.load(prefix + "q_weight"),
            reader.load(prefix + "k_weight"),
            args.norm_eps,
        )
        released_ids, native_ids = ids.cuda(), ids[0].cuda()
        for stream in ("official", "native"):
            saved = artifact[f"{stream}_mhc"][step][1]
            hidden = saved["layer_input"].cuda().reshape(-1, args.hc_mult, args.dim)
            tensors = (
                hidden,
                rows,
                native_ids,
                native_owner.q_weight,
                native_owner.k_weight,
            )
            before = [value.clone() for value in tensors]
            projections = {}
            released_linear, native_linear = (
                released_owner.wkv.forward,
                native_owner.wkv.forward,
            )

            def capture_released(value):
                result = released_linear(value)
                projections["released"] = result[0].clone()
                return result

            def capture_native(value):
                result = native_linear(value)
                projections["native"] = result.clone()
                return result

            with patch.object(released_owner.wkv, "forward", capture_released):
                released = released_owner(hidden[None].clone(), released_ids).clone()
            with patch.object(native_owner.wkv, "forward", capture_native):
                baseline = native_owner(hidden.clone(), native_ids).clone()
            held = []
            for projection in ("native", "released"):
                kv = projections[projection]
                kv_before = kv.clone()
                with patch.object(
                    released_owner.wkv, "forward", lambda value: kv[None]
                ):
                    reference = released_owner(
                        hidden[None].clone(), released_ids
                    ).clone()
                with patch.object(native_owner.wkv, "forward", lambda value: kv):
                    fixed_baseline = native_owner(hidden.clone(), native_ids).clone()
                    working = hidden.clone()
                    with engram_gate_reference(True):
                        candidate = native_owner(working, native_ids)
                    if candidate.data_ptr() != working.data_ptr():
                        raise AssertionError(
                            "Engram candidate lost native hidden writeback"
                        )
                    candidate = candidate.clone()
                    restored = native_owner(hidden.clone(), native_ids).clone()
                held.append(
                    (
                        projection,
                        reference,
                        fixed_baseline,
                        candidate,
                        restored,
                        kv_before,
                        kv,
                    )
                )
            with engram_gate_reference(True):
                gate_only = native_owner(hidden.clone(), native_ids).clone()
            substituted = []

            def capture_substituted(value):
                result = native_linear(value)
                substituted.append(result.clone())
                return result

            with patch.object(native_owner.wkv, "forward", capture_substituted):
                with engram_projection_reference(True, kernel=kernel):
                    projection_only = native_owner(hidden.clone(), native_ids).clone()
                    working = hidden.clone()
                    with engram_gate_reference(True):
                        combined = native_owner(working, native_ids)
                    if combined.data_ptr() != working.data_ptr():
                        raise AssertionError(
                            "Combined Engram diagnostic lost writeback"
                        )
                    combined = combined.clone()
            restored_owner = native_owner(hidden.clone(), native_ids).clone()
            label = f"{stream}.step{step}.layer01.engram"
            # Compare/transfer only after all forwards on these held inputs.
            if stream == "official":
                require_exact(
                    saved["attention_residual"], released, label + ".released_control"
                )
                official_controls += 1
            else:
                require_exact(
                    saved["attention_residual"], baseline, label + ".native_control"
                )
                native_controls += 1
            require_exact(baseline, held[0][2], label + ".fixed_projection_control")
            projection_controls += 1
            require_exact(held[0][3], gate_only, label + ".actual_gate_control")
            gate_controls += 1
            for index, value in enumerate(substituted):
                require_exact(
                    projections["released"], value, f"{label}.substituted_wkv{index}"
                )
                substituted_projections += 1
            require_exact(released, combined, label + ".combined_owner")
            combined_controls += 1
            require_exact(baseline, restored_owner, label + ".restored_owner")
            owner_restorations += 1
            for index, (left, right) in enumerate(zip(before, tensors)):
                require_source_exact(left, right, f"{label}.source{index}")
                source_controls += 1
            for (
                projection,
                reference,
                fixed,
                candidate,
                restored,
                before_kv,
                kv,
            ) in held:
                candidate_diff = require_exact(
                    reference, candidate, label + f".{projection}.candidate"
                )
                require_exact(fixed, restored, label + f".{projection}.restored")
                require_exact(before_kv, kv, label + f".{projection}.projection_source")
                reference_controls += 1
                restored_controls += 1
                source_controls += 1
                cases.append(
                    {
                        "stream": stream,
                        "step": step,
                        "layer": 1,
                        "projection": projection,
                        "native_vs_released_projection": difference(
                            projections["released"], projections["native"]
                        ),
                        "native_vs_released_residual": difference(released, baseline),
                        "fixed_gate_vs_released": difference(reference, fixed),
                        "candidate_vs_released": candidate_diff,
                        "gate_only_vs_released_owner": difference(released, gate_only),
                        "projection_only_vs_released_owner": difference(
                            released, projection_only
                        ),
                        "combined_vs_released_owner": difference(released, combined),
                    }
                )
    expected_rows = [
        row
        for row in history["selected_row_reads"]
        if row["name"].startswith(prefix + "embed.")
    ]
    if reader.selected_row_reads != expected_rows:
        raise ValueError("Selected Engram rows differ from qualified history")
    for name, digest in reader.hashes.items():
        if history["weight_sha256"].get(name) != digest:
            raise ValueError(f"History Engram weight changed: {name}")
    if (
        native_controls,
        official_controls,
        projection_controls,
        reference_controls,
        restored_controls,
        source_controls,
    ) != (3, 3, 6, 12, 12, 42):
        raise AssertionError("Engram replay control coverage is incomplete")
    if (
        substituted_projections,
        combined_controls,
        owner_restorations,
        gate_controls,
    ) != (12, 6, 6, 6):
        raise AssertionError("Engram projection candidate coverage is incomplete")
    source_names = (
        "rtp_llm/tools/dsv41_engram_boundary_replay.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_engram_precision.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_expert_precision.py",
        "rtp_llm/models_py/modules/dsv4/v41_mxfp8.py",
        "rtp_llm/models_py/modules/dsv4/engram.py",
        "rtp_llm/models_py/modules/dsv4/_engram_inject_triton.py",
        "rtp_llm/models_py/modules/dsv4/utils.py",
    )
    result = {
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
        "history_revision": history["native_revision"],
        "history_report_sha256": sha256(cli.history_report),
        "history_artifact_sha256": sha256(cli.history_artifact),
        "prompt_index": history["prompt_index"],
        "recipe": history["native_recipe"],
        "native_source_sha256": {name: sha256(repo / name) for name in source_names},
        "official_source_sha256": history["official_source_sha256"],
        "weight_sha256": reader.hashes,
        "selected_row_reads": reader.selected_row_reads,
        "native_engram_controls_exact": native_controls,
        "official_engram_controls_exact": official_controls,
        "held_projection_controls_exact": projection_controls,
        "candidate_gate_comparisons_exact": reference_controls,
        "restored_native_controls_exact": restored_controls,
        "source_tensor_checks_exact": source_controls,
        "substituted_projection_controls_exact": substituted_projections,
        "combined_owner_comparisons_exact": combined_controls,
        "restored_owner_controls_exact": owner_restorations,
        "actual_gate_controls_exact": gate_controls,
        "cases": cases,
        "limitations": [
            "L1 Engram only, with held official/native inputs at three saved steps.",
            "Selected table rows and actual small projection; no full table staging.",
            "Changed residuals are not propagated through experts/attention/cache history.",
            "No full-model token, serving, model graph or performance claim.",
        ],
    }
    with cli.report.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
