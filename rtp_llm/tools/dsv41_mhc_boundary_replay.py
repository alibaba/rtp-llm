"""Replay delayed-mHC writeback with held official/native history inputs.

Every saved native post must reproduce before replacement results are accepted.
This diagnostic does not propagate changed residuals or re-execute experts.
"""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import (
    mhc_post_reference,
)
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.tools.dsv41_attention_history_replay import (
    require_exact,
    validate_native_recipe,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_reference_probe import model_args


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_history(root, repo, history, artifact):
    if sha256(artifact) != history["input_artifact_sha256"]:
        raise ValueError("History tensor artifact digest differs")
    for field in ("official_reconstruction", "native_reconstruction"):
        if len(history[field]) != 9 or not all(
            row["bitwise_equal"] for row in history[field].values()
        ):
            raise ValueError("History requires all nine exact FFN input controls")
    for name, digest in history["official_source_sha256"].items():
        if sha256(root / "inference" / name) != digest:
            raise ValueError(f"Released source changed: {name}")
    for name, digest in history["native_source_sha256"].items():
        # This scoped post extension is the candidate under investigation.
        if name.endswith("/test/v41_mhc_precision.py"):
            continue
        if sha256(repo / name) != digest:
            raise ValueError(f"History owner or reconstruction source changed: {name}")
    for field in ("reference_trace_sha256", "native_trace_sha256"):
        for name, digest in history[field].items():
            if sha256(Path(name)) != digest:
                raise ValueError(f"History trace changed: {name}")
    traces = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in history["native_trace_sha256"]
    ]
    recipe = validate_native_recipe(
        traces, reference_kernel_sha256=sha256(root / "inference/kernel.py")
    )
    if recipe != history["native_recipe"]:
        raise ValueError("History recipe differs from saved traces")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("history_report", type=Path)
    parser.add_argument("history_artifact", type=Path)
    parser.add_argument("report", type=Path)
    cli = parser.parse_args()
    if cli.report.exists():
        raise ValueError("Replay report must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("mHC boundary replay requires SM120 under GPU lock")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    root = cli.checkpoint.resolve()
    repo = Path(__file__).resolve().parents[2]
    history = json.loads(cli.history_report.read_text())
    validate_history(root, repo, history, cli.history_artifact)
    artifact = torch.load(cli.history_artifact, map_location="cpu", weights_only=True)
    if (
        artifact["native_recipe"] != history["native_recipe"]
        or artifact["identities"] != history["identities"]
    ):
        raise ValueError("History tensor artifact has a different recipe/prefix")
    sys.path.insert(0, str(root / "inference"))
    official = importlib.import_module("model")
    if Path(official.__file__).resolve() != root / "inference/model.py":
        raise ValueError("Unexpected released model module")
    args = model_args(official, root, 256)
    reader, owners = NASReader(root), {}
    for layer in range(3):
        for kind in ("attn", "ffn"):
            owners[layer, kind] = DelayedHCUnit(
                reader.load(f"layers.{layer}.hc_{kind}_fn").float(),
                reader.load(f"layers.{layer}.hc_{kind}_base").float(),
                reader.load(f"layers.{layer}.hc_{kind}_scale").float(),
                dim=args.dim,
                hc_mult=args.hc_mult,
                hc_sinkhorn_iters=args.hc_sinkhorn_iters,
                hc_eps=args.hc_eps,
                norm_eps=args.norm_eps,
                layer_id=layer,
                name=kind,
            )
    for name, digest in reader.hashes.items():
        if history["weight_sha256"].get(name) != digest:
            raise ValueError(f"History mHC weight changed: {name}")
    block = SimpleNamespace(hc_mult=args.hc_mult)
    cases = []
    native_controls, official_controls, reference_checks, unchanged_inputs = 0, 0, 0, 0
    for stream in ("official", "native"):
        steps = artifact[f"{stream}_mhc"]
        if len(steps) != 3 or any(len(step) != 3 for step in steps):
            raise ValueError("mHC replay requires exactly three layers/steps")
        for step, layers in enumerate(steps):
            for layer, values in enumerate(layers):
                for kind in ("attn", "ffn"):
                    if kind == "ffn" and layer == 2:
                        continue
                    owner = owners[layer, kind]
                    x_name = "attention_output" if kind == "attn" else "ffn_output"
                    r_name = "attention_residual" if kind == "attn" else "attn_post"
                    x = values[x_name].cuda().reshape(1, -1, args.dim)
                    residual = (
                        values[r_name].cuda().reshape(1, -1, args.hc_mult, args.dim)
                    )
                    post = (
                        values[f"{kind}_post_mix"]
                        .cuda()
                        .reshape(1, -1, args.hc_mult, 1)
                    )
                    comb = (
                        values[f"{kind}_comb_mix"]
                        .cuda()
                        .reshape(1, -1, args.hc_mult, args.hc_mult)
                    )
                    before = [value.clone() for value in (x, residual, post, comb)]
                    label = f"{stream}.step{step}.layer{layer:02d}.{kind}.post"
                    baseline = owner.post(x, residual.clone(), post, comb).clone()
                    if stream == "native":
                        require_exact(
                            values[f"{kind}_post"], baseline, label + ".control"
                        )
                        native_controls += 1
                    released = official.Block.hc_post(
                        block, x, residual, post.squeeze(-1), comb
                    ).clone()
                    if stream == "official":
                        require_exact(
                            values[f"{kind}_post"], released, label + ".released"
                        )
                        official_controls += 1
                    working = residual.clone()
                    with mhc_post_reference(True):
                        candidate = owner.post(x, working, post, comb)
                    if candidate.data_ptr() != working.data_ptr():
                        raise AssertionError(
                            f"Native post writeback alias changed: {label}"
                        )
                    require_exact(released, candidate, label + ".candidate")
                    reference_checks += 1
                    restored = owner.post(x, residual.clone(), post, comb)
                    require_exact(baseline, restored, label + ".restored")
                    for index, (left, right) in enumerate(
                        zip(before, (x, residual, post, comb))
                    ):
                        require_exact(left, right, f"{label}.source{index}")
                        unchanged_inputs += 1
                    cases.append(
                        {
                            "stream": stream,
                            "step": step,
                            "layer": layer,
                            "kind": kind,
                            "native_vs_released": difference(released, baseline),
                            "candidate_vs_released": difference(released, candidate),
                        }
                    )
    if (native_controls, official_controls, reference_checks, unchanged_inputs) != (
        15,
        15,
        30,
        120,
    ):
        raise AssertionError("mHC replay is incomplete")
    sources = (
        "rtp_llm/tools/dsv41_mhc_boundary_replay.py",
        "rtp_llm/models_py/modules/dsv4/fp8/test/v41_mhc_precision.py",
        "rtp_llm/models_py/modules/dsv4/hc/delayed.py",
        "rtp_llm/models_py/modules/dsv4/hc/base.py",
        "rtp_llm/models_py/modules/dsv4/hc/tilelang_impl.py",
        "rtp_llm/models_py/modules/dsv4/hc/mhc_tilelang.py",
        "rtp_llm/models_py/3rdparty/tile_kernels/modeling/mhc/ops/post.py",
        "rtp_llm/models_py/3rdparty/tile_kernels/mhc/post_kernel.py",
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
        "recipe": history["native_recipe"],
        "native_source_sha256": {name: sha256(repo / name) for name in sources},
        "official_source_sha256": history["official_source_sha256"],
        "weight_sha256": reader.hashes,
        "native_post_controls_exact": native_controls,
        "official_post_controls_exact": official_controls,
        "candidate_post_comparisons_exact": reference_checks,
        "source_tensor_checks_exact": unchanged_inputs,
        "restored_post_controls_exact": len(cases),
        "cases": cases,
        "limitations": [
            "Held official/native post inputs; changed residuals are not propagated.",
            "No expert execution, full-model accuracy or performance claim.",
            "Graphs qualify only the post component; model graph/serving gates remain separate.",
        ],
    }
    with cli.report.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "native_revision",
                    "native_post_controls_exact",
                    "official_post_controls_exact",
                    "candidate_post_comparisons_exact",
                    "source_tensor_checks_exact",
                    "restored_post_controls_exact",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
