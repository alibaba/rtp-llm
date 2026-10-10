"""Isolate selected-key order and invalid-slot placement on held prefill inputs."""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import triton

from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120 import (
    sparse_prefill_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_output_owner_replay import admit_projection
from rtp_llm.tools.dsv41_output_projection_replay import admit_owner
from rtp_llm.tools.dsv41_query_boundary_replay import (
    KERNEL_SHA256,
    require_bytes,
    sha256,
)
from rtp_llm.tools.dsv41_sparse_precision_replay import (
    _precision_sparse,
    released_attention,
)

ROLES = ("maintained", "released_tp8", "libdevice_exp_fusion_on")
PLANS = ("native", "reference_compact", "reference_layout")
EVIDENCE = (
    "history_report",
    "history_artifact",
    "query_report",
    "query_artifact",
    "owner_report",
    "owner_artifact",
    "projection_report",
    "projection_artifact",
    "precision_report",
    "precision_artifact",
)


def reference_plans(reference, operands):
    """Map released logical positions to native physical KV without changing keys."""
    indices, lengths = operands["indices"], operands["lengths"]
    rows, _, width = indices.shape
    global_rows = operands["kv"].shape[0] - rows
    selected = reference["indices"].reshape(rows, -1)
    if global_rows < 0 or selected.shape[1] > width:
        raise ValueError("Released plan does not fit the held native KV/width")
    mapped = torch.where(selected < rows, selected + global_rows, selected - rows)
    mapped = torch.where(selected >= 0, mapped, -1).to(indices.dtype)
    compact, layout = torch.full_like(indices, -1), torch.full_like(indices, -1)
    layout[:, 0, : mapped.shape[1]] = mapped
    for row, limit in enumerate(lengths.tolist()):
        original = indices[row, 0, :limit]
        wanted = mapped[row][mapped[row] >= 0]
        actual = original[original >= 0]
        if (
            len(wanted) != limit
            or sorted(wanted.tolist()) != sorted(actual.tolist())
            or len(set(wanted.tolist())) != len(wanted)
            or (wanted >= operands["kv"].shape[0]).any()
        ):
            raise AssertionError("Reference order changes held causal key selection")
        compact[row, 0, :limit] = wanted
    return {
        "native": {"indices": indices.clone(), "lengths": lengths.clone()},
        "reference_compact": {"indices": compact, "lengths": lengths.clone()},
        "reference_layout": {
            "indices": layout,
            "lengths": torch.full_like(lengths, mapped.shape[1]),
        },
    }


def admit_precision(cli, history, repo):
    report = json.loads(cli.precision_report.read_text())
    tensor = torch.load(cli.precision_artifact, map_location="cpu", weights_only=True)
    if (
        report["tensor_artifact_sha256"] != sha256(cli.precision_artifact)
        or report["identities"] != history["identities"]
        or tensor["identities"] != history["identities"]
        or report["native_recipe"] != history["native_recipe"]
        or report["reference_kernel_sha256"] != KERNEL_SHA256
    ):
        raise ValueError("Sparse order requires the admitted precision evidence")
    for name in EVIDENCE[:-2]:
        path = getattr(cli, name)
        if report["evidence_sha256"][str(path)] != sha256(path):
            raise ValueError(f"Sparse precision evidence changed: {name}")
    for name, expected in report["source_sha256"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Qualified sparse precision source changed: {name}")
    cases = {(row["stream"], row["layer"], row["step"]): row for row in tensor["cases"]}
    wanted = {
        (stream, layer, 0) for stream in ("official", "native") for layer in range(3)
    }
    if set(cases) != wanted or len(tensor["cases"]) != 6:
        raise ValueError("Sparse order requires all six held prefill cases")
    return report, cases


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", *EVIDENCE, "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Sparse order evidence must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Sparse order replay requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history, inputs, owner_report, owners, _ = admit_owner(cli, root, repo)
    admit_projection(cli, history, owner_report, repo)
    precision_report, preceding = admit_precision(cli, history, repo)
    sys.path.insert(0, str(root / "inference"))
    kernel = importlib.import_module("kernel")
    if Path(kernel.__file__).resolve() != root / "inference/kernel.py":
        raise ValueError("Unexpected released sparse module")
    reader, results, raw = NASReader(root), [], []
    for stream in ("official", "native"):
        for layer in range(3):
            key, label = (stream, layer, 0), f"{stream}.layer{layer}.prefill"
            saved = preceding[key]
            captured = owners["results"][stream][str(layer)]["query_sparse_decode"][0][
                "capture"
            ]
            reference = inputs["official_attention"][0][layer]
            before = saved["operands_before"]
            for name, value in before.items():
                require_bytes(value, saved["operands_after"][name], label + ".admitted")
                if name != "sink":
                    require_bytes(value, captured[name], label + ".owner")
            sink_name = f"layers.{layer}.attn.attn_sink"
            require_bytes(before["sink"], reader.load(sink_name).cpu(), label + ".sink")
            if reader.hashes[sink_name] != owner_report["weight_sha256"][sink_name]:
                raise ValueError("Sparse order sink identity differs")
            plans = reference_plans(reference, before)
            operands = {
                name: value.cuda().contiguous().clone()
                for name, value in before.items()
            }
            outputs, controls, plan_after = {}, {}, {}
            for plan_name in PLANS:
                current = dict(operands)
                current.update(
                    {
                        name: value.cuda().contiguous()
                        for name, value in plans[plan_name].items()
                    }
                )
                expected = dict(before, **plans[plan_name])
                outputs[plan_name], controls[plan_name] = {}, {}
                for role in ROLES:
                    if role == "maintained":
                        with sparse_sink_div_rn(True):
                            output = sparse_prefill_attention(*current.values())
                    elif role == "released_tp8":
                        output, _ = released_attention(kernel, current)
                    else:
                        output = torch.empty_like(current["q"])
                        _precision_sparse[
                            (output.shape[0], triton.cdiv(output.shape[1], 16))
                        ](
                            *current.values(),
                            output,
                            current["kv"].shape[0],
                            HEADS=output.shape[1],
                            WIDTH=current["indices"].shape[-1],
                            EXP=1,
                            enable_fp_fusion=True,
                            num_warps=4,
                            num_stages=1,
                        )
                    outputs[plan_name][role] = output.cpu().clone()
                    controls[plan_name][role] = {
                        name: require_bytes(
                            value,
                            current[name].cpu(),
                            f"{label}.{plan_name}.{role}.{name}",
                        )
                        for name, value in expected.items()
                    }
                    if plan_name == "native":
                        require_bytes(
                            saved["outputs"][role],
                            outputs[plan_name][role],
                            label + ".preceding." + role,
                        )
                plan_after[plan_name] = {
                    name: current[name].cpu().clone() for name in ("indices", "lengths")
                }
            original_after = {
                name: value.cpu().clone() for name, value in operands.items()
            }
            for name, value in before.items():
                require_bytes(value, original_after[name], label + ".original." + name)
            identical = difference(reference["q"], before["q"])["bitwise_equal"]
            global_rows = before["kv"].shape[0] - before["q"].shape[0]
            aligned_kv = torch.cat(
                (before["kv"][global_rows:], before["kv"][:global_rows])
            ).reshape(reference["kv"].shape)
            identical = (
                identical and difference(reference["kv"], aligned_kv)["bitwise_equal"]
            )
            if identical:
                output = outputs["reference_layout"]["released_tp8"]
                require_bytes(
                    reference["attention"].reshape(output.shape),
                    output,
                    label + ".released_layout_history",
                )
            metrics = {
                plan: {
                    role: {
                        "maintained": difference(group["maintained"], output),
                        "released_tp8": difference(group["released_tp8"], output),
                        "official_history": difference(reference["attention"], output),
                    }
                    for role, output in group.items()
                }
                for plan, group in outputs.items()
            }
            record = {
                "stream": stream,
                "layer": layer,
                "step": 0,
                "identity": saved["identity"],
                "released_layout_history_required": identical,
                "controls": controls,
                "outputs": metrics,
            }
            results.append(record)
            raw.append(
                {
                    "stream": stream,
                    "layer": layer,
                    "step": 0,
                    "identity": saved["identity"],
                    "operands_before": before,
                    "operands_after": original_after,
                    "plans_before": plans,
                    "plans_after": plan_after,
                    "outputs": outputs,
                }
            )
            print(
                json.dumps(
                    {
                        "case": label,
                        "different_values": {
                            plan: {
                                role: {
                                    target: result["different_values"]
                                    for target, result in measured.items()
                                }
                                for role, measured in group.items()
                            }
                            for plan, group in metrics.items()
                        },
                    }
                ),
                flush=True,
            )
    with cli.tensor_artifact.open("xb") as output:
        torch.save({"cases": raw, "identities": history["identities"]}, output)
    source_names = set(precision_report["source_sha256"]) | {
        "rtp_llm/tools/dsv41_sparse_order_replay.py"
    }
    report = {
        "cases": results,
        "identities": history["identities"],
        "native_recipe": history["native_recipe"],
        "prompt_index": history["prompt_index"],
        "weight_sha256": reader.hashes,
        "reference_kernel_sha256": KERNEL_SHA256,
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
            str(getattr(cli, name)): sha256(getattr(cli, name)) for name in EVIDENCE
        },
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "plans": PLANS,
        "roles": ROLES,
        "limitations": [
            "Held prefill inputs; no cache, experts, decode, graph, performance or production promotion.",
            "Compact order and exact invalid-slot layout are independent controlled plans.",
        ],
    }
    with cli.report.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
