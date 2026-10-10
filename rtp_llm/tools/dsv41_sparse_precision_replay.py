"""Isolate sparse exponential and fusion arithmetic on admitted prefill inputs.

The maintained direct-sink kernel must reproduce the saved owner. Candidates
keep its tile, key order, BF16 probability contraction and division fixed.
Released attention consumes the same operands in eight serial head shards.
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120 import (
    sparse_prefill_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    aligned_sparse_inputs,
    difference,
)
from rtp_llm.tools.dsv41_output_owner_replay import admit_projection
from rtp_llm.tools.dsv41_output_projection_replay import admit_owner
from rtp_llm.tools.dsv41_query_boundary_replay import (
    KERNEL_SHA256,
    require_bytes,
    sha256,
)


@triton.jit
def _exponential(value, EXP: tl.constexpr):
    if EXP == 0:
        return tl.exp(value)
    elif EXP == 1:
        return libdevice.exp(value)
    else:
        return libdevice.fast_expf(value)


@triton.jit
def _precision_sparse(
    Q,
    KV,
    IDX,
    LENS,
    SINK,
    OUT,
    COUNT,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    EXP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    address = row * HEADS * 512 + heads[:, None] * 512 + columns[None, :]
    q = tl.load(Q + address, heads[:, None] < HEADS, other=0)
    limit = tl.load(LENS + row)
    acc = tl.full((16, 512), 0, tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    total = tl.full((16,), 0, tl.float32)
    for tile in range(tl.cdiv(WIDTH, 32)):
        offset = tile * 32 + tl.arange(0, 32)
        slot = tl.load(IDX + row * WIDTH + offset, offset < WIDTH, other=-1).to(
            tl.int64
        )
        live = (offset < WIDTH) & (offset < limit)
        tl.device_assert(
            ~live | ((slot >= -1) & (slot < COUNT)), "V4.1 invalid precision KV index"
        )
        valid = live & (slot >= 0) & (slot < COUNT)
        kv = tl.load(
            KV + slot[:, None] * 512 + columns[None, :], valid[:, None], other=0
        )
        scores = tl.dot(q, tl.trans(kv)) * (512**-0.5)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        finite_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = _exponential(maximum - finite_max, EXP)
        probabilities = _exponential(scores - finite_max[:, None], EXP)
        acc = tl.dot(probabilities.to(tl.bfloat16), kv, acc * alpha[:, None])
        total = total * alpha + tl.sum(probabilities, 1)
        maximum = next_max
    sink = tl.load(SINK + heads, heads < HEADS, other=0)
    has_keys = total > 0
    finite_max = tl.where(has_keys, maximum, 0.0)
    denominator = total + _exponential(sink - finite_max, EXP)
    result = tl.where(has_keys[:, None], tl.div_rn(acc, denominator[:, None]), 0.0)
    tl.store(OUT + address, result, heads[:, None] < HEADS)


def candidates():
    return [
        (f"{name}_fusion_{'on' if fusion else 'off'}", exp, fusion)
        for name, exp in (("triton_exp", 0), ("libdevice_exp", 1), ("fast_expf", 2))
        for fusion in (True, False)
    ]


def released_attention(kernel, operands):
    query, kv, indices, lengths, sinks = (
        operands[name] for name in ("q", "kv", "indices", "lengths", "sink")
    )
    rows, heads, _ = query.shape
    if heads != 64:
        raise ValueError("Held TP8 sparse replay requires 64 heads")
    offsets = torch.arange(indices.shape[-1], device=query.device)
    masked = torch.where(
        offsets[None, None, :] < lengths[:, None, None], indices, -1
    ).contiguous()
    outputs = []
    for rank in range(8):
        start = rank * 8
        outputs.append(
            kernel.sparse_attn(
                query[:, start : start + 8].reshape(1, rows, 8, 512).contiguous(),
                kv.reshape(1, -1, 512),
                sinks[start : start + 8].contiguous(),
                masked.reshape(1, rows, -1),
                512**-0.5,
            )[0]
        )
    return torch.cat(outputs, dim=1).contiguous(), masked


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    evidence = (
        "history_report",
        "history_artifact",
        "query_report",
        "query_artifact",
        "owner_report",
        "owner_artifact",
        "projection_report",
        "projection_artifact",
    )
    for name in ("checkpoint", *evidence, "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Sparse precision evidence must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Sparse precision replay requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    root, repo = cli.checkpoint.resolve(), Path(__file__).resolve().parents[2]
    history, inputs, owner_report, owners, _ = admit_owner(cli, root, repo)
    projection_report, projections = admit_projection(cli, history, owner_report, repo)
    if history["identities"][0]["start_pos"] != 0:
        raise ValueError("Sparse precision requires the held initial prefill")
    sys.path.insert(0, str(root / "inference"))
    kernel = importlib.import_module("kernel")
    if (
        Path(kernel.__file__).resolve() != root / "inference/kernel.py"
        or sha256(Path(kernel.__file__)) != KERNEL_SHA256
    ):
        raise ValueError("Sparse precision requires the pinned released kernel")
    reader, results, raw = NASReader(root), [], []
    for stream in ("official", "native"):
        for layer in range(3):
            identity = history["identities"][0]
            captured = owners["results"][stream][str(layer)]["query_sparse_decode"][0][
                "capture"
            ]
            reference = inputs["official_attention"][0][layer]
            held = projections[(stream, layer, 0)]
            label = f"{stream}.layer{layer}.prefill"
            require_bytes(
                captured["attention"], held["attention_before"], label + ".projection"
            )
            name = f"layers.{layer}.attn.attn_sink"
            operands = {
                key: captured[key].cuda().contiguous().clone()
                for key in ("q", "kv", "indices", "lengths")
            }
            operands["sink"] = reader.load(name).contiguous().clone()
            if reader.hashes[name] != owner_report["weight_sha256"][name]:
                raise ValueError("Held sparse sink weight differs from owner evidence")
            before = {name: value.cpu().clone() for name, value in operands.items()}
            controls, outputs = {}, {}

            def retain(name, output):
                outputs[name] = output.cpu().clone()
                controls[name] = {
                    key: require_bytes(
                        value, operands[key].cpu(), f"{label}.{name}.{key}"
                    )
                    for key, value in before.items()
                }

            with sparse_sink_div_rn(True):
                output = sparse_prefill_attention(*operands.values())
            retain("maintained", output)
            require_bytes(
                captured["attention"], outputs["maintained"], label + ".saved"
            )
            released, masked = released_attention(kernel, operands)
            retain("released_tp8", released)
            q_control = difference(reference["q"], captured["q"])
            kv_control, selection = aligned_sparse_inputs(
                reference, captured, len(identity["input_ids"])
            )
            identical_inputs = (
                q_control["bitwise_equal"]
                and kv_control["bitwise_equal"]
                and all(row["order_equal"] for row in selection)
            )
            if identical_inputs:
                require_bytes(
                    reference["attention"].reshape(released.shape),
                    outputs["released_tp8"],
                    label + ".released_history",
                )
            failures = {}
            for name, exp, fusion in candidates():
                output = torch.empty_like(operands["q"])
                try:
                    _precision_sparse[
                        (output.shape[0], triton.cdiv(output.shape[1], 16))
                    ](
                        *operands.values(),
                        output,
                        operands["kv"].shape[0],
                        HEADS=output.shape[1],
                        WIDTH=operands["indices"].shape[-1],
                        EXP=exp,
                        enable_fp_fusion=fusion,
                        num_warps=4,
                        num_stages=1,
                    )
                except triton.runtime.errors.OutOfResources as error:
                    failures[name] = {"status": "unsupported", "reason": str(error)}
                    continue
                retain(name, output)
                if name == "triton_exp_fusion_on":
                    require_bytes(outputs["maintained"], outputs[name], label + ".copy")
            metrics = {
                name: {
                    "maintained": difference(outputs["maintained"], value),
                    "released_tp8": difference(outputs["released_tp8"], value),
                    "official_history": difference(reference["attention"], value),
                }
                for name, value in outputs.items()
            }
            record = {
                "stream": stream,
                "layer": layer,
                "step": 0,
                "identity": identity,
                "held_q": q_control,
                "held_logical_kv": kv_control,
                "selected_positions": selection,
                "released_history_required": identical_inputs,
                "controls": controls,
                "outputs": metrics,
                "unsupported": failures,
            }
            results.append(record)
            raw.append(
                {
                    "stream": stream,
                    "layer": layer,
                    "step": 0,
                    "identity": identity,
                    "operands_before": before,
                    "operands_after": {
                        name: value.cpu().clone() for name, value in operands.items()
                    },
                    "released_masked_indices": masked.cpu(),
                    "outputs": outputs,
                }
            )
            print(
                json.dumps(
                    {"case": label, "outputs": metrics, "unsupported": failures}
                ),
                flush=True,
            )
    if len(raw) != 6 or len(reader.hashes) != 3:
        raise AssertionError("Sparse precision case or sink coverage differs")
    with cli.tensor_artifact.open("xb") as output:
        torch.save({"cases": raw, "identities": history["identities"]}, output)
    source_names = set(projection_report["source_sha256"]) | {
        "rtp_llm/tools/dsv41_sparse_precision_replay.py",
        "rtp_llm/tools/dsv41_output_owner_replay.py",
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
            str(getattr(cli, name)): sha256(getattr(cli, name)) for name in evidence
        },
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "arithmetic": {
            "key_tile": 32,
            "warps": 4,
            "stages": 1,
            "probability_contraction": "bfloat16",
            "sink_division": "div_rn",
            "candidates": candidates(),
            "released_serial_head_shards": 8,
        },
        "limitations": [
            "Six held prefill cases per prompt; no retained cache or expert execution.",
            "Released comparisons use identical held operands and eight serial head shards.",
            "No production promotion, performance or full-model accuracy claim.",
        ],
    }
    with cli.report.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
