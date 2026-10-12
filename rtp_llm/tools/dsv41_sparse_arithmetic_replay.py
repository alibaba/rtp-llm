"""Test-only sparse tile/sink arithmetic on saved V4.1 actual inputs.

The baseline copies the maintained SM120 BF16 prefill kernel. Candidates vary
key tile width and sink normalization, following the released reference's
direct denominator addition/division. Production providers are never patched.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120 import (
    sparse_prefill_attention,
)
from rtp_llm.tools.dsv41_attention_history_replay import require_exact
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference


@triton.jit
def _arithmetic_sparse(
    Q,
    KV,
    IDX,
    LENS,
    SINK,
    OUT,
    COUNT,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    EPILOGUE: tl.constexpr,
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
    for tile in range(tl.cdiv(WIDTH, BLOCK)):
        offset = tile * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(IDX + row * WIDTH + offset, offset < WIDTH, other=-1).to(
            tl.int64
        )
        live = (offset < WIDTH) & (offset < limit)
        tl.device_assert(
            ~live | ((slot >= -1) & (slot < COUNT)), "V4.1 invalid replay KV index"
        )
        valid = live & (slot >= 0) & (slot < COUNT)
        kv = tl.load(
            KV + slot[:, None] * 512 + columns[None, :], valid[:, None], other=0
        )
        scores = tl.dot(q, tl.trans(kv)) * (512**-0.5)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        finite_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - finite_max)
        probabilities = tl.exp(scores - finite_max[:, None])
        acc = tl.dot(probabilities.to(tl.bfloat16), kv, acc * alpha[:, None])
        total = total * alpha + tl.sum(probabilities, 1)
        maximum = next_max
    sink = tl.load(SINK + heads, heads < HEADS, other=0)
    if EPILOGUE == 0:
        next_max = tl.maximum(maximum, sink)
        alpha = tl.exp(maximum - next_max)
        denominator = total * alpha + tl.exp(sink - next_max)
        result = acc * (alpha / denominator)[:, None]
    else:
        # The released kernel keeps the key maximum and adds the sink's mass
        # directly, then divides the output accumulator. Round-to-nearest is
        # an independent candidate; it is not assumed from source syntax.
        denominator = total + tl.exp(sink - maximum)
        if EPILOGUE == 1:
            result = acc / denominator[:, None]
        else:
            result = tl.div_rn(acc, denominator[:, None])
    tl.store(OUT + address, result, heads[:, None] < HEADS)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("history_artifact", type=Path)
    parser.add_argument("report", type=Path)
    cli = parser.parse_args()
    if cli.report.exists():
        raise ValueError("Sparse replay report must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Sparse arithmetic replay requires SM120 under GPU lock")
    torch.set_num_threads(1)
    artifact = torch.load(cli.history_artifact, map_location="cpu", weights_only=True)
    reader = NASReader(cli.checkpoint.resolve())
    cases, results = ((0, 0), (1, 0), (2, 0), (0, 2)), {}
    candidates = [
        (f"tile{block}_warps{warps}_{name}", block, warps, epilogue)
        for block, warps in ((32, 4), (64, 4), (64, 8))
        for name, epilogue in (
            ("sink_rescale", 0),
            ("sink_divide", 1),
            ("sink_divide_rn", 2),
        )
    ]
    for layer, step in cases:
        reference = artifact["official_attention"][step][layer]
        captured = artifact["official_input_history"]["tp8_recipe_official_q"][
            str(layer)
        ][step]
        q = reference["q"].reshape(-1, 64, 512).cuda().contiguous()
        kv = captured["kv"].cuda().contiguous()
        indices = captured["indices"].cuda().contiguous()
        lengths = captured["lengths"].cuda().contiguous()
        sink = reader.load(f"layers.{layer}.attn.attn_sink").contiguous()
        baseline = sparse_prefill_attention(q, kv, indices, lengths, sink)
        label = f"layer{layer}.step{step}"
        baseline_check = require_exact(
            captured["attention"], baseline, f"{label}.saved_native_sparse"
        )
        record = {
            "identity": artifact["identities"][step],
            "saved_native_sparse_reconstruction": baseline_check,
            "baseline_reference": difference(reference["attention"], baseline),
            "sinks": {"minimum": float(sink.min()), "maximum": float(sink.max())},
            "candidates": {},
        }
        for name, block, warps, epilogue in candidates:
            output = torch.empty_like(q)
            try:
                _arithmetic_sparse[(q.shape[0], triton.cdiv(q.shape[1], 16))](
                    q,
                    kv,
                    indices,
                    lengths,
                    sink,
                    output,
                    kv.shape[0],
                    HEADS=q.shape[1],
                    WIDTH=indices.shape[-1],
                    BLOCK=block,
                    EPILOGUE=epilogue,
                    num_warps=warps,
                    num_stages=1,
                )
            except triton.runtime.errors.OutOfResources as error:
                # Resource admission fails before launch. Preserve the concrete
                # rejection and continue independent candidates; arithmetic
                # failures and nonzero baseline assertions still propagate.
                result = {"status": "unsupported", "reason": str(error)}
                record["candidates"][name] = result
                print(
                    json.dumps({"case": label, "candidate": name, **result}), flush=True
                )
                continue
            result = difference(reference["attention"], output)
            if block == 32 and warps == 4 and epilogue == 0:
                record["copied_baseline_reconstruction"] = require_exact(
                    baseline, output, f"{label}.copied_baseline"
                )
            record["candidates"][name] = result
            print(json.dumps({"case": label, "candidate": name, **result}), flush=True)
        results[label] = record
    repo = Path(__file__).resolve().parents[2]
    sources = (
        "rtp_llm/tools/dsv41_sparse_arithmetic_replay.py",
        "rtp_llm/tools/dsv41_attention_replay.py",
        "rtp_llm/tools/dsv41_attention_history_replay.py",
        "rtp_llm/models_py/modules/dsv4/fp8/_v41_prefill_attention_sm120.py",
    )
    report = {
        "cases": results,
        "input_artifact_sha256": hashlib.sha256(
            cli.history_artifact.read_bytes()
        ).hexdigest(),
        "checkpoint_weight_sha256": reader.hashes,
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
            "Four fixed actual-input cases and nine arithmetic candidates, without model recomputation.",
            "Official Q is injected and native recipe KV/index plans are held fixed.",
            "Tile/key order, Q/projections, FFNs and distributed collective behavior remain separate.",
            "No production provider change or latency/full-model accuracy claim.",
        ],
    }
    with cli.report.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
