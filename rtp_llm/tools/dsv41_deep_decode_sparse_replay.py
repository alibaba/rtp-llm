"""Isolate decode exponential arithmetic on qualified deep Q/KV operands."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_attention_triton
from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import _unpack_fp4
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_mhc_collapse_replay import admit, exact
from rtp_llm.tools.dsv41_query_boundary_replay import KERNEL_SHA256, sha256
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_sparse_precision_replay import (
    _exponential,
    _precision_sparse,
    released_attention,
)


# Diagnostic copy preserves the maintained SWA/GLOBAL loops and launch geometry.
@triton.jit
def _held_pool(
    q,
    acc,
    maximum,
    total,
    pool,
    indices,
    row,
    stride,
    pages,
    WIDTH: tl.constexpr,
    ENTRIES: tl.constexpr,
    GLOBAL: tl.constexpr,
    EXP: tl.constexpr,
):
    columns = tl.arange(0, 512)
    for tile in range(tl.cdiv(WIDTH, 32)):
        offset = tile * 32 + tl.arange(0, 32)
        slot = tl.load(indices + row * WIDTH + offset, offset < WIDTH, other=-1).to(
            tl.int64
        )
        tl.device_assert(
            (slot >= -1) & (slot < pages * ENTRIES), "V4.1 attention invalid slot"
        )
        valid = (slot >= 0) & (slot < pages * ENTRIES)
        base = (slot // ENTRIES) * stride.to(tl.int64)
        entry = slot % ENTRIES
        if GLOBAL:
            packed = tl.load(
                pool + base[:, None] + entry[:, None] * 256 + columns[None, :] // 2,
                valid[:, None],
                other=0,
            )
            codes = (packed.to(tl.int32) >> ((columns[None, :] % 2) * 4)) & 15
            values = _unpack_fp4(codes)
            scales = (
                tl.load(
                    pool
                    + base[:, None]
                    + ENTRIES * 256
                    + entry[:, None] * 32
                    + columns[None, :] // 16,
                    valid[:, None],
                    other=0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
        else:
            values = (
                tl.load(
                    pool + base[:, None] + entry[:, None] * 512 + columns[None, :],
                    valid[:, None],
                    other=0,
                )
                .to(tl.float8e4nv, bitcast=True)
                .to(tl.float32)
            )
            scale_bytes = tl.load(
                pool
                + base[:, None]
                + ENTRIES * 512
                + entry[:, None] * 16
                + columns[None, :] // 32,
                valid[:, None],
                other=127,
            )
            bits = scale_bytes.to(tl.uint32) << 23
            bits = tl.where(scale_bytes == 0, 0x00400000, bits)
            bits = tl.where(scale_bytes == 255, 0x7FC00000, bits)
            scales = bits.to(tl.uint32).to(tl.float32, bitcast=True)
        kv = (values * scales).to(tl.bfloat16)
        scores = tl.dot(q, tl.trans(kv)) * (512**-0.5)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        finite_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = _exponential(maximum - finite_max, EXP)
        probabilities = _exponential(scores - finite_max[:, None], EXP)
        acc = acc * alpha[:, None]
        acc = tl.dot(probabilities.to(tl.bfloat16), kv, acc)
        total = total * alpha + tl.sum(probabilities, 1)
        maximum = next_max
    return acc, maximum, total


@triton.jit(
    do_not_specialize=["swa_stride", "swa_pages", "global_stride", "global_pages"]
)
def _held_attention(
    query,
    sinks,
    swa,
    swa_slots,
    valid_rows,
    global_pool,
    global_slots,
    out,
    swa_stride,
    swa_pages,
    global_stride,
    global_pages,
    HEADS: tl.constexpr,
    SWA_WIDTH: tl.constexpr,
    SWA_ENTRIES: tl.constexpr,
    GLOBAL_WIDTH: tl.constexpr,
    GLOBAL_ENTRIES: tl.constexpr,
    DIRECT_SINK_DIV_RN: tl.constexpr = False,
    EXP: tl.constexpr = 0,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    addresses = row * HEADS * 512 + heads[:, None] * 512 + columns[None, :]
    if not tl.load(valid_rows + row):
        tl.store(out + addresses, 0, heads[:, None] < HEADS)
        return
    q = tl.load(query + addresses, heads[:, None] < HEADS, other=0)
    acc = tl.full((16, 512), 0, tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    total = tl.full((16,), 0, tl.float32)
    acc, maximum, total = _held_pool(
        q,
        acc,
        maximum,
        total,
        swa,
        swa_slots,
        row,
        swa_stride,
        swa_pages,
        SWA_WIDTH,
        SWA_ENTRIES,
        False,
        EXP,
    )
    if GLOBAL_WIDTH:
        acc, maximum, total = _held_pool(
            q,
            acc,
            maximum,
            total,
            global_pool,
            global_slots,
            row,
            global_stride,
            global_pages,
            GLOBAL_WIDTH,
            GLOBAL_ENTRIES,
            True,
            EXP,
        )
    sink = tl.load(sinks + heads, heads < HEADS, other=0)
    if DIRECT_SINK_DIV_RN:
        has_keys = total > 0
        finite_max = tl.where(has_keys, maximum, 0.0)
        denominator = total + _exponential(sink - finite_max, EXP)
        result = tl.where(has_keys[:, None], tl.div_rn(acc, denominator[:, None]), 0.0)
    else:
        next_max = tl.maximum(maximum, sink)
        alpha = _exponential(maximum - next_max, EXP)
        denominator = total * alpha + _exponential(sink - next_max, EXP)
        result = acc * (alpha / denominator)[:, None]
    tl.store(out + addresses, result, heads[:, None] < HEADS)


def equal_bytes(left, right):
    return (
        left.dtype == right.dtype
        and left.numel() == right.numel()
        and torch.equal(
            left.cpu().contiguous().reshape(-1).view(torch.uint8),
            right.cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def reconstruct_pages(operands, saved, global_rows, label):
    """Re-encode held rows at saved slots; original packed bytes were not saved."""
    caches, read_slots, decoded, plans = {}, {}, {}, {}
    indices = operands["indices"].reshape(1, -1)
    for tag, name, logical, values in (
        (
            "swa_kv",
            "swa",
            indices[:, :128] - global_rows,
            operands["kv"][global_rows:].reshape(128, 512),
        ),
        (
            "csa_kv",
            "global",
            indices[:, 128:],
            operands["kv"][:global_rows].reshape(global_rows, 512),
        ),
    ):
        if name == "global" and not global_rows:
            continue
        slots = saved[name + "_slots"].cuda().reshape(1, -1).contiguous().clone()
        live = slots >= 0
        selected = slots[live] - logical[live]
        layout = dsv41_page_layouts(0)[tag]
        if (
            not selected.numel()
            or not torch.equal(selected, selected[:1].expand_as(selected))
            or int(selected[0]) % layout.entries
            or int(logical[live].min()) < 0
            or int(logical[live].max()) >= values.shape[0]
        ):
            raise ValueError("Held decode physical plan differs: " + label + name)
        base = int(selected[0])
        reads = torch.arange(
            base, base + values.shape[0], device="cuda", dtype=torch.int32
        )
        pages = triton.cdiv(base + values.shape[0], layout.entries)
        cache = V41PagedCache(
            torch.zeros((pages, layout.stride_bytes), device="cuda", dtype=torch.uint8),
            tag,
            gen_num_per_cycle=0,
        )
        cache.write(values.contiguous(), reads)
        decoded[name] = cache.read(reads)
        exact(values, decoded[name], label + ".reconstructed." + name)
        caches[name], read_slots[name], plans[name] = cache, reads, slots
    return caches, read_slots, decoded, plans


def paged_attention(kernel, operands, caches, plans, valid_rows, *, exp=None):
    swa, global_cache = caches["swa"], caches.get("global")
    output = torch.empty_like(operands["q"])
    options = {"EXP": exp} if exp is not None else {}
    kernel[(1, triton.cdiv(64, 16))](
        operands["q"],
        operands["sink"],
        swa.raw,
        plans["swa"],
        valid_rows,
        global_cache.raw if global_cache is not None else swa.raw,
        plans.get("global", plans["swa"]),
        output,
        swa.raw.stride(0),
        swa.raw.shape[0],
        global_cache.raw.stride(0) if global_cache is not None else 0,
        global_cache.raw.shape[0] if global_cache is not None else 0,
        HEADS=64,
        SWA_WIDTH=128,
        SWA_ENTRIES=swa.layout.entries,
        GLOBAL_WIDTH=512 if global_cache is not None else 0,
        GLOBAL_ENTRIES=global_cache.layout.entries if global_cache is not None else 0,
        DIRECT_SINK_DIV_RN=True,
        enable_fp_fusion=True,
        num_warps=4,
        num_stages=1,
        **options,
    )
    return output


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "history_qualification",
        "history_report",
        "history_artifact",
        "admission_manifest",
        "report",
    ):
        parser.add_argument(name, type=Path)
    parser.add_argument("--tensor-artifact", type=Path, required=True)
    cli = parser.parse_args()
    if cli.report.exists() or cli.tensor_artifact.exists():
        raise ValueError("Decode sparse artifacts must be fresh")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Decode sparse requires leased SM120 hardware")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    manifest, history, raw = admit(cli)
    sys.path.insert(0, str(cli.checkpoint / "inference"))
    kernel = importlib.import_module("kernel")
    if (
        Path(kernel.__file__).resolve() != cli.checkpoint / "inference/kernel.py"
        or sha256(Path(kernel.__file__)) != KERNEL_SHA256
    ):
        raise ValueError("Decode sparse released kernel differs")
    reader, cases, records = NASReader(cli.checkpoint), [], []
    native_controls = restored_controls = source_controls = history_controls = 0
    copied_controls = reconstructed_controls = unchanged_page_controls = corrected = 0
    for step in range(1, len(raw["identities"])):
        identity = raw["identities"][step]
        if len(identity["input_ids"]) != 1:
            raise ValueError("Deep sparse requires single-token decode")
        end = identity["start_pos"] + 1
        for layer in range(7):
            ratio = 0 if layer < 2 else 2
            global_rows = end // ratio if ratio else 0
            width = 640 if ratio else 128
            saved = raw["native_attention"][step][layer]
            sink_name = f"layers.{layer}.attn.attn_sink"
            sink = reader.load(sink_name)
            if reader.hashes[sink_name] != history["weight_sha256"][sink_name]:
                raise ValueError("Decode sink weight differs from admitted history")
            operands = {
                "q": saved["q"].cuda().reshape(1, 64, 512).contiguous().clone(),
                "kv": saved["kv"].cuda().reshape(-1, 1, 512).contiguous().clone(),
                "indices": saved["indices"].cuda().contiguous().clone(),
                "lengths": saved["lengths"].cuda().contiguous().clone(),
                "sink": sink.contiguous().clone(),
            }
            if (
                tuple(operands["indices"].shape) != (1, 1, width)
                or operands["lengths"].tolist() != [width]
                or operands["kv"].shape[0] != 128 + global_rows
                or operands["indices"].dtype != torch.int32
                or operands["lengths"].dtype != torch.int32
            ):
                raise ValueError("Decode sparse plan scope differs")
            before = {name: value.cpu().clone() for name, value in operands.items()}
            label = f"step{step}.layer{layer:02d}"
            caches, read_slots, decoded, plans = reconstruct_pages(
                operands, saved, global_rows, label
            )
            reconstructed_controls += len(caches)
            valid_rows = torch.ones(1, device="cuda", dtype=torch.bool)
            pages_before = {
                name: cache.raw.cpu().clone() for name, cache in caches.items()
            }
            plans_before = {name: value.cpu().clone() for name, value in plans.items()}

            def precision(exp):
                output = torch.empty_like(operands["q"])
                _precision_sparse[(1, triton.cdiv(64, 16))](
                    *operands.values(),
                    output,
                    operands["kv"].shape[0],
                    HEADS=64,
                    WIDTH=width,
                    EXP=exp,
                    enable_fp_fusion=True,
                    num_warps=4,
                    num_stages=1,
                )
                return output

            baseline = paged_attention(
                _v41_attention_triton._attention, operands, caches, plans, valid_rows
            )
            exact(saved["attention"], baseline, label + ".native_control")
            native_controls += 1
            copied = paged_attention(
                _held_attention, operands, caches, plans, valid_rows, exp=0
            )
            exact(baseline, copied, label + ".copied_control")
            copied_controls += 1
            released, masked = released_attention(kernel, operands)
            candidate = paged_attention(
                _held_attention, operands, caches, plans, valid_rows, exp=1
            )
            gathered = precision(0)
            restored = paged_attention(
                _v41_attention_triton._attention, operands, caches, plans, valid_rows
            )
            exact(baseline, restored, label + ".restored")
            restored_controls += 1
            after = {name: value.cpu().clone() for name, value in operands.items()}
            for name, value in before.items():
                exact(value, after[name], label + ".source." + name)
                source_controls += 1
            pages_after = {
                name: cache.raw.cpu().clone() for name, cache in caches.items()
            }
            plans_after = {name: value.cpu().clone() for name, value in plans.items()}
            for name in caches:
                exact(pages_before[name], pages_after[name], label + ".pages." + name)
                exact(plans_before[name], plans_after[name], label + ".plans." + name)
                unchanged_page_controls += 1
            reference = raw["official_attention"][step][layer]
            aligned_kv = torch.cat(
                (before["kv"][global_rows:], before["kv"][:global_rows])
            )
            mapped = torch.where(
                before["indices"] < global_rows,
                before["indices"] + 128,
                before["indices"] - global_rows,
            )
            mapped = torch.where(before["indices"] >= 0, mapped, -1)
            same_input = all(
                (
                    equal_bytes(reference["q"], before["q"]),
                    equal_bytes(reference["kv"], aligned_kv),
                    equal_bytes(reference["indices"], mapped),
                )
            )
            if same_input:
                exact(reference["attention"], released, label + ".released_history")
                history_controls += 1
            outputs = {
                name: value.cpu().clone()
                for name, value in (
                    ("baseline", baseline),
                    ("copied_baseline", copied),
                    ("released", released),
                    ("candidate", candidate),
                    ("restored", restored),
                    ("gathered_baseline", gathered),
                )
            }
            metrics = {
                name: {
                    "released": difference(outputs["released"], value),
                    "history": difference(reference["attention"], value),
                }
                for name, value in outputs.items()
            }
            corrected += int(equal_bytes(outputs["released"], outputs["candidate"]))
            record = {
                "step": step,
                "layer": layer,
                "identity": identity,
                "global_rows": global_rows,
                "released_history_control_required": same_input,
                "metrics": metrics,
            }
            records.append(record)
            cases.append(
                {key: value for key, value in record.items() if key != "metrics"}
                | {
                    "operands_before": before,
                    "operands_after": after,
                    "outputs": outputs,
                    "released_masked_indices": masked.cpu().clone(),
                    "pages_before": pages_before,
                    "pages_after": pages_after,
                    "plans_before": plans_before,
                    "plans_after": plans_after,
                    "read_slots": {
                        name: value.cpu().clone() for name, value in read_slots.items()
                    },
                    "decoded_rows": {
                        name: value.cpu().clone() for name, value in decoded.items()
                    },
                    "valid_rows": valid_rows.cpu().clone(),
                }
            )
    count = 7 * (len(raw["identities"]) - 1)
    if (len(cases), native_controls, restored_controls, source_controls) != (
        count,
        count,
        count,
        5 * count,
    ):
        raise AssertionError("Deep sparse held coverage differs")
    if (
        copied_controls != count
        or reconstructed_controls != 12 * (len(raw["identities"]) - 1)
        or unchanged_page_controls != reconstructed_controls
    ):
        raise AssertionError("Deep sparse reconstructed paged controls differ")
    if {
        name: tensor_digest(value) for name, value in reader.values.items()
    } != reader.hashes:
        raise AssertionError("Deep sparse cached sink weights changed")
    with cli.tensor_artifact.open("xb") as stream:
        torch.save(
            {
                "identities": raw["identities"],
                "native_recipe": raw["native_recipe"],
                "cases": cases,
            },
            stream,
        )
    report = {
        "revision": manifest["revision"],
        "prompt_index": history["prompt_index"],
        "identities": raw["identities"],
        "native_recipe": raw["native_recipe"],
        "cases": records,
        "native_controls": native_controls,
        "restored_controls": restored_controls,
        "source_controls": source_controls,
        "copied_baseline_controls": copied_controls,
        "reconstructed_row_controls": reconstructed_controls,
        "unchanged_page_controls": unchanged_page_controls,
        "cache_evidence": "reconstructed-pages-exact-decoded-rows",
        "released_history_controls": history_controls,
        "candidate_released_byte_matches": corrected,
        "candidate_exact_on_same_inputs": corrected == count,
        "weight_sha256": reader.hashes,
        "source_sha256": manifest["source_sha256"],
        "history_qualification_sha256": manifest["history_qualification_sha256"],
        "tensor_artifact_sha256": sha256(cli.tensor_artifact),
        "numerical_promotion": False,
    }
    with cli.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "DSV41_DEEP_DECODE_SPARSE_CONTROLS "
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "native_controls",
                    "restored_controls",
                    "source_controls",
                    "copied_baseline_controls",
                    "reconstructed_row_controls",
                    "unchanged_page_controls",
                    "released_history_controls",
                    "candidate_released_byte_matches",
                    "candidate_exact_on_same_inputs",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
