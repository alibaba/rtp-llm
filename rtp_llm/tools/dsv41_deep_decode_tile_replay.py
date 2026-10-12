"""Isolate decode key tiles while reproducing qualified 32-key outputs."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_attention_triton
from rtp_llm.models_py.modules.dsv4.fp8._v41_cache_triton import _unpack_fp4
from rtp_llm.tools.dsv41_attention_replay import NASReader, difference
from rtp_llm.tools.dsv41_deep_decode_sparse_replay import (
    equal_bytes,
    paged_attention,
    reconstruct_pages,
)
from rtp_llm.tools.dsv41_mhc_collapse_replay import admit, exact
from rtp_llm.tools.dsv41_query_boundary_replay import KERNEL_SHA256, sha256
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_sparse_precision_replay import _exponential, released_attention


# Diagnostic tile copy preserves the admitted separate SWA/GLOBAL loops.
@triton.jit
def _tile_pool(
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
    TILE: tl.constexpr,
):
    columns = tl.arange(0, 512)
    for tile in range(tl.cdiv(WIDTH, TILE)):
        offset = tile * TILE + tl.arange(0, TILE)
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


@triton.jit(do_not_specialize=["stride", "pages"])
def _tile_phase(
    query,
    pool,
    indices,
    valid_rows,
    accumulator,
    maxima,
    totals,
    stride,
    pages,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    ENTRIES: tl.constexpr,
    GLOBAL: tl.constexpr,
    FIRST: tl.constexpr,
    EXP: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    addresses = row * HEADS * 512 + heads[:, None] * 512 + columns[None, :]
    stats = row * HEADS + heads
    if not tl.load(valid_rows + row):
        tl.store(accumulator + addresses, 0.0, heads[:, None] < HEADS)
        tl.store(maxima + stats, -float("inf"), heads < HEADS)
        tl.store(totals + stats, 0.0, heads < HEADS)
        return
    q = tl.load(query + addresses, heads[:, None] < HEADS, other=0)
    if FIRST:
        acc = tl.full((16, 512), 0, tl.float32)
        maximum = tl.full((16,), -float("inf"), tl.float32)
        total = tl.full((16,), 0, tl.float32)
    else:
        acc = tl.load(accumulator + addresses, heads[:, None] < HEADS, other=0)
        maximum = tl.load(maxima + stats, heads < HEADS, other=-float("inf"))
        total = tl.load(totals + stats, heads < HEADS, other=0)
    acc, maximum, total = _tile_pool(
        q,
        acc,
        maximum,
        total,
        pool,
        indices,
        row,
        stride,
        pages,
        WIDTH,
        ENTRIES,
        GLOBAL,
        EXP,
        TILE,
    )
    tl.store(accumulator + addresses, acc, heads[:, None] < HEADS)
    tl.store(maxima + stats, maximum, heads < HEADS)
    tl.store(totals + stats, total, heads < HEADS)


@triton.jit
def _tile_sink(
    accumulator, maxima, totals, sinks, out, HEADS: tl.constexpr, EXP: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * 16 + tl.arange(0, 16)
    columns = tl.arange(0, 512)
    addresses = row * HEADS * 512 + heads[:, None] * 512 + columns[None, :]
    stats = row * HEADS + heads
    acc = tl.load(accumulator + addresses, heads[:, None] < HEADS, other=0)
    maximum = tl.load(maxima + stats, heads < HEADS, other=-float("inf"))
    total = tl.load(totals + stats, heads < HEADS, other=0)
    sink = tl.load(sinks + heads, heads < HEADS, other=0)
    has_keys = total > 0
    finite_max = tl.where(has_keys, maximum, 0.0)
    denominator = total + _exponential(sink - finite_max, EXP)
    result = tl.where(has_keys[:, None], tl.div_rn(acc, denominator[:, None]), 0.0)
    tl.store(out + addresses, result, heads[:, None] < HEADS)


LAUNCH_METADATA = {}
RESOURCE_LIMIT_BYTES = None


class _UnavailableKeyTile(ValueError):
    pass


def record_launch(key, kernel):
    value = {
        "shared_bytes": kernel.metadata.shared,
        "num_warps": kernel.metadata.num_warps,
        "num_stages": kernel.metadata.num_stages,
        "compiled_hash": kernel.hash,
    }
    if key in LAUNCH_METADATA and LAUNCH_METADATA[key] != value:
        raise AssertionError("Key-tile compiled launch metadata changed")
    LAUNCH_METADATA[key] = value


class _KeyTileKernel:
    def __init__(self, tile):
        self.tile = tile

    def __getitem__(self, grid):
        def tiled(*args, **kwargs):
            query, sinks, swa, swa_slots, valid_rows, global_pool, global_slots, out = (
                args[:8]
            )
            swa_stride, swa_pages, global_stride, global_pages = args[8:]
            if not kwargs["DIRECT_SINK_DIV_RN"]:
                raise ValueError("Split tile diagnostic requires direct sink division")
            heads, exp = kwargs["HEADS"], kwargs["EXP"]
            accumulator = torch.empty_like(query, dtype=torch.float32)
            maxima = torch.empty(
                query.shape[:2], device=query.device, dtype=torch.float32
            )
            totals = torch.empty_like(maxima)
            options = {
                name: kwargs[name]
                for name in ("num_warps", "num_stages", "enable_fp_fusion")
            }
            prefix = f"tile{self.tile}.exp{exp}."
            kernel = _tile_phase[grid](
                query,
                swa,
                swa_slots,
                valid_rows,
                accumulator,
                maxima,
                totals,
                swa_stride,
                swa_pages,
                HEADS=heads,
                WIDTH=kwargs["SWA_WIDTH"],
                ENTRIES=kwargs["SWA_ENTRIES"],
                GLOBAL=False,
                FIRST=True,
                EXP=exp,
                TILE=self.tile,
                **options,
            )
            record_launch(prefix + "swa", kernel)
            if kwargs["GLOBAL_WIDTH"]:
                kernel = _tile_phase.warmup(
                    query,
                    global_pool,
                    global_slots,
                    valid_rows,
                    accumulator,
                    maxima,
                    totals,
                    global_stride,
                    global_pages,
                    HEADS=heads,
                    WIDTH=kwargs["GLOBAL_WIDTH"],
                    ENTRIES=kwargs["GLOBAL_ENTRIES"],
                    GLOBAL=True,
                    FIRST=False,
                    EXP=exp,
                    TILE=self.tile,
                    grid=grid,
                    **options,
                )
                record_launch(prefix + "global", kernel)
                if kernel.metadata.shared > RESOURCE_LIMIT_BYTES:
                    raise _UnavailableKeyTile(
                        f"tile{self.tile}.exp{exp}.global requires "
                        f"{kernel.metadata.shared} shared bytes; limit {RESOURCE_LIMIT_BYTES}"
                    )
                _tile_phase[grid](
                    query,
                    global_pool,
                    global_slots,
                    valid_rows,
                    accumulator,
                    maxima,
                    totals,
                    global_stride,
                    global_pages,
                    HEADS=heads,
                    WIDTH=kwargs["GLOBAL_WIDTH"],
                    ENTRIES=kwargs["GLOBAL_ENTRIES"],
                    GLOBAL=True,
                    FIRST=False,
                    EXP=exp,
                    TILE=self.tile,
                    **options,
                )
            kernel = _tile_sink[grid](
                accumulator, maxima, totals, sinks, out, HEADS=heads, EXP=exp, **options
            )
            record_launch(prefix + "sink", kernel)
            return kernel

        return tiled


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
    global RESOURCE_LIMIT_BYTES
    shared_limit = triton.runtime.driver.active.utils.get_device_properties(
        torch.cuda.current_device()
    )["max_shared_mem"]
    RESOURCE_LIMIT_BYTES = shared_limit
    manifest, history, raw = admit(cli)
    preceding_report = json.loads(Path(manifest["sparse_parent_report"]).read_text())
    preceding_raw = torch.load(
        manifest["sparse_parent_artifact"], map_location="cpu", weights_only=True
    )
    preceding_sha = sha256(Path(manifest["sparse_parent_artifact"]))
    if (
        preceding_report["tensor_artifact_sha256"] != preceding_sha
        or preceding_raw["identities"] != raw["identities"]
    ):
        raise ValueError("Key-tile preceding held dataset differs")
    preceding_cases = {
        (row["step"], row["layer"]): row for row in preceding_raw["cases"]
    }
    if len(preceding_cases) != 7 * (len(raw["identities"]) - 1):
        raise ValueError("Key-tile preceding case coverage differs")
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
    preceding_candidate_controls = tile_triton_matches = unavailable_cases = 0
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
            preceding = preceding_cases[step, layer]
            if preceding["identity"] != identity:
                raise ValueError("Key-tile preceding identity differs")
            for name, value in before.items():
                exact(
                    preceding["operands_before"][name],
                    value,
                    label + ".preceding_operands." + name,
                )
            caches, read_slots, decoded, plans = reconstruct_pages(
                operands, saved, global_rows, label
            )
            reconstructed_controls += len(caches)
            valid_rows = torch.ones(1, device="cuda", dtype=torch.bool)
            pages_before = {
                name: cache.raw.cpu().clone() for name, cache in caches.items()
            }
            plans_before = {name: value.cpu().clone() for name, value in plans.items()}

            for name in caches:
                for field, value in (
                    ("pages_before", pages_before[name]),
                    ("plans_before", plans_before[name]),
                    ("read_slots", read_slots[name]),
                    ("decoded_rows", decoded[name]),
                ):
                    exact(
                        preceding[field][name],
                        value,
                        label + ".preceding." + field + "." + name,
                    )

            baseline = paged_attention(
                _v41_attention_triton._attention, operands, caches, plans, valid_rows
            )
            exact(saved["attention"], baseline, label + ".native_control")
            exact(
                preceding["outputs"]["baseline"],
                baseline,
                label + ".preceding_baseline",
            )
            native_controls += 1
            copied = paged_attention(
                _KeyTileKernel(32), operands, caches, plans, valid_rows, exp=0
            )
            exact(baseline, copied, label + ".copied_control")
            copied_controls += 1
            preceding_candidate = paged_attention(
                _KeyTileKernel(32), operands, caches, plans, valid_rows, exp=1
            )
            exact(
                preceding["outputs"]["candidate"],
                preceding_candidate,
                label + ".preceding_candidate",
            )
            preceding_candidate_controls += 1
            released, masked = released_attention(kernel, operands)
            exact(
                preceding["outputs"]["released"],
                released,
                label + ".preceding_released",
            )
            unavailable = {}
            variants = {}
            for name, exp in (("candidate", 1), ("tile64_triton_exp", 0)):
                try:
                    variants[name] = paged_attention(
                        _KeyTileKernel(64), operands, caches, plans, valid_rows, exp=exp
                    )
                except _UnavailableKeyTile as error:
                    variants[name] = None
                    unavailable[name] = str(error)
            candidate, tile_triton = (
                variants["candidate"],
                variants["tile64_triton_exp"],
            )
            unavailable_cases += int(bool(unavailable))
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
                name: value.cpu().clone() if value is not None else None
                for name, value in (
                    ("baseline", baseline),
                    ("copied_baseline", copied),
                    ("released", released),
                    ("candidate", candidate),
                    ("restored", restored),
                    ("tile64_triton_exp", tile_triton),
                    ("preceding_candidate", preceding_candidate),
                )
            }
            metrics = {
                name: (
                    {
                        "released": difference(outputs["released"], value),
                        "history": difference(reference["attention"], value),
                    }
                    if value is not None
                    else None
                )
                for name, value in outputs.items()
            }
            corrected += int(
                outputs["candidate"] is not None
                and equal_bytes(outputs["released"], outputs["candidate"])
            )
            tile_triton_matches += int(
                outputs["tile64_triton_exp"] is not None
                and equal_bytes(outputs["released"], outputs["tile64_triton_exp"])
            )
            record = {
                "step": step,
                "layer": layer,
                "identity": identity,
                "global_rows": global_rows,
                "unavailable_tiles": unavailable,
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
        preceding_candidate_controls != count
        or copied_controls != count
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
    if len(LAUNCH_METADATA) != 12:
        raise AssertionError("Split tile launch resource coverage differs")
    report = {
        "revision": manifest["revision"],
        "split_phase_fp32_state": True,
        "unavailable_tile64_cases": unavailable_cases,
        "shared_memory_limit_bytes": shared_limit,
        "launch_metadata": LAUNCH_METADATA,
        "prompt_index": history["prompt_index"],
        "identities": raw["identities"],
        "native_recipe": raw["native_recipe"],
        "cases": records,
        "native_controls": native_controls,
        "restored_controls": restored_controls,
        "source_controls": source_controls,
        "copied_baseline_controls": copied_controls,
        "preceding_candidate_controls": preceding_candidate_controls,
        "preceding_tensor_artifact_sha256": preceding_sha,
        "tile64_triton_released_byte_matches": tile_triton_matches,
        "tile64_triton_exact_on_same_inputs": tile_triton_matches == count,
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
        "DSV41_DEEP_DECODE_TILE_CONTROLS "
        + json.dumps(
            {
                key: report[key]
                for key in (
                    "native_controls",
                    "restored_controls",
                    "source_controls",
                    "copied_baseline_controls",
                    "preceding_candidate_controls",
                    "tile64_triton_released_byte_matches",
                    "unavailable_tile64_cases",
                    "tile64_triton_exact_on_same_inputs",
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
