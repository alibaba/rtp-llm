"""Measure the in-block dynamic range of a checkpoint's routed experts.

The FP4 -> FP8 rewrite in ``convert_fp4_experts_to_fp8.py`` is exact only while a
128x128 block's dynamic range stays inside e4m3's normal range. The block scale
puts the block max in ``(2^7, 2^8]`` and the smallest e4m3 normal is ``2^-6``, so
the bound is ``2^14``; between ``2^14`` and ``2^17`` values are subnormal and lose
mantissa bits, and past ``2^17`` they flush to zero.

This reports the margin against that bound for an already-converted checkpoint,
which is the evidence behind the converter's exactness claim. Two numbers matter:

  * the largest in-block dynamic range over the routed experts;
  * how many surviving nonzero weights are subnormal -- zero means no value lost
    mantissa bits, whatever the spans say.

On the released DeepSeek-V4-Flash checkpoint (3 shards, 4096 blocks, 59.3M nonzero
weights) those are 48 = 2^5.58 and 0.

Only routed experts are measured. ``shared_experts`` were FP8 in the source and are
copied through untouched, and their spans are wider (2^17.8 on layer 0's w1), which
says nothing about this rewrite -- an early version of this script matched them by
accident and produced a misleading number.

Reads the safetensors headers directly and decodes e4m3 in numpy, so it needs
neither a GPU nor a torch build carrying the float8 dtypes.

Usage:
    python3 measure_expert_block_span.py CHECKPOINT_DIR [--shards N] [--tensors N]
"""

import argparse
import glob
import json
import os
import struct

import numpy as np

FP8_BLOCK = 128
E4M3_SMALLEST_NORMAL = 2.0**-6
EXACT_SPAN_BOUND = 2.0**14


def read_header(path):
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length)), 8 + length


def load_bytes(path, spec, data_start):
    off0, off1 = spec["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(data_start + off0)
        return np.frombuffer(handle.read(off1 - off0), dtype=np.uint8)


def e4m3_to_float32(raw):
    """Decode ``float8_e4m3fn`` bytes; numpy has no such dtype.

    ``fn`` means finite-and-NaN: there are no infinities, and the single NaN
    encoding is exponent 0x0F with mantissa 0x07. Decoding that as a number gives
    1.875 * 2^8 = 480, which would fold into a block's max and let a checkpoint
    carrying NaN report a comfortable span.
    """
    bits = raw.astype(np.uint32)
    sign = np.where(bits & 0x80, -1.0, 1.0).astype(np.float32)
    exponent = ((bits >> 3) & 0x0F).astype(np.int32)
    mantissa_bits = (bits & 0x07).astype(np.int32)
    mantissa = mantissa_bits.astype(np.float32)
    value = np.where(
        exponent == 0,
        mantissa / 8.0 * E4M3_SMALLEST_NORMAL,
        (1.0 + mantissa / 8.0) * np.exp2((exponent - 7).astype(np.float32)),
    )
    value = np.where((exponent == 0x0F) & (mantissa_bits == 0x07), np.nan, value)
    return (sign * value).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--tensors", type=int, default=4,
                        help="routed-expert tensors sampled per shard")
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.checkpoint, "*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors shards under {args.checkpoint}")

    worst_span, worst_name = 0.0, ""
    blocks_seen = nonzero_total = subnormal_total = nan_total = 0
    tensors_seen = 0
    for path in shards[: max(args.shards, 1)]:
        header, data_start = read_header(path)
        names = [
            name
            for name, spec in header.items()
            if name != "__metadata__"
            # The leading dot is what excludes ``shared_experts``.
            and ".experts." in name
            and spec.get("dtype") == "F8_E4M3"
            and len(spec.get("shape", ())) == 2
        ]
        for name in names[: max(args.tensors, 1)]:
            spec = header[name]
            rows, cols = spec["shape"]
            if rows % FP8_BLOCK or cols % FP8_BLOCK:
                continue
            decoded = e4m3_to_float32(load_bytes(path, spec, data_start)).reshape(
                rows, cols
            )
            nan_total += int(np.isnan(decoded).sum())
            magnitude = np.abs(decoded)
            tensors_seen += 1
            blocks = magnitude.reshape(
                rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK
            )
            block_max = blocks.max(axis=(1, 3))
            block_min = np.where(blocks > 0, blocks, np.inf).min(axis=(1, 3))
            live = np.isfinite(block_min) & (block_max > 0)
            span = np.where(live, block_max / np.maximum(block_min, 1e-45), 1.0)
            blocks_seen += int(live.sum())
            nonzero_total += int((magnitude > 0).sum())
            subnormal_total += int(
                ((magnitude > 0) & (magnitude < E4M3_SMALLEST_NORMAL)).sum()
            )
            if float(span.max()) > worst_span:
                worst_span = float(span.max())
                worst_name = f"{os.path.basename(path)}:{name}"

    if not blocks_seen:
        raise SystemExit(
            "no FP8 routed-expert tensors found: is this an unconverted (FP4) "
            "checkpoint?"
        )

    total_shards = len(shards)
    used_shards = min(max(args.shards, 1), total_shards)
    print(
        f"sampled                : {used_shards}/{total_shards} shards, "
        f"{tensors_seen} routed-expert tensors "
        f"(--shards/--tensors raise this)"
    )
    print(f"blocks measured        : {blocks_seen}")
    print(f"nonzero weights        : {nonzero_total}")
    print(f"NaN weights            : {nan_total}")
    print(
        f"largest in-block span  : {worst_span:.4g} = "
        f"2^{np.log2(worst_span):.2f}  ({worst_name})"
    )
    print(f"exactness bound        : {EXACT_SPAN_BOUND:.0f} = 2^14")
    print(f"subnormal nonzeros     : {subnormal_total}  (0 => no mantissa loss)")
    if nan_total:
        print(f"\nVERDICT: {nan_total} NaN weights -- this is not a valid checkpoint.")
        return 1
    if worst_span > EXACT_SPAN_BOUND or subnormal_total:
        print("\nVERDICT: at least one block exceeds the exact range.")
        return 1
    print(
        f"\nVERDICT (over the sample above): clear by "
        f"2^{np.log2(EXACT_SPAN_BOUND / worst_span):.2f}; every value kept its "
        "mantissa."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
