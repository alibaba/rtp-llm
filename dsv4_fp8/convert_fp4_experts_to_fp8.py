"""Rewrite DeepSeek-V4-Flash routed experts from FP4 to FP8 so they run on Hopper.

The checkpoint is already FP8 everywhere except the routed experts: attention and
shared-expert weights ship as ``F8_E4M3 [N, K]`` plus a ``F8_E8M0 [N/128, K/128]``
block scale, exactly the layout ``deep_gemm``'s FP8 GEMMs consume. Only the routed
experts use FP4 -- ``I8 [N, K/2]`` nibble pairs plus a ``F8_E8M0 [N, K/32]``
group-32 scale -- and every FP4 kernel in DeepGEMM requires SM100. H20 is SM90 and
has no FP4 tensor cores, so those experts are the single thing standing between
this checkpoint and this fleet.

This converter rewrites just those tensors into the same FP8 layout the rest of
the checkpoint already uses, leaving every other byte untouched.

The rewrite is value-exact, not merely close. Dequantised FP4 is
``v * 2^e`` where ``v`` comes from a 16-entry e2m1 table and ``e`` is a UE8M0
exponent. Every table entry needs at most four significant bits, which is exactly
what e4m3 carries, so re-expressing a weight against a *power-of-two* block scale
only shifts the exponent and never touches the mantissa. Values survive bit for
bit as long as the shift keeps them inside e4m3's range, which the block scale is
chosen to guarantee; ``--verify`` reports the fraction that did.

Usage:
    python3 convert_fp4_experts_to_fp8.py --src DIR --dst DIR [--verify] [--limit N]
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time

import numpy as np
import torch

# e2m1: 4-bit code -> value. Signed zero collapses to 0.0, which is what the
# reference unpacker in rtp_llm/models_py/modules/dsv4/qlinear.py produces too.
FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

FP4_GROUP = 32          # columns per UE8M0 scale in the source layout
FP8_BLOCK = 128         # square block per UE8M0 scale in the target layout
UE8M0_BIAS = 127
# e4m3 tops out at 448. Its largest FP4 input is 6, so the block scale may lift a
# block by at most 2^6 before overflow. Aiming the block max into (2^7, 2^8]
# stays well inside that while leaving the widest possible room underneath for
# small values, which are what a too-coarse scale would flush to zero.
E4M3_HEADROOM_EXP = 8

DTYPE_ITEMSIZE = {
    "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}


def read_header(path):
    """Return (header_dict, byte offset where the data blob starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def numel(shape):
    out = 1
    for s in shape:
        out *= s
    return out


def is_expert_weight(name, spec):
    return (
        ".experts." in name
        and name.endswith(".weight")
        and spec["dtype"] == "I8"
        and len(spec["shape"]) == 2
    )


def is_expert_scale(name, spec):
    return (
        ".experts." in name
        and name.endswith(".scale")
        and spec["dtype"] == "F8_E8M0"
        and len(spec["shape"]) == 2
    )


def convert_expert(weight_bytes, scale_bytes, n, k_packed, device):
    """FP4 (I8 nibbles + group-32 UE8M0) -> FP8 (e4m3 + 128x128 UE8M0).

    Returns (weight_e4m3_bytes, scale_ue8m0_bytes, exact_fraction, max_rel_err).
    """
    k = k_packed * 2
    if n % FP8_BLOCK or k % FP8_BLOCK:
        raise ValueError(f"shape [{n}, {k}] is not a multiple of {FP8_BLOCK}")

    packed = torch.from_numpy(weight_bytes).to(device).view(n, k_packed)
    # Nibble order matches qlinear._fp4_unpack_to_fp32: low half-byte first.
    codes = torch.empty(n, k, dtype=torch.long, device=device)
    codes[:, 0::2] = (packed & 0x0F).long()
    codes[:, 1::2] = ((packed >> 4) & 0x0F).long()
    values = FP4_LUT.to(device)[codes]

    group_exp = torch.from_numpy(scale_bytes).to(device).view(n, k // FP4_GROUP)
    group_scale = torch.exp2((group_exp.to(torch.int32) - UE8M0_BIAS).float())
    dequant = values * group_scale.repeat_interleave(FP4_GROUP, dim=1)

    # One power-of-two scale per 128x128 block, placed so the block's largest
    # magnitude lands just under e4m3's ceiling.
    blocks = dequant.view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3))
    block_exp = torch.where(
        amax > 0,
        torch.ceil(torch.log2(amax)).to(torch.int32) - E4M3_HEADROOM_EXP,
        torch.zeros_like(amax, dtype=torch.int32),
    )
    biased = block_exp + UE8M0_BIAS
    if int(biased.min()) < 0 or int(biased.max()) > 255:
        raise ValueError(f"block exponent {int(block_exp.min())}..{int(block_exp.max())} escapes UE8M0")

    inv = torch.exp2(-block_exp.float())
    scaled = dequant * inv.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    quant = scaled.to(torch.float8_e4m3fn)

    exact_fraction, max_rel_err = None, None
    if VERIFY:
        recon = quant.float() * torch.exp2(block_exp.float()).repeat_interleave(
            FP8_BLOCK, 0
        ).repeat_interleave(FP8_BLOCK, 1)
        exact_fraction = float((recon == dequant).float().mean())
        denom = dequant.abs().clamp_min(1e-30)
        max_rel_err = float(((recon - dequant).abs() / denom).max())

    weight_out = quant.view(torch.uint8).cpu().numpy().tobytes()
    scale_out = biased.to(torch.uint8).cpu().numpy().tobytes()
    return weight_out, scale_out, exact_fraction, max_rel_err


def plan_file(header):
    """Decide the output spec for every tensor, preserving source order."""
    plan = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if is_expert_weight(name, spec):
            n, k_packed = spec["shape"]
            plan.append((name, spec, "weight", "F8_E4M3", [n, k_packed * 2]))
        elif is_expert_scale(name, spec):
            n, k_groups = spec["shape"]
            k = k_groups * FP4_GROUP
            plan.append((name, spec, "scale", "F8_E8M0", [n // FP8_BLOCK, k // FP8_BLOCK]))
        else:
            plan.append((name, spec, "copy", spec["dtype"], spec["shape"]))
    return plan


def convert_file(src, dst, device, stats):
    header, data_start = read_header(src)
    metadata = header.get("__metadata__")
    plan = plan_file(header)

    out_header = {}
    cursor = 0
    for name, _spec, _kind, out_dtype, out_shape in plan:
        nbytes = numel(out_shape) * DTYPE_ITEMSIZE[out_dtype]
        out_header[name] = {
            "dtype": out_dtype,
            "shape": out_shape,
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    if metadata is not None:
        out_header["__metadata__"] = metadata

    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    # safetensors wants the data blob 8-byte aligned.
    blob += b" " * (-len(blob) % 8)

    # A shard lists an expert's scale before its weight, so the pair is resolved
    # on first sight of either member rather than in stream order.
    scale_of, weight_of = {}, {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if is_expert_scale(name, spec):
            scale_of[name[: -len(".scale")]] = spec
        elif is_expert_weight(name, spec):
            weight_of[name[: -len(".weight")]] = spec

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)

        def raw(spec):
            lo, hi = spec["data_offsets"]
            fin.seek(data_start + lo)
            return np.frombuffer(fin.read(hi - lo), dtype=np.uint8)

        converted = {}

        def pair(stem):
            if stem not in converted:
                w_spec = weight_of[stem]
                n, k_packed = w_spec["shape"]
                weight_out, scale_out, exact, rel = convert_expert(
                    raw(w_spec).copy(), raw(scale_of[stem]).copy(), n, k_packed, device
                )
                converted[stem] = {"weight": weight_out, "scale": scale_out}
                stats["tensors"] += 1
                if exact is not None:
                    stats["exact_min"] = min(stats["exact_min"], exact)
                    stats["exact_sum"] += exact
                    stats["rel_max"] = max(stats["rel_max"], rel)
                    stats["verified"] += 1
            return converted[stem]

        for name, spec, kind, _out_dtype, out_shape in plan:
            if kind == "copy":
                fout.write(raw(spec).tobytes())
                continue
            stem = name[: -len(f".{kind}")]
            entry = pair(stem)
            fout.write(entry.pop(kind))
            if not entry:
                del converted[stem]

        if converted:
            raise RuntimeError(f"unpaired expert tensors: {sorted(converted)[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="measure per-tensor exactness (adds a reconstruction pass)")
    ap.add_argument("--limit", type=int, default=0,
                    help="convert only the first N shards, for a quick check")
    args = ap.parse_args()

    global VERIFY
    VERIFY = args.verify

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.dst, exist_ok=True)

    shards = sorted(f for f in os.listdir(args.src) if f.endswith(".safetensors"))
    if args.limit:
        shards = shards[: args.limit]

    # Everything that is not a shard rides along unchanged, except that the
    # config must now advertise FP8 experts so the loader picks the FP8 path.
    # Subdirectories are copied whole, not skipped: V4 ships executable pieces
    # of the checkpoint (encoding/encoding_dsv4.py, inference/) that the loader
    # imports by path, so a converted checkpoint missing them cannot start.
    for entry in sorted(os.listdir(args.src)):
        src_path = os.path.join(args.src, entry)
        if entry.endswith(".safetensors"):
            continue
        dst_path = os.path.join(args.dst, entry)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        elif entry == "config.json":
            cfg = json.load(open(src_path))
            cfg["expert_dtype"] = "fp8"
            json.dump(cfg, open(dst_path, "w"), indent=2)
            print(f"config.json: expert_dtype -> fp8")
        else:
            shutil.copy2(src_path, dst_path)

    stats = {"tensors": 0, "verified": 0, "exact_min": 1.0, "exact_sum": 0.0, "rel_max": 0.0}
    t0 = time.time()
    for i, shard in enumerate(shards, 1):
        t1 = time.time()
        convert_file(os.path.join(args.src, shard), os.path.join(args.dst, shard),
                     device, stats)
        out_gb = os.path.getsize(os.path.join(args.dst, shard)) / 2**30
        print(f"[{i}/{len(shards)}] {shard}  {out_gb:.2f} GiB  "
              f"{time.time() - t1:.1f}s  (total {(time.time() - t0) / 60:.1f}m)",
              flush=True)

    print(f"\nconverted {stats['tensors']} expert weights in {(time.time() - t0) / 60:.1f} min")
    if stats["verified"]:
        print(f"exactness: mean {stats['exact_sum'] / stats['verified']:.6f}, "
              f"worst tensor {stats['exact_min']:.6f}, max rel err {stats['rel_max']:.3e}")


if __name__ == "__main__":
    VERIFY = False
    sys.exit(main())
