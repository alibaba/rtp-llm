"""CUDA-hidden exact-byte CPU contracts for the dispatch packet sidecar."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from unittest import mock

import torch

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must stay hidden"
os.environ["TRITON_INTERPRET"] = "1"  # Must precede sidecar/Triton import.

HERE = pathlib.Path(__file__).resolve().parent
WT = HERE.parents[4]
PACK = HERE / "_nccl_ep_mxfp8_dispatch_pack.py"
STRATEGY = HERE / "strategies/nccl_ep_mxfp8.py"
PKG = "rtp_llm.models_py.modules.dsv4.moe"
SPKG = PKG + ".strategies"
CHECKS = []


def check(name, condition):
    CHECKS.append((name, bool(condition)))
    if not condition:
        raise AssertionError(name)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


# Source-load the actual legacy method with only its unrelated heavy siblings
# doubled.  This is not a copied reference implementation.
for name in (
    "rtp_llm",
    "rtp_llm.models_py",
    "rtp_llm.models_py.modules",
    "rtp_llm.models_py.modules.dsv4",
    PKG,
    SPKG,
):
    package(name)
prof = types.ModuleType("rtp_llm.models_py.modules.dsv4._profiler")
prof.record_function_range = lambda *_a, **_k: None
sys.modules[prof.__name__] = prof
combine = types.ModuleType(PKG + "._nccl_ep_mxfp8_combine")
combine.SCALE_BLOCK, combine.mxfp8_dequant_peer_sum = 32, lambda *_a, **_k: None
sys.modules[combine.__name__] = combine
warmup = types.ModuleType(PKG + ".warmup_sync")
warmup.cuda_graph_warmup_forward_enabled = lambda: False
sys.modules[warmup.__name__] = warmup
plan = types.ModuleType(PKG + ".forward_ep_plan")
plan.current_scope = lambda: None
sys.modules[plan.__name__] = plan
base = types.ModuleType(SPKG + ".base")
base.MoeCfg = object
base.RoutedExpertsStrategy = object
base.register_strategy = lambda cls: cls
sys.modules[base.__name__] = base
gfp4 = types.ModuleType(SPKG + ".grouped_fp4")
gfp4.GroupedFP4Strategy, gfp4._has_fp8_fp4_grouped_kernel = object, lambda: False
sys.modules[gfp4.__name__] = gfp4
loop = types.ModuleType(SPKG + ".local_loop")
loop.LocalLoopStrategy = object
sys.modules[loop.__name__] = loop
utils = package("rtp_llm.models_py.utils")
arch = types.ModuleType("rtp_llm.models_py.utils.arch")
arch.is_sm120 = lambda *_a, **_k: False
sys.modules[arch.__name__] = arch
setattr(utils, "arch", arch)

packet = load(PKG + "._nccl_ep_mxfp8_dispatch_pack", PACK)
legacy = load(SPKG + "._legacy_pack_actual", STRATEGY)


def source_inputs(n, d=128, k=6):
    q = torch.arange(n * d, dtype=torch.int64).reshape(n, d).to(torch.uint8)
    scale = (
        torch.arange(n * (d // 32), dtype=torch.int64).reshape(n, d // 32) + 101
    ).to(torch.uint8)
    ids = torch.empty((n, k), dtype=torch.int64)
    for row in range(n):
        ids[row] = torch.tensor([-1, 0, 7, 8, 15, 31], dtype=torch.int64)
        if row % 3 == 1:
            ids[row, 0] = -9
            ids[row, 2] = 16
    weights = torch.arange(n * k, dtype=torch.float32).reshape(n, k) / 7.0
    if n:
        # Positive and signaling-ish NaN payload bytes are intentionally data,
        # not a numerical-quality case.  The packet must reproduce legacy bytes.
        raw = weights.view(torch.int32)
        raw[0, 0] = 0x7FC01234
        raw[0, 1] = 0x7FA00001
        raw[0, 2] = -1
        raw[0, 3] = -2147483648
        # An owned zero route weight remains an owned ID; masking must not
        # discard it before matching the legacy packet writer.
        weights[0, 4] = 0.0
    return q.contiguous(), scale.contiguous(), weights.contiguous(), ids.contiguous()


def old_pack(q, scale, weights, ids, experts_per_rank=8):
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.mxfp8_quantize = lambda x, is_sf_swizzled_layout=False: (q, scale)
    before = sys.modules.get("flashinfer")
    sys.modules["flashinfer"] = flashinfer
    try:
        self = types.SimpleNamespace(
            cfg=types.SimpleNamespace(n_local_experts=experts_per_rank),
            _mask_for_peer=legacy.NcclEpMxfp8Strategy._mask_for_peer,
        )
        x = torch.empty((q.shape[0], q.shape[1]), dtype=torch.bfloat16)
        return legacy.NcclEpMxfp8Strategy._pack_dispatch(
            self,
            x,
            weights,
            ids,
            q.shape[1],
            scale.shape[1],
            ids.shape[1],
            q.shape[1] + scale.shape[1] + 8 * ids.shape[1],
            4,
        )
    finally:
        if before is None:
            sys.modules.pop("flashinfer", None)
        else:
            sys.modules["flashinfer"] = before


for n in (0, 1, 1024):
    q, scale, weights, ids = source_inputs(n)
    actual = old_pack(q, scale, weights, ids)
    oracle = packet.pack_dispatch_packet_torch(
        q, scale, weights, ids, experts_per_rank=8
    )
    check("legacy-byte-equality-n%d" % n, torch.equal(oracle, actual))
    check("shape-n%d" % n, tuple(oracle.shape) == (4 * n, 180))
    if n:
        packed = oracle.view(4, n, 180)
        check(
            "q-replicated-n%d" % n,
            all(torch.equal(packed[dst, :, :128], q) for dst in range(4)),
        )
        check(
            "scale-replicated-n%d" % n,
            all(torch.equal(packed[dst, :, 128:132], scale) for dst in range(4)),
        )
        # All destinations own exactly their legacy route lanes; zero weight is
        # still retained when its ID is owned, matching the existing pack loop.
        for dst in range(4):
            got_w = packed[dst, :, 132:156].view(torch.float32)
            got_i = packed[dst, :, 156:].view(torch.int32).to(torch.int64)
            owned = (ids >= 0) & (torch.div(ids, 8, rounding_mode="floor") == dst)
            expected_i = (
                torch.where(owned, ids, torch.full_like(ids, -1))
                .to(torch.int32)
                .to(torch.int64)
            )
            check("ids-owner-n%d-d%d" % (n, dst), torch.equal(got_i, expected_i))
            check(
                "weights-raw-n%d-d%d" % (n, dst),
                torch.equal(
                    got_w.view(torch.uint8),
                    torch.where(owned, weights, torch.zeros_like(weights)).view(
                        torch.uint8
                    ),
                ),
            )

q, scale, weights, ids = source_inputs(2)


def noncontiguous_same_shape(t):
    return t.t().contiguous().t()


for name, bad in (
    ("q", noncontiguous_same_shape(q)),
    ("scale", noncontiguous_same_shape(scale)),
    ("weights", noncontiguous_same_shape(weights)),
    ("ids", noncontiguous_same_shape(ids)),
):
    args = dict(q=q, scale=scale, weights=weights, ids=ids, experts_per_rank=8)
    args[name] = bad
    try:
        packet.pack_dispatch_packet_torch(**args)
    except ValueError as exc:
        check("noncontiguous-rejected-" + name, "contiguous" in str(exc))
    else:
        check("noncontiguous-rejected-" + name, False)

# ABI gates deliberately match the production MXFP8 dispatch packet rather
# than accepting a broader layout whose typed legacy router fields misalign.
for name, args, text in (
    (
        "d128",
        dict(
            q=q[:, :32].contiguous(),
            scale=scale[:, :1].contiguous(),
            weights=weights,
            ids=ids,
            experts_per_rank=8,
        ),
        "divisible by 128",
    ),
    (
        "experts-positive-int",
        dict(q=q, scale=scale, weights=weights, ids=ids, experts_per_rank=0),
        "positive integer",
    ),
    (
        "experts-type",
        dict(q=q, scale=scale, weights=weights, ids=ids, experts_per_rank=8.0),
        "positive integer",
    ),
    (
        "topk-positive",
        dict(
            q=q,
            scale=scale,
            weights=weights[:, :0].contiguous(),
            ids=ids[:, :0].contiguous(),
            experts_per_rank=8,
        ),
        "topk",
    ),
):
    try:
        packet.pack_dispatch_packet_torch(**args)
    except ValueError as exc:
        check("abi-rejected-" + name, text in str(exc))
    else:
        check("abi-rejected-" + name, False)

# Parent-owned hook: verify actual caller OFF remains legacy and ON selects
# the sidecar, using only a CPU oracle leaf double in this CUDA-hidden test.
q_hook, scale_hook, weights_hook, ids_hook = source_inputs(2)
with_env = os.environ.get("DSV4_NCCL_EP_MXFP8_DISPATCH_PACK")
os.environ["DSV4_NCCL_EP_MXFP8_DISPATCH_PACK"] = "0"
off_hook = old_pack(q_hook, scale_hook, weights_hook, ids_hook)
with mock.patch.object(
    packet, "pack_dispatch_packet", wraps=packet.pack_dispatch_packet_torch
) as on_spy:
    os.environ["DSV4_NCCL_EP_MXFP8_DISPATCH_PACK"] = "1"
    on_hook = old_pack(q_hook, scale_hook, weights_hook, ids_hook)
    check("actual-caller-on-selected-sidecar-cpu-double", on_spy.call_count == 1)
check("actual-caller-off-on-byte-equal", torch.equal(off_hook, on_hook))
if with_env is None:
    os.environ.pop("DSV4_NCCL_EP_MXFP8_DISPATCH_PACK", None)
else:
    os.environ["DSV4_NCCL_EP_MXFP8_DISPATCH_PACK"] = with_env

# Regression witness for the pre-fix token-major mapping. N=2 is essential:
# launch row 1 formerly read (dst=1,row=0) but legacy output row 1 is (dst=0,row=1).
q2, scale2, weights2, ids2 = source_inputs(2)
legacy2 = old_pack(q2, scale2, weights2, ids2)
payload2 = legacy2.shape[1]
buggy = torch.empty_like(legacy2)
for row_dst in range(8):
    old_dst, old_row = row_dst % 4, row_dst // 4
    buggy[row_dst].copy_(legacy2[old_dst * 2 + old_row])
check("old-token-major-red-reproduction-n2", not torch.equal(buggy, legacy2))

# Exercise the ACTUAL decorated kernel directly under Triton's CPU interpreter.
# This deliberately bypasses only pack_dispatch_packet's production CUDA guard.
if packet.triton is None:
    raise AssertionError("TRITON_INTERPRET import did not provide Triton")
interpreter_out = torch.empty_like(legacy2)
packet._dispatch_packet_write_kernel[(8, packet.triton.cdiv(132, 256))](
    q2,
    scale2,
    weights2,
    ids2,
    interpreter_out,
    2,
    128,
    4,
    6,
    payload2,
    8,
    WORLD=4,
    BLOCK_BYTES=256,
    BLOCK_K=8,
    num_warps=1,
)
check(
    "actual-triton-interpreter-fixed-peer-major-n2",
    torch.equal(interpreter_out, legacy2),
)
check(
    "actual-triton-interpreter-all-four-dest-n2",
    torch.equal(interpreter_out.view(4, 2, payload2), legacy2.view(4, 2, payload2)),
)

old = os.environ.pop("DSV4_NCCL_EP_MXFP8_DISPATCH_PACK", None)
check("strict-default-off", not packet.dispatch_pack_experimental_enabled())
os.environ["DSV4_NCCL_EP_MXFP8_DISPATCH_PACK"] = "1"
check("proposal-hook-on", packet.dispatch_pack_experimental_enabled())
if old is None:
    os.environ.pop("DSV4_NCCL_EP_MXFP8_DISPATCH_PACK", None)
else:
    os.environ["DSV4_NCCL_EP_MXFP8_DISPATCH_PACK"] = old
check(
    "parent-strategy-default-off-hook-wired",
    "DSV4_NCCL_EP_MXFP8_DISPATCH_PACK" in STRATEGY.read_text(),
)

print("dispatch packet CPU checks: %d PASS" % len(CHECKS))
