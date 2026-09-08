"""T183: M890P C128 fused-compressor warp-count microgate.

This is a standalone diagnostic, not part of the default unit-test suite.  It
uses the production B4 x 4096 all-raw prefill geometry and checks that changing
only Triton's launch warp count preserves the emitted KV bytes.
"""

from __future__ import annotations

import json

import torch

from internal_source.rtp_llm.models_py.modules.dsv4.ppu_provider import (
    register_m890p_dsv4_provider,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _compressor_vllm_triton as kernels


N_REQ = 4
SEQ = 4096
N = N_REQ * SEQ
HEAD = 512
ROPE = 64
RATIO = 128
STATE_PAGE = 256
KV_SLOTS = N // RATIO
TOKEN_STRIDE = 576
SCALE_DIM = 8


def _allocate_kv_cache() -> tuple[torch.Tensor, torch.Tensor]:
    block_bytes = KV_SLOTS * (TOKEN_STRIDE + SCALE_DIM)
    backing = torch.zeros((2, block_bytes), dtype=torch.uint8, device="cuda")
    cache = backing.as_strided(
        size=(2, KV_SLOTS, TOKEN_STRIDE),
        stride=(block_bytes, TOKEN_STRIDE, 1),
    )
    return cache, backing


def main() -> None:
    torch.cuda.set_device(0)
    register_m890p_dsv4_provider(device_name="ZW-M890P", ep_size=8)
    generator = torch.Generator(device="cuda").manual_seed(20260826)

    positions = torch.arange(SEQ, dtype=torch.int64, device="cuda").repeat(N_REQ)
    token_to_req = torch.arange(N_REQ, dtype=torch.int32, device="cuda").repeat_interleave(SEQ)
    state_slots = torch.zeros(N, dtype=torch.int64, device="cuda")
    block_table = torch.zeros((N_REQ, SEQ // STATE_PAGE), dtype=torch.int32, device="cuda")
    kv_slots = torch.full((N,), -1, dtype=torch.int64, device="cuda")
    boundary = ((positions + 1) % RATIO) == 0
    boundary_ordinal = torch.arange(KV_SLOTS, dtype=torch.int64, device="cuda")
    kv_slots[boundary] = KV_SLOTS + boundary_ordinal

    state_cache = torch.zeros((2, STATE_PAGE, 2 * HEAD), dtype=torch.float32, device="cuda")
    kv_raw = torch.randn((N, HEAD), dtype=torch.float32, device="cuda", generator=generator) * 0.1
    score_raw = torch.randn((N, HEAD), dtype=torch.float32, device="cuda", generator=generator) * 0.1
    ape = torch.randn((RATIO, HEAD), dtype=torch.float32, device="cuda", generator=generator) * 0.1
    rms = torch.ones(HEAD, dtype=torch.bfloat16, device="cuda")
    cos_sin = torch.zeros((SEQ, ROPE), dtype=torch.float32, device="cuda")
    cos_sin[:, : ROPE // 2] = 1.0
    seq_start_per_req = torch.zeros(N_REQ, dtype=torch.int64, device="cuda")
    cu_seq_per_req = torch.arange(N_REQ + 1, dtype=torch.int64, device="cuda") * SEQ

    original_picker = kernels._fused_num_warps
    results = []
    reference = None
    try:
        for num_warps in (4, 8, 16):
            kernels._fused_num_warps = lambda _h, _r, _cfg, nw=num_warps: nw
            kv_cache, backing = _allocate_kv_cache()

            def launch() -> None:
                kernels.run_fused_compress_kv_write(
                    state_cache,
                    token_to_req,
                    positions,
                    state_slots,
                    block_table,
                    rms,
                    1e-6,
                    cos_sin,
                    kv_cache,
                    kv_slots,
                    kv_raw,
                    score_raw,
                    ape,
                    0,
                    head_dim=HEAD,
                    rope_head_dim=ROPE,
                    compress_ratio=RATIO,
                    overlap=False,
                    seq_start_per_req=seq_start_per_req,
                    cu_seq_per_req=cu_seq_per_req,
                    state_tokens_per_block=STATE_PAGE,
                )

            for _ in range(3):
                launch()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                launch()
            end.record()
            end.synchronize()
            mean_ms = start.elapsed_time(end) / 20.0
            output = backing[1].clone()
            byte_equal = True if reference is None else bool(torch.equal(output, reference))
            if reference is None:
                reference = output
            results.append(
                {"num_warps": num_warps, "mean_ms": mean_ms, "byte_equal_to_w4": byte_equal}
            )
    finally:
        kernels._fused_num_warps = original_picker

    print(json.dumps({"shape": [N_REQ, SEQ, HEAD], "results": results}, indent=2))


if __name__ == "__main__":
    main()
