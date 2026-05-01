"""Probe FlashMLA wheel's ``flash_mla_with_kvcache`` (sparse FP8 decode) with
several KV-cache layout variants, to find which one the locally-installed
wheel actually accepts.

Run::

    bazelisk test //rtp_llm/models_py/modules/dsv4/decode/test:flash_mla_layout_probe_test \
        --config=cuda13 --test_output=streamed

Each variant call is wrapped in try/except so one failure doesn't mask the
others — output reports per-variant: assertion text (if any), output shape,
and finite-ness check. Pure probe; no golden compare.
"""

from __future__ import annotations

import logging
import unittest

import torch

# DSv4 packed FP8 layout constants (mirror fp8_kv_quant_decode_op.py).
NOPE_DIM = 448
ROPE_DIM = 64
NOPE_BYTES = NOPE_DIM  # fp8_e4m3 -> 1 B/elt
ROPE_BYTES = ROPE_DIM * 2  # bf16     -> 2 B/elt
SCALE_BYTES_DSV4 = 8  # 7 ue8m0 + 1 pad
ENTRY_BYTES_DSV4 = NOPE_BYTES + ROPE_BYTES + SCALE_BYTES_DSV4  # 584

# V3.2 layout (per wheel docstring).
ENTRY_BYTES_V32 = 512 + 16 + 128  # 656

# MODEL1 (per wheel docstring): 576 B per-token data region; scales 8 B/token in
# trailing block region. Effective avg per token = 576 + 8 = 584 B.
ENTRY_BYTES_MODEL1_DATA = NOPE_BYTES + ROPE_BYTES  # 576

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _try_call(label: str, q, k_cache, topk_3d, softmax_scale, head_dim_v=512):
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata  # type: ignore

    B, q_len, H, _ = q.shape
    topk = topk_3d.shape[-1]
    sched_meta, num_splits = get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=B * q_len * H,
        topk=topk,
        num_heads_q=H,
        num_heads_k=1,
        is_fp8_kvcache=True,
    )
    block_table = torch.arange(
        k_cache.shape[0], dtype=torch.int32, device=q.device
    ).unsqueeze(1)
    cache_seqlens = torch.full(
        (B,),
        min(topk, k_cache.shape[0] * k_cache.shape[1]),
        dtype=torch.int32,
        device=q.device,
    )
    print(
        f"\n[{label}] q={tuple(q.shape)} k_cache={tuple(k_cache.shape)} "
        f"k_cache.stride={tuple(k_cache.stride())} "
        f"k_batch_stride={k_cache.stride(0)} "
        f"k_batch_stride%576={k_cache.stride(0)%576} "
        f"k_batch_stride%656={k_cache.stride(0)%656}"
    )
    try:
        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=block_table,
            head_dim_v=head_dim_v,
            cache_seqlens=cache_seqlens,
            tile_scheduler_metadata=sched_meta,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=topk_3d,
            softmax_scale=softmax_scale,
        )
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out).all().item())
        nz = bool((out != 0).any().item())
        print(
            f"[{label}] OK out.shape={tuple(out.shape)} finite={finite} any_nonzero={nz}"
        )
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        # Truncate the long traceback noise.
        head = msg.splitlines()[0] if msg else type(e).__name__
        print(f"[{label}] FAIL {type(e).__name__}: {head[:200]}")
        return False


@unittest.skipUnless(DEVICE == "cuda", "requires CUDA")
class FlashMlaLayoutProbe(unittest.TestCase):
    """Sweep candidate FP8 KV cache layouts and report which the wheel accepts."""

    def setUp(self) -> None:
        try:
            import flash_mla  # type: ignore
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"flash_mla not importable: {e}")
        try:
            from importlib.metadata import version as _v

            print(
                f"\n[flash_mla] using version={_v('flash_mla')} module={flash_mla.__file__}"
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"\n[flash_mla] version probe failed: {e}; module={flash_mla.__file__}"
            )
        self.B = 1
        self.q_len = 1
        self.H = 64  # DSv4 n_heads
        self.qk_head_dim = 512  # NoPE 448 + RoPE 64 (element count)
        self.head_dim_v = 512
        self.topk = 128
        self.page_block_size = 128
        self.softmax_scale = self.qk_head_dim**-0.5

    def _q_topk(self, num_blocks: int):
        """Build a Q tensor + topk_idxs that point at valid slots."""
        q = torch.randn(
            self.B,
            self.q_len,
            self.H,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        max_slot = num_blocks * self.page_block_size
        topk_idxs = torch.randint(
            0,
            max_slot,
            (self.B, self.q_len, self.topk),
            dtype=torch.int32,
            device="cuda",
        )
        return q, topk_idxs

    # ------------------------------------------------------------------
    # Variant A: DSv4 packed (current production layout).
    # k_cache last-dim = 584 = NoPE | RoPE | per-slot scales (ue8m0).
    # k_batch_stride = page_block_size * 1 * 584 = 74752.
    # 74752 % 576 = 448, % 656 = 224 → assertion expected to fail.
    # ------------------------------------------------------------------
    def test_A_dsv4_packed_584(self):
        num_blocks = 1
        kv_packed = torch.zeros(
            num_blocks,
            self.page_block_size,
            ENTRY_BYTES_DSV4,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_packed.random_(0, 255)
        kv_4d = kv_packed.unsqueeze(-2)  # (B, page_block, 1, 584)
        q, topk = self._q_topk(num_blocks)
        _try_call("A: DSv4 packed 584", q, kv_4d, topk, self.softmax_scale)

    # ------------------------------------------------------------------
    # Variant B: V3.2 layout — 656 B/slot per-token (NoPE 512 fp8 |
    # 16B fp32 scales | RoPE 128B). NoPE element count = 512 (vs DSv4 448).
    # k_batch_stride = 128 * 656 = 83968 → 83968 % 656 == 0 ✓.
    # ------------------------------------------------------------------
    def test_B_v32_packed_656(self):
        num_blocks = 1
        kv_packed = torch.zeros(
            num_blocks,
            self.page_block_size,
            ENTRY_BYTES_V32,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_packed.random_(0, 255)
        kv_4d = kv_packed.unsqueeze(-2)
        q = torch.randn(
            self.B,
            self.q_len,
            self.H,
            576,  # V3.2 qk_head_dim=576
            dtype=torch.bfloat16,
            device="cuda",
        )
        max_slot = num_blocks * self.page_block_size
        topk = torch.randint(
            0,
            max_slot,
            (self.B, self.q_len, self.topk),
            dtype=torch.int32,
            device="cuda",
        )
        _try_call("B: V3.2 packed 656", q, kv_4d, topk, 576**-0.5)

    # ------------------------------------------------------------------
    # Variant C: MODEL1 docstring layout — per-block split.
    # Underlying buffer is 584 B/token total but viewed for the kernel as
    # (num_blocks, page_block_size, 1, 576) (data region only).
    # k_batch_stride = 128 * 576 = 73728 → 73728 % 576 == 0 ✓.
    # If the kernel reads scales from offset 576*page_block_size within the
    # block, this is the DeepSeek MODEL1 contract.
    # ------------------------------------------------------------------
    def test_C_model1_blocksplit_view576(self):
        num_blocks = 1
        # Allocate the full 584 B/token region so trailing scales are mapped.
        full_block_bytes = self.page_block_size * ENTRY_BYTES_DSV4  # 74752
        underlying = torch.zeros(
            num_blocks,
            full_block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        underlying.random_(0, 255)
        # View only the data region (576 B/token).
        data_region = underlying[:, : self.page_block_size * ENTRY_BYTES_MODEL1_DATA]
        kv_4d = data_region.view(
            num_blocks, self.page_block_size, 1, ENTRY_BYTES_MODEL1_DATA
        )
        # NB: kv_4d.stride(0) must be 73728 here (data region only). Print
        # confirms whether torch keeps the original underlying stride.
        q, topk = self._q_topk(num_blocks)
        _try_call(
            "C: MODEL1 view 576 (block-split)", q, kv_4d, topk, self.softmax_scale
        )

    # ------------------------------------------------------------------
    # Variant D: pad each slot to 640 B (next 64-multiple of 584). 128*640=
    # 81920. 81920 % 576 = 128, % 656 = 384 → assertion still fails. Probe
    # to confirm "% 576 / % 656" is really the constraint.
    # ------------------------------------------------------------------
    def test_D_pad_slot_640(self):
        num_blocks = 1
        kv_packed = torch.zeros(
            num_blocks,
            self.page_block_size,
            640,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_packed.random_(0, 255)
        kv_4d = kv_packed.unsqueeze(-2)
        q, topk = self._q_topk(num_blocks)
        _try_call("D: pad slot 640", q, kv_4d, topk, self.softmax_scale)

    # ------------------------------------------------------------------
    # Variant E: keep 584 B/slot, change page_block_size to 72 (smallest
    # value such that 584 * page_block_size is a multiple of 576).
    # 72*584=42048, 42048%576=0 ✓.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Variant F: user-requested probe — page_block_size=256, slot=584.
    # 256 * 584 = 149504. 149504 % 576 = 320 ≠ 0  → expect stride assertion.
    # ------------------------------------------------------------------
    def test_F_page_block_256_slot_584(self):
        num_blocks = 1
        page_block_size = 256
        kv_packed = torch.zeros(
            num_blocks,
            page_block_size,
            ENTRY_BYTES_DSV4,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_packed.random_(0, 255)
        kv_4d = kv_packed.unsqueeze(-2)
        q = torch.randn(
            self.B,
            self.q_len,
            self.H,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        topk = torch.randint(
            0,
            num_blocks * page_block_size,
            (self.B, self.q_len, 128),
            dtype=torch.int32,
            device="cuda",
        )
        _try_call(
            "F: page_block 256 × slot 584",
            q,
            kv_4d,
            topk,
            self.softmax_scale,
        )

    # ------------------------------------------------------------------
    # Variant G: nearest 72-multiple to 256 → 288.
    # 288 * 584 = 168192, 168192 % 576 = 0  → stride should pass.
    # topk=128 → 128 % 64 = 0 → topk assertion should pass too.
    # ------------------------------------------------------------------
    def test_G_page_block_288_slot_584_topk128(self):
        num_blocks = 1
        page_block_size = 288
        kv_packed = torch.zeros(
            num_blocks,
            page_block_size,
            ENTRY_BYTES_DSV4,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_packed.random_(0, 255)
        kv_4d = kv_packed.unsqueeze(-2)
        q = torch.randn(
            self.B,
            self.q_len,
            self.H,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        topk = torch.randint(
            0,
            num_blocks * page_block_size,
            (self.B, self.q_len, 128),
            dtype=torch.int32,
            device="cuda",
        )
        _try_call(
            "G: page_block 288 × slot 584 (topk=128)",
            q,
            kv_4d,
            topk,
            self.softmax_scale,
        )

    # ------------------------------------------------------------------
    # Variant H: per assertion string "Currently page_block_size must be 64"
    # found inside the .so, the SM100 sparse FP8 decode kernel is hardcoded
    # for page_block_size=64.  64*584=37376 — `% 576 = 512` — but the kernel
    # may take a different stride path when page_block matches its hardcoded
    # value.
    # ------------------------------------------------------------------
    def test_H_page_block_64_slot_584(self):
        num_blocks = 2  # need >1 so topk has slots; 2*64*584=74752
        page_block_size = 64
        kv_packed = torch.zeros(
            num_blocks,
            page_block_size,
            ENTRY_BYTES_DSV4,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_packed.random_(0, 255)
        kv_4d = kv_packed.unsqueeze(-2)
        q = torch.randn(
            self.B,
            self.q_len,
            self.H,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        topk = torch.randint(
            0,
            num_blocks * page_block_size,
            (self.B, self.q_len, 128),
            dtype=torch.int32,
            device="cuda",
        )
        _try_call(
            "H: page_block 64 × slot 584",
            q,
            kv_4d,
            topk,
            self.softmax_scale,
        )

    def test_E_page_block_72_slot_584(self):
        num_blocks = 1
        page_block_size = 72
        kv_packed = torch.zeros(
            num_blocks,
            page_block_size,
            ENTRY_BYTES_DSV4,
            dtype=torch.uint8,
            device="cuda",
        )
        kv_packed.random_(0, 255)
        kv_4d = kv_packed.unsqueeze(-2)
        q = torch.randn(
            self.B,
            self.q_len,
            self.H,
            self.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        max_slot = num_blocks * page_block_size
        topk = torch.randint(
            0,
            max_slot,
            (self.B, self.q_len, min(self.topk, max_slot)),
            dtype=torch.int32,
            device="cuda",
        )
        _try_call(
            "E: page_block 72 × slot 584",
            q,
            kv_4d,
            topk,
            self.softmax_scale,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
