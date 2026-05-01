"""Unit tests for DSv4 prefill paged KV write.

Verifies that ``write_paged_kv_per_req`` and ``write_paged_kv_swa_per_req``
write into the BlockPool layout exactly the way the legacy gather/scatter
flow does, so swapping the dense ``register_buffer`` intermediate for
direct paged writes is a no-op semantic change.

Pure-torch ops; CPU and CUDA both work.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.paged_kv_write import (
    _pool_as_dense_view,
    gather_paged_kv_per_req,
    write_paged_kv_per_req,
    write_paged_kv_swa_per_req,
)


def _make_pool(num_blocks: int, tokens_per_block: int, head_dim: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Allocate a BlockPool tensor in the same layout DSV4ConfigCreator emits.

    [num_blocks, tokens_per_block * head_dim * esize] uint8 — zero-initialized.
    """
    esize = torch.empty((), dtype=dtype).element_size()
    stride = tokens_per_block * head_dim * esize
    return torch.zeros(num_blocks, stride, dtype=torch.uint8, device=device)


def _tagged(rows: int, head_dim: int, base: float, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    r_off = torch.arange(rows, device=device, dtype=torch.float32).view(rows, 1)
    d_off = (
        torch.arange(head_dim, device=device, dtype=torch.float32).view(1, head_dim)
        * 0.01
    )
    return (base + r_off + d_off).to(dtype)


class TestPoolView(unittest.TestCase):
    def test_view_shape_and_stride(self):
        # CSA_KV: 64 entries × 512 head_dim × 2 bytes = 65536 bytes/block
        pool = _make_pool(num_blocks=4, tokens_per_block=64, head_dim=512)
        view = _pool_as_dense_view(pool, 64, 512, torch.bfloat16)
        self.assertEqual(view.shape, (4, 64, 512))
        self.assertEqual(view.dtype, torch.bfloat16)

    def test_view_rejects_stride_mismatch(self):
        # stride 1024 != 64 * 512 * 2 = 65536
        pool = torch.zeros(4, 1024, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            _pool_as_dense_view(pool, 64, 512, torch.bfloat16)

    def test_view_rejects_wrong_dtype(self):
        pool = torch.zeros(4, 65536, dtype=torch.float32)
        with self.assertRaises(ValueError):
            _pool_as_dense_view(pool, 64, 512, torch.bfloat16)


class TestPagedKvWriteSequential(unittest.TestCase):
    """CSA / HCA / INDEXER compressed-K writes are sequential
    (no ring): global slot = offset + i.
    """

    def test_single_block_csa(self):
        # CSA_KV: 64 tokens/block × head_dim=512 × bf16
        device = torch.device("cpu")
        tpb, hd = 64, 512
        pool = _make_pool(num_blocks=8, tokens_per_block=tpb, head_dim=hd, device=device)
        # Request gets 4 physical blocks: [3, 1, 5, 7]
        bt = torch.tensor([3, 1, 5, 7], dtype=torch.int32, device=device)
        # Write 10 tokens starting at slot 0 (fresh prefill, prefix=0)
        src = _tagged(10, hd, base=100.0, device=device)
        write_paged_kv_per_req(src, pool, bt, slot_offset=0, tokens_per_block=tpb, head_dim=hd)
        # Read back via gather
        slots = torch.arange(10, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        self.assertTrue(torch.equal(got, src))

    def test_multi_block_csa_continuation(self):
        """Continuation prefill: offset starts mid-block, writes span block boundary."""
        device = torch.device("cpu")
        tpb, hd = 64, 512
        pool = _make_pool(num_blocks=8, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([3, 1, 5, 7], dtype=torch.int32, device=device)
        # Pre-populate first block by writing 50 tokens at offset 0
        prefill_src = _tagged(50, hd, base=10.0, device=device)
        write_paged_kv_per_req(prefill_src, pool, bt, slot_offset=0, tokens_per_block=tpb, head_dim=hd)
        # Continuation: 30 more tokens starting at slot 50 → spans into second block
        cont_src = _tagged(30, hd, base=200.0, device=device)
        write_paged_kv_per_req(cont_src, pool, bt, slot_offset=50, tokens_per_block=tpb, head_dim=hd)

        # Verify: first 50 still match prefill_src, next 30 match cont_src
        slots = torch.arange(80, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        self.assertTrue(torch.equal(got[:50], prefill_src))
        self.assertTrue(torch.equal(got[50:], cont_src))

    def test_hca_two_entries_per_block(self):
        # HCA: only 2 entries/block — exercises the multi-block dispatch
        device = torch.device("cpu")
        tpb, hd = 2, 512
        pool = _make_pool(num_blocks=16, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32, device=device)
        src = _tagged(8, hd, base=500.0, device=device)
        # 8 entries, 2 per block, span 4 blocks
        write_paged_kv_per_req(src, pool, bt, slot_offset=0, tokens_per_block=tpb, head_dim=hd)
        slots = torch.arange(8, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        self.assertTrue(torch.equal(got, src))


class TestPagedKvWriteSwaRing(unittest.TestCase):
    """SWA writes use ring modulo: slot = (start_pos + i) % win.

    BlockPool SWA layout: each request gets fixed_blocks=2 of
    tokens_per_block=256 each, but only the first block's [0..win) slot
    range is actively used (V4 sliding_window=128).
    """

    def test_fresh_prefill_short(self):
        device = torch.device("cpu")
        win, tpb, hd = 8, 16, 4  # tiny shapes for unit-test speed
        pool = _make_pool(num_blocks=4, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([2, 3], dtype=torch.int32, device=device)
        # Fresh prefill, 5 tokens → write to ring slots [0..5)
        src = _tagged(5, hd, base=1.0, device=device)
        write_paged_kv_swa_per_req(
            src, pool, bt, start_pos=0, seqlen=5, win=win, head_dim=hd, tokens_per_block=tpb
        )
        slots = torch.arange(5, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        self.assertTrue(torch.equal(got, src))

    def test_continuation_within_window(self):
        device = torch.device("cpu")
        win, tpb, hd = 8, 16, 4
        pool = _make_pool(num_blocks=4, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([2, 3], dtype=torch.int32, device=device)
        # Phase 1: 3 tokens at start_pos=0 → ring slots 0,1,2
        ph1 = _tagged(3, hd, base=10.0, device=device)
        write_paged_kv_swa_per_req(ph1, pool, bt, start_pos=0, seqlen=3, win=win, head_dim=hd, tokens_per_block=tpb)
        # Phase 2: 2 more tokens at start_pos=3 → ring slots 3,4
        ph2 = _tagged(2, hd, base=20.0, device=device)
        write_paged_kv_swa_per_req(ph2, pool, bt, start_pos=3, seqlen=2, win=win, head_dim=hd, tokens_per_block=tpb)
        slots = torch.arange(5, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        self.assertTrue(torch.equal(got[:3], ph1))
        self.assertTrue(torch.equal(got[3:], ph2))

    def test_ring_wrap(self):
        """seqlen > win: last `win` rows survive at correct ring slots."""
        device = torch.device("cpu")
        win, tpb, hd = 4, 8, 4
        pool = _make_pool(num_blocks=2, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([1], dtype=torch.int32, device=device)
        # Write 10 tokens fresh: first 6 get overwritten, last 4 survive
        src = _tagged(10, hd, base=0.0, device=device)
        write_paged_kv_swa_per_req(src, pool, bt, start_pos=0, seqlen=10, win=win, head_dim=hd, tokens_per_block=tpb)
        slots = torch.arange(win, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        # ring slot s ∈ [0, win): last token to land there has global pos
        # = max(t in [0,10) where t%win==s) = (10-1) - ((10-1-s) % win)
        # For win=4, slot 0 → t=8 (last t with t%4==0 in [0,10) is 8); slot 1 → 9; slot 2 → 6; slot 3 → 7.
        expected = torch.stack([src[8], src[9], src[6], src[7]])
        self.assertTrue(torch.equal(got, expected))

    def test_continuation_with_wrap(self):
        """Continuation prefill that crosses ring boundary."""
        device = torch.device("cpu")
        win, tpb, hd = 4, 8, 4
        pool = _make_pool(num_blocks=2, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([1], dtype=torch.int32, device=device)
        # Prefix 3 tokens at slots 0,1,2
        ph1 = _tagged(3, hd, base=10.0, device=device)
        write_paged_kv_swa_per_req(ph1, pool, bt, start_pos=0, seqlen=3, win=win, head_dim=hd, tokens_per_block=tpb)
        # Continuation 4 more tokens at start_pos=3 → slots 3,0,1,2 (wrap)
        ph2 = _tagged(4, hd, base=20.0, device=device)
        write_paged_kv_swa_per_req(ph2, pool, bt, start_pos=3, seqlen=4, win=win, head_dim=hd, tokens_per_block=tpb)
        slots = torch.arange(win, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        # Slot 0 ← ph2[1] (g=4, last); slot 1 ← ph2[2] (g=5); slot 2 ← ph2[3] (g=6); slot 3 ← ph2[0] (g=3).
        expected = torch.stack([ph2[1], ph2[2], ph2[3], ph2[0]])
        self.assertTrue(torch.equal(got, expected))


class TestPagedKvParity(unittest.TestCase):
    """The dense gather/scatter flow this op replaces would compose as
    ``scatter(dense) → gather(dense)`` — verify our paged write +
    paged gather composes to the same semantics.
    """

    def test_indexer_kv_parity(self):
        """INDEXER_KV: tpb=64, head_dim=128 (idx_head_dim), bf16."""
        device = torch.device("cpu")
        tpb, hd = 64, 128
        pool = _make_pool(num_blocks=8, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([2, 5, 1, 7], dtype=torch.int32, device=device)
        prefill = _tagged(80, hd, base=1.0, device=device)
        # Write 80 tokens fresh
        write_paged_kv_per_req(prefill, pool, bt, slot_offset=0, tokens_per_block=tpb, head_dim=hd)
        # Gather back arbitrary indices: out-of-order, repeated
        idx = torch.tensor([5, 0, 79, 64, 50, 0], device=device)
        got = gather_paged_kv_per_req(pool, bt, idx, tpb, hd)
        self.assertTrue(torch.equal(got, prefill[idx]))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestPagedKvCuda(unittest.TestCase):
    def test_csa_kv_cuda_roundtrip(self):
        device = torch.device("cuda:0")
        tpb, hd = 64, 512
        pool = _make_pool(num_blocks=16, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([3, 11, 5, 7], dtype=torch.int32, device=device)
        src = torch.randn(120, hd, dtype=torch.bfloat16, device=device)
        write_paged_kv_per_req(src, pool, bt, slot_offset=0, tokens_per_block=tpb, head_dim=hd)
        slots = torch.arange(120, device=device)
        got = gather_paged_kv_per_req(pool, bt, slots, tpb, hd)
        self.assertTrue(torch.equal(got, src))

    def test_swa_cuda_ring(self):
        device = torch.device("cuda:0")
        win, tpb, hd = 128, 256, 512
        pool = _make_pool(num_blocks=4, tokens_per_block=tpb, head_dim=hd, device=device)
        bt = torch.tensor([1, 2], dtype=torch.int32, device=device)
        src = torch.randn(80, hd, dtype=torch.bfloat16, device=device)
        # Continuation: prefix 50 tokens, write 80 new tokens at start_pos=50
        # Total 130 > win=128: ring wraps, slot 0 holds token 128 (g=128),
        # slot 1 → g=129, slot 2 → g=2, slot 3 → g=3, ...
        write_paged_kv_swa_per_req(src, pool, bt, start_pos=50, seqlen=80, win=win, head_dim=hd, tokens_per_block=tpb)
        # Slot 78 ← g=78 (50 + 28) which is src[28]
        slot_78 = gather_paged_kv_per_req(pool, bt, torch.tensor([78], device=device), tpb, hd)
        self.assertTrue(torch.equal(slot_78[0], src[28]))


if __name__ == "__main__":
    unittest.main()
