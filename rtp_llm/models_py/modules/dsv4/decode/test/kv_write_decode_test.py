"""Unit tests for DSv4 decode KV-write ops.

Pure-torch ops, runnable on CPU and CUDA. CUDA-only tests gated by
``unittest.skipUnless(torch.cuda.is_available(), ...)``.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    build_decode_metadata,
)
from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
    write_compressed_k_decode,
    write_swa_k_decode,
)


def _tagged(
    rows: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build a recognizable [rows, head_dim] tensor — row r's payload is
    base + r + 0.01 * arange(head_dim). Distinct enough for == comparisons
    even in bf16 at the magnitudes we use.
    """
    r_off = torch.arange(rows, device=device, dtype=torch.float32).view(rows, 1)
    d_off = (
        torch.arange(head_dim, device=device, dtype=torch.float32).view(1, head_dim)
        * 0.01
    )
    return (base + r_off + d_off).to(dtype)


class TestWriteSwaKDecode(unittest.TestCase):
    def test_b1_qlen1_basic(self):
        """B=1, q_len=1, window=8, head_dim=4, start_pos=3 → slot 3."""
        device = torch.device("cpu")
        B, q_len, window, head_dim = 1, 1, 8, 4
        T = B * q_len
        head_dim_dtype = torch.bfloat16

        swa = torch.zeros(B, window, head_dim, dtype=head_dim_dtype, device=device)
        k_state = _tagged(T, head_dim, base=100.0, device=device)  # [1, 4]
        slot = torch.tensor([3], dtype=torch.int32, device=device)

        write_swa_k_decode(k_state, slot, swa)

        # Slot 3 written.
        self.assertTrue(torch.equal(swa[0, 3], k_state[0]))
        # All other rows untouched.
        for s in range(window):
            if s == 3:
                continue
            self.assertTrue(
                torch.all(swa[0, s] == 0),
                msg=f"swa[0, {s}] should be zero, got {swa[0, s]}",
            )

    def test_b4_different_start_pos(self):
        """B=4, window=8, start_pos=[0,5,7,10] → flat slots [0, 13, 23, 26]."""
        device = torch.device("cpu")
        B, q_len, window, head_dim = 4, 1, 8, 4
        T = B * q_len
        head_dim_dtype = torch.bfloat16

        swa = torch.zeros(B, window, head_dim, dtype=head_dim_dtype, device=device)
        k_state = _tagged(T, head_dim, base=200.0, device=device)
        # Per-request ring positions: [0, 5, 7, 2]; per-request stride = window = 8.
        slot = torch.tensor([0, 13, 23, 26], dtype=torch.int32, device=device)

        write_swa_k_decode(k_state, slot, swa)

        expected = {
            (0, 0): k_state[0],
            (1, 5): k_state[1],
            (2, 7): k_state[2],
            (3, 2): k_state[3],
        }
        for r in range(B):
            for s in range(window):
                if (r, s) in expected:
                    self.assertTrue(
                        torch.equal(swa[r, s], expected[(r, s)]),
                        msg=f"swa[{r}, {s}] mismatch: {swa[r, s]} vs {expected[(r, s)]}",
                    )
                else:
                    self.assertTrue(
                        torch.all(swa[r, s] == 0),
                        msg=f"swa[{r}, {s}] should be zero, got {swa[r, s]}",
                    )

    def test_qlen3_spec_decode(self):
        """B=2, q_len=3, window=8, start_pos=[5, 9].

        Per-request abs positions:
          req 0: 5, 6, 7   → ring slots 5, 6, 7   → flat 5, 6, 7
          req 1: 9, 10, 11 → ring slots 1, 2, 3   → flat 9, 10, 11
        """
        device = torch.device("cpu")
        B, q_len, window, head_dim = 2, 3, 8, 4
        T = B * q_len
        head_dim_dtype = torch.bfloat16

        swa = torch.zeros(B, window, head_dim, dtype=head_dim_dtype, device=device)
        k_state = _tagged(T, head_dim, base=300.0, device=device)
        slot = torch.tensor([5, 6, 7, 9, 10, 11], dtype=torch.int32, device=device)

        write_swa_k_decode(k_state, slot, swa)

        # Each request's expected (ring_slot, k_state index) pairs.
        layout = [
            (0, 5, 0),
            (0, 6, 1),
            (0, 7, 2),
            (1, 1, 3),
            (1, 2, 4),
            (1, 3, 5),
        ]
        seen = set()
        for r, s, ki in layout:
            self.assertTrue(
                torch.equal(swa[r, s], k_state[ki]),
                msg=f"swa[{r}, {s}] mismatch (k_state[{ki}])",
            )
            seen.add((r, s))
        for r in range(B):
            for s in range(window):
                if (r, s) in seen:
                    continue
                self.assertTrue(
                    torch.all(swa[r, s] == 0),
                    msg=f"swa[{r}, {s}] should be zero, got {swa[r, s]}",
                )

    def test_uses_metadata_builder(self):
        """End-to-end: build slot mapping via build_decode_metadata, write
        through the op, compare against a pure-Python oracle."""
        device = torch.device("cpu")
        B, q_len, window, head_dim = 4, 2, 8, 4
        head_dim_dtype = torch.bfloat16

        start_pos = torch.tensor([0, 3, 7, 14], dtype=torch.int32, device=device)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=q_len,
            window_size=window,
            head_dim=head_dim,
            max_seq_len=64,
            compress_ratios=[4],
            index_topk=8,
            device=device,
        )

        T = B * q_len
        swa = torch.zeros(B, window, head_dim, dtype=head_dim_dtype, device=device)
        k_state = _tagged(T, head_dim, base=400.0, device=device)

        write_swa_k_decode(k_state, meta.slot_mapping_swa, swa)

        # Oracle: flat slot mapping per (r, s).
        oracle = torch.zeros_like(swa)
        sp = start_pos.tolist()
        for r in range(B):
            for s in range(q_len):
                ring = (sp[r] + s) % window
                t_idx = r * q_len + s
                oracle[r, ring] = k_state[t_idx]
        self.assertTrue(torch.equal(swa, oracle))

    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    def test_cuda_matches_cpu(self):
        """Parity: same inputs on CUDA produce same buffer state as CPU."""
        cpu = torch.device("cpu")
        cuda = torch.device("cuda:0")
        B, q_len, window, head_dim = 3, 2, 8, 4
        T = B * q_len

        start_pos_cpu = torch.tensor([0, 5, 13], dtype=torch.int32, device=cpu)
        start_pos_cuda = start_pos_cpu.cuda()

        meta_cpu = build_decode_metadata(
            start_pos=start_pos_cpu,
            q_len=q_len,
            window_size=window,
            head_dim=head_dim,
            max_seq_len=64,
            compress_ratios=[4],
            index_topk=8,
            device=cpu,
        )
        meta_cuda = build_decode_metadata(
            start_pos=start_pos_cuda,
            q_len=q_len,
            window_size=window,
            head_dim=head_dim,
            max_seq_len=64,
            compress_ratios=[4],
            index_topk=8,
            device=cuda,
        )

        k_cpu = _tagged(T, head_dim, base=500.0, device=cpu)
        k_cuda = k_cpu.cuda()
        swa_cpu = torch.zeros(B, window, head_dim, dtype=torch.bfloat16, device=cpu)
        swa_cuda = torch.zeros(B, window, head_dim, dtype=torch.bfloat16, device=cuda)

        write_swa_k_decode(k_cpu, meta_cpu.slot_mapping_swa, swa_cpu)
        write_swa_k_decode(k_cuda, meta_cuda.slot_mapping_swa, swa_cuda)
        self.assertTrue(torch.equal(swa_cpu, swa_cuda.cpu()))

    def test_t_total_zero_noop(self):
        """T_total == 0 → no-op; buffer untouched."""
        device = torch.device("cpu")
        swa = torch.zeros(2, 8, 4, dtype=torch.bfloat16, device=device)
        k_state = torch.zeros(0, 4, dtype=torch.bfloat16, device=device)
        slot = torch.zeros(0, dtype=torch.int32, device=device)
        write_swa_k_decode(k_state, slot, swa)
        self.assertTrue(torch.all(swa == 0))


class TestWriteCompressedKDecode(unittest.TestCase):
    def test_only_writes_on_boundary(self):
        """ratio=4, B=4, start_pos=[2, 3, 4, 11], q_len=1.

        abs+1 = [3, 4, 5, 12]; ratio=4 boundaries at abs+1 in {4, 12}.
        So requests 1 and 3 hit a boundary. With max_seq_len=16, ratio=4:
        stride = 4. In-request slots:
          req 1: in_req = 4//4 - 1 = 0 → flat 1*4 + 0 = 4
          req 3: in_req = 12//4 - 1 = 2 → flat 3*4 + 2 = 14
        """
        device = torch.device("cpu")
        ratio = 4
        B, q_len, head_dim, max_seq_len = 4, 1, 4, 16
        T = B * q_len
        T_dim = max_seq_len // ratio  # 4

        cmp_buf = torch.zeros(B, T_dim, head_dim, dtype=torch.bfloat16, device=device)
        k_state = _tagged(T, head_dim, base=600.0, device=device)
        # slot mapping with -1 sentinels for non-boundary tokens.
        slot = torch.tensor([-1, 4, -1, 14], dtype=torch.int32, device=device)

        write_compressed_k_decode(k_state, slot, cmp_buf)

        expected_writes = {
            (1, 0): k_state[1],
            (3, 2): k_state[3],
        }
        for r in range(B):
            for s in range(T_dim):
                if (r, s) in expected_writes:
                    self.assertTrue(
                        torch.equal(cmp_buf[r, s], expected_writes[(r, s)]),
                        msg=f"cmp_buf[{r}, {s}] mismatch",
                    )
                else:
                    self.assertTrue(
                        torch.all(cmp_buf[r, s] == 0),
                        msg=f"cmp_buf[{r}, {s}] should be zero, got {cmp_buf[r, s]}",
                    )

    def test_qlen3_with_mid_step_boundary(self):
        """B=1, q_len=3, ratio=4, start_pos=[2].

        abs in {2, 3, 4} → abs+1 in {3, 4, 5}. Only abs+1=4 (mid-step,
        token index t=1) is a ratio=4 boundary; in_req=0, flat slot=0.
        """
        device = torch.device("cpu")
        ratio = 4
        B, q_len, head_dim, max_seq_len = 1, 3, 4, 16
        T = B * q_len
        T_dim = max_seq_len // ratio  # 4

        cmp_buf = torch.zeros(B, T_dim, head_dim, dtype=torch.bfloat16, device=device)
        k_state = _tagged(T, head_dim, base=700.0, device=device)
        slot = torch.tensor([-1, 0, -1], dtype=torch.int32, device=device)

        write_compressed_k_decode(k_state, slot, cmp_buf)

        # Only slot 0 (t=1) written.
        self.assertTrue(torch.equal(cmp_buf[0, 0], k_state[1]))
        for s in range(1, T_dim):
            self.assertTrue(
                torch.all(cmp_buf[0, s] == 0),
                msg=f"cmp_buf[0, {s}] should be zero",
            )

    def test_uses_metadata_builder(self):
        """Build slot mapping via build_decode_metadata for ratio=4,
        write through op, verify against pure-Python oracle."""
        device = torch.device("cpu")
        ratio = 4
        B, q_len, window, head_dim, max_seq_len = 3, 2, 8, 4, 16
        T = B * q_len
        T_dim = max_seq_len // ratio  # 4

        # Span various boundary alignments.
        start_pos = torch.tensor([2, 3, 6], dtype=torch.int32, device=device)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=q_len,
            window_size=window,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            compress_ratios=[ratio],
            index_topk=8,
            device=device,
        )

        cmp_buf = torch.zeros(B, T_dim, head_dim, dtype=torch.bfloat16, device=device)
        k_state = _tagged(T, head_dim, base=800.0, device=device)

        write_compressed_k_decode(
            k_state,
            meta.slot_mapping_compressed[ratio],
            cmp_buf,
        )

        # Oracle.
        oracle = torch.zeros_like(cmp_buf)
        sp = start_pos.tolist()
        for r in range(B):
            for s in range(q_len):
                abs_p1 = sp[r] + s + 1
                if abs_p1 % ratio == 0:
                    in_req = abs_p1 // ratio - 1
                    t_idx = r * q_len + s
                    oracle[r, in_req] = k_state[t_idx]
        self.assertTrue(torch.equal(cmp_buf, oracle))

    def test_t_total_zero_noop(self):
        device = torch.device("cpu")
        cmp_buf = torch.zeros(2, 4, 4, dtype=torch.bfloat16, device=device)
        k_state = torch.zeros(0, 4, dtype=torch.bfloat16, device=device)
        slot = torch.zeros(0, dtype=torch.int32, device=device)
        write_compressed_k_decode(k_state, slot, cmp_buf)
        self.assertTrue(torch.all(cmp_buf == 0))

    def test_all_slots_negative_noop(self):
        """All slots -1 → buffer must remain unchanged."""
        device = torch.device("cpu")
        cmp_buf = torch.zeros(2, 4, 4, dtype=torch.bfloat16, device=device)
        k_state = _tagged(3, 4, base=900.0, device=device)
        slot = torch.tensor([-1, -1, -1], dtype=torch.int32, device=device)
        write_compressed_k_decode(k_state, slot, cmp_buf)
        self.assertTrue(torch.all(cmp_buf == 0))


if __name__ == "__main__":
    unittest.main()
