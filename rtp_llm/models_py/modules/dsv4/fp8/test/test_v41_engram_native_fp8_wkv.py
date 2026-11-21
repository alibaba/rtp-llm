# SPDX-License-Identifier: Apache-2.0
"""Regression for the direct-FP8 Engram WKV input (DSV41_ENGRAM_WKV_INPUT).

The native path must stay inert unless explicitly enabled AND applicable: the
Triton kernel module must not be imported when the option is off (CPU-only
deployments have no Triton), and any ineligible geometry / non-native projection /
pending prefetch / wrong device must fall back to the baseline `_rows_for` path.

The realistic layout is n_heads=8 x max_ngram_size=4 -> n_hash_cols=24 hash columns,
head_dim=256. The operand-parity test (SM120) checks the fused gather+decode+quantize
kernel against the baseline lookup + native group-32 quantizer, bit-exact.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import engram
from rtp_llm.models_py.modules.dsv4.engram import Engram

ENV = "DSV41_ENGRAM_WKV_INPUT"

# Realistic V4.1-Flash Engram layout: 8 heads x (4-1) ngrams = 24 hash columns.
N_HASH_COLS = 24
HEAD_DIM = 256
TOKENS = 2


def _layout(n_hash_cols=N_HASH_COLS, head_dim=HEAD_DIM):
    return SimpleNamespace(
        n_heads=8, max_ngram_size=4, n_hash_cols=n_hash_cols, head_dim=head_dim
    )


def _owner(*, layout=None, native_wkv=True, prefetch=None, cuda=False):
    """A minimal Engram-like owner with the real _forward_rows/_rows_for/eligible bound.

    The wkv records which path ran: forward -> ("baseline", rows), forward_quantized ->
    ("native", x_q, x_s). gated_engram_residual is patched to the identity on `kv` by the
    tests, so the _forward_rows return value is exactly that record tuple.
    """
    device = "cuda" if cuda else "cpu"
    rows_sentinel = torch.zeros(
        (TOKENS, N_HASH_COLS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )

    class _BaselineWkv:
        def forward(self, rows):
            return ("baseline", rows)

        __call__ = forward

    class _NativeWkv:
        def forward(self, rows):
            return ("baseline", rows)

        __call__ = forward

        def forward_quantized(self, x_q, x_s):
            return ("native", x_q, x_s)

    owner = SimpleNamespace(
        layout=layout or _layout(),
        embed_tokens=lambda ids, dev: rows_sentinel,
        wkv=_NativeWkv() if native_wkv else _BaselineWkv(),
        q_weight=torch.ones(2, 4),
        k_weight=torch.ones(2, 4),
        eps=1e-6,
        _prefetched_rows=prefetch,
    )
    owner._rows_for = lambda h, i: Engram._rows_for(owner, h, i)
    owner._wkv_native_fp8_eligible = lambda h, i: Engram._wkv_native_fp8_eligible(
        owner, h, i
    )
    return owner


def _run(owner, hidden, hash_ids):
    """Run _forward_rows with gated_engram_residual returning the kv record tuple."""
    with patch.object(engram, "gated_engram_residual", lambda hidden, kv, *a, **k: kv):
        return Engram._forward_rows(owner, hidden, hash_ids, None)


def _hidden(cuda=False):
    return torch.zeros(
        (TOKENS, 2, 4), dtype=torch.bfloat16, device="cuda" if cuda else "cpu"
    )


def _hash_ids(cuda=False, dtype=torch.int64, cols=N_HASH_COLS):
    return torch.zeros((TOKENS, cols), dtype=dtype, device="cuda" if cuda else "cpu")


class NativeFp8ImportGuardTest(unittest.TestCase):
    """OFF option must never import the Triton kernel module (CPU-only safe)."""

    def test_off_option_does_not_import_kernel_module(self):
        sys.modules.pop("rtp_llm.models_py.modules.dsv4._engram_fp8_wkv", None)
        owner = _owner(native_wkv=True)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)  # option absent -> baseline
            # Make the kernel module unimportable; the baseline path must not care.
            with patch.dict(
                sys.modules, {"rtp_llm.models_py.modules.dsv4._engram_fp8_wkv": None}
            ):
                result = _run(owner, _hidden(), _hash_ids())
        self.assertEqual(result[0], "baseline")

    def test_gate_returns_false_when_option_off(self):
        owner = _owner()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            self.assertFalse(owner._wkv_native_fp8_eligible(_hidden(), _hash_ids()))


class NativeFp8FallbackTest(unittest.TestCase):
    """Option ON but any ineligibility -> baseline path (no native operand consumed)."""

    def _check_baseline(self, owner, hidden, hash_ids):
        with patch.dict(os.environ, {ENV: "native_fp8"}):
            result = _run(owner, hidden, hash_ids)
        self.assertEqual(result[0], "baseline")

    def test_cpu_device_falls_back(self):
        self._check_baseline(_owner(native_wkv=True), _hidden(False), _hash_ids(False))

    def test_non_native_wkv_falls_back(self):
        self._check_baseline(
            _owner(native_wkv=False, cuda=True), _hidden(True), _hash_ids(True)
        )

    def test_wrong_hash_col_count_falls_back(self):
        # e.g. a 4-head layout (n_hash_cols=12) is unsupported by the 24-column kernel
        self._check_baseline(
            _owner(layout=_layout(n_hash_cols=12), cuda=True),
            _hidden(True),
            _hash_ids(True, cols=12),
        )

    def test_wrong_head_width_falls_back(self):
        self._check_baseline(
            _owner(layout=_layout(head_dim=128), cuda=True),
            _hidden(True),
            _hash_ids(True),
        )

    def test_non_integer_hash_dtype_falls_back(self):
        self._check_baseline(
            _owner(native_wkv=True, cuda=True),
            _hidden(True),
            _hash_ids(True, dtype=torch.float32),
        )

    def test_pending_prefetch_falls_back_and_is_consumed(self):
        if not torch.cuda.is_available():
            self.skipTest("prefetch event requires CUDA")
        prefetch_rows = torch.zeros(
            (TOKENS, N_HASH_COLS, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )
        event = torch.cuda.Event()
        owner = _owner(native_wkv=True, cuda=True, prefetch=(prefetch_rows, event))
        with patch.dict(os.environ, {ENV: "native_fp8"}):
            result = _run(owner, _hidden(True), _hash_ids(True))
        self.assertEqual(result[0], "baseline")
        self.assertIsNone(owner._prefetched_rows)  # prefetch consumed by _rows_for


class NativeFp8EngagementTest(unittest.TestCase):
    """Positive on-path engagement: eligible inputs take the native forward_quantized path."""

    def setUp(self):
        if not (
            torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
        ):
            self.skipTest("native engagement requires an SM120 CUDA device")

    def test_eligible_input_engages_native_path(self):
        owner = _owner(
            native_wkv=True, cuda=True
        )  # realistic 24-col / 256-width layout
        with patch.dict(os.environ, {ENV: "native_fp8"}):
            self.assertTrue(
                owner._wkv_native_fp8_eligible(_hidden(True), _hash_ids(True))
            )
        fake_operands = (
            torch.zeros(
                (TOKENS, N_HASH_COLS * HEAD_DIM),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            ),
            torch.zeros((TOKENS, 48), dtype=torch.int32, device="cuda"),
        )
        with patch.dict(os.environ, {ENV: "native_fp8"}):
            with patch(
                "rtp_llm.models_py.modules.dsv4._engram_fp8_wkv.operands_from_embedding",
                return_value=fake_operands,
            ):
                result = _run(owner, _hidden(True), _hash_ids(True))
        self.assertEqual(result[0], "native")  # forward_quantized path taken


class NativeFp8OperandParityTest(unittest.TestCase):
    """Fused gather+decode+quantize == baseline lookup + native group32 quantizer.

    Uses scale byte 127 (group scale == 1.0) and full-range payloads so every group's
    amax sits far above both quantizers' eps floors (1e-4 baseline, fp32-tiny native),
    where the two recipes are bit-identical (power-of-2 scale: x/scale == x*(1/scale)).
    """

    def setUp(self):
        if not (
            torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
        ):
            self.skipTest("operand parity requires an SM120 CUDA device")

    def test_fused_operands_match_baseline(self):
        from rtp_llm.models_py.modules.dsv4 import _engram_fp8_wkv
        from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32

        rows, cols, dim = 9, N_HASH_COLS, HEAD_DIM
        # Full-range E4M3 payloads with a guaranteed healthy per-group max (every row
        # starts at the max finite E4M3 magnitude 0x7E), and unit group scales (127).
        weight = (
            torch.arange(dim, dtype=torch.uint8).repeat(rows, 1) % 0x7F
        ).contiguous()
        weight[:, 0] = 0x7E  # guarantee a large group max so amax >> eps floors
        weight = weight.cuda()
        scales = torch.full((rows, dim // 32), 127, dtype=torch.uint8, device="cuda")
        hash_ids = torch.randint(0, rows, (3, cols), dtype=torch.int64, device="cuda")

        # native fused path
        x_q, x_s = _engram_fp8_wkv.fused_lookup_quant_rows(
            weight, scales, hash_ids, rows
        )

        # baseline: decode FP8 -> BF16, then the production group-32 quantizer
        val = weight[hash_ids].view(torch.float8_e4m3fn).float()  # [t,cols,dim]
        sc = scales[hash_ids].view(torch.float8_e8m0fnu).float()  # [t,cols,dim/32]
        bf16 = (val * sc.repeat_interleave(32, dim=-1)).to(
            torch.bfloat16
        )  # [t,cols,dim]
        flat = bf16.flatten(1).contiguous()  # [t, 6144]
        x_q_ref, s_ref = quantize_group32(flat, backend="triton")

        # payload must be bit-exact
        self.assertTrue(torch.equal(x_q.view(torch.uint8), x_q_ref.view(torch.uint8)))
        # scales: x_s packs column c groups [0..3]->word[2c], [4..7]->word[2c+1], little-endian
        t_n = hash_ids.shape[0]
        got = torch.zeros((t_n, cols * 8), dtype=torch.uint8, device=x_s.device)
        words = x_s.contiguous().view(t_n, 2 * cols).to(torch.int32)
        for c in range(cols):
            w0 = words[:, 2 * c]
            w1 = words[:, 2 * c + 1]
            for g in range(4):
                got[:, c * 8 + g] = ((w0 >> (8 * g)) & 0xFF).to(torch.uint8)
                got[:, c * 8 + 4 + g] = ((w1 >> (8 * g)) & 0xFF).to(torch.uint8)
        self.assertTrue(torch.equal(got, s_ref.contiguous().view(t_n, cols * 8)))


if __name__ == "__main__":
    unittest.main()
