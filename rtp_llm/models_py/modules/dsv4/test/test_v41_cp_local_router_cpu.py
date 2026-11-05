"""CPU tests for the D1 CP-local-router flag: env parsing, padded-GEMM math,
guard rejection. Bitwise GPU evidence lives in r7_router_bitwise_v2.json
(captures from the reference cluster); these tests cover the plumbing only.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "_cp_local_router_under_test", ROOT / "_v41_cp_local_router.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clr = load_module()


class StubGate:
    """Row-independent _forward_local stand-in (mirrors the per-row
    determinism of the real FP32 GEMM + fused epilogue)."""

    def __init__(self, topk=2, hash=False, fp32_logits=True):
        self.hash = hash
        self.fp32_logits = fp32_logits
        self.topk = topk
        self.seen = []

    def _forward_local(self, x, input_ids):
        assert input_ids is None
        self.seen.append(x)
        scores = x.float()
        weights = scores.sum(-1, keepdim=True).expand(-1, self.topk).contiguous()
        indices = scores.topk(self.topk, dim=-1)[1]
        return weights, indices


def fake_cp(t, padded=None, cp_size=4, sharded=False):
    return SimpleNamespace(
        cp_size=cp_size,
        chunk_length=t,
        padded_seq_len=(4 * t if padded is None else padded),
        kv_cache_sharded=sharded,
    )


class FlagParsingTest(unittest.TestCase):
    def test_default_on(self):
        # R8 decision 1: padded local router is the default (byte-exact).
        env = dict(os.environ)
        env.pop("DSV41_CP_LOCAL_ROUTER", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(clr.cp_local_router_enabled())

    def test_one_enables(self):
        with mock.patch.dict(os.environ, {"DSV41_CP_LOCAL_ROUTER": "1"}):
            self.assertTrue(clr.cp_local_router_enabled())

    def test_escape_hatch_and_other_values_disable(self):
        for value in ("0", "true", "yes", ""):
            with mock.patch.dict(os.environ, {"DSV41_CP_LOCAL_ROUTER": value}):
                self.assertFalse(clr.cp_local_router_enabled())


class PaddedMathTest(unittest.TestCase):
    def test_identity_when_m_full_equals_rows(self):
        gate = StubGate()
        x = torch.randn(5, 8).bfloat16()
        sentinel = (torch.zeros(5, 2), torch.zeros(5, 2, dtype=torch.long))
        gate._forward_local = lambda xin, ids: (gate.seen.append(xin), sentinel)[1]
        out = clr.padded_local_routes(gate, x, 5)
        self.assertIs(out, sentinel)
        self.assertIs(gate.seen[0], x)

    def test_m_full_below_rows_rejected(self):
        gate = StubGate()
        x = torch.randn(5, 8).bfloat16()
        with self.assertRaises(ValueError):
            clr.padded_local_routes(gate, x, 4)

    def test_zero_pad_and_front_slice(self):
        gate = StubGate(topk=3)
        t, dim, m_full = 3, 8, 12
        x = torch.randn(t, dim).bfloat16()
        weights, indices = clr.padded_local_routes(gate, x, m_full)
        # the stub saw the padded batch: exact x rows in front, zeros behind
        seen = gate.seen[0]
        self.assertEqual(tuple(seen.shape), (m_full, dim))
        self.assertTrue(torch.equal(seen[:t], x))
        self.assertTrue(torch.equal(seen[t:], torch.zeros(m_full - t, dim, dtype=seen.dtype)))
        # outputs are the front slice, contiguous, correctly typed/shaped
        self.assertEqual(tuple(weights.shape), (t, 3))
        self.assertEqual(tuple(indices.shape), (t, 3))
        self.assertTrue(weights.is_contiguous() and indices.is_contiguous())
        # row-independent routing: equals routing the hand-built pad directly
        pad = torch.zeros(m_full, dim, dtype=x.dtype)
        pad[:t] = x
        ref_gate = StubGate(topk=3)
        ref_w, ref_i = ref_gate._forward_local(pad, None)
        self.assertTrue(torch.equal(weights, ref_w[:t]))
        self.assertTrue(torch.equal(indices, ref_i[:t]))


class GuardTest(unittest.TestCase):
    def _x(self, t=4, dim=5120):
        return torch.zeros(t, dim, dtype=torch.bfloat16)

    def test_cpu_tensor_rejected(self):
        gate = StubGate()
        with self.assertRaises(NotImplementedError):
            clr.cp_local_padded_routes(gate, self._x(), fake_cp(4))

    def test_hash_gate_rejected(self):
        gate = StubGate(hash=True)
        with self.assertRaises(NotImplementedError):
            clr.cp_local_padded_routes(gate, self._x(), fake_cp(4))

    def test_non_fp32_logits_rejected(self):
        gate = StubGate(fp32_logits=False)
        with self.assertRaises(NotImplementedError):
            clr.cp_local_padded_routes(gate, self._x(), fake_cp(4))

    def test_sharded_kv_cache_rejected(self):
        gate = StubGate()
        with self.assertRaises(NotImplementedError):
            clr.cp_local_padded_routes(gate, self._x(), fake_cp(4, sharded=True))

    def test_cp_size_rejected(self):
        gate = StubGate()
        with self.assertRaises(NotImplementedError):
            clr.cp_local_padded_routes(
                gate, self._x(), fake_cp(4, cp_size=2)
            )

    def test_non_contiguous_rejected(self):
        gate = StubGate()
        x = torch.zeros(4, 2 * 5120, dtype=torch.bfloat16)[:, :5120]
        self.assertFalse(x.is_contiguous())
        with self.assertRaises(NotImplementedError):
            clr.cp_local_padded_routes(gate, x, fake_cp(4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
