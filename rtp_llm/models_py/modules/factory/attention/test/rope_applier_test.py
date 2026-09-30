"""Contract tests for the rope applier layer.

Pins the applier protocol (rope output, KV write and params ownership) and the
py-flashinfer prefill selection, so a rope change cannot silently rewire a
prefill core or drop a rope method.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    FlashinferRopeApplier,
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillImplBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.rope_applier import (
    FusedRopeQKVOutApplier,
)
from rtp_llm.models_py.modules.factory.attention.rope_applier import RopeApplier
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import FusedRopeKVCachePrefillOpQKVOut


def _attn_configs(style: RopeStyle, mrope_interleaved: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        size_per_head=128,
        kernel_tokens_per_block=16,
        max_seq_len=4096,
        gen_num_per_cycle=1,
        head_num=8,
        kv_head_num=2,
        rope_config=SimpleNamespace(
            style=style,
            base=10000.0,
            dim=128,
            mrope_interleaved=mrope_interleaved,
        ),
    )


class RopeApplierContractTest(unittest.TestCase):
    def test_fused_applier_is_the_fused_operator(self):
        # The applier adds rope metadata to the existing operator instead of
        # wrapping it, so callers keep the operator API unchanged.
        applier = FusedRopeQKVOutApplier(SimpleNamespace())
        self.assertIsInstance(applier, FusedRopeKVCachePrefillOpQKVOut)
        self.assertIsInstance(applier, RopeApplier)
        self.assertTrue(FusedRopeQKVOutApplier.fused_kv_write)
        self.assertTrue(FusedRopeQKVOutApplier.owns_params)
        with mock.patch.object(applier, "forward") as op_forward:
            op_forward.return_value = "roped"
            self.assertEqual(applier.apply("qkv", "cache", "params"), "roped")
        op_forward.assert_called_once_with("qkv", "cache", "params")

    def test_flashinfer_applier_is_the_rotary_operator(self):
        applier = FlashinferRopeApplier(_attn_configs(RopeStyle.Base))
        self.assertIsInstance(applier, MhaRotaryEmbeddingOp)
        self.assertIsInstance(applier, RopeApplier)
        # The KV write stays with the caller, and positions come from the
        # shared FMHA params instead of applier-owned params.
        self.assertFalse(FlashinferRopeApplier.fused_kv_write)
        self.assertFalse(FlashinferRopeApplier.owns_params)
        self.assertIsNone(RopeApplier.prepare(applier, None))
        with mock.patch.object(applier, "forward") as op_forward:
            op_forward.return_value = ("q", "k", "v")
            self.assertEqual(applier.apply("qkv"), ("q", "k", "v"))
        op_forward.assert_called_once_with("qkv")

    def test_py_flashinfer_prefill_selection(self):
        self.assertIsNone(
            PyFlashinferPrefillImplBase._create_rope_applier(
                None, _attn_configs(RopeStyle.No)
            )
        )
        for style in (RopeStyle.Base, RopeStyle.Yarn):
            applier = PyFlashinferPrefillImplBase._create_rope_applier(
                None, _attn_configs(style)
            )
            self.assertIsInstance(applier, FlashinferRopeApplier)


if __name__ == "__main__":
    unittest.main()
