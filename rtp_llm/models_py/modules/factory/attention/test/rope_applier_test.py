"""Contract tests for the rope applier layer.

Pins the applier protocol (rope output, KV write and params ownership), the
rope-module selection for the py-flashinfer prefill cores, and the factory's
native-first dispatch (an impl that would swap its rope module never steals
the selection from kernels whose rope is native), so a rope change cannot
silently rewire a prefill core or flip the dispatch.
"""

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    FlashinferRopeApplier,
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPagedPrefillImpl,
    PyFlashinferPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.rope_applier import (
    FusedRopeQKVOutApplier,
    create_prefill_rope_applier,
    prefill_rope_is_fused,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
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


def _attn_inputs(has_paged_blocks: bool) -> SimpleNamespace:
    prefix_lengths = torch.zeros(0, dtype=torch.int32)
    empty = torch.zeros(0, dtype=torch.int32)
    blocks = torch.ones(4, dtype=torch.int32) if has_paged_blocks else empty
    return SimpleNamespace(
        prefix_lengths=prefix_lengths,
        kv_cache_kernel_block_id=blocks,
        kv_cache_kernel_block_id_device=empty,
    )


def _patch_sm10x(value: bool):
    return mock.patch(
        "rtp_llm.models_py.modules.factory.attention.cuda_impl"
        ".py_flashinfer_mha.is_sm10x",
        return_value=value,
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

    def test_rope_module_selected_by_config(self):
        self.assertIsNone(create_prefill_rope_applier(_attn_configs(RopeStyle.No)))
        for style in (RopeStyle.Base, RopeStyle.Yarn):
            self.assertIsInstance(
                create_prefill_rope_applier(_attn_configs(style)),
                FlashinferRopeApplier,
                style,
            )
        self.assertIsInstance(
            create_prefill_rope_applier(_attn_configs(RopeStyle.Mrope)),
            FusedRopeQKVOutApplier,
        )
        self.assertTrue(prefill_rope_is_fused(_attn_configs(RopeStyle.Mrope)))
        self.assertFalse(
            prefill_rope_is_fused(
                _attn_configs(RopeStyle.Mrope, mrope_interleaved=False)
            )
        )
        for style in (RopeStyle.No, RopeStyle.Base, RopeStyle.Yarn):
            self.assertFalse(prefill_rope_is_fused(_attn_configs(style)), style)


class PyFlashinferRopeSupportTest(unittest.TestCase):
    def test_paged_core_accepts_interleaved_mrope(self):
        # MRoPE composes with the stock paged core on every GPU the factory
        # needs it on, including sm_10x where no TRT-LLM cubin exists.
        for sm10x in (True, False):
            with _patch_sm10x(sm10x):
                self.assertTrue(
                    PyFlashinferPagedPrefillImpl.support(
                        _attn_configs(RopeStyle.Mrope), _attn_inputs(True)
                    ),
                    f"sm10x={sm10x}",
                )
                # Non-interleaved MRoPE has no rope kernel: still rejected.
                self.assertFalse(
                    PyFlashinferPagedPrefillImpl.support(
                        _attn_configs(RopeStyle.Mrope, mrope_interleaved=False),
                        _attn_inputs(True),
                    )
                )

    def test_paged_core_keeps_non_mrope_gates(self):
        with _patch_sm10x(True):
            for style in (RopeStyle.No, RopeStyle.Base, RopeStyle.Yarn):
                self.assertFalse(
                    PyFlashinferPagedPrefillImpl.support(
                        _attn_configs(style), _attn_inputs(True)
                    ),
                    style,
                )
        with _patch_sm10x(False):
            for style in (RopeStyle.No, RopeStyle.Base, RopeStyle.Yarn):
                self.assertTrue(
                    PyFlashinferPagedPrefillImpl.support(
                        _attn_configs(style), _attn_inputs(True)
                    ),
                    style,
                )

    def test_ragged_core_serves_mrope_only_without_block_table(self):
        # With a paged block table the paged core owns the serving layout; the
        # ragged core is the embedding / classifier scoring path.
        self.assertFalse(
            PyFlashinferPrefillImpl.support(
                _attn_configs(RopeStyle.Mrope), _attn_inputs(True)
            )
        )
        self.assertTrue(
            PyFlashinferPrefillImpl.support(
                _attn_configs(RopeStyle.Mrope), _attn_inputs(False)
            )
        )
        self.assertFalse(
            PyFlashinferPrefillImpl.support(
                _attn_configs(RopeStyle.Mrope, mrope_interleaved=False),
                _attn_inputs(False),
            )
        )
        for style in (RopeStyle.No, RopeStyle.Base, RopeStyle.Yarn):
            self.assertTrue(
                PyFlashinferPrefillImpl.support(
                    _attn_configs(style), _attn_inputs(False)
                ),
                style,
            )

    def test_rope_is_composed_marks_only_the_swapped_configs(self):
        for impl in (PyFlashinferPagedPrefillImpl, PyFlashinferPrefillImpl):
            self.assertTrue(impl.rope_is_composed(_attn_configs(RopeStyle.Mrope)))
            self.assertFalse(
                impl.rope_is_composed(
                    _attn_configs(RopeStyle.Mrope, mrope_interleaved=False)
                )
            )
            for style in (RopeStyle.No, RopeStyle.Base, RopeStyle.Yarn):
                self.assertFalse(impl.rope_is_composed(_attn_configs(style)), style)
        # Other impl families keep the default: the rope is part of the kernel.
        self.assertFalse(FMHAImplBase.rope_is_composed(_attn_configs(RopeStyle.Mrope)))


class _StubImpl:
    accepts_fmha_config = False

    def __init__(self, *_args, **_kwargs):
        pass

    @classmethod
    def support(cls, attn_configs, attn_inputs):
        return True

    @classmethod
    def support_parallelism_config(cls, parallelism_config):
        return True

    @classmethod
    def rope_is_composed(cls, attn_configs):
        return False

    def support_cuda_graph(self):
        return True


class _ComposedRopeStubImpl(_StubImpl):
    @classmethod
    def rope_is_composed(cls, attn_configs):
        return True


class _NativeRopeStubImpl(_StubImpl):
    pass


class FactoryNativeRopePriorityTest(unittest.TestCase):
    def _select(self, implementations):
        attn_configs = SimpleNamespace(
            rope_config=SimpleNamespace(style=None), need_rope_kv_cache=False
        )
        attn_inputs = SimpleNamespace(is_prefill=True)
        with patch.object(attn_factory, "PREFILL_MHA_IMPS", implementations):
            return attn_factory.get_fmha_impl(
                attn_configs, None, attn_inputs, is_cuda_graph=False
            )

    def test_native_rope_impl_wins_over_a_preceding_composed_impl(self):
        # The py-flashinfer paged impls sit before the TRT-LLM ones; a config
        # they would only serve by swapping the rope must not steal the
        # selection from kernel-native impls.
        self.assertIsInstance(
            self._select([_ComposedRopeStubImpl, _NativeRopeStubImpl]),
            _NativeRopeStubImpl,
        )

    def test_composed_impl_serves_when_it_is_the_only_candidate(self):
        self.assertIsInstance(
            self._select([_ComposedRopeStubImpl]), _ComposedRopeStubImpl
        )

    def test_composed_impl_serves_when_native_candidates_reject(self):
        # sm_10x: the TRT-LLM impls reject by architecture (their cubins do not
        # cover it), so the deferral must fall back to the composed
        # py-flashinfer impl.
        class _ArchGatedNativeStubImpl(_NativeRopeStubImpl):
            @classmethod
            def support(cls, attn_configs, attn_inputs):
                return False

        self.assertIsInstance(
            self._select([_ArchGatedNativeStubImpl, _ComposedRopeStubImpl]),
            _ComposedRopeStubImpl,
        )

    def test_native_impl_still_serves_without_a_composed_candidate(self):
        self.assertIsInstance(self._select([_NativeRopeStubImpl]), _NativeRopeStubImpl)

    def test_unsupported_composed_impl_is_not_selected(self):
        class _UnsupportedComposedRopeStubImpl(_ComposedRopeStubImpl):
            @classmethod
            def support(cls, attn_configs, attn_inputs):
                return False

        with self.assertRaisesRegex(Exception, "can not find mha type"):
            self._select([_UnsupportedComposedRopeStubImpl])


if __name__ == "__main__":
    unittest.main()
