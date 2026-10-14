"""Tests for FlashAttn2PrefillImpl (sm120 eager varlen prefill, no prefix).

Correctness follows the same reference path as the fmha_v2 impl tests:
apply_base_rope_to_qkv_reference + compute_pytorch_prefill_reference. The
impl is constructed directly — support() is sm12x-gated for factory
routing, but the FA2 kernels are arch-portable (Ampere and later), so the
math is validated on whichever GPU runs the test.
"""

import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_attn_2 import (
    FlashAttn2PrefillImpl,
    _HAS_FLASH_ATTN_2,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.test_trt_base import (
    TRTLLMFMHAv2TestBase,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.trt_tests.trt_test_utils import (
    apply_base_rope_to_qkv_reference,
    compute_pytorch_prefill_reference,
)
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import get_typemeta


class TestSm120KernelGuard(unittest.TestCase):
    """The shared version guard gates this impl's support() (round-6 fix).

    Covers both branches: reject pre-2.8.4 builds (2.8.3.post1 imported
    cleanly but has no sm_120 cubins) and accept 2.8.4+ local builds.
    """

    def test_guard_rejects_283_and_accepts_284(self):
        fake = mock.MagicMock()
        for version, expected in (
            ("2.8.3.post1+cu13torch2.11cxx11abitrue.r1", False),
            ("2.8.4+b27cdc3.cu132", True),
        ):
            fake.__version__ = version
            with mock.patch.dict("sys.modules", {"flash_attn": fake}):
                from rtp_llm.models_py.modules.factory.attention.cuda_impl import (
                    flash_attn_2,
                )
                from rtp_llm.utils.flash_attn_utils import (
                    _flash_attn_has_sm120_kernels,
                )

                self.assertEqual(
                    _flash_attn_has_sm120_kernels(),
                    expected,
                    msg=version,
                )
                # the impl-level guard additionally requires the import to
                # have succeeded; emulate the loaded state
                with mock.patch.object(flash_attn_2, "_HAS_FLASH_ATTN_2", True):
                    self.assertEqual(
                        flash_attn_2._has_sm120_kernels(), expected, msg=version
                    )


class TestSupportWiring(unittest.TestCase):
    """support() wiring on the sm120 path (round-6 must-fix 3).

    Reaches past the arch/version guards with mocks so the rest of the
    predicate (dtype/head-dim/GQA/prefix checks - and the module-level
    constants they reference) is exercised on any host GPU. CI run
    79765748 caught a NameError here that the impl-construction tests
    cannot see: a free name in support() only fails on the factory path.
    """

    def _make_inputs(self, batch, lengths, prefix=None):
        # minimal prefill inputs; only the fields support() reads matter
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.input_lengths = torch.tensor(lengths, dtype=torch.int32)
        if prefix is None:
            inputs.prefix_lengths = torch.zeros(batch, dtype=torch.int32)
        else:
            inputs.prefix_lengths = torch.tensor(
                prefix, dtype=torch.int32
            )
        return inputs

    def _support(self, configs, inputs):
        from rtp_llm.models_py.modules.factory.attention.cuda_impl import (
            flash_attn_2,
        )

        with mock.patch.object(
            flash_attn_2, "is_sm12x", return_value=True
        ), mock.patch.object(
            flash_attn_2, "_has_sm120_kernels", return_value=True
        ):
            return flash_attn_2.FlashAttn2PrefillImpl.support(configs, inputs)

    def test_support_accepts_gate_config(self):
        configs = self._configs(head_dim=64, dtype=torch.bfloat16)
        inputs = self._make_inputs(1, [128])
        self.assertTrue(self._support(configs, inputs))

    def test_support_rejects_unsupported_head_dim_and_dtype(self):
        inputs = self._make_inputs(1, [128])
        configs = self._configs(head_dim=63, dtype=torch.bfloat16)
        self.assertFalse(self._support(configs, inputs))
        configs = self._configs(head_dim=64, dtype=torch.float32)
        self.assertFalse(self._support(configs, inputs))

    def test_support_rejects_prefix(self):
        configs = self._configs(head_dim=64, dtype=torch.bfloat16)
        inputs = self._make_inputs(
            1, [128], prefix=[64]
        )
        self.assertFalse(self._support(configs, inputs))

    def _configs(self, head_dim, dtype):
        from rtp_llm.ops import AttentionConfigs, KvCacheDataType

        configs = AttentionConfigs()
        configs.head_num = 32
        configs.kv_head_num = 8
        configs.size_per_head = head_dim
        configs.dtype = dtype
        configs.kv_cache_dtype = KvCacheDataType.BASE
        configs.use_mla = False
        configs.is_sparse = False
        return configs


@unittest.skipUnless(_HAS_FLASH_ATTN_2, "flash_attn 2 not available")
class TestFlashAttn2PrefillImpl(TRTLLMFMHAv2TestBase):
    def _run_impl_rope_correctness(
        self,
        rope_style,
        input_lengths,
        head_num,
        head_num_kv,
        head_dim,
        data_type="fp16",
    ):
        tokens_per_block = 64
        rope_base = 10000
        batch_size = len(input_lengths)
        total_tokens = sum(input_lengths)

        attn_configs = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=head_dim,
            seq_size_per_block=tokens_per_block,
            data_type=data_type,
        )
        attn_configs.rope_config.style = rope_style
        attn_configs.rope_config.dim = head_dim
        attn_configs.rope_config.base = rope_base
        attn_configs.rope_config.max_pos = max(input_lengths)
        attn_configs.max_seq_len = max(input_lengths)
        attn_inputs = self._create_prefill_attention_inputs(
            batch_size,
            input_lengths,
            tokens_per_block,
            prefix_lengths=None,
        )

        qkv = self._create_qkv_tensor(
            total_tokens,
            head_num,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        attn_inputs.dtype = get_typemeta(qkv)

        expected_qkv = qkv
        if rope_style != RopeStyle.No:
            expected_qkv = apply_base_rope_to_qkv_reference(
                qkv,
                input_lengths,
                head_num,
                head_num_kv,
                head_dim,
                rope_base,
            )

        total_blocks = self._calculate_total_blocks(
            input_lengths, tokens_per_block
        )
        kv_cache, _, _ = self._create_kv_cache(
            total_blocks,
            tokens_per_block,
            head_num_kv,
            head_dim,
            dtype=attn_configs.dtype,
        )
        kv_cache.kv_cache_base = kv_cache.kv_cache_base.flatten(1)

        impl = FlashAttn2PrefillImpl(attn_configs, attn_inputs)
        actual = impl.forward(qkv.clone(), kv_cache, layer_idx=0)

        expected = compute_pytorch_prefill_reference(
            expected_qkv,
            input_lengths,
            head_num,
            head_num_kv,
            head_dim,
        )
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)

    def test_impl_without_rope_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.No, [128, 64], head_num=32, head_num_kv=8, head_dim=64
        )

    def test_impl_with_rope_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.Base, [128, 64], head_num=32, head_num_kv=8, head_dim=64
        )

    def test_impl_gqa_head128_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.Base, [96, 33], head_num=8, head_num_kv=2, head_dim=128
        )

    def test_impl_mha_matches_torch(self):
        self._run_impl_rope_correctness(
            RopeStyle.Base, [64], head_num=8, head_num_kv=8, head_dim=64
        )

    def test_impl_bf16_gqa_hd64_matches_torch(self):
        # The production gate configuration (d1-3B: bf16, GQA 32:8, hd 64,
        # sm120) - kept as a numeric golden so the exact gate config has
        # direct coverage, not only fp16 variants.
        self._run_impl_rope_correctness(
            RopeStyle.Base,
            [128, 64],
            head_num=32,
            head_num_kv=8,
            head_dim=64,
            data_type="bf16",
        )


if __name__ == "__main__":
    unittest.main()
