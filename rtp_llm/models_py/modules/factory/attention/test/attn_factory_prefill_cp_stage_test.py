import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    fill_paged_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import CPRotateMethod, FMHAConfig, ParallelismConfig, RopeStyle
from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCachePrefillOpQKVOut


class _PlainMHA:
    accepts_fmha_config = False
    support_parallelism_config = classmethod(
        FMHAImplBase.support_parallelism_config.__func__
    )

    @classmethod
    def support_prefill_cp(cls):
        return False

    @classmethod
    def support(cls, *_args):
        return True

    def __init__(self, *_args):
        pass


class _UnsupportedMHA(_PlainMHA):
    @classmethod
    def support(cls, *_args):
        return False


class AttentionFactoryPrefillCPStageTest(unittest.TestCase):
    def setUp(self):
        self.parallelism = ParallelismConfig()
        self.parallelism.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        self.config = types.SimpleNamespace(
            rope_config=types.SimpleNamespace(style=None)
        )

    def select(self, is_prefill, impl):
        registry = "PREFILL_MHA_IMPS" if is_prefill else "DECODE_MHA_IMPS"
        with patch.object(attn_factory, registry, [impl]):
            return attn_factory.get_fmha_impl(
                self.config,
                None,
                types.SimpleNamespace(is_prefill=is_prefill),
                parallelism_config=self.parallelism,
            )

    def test_prefill_still_requires_cp_support(self):
        with self.assertRaisesRegex(Exception, "can not find mha type"):
            self.select(True, _PlainMHA)

    def test_decode_is_not_filtered_by_prefill_cp(self):
        for tp_size in (1, 2, 4):
            with self.subTest(tp_size=tp_size):
                self.parallelism.tp_size = tp_size
                self.assertIsInstance(self.select(False, _PlainMHA), _PlainMHA)

    def test_decode_still_checks_kernel_support(self):
        with self.assertRaisesRegex(Exception, "can not find mha type"):
            self.select(False, _UnsupportedMHA)


class AttentionFactoryMropeDecodeCUDATest(BaseAttentionTest):
    def test_bf16_mrope_decode_with_prefill_cp(self):
        config = self._create_config(head_num=32, data_type="bf16")
        cfg = config.attn_configs
        cfg.need_rope_kv_cache = True
        cfg.max_seq_len = 256
        cfg.rope_config.style = RopeStyle.Mrope
        cfg.rope_config.dim = 128
        cfg.rope_config.base = 5000000
        cfg.rope_config.max_pos = 256
        cfg.rope_config.index_factor = 3
        cfg.rope_config.mrope_dim1 = 24
        cfg.rope_config.mrope_dim2 = 20
        cfg.rope_config.mrope_dim3 = 20
        cfg.rope_config.mrope_interleaved = True
        config.parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        lengths = [13, 65]
        inputs = self._create_attention_inputs_base(
            2, lengths, config.seq_size_per_block, dtype=torch.bfloat16
        )
        inputs.is_cuda_graph = False
        # Production PositionIdsGenerator advances all three axes together in decode.
        positions = torch.tensor(
            [[9, 9, 9], [31, 31, 31]], device="cuda", dtype=torch.int32
        )
        inputs.combo_position_ids = positions.flatten()
        qkv = torch.randn(2, (32 + 2 * 8) * 128, device="cuda", dtype=torch.bfloat16)
        q, k, v = qkv.split([32 * 128, 8 * 128, 8 * 128], dim=-1)
        q = q.reshape(2, 32, 128)
        k = k.reshape(2, 8, 128)
        v = v.reshape(2, 8, 128)
        # Equal T/H/W decode positions make the interleaved MRoPE frequency
        # selection identical on all axes. Use a float32 RoPE reference.
        inv_freq = 1.0 / (
            5000000 ** (torch.arange(0, 128, 2, device="cuda").float() / 128)
        )
        freqs = positions.float()[:, :, None] * inv_freq[None, None, :]
        mixed = freqs[:, 0].clone()
        emb = torch.cat([mixed, mixed], dim=-1)[:, None, :]

        def rotate(x):
            xf = x.float()
            half = torch.cat([-xf[..., 64:], xf[..., :64]], dim=-1)
            return (xf * emb.cos() + half * emb.sin()).to(x.dtype)

        qr, kr = rotate(q), rotate(k)
        keys = [
            torch.randn(n, 8, 128, device="cuda", dtype=torch.bfloat16) for n in lengths
        ]
        values = [torch.randn_like(x) for x in keys]
        cache = fill_paged_kv_cache(
            keys,
            values,
            [n - 1 for n in lengths],
            inputs.kv_cache_kernel_block_id,
            64,
            8,
            128,
            torch.bfloat16,
            self.device,
            total_pages=3,
        )
        with patch.object(attn_factory, "DECODE_MHA_IMPS", [PyFlashinferDecodeImpl]):
            selected = attn_factory.get_fmha_impl(
                cfg, None, inputs, parallelism_config=config.parallelism_config
            )
        self.assertIsInstance(selected, PyFlashinferDecodeImpl)
        rotated_q = selected.rope_impl.forward(qkv.clone(), cache, selected.rope_params)
        torch.testing.assert_close(rotated_q.reshape_as(qr), qr, atol=1e-2, rtol=1e-2)
        for i, n in enumerate(lengths):
            page = inputs.kv_cache_kernel_block_id[i, (n - 1) // 64].item()
            offset = (n - 1) % 64
            torch.testing.assert_close(
                cache.kv_cache_base[page, 0, :, offset], kr[i], atol=1e-2, rtol=1e-2
            )
            torch.testing.assert_close(
                cache.kv_cache_base[page, 1, :, offset], v[i], atol=0, rtol=0
            )
            keys[i][-1], values[i][-1] = kr[i], v[i]
        actual = selected.forward(qkv.clone(), cache).reshape_as(qr)
        for i in range(2):
            expected = (
                F.scaled_dot_product_attention(
                    qr[i].float()[None, :, None, :],
                    keys[i].float().transpose(0, 1).repeat_interleave(4, dim=0)[None],
                    values[i].float().transpose(0, 1).repeat_interleave(4, dim=0)[None],
                )
                .to(torch.bfloat16)
                .reshape(32, 128)
            )
            torch.testing.assert_close(actual[i], expected, atol=1e-2, rtol=1e-2)

    def test_qwen3_prefill_unequal_mrope_axes(self):
        config = self._create_config(head_num=32, data_type="bf16")
        cfg = config.attn_configs
        cfg.need_rope_kv_cache = True
        cfg.max_seq_len = 256
        cfg.rope_config.style = RopeStyle.Mrope
        cfg.rope_config.dim = 128
        cfg.rope_config.base = 5000000
        cfg.rope_config.max_pos = 256
        cfg.rope_config.index_factor = 3
        sections = [24, 20, 20]
        (
            cfg.rope_config.mrope_dim1,
            cfg.rope_config.mrope_dim2,
            cfg.rope_config.mrope_dim3,
        ) = sections
        cfg.rope_config.mrope_interleaved = True
        inputs = self._create_prefill_attention_inputs(1, [2], 64, dtype=torch.bfloat16)
        inputs.is_cuda_graph = False
        positions = torch.tensor(
            [[2, 5, 9], [31, 17, 23]], device="cuda", dtype=torch.int32
        )
        inputs.combo_position_ids = positions.flatten()
        qkv = torch.randn(2, (32 + 2 * 8) * 128, device="cuda", dtype=torch.bfloat16)
        reference = qkv.clone().reshape(2, 48, 128)
        inv_freq = 1.0 / (
            cfg.rope_config.base
            ** (torch.arange(0, 128, 2, device="cuda").float() / 128)
        )
        freqs = positions.float()[:, :, None] * inv_freq[None, None, :]
        mixed = freqs[:, 0].clone()
        mixed[:, 1:60:3] = freqs[:, 1, 1:60:3]
        mixed[:, 2:60:3] = freqs[:, 2, 2:60:3]
        emb = torch.cat([mixed, mixed], dim=-1)[:, None, :]
        qk = reference[:, :40].float()
        rotated_half = torch.cat([-qk[..., 64:], qk[..., :64]], dim=-1)
        reference[:, :40] = (qk * emb.cos() + rotated_half * emb.sin()).to(
            torch.bfloat16
        )
        op = FusedRopeKVCachePrefillOpQKVOut(cfg)
        actual = op.forward(qkv.clone(), None, op.prepare(inputs)).reshape_as(reference)
        torch.testing.assert_close(
            actual[:, :40], reference[:, :40], atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(actual[:, 40:], reference[:, 40:], atol=0, rtol=0)

    def test_disabled_decode_kernel_stays_disabled(self):
        config = self._create_config(head_num=32, data_type="bf16")
        config.parallelism_config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
        inputs = self._create_attention_inputs_base(1, [8], 64, dtype=torch.bfloat16)
        fmha_config = FMHAConfig()
        fmha_config.disable_flashinfer_native = True
        with patch.object(attn_factory, "DECODE_MHA_IMPS", [PyFlashinferDecodeImpl]):
            with self.assertRaisesRegex(Exception, "can not find mha type"):
                attn_factory.get_fmha_impl(
                    config.attn_configs,
                    None,
                    inputs,
                    fmha_config=fmha_config,
                    parallelism_config=config.parallelism_config,
                )


if __name__ == "__main__":
    unittest.main()
