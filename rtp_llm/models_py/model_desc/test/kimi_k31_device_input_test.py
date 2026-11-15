"""Prepare device-only Prefill metadata before the eager KDA consumer."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaPagedPrefill
from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import HybridAttentionType
from rtp_llm.ops.compute_ops import PyAttentionInputs


class KimiK31DeviceInputTest(unittest.TestCase):
    def _warmup_attention(self):
        attention = SimpleNamespace(
            is_target_verify=False,
            is_mtp_draft_update=False,
            kv_cache_kernel_block_id_device=None,
            padding_offset=None,
        )
        attention.is_prefill = attention.is_s_padded = attention.is_warmup = True
        attention.input_lengths = torch.tensor([4], dtype=torch.int32)
        attention.prefix_lengths = torch.tensor([0], dtype=torch.int32)
        attention.input_lengths_device = attention.input_lengths.cuda()
        attention.prefix_lengths_device = attention.prefix_lengths.cuda()
        attention.cu_seqlens_device = torch.tensor(
            [0, 4], dtype=torch.int32, device="cuda"
        )
        attention.logical_token_count = attention.physical_token_count = 4
        return attention

    def test_prefill_requires_explicit_warmup_cache_backing(self):
        attention = self._warmup_attention()
        metadata = PageRRMlaMetadata(4096, 128, 1, 0)
        with self.assertRaisesRegex(ValueError, "page table"):
            metadata.prepare_prefill(attention)

    def test_dsa_warmup_accepts_reserved_zero_pages(self):
        attention = self._warmup_attention()
        attention.kv_cache_kernel_block_id_device = torch.zeros(
            (1, 32), dtype=torch.int32, device="cuda"
        )
        parallelism = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            prefill_cp_config=SimpleNamespace(
                kv_cache_sharded=True, is_enabled=lambda: False
            ),
        )
        params = DsaPageRRParams(
            SimpleNamespace(tokens_per_block=4096, kernel_tokens_per_block=128),
            parallelism,
        )
        params.prepare(attention)
        self.assertTrue(params.is_warmup)
        self.assertEqual(params.slot_mapping.tolist(), [0, 1, 2, 3])

    def test_graph_padding_alone_does_not_replace_real_kda_cache(self):
        attention = PyAttentionInputs()
        attention.is_s_padded = True
        cache = torch.ones((2, 128), device="cuda")
        original, backing = KdaPagedPrefill._warmup_cache_inputs(attention, cache)
        self.assertIs(original, attention)
        self.assertIs(backing, cache)
        attention.is_warmup = True
        attention.kv_cache_kernel_block_id = torch.zeros((1, 2), dtype=torch.int32)
        scratch_view, scratch = KdaPagedPrefill._warmup_cache_inputs(attention, cache)
        self.assertEqual(scratch_view.kv_cache_kernel_block_id.tolist(), [[1, 1]])
        self.assertEqual(attention.kv_cache_kernel_block_id.tolist(), [[0, 0]])
        self.assertNotEqual(scratch.data_ptr(), cache.data_ptr())
        self.assertTrue(torch.all(cache == 1).item())

    def test_eager_kda_preparation_receives_complete_host_mirrors(self):
        self.assertTrue(torch.cuda.is_available(), "requires the CUDA CI worker")
        attention = PyAttentionInputs()
        attention.is_prefill = True
        attention.cu_seqlens = torch.empty(0, dtype=torch.int32)
        attention.cu_seqlens_device = torch.tensor(
            [0, 4], dtype=torch.int32, device="cuda"
        )
        attention.prefix_lengths = torch.zeros(1, dtype=torch.int32, device="cuda")
        attention.kv_cache_block_id = torch.tensor(
            [[1]], dtype=torch.int32, device="cuda"
        )
        attention.kv_cache_kernel_block_id = attention.kv_cache_block_id
        attention.logical_token_count = attention.physical_token_count = 4
        attention.logical_request_count = 1
        inputs = SimpleNamespace(
            attention_inputs=attention,
            input_ids=torch.ones(4, dtype=torch.int32, device="cuda"),
        )
        info = SimpleNamespace(
            prefill_actual_input_lengths_cpu=torch.tensor([4], dtype=torch.int32),
            prefill_prefix_lengths_cpu=torch.tensor([0], dtype=torch.int32),
            prefill_qkv_restore_indice=torch.arange(4, device="cuda"),
            prefill_qkv_padding_mask=torch.ones(4, dtype=torch.bool, device="cuda"),
        )
        layout = QueryLayout(info, 2, 0, "cuda")
        prefill = object.__new__(KdaPagedPrefill)
        nn.Module.__init__(prefill)
        prefill.local_num_v_heads = 1
        prefill.head_k_dim = prefill.head_v_dim = 128
        prefill.linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=1,
            head_v_dim=128,
            head_k_dim=128,
            ssm_state_dtype=torch.float32,
            linear_conv_kernel_dim=4,
            qkv_size=384,
            conv_state_dtype=torch.bfloat16,
        )
        cache = SimpleNamespace(
            tag="linear",
            seq_size_per_block=128,
            kv_cache_base=torch.zeros(
                (2, prefill.linear_cache_converter.block_size_bytes // 2),
                dtype=torch.bfloat16,
                device="cuda",
            ),
        )
        model = object.__new__(KimiK3Model)
        nn.Module.__init__(model)
        model.dsa_workspace = object()
        model._query_layout, model._outer_layout = layout, None
        model.layers = [
            SimpleNamespace(
                layer_type=HybridAttentionType.LINEAR,
                attention=SimpleNamespace(prefill=prefill),
            )
        ]
        model.kv_cache = SimpleNamespace(get_layer_cache=lambda index: cache)
        prepared_attention = object()
        # DSA projection/kernel preparation is independent of the subsequent
        # real KDA planner; retain its selected QueryLayout and FMHA identity.
        model._prepare_dsa = lambda inputs, is_cuda_graph: prepared_attention

        self.assertIs(model.prepare_fmha_impl(inputs), prepared_attention)
        _, metadata, tag, _ = model._prepared_kda_prefill
        self.assertEqual(tag, "linear")
        self.assertEqual(len(metadata.prefill_cuda_inputs), 1)
        for name in (
            "cu_seqlens",
            "prefix_lengths",
            "kv_cache_block_id",
            "kv_cache_kernel_block_id",
        ):
            self.assertEqual(getattr(attention, name).device.type, "cpu", name)
        torch.testing.assert_close(
            attention.cu_seqlens, torch.tensor([0, 4], dtype=torch.int32)
        )
        self.assertEqual(
            attention.kv_cache_block_id.data_ptr(),
            attention.kv_cache_kernel_block_id.data_ptr(),
        )
        self.assertTrue(attention.cu_seqlens_device.is_cuda)


if __name__ == "__main__":
    unittest.main()
