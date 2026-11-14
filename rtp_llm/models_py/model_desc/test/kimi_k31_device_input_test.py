"""Prepare device-only Prefill metadata before the eager KDA consumer."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaPagedPrefill
from rtp_llm.models_py.modules.hybrid.query_layout import QueryLayout
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import HybridAttentionType
from rtp_llm.ops.compute_ops import PyAttentionInputs


class KimiK31DeviceInputTest(unittest.TestCase):
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
