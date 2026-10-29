import unittest

import torch
import torch.nn.functional as F
from base_attention_test import BaseAttentionTest

from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa import XQADecodeImpl
from rtp_llm.ops import KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache


class DynamicDecodeXQATest(BaseAttentionTest):
    def _check_replay(self, fp8):
        config = self._create_config(
            head_num=8,
            head_num_kv=4,
            size_per_head=128,
            seq_size_per_block=64,
            data_type="bf16",
        )
        config.attn_configs.rope_config.style = RopeStyle.No
        config.attn_configs.need_rope_kv_cache = False
        config.attn_configs.kv_cache_dtype = (
            KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
        )
        inputs = self._create_attention_inputs_base(2, [128, 128], 64, torch.bfloat16)
        inputs.decode_cu_seqlens = torch.arange(3, dtype=torch.int32)
        inputs.is_cuda_graph = True
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        data = torch.randn(4, 2, 4, 64, 128, device="cuda").to(dtype)
        cache = LayerKVCache(data, 64, 0, "default")
        q = torch.randn(2, 8, 128, dtype=torch.bfloat16, device="cuda")
        impl = XQADecodeImpl(config.attn_configs, inputs, config.parallelism_config)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                impl.forward(q, cache)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = impl.forward(q, cache)

        for length in (63, 65, 127):
            inputs.sequence_lengths.fill_(length - 1)
            q.normal_()
            impl.prepare_cuda_graph(inputs)
            graph.replay()
            torch.cuda.synchronize()
            actual = captured.clone()
            eager = impl.forward(q, cache).clone()
            torch.testing.assert_close(actual, eager, rtol=1e-3, atol=1e-3)
            for batch in range(2):
                # Read the same paged cache, including FP8 rounding, for SDPA.
                pages = data[batch * 2 : (batch + 1) * 2].float()
                k, v = [
                    pages[:, kind].permute(1, 0, 2, 3).reshape(4, 128, 128)
                    for kind in (0, 1)
                ]
                expected = F.scaled_dot_product_attention(
                    q[batch].float().unsqueeze(1),
                    k[:, :length].repeat_interleave(2, dim=0),
                    v[:, :length].repeat_interleave(2, dim=0),
                ).squeeze(1)
                # Match the existing FP8 decode suite's quantized-math tolerance.
                tolerance = 0.04 if fp8 else 0.02
                torch.testing.assert_close(
                    actual.reshape(2, 8, 128)[batch].float(),
                    expected,
                    rtol=tolerance,
                    atol=tolerance,
                )
        del graph, impl

    def test_bf16_kv_capture_replay(self):
        self._check_replay(False)

    def test_fp8_kv_capture_replay(self):
        self._check_replay(True)


if __name__ == "__main__":
    unittest.main()
