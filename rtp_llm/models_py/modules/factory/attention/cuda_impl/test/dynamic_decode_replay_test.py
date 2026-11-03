import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from base_attention_test import BaseAttentionTest

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeImpl,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
    bench_backend_grid,
)
from rtp_llm.ops import KvCacheDataType, RopeStyle, get_rope_cache_once
from rtp_llm.ops.compute_ops import LayerKVCache


class DynamicDecodeReplayTest(BaseAttentionTest):
    def test_00_recorder_preserves_serving_rope_cache(self):
        from rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder import (
            TensorRecorder,
        )

        config, _, _ = self.fixture()
        config.attn_configs.rope_config.style = RopeStyle.Base
        config.attn_configs.rope_config.dim = 64
        model = SimpleNamespace(
            config=SimpleNamespace(
                getAttentionConfigs=lambda size: config.attn_configs
            ),
            parallelism_config=config.parallelism_config,
            prepare_fmha_impl=lambda: None,
        )
        recorder = TensorRecorder(model)
        config.attn_configs.rope_config.style = RopeStyle.No
        cache = get_rope_cache_once(config.attn_configs.rope_config, 128)
        self.assertTrue(cache.used)
        self.assertEqual(cache.dim, 64)
        self.assertTrue(torch.isfinite(cache.data).all())
        recorder.close()

    def fixture(self, fp8=False):
        config = self._create_config(
            head_num=4,
            head_num_kv=4,
            size_per_head=64,
            seq_size_per_block=64,
            data_type="bf16",
        )
        config.attn_configs.max_seq_len = 128
        config.attn_configs.rope_config.style = RopeStyle.No
        config.attn_configs.need_rope_kv_cache = True
        config.attn_configs.kv_cache_dtype = (
            KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
        )
        inputs = self._create_attention_inputs_base(2, [128, 128], 64, torch.bfloat16)
        inputs.sequence_lengths_plus_1_device = torch.full(
            (2,), 128, dtype=torch.int32, device="cuda"
        )
        inputs.is_cuda_graph = True
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
        data = torch.randn(4, 2, 4, 64, 64, device="cuda").to(dtype)
        scale = torch.ones(4, 2 * 4 * 64, device="cuda") if fp8 else None
        return config, inputs, LayerKVCache(data, 64, 0, "default", scale)

    def test_benchmark_calls_prepare_and_preserves_engine_inputs(self):
        for fp8 in (False, True):
            config, inputs, cache = self.fixture(fp8)
            before = inputs.sequence_lengths.clone()
            cache_before = cache.kv_cache_base.float().clone()
            calls = []
            original = PyFlashinferDecodeImpl.prepare_cuda_graph

            def prepare(instance, value):
                calls.append(value.sequence_lengths.tolist())
                return original(instance, value)

            with patch.object(PyFlashinferDecodeImpl, "prepare_cuda_graph", prepare):
                times = bench_backend_grid(
                    PyFlashinferDecodeImpl,
                    config.attn_configs,
                    inputs,
                    cache,
                    config.parallelism_config,
                    [63, 64, 127],
                    warmup=2,
                    iters=3,
                    attention_layer_count=2,
                    l2_fill_mode="none",
                )
            self.assertEqual(len(times), 3)
            self.assertTrue(all(value > 0 for value in times))
            self.assertEqual(len(calls), 15)
            self.assertTrue(torch.equal(inputs.sequence_lengths, before))
            self.assertTrue(torch.equal(cache.kv_cache_base.float(), cache_before))

    def test_actual_replay_obeys_tensor_and_cuda_core_plan_contract(self):
        for fp8 in (False, True):
            config, inputs, cache = self.fixture(fp8)
            impl = PyFlashinferDecodeImpl(
                config.attn_configs, inputs, config.parallelism_config
            )
            qkv = torch.randn(2, 12 * 64, dtype=torch.bfloat16, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    impl.forward(qkv, cache)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = impl.forward(qkv, cache)
            torch.cuda.synchronize()
            op = impl.fmha_impl
            with patch.object(
                op, "_plan_decode_wrapper", wraps=op._plan_decode_wrapper
            ) as plan:
                for length in (62, 63, 64, 65):
                    inputs.sequence_lengths.fill_(length)
                    inputs.sequence_lengths_plus_1_device.fill_(length + 1)
                    impl.prepare_cuda_graph(inputs)
                    graph.replay()
                    torch.cuda.synchronize()
                    actual = captured.clone()
                    expected = impl.forward(qkv, cache).clone()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
                self.assertEqual(plan.call_count, 4 if op.use_tensor_core else 2)
            del graph, impl
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
