"""Mega CSA offload parity on CUDA 13 SM100/SM103 workers."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.offload.test.dsv4_offload_gpu_test import make_runtime
from rtp_llm.models_py.modules.offload.plan import LayerOffloadPlan


class Dsv4OffloadMegaGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        capability = torch.cuda.get_device_capability()
        if capability not in ((10, 0), (10, 3)):
            raise unittest.SkipTest(f"Mega CSA requires SM100/SM103, got {capability}")

    def setUp(self):
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.context = torch.cuda.stream(self.stream)
        self.context.__enter__()

    def tearDown(self):
        self.stream.synchronize()
        self.context.__exit__(None, None, None)

    def test_verify_and_graph_match_resident_at_deployment_batch(self):
        for batch_size, q_len in ((2, 4), (64, 4), (150, 1)):
            with self.subTest(batch_size=batch_size, q_len=q_len):
                self._compare_mega_csa(batch_size, q_len)

    def test_deployment_verify_shape_uses_non_mega_path(self):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_adapter import (
            MegaCSAAdapter,
        )
        hidden = torch.empty((150, 4, 1, 1), device="cuda")
        self.assertFalse(
            MegaCSAAdapter.supports_decode_shape(
                hidden, SimpleNamespace(batch_size=150, q_len_per_req=4)
            )
        )

    @torch.inference_mode()
    def _compare_mega_csa(self, batch_size, q_len):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            build_decode_metadata_fp8,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
            FLASH_GEOMETRY,
            HC,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.test.test_mega_csa_rtp_eager import (
            MegaCSARTPEagerTest,
            _make_pools,
            _fill_random_context,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV

        device = torch.device("cuda", torch.cuda.current_device())
        block, mega_runtime, adapter = MegaCSARTPEagerTest._make_layer(
            device, FLASH_GEOMETRY
        )
        native = _make_pools(device, batch_size=batch_size, max_seq_len=1024)
        offloaded = _make_pools(device, batch_size=batch_size, max_seq_len=1024)
        _fill_random_context(native, device, seed=123)
        _fill_random_context(offloaded, device, seed=123)
        runtime, host_raw, resident_raw = make_runtime(
            maximum=batch_size * 4 * 2048, page=64, total=offloaded.tensors[CSA_KV].shape[0]
        )
        host_raw.copy_(offloaded.tensors[CSA_KV][2:].cpu())
        resident_raw[:2].copy_(offloaded.tensors[CSA_KV][:2])
        hidden = (
            torch.randn(
                (batch_size, q_len, HC, FLASH_GEOMETRY.dim),
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.05
        )

        def metadata(pools):
            return build_decode_metadata_fp8(
                attention_inputs=torch.full(
                    (batch_size,), 765, device=device, dtype=torch.int32
                ),
                q_len=q_len,
                window_size=128,
                head_dim=512,
                max_seq_len=1024,
                compress_ratios=[4],
                index_topk=512,
                device=device,
                paged_block_tables=pools.block_tables,
                paged_pool_entries_per_block=pools.entries_per_block,
                paged_pool_tokens_per_block=pools.tokens_per_block,
            )

        native_meta, host_meta = metadata(native), metadata(offloaded)
        self.assertTrue(adapter.supports_decode_shape(hidden, host_meta))
        mega_runtime.begin_decode(native_meta)
        expected = adapter.forward_attention_sublayer(
            block, hidden.clone(), native_meta, kv_cache=native.kv_cache
        ).clone()
        block.attn.configure_sparse_offload(
            LayerOffloadPlan(runtime, 0, True, True), batch_size * 4
        )
        mega_runtime.begin_decode(host_meta)
        actual = adapter.forward_attention_sublayer(
            block, hidden.clone(), host_meta, kv_cache=offloaded.kv_cache
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue((runtime.mapping[128:] >= 0).any().item())
        runtime.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            mega_runtime.begin_decode(host_meta)
            captured = adapter.forward_attention_sublayer(
                block, hidden, host_meta, kv_cache=offloaded.kv_cache
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)

if __name__ == "__main__":
    unittest.main(verbosity=2)

