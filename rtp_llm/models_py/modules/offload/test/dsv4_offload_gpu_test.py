"""Native V4 striped FP8 pages: host fetch, graph invalidation and writes."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.test_compressor_fp8_per_token import (
    _build_compressor,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_flash_mla_decode_kernel_layout import (
    _call_flash_mla,
    _make_q,
    _pack_model1_fp8_cache,
)
from rtp_llm.models_py.modules.offload.layout import PagedRowLayout
from rtp_llm.models_py.modules.offload.plan import LayerOffloadPlan
from rtp_llm.models_py.modules.offload.runtime import KVWrite, SparseOffloadRuntime


def make_runtime(maximum=1024, page=32, total=16):
    block, hbm = 64, 2
    stride = ((page * 584 + 575) // 576) * 576

    def allocate(pages, host):
        raw = torch.zeros(
            (pages, stride),
            dtype=torch.uint8,
            device="cpu" if host else "cuda",
            pin_memory=host,
        )
        payload = raw.as_strided((pages, page, 576), (stride, 576, 1))
        scales = raw.as_strided((pages, page, 8), (stride, 8, 1), page * 576)
        return raw, (payload, scales)

    host_raw, host = allocate((total - hbm) * block // page, True)
    resident_raw, resident = allocate((hbm * block + maximum) // page, False)
    storage = SimpleNamespace(
        tag="csa_kv",
        owner_layer=0,
        total_blocks=total,
        hbm_blocks=hbm,
        rows_per_block=block,
        rows_per_page=page,
        max_selected_rows=maximum,
        host_planes=host,
        resident_planes=resident,
        generations=torch.ones(total, dtype=torch.int64, device="cuda"),
    )
    return SparseOffloadRuntime((storage,), PagedRowLayout()), host_raw, resident_raw


class Dsv4OffloadGpuTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.context = torch.cuda.stream(self.stream)
        self.context.__enter__()

    def tearDown(self):
        self.stream.synchronize()
        self.context.__exit__(None, None, None)

    def test_host_fetch_matches_flash_mla_including_scales(self):
        runtime, host_raw, resident_raw = make_runtime()
        kv = torch.randn(1024, 512, dtype=torch.bfloat16, device="cuda") * 0.5
        reference = _pack_model1_fp8_cache(kv, 32)
        reference_raw = reference.as_strided((32, 19008), (19008, 1))
        host_raw.copy_(reference_raw[4:], non_blocking=False)
        resident_raw[:4].copy_(reference_raw[:4])
        ids = torch.randint(0, 1024, (2, 4, 128), device="cuda", dtype=torch.int32)
        ids[:, :, -5:] = -1
        runtime.begin(ids)
        view = runtime.acquire(0)
        resident_cache = resident_raw.as_strided(
            (resident_raw.shape[0], 32, 1, 584), (19008, 584, 584, 1)
        )
        q = _make_q(2, 4)
        expected = _call_flash_mla(q, reference, ids)
        actual = _call_flash_mla(q, resident_cache, view.indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_graph_replay_after_host_reimport_and_write(self):
        runtime, host_raw, resident_raw = make_runtime(128)
        storage = runtime.storages[0]
        ids = torch.arange(128, 256, device="cuda", dtype=torch.int32).view(1, 128)
        values = (
            torch.randint(0, 256, (128, 576), device="cuda", dtype=torch.uint8),
            torch.randint(0, 256, (128, 8), device="cuda", dtype=torch.uint8),
        )
        runtime.write(0, KVWrite(ids.reshape(-1), values))
        runtime.begin(ids)
        runtime.acquire(0)
        runtime.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            runtime.begin(ids)
            view = runtime.acquire(0)
            payload = view.planes[0][
                view.indices.long() // 32, view.indices.long() % 32
            ].clone()
            scales = view.planes[1][
                view.indices.long() // 32, view.indices.long() % 32
            ].clone()
        for offset in (0, 1, 7):
            # Simulate P/D importing new bytes into recycled host blocks.
            for plane, data in zip(storage.host_planes, values):
                plane[:4].copy_((data + offset).reshape_as(plane[:4]).cpu())
            storage.generations[2:4].add_(1)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(payload[0], values[0] + offset, rtol=0, atol=0)
            torch.testing.assert_close(scales[0], values[1] + offset, rtol=0, atol=0)

    def test_compressor_write_matches_native_pages_for_verify(self):
        compressor = _build_compressor(
            dim=64, head_dim=512, rope_head_dim=64, compress_ratio=4
        )
        runtime, host_raw, resident_raw = make_runtime()
        reference_raw = torch.zeros((32, 19008), device="cuda", dtype=torch.uint8)
        reference = reference_raw.as_strided((32, 32, 584), (19008, 584, 1))
        state = torch.randn(5, 256, 2048, device="cuda", dtype=torch.float32)
        saved_state = state.clone()
        state_bt = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int32)
        kv_bt = torch.tensor([[5, 6], [9, 10]], device="cuda", dtype=torch.int32)
        compressor.set_pool_context(
            reference,
            kv_bt,
            32,
            state,
            state_bt,
            256,
            state_tokens_per_block=256,
            kv_tokens_per_block=128,
        )
        positions = torch.tensor(
            [123, 124, 125, 126, 251, 252, 253, 254], device="cuda"
        )
        b_idx = torch.arange(2, device="cuda").repeat_interleave(4)
        meta = compressor.prepare_metadata(
            positions,
            b_idx,
            has_prefix=True,
            is_batched=True,
            seq_start_per_req=positions.view(2, 4)[:, 0].contiguous(),
            cu_seq_per_req=torch.tensor([0, 4, 8], device="cuda"),
        )
        kv = torch.randn(8, 1024, device="cuda", dtype=torch.float32)
        score = torch.randn_like(kv)
        compressor._launch(kv, score, meta)
        state.copy_(saved_state)
        compressor.offload_plan = LayerOffloadPlan(runtime, 0, True, True)
        raw = torch.zeros((1, 5184), device="cuda", dtype=torch.uint8)
        compressor.offload_staging = raw.as_strided((1, 8, 584), (5184, 584, 1))
        compressor._launch(kv, score, meta)
        valid = meta.kv_slots >= 0
        ids = meta.kv_slots[valid].to(torch.int32).contiguous()
        runtime.begin(ids)
        view = runtime.acquire(0)
        for resident, width, start in zip(view.planes, (576, 8), (0, 32 * 576)):
            expected_plane = reference_raw.as_strided(
                (32, 32, width), (19008, width, 1), start
            )
            actual = resident[view.indices.long() // 32, view.indices.long() % 32]
            expected = expected_plane[ids.long() // 32, ids.long() % 32]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_mega_csa_verify_and_graph_match_resident_path(self):
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
        native = _make_pools(device, batch_size=2, max_seq_len=1024)
        offloaded = _make_pools(device, batch_size=2, max_seq_len=1024)
        _fill_random_context(native, device, seed=123)
        _fill_random_context(offloaded, device, seed=123)
        runtime, host_raw, resident_raw = make_runtime(
            maximum=4096, page=64, total=offloaded.tensors[CSA_KV].shape[0]
        )
        host_raw.copy_(offloaded.tensors[CSA_KV][2:].cpu())
        resident_raw[:2].copy_(offloaded.tensors[CSA_KV][:2])
        hidden = (
            torch.randn(
                (2, 4, HC, FLASH_GEOMETRY.dim), device=device, dtype=torch.bfloat16
            )
            * 0.05
        )

        def metadata(pools):
            return build_decode_metadata_fp8(
                attention_inputs=torch.full(
                    (2,), 765, device=device, dtype=torch.int32
                ),
                q_len=4,
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
        mega_runtime.begin_decode(native_meta)
        expected = adapter.forward_attention_sublayer(
            block, hidden.clone(), native_meta, kv_cache=native.kv_cache
        ).clone()
        block.attn.configure_sparse_offload(LayerOffloadPlan(runtime, 0, True, True), 8)
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
