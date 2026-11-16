"""Native V4 striped FP8 pages: host fetch, graph invalidation and writes."""

import unittest
import weakref
from types import MethodType, SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test.test_compressor_fp8_per_token import (
    _build_compressor,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_flash_mla_decode_kernel_layout import (
    _call_flash_mla,
    _make_q,
    _pack_model1_fp8_cache,
)
from rtp_llm.models_py.modules.offload.layout import FetchRows, PagedRowLayout
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

    def test_fp8_fetch_preserves_all_other_bytes_eager_and_graph(self):
        # Different page/row strides and nonzero plane offsets must be respected.
        # Unaligned pointers and strided bytes exercise the generic fallback.
        for page, offset, step, row_padding in (
            (32, 4, 1, 0),
            (64, 12, 1, 0),
            (32, 5, 1, 0),
            (32, 4, 2, 0),
            (32, 4, 1, 4),
        ):
            with self.subTest(page=page, offset=offset, step=step, padding=row_padding):
                hbm, n = 128, 37

                def planes(raw, base, padding):
                    pages, stride = raw.shape
                    return tuple(
                        raw.as_strided(
                            (pages, page, width),
                            (stride, (width + padding) * step, step),
                            base + start,
                        )
                        for width, start in (
                            (576, 0), (8, page * (576 + padding) * step)
                        )
                    )

                stride = offset + page * (584 + 2 * row_padding) * step + 320
                stride = (stride + 3) // 4 * 4
                host_raw = torch.randint(
                    0, 256, (8, stride), dtype=torch.uint8, pin_memory=True
                )
                original_host = host_raw.clone()
                resident_raw = torch.empty(
                    (8, stride + 64 + page * 2 * row_padding * step),
                    dtype=torch.uint8,
                    device="cuda",
                )
                hosts = planes(host_raw, offset, row_padding)
                residents = planes(resident_raw, offset + 8, 2 * row_padding)
                source = torch.randperm(8 * page)[:n]
                target = torch.randperm(8 * page - hbm)[:n] + hbm
                positions_cpu = torch.randperm(n).to(torch.int32)
                ids = (source + hbm).to(device="cuda", dtype=torch.int32)
                slots = target.to(device="cuda", dtype=torch.int32)
                count = torch.zeros((), dtype=torch.int32, device="cuda")
                misses = FetchRows(ids, slots, positions_cpu.cuda(), count)
                storage = SimpleNamespace(
                    host_planes=hosts,
                    resident_planes=residents,
                    rows_per_page=page,
                    hbm_blocks=2,
                    rows_per_block=64,
                )
                layout = PagedRowLayout()
                layout.fetch(storage, misses)
                self.stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    layout.fetch(storage, misses)
                for captured in (False, True):
                    for miss_count in (0, 1, 7, n):
                        resident_raw.fill_(0xA5)
                        count.fill_(miss_count)
                        if captured:
                            graph.replay()
                        else:
                            layout.fetch(storage, misses)
                        expected = torch.full_like(resident_raw, 0xA5, device="cpu")
                        selected = positions_cpu[:miss_count].long()
                        src, dst = source[selected], target[selected]
                        for host, reference in zip(
                            hosts, planes(expected, offset + 8, 2 * row_padding)
                        ):
                            reference[dst // page, dst % page] = host[
                                src // page, src % page
                            ]
                        self.stream.synchronize()
                        # Includes HBM prefix, unselected rows, plane gaps and padding.
                        self.assertTrue(torch.equal(resident_raw.cpu(), expected))
                        self.assertTrue(torch.equal(host_raw, original_host))

    def test_warmup_cache_release_and_production_rebind(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        model = DeepSeekV4Model.__new__(DeepSeekV4Model)
        torch.nn.Module.__init__(model)
        model._materialized = True
        model.fp8_kv_cache = True
        model._max_generate_batch_size = 4
        model._gen_num_per_cycle = 3
        model.config = SimpleNamespace(
            kv_cache_spec_descs=[
                [
                    SimpleNamespace(
                        tag="csa_kv",
                        offload=SimpleNamespace(enabled=True),
                        offload_runtime_group=None,
                    )
                ]
            ]
        )
        attn = SimpleNamespace(compress_ratio=4, compressor=SimpleNamespace())
        attn.configure_sparse_offload = MethodType(
            AttentionFP8.configure_sparse_offload, attn
        )
        model.v4 = SimpleNamespace(layers=[SimpleNamespace(attn=attn)])

        def make_cache():
            runtime, _, _ = make_runtime(128, total=3)
            storage = runtime.storages[0]
            return SimpleNamespace(
                layer_count=1,
                group_tags=["csa_kv"],
                get_offload_storage=lambda layer, tag: storage,
            )

        warmup = make_cache()
        model.initialize(SimpleNamespace(kv_cache=warmup))
        runtime_ref = weakref.ref(model.offload_plans[0, "csa_kv"].runtime)
        staging_ref = weakref.ref(attn.compressor.offload_staging)
        self.assertEqual(staging_ref().shape, (1, 16, 584))
        model.initialize(SimpleNamespace(kv_cache=warmup))
        self.assertIs(model.offload_plans[0, "csa_kv"].runtime, runtime_ref())
        attn._kv_cache = warmup
        attn._block_tables_by_type = {"csa_kv": torch.zeros(1, device="cuda")}
        model.release_sparse_kv_offload()
        self.assertIsNone(runtime_ref())
        self.assertIsNone(staging_ref())
        self.assertIs(attn._kv_cache, warmup)
        self.assertIsNotNone(attn._block_tables_by_type)
        model.initialize(SimpleNamespace(kv_cache=None))
        self.assertIsNone(attn._kv_cache)
        self.assertIsNone(attn._block_tables_by_type)
        self.assertIsNone(model._offload_signature)
        production = make_cache()
        model.initialize(SimpleNamespace(kv_cache=production))
        self.assertIs(
            model.offload_plans[0, "csa_kv"].storage,
            production.get_offload_storage(0, "csa_kv"),
        )
        model.release_kv_cache()

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
        for batch_size in (2, 150):
            with self.subTest(batch_size=batch_size):
                self._compare_compressor_write(batch_size)

    def _compare_compressor_write(self, batch_size):
        tokens = batch_size * 4
        compressor = _build_compressor(
            dim=64, head_dim=512, rope_head_dim=64, compress_ratio=4
        )
        runtime, _, _ = make_runtime(
            maximum=((tokens + 31) // 32) * 32, total=batch_size + 2
        )
        pages = (batch_size + 2) * 2
        reference_raw = torch.zeros((pages, 19008), device="cuda", dtype=torch.uint8)
        reference = reference_raw.as_strided((pages, 32, 584), (19008, 584, 1))
        state = torch.randn(2 * batch_size + 1, 256, 2048, device="cuda", dtype=torch.float32)
        saved_state = state.clone()
        state_bt = torch.arange(
            1, 2 * batch_size + 1, device="cuda", dtype=torch.int32
        ).view(batch_size, 2)
        kv_bt = state_bt.clone()
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
        positions = (
            123 + (torch.arange(batch_size, device="cuda") % 2)[:, None] * 128
            + torch.arange(4, device="cuda")
        ).flatten()
        b_idx = torch.arange(batch_size, device="cuda").repeat_interleave(4)
        meta = compressor.prepare_metadata(
            positions,
            b_idx,
            has_prefix=True,
            is_batched=True,
            seq_start_per_req=positions.view(batch_size, 4)[:, 0].contiguous(),
            cu_seq_per_req=torch.arange(0, tokens + 1, 4, device="cuda"),
        )
        kv = torch.randn(tokens, 1024, device="cuda", dtype=torch.float32)
        score = torch.randn_like(kv)
        compressor._launch(kv, score, meta)
        state.copy_(saved_state)
        compressor.offload_plan = LayerOffloadPlan(runtime, 0, True, True)
        stride = ((tokens * 584 + 575) // 576) * 576
        raw = torch.zeros((1, stride), device="cuda", dtype=torch.uint8)
        compressor.offload_staging = raw.as_strided((1, tokens, 584), (stride, 584, 1))
        valid = meta.kv_slots >= 0
        ids = meta.kv_slots[valid].to(torch.int32).contiguous()

        class DelayedLayout(PagedRowLayout):
            def fetch(self, storage, misses):
                torch.cuda._sleep(20_000_000)
                super().fetch(storage, misses)

        runtime.layout = DelayedLayout()

        def run():
            state.copy_(saved_state)
            runtime.begin(ids)
            compressor._launch(kv, score, meta)
            view = runtime.acquire(0)
            return tuple(
                plane[view.indices.long() // 32, view.indices.long() % 32].clone()
                for plane in view.planes
            )

        # Old Host bytes may be fetched while the new boundary KV is computed.
        # The writer must join the fetch before publishing the new bytes.
        run()
        self.stream.synchronize()
        runtime.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            captured = run()
        for replay in (False, True, True):
            self.stream.synchronize()
            for host in runtime.storages[0].host_planes:
                host.zero_()
            runtime.storages[0].generations.add_(1)
            if replay:
                graph.replay()
                actual = captured
            else:
                actual = run()
            for observed, width, start in zip(actual, (576, 8), (0, 32 * 576)):
                expected_plane = reference_raw.as_strided(
                    (pages, 32, width), (19008, width, 1), start
                )
                expected = expected_plane[ids.long() // 32, ids.long() % 32]
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
