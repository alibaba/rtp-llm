"""Captured target adapter state refresh, rejection and physical-page migration."""

import unittest
from types import SimpleNamespace

import test_attention
import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import (
    PAIR_OWNERS,
    CacheLayout,
    CacheRegion,
    RegionSlot,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.compact_writer import write_compact
from rtp_llm.models_py.modules.dsv41.decode_compressor import (
    PAIR_SNAPSHOT_BYTES,
    V41DecodePairState,
)
from rtp_llm.models_py.modules.dsv41.decode_fmha_impl import V41DecodeFmhaImpl
from rtp_llm.ops.compute_ops import KVCacheRegionName


class DecodeFmhaGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_attention.AttentionGpuTest.setUpClass()

    @classmethod
    def tearDownClass(cls):
        test_attention.AttentionGpuTest.tearDownClass()

    def model(self, width):
        fixture = test_attention.AttentionGpuTest()
        fixture.setUp()
        # CP1 is a local byte-movement fixture; distributed topology is tested
        # by the real CP8 P/D service, not represented by this two-slot graph.
        fixture.layout = layout = CacheLayout(
            cp_size=1,
            speculative_tokens=5 if width == 6 else 0,
            draft_enabled=width == 6,
        )
        model = SimpleNamespace(
            layout=layout,
            config=SimpleNamespace(max_seq_len=256, dsv41_replay_mode="full"),
            target=SimpleNamespace(
                embedding=torch.empty((0, 5120), dtype=torch.bfloat16, device="cuda"),
                blocks=[
                    SimpleNamespace(attention=fixture.model(layer))
                    for layer in range(40)
                ],
            ),
            _groups={},
            _pages={},
            _pair_pools={},
        )
        units = []
        names = {
            CacheRegion.SWA: KVCacheRegionName.SWA_KV,
            CacheRegion.GLOBAL: KVCacheRegionName.DSV41_GLOBAL_KV,
            CacheRegion.INDEX_K: KVCacheRegionName.DSV41_INDEX_KV,
        }
        for page in layout.pages:
            pages = CompactPages(
                torch.zeros(
                    (5, page.page_stride_bytes), dtype=torch.uint8, device="cuda"
                ),
                page.slot.region,
                page.entries,
            )
            model._pages[page.slot] = pages
            model._groups[(page.slot.owner_layer, int(names[page.slot.region]))] = len(
                units
            )
            units.append(
                layout.reuse_unit
                if page.slot.region == CacheRegion.SWA
                else layout.token_block_size
            )
            dimension = 128 if page.slot.region == CacheRegion.INDEX_K else 512
            values = torch.zeros(
                (4 * page.entries, dimension), dtype=torch.bfloat16, device="cuda"
            )
            if dimension == 128:
                values.fill_(1)
            else:
                values[:, :448] = (512 / 448) ** 0.5
            write_compact(
                values,
                pages,
                torch.arange(
                    page.entries, 5 * page.entries, dtype=torch.int64, device="cuda"
                ),
            ).check()
        snapshots = layout.speculative_tokens + 2
        for layer in PAIR_OWNERS:
            pool = torch.zeros(
                (5, ((snapshots * PAIR_SNAPSHOT_BYTES + 511) // 512) * 512),
                dtype=torch.uint8,
                device="cuda",
            )
            initial = V41DecodePairState(
                pool[
                    :,
                    (snapshots - 1)
                    * PAIR_SNAPSHOT_BYTES : snapshots
                    * PAIR_SNAPSHOT_BYTES,
                ]
            )
            initial.positions[1] = 127
            initial.valid[1] = 1
            initial.partial_kv[1, :448] = 1
            model._pair_pools[layer] = pool
            model._groups[(layer, int(KVCacheRegionName.DSV41_PAIR_STATE))] = len(units)
            units.append(layout.reuse_unit)
        model.kv_cache = SimpleNamespace(group_seq_size_per_block=units)
        return fixture, model

    @staticmethod
    def inputs(model, width, start, token, reload=False):
        values = torch.arange(token, token + width, dtype=torch.int32, device="cuda")
        ids = torch.cat((values, torch.zeros_like(values)))
        history = torch.zeros((2, width, 3), dtype=torch.int32, device="cuda")
        chain = torch.cat(
            (torch.tensor([71, 72, 73], dtype=torch.int32, device="cuda"), values)
        )
        for row in range(width):
            history[0, row] = chain[row : row + 3]
        valid = torch.tensor(
            [True] * width + [False] * width, dtype=torch.bool, device="cuda"
        )
        table = torch.tensor(
            [[3, 4] if reload else [1, 2], [0, 0]], dtype=torch.int32, device="cuda"
        )
        ranges = torch.tensor(
            [[[max(start - 128, 0), start, 0]] * 43, [[0, 0, 0]] * 43],
            dtype=torch.int64,
        )
        return SimpleNamespace(
            input_ids=ids,
            request_id=torch.tensor([101, -1], dtype=torch.int64),
            v41_is_fake=torch.tensor([False, True]),
            v41_state_ready=torch.tensor([True, False]),
            v41_execution_context=torch.tensor(
                [[0, start, start, start], [0, 0, 0, 0]], dtype=torch.int64
            ),
            v41_swa_ranges=ranges,
            v41_token_types=torch.full_like(ids, -1),
            v41_token_valid=valid,
            engram_history_ids=history.flatten(0, 1),
            engram_history_valid=valid[:, None].expand(-1, 3).contiguous(),
            attention_inputs=SimpleNamespace(
                is_prefill=width == 6,
                is_target_verify=width == 6,
                context_parallel_info=None,
                input_lengths=torch.tensor([width, 0], dtype=torch.int32),
                prefix_lengths=(
                    torch.tensor([start, 0], dtype=torch.int32)
                    if width == 6
                    else torch.empty(0, dtype=torch.int32)
                ),
                sequence_lengths=(
                    torch.tensor([start, 0], dtype=torch.int32)
                    if width == 1
                    else torch.empty(0, dtype=torch.int32)
                ),
                kv_cache_kernel_block_id_device_by_group=[
                    table.clone() for _ in model.kv_cache.group_seq_size_per_block
                ],
            ),
        )

    def test_q1_and_q6_replay_rebinds_rows_pages_and_retained_snapshots(self):
        for width in (1, 6):
            with self.subTest(width=width), torch.inference_mode():
                fixture, model = self.model(width)
                original = self.inputs(model, width, 127, 100)
                impl = V41DecodeFmhaImpl(model, original, query_width=width)
                hidden = fixture.hidden(2 * width)

                def run():
                    impl.begin_forward()
                    result = torch.stack(
                        [
                            impl.context.layers[layer](hidden, impl.context)
                            for layer in range(40)
                        ]
                    )
                    impl.finish_forward()
                    return result

                before = [pages.data.clone() for pages in model._pages.values()]
                before += [pool.clone() for pool in model._pair_pools.values()]
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = run()
                after = [pages.data for pages in model._pages.values()] + list(
                    model._pair_pools.values()
                )
                for actual, expected in zip(after, before):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for start, token, reload in ((127, 100, False), (128, 200, True)):
                    if reload:
                        for pages in model._pages.values():
                            pages.data[3:5].copy_(pages.data[1:3])
                            pages.data[1:3].zero_()
                        for pool in model._pair_pools.values():
                            pool[3:5].copy_(pool[1:3])
                            pool[1:3].zero_()
                    inputs = self.inputs(model, width, start, token, reload)
                    impl.prepare_model_inputs(inputs)
                    graph.replay()
                    self.assertEqual(int(impl.rows.token_ids[0]), token)
                    self.assertFalse(bool(output[:, width:].any()))
                    torch.testing.assert_close(
                        output[0, :width],
                        fixture.expected(
                            torch.arange(start, start + width, device="cuda")
                        ),
                        rtol=0,
                        atol=0,
                    )
                    retained = 1 if start == 127 or width == 1 else 3
                    if width == 6:
                        self.assertEqual(impl.get_execution_states(inputs), [])
                        for layer in PAIR_OWNERS:
                            snapshots = V41DecodePairState(
                                impl.context.layers[layer].compressor.pair_snapshots
                            )
                            torch.testing.assert_close(
                                snapshots.positions[0],
                                torch.arange(start, start + 7, device="cuda"),
                                rtol=0,
                                atol=0,
                            )
                        impl.commit_retained_rows(
                            torch.tensor(
                                [retained, 0], dtype=torch.int32, device="cuda"
                            ),
                            draft_committed=True,
                        )
                    states = impl.get_execution_states(inputs)
                    self.assertEqual(len(states), 1)
                    self.assertEqual(states[0].materialized_end, start + retained)
                    if width == 6:
                        self.assertTrue(states[0].draft_committed)
                        self.assertLessEqual(
                            states[0].aux_valid_start, max(0, start + retained - 128)
                        )
                        self.assertEqual(states[0].aux_valid_end, start + retained)
                    else:
                        self.assertEqual(
                            (states[0].aux_valid_start, states[0].aux_valid_end), (0, 0)
                        )
                    self.assertEqual(
                        len(states[0].swa_valid_end), 43 if width == 6 else 40
                    )
                    self.assertTrue(
                        all(end == start + retained for end in states[0].swa_valid_end)
                    )
                    if retained == 3:
                        self.assertEqual(
                            states[0].history_token_ids, [token, token + 1, token + 2]
                        )
                    for layer in PAIR_OWNERS:
                        pool = model._pair_pools[layer]
                        page_id = (3 if reload else 1) + (
                            (start + retained - 1) // model.layout.reuse_unit
                        )
                        offset = (
                            model.layout.speculative_tokens + 1
                        ) * PAIR_SNAPSHOT_BYTES
                        pair = V41DecodePairState(
                            pool[
                                page_id : page_id + 1,
                                offset : offset + PAIR_SNAPSHOT_BYTES,
                            ]
                        )
                        self.assertEqual(int(pair.positions[0]), start + retained)
                        self.assertEqual(int(pair.valid[0]), (start + retained) % 2)


if __name__ == "__main__":
    unittest.main()
