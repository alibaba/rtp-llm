"""A captured forty-layer attention schedule with refreshed compact-page IDs."""

import unittest
from dataclasses import replace

import test_attention
import torch
from rtp_llm.models_py.modules.dsv41.cache_layout import PAIR_OWNERS, layer_sources
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.decode_attention import (
    V41DecodeAttention,
    V41DecodeAttentionContext,
)
from rtp_llm.models_py.modules.dsv41.decode_compressor import (
    PAIR_SNAPSHOT_BYTES,
    V41DecodePairState,
)


class DecodeAttentionGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_attention.AttentionGpuTest.setUpClass()

    @classmethod
    def tearDownClass(cls):
        test_attention.AttentionGpuTest.tearDownClass()

    def setUp(self):
        self.fixture = test_attention.AttentionGpuTest()
        self.fixture.setUp()

    @staticmethod
    def add_reload_pages(pages):
        return CompactPages(
            torch.cat((pages.data, torch.zeros_like(pages.data[1:])), dim=0),
            pages.region,
            pages.entries_per_page,
        )

    @staticmethod
    def move_to_new_pages(pages, page_ids):
        shift = (pages.data.shape[0] - 1) // 2
        pages.data[shift + 1 :].copy_(pages.data[1 : shift + 1])
        pages.data[1 : shift + 1].zero_()
        page_ids.add_(shift)

    def test_producer_consumer_graph_reloads_and_rolls_back_without_recapture(self):
        fixture = self.fixture
        cache = fixture.cache(384)
        models = [fixture.model(layer) for layer in range(40)]
        prefill = cache.begin_forward(epoch=0, start=0, end=129)
        with torch.inference_mode():
            for model in models:
                model(fixture.hidden(129), prefill)
            for layer in range(40):
                binding = cache.swa[layer]
                cache.swa[layer] = replace(
                    binding, pages=self.add_reload_pages(binding.pages)
                )
            for owner in cache.owners.values():
                owner.global_kv = replace(
                    owner.global_kv,
                    pages=self.add_reload_pages(owner.global_kv.pages),
                )
                owner.index_pages = self.add_reload_pages(owner.index_pages)
            swa = {layer: cache.swa[layer] for layer in range(40)}
            context = V41DecodeAttentionContext(
                cache.layout,
                swa,
                cache.owners,
                batch_size=1,
                query_width=6,
                max_tokens=384,
            )
            wrapped = [V41DecodeAttention(model, context) for model in models]
            pairs = {}
            for layer in PAIR_OWNERS:
                pair = cache.owners[layer].pair
                raw = torch.zeros(
                    (1, PAIR_SNAPSHOT_BYTES), dtype=torch.uint8, device="cuda"
                )
                state = V41DecodePairState(raw)
                state.partial_kv[0].copy_(pair.partial_kv)
                state.partial_score[0].copy_(pair.partial_score)
                state.positions.fill_(129)
                state.valid.fill_(1)
                pairs[layer] = state
            starts = torch.tensor([129], dtype=torch.int64, device="cuda")
            counts = torch.tensor([6], dtype=torch.int32, device="cuda")
            hidden = fixture.hidden(6)
            context.prepare(
                starts,
                counts,
                swa=swa,
                owners=cache.owners,
                pair_states=pairs,
            )

            def run():
                context.begin_forward()
                return torch.stack([model(hidden, context) for model in wrapped])

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = run()
            addresses = [context.swa[layer].page_ids.data_ptr() for layer in range(40)]
            addresses += [
                owner.global_kv.page_table.data_ptr()
                for owner in context.owners.values()
            ]
            for position, count in ((129, 6), (130, 6), (131, 6), (132, 0)):
                starts.fill_(position)
                counts.fill_(count)
                for binding in swa.values():
                    binding.valid_ends.fill_(position)
                context.prepare(
                    starts,
                    counts,
                    swa=swa,
                    owners=cache.owners,
                    pair_states=pairs,
                )
                graph.replay()
                context.check()
                for layer in range(40):
                    expected = (
                        fixture.expected(
                            torch.arange(position, position + 6, device="cuda"),
                            ratio=layer_sources(layer).ratio,
                        )
                        if count
                        else torch.zeros_like(output[layer])
                    )
                    torch.testing.assert_close(output[layer], expected, rtol=0, atol=0)
                retained = torch.tensor(
                    [1 if count else 0], dtype=torch.int32, device="cuda"
                )
                for layer in PAIR_OWNERS:
                    pairs[layer] = wrapped[layer].compressor.select_pair(retained)
                    self.assertEqual(
                        int(pairs[layer].positions[0]), position + int(retained[0])
                    )
                for layer, binding in swa.items():
                    binding.valid_starts.copy_(context.swa[layer].valid_starts)
                if position == 129:
                    for binding in swa.values():
                        self.move_to_new_pages(binding.pages, binding.page_ids)
                    for owner in cache.owners.values():
                        self.move_to_new_pages(
                            owner.global_kv.pages, owner.global_kv.page_table
                        )
                        self.move_to_new_pages(owner.index_pages, owner.index_table)
            current = [context.swa[layer].page_ids.data_ptr() for layer in range(40)]
            current += [
                owner.global_kv.page_table.data_ptr()
                for owner in context.owners.values()
            ]
            self.assertEqual(addresses, current)


if __name__ == "__main__":
    unittest.main()
