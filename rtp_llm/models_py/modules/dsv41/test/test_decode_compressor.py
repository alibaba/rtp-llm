"""Fixed-width decode compression against the existing eager owner path."""

import unittest

import torch
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    ENCODINGS,
    CacheIdentity,
    CacheLayout,
    CacheRegion,
)
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact
from rtp_llm.models_py.modules.dsv41.compressor import (
    OwnerCompressor,
    PairCarry,
    prepare_owner_kv,
)
from rtp_llm.models_py.modules.dsv41.decode_compressor import (
    PAIR_SNAPSHOT_BYTES,
    V41DecodeOwnerCompressor,
    V41DecodePairState,
)


class DecodeCompressorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("decode compressor requires an actual Blackwell GPU")
        cls.prior_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.prior_tf32

    def model(self, layer, query_width=6):
        layout = CacheLayout()
        dtype = torch.bfloat16 if layer == 20 else torch.float32
        weight = torch.zeros((512, 5120), dtype=dtype, device="cuda")
        gate = torch.zeros_like(weight, dtype=torch.float32)
        index = torch.arange(512, device="cuda")
        # Exact powers-of-two projections isolate state/shape bugs from GEMM
        # reduction-order differences between eager variable rows and a bucket.
        weight[index, index] = 0.5
        weight[index, index + 512] = -0.25
        gate[index, index + 1024] = 0.25
        owner = OwnerCompressor(
            layer,
            weight,
            torch.ones(512, dtype=torch.bfloat16, device="cuda"),
            None if layer == 20 else gate,
            layout=layout,
        )
        index_weight = torch.zeros((128, 512), dtype=torch.bfloat16, device="cuda")
        index_weight[torch.arange(128), torch.arange(128)] = 0.5
        model = V41DecodeOwnerCompressor.from_owner(
            owner,
            index_weight,
            torch.ones(128, dtype=torch.bfloat16, device="cuda"),
            batch_size=3,
            query_width=query_width,
        )
        return model, CacheIdentity("fixed-revision", layout.fingerprint, "full")

    def inputs(self, model, starts, counts):
        torch.manual_seed(20260913)
        start = torch.tensor(starts, dtype=torch.int64, device="cuda")
        count = torch.tensor(counts, dtype=torch.int32, device="cuda")
        hidden = torch.randn(
            (3, model.query_width, 5120), dtype=torch.bfloat16, device="cuda"
        )
        valid = torch.arange(model.query_width, device="cuda")[None, :] < count[:, None]
        hidden.masked_fill_(~valid[:, :, None], float("nan"))
        state = None
        if model.owner.ratio == 2:
            state = V41DecodePairState(
                torch.zeros((3, PAIR_SNAPSHOT_BYTES), dtype=torch.uint8, device="cuda")
            )
            state.positions.copy_(start)
            state.valid.copy_(start % 2)
            state.partial_kv.copy_(
                torch.randn((3, 512), device="cuda") * (start % 2)[:, None]
            )
            state.partial_score.copy_(
                torch.randn((3, 512), device="cuda") * (start % 2)[:, None]
            )
        return hidden, start, count, state

    def eager(self, model, identity, hidden, starts, counts, state, batch):
        start, count = int(starts[batch]), int(counts[batch])
        pair = None
        if model.owner.ratio == 2:
            pair = PairCarry(
                model.owner.owner_layer,
                str(batch),
                identity,
                start,
                state.partial_kv[batch] if start % 2 else None,
                state.partial_score[batch] if start % 2 else None,
            )
        return model.owner(
            hidden[batch, :count].contiguous(),
            start_pos=start,
            request_id=str(batch),
            identity=identity,
            pair=pair,
        )

    def exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def check_outputs(self, model, identity, hidden, starts, counts, state):
        results = []
        for batch in range(3):
            result = self.eager(model, identity, hidden, starts, counts, state, batch)
            results.append(result)
            emitted = model.emitted[batch]
            self.exact(model.unrotated[batch, emitted], result.unrotated)
            self.exact(model.group_positions[batch, emitted], result.group_positions)
            self.exact(
                model.visible_lengths[batch, : int(counts[batch])],
                result.visible_lengths,
            )
            self.assertFalse(bool(model.emitted[batch, int(counts[batch]) :].any()))
            values = prepare_owner_kv(result, model.index_weight, model.index_norm)
            self.exact(model.global_values[batch, emitted], values.global_values)
            self.exact(model.index_values[batch, emitted], values.index_values)
        return results

    def check_pair(self, actual, expected, batch):
        self.assertEqual(int(actual.positions[batch]), expected.next_position)
        self.assertEqual(int(actual.valid[batch]), expected.next_position % 2)
        if expected.next_position % 2:
            self.exact(actual.partial_kv[batch], expected.partial_kv)
            self.exact(actual.partial_score[batch], expected.partial_score)
        else:
            self.assertEqual(int(torch.count_nonzero(actual.partial_kv[batch])), 0)
            self.assertEqual(int(torch.count_nonzero(actual.partial_score[batch])), 0)

    def test_all_owners_and_each_accepted_snapshot(self):
        for layer in (2, 8, 14, 20):
            with self.subTest(owner=layer):
                model, identity = self.model(layer)
                args = self.inputs(model, (0, 127, 128), (6, 5, 0))
                hidden, starts, counts, state = args
                model.prepare(starts, counts, pair_state=state)
                model(hidden)
                self.check_outputs(model, identity, *args)
                if layer == 20:
                    continue
                original = state.storage.clone()
                for retained in range(7):
                    kept = counts.clamp_max(retained)
                    selected = model.select_pair(kept)
                    for batch in range(3):
                        expected = self.eager(
                            model, identity, hidden, starts, kept, state, batch
                        )
                        self.check_pair(selected, expected.next_pair, batch)
                self.exact(state.storage, original)

    def pages(self, model, region):
        entries = model.owner.layout.token_block_size // model.owner.ratio
        width = (entries * ENCODINGS[region].entry_bytes + 511) // 512 * 512
        return CompactPages(
            torch.full((4, width), 91, dtype=torch.uint8, device="cuda"),
            region,
            entries,
        )

    def test_graph_replay_updates_parity_pages_padding_and_accepted_pair(self):
        model, identity = self.model(14)
        initial = self.inputs(model, (0, 127, 128), (6, 5, 0))
        hidden, starts, counts, state = initial
        model.prepare(starts, counts, pair_state=state)
        global_pages = self.pages(model, CacheRegion.GLOBAL)
        index_pages = self.pages(model, CacheRegion.INDEX_K)
        slots = torch.full((3, 6), -1, dtype=torch.int64, device="cuda")
        kept = counts.clamp_max(1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                model(hidden)
                model.store(global_pages, index_pages, slots, slots)
                model.select_pair(kept)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            model(hidden)
            writes = model.store(global_pages, index_pages, slots, slots)
            selected = model.select_pair(kept)
        addresses = (
            model.global_values.data_ptr(),
            model.pair_snapshots.data_ptr(),
            model.selected_pair.data_ptr(),
        )
        for positions, valid, retained in (
            ((0, 127, 128), (6, 5, 0), (1, 1, 0)),
            ((129, 128, 1), (4, 0, 6), (2, 0, 6)),
        ):
            args = self.inputs(model, positions, valid)
            hidden.copy_(args[0])
            starts, counts, state = args[1:]
            kept.copy_(torch.tensor(retained, dtype=torch.int32, device="cuda"))
            model.prepare(starts, counts, pair_state=state)
            absolute = starts[:, None] + torch.arange(6, device="cuda")[None, :]
            slots.copy_(
                torch.tensor((3, 1, 2), device="cuda")[:, None]
                * global_pages.entries_per_page
                + absolute // 2 % global_pages.entries_per_page
            )
            global_pages.data.fill_(91)
            index_pages.data.fill_(91)
            graph.replay()
            for write in writes:
                write.check()
            self.check_outputs(model, identity, hidden, starts, counts, state)
            for batch in range(3):
                expected = self.eager(
                    model, identity, hidden, starts, kept, state, batch
                )
                self.check_pair(selected, expected.next_pair, batch)
            for pages, values in (
                (global_pages, model.global_values),
                (index_pages, model.index_values),
            ):
                encoded = encode_compact(values.flatten(0, 1), pages.region)
                encoded.check()
                expected_bytes = torch.full_like(pages.data, 91)
                for row in range(18):
                    if bool(model.emitted.flatten()[row]):
                        page, offset = divmod(
                            int(slots.flatten()[row]), pages.entries_per_page
                        )
                        begin = offset * encoded.output.shape[1]
                        expected_bytes[
                            page, begin : begin + encoded.output.shape[1]
                        ] = encoded.output[row]
                self.exact(pages.data, expected_bytes)
            self.assertEqual(
                addresses,
                (
                    model.global_values.data_ptr(),
                    model.pair_snapshots.data_ptr(),
                    model.selected_pair.data_ptr(),
                ),
            )


if __name__ == "__main__":
    unittest.main()
