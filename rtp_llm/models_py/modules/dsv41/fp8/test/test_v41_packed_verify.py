"""Packed target geometry, physical cache ownership and live graph metadata."""

import itertools
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.attn_type import DECODER_SWA_KV, SWA_KV
from rtp_llm.models_py.modules.dsv41.fp8.decode.packed_metadata import (
    allocate_packed_decode_metadata,
    packed_pool_slots,
    packed_row_mapping,
    update_packed_decode_metadata,
)


def _fixture(device, lengths=(1, 2, 6)):
    batch, width = len(lengths), sum(lengths) // len(lengths)
    specs = {SWA_KV: (256, 128, 5), DECODER_SWA_KV: (140, 128, 5)}
    meta = allocate_packed_decode_metadata(
        max_batch_size=batch,
        q_len=width,
        window_size=128,
        head_dim=512,
        max_seq_len=640,
        compress_ratios=[0],
        index_topk=512,
        device=torch.device(device),
        paged_pool_specs=specs,
    )
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    prefixes = torch.tensor(
        [127, 255, 7, 510][:batch], dtype=torch.int32, device=device
    )
    inputs = SimpleNamespace(
        input_lengths=torch.full((batch,), -123, dtype=torch.int32),
        prefix_lengths=torch.full((batch,), -456, dtype=torch.int32),
        input_lengths_device=lens,
        prefix_lengths_device=prefixes,
        cu_seqlens_device=torch.cat((lens.new_zeros(1), lens.cumsum(0).int())),
    )
    tables = {
        tag: torch.arange(1, batch * 5 + 1, dtype=torch.int32, device=device).reshape(
            batch, 5
        )
        + index * batch * 5
        for index, tag in enumerate(specs)
    }
    entries = {tag: spec[0] for tag, spec in specs.items()}
    spans = {tag: spec[1] for tag, spec in specs.items()}
    return meta, inputs, tables, entries, spans


class PackedVerifyCPU(unittest.TestCase):
    def test_budget_metadata_banks_keep_rank_and_padding_state_independent(self):
        # Each captured budget owns a different static token capacity. Updating
        # one rank/budget must neither resize it nor overwrite another graph's
        # request/position/cache metadata when async preparation switches back.
        banks = {
            (rank, budget): _fixture("cpu", (budget + 1,) * 4)
            for rank in range(2)
            for budget in range(6)
        }

        def watched(meta):
            return (
                meta.position_ids,
                meta.req_id_per_token,
                meta.input_lengths,
                meta.valid_requests,
                meta.swa_global_slots,
                meta.decoder_swa_metadata.swa_global_slots,
                *meta.pool_block_tables.values(),
                *meta.pool_write_slot_mappings.values(),
            )

        pointers = {}
        snapshots = {}
        for key, (meta, inputs, tables, entries, spans) in banks.items():
            update_packed_decode_metadata(meta, inputs, tables, entries, spans)
            pointers[key] = [value.data_ptr() for value in watched(meta)]
            snapshots[key] = [value.clone() for value in watched(meta)]

        layouts = ([1, 1], [1, 3], [1, 2, 6], [2, 4, 6], [4, 6], [6])
        for step, budgets in enumerate(
            ((0, 5), (4, 1), (1, 3), (5, 0), (2, 4), (3, 2), (0, 5))
        ):
            for rank, budget in enumerate(budgets):
                with self.subTest(step=step, rank=rank, budget=budget):
                    key = (rank, budget)
                    meta, inputs, tables, entries, spans = banks[key]
                    real_lengths = list(layouts[budget])
                    if step % 2:
                        real_lengths.reverse()
                    live = len(real_lengths)
                    lengths = real_lengths + [budget + 1] * (4 - live)
                    prefixes = [7 + 31 * rank + 9 * step + b for b in range(live)]
                    prefixes += [0] * (4 - live)
                    inputs.input_lengths_device.copy_(torch.tensor(lengths))
                    inputs.prefix_lengths_device.copy_(torch.tensor(prefixes))
                    inputs.cu_seqlens_device.copy_(
                        torch.tensor([0, *itertools.accumulate(lengths)])
                    )
                    for index, table in enumerate(tables.values()):
                        table.fill_(1 + index * 100 + rank * 30 + step * 3)
                        table[live:].zero_()

                    update_packed_decode_metadata(
                        meta,
                        inputs,
                        tables,
                        entries,
                        spans,
                        valid_request_count=live,
                    )
                    expected = [
                        (request, prefix + offset)
                        for request, (length, prefix) in enumerate(
                            zip(lengths, prefixes)
                        )
                        for offset in range(length)
                    ]
                    self.assertEqual(
                        list(
                            zip(
                                meta.req_id_per_token.tolist(),
                                meta.position_ids.tolist(),
                            )
                        ),
                        expected,
                    )
                    self.assertEqual(
                        meta.valid_requests.tolist(),
                        [True] * live + [False] * (4 - live),
                    )
                    self.assertEqual(
                        [value.data_ptr() for value in watched(meta)], pointers[key]
                    )
                    self.assertIs(
                        meta.decoder_swa_metadata.input_lengths, meta.input_lengths
                    )
                    real_rows = sum(real_lengths)
                    for tag, writes in meta.pool_write_slot_mappings.items():
                        expected_writes = [
                            (
                                int(tables[tag][request, position // spans[tag]])
                                * entries[tag]
                                + position % entries[tag]
                                if request < live
                                else -1
                            )
                            for request, position in expected
                        ]
                        self.assertEqual(writes.tolist(), expected_writes)
                    self.assertTrue(meta.swa_global_slots[real_rows:].eq(-1).all())
                    self.assertTrue(
                        meta.decoder_swa_metadata.swa_global_slots[real_rows:]
                        .eq(-1)
                        .all()
                    )
                    snapshots[key] = [value.clone() for value in watched(meta)]
                    for other_key, (other_meta, *_rest) in banks.items():
                        for actual, previous in zip(
                            watched(other_meta), snapshots[other_key]
                        ):
                            torch.testing.assert_close(actual, previous, rtol=0, atol=0)

    def test_zero_budget_and_full_width_keep_exact_request_positions(self):
        for lengths in ((1, 1, 1), (6, 6, 6)):
            meta, inputs, tables, entries, spans = _fixture("cpu", lengths)
            update_packed_decode_metadata(meta, inputs, tables, entries, spans)
            self.assertEqual(meta.total_tokens, sum(lengths))
            self.assertEqual(meta.q_len_per_req, lengths[0])
            self.assertEqual(
                meta.position_ids.tolist(),
                [
                    int(inputs.prefix_lengths_device[b]) + j
                    for b, length in enumerate(lengths)
                    for j in range(length)
                ],
            )

    def test_all_integral_budget_layouts_include_zero_one_and_gamma(self):
        for batch in range(1, 5):
            prefixes = [127, 255, 7, 510][:batch]
            for lengths in itertools.product(range(1, 7), repeat=batch):
                if sum(lengths) % batch:
                    continue
                request, positions, cu = packed_row_mapping(
                    torch.tensor(lengths), torch.tensor(prefixes), sum(lengths)
                )
                expected = [
                    (b, prefixes[b] + offset)
                    for b, length in enumerate(lengths)
                    for offset in range(length)
                ]
                self.assertEqual(
                    list(zip(request.tolist(), positions.tolist())), expected
                )
                self.assertEqual(cu[-1].item(), sum(lengths))

    def test_page_and_ring_mapping_uses_real_request_not_execution_group(self):
        request, positions, _ = packed_row_mapping(
            torch.tensor([1, 5]), torch.tensor([127, 255]), 6
        )
        table = torch.tensor([[7, 9, 0, 11], [17, 19, 23, 29]])
        actual = packed_pool_slots(table, request, positions, 140, 128)
        expected = [
            int(table[b, p // 128]) * 140 + p % 140 for b, p in zip(request, positions)
        ]
        self.assertEqual(actual.tolist(), expected)
        uniform = torch.arange(2).repeat_interleave(3)
        self.assertFalse(
            torch.equal(actual, packed_pool_slots(table, uniform, positions, 140, 128))
        )
        table[1, 2] = 0
        self.assertEqual(
            packed_pool_slots(table, request, positions, 140, 128)[-1].item(), -1
        )

    def test_metadata_rebuild_changes_layout_at_fixed_addresses(self):
        meta, inputs, tables, entries, spans = _fixture("cpu")
        watched = (
            meta.position_ids,
            meta.req_id_per_token,
            meta.swa_global_slots,
            meta.input_lengths,
        )
        pointers = [tensor.data_ptr() for tensor in watched]
        for lengths in ([1, 2, 6], [6, 2, 1], [3, 3, 3]):
            inputs.input_lengths_device.copy_(torch.tensor(lengths))
            inputs.cu_seqlens_device.copy_(
                torch.tensor([0, *itertools.accumulate(lengths)])
            )
            update_packed_decode_metadata(
                meta, inputs, tables, entries, spans, valid_request_count=2
            )
            self.assertEqual([tensor.data_ptr() for tensor in watched], pointers)
            expected = [b for b, length in enumerate(lengths) for _ in range(length)]
            self.assertEqual(meta.req_id_per_token.tolist(), expected)
            self.assertEqual(meta.valid_requests.tolist(), [True, True, False])
            self.assertIs(meta.decoder_swa_metadata.input_lengths, meta.input_lengths)
            self.assertIs(
                meta.decoder_swa_metadata.req_id_per_token, meta.req_id_per_token
            )
            self.assertFalse(
                torch.equal(
                    meta.swa_global_slots, meta.decoder_swa_metadata.swa_global_slots
                )
            )
            for row, b in enumerate(expected):
                position = int(meta.position_ids[row])
                valid_positions = range(max(0, position - 127), position + 1)
                for tag, selected in (
                    (SWA_KV, meta),
                    (DECODER_SWA_KV, meta.decoder_swa_metadata),
                ):
                    oracle = [
                        int(tables[tag][b, p // 128]) * entries[tag] + p % entries[tag]
                        for p in valid_positions
                    ]
                    oracle += [-1] * (128 - len(oracle))
                    self.assertEqual(selected.swa_global_slots[row].tolist(), oracle)


class PackedTaggedTableCapacityCPU(unittest.TestCase):
    @staticmethod
    def _fixture(bounded=True):
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplConfigFP8,
        )
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        # Live TP1 service: physical reserve expands to a 34-column native
        # kernel table; the former max_seq/kernel+1 estimate was only 33.
        geometry = {
            SWA_KV: (160, 256),
            "global_kv_2": (64, 128),
            "indexer_kv": (128, 128),
            "csa_state": (4, 256),
            "global_kv_1": (128, 128),
            DECODER_SWA_KV: (160, 256),
        }
        if not bounded:
            geometry.pop(DECODER_SWA_KV)
        config = DSv4DecodeFmhaImplConfigFP8(
            max_batch_size=4,
            q_len=3,
            window_size=128,
            head_dim=512,
            max_seq_len=4096,
            compress_ratios=[0],
            index_topk=512,
            paged_pool_specs={tag: (*shape, 33) for tag, shape in geometry.items()},
            group_tags=list(geometry),
            packed_target_verify=True,
        )
        tagged = {}
        for index, tag in enumerate(geometry):
            attn = PyAttentionInputs()
            attn.input_lengths = torch.tensor([1, 2, 3, 6], dtype=torch.int32)
            attn.prefix_lengths = torch.tensor([127, 255, 510, 4090], dtype=torch.int32)
            attn.cu_seqlens_device = torch.tensor([0, 1, 3, 6, 12], dtype=torch.int32)
            attn.kv_cache_kernel_block_id_device = (
                torch.arange(1, 137, dtype=torch.int32).reshape(4, 34) + index * 1000
            )
            attn.is_target_verify = True
            attn.is_packed_target_verify = True
            tagged[tag] = attn
        return config, tagged

    def _assert_swa_slots(self, meta, tagged):
        children = [(SWA_KV, meta)]
        if DECODER_SWA_KV in tagged:
            children.append((DECODER_SWA_KV, meta.decoder_swa_metadata))
        for tag, selected in children:
            table = tagged[tag].kv_cache_kernel_block_id_device
            requests = meta.req_id_per_token.tolist()
            positions = meta.position_ids.tolist()
            expected = [
                (
                    int(table[r, p // 256]) * 160 + p % 160
                    if table[r, p // 256] > 0
                    else -1
                )
                for r, p in zip(requests, positions)
            ]
            self.assertEqual(meta.pool_write_slot_mappings[tag].tolist(), expected)
            # Last real position is 4095; native reserve columns are retained
            # byte-for-byte but do not enter valid request reads/writes.
            for row, (request, position) in enumerate(zip(requests, positions)):
                visible = range(max(0, position - 127), position + 1)
                expected_reads = [
                    (
                        int(table[request, p // 256]) * 160 + p % 160
                        if table[request, p // 256] > 0
                        else -1
                    )
                    for p in visible
                ]
                expected_reads += [-1] * (128 - len(expected_reads))
                self.assertEqual(
                    selected.swa_global_slots[row].tolist(), expected_reads
                )

    def test_real_tagged_constructor_and_replay_preserve_full_native_table(self):
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )

        config, tagged = self._fixture()
        impl = DSv4DecodeFmhaImplFP8(config, torch.device("cpu"), tagged)
        meta = impl.metadata
        pointers = {
            tag: value.data_ptr() for tag, value in meta.pool_block_tables.items()
        }
        for tag, value in meta.pool_block_tables.items():
            self.assertEqual(tuple(value.shape), (4, 34))
            torch.testing.assert_close(
                value, tagged[tag].kv_cache_kernel_block_id_device
            )
        self._assert_swa_slots(meta, tagged)
        for attn in tagged.values():
            attn.input_lengths.copy_(torch.tensor([6, 3, 2, 1]))
            attn.prefix_lengths.copy_(torch.tensor([510, 1023, 2047, 4090]))
            attn.cu_seqlens_device.copy_(torch.tensor([0, 6, 9, 11, 12]))
            attn.kv_cache_kernel_block_id_device.add_(10000)
            attn.kv_cache_kernel_block_id_device[3].zero_()
        impl.prepare_cuda_graph(tagged, valid_request_count=3)
        self.assertEqual(meta.valid_requests.tolist(), [True, True, True, False])
        for tag, value in meta.pool_block_tables.items():
            self.assertEqual(value.data_ptr(), pointers[tag])
            torch.testing.assert_close(
                value, tagged[tag].kv_cache_kernel_block_id_device
            )
        self._assert_swa_slots(meta, tagged)

    def test_eager_uses_same_native_capacity_and_rejects_missing_unknown_or_growth(
        self,
    ):
        from rtp_llm.models_py.modules.dsv41.decode.forward import build_metadata_eager
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
            as_tagged_attention_inputs,
        )

        config, tagged = self._fixture()
        args = SimpleNamespace(
            max_seq_len=4096,
            v41_config=object(),
            window_size=128,
            head_dim=512,
            compress_ratios=[0],
            n_layers=1,
            index_topk=512,
        )
        meta = build_metadata_eager(
            args,
            as_tagged_attention_inputs(tagged),
            torch.device("cpu"),
            config.paged_pool_specs,
            fp8_kv_cache=True,
            total_tokens=12,
        )
        self._assert_swa_slots(meta, tagged)
        tables = {
            tag: value.kv_cache_kernel_block_id_device for tag, value in tagged.items()
        }
        entries = {tag: spec[0] for tag, spec in config.paged_pool_specs.items()}
        spans = {tag: spec[1] for tag, spec in config.paged_pool_specs.items()}
        attn = as_tagged_attention_inputs(tagged)
        for bad, message in (
            (
                {tag: value for tag, value in tables.items() if tag != SWA_KV},
                "complete block table for swa_kv",
            ),
            (
                {**tables, "unregistered_pool": torch.ones(4, 34)},
                "unknown block table tags",
            ),
            (
                {**tables, SWA_KV: torch.ones(4, 35)},
                "swa_kv: src_shape=\\(4, 35\\), dst_shape=\\(4, 34\\)",
            ),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                update_packed_decode_metadata(meta, attn, bad, entries, spans)
        for tag, value in meta.pool_block_tables.items():
            self.assertEqual(value.shape[1], 34)
            torch.testing.assert_close(value, tables[tag])

    def test_unbounded_layout_without_decoder_swa_uses_native_capacity(self):
        from rtp_llm.models_py.modules.dsv41.decode.forward import build_metadata_eager
        from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
            as_tagged_attention_inputs,
        )

        config, tagged = self._fixture(bounded=False)
        impl = DSv4DecodeFmhaImplFP8(config, torch.device("cpu"), tagged)
        impl.prepare_cuda_graph(tagged, valid_request_count=4)
        self.assertIsNone(impl.metadata.decoder_swa_metadata)
        self._assert_swa_slots(impl.metadata, tagged)
        args = SimpleNamespace(
            max_seq_len=4096,
            v41_config=object(),
            window_size=128,
            head_dim=512,
            compress_ratios=[0],
            n_layers=1,
            index_topk=512,
        )
        eager = build_metadata_eager(
            args,
            as_tagged_attention_inputs(tagged),
            torch.device("cpu"),
            config.paged_pool_specs,
            fp8_kv_cache=True,
            total_tokens=12,
        )
        self.assertIsNone(eager.decoder_swa_metadata)
        self.assertEqual(set(eager.pool_block_tables), set(tagged))
        for tag, value in eager.pool_block_tables.items():
            self.assertEqual(tuple(value.shape), (4, 34))
            torch.testing.assert_close(value, impl.metadata.pool_block_tables[tag])
        self._assert_swa_slots(eager, tagged)

    def test_eager_metadata_tracks_changing_budget_for_same_requests(self):
        from rtp_llm.models_py.modules.dsv41.decode.forward import build_metadata_eager
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
            as_tagged_attention_inputs,
        )

        config, tagged = self._fixture()
        args = SimpleNamespace(
            max_seq_len=4096,
            v41_config=object(),
            window_size=128,
            head_dim=512,
            compress_ratios=[0],
            n_layers=1,
            index_topk=512,
        )
        # Keep B=4 and the physical tables fixed while crossing every average
        # budget, including shrinking after full width and returning to b=2.
        for lengths in (
            [1, 2, 3, 6],
            [1, 1, 1, 1],
            [1, 1, 1, 5],
            [4, 6, 5, 5],
            [6, 6, 6, 6],
            [2, 6, 3, 5],
            [6, 3, 2, 1],
        ):
            with self.subTest(lengths=lengths):
                for attn in tagged.values():
                    attn.input_lengths.copy_(torch.tensor(lengths))
                    attn.cu_seqlens_device.copy_(
                        torch.tensor([0, *itertools.accumulate(lengths)])
                    )
                meta = build_metadata_eager(
                    args,
                    as_tagged_attention_inputs(tagged),
                    torch.device("cpu"),
                    config.paged_pool_specs,
                    fp8_kv_cache=True,
                    total_tokens=sum(lengths),
                )
                self.assertFalse(meta.is_cuda_graph)
                self.assertEqual(meta.q_len_per_req, sum(lengths) // 4)
                self.assertEqual(meta.total_tokens, sum(lengths))
                self.assertEqual(
                    meta.req_id_per_token.tolist(),
                    [r for r, length in enumerate(lengths) for _ in range(length)],
                )
                self._assert_swa_slots(meta, tagged)


class PackedHiddenHandoffCPU(unittest.TestCase):
    @staticmethod
    def _capture(rows, capacity=None):
        from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

        capture = SimpleNamespace(
            layers=[None] * 40,
            _mtp_hidden_buffer=torch.full((capacity or rows, 6), -999.0),
            _mtp_hidden_valid_tokens=111,
        )
        V4Transformer.set_aux_hidden_capture_layer_ids(capture, (37, 38, 39))
        return capture

    def test_capture_hc_mean_layer_columns_and_explicit_graph_rows(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_base_model import DeepSeekV4Model
        from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

        capture = self._capture(9, capacity=12)
        address = capture._mtp_hidden_buffer.data_ptr()
        # Arrival order deliberately differs from the configured column order.
        for rows, shape in ((9, (3, 3, 4, 2)), (6, (6, 4, 2))):
            expected = []
            for layer in (39, 37, 38):
                base = torch.arange(rows * 2).reshape(rows, 1, 2).float()
                hidden = base + torch.tensor([0, 2, 8, 14]).view(1, 4, 1)
                hidden += 100 * layer + rows
                V4Transformer.capture_aux_hidden(capture, layer, hidden.reshape(shape))
            for layer in (37, 38, 39):
                expected.append(
                    torch.arange(rows * 2).reshape(rows, 2) + 100 * layer + rows + 6
                )
            expected = torch.cat(expected, dim=-1).float()
            view = DeepSeekV4Model.get_mtp_target_hidden_states(
                SimpleNamespace(v4=capture), rows
            )
            torch.testing.assert_close(view, expected, rtol=0, atol=0)
            self.assertEqual(view.data_ptr(), address)
            self.assertEqual(capture._mtp_hidden_valid_tokens, 111)
            # Explicit graph row count cannot expose stale capacity/tail rows.
            self.assertEqual(tuple(view.shape), (rows, 6))
        self.assertTrue((capture._mtp_hidden_buffer[9:] == -999).all())

    def test_packed_capture_matches_selected_dense_aux_for_same_layer_inputs(self):
        from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

        dense = self._capture(12)
        packed = self._capture(6)
        # Real requests [1, 5] cross the artificial execution-group boundary.
        rows = torch.tensor([0, 6, 7, 8, 9, 10])
        for layer in (38, 39, 37):
            hidden = torch.arange(12 * 4 * 2).float().reshape(2, 6, 4, 2)
            hidden = hidden * (layer - 35) + layer * 100
            selected = hidden.flatten(0, 1).index_select(0, rows)
            V4Transformer.capture_aux_hidden(dense, layer, hidden)
            V4Transformer.capture_aux_hidden(
                packed, layer, selected.reshape(2, 3, 4, 2)
            )
        torch.testing.assert_close(
            packed._mtp_hidden_buffer,
            dense._mtp_hidden_buffer.index_select(0, rows),
            rtol=0,
            atol=0,
        )
        self.assertFalse(
            torch.equal(packed._mtp_hidden_buffer, dense._mtp_hidden_buffer[:6])
        )

    def test_real_decode_loop_captures_layer_input_before_inplace_update(self):
        from unittest.mock import patch

        from rtp_llm.models_py.modules.dsv41.decode import forward as decode
        from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

        class Layer:
            def __init__(self, layer_id):
                self.layer_id = layer_id
                self.engram = None

            def forward_decode(self, hidden, *args, **kwargs):
                return hidden.add_(self.layer_id + 1)

        capture = self._capture(9)
        capture.layers = [Layer(i) for i in range(5)]
        V4Transformer.set_aux_hidden_capture_layer_ids(capture, (1, 3, 4))
        capture.capture_aux_hidden = (
            lambda layer, hidden: V4Transformer.capture_aux_hidden(
                capture, layer, hidden
            )
        )
        capture._note_aux_hidden_rows = (
            lambda rows, is_cuda_graph: V4Transformer._note_aux_hidden_rows(
                capture, rows, is_cuda_graph
            )
        )
        capture._hc_head_reduce = lambda hidden: hidden.mean(dim=-2)
        capture.norm = lambda hidden: hidden
        original = torch.arange(3 * 3 * 4 * 2).reshape(3, 3, 4, 2).float()
        meta = SimpleNamespace(batch_size=3, q_len_per_req=3, is_cuda_graph=True)
        with (
            patch.object(decode._rt, "ENABLED", False),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(
                decode, "prepare_model_gate_routing_metadata", return_value=None
            ),
            patch.object(
                decode, "forward_decode_layer", wraps=decode.forward_decode_layer
            ) as forward,
        ):
            decode.forward_layers(
                capture,
                None,
                torch.arange(9),
                meta,
                expert_token_counts=None,
                prepare_hidden_fn=lambda **kwargs: original.clone(),
            )
        pooled = original.mean(dim=-2).reshape(9, 2)
        expected = torch.cat([pooled + delta for delta in (1, 6, 10)], dim=-1)
        torch.testing.assert_close(capture._mtp_hidden_buffer, expected, rtol=0, atol=0)
        # Capture the resolved layer entry through the callback without forcing
        # the preceding layer to materialize its delayed MHC output.
        self.assertEqual(
            [c.kwargs["preserve_output"] for c in forward.call_args_list],
            [False, False, False, False, False],
        )
        self.assertEqual(
            [c.kwargs["capture_input"] is not None for c in forward.call_args_list],
            [False, True, False, True, True],
        )
        self.assertEqual(capture._mtp_hidden_valid_tokens, 111)

    def test_capture_to_commit_preserves_rows_after_reorder_shrink_and_padding(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model import (
            DeepSeekV4DSparkModel,
        )
        from rtp_llm.models_py.modules.dsv41.transformer import V4Transformer

        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        model.kv_cache = None
        model._dspark_commit_cp_enabled = False
        model._dspark_aux_feature_dim = model._dspark_hidden_dim = 6
        model.combine_hidden_states = lambda features: features
        seen = []
        model.commit_feature_rows = (
            lambda hidden, req, pos, ends, inputs, **kwargs: seen.append(
                (hidden.clone(), req.tolist(), pos.tolist(), ends.tolist())
            )
        )
        capture = self._capture(18, capacity=19)
        address = capture._mtp_hidden_buffer.data_ptr()
        # True request identity is encoded in features. Host mirrors are poison;
        # device metadata is mutated in place at equal shape on the first two calls.
        lens_buffer = torch.zeros(3, dtype=torch.int32)
        prefix_buffer = torch.zeros(3, dtype=torch.int32)
        rounds = [
            ([41, 7, 9], [1, 2, 6], [127, 255, 7]),
            ([9, 41, 7], [6, 1, 2], [10, 128, 256]),
            ([9, 41, 7], [1, 1, 1], [10, 128, 256]),
            ([9, 41, 7], [1, 1, 4], [10, 128, 256]),
            ([9, 41, 7], [6, 6, 6], [10, 128, 256]),
            ([9, 41, 7], [6, 3, 6], [10, 128, 256]),
            ([9, 41, 7], [5, 1, 6], [10, 128, 256]),
            ([9, 41, 7], [6, 1, 2], [10, 128, 256]),
            ([7, 9], [1, 5], [258, 11]),
            ([7], [1], [259]),
            ([9], [6], [16]),
        ]
        for identities, lengths, prefixes in rounds:
            batch = len(lengths)
            rows = [
                (request, prefixes[request] + offset, stable)
                for request, (stable, length) in enumerate(zip(identities, lengths))
                for offset in range(length)
            ]
            for layer in (39, 37, 38):
                values = torch.tensor(
                    [
                        [
                            stable * 100 + pos * 2 + channel + layer
                            for channel in range(2)
                        ]
                        for _, pos, stable in rows
                    ]
                ).float()
                hidden = values[:, None, :] + torch.tensor([0, 2, 4, 6]).view(1, 4, 1)
                V4Transformer.capture_aux_hidden(capture, layer, hidden)
            lens_buffer[:batch].copy_(torch.tensor(lengths))
            prefix_buffer[:batch].copy_(torch.tensor(prefixes))
            attn = SimpleNamespace(
                input_lengths=torch.full((batch,), 777, dtype=torch.int32),
                prefix_lengths=torch.full((batch,), 888, dtype=torch.int32),
                input_lengths_device=lens_buffer[:batch],
                prefix_lengths_device=prefix_buffer[:batch],
                context_parallel_info=None,
            )
            # One graph-capacity tail row must be masked by the real row windows.
            inputs = SimpleNamespace(
                attention_inputs=attn,
                input_hiddens=capture._mtp_hidden_buffer[: len(rows) + 1],
            )
            model.run_commit_step(inputs, torch.device("cpu"))
            actual, req, positions, ends = seen[-1]
            self.assertEqual(req, [row[0] for row in rows] + [-1])
            self.assertEqual(positions, [row[1] for row in rows] + [-1])
            self.assertEqual(ends, [p + n for p, n in zip(prefixes, lengths)])
            expected = torch.tensor(
                [
                    [
                        stable * 100 + pos * 2 + channel + layer + 3
                        for layer in (37, 38, 39)
                        for channel in range(2)
                    ]
                    for _, pos, stable in rows
                ]
            ).float()
            torch.testing.assert_close(actual[:-1], expected, rtol=0, atol=0)
            self.assertEqual(capture._mtp_hidden_buffer.data_ptr(), address)

    def test_next_proposal_anchor_and_visible_context_follow_accepted_prefix(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_base_dspark_model import (
            DeepSeekV4DSparkModel,
        )

        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        model.kv_cache = None
        model._dspark_width = model._dspark_query_width = model._gen_num_per_cycle = 5
        model._dspark_noise_token_id = 123
        model._v4_args = SimpleNamespace(window_size=128)
        observed = []
        model.forward_query_block = (
            lambda ids, pos, prefixes, active, *args: observed.append(
                (ids.clone(), pos.clone(), prefixes.clone(), active.clone())
            )
        )
        model.compute_draft_outputs = lambda hidden: hidden
        # Previous lengths [1,2,6] and accepted [0,0,5] produce prefix advances
        # [1,1,6]. Reorder and shrink to request C, A; correction/bonus is anchor.
        prefixes = torch.tensor([13, 128], dtype=torch.int32)
        anchors = torch.tensor([[9003, 91, 92, 93, 94], [9001, 81, 82, 83, 84]])
        attn = SimpleNamespace(
            prefix_lengths=torch.tensor([-99, -99]), prefix_lengths_device=prefixes
        )
        model.run_propose_step(
            SimpleNamespace(input_ids=anchors.flatten(), attention_inputs=attn),
            None,
            torch.device("cpu"),
        )
        ids, positions, actual_prefix, active = observed[-1]
        self.assertEqual(
            ids.tolist(), [[9003, 123, 123, 123, 123], [9001, 123, 123, 123, 123]]
        )
        self.assertEqual(
            positions.tolist(), [[13, 14, 15, 16, 17], [128, 129, 130, 131, 132]]
        )
        table = torch.tensor([[9, 19], [41, 51]], dtype=torch.int32)
        slots, lengths = model._build_noncausal_indices(
            actual_prefix, active, table, 256, 256
        )
        for request, prefix in enumerate(prefixes.tolist()):
            visible = range(max(0, prefix - 128), prefix + 5)
            expected = [int(table[request, p // 256]) * 256 + p % 256 for p in visible]
            expected += [-1] * (slots.shape[-1] - len(expected))
            for query in range(5):
                self.assertEqual(slots[request * 5 + query].tolist(), expected)
            self.assertEqual(lengths[request].item(), min(prefix, 128) + 5)


class PackedVerifyCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("SM10x CUDA required")

    @torch.inference_mode()
    def test_graph_reads_changed_request_lengths_and_page_tables(self):
        meta, inputs, tables, entries, spans = _fixture("cuda")
        update_packed_decode_metadata(meta, inputs, tables, entries, spans)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            positions = meta.position_ids.clone()
            requests = meta.req_id_per_token.clone()
            slots = meta.swa_global_slots.clone()
            decoder_slots = meta.decoder_swa_metadata.swa_global_slots.clone()
        for lengths, live in (
            ([1, 2, 6], 3),
            ([6, 2, 1], 2),
            ([3, 3, 3], 1),
            ([2, 6, 1], 3),
        ):
            inputs.input_lengths_device.copy_(torch.tensor(lengths, device="cuda"))
            inputs.cu_seqlens_device.copy_(
                torch.tensor([0, *itertools.accumulate(lengths)], device="cuda")
            )
            inputs.prefix_lengths_device.add_(1)
            for table in tables.values():
                table.copy_(table.roll(1, dims=1))
            update_packed_decode_metadata(
                meta, inputs, tables, entries, spans, valid_request_count=live
            )
            graph.replay()
            torch.testing.assert_close(positions, meta.position_ids, rtol=0, atol=0)
            torch.testing.assert_close(requests, meta.req_id_per_token, rtol=0, atol=0)
            torch.testing.assert_close(slots, meta.swa_global_slots, rtol=0, atol=0)
            torch.testing.assert_close(
                decoder_slots,
                meta.decoder_swa_metadata.swa_global_slots,
                rtol=0,
                atol=0,
            )

    @torch.inference_mode()
    def test_compressor_matches_dense_selected_prefix_cache_bytes(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import CSA_STATE, INDEXER_KV
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_global as producer
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_global import (
            _case,
            _clone_case,
        )

        lengths = [1, 2, 6]
        for ratio in (1, 2):
            dense = _case("cuda", ratio, 3, 6)
            packed = _clone_case(dense)
            chosen = torch.tensor(
                [b * 6 + j for b, length in enumerate(lengths) for j in range(length)],
                device="cuda",
            )
            packed.x = (
                dense.x.flatten(0, 1)
                .index_select(0, chosen)
                .reshape(3, 3, -1)
                .contiguous()
            )
            packed.req_ids = dense.req_ids.index_select(0, chosen)
            packed.positions = dense.positions.index_select(0, chosen)
            initial = {
                tag: pool.clone() for tag, pool in packed.owner._test_pools.items()
            }
            self.assertTrue(
                producer.try_produce_global(
                    dense.owner, dense.x, dense.positions, dense.req_ids, dense.starts
                )
            )
            self.assertTrue(
                producer.try_produce_global(
                    packed.owner,
                    packed.x,
                    packed.positions,
                    packed.req_ids,
                    packed.starts,
                    input_lengths=torch.tensor(lengths, device="cuda"),
                )
            )
            for tag, actual in packed.owner._test_pools.items():
                mask = torch.zeros_like(actual, dtype=torch.bool)
                table = packed.owner._block_tables_by_type[tag]
                for b, length in enumerate(lengths):
                    for position in range(
                        int(packed.starts[b]), int(packed.starts[b]) + length
                    ):
                        if tag == CSA_STATE:
                            if ratio == 2:
                                block = int(table[b, (position // 256) % 2])
                                mask[block * 8 + position % 8] = True
                        elif (position + 1) % ratio == 0:
                            block = int(table[b, position // 128])
                            offset = position % 128 // ratio
                            if tag == INDEXER_KV:
                                flat = mask[block].view(-1)
                                flat[offset * 64 : (offset + 1) * 64] = True
                                flat[
                                    128 * 64 + offset * 4 : 128 * 64 + (offset + 1) * 4
                                ] = True
                            else:
                                mask[block, offset] = True
                torch.testing.assert_close(
                    actual[mask], dense.owner._test_pools[tag][mask], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual[~mask], initial[tag][~mask], rtol=0, atol=0
                )

    @torch.inference_mode()
    def test_indexer_fused_and_uncached_match_per_query_table_oracle(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as indexer
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_indexer import (
            make_case,
        )

        for ratio, logical in ((1, 128), (2, 64)):
            case, _ = make_case(2, 3, 2053, 128, logical, ratio)
            request = torch.tensor([0, 1, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
            positions = case["context_lens"].flatten().long() * ratio - 1
            keys = indexer.DecodeIndexerKeys(
                case["pool"], case["block_table"], 2053, logical
            )
            plan = indexer.get_decode_metadata(
                {},
                keys,
                positions,
                batch_size=2,
                query_len=3,
                compress_ratio=ratio,
                request_ids=request,
            )
            self.assertIsNotNone(plan)
            torch.testing.assert_close(plan.paged.request_ids, request, rtol=0, atol=0)
            torch.testing.assert_close(
                plan.paged.tables,
                case["block_table"].clamp_min(0).int()[request.long()],
                rtol=0,
                atol=0,
            )
            oracle = dict(case)
            oracle["q"] = case["q"].reshape(6, 1, 32, 128)
            oracle["weights"] = case["weights"].reshape(6, 1, 32)
            oracle["block_table"] = case["block_table"].index_select(0, request.long())
            oracle["context_lens"] = case["context_lens"].reshape(6, 1)
            expected = indexer.score_decode_indexer(**oracle, preserve_score_dtype=True)
            for metadata in (None, plan):
                actual = indexer.score_decode_indexer(
                    **case,
                    metadata=metadata,
                    request_ids=request,
                    preserve_score_dtype=True,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_sparse_plan_uses_real_request_pages_and_validity(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_decode as sparse

        table = torch.tensor(
            [[11, 13, 17, 19], [23, 29, 31, 37]], device="cuda", dtype=torch.int32
        )
        request = torch.tensor([0, 1, 1, 1, 1, 1], device="cuda", dtype=torch.int32)
        lengths = torch.tensor(
            [1, 31, 127, 128, 129, 511], device="cuda", dtype=torch.int32
        )
        candidates = torch.full((6, 512), -1, device="cuda", dtype=torch.int32)
        for row, length in enumerate(lengths.tolist()):
            count = (length + 7) // 8
            candidates[row, :count] = torch.arange(
                count, device="cuda", dtype=torch.int32
            ).flip(0)
        self.assertTrue(sparse.is_supported(candidates))
        valid = torch.tensor([True, False], device="cuda")
        plan = sparse.make_plan(
            candidates,
            lengths,
            table,
            query_len=3,
            request_ids=request,
            valid_requests=valid,
        )
        torch.testing.assert_close(
            plan.table, table.index_select(0, request.long()), rtol=0, atol=0
        )
        torch.testing.assert_close(
            plan.compact_lengths,
            torch.tensor([1, 0, 0, 0, 0, 0], device="cuda", dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        for row, length in enumerate(lengths.tolist()):
            torch.testing.assert_close(
                plan.blocks[row][plan.active[row]],
                torch.arange((length + 7) // 8, device="cuda", dtype=torch.int32),
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
