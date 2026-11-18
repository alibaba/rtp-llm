"""Exercise native draft graph layouts independently of hidden-state width."""

import copy
import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests import cuda_graph_tagged_cache_test as fixtures
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
    draft_graph_rows,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelOutputs, rtp_llm_ops


class PackedDraftCapacityModel:
    def __init__(self, hc_mult=4):
        self.shapes = set()
        self.hc_mult = hc_mult

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return None

    def forward(self, inputs, fmha_impl=None):
        rows = inputs.input_ids.numel()
        metadata = get_primary_attention_inputs(inputs)
        assert metadata.is_draft_extend
        batch = metadata.input_lengths_device.numel()
        self.shapes.add((rows, batch))
        hidden = inputs.input_hiddens.reshape(rows, self.hc_mult, 4).sum(1)
        return PyModelOutputs(hidden + inputs.input_ids.to(hidden.dtype)[:, None])


class DraftCapacityModel(PackedDraftCapacityModel):
    def __init__(self, num_tokens_per_request=4, hc_mult=4):
        super().__init__(hc_mult)
        self.num_tokens_per_request = num_tokens_per_request

    def cuda_graph_draft_prefill_uses_request_lanes(self):
        return True

    def forward(self, inputs, fmha_impl=None):
        rows = inputs.input_ids.numel()
        metadata = get_primary_attention_inputs(inputs)
        assert (
            rows == metadata.input_lengths_device.numel() * self.num_tokens_per_request
        )
        outputs = super().forward(inputs, fmha_impl)
        _, _, _, positions, _ = draft_graph_rows(metadata, rows)
        return PyModelOutputs(
            outputs.hidden_states + positions.to(outputs.hidden_states.dtype)[:, None]
        )


class MtpGraphCapacityTest(unittest.TestCase):
    def test_native_mtp_capture_initializes_device_metadata_token_capacity(self):
        from unittest.mock import Mock

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_sparse_mla import (
            SparseMlaImpl,
            SparseMlaOp,
        )

        owner = self

        class MetadataModel:
            def cuda_graph_draft_prefill_uses_request_lanes(self):
                return True

            def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
                metadata = get_primary_attention_inputs(inputs)
                # Cover the native eager datatype probe and every graph bucket,
                # including CaptureMemoryHold's single-tag anchor copy.
                owner.assertGreater(metadata.total_tokens, 0)
                owner.assertEqual(metadata.total_tokens, inputs.input_ids.numel())
                groups = inputs.attention_inputs
                if isinstance(groups, dict):
                    for tagged in groups.values():
                        owner.assertEqual(tagged.total_tokens, metadata.total_tokens)
                backend = object.__new__(SparseMlaImpl)
                backend.seq_size_per_block = 128
                backend._device_graph_metadata = True
                backend._cuda_dag_indexer_metadata = None
                backend.fmha_params = rtp_llm_ops.SparseMlaParams()
                backend.fmha_impl = object.__new__(SparseMlaOp)
                backend.fmha_impl.plan = Mock()
                backend._refresh_paged_mqa_schedule_metadata = Mock()
                backend.prepare(metadata)
                owner.assertEqual(
                    backend.fmha_params.multi_token_decode_total_tokens,
                    metadata.total_tokens,
                )
                owner.assertFalse(backend.cuda_graph_requires_host_metadata())
                return backend

            def forward(self, inputs, fmha_impl=None):
                positions = fmha_impl.fmha_params.positions_d
                return PyModelOutputs(
                    inputs.input_hiddens + positions.to(torch.bfloat16)[:, None]
                )

        for draft, tags in (
            (False, ["full"]),
            (False, ["full", "aux"]),
            (True, ["full"]),
        ):
            with self.subTest(draft=draft, tags=tags):
                runner = CudaGraphRunner()
                query = 4 if draft else 5
                if draft:
                    runner.init_mtp_draft_prefill(MetadataModel(), 4, 4, query, 1)
                else:
                    runner.init_decode(
                        MetadataModel(), 4, 81920, 640, [1, 2, 4], tags, True, query
                    )
                for batch in (4, 1, 3, 2, 4):
                    inputs = fixtures._build_target_verify_inputs(
                        tags, {tag: 1 for tag in tags}, batch, query, 11
                    )
                    groups = inputs.attention_inputs
                    for metadata in groups.values():
                        metadata.is_target_verify = not draft
                        metadata.is_draft_extend = draft
                    if len(tags) == 1:
                        inputs.attention_inputs = groups[tags[0]]
                    inputs.input_hiddens = torch.ones(
                        (batch * query, 4), dtype=torch.bfloat16, device="cuda"
                    )
                    positions = torch.arange(
                        11, 11 + query, device="cuda", dtype=torch.int32
                    ).repeat(batch)
                    if draft:
                        inputs.combo_position_ids = positions
                    self.assertTrue(runner.canRun(inputs))
                    actual = runner.forward(inputs).hidden_states.clone()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        actual,
                        inputs.input_hiddens + positions[:, None],
                        rtol=0,
                        atol=0,
                    )

    def test_empty_multi_token_batch_keeps_native_host_fallback(self):
        from unittest.mock import Mock

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_sparse_mla import (
            SparseMlaImpl,
            SparseMlaOp,
        )

        metadata = PyAttentionInputs()
        metadata.is_prefill = True
        metadata.is_target_verify = True
        metadata.total_tokens = 0
        metadata.input_lengths = torch.empty(0, dtype=torch.int32)
        metadata.prefix_lengths = torch.empty(0, dtype=torch.int32)
        metadata.sequence_lengths = torch.empty(0, dtype=torch.int32)
        metadata.kv_cache_kernel_block_id = torch.empty((0, 1), dtype=torch.int32)
        metadata.kv_cache_kernel_block_id_device = (
            metadata.kv_cache_kernel_block_id.cuda()
        )
        model = object.__new__(SparseMlaImpl)
        model.seq_size_per_block = 128
        model.fmha_params = rtp_llm_ops.SparseMlaParams()
        model.fmha_impl = object.__new__(SparseMlaOp)
        model.fmha_impl.plan = Mock()
        model._refresh_paged_mqa_schedule_metadata = Mock()
        model.prepare(metadata)
        self.assertEqual(model.fmha_params.expanded_seq_lens.numel(), 0)
        self.assertEqual(model.fmha_params.multi_token_decode_total_tokens, 0)
        model.fmha_impl.plan.assert_called_once()

    def test_sparse_decode_device_plan_refreshes_cache_write_slots(self):
        params = rtp_llm_ops.SparseMlaParams()
        metadata = PyAttentionInputs()
        metadata.is_prefill = False
        metadata.input_lengths = torch.ones(3, dtype=torch.int32)
        metadata.prefix_lengths = torch.empty(0, dtype=torch.int32)
        metadata.sequence_lengths = torch.tensor([2, 33, 64], dtype=torch.int32)
        metadata.kv_cache_kernel_block_id = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32
        )
        params.fill_params(metadata, 32)
        table = metadata.kv_cache_kernel_block_id.cuda()
        for positions in ([63, 31, 32], [0, 65, 14]):
            lengths = torch.tensor(positions, dtype=torch.int32, device="cuda") + 1
            params.fill_sparse_mla_decode_cuda_graph_params(lengths, table, 32)
            expected = [
                int(metadata.kv_cache_kernel_block_id[i, p // 32]) * 32 + p % 32
                for i, p in enumerate(positions)
            ]
            self.assertEqual(params.positions_d.cpu().tolist(), positions)
            self.assertEqual(params.slot_mapping.cpu().tolist(), expected)

    def test_context_kv_extent_preserves_live_prefixes_and_explicit_override(self):
        metadata = PyAttentionInputs()
        metadata.context_total_kv_length = 11
        self.assertEqual(metadata.context_total_kv_length, 11)
        cumulative = torch.tensor(
            [0, 81925, 163854, 163859], device="cuda", dtype=torch.int32
        )
        metadata.context_total_kv_length_device = cumulative[2:3]
        alias = copy.copy(metadata)
        # A graph may append dummy requests; the scalar still describes live KV.
        self.assertEqual(alias.context_total_kv_length, 163854)
        cumulative[2].add_(7)
        self.assertEqual(alias.context_total_kv_length, 163861)
        metadata.context_total_kv_length = 23
        self.assertEqual(metadata.context_total_kv_length, 23)
        self.assertEqual(alias.context_total_kv_length, 163861)

    def test_multi_token_device_metadata_matches_ragged_reference(self):
        for span in (32, 128):
            params = rtp_llm_ops.SparseMlaParams()
            # Includes >one warp of requests, a zero-token row, 80K prefixes,
            # acceptance shrink/grow and reuse of larger allocated buffers.
            for batch, lengths, prefixes in (
                (48, [5] * 48, [81920 + i for i in range(48)]),
                (3, [3, 0, 1], [81921, 0, 7]),
                (1, [2], [0]),
                (33, [1 + i % 5 for i in range(33)], [37 * i for i in range(33)]),
            ):
                with self.subTest(span=span, batch=batch):
                    width = (
                        max(p + n for p, n in zip(prefixes, lengths)) + span - 1
                    ) // span
                    width = max(width, 1)
                    table = torch.arange(
                        1, batch * width + 1, dtype=torch.int32
                    ).reshape(batch, width)
                    q = torch.tensor(lengths, dtype=torch.int32, device="cuda")
                    prefix = torch.tensor(prefixes, dtype=torch.int32, device="cuda")
                    device_table = table.cuda()
                    params.fill_multi_token_decode_params(
                        q, prefix, device_table, span, sum(lengths)
                    )
                    pointers = {
                        name: getattr(params, name).data_ptr()
                        for name in (
                            "positions_d",
                            "slot_mapping",
                            "expanded_seq_lens",
                            "ks",
                            "ke",
                        )
                    }
                    # Replay uses fresh device values without reallocating the
                    # metadata captured by attention and cache-write kernels.
                    params.fill_multi_token_decode_cuda_graph_params(
                        q, prefix, device_table, span
                    )
                    self.assertEqual(
                        pointers,
                        {name: getattr(params, name).data_ptr() for name in pointers},
                    )
                    position, batch_index, slots, expanded, ks, ke, pages = (
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                    )
                    qo, kv_ptr, page_ptr, kv_lengths, last_page = [0], [0], [0], [], []
                    for row, (length, base) in enumerate(zip(lengths, prefixes)):
                        kv = length + base
                        count = (kv + span - 1) // span
                        pages.extend(table[row, :count].tolist())
                        page_ptr.append(len(pages))
                        kv_lengths.append(kv)
                        last_page.append((kv - 1) % span + 1 if kv else 0)
                        for offset in range(length):
                            pos = base + offset
                            position.append(pos)
                            batch_index.append(row)
                            slots.append(
                                int(table[row, pos // span]) * span + pos % span
                            )
                            expanded.append(pos + 1)
                            ks.append(kv_ptr[-1])
                            ke.append(kv_ptr[-1] + pos + 1)
                        qo.append(len(position))
                        kv_ptr.append(kv_ptr[-1] + kv)
                    expected = {
                        "positions_d": position,
                        "batch_indice_d": batch_index,
                        "slot_mapping": slots,
                        "expanded_seq_lens": expanded,
                        "topk_indices_offset": [0] * sum(lengths),
                        "ks": ks,
                        "ke": ke,
                        "qo_indptr_d": qo,
                        "kvlen_d": kv_lengths,
                        "prefill_ragged_kv_len_indptr_d": kv_ptr,
                        "decode_page_indptr_d": page_ptr,
                        "paged_kv_last_page_len_d": last_page,
                    }
                    for name, values in expected.items():
                        self.assertEqual(
                            getattr(params, name).cpu().tolist(), values, name
                        )
                    self.assertEqual(
                        params.page_indice_d[: len(pages)].cpu().tolist(), pages
                    )

    def test_multi_token_replay_clears_unused_packed_token_capacity(self):
        params = rtp_llm_ops.SparseMlaParams()
        q = torch.full((3,), 5, dtype=torch.int32, device="cuda")
        prefix = torch.full((3,), 81920, dtype=torch.int32, device="cuda")
        table = torch.arange(1, 3 * 645 + 1, dtype=torch.int32, device="cuda").reshape(
            3, 645
        )
        params.fill_multi_token_decode_params(q, prefix, table, 128, 15)
        pointer = params.slot_mapping.data_ptr()
        with self.assertRaisesRegex(RuntimeError, "exceeds captured capacity"):
            params.fill_multi_token_decode_cuda_graph_params(
                torch.ones(4, dtype=torch.int32, device="cuda"),
                torch.zeros(4, dtype=torch.int32, device="cuda"),
                torch.ones((4, 645), dtype=torch.int32, device="cuda"),
                128,
            )
        for lengths, prefixes in (
            ([1, 3, 0], [81920, 7, 0]),
            ([0, 5, 5], [0, 33, 64]),
            ([5, 5, 5], [3, 8, 13]),
        ):
            q.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
            prefix.copy_(torch.tensor(prefixes, dtype=torch.int32, device="cuda"))
            params.fill_multi_token_decode_cuda_graph_params(q, prefix, table, 128)
            positions, slots, expanded = [], [], []
            for row, (length, base) in enumerate(zip(lengths, prefixes)):
                for offset in range(length):
                    position = base + offset
                    positions.append(position)
                    slots.append(
                        int(table[row, position // 128]) * 128 + position % 128
                    )
                    expanded.append(position + 1)
            tail = 15 - sum(lengths)
            self.assertEqual(params.slot_mapping.data_ptr(), pointer)
            self.assertEqual(params.positions_d.cpu().tolist(), positions + [0] * tail)
            self.assertEqual(params.slot_mapping.cpu().tolist(), slots + [-1] * tail)
            self.assertEqual(
                params.expanded_seq_lens.cpu().tolist(), expanded + [0] * tail
            )
            self.assertEqual(
                params.qo_indptr_d.cpu().tolist(),
                [0, lengths[0], sum(lengths[:2]), sum(lengths)],
            )

    def test_device_graph_replay_keeps_metadata_in_device_coordinates(self):
        class DeviceOnlyPlan:
            def cuda_graph_requires_host_metadata(self):
                return False

            def prepare_cuda_graph(self, groups):
                # Planning and all captured model consumers use device tables.
                for metadata in groups.values():
                    assert metadata.kv_cache_kernel_block_id_device.is_cuda

        class DeviceModel(fixtures.TaggedBlockRowModel):
            def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
                return DeviceOnlyPlan()

        model = DeviceModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model, fixtures.HIDDEN_SIZE, 81920, 32, [4], fixtures.GROUP_TAGS
        )
        for batch in (4, 1, 3, 2, 4):
            inputs = fixtures._build_decode_inputs(
                fixtures.GROUP_TAGS, {"full": 0, "aux": 0}, batch_size=batch
            )
            groups = inputs.attention_inputs
            expected = 0
            for index, tag in enumerate(fixtures.GROUP_TAGS):
                table = torch.full(
                    (batch, 32), batch + index + 1, device="cuda", dtype=torch.int32
                )
                # The device plan does not consume legacy host geometry.
                groups[tag].kv_cache_kernel_block_id = torch.full_like(
                    table, -777, device="cpu"
                ).pin_memory()
                groups[tag].kv_cache_kernel_block_id_device = table
                expected += (1 if index == 0 else 16) * 32 * (batch + index + 1)
            inputs.attention_inputs = groups
            self.assertTrue(runner.canRun(inputs))
            output = runner.forward(inputs).hidden_states.clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output, torch.full_like(output, expected), rtol=0, atol=0
            )

    def test_seed_rows_keep_packed_coordinates_after_batch_shrink_and_grow(self):
        class SeedRowsModel(DraftCapacityModel):
            def forward(self, inputs, fmha_impl=None):
                outputs = super().forward(inputs, fmha_impl)
                metadata = get_primary_attention_inputs(inputs)
                rows = outputs.hidden_states.shape[0]
                _, _, _, positions, _ = draft_graph_rows(metadata, rows)
                selected_positions = positions.index_select(
                    0, metadata.mtp_indexer_seed_rows.long()
                )
                request = (
                    (
                        torch.arange(rows, device="cuda")[:, None]
                        >= metadata.cu_seqlens_device[1:][None, :]
                    )
                    .sum(1)
                    .clamp(max=selected_positions.numel() - 1)
                )
                per_token_seed = selected_positions.index_select(0, request)
                return PyModelOutputs(
                    outputs.hidden_states
                    + per_token_seed.to(outputs.hidden_states.dtype)[:, None]
                )

        torch.cuda.set_device(0)
        lanes = 3
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(SeedRowsModel(lanes, 1), 4, 4, lanes, 1)
        for lengths in ([3, 1, 2], [1], [1, 3], [2, 2, 1], [3]):
            with self.subTest(lengths=lengths):
                inputs = fixtures._build_prefill_inputs(["full"], {"full": 1}, lengths)
                inputs.attention_inputs = get_primary_attention_inputs(inputs)
                rows = sum(lengths)
                inputs.input_hiddens = torch.ones(
                    (rows, 4), dtype=torch.bfloat16, device="cuda"
                )
                metadata = get_primary_attention_inputs(inputs)
                length_tensor = torch.tensor(lengths, dtype=torch.int32, device="cuda")
                metadata.mtp_indexer_seed_rows = (
                    length_tensor.cumsum(0).to(torch.int32) - 1
                )
                prefixes = [7 + 11 * index for index in range(len(lengths))]
                metadata.prefix_lengths.copy_(torch.tensor(prefixes, dtype=torch.int32))
                metadata.context_total_kv_length = rows + sum(prefixes)
                kv_boundaries = [0]
                for length, prefix in zip(lengths, prefixes):
                    kv_boundaries.append(kv_boundaries[-1] + prefix + length)
                metadata.cu_kv_seqlens_device.copy_(
                    torch.tensor(kv_boundaries, dtype=torch.int32, device="cuda")
                )
                # Accepted rows must index packed positions, even with fixed
                # capacity and padding between the graph's logical requests.
                positions = torch.cat(
                    [
                        torch.arange(
                            prefix, prefix + length, dtype=torch.int32, device="cuda"
                        )
                        for length, prefix in zip(lengths, prefixes)
                    ]
                )
                inputs.combo_position_ids = positions
                accepted_positions = torch.tensor(
                    [prefix + length - 1 for length, prefix in zip(lengths, prefixes)],
                    dtype=torch.int32,
                    device="cuda",
                )
                per_token_seed = accepted_positions.repeat_interleave(
                    length_tensor.long()
                )
                expected = (
                    inputs.input_hiddens
                    + (inputs.input_ids + positions + per_token_seed).to(
                        torch.bfloat16
                    )[:, None]
                )
                self.assertTrue(runner.canRun(inputs))
                actual = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_request_capacity_shrink_and_grow_with_variable_acceptance(self):
        # HC width and speculative request lanes are independent dimensions.
        for lanes, hc_mult in ((1, 4), (2, 4), (3, 1), (4, 4)):
            with self.subTest(lanes=lanes, hc_mult=hc_mult):
                self._check_request_capacity(lanes, hc_mult)

    def _check_request_capacity(self, lanes, hc_mult):
        torch.cuda.set_device(0)
        model = DraftCapacityModel(lanes, hc_mult)
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4, lanes, hc_mult)
        self.assertEqual(
            model.shapes, {(lanes * batch, batch) for batch in range(1, 9)}
        )

        # The same graph keys are revisited after both larger and smaller
        # batches. Even one-token requests retain the configured request lanes.
        for lengths in (
            [lanes] * 8,
            [1],
            [lanes],
            [1] * 8,
            [lanes, 1, max(1, lanes - 1)],
            [1, 1],
            [min(2, lanes)],
            [max(1, lanes - 1)] * 7,
            [1],
        ):
            with self.subTest(lengths=lengths):
                rows = sum(lengths)
                inputs = fixtures._build_prefill_inputs(["full"], {"full": 1}, lengths)
                # A single-group graph uses the anchor ABI, not a one-entry tag map.
                inputs.attention_inputs = get_primary_attention_inputs(inputs)
                inputs.input_hiddens = (
                    torch.arange(rows * hc_mult * 4, device="cuda", dtype=torch.float32)
                    .remainder(11)
                    .reshape(rows, hc_mult * 4)
                    .to(torch.bfloat16)
                )
                metadata = get_primary_attention_inputs(inputs)
                metadata.prefix_lengths.fill_(7)
                metadata.context_total_kv_length = rows + 7 * len(lengths)
                cu_kv = [0]
                for length in lengths:
                    cu_kv.append(cu_kv[-1] + 7 + length)
                metadata.cu_kv_seqlens_device.copy_(
                    torch.tensor(cu_kv, device="cuda", dtype=torch.int32)
                )
                positions = torch.tensor(
                    [7 + offset for length in lengths for offset in range(length)],
                    device="cuda",
                    dtype=torch.int32,
                )
                inputs.combo_position_ids = positions
                expected = inputs.input_hiddens.reshape(rows, hc_mult, 4).sum(1)
                expected = (
                    expected
                    + (positions + inputs.input_ids).to(expected.dtype)[:, None]
                )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), lanes * len(lengths))
                output = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

        overflow = fixtures._build_prefill_inputs(["full"], {"full": 1}, [1] * 9)
        overflow.attention_inputs = get_primary_attention_inputs(overflow)
        self.assertFalse(runner.canRun(overflow))

    def test_undeclared_layout_keeps_packed_metadata_with_hc(self):
        torch.cuda.set_device(0)
        model = PackedDraftCapacityModel(hc_mult=4)
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4, 3, 4)
        self.assertEqual(model.shapes, {(3 * batch, 8) for batch in range(1, 9)})
        for lengths in ([1] * 8, [3], [3, 1, 2]):
            with self.subTest(lengths=lengths):
                rows = sum(lengths)
                inputs = fixtures._build_prefill_inputs(["full"], {"full": 1}, lengths)
                inputs.attention_inputs = get_primary_attention_inputs(inputs)
                inputs.input_hiddens = torch.ones(
                    (rows, 16), dtype=torch.bfloat16, device="cuda"
                )
                inputs.combo_position_ids = torch.tensor(
                    [offset for length in lengths for offset in range(length)],
                    dtype=torch.int32,
                    device="cuda",
                )
                expected = 4 + inputs.input_ids.to(torch.bfloat16)[:, None].expand(
                    -1, 4
                )
                self.assertTrue(runner.canRun(inputs))
                output = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_undeclared_layout_rejects_batches_beyond_captured_rows(self):
        # max_generate_batch_size can admit decode batches wider than the
        # captured max_bs_ rows; per-row acceptance packs few enough tokens
        # to still fit a captured bucket, so the row guard is what keeps the
        # replay out of graph preparation and its block-table row check.
        torch.cuda.set_device(0)
        model = PackedDraftCapacityModel(hc_mult=4)
        runner = CudaGraphRunner()
        runner.init_mtp_draft_prefill(model, 8, 4, 3, 4)
        overflow = fixtures._build_prefill_inputs(["full"], {"full": 1}, [1] * 9)
        overflow.attention_inputs = get_primary_attention_inputs(overflow)
        # 9 rows with 9 packed tokens fit the 9-token capture bucket; only
        # the row count exceeds the captured 8 rows.
        overflow.combo_position_ids = torch.zeros(9, dtype=torch.int32, device="cuda")
        self.assertFalse(runner.canRun(overflow))
        fitting = fixtures._build_prefill_inputs(["full"], {"full": 1}, [1] * 8)
        fitting.attention_inputs = get_primary_attention_inputs(fitting)
        fitting.combo_position_ids = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.assertTrue(runner.canRun(fitting))

    def test_nonpositive_request_lanes_rejected_before_capture(self):
        for lanes in (0, -1):
            with self.subTest(lanes=lanes):
                with self.assertRaisesRegex(RuntimeError, "int32 token/padding"):
                    CudaGraphRunner().init_mtp_draft_prefill(
                        DraftCapacityModel(), 8, 4, lanes, 4
                    )


if __name__ == "__main__":
    unittest.main()
