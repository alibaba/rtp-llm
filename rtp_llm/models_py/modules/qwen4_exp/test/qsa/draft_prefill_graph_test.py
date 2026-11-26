import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
    DraftPrefillGraphReplay,
    draft_graph_lanes,
    draft_graph_rows,
)


class DraftPrefillGraphTest(unittest.TestCase):
    def test_fixed_capacity_mapping_handles_padding_and_changed_lengths(self):
        for lengths, prefixes in (([1, 4, 0], [7, 127, 0]), ([3, 2, 1], [125, 5, 9])):
            cu = [0]
            for length in lengths:
                cu.append(cu[-1] + length)
            inputs = SimpleNamespace(
                input_lengths_device=torch.tensor(lengths, dtype=torch.int32),
                prefix_lengths_device=torch.tensor(prefixes, dtype=torch.int32),
                cu_seqlens_device=torch.tensor(cu, dtype=torch.int32),
            )
            sources, inverse, valid, positions, visible = draft_graph_rows(inputs, 12)
            packed = torch.arange(12)
            recovered = packed.index_select(0, sources).index_select(0, inverse)
            torch.testing.assert_close(recovered[valid], packed[: cu[-1]])
            expected_positions = [
                p + i for p, n in zip(prefixes, lengths) for i in range(n)
            ]
            self.assertEqual(positions[valid].tolist(), expected_positions)
            self.assertTrue(torch.all(visible[inputs.input_lengths_device == 0] == 0))

    def test_short_capacity_mapping_and_prewrite_bounds(self):
        for lanes in (1, 2, 3):
            lengths = [lanes, 1, 0]
            inputs = SimpleNamespace(
                input_lengths_device=torch.tensor(lengths, dtype=torch.int32),
                prefix_lengths_device=torch.tensor([127, 5, 0], dtype=torch.int32),
                cu_seqlens_device=torch.tensor(
                    [0, lanes, lanes + 1, lanes + 1], dtype=torch.int32
                ),
            )
            sources, inverse, valid, _, visible = draft_graph_rows(inputs, 3 * lanes)
            packed = torch.arange(3 * lanes)
            recovered = packed.index_select(0, sources).index_select(0, inverse)
            torch.testing.assert_close(recovered[valid], packed[: lanes + 1])
            self.assertEqual(visible.shape, (3, lanes))
        inputs.input_lengths_device = torch.zeros(0, dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            draft_graph_lanes(inputs, 4)
        inputs.input_lengths_device = torch.ones(3, dtype=torch.int32)
        for bad in (0, 5, 15):
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                draft_graph_lanes(inputs, bad)
        hook, groups, delegate = self._replay()
        groups["full"].input_lengths[1] = 5
        with self.assertRaisesRegex(RuntimeError, "within request capacity"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def _replay(self, *, device="cpu", state_page_size=128):
        lengths = torch.tensor([1, 4, 0], dtype=torch.int32)
        prefixes = torch.tensor([7, 127, 0], dtype=torch.int32)
        cu = torch.tensor([0, 1, 5, 5], dtype=torch.int32)
        positions = (
            torch.tensor(
                [7, 127, 128, 129, 130] + [0] * 7, device=device, dtype=torch.int32
            )
            .unsqueeze(1)
            .expand(-1, 3)
            .contiguous()
            .flatten()
        )
        tables = torch.tensor([[1, 2], [3, 4], [0, 0]], dtype=torch.int32)
        groups = {}
        for tag in ("full", INDEXER_KV_TAG, INDEXER_STATE_TAG):
            groups[tag] = SimpleNamespace(
                input_lengths=lengths,
                prefix_lengths=prefixes,
                cu_seqlens=cu,
                cu_seqlens_device=cu.to(device).clone(),
                input_lengths_device=lengths.to(device),
                prefix_lengths_device=prefixes.to(device),
                combo_position_ids=positions,
                kv_cache_kernel_block_id=tables.clone(),
                kv_cache_block_id_device=tables.to(device).clone(),
            )
        delegate = Mock()
        hook = self._capture(groups, delegate, state_page_size=state_page_size)
        return hook, groups, delegate

    def _capture(self, groups, delegate, *, state_page_size=128):
        return DraftPrefillGraphReplay(
            {"full": delegate},
            [
                ("full", 12, 128, False),
                (INDEXER_KV_TAG, 12, 32, True),
                (INDEXER_STATE_TAG, 12, state_page_size, False),
            ],
            3,
            12,
            capture_groups=groups,
        )

    def test_capture_rejects_invalid_fixed_layout(self):
        _, groups, delegate = self._replay()
        anchor = groups["full"]
        bad_layouts = (
            ("combo_position_ids", None, "capacity"),
            ("combo_position_ids", torch.zeros(35, dtype=torch.int32), "capacity"),
            ("cu_seqlens_device", None, "cu_seqlens"),
            ("cu_seqlens_device", torch.zeros(4, dtype=torch.int64), "cu_seqlens"),
            ("cu_seqlens_device", torch.zeros(3, dtype=torch.int32), "cu_seqlens"),
            ("cu_seqlens_device", torch.zeros(4, 1, dtype=torch.int32), "cu_seqlens"),
        )
        for field, value, error in bad_layouts:
            with self.subTest(field=field, value=value):
                bad_anchor = SimpleNamespace(**(vars(anchor) | {field: value}))
                with self.assertRaisesRegex(RuntimeError, f"capture.*{error}"):
                    self._capture(dict(groups, full=bad_anchor), delegate)

        del groups[INDEXER_STATE_TAG]
        with self.assertRaisesRegex(RuntimeError, "capture.*raw-state cache group"):
            self._capture(groups, delegate)
        delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_stages_host_reference_with_cuda_default_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        replay, groups, delegate = self._replay(device=device)
        with torch.device("cuda"):
            replay.prepare_cuda_graph(groups)
        reference = replay._workspace.reference_host
        self.assertEqual(reference.device.type, "cpu")
        self.assertEqual(reference.is_pinned(), device == "cuda")
        delegate.prepare_cuda_graph.assert_called_once_with(groups["full"])

    def test_replay_checks_all_pools_before_delegate(self):
        hook, groups, delegate = self._replay()
        self.assertNotIsInstance(hook, dict)
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once_with(groups["full"])
        delegate.reset_mock()
        groups[INDEXER_KV_TAG].kv_cache_kernel_block_id[1, 0] = 0
        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_revalidates_visible_pages_changed_in_place(self):
        for tag, column in (("full", 1), (INDEXER_KV_TAG, 0)):
            for invalid in (0, -1, 12, 13):
                with self.subTest(tag=tag, invalid=invalid):
                    hook, groups, delegate = self._replay()
                    table = groups[tag].kv_cache_kernel_block_id
                    storage = table.data_ptr()
                    hook.prepare_cuda_graph(groups)
                    delegate.reset_mock()
                    original = int(table[1, column])
                    table[1, column] = invalid
                    with self.assertRaisesRegex(RuntimeError, "physical block"):
                        hook.prepare_cuda_graph(groups)
                    self.assertEqual(table.data_ptr(), storage)
                    delegate.prepare_cuda_graph.assert_not_called()
                    table[1, column] = original
                    hook.prepare_cuda_graph(groups)
                    delegate.prepare_cuda_graph.assert_called_once()

    def test_replay_checks_only_visible_pages_at_full_and_compressed_boundaries(self):
        for tag, prefix in (("full", 127), (INDEXER_KV_TAG, 130)):
            with self.subTest(tag=tag):
                hook, groups, delegate = self._replay()
                table = groups[tag].kv_cache_kernel_block_id
                # Keep a non-contiguous host table, including invalid padding
                # and a second page that is not visible until the next step.
                backing = torch.zeros(3, 4, dtype=torch.int32)
                backing[:, ::2].copy_(table)
                table = backing[:, ::2]
                groups[tag].kv_cache_kernel_block_id = table
                table[0, 1] = 0
                groups["full"].prefix_lengths[0] = prefix
                hook.prepare_cuda_graph(groups)
                delegate.reset_mock()
                groups["full"].prefix_lengths[0] += 1
                with self.assertRaisesRegex(RuntimeError, "physical block"):
                    hook.prepare_cuda_graph(groups)
                delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_visible_tail_checks_short_and_long_strided_rows(self):
        for tag in ("full", INDEXER_KV_TAG):
            for pages in (0, 1, 64, 65, 256):
                if tag == "full" and pages == 0:
                    continue
                with self.subTest(tag=tag, pages=pages):
                    hook, groups, delegate = self._replay()
                    width = pages + 2
                    for name in ("full", INDEXER_KV_TAG):
                        storage = torch.ones(3, 2 * width, dtype=torch.int32)
                        table = storage[:, ::2]
                        table[2].fill_(-1)  # Inactive rows are outside visibility.
                        groups[name].kv_cache_kernel_block_id = table
                    groups[INDEXER_STATE_TAG].kv_cache_block_id_device = torch.ones(
                        3, width, dtype=torch.int32
                    )
                    groups[tag].kv_cache_kernel_block_id[0, pages] = 0
                    prefix = pages * 128 - 1 if tag == "full" else pages * 128 + 2
                    groups["full"].prefix_lengths[0] = prefix
                    hook.prepare_cuda_graph(groups)
                    delegate.prepare_cuda_graph.assert_called_once()
                    delegate.reset_mock()
                    groups["full"].prefix_lengths[0] += 1
                    with self.assertRaisesRegex(RuntimeError, "physical block"):
                        hook.prepare_cuda_graph(groups)
                    delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_resolves_state_pages_from_refreshed_kernel_table(self):
        hook, groups, delegate = self._replay(state_page_size=256)
        state = groups[INDEXER_STATE_TAG]
        physical = state.kv_cache_block_id_device
        expanded = physical[..., None] * 2 + torch.arange(2, dtype=torch.int32)
        state.kv_cache_kernel_block_id_device = torch.where(
            physical[..., None] > 0, expanded, 0
        ).flatten(1)
        state.kv_cache_block_id_device = None
        state.kv_cache_block_id = torch.zeros_like(physical)
        state.is_cuda_graph = True
        hook.state_cache = SimpleNamespace(
            get_seq_size_per_block=lambda tag: 256,
            get_kernel_seq_size_per_block=lambda tag: 128,
        )
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once_with(groups["full"])
        delegate.reset_mock()
        state.kv_cache_kernel_block_id_device[1, :2].zero_()
        with self.assertRaisesRegex(RuntimeError, "raw-state pages"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_ignores_unused_host_cu_padding_tail(self):
        hook, groups, delegate = self._replay()
        groups["full"].cu_seqlens[-1] = 12
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay mirrors are required")
    def test_device_cu_validation_uses_existing_verdict_without_host_copy(self):
        hook, groups, delegate = self._replay(device="cuda")
        groups["full"].cu_seqlens[1] = 999  # Public host mirror can be stale.
        reads = []
        original_item, original_tolist = torch.Tensor.item, torch.Tensor.tolist

        def item(tensor, *args, **kwargs):
            if tensor.is_cuda:
                reads.append(1)
            return original_item(tensor, *args, **kwargs)

        def tolist(tensor, *args, **kwargs):
            self.assertFalse(tensor.is_cuda, "draft graph copied device CU to host")
            return original_tolist(tensor, *args, **kwargs)

        with (
            patch.object(torch.Tensor, "item", item),
            patch.object(torch.Tensor, "tolist", tolist),
        ):
            hook.prepare_cuda_graph(groups)
        self.assertEqual(len(reads), 1)
        delegate.prepare_cuda_graph.assert_called_once()
        delegate.reset_mock()
        groups["full"].cu_seqlens_device[1] = 999
        with self.assertRaisesRegex(RuntimeError, "cu_seqlens"):
            hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_canonicalizes_transported_media_positions_in_place(self):
        hook, groups, delegate = self._replay()
        anchor = groups["full"]
        storage = anchor.combo_position_ids.data_ptr()
        for prefixes in ([7, 127, 0], [5, 125, 0]):
            anchor.prefix_lengths.copy_(torch.tensor(prefixes, dtype=torch.int32))
            anchor.combo_position_ids.fill_(999)
            hook.prepare_cuda_graph(groups)
            expected = [prefixes[0], *range(prefixes[1], prefixes[1] + 4)] + [0] * 7
            torch.testing.assert_close(
                anchor.combo_position_ids,
                torch.tensor(expected, dtype=torch.int32).repeat_interleave(3),
            )
            self.assertEqual(anchor.combo_position_ids.data_ptr(), storage)
            delegate.prepare_cuda_graph.assert_called_once_with(anchor)
            delegate.reset_mock()

    def test_replay_reuses_scratch_and_refreshes_all_padding(self):
        hook, groups, _ = self._replay()
        anchor = groups["full"]
        hook.prepare_cuda_graph(groups)
        workspace = hook._workspace
        pointers = (workspace.reference_host.data_ptr(), workspace.reference.data_ptr())
        # Shrink the live tokens without changing this bucket's storage.
        anchor.input_lengths.copy_(torch.tensor([2, 0, 0], dtype=torch.int32))
        anchor.prefix_lengths.copy_(torch.tensor([9, 0, 0], dtype=torch.int32))
        anchor.cu_seqlens_device.copy_(torch.tensor([0, 2, 2, 2], dtype=torch.int32))
        anchor.combo_position_ids.fill_(999)
        hook.prepare_cuda_graph(groups)
        self.assertIs(hook._workspace, workspace)
        self.assertEqual(
            (workspace.reference_host.data_ptr(), workspace.reference.data_ptr()),
            pointers,
        )
        torch.testing.assert_close(
            anchor.combo_position_ids,
            torch.tensor([9, 10] + [0] * 10, dtype=torch.int32).repeat_interleave(3),
        )
        other, other_groups, _ = self._replay()
        other.prepare_cuda_graph(other_groups)
        self.assertIsNot(other._workspace, workspace)
        self.assertNotEqual(other._workspace.reference.data_ptr(), pointers[-1])

    def test_replay_discards_workspace_and_recovers_after_failure(self):
        for defect in ("host", "verdict", "delegate"):
            with self.subTest(defect=defect):
                hook, groups, delegate = self._replay()
                hook.prepare_cuda_graph(groups)
                first = hook._workspace
                delegate.reset_mock()
                if defect == "host":
                    groups["full"].kv_cache_kernel_block_id[0, 0] = 0
                elif defect == "verdict":
                    groups["full"].cu_seqlens_device[1] = 999
                else:
                    delegate.prepare_cuda_graph.side_effect = KeyboardInterrupt
                error = KeyboardInterrupt if defect == "delegate" else RuntimeError
                with self.assertRaises(error):
                    hook.prepare_cuda_graph(groups)
                self.assertIsNone(hook._workspace)
                if defect != "delegate":
                    delegate.prepare_cuda_graph.assert_not_called()
                groups["full"].kv_cache_kernel_block_id[0, 0] = 1
                groups["full"].cu_seqlens_device[1] = 1
                delegate.prepare_cuda_graph.side_effect = None
                delegate.reset_mock()
                hook.prepare_cuda_graph(groups)
                self.assertIsNot(hook._workspace, first)
                delegate.prepare_cuda_graph.assert_called_once_with(groups["full"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA staging requires a GPU")
    def test_replay_scratch_is_stream_local_and_recovers_after_failure(self):
        hook, groups, _ = self._replay(device="cuda")
        hook.prepare_cuda_graph(groups)
        first = hook._workspace
        groups["full"].cu_seqlens_device[1] = 999
        with self.assertRaisesRegex(RuntimeError, "cu_seqlens"):
            hook.prepare_cuda_graph(groups)
        self.assertIsNone(hook._workspace)
        groups["full"].cu_seqlens_device[1] = 1
        hook.prepare_cuda_graph(groups)
        recovered = hook._workspace
        self.assertIsNot(recovered, first)
        hook.prepare_cuda_graph(groups)
        self.assertIs(hook._workspace, recovered)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            hook.prepare_cuda_graph(groups)
            second = hook._workspace
        torch.cuda.current_stream().wait_stream(stream)
        self.assertIsNot(recovered, second)
        self.assertNotEqual(recovered.reference.data_ptr(), second.reference.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA staging requires a GPU")
    def test_replay_failure_before_verdict_preserves_pending_copies(self):
        hook, groups, delegate = self._replay(device="cuda")
        hook.prepare_cuda_graph(groups)
        spares = [
            torch.empty((2, 3), dtype=torch.int64, pin_memory=True) for _ in range(2)
        ]
        del spares

        def fail_before_verdict(tensor):
            self.assertTrue(tensor.is_cuda)
            raise RuntimeError("injected failure before verdict")

        pending = torch.cuda.Event()
        for _ in range(3):
            self.assertIsNotNone(hook._workspace.invalid)
            old_reference = hook._workspace.reference
            old_host_pointer = hook._workspace.reference_host.data_ptr()
            delegate.reset_mock()

            torch.cuda._sleep(200_000_000)
            # A pending event ahead of the copy proves the old host buffer
            # is still in use when the allocator supplies another buffer.
            pending.record()
            with patch.object(torch.Tensor, "item", fail_before_verdict):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    hook.prepare_cuda_graph(groups)
            self.assertIsNone(hook._workspace)
            delegate.prepare_cuda_graph.assert_not_called()
            poisoned = torch.empty((2, 3), dtype=torch.int64, pin_memory=True)
            if not pending.query():
                self.assertNotEqual(poisoned.data_ptr(), old_host_pointer)
                poisoned.fill_(-1)
                break
            hook.prepare_cuda_graph(groups)
        else:
            self.skipTest("CUDA delay ended before pinned allocation could be checked")

        anchor = groups["full"]
        anchor.input_lengths.copy_(torch.tensor([2, 0, 0], dtype=torch.int32))
        anchor.prefix_lengths.copy_(torch.tensor([9, 0, 0], dtype=torch.int32))
        for inputs in groups.values():
            inputs.input_lengths_device.zero_()
            inputs.input_lengths_device[:1].fill_(2)
            inputs.prefix_lengths_device.zero_()
            inputs.prefix_lengths_device[:1].fill_(9)
            inputs.cu_seqlens_device.zero_()
            inputs.cu_seqlens_device[1:].fill_(2)
        hook.prepare_cuda_graph(groups)
        torch.testing.assert_close(
            old_reference,
            torch.tensor([[1, 4, 0], [7, 127, 0]], device="cuda", dtype=torch.int64),
        )
        torch.testing.assert_close(
            anchor.combo_position_ids,
            torch.tensor(
                [9, 10] + [0] * 10, device="cuda", dtype=torch.int32
            ).repeat_interleave(3),
        )
        delegate.prepare_cuda_graph.assert_called_once_with(anchor)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA state projection requires a GPU"
    )
    def test_replay_refreshes_cuda_state_projection_workspace(self):
        for factor in (2, 4):
            with self.subTest(factor=factor):
                hook, groups, delegate = self._replay(
                    device="cuda", state_page_size=128 * factor
                )
                state = groups[INDEXER_STATE_TAG]
                physical = state.kv_cache_block_id_device.clone()
                offsets = torch.arange(factor, device="cuda", dtype=torch.int32)

                def expanded():
                    return torch.where(
                        physical[..., None] > 0,
                        physical[..., None] * factor + offsets,
                        0,
                    ).flatten(1)

                state.kv_cache_kernel_block_id_device = expanded()
                state.kv_cache_block_id_device = None
                state.kv_cache_block_id = torch.zeros(3, 2, dtype=torch.int32)
                state.is_cuda_graph = True
                hook.state_cache = SimpleNamespace(
                    get_seq_size_per_block=lambda tag: 128 * factor,
                    get_kernel_seq_size_per_block=lambda tag: 128,
                )
                hook.prepare_cuda_graph(groups)
                workspace = hook._workspace
                view = workspace.state_inputs[INDEXER_STATE_TAG][1]
                projection = view.kv_cache_block_id_device
                torch.testing.assert_close(projection, physical)
                physical[1, 0] = 6
                state.kv_cache_kernel_block_id_device.copy_(expanded())
                hook.prepare_cuda_graph(groups)
                self.assertIs(hook._workspace, workspace)
                self.assertIs(workspace.state_inputs[INDEXER_STATE_TAG][1], view)
                self.assertEqual(
                    view.kv_cache_block_id_device.data_ptr(), projection.data_ptr()
                )
                torch.testing.assert_close(projection, physical)
                delegate.reset_mock()
                state.kv_cache_kernel_block_id_device[1, :factor].zero_()
                with self.assertRaisesRegex(RuntimeError, "raw-state pages"):
                    hook.prepare_cuda_graph(groups)
                self.assertIsNone(hook._workspace)
                delegate.prepare_cuda_graph.assert_not_called()
                state.kv_cache_kernel_block_id_device.copy_(expanded())
                hook.prepare_cuda_graph(groups)
                self.assertIsNot(hook._workspace, workspace)
                torch.testing.assert_close(
                    hook._workspace.state_inputs[INDEXER_STATE_TAG][
                        1
                    ].kv_cache_block_id_device,
                    physical,
                )
                delegate.prepare_cuda_graph.assert_called_once_with(groups["full"])

    def test_replay_rejects_state_pages_positions_and_length_changes(self):
        for defect in ("state", "position_capacity", "length", "cu"):
            with self.subTest(defect=defect):
                hook, groups, delegate = self._replay()
                anchor = groups["full"]
                if defect == "state":
                    groups[INDEXER_STATE_TAG].kv_cache_block_id_device[1, 1] = 12
                elif defect == "position_capacity":
                    anchor.combo_position_ids = anchor.combo_position_ids[:-1]
                elif defect == "length":
                    anchor.input_lengths[2] = 5
                else:
                    anchor.cu_seqlens_device[2] = 6
                with self.assertRaises(RuntimeError):
                    hook.prepare_cuda_graph(groups)
                delegate.prepare_cuda_graph.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay mirrors are required")
    def test_replay_uses_device_state_table_with_stale_host_table(self):
        hook, groups, delegate = self._replay(device="cuda")
        groups[INDEXER_STATE_TAG].kv_cache_block_id = torch.zeros(
            3, 2, dtype=torch.int32
        )
        hook.prepare_cuda_graph(groups)
        delegate.prepare_cuda_graph.assert_called_once()
        groups[INDEXER_STATE_TAG].kv_cache_block_id_device[1, 1] = 0
        with self.assertRaisesRegex(RuntimeError, "raw-state pages"):
            hook.prepare_cuda_graph(groups)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay metadata is required")
    def test_device_prepare_refreshes_independent_group_positions_and_padding(self):
        hook, groups, delegate = self._replay(device="cuda")
        media_positions = groups["full"].combo_position_ids.clone().fill_(999)
        for inputs in groups.values():
            inputs.combo_position_ids = media_positions.clone()
        pointers = [inputs.combo_position_ids.data_ptr() for inputs in groups.values()]
        for lengths, prefixes in (
            ([1, 4, 0], [7, 127, 0]),
            ([2, 0, 0], [9, 0, 0]),
            ([0, 0, 0], [0, 0, 0]),
        ):
            cu = [0]
            for length in lengths:
                cu.append(cu[-1] + length)
            for inputs in groups.values():
                inputs.input_lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
                inputs.prefix_lengths.copy_(torch.tensor(prefixes, dtype=torch.int32))
                inputs.input_lengths_device.copy_(inputs.input_lengths)
                inputs.prefix_lengths_device.copy_(inputs.prefix_lengths)
                inputs.cu_seqlens_device.copy_(torch.tensor(cu, dtype=torch.int32))
            incoming_cu = groups["full"].cu_seqlens_device.clone()
            hook.prepare_cuda_graph(groups)
            expected = [p + i for p, n in zip(prefixes, lengths) for i in range(n)]
            expected += [0] * (12 - len(expected))
            for inputs in groups.values():
                torch.testing.assert_close(
                    inputs.combo_position_ids,
                    torch.tensor(
                        expected, dtype=torch.int32, device="cuda"
                    ).repeat_interleave(3),
                )
            torch.testing.assert_close(groups["full"].cu_seqlens_device, incoming_cu)
            self.assertIsNotNone(hook._workspace.invalid)
            delegate.reset_mock()
        self.assertEqual(
            [inputs.combo_position_ids.data_ptr() for inputs in groups.values()],
            pointers,
        )
        self.assertTrue(torch.all(media_positions == 999))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay metadata is required")
    def test_device_prepare_fallback_uses_current_host_mirrors(self):
        for defect in ("strided_table", "length_dtype", "page_size"):
            with self.subTest(defect=defect):
                hook, groups, delegate = self._replay(
                    device="cuda", state_page_size=127 if defect == "page_size" else 128
                )
                anchor = groups["full"]
                expected = anchor.combo_position_ids.clone()
                if defect == "strided_table":
                    state = groups[INDEXER_STATE_TAG]
                    table = state.kv_cache_block_id_device
                    state.kv_cache_block_id_device = table.T.contiguous().T
                elif defect == "length_dtype":
                    anchor.input_lengths_device = anchor.input_lengths_device.long()
                anchor.combo_position_ids.fill_(999)
                hook.prepare_cuda_graph(groups)
                self.assertIsNone(hook._workspace.invalid)
                torch.testing.assert_close(
                    hook._workspace.reference.cpu(),
                    torch.stack((anchor.input_lengths, anchor.prefix_lengths)).long(),
                )
                torch.testing.assert_close(anchor.combo_position_ids, expected)
                delegate.prepare_cuda_graph.assert_called_once_with(anchor)
                delegate.reset_mock()
                anchor.cu_seqlens_device[1] = 999
                with self.assertRaisesRegex(RuntimeError, "cu_seqlens"):
                    hook.prepare_cuda_graph(groups)
                delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_rejects_host_device_mismatch_before_delegate(self):
        layouts = ["cpu"]
        if torch.cuda.is_available():
            layouts += ["cuda", "cuda_strided"]
        for layout in layouts:
            for field, values in (
                ("input_lengths_device", [2, 3, 0]),
                ("prefix_lengths_device", [8, 127, 0]),
                ("cu_seqlens_device", [0, 2, 5, 5]),
            ):
                with self.subTest(layout=layout, field=field):
                    device = "cpu" if layout == "cpu" else "cuda"
                    hook, groups, delegate = self._replay(device=device)
                    if layout == "cuda_strided":
                        state = groups[INDEXER_STATE_TAG]
                        state.kv_cache_block_id_device = (
                            state.kv_cache_block_id_device.T.contiguous().T
                        )
                    value = torch.tensor(values, device=device, dtype=torch.int32)
                    anchor = groups["full"]
                    groups["full"] = SimpleNamespace(**(vars(anchor) | {field: value}))
                    with self.assertRaisesRegex(
                        RuntimeError, "raw-state pages, cu_seqlens"
                    ):
                        hook.prepare_cuda_graph(groups)
                    delegate.prepare_cuda_graph.assert_not_called()

    def test_replay_checks_interior_raw_state_pages(self):
        for page_size in (1, 2, 3):
            with self.subTest(page_size=page_size):
                hook, groups, delegate = self._replay(state_page_size=page_size)
                anchor = groups["full"]
                anchor.input_lengths.copy_(torch.tensor([4, 0, 0]))
                anchor.prefix_lengths.copy_(torch.tensor([7, 0, 0]))
                anchor.cu_seqlens_device.copy_(torch.tensor([0, 4, 4, 4]))
                state = groups[INDEXER_STATE_TAG]
                table = state.kv_cache_block_id_device = torch.ones(
                    3, 11, dtype=torch.int32
                )
                hook.prepare_cuda_graph(groups)
                delegate.reset_mock()
                # The raw-state window spans positions 4 through 10 inclusive.
                table[0, 4 // page_size + 1] = 0
                with self.assertRaisesRegex(RuntimeError, "raw-state pages"):
                    hook.prepare_cuda_graph(groups)
                delegate.prepare_cuda_graph.assert_not_called()


if __name__ == "__main__":
    unittest.main()
