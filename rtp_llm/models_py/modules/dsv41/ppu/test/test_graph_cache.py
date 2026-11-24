"""Native packed layout and changing-input contracts for PPU decode Graph."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import (
    NativeGraphCache,
    PPUDecodeGraphImpl,
)
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import NativeCacheBridge
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import FRAMEWORK_POOL_TAGS
from rtp_llm.models_py.modules.dsv41.ppu.test.test_native_cache import NativePools
from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config


class GraphCacheTest(unittest.TestCase):
    def make_cache(self):
        config = synthetic_config()
        native = NativePools(config)
        cache = NativeGraphCache(config, native, native.inputs, device="cpu")
        return config, native, cache

    def test_slots_use_independent_tag_tables_and_raw_indexer_capacity(self):
        _, native, cache = self.make_cache()
        positions = torch.tensor([[0, 2, 7, 8], [8, 10, 15, 16]])
        for owner, tag in cache.geometry.pool_keys():
            slots = cache.slots(owner, tag, positions)
            table = cache.tables[tag]
            block = torch.stack(
                [table[row, positions[row] // cache.spans[tag]] for row in range(2)]
            )
            offset = (
                positions.remainder(cache.spans[tag]) // cache.geometry.ratio(owner)
                if tag in ("global_kv_1", "global_kv_2", "indexer_kv")
                else positions.remainder(cache.entries[tag])
            )
            self.assertTrue(torch.equal(slots, block * cache.entries[tag] + offset))
            self.assertEqual(
                cache.pool(owner, tag).data_ptr(),
                native.handles[
                    (owner, FRAMEWORK_POOL_TAGS[tag])
                ].kv_cache_base.data_ptr(),
            )
        # Ratio 2 does not halve the indexer payload/scale plane's raw capacity.
        ratio2 = next(
            owner
            for owner in cache.geometry.kv_source_layer_ids
            if cache.geometry.ratio(owner) == 2
        )
        self.assertEqual(cache.entries["indexer_kv"], native.span)
        self.assertEqual(cache.entries["global_kv_2"], native.span // 2)
        self.assertEqual(
            int(cache.slots(ratio2, "indexer_kv", torch.tensor([2, 2]))[0])
            % native.span,
            1,
        )

    def test_reads_and_writes_match_native_bridge_without_touching_padding(self):
        config, native, cache = self.make_cache()
        bridge = NativeCacheBridge(config, native, native.inputs)
        for owner, tag in cache.geometry.pool_keys():
            # A round-trip batch must address distinct entries. CSA has four
            # ring entries per physical block: raw positions 2 and 6 alias.
            positions = torch.tensor(
                [[0, 1, 2, 8], [8, 9, 10, 16]]
                if tag == "csa_state"
                else [[0, 2, 6, 8], [8, 10, 14, 16]]
            )
            pool = cache.pool(owner, tag)
            stride_tail = pool[:, -16:].clone()
            slots = cache.slots(owner, tag, positions)
            width = 1024 if tag == "csa_state" else {"indexer_kv": 68, "swa_kv": 528}.get(tag, 288)
            values = (
                torch.arange(positions.numel() * width)
                .reshape(*positions.shape, width)
                .remainder(149)
                .to(pool.dtype)
            )
            cache.write_rows(owner, tag, slots, values)
            self.assertTrue(torch.equal(cache.read_rows(owner, tag, slots), values), (owner, tag))
            for row in range(positions.shape[0]):
                actual = bridge._copy_rows((owner, tag), row, positions[row])
                self.assertTrue(torch.equal(actual, values[row]), (owner, tag, row))
            self.assertTrue(torch.equal(pool[:, -16:], stride_tail), (owner, tag))

    def test_csa_ring_wrap_replaces_the_same_slot_in_decode_order(self):
        config, native, cache = self.make_cache()
        owner = next(
            owner
            for owner in cache.geometry.kv_source_layer_ids
            if cache.geometry.ratio(owner) == 2
        )
        tag = "csa_state"
        first_positions = torch.tensor([0, 8])
        wrapped_positions = first_positions + cache.entries[tag]
        first_slots = cache.slots(owner, tag, first_positions)
        wrapped_slots = cache.slots(owner, tag, wrapped_positions)
        self.assertTrue(torch.equal(first_slots, wrapped_slots))
        adjacent_slots = cache.slots(owner, tag, first_positions + 1)
        first = torch.arange(2 * 1024, dtype=torch.float32).reshape(2, 1024)
        adjacent = first + 10000
        latest = first + 20000
        cache.write_rows(owner, tag, first_slots, first)
        cache.write_rows(owner, tag, adjacent_slots, adjacent)
        cache.write_rows(owner, tag, wrapped_slots, latest)
        # A later decode step owns the ring entry; the old raw position now
        # sees the latest state, while adjacent entries retain their contents.
        self.assertTrue(torch.equal(cache.read_rows(owner, tag, first_slots), latest))
        self.assertTrue(
            torch.equal(cache.read_rows(owner, tag, adjacent_slots), adjacent)
        )
        bridge = NativeCacheBridge(config, native, native.inputs)
        for row in range(2):
            self.assertTrue(
                torch.equal(
                    bridge._copy_rows(
                        (owner, tag), row, wrapped_positions[row : row + 1]
                    ),
                    latest[row : row + 1],
                )
            )

    def test_negative_positions_and_reserved_block_zero_never_write(self):
        _, native, cache = self.make_cache()
        owner, tag = cache.geometry.pool_keys()[0]
        cache.tables[tag][1, 0] = 0
        positions = torch.tensor([[-1, 2], [0, 2]])
        slots = cache.slots(owner, tag, positions)
        self.assertEqual(int(slots[0, 0]), -1)
        self.assertTrue(torch.equal(slots[1], torch.tensor([-1, -1])))
        pool = cache.pool(owner, tag)
        before = pool[0].clone()
        rows = cache.read_rows(owner, tag, slots)
        self.assertTrue(torch.equal(rows[1], torch.zeros_like(rows[1])))
        cache.write_rows(owner, tag, slots, torch.ones_like(rows))
        self.assertTrue(torch.equal(pool[0], before))

    def test_changed_table_contents_are_seen_without_rebinding(self):
        _, native, cache = self.make_cache()
        owner, tag = cache.geometry.pool_keys()[0]
        original_pointer = cache.tables[tag].data_ptr()
        first = cache.slots(owner, tag, torch.tensor([0, 0]))
        cache.tables[tag][:, 0].sub_(2)
        cache.prepare(native.inputs)
        second = cache.slots(owner, tag, torch.tensor([0, 0]))
        self.assertEqual(cache.tables[tag].data_ptr(), original_pointer)
        self.assertTrue(torch.equal(second, first - 2 * cache.entries[tag]))

    def test_pool_reallocation_is_rejected(self):
        _, native, cache = self.make_cache()
        key = next(iter(native.handles))
        native.handles[key].kv_cache_base = native.handles[key].kv_cache_base.clone()
        with self.assertRaisesRegex(RuntimeError, "allocation changed"):
            cache.prepare(native.inputs)

    def test_physical_and_kernel_table_names_remain_distinct(self):
        config, native, _ = self.make_cache()
        for entry in native.inputs.values():
            entry.kv_cache_block_id_device = entry.kv_cache_block_id_device.clone()
            entry.kv_cache_block_id_device[:, 0] = 3
            entry.kv_cache_kernel_block_id_device[:, 0] = 5
        cache = NativeGraphCache(config, native, native.inputs, device="cpu")
        self.assertEqual(int(cache.tables["swa_kv"][0, 0]), 3)
        self.assertEqual(int(cache.tables["csa_state"][0, 0]), 3)
        self.assertEqual(int(cache.tables["indexer_kv"][0, 0]), 5)


class DecodeGraphMetadataTest(unittest.TestCase):
    def make_impl(self):
        config = synthetic_config(max_position_embeddings=256)
        native = NativePools(config)
        for entry in native.inputs.values():
            entry.input_lengths = torch.ones(2, dtype=torch.int32)
            entry.sequence_lengths = torch.tensor([-123, -123], dtype=torch.int32)
            entry.sequence_lengths_plus_1_device = torch.tensor([8, 18], dtype=torch.int32)
        impl = PPUDecodeGraphImpl(config, native, native.inputs, device="cpu")
        return native, impl

    def test_device_lengths_and_live_rows_change_in_fixed_output_storage(self):
        native, impl = self.make_impl()
        metadata = impl.metadata
        pointers = metadata.positions.data_ptr(), metadata.valid_mask.data_ptr()
        self.assertTrue(torch.equal(metadata.positions, torch.tensor([7, 17])))
        for lengths, count in (([129, 256], 1), ([3, 6], 0), ([8, 9], 2)):
            for entry in native.inputs.values():
                entry.sequence_lengths_plus_1_device.copy_(torch.tensor(lengths))
            impl.prepare_cuda_graph(native.inputs, valid_request_count=count)
            impl.prepare_in_graph(native.inputs)
            self.assertEqual(
                (metadata.positions.data_ptr(), metadata.valid_mask.data_ptr()),
                pointers,
            )
            self.assertTrue(torch.equal(metadata.positions, torch.tensor(lengths) - 1))
            self.assertTrue(torch.equal(metadata.valid_mask, torch.arange(2) < count))

    def test_explicit_count_is_required_and_capture_warmup_lengths_are_bounded(self):
        native, impl = self.make_impl()
        for entry in native.inputs.values():
            entry.sequence_lengths_plus_1_device.fill_(10000)
        with self.assertRaisesRegex(ValueError, "explicit live request count"):
            impl.prepare_cuda_graph(native.inputs)
        impl.prepare_cuda_graph(native.inputs, valid_request_count=2)
        self.assertTrue(torch.equal(impl.metadata.positions, torch.tensor([255, 255])))
        with self.assertRaisesRegex(ValueError, "exceeds captured batch"):
            impl.prepare_cuda_graph(native.inputs, valid_request_count=3)

    def test_device_count_input_is_consumed_without_host_extraction(self):
        native, impl = self.make_impl()
        count = torch.tensor(1, dtype=torch.int32)
        impl.prepare_cuda_graph(native.inputs, valid_request_count=count)
        self.assertTrue(torch.equal(impl.metadata.valid_mask, torch.tensor([True, False])))
        self.assertEqual(impl.cuda_graph_valid_request_count_device.dtype, torch.int32)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_GRAPH_DEVICE") == "1" and torch.cuda.is_available(),
    "requires explicit PPU/CUDA table-geometry qualification",
)
class VerifyTableGeometryDeviceTest(unittest.TestCase):
    """Changing page widths must retain exact checks without new JIT variants."""

    @unittest.skipUnless(os.environ.get("RTP_V41_PPU_ZERO_OVERHEAD_SCHEDULER") == "1", "PPU scheduler receipt")
    def test_local_publication_receipt_orders_d2h(self):
        from librtp_compute_ops import _PpuPublicationReceipt

        receipt = _PpuPublicationReceipt()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        host = torch.empty(1, dtype=torch.int32, pin_memory=True)
        for generation in range(32):
            with torch.cuda.stream(streams[generation % 2]):
                result = torch.full((1,), generation % 5, device="cuda", dtype=torch.int32)
                host.fill_(-1)
                host.copy_(result, non_blocking=True)
                receipt.record()
            receipt.wait()
            self.assertEqual(host.item(), generation % 5)

    @patch.dict(os.environ, {"RTP_V41_PPU_DSPARK_PRIVATE_DRAFT_KV": "1"})
    def test_private_draft_receipt_is_bound_to_exact_capture(self):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_model import DeepSeekV41PPUDSparkModel

        draft = DeepSeekV41PPUDSparkModel
        captures = {
            batch: SimpleNamespace(
                cuda_graph_preparation_in_graph=True,
                stage_private_draft_commit=lambda inputs: inputs,
            )
            for batch in (4, 8, 16, 32)
        }
        model = SimpleNamespace(
            _private_draft_captures=captures, _private_draft_armed_ids=None, _private_draft_pending=None,
        )
        for batch in (32, 4, 16, 8):
            ids = torch.arange(100, 100 + batch, dtype=torch.int64)
            inputs = SimpleNamespace(request_id=ids, warmup=False, is_fake_stream=False)
            self.assertTrue(draft.arm_private_draft_commit(model, ids))
            with self.assertRaisesRegex(ValueError, "does not match the armed requests"):
                draft.note_private_draft_replay(model, captures[4 if batch != 4 else 8], inputs)
            draft.note_private_draft_replay(model, captures[batch], inputs)
            self.assertTrue(draft.finish_private_draft_stage(model))
            order = tuple(ids.tolist())
            self.assertIs(draft.private_draft_commit_receipt(model, order), inputs)
            with self.assertRaisesRegex(ValueError, "request order changed"):
                draft.private_draft_commit_receipt(model, order[::-1])
            draft.abort_private_draft_commit(model)
        self.assertFalse(draft.arm_private_draft_commit(model, torch.arange(5)))

    @torch.inference_mode()
    def test_joint_verify_pair_publishes_dense_buckets(self):
        from rtp_llm.models_py.modules.dsv41.ppu import graph_cache_kernels
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
            BatchedVerifyPublication,
            VerifyPublicationReceipt,
            prepare_verify_pair,
            try_enqueue_verify_table_check,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.verify_transaction import (
            _QueuedGraphVerifyReceipt,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        for batch in (4, 8, 16, 32):
            with self.subTest(batch=batch):
                width, span, row_bytes = 4, 128, 16
                rows = batch * width
                bases = tuple(77824 + 257 * index for index in range(batch))
                counts = tuple(index % (width + 1) for index in range(batch))
                prefixes = torch.tensor(bases, dtype=torch.long, device=device)
                lengths = torch.full((batch,), width, dtype=torch.int32, device=device)
                offsets = torch.arange(width, device=device)
                requests = torch.arange(batch, device=device).repeat_interleave(width)
                raw = (prefixes[:, None] + offsets).flatten()
                valid = torch.ones(rows, dtype=torch.bool, device=device)
                accepted = (offsets < torch.tensor(counts, device=device)[:, None]).flatten()

                def make_journal(name, seed):
                    table = (1 + torch.arange(batch, dtype=torch.int32, device=device))[:, None]
                    table = table.expand(batch, 768).contiguous()
                    live_table = table[:, :704].clone()
                    pool = torch.zeros((batch + 1, span * row_bytes), dtype=torch.uint8, device=device)
                    payload = (
                        (torch.arange(rows * row_bytes, device=device) + seed) % 251
                    ).to(torch.uint8).view(rows, row_bytes)
                    publisher = BatchedVerifyPublication.try_create(
                        [(raw, payload, pool, table, span, span, 1, False, False, False)],
                        rows, device,
                    )
                    self.assertIsNotNone(publisher)
                    slots = table[requests, raw // span].to(torch.int64) * span + raw % span
                    expected = torch.zeros_like(pool)
                    expected.view(-1, row_bytes)[slots[accepted]] = payload[accepted]
                    binding = ((name, table, live_table),)
                    return (publisher, requests, prefixes, valid, binding), pool, expected

                target, target_pool, target_expected = make_journal("target", 1)
                draft, draft_pool, draft_expected = make_journal("draft", 97)
                prepared = prepare_verify_pair(target, (*draft, lengths), prefixes)
                self.assertIsNotNone(prepared)

                with self.assertRaisesRegex(ValueError, "stale acceptance"):
                    prepared.publish(counts, (*bases[:-1], bases[-1] + 1))
                live_draft = draft[4][0][2]
                live_draft[-1, -1].add_(1)
                with self.assertRaisesRegex(RuntimeError, "block table changed for draft"):
                    prepared.publish(counts, bases)
                live_draft[-1, -1].sub_(1)
                self.assertTrue(prepared.publish((0,) * batch, bases))
                self.assertFalse(bool(target_pool.any() or draft_pool.any()))
                self.assertTrue(prepared.publish(counts, bases))
                self.assertTrue(torch.equal(target_pool, target_expected))
                self.assertTrue(torch.equal(draft_pool, draft_expected))

                target_pool.zero_()
                draft_pool.zero_()
                tokens = tuple(100 + row for row in range(rows))
                windows = tuple(
                    tokens[row - column] if row % width >= column else -1
                    for row in range(rows)
                    for column in range(3)
                )
                raw_receipt = torch.tensor(
                    (2, 0, *((width,) * batch), *bases, *tokens, *windows),
                    dtype=torch.long, device=device,
                )
                ready = torch.cuda.Event()
                ready.record()
                replay = _QueuedGraphVerifyReceipt(
                    ready, torch.empty(0), (raw_receipt,), rows, batch, batch, 3
                )
                bindings = try_enqueue_verify_table_check((*target[4], *draft[4]), batch)
                self.assertIsNotNone(bindings)
                receipt = VerifyPublicationReceipt(replay, bindings, 1, 98304, 90112, 3)
                raw_receipt[-1].add_(1)
                self.assertFalse(prepared.publish(counts, replay_check=receipt))
                self.assertFalse(bool(target_pool.any() or draft_pool.any()))
                raw_receipt[-1].sub_(1)
                self.assertTrue(prepared.publish(counts, replay_check=receipt))
                self.assertTrue(torch.equal(target_pool, target_expected))
                self.assertTrue(torch.equal(draft_pool, draft_expected))

                device_counts = torch.tensor([max(1, count) for count in counts], dtype=torch.int32, device=device)
                emitted = torch.tensor(tokens, dtype=torch.int32, device=device).view(batch, width)
                target_pool.zero_()
                draft_pool.zero_()
                live_draft[-1, -1].add_(1)
                self.assertFalse(prepared.publish(
                    device_counts, replay_check=receipt, accepted_tokens=emitted, vocab_size=1024,
                ))
                live_draft[-1, -1].sub_(1)
                emitted[0, 0] = 1024
                self.assertFalse(prepared.publish(
                    device_counts, replay_check=receipt, accepted_tokens=emitted, vocab_size=1024,
                ))
                emitted[0, 0] = tokens[0]
                device_counts[0] = 0
                self.assertFalse(prepared.publish(
                    device_counts, replay_check=receipt, accepted_tokens=emitted, vocab_size=1024,
                ))
                device_counts[0] = 1
                for completion, eos, expected_success in (
                    (1, -1, False), (2, tokens[0], False), (2, tokens[1], True),
                ):
                    raw_receipt[0] = completion
                    snapshots = []
                    with patch.dict(os.environ, {"RTP_V41_PPU_ZERO_OVERHEAD_SCHEDULER": "1"}), patch.object(
                        graph_cache_kernels, "_wait_for_joint_publication_status",
                        wraps=graph_cache_kernels._wait_for_joint_publication_status,
                    ) as wait_for_status:
                        def after_enqueue(status):
                            wait_for_status.assert_not_called()
                            self.assertEqual(status.device, device)
                            self.assertEqual(status.dtype, torch.int32)
                            self.assertEqual(torch.cuda.current_stream(device), prepared.stream)
                            snapshots.append((status.clone(), target_pool.clone(), draft_pool.clone()))

                        self.assertEqual(
                            prepared.publish(
                                device_counts, replay_check=receipt, accepted_tokens=emitted, vocab_size=1024,
                                on_publication_enqueued=after_enqueue, lookahead_eos=eos,
                            ),
                            expected_success,
                        )
                        wait_for_status.assert_called_once()
                    self.assertEqual(len(snapshots), 1)
                    status, target_snapshot, draft_snapshot = snapshots[0]
                    self.assertEqual(status.item() == 0, expected_success)
                    self.assertEqual(bool(target_snapshot.any() and draft_snapshot.any()), expected_success)
                    self.assertTrue(torch.equal(target_snapshot, target_pool))
                    self.assertTrue(torch.equal(draft_snapshot, draft_pool))
                device_result = (target_pool.clone(), draft_pool.clone())
                target_pool.zero_()
                draft_pool.zero_()
                self.assertTrue(prepared.publish(tuple(device_counts.tolist()), replay_check=receipt))
                self.assertTrue(torch.equal(device_result[0], target_pool))
                self.assertTrue(torch.equal(device_result[1], draft_pool))

                for early, eos, success in ((True, -1, True), (True, tokens[0], False), (False, -1, True)):
                    target_pool.zero_()
                    draft_pool.zero_()
                    observed = []

                    def before_write(status, enqueue_writes):
                        observed.append((status.clone(), target_pool.clone(), draft_pool.clone()))
                        if early:
                            enqueue_writes()
                            enqueue_writes()

                    with patch.object(graph_cache_kernels, "_copy_verify_pair",
                                      wraps=graph_cache_kernels._copy_verify_pair) as copy_pair:
                        self.assertEqual(prepared.publish(
                            device_counts, replay_check=receipt, accepted_tokens=emitted, vocab_size=1024,
                            on_publication_validated=before_write, lookahead_eos=eos,
                        ), success)
                        self.assertEqual(copy_pair.call_count, 2)
                    (status, before_target, before_draft), = observed
                    self.assertEqual(int(status.item()) == 0, success)
                    self.assertFalse(bool(before_target.any() or before_draft.any()))
                    if success:
                        self.assertTrue(torch.equal(target_pool, device_result[0]))
                        self.assertTrue(torch.equal(draft_pool, device_result[1]))
                    else:
                        self.assertFalse(bool(target_pool.any() or draft_pool.any()))

    @patch.dict(os.environ, {"RTP_V41_PPU_FUSED_DSPARK_COMMIT_METADATA": "1"})
    @torch.inference_mode()
    def test_private_draft_row_mapper_replays_dense_buckets(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import try_stage_private_draft_rows

        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=device)
        for batch in (4, 8, 16, 32):
            with self.subTest(batch=batch), torch.cuda.stream(stream):
                ends = 81920 + torch.arange(batch, dtype=torch.int32, device=device)
                active = torch.ones(batch, dtype=torch.bool, device=device)
                requests = torch.arange(batch, dtype=torch.int32, device=device).repeat_interleave(4)
                positions = (ends[:, None] - torch.arange(4, 0, -1, dtype=torch.int32, device=device)).flatten()

                def stage():
                    return try_stage_private_draft_rows(
                        requests, positions, ends, active, window=128, max_seq_len=98304,
                    )

                self.assertIsNotNone(stage())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    mapped = stage()
                requests[0] = -1
                requests[-1] = batch
                active[1] = False
                positions[2] = 98304
                positions[3] = -1
                graph.replay()
                safe = requests.long().clamp(0, batch - 1)
                live = (
                    (requests >= 0) & (requests < batch) & active[safe]
                    & (positions >= ends[safe] - 128) & (positions < ends[safe])
                )
                reference = (
                    safe, positions.long().clamp(0, 98303),
                    torch.where(live, positions.long(), -1), live, live.sum(),
                )
                for actual, expected in zip(mapped, reference):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_single_table_growth_strides_masks_and_jit_reuse(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import _verify_table_check

        single, _, cdiv = _verify_table_check()
        device = torch.cuda.current_device()
        for dtype in (torch.int32, torch.int64):
            captured = torch.arange(8 * 768 * 2, dtype=dtype, device="cuda").reshape(8, -1)[:, ::2]
            variants = None
            for width in (513, 1, 17, 511, 512, 514, 527, 528, 529, 544, 641):
                # Both the row stride and column stride change after warmup.
                storage = torch.empty((8, 2 * width + 17), dtype=dtype, device="cuda")
                live = storage[:, 1:1 + 2 * width:2]
                live.copy_(captured[:, :width])
                count = cdiv(8 * width, 256)
                flags = torch.empty(count + 1, dtype=torch.int32, device="cuda")[1:]

                def check():
                    single[(count,)](captured, live, flags, 8, width,
                                      *captured.stride(), *live.stride(), 256, num_warps=4)

                check()
                self.assertEqual(int(flags.count_nonzero()), 0)
                live[-1, -1].add_(1)
                check()
                self.assertEqual(int(flags.count_nonzero()), 1)
                current = len(single.device_caches[device][0])
                if variants is None:
                    variants = current
                self.assertEqual(current, variants, f"page width {width} recompiled the single-table check")

    @torch.inference_mode()
    def test_joint_table_growth_offsets_and_jit_reuse(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
            _launch_joint_table_checks, _verify_table_check,
        )

        single, batched, cdiv = _verify_table_check()
        device = torch.cuda.current_device()
        for dtype in (torch.int32, torch.int64):
            captured = tuple(torch.arange(8 * 768, dtype=dtype, device="cuda").reshape(8, -1)
                             for _ in range(7))
            variants = None
            for width in (513, 514, 527, 528, 529, 544, 641):
                widths = (width, width - 1, 2, width, width + 1, 3, width)
                live = tuple(torch.empty((8, 2 * columns + 17), dtype=dtype, device="cuda")
                             [:, 1:1 + 2 * columns:2] for columns in widths)
                for snapshot, table in zip(captured, live):
                    table.copy_(snapshot[:, :table.size(1)])
                geometry = tuple((table.shape[1], *snapshot.stride(), *table.stride())
                                 for snapshot, table in zip(captured, live))
                offsets = [0]
                for columns in widths:
                    offsets.append(offsets[-1] + cdiv(8 * columns, 256))
                flags = torch.empty(offsets[-1] + 1, dtype=torch.int32, device="cuda")[1:]

                def check():
                    _launch_joint_table_checks(single, batched, captured, live, geometry,
                                               tuple(offsets), flags, 8, 256,
                                               torch.cuda.current_stream())

                check()
                self.assertEqual(int(flags.count_nonzero()), 0)
                live[-1][-1, -1].add_(1)
                check()
                self.assertEqual(int(flags.count_nonzero()), 1)
                self.assertTrue(bool(flags[offsets[-2]:].any()))
                current = len(batched.device_caches[device][0])
                if variants is None:
                    variants = current
                self.assertEqual(current, variants, f"page width {width} recompiled the joint check")

    @torch.inference_mode()
    def test_public_pending_receipt_still_rejects_changed_table(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import try_enqueue_verify_table_check

        captured = torch.arange(8 * 768, device="cuda", dtype=torch.int32).reshape(8, -1)
        for width in (513, 529, 641):
            live = captured[:, :width].clone()
            pending = try_enqueue_verify_table_check((("pool", captured, live),), 8)
            self.assertIsNotNone(pending)
            pending.validate()
            live[-1, -1].add_(1)
            pending = try_enqueue_verify_table_check((("pool", captured, live),), 8)
            with self.assertRaisesRegex(RuntimeError, "native block table changed"):
                pending.validate()


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_GRAPH_DEVICE") == "1" and torch.cuda.is_available(),
    "requires explicit PPU/CUDA Graph device qualification",
)
class GraphCacheDeviceTest(unittest.TestCase):
    @torch.inference_mode()
    def test_packed_verify_replay_restores_reserved_pages(self):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import VerifyGraphJournal
        from rtp_llm.models_py.modules.dsv41.ppu.graph_attention_layer import _shared_decode_attention
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import GraphPackedAttention
        from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
        from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import DEPENDENCY_ENV

        if not os.environ.get(DEPENDENCY_ENV):
            self.skipTest("requires the pinned packed FlashMLA provider")
        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(5531)
        keys = ((0, "swa_kv"), (0, "global_kv_2"))
        cache = NativeGraphCache.__new__(NativeGraphCache)
        cache.device, cache.max_seq_len, cache.native = device, 512, None
        cache.geometry = SimpleNamespace(pool_keys=lambda: keys, ratio=lambda owner: 2)
        cache.entries = {"swa_kv": 128, "global_kv_2": 64}
        cache.spans = dict.fromkeys(cache.entries, 128)
        cache._framework_tags = {key: key[1] for key in keys}
        cache.tables = {
            "swa_kv": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device, dtype=torch.int32),
            "global_kv_2": torch.tensor([[2, 3, 4, 1], [6, 7, 8, 5]], device=device, dtype=torch.int32),
        }

        def encoded(count, tag):
            kv = torch.randn(count, 512, device=device, dtype=torch.bfloat16) * 0.25
            return encode_kv(kv, pool_tag=tag)

        cache._pools = {
            (owner, tag): encoded(9 * cache.entries[tag], tag).reshape(9, -1)
            for owner, tag in keys
        }
        before = {key: cache.pool(*key).clone() for key in keys}
        rows = 8
        journal = VerifyGraphJournal(cache, rows)
        geometry = SimpleNamespace(window=128, head_dim=512, n_heads=64, sm_scale=lambda: 512**-0.5)
        packed = GraphPackedAttention(cache, geometry, journal=journal)
        sink = torch.zeros(64, device=device)
        layer = SimpleNamespace(
            layer_id=0, kv_source_layer_id=0, index_source_layer_id=0, compress_ratio=2,
            geometry=geometry, attn_sink=sink,
        )
        positions = torch.zeros(rows, device=device, dtype=torch.int64)
        valid = torch.ones(rows, device=device, dtype=torch.bool)
        prefixes = torch.zeros(2, device=device, dtype=torch.int64)
        starts = torch.tensor([0, 4], device=device)
        requests = torch.arange(rows, device=device) // 4
        metadata = SimpleNamespace(positions=positions, valid_mask=valid)
        selected = torch.arange(512, device=device, dtype=torch.int32).repeat(rows, 1)
        q = torch.randn(rows, 64, 512, device=device, dtype=torch.bfloat16)
        payloads = {key: encoded(rows, key[1]) for key in keys}

        def step():
            journal.bind(requests, starts, prefixes, valid)
            for key in keys:
                if key[1] == "swa_kv":
                    raw = torch.where(valid, positions, -1)
                else:
                    raw = torch.where(valid & ((positions & 1) == 1), positions - 1, -1)
                journal.write_rows(*key, journal.slots(*key, raw), payloads[key])
            shared = {}
            output = packed.forward(layer, q, selected, metadata, shared)
            swa, _ = packed._swa_indices(0, positions, valid, shared)
            global_slots, _ = packed._global_indices(
                0, 0, "global_kv_2", 2, selected, positions, valid, shared,
            )
            reference = _shared_decode_attention(
                q, selected, cache=journal, layer_id=0, source_layer=0, ratio=2,
                positions=positions, valid_mask=valid, window=128,
                sm_scale=geometry.sm_scale(), attn_sink=sink,
            )
            return output, reference, swa, global_slots

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            step()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output, reference, swa, global_slots = step()
        for bases, count in (((127, 129), 7), ((132, 5), 4), ((255, 253), 8)):
            prefixes.copy_(torch.tensor(bases, device=device))
            positions.copy_(prefixes[requests] + torch.arange(rows, device=device) % 4)
            valid.copy_(torch.arange(rows, device=device) < count)
            graph.replay()
            torch.testing.assert_close(output, reference, rtol=0.02, atol=0.005)
            self.assertEqual(int(torch.count_nonzero(output[~valid])), 0)
            for key in keys:
                self.assertTrue(torch.equal(cache.pool(*key), before[key]), (bases, key))
            if bases == (127, 129):
                self.assertEqual(swa[0, 0, -2:].tolist(), [254, 0])
                self.assertEqual(swa[1, 0, -2:].tolist(), [0, 1])
                self.assertEqual(int(global_slots[0, 0, 63]), 0)
                self.assertEqual(int(global_slots[4, 0, 64]), 4)
                self.assertTrue(bool((swa[7] == -1).all() and (global_slots[7] == -1).all()))

    @torch.inference_mode()
    def test_packed_scheduler_replay_shares_only_equal_prefixes(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import (
            GRAPH_SHARED_MLA_METADATA_ENV,
            GraphPackedAttention,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
        from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import DEPENDENCY_ENV

        if not os.environ.get(DEPENDENCY_ENV):
            self.skipTest("requires the pinned packed FlashMLA provider")

        torch.manual_seed(5530)
        device = torch.device("cuda", torch.cuda.current_device())
        ratios = (0, 1, 1, 2, 2)
        pages, span = 8, 128
        for rows in (1, 8):
            with self.subTest(rows=rows):
                cache = NativeGraphCache.__new__(NativeGraphCache)
                cache.device = device
                cache.entries, cache.spans, cache.tables = {}, {}, {}
                cache._framework_tags, cache._pools = {}, {}
                page_ids = torch.arange(1, pages, dtype=torch.int32, device=device)
                for owner, ratio in enumerate(ratios):
                    global_tag = f"global_kv_{ratio}"
                    tags = ("swa_kv", global_tag) if ratio else ("swa_kv",)
                    for tag in tags:
                        domain = (
                            "decoder_swa_kv" if tag == "swa_kv" and owner % 2 else tag
                        )
                        entries = span // ratio if tag == global_tag else span
                        cache._framework_tags[owner, tag] = domain
                        cache.entries[domain], cache.spans[domain] = entries, span
                        mapping = page_ids if domain != "decoder_swa_kv" else page_ids.flip(0)
                        cache.tables[domain] = mapping.repeat(rows, 1)
                        kv = torch.randn(
                            pages * entries, 512, device=device, dtype=torch.bfloat16
                        ) * 0.25
                        cache._pools[owner, tag] = encode_kv(kv, pool_tag=tag).reshape(pages, -1)
                cache.geometry = SimpleNamespace(
                    pool_keys=lambda: tuple(cache._pools), ratio=lambda owner: ratios[owner]
                )
                geometry = SimpleNamespace(
                    window=128, head_dim=512, n_heads=64, sm_scale=lambda: 512**-0.5
                )
                with patch.dict(os.environ, {GRAPH_SHARED_MLA_METADATA_ENV: "1"}):
                    packed = GraphPackedAttention(cache, geometry)
                metadata_factory = packed.metadata_factory
                schedules = []

                def record_scheduler():
                    meta, num_splits = metadata_factory()
                    schedules.append(meta)
                    return meta, num_splits

                packed.metadata_factory = record_scheduler
                sink = torch.zeros(64, device=device)
                layers = [
                    SimpleNamespace(
                        layer_id=layer, kv_source_layer_id=layer, index_source_layer_id=layer,
                        compress_ratio=ratio, geometry=geometry, attn_sink=sink,
                    )
                    for layer, ratio in enumerate(ratios)
                ]
                q = torch.randn(rows, 64, 512, device=device, dtype=torch.bfloat16)
                positions = torch.zeros(rows, device=device, dtype=torch.int64)
                valid = torch.ones(rows, device=device, dtype=torch.bool)
                metadata = SimpleNamespace(positions=positions, valid_mask=valid)
                ordinal = torch.arange(512, device=device, dtype=torch.int32).repeat(rows, 1)
                selected = (None, ordinal, ordinal.roll(3, 1), ordinal, ordinal.roll(7, 1))

                def step(share):
                    shared = {}
                    schedules.clear()
                    outputs = [
                        packed.forward(layer, q, ids, metadata, shared, share_scheduler=share)
                        for layer, ids in zip(layers, selected)
                    ]
                    return outputs, tuple(schedules)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    self.assertEqual(len(step(True)[1]), len(layers))
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    reference, reference_schedules = step(False)
                    actual, shared_schedules = step(True)
                self.assertEqual(len(reference_schedules), 5)
                self.assertEqual(len(shared_schedules), 3)

                live_rows = torch.arange(rows, device=device)
                for start, count in ((0, rows), (387, rows), (6, rows - 1), (513, rows)):
                    positions.copy_(start + 3 * live_rows)
                    valid.copy_(live_rows < count)
                    graph.replay()
                    for index, group in enumerate((0, 1, 1, 2, 2)):
                        torch.testing.assert_close(
                            actual[index], reference[index], rtol=0.005, atol=0.005
                        )
                        self.assertEqual(int(torch.count_nonzero(actual[index][~valid])), 0)
                        torch.testing.assert_close(
                            shared_schedules[group].num_splits,
                            reference_schedules[index].num_splits,
                            rtol=0, atol=0,
                        )

    def test_replay_refreshes_all_pool_formats_and_independent_tables(self):
        config = synthetic_config(max_position_embeddings=256)
        native, reference = NativePools(config), NativePools(config)
        device = torch.device("cuda", torch.cuda.current_device())
        for handle in native.handles.values():
            handle.kv_cache_base = handle.kv_cache_base.to(device)
        for entry in native.inputs.values():
            entry.kv_cache_kernel_block_id_device = entry.kv_cache_kernel_block_id_device.to(device)
            entry.kv_cache_block_id_device = entry.kv_cache_block_id_device.to(device).clone()
            entry.input_lengths = torch.ones(2, dtype=torch.int32)
            # Poisoned host lengths must never become the replay source.
            entry.sequence_lengths = torch.tensor([-123, -123], dtype=torch.int32)
            entry.sequence_lengths_plus_1_device = torch.tensor(
                [3, 11], dtype=torch.int32, device=device
            )
        for entry in reference.inputs.values():
            entry.kv_cache_block_id_device = entry.kv_cache_block_id_device.clone()
        impl = PPUDecodeGraphImpl(config, native, native.inputs, device=device)
        cache = impl.cache
        keys = cache.geometry.pool_keys()
        values, outputs = {}, {}
        for owner, tag in keys:
            width = 1024 if tag == "csa_state" else {"indexer_kv": 68, "swa_kv": 528}.get(tag, 288)
            values[(owner, tag)] = (
                torch.arange(2 * width, device=device)
                .reshape(2, width)
                .remainder(149)
                .to(cache.pool(owner, tag).dtype)
            )

        def body():
            impl.prepare_in_graph(native.inputs)
            positions = torch.where(
                impl.metadata.valid_mask, impl.metadata.positions,
                torch.full_like(impl.metadata.positions, -1),
            )
            for owner, tag in keys:
                slots = cache.slots(owner, tag, positions)
                cache.write_rows(owner, tag, slots, values[(owner, tag)])
                outputs[(owner, tag)] = cache.read_rows(owner, tag, slots)

        # Compile every format before capture. Restore pools afterwards so the
        # reference compares the exact sequence of real replay side effects.
        for _ in range(2):
            body()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        pointers = impl.metadata.positions.data_ptr(), impl.metadata.valid_mask.data_ptr()
        for handle in native.handles.values():
            handle.kv_cache_base.fill_(173)
        cases = (([2, 10], 2), ([9, 17], 1), ([16, 23], 0), ([5, 14], 2))
        for step, (positions, count) in enumerate(cases):
            for tag, entry in native.inputs.items():
                # Distinct table families make accidental physical/kernel
                # aliasing observable for SWA and unpaired CSA state.
                kernel = torch.tensor(
                    [[1 + step, 2, 3, 4, 5, 6], [7 + step, 8, 9, 10, 11, 1]],
                    dtype=entry.kv_cache_kernel_block_id_device.dtype,
                )
                physical = torch.tensor(
                    [[4 + step, 5, 6, 7, 8, 9], [8 + step, 9, 10, 11, 1, 2]],
                    dtype=entry.kv_cache_block_id_device.dtype,
                )
                entry.kv_cache_kernel_block_id_device.copy_(kernel)
                entry.kv_cache_block_id_device.copy_(physical)
                entry.sequence_lengths_plus_1_device.copy_(
                    torch.tensor(positions, dtype=torch.int32) + 1
                )
                reference.inputs[tag].kv_cache_kernel_block_id_device.copy_(kernel)
                reference.inputs[tag].kv_cache_block_id_device.copy_(physical)
            for rows in values.values():
                rows.add_(1)
            impl.prepare_cuda_graph(native.inputs, valid_request_count=count)
            graph.replay()
            torch.cuda.synchronize(device)
            self.assertEqual(
                (impl.metadata.positions.data_ptr(), impl.metadata.valid_mask.data_ptr()),
                pointers,
            )
            self.assertTrue(torch.equal(impl.metadata.positions.cpu(), torch.tensor(positions)))
            self.assertTrue(torch.equal(impl.metadata.valid_mask.cpu(), torch.arange(2) < count))
            reference_bridge = NativeCacheBridge(config, reference, reference.inputs)
            for owner, tag in keys:
                cpu_rows = values[(owner, tag)].cpu()
                expected_output = torch.zeros_like(cpu_rows)
                for row in range(count):
                    reference_bridge._copy_rows(
                        (owner, tag), row, torch.tensor([positions[row]]),
                        cpu_rows[row:row + 1],
                    )
                    expected_output[row] = cpu_rows[row]
                self.assertTrue(
                    torch.equal(outputs[(owner, tag)].cpu(), expected_output),
                    (step, owner, tag),
                )
                actual_pool = cache.pool(owner, tag).cpu()
                expected_pool = reference.handles[(owner, FRAMEWORK_POOL_TAGS[tag])].kv_cache_base
                self.assertTrue(torch.equal(actual_pool, expected_pool), (step, owner, tag))


if __name__ == "__main__":
    unittest.main()
