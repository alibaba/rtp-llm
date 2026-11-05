"""Native packed layout and changing-input contracts for PPU decode Graph."""

import os
import unittest

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
        positions = torch.tensor([[0, 2, 6, 8], [8, 10, 14, 16]])
        for owner, tag in cache.geometry.pool_keys():
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
    "requires explicit PPU/CUDA Graph device qualification",
)
class GraphCacheDeviceTest(unittest.TestCase):
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
