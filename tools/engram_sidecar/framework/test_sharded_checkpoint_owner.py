# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for real checkpoint weight/scale row partitioning."""
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.engram_sidecar.framework.sharded_checkpoint_owner import load_rows, tensor_shard_layout, layer_numa_placement
from rtp_llm.models_py.modules.dsv41.engram_checkpoint import CheckpointError, TensorPlan, _header
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards


class ShardedCheckpointLayoutTest(unittest.TestCase):
    def test_placement_is_complete_and_never_silently_falls_back(self):
        self.assertEqual(layer_numa_placement(None, [1, 14], 0), {1: 0, 14: 0})
        self.assertEqual(layer_numa_placement('{"1":0,"14":1}', [1, 14], 0), {1: 0, 14: 1})
        for value in ('{}', '{"1":0}', '{"1":0,"14":1,"2":0}',
                      '{"1":true,"14":1}', '{"1":0,"14":-1}',
                      '{"1":0,"14":1.0}', '{"1":0,"1":1,"14":0}', '[]'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                layer_numa_placement(value, [1, 14], 0)

    def test_fp8_scale_uses_weight_partition_on_every_owner(self):
        rows, dim, granularity = 384006168, 256, 2 << 20
        tensors = [SimpleNamespace(name="layers.1.engram.embed." + role, shape=(rows, width))
                   for role, width in (("weight", dim), ("scale", dim // 32))]
        for count in (2, 4):
            plan = plan_shards(rows, dim, count, granularity=granularity)
            previous_end = 0
            for shard in range(count):
                weight = tensor_shard_layout(tensors[0], (rows, dim), shard, count, granularity)
                scale = tensor_shard_layout(tensors[1], (rows, dim), shard, count, granularity)
                self.assertEqual(weight[:2], scale[:2])
                self.assertEqual(weight[0], previous_end)
                previous_end = weight[1]
                self.assertEqual(weight[2], plan.weight_bytes)
                self.assertEqual(scale[2], plan.scale_bytes)
                self.assertEqual(weight[2], scale[2] * 32)
                self.assertEqual(scale[2] % granularity, 0)
            self.assertEqual(previous_end, rows)

    def test_mismatched_scale_rows_or_width_rejected(self):
        for shape in ((99, 8), (100, 7)):
            tensor = SimpleNamespace(name="layers.1.engram.embed.scale", shape=shape)
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                tensor_shard_layout(tensor, (100, 256), 0, 2, 4096)


class ShardedCheckpointStreamingTest(unittest.TestCase):
    def fixture(self, root, columns):
        prefix, payload, suffix = b"prefix!", bytes(range(12 * columns)), b"suffix"
        header = json.dumps({"table": {"dtype": "U8", "shape": [12, columns],
                                     "data_offsets": [len(prefix), len(prefix) + len(payload)]}}).encode()
        path = root / "table.safetensors"
        path.write_bytes(len(header).to_bytes(8, "little") + header + prefix + payload + suffix)
        _, _, snapshot = _header(path, path.name)
        tensor = TensorPlan("table", path.name, (12, columns), "U8",
                            (len(prefix), len(prefix) + len(payload)), 0)
        return path, tensor, snapshot, payload

    def test_each_shard_streams_exact_bytes_with_bounded_chunks(self):
        for columns in (1, 8):
            with self.subTest(columns=columns), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                _, tensor, snapshot, payload = self.fixture(root, columns)
                for start, end in ((0, 7), (7, 12)):
                    calls = []
                    owner = SimpleNamespace(load=lambda data, offset: calls.append((data, offset)))
                    digest = load_rows(owner, root, tensor, snapshot, start, end, columns,
                                       chunk_bytes=columns * 3)
                    expected = payload[start * columns:end * columns]
                    self.assertEqual(b"".join(data for data, _ in calls), expected)
                    self.assertEqual(digest, "sha256:" + hashlib.sha256(expected).hexdigest())
                    cursor = 0
                    for data, offset in calls:
                        self.assertEqual(offset, cursor)
                        self.assertLessEqual(len(data), columns * 3)
                        cursor += len(data)

    def test_changed_source_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path, tensor, snapshot, _ = self.fixture(root, 8)
            path.write_bytes(path.read_bytes()[:-1])
            calls = []
            with self.assertRaisesRegex(CheckpointError, "CHECKPOINT_CHANGED"):
                load_rows(SimpleNamespace(load=lambda *args: calls.append(args)), root,
                          tensor, snapshot, 0, 7, 8)
            self.assertEqual(calls, [])

    def test_invalid_ranges_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            _, tensor, snapshot, _ = self.fixture(root, 8)
            for start, end, columns in ((-1, 3, 8), (0, 13, 8), (3, 3, 8), (0, 3, 7)):
                with self.subTest(start=start, end=end, columns=columns), self.assertRaises(ValueError):
                    load_rows(None, root, tensor, snapshot, start, end, columns)


if __name__ == "__main__":
    unittest.main()
