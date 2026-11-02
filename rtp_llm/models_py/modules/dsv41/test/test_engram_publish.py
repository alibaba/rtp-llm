# SPDX-License-Identifier: Apache-2.0
import unittest

from rtp_llm.models_py.modules.dsv41 import engram_publish as p
from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards


class Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = tuple(shape)


class ShardLayoutTest(unittest.TestCase):
    def test_weight_and_scale_share_the_row_partition(self):
        rows, dim, shards, granularity = 100000, 64, 2, 4096
        plan = plan_shards(rows, dim, shards, granularity=granularity)
        weight = Tensor('layers.1.engram.embed.weight', (rows, dim))
        scale = Tensor('layers.1.engram.embed.scale', (rows, dim // plan.block))
        for shard in range(shards):
            start, end, weight_bytes = p.shard_layout(weight, (rows, dim), shard, shards, granularity)
            self.assertEqual((start, end), plan.row_range(shard))
            self.assertEqual(weight_bytes, plan.weight_bytes)
            start, end, scale_bytes = p.shard_layout(scale, (rows, dim), shard, shards, granularity)
            self.assertEqual(scale_bytes, plan.scale_bytes)

    def test_mismatched_shapes_are_rejected(self):
        weight = Tensor('layers.1.engram.embed.weight', (100, 64))
        with self.assertRaises(ValueError):
            p.shard_layout(weight, (200, 64), 0, 2, 4096)

    def test_unknown_role_is_rejected(self):
        tensor = Tensor('layers.1.engram.embed.bias', (100, 64))
        with self.assertRaises(ValueError):
            p.shard_layout(tensor, (100, 64), 0, 2, 4096)

    def test_release_of_an_empty_generation_is_a_noop(self):
        self.assertEqual(p.release('missing-generation'), [])


class PlacementMapTest(unittest.TestCase):
    def test_absent_map_uses_the_scalar_default(self):
        self.assertEqual(p._placement_map(None, [1, 2], 3, 'device'), {1: 3, 2: 3})

    def test_partial_or_non_integer_maps_are_rejected(self):
        for raw in ({1: 0}, {1: 0, 2: 0, 3: 1}, {1: 0, 2: True}, {1: 0, 2: -1}, 'cuda:1'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                p._placement_map(raw, [1, 2], 0, 'NUMA')

    def test_explicit_map_is_honoured_per_layer(self):
        self.assertEqual(p._placement_map({1: 2, 2: 3}, [1, 2], 0, 'device'), {1: 2, 2: 3})


class LoadRowsBoundsTest(unittest.TestCase):
    def test_row_range_and_chunk_bounds_are_checked_before_any_read(self):
        tensor = Tensor('layers.1.engram.embed.weight', (100, 64))
        for start, end, columns, chunk in [(1, 1, 64, 4096), (0, 101, 64, 4096),
                                           (0, 10, 32, 4096), (0, 10, 64, 0),
                                           (0, 10, 64, (64 << 20) + 1)]:
            with self.subTest(case=(start, end, columns, chunk)), self.assertRaises(ValueError):
                p.load_rows(None, None, tensor, None, start, end, columns, chunk_bytes=chunk)

class PublishFailureTest(unittest.TestCase):
    def test_chunk_deadline_checked_after_read_before_native_write(self):
        import io
        import sys
        import time
        from contextlib import contextmanager
        from types import SimpleNamespace as NS
        from unittest.mock import patch

        @contextmanager
        def opened(*args):
            yield io.BytesIO(bytes(512)), 0
        now = [0.0]
        def read(source, count):
            now[0] += 2
            return source.read(count)
        checkpoint = NS(_cache_page_size=lambda _: None,
                        _discard_consumed_pages=lambda *args: None,
                        _open_tensor_source=opened, _read_exact=read)
        writes = []
        tensor = NS(shape=(8, 64), data_range=(0, 512))
        with patch.dict(sys.modules, {'rtp_llm.models_py.modules.dsv41.engram_checkpoint': checkpoint}), \
             patch.object(p.time, 'monotonic', lambda: now[0]):
            with self.assertRaisesRegex(ValueError, 'LOCAL_SHARD_PUBLISH_TIMEOUT'):
                p.load_rows(NS(load=lambda *args: writes.append(args)), None, tensor,
                            None, 0, 8, 64, chunk_bytes=64, deadline=1)
        self.assertEqual(writes, [])

    def test_local_budget_preflight_and_cleanup_retry(self):
        import json
        import sys
        import tempfile
        from contextlib import nullcontext
        from pathlib import Path
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from rtp_llm.models_py.modules.dsv41 import engram_bootstrap as b

        made = []
        class Owner:
            def __init__(self, *args): self.fail_close = True; made.append(self)
            def close(self):
                if self.fail_close: raise RuntimeError('injected close failure')
        native = NS(Owner=Owner, max_load_chunk_bytes=4096, max_allocation_bytes=1 << 30,
                    host_numa_granularity=lambda *args: 4096)
        layout = NS(layer_ids=(1,), num_embeddings=(100000,), head_dim=64)
        layers = [NS(layer_id=1, tensors=[NS(name='layers.1.engram.embed.' + role,
                   shape=(100000, width), relative_path='weights')
                  for role, width in [('weight', 64), ('scale', 2)]])]
        checkpoint = NS(inspect_checkpoint=lambda *args, **kwargs: NS(layers=layers, files=[NS(relative_path='weights')]),
                        _cache_page_size=lambda _: None,
                        _discard_consumed_pages=None, _open_tensor_source=None)
        model = NS(EngramLayout=lambda _: layout, NgramHashState=lambda *args: None)
        modules = {'torch': NS(cuda=NS(device=lambda _: nullcontext())),
                   'rtp_llm.models_py.modules.dsv41.engram': model,
                   'rtp_llm.models_py.modules.dsv41.engram_checkpoint': checkpoint}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'config.json').write_text('{}')
            with patch.dict(sys.modules, modules), patch.object(b, '_load_native', lambda: native), \
                 patch.object(b, 'hash_state_digest', lambda _: 'sha256:' + '0' * 64):
                with self.assertRaisesRegex(ValueError, 'host budget'):
                    p.publish_shard(directory, 'rev', 'host', 'budget-test', 0, 2, budget_bytes=4096)
                self.assertEqual(made, [])  # fail before even the first allocation
                with patch.object(p, 'load_rows', side_effect=OSError('injected I/O failure')):
                    with self.assertRaisesRegex(RuntimeError, 'CLEANUP_INCOMPLETE'):
                        p.publish_shard(directory, 'rev', 'host', 'cleanup-test', 0, 2)
                self.assertEqual(len(p._KEEPERS['cleanup-test']), 1)
                self.assertEqual(len(p.release('cleanup-test')), 1)
                made[0].fail_close = False
                self.assertEqual(p.release('cleanup-test'), [])
                self.assertNotIn('cleanup-test', p._KEEPERS)


if __name__ == '__main__': unittest.main(verbosity=2)
