#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real CUDA regression; small local allocations, not multi-host evidence.

ENGRAM_RUN_CUDA_TESTS=1 python3 -m unittest \
  tools.engram_sidecar.native.test_sharded_lifecycle_cuda
Requires the prebuilt native module on PYTHONPATH. Never builds/restarts services.
"""
import gc
import os
import unittest


@unittest.skipUnless(os.environ.get('ENGRAM_RUN_CUDA_TESTS') == '1', 'explicit CUDA test opt-in required')
class ShardedCudaLifecycleTest(unittest.TestCase):
    def test_real_aliases_graph_and_repeated_consumers(self):
        import torch
        from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native
        from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
        from rtp_llm.models_py.modules.dsv41.engram_shard_storage import CudaDriverOps, ShardedLayerMapping
        from rtp_llm.models_py.modules.dsv41.engram_storage import StorageBusyError

        device = int(os.environ.get('ENGRAM_TEST_DEVICE', '0'))
        numa = int(os.environ.get('ENGRAM_TEST_NUMA', '0'))
        torch.cuda.set_device(device)
        native = _load_native()
        baseline = native.resource_counters()
        granularity = native.host_numa_granularity(numa, device)

        class CountingOps(CudaDriverOps):
            imports = releases = 0
            def import_handle(self, raw):
                result = super().import_handle(raw)
                self.imports += 1
                return result
            def release(self, handle):
                super().release(handle)
                self.releases += 1

        for p in (2, 3):
            plan = plan_shards(p * (granularity // 8) - 7, 256, p, granularity=granularity)
            owners, weights, scales = [], [], []
            try:
                for shard in range(p):
                    for size, value, handles in ((plan.weight_bytes, shard + 1, weights),
                                                 (plan.scale_bytes, shard + 101, scales)):
                        owner = native.Owner(size, numa, device)
                        owners.append(owner)
                        for offset in range(0, size, 8 << 20):
                            owner.load(bytes([value]) * min(8 << 20, size - offset), offset)
                        handles.append(owner.export_handle())
                ops = [CountingOps(device), CountingOps(device)]
                mappings = [ShardedLayerMapping(plan, weights, scales, device, ops=op).attach() for op in ops]
                for mapping in mappings:
                    view = mapping.weight_view()
                    self.assertEqual(view.data_ptr(), mapping._weight_ptr)
                    self.assertEqual(view.device.index, device)
                    alias = view[1:2]
                    del view
                    gc.collect()
                    self.assertEqual(mapping.info()['active_views'], 1)
                    with self.assertRaises(StorageBusyError):
                        mapping.close()
                    del alias
                    gc.collect()
                    self.assertEqual(mapping.info()['active_views'], 0)

                first, second = mappings
                first.close()
                # Closing one independent mapping must not affect the other.
                w, s = second.weight_view(), second.scale_view()
                rows = torch.tensor([plan.row_range(i)[0] for i in range(p)], device=device)
                expected_w = list(range(1, p + 1))
                expected_s = list(range(101, p + 101))
                self.assertEqual(w.index_select(0, rows)[:, 0].cpu().tolist(), expected_w)
                self.assertEqual(s.index_select(0, rows)[:, 0].cpu().tolist(), expected_s)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    output = w.index_select(0, rows)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = w.index_select(0, rows)
                for _ in range(3):
                    graph.replay()
                    self.assertEqual(output[:, 0].cpu().tolist(), expected_w)
                torch.cuda.synchronize(device)
                graph.reset()
                del graph, w, s, output, rows
                gc.collect()
                second.close()
                self.assertTrue(all(op.imports == op.releases == 2 * p for op in ops))
            finally:
                # Failures must terminate this isolated probe rather than force
                # releasing owners while an unknown Graph/view may still exist.
                if not all(m.state == 'CLOSED' for m in locals().get('mappings', [])):
                    raise RuntimeError('probe mappings remain live; terminate probe process')
                for owner in reversed(owners):
                    owner.close()
            self.assertEqual(native.resource_counters(), baseline)


if __name__ == '__main__':
    unittest.main(verbosity=2)
