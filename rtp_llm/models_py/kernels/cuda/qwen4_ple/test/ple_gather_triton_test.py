import gc
import os
import subprocess
import sys
import unittest
import uuid
import weakref
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpNGramEmbedding


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PLEGatherTritonTest(unittest.TestCase):
    def _module(self):
        device = torch.device("cuda")
        shards = [
            torch.arange(5 * 160, device=device, dtype=torch.float32)
            .reshape(5, 160)
            .add(i * 1000)
            .to(torch.bfloat16)
            for i in range(2)
        ]
        module = Qwen4ExpNGramEmbedding(
            shards,
            torch.tensor([8, 8], dtype=torch.int64),
            torch.tensor([0, 8], dtype=torch.int64),
            torch.tensor([3, 5], dtype=torch.int64),
            ngram_size=2,
            eos_token_id=0,
            shard_indices=[1, 3],
            total_shards=4,
            distributed_reduce=True,
        )
        return module

    def test_local_and_remote_shards_match_boolean_indexing(self):
        module = self._module()
        ids = torch.tensor([0, 5, 7, 9, 10, 16, 19], device="cuda").reshape(1, 7, 1)
        expected = module._gather_torch(ids)
        actual = module._gather_local(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_graph_replay_uses_updated_ids(self):
        module = self._module()
        ids = torch.tensor([5, 0, 15], device="cuda", dtype=torch.int64)
        module._gather_local(ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module._gather_local(ids)
        ids.copy_(torch.tensor([16, 9, 2], device="cuda"))
        graph.replay()
        expected = module._gather_torch(ids)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_cpu_fp8_prefill_tiling_preserves_all_bit_patterns(self):
        from rtp_llm.models_py.kernels.cuda.qwen4_ple.gather import (
            gather_local,
            prepare_cpu_uva_shards,
        )

        raw = (
            torch.arange(256, dtype=torch.int32)
            .to(torch.uint8)
            .repeat(5)
            .reshape(8, 160)
        )
        shards, pointers = prepare_cpu_uva_shards(
            [raw.view(torch.float8_e4m3fn)],
            [0],
            2,
            torch.device("cuda", torch.cuda.current_device()),
        )
        for n in (17, 65536, 131072, 196608):
            ids = torch.arange(n, device="cuda", dtype=torch.int64) % 20 - 2
            for value in (0.00003125, 0.119713, 8.0):
                scale = torch.tensor([value], device="cuda")
                expected = torch.zeros((n, 160), device="cuda", dtype=torch.bfloat16)
                valid = (ids >= 0) & (ids < 8)
                expected[valid] = (
                    raw.to(device="cuda")
                    .view(torch.float8_e4m3fn)
                    .to(torch.bfloat16)
                    .index_select(0, ids[valid])
                    .mul(scale.to(torch.bfloat16))
                )
                actual = gather_local(ids, shards, (0,), 2, pointers, scale=scale)
                self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(expected)))
                finite = ~torch.isnan(expected)
                self.assertTrue(
                    torch.equal(
                        actual[finite].view(torch.int16),
                        expected[finite].view(torch.int16),
                    )
                )

    def test_cpu_fp8_prefill_tiling_graph_refreshes_ids_and_scale(self):
        from rtp_llm.models_py.kernels.cuda.qwen4_ple.gather import (
            gather_local,
            prepare_cpu_uva_shards,
        )

        shard = (
            torch.arange(8 * 160, dtype=torch.float32)
            .reshape(8, 160)
            .remainder(63)
            .to(torch.float8_e4m3fn)
        )
        shards, pointers = prepare_cpu_uva_shards(
            [shard], [0], 2, torch.device("cuda", torch.cuda.current_device())
        )
        ids = torch.arange(65536, device="cuda", dtype=torch.int64) % 20 - 2
        scale = torch.tensor([0.119713], device="cuda")
        for _ in range(3):
            gather_local(ids, shards, (0,), 2, pointers, scale=scale)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gather_local(ids, shards, (0,), 2, pointers, scale=scale)
        for generation in range(8):
            ids.copy_(
                (torch.arange(65536, device="cuda", dtype=torch.int64) + generation)
                % 20
                - 2
            )
            scale.fill_((generation + 1) * 0.119713)
            graph.replay()
            expected = gather_local(ids, shards, (0,), 2, pointers, scale=scale)
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )
        graph.reset()


class _HostWeights:
    def __init__(self):
        self.weights = [
            {"table": torch.arange(512, dtype=torch.float32).reshape(16, 32)}
        ]
        self.global_weights = {}


def _resident_ple(cold):
    from rtp_llm.models_py.kernels.cuda.qwen4_ple.gather import prepare_cpu_uva_shards
    from rtp_llm.resident_weights import ResidentWeights

    manager = ResidentWeights("target", 0, {"model_type": "ple-test"}, False)
    weights = _HostWeights()
    if cold:
        manager.move_to_vmm(weights)
    else:
        loaded = manager.attach()
        assert loaded is not None
        weights.weights = [{}]
        for item in loaded:
            weights.weights[item.scope][item.name] = item.tensor
        manager.bind(weights)
    shard = weights.weights[0]["table"]
    assert shard.is_pinned() == (
        not cold
    )  # hot re-registers the published access requirement
    owners, pointers = prepare_cpu_uva_shards([shard], [0], 1, torch.device("cuda", 0))
    assert owners[0] is shard and owners[0].data_ptr() == shard.data_ptr()
    assert shard.is_pinned()
    manager.finalize()
    return manager, weights, owners, pointers


def _ple_shared_process(mode):
    from rtp_llm.models_py.kernels.cuda.qwen4_ple.gather import gather_local

    manager, weights, owners, pointers = _resident_ple(mode == "ple_cold")
    ids = torch.tensor([3, 7, 15], dtype=torch.int64, device="cuda")
    expected = weights.weights[0]["table"].index_select(0, ids.cpu()).to("cuda")
    actual = gather_local(ids, owners, (0,), 1, pointers)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_local(ids, owners, (0,), 1, pointers)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph.reset()
    if mode == "ple_gpu_write":

        class CudaView:
            __cuda_array_interface__ = {
                "shape": (512,),
                "typestr": "<f4",
                "data": (pointers[0].item(), False),
                "version": 3,
            }

        torch.as_tensor(CudaView(), device="cuda").add_(1)
        torch.cuda.synchronize()  # expected illegal access, in this isolated process only


class ResidentPLEGatherTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.resident_weights import store

        self.environment = patch.dict(
            os.environ, RTP_LLM_RESIDENT_STORE=f"resident-ple-test.{uuid.uuid4().hex}"
        )
        self.environment.start()
        self.store = store
        self.name = store.default_store_name()
        store.ensure_running(self.name, owner_pid=os.getpid())

    def tearDown(self):
        self.store.stop_store(self.name)
        self.environment.stop()

    def test_cold_exit_hot_graph_and_gpu_readonly_fault(self):
        for mode in ("ple_cold", "ple_hot", "ple_gpu_write", "ple_hot"):
            child = subprocess.run(
                [sys.executable, __file__, mode], capture_output=True, text=True
            )
            if mode == "ple_gpu_write":
                self.assertNotEqual(child.returncode, 0)
                self.assertIn("illegal memory access", child.stdout + child.stderr)
            else:
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)

    def test_async_reader_finishes_before_last_host_view_is_unmapped(self):
        from rtp_llm.models_py.kernels.cuda.qwen4_ple.gather import gather_local

        manager, weights, owners, pointers = _resident_ple(True)
        shard = weights.weights[0]["table"]
        ref = weakref.ref(manager.allocation_of(shard))
        ids = torch.tensor([3, 7, 15], dtype=torch.int64, device="cuda")
        expected = shard.index_select(0, ids.cpu()).to("cuda")
        gather_local(
            ids, owners, (0,), 1, pointers
        )  # compile outside the queued interval
        torch.cuda.synchronize()
        torch.cuda._sleep(20_000_000)
        actual = gather_local(ids, owners, (0,), 1, pointers)
        del (
            owners,
            weights,
            shard,
        )  # no pre-delete sync: the buffer destructor must wait
        gc.collect()
        self.assertIsNone(ref())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_release_inside_graph_capture_is_deferred(self):
        from rtp_llm.models_py.kernels.cuda.qwen4_ple.gather import gather_local
        from rtp_llm.resident_weights import cpu

        manager, weights, owners, pointers = _resident_ple(True)
        ids = torch.tensor([3, 7, 15], dtype=torch.int64, device="cuda")
        expected = owners[0].index_select(0, ids.cpu()).to("cuda")
        gather_local(ids, owners, (0,), 1, pointers)
        torch.cuda.synchronize()
        before = len(cpu._deferred_releases)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gather_local(ids, owners, (0,), 1, pointers)
            del owners, weights
            gc.collect()
        self.assertEqual(len(cpu._deferred_releases), before + 1)
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        graph.reset()
        # A later registered release outside capture drains deferred mappings, as for VMM.
        manager2, weights2, owners2, pointers2 = _resident_ple(True)
        del owners2, weights2
        gc.collect()
        self.assertEqual(len(cpu._deferred_releases), 0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("ple_"):
        _ple_shared_process(sys.argv[1])
    else:
        unittest.main()
