"""Two-GPU lifecycle tests for ProcessGroup-owned HIPGraph RCCL handles."""

import multiprocessing as mp
import sys
import traceback
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.models_py.distributed import rocm_rccl
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager


def _config(rank: int) -> ParallelismConfig:
    config = ParallelismConfig()
    config.world_rank = rank
    config.world_size = 2
    config.local_rank = rank
    config.tp_size = 2
    config.dp_size = 1
    return config


def _comm_config() -> NcclCommConfig:
    return NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=1,
        dp_tp_nccl_port=2,
        ffn_tp_nccl_port=3,
    )


def _fresh_world_worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    config = _config(rank)
    ct.init_distributed_environment(
        config,
        _comm_config(),
        port,
        graph_required=True,
        timeout=60,
    )
    descriptor = rocm_rccl._graph_comm_manager.require_ready(
        ct._get_group_record(ct.Group.TP), rank
    )
    assert descriptor.handle != 0
    assert descriptor.source_group.process_group is torch.distributed.group.WORLD
    assert descriptor.generation == ct._distributed_generation

    token = rocm_rccl._graph_comm_manager.acquire_graph_owner(rank + 100)
    graphs = []
    try:
        rocm_rccl.begin_capture_planning(token.token_id, token.generation)
        trt_source = torch.full((2, 1024), float(rank + 1), device=f"cuda:{rank}")
        raw_source = torch.full((2, 3), float(rank + 1), device=f"cuda:{rank}")
        # Plan both occurrences of each all-gather signature. Repeated calls in
        # one graph must receive independent stable output buffers.
        for source in (trt_source, raw_source):
            ct.all_gather(source, ct.Group.TP)
            ct.all_gather(source, ct.Group.TP)
        rocm_rccl.prepare_capture_arena(token.token_id, token.generation)

        from rtp_llm.models_py.modules.base.rocm import trt_allreduce

        trt_ready = trt_allreduce.is_trt_allreduce_ready()
        assert trt_ready, "supported TRT shape did not initialize its IPC workspace"

        def capture(source):
            stream = torch.cuda.Stream(device=rank)
            work = torch.empty_like(source)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                stream.synchronize()
                rocm_rccl.enter_graph_capture_mode(token.token_id, token.generation)
                try:
                    # C++ CudaGraphCaptureGuard owns the native flag in
                    # production. This Python-only harness supplies the same
                    # signal while exercising the real collective_torch route.
                    with patch.object(
                        rocm_rccl, "_is_hipgraph_capture_active", return_value=True
                    ), torch.cuda.graph(graph, stream=stream):
                        work.copy_(source)
                        reduced = ct.all_reduce(work, ct.Group.TP)
                        first_gathered = ct.all_gather(reduced, ct.Group.TP)
                        second_input = reduced + 5.0
                        gathered = ct.all_gather(second_input, ct.Group.TP)
                finally:
                    rocm_rccl.exit_graph_capture_mode(token.token_id, token.generation)
            stream.synchronize()
            graphs.append(graph)
            return graph, stream, reduced, first_gathered, gathered

        trt_graph, trt_stream, trt_reduced, trt_first, trt_gathered = capture(
            trt_source
        )
        assert trt_allreduce.has_pending_capture(), "supported shape did not use TRT"
        rocm_rccl.finish_hipgraph_capture_session(token.token_id, token.generation)
        assert not trt_allreduce.has_pending_capture()

        raw_graph, raw_stream, raw_reduced, raw_first, raw_gathered = capture(
            raw_source
        )
        assert (
            not trt_allreduce.has_pending_capture()
        ), "raw fallback unexpectedly used TRT"
        for _ in range(60):
            trt_graph.replay()
            trt_stream.synchronize()
            raw_graph.replay()
            raw_stream.synchronize()
        torch.testing.assert_close(trt_reduced, torch.full_like(trt_source, 3.0))
        assert trt_first.data_ptr() != trt_gathered.data_ptr()
        torch.testing.assert_close(
            trt_first, torch.full((4, 1024), 3.0, device=f"cuda:{rank}")
        )
        torch.testing.assert_close(
            trt_gathered, torch.full((4, 1024), 8.0, device=f"cuda:{rank}")
        )
        torch.testing.assert_close(raw_reduced, torch.full_like(raw_source, 3.0))
        assert raw_first.data_ptr() != raw_gathered.data_ptr()
        torch.testing.assert_close(
            raw_first, torch.full((4, 3), 3.0, device=f"cuda:{rank}")
        )
        torch.testing.assert_close(
            raw_gathered, torch.full((4, 3), 8.0, device=f"cuda:{rank}")
        )

        try:
            ct.destroy_distributed_environment()
        except RuntimeError as exc:
            assert "live graph owners" in str(exc)
        else:
            raise AssertionError("live graph owner did not block teardown")
    finally:
        for graph in graphs:
            graph.reset()
        if graphs:
            torch.cuda.synchronize(rank)
        rocm_rccl.release_graph_owner(token.token_id, token.generation)
    ct.destroy_distributed_environment()
    assert not torch.distributed.is_initialized()


def _external_world_worker(rank: int, ports) -> None:
    torch.cuda.set_device(rank)
    config = _config(rank)
    rocm_rccl.prepare_distributed_environment(config, graph_required=True)
    torch.distributed.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{ports[0]}",
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
    )
    external_world = torch.distributed.group.WORLD
    try:
        ct.init_distributed_environment(
            config,
            _comm_config(),
            ports[1],
            graph_required=True,
            timeout=60,
        )
        tp = ct._get_group_record(ct.Group.TP)
        assert tp.process_group is not external_world
        assert tp.owned_by_rtp
        assert not ct._get_group_record(ct.Group.DP_AND_TP).owned_by_rtp
        descriptor = rocm_rccl._graph_comm_manager.require_ready(tp, rank)
        assert descriptor.source_group == tp

        ct.destroy_distributed_environment()
        assert torch.distributed.is_initialized()
        value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
        torch.distributed.all_reduce(value)
        assert value.item() == 3.0
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _extraction_failure_worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    config = _config(rank)
    original = rocm_rccl._graph_comm_manager._accessor.extract
    if rank == 1:
        rocm_rccl._graph_comm_manager._accessor.extract = lambda record, buffers=None: (
            _ for _ in ()
        ).throw(RuntimeError("injected zero pointer"))
    try:
        try:
            ct.init_distributed_environment(
                config,
                _comm_config(),
                port,
                timeout=60,
                graph_required=True,
            )
        except RuntimeError as exc:
            assert "Uniform RCCL" in str(exc)
            assert "injected zero pointer" in str(exc)
        else:
            raise AssertionError("one-rank extraction failure was published")
        assert rocm_rccl._graph_comm_manager.descriptor is None
        assert rocm_rccl._graph_comm_manager.state == rocm_rccl.ManagerState.FAILED
    finally:
        rocm_rccl._graph_comm_manager._accessor.extract = original
        if torch.distributed.is_initialized():
            ct.destroy_distributed_environment()


def _worker_entry(worker, rank, ports) -> None:
    try:
        worker(rank, ports)
    except BaseException:
        print(f"rank {rank}:\n{traceback.format_exc()}", file=sys.stderr)
        raise


class HipGraphProcessGroupIntegrationTest(unittest.TestCase):
    def setUp(self):
        if not rocm_rccl.is_rocm_runtime():
            self.skipTest("requires a PyTorch ROCm runtime")
        if torch.cuda.device_count() < 2:
            self.skipTest(
                f"requires two visible ROCm GPUs; found {torch.cuda.device_count()}"
            )
        self.ports = PortManager()
        mp.set_start_method("spawn", force=True)

    def _run(self, worker):
        port_count = 2 if worker is _external_world_worker else 1
        ports, locks = self.ports.get_consecutive_ports(port_count)
        worker_ports = ports if port_count == 2 else ports[0]
        processes = []
        try:
            processes = [
                mp.Process(
                    target=_worker_entry,
                    args=(worker, rank, worker_ports),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=120)
                self.assertIsNotNone(process.exitcode, "worker timed out after 120s")
                if process.exitcode != 0:
                    self.fail(f"worker pid={process.pid} exited {process.exitcode}")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_fresh_world_materialize_lease_and_teardown(self):
        self._run(_fresh_world_worker)

    def test_external_world_uses_owned_tp_group_and_survives(self):
        self._run(_external_world_worker)

    def test_one_rank_extraction_failure_is_uniform(self):
        self._run(_extraction_failure_worker)


if __name__ == "__main__":
    unittest.main()
