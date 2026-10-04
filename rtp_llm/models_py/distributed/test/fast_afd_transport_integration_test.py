"""Three/four-GPU regressions for Gloo control and NCCL FastAFD payloads.

No model weights or server are needed. A small CUDA expert consumes all three
payload tensors, allowing transport correctness and lifecycle failures to be
checked independently of a particular MoE backend.
"""

import multiprocessing as mp
import time
import unittest
from datetime import timedelta
from multiprocessing.connection import wait

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed.fast_afd import FastAFDClient, FastAFDExpertService
from rtp_llm.ops import NcclCommConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager

_WORLD_SIZE = 3
_EXPERT_RANK = 2
_MULTI_EXPERT_RANKS = (2, 3)
_HIDDEN_SIZE = 64
_TOP_K = 2
_PROCESS_TIMEOUT_SECONDS = 180

# Each pair contains the requests for AG0 and AG1 in one engine step.
_STEPS = (
    ((), ()),
    (((0, 3), (1, 1)), ()),
    (((0, 1), (1, 3)), ((0, 5), (1, 2))),
    (((0, 0), (1, 9)), ((1, 4),)),
    (((0, 128), (1, 17), (0, 2)), ((1, 7), (0, 4))),
    ((), ()),
)
_IDLE_FLAGS = (True, False, False, False, False, True)


def _inputs(rank, step, tokens, dtype, device):
    # Strided inputs also exercise the client's contiguous payload copies.
    hidden = torch.arange(
        tokens * _HIDDEN_SIZE * 2, dtype=torch.float32, device=device
    ).reshape(tokens, _HIDDEN_SIZE * 2)
    hidden = (hidden.remainder(64) / 4 + rank * 2 + step).to(dtype)[:, ::2]
    ids_dtype = torch.int32 if rank == 0 else torch.int64
    ids = torch.arange(tokens * _TOP_K * 2, dtype=ids_dtype, device=device)
    ids = ((ids + rank) % 4).reshape(tokens, _TOP_K * 2)[:, ::2]
    weights = torch.tensor(
        [[0.25, 0.0, 0.75, 0.0]], device=device, dtype=torch.float32
    ).expand(tokens, -1)[:, ::2]
    return hidden, ids, weights


def _expert_result(hidden, ids, weights, layer):
    routed = ((ids.float() + 1) * weights).sum(dim=1, keepdim=True)
    return hidden * 2 + routed.to(hidden.dtype) + layer * 8


class _CudaExpert(torch.nn.Module):
    topk_ids_dtype = torch.int64
    includes_shared_expert = False
    expert_num = 4

    def __init__(self, layer, fail=False):
        super().__init__()
        self.layer = layer
        self.fail = fail
        self.batch_sizes = []

    def forward(self, hidden_states, topk_weights, topk_ids, activation):
        assert activation == "SiGLU"
        assert hidden_states.is_cuda and topk_ids.is_cuda and topk_weights.is_cuda
        assert topk_ids.dtype == self.topk_ids_dtype
        assert topk_weights.dtype == torch.float32
        self.batch_sizes.append(hidden_states.shape[0])
        if self.fail:
            # A recoverable Python exception exercises protocol propagation
            # without deliberately corrupting a CUDA context or communicator.
            raise RuntimeError("injected expert failure")
        return _expert_result(hidden_states, topk_ids, topk_weights, self.layer)


def _sharded_inputs(rank, step, tokens, dtype, device):
    hidden, ids, weights = _inputs(rank, step, tokens, dtype, device)
    # Keep all arithmetic exactly representable in both activation dtypes,
    # including reduction of the two independently rounded shard outputs.
    hidden.remainder_(4)
    if step % 3:
        # Include batches with no tokens routed to one of the expert shards.
        ids[:, 0] = 0 if step % 3 == 1 else 2
        ids[:, 1] = 1 if step % 3 == 1 else 3
    return hidden, ids, weights


def _sharded_expert_result(hidden, ids, weights, layer, expert_range=None):
    routed_weights = weights
    if expert_range is not None:
        start, end = expert_range
        routed_weights = weights * ((ids >= start) & (ids < end))
    expert_outputs = (
        hidden.float().unsqueeze(1) * 4
        + (ids.float() + 1).unsqueeze(-1) * 4
        + layer * 8
    )
    return (expert_outputs * routed_weights.unsqueeze(-1)).sum(1).to(hidden.dtype)


class _ShardedCudaExpert(_CudaExpert):
    def __init__(self, layer, rank, fail=False):
        super().__init__(layer, fail=fail)
        self.expert_range = (2 * (rank - 2), 2 * (rank - 1))
        self.seen_ids = set()
        self.empty_shard_batches = 0

    def forward(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        activation,
        skip_tp_allreduce=False,
    ):
        assert skip_tp_allreduce, "EG service must reduce only the expert group"
        assert activation == "SiGLU"
        assert hidden_states.is_cuda and topk_ids.is_cuda and topk_weights.is_cuda
        assert topk_ids.dtype == self.topk_ids_dtype
        assert topk_weights.dtype == torch.float32
        self.batch_sizes.append(hidden_states.shape[0])
        # Each EG receives global IDs, including IDs owned by the other shard.
        observed_ids = set(topk_ids.cpu().flatten().tolist())
        assert observed_ids <= {0, 1, 2, 3}, observed_ids
        self.seen_ids.update(observed_ids)
        start, end = self.expert_range
        if hidden_states.shape[0] and not observed_ids.intersection(range(start, end)):
            self.empty_shard_batches += 1
        if self.fail:
            raise RuntimeError("injected follower expert failure")
        return _sharded_expert_result(
            hidden_states, topk_ids, topk_weights, self.layer, self.expert_range
        )


def _check_groups(endpoint):
    control_group = collective.get_fast_afd_control_group()
    assert endpoint._control_transport.group is control_group
    assert collective.get_fast_afd_control_group() is control_group
    assert dist.get_backend(control_group) == "gloo"
    assert dist.get_backend(collective._get_group(collective.Group.DP_AND_TP)) == "nccl"


def _client(rank, dtype, expert_ranks=None):
    kwargs = {} if expert_ranks is None else {"expert_ranks": expert_ranks}
    client = FastAFDClient(
        _EXPERT_RANK,
        _HIDDEN_SIZE,
        _TOP_K,
        torch.device("cuda", rank),
        dtype,
        4,
        topk_ids_dtype=torch.int32 if rank == 0 else torch.int64,
        **kwargs,
    )
    _check_groups(client)
    return client


def _service(dtype, fail=False, rank=_EXPERT_RANK, expert_ranks=None):
    kwargs = {} if expert_ranks is None else {"expert_ranks": expert_ranks}
    if expert_ranks is None:
        layers = {layer: _CudaExpert(layer, fail=fail) for layer in (0, 1)}
    else:
        layers = {layer: _ShardedCudaExpert(layer, rank, fail=fail) for layer in (0, 1)}
    service = FastAFDExpertService(
        [0, 1], layers, _HIDDEN_SIZE, _TOP_K, f"cuda:{rank}", dtype, 4, **kwargs
    )
    _check_groups(service)
    return service, layers


def _request(client, rank, step, layer, tokens, dtype, expert_ranks=None):
    inputs = _inputs if expert_ranks is None else _sharded_inputs
    reference = _expert_result if expert_ranks is None else _sharded_expert_result
    result = client.forward(
        layer, *inputs(rank, step, tokens, dtype, torch.device("cuda", rank))
    )
    # Consume the received tensor immediately on the current CUDA stream,
    # before any synchronize/assertion. This checks NCCL's data dependencies
    # after the CPU completion status has already arrived.
    consumed = result * 2 + 3
    expected = reference(*inputs(rank, step, tokens, dtype, "cpu"), layer)
    return consumed, expected * 2 + 3


def _check_results(results):
    for actual, expected in results:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _healthy_attention(rank, dtype, first_rank_stopped, expert_ranks=None):
    client = _client(rank, dtype, expert_ranks)
    for step, requests in enumerate(_STEPS):
        client.begin_step()
        results = [
            _request(client, rank, step, layer, tokens, dtype, expert_ranks)
            for layer, tokens in requests[rank]
        ]
        client.finish()
        assert client.global_idle == _IDLE_FLAGS[step]
        _check_results(results)

    if rank == 0:
        client.stop()
        first_rank_stopped.set()
        client.stop()  # Idempotent shutdown must not send a second STOP.
    else:
        # AG1 deliberately cannot send its next header until AG0's STOP ACK
        # arrives. A group-wide STOP barrier would deadlock this sequence.
        assert first_rank_stopped.wait(timeout=30), "AG0 STOP waited for AG1"
        client.begin_step()
        results = [_request(client, rank, len(_STEPS), 0, 2, dtype, expert_ranks)]
        client.finish()
        assert not client.global_idle
        _check_results(results)
        client.stop()


def _healthy_service(dtype, rank=_EXPERT_RANK, expert_ranks=None):
    service, layers = _service(dtype, rank=rank, expert_ranks=expert_ranks)
    idle_flags = []
    while not service.serve_until_done():
        idle_flags.append(service.global_idle)
    assert idle_flags == [*_IDLE_FLAGS, False], idle_flags
    # The aligned second active step must aggregate both AGs before executing.
    assert 6 in layers[0].batch_sizes, layers[0].batch_sizes
    assert 5 in layers[1].batch_sizes, layers[1].batch_sizes
    if expert_ranks is None or rank == expert_ranks[0]:
        assert not service._live_attention_ranks
    if expert_ranks is not None:
        assert layers[0].seen_ids == {0, 1, 2, 3}, layers[0].seen_ids
        assert sum(layer.empty_shard_batches for layer in layers.values()) > 0
    assert service.serve_until_done()  # No communication after all ranks stop.


def _failure_attention(rank, scenario, expert_ranks=None):
    client = _client(rank, torch.bfloat16, expert_ranks)
    case = unittest.TestCase()
    client.begin_step()
    if scenario == "expert":
        with case.assertRaisesRegex(RuntimeError, "EXPERT_FAILED"):
            _request(client, rank, 0, 0, rank + 1, torch.bfloat16, expert_ranks)
    elif rank == 0 and scenario == "rejected":
        with case.assertRaisesRegex(RuntimeError, "BAD_LAYER"):
            _request(client, rank, 0, 99, 1, torch.bfloat16, expert_ranks)
    elif rank == 0 and scenario == "abort":
        hidden, ids, weights = _inputs(rank, 0, 1, torch.bfloat16, "cuda:0")
        with case.assertRaisesRegex(ValueError, "topk_ids must have shape"):
            client.forward(0, hidden, ids[:, :1], weights)
    else:
        # A rejected header must not prevent draining the healthy AG payload;
        # local ABORT must also wake an idle AG waiting at FINISH.
        if scenario == "rejected":
            _check_results(
                [_request(client, rank, 0, 0, 2, torch.bfloat16, expert_ranks)]
            )
        with case.assertRaisesRegex(RuntimeError, "STEP_FAILED"):
            client.finish()
    client.finish()  # Failed/rejected clients are already finished.
    with case.assertRaisesRegex(RuntimeError, "failed in a previous step"):
        client.begin_step()
    client.stop()  # No receiver remains after the service reports the error.


def _failure_service(scenario, rank=_EXPERT_RANK, expert_ranks=None):
    failing_rank = _EXPERT_RANK if expert_ranks is None else expert_ranks[-1]
    service, _ = _service(
        torch.bfloat16,
        fail=scenario == "expert" and rank == failing_rank,
        rank=rank,
        expert_ranks=expert_ranks,
    )
    expected = {
        "expert": "expert.*fail|fail.*expert",
        "rejected": "BAD_LAYER|STEP_FAILED",
        "abort": "aborted|STEP_FAILED",
    }
    with unittest.TestCase().assertRaisesRegex(RuntimeError, expected[scenario]):
        service.serve_until_done()


def _worker(rank, port, stop_events, expert_ranks=None):
    world_size = _WORLD_SIZE if expert_ranks is None else max(expert_ranks) + 1
    torch.cuda.set_device(rank)
    torch.set_default_device(torch.device("cuda", rank))
    config = ParallelismConfig()
    config.world_rank = rank
    config.world_size = world_size
    config.local_rank = rank
    config.local_world_size = world_size
    config.tp_size = 1
    config.dp_size = world_size
    config.use_ub_comm = False
    base_port = port + 11
    nccl_config = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=base_port - 2,
        dp_tp_nccl_port=base_port - 10,
        ffn_tp_nccl_port=base_port - 5,
    )
    collective.init_distributed_environment(config, nccl_config, port)
    try:
        cached_control_group = None
        # Use a non-default CUDA stream to cover the send/recv stream waits.
        with torch.cuda.stream(torch.cuda.Stream(device=rank)):
            for dtype, stop_event in zip((torch.bfloat16, torch.float16), stop_events):
                if rank >= _EXPERT_RANK:
                    print(f"FastAFD transport: healthy lifecycle {dtype}", flush=True)
                    _healthy_service(dtype, rank, expert_ranks)
                else:
                    _healthy_attention(rank, dtype, stop_event, expert_ranks)
                control_group = collective.get_fast_afd_control_group()
                if cached_control_group is None:
                    cached_control_group = control_group
                assert control_group is cached_control_group
                dist.monitored_barrier(
                    group=control_group, timeout=timedelta(seconds=30)
                )

            for scenario in ("expert", "rejected", "abort"):
                if rank >= _EXPERT_RANK:
                    print(
                        f"FastAFD transport: failure propagation {scenario}", flush=True
                    )
                    _failure_service(scenario, rank, expert_ranks)
                else:
                    _failure_attention(rank, scenario, expert_ranks)
                assert collective.get_fast_afd_control_group() is cached_control_group
                dist.monitored_barrier(
                    group=cached_control_group, timeout=timedelta(seconds=30)
                )
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        collective.destroy_distributed_environment()


class FastAFDTransportIntegrationTest(unittest.TestCase):
    def test_three_gpu_control_data_and_lifecycle(self):
        self._run_workers()

    def _run_workers(self, expert_ranks=None):
        world_size = _WORLD_SIZE if expert_ranks is None else max(expert_ranks) + 1
        self.assertTrue(torch.cuda.is_available(), "requires CUDA")
        self.assertGreaterEqual(
            torch.cuda.device_count(), world_size, f"requires {world_size} GPUs"
        )
        self.assertTrue(dist.is_gloo_available(), "requires Gloo CPU control support")
        self.assertTrue(dist.is_nccl_available(), "requires NCCL GPU payload support")

        ports, locks = PortManager().get_consecutive_ports(1)
        context = mp.get_context("spawn")
        stop_events = [context.Event(), context.Event()]
        processes = []
        try:
            for rank in range(world_size):
                process = context.Process(
                    target=_worker,
                    args=(rank, ports[0], stop_events, expert_ranks),
                    name=f"fast-afd-rank-{rank}",
                )
                process.start()
                processes.append(process)
            pending = {process.sentinel: process for process in processes}
            deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
            while pending:
                ready = wait(pending, timeout=max(0.0, deadline - time.monotonic()))
                self.assertTrue(ready, "FastAFD workers exceeded the test deadline")
                for sentinel in ready:
                    process = pending.pop(sentinel)
                    process.join()
                    self.assertEqual(process.exitcode, 0, f"{process.name} failed")
        finally:
            # Production idle waits intentionally have no short deadline.
            # Reap only this test's children if any rank fails or hangs.
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for lock in locks:
                lock.__exit__(None, None, None)


class FastAFDMultiExpertTransportIntegrationTest(unittest.TestCase):
    def test_four_gpu_shard_reduction_control_and_lifecycle(self):
        FastAFDTransportIntegrationTest._run_workers(self, _MULTI_EXPERT_RANKS)


if __name__ == "__main__":
    unittest.main()
