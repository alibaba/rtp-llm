"""Run with torchrun --standalone --nproc_per_node=2 with the CUDA 13 native extension."""

import json
import os
import statistics
import tempfile
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.qwen4_exp.moe.reduce_gate import (
    prepare_model_reduce,
)
from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer
from rtp_llm.models_py.triton_kernels.common.moe_gating import (
    sigmoid_gate_scale_add_triton,
)


def main():
    if "LOCAL_RANK" in os.environ:
        run_worker(int(os.environ["LOCAL_RANK"]))
    else:
        if torch.cuda.device_count() < 2:
            raise RuntimeError("The LSA regression requires two CUDA devices")
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(run_worker, args=(f"file://{tmp}/rendezvous",), nprocs=2)


def run_worker(rank, rendezvous=None):
    torch.cuda.set_device(rank)
    group_options = (
        {} if rendezvous is None else dict(init_method=rendezvous, rank=rank, world_size=2)
    )
    if os.environ.get("RTP_TEST_LSA_MISSING_PEER") == "1":
        run_missing_peer(rank, group_options)
        return
    dist.init_process_group("nccl", **group_options)
    device = torch.device("cuda", rank)
    results = []
    for hidden_size in (8, 264, 2048, 2560, 4096, 8192):
        results.extend(run_hidden_size(device, rank, hidden_size))
    result_path = os.environ.get("RTP_QWEN4_MOE_LSA_RESULTS")
    if rank == 0 and result_path:
        Path(result_path).write_text(
            json.dumps(results, indent=2) + "\n"
        )
    dist.destroy_process_group()
    if rank == 0:
        print("LSA_MODEL_ENTRY_DONE", flush=True)


def run_missing_peer(rank, group_options):
    from rtp_llm.ops import rtp_llm_ops

    dist.init_process_group("gloo", **group_options)
    ops = rtp_llm_ops._qwen38_lsa
    ids = [ops.unique_id() if rank == 0 else None]
    dist.broadcast_object_list(ids, src=0)
    hidden = 2560
    handle = ops.create(rank, rank, hidden, ids[0])
    value = torch.zeros(1, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.full_like(value, float("nan"))
    gate = torch.zeros(1, 1, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    dist.barrier()
    if rank == 0:
        start = time.monotonic()
        ops.apply(
            handle,
            value.data_ptr(),
            value.data_ptr(),
            gate.data_ptr(),
            output.data_ptr(),
            1,
            stream.cuda_stream,
        )
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "unspecified launch failure"
        ):
            stream.synchronize()
        assert time.monotonic() - start < 30, "LSA timeout did not bound the wait"
        print("LSA_MISSING_PEER_TIMEOUT", flush=True)
    dist.barrier()
    # A timeout makes the CUDA context unusable. Let the subprocess exit
    # without attempting to reuse or destroy the test communicator.
    os._exit(0)


def run_hidden_size(device, rank, hidden_size):
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
        PureTpRouterNvfp4Cutlass,
    )

    mlp = SimpleNamespace(
        fused_moe=SimpleNamespace(
            router=PureTpRouterNvfp4Cutlass.__new__(PureTpRouterNvfp4Cutlass)
        ),
        shared_expert=object(),
        shared_expert_gate=SimpleNamespace(
            weight=torch.empty(1, hidden_size, device=device, dtype=torch.bfloat16)
        ),
        use_ep_shared_allreduce=False,
        use_unified_tp_allreduce=False,
        _shared_stream=None,
    )
    model = SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])
    config = SimpleNamespace(
        is_mtp=False,
        hidden_size=hidden_size,
        _qwen4_dense_storage="source_bf16",
        eplb_config=SimpleNamespace(enable_eplb=lambda: False),
    )
    parallelism = SimpleNamespace(
        role_type="DECODE",
        tp_size=2,
        ep_size=2,
        dp_size=1,
        pp_size=1,
        local_world_size=2,
        get_ffn_tp_size=lambda: 1,
    )
    with patch.object(collective_torch, "_get_group", return_value=dist.group.WORLD):
        if hidden_size == 2560:
            with patch.dict(os.environ, {"NCCL_P2P_DISABLE": str(rank)}):
                assert prepare_model_reduce(model, config, parallelism) is None
            # Decline before native communicator creation on either rank.
            if rank == 1:
                with patch("torch.cuda.mem_get_info", return_value=(0, 0)):
                    assert prepare_model_reduce(model, config, parallelism) is None
            else:
                assert prepare_model_reduce(model, config, parallelism) is None
        # A single rank declining must prevent either from creating a communicator.
        parallelism.ep_size = 4 if rank == 1 else 2
        assert prepare_model_reduce(model, config, parallelism) is None
        parallelism.ep_size = 2
        config.hidden_size = hidden_size + 8 * rank
        mlp.shared_expert_gate.weight = torch.empty(
            1, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        assert prepare_model_reduce(model, config, parallelism) is None
        config.hidden_size = hidden_size
        mlp.shared_expert_gate.weight = torch.empty(
            1, hidden_size, device=device, dtype=torch.bfloat16
        )
        owner = prepare_model_reduce(model, config, parallelism)
    assert owner is not None, "SM120 CUDA 13 native LSA unavailable"
    assert mlp._reduce_shared_gate is owner
    assert not owner.supports(
        torch.empty(33, hidden_size, device=device, dtype=torch.bfloat16)
    )
    assert not owner.supports(
        torch.empty(1, hidden_size + 8, device=device, dtype=torch.bfloat16)
    )
    uid = owner.ops.unique_id()
    for invalid_hidden in (0, 7, (2**31 - 1) // 32 + 1):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "hidden size"):
            owner.ops.create(rank, device.index, invalid_hidden, uid)
    results = []
    for rows in (1, 2, 4, 8, 16, 32):
        torch.manual_seed(300 + rank + rows)
        partial = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
        shared = torch.randn_like(partial)
        gate = torch.randn(rows, 1, device=device, dtype=torch.bfloat16)
        hidden = torch.zeros_like(partial)

        class Shared(torch.nn.Module):
            def forward(self, value, *, skip_allreduce):
                assert skip_allreduce
                return shared

        class Gate(torch.nn.Module):
            def forward(self, value):
                return gate

        layer = Qwen4MoeLayer.__new__(Qwen4MoeLayer)
        torch.nn.Module.__init__(layer)
        layer.shared_expert = Shared()
        layer.shared_expert_gate = Gate()
        layer._reduce_shared_gate = owner

        def routed(self, value, *, skip_tp_allreduce=False):
            assert skip_tp_allreduce
            return partial

        layer._forward_routed = MethodType(routed, layer)

        def baseline():
            output = torch.empty_like(partial)
            owner.ops.reference(
                owner.handle,
                partial.data_ptr(),
                output.data_ptr(),
                rows,
                torch.cuda.current_stream().cuda_stream,
            )
            reference_shared = shared if shared.data_ptr() % 16 == 0 else shared.clone()
            return sigmoid_gate_scale_add_triton(gate, reference_shared, output)

        def fused():
            return layer(hidden)

        if rows == 2:
            aligned_partial, aligned_shared, aligned_hidden = partial, shared, hidden
            if rank == 1:
                partial, shared, hidden = (
                    torch.empty(
                        rows * hidden_size + 1, device=device, dtype=torch.bfloat16
                    )[1:].view(rows, hidden_size)
                    for _ in range(3)
                )
                assert all(value.data_ptr() % 16 for value in (partial, shared, hidden))
            for value in (partial, shared, gate):
                value.normal_()
            assert owner.supports(hidden)
            expected = baseline()
            actual = fused()
            torch.cuda.synchronize()
            assert torch.equal(actual, expected), (rank, hidden_size, "offset eager")
            offset_graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(offset_graph):
                captured_offset = fused()
            for value in (partial, shared, gate):
                value.normal_()
            expected = baseline()
            offset_graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured_offset, expected), (
                rank,
                hidden_size,
                "offset graph",
            )
            offset_graph.reset()
            partial, shared, hidden = aligned_partial, aligned_shared, aligned_hidden

        for pattern in range(12):
            partial.normal_(mean=pattern / 10)
            shared.normal_()
            gate.normal_(std=pattern + 1)
            reference = baseline()
            actual = fused()
            torch.cuda.synchronize()
            assert torch.equal(actual, reference), (rank, rows, pattern)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            captured = fused()
        for _ in range(12):
            partial.normal_()
            shared.normal_()
            gate.normal_(std=4)
            reference = baseline()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, reference), (rank, rows, "dynamic graph")
        timings = {}
        for name, call in (("nccl_triton", baseline), ("staged_lsa_fused", fused)):
            dist.barrier()
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            timed = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timed):
                for _ in range(64):
                    output = call()
            samples = []
            for _ in range(15):
                dist.barrier()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                timed.replay()
                end.record()
                end.synchronize()
                duration = torch.tensor(
                    start.elapsed_time(end) * 1000 / 64, device=device
                )
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
                samples.append(duration.item())
            timings[name] = statistics.median(samples)
        results.append(
            dict(hidden_size=hidden_size, rows=rows, bitwise=True, times_us=timings)
        )
        if rank == 0:
            print("LSA_MODEL_ENTRY_CASE", json.dumps(results[-1]), flush=True)
    # Model close retains contexts until process exit. Tests explicitly destroy
    # only after both ranks synchronize and all graph users have finished.
    torch.cuda.synchronize()
    dist.barrier()
    owner.ops.destroy(owner.handle)
    owner.handle = None
    owner.close()
    return results


if __name__ == "__main__":
    main()
