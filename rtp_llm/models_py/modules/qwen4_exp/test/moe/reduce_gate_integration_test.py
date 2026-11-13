"""Run with torchrun --standalone --nproc_per_node=2 and the pinned LSA library."""

import ctypes
import json
import os
import statistics
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.qwen4_exp.moe.reduce_gate import (
    load_library,
    prepare_model_reduce,
)
from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer
from rtp_llm.models_py.triton_kernels.common.moe_gating import (
    sigmoid_gate_scale_add_triton,
)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank)
    library = load_library(
        os.environ["RTP_QWEN4_MOE_LSA_LIBRARY"],
        os.environ["RTP_QWEN4_MOE_LSA_LIBRARY_SHA256"],
    )
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
            weight=torch.empty(1, device=device, dtype=torch.bfloat16)
        ),
        use_ep_shared_allreduce=False,
        use_unified_tp_allreduce=False,
        _shared_stream=None,
    )
    model = SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])
    config = SimpleNamespace(
        is_mtp=False,
        hidden_size=2560,
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
        with patch.dict(os.environ, {"RTP_QWEN4_MOE_REDUCE_GATE_FUSION": "0"}):
            assert prepare_model_reduce(model, config, parallelism) is None
        with patch.dict(os.environ, {"RTP_QWEN4_MOE_REDUCE_GATE_FUSION": str(rank)}):
            try:
                prepare_model_reduce(model, config, parallelism)
                raise AssertionError("Mismatched rank modes accepted")
            except ValueError:
                pass
        with patch.dict(os.environ, {"RTP_QWEN4_MOE_REDUCE_GATE_FUSION": "1"}):
            parallelism.ep_size = 4
            try:
                prepare_model_reduce(model, config, parallelism)
                raise AssertionError("Unsupported EP layout accepted")
            except RuntimeError:
                pass
            parallelism.ep_size = 2
            owner = prepare_model_reduce(model, config, parallelism)
    assert mlp._reduce_shared_gate is owner
    assert not owner.supports(
        torch.empty(33, 2560, device=device, dtype=torch.bfloat16)
    )
    assert not owner.supports(torch.empty(2, 2560, device=device, dtype=torch.float32))
    library.qwen38_lsa_reference.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    results = []
    for rows in (1, 2, 4, 8, 16, 32):
        torch.manual_seed(300 + rank + rows)
        partial = torch.randn(rows, 2560, device=device, dtype=torch.bfloat16)
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
            owner._check(
                library.qwen38_lsa_reference(
                    owner.handle,
                    partial.data_ptr(),
                    output.data_ptr(),
                    rows,
                    torch.cuda.current_stream().cuda_stream,
                )
            )
            return sigmoid_gate_scale_add_triton(gate, shared, output)

        def fused():
            return layer(hidden)

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
        results.append(dict(rows=rows, bitwise=True, times_us=timings))
        if rank == 0:
            print("LSA_MODEL_ENTRY_CASE", json.dumps(results[-1]), flush=True)
    if rank == 0:
        Path(os.environ["RTP_QWEN4_MOE_LSA_RESULTS"]).write_text(
            json.dumps(results, indent=2) + "\n"
        )
    # Model close retains contexts until process exit. Tests explicitly destroy
    # only after both ranks synchronize and all graph users have finished.
    torch.cuda.synchronize()
    dist.barrier()
    owner.library.qwen38_lsa_destroy(owner.handle)
    owner.handle = None
    owner.close()
    dist.destroy_process_group()
    if rank == 0:
        print("LSA_MODEL_ENTRY_DONE", flush=True)


if __name__ == "__main__":
    main()
