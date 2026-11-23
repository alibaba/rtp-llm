"""Run with torchrun --standalone --nproc_per_node=2 with the CUDA 13 native extension."""

import gc
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
import weakref
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


# Independent CUDA/NCCL/Triton implementations may differ at BF16 rounding
# boundaries. Keep the existing MoE gating tolerance, and report the actual
# differences rather than claiming a cross-implementation bitwise contract.
def check_reference(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    return {
        "unequal_elements": int(torch.count_nonzero(actual != expected)),
        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
    }


def main():
    if sys.argv[1:2] == ["--manage-failing-tp-group"]:
        manage_failing_tp_group(Path(sys.argv[2]))
        return
    if os.environ.get("RTP_TEST_LSA_TP_RECOVERY") == "1":
        run_tp_recovery()
        return
    if os.environ.get("RTP_TEST_LSA_PEER_EXIT") == "1":
        run_peer_exit()
        return
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
        {}
        if rendezvous is None
        else dict(init_method=rendezvous, rank=rank, world_size=2)
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
        Path(result_path).write_text(json.dumps(results, indent=2) + "\n")
    dist.destroy_process_group()
    if rank == 0:
        print("LSA_MODEL_ENTRY_DONE", flush=True)


def prepare_fault_case(rank, group_options, hidden, rows, timeout_ms):
    from rtp_llm.ops import rtp_llm_ops

    dist.init_process_group("gloo", **group_options)
    ops = rtp_llm_ops._qwen38_lsa
    ids = [ops.unique_id() if rank == 0 else None]
    dist.broadcast_object_list(ids, src=0)
    handle = ops.allocate(rank, hidden)
    ops.set_timeout(handle, timeout_ms)
    ops.connect(handle, rank, ids[0])
    ops.check_capability(handle)
    ops.activate(handle)
    ops.check_activation(handle)
    value = torch.zeros(rows, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.full_like(value, float("nan"))
    gate = torch.zeros(rows, 1, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    stream.synchronize()
    dist.barrier()
    return ops, handle, value, output, gate, stream


def run_peer_exit():
    if "LOCAL_RANK" in os.environ or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "The peer-exit regression needs two GPUs and its own parent process"
        )
    ctx = mp.get_context("spawn")
    submitted = ctx.Event()
    peer_exited = ctx.Event()
    with tempfile.TemporaryDirectory() as tmp:
        rendezvous = f"file://{tmp}/rendezvous"
        survivor = ctx.Process(
            target=peer_exit_worker, args=(0, rendezvous, submitted, peer_exited)
        )
        peer = ctx.Process(
            target=peer_exit_worker, args=(1, rendezvous, submitted, peer_exited)
        )
        workers = (survivor, peer)
        try:
            for worker in workers:
                worker.start()
            peer.join(45)
            assert peer.exitcode == 42 and submitted.is_set(), peer.exitcode
            peer_exited.set()
            survivor.join(20)
            assert survivor.exitcode == 0, survivor.exitcode
            print("LSA_PEER_EXIT_TEST_PASSED peer_exit=42 survivor_exit=0", flush=True)
        finally:
            for worker in workers:
                if worker.pid is not None:
                    if worker.is_alive():
                        worker.kill()
                    worker.join()


def peer_exit_worker(rank, rendezvous, submitted, peer_exited):
    torch.cuda.set_device(rank)
    rows, hidden = 32, 8192
    ops, handle, value, output, gate, stream = prepare_fault_case(
        rank, dict(init_method=rendezvous, rank=rank, world_size=2), hidden, rows, 0
    )
    if rank == 1:
        assert submitted.wait(20)
        # A real process exit with live NCCL windows and no matching launch.
        os._exit(42)

    ops.apply(
        handle,
        value.data_ptr(),
        value.data_ptr(),
        gate.data_ptr(),
        output.data_ptr(),
        rows,
        stream.cuda_stream,
    )
    submitted.set()
    assert peer_exited.wait(25)
    try:
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            assert not stream.query(), "LSA completed without the peer contributing"
            time.sleep(0.01)
    except RuntimeError as error:
        # Some drivers may invalidate the dead peer's mapping immediately.
        if not any(
            message in str(error)
            for message in ("unspecified launch failure", "illegal memory access")
        ):
            raise
        print("LSA_PEER_EXIT_DEVICE_FAULT", flush=True)
    else:
        ops.abort(handle)
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "unspecified launch failure|illegal memory access"
        ):
            stream.synchronize()
        print("LSA_PEER_EXIT_PENDING_THEN_CANCELLED", flush=True)
    os._exit(0)


def run_tp_recovery():
    if "LOCAL_RANK" in os.environ or torch.cuda.device_count() < 2:
        raise RuntimeError("TP recovery needs two GPUs and its own process manager")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        command = [sys.executable, __file__, "--manage-failing-tp-group", tmp]
        with subprocess.Popen(command, start_new_session=True) as supervisor:
            try:
                assert supervisor.wait(timeout=75) == 1
            finally:
                if supervisor.poll() is None:
                    os.killpg(supervisor.pid, signal.SIGKILL)
        old = json.loads((root / "failed-group.json").read_text())
        assert old["peer_exit"] == 42, old
        assert old["survivor_exit"] == -signal.SIGKILL, old
        assert old["failure_detected"] and old["launch_pending"], old
        assert old["pids"][0] != old["pids"][1], old
        print("LSA_TP_GROUP_REAPED", json.dumps(old), flush=True)

        # A deployment supervisor starts a new service after the manager's
        # nonzero exit. Reproduce that handoff with a fresh rendezvous here.
        rendezvous = f"file://{root}/restarted-rendezvous"
        mp.spawn(restarted_tp_worker, args=(rendezvous, root), nprocs=2)
        new = [
            json.loads((root / f"ready-rank{rank}.json").read_text())
            for rank in range(2)
        ]
        assert {entry["rank"] for entry in new} == {0, 1}, new
        assert len({entry["pid"] for entry in new}) == 2, new
        assert not {entry["pid"] for entry in new}.intersection(old["pids"]), new
        print("LSA_TP_GROUP_RESTARTED", json.dumps(new), flush=True)


def manage_failing_tp_group(root):
    from rtp_llm.utils.process_manager import ProcessManager

    ctx = mp.get_context("spawn")
    submitted = ctx.Event()
    rendezvous = f"file://{root}/failed-rendezvous"
    workers = [
        ctx.Process(
            target=failing_tp_worker,
            args=(rank, rendezvous, submitted),
            name=f"lsa-tp-rank{rank}",
        )
        for rank in range(2)
    ]

    def record_exit():
        (root / "failed-group.json").write_text(
            json.dumps(
                {
                    "pids": [worker.pid for worker in workers],
                    "survivor_exit": workers[0].exitcode,
                    "peer_exit": workers[1].exitcode,
                    "failure_detected": manager.failure_detected,
                    "launch_pending": submitted.is_set(),
                }
            )
        )

    manager = ProcessManager(
        monitor_interval=1, allow_defer_first_sigterm=True, pre_exit_cleanup=record_exit
    )
    try:
        for worker in workers:
            worker.start()
        # Matches the production backend manager's registration and failure path.
        manager.set_processes(workers, shutdown_group="backend")
        manager.monitor_and_release_processes()
        raise AssertionError("The failed TP group must make its manager exit nonzero")
    finally:
        for worker in workers:
            if worker.pid is not None and worker.is_alive():
                worker.kill()
                worker.join()


def failing_tp_worker(rank, rendezvous, submitted):
    # Exercise the service manager's hard-kill path when a rank stuck in
    # the CUDA call cannot complete its graceful shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    torch.cuda.set_device(rank)
    rows = 32
    ops, handle, value, output, gate, stream = prepare_fault_case(
        rank, dict(init_method=rendezvous, rank=rank, world_size=2), 8192, rows, 0
    )
    if rank == 1:
        assert submitted.wait(20), "The survivor never entered LSA"
        os._exit(42)

    ops.apply(
        handle,
        value.data_ptr(),
        value.data_ptr(),
        gate.data_ptr(),
        output.data_ptr(),
        rows,
        stream.cuda_stream,
    )
    assert not stream.query(), "Unmatched LSA launch completed before peer exit"
    submitted.set()
    try:
        stream.synchronize()
    except RuntimeError as error:
        if not any(
            reason in str(error)
            for reason in ("unspecified launch failure", "illegal memory access")
        ):
            raise
        # A driver may report a device fault first. Keep the process alive so
        # the test can still verify the manager owns group cleanup.
        print("LSA_TP_SURVIVOR_DEVICE_FAULT", flush=True)
        signal.pause()
    raise AssertionError("Unmatched LSA completed or the survivor escaped cleanup")


def restarted_tp_worker(rank, rendezvous, root):
    torch.cuda.set_device(rank)
    rows = 32
    ops, handle, value, output, gate, stream = prepare_fault_case(
        rank, dict(init_method=rendezvous, rank=rank, world_size=2), 8192, rows, 0
    )
    shared = torch.zeros_like(value)

    def launch():
        ops.apply(
            handle,
            value.data_ptr(),
            shared.data_ptr(),
            gate.data_ptr(),
            output.data_ptr(),
            rows,
            torch.cuda.current_stream().cuda_stream,
        )

    value.fill_(rank + 1)
    launch()
    stream.synchronize()
    assert torch.equal(output, torch.full_like(output, 3))

    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        launch()
    for multiplier in (2, 3):
        value.fill_((rank + 1) * multiplier)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, torch.full_like(output, 3 * multiplier))
    dist.barrier()
    dist.destroy_process_group()
    (root / f"ready-rank{rank}.json").write_text(
        json.dumps({"rank": rank, "pid": os.getpid(), "eager": 1, "graph_replays": 2})
    )


def run_missing_peer(rank, group_options):
    abort_default = os.environ.get("RTP_TEST_LSA_DEFAULT_ABORT") == "1"
    hidden, rows = (8192, 32) if abort_default else (2560, 1)
    ops, handle, value, output, gate, stream = prepare_fault_case(
        rank, group_options, hidden, rows, 0 if abort_default else 1000
    )
    if rank == 0:
        start = time.monotonic()
        ops.apply(
            handle,
            value.data_ptr(),
            value.data_ptr(),
            gate.data_ptr(),
            output.data_ptr(),
            rows,
            stream.cuda_stream,
        )
        if abort_default:
            # Rank 1 deliberately never submits its matching kernel. Cancel
            # rank 0's default untimed wait through the independent H2D stream
            # and the real NCCL abort API. This shape uses all 64 barrier CTAs.
            # This case runs in its own process and never reuses the context.
            time.sleep(0.2)
            ops.abort(handle)
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "unspecified launch failure|illegal memory access"
        ):
            stream.synchronize()
        elapsed = time.monotonic() - start
        lower = 0.1 if abort_default else 0.5
        assert lower <= elapsed < 15, f"LSA failed to stop: {elapsed}s"
        print(
            f"LSA_MISSING_PEER_STOP default_abort={abort_default} rows={rows} hidden={hidden} elapsed_s={elapsed:.6f}",
            flush=True,
        )
    dist.barrier()
    # A timeout makes the CUDA context unusable. Let the subprocess exit
    # without attempting to reuse or destroy the test communicator.
    os._exit(0)


def check_startup_fallback(model, config, parallelism, rank, device):
    from rtp_llm.ops import rtp_llm_ops

    ops = rtp_llm_ops._qwen38_lsa
    for stage in ("allocate", "check_capability", "check_activation"):
        with (
            patch.object(ops, "allocate", wraps=ops.allocate) as allocate,
            patch.object(ops, "connect", wraps=ops.connect) as connect,
            patch.object(ops, "check_capability", wraps=ops.check_capability) as check,
            patch.object(ops, "activate", wraps=ops.activate) as activate,
            patch.object(
                ops, "check_activation", wraps=ops.check_activation
            ) as check_activation,
            patch.object(ops, "destroy", wraps=ops.destroy) as destroy,
        ):
            if rank == 1:
                failing = {
                    "allocate": allocate,
                    "check_capability": check,
                    "check_activation": check_activation,
                }[stage]
                failing.side_effect = RuntimeError(
                    f"injected local LSA {stage} failure"
                )
            assert prepare_model_reduce(model, config, parallelism) is None
            assert all(
                not hasattr(layer.mlp, "_reduce_shared_gate") for layer in model.layers
            )
            allocate.assert_called_once_with(device.index, config.hidden_size)
            if stage == "allocate":
                connect.assert_not_called()
                check.assert_not_called()
            else:
                connect.assert_called_once()
                check.assert_called_once()
            if stage == "check_activation":
                activate.assert_called_once()
                check_activation.assert_called_once()
            else:
                activate.assert_not_called()
                check_activation.assert_not_called()
            if stage == "allocate" and rank == 1:
                destroy.assert_not_called()
            else:
                destroy.assert_called_once()
                assert destroy.call_args.args[0] > 0

        # The original TP communicator is still usable on both ranks.
        value = torch.full((1,), rank + 1, device=device, dtype=torch.bfloat16)
        dist.all_reduce(value)
        assert value.item() == 3
        if rank == 0:
            print(f"LSA_STARTUP_SYMMETRIC_FALLBACK {stage}", flush=True)


def check_shared_packing(rank, device):
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
        CudaF16Linear,
    )
    from rtp_llm.models_py.modules.qwen4_exp.moe import shared_gemm
    from rtp_llm.utils.model_weight import W
    from rtp_llm.model_loader.model_weight_info import ModelWeights

    for fail_rank in (0, 1):
        for failure in ("memory", "layout", "mid_pack", None):
            torch.manual_seed(816)
            sources = [
                {
                    W.ffn_w13: torch.randn(
                        2560, 1280, device=device, dtype=torch.bfloat16
                    ),
                    W.ffn_w2: torch.randn(
                        640, 2560, device=device, dtype=torch.bfloat16
                    ),
                }
                for _ in range(2)
            ]
            layers = [
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        shared_expert=SimpleNamespace(
                            up_proj=CudaF16Linear(source[W.ffn_w13]),
                            down_proj=CudaF16Linear(source[W.ffn_w2]),
                        )
                    )
                )
                for source in sources
            ]
            original = [
                (layer.mlp.shared_expert.up_proj, layer.mlp.shared_expert.down_proj)
                for layer in layers
            ]
            if rank == fail_rank and failure == "layout":
                sources[0][W.ffn_w13] = sources[0][W.ffn_w13].clone()
            pack = shared_gemm._PackedSharedLinear
            count = 0
            packed_refs = []

            def allocate(*args):
                nonlocal count
                count += 1
                if rank == fail_rank and failure == "mid_pack" and count == 3:
                    raise torch.OutOfMemoryError("injected shared packing OOM")
                result = pack(*args)
                packed_refs.append(weakref.ref(result.packed_weight))
                return result

            memory = torch.cuda.mem_get_info(device)
            with (
                patch.object(shared_gemm, "_PackedSharedLinear", side_effect=allocate),
                patch(
                    "torch.cuda.mem_get_info",
                    return_value=(
                        (0, memory[1])
                        if rank == fail_rank and failure == "memory"
                        else memory
                    ),
                ),
            ):
                # Qualification still needs the real projection type.
                success = shared_gemm.prepare_shared_gemm(
                    layers, sources, group=dist.group.WORLD
                )
            assert success == (failure is None), (rank, fail_rank, failure)
            if not success:
                gc.collect()
                assert all(
                    ref() is None for ref in packed_refs
                ), "Failed packing retained a private copy"
            for layer, source, before in zip(layers, sources, original):
                shared = layer.mlp.shared_expert
                for index, key in enumerate((W.ffn_w13, W.ffn_w2)):
                    assert (
                        bool(
                            getattr(
                                source[key], "_rtp_requires_restart_for_update", False
                            )
                        )
                        == success
                    )
                    projection = (shared.up_proj, shared.down_proj)[index]
                    if not success:
                        assert projection is before[index]
                value = torch.ones(4, 2560, device=device, dtype=torch.bfloat16)
                output = shared.up_proj(value)
                other = output.clone()
                dist.broadcast(other, src=0)
                assert torch.equal(other, output), (rank, fail_rank, failure)
            manager = ModelWeights(2, str(device), torch.bfloat16)
            manager.weights = sources
            for index, source in enumerate(sources):
                before = source[W.ffn_w13].clone()
                replacement = torch.zeros_like(before)
                if success:
                    with unittest.TestCase().assertRaises(PermissionError):
                        manager.update_layer_weight(index, W.ffn_w13, replacement)
                    assert torch.equal(source[W.ffn_w13], before)
                else:
                    manager.update_layer_weight(index, W.ffn_w13, replacement)
                    assert torch.equal(source[W.ffn_w13], replacement)
            dist.barrier()
            if rank == 0:
                print(
                    "SHARED_PACKING_AGREEMENT", fail_rank, failure, success, flush=True
                )


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
            with patch(
                "rtp_llm.models_py.modules.qwen4_exp.moe.reduce_gate.exchange_group_state"
            ) as exchange:
                config.is_mtp = True
                assert prepare_model_reduce(model, config, parallelism) is None
                config.is_mtp = False
                exchange.assert_not_called()
            config.quant_config = SimpleNamespace(decode_lsa_enabled=rank == 0)
            assert prepare_model_reduce(model, config, parallelism) is None
            config.quant_config = SimpleNamespace(
                decode_lsa_startup_timeout_s=120 + rank
            )
            assert prepare_model_reduce(model, config, parallelism) is None
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
        if hidden_size == 2560:
            config.quant_config = SimpleNamespace(decode_lsa_timeout_ms=1000 * rank)
            assert prepare_model_reduce(model, config, parallelism) is None
            config.quant_config = SimpleNamespace(decode_shared_gemm_nk=bool(rank))
            assert prepare_model_reduce(model, config, parallelism) is None
            config.quant_config = SimpleNamespace(decode_lsa_timeout_ms=2000)
            check_startup_fallback(model, config, parallelism, rank, device)
            check_shared_packing(rank, device)
        owner = prepare_model_reduce(model, config, parallelism)
    assert owner is not None, "SM120 CUDA 13 native LSA unavailable"
    assert mlp._reduce_shared_gate is owner
    assert not owner.supports(
        torch.empty(33, hidden_size, device=device, dtype=torch.bfloat16)
    )
    assert not owner.supports(
        torch.empty(1, hidden_size + 8, device=device, dtype=torch.bfloat16)
    )
    for invalid_hidden in (0, 7, (2**31 - 1) // 32 + 1):
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "hidden size"):
            owner.ops.allocate(device.index, invalid_hidden)
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
                return shared.clone()

        class Gate(torch.nn.Module):
            weight = gate

            def forward(self, value):
                return gate.clone()

        layer = Qwen4MoeLayer.__new__(Qwen4MoeLayer)
        torch.nn.Module.__init__(layer)
        layer.shared_expert = Shared()
        layer.shared_expert_gate = Gate()
        layer._reduce_shared_gate = owner
        layer.parallelism_config = parallelism
        layer.prepare_reduce_gate_overlap()

        def routed(self, value, *, skip_tp_allreduce=False):
            assert skip_tp_allreduce
            return partial

        layer._forward_routed = MethodType(routed, layer)

        def baseline():
            output = torch.empty_like(partial, memory_format=torch.contiguous_format)
            reference_partial = partial.contiguous()
            owner.ops.reference(
                owner.handle,
                reference_partial.data_ptr(),
                output.data_ptr(),
                rows,
                torch.cuda.current_stream().cuda_stream,
            )
            reference_shared = (
                shared
                if shared.is_contiguous() and shared.data_ptr() % 16 == 0
                else shared.clone(memory_format=torch.contiguous_format)
            )
            return sigmoid_gate_scale_add_triton(gate, reference_shared, output)

        def fused():
            return layer(hidden)

        if rows in (2, 4):
            aligned_partial, aligned_shared, aligned_hidden = partial, shared, hidden
            if rank == 1:
                partial, shared, hidden = (
                    torch.empty(
                        rows * hidden_size + 1, device=device, dtype=torch.bfloat16
                    )[1:].view(rows, hidden_size)
                    for _ in range(3)
                )
                if rows == 4:
                    partial, shared, hidden = (
                        torch.empty(
                            hidden_size, rows, device=device, dtype=torch.bfloat16
                        ).t()
                        for _ in range(3)
                    )
                    assert all(
                        not value.is_contiguous() for value in (partial, shared, hidden)
                    )
                else:
                    assert all(
                        value.data_ptr() % 16 for value in (partial, shared, hidden)
                    )
            for value in (partial, shared, gate):
                value.normal_()
            assert owner.supports(hidden)
            expected = baseline()
            actual = fused()
            torch.cuda.synchronize()
            check_reference(actual, expected)
            offset_graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(offset_graph):
                captured_offset = fused()
            for value in (partial, shared, gate):
                value.normal_()
            expected = baseline()
            offset_graph.replay()
            torch.cuda.synchronize()
            check_reference(captured_offset, expected)
            offset_graph.reset()
            partial, shared, hidden = aligned_partial, aligned_shared, aligned_hidden

        if hidden_size == 2560 and rows == 1:
            # A legal late arrival below the opt-in deadline must not fault.
            expected = baseline()
            dist.barrier()
            if rank == 1:
                time.sleep(0.1)
            actual = fused()
            torch.cuda.synchronize()
            check_reference(actual, expected)
            if rank == 0:
                print(
                    "LSA_DELAYED_PEER_PASSED timeout_ms=2000 delay_ms=100", flush=True
                )

        comparisons = []
        for pattern in range(12):
            partial.normal_(mean=pattern / 10)
            shared.normal_()
            gate.normal_(std=pattern + 1)
            reference = baseline()
            actual = fused()
            torch.cuda.synchronize()
            comparisons.append(check_reference(actual, reference))
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
            comparisons.append(check_reference(captured, reference))
            # The same fused implementation must remain byte-identical between
            # eager and Graph for these identical inputs and reduction order.
            eager = fused()
            torch.cuda.synchronize()
            assert torch.equal(captured, eager), (rank, rows, "eager/graph")
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
            dict(
                hidden_size=hidden_size,
                rows=rows,
                times_us=timings,
                reference_atol=1e-2,
                reference_rtol=1e-2,
                eager_graph_bitwise=True,
                reference_unequal_elements=sum(
                    x["unequal_elements"] for x in comparisons
                ),
                reference_max_abs_error=max(x["max_abs_error"] for x in comparisons),
            )
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
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "closed"):
        owner.prepare_output(hidden)
    return results


if __name__ == "__main__":
    main()
