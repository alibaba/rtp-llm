"""Opt-in eight-GPU test of CMP's real RTP TP reduction, eager and captured.

Run only with eight reserved GPUs and RTP_GLM5_TP8_DISTRIBUTED_TEST=1.
Projection spies exercise both post branches without loading model weights;
the collective, process-group initialization and graph replay are production
implementations. This is not a full-model or projection-precision test.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

WORLD_SIZE = 8
WATCHDOG_SECONDS = 180
ENABLE_ENV = "RTP_GLM5_TP8_DISTRIBUTED_TEST"


def _add_repo_root():
    for root in Path(__file__).resolve().parents:
        if (root / "rtp_llm/models_py/modules/hybrid/glm5_cmp.py").is_file():
            sys.path.insert(0, str(root))
            return
    raise FileNotFoundError("RTP-LLM source root")


def _worker(rank, port):
    _add_repo_root()
    import torch
    import torch.distributed as dist

    from rtp_llm.models_py.distributed import collective_torch as collective
    from rtp_llm.models_py.distributed import symm_mem
    from rtp_llm.models_py.modules.hybrid.glm5_cmp import Glm5Cmp
    from rtp_llm.ops import NcclCommConfig, ParallelismConfig
    from rtp_llm.utils.model_weight import W

    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    config = ParallelismConfig()
    config.world_size = config.local_world_size = config.tp_size = WORLD_SIZE
    config.dp_size = 1
    config.world_rank = config.local_rank = config.tp_rank = rank
    config.dp_rank = 0
    config.use_ub_comm = False
    config.prefill_cp_config.kv_cache_sharded = False
    nccl_config = NcclCommConfig(nccl_ip="127.0.0.1")
    try:
        # RTP sets its group map, symmetric-memory communicator and C++
        # callbacks here. dist.init_process_group alone would not test this ABI.
        collective.init_distributed_environment(
            config, nccl_comm_config=nccl_config, nccl_init_port=port
        )
        group = collective._get_group(collective.Group.TP)
        assert dist.get_world_size(group) == WORLD_SIZE
        assert config.get_attn_tp_size() == WORLD_SIZE
        communicator = symm_mem.get_symm_mem_communicator()
        cmp = object.__new__(Glm5Cmp)
        cmp.parallelism_config = config
        cmp.layer_idx = 0
        cmp._is_moe_layer = False
        cmp._output_projection = (None, None)
        stream = torch.cuda.Stream(device=device)
        completed = []
        for rows in (1, 4, 6, 64, 256):
            source = torch.empty((rows, 6144), dtype=torch.bfloat16, device=device)
            local = torch.empty_like(source)
            residual = torch.zeros_like(source)
            latent = torch.zeros((rows, 8, 512), dtype=torch.bfloat16, device=device)
            expanded = torch.zeros((rows, 8, 256), dtype=torch.bfloat16, device=device)
            implementation = SimpleNamespace(
                weights=[{W.mla_vc: None}],
                fmha_params=None,
                _apply_output_bmm=lambda *_: expanded,
            )
            eligible = (
                communicator is not None
                and communicator.should_torch_symm_mem_allreduce(source)
            )
            route = "symm_mem_multimem" if eligible else "torch_nccl"
            if (
                eligible
                and WORLD_SIZE
                not in communicator._WORLD_SIZES_MULTIMEM[
                    communicator.device_capability
                ]
            ):
                route = "symm_mem_two_shot"
            print(
                json.dumps({"rank": rank, "rows": rows, "collective_route": route}),
                flush=True,
            )

            for mode, multiplier in (
                ("direct", 1),
                ("fused_post_spy", 2),
                ("standard_post_spy", 3),
            ):
                cmp.disable_attention_post_moe_pre = mode == "standard_post_spy"

                # Each projection writes a fresh local partial; NCCL's in-place
                # output cannot corrupt the next replay's source inputs.
                def projection(*_):
                    return torch.mul(source, multiplier, out=local)

                cmp.ops = SimpleNamespace(
                    mla_absorbed_output_bmm_quant=lambda *_: (source, None),
                    project_attention_output=projection,
                )
                cmp.self_attn = SimpleNamespace(o_proj=projection)

                def forward():
                    if mode == "direct":
                        local.copy_(source)
                        return cmp._reduce_attention_output(local)
                    output, returned_residual = cmp.mla_post_moe_pre(
                        latent, residual, implementation
                    )
                    assert returned_residual is residual
                    return output

                def check(output, step):
                    # All values/sums are small integers exactly representable
                    # in BF16, so no arithmetic tolerance can hide bad routing.
                    expected_value = multiplier * (36 + WORLD_SIZE * step)
                    expected = torch.full_like(source, expected_value)
                    assert torch.equal(output, expected), (rank, rows, mode, step)
                    assert torch.equal(source, torch.full_like(source, rank + step + 1))

                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for step in range(3):
                        source.fill_(rank + step + 1)
                        check(forward(), step)
                stream.synchronize()
                dist.barrier(group=group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    graph_output = forward()
                for step in (2, 0, 1):
                    source.fill_(rank + step + 1)
                    graph.replay()
                    check(graph_output, step)
                torch.cuda.synchronize(device)
                dist.barrier(group=group)
                completed.append((rows, mode))
                del graph, graph_output
        print(
            "TP8_CMP_DISTRIBUTED_PASS "
            + json.dumps({"rank": rank, "cases": len(completed)}),
            flush=True,
        )
    finally:
        if dist.is_initialized():
            # This can also block on a failed peer; the outer process-group
            # watchdog remains active until all workers and cleanup exit.
            collective.destroy_distributed_environment()


def _worker_group():
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError(
            "Expose exactly eight reserved GPUs with CUDA_VISIBLE_DEVICES"
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    os.environ["RTP_LLM_CPU_TP_BROADCASTER_ID"] = f"cmp_tp8_test_{os.getpid()}_{port}"
    mp.spawn(_worker, args=(port,), nprocs=WORLD_SIZE, join=True)


def _signal_owned_group(process, sig):
    # start_new_session=True below creates this dedicated process group; all
    # eight spawned children inherit it. Never signal an existing GPU job.
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


class Glm5CmpTp8DistributedTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get(ENABLE_ENV) == "1", "Requires explicit reservation of eight GPUs"
    )
    def test_real_tp_reduction_eager_and_graph(self):
        deadline = time.monotonic() + WATCHDOG_SECONDS
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "2")
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-group"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            # Reserve four seconds for TERM/KILL/drain within the watchdog.
            output, _ = process.communicate(
                timeout=max(0.1, deadline - time.monotonic() - 4)
            )
        except subprocess.TimeoutExpired as error:
            _signal_owned_group(process, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                _signal_owned_group(process, signal.SIGKILL)
                try:
                    output, _ = process.communicate(
                        timeout=max(0.1, deadline - time.monotonic())
                    )
                except subprocess.TimeoutExpired:
                    output = str(error.output or "")
            self.fail(
                f"180-second TP8 watchdog expired; signalled only owned process group {process.pid}.\n{output}"
            )
        finally:
            # Also stop surviving peers if spawn's error handling exits its
            # driver early. This PGID belongs exclusively to this test run.
            _signal_owned_group(process, signal.SIGKILL)
        print(output, end="", flush=True)
        self.assertEqual(process.returncode, 0, output)
        self.assertEqual(output.count("TP8_CMP_DISTRIBUTED_PASS "), WORLD_SIZE, output)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker-group"]:
        _worker_group()
    else:
        unittest.main()
