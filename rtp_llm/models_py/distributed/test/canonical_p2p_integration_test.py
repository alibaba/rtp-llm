# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.distributed.canonical_p2p import init_canonical_p2p


def _worker():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
    group = dist.group.WORLD
    device = torch.device("cuda", rank)
    for label, enabled, disabled in (
        ("default", False, None),
        ("rank_mismatch", rank != 7, None),
        ("explicit_disabled", True, True),
    ):
        with patch.dict(
            os.environ, {"RTP_LLM_CANONICAL_P2P_ALL_REDUCE": "1" if enabled else "0"}
        ):
            result = init_canonical_p2p(
                group, device, disable_custom_all_reduce=disabled
            )
            if result is not None:
                raise RuntimeError("init fallback failed " + label)
    # One rank without peer access must vote the whole group back to NCCL.
    with patch.dict(os.environ, {"RTP_LLM_CANONICAL_P2P_ALL_REDUCE": "1"}):
        if rank == 7:
            with patch("torch.cuda.can_device_access_peer", return_value=False):
                result = init_canonical_p2p(group, device)
        else:
            result = init_canonical_p2p(group, device)
        if result is not None:
            raise RuntimeError("peer availability vote split the group")
        communicator = init_canonical_p2p(group, device)
    if communicator is None:
        raise RuntimeError("requested supported communicator unavailable")
    from types import SimpleNamespace

    collective._canonical_p2p = communicator
    collective._group_map[collective.Group.TP] = group
    collective._group_map[collective.Group.DP_AND_TP] = group
    collective._initialized = True
    collective._parallelism_config = SimpleNamespace(
        tp_size=8,
        dp_size=1,
        world_size=8,
        local_world_size=8,
        tp_rank=rank,
        local_rank=rank,
        world_rank=rank,
        dp_rank=0,
    )
    import sys

    compute_ops = SimpleNamespace(register_comm_ops=MagicMock())
    with patch.dict(sys.modules, {"librtp_compute_ops": compute_ops}):
        collective._register_process_groups_to_cpp()
    cpp_sum = compute_ops.register_comm_ops.call_args.args[1]
    with patch.object(
        communicator, "all_reduce", wraps=communicator.all_reduce
    ) as fast:
        for has_dest in (False, True):
            x = torch.full((257,), rank + 1.0, dtype=torch.bfloat16, device=device)
            dest = torch.empty_like(x) if has_dest else None
            output = cpp_sum(x, 0, 0, dest)
            if output is not (dest if has_dest else x):
                raise RuntimeError("C++ callback output identity changed")
            torch.testing.assert_close(
                output, torch.full_like(output, 36), atol=0, rtol=0
            )
            if has_dest:
                torch.testing.assert_close(
                    x, torch.full_like(x, rank + 1), atol=0, rtol=0
                )
        if fast.call_count != 2:
            raise RuntimeError("C++ TP SUM did not use canonical peer sums")
        x = torch.full((257,), rank + 1.0, dtype=torch.bfloat16, device=device)
        output = cpp_sum(x, 2, 0, None)
        torch.testing.assert_close(output, torch.full_like(output, 8), atol=0, rtol=0)
        x.fill_(rank + 1)
        output = cpp_sum(x, 0, 2, None)
        torch.testing.assert_close(output, torch.full_like(output, 36), atol=0, rtol=0)
        if fast.call_count != 2:
            raise RuntimeError("non-SUM or non-TP callback selected canonical path")
    # The model uses full tiles. Dynamic masks previously split each BF16 pair
    # into scalar peer reads; verify the optimized specialization's PTX.
    compiled = communicator.kernel.warmup(
        *communicator.peers,
        torch.empty(2560, dtype=torch.bfloat16, device=device),
        2560,
        BLOCK=256,
        ALIGNED=True,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(10,),
    )
    ptx = compiled.asm["ptx"]
    if "ld.global.cg.b32" not in ptx or "ld.global.cg.b16" in ptx:
        raise RuntimeError("full-tile peer reads were not vectorized")
    rows = []
    generator = torch.Generator(device=device).manual_seed(12433 + rank)
    for count in (1, 257, 2560, 10240, 20480):
        for inplace in (True, False):
            x = torch.randn(
                count, dtype=torch.bfloat16, device=device, generator=generator
            )
            old = x.clone()
            shards = [torch.empty_like(x) for _ in range(8)]
            dist.all_gather(shards, x)
            reference = torch.stack(shards).float().sum(0).bfloat16()
            output = collective.all_reduce(x, collective.Group.TP, inplace=inplace)
            torch.testing.assert_close(output, reference, atol=0.001, rtol=0.001)
            if inplace and output is not x:
                raise RuntimeError("inplace identity changed")
            if not inplace and (
                output.data_ptr() == x.data_ptr() or not torch.equal(x, old)
            ):
                raise RuntimeError("out-of-place input changed")
            outputs = [torch.empty_like(output) for _ in range(8)]
            dist.all_gather(outputs, output)
            if not all(torch.equal(output, y) for y in outputs):
                raise RuntimeError("rank result mismatch")
            rows.append(
                dict(
                    count=count,
                    inplace=inplace,
                    max_abs=(output.float() - reference.float()).abs().max().item(),
                )
            )
        for path in ("python", "cpp"):
            for inplace in (True, False):
                x = torch.zeros(count, dtype=torch.bfloat16, device=device)
                barrier = torch.zeros(1, device=device)
                dest = torch.empty_like(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    dist.all_reduce(barrier)
                    output = (
                        collective.all_reduce(x, collective.Group.TP, inplace=inplace)
                        if path == "python"
                        else cpp_sum(x, 0, 0, None if inplace else dest)
                    )
                for iteration in range(16):
                    x.fill_((rank + iteration) % 9 - 4)
                    expected = sum((r + iteration) % 9 - 4 for r in range(8))
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        output, torch.full_like(output, expected), atol=0, rtol=0
                    )
                del graph
    # Unsupported inputs retain the existing backend, including its errors.
    fallback_contracts = []
    for x in (
        torch.full((257,), rank + 1.0, device=device),
        torch.full((20481,), rank + 1.0, dtype=torch.bfloat16, device=device),
        torch.full((2, 257), rank + 1.0, dtype=torch.bfloat16, device=device).t(),
    ):
        if communicator.should_use(x):
            raise RuntimeError("unsupported input selected")
        outputs, errors = [], []
        for selected in (None, communicator):
            collective._canonical_p2p = selected
            try:
                outputs.append(
                    collective.all_reduce(x, collective.Group.TP, inplace=False)
                )
                errors.append(None)
            except (ValueError, RuntimeError) as error:
                outputs.append(None)
                errors.append((type(error).__name__, str(error)))
        collective._canonical_p2p = communicator
        if errors[0] != errors[1]:
            raise RuntimeError("fallback error contract changed")
        if errors[0] is None:
            torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
            torch.testing.assert_close(
                outputs[1], torch.full_like(outputs[1], 36), atol=0, rtol=0
            )
        torch.testing.assert_close(x, torch.full_like(x, rank + 1), atol=0, rtol=0)
        fallback_contracts.append(
            dict(
                dtype=str(x.dtype),
                elements=x.numel(),
                contiguous=x.is_contiguous(),
                baseline_error=errors[0],
                candidate_error=errors[1],
            )
        )
    all_rows = [None] * 8
    dist.all_gather_object(all_rows, rows)
    if rank == 0:
        out = os.environ["RTP_CANONICAL_P2P_TEST_OUT"]
        open(out, "w").write(
            json.dumps(
                dict(
                    passed=True,
                    ranks=8,
                    full_tile_vector_load_bits=32,
                    rows=all_rows,
                    changing_graph_replays=320,
                    initialization_fallback_cases=4,
                    unsupported_cases=3,
                    cpp_sum_cases=2,
                    cpp_other_op_mode_cases=2,
                    fallback_contracts=fallback_contracts,
                ),
                indent=2,
            )
        )
        print("integration GPU gates passed", flush=True)
    collective._canonical_p2p = None
    del communicator
    dist.destroy_process_group()


class CanonicalPeerAllReduceIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.device_count() == 8
        and torch.version.hip is None
        and all(torch.cuda.get_device_capability(r) == (12, 0) for r in range(8)),
        "requires eight visible SM120 GPUs",
    )
    def test_precision_graph_reuse_and_fallback(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["RTP_CANONICAL_P2P_TEST_OUT"] = str(Path(directory) / "result.json")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    "--nproc-per-node=8",
                    __file__,
                ],
                env=env,
                check=True,
                timeout=600,
            )
            result = json.loads(Path(env["RTP_CANONICAL_P2P_TEST_OUT"]).read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["ranks"], 8)


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker()
    else:
        unittest.main()
