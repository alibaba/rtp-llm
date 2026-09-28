"""Two-GPU NCCL correctness checks for TP MoE chunk scheduling.

This deliberately uses real CUDA math and NCCL, rather than mocks.  The tiny
module models the algebra used by ``GenericMoeLayer`` in unified pure-TP mode:
each rank creates a routed-expert partial result plus a gated shared-expert
partial result, adds them, and all-reduces that sum exactly once.  Its small
weights make it suitable for a two-GPU test without claiming Qwen performance.
"""

from __future__ import annotations

import os
import socket
import unittest
from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _TinyRoutedSharedMoe:
    """Real GPU routed/shared/gate compute with deterministic rank-local weights."""

    rank: int
    hidden_size: int
    experts: int
    device: torch.device

    def __post_init__(self) -> None:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(20260928 + self.rank)
        self.routed_weight = torch.randn(
            self.experts,
            self.hidden_size,
            self.hidden_size,
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.shared_weight = torch.randn(
            self.hidden_size,
            self.hidden_size,
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        )
        replicated = torch.Generator(device=self.device)
        replicated.manual_seed(20260928)
        self.gate_weight = torch.randn(
            self.hidden_size,
            1,
            generator=replicated,
            device=self.device,
            dtype=torch.bfloat16,
        )

    def partial(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Gathered expert weights make routing deliberately skew-sensitive.
        routed = torch.bmm(x.unsqueeze(1), self.routed_weight[expert_ids]).squeeze(1)
        shared = x @ self.shared_weight
        gate = torch.sigmoid(
            x @ self.gate_weight if gate_output is None else gate_output
        )
        return routed + gate * shared


def _install_tp_group() -> None:
    # The production async helper intentionally bypasses custom all-reduce.  Map
    # its TP group to this worker's real NCCL WORLD group without initializing
    # RTP's broader engine distributed state.
    from rtp_llm.models_py.distributed import collective_torch

    collective_torch._group_map[collective_torch.Group.TP] = dist.group.WORLD
    # Keep this isolated test out of RTP engine setup while exercising the
    # production async helper with an actual NCCL process group.
    collective_torch._get_group = lambda _group: dist.group.WORLD


def _run_chunked(
    module: _TinyRoutedSharedMoe,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    chunks: int,
    *,
    mode: str,
    direct_output: bool = False,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    from rtp_llm.models_py.model_desc import generic_moe

    class _ChunkHarness(generic_moe.GenericMoeLayer):
        def __init__(self) -> None:
            # Do not construct model weights; inherited _forward_tp_chunks is
            # the production code under test and _forward_impl below supplies
            # real small routed/shared/gate CUDA math for each chunk.
            torch.nn.Module.__init__(self)
            self.tp_chunk_config = SimpleNamespace(chunks=chunks, mode=mode)
            self.tp_prefill_config = SimpleNamespace(
                backend="default", direct_output=direct_output
            )
            self.shared_expert_gate = lambda hidden: hidden @ module.gate_weight

        def _route(self, hidden_states):
            return (
                torch.ones_like(expert_ids[:, None], dtype=torch.float32),
                expert_ids[:, None],
            )

        def _forward_impl(
            self,
            hidden_states: torch.Tensor,
            *,
            skip_final_allreduce: bool = False,
            routing=None,
            shared_gate_output=None,
            tp_prefill_backend="default",
            output_tensor=None,
        ) -> torch.Tensor:
            assert skip_final_allreduce
            ids = routing[1].flatten()
            self.token_offset += hidden_states.size(0)
            # This is the GenericMoe unified boundary: routed + gated shared
            # before the collective, never an MoE-only reduction plus shared add.
            partial = module.partial(
                hidden_states, ids, shared_gate_output
            ).contiguous()
            if output_tensor is not None:
                output_tensor.copy_(partial)
                return output_tensor
            return partial

    harness = _ChunkHarness()
    harness.token_offset = 0
    launched: list[dict[str, object]] = []
    real_all_reduce_async = generic_moe.all_reduce_async

    def _track_real_nccl(tensor: torch.Tensor, group, *, inplace: bool = True):
        # Record scalar metadata only. Holding the tensor here would make this
        # test accidentally satisfy the production PendingAllReduce lifetime
        # contract. The wrapped function still launches the real NCCL work.
        launched.append(
            {
                "shape": tuple(tensor.shape),
                "is_cuda": tensor.is_cuda,
                "data_ptr": tensor.data_ptr(),
                "storage_ptr": tensor.untyped_storage().data_ptr(),
            }
        )
        return real_all_reduce_async(tensor, group, inplace=inplace)

    generic_moe.all_reduce_async = _track_real_nccl
    try:
        result = harness._forward_tp_chunks(x, use_fusion=direct_output)
    finally:
        generic_moe.all_reduce_async = real_all_reduce_async
    assert harness.token_offset == x.size(0)
    # ``launched`` is metadata for the actual NCCL inputs; it does not extend
    # their lifetime beyond the production PendingAllReduce handles.
    return result, launched


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        device = torch.device("cuda", rank)
        _install_tp_group()
        tokens, hidden, experts, chunk_size = 13, 32, 4, 4
        # Same post-norm hidden states on every TP rank; rank-local weights model
        # TP shards.  The last singleton chunk exercises the tail path.
        generator = torch.Generator(device=device)
        generator.manual_seed(73)
        x = torch.randn(
            tokens,
            hidden,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        # 10/13 tokens choose expert 0; the remaining ids prove skew does not
        # change the chunked algebra.
        expert_ids = torch.tensor([0] * 10 + [1, 3, 2], device=device, dtype=torch.long)
        module = _TinyRoutedSharedMoe(rank, hidden, experts, device)

        reference = module.partial(x, expert_ids).contiguous()
        dist.all_reduce(reference, group=dist.group.WORLD)
        serial, serial_launches = _run_chunked(
            module, x, expert_ids, chunk_size, mode="serial"
        )
        overlap, overlap_launches = _run_chunked(
            module, x, expert_ids, chunk_size, mode="overlap"
        )

        assert len(overlap_launches) == 4
        assert overlap_launches[-1]["shape"] == (1, hidden)
        # All overlap chunks are concurrently in flight, so each must have
        # distinct input storage. The list contains no tensor references.
        assert len({entry["data_ptr"] for entry in overlap_launches}) == 4
        assert all(
            bool(entry["is_cuda"]) for entry in serial_launches + overlap_launches
        )
        # Results are already concatenated by _forward_tp_chunks. Allocate
        # unrelated storage before comparison without retaining async inputs.
        _ = torch.empty((1024, hidden), device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(serial, reference, rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(overlap, reference, rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(overlap, serial, rtol=0, atol=0)
        for mode in ("serial", "overlap"):
            direct, launches = _run_chunked(
                module, x, expert_ids, chunk_size, mode=mode, direct_output=True
            )
            assert len(launches) == 4
            assert len({item["data_ptr"] for item in launches}) == 4
            assert len({item["storage_ptr"] for item in launches}) == 1
            assert direct.data_ptr() == launches[0]["data_ptr"]
            torch.testing.assert_close(direct, serial, rtol=0, atol=0)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class MoeTpChunkingCudaTest(unittest.TestCase):
    def test_real_nccl_tp2_chunked_routed_shared_gate(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA GPUs for real NCCL TP2 coverage")
        mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
