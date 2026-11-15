"""Real SM103 TP2 native AG, independent NCCL reference, no skip-green."""

import contextlib
import datetime
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _nccl_reference(value, group):
    # Independent of both GLM dispatch and TorchSymmMemCommunicator.
    contiguous = value.contiguous()
    output = torch.empty(
        (dist.get_world_size(group) * value.shape[0], *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_gather_into_tensor(output, contiguous, group=group)
    return output


def _assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(actual, expected), "native/fallback differs from NCCL bits"


def _worker(rank, store):
    from rtp_llm.models_py.distributed import collective_torch as collective
    from rtp_llm.models_py.distributed import glm53_collective_gemm as glm
    from rtp_llm.models_py.distributed import symm_mem
    from rtp_llm.models_py.modules.dsv4 import forward_metadata

    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability(rank) != (10, 3):
        raise RuntimeError("UNSUPPORTED: this target requires SM103 on both GPUs")
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + store,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    group = dist.group.WORLD
    pc = SimpleNamespace(
        world_size=2,
        world_rank=rank,
        local_world_size=2,
        local_rank=rank,
        tp_size=2,
        tp_rank=rank,
        dp_size=1,
        pp_size=1,
        role_type="PREFILL",
    )
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(glm, "_STATE", None))
            stack.enter_context(patch.object(collective, "_initialized", True))
            stack.enter_context(patch.object(collective, "_parallelism_config", pc))
            stack.enter_context(
                patch.dict(
                    collective._group_map,
                    {collective.Group.WORLD: group, collective.Group.TP: group},
                    clear=True,
                )
            )
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {
                        "GLM53_PREFILL_AG_GEMM": "1",
                        "GLM53_PREFILL_GEMM_RS": "0",
                        "GLM53_PREFILL_STABLE_RS": "0",
                        "GLM53_COLLECTIVE_MAX_TOKENS": "4096",
                    },
                )
            )
            token = forward_metadata._FORWARD_METADATA.set({})
            stack.callback(forward_metadata._FORWARD_METADATA.reset, token)
            # Exercise the actual startup topology agreement and communicator
            # once. No model loading and no fabricated multicast capability.
            glm.configure_glm53_collective_gemm(group, 128, pc, enable_cuda_graph=False)
            owner = glm._STATE.native_all_gather
            if owner is None or owner.communicator.disabled:
                raise RuntimeError(
                    "UNSUPPORTED: native AG startup declined; SM103 TP2 needs "
                    "actual same-node multicast and multimem_all_gather_out"
                )
            communicator = owner.communicator
            assert isinstance(communicator, symm_mem.TorchSymmMemCommunicator)
            assert symm_mem.get_symm_mem_communicator() is None

            with (
                patch.object(owner, "gather", wraps=owner.gather) as native,
                patch.object(
                    collective, "all_gather", wraps=collective.all_gather
                ) as fallback,
            ):
                for step, shape in enumerate(((8,), (7, 32), (2, 3, 8))):
                    value = (
                        torch.arange(1, 1 + math.prod(shape), device=device)
                        .reshape(shape)
                        .to(torch.bfloat16)
                    )
                    value.add_(rank * 17 + step * 5)
                    original = value.clone()
                    print(f"rank={rank} native shape={shape}", flush=True)
                    expected = _nccl_reference(value, group)
                    assert communicator.should_torch_symm_mem_allgather(value)
                    before = native.call_count
                    actual = glm.all_gather(value, collective.Group.TP)
                    _assert_exact(actual, expected)
                    _assert_exact(value, original)
                    assert native.call_count == before + 1
                    assert actual.data_ptr() != communicator.buffer.data_ptr()
                fallback.assert_not_called()

                # Matching logical payloads may have different storage offsets
                # on different ranks. Rank1 needs local aligned staging, while
                # both ranks must still enter the same native collective.
                value = torch.arange(8 + rank, device=device).to(torch.bfloat16)[rank:]
                assert value.is_contiguous() and value.numel() == 8
                assert value.data_ptr() % 16 == rank * 2
                expected = _nccl_reference(value, group)
                before = native.call_count
                actual = glm.all_gather(value, collective.Group.TP)
                _assert_exact(actual, expected)
                assert native.call_count == before + 1
                fallback.assert_not_called()

                value = torch.full(
                    (5, 8), rank + 1, dtype=torch.bfloat16, device=device
                )
                expected = _nccl_reference(value, group)
                for logical in (0, 7, 10):
                    before = native.call_count
                    actual = glm.all_gather_trim(value, logical, collective.Group.TP)
                    _assert_exact(actual, expected[:logical])
                    assert native.call_count == before + 1
                fallback.assert_not_called()

                # All ranks use the same miss. Check the actual ordinary main
                # collective, including typed payloads and zero-size tensors.
                misses = [
                    torch.full((3,), rank + 1, dtype=torch.bfloat16, device=device),
                    torch.full(
                        (1025, 2), rank + 1, dtype=torch.bfloat16, device=device
                    ),
                    torch.full((3, 8), rank + 1, dtype=torch.float32, device=device),
                    torch.full((3, 8), 127 + rank, dtype=torch.uint8, device=device),
                    torch.empty((0, 8), dtype=torch.bfloat16, device=device),
                ]
                # Strict capacity boundary, not a patched predicate/buffer size.
                shard_elements = communicator.max_size // (2 * torch.bfloat16.itemsize)
                misses.append(
                    torch.full(
                        (1, shard_elements),
                        rank + 3,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                for value in misses:
                    assert (
                        value.numel() == 0
                        or not communicator.should_torch_symm_mem_allgather(value)
                    )

                # Preserve the original failing 8-byte shard, plus the other
                # sub-vector sizes.
                # The shared communicator's 4-byte gate admits these; the
                # GLM wrapper must decline before launching a vectorized op.
                alignment_misses = [
                    torch.full((n,), rank + 5, dtype=torch.bfloat16, device=device)
                    for n in (2, 4, 6)
                ]
                for value in alignment_misses:
                    assert communicator.should_torch_symm_mem_allgather(value)
                    assert value.numel() * value.element_size() % 16 != 0
                for value in misses + alignment_misses:
                    print(
                        f"rank={rank} fallback shape={tuple(value.shape)} "
                        f"dtype={value.dtype} offset={value.storage_offset()}",
                        flush=True,
                    )
                    expected = _nccl_reference(value, group)
                    before_native = native.call_count
                    before_fallback = fallback.call_count
                    actual = glm.all_gather(value, collective.Group.TP)
                    _assert_exact(actual, expected)
                    assert native.call_count == before_native
                    assert fallback.call_count == before_fallback + 1
                    assert fallback.call_args.args[0] is value

                # Main rejects a non-contiguous shard. Preserve that error
                # contract, rather than changing the input to make it succeed.
                value = (
                    torch.arange(24, device=device).reshape(3, 8).to(torch.bfloat16).T
                )
                before_native = native.call_count
                case = unittest.TestCase()
                with case.assertRaisesRegex(ValueError, "Tensors must be contiguous"):
                    collective.all_gather(value, collective.Group.TP)
                before_fallback = fallback.call_count
                with case.assertRaisesRegex(ValueError, "Tensors must be contiguous"):
                    glm.all_gather(value, collective.Group.TP)
                assert native.call_count == before_native
                assert fallback.call_count == before_fallback + 1
                assert fallback.call_args.args[0] is value

                # No reference collective, host read, or stream sync between
                # these submissions: they must be ordered by the real owner.
                # Keep all outputs alive so later writes can expose aliases.
                torch.cuda.synchronize(device)
                streams = [torch.cuda.Stream(device=device) for _ in range(2)]
                retained = []
                before_native, before_fallback = native.call_count, fallback.call_count
                for step in range(24):
                    with torch.cuda.stream(streams[step % 2]):
                        value = torch.full(
                            (33 + step % 3, 512),
                            rank + step * 4 + 1,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        actual = glm.all_gather(value, collective.Group.TP)
                        retained.append((value, actual))
                for stream in streams:
                    stream.synchronize()
                assert native.call_count == before_native + len(retained)
                assert fallback.call_count == before_fallback
                assert len({out.data_ptr() for _, out in retained}) == len(retained)
                for value, actual in retained:
                    _assert_exact(actual, _nccl_reference(value, group))
                    assert actual.data_ptr() != communicator.buffer.data_ptr()
                torch.cuda.synchronize(device)
            assert symm_mem.get_symm_mem_communicator() is None
    finally:
        dist.destroy_process_group()


class Glm53NativeAllGatherIntegrationTest(unittest.TestCase):
    def test_sm103_tp2_native_fallback_trim_and_cross_stream_ownership(self):
        # Failure is deliberate on a misrouted pool or unavailable multicast;
        # this manual integration target must never report skip-green.
        self.assertEqual(os.environ.get("GPU_COUNT"), "2")
        self.assertIsNone(torch.version.hip)
        self.assertTrue(torch.cuda.is_available(), "UNSUPPORTED: CUDA unavailable")
        self.assertEqual(torch.cuda.device_count(), 2, "requires two locked GPUs")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _worker,
                args=(str(Path(directory) / "store"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
