"""Real two-rank IPC, vocabulary and primitive Graph tests for all N=1..64.

These are primitive Graphs, not additional model capture buckets.
"""

import gc
import os
import socket
import sys
import unittest
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _embedding(owners, reducer, vocab_size=129):
    from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
        DecodeVocabEmbedding,
    )
    from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX
    from rtp_llm.utils.model_weight import W

    return DecodeVocabEmbedding(
        owners[W.embedding],
        reducer,
        rank=dist.get_rank(),
        vocab_size=vocab_size,
        scales=owners.get(W.embedding + FP8_STORAGE_SCALE_SUFFIX),
    )


def _loaded_vocab(full, rank, fp8=False):
    from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4VocabWeight
    from rtp_llm.utils.model_weight import CkptWeightInfo, W

    descriptor = Qwen4VocabWeight(
        W.embedding,
        [CkptWeightInfo("model.embed_tokens.weight")],
        vocab_size=full.shape[0],
        hidden_size=full.shape[1],
        fp8_storage=fp8,
    )
    config = SimpleNamespace(tp_size=2, tp_rank=rank)
    # Loader behavior is exercised independently with TensorSource.load tests.
    with torch.inference_mode():
        loaded = descriptor._split(full, config)
    return loaded, descriptor, config


def _check_vocab_hot_updates(rank, reducer, device):
    from rtp_llm.model_loader import weight_manager as manager_module
    from rtp_llm.model_loader.model_weight_info import ModelWeights
    from rtp_llm.utils.model_weight import W

    vocab, width = 129, 2560
    full = (
        torch.arange(vocab * width, device=device).reshape(vocab, width) % 11
    ).bfloat16()
    owners, descriptor, config = _loaded_vocab(full, rank)
    weights = ModelWeights(0, str(device), torch.bfloat16)
    weights.global_weights = owners
    target, draft = _embedding(owners, reducer), _embedding(owners, reducer)
    pointer = target.weight.data_ptr()
    assert target.weight is draft.weight is owners[W.embedding]
    loader = SimpleNamespace(
        _model_weights_info=SimpleNamespace(weights=[descriptor]),
        get_load_config=lambda: config,
        is_global_weight_local=lambda name: True,
    )
    manager = manager_module.WeightManager(device, weights, loader)
    request = dict(name="model.embed_tokens.weight", method="cuda_ipc", desc="00")
    ids = torch.tensor([0, 64, 128, 3], device=device)
    target(ids)
    draft(ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        target_output, draft_output = target(ids), draft(ids)
    reducer.finish_graph_capture()
    try:

        def update_local(value):
            # Deliver each rank's RPC separately: no update may wait for peers.
            with ExitStack() as stack:
                for name in (
                    "all_reduce",
                    "all_to_all_single",
                    "all_gather_object",
                    "all_gather",
                    "all_gather_into_tensor",
                    "barrier",
                ):
                    stack.enter_context(
                        patch.object(
                            dist,
                            name,
                            side_effect=AssertionError("update used collective"),
                        )
                    )
                stack.enter_context(
                    patch.object(manager, "_load_update_tensor", return_value=value)
                )
                manager.update(request)

        for gain in (2, 3):
            updated = full * gain
            incoming = updated if gain == 2 else updated.cpu().t().contiguous().t()
            for owner in (0, 1):
                if rank == owner:
                    update_local(incoming)
                dist.barrier()
            graph.replay()
            torch.testing.assert_close(target_output, updated[ids], rtol=0, atol=0)
            torch.testing.assert_close(draft_output, updated[ids], rtol=0, atol=0)
            assert pointer == target.weight.data_ptr() == draft.weight.data_ptr()
            if rank == 1:
                assert not bool(target.weight[-1].count_nonzero())

        before = target.weight.clone()
        for invalid in (full[:-1], full.float(), torch.nn.functional.pad(full, (0, 1))):
            for owner in (0, 1):
                if rank == owner:
                    try:
                        update_local(invalid)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("invalid full weight must be rejected")
                    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)
                dist.barrier()
        # Shared draft cannot update a non-owning alias.
        alias_manager = manager_module.WeightManager(
            device, weights, loader, (W.embedding,)
        )
        with patch.object(
            manager_module.CudaIpcHelper, "build_from_meta", return_value=full
        ):
            try:
                alias_manager.update(request)
            except PermissionError:
                pass
            else:
                raise AssertionError("alias update must be rejected")
            manager.update(request)
        graph.replay()
        torch.testing.assert_close(target_output, full[ids], rtol=0, atol=0)
        torch.testing.assert_close(draft_output, full[ids], rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()

    # The loading descriptor rejects FP8 updates before changing storage.
    fp8_owners, fp8_descriptor, _ = _loaded_vocab(full, rank, True)
    weights.global_weights = fp8_owners
    loader._model_weights_info.weights = [fp8_descriptor]
    before = fp8_owners[W.embedding].view(torch.uint8).clone()
    with patch.object(manager, "_load_update_tensor", return_value=full):
        try:
            manager.update(request)
        except PermissionError:
            pass
        else:
            raise AssertionError("quantized update must remain guarded")
    torch.testing.assert_close(
        fp8_owners[W.embedding].view(torch.uint8), before, rtol=0, atol=0
    )


def _worker(rank, port, hot_update_only=False):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
        Qwen4DecodeReduce,
    )

    collective_torch._group_map[collective_torch.Group.TP] = dist.group.WORLD
    collective_torch._group_map[collective_torch.Group.WORLD] = dist.group.WORLD
    collective_torch._group_map[collective_torch.Group.STAGE] = dist.group.WORLD
    collective_torch._initialized = True
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=2, dp_size=1, pp_size=1, world_size=2
    )
    group = collective_torch.Group.TP
    device = torch.device("cuda", rank)
    from flashinfer.comm import cuda_ipc, vllm_ar

    from rtp_llm.models_py.modules.qwen4_exp import decode_communication as module

    communicator = SimpleNamespace(
        group=dist.group.WORLD, device=device, disabled=False, cpp_small_enabled=True
    )
    parallelism = SimpleNamespace(
        role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
    )
    # Admission must reject incompatible peer policies before entering IPC setup.
    for field, local in (
        (
            "_qwen4_decode_communication_mode",
            "default" if rank == 0 else "vocab_parallel_ipc",
        ),
        ("_qwen4_target_decode_reduce", None),
    ):
        config = SimpleNamespace(hidden_size=2560)
        if field.endswith("mode") or rank == 0:
            setattr(config, field, local)
        with patch.object(module, "get_all_reduce", return_value=communicator):
            try:
                module.prepare_decode_reduce(config, parallelism)
            except ValueError:
                pass
            else:
                raise AssertionError("inconsistent rank admission must be rejected")
    # Fail allocation on just one rank, inside the real coordinated constructor.
    # Both ranks retain the loaded layout and remain able to use NCCL.
    malloc = cuda_ipc.cudart.cudaMalloc

    def allocate(size):
        if rank == 0:
            raise RuntimeError("injected rank-local CUDA allocation failure")
        return malloc(size)

    with (
        patch.object(module, "get_all_reduce", return_value=communicator),
        patch.object(cuda_ipc.cudart, "cudaMalloc", side_effect=allocate),
    ):
        assert (
            module.prepare_decode_reduce(SimpleNamespace(hidden_size=2560), parallelism)
            is None
        )
    register = vllm_ar.register_buffer

    def register_buffer(*args):
        if rank == 1:
            raise RuntimeError("injected rank-local IPC registration failure")
        return register(*args)

    with (
        patch.object(module, "get_all_reduce", return_value=communicator),
        patch.object(vllm_ar, "register_buffer", side_effect=register_buffer),
    ):
        assert (
            module.prepare_decode_reduce(SimpleNamespace(hidden_size=2560), parallelism)
            is None
        )
    value = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(value)
    torch.testing.assert_close(value, torch.tensor([3.0], device=device))
    draft_config = SimpleNamespace(hidden_size=2560, _qwen4_target_decode_reduce=None)
    with (
        patch.object(module, "get_all_reduce", return_value=communicator),
        patch.object(
            Qwen4DecodeReduce,
            "__init__",
            side_effect=AssertionError("draft retried failed IPC"),
        ),
    ):
        assert module.prepare_decode_reduce(draft_config, parallelism) is None

    reducer = Qwen4DecodeReduce(communicator)
    graphs = []
    try:
        if hot_update_only:
            _check_vocab_hot_updates(rank, reducer, device)
            if rank == 0:
                print(
                    "PASS: rank-local BF16 hot update, target/draft Graph replay, stable pointers, local failure and retry",
                    flush=True,
                )
            return
        # The loaded owner is consumed directly, including after IPC fallback.
        from rtp_llm.utils.model_weight import W

        full = torch.ones(4096, 2560, device=device, dtype=torch.bfloat16)
        owners, _, _ = _loaded_vocab(full, rank)
        del full
        original = owners[W.embedding]
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        target = _embedding(owners, reducer, 4096)
        draft_embedding = _embedding(owners, reducer, 4096)
        fallback = _embedding(owners, None, 4096)
        assert target.weight is draft_embedding.weight is fallback.weight is original
        assert torch.cuda.memory_allocated() == allocated_before
        ids = torch.tensor([0, 2048, 4095], device=device)
        expected = torch.ones(3, 2560, device=device, dtype=torch.bfloat16)
        for embedding in (target, draft_embedding, fallback):
            torch.testing.assert_close(embedding(ids), expected, rtol=0, atol=0)
        del target, draft_embedding, fallback, original, owners
        for rows in range(1, 65):
            value = torch.full(
                (rows, 2560), rank + 1, device=device, dtype=torch.bfloat16
            )
            original = value.clone()
            eager = reducer(value, group)
            torch.testing.assert_close(eager, torch.full_like(value, 3), rtol=0, atol=0)
            torch.testing.assert_close(value, original, rtol=0, atol=0)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = reducer(value, group)
            reducer.finish_graph_capture()
            graphs.append((graph, value, out))
        # Different sizes repeatedly reuse one workspace; inputs stay live and
        # are changed between replays to detect stale registration/staging.
        for sequence in (range(64), reversed(range(64)), range(0, 64, 3)):
            for index in sequence:
                graph, value, out = graphs[index]
                value.fill_(rank + index % 7)
                graph.replay()
                torch.testing.assert_close(
                    out, torch.full_like(out, 2 * (index % 7) + 1), rtol=0, atol=0
                )
        # Target and draft capture/register separate graphs on one ordered IPC
        # workspace. Closing either wrapper must leave the other graph usable.
        draft_config._qwen4_target_decode_reduce = reducer
        with patch.object(module, "get_all_reduce", return_value=communicator):
            draft = module.prepare_decode_reduce(draft_config, parallelism)
        assert draft is not reducer and draft.workspace is reducer.workspace
        for rows in (2, 8, 16, 32, 64):
            value = torch.full(
                (rows, 2560), rank + 1, device=device, dtype=torch.bfloat16
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = draft(value, group)
            draft.finish_graph_capture()
            graphs.append((graph, value, out))
            for shift in range(3):
                graphs[rows - 1][0].replay()
                value.fill_(rank + shift)
                graph.replay()
                torch.testing.assert_close(
                    out, torch.full_like(out, 2 * shift + 1), rtol=0, atol=0
                )
        draft.close()
        draft.close()
        assert not reducer.closed
        graphs[0][0].replay()
        torch.testing.assert_close(
            graphs[0][2], torch.ones_like(graphs[0][2]), rtol=0, atol=0
        )

        for noncontiguous in (False, True):
            storage = torch.full(
                (8 * 2560 * 2 + 8,), rank + 1, device=device, dtype=torch.bfloat16
            )
            value = (
                storage[rank : rank + 8 * 2560 * 2 : 2].view(8, 2560)
                if noncontiguous
                else storage[rank : rank + 8 * 2560].view(8, 2560)
            )
            with patch.object(
                collective_torch,
                "all_reduce",
                side_effect=AssertionError("rank-local layout changed backend"),
            ):
                actual = reducer(value, group)
                torch.testing.assert_close(
                    actual, torch.full_like(actual, 3), rtol=0, atol=0
                )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = reducer(value, group)
                reducer.finish_graph_capture()
                value.fill_(rank + 3)
                graph.replay()
                torch.testing.assert_close(
                    actual, torch.full_like(actual, 7), rtol=0, atol=0
                )
                graphs.append((graph, value, actual))

        # Both oversized and wrong-dtype calls preserve the NCCL fallback.
        for rows, dtype in ((65, torch.bfloat16), (8, torch.float32)):
            value = torch.full((rows, 2560), rank + 1, device=device, dtype=dtype)
            torch.testing.assert_close(
                reducer(value, group), torch.full_like(value, 3), rtol=0, atol=0
            )

        for fp8 in (False, True):
            vocab, half_width = 129, 1280
            full = (
                (
                    torch.arange(vocab * half_width * 2, device=device).reshape(
                        vocab, half_width * 2
                    )
                    % 15
                )
                - 7
            ).to(torch.bfloat16)
            owners, _, _ = _loaded_vocab(full, rank, fp8)
            if fp8:
                parts = []
                for source in full.chunk(2, dim=1):
                    scale = source.float().abs().amax().clamp_min(1e-12) / 448.0
                    raw = (
                        (source.float() / scale)
                        .clamp(-448, 448)
                        .to(torch.float8_e4m3fn)
                    )
                    parts.append((raw.bfloat16() * scale.bfloat16()).bfloat16())
                expected_weight = torch.cat(parts, dim=1)
            else:
                expected_weight = full
            embedding = _embedding(owners, reducer)
            shared = _embedding(owners, reducer)
            assert (
                shared.weight is embedding.weight
            ), "target/draft must share vocabulary storage"
            fallback = _embedding(owners, None)
            ids = torch.tensor([0, 64, 128, 3], device=device)
            fallback(ids)
            torch.cuda.synchronize()
            fallback_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(fallback_graph):
                fallback_output = fallback(ids)
            for token in (0, 64, 128):
                ids.fill_(token)
                fallback_graph.replay()
                torch.testing.assert_close(
                    fallback_output, expected_weight[ids], rtol=0, atol=0
                )
            graphs.append((fallback_graph, ids, fallback_output, fallback))
            for rows in range(1, 65):
                ids = (
                    torch.arange(rows, device=device, dtype=torch.int64)
                    * 17
                    % (vocab + 3)
                    - 1
                )
                mask = torch.arange(rows, device=device) % 5 != 0
                wanted = expected_weight[ids.clamp(0, vocab - 1)].clone()
                wanted[(ids < 0) | (ids >= vocab) | ~mask] = 0
                actual = embedding(ids, text_tokens_mask=mask)
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = embedding(ids, text_tokens_mask=mask)
                reducer.finish_graph_capture()
                for shift in (0, 1, 63, 64, 128):
                    ids.fill_(shift)
                    graph.replay()
                    wanted = expected_weight[ids].clone()
                    wanted[~mask] = 0
                    torch.testing.assert_close(output, wanted, rtol=0, atol=0)
                graphs.append((graph, ids, mask, output, embedding))
            empty = embedding(torch.empty(0, device=device, dtype=torch.int64))
            assert empty.shape == (0, 2560)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                "PASS: N=1..64 BF16 reduction, BF16/FP8 embedding, repeated Graph replay, boundary/fallback/shared-weight cases",
                flush=True,
            )
    finally:
        # On failure, let mp.spawn report the exception and terminate peers;
        # a cleanup collective would hide the original error behind a timeout.
        if sys.exc_info()[0] is None:
            # Graphs keep registered pointers alive until both ranks finish.
            torch.cuda.synchronize()
            dist.barrier()
            # NCCL teardown waits for captured graph callbacks. Local variables
            # can retain graphs even after clearing the ownership list.
            for captured in graphs:
                captured[0].reset()
            graphs.clear()
            reducer.workspace.destroy()
            dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and torch.cuda.get_device_capability(0) == (12, 0),
    "requires two SM120 GPUs",
)
class DecodeCommunicationSm120Test(unittest.TestCase):
    def test_vocab_store_mask_does_not_overflow_large_logical_rows(self):
        from rtp_llm.models_py.triton_kernels.common.decode_vocab_lookup import (
            _vocab_rows,
        )

        tokens = torch.zeros(1, device="cuda", dtype=torch.int32)
        weight = torch.ones(1, 2560, device="cuda", dtype=torch.bfloat16)
        # Launch only the first CTA: every access is within these small buffers,
        # while the runtime row count exercises the int32 N*D boundary.
        for rows in (64, 838860, 838861):
            output = torch.full((1024,), -7, device="cuda", dtype=torch.bfloat16)
            _vocab_rows[(1,)](
                tokens,
                weight,
                tokens,
                tokens,
                output,
                rows,
                START=0,
                LOCAL_V=1,
                V=1,
                D=2560,
                FP8=False,
                TOKEN_STRIDE=1,
                MASK_STRIDE=0,
                HAS_MASK=False,
                BLOCK=1024,
                num_warps=4,
            )
            torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)

    def test_two_rank_decode_communication(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=2, join=True)

    def test_two_rank_vocab_hot_updates(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port, True), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
