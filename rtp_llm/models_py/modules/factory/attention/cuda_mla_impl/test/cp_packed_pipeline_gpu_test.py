import os
import socket
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_merged_cache_write(rank: int) -> None:
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.models_py.distributed.pynccl_cp import warmup_current_cp
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
        SparseMlaFp8CPOp,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
        MlaKVCacheWriteOp,
    )
    from rtp_llm.ops import KvCacheDataType
    from rtp_llm.ops.compute_ops import LayerKVCache

    count = 9 * dist.get_world_size()
    ckv = (
        torch.arange(count * 512, device="cuda").view(count, 512).remainder(37) * 0.37
    ).bfloat16()
    rope = torch.arange(count * 64, device="cuda").view(count, 64).bfloat16()
    mapping = torch.arange(count - 1, -1, -1, device="cuda", dtype=torch.int64)
    slots = torch.arange(count, device="cuda", dtype=torch.int64)
    slots[slots.remainder(dist.get_world_size()) != rank] = -1
    actual, expected = LayerKVCache(), LayerKVCache()
    actual.kv_cache_base = torch.zeros(
        ((count + 15) // 16, 16, 656), device="cuda", dtype=torch.uint8
    )
    expected.kv_cache_base = torch.zeros_like(actual.kv_cache_base)
    writer = MlaKVCacheWriteOp(KvCacheDataType.FP8)
    params = SimpleNamespace(slot_mapping=slots)
    writer.forward(ckv[mapping], rope[mapping], expected, params)
    op = SimpleNamespace(
        use_cuda_graph=False,
        kv_restore_unpad_indices=mapping,
        kv_cache_write_op=writer,
        mla_params=params,
        kv_cache_sharded=True,
        sharded_slot_mapping=slots,
        write_cache_store_impl=None,
        attn_inputs=SimpleNamespace(is_prefill=True, cache_store_inputs=None),
    )
    stream = warmup_current_cp(Group.TP)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        SparseMlaFp8CPOp.write_kv(
            op, ckv[rank * 9 : (rank + 1) * 9], rope[rank * 9 : (rank + 1) * 9], actual
        )
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(
        actual.kv_cache_base, expected.kv_cache_base, rtol=0, atol=0
    )


def check_joint_read(rank: int, scale_bytes: int) -> None:
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_indexer_transport import (
        start_indexer_read,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_packed_read import (
        KvReadTarget,
        PackedKvRead,
    )
    from rtp_llm.models_py.triton_kernels.sparse_mla.cp_cache_transport import PagedRows

    local, page, head = 9, 16, 128
    total = local * dist.get_world_size()
    values = (
        torch.arange(total * 512, device="cuda")
        .view(total, 512)
        .remainder(23)
        .to(torch.float8_e4m3fn)
    )
    scales = (
        torch.arange(total * 4, device="cuda").view(total, 4).remainder(4).float() + 1
    )
    rope = torch.arange(total * 64, device="cuda").view(total, 64).bfloat16()
    packed = torch.cat(
        (values.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)),
        dim=1,
    )
    source = torch.zeros((2, page, 656), device="cuda", dtype=torch.uint8)
    source[1, :local] = packed[rank * local : (rank + 1) * local]
    index_raw = (
        torch.arange(total * (head + scale_bytes), device="cuda")
        .to(torch.uint8)
        .view(total, head + scale_bytes)
    )
    index_cache = torch.zeros(
        (2, page, head + scale_bytes), device="cuda", dtype=torch.uint8
    )
    block = index_cache[1].view(-1)
    block[: local * head] = index_raw[rank * local : (rank + 1) * local, :head].reshape(
        -1
    )
    block[page * head : page * head + local * scale_bytes] = index_raw[
        rank * local : (rank + 1) * local, head:
    ].reshape(-1)
    table = torch.ones((1, 1), device="cuda", dtype=torch.int32)
    rows = torch.arange(local, device="cuda", dtype=torch.int64)
    mapping = torch.arange(total - 1, -1, -1, device="cuda", dtype=torch.int64)
    output = torch.empty((total, 576), device="cuda", dtype=torch.bfloat16)
    read = PackedKvRead(
        PagedRows(source, table, rows), KvReadTarget(mapping, output), True
    )
    params = SimpleNamespace(
        kv_cache_sharded=True,
        cp_size=dist.get_world_size(),
        total_kv_len=total,
        indexer_cp_plan=SimpleNamespace(restore_indices=mapping),
        indexer_transport_rows=rows,
        indexer_transport_table=table,
        packed_kv_transport=read,
    )
    pending = start_indexer_read(index_cache, params, head)
    assert pending is not None
    independent = torch.ones((64, 64), device="cuda")
    independent = independent @ independent
    key, sf = pending.wait()
    read.consume()
    expected = torch.cat(
        (
            (values.float().view(-1, 4, 128) * scales[:, :, None])
            .view(-1, 512)
            .bfloat16(),
            rope,
        ),
        dim=1,
    )[mapping]
    from flash_mla import flash_mla_sparse_fwd

    query = torch.randn((3, 64, 576), device="cuda", dtype=torch.bfloat16) * 0.01
    indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(3, 1)
    indices[indices >= total] = -1
    actual_attention = flash_mla_sparse_fwd(
        query, output[:, None], indices[:, None], 576**-0.5, d_v=512
    )[0]
    reference_attention = flash_mla_sparse_fwd(
        query, expected[:, None], indices[:, None], 576**-0.5, d_v=512
    )[0]
    torch.testing.assert_close(actual_attention, reference_attention, rtol=0, atol=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        key.view(torch.uint8), index_raw[mapping, :head], rtol=0, atol=0
    )
    torch.testing.assert_close(sf, index_raw[mapping, head:], rtol=0, atol=0)
    torch.testing.assert_close(independent, torch.full_like(independent, 64))
    read.close()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unsupported = start_indexer_read(index_cache, params, head)
    assert unsupported is None


def worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")
    size = int(os.environ.get("CP_TEST_WORLD_SIZE", "2"))
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=120),
    )
    from rtp_llm.models_py.distributed import collective_torch, pynccl_cp
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.cp_kv_prefetch_gpu_test import (
        run_case,
    )

    collective_torch._group_map[collective_torch.Group.DP_AND_TP] = dist.group.WORLD
    collective_torch._initialized = True
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=size, dp_size=1, world_size=size
    )
    producer = pynccl_cp.warmup_kv_overlap(dist.group.WORLD)
    pynccl_cp.warmup(dist.group.WORLD)
    real_event = torch.cuda.Event

    def timed_event(*args, **kwargs):
        return real_event(*args, **(kwargs | {"enable_timing": True}))

    try:
        with patch.object(torch.cuda, "Event", side_effect=timed_event):
            for cache_dtype in (torch.uint8, torch.float8_e4m3fn):
                for prefixes, lengths in (
                    ([1, 511, 1025, 0], [6, 513, 13, 0]),
                    ([512], [1]),
                    ([0], [1]),
                ):
                    result = run_case(rank, prefixes, lengths, producer, cache_dtype)
                    assert result["bit_exact"]
            for scale_bytes in (4, 16):
                check_joint_read(rank, scale_bytes)
            check_merged_cache_write(rank)
        dist.barrier()
        if rank == 0:
            print(
                "CP packed pipeline: staging, joint packets and real merged cache write passed"
            )
    finally:
        pynccl_cp.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["RTP_LLM_CP_PACKED_PIPELINE"] = "1"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        worker,
        args=(port,),
        nprocs=int(os.environ.get("CP_TEST_WORLD_SIZE", "2")),
        join=True,
    )
