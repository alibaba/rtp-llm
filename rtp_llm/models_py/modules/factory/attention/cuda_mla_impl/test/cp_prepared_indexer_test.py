from types import SimpleNamespace
from unittest import SkipTest

import torch


def test_prepared_cp_keys_preserve_query_scale_order_and_exact_topk() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise SkipTest("Hopper or newer is required for FP8 MQA")
    from deep_gemm.utils import per_token_cast_to_fp8
    from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
    from rtp_llm.ops.compute_ops import LayerKVCache

    op = IndexerOp(32, 128, 512, 64)
    keys = torch.zeros((1024, 128), device="cuda", dtype=torch.bfloat16)
    keys[:512, :32] = 1
    keys[512:, :32] = -1
    keys[:512, -32:] = -1
    keys[512:, -32:] = 1
    key_fp8, key_sf = per_token_cast_to_fp8(
        keys,
        use_ue8m0=op.use_mxfp8,
        gran_k=32 if op.use_mxfp8 else 128,
        use_packed_ue8m0=op.use_mxfp8,
    )
    queries = torch.zeros((2, 32, 128), device="cuda", dtype=torch.bfloat16)
    queries[0, :, :32] = 8
    queries[0, :, -32:] = 1
    queries[1, :, :32] = 1
    queries[1, :, -32:] = 8
    query_fp8, query_sf = op.quant_q_only(queries)
    weights = torch.ones(
        (2, 32, 1),
        device="cuda",
        dtype=torch.bfloat16 if op.use_mxfp8 else torch.float32,
    )
    if not op.use_mxfp8:
        weights = weights * query_sf
    selected = torch.tensor([1, 0], device="cuda", dtype=torch.int64)
    starts = torch.zeros(2, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, 1024)
    inputs = SimpleNamespace(
        kv_cache_kernel_block_id_device=torch.zeros(
            (1, 1), device="cuda", dtype=torch.int32
        )
    )
    actual = op._get_topk_ragged_cp(
        query_fp8,
        weights,
        LayerKVCache(),
        SimpleNamespace(prefill_max_kv_len=1024),
        inputs,
        selected,
        torch.tensor([0, 1024], device="cuda", dtype=torch.int32),
        1024,
        starts,
        ends,
        ends,
        starts,
        kv_cache_sharded=True,
        q_scale=query_sf if op.use_mxfp8 else None,
        prepared_kv=(key_fp8, key_sf.contiguous().view(torch.uint8)),
    )
    expected = torch.stack(
        (torch.arange(512, 1024, device="cuda"), torch.arange(512, device="cuda"))
    ).int()
    torch.testing.assert_close(actual.sort(dim=1).values, expected, rtol=0, atol=0)


if __name__ == "__main__":
    test_prepared_cp_keys_preserve_query_scale_order_and_exact_topk()
    print("CP prepared Indexer: exact ranking and query-scale order passed")
