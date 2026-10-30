"""Role-gated MegaA2A and Query-CP local Decode through the DCP fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from rtp_llm.models.kimi_k3.kimi_k31 import KimiK31ModelConfig
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    destroy_distributed_environment,
    init_distributed_environment,
    reduce_scatter_bf16,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_execution import (
    DsaKVExecution,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
    DsaPageRRParams,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_page_rr_impl import (
    PageRRSparseMlaOp,
    QueryCpLayout,
)
from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA
from rtp_llm.models_py.modules.kimi_k3.dsa_model_adapter import KimiK31ModelAdapter
from rtp_llm.ops import (
    AttentionConfigs,
    KvCacheDataType,
    NcclCommConfig,
    ParallelismConfig,
    RoleType,
)
from rtp_llm.test.utils.port_util import PortManager

HEADS, LATENT, ROPE, VALUE, PAGE, TOPK = 96, 512, 64, 128, 128, 2048


def _make_owner(parallel, rows, *, indexer_use_fp8=True):
    model = torch.nn.Module()
    cp = parallel.prefill_cp_config
    model.prefill_query_cp = (
        parallel.role_type == RoleType.PREFILL
        and cp.is_enabled()
        and cp.preserve_tp_layout
    )
    model.config = SimpleNamespace(
        is_mtp=False,
        k3_runtime_config=SimpleNamespace(
            attn_res_block_size=0,
            dsa=SimpleNamespace(
                index_num_heads=32,
                index_head_dim=128,
                indexer_use_fp8=indexer_use_fp8,
                index_topk=TOPK,
            ),
        ),
        attn_config=SimpleNamespace(
            kv_cache_dtype=KvCacheDataType.BASE,
            head_num=HEADS,
            kv_lora_rank=LATENT,
            rope_head_dim=ROPE,
            v_head_dim=VALUE,
            q_lora_rank=0,
            nope_head_dim=128,
        ),
        hidden_size=0,
        num_layers=0,
        prefill_kv_workspace_budget_bytes=8 * 1024**3,
        gen_num_per_cycle=0,
    )
    model.parallelism_config = parallel
    model._max_generate_batch_size = rows
    model.layers = []
    model.weight = SimpleNamespace(weights=[])
    KimiK31ModelAdapter._setup_dsa(model)
    return model.dsa_workspace


def _make_operator(parallel, rows, *, graph):
    attention = AttentionConfigs()
    attention.tokens_per_block = attention.kernel_tokens_per_block = PAGE
    owner = _make_owner(parallel, rows)
    expect_mega = parallel.role_type == RoleType.DECODE
    assert owner.mega_a2a == expect_mega
    assert owner.query_heads == HEADS
    weights = torch.zeros((HEADS, LATENT, VALUE), device="cuda", dtype=torch.bfloat16)
    head_scale = torch.arange(1, HEADS + 1, device="cuda", dtype=torch.float32) / 16
    weights[:, 0, :] = head_scale[:, None]
    owner.value_weights[0] = weights

    table = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")[:, None]
    inputs = SimpleNamespace(
        is_prefill=False,
        is_target_verify=False,
        is_mtp_draft_update=False,
        is_cuda_graph=graph,
        is_warmup=graph,
        input_lengths=torch.ones(rows, dtype=torch.int32),
        physical_token_count=rows,
        sequence_lengths_plus_1_device=torch.full(
            (rows,), PAGE + 1, dtype=torch.int32, device="cuda"
        ),
        kv_cache_kernel_block_id_device=table,
        valid_token_mask=None,
    )
    params = DsaPageRRParams(
        attention,
        parallel,
        max_seq_len=PAGE * 2,
        workspace_owner=owner,
        is_cuda_graph=graph,
    )
    params.prepare(inputs)
    params.sfa_execution = DsaKVExecution(params, "sfa")
    assert not params.is_prefill
    assert params.sfa_execution.blocks
    op = PageRRSparseMlaOp(
        num_heads=HEADS if owner.prefill_query_cp else HEADS // parallel.tp_size,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        qk_nope_head_dim=128,
        page_size=PAGE,
        softmax_extra_scale=1.0,
        top_k=TOPK,
        parallelism_config=parallel,
        workspace_owner=owner,
    )
    op.plan(params, params.block_table)
    assert isinstance(op.communicator, QueryCpLayout) == owner.prefill_query_cp
    assert (op.mega_a2a is not None) == expect_mega
    q = torch.zeros((rows, HEADS, LATENT + ROPE), dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(
        ((rows + 1) * PAGE, LATENT + ROPE), dtype=torch.bfloat16, device="cuda"
    )
    # Q=0 gives two globally selected keys equal attention weight. Page 0 is
    # owned by rank 0, page 1 by rank 1. Each request has its own physical page,
    # so the oracle checks both head identity and request-row identity.
    _fill_cache(kv, rows, 2 + parallel.tp_rank * 4)
    indices = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
    indices[:, 0], indices[:, 1] = 0, PAGE
    return op, q, kv, indices, head_scale


def _fill_cache(kv, rows, value):
    row_scale = torch.arange(1, rows + 1, dtype=torch.float32, device="cuda")
    kv.view(rows + 1, PAGE, LATENT + ROPE)[1:, :, 0] = value * row_scale[:, None]


def _reference(rows, head_scale, values=(2, 6)):
    row_scale = torch.arange(1, rows + 1, dtype=torch.float32, device="cuda")
    product = row_scale[:, None] * head_scale[None, :]
    # Both merge paths consume BF16 V-up partials before the FP32 equal-LSE merge.
    partials = [torch.mul(product, value).bfloat16().float() for value in values]
    merged = ((partials[0] + partials[1]) / 2).bfloat16()
    return merged[:, :, None].expand(rows, HEADS, VALUE)


def _check_gate_and_projection(rank, world, values, expected):
    # Isolate only the QKV producer boundary. The SFA result above came through
    # the real FlashMLA kernel, V-up, slot mapper and standard DCP merge.
    # Run the production gate/projection/row-selection consumer unchanged.
    attention = object.__new__(KimiK31DSA)
    torch.nn.Module.__init__(attention)
    attention.prologue = "integration"
    attention.prefill_query_cp = True
    attention.tp_size, attention.tp_rank = world, rank
    attention._fp8_output_gate = False
    attention._mtp_bf16_collectives = None
    attention.output = torch.nn.Identity()
    flattened = values.reshape(values.shape[0], HEADS * VALUE)
    attention._decode_prologue = lambda *_: (flattened, torch.zeros_like(flattened))
    actual = attention(
        flattened,
        {"full": SimpleNamespace(workspace_owner=None)},
        None,
        attention_inputs=SimpleNamespace(valid_token_mask=None),
        metadata=SimpleNamespace(query_layout=None, outer_layout=None),
    )
    expected = expected.reshape(values.shape[0], HEADS * VALUE)
    expected = expected.chunk(world, 0)[rank] / 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03125)


@torch.inference_mode()
def _worker(rank, port):
    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_size = parallel.tp_size = parallel.ep_size = parallel.ffn_tp_size = 2
    parallel.local_world_size = 2
    parallel.world_rank = parallel.local_rank = parallel.tp_rank = rank
    parallel.ep_rank = rank
    parallel.ffn_tp_rank = rank
    parallel.role_type = RoleType.PREFILL
    parallel.prefill_cp_config.kv_cache_sharded = True
    # Exercise the actual model-level role/layout policy, rather than overriding
    # attention heads or Query-CP on a DECODE-role model.
    KimiK31ModelConfig.configure_parallelism(None, parallel)
    init_distributed_environment(
        parallel, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=120
    )
    graph = None
    try:
        # The ordinary DECODE/DCP contract keeps only this rank's heads.
        # Verify that control with the real MegaA2A kernel before the Query-CP
        # fallback, so a kernel/setup failure cannot masquerade as red.
        decode_parallel = ParallelismConfig()
        decode_parallel.world_size = decode_parallel.tp_size = 2
        decode_parallel.ep_size = 2
        decode_parallel.ffn_tp_size = decode_parallel.local_world_size = 2
        decode_parallel.world_rank = decode_parallel.tp_rank = rank
        decode_parallel.local_rank = rank
        decode_parallel.ep_rank = decode_parallel.ffn_tp_rank = rank
        decode_parallel.role_type = RoleType.DECODE
        decode_parallel.prefill_cp_config.kv_cache_sharded = True
        op, q, kv, indices, head_scale = _make_operator(decode_parallel, 4, graph=False)
        control = op.forward(q, kv, indices)
        assert tuple(control.shape) == (4, HEADS // 2, VALUE), tuple(control.shape)
        expected_control = _reference(4, head_scale).chunk(2, dim=1)[rank]
        torch.testing.assert_close(control, expected_control, rtol=0, atol=0.03125)
        # A standalone SFA call has not run o_proj's consuming TP collective.
        # Use the production ReduceScatter transport before another operator
        # reuses MegaA2A's symmetric sources; a rank-local assertion is not enough.
        reduce_scatter_bf16(control.reshape(4, -1), Group.TP)

        for rows in (2, 4):
            op, q, kv, indices, head_scale = _make_operator(parallel, rows, graph=False)
            actual = op.forward(q, kv, indices)
            # PREFILL-role local Decode must use standard DCP and retain every
            # head, including when rows=TP could hide a bad reshape by broadcasting.
            assert tuple(actual.shape) == (rows, HEADS, VALUE), tuple(actual.shape)
            expected = _reference(rows, head_scale)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0.03125)
            _check_gate_and_projection(rank, parallel.tp_size, actual, expected)
            prepared = torch.nn.functional.pad(q, (0, 0, 0, op.kernel_heads - HEADS))
            actual_prepared = op.forward(
                q=prepared, kv=kv, topk_indices=indices, prepared_query=True
            )
            torch.testing.assert_close(actual_prepared, expected, rtol=0, atol=0.03125)

        # Capture/replay the same non-Prefill phase that a PREFILL wrapper can
        # receive during ordinary decode graph initialization or local Decode.
        op, q, kv, indices, head_scale = _make_operator(parallel, 4, graph=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                op.forward(q, kv, indices)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = op.forward(q, kv, indices)
        torch.cuda.current_stream().wait_stream(stream)
        _fill_cache(kv, 4, 3 + rank * 6)
        graph.replay()
        expected = _reference(4, head_scale, (3, 9))
        torch.testing.assert_close(captured, expected, rtol=0, atol=0.0625)
    finally:
        # Release NCCL's captured communicator on success and assertion failure.
        # Reset explicitly: an exception traceback may still hold the Python
        # graph even after this worker scope exits.
        if graph is not None:
            graph.reset()
            graph = None
    # On failure let mp.spawn terminate the other worker instead of hiding the
    # original exception in graceful shutdown while a peer waits in a collective.
    destroy_distributed_environment()


class KimiK31MegaQueryCpIntegrationTest(unittest.TestCase):
    def test_mega_a2a_requires_decode_role_fp8_indexer_and_multiple_ranks(self):
        for role, tp, fp8, enabled in (
            (RoleType.DECODE, 2, True, True),
            (RoleType.PREFILL, 2, True, False),
            (RoleType.PDFUSION, 2, True, False),
            (RoleType.DECODE, 1, True, False),
            (RoleType.DECODE, 2, False, False),
        ):
            with self.subTest(role=role, tp=tp, fp8=fp8):
                parallel = ParallelismConfig()
                parallel.role_type, parallel.tp_size = role, tp
                parallel.prefill_cp_config.kv_cache_sharded = True
                KimiK31ModelConfig.configure_parallelism(None, parallel)
                with patch("torch.cuda.current_device", return_value=0):
                    owner = _make_owner(parallel, 4, indexer_use_fp8=fp8)
                self.assertEqual(owner.mega_a2a, enabled)

    def test_prefill_role_local_decode_falls_back_with_every_head(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2, "requires two SM10x GPUs")
        ports, locks = PortManager().get_consecutive_ports(1)
        try:
            mp.spawn(_worker, args=(ports[0],), nprocs=2, join=True)
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
