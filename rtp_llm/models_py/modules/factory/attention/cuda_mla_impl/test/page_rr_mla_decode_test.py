"""Eight-rank Page-RR histories against dense MLA math and native FP8 attention."""

import itertools
import os
import sys
import unittest
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import torch
import torch.multiprocessing as mp

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    destroy_distributed_environment,
    get_process_group,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_decode import (
    PageRRMlaDecodeImpl,
)
from rtp_llm.ops import (
    AttentionConfigs,
    CPRotateMethod,
    HybridAttentionType,
    KvCacheDataType,
    NcclCommConfig,
    ParallelismConfig,
    RoleType,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.test.utils.port_util import PortManager
from rtp_llm.utils.model_weight import W

# Give DeepGEMM JIT an absolute, writable cache inside the Bazel test sandbox.
_TEST_TMPDIR = os.environ.get("TEST_TMPDIR")
if _TEST_TMPDIR:
    os.environ.setdefault("DG_JIT_CACHE_DIR", os.path.join(_TEST_TMPDIR, "deep_gemm"))


def _run_history(rank, port, queries=7, mla_layers=1, fp8=False):
    from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner

    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_rank = parallel.local_rank = parallel.tp_rank = rank
    parallel.world_size = parallel.tp_size = 8
    parallel.dp_size = 1
    init_distributed_environment(
        parallel, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=120
    )
    try:
        torch.manual_seed(907)
        batch, heads, page, kernel_page = 2, 96, 4096, 128
        pages_per_block = page // kernel_page
        table_width = 2 * pages_per_block
        max_seq_len = 9 * page
        latent, nope, rope, value = 512, 128, 64, 128
        tokens = batch * queries
        dtype, device = torch.bfloat16, torch.device("cuda", rank)
        cache_dtype = torch.float8_e4m3fn if fp8 else dtype
        q_scale, kv_scale = (0.5, 0.25) if fp8 else (1.0, 1.0)
        if fp8:
            from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
                _get_tokenspeed_workspace,
                _load_tokenspeed_mla,
            )
            assert _load_tokenspeed_mla()
            from tokenspeed_mla.mla_decode import tokenspeed_mla_decode

            dense_workspace = _get_tokenspeed_workspace(device, heads, latent, 1)
            owned_positions = [p for p in range(max_seq_len) if p // page % 8 == rank]
            dense_pages = torch.arange(
                batch * len(owned_positions) // kernel_page, dtype=torch.int32, device=device
            ).view(batch, -1).repeat_interleave(queries, dim=0)
        base_cache = torch.randn(batch, max_seq_len, latent + rope, device=device).to(
            dtype
        )
        query = torch.randn(tokens, heads, nope + rope, device=device).to(dtype)
        append = torch.randn(tokens, latent + rope, device=device).to(dtype)
        kc = (torch.randn(heads, nope, latent, device=device) * 0.03).to(dtype)
        vc = (torch.randn(heads, latent, value, device=device) * 0.03).to(dtype)
        base_caches = [base_cache * (0.5**i) for i in range(mla_layers)]
        kcs = [kc * (0.5**i) for i in range(mla_layers)]
        vcs = [vc * (0.5**i) for i in range(mla_layers)]
        angles = torch.arange(max_seq_len, device=device)[:, None] * (
            10000 ** (-torch.arange(0, rope, 2, device=device).float() / rope)
        )
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

        def rotate(x, positions):
            cos, sin = cos_sin[positions].chunk(2, dim=-1)
            if x.ndim == 3:
                cos, sin = cos[:, None], sin[:, None]
            left, right = x.float().chunk(2, dim=-1)
            return torch.cat(
                (left * cos - right * sin, right * cos + left * sin), -1
            ).to(dtype)

        for replicated in (False, True):
            parallel.prefill_cp_config.method = (
                CPRotateMethod.ALL_GATHER if replicated else CPRotateMethod.DISABLED
            )
            selected = slice(None) if replicated else slice(rank * 12, (rank + 1) * 12)
            config = AttentionConfigs()
            config.use_mla = True
            config.kv_cache_dtype = KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
            config.mla_fp8_compute = fp8
            config.mla_fp8_q_scale, config.mla_fp8_kv_scale = q_scale, kv_scale
            config.head_num = 96 if replicated else 12
            config.kv_lora_rank, config.nope_head_dim = latent, nope
            config.rope_head_dim = rope
            config.tokens_per_block = page
            config.kernel_tokens_per_block = kernel_page
            config.softmax_extra_scale = 1.0
            config.rope_config.is_neox_style = True
            model_config = ModelConfig()
            model_config.attn_config = config
            model_config.attn_config.head_num = heads
            model_config.attn_config.kv_head_num = 1
            model_config.num_layers = 2 + mla_layers
            model_config.max_seq_len = max_seq_len
            model_config.hybrid_attention_config.enable_hybrid_attention = True
            model_config.hybrid_attention_config.enable_independent_kv_cache_pools = (
                True
            )
            model_config.hybrid_attention_config.hybrid_attention_types = [
                HybridAttentionType.LINEAR,
                HybridAttentionType.LINEAR,
            ] + [HybridAttentionType.NONE] * mla_layers
            model_config.quant_config = None
            layer_ids = tuple(range(2, 2 + mla_layers))
            layer_to_group = [1, 1] + [0] * mla_layers
            weights = ModelWeights(model_config.num_layers, str(device), dtype)
            for i, layer_id in enumerate(layer_ids):
                weights.weights[layer_id] = {
                    W.mla_kc: kcs[i][selected],
                    W.mla_vc: vcs[i][selected],
                }
            weights.set_global_weight(W.rope_cos_sin_cache, cos_sin)
            parallel.role_type = RoleType.PDFUSION if replicated else RoleType.DECODE
            parallel.decode_cp_kv_cache_sharded = True
            parallel.prefill_cp_config.kv_cache_sharded = replicated
            inputs = PyAttentionInputs()
            inputs.is_prefill = inputs.is_target_verify = queries > 1
            inputs.is_cuda_graph = True
            inputs.total_tokens = tokens
            inputs.input_lengths = torch.full(
                (batch,), queries, dtype=torch.int32, device=device
            )
            inputs.prefix_lengths = torch.empty(
                batch if queries > 1 else 0, dtype=torch.int32, device=device
            )
            inputs.sequence_lengths = torch.empty_like(inputs.input_lengths)
            inputs.sequence_lengths_plus_1_d = torch.empty_like(inputs.input_lengths)
            table_storage = torch.full(
                (batch, table_width + 3), -1, dtype=torch.int32, device=device
            )
            full_table = table_storage[:, :table_width]
            linear_table = torch.zeros_like(full_table)
            inputs.kv_cache_kernel_block_id_device_by_group = [full_table, linear_table]
            inputs.kv_cache_layer_to_group = torch.tensor(
                layer_to_group, dtype=torch.int32
            )
            if not replicated:
                inputs.kv_cache_layer_to_group_host = inputs.kv_cache_layer_to_group
            inputs.kv_cache_kernel_block_id_device = linear_table
            layer_caches = []
            # Match the cache producer's per-kernel-page padding and layer offsets.
            cache_storage = torch.full(
                (mla_layers + 1, 5 * pages_per_block, kernel_page * (latent + rope) + 64),
                float("nan"), dtype=cache_dtype, device=device,
            )
            for i, _ in enumerate(layer_ids):
                layer_cache = LayerKVCache()
                layer_cache.kv_cache_base = cache_storage[
                    i + 1, :, :kernel_page * (latent + rope)
                ].view(-1, kernel_page, latent + rope)
                layer_caches.append(layer_cache)
            q = torch.empty_like(query[:, selected].contiguous())
            ckv = append[:, :latent].contiguous()
            kpe = torch.empty_like(append[:, latent:])

            def prepare(step, layer_index):
                starts = (
                    ((8 * page - 3, page - 3), (8 * page, page), (1, 8 * page - 1))
                    if queries > 1
                    else ((8 * page - 1, page - 1), (8 * page, page), (1, 8 * page - 1))
                )[step]
                physical = ((3, 1), (4, 2)) if step != 1 else ((1, 3), (2, 4))
                lengths = torch.tensor(starts, device=device, dtype=torch.int32)
                if queries > 1:
                    inputs.prefix_lengths.copy_(lengths)
                inputs.sequence_lengths.copy_(lengths)
                inputs.sequence_lengths_plus_1_d.copy_(lengths + 1)
                physical_pages = torch.tensor(
                    physical, device=device, dtype=torch.int32
                )
                full_table.copy_(
                    (
                        physical_pages[..., None] * pages_per_block
                        + torch.arange(
                            pages_per_block, device=device, dtype=torch.int32
                        )
                    ).flatten(1)
                )
                layer_cache = layer_caches[layer_index]
                layer_cache.kv_cache_base.zero_()
                dense = base_caches[layer_index].clone()
                positions = torch.tensor(
                    [s + j for s in starts for j in range(queries)], device=device
                )
                rotated_key = rotate(append[:, latent:], positions)
                for b, start in enumerate(starts):
                    prefix_indices = torch.arange(start, device=device)
                    owned = prefix_indices[(prefix_indices // page) % 8 == rank]
                    slots = (
                        physical_pages[b, owned // (page * 8)].long() * page
                        + owned % page
                    )
                    prefix = dense[b, owned]
                    if fp8:
                        prefix = (prefix.float() / kv_scale).clamp(-448, 448).to(cache_dtype)
                    layer_cache.kv_cache_base[slots // kernel_page, slots % kernel_page] = prefix
                    dense[b, start : start + queries, :latent] = ckv[
                        b * queries : (b + 1) * queries
                    ]
                    dense[b, start : start + queries, latent:] = rotated_key[
                        b * queries : (b + 1) * queries
                    ]
                q.copy_(query[:, selected])
                kpe.copy_(append[:, latent:])
                q_rotated = rotate(query[:, :, nope:], positions)
                absorbed = torch.bmm(
                    query[:, :, :nope].transpose(0, 1), kcs[layer_index]
                ).transpose(0, 1)
                q_dense = torch.cat((absorbed, q_rotated), -1).float()
                if fp8:
                    q_dense = (q_dense / q_scale).clamp(-448, 448).to(cache_dtype)
                    dense = (dense.float() / kv_scale).clamp(-448, 448).to(cache_dtype)
                    # FP8 rounds P within local attention. Use contiguous owner KV
                    # with the native kernel, then merge independently in PyTorch.
                    local_lengths = torch.tensor(
                        [bisect_right(owned_positions, start + j)
                         for start in starts for j in range(queries)],
                        dtype=torch.int32, device=device,
                    )
                    partial, lse = tokenspeed_mla_decode(
                        query=q_dense.view(tokens, 1, heads, latent + rope),
                        kv_cache=dense[:, owned_positions].contiguous().view(-1, kernel_page, latent + rope),
                        workspace_buffer=dense_workspace,
                        kv_lora_rank=latent, qk_rope_head_dim=rope,
                        block_tables=dense_pages,
                        seq_lens=local_lengths,
                        max_seq_len=len(owned_positions),
                        softmax_scale=(nope + rope)**-0.5 * q_scale * kv_scale,
                        output_scale=kv_scale, is_var_seq=True, causal_mask=True,
                        enable_pdl=False, return_lse=True,
                    )
                    partial = torch.where(local_lengths[:, None, None] > 0,
                                          partial.view(tokens, heads, latent), 0.0)
                    lse = torch.where(local_lengths[:, None] > 0,
                                      lse.view(tokens, heads), -torch.inf)
                    partials = [torch.empty_like(partial) for _ in range(8)]
                    lses = [torch.empty_like(lse) for _ in range(8)]
                    torch.distributed.all_gather(partials, partial, group=get_process_group(Group.TP))
                    torch.distributed.all_gather(lses, lse, group=get_process_group(Group.TP))
                    all_lse = torch.stack(lses)
                    factors = torch.exp2(all_lse - all_lse.amax(dim=0))
                    factors /= factors.sum(dim=0)
                    reference = (torch.stack(partials).float() * factors[..., None]).sum(dim=0)
                else:
                    reference = []
                    for b, start in enumerate(starts):
                        for j in range(queries):
                            keys = dense[b, : start + j + 1].float()
                            scores = (
                                q_dense[b * queries + j] @ keys.T * ((nope + rope) ** -0.5)
                            )
                            reference.append(torch.softmax(scores, -1) @ keys[:, :latent])
                    reference = torch.stack(reference)
                merged = reference.to(dtype)[:, selected]
                output = torch.bmm(
                    merged.transpose(0, 1), vcs[layer_index][selected]
                ).transpose(0, 1)
                return output, dense, starts, physical

            expected = [prepare(0, i) for i in range(mla_layers)]
            if replicated:
                host_map = inputs.kv_cache_layer_to_group
                inputs.kv_cache_layer_to_group = host_map.to(device)
                with unittest.TestCase().assertRaisesRegex(ValueError, "host layer"):
                    AttnImplFactory.get_fmha_impl(
                        model_config, parallel, weights, inputs, is_cuda_graph=True
                    )
                inputs.kv_cache_layer_to_group = host_map
            model_config.attn_config.is_sparse = True
            with unittest.TestCase().assertRaisesRegex(
                ValueError, "requires dense MLA"
            ):
                AttnImplFactory.get_fmha_impl(
                    model_config, parallel, weights, inputs, is_cuda_graph=True
                )
            model_config.attn_config.is_sparse = False
            impl = AttnImplFactory.get_fmha_impl(
                model_config, parallel, weights, inputs, is_cuda_graph=True
            )
            assert isinstance(impl, PageRRMlaDecodeImpl)
            assert impl.cache_group_id == 0
            assert impl.fmha_params.block_tables.data_ptr() == full_table.data_ptr()
            assert (
                inputs.kv_cache_kernel_block_id_device.data_ptr()
                == linear_table.data_ptr()
            )

            def forward_layers(fmha, layer_q, layer_ckv, layer_kpe):
                metadata = fmha.fmha_params
                table_pointer = metadata.query_block_tables.data_ptr()
                outputs = []
                with mock.patch.object(
                    metadata, "prepare", wraps=metadata.prepare
                ) as spy:
                    for layer_id, cache in zip(layer_ids, layer_caches):
                        outputs.append(
                            fmha.forward(
                                layer_q.clone(),
                                layer_ckv,
                                layer_kpe.clone(),
                                cache,
                                layer_id,
                            )
                        )
                        assert fmha.fmha_params is metadata
                        assert metadata.query_block_tables.data_ptr() == table_pointer
                    spy.assert_not_called()
                return outputs

            def invoke():
                return forward_layers(impl, q, ckv, kpe)

            def check(actual, expected, layer_index, actual_batch=batch):
                output, dense, starts, physical = expected
                real_tokens = actual_batch * queries
                torch.testing.assert_close(
                    actual, output[:real_tokens], atol=0.015, rtol=0.03
                )
                relative_error = (
                    (actual.float() - output[:real_tokens].float()).abs().max()
                    / output[:real_tokens].float().abs().max()
                ).item()
                assert relative_error < 2e-2, f"normalized max error: {relative_error}"
                for b, start in enumerate(starts[:actual_batch]):
                    positions = torch.arange(start + queries, device=device)
                    owned = positions[(positions // page) % 8 == rank]
                    physical_pages = torch.tensor(physical[b], device=device)
                    slots = physical_pages[owned // (page * 8)] * page + owned % page
                    torch.testing.assert_close(
                        layer_caches[layer_index].kv_cache_base[
                            slots // kernel_page, slots % kernel_page
                        ].float() * kv_scale,
                        dense[b, owned].float() * kv_scale,
                        atol=0.015,
                        rtol=0.01,
                    )
                assert torch.all(table_storage[:, table_width:] == -1)
                assert torch.isnan(cache_storage[:, :, kernel_page * (latent + rope):].float()).all()

            for i, actual in enumerate(invoke()):
                check(actual, expected[i], i)
            for _ in range(3):
                for i in range(mla_layers):
                    prepare(0, i)
                impl.prepare_cuda_graph(inputs)
                invoke()
            for i in range(mla_layers):
                prepare(0, i)
            impl.prepare_cuda_graph(inputs)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    captured = invoke()
                for step in range(3):
                    expected = [prepare(step, i) for i in range(mla_layers)]
                    impl.prepare_cuda_graph(inputs)
                    if fp8:
                        eager = [output.clone() for output in invoke()]
                        # Undo eager writes so replay must append the current KV itself.
                        expected = [prepare(step, i) for i in range(mla_layers)]
                        impl.prepare_cuda_graph(inputs)
                    graph.replay()
                    for i, actual in enumerate(captured):
                        if fp8:
                            torch.testing.assert_close(actual, eager[i], atol=0, rtol=0)
                        check(actual, expected[i], i)
            finally:
                graph.reset()

            # Native runner owns capture padding, group copies and preparation.
            local_heads = heads if replicated else heads // 8
            query_width = local_heads * (nope + rope)
            input_width = query_width + latent + rope
            hidden_width = max(input_width, mla_layers * local_heads * value)

            class RunnerModel(GptModelBase):
                def forward(self, model_inputs, fmha_impl=None):
                    if fmha_impl is None:
                        fmha_impl = self.prepare_fmha_impl(model_inputs)
                    self.captured_metadata = fmha_impl.fmha_params
                    hidden = model_inputs.input_hiddens.to(dtype)
                    outputs = forward_layers(
                        fmha_impl,
                        hidden[:, :query_width].reshape(-1, local_heads, nope + rope),
                        hidden[:, query_width : query_width + latent].contiguous(),
                        hidden[:, query_width + latent : input_width].contiguous(),
                    )
                    output = torch.cat([x.flatten(1) for x in outputs], dim=1)
                    return PyModelOutputs(
                        torch.nn.functional.pad(
                            output, (0, hidden_width - output.shape[1])
                        ),
                        fmha_impl.fmha_params,
                    )

            runner_model = RunnerModel(
                model_config, parallel, weights, max_generate_batch_size=batch
            )
            runner = CudaGraphRunner()
            try:
                runner.init_decode(
                    runner_model,
                    hidden_size=hidden_width,
                    max_seq_len=max_seq_len,
                    tokens_per_block=page,
                    kernel_tokens_per_block=kernel_page,
                    decode_capture_batch_sizes=[batch],
                    num_tokens_per_bs=queries,
                    is_target_verify=queries > 1,
                    max_context_batch_size=batch,
                    kv_cache_layer_to_group=layer_to_group,
                    kv_cache_group_num=2,
                )
                captured_table = runner_model.captured_metadata.query_block_tables
                table_pointer = captured_table.data_ptr()
                # Native capture reserves the maximum sequence width, beyond this batch's live pages.
                captured_width = runner_model.captured_metadata.block_tables.shape[1]
                compute_stream = torch.cuda.current_stream()
                prepare_stream = torch.cuda.Stream()

                def prepare_on_worker(replay):
                    torch.cuda.set_device(rank)
                    with torch.cuda.stream(prepare_stream):
                        runner.prepareAttentionInputs(replay)

                with ThreadPoolExecutor(max_workers=1) as worker:
                    for step, actual_batch in enumerate((2, 1, 2)):
                        expected = [prepare(step, i) for i in range(mla_layers)]
                        real_tokens = actual_batch * queries
                        before = [cache.kv_cache_base.clone() for cache in layer_caches]
                        replay = PyModelInputs()
                        replay.input_ids = torch.arange(
                            real_tokens, dtype=torch.int32, device=device
                        )
                        replay.input_hiddens = torch.nn.functional.pad(
                            torch.cat(
                                (
                                    query[:real_tokens, selected].flatten(1),
                                    append[:real_tokens],
                                ),
                                dim=1,
                            ).to(torch.float16),
                            (0, hidden_width - input_width),
                        )
                        live = PyAttentionInputs()
                        live.is_prefill = live.is_target_verify = queries > 1
                        live.is_cuda_graph = True
                        live.total_tokens = real_tokens
                        live.input_lengths = inputs.input_lengths[:actual_batch]
                        live.prefix_lengths = inputs.prefix_lengths[:actual_batch]
                        live.sequence_lengths = inputs.sequence_lengths[:actual_batch]
                        live.sequence_lengths_plus_1_d = (
                            inputs.sequence_lengths_plus_1_d[:actual_batch]
                        )
                        live.input_lengths_host = torch.full(
                            (actual_batch,), queries, dtype=torch.int32
                        ).pin_memory()
                        starts = expected[0][2][:actual_batch]
                        live.prefix_lengths_host = torch.tensor(
                            starts if queries > 1 else [], dtype=torch.int32
                        ).pin_memory()
                        live.sequence_lengths_host = torch.tensor(
                            starts, dtype=torch.int32
                        ).pin_memory()
                        live.decode_cu_seqlens_d = torch.arange(
                            0,
                            real_tokens + 1,
                            queries,
                            dtype=torch.int32,
                            device=device,
                        )
                        live.cu_seqlens = live.decode_cu_seqlens_d
                        live.cu_seqlens_host = live.cu_seqlens.cpu().pin_memory()
                        live.cu_kv_seqlens = live.cu_seqlens.clone()
                        live.kv_cache_layer_to_group = inputs.kv_cache_layer_to_group
                        live.kv_cache_layer_to_group_host = (
                            inputs.kv_cache_layer_to_group
                        )
                        live.kv_cache_kernel_block_id_device = linear_table[
                            :actual_batch
                        ]
                        live.kv_cache_kernel_block_id_device_by_group = [
                            full_table[:actual_batch],
                            linear_table[:actual_batch],
                        ]
                        live.kv_cache_kernel_block_id_host = (
                            linear_table[:actual_batch].cpu().pin_memory()
                        )
                        live.kv_cache_kernel_block_id_host_by_group = [
                            full_table[:actual_batch].cpu().pin_memory(),
                            live.kv_cache_kernel_block_id_host,
                        ]
                        replay.attention_inputs = live
                        assert runner.canRun(replay)
                        if step == 1:
                            prepare_stream.wait_stream(compute_stream)
                            worker.submit(prepare_on_worker, replay).result()
                            compute_stream.wait_stream(prepare_stream)
                        elif step == 2:
                            runner.prepareAttentionInputs(replay)
                        actual = runner.forward(replay).hidden_states
                        for i in range(mla_layers):
                            width = local_heads * value
                            check(
                                actual[:, i * width : (i + 1) * width]
                                .reshape(real_tokens, local_heads, value)
                                .to(dtype),
                                expected[i],
                                i,
                                actual_batch,
                            )
                            if actual_batch < batch:
                                # The inactive request's physical pages remain untouched.
                                physical = expected[i][3][1]
                                for p in physical:
                                    begin = p * pages_per_block
                                    torch.testing.assert_close(
                                        layer_caches[i].kv_cache_base[
                                            begin : begin + pages_per_block
                                        ].float(),
                                        before[i][begin : begin + pages_per_block].float(),
                                        atol=0,
                                        rtol=0,
                                    )
                        assert captured_table.data_ptr() == table_pointer
                        assert captured_table.shape == (batch * queries, captured_width)
                        torch.testing.assert_close(
                            captured_table[:real_tokens, :table_width],
                            full_table[:actual_batch].repeat_interleave(queries, dim=0),
                            atol=0,
                            rtol=0,
                        )
                        assert torch.all(captured_table[:real_tokens, table_width:] == 0)
            finally:
                del runner
            torch.cuda.synchronize()
            torch.distributed.barrier()
    finally:
        # Let mp.spawn report failures and terminate peers still inside collectives.
        if sys.exc_info()[0] is None:
            destroy_distributed_environment()


def _run_merge_history(rank, port):
    world_size = tp_size = 8
    dp_size = 1
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_collective import (
        merge_page_rr_attention,
    )
    from rtp_llm.ops import CPRotateMethod

    parallelism = ParallelismConfig()
    parallelism.world_rank = parallelism.local_rank = rank
    parallelism.world_size = world_size
    parallelism.tp_size, parallelism.dp_size = tp_size, dp_size
    parallelism.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
    torch.cuda.set_device(rank)
    init_distributed_environment(
        parallelism, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=60
    )
    try:
        assert parallelism.get_attn_tp_size() == 1
        assert torch.distributed.get_world_size(get_process_group(Group.TP)) == tp_size
        tokens, heads, dim = 14, 96, 512
        # Identical independent global references on all ranks; only each
        # rank's own partial is passed into the real NCCL merge.
        torch.manual_seed(301)
        values = torch.randn(
            (world_size, tokens, heads, dim), device="cuda", dtype=torch.bfloat16
        )
        logits = torch.randn((world_size, tokens, heads), device="cuda")
        partial = torch.empty_like(values[rank])
        lse = torch.empty_like(logits[rank])

        def prepare(step):
            all_lse = logits + 1000 + step
            all_lse[step % world_size].fill_(-float("inf"))
            if step == 1:
                all_lse[:, 0].fill_(-float("inf"))
            all_values = values.clone()
            all_values[step % world_size].fill_(float("nan"))
            if step == 2:
                all_values[0, -1, 0, 0] = float("nan")
            partial.copy_(all_values[rank])
            lse.copy_(all_lse[rank])
            weights = torch.softmax(all_lse * 0.6931471805599453, dim=0)
            weights = torch.where(
                torch.all(all_lse == -float("inf"), dim=0)[None], 0, weights
            )
            all_values = torch.where(
                (all_lse == -float("inf"))[..., None], 0, all_values.float()
            )
            return (weights[..., None] * all_values).sum(0).permute(1, 0, 2)

        for replicated in (False, True):

            def invoke():
                return merge_page_rr_attention(
                    partial, lse, replicated_heads=replicated
                )

            def check(actual, expected):
                if not replicated:
                    expected = expected[
                        rank * (heads // tp_size) : (rank + 1) * (heads // tp_size)
                    ]
                torch.testing.assert_close(
                    actual, expected, atol=2e-4, rtol=2e-4, equal_nan=True
                )

            expected = prepare(0)
            check(invoke(), expected)
            for _ in range(3):
                invoke()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    captured = invoke()
                for step in range(3):
                    expected = prepare(step)
                    graph.replay()
                    check(captured, expected)
            finally:
                # Captured NCCL work retains communicator resources. Release
                # the graph before destroying the distributed environment.
                graph.reset()
            torch.distributed.barrier()
        torch.cuda.synchronize()
    finally:
        if sys.exc_info()[0] is None:
            destroy_distributed_environment()


def _run_prefill_gather(rank, port):
    from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
        flashmla_dense_prefill,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_fp8_kernels import (
        quantize_fp8,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.flashmla_forward_test_utils import (
        DeterministicPackedProjection,
    )
    from rtp_llm.ops.compute_ops import CacheGroupType, KVCache

    torch.cuda.set_device(rank)
    parallel = ParallelismConfig()
    parallel.world_rank = parallel.local_rank = parallel.tp_rank = rank
    parallel.world_size = parallel.tp_size = 8
    parallel.role_type = RoleType.PREFILL
    init_distributed_environment(
        parallel, NcclCommConfig(nccl_ip="127.0.0.1"), port, timeout=120
    )
    try:
        page, kernel_page, width = 4096, 128, 576
        pages_per_block = page // kernel_page
        prefixes, lengths = [0, page + 5, 9 * page + 3], [3, 1, 2]
        q_offsets = [0, 3, 4, 6]
        config = ModelConfig()
        config.num_layers, config.max_seq_len = 3, 10 * page
        config.quant_config = None
        attn = config.attn_config
        attn.use_mla, attn.is_sparse = True, False
        attn.head_num, attn.kv_head_num = 96, 1
        attn.kv_lora_rank, attn.nope_head_dim = 512, 128
        attn.rope_head_dim, attn.v_head_dim = 64, 128
        attn.tokens_per_block, attn.kernel_tokens_per_block = page, kernel_page
        attn.kv_cache_dtype = KvCacheDataType.BASE
        hybrid = config.hybrid_attention_config
        hybrid.enable_hybrid_attention = hybrid.enable_independent_kv_cache_pools = True
        hybrid.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
            HybridAttentionType.NONE,
        ]
        weights = ModelWeights(3, "cuda", torch.bfloat16)
        weights.set_global_weight(
            W.rope_cos_sin_cache, torch.zeros(config.max_seq_len, 64, device="cuda")
        )
        torch.manual_seed(191)
        history = [
            torch.randn(n, width, device="cuda", dtype=torch.bfloat16) for n in prefixes
        ]
        fresh = torch.randn(sum(lengths), width, device="cuda", dtype=torch.bfloat16)
        query = (
            torch.randn(sum(lengths), 12, 192, device="cuda", dtype=torch.bfloat16)
            * 0.125
        )

        def reference(history_rows, suffix):
            outputs = []
            for request, old in enumerate(history_rows):
                begin, end = q_offsets[request : request + 2]
                kv = torch.cat((old, suffix[begin:end])).float()
                key = torch.cat((kv[:, :128], kv[:, 512:]), -1)
                scores = torch.einsum("qhd,kd->hqk", query[begin:end].float(), key) * (
                    192**-0.5
                )
                q_positions = old.shape[0] + torch.arange(
                    lengths[request], device="cuda"
                )
                k_positions = torch.arange(kv.shape[0], device="cuda")
                scores.masked_fill_(
                    k_positions[None, :] > q_positions[:, None], -torch.inf
                )
                outputs.append(
                    torch.einsum("hqk,kd->qhd", scores.softmax(-1), kv[:, 128:256])
                )
            return torch.cat(outputs)

        native_unit = {}
        table_width = (max(prefixes) + max(lengths) + kernel_page - 1) // kernel_page
        saw_empty_owner = False
        kv_scale = 16.0
        # Compare every lifecycle scenario against the same non-sharded attention path.
        for fp8, sharded in itertools.product((False, True), repeat=2):
            attn.kv_cache_dtype = (
                KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE
            )
            attn.mla_fp8_compute = fp8
            attn.mla_fp8_kv_scale = kv_scale
            cache_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
            parallel.prefill_cp_config.kv_cache_sharded = sharded
            cp_size = 8 if sharded else 1
            # Preserve the C++ producer's kernel-page padding and layer offsets.
            padding = 64
            raw_stride = page * width + pages_per_block * padding
            raw = torch.full(
                (3, 16, raw_stride), float("nan"), device="cuda", dtype=cache_dtype
            )
            kv_cache = KVCache()
            kv_cache.kv_cache_base_by_layer = [raw[i] for i in range(3)]
            kv_cache.seq_size_per_block = page
            kv_cache.kernel_seq_size_per_block = kernel_page
            kv_cache.use_mla = True
            kv_cache.kv_lora_rank, kv_cache.rope_head_dim = 512, 64
            kv_cache.layer_group_types = [
                CacheGroupType.LINEAR,
                CacheGroupType.FULL,
                CacheGroupType.FULL,
            ]
            kv_cache.layer_region_to_group_id = [[0], [1], [1]]
            table_storage = torch.full(
                (3, table_width + 5), -1, dtype=torch.int32, device="cuda"
            )
            table = table_storage[:, :table_width]
            linear_table = torch.zeros((3, 1), dtype=torch.int32, device="cuda")
            layer_caches = {i: kv_cache.get_layer_cache(i) for i in (1, 2)}
            for i, cache in layer_caches.items():
                view = cache.kv_cache_base
                assert view.shape == (16 * pages_per_block, kernel_page, width)
                assert view.stride() == (kernel_page * width + padding, width, 1)
                assert view.data_ptr() == raw[i].data_ptr()
                assert view.storage_offset() == raw[i].storage_offset()

            def make_inputs(prefix_lengths):
                inputs = PyAttentionInputs()
                inputs.is_prefill, inputs.total_tokens = True, sum(lengths)
                inputs.input_lengths_host = torch.tensor(lengths, dtype=torch.int32)
                inputs.prefix_lengths_host = torch.tensor(
                    prefix_lengths, dtype=torch.int32
                )
                inputs.input_lengths = inputs.input_lengths_host.cuda()
                inputs.prefix_lengths = inputs.prefix_lengths_host.cuda()
                inputs.cu_seqlens = torch.tensor(
                    q_offsets, dtype=torch.int32, device="cuda"
                )
                inputs.cu_kv_seqlens = torch.tensor(
                    [0] + [p + q for p, q in zip(prefix_lengths, lengths)],
                    dtype=torch.int32,
                    device="cuda",
                ).cumsum(0, dtype=torch.int32)
                inputs.padding_offset = torch.tensor(
                    [0, 0, 0, 0, 2, 2], dtype=torch.int32, device="cuda"
                )
                inputs.kv_cache_kernel_block_id_device = (
                    linear_table if sharded else table
                )
                inputs.kv_cache_kernel_block_id_device_by_group = [linear_table, table]
                inputs.kv_cache_layer_to_group_host = torch.tensor(
                    [0, 1, 1], dtype=torch.int32
                )
                inputs.kv_cache_layer_to_group = inputs.kv_cache_layer_to_group_host
                return inputs

            def populate(prefix_lengths, scale, remap):
                raw.fill_(float("nan"))
                table.zero_()
                logical = {
                    i: [
                        old[:n] * (scale * 0.5 ** (i - 1))
                        for old, n in zip(history, prefix_lengths)
                    ]
                    for i in (1, 2)
                }
                cache_by_layer = {
                    i: [quantize_fp8(row, kv_scale) if fp8 else row for row in logical[i]]
                    for i in (1, 2)
                }
                history_by_layer = {
                    i: [(row.float() * kv_scale).bfloat16() if fp8 else row
                        for row in cache_by_layer[i]]
                    for i in (1, 2)
                }
                fresh_by_layer = {
                    i: fresh * (scale * 0.5 ** (i - 1)) for i in (1, 2)
                }
                physical_page = 15 - remap
                for request, n in enumerate(prefix_lengths):
                    for global_page in range((n + page - 1) // page):
                        if sharded and global_page % cp_size != rank:
                            continue
                        local_page = global_page // cp_size
                        first = local_page * pages_per_block
                        destination = table[request, first : first + pages_per_block]
                        destination.copy_(
                            torch.arange(
                                physical_page * pages_per_block,
                                (physical_page + 1) * pages_per_block,
                                device="cuda",
                                dtype=torch.int32,
                            )[: destination.numel()]
                        )
                        count = min(page, n - global_page * page)
                        rows = torch.arange(count, device="cuda")
                        for i, cache in layer_caches.items():
                            cache.kv_cache_base[
                                physical_page * pages_per_block + rows // kernel_page,
                                rows % kernel_page,
                                :width,
                            ] = cache_by_layer[i][request][
                                global_page * page : global_page * page + count
                            ]
                        physical_page -= 1
                return history_by_layer, fresh_by_layer

            for capacity in (0, page):
                config.attn_config.mla_prefill_expanded_kv_budget_bytes = (
                    capacity * 12 * 320 * 2
                )
                impl = None
                scenarios = (
                    (prefixes, 1.0),
                    (prefixes, 0.125),
                    ([0, 0, 0], 0.125),
                    ([0, 129, page - 3], 0.125),
                )
                for scenario, (prefix_lengths, scale) in enumerate(scenarios):
                    histories, suffixes = populate(prefix_lengths, scale, scenario % 2)
                    inputs = make_inputs(prefix_lengths)
                    with mock.patch.object(
                        flashmla_dense_prefill,
                        "_build_cp_gather_plan",
                        wraps=flashmla_dense_prefill._build_cp_gather_plan,
                    ) as builder:
                        if impl is None:
                            impl = AttnImplFactory.get_fmha_impl(
                                config, parallel, weights, inputs
                            )
                        else:
                            impl.prepare(inputs)
                        op = impl.fmha_impl
                        plans = dict(op._cp_gather_plans)
                        oracle_counts = {}
                        for key, plan in plans.items():
                            counts = [0] * cp_size
                            for item in key[0]:
                                for position in range(
                                    item.prefix_start,
                                    item.prefix_start + item.prefix_len,
                                ):
                                    counts[position // page % cp_size] += 1
                            oracle_counts[key] = counts
                            assert plan.stride == max(counts)
                            assert plan.local_count == counts[rank]
                            assert plan.total_count == sum(counts)
                            coordinates = (
                                plan.pack_request,
                                plan.pack_column,
                                plan.pack_offset,
                                plan.restore_source,
                                plan.restore_target,
                            )
                            assert sum(
                                t.numel() * t.element_size() for t in coordinates
                            ) == (8 * (3 * counts[rank] + 2 * sum(counts)))
                        plan_pointers = {
                            key: tuple(
                                getattr(plan, field).data_ptr()
                                for field in (
                                    "pack_request",
                                    "pack_column",
                                    "pack_offset",
                                    "restore_source",
                                    "restore_target",
                                )
                            )
                            for key, plan in plans.items()
                        }
                        assert builder.call_count == len(plans)
                        build_count = builder.call_count
                        gather_calls = []
                        buffers = None
                        original_gather = op._gather_cp_prefix

                        def checked_gather(
                            cache, slices, offsets, latent_out, rope_out
                        ):
                            sentinel = -123.0
                            latent_out.fill_(sentinel)
                            rope_out.fill_(sentinel)
                            original_gather(
                                cache, slices, offsets, latent_out, rope_out
                            )
                            untouched = torch.ones(
                                latent_out.shape[0], dtype=torch.bool, device="cuda"
                            )
                            for item, destination in zip(slices, offsets):
                                count = item.prefix_len
                                expected = active_history[item.request_idx][
                                    item.prefix_start : item.prefix_start + count
                                ]
                                torch.testing.assert_close(
                                    latent_out[destination : destination + count],
                                    expected[:, :512],
                                    atol=0,
                                    rtol=0,
                                )
                                expected_rope = expected[:, 512:]
                                if rope_out.ndim == 3:
                                    expected_rope = expected_rope[:, None].expand(
                                        -1, rope_out.shape[1], -1
                                    )
                                torch.testing.assert_close(
                                    rope_out[destination : destination + count],
                                    expected_rope,
                                    atol=0,
                                    rtol=0,
                                )
                                untouched[destination : destination + count] = False
                            assert torch.all(latent_out[untouched] == sentinel)
                            assert torch.all(rope_out[untouched] == sentinel)
                            gather_calls.append((slices, tuple(offsets), rope_out.ndim))

                        with (
                            mock.patch.object(
                                op,
                                "_create_kv_b_proj",
                                return_value=DeterministicPackedProjection(),
                            ),
                            mock.patch.object(
                                op, "_gather_cp_prefix", side_effect=checked_gather
                            ),
                            mock.patch.object(
                                flashmla_dense_prefill,
                                "all_gather_into",
                                wraps=flashmla_dense_prefill.all_gather_into,
                            ) as collective,
                        ):
                            for layer_id in (1, 2):
                                select_block_map_for_layer(inputs, layer_id)
                                active_history = histories[layer_id]
                                suffix = suffixes[layer_id]
                                output = op.forward(
                                    query,
                                    suffix[:, :512].contiguous(),
                                    suffix[:, 512:],
                                    layer_caches[layer_id],
                                    layer_id,
                                ).clone()
                                key = (fp8, capacity, scenario, layer_id)
                                if sharded:
                                    torch.testing.assert_close(
                                        output, native_unit[key], atol=0, rtol=0
                                    )
                                else:
                                    native_unit[key] = output
                                if not fp8 and scenario != 0:
                                    torch.testing.assert_close(
                                        output.float(),
                                        reference(active_history, suffix),
                                        atol=1e-3,
                                        rtol=2e-2,
                                    )
                                assert builder.call_count == build_count
                                assert all(
                                    op._cp_gather_plans[k] is v
                                    for k, v in plans.items()
                                )
                                for key, plan in plans.items():
                                    current = tuple(
                                        getattr(plan, field).data_ptr()
                                        for field in (
                                            "pack_request",
                                            "pack_column",
                                            "pack_offset",
                                            "restore_source",
                                            "restore_target",
                                        )
                                    )
                                    assert current == plan_pointers[key]
                                    saw_empty_owner |= (
                                        plan.local_count == 0 and plan.stride > 0
                                    )
                                if sharded and any(prefix_lengths):
                                    current = tuple(
                                        t.data_ptr() for t in op._cp_gather_buffers
                                    )
                                    if buffers is not None:
                                        assert current == buffers
                                    buffers = current
                                    for buffer in op._cp_gather_buffers:
                                        buffer.fill_(float("nan"))
                                padding_view = raw[layer_id].view(
                                    16 * pages_per_block, -1
                                )
                                assert torch.isnan(
                                    padding_view[:, kernel_page * width :]
                                ).all()
                            if sharded and any(prefix_lengths):
                                calls_per_layer = (
                                    len(op._prefix_runtime_launches)
                                    if op._forward_plan.route.value == "hybrid"
                                    else 1
                                )
                                assert len(gather_calls) == 2 * calls_per_layer
                                assert collective.call_count == len(gather_calls)
                                for call in collective.call_args_list:
                                    local, gathered, group = call.args
                                    assert group == Group.TP
                                    assert (
                                        local.is_contiguous()
                                        and gathered.is_contiguous()
                                    )
                                    assert local.shape[1] == width
                                    assert gathered.shape == (8 * local.shape[0], width)
                                maximum_stride = max(
                                    max(counts) for counts in oracle_counts.values()
                                )
                                maximum_count = max(
                                    sum(counts) for counts in oracle_counts.values()
                                )
                                expected_shapes = (
                                    (maximum_stride, width),
                                    (8 * maximum_stride, width),
                                    (maximum_count, width),
                                )
                                assert (
                                    tuple(t.shape for t in op._cp_gather_buffers)
                                    == expected_shapes
                                )
                                if capacity and scenario == 0:
                                    assert any(
                                        item.prefix_start > 0
                                        for slices, _, _ in gather_calls
                                        for item in slices
                                    )
                                    assert {ndim for _, _, ndim in gather_calls} == {3}
                                if scenario == 1:
                                    impl.release_forward_workspace()
                                    assert op._cp_gather_buffers is None
                                    assert op._forward_workspace is None
                                    assert all(
                                        op._cp_gather_plans[key] is plan
                                        for key, plan in plans.items()
                                    )
                                    repeated = op.forward(
                                        query,
                                        suffix[:, :512].contiguous(),
                                        suffix[:, 512:],
                                        layer_caches[2],
                                        2,
                                    ).clone()
                                    torch.testing.assert_close(
                                        repeated, output, atol=0, rtol=0
                                    )
                                    assert builder.call_count == build_count
                            else:
                                assert not gather_calls
                                collective.assert_not_called()
                                assert not plans and op._cp_gather_buffers is None
                        assert torch.all(table_storage[:, table_width:] == -1)
                        if sharded and scenario == 1:
                            from rtp_llm.config.quant_config import init_quant_config
                            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
                                CudaFp8DeepGEMMLinear,
                            )
                            from rtp_llm.models_py.modules.kimi_k3.fp8_producers import (
                                Fp8RMSNorm,
                            )

                            # The production producer retains BF16 for cache writes;
                            # both current and historical KV still reach real KV-B GEMM.
                            payload = Fp8RMSNorm(
                                torch.ones(512, dtype=torch.bfloat16, device="cuda"),
                                1e-6,
                                retain_bf16=True,
                            )(suffixes[2][:, :512])
                            projection = CudaFp8DeepGEMMLinear(
                                weight=(
                                    torch.randn(3072, 512, device="cuda") * 0.03
                                ).to(torch.float8_e4m3fn),
                                weight_scales=torch.full(
                                    (1, 3072),
                                    0x7F7F7F7F,
                                    dtype=torch.int32,
                                    device="cuda",
                                ).T,
                                input_scales=None,
                                bias=None,
                                quant_config=init_quant_config("FP8_PER_BLOCK"),
                            )
                            with mock.patch.object(
                                op, "_create_kv_b_proj", return_value=projection
                            ):
                                expected_quantized = op.forward(
                                    query,
                                    payload.bf16,
                                    suffixes[2][:, 512:],
                                    layer_caches[2],
                                    2,
                                ).clone()
                                actual_quantized = op.forward(
                                    query,
                                    payload,
                                    suffixes[2][:, 512:],
                                    layer_caches[2],
                                    2,
                                ).clone()
                            torch.testing.assert_close(
                                actual_quantized, expected_quantized, atol=0, rtol=0
                            )
                            assert builder.call_count == build_count
                    del plans
                torch.distributed.barrier()
        empties = torch.tensor(int(saw_empty_owner), device="cuda")
        torch.distributed.all_reduce(empties)
        assert int(empties) > 0
    finally:
        if sys.exc_info()[0] is None:
            destroy_distributed_environment()


class PageRRMlaDecodeTest(unittest.TestCase):
    def _run(self, worker, *args):
        self.assertEqual(
            torch.cuda.device_count(), 8, "native target requires eight GPUs"
        )
        ports, locks = PortManager().get_consecutive_ports(1)
        try:
            mp.spawn(worker, args=(ports[0], *args), nprocs=8, join=True)
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_writer_projection_attention_graph_history(self):
        self._run(_run_history)

    def test_q1_writer_projection_attention_graph_history(self):
        self._run(_run_history, 1)

    def test_two_mla_layers_share_prepared_query_tables(self):
        self._run(_run_history, 7, 2)

    def test_fp8_q1_writer_projection_attention_graph_history(self):
        self._run(_run_history, 1, 1, True)

    def test_fp8_verify_two_layers_writer_projection_attention_graph_history(self):
        self._run(_run_history, 7, 2, True)

    def test_partial_merge_graph_history(self):
        self._run(_run_merge_history)

    def test_prefill_prefix_gather_from_owners(self):
        self._run(_run_prefill_gather)


if __name__ == "__main__":
    unittest.main()
