"""Eight-rank Page-RR adapter history against independent dense MLA math."""

import unittest
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


def _run_history(rank, port, queries=7, mla_layers=1):
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
            config.kv_cache_dtype = KvCacheDataType.BASE
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
            model_config.hybrid_attention_config.enable_independent_kv_cache_pools = True
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
                    W.mla_kc: kcs[i][selected], W.mla_vc: vcs[i][selected]
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
            for _ in layer_ids:
                layer_cache = LayerKVCache()
                layer_cache.kv_cache_base = torch.empty(
                    (5 * pages_per_block, kernel_page, latent + rope),
                    dtype=dtype, device=device,
                )
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
                    layer_cache.kv_cache_base.view(-1, latent + rope)[slots] = dense[
                        b, owned
                    ]
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
                reference = []
                for b, start in enumerate(starts):
                    for j in range(queries):
                        keys = dense[b, : start + j + 1].float()
                        scores = (
                            q_dense[b * queries + j] @ keys.T * ((nope + rope) ** -0.5)
                        )
                        reference.append(torch.softmax(scores, -1) @ keys[:, :latent])
                merged = torch.stack(reference).to(dtype)[:, selected]
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
            with unittest.TestCase().assertRaisesRegex(ValueError, "requires dense MLA"):
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
                with mock.patch.object(metadata, "prepare", wraps=metadata.prepare) as spy:
                    for layer_id, cache in zip(layer_ids, layer_caches):
                        outputs.append(
                            fmha.forward(
                                layer_q.clone(), layer_ckv, layer_kpe.clone(),
                                cache, layer_id,
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
                        layer_caches[layer_index].kv_cache_base.view(-1, latent + rope)[slots],
                        dense[b, owned], atol=0.015, rtol=0.01,
                    )
                assert torch.all(table_storage[:, table_width:] == -1)

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
                    graph.replay()
                    for i, actual in enumerate(captured):
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
                    runner_model, hidden_size=hidden_width, max_seq_len=max_seq_len,
                    tokens_per_block=page, kernel_tokens_per_block=kernel_page,
                    decode_capture_batch_sizes=[batch], num_tokens_per_bs=queries,
                    is_target_verify=queries > 1, max_context_batch_size=batch,
                    kv_cache_layer_to_group=layer_to_group, kv_cache_group_num=2,
                )
                captured_table = runner_model.captured_metadata.query_block_tables
                table_pointer = captured_table.data_ptr()
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
                                (query[:real_tokens, selected].flatten(1), append[:real_tokens]),
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
                            0, real_tokens + 1, queries, dtype=torch.int32, device=device
                        )
                        live.cu_seqlens = live.decode_cu_seqlens_d
                        live.cu_seqlens_host = live.cu_seqlens.cpu().pin_memory()
                        live.cu_kv_seqlens = live.cu_seqlens.clone()
                        live.kv_cache_layer_to_group = inputs.kv_cache_layer_to_group
                        live.kv_cache_layer_to_group_host = inputs.kv_cache_layer_to_group
                        live.kv_cache_kernel_block_id_device = linear_table[:actual_batch]
                        live.kv_cache_kernel_block_id_device_by_group = [
                            full_table[:actual_batch], linear_table[:actual_batch]
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
                                .reshape(real_tokens, local_heads, value).to(dtype),
                                expected[i], i, actual_batch,
                            )
                            if actual_batch < batch:
                                # The inactive request's physical pages remain untouched.
                                physical = expected[i][3][1]
                                for p in physical:
                                    begin = p * pages_per_block
                                    torch.testing.assert_close(
                                        layer_caches[i].kv_cache_base[begin : begin + pages_per_block],
                                        before[i][begin : begin + pages_per_block],
                                        atol=0, rtol=0,
                                    )
                        assert captured_table.data_ptr() == table_pointer
                        assert captured_table.shape == (batch * queries, table_width)
                        torch.testing.assert_close(
                            captured_table[:real_tokens],
                            full_table[:actual_batch].repeat_interleave(queries, dim=0),
                            atol=0, rtol=0,
                        )
            finally:
                del runner
            torch.cuda.synchronize()
            torch.distributed.barrier()
    finally:
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

    def test_partial_merge_graph_history(self):
        self._run(_run_merge_history)


if __name__ == "__main__":
    unittest.main()
