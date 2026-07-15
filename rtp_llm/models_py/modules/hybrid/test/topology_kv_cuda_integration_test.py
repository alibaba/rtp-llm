"""H20 integration coverage for topology-aware Indexer candidates.

This test intentionally crosses the production CUDA boundary.  Candidate generation
comes from ``Indexer.forward`` (including the ragged and CP kernels) and candidates are
consumed by ``SparseMlaOp``.  The policy's opt-in host synchronization is inside the
timed region; benchmark-only attention helpers are not used here.
"""

import math
import os
import time
from unittest import SkipTest, TestCase, main, mock, skipIf

import torch
import torch.nn.functional as F


def _cuda_flashmla_available() -> bool:
    try:
        if not torch.version.cuda:
            return False
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) < (12, 9):
            return False
        from flash_mla import (  # noqa: F401
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        return torch.cuda.is_available()
    except (ImportError, AttributeError, ValueError):
        return False


CUDA_FLASHMLA_OK = _cuda_flashmla_available()
SKIP_REASON = "requires an H20-class CUDA >= 12.9 worker with flash_mla"

if CUDA_FLASHMLA_OK:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.metrics import GaugeMetrics, kmonitor
    from rtp_llm.models_py.distributed.collective_torch import (
        destroy_distributed_environment,
        init_distributed_environment,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_cp_impl import (
        SparseMlaCpImpl,
    )
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
        SparseMlaImpl,
        SparseMlaOp,
    )
    from rtp_llm.models_py.modules.hybrid.indexer import Indexer
    from rtp_llm.ops import (
        CPRotateMethod,
        KvCacheDataType,
        NcclCommConfig,
        ParallelismConfig,
        PrefillCPConfig,
    )
    from rtp_llm.ops.compute_ops import (
        LayerKVCache,
        PyAttentionInputs,
        PyContextParallelParams,
        rtp_llm_ops,
    )
    from rtp_llm.test.utils.port_util import PortManager
    from rtp_llm.utils.model_weight import W


def _block_table(total_length: int, page_size: int) -> torch.Tensor:
    return torch.arange(math.ceil(total_length / page_size), dtype=torch.int32).view(
        1, -1
    )


def _reference_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    global_indices: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """Small sampled PyTorch oracle; ``-1`` remains masked, never gathered."""
    q_fp32 = q.float()
    kv_fp32 = kv.float().squeeze(1)
    invalid = (global_indices < 0) | (global_indices >= kv_fp32.size(0))
    safe = global_indices.clamp(0, kv_fp32.size(0) - 1)
    gathered = kv_fp32[safe].unsqueeze(1).expand(-1, q.size(1), -1, -1)
    scores = torch.matmul(q_fp32.unsqueeze(2), gathered.transpose(-1, -2))
    scores = scores.squeeze(2).mul(scale).masked_fill(invalid.unsqueeze(1), -torch.inf)
    probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    return (
        torch.matmul(probabilities.unsqueeze(2), gathered[..., :kv_lora_rank])
        .squeeze(2)
        .to(q.dtype)
    )


@skipIf(not CUDA_FLASHMLA_OK, SKIP_REASON)
class TopologyKvCudaProductionBoundaryTest(TestCase):
    PAGE_SIZE = 64
    TOPK = 128
    HIDDEN_SIZE = 256
    Q_LORA_RANK = 128
    INDEX_HEADS = 64
    INDEX_HEAD_DIM = 128
    MLA_HEADS = 64
    KV_LORA_RANK = 512
    ROPE_DIM = 64
    NOPE_DIM = 512

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise SkipTest(SKIP_REASON)
        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)
        os.environ["RTP_LLM_TOPOLOGY_KV_ALLOW_CUDA_SYNC"] = "1"
        cls._port_manager = PortManager()
        ports, cls._port_locks = cls._port_manager.get_consecutive_ports(1)
        base = ports[0] + 11
        distributed = cls._make_parallelism(True)
        init_distributed_environment(
            distributed,
            NcclCommConfig(
                nccl_ip="127.0.0.1",
                tp_nccl_port=base - 2,
                dp_tp_nccl_port=base - 10,
                ffn_tp_nccl_port=base - 5,
            ),
            nccl_init_port=base - 11,
            backend="nccl",
            timeout=120,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        destroy_distributed_environment()
        for lock in cls._port_locks:
            lock.__exit__(None, None, None)

    def setUp(self) -> None:
        torch.manual_seed(20260715)
        torch.cuda.manual_seed_all(20260715)

    @staticmethod
    def _make_parallelism(cp: bool) -> "ParallelismConfig":
        config = ParallelismConfig()
        config.world_rank = 0
        config.world_size = 1
        config.local_rank = 0
        config.tp_rank = 0
        config.tp_size = 1
        if cp:
            config.prefill_cp_config = PrefillCPConfig()
            config.prefill_cp_config.method = CPRotateMethod.ALL_GATHER
            config.prefill_cp_config.comm_buffer_size = 0
        return config

    def _model_and_weights(self):
        config = ModelConfig()
        config.hidden_size = self.HIDDEN_SIZE
        config.max_seq_len = 1024
        config.layernorm_eps = 1e-6
        config.quant_config = None
        attn = config.attn_config
        attn.head_num = self.MLA_HEADS
        attn.size_per_head = self.NOPE_DIM + self.ROPE_DIM
        attn.nope_head_dim = self.NOPE_DIM
        attn.rope_head_dim = self.ROPE_DIM
        attn.kv_lora_rank = self.KV_LORA_RANK
        attn.v_head_dim = self.NOPE_DIM
        attn.q_lora_rank = self.Q_LORA_RANK
        attn.tokens_per_block = self.PAGE_SIZE
        attn.kernel_tokens_per_block = self.PAGE_SIZE
        attn.softmax_extra_scale = 1.0
        attn.use_mla = True
        attn.indexer_head_num = self.INDEX_HEADS
        attn.indexer_head_dim = self.INDEX_HEAD_DIM
        attn.indexer_topk = self.TOPK
        attn.kv_cache_dtype = KvCacheDataType.FP8

        weights = {
            W.mla_indexer_qb_w: torch.randn(
                self.Q_LORA_RANK,
                self.INDEX_HEADS * self.INDEX_HEAD_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            W.mla_indexer_k_w: torch.randn(
                self.HIDDEN_SIZE,
                self.INDEX_HEAD_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            W.mla_indexer_k_norm_w: torch.ones(
                self.INDEX_HEAD_DIM, dtype=torch.bfloat16, device=self.device
            ),
            W.mla_indexer_k_norm_b: torch.zeros(
                self.INDEX_HEAD_DIM, dtype=torch.bfloat16, device=self.device
            ),
            W.mla_indexer_weights_proj_w: torch.randn(
                self.HIDDEN_SIZE,
                self.INDEX_HEADS,
                dtype=torch.float32,
                device=self.device,
            ),
            W.rope_cos_sin_cache: torch.randn(
                config.max_seq_len,
                self.ROPE_DIM,
                dtype=torch.float32,
                device=self.device,
            ),
            W.mla_kc: torch.randn(
                self.MLA_HEADS,
                self.NOPE_DIM,
                self.KV_LORA_RANK,
                dtype=torch.bfloat16,
                device=self.device,
            ).mul_(0.01),
            W.mla_vc: torch.randn(
                self.MLA_HEADS,
                self.KV_LORA_RANK,
                self.NOPE_DIM,
                dtype=torch.bfloat16,
                device=self.device,
            ).mul_(0.01),
        }
        return config, weights

    def _attention_inputs(self, query_length: int, prefix_length: int):
        total_length = query_length + prefix_length
        table = _block_table(total_length, self.PAGE_SIZE)
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.input_lengths = torch.tensor([query_length], dtype=torch.int32)
        inputs.sequence_lengths = torch.tensor([total_length], dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor([prefix_length], dtype=torch.int32)
        inputs.cu_seqlens_device = torch.tensor(
            [0, query_length], dtype=torch.int32, device=self.device
        )
        inputs.cu_kv_seqlens_device = torch.tensor(
            [0, total_length], dtype=torch.int32, device=self.device
        )
        inputs.kv_cache_block_id = table
        inputs.kv_cache_block_id_device = table.to(self.device)
        inputs.kv_cache_kernel_block_id = table
        inputs.kv_cache_kernel_block_id_device = table.to(self.device)
        params = rtp_llm_ops.SparseMlaParams()
        params.fill_params(inputs, self.PAGE_SIZE)
        return inputs, params

    def _index_cache(self, total_length: int):
        cache = LayerKVCache()
        cache.kv_scale_base = torch.empty(
            math.ceil(total_length / self.PAGE_SIZE),
            self.PAGE_SIZE,
            self.INDEX_HEAD_DIM + 4,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        return cache

    def _prime_prefix(self, indexer, cache, prefix_length: int) -> None:
        if prefix_length == 0:
            return
        prefix_inputs, prefix_params = self._attention_inputs(prefix_length, 0)
        hidden = torch.randn(
            prefix_length,
            self.HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=self.device,
        )
        indexer(
            hidden_states=hidden,
            q_lora=torch.empty(
                prefix_length,
                self.Q_LORA_RANK,
                dtype=torch.bfloat16,
                device=self.device,
            ),
            kv_cache=cache,
            fmha_params=prefix_params,
            attention_inputs=prefix_inputs,
            use_fast_path=True,
        )

    def _sparse_operator(self, params, inputs):
        op = SparseMlaOp(
            num_heads=self.MLA_HEADS,
            kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.ROPE_DIM,
            qk_nope_head_dim=self.NOPE_DIM,
            page_size=self.PAGE_SIZE,
            softmax_extra_scale=1.0,
            top_k=self.TOPK,
        )
        op.plan(params, inputs.kv_cache_kernel_block_id_device, inputs)
        return op

    def _assert_sparse_output_matches_reference(self, op, output, q, kv, topk):
        global_indices = op._convert_topk_indices_to_global(topk)[:, 0, :]
        sample = torch.arange(max(0, q.size(0) - 2), q.size(0), device=self.device)
        reference = _reference_sparse_mla(
            q[sample],
            kv,
            global_indices[sample],
            op.scale,
            self.KV_LORA_RANK,
        )
        actual = output[sample]
        actual_normalized = actual / (torch.linalg.vector_norm(actual) + 1e-8)
        reference_normalized = reference / (torch.linalg.vector_norm(reference) + 1e-8)
        self.assertTrue(
            torch.allclose(
                actual_normalized, reference_normalized, atol=1e-2, rtol=1e-2
            )
        )
        self.assertGreater(
            F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item(),
            0.99,
        )

    def _run_case(self, query_length: int, prefix_length: int):
        total_length = query_length + prefix_length
        config, weights = self._model_and_weights()
        parallelism = self._make_parallelism(False)
        indexer = Indexer(
            attn_config=config.attn_config,
            weights=weights,
            global_weights=weights,
            layer_idx=0,
            layernorm_eps=config.layernorm_eps,
            quant_config=config.quant_config,
            parallelism_config=parallelism,
            scale_fmt="ue8m0",
        )
        cache = self._index_cache(total_length)
        self._prime_prefix(indexer, cache, prefix_length)
        inputs, params = self._attention_inputs(query_length, prefix_length)
        hidden = torch.randn(
            query_length,
            self.HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_lora = torch.randn(
            query_length,
            self.Q_LORA_RANK,
            dtype=torch.bfloat16,
            device=self.device,
        )

        indexer.topology_kv_policy = "disabled"
        torch.cuda.synchronize()
        baseline_started = time.perf_counter()
        baseline_topk = indexer(hidden, q_lora, cache, params, inputs, False, None)

        sparse_op = self._sparse_operator(params, inputs)
        q = torch.randn(
            query_length,
            self.MLA_HEADS,
            self.NOPE_DIM + self.ROPE_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        ).mul_(0.1)
        physical_tokens = math.ceil(total_length / self.PAGE_SIZE) * self.PAGE_SIZE
        kv = torch.randn(
            physical_tokens,
            1,
            self.KV_LORA_RANK + self.ROPE_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        ).mul_(0.1)
        baseline_output = sparse_op.forward(q, kv, baseline_topk)
        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - baseline_started) * 1000.0

        indexer.topology_kv_policy = "topology_only"
        torch.cuda.synchronize()
        policy_started = time.perf_counter()
        # The reporter spy observes production telemetry without replacing the
        # real Indexer, ragged top-k, coordinate conversion, or sparse MLA path.
        with mock.patch.object(kmonitor, "report") as metric_report:
            policy_topk = indexer(hidden, q_lora, cache, params, inputs, False, None)
        policy_output = sparse_op.forward(q, kv, policy_topk)
        torch.cuda.synchronize()
        policy_ms = (time.perf_counter() - policy_started) * 1000.0

        over_topk = params.expanded_seq_lens > self.TOPK
        self.assertTrue(
            torch.any(over_topk).item(), "scenario must exercise truncation"
        )
        self.assertFalse(torch.equal(baseline_topk[over_topk], policy_topk[over_topk]))
        self.assertTrue(torch.all((policy_topk >= 0).any(dim=1)).item())
        reported_metrics = {call.args[0] for call in metric_report.call_args_list}
        self.assertTrue(
            {
                GaugeMetrics.TOPOLOGY_KV_POLICY_SCHEDULE_MS_METRIC,
                GaugeMetrics.TOPOLOGY_KV_POLICY_SELECTED_TOKENS_METRIC,
                GaugeMetrics.TOPOLOGY_KV_POLICY_EVICTED_TOKENS_METRIC,
            }.issubset(reported_metrics)
        )
        self._assert_sparse_output_matches_reference(
            sparse_op, baseline_output, q, kv, baseline_topk
        )
        self._assert_sparse_output_matches_reference(
            sparse_op, policy_output, q, kv, policy_topk
        )
        return baseline_ms, policy_ms

    def test_normal_prefill_actual_indexer_to_sparse_mla(self):
        baseline_ms, policy_ms = self._run_case(129, 0)
        print(
            "topology_kv_e2e normal_prefill "
            f"baseline_ms={baseline_ms:.3f} policy_ms={policy_ms:.3f} "
            f"host_sync_and_schedule_overhead_ms={policy_ms - baseline_ms:.3f}"
        )

    def test_prefix_prefill_actual_indexer_to_sparse_mla(self):
        self._run_case(4, 192)

    def test_cp_indexer_coordinates_reach_sparse_mla(self):
        self.assertTrue(issubclass(SparseMlaCpImpl, SparseMlaImpl))
        self.assertTrue(SparseMlaCpImpl.support_prefill_cp())
        query_length = 129
        config, weights = self._model_and_weights()
        parallelism = self._make_parallelism(True)
        inputs, params = self._attention_inputs(query_length, 0)
        cp_info = PyContextParallelParams()
        cp_info.prefill_cp_chunk_lengths = torch.tensor(
            [65, 64], dtype=torch.int32, device=self.device
        )
        cp_info.prefill_qkv_restore_indice = torch.arange(
            query_length, dtype=torch.long, device=self.device
        )
        cp_info.prefill_qkv_padding_mask = torch.ones(
            query_length, dtype=torch.int32, device=self.device
        )
        cp_info.prefill_actual_input_lengths_cpu = torch.tensor(
            [query_length], dtype=torch.int32
        )
        inputs.context_parallel_info = cp_info

        attn_configs = config.getAttentionConfigs(1)
        cp_impl = SparseMlaCpImpl(
            attn_configs,
            inputs,
            [weights],
            weights[W.rope_cos_sin_cache],
            quant_config=None,
            parallelism_config=parallelism,
        )
        indexer = Indexer(
            attn_config=config.attn_config,
            weights=weights,
            global_weights=weights,
            layer_idx=0,
            layernorm_eps=config.layernorm_eps,
            quant_config=None,
            parallelism_config=parallelism,
            scale_fmt="ue8m0",
        )
        index_cache = self._index_cache(query_length)
        hidden = torch.randn(
            query_length,
            self.HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=self.device,
        )
        q_lora = torch.randn(
            query_length,
            self.Q_LORA_RANK,
            dtype=torch.bfloat16,
            device=self.device,
        )
        indexer.topology_kv_policy = "disabled"
        baseline_topk = indexer(
            hidden, q_lora, index_cache, params, inputs, False, cp_impl.cp_params
        )
        indexer.topology_kv_policy = "topology_only"
        policy_topk = indexer(
            hidden, q_lora, index_cache, params, inputs, False, cp_impl.cp_params
        )
        self.assertTrue(torch.any(cp_impl.cp_params.precomputed_lengths > self.TOPK))
        self.assertFalse(torch.equal(baseline_topk, policy_topk))
        self.assertTrue(torch.all((policy_topk >= 0).any(dim=1)).item())

        q = torch.randn(
            query_length,
            self.MLA_HEADS,
            self.NOPE_DIM + self.ROPE_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        ).mul_(0.1)
        compressed_kv = torch.randn(
            query_length,
            self.KV_LORA_RANK,
            dtype=torch.bfloat16,
            device=self.device,
        ).mul_(0.1)
        k_pe = torch.randn(
            query_length,
            self.ROPE_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        ).mul_(0.1)

        def cache():
            value = LayerKVCache()
            value.kv_cache_base = torch.empty(
                math.ceil(query_length / self.PAGE_SIZE),
                self.PAGE_SIZE,
                656,
                dtype=torch.uint8,
                device=self.device,
            )
            return value

        def compare_with_non_cp(topk_cp):
            fresh_cp_impl = SparseMlaCpImpl(
                attn_configs,
                inputs,
                [weights],
                weights[W.rope_cos_sin_cache],
                quant_config=None,
                parallelism_config=parallelism,
            )
            self.assertTrue(
                torch.equal(
                    fresh_cp_impl.cp_params.total_local_ids,
                    cp_impl.cp_params.total_local_ids,
                )
            )
            plain_inputs, _ = self._attention_inputs(query_length, 0)
            plain_impl = SparseMlaImpl(
                attn_configs,
                plain_inputs,
                [weights],
                weights[W.rope_cos_sin_cache],
                quant_config=None,
                parallelism_config=self._make_parallelism(False),
            )
            topk_plain = torch.empty_like(topk_cp)
            topk_plain[fresh_cp_impl.cp_params.total_local_ids] = topk_cp
            cp_output = fresh_cp_impl.forward(
                q.clone(),
                compressed_kv.clone(),
                k_pe.clone(),
                cache(),
                0,
                topk_cp,
            )
            plain_output = plain_impl.forward(
                q.clone(),
                compressed_kv.clone(),
                k_pe.clone(),
                cache(),
                0,
                topk_plain,
            )
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(cp_output, plain_output, atol=0.15, rtol=0.05),
                f"CP/non-CP mismatch: max_abs={torch.max(torch.abs(cp_output - plain_output)).item()}",
            )

        compare_with_non_cp(baseline_topk)
        compare_with_non_cp(policy_topk)


if __name__ == "__main__":
    main()
