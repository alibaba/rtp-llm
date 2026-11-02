"""Qwen4Exp decoder layer construction and residual-stream wiring.

The gated-residual math itself is covered by
models_py/modules/hybrid/test/gated_residual_test.py (which checks it against
an independent transcription of upstream). This test covers the wiring: that the
layer picks up the right weight tags and threads the wide stream through both
sublayers in the right order.
"""

from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.model_desc import qwen3_next, qwen4_exp
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from rtp_llm.ops import HybridAttentionType, LinearAttentionConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

_HC = 4
_HIDDEN = 8
_LOWRANK = 3
_EXPERTS = 4
_EPS = 1e-6


class _Attention(nn.Module):
    """Records what the attention sublayer was handed."""

    def __init__(self):
        super().__init__()
        self.seen = None
        config = LinearAttentionConfig()
        config.linear_num_key_heads = config.linear_num_value_heads = 1
        config.linear_key_head_dim = config.linear_value_head_dim = _HIDDEN
        config.linear_conv_kernel_dim = 4
        self.prefill_gdn = qwen3_next.Qwen3NextGatedDeltaNetPrefill(
            config,
            ParallelismConfig(),
            {
                W.linear_attn_conv1d_w: torch.zeros(3 * _HIDDEN, 1, 4),
                W.linear_attn_dt_b: torch.zeros(1),
                W.linear_attn_alog: torch.zeros(1),
            },
        )

    def forward(self, hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta):
        self.seen = hidden_states
        return hidden_states * 3.0


class _RoutedBackend(nn.Module):
    includes_shared_expert = False
    fused_experts = None
    topk_ids_dtype = torch.int32
    router = SimpleNamespace(tp_collective_size=1, supports_skip_tp_allreduce=False)

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        return hidden_states * 0.0 + 1.0


class _QsaMainCachePhase:
    def begin_qsa_cache_transaction(self):
        self.main_cache_mutation_started = False

    def qsa_main_cache_mutation_started(self):
        return self.main_cache_mutation_started


def _select_topk(logits, ids, weights):
    values, selected = logits.softmax(dim=-1).topk(ids.shape[-1])
    ids.copy_(selected)
    weights.copy_(values / values.sum(dim=-1, keepdim=True))


_select_topk.config = SimpleNamespace(expert_num=_EXPERTS, moe_k=2, has_moe_norm=True)


def _hc_weights(seed):
    torch.manual_seed(seed)
    hc_hidden = _HC * _HIDDEN
    return (
        torch.randn(hc_hidden) * 0.1,
        torch.randn(_LOWRANK, hc_hidden) * 0.05,
        torch.randn(hc_hidden, _LOWRANK) * 0.05,
        torch.randn(_HC, hc_hidden) * 0.05,
    )


def _hybrid_types():
    return [
        HybridAttentionType.LINEAR,
        HybridAttentionType.LINEAR,
        HybridAttentionType.LINEAR,
        HybridAttentionType.NONE,
    ] * 2


def _spec_config():
    config = ModelConfig()
    config.num_layers = 8
    config.hidden_size = _HIDDEN
    config.hc_mult = _HC
    config.hybrid_attention_config.hybrid_attention_types = _hybrid_types()
    return config


def _fake_kv_cache(descs):
    tags_per_layer = [[d.tag for d in layer] for layer in descs]

    class _FakeKVCache:
        def __init__(self):
            self.layers = [
                [SimpleNamespace(tag=tag) for tag in tags] for tags in tags_per_layer
            ]

        def get_layer_cache_groups(self, layer_idx):
            return self.layers[layer_idx]

        def get_layer_cache(self, layer_idx, tag):
            return next(cache for cache in self.layers[layer_idx] if cache.tag == tag)

    return _FakeKVCache()


def _bare_model(descs, layer_types=None):
    """A ``Qwen4ExpModel`` without ``__init__``, wired to a fake tag topology."""
    model = qwen4_exp.Qwen4ExpModel.__new__(qwen4_exp.Qwen4ExpModel)
    model.kv_cache = _fake_kv_cache(descs)
    if layer_types is not None:
        model.layers = [SimpleNamespace(layer_type=t) for t in layer_types]
    return model


class PLERuntimeFixture:
    _PAGE = 8

    def setUp(self):
        torch.manual_seed(7)
        embedding = Qwen4ExpNGramEmbedding(
            [torch.randn(12, 2, dtype=torch.bfloat16)],
            torch.tensor([5, 7], dtype=torch.long),
            torch.tensor([0, 5], dtype=torch.long),
            torch.tensor([3, 5], dtype=torch.long),
            ngram_size=2,
            eos_token_id=7,
        )
        hc_hidden = _HC * _HIDDEN
        self.ple = Qwen4ExpPLELayer(
            embedding,
            torch.randn(hc_hidden, 4, dtype=torch.bfloat16),
            torch.randn(_HIDDEN, 4, dtype=torch.bfloat16),
            torch.randn(hc_hidden, 1, 2, dtype=torch.bfloat16),
            torch.randn(hc_hidden, dtype=torch.bfloat16),
            torch.randn(hc_hidden, dtype=torch.bfloat16),
            torch.randn(hc_hidden, dtype=torch.bfloat16),
            hc_mult=_HC,
            hidden_size=_HIDDEN,
            conv_kernel_size=2,
            norm_eps=_EPS,
        )
        self.state_base = torch.zeros(
            12,
            self.ple.short_conv_state_len * hc_hidden,
            dtype=torch.bfloat16,
        )
        self.ctx_base = torch.zeros(
            12, self.ple.ple_embedding.context_len, dtype=torch.int64
        )

        class _Cache:
            def __init__(cache_self, outer):
                cache_self.layers = {
                    PLE_STATE_TAG: SimpleNamespace(
                        tag=PLE_STATE_TAG,
                        kv_cache_base=outer.state_base,
                        seq_size_per_block=outer._PAGE,
                    ),
                    PLE_NGRAM_CTX_TAG: SimpleNamespace(
                        tag=PLE_NGRAM_CTX_TAG,
                        kv_cache_base=outer.ctx_base,
                        seq_size_per_block=outer._PAGE,
                    ),
                }

            def get_layer_cache(cache_self, layer_idx, tag):
                self.assertEqual(layer_idx, 1)
                return cache_self.layers[tag]

        self.model = qwen4_exp.Qwen4ExpModel.__new__(qwen4_exp.Qwen4ExpModel)
        nn.Module.__init__(self.model)
        self.model.ple_layers = nn.ModuleDict({"1": self.ple})
        self.model._ple_target_transaction = None
        self.model._ple_graph_target_transactions = {}
        self.model.kv_cache = _Cache(self)
        self.model.config = SimpleNamespace(
            special_tokens=SimpleNamespace(eos_token_id=7)
        )
        self.model.parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False, is_prefill_enabled=lambda: False
            )
        )
        self.state_blocks = torch.tensor([[1, 5, 7], [2, 4, 8]], dtype=torch.int32)
        # Independent pools deliberately use different physical block ids.
        self.ctx_blocks = torch.tensor([[3, 1, 9], [4, 2, 10]], dtype=torch.int32)

    @staticmethod
    def _side_inputs(
        is_prefill, input_lengths, prefix_lengths, sequence_lengths, blocks
    ):
        return SimpleNamespace(
            is_prefill=is_prefill,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            prefix_lengths_device=prefix_lengths.to(blocks.device),
            sequence_lengths=sequence_lengths,
            kv_cache_block_id_device=blocks,
            kv_cache_block_id=blocks,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
        )

    def _inputs_by_tag(
        self,
        *,
        is_prefill,
        input_lengths,
        prefix_lengths,
        sequence_lengths,
    ):
        return {
            PLE_STATE_TAG: self._side_inputs(
                is_prefill,
                input_lengths,
                prefix_lengths,
                sequence_lengths,
                self.state_blocks,
            ),
            PLE_NGRAM_CTX_TAG: self._side_inputs(
                is_prefill,
                input_lengths.clone(),
                prefix_lengths.clone(),
                sequence_lengths.clone(),
                self.ctx_blocks,
            ),
        }

    def _target_inputs(self, prefixes, query_len):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.full((len(prefixes),), query_len, dtype=torch.int32),
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        for value in inputs.values():
            value.is_target_verify = True
        return inputs

    def _seed_target_history(self, prefixes):
        state_rows = []
        context_rows = []
        for request_idx, prefix in enumerate(prefixes):
            page = (prefix - 1) // self._PAGE
            state_row = int(self.state_blocks[request_idx, page])
            context_row = int(self.ctx_blocks[request_idx, page])
            state = torch.randn(
                self.ple.short_conv_state_len,
                _HC * _HIDDEN,
                dtype=torch.bfloat16,
            )
            context = torch.tensor([request_idx + 2], dtype=torch.int64)
            self.state_base[state_row].copy_(state.reshape(-1))
            self.ctx_base[context_row].copy_(context)
            state_rows.append(state)
            context_rows.append(context)
        return torch.stack(state_rows), torch.stack(context_rows)

    def _target_baseline(self, hyper, ids, initial_state, initial_context):
        batch, query_len = ids.shape
        state = initial_state
        context = initial_context
        outputs = []
        states = []
        contexts = []
        for step in range(query_len):
            step_ids = ids[:, step : step + 1]
            output, state = self.ple.decode_step(
                hyper[:, step : step + 1],
                torch.cat([context, step_ids], dim=1),
                state,
            )
            context = torch.cat([context, step_ids], dim=1)[
                :, -self.ple.ple_embedding.context_len :
            ]
            outputs.append(output)
            states.append(state.clone())
            contexts.append(context.clone())
        return torch.cat(outputs, dim=1), states, contexts

    def _prefill(self):
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        hyper = torch.randn(4, _HC * _HIDDEN, dtype=torch.bfloat16)
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([3, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        got = self.model._apply_ple(1, hyper, ids, inputs)
        return ids, hyper, got

    def _move_ple_to_cuda(self, device, cpu_tables=False):
        embedding = self.ple.ple_embedding
        gpu_embedding = Qwen4ExpNGramEmbedding(
            (
                embedding.shards
                if cpu_tables
                else [shard.to(device) for shard in embedding.shards]
            ),
            embedding.vocab_sizes.to(device),
            embedding.offsets.to(device),
            embedding.multipliers.to(device),
            ngram_size=embedding.ngram_size,
            eos_token_id=embedding.eos_token_id,
            inline_cpu_uva_graph=cpu_tables,
        )
        if cpu_tables:
            self.assertTrue(gpu_embedding._cpu_uva_enabled)
            self.assertTrue(all(shard.is_pinned() for shard in gpu_embedding.shards))
        self.ple = Qwen4ExpPLELayer(
            gpu_embedding,
            self.ple.key_proj.to(device),
            self.ple.value_proj.to(device),
            self.ple.conv_weight.to(device),
            self.ple.norm_key.to(device),
            self.ple.norm_query.to(device),
            self.ple.norm_conv.to(device),
            hc_mult=_HC,
            hidden_size=_HIDDEN,
            conv_kernel_size=2,
            norm_eps=_EPS,
        )
        self.model.ple_layers = nn.ModuleDict({"1": self.ple})
        self.state_base = self.state_base.to(device)
        self.ctx_base = self.ctx_base.to(device)
        self.model.kv_cache.layers[PLE_STATE_TAG].kv_cache_base = self.state_base
        self.model.kv_cache.layers[PLE_NGRAM_CTX_TAG].kv_cache_base = self.ctx_base
        self.state_blocks = self.state_blocks.to(device)
        self.ctx_blocks = self.ctx_blocks.to(device)
        return gpu_embedding

    def _enable_async_cpu_ple(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("NVIDIA CUDA is required for asynchronous PLE")
        embedding = self._move_ple_to_cuda(torch.device("cuda"), cpu_tables=True)
        self.assertTrue(embedding.supports_prefetch)
        return embedding
