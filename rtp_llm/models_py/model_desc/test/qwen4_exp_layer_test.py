"""Qwen4Exp decoder layer construction and residual-stream wiring.

The gated-residual math itself is covered by
models_py/modules/qwen4_exp/test/gated_residual_test.py (which checks it against
an independent transcription of upstream). This test covers the wiring: that the
layer picks up the right weight tags and threads the wide stream through both
sublayers in the right order.
"""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3NextBase
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
    PLE_NGRAM_CTX_TAG,
    PLE_STATE_TAG,
    build_qwen4_exp_kv_cache_spec_descs,
)
from rtp_llm.models_py.model_desc import generic_moe, qwen3_next, qwen4_exp
from rtp_llm.models_py.modules.qwen4_exp.gated_residual import (
    Qwen4ExpGatedResidual,
    inject_into_residual,
)
from rtp_llm.models_py.modules.qwen4_exp.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
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

    def forward(self, hidden_states, fmha_impl, kv_cache, attention_inputs, attn_meta):
        self.seen = hidden_states
        return hidden_states * 3.0


class _RoutedBackend(nn.Module):
    includes_shared_expert = False
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


class Qwen4ExpDecoderLayerTest(TestCase):
    def setUp(self):
        self.hc_hidden = _HC * _HIDDEN
        self.config = ModelConfig()
        self.config.num_layers = 1
        self.config.hidden_size = _HIDDEN
        self.config.max_seq_len = 16
        self.config.layernorm_eps = _EPS
        self.config.activation_type = "SiGLU"
        self.config.hc_mult = _HC
        self.config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR
        ]
        Qwen3NextBase._parse_moe_config(
            {
                "num_experts": _EXPERTS,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 0,
            },
            self.config,
        )

        self.attn_hc = _hc_weights(0)
        self.mlp_hc = _hc_weights(1)
        self.weights = {
            W.qwen4_hc_attn_norm: self.attn_hc[0],
            W.qwen4_hc_attn_mix_down: self.attn_hc[1],
            W.qwen4_hc_attn_mix_up: self.attn_hc[2],
            W.qwen4_hc_attn_inject: self.attn_hc[3],
            W.qwen4_hc_mlp_norm: self.mlp_hc[0],
            W.qwen4_hc_mlp_mix_down: self.mlp_hc[1],
            W.qwen4_hc_mlp_mix_up: self.mlp_hc[2],
            W.qwen4_hc_mlp_inject: self.mlp_hc[3],
            W.moe_gate: torch.ones(_EXPERTS, _HIDDEN),
            W.moe_w1: torch.ones(_EXPERTS, _HIDDEN, _HIDDEN),
            W.moe_w2: torch.ones(_EXPERTS, _HIDDEN, 4),
        }

    def _run(self, hyper_states):
        attention = _Attention()
        with (
            patch.object(qwen3_next, "Qwen3NextGatedDeltaNet", return_value=attention),
            patch.object(
                generic_moe.LinearFactory,
                "create_linear_from_weights",
                return_value=nn.Linear(_HIDDEN, _EXPERTS, bias=False),
            ),
            patch.object(generic_moe, "SelectTopk", return_value=_select_topk),
            patch.object(
                generic_moe.FusedMoeFactory,
                "create_fused_moe",
                return_value=_RoutedBackend(),
            ),
        ):
            layer = qwen4_exp.Qwen4ExpDecoderLayer(
                self.config, ParallelismConfig(), self.weights, 0, MoeConfig()
            )
            output = layer(hyper_states, None)
        return layer, attention, output

    def test_layer_has_gated_residual_units_not_layernorms(self):
        layer, _, _ = self._run(torch.randn(3, self.hc_hidden))

        self.assertIsInstance(layer.attn_hyper_connection, Qwen4ExpGatedResidual)
        self.assertIsInstance(layer.mlp_hyper_connection, Qwen4ExpGatedResidual)
        self.assertFalse(hasattr(layer, "input_layernorm"))
        self.assertFalse(hasattr(layer, "post_attention_layernorm"))

    def test_stream_width_is_preserved(self):
        hyper_states = torch.randn(3, self.hc_hidden)

        _, _, output = self._run(hyper_states)

        self.assertEqual(output.shape, hyper_states.shape)

    def test_attention_receives_the_collapsed_stream(self):
        hyper_states = torch.randn(3, self.hc_hidden)

        _, attention, _ = self._run(hyper_states)

        self.assertEqual(attention.seen.shape, (3, _HIDDEN))

    def test_wiring_matches_a_hand_rolled_two_stage_reference(self):
        hyper_states = torch.randn(3, self.hc_hidden)

        _, attention, output = self._run(hyper_states)

        attn_unit = Qwen4ExpGatedResidual(*self.attn_hc, hc_mult=_HC, norm_eps=_EPS)
        mlp_unit = Qwen4ExpGatedResidual(*self.mlp_hc, hc_mult=_HC, norm_eps=_EPS)
        x, hyper, w = attn_unit(hyper_states)
        hyper = inject_into_residual(hyper, x * 3.0, w)
        x, hyper, w = mlp_unit(hyper)
        expected = inject_into_residual(hyper, torch.ones_like(x), w)

        torch.testing.assert_close(output, expected)

    def test_rejects_a_narrow_stream(self):
        with self.assertRaisesRegex(ValueError, "gated residual features"):
            self._run(torch.randn(3, _HIDDEN))

    def test_linear_layer_forwards_the_configured_norm_activation(self):
        # Qwen4-Exp checkpoints gate the GDN output norm with output_gate_type
        # ("sigmoid"); the layer must forward it instead of the silu default.
        self.config.linear_attn_norm_activation = "sigmoid"
        attention = _Attention()
        with (
            patch.object(
                qwen3_next, "Qwen3NextGatedDeltaNet", return_value=attention
            ) as gdn_cls,
            patch.object(
                generic_moe.LinearFactory,
                "create_linear_from_weights",
                return_value=nn.Linear(_HIDDEN, _EXPERTS, bias=False),
            ),
            patch.object(generic_moe, "SelectTopk", return_value=_select_topk),
            patch.object(
                generic_moe.FusedMoeFactory,
                "create_fused_moe",
                return_value=_RoutedBackend(),
            ),
        ):
            qwen4_exp.Qwen4ExpDecoderLayer(
                self.config, ParallelismConfig(), self.weights, 0, MoeConfig()
            )
        self.assertEqual(gdn_cls.call_args.kwargs.get("norm_activation"), "sigmoid")


class Qwen4ExpSpecFaultInjectionTest(TestCase):
    """The env-gated bring-up fault injection used by the multi-rank runs.

    ``RTP_LLM_QWEN4_SPEC_FAULT_INJECT=<rank>:<phase>`` must raise on exactly
    that TP rank and phase, and stay inert for every other rank/phase or when
    unset.
    """

    def _stub(self, tp_rank):
        return SimpleNamespace(
            parallelism_config=SimpleNamespace(get_attn_tp_rank=lambda: tp_rank)
        )

    def test_injection_matches_only_the_configured_rank_and_phase(self):
        call = qwen4_exp.Qwen4ExpModel._maybe_inject_speculative_fault
        stub = self._stub(3)

        with patch.dict("os.environ", {qwen4_exp._SPEC_FAULT_INJECT_ENV: "3:commit"}):
            with self.assertRaisesRegex(RuntimeError, "commit fault injection"):
                call(stub, "commit")
            call(stub, "prepare")  # other phase: inert

        with patch.dict("os.environ", {qwen4_exp._SPEC_FAULT_INJECT_ENV: "5:commit"}):
            call(stub, "commit")  # other rank: inert

        with patch.dict("os.environ", {}, clear=True):
            call(stub, "commit")  # unset: inert


class Qwen4ExpModelConstructionTest(TestCase):
    """The real construction path: GptModelBase.__init__ then Qwen4ExpModel.

    Every other test in this file fakes the base init, which hid a startup
    regression: GptModelBase already owns `_mtp_target_hidden_states` as a plain
    attribute, so re-registering it as a buffer raised
    "attribute ... already exists" on every server start.
    """

    class _FakeModelWeights:
        def __init__(self, embedding, mixer_norm, mix_down, mix_up):
            self._globals = {
                W.embedding: embedding,
                W.qwen4_hc_mixer_norm: mixer_norm,
                W.qwen4_hc_mixer_mix_down: mix_down,
                W.qwen4_hc_mixer_mix_up: mix_up,
            }
            self.weights: list = []

        def get_global_weight(self, name):
            return self._globals[name]

    def test_model_init_reuses_the_base_mtp_capture_attributes(self):
        hidden, vocab = _HIDDEN, 16
        hc_hidden = _HC * hidden
        config = ModelConfig()
        config.num_layers = 0
        config.hidden_size = hidden
        config.vocab_size = vocab
        config.hc_mult = _HC
        config.layernorm_eps = _EPS
        config.capture_aux_hidden_layer_ids = None
        mixer_norm, mix_down, mix_up, _ = _hc_weights(0)
        weights = self._FakeModelWeights(
            torch.zeros(vocab, hidden),
            mixer_norm,
            mix_down,
            mix_up,
        )

        model = qwen4_exp.Qwen4ExpModel(
            config, ParallelismConfig(), weights, MoeConfig(), 0
        )

        self.assertIsNone(model._mtp_target_hidden_states)
        self.assertFalse(model._capture_mtp_target_hidden)
        self.assertEqual(model.hyper_connection_mixer.hc_hidden_size, hc_hidden)


class Qwen4ExpQSAConstructionTest(TestCase):
    @staticmethod
    def _fake_base_init(module, *args, **kwargs):
        nn.Module.__init__(module)

    def _config(self, *, is_mtp=False):
        return SimpleNamespace(
            enable_qwen4_qsa=True,
            is_mtp=is_mtp,
            _qwen4_indexer_head_dim=8,
            _qwen4_indexer_compress_ratio=4,
            _qwen4_indexer_budget=8,
            _qwen4_indexer_kv_heads=1,
            attn_config=SimpleNamespace(
                indexer_head_num=4,
                rope_config=SimpleNamespace(
                    index_factor=3,
                    dim=8,
                    mrope_dim1=2,
                    mrope_dim2=1,
                    mrope_dim3=1,
                    mrope_interleaved=True,
                    base=10_000,
                ),
            ),
        )

    def _weights(self):
        return {
            W.qwen4_indexer_qk_proj_w: torch.randn(40, 16),
            W.qwen4_indexer_q_ln_gamma: torch.randn(8),
            W.qwen4_indexer_k_ln_gamma: torch.randn(8),
        }

    def test_constructs_checkpoint_backed_indexer(self):
        weights = self._weights()
        with patch.object(
            qwen3_next.Qwen3NextAttention,
            "__init__",
            self._fake_base_init,
        ):
            attention = qwen4_exp.Qwen4ExpAttention(
                SimpleNamespace(),
                SimpleNamespace(),
                weights,
                _EPS,
                self._config(),
                layer_idx=3,
            )

        self.assertIs(attention.qsa_indexer.qk_proj, weights[W.qwen4_indexer_qk_proj_w])
        self.assertEqual(attention.qsa_indexer.max_selected, 11)
        self.assertEqual(attention.layer_idx, 3)

    @staticmethod
    def _prefill_inputs():
        positions = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [0, 0, 0],
                [1, 3, 5],
            ],
            dtype=torch.int32,
        )
        return SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=torch.tensor([3, 2], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            combo_position_ids=positions.reshape(-1),
        )

    @staticmethod
    def _decode_inputs():
        return SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=torch.tensor([5, 7], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([5, 7], dtype=torch.int32),
            combo_position_ids=torch.tensor(
                [[5, 5, 5], [7, 7, 7]], dtype=torch.int32
            ).reshape(-1),
        )

    def _attention(self, *, is_mtp=False):
        with patch.object(
            qwen3_next.Qwen3NextAttention,
            "__init__",
            self._fake_base_init,
        ):
            attention = qwen4_exp.Qwen4ExpAttention(
                SimpleNamespace(),
                SimpleNamespace(),
                self._weights(),
                _EPS,
                self._config(is_mtp=is_mtp),
                layer_idx=3,
            )
        attention.qkv_proj = nn.Identity()
        attention.qk_fuse_norm = None
        attention.gate = nn.Linear(16, 16, bias=False)
        attention.gate.weight.data.zero_()
        attention.o_proj = nn.Identity()
        attention.tp_size = 1
        return attention

    def _runtime(self, attention, inputs):
        class _Runtime:
            main_cache = None
            main_inputs = inputs
            is_mtp_draft = attention.is_mtp_draft
            rollback_count = 0
            finalize_count = 0

            def rollback_side_cache(runtime_self):
                runtime_self.rollback_count += 1

            def finalize_side_cache(runtime_self):
                runtime_self.finalize_count += 1

            def validate_before_projection(runtime_self, **kwargs):
                runtime_self.validated_before_projection = True

            def write_prefill_indexer_cache(runtime_self, raw_keys, **kwargs):
                runtime_self.raw_keys = raw_keys
                lengths = [int(value) for value in inputs.input_lengths.tolist()]
                max_len = max(lengths)
                rope_dim = int(attention.qsa_rope_config.dim)
                cos = torch.ones(len(lengths), max_len, rope_dim)
                sin = torch.zeros_like(cos)
                return lengths, cos, sin, {}

            def select_decode_tokens(runtime_self, q, raw_keys, **kwargs):
                runtime_self.q = q
                runtime_self.raw_keys = raw_keys
                return torch.zeros(
                    q.shape[0],
                    attention.qsa_indexer.max_selected,
                    dtype=torch.int32,
                    device=q.device,
                )

            def select_target_verify_tokens(runtime_self, q, raw_keys, **kwargs):
                runtime_self.q = q
                runtime_self.raw_keys = raw_keys
                return torch.zeros(
                    q.shape[0],
                    attention.qsa_indexer.max_selected,
                    dtype=torch.int32,
                    device=q.device,
                )

            def select_draft_incremental_prefill_tokens(
                runtime_self, q, raw_keys, **kwargs
            ):
                runtime_self.draft_q = q
                runtime_self.raw_keys = raw_keys
                return torch.zeros(
                    q.shape[0],
                    attention.qsa_indexer.max_selected,
                    dtype=torch.int32,
                    device=q.device,
                )

        return _Runtime()

    def test_ragged_prefill_hands_packed_indexer_selection_to_sparse_fmha(self):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.selected = None

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention()
        fmha = _SparseFmha()
        hidden = torch.randn(5, 16)
        inputs = self._prefill_inputs()
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            output = attention(hidden, fmha, None, inputs, qsa_runtime=runtime)

        self.assertEqual(output.shape, hidden.shape)
        project.assert_called_once_with(hidden)
        self.assertEqual(runtime.raw_keys.shape, (5, 8))
        self.assertEqual(fmha.selected.shape, (5, 11))
        self.assertEqual(fmha.selected.dtype, torch.int32)
        self.assertEqual(runtime.rollback_count, 0)
        self.assertEqual(runtime.finalize_count, 1)
        # Each request owns a local token-index space; the second request must
        # restart at zero instead of indexing into the first request's K/V.
        self.assertLessEqual(int(fmha.selected[:3].max().item()), 2)
        self.assertLessEqual(int(fmha.selected[3:].max().item()), 1)

    def test_main_attention_failure_rolls_back_qsa_side_transaction(self):
        class _FailingSparseFmha(_QsaMainCachePhase):
            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                raise RuntimeError("injected main attention failure")

        attention = self._attention()
        inputs = self._prefill_inputs()
        runtime = self._runtime(attention, inputs)

        with self.assertRaisesRegex(RuntimeError, "injected main attention failure"):
            attention(
                torch.randn(5, 16),
                _FailingSparseFmha(),
                None,
                inputs,
                qsa_runtime=runtime,
            )

        self.assertEqual(runtime.rollback_count, 1)
        self.assertEqual(runtime.finalize_count, 0)

    def test_failure_after_main_cache_write_preserves_side_cache(self):
        class _FailingSparseFmha(_QsaMainCachePhase):
            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                self.main_cache_mutation_started = True
                raise RuntimeError("injected failure after main cache write")

        attention = self._attention()
        inputs = self._prefill_inputs()
        runtime = self._runtime(attention, inputs)

        with self.assertRaisesRegex(RuntimeError, "after main cache write"):
            attention(
                torch.randn(5, 16),
                _FailingSparseFmha(),
                None,
                inputs,
                qsa_runtime=runtime,
            )

        self.assertEqual(runtime.rollback_count, 0)
        self.assertEqual(runtime.finalize_count, 1)

    def test_decode_projects_once_and_hands_paged_selection_to_sparse_fmha(self):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.selected = None

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention()
        fmha = _SparseFmha()
        hidden = torch.randn(2, 16)
        inputs = self._decode_inputs()
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            output = attention(hidden, fmha, None, inputs, qsa_runtime=runtime)

        self.assertEqual(output.shape, hidden.shape)
        project.assert_called_once_with(hidden)
        self.assertEqual(runtime.q.shape, (2, 4, 8))
        self.assertEqual(runtime.raw_keys.shape, (2, 8))
        self.assertEqual(fmha.selected.shape, (2, 11))
        self.assertEqual(fmha.selected.dtype, torch.int32)

    def test_target_verify_projects_once_and_hands_paged_selection_to_sparse_fmha(self):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.selected = None

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention()
        fmha = _SparseFmha()
        inputs = self._prefill_inputs()
        inputs.is_target_verify = True
        inputs.input_lengths = torch.tensor([4], dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            output = attention(
                torch.randn(4, 16), fmha, None, inputs, qsa_runtime=runtime
            )

        self.assertEqual(output.shape, (4, 16))
        project.assert_called_once()
        self.assertTrue(runtime.validated_before_projection)
        self.assertEqual(runtime.raw_keys.shape, (4, 8))
        self.assertEqual(fmha.selected.shape, (4, 11))

    def test_mtp_draft_nonzero_prefix_sets_explicit_gate_and_uses_incremental_path(
        self,
    ):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.is_mtp_draft = False
                self.selected = None

            def set_mtp_draft_mode(self, enabled):
                self.is_mtp_draft = enabled

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention(is_mtp=True)
        fmha = _SparseFmha()
        inputs = self._prefill_inputs()
        inputs.prefix_lengths = torch.tensor([7, 11], dtype=torch.int32)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        runtime = self._runtime(attention, inputs)

        output = attention(torch.randn(5, 16), fmha, None, inputs, qsa_runtime=runtime)

        self.assertEqual(output.shape, (5, 16))
        self.assertTrue(fmha.is_mtp_draft)
        self.assertEqual(runtime.draft_q.shape, (5, 4, 8))
        self.assertEqual(runtime.raw_keys.shape, (5, 8))
        self.assertEqual(fmha.selected.shape, (5, 11))

    def test_target_verify_validation_fails_before_projection_and_main_writer(self):
        events = []

        class _SparseFmha(_QsaMainCachePhase):
            def set_selected_indices(self, selected):
                events.append("selection")

            def validate_qsa_before_side_write(self, runtime, indexer, hidden):
                events.append("validation")
                raise RuntimeError("invalid target geometry")

            def forward(self, qkv, kv_cache, layer_idx):
                events.append("main_writer")
                return qkv

        attention = self._attention()
        inputs = self._prefill_inputs()
        inputs.is_target_verify = True
        inputs.input_lengths = torch.tensor([4], dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            with self.assertRaisesRegex(RuntimeError, "invalid target geometry"):
                attention(
                    torch.randn(4, 16),
                    _SparseFmha(),
                    None,
                    inputs,
                    qsa_runtime=runtime,
                )

        project.assert_not_called()
        self.assertEqual(events, ["validation"])

    def test_enabled_indexer_rejects_dense_fmha(self):
        attention = self._attention()

        with self.assertRaisesRegex(RuntimeError, "dense fallback is forbidden"):
            attention(torch.randn(5, 16), object(), None, self._prefill_inputs())

    def test_enabled_indexer_requires_side_cache_context(self):
        class _SparseFmha:
            def set_selected_indices(self, selected):
                pass

        attention = self._attention()
        with self.assertRaisesRegex(RuntimeError, "side-cache regions"):
            attention(torch.randn(5, 16), _SparseFmha(), None, self._prefill_inputs())


class Qwen4ExpAttentionTagRoutingTest(TestCase):
    """The PLE layer owns three cache regions, so attention must be tag-routed.

    Tags come from the real desc builder so this fails if either side renames a
    region.
    """

    def _descs(self):
        return build_qwen4_exp_kv_cache_spec_descs(
            _spec_config(), ple_layer_indices=[1], ple_conv_kernel_size=4, ngram_size=3
        )

    def test_ple_layer_resolves_its_single_attention_region(self):
        descs = self._descs()
        model = _bare_model(descs)

        self.assertEqual(len(descs[1]), 3)
        self.assertEqual(model._attention_tag(1), descs[1][0].tag)
        self.assertNotIn(model._attention_tag(1), (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG))

    def test_plain_layers_resolve_their_own_region(self):
        descs = self._descs()
        model = _bare_model(descs)

        for layer_idx in range(len(descs)):
            if layer_idx == 1:
                continue
            self.assertEqual(model._attention_tag(layer_idx), descs[layer_idx][0].tag)

    def test_rejects_a_layer_with_two_attention_regions(self):
        descs = self._descs()
        descs[0] = descs[0] + [descs[3][0]]
        model = _bare_model(descs)

        with self.assertRaisesRegex(RuntimeError, "exactly one attention region"):
            model._attention_tag(0)


class Qwen4ExpFmhaRoutingTest(TestCase):
    """Indexer regions must not leak into FMHA routing.

    With the indexer enabled a full-attention layer owns three regions. Without
    a filter ``_get_fmha_group_tags`` would ask the factory to build impls for
    the indexer pools and ``_layer_fmha_impl`` would hand ``self_attn`` a list
    of every region's impl instead of the attention one.
    """

    def _descs(self):
        return build_qwen4_exp_kv_cache_spec_descs(
            _spec_config(),
            ple_layer_indices=[1],
            ple_conv_kernel_size=4,
            ngram_size=3,
            indexer_head_dim=128,
            indexer_compress_ratio=4,
        )

    def _full_layer(self):
        return SimpleNamespace(layer_type=HybridAttentionType.NONE)

    def test_the_indexer_really_adds_two_regions(self):
        descs = self._descs()

        self.assertEqual(
            [d.tag for d in descs[3]], ["full", INDEXER_KV_TAG, INDEXER_STATE_TAG]
        )

    def test_fmha_impls_build_only_for_the_attention_region(self):
        model = _bare_model(self._descs(), _hybrid_types())

        self.assertEqual(model._get_fmha_group_tags(), ["full"])

    def test_full_attention_layer_resolves_its_single_impl(self):
        model = _bare_model(self._descs(), _hybrid_types())
        impls = {
            "full": object(),
            INDEXER_KV_TAG: object(),
            INDEXER_STATE_TAG: object(),
        }

        chosen = model._layer_fmha_impl(self._full_layer(), impls, 3)

        self.assertIs(chosen, impls["full"])

    def test_linear_layer_gets_no_impl(self):
        model = _bare_model(self._descs(), _hybrid_types())

        chosen = model._layer_fmha_impl(
            SimpleNamespace(layer_type=HybridAttentionType.LINEAR),
            {"full": object()},
            0,
        )

        self.assertIsNone(chosen)

    def test_an_untagged_impl_passes_through(self):
        model = _bare_model(self._descs(), _hybrid_types())
        impl = object()

        self.assertIs(model._layer_fmha_impl(self._full_layer(), impl, 3), impl)

    def test_rejects_a_mapping_without_the_attention_tag(self):
        model = _bare_model(self._descs(), _hybrid_types())

        with self.assertRaisesRegex(RuntimeError, "FMHA impl for tag"):
            model._layer_fmha_impl(self._full_layer(), {INDEXER_KV_TAG: object()}, 3)

    def test_qsa_runtime_context_resolves_all_three_tag_local_regions(self):
        descs = self._descs()
        model = _bare_model(descs)
        model.config = SimpleNamespace(is_mtp=True)
        main_cache = model.kv_cache.get_layer_cache(3, "full")
        main_inputs = object()
        kv_inputs = object()
        state_inputs = object()
        attention_inputs = {
            "full": main_inputs,
            INDEXER_KV_TAG: kv_inputs,
            INDEXER_STATE_TAG: state_inputs,
        }

        context = model._qsa_runtime_context(
            3, main_cache, main_inputs, attention_inputs
        )

        self.assertIs(context.main_cache, main_cache)
        self.assertIs(context.main_inputs, main_inputs)
        self.assertEqual(context.indexer_kv_cache.tag, INDEXER_KV_TAG)
        self.assertIs(context.indexer_kv_inputs, kv_inputs)
        self.assertEqual(context.indexer_state_cache.tag, INDEXER_STATE_TAG)
        self.assertIs(context.indexer_state_inputs, state_inputs)
        self.assertTrue(context.is_mtp_draft)


class Qwen4ExpPLERuntimeTest(TestCase):
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

    def test_ragged_prefill_and_decode_use_tag_local_state_pools(self):
        ids, hyper, got = self._prefill()
        expected = []
        offset = 0
        for length in (3, 1):
            seq = ids[offset : offset + length].view(1, length)
            history = torch.cat([seq.new_full((1, 1), 7), seq], dim=1)
            expected.append(
                self.ple(hyper[offset : offset + length].view(1, length, -1), history)[
                    0
                ]
            )
            offset += length
        torch.testing.assert_close(got, hyper + torch.cat(expected))
        torch.testing.assert_close(self.ctx_base[3], torch.tensor([3]))
        torch.testing.assert_close(self.ctx_base[4], torch.tensor([4]))

        decode_ids = torch.tensor([5, 6], dtype=torch.long)
        decode_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        old_state = self.state_base[[1, 2]].view(2, self.ple.short_conv_state_len, -1)
        old_ctx = self.ctx_base[[3, 4]]
        history = torch.cat([old_ctx, decode_ids.view(2, 1)], dim=1)
        expected_out, expected_state = self.ple.decode_step(
            decode_hyper.view(2, 1, -1), history, old_state
        )
        inputs = self._inputs_by_tag(
            is_prefill=False,
            # Decode input_lengths carries each request's original prompt
            # length; one current token per row is represented by ids.numel().
            input_lengths=torch.tensor([3, 1], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([3, 1], dtype=torch.int32),
        )
        got = self.model._apply_ple(1, decode_hyper, decode_ids, inputs)
        torch.testing.assert_close(got, decode_hyper + expected_out.squeeze(1))
        torch.testing.assert_close(
            self.state_base[[1, 2]].view_as(expected_state), expected_state
        )
        torch.testing.assert_close(self.ctx_base[[3, 4]], decode_ids.view(2, 1))

    def test_target_verify_commit_finalize_and_next_decode_match_baseline(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        query_len = 5
        accepted = torch.tensor([1, query_len], dtype=torch.int32)
        initial_state, initial_context = self._seed_target_history(prefixes)
        ids = torch.tensor([[4, 5, 6, 8, 9], [3, 4, 5, 6, 8]], dtype=torch.long)
        hyper = torch.randn(2, query_len, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_output, states, contexts = self._target_baseline(
            hyper, ids, initial_state, initial_context
        )
        inputs = self._target_inputs(prefixes, query_len)
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()

        got = self.model._apply_ple(
            1, hyper.reshape(-1, _HC * _HIDDEN), ids.reshape(-1), inputs
        )

        torch.testing.assert_close(got.reshape_as(hyper), hyper + expected_output)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

        self.model.prepare_speculative_target_commit(accepted)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

        self.model.finish_speculative_target_commit(True)
        final_positions = torch.tensor(prefixes) + accepted.to(torch.long) - 1
        final_pages = torch.div(final_positions, self._PAGE, rounding_mode="floor")
        state_destinations = self.state_blocks.gather(
            1, final_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        context_destinations = self.ctx_blocks.gather(
            1, final_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        expected_state = torch.stack(
            [states[int(accepted[b]) - 1][b] for b in range(2)]
        )
        expected_context = torch.stack(
            [contexts[int(accepted[b]) - 1][b] for b in range(2)]
        )
        torch.testing.assert_close(
            self.state_base.index_select(0, state_destinations).view_as(expected_state),
            expected_state,
        )
        torch.testing.assert_close(
            self.ctx_base.index_select(0, context_destinations), expected_context
        )
        self.assertIsNotNone(self.model._ple_target_transaction)

        self.model.finalize_speculative_target_commit()
        self.assertIsNone(self.model._ple_target_transaction)

        next_ids = torch.tensor([10, 11], dtype=torch.long)
        next_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_next, expected_next_state = self.ple.decode_step(
            next_hyper.view(2, 1, -1),
            torch.cat([expected_context, next_ids.view(2, 1)], dim=1),
            expected_state,
        )
        committed_lengths = torch.tensor(prefixes) + accepted
        decode_inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=committed_lengths.to(torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=committed_lengths.to(torch.int32),
        )
        got_next = self.model._apply_ple(1, next_hyper, next_ids, decode_inputs)
        torch.testing.assert_close(got_next, next_hyper + expected_next.squeeze(1))
        next_write_pages = torch.div(
            committed_lengths, self._PAGE, rounding_mode="floor"
        )
        next_state_destinations = self.state_blocks.gather(
            1, next_write_pages.unsqueeze(1).to(torch.long)
        ).squeeze(1)
        torch.testing.assert_close(
            self.state_base.index_select(0, next_state_destinations).view_as(
                expected_next_state
            ),
            expected_next_state,
        )

    def test_target_verify_tentative_commit_can_be_rolled_back_idempotently(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        query_len = 3
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, query_len)
        hyper = torch.randn(2, query_len, _HC * _HIDDEN, dtype=torch.bfloat16)
        ids = torch.tensor([[4, 5, 6], [8, 9, 10]], dtype=torch.long)
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()
        self.model._apply_ple(
            1, hyper.reshape(-1, _HC * _HIDDEN), ids.reshape(-1), inputs
        )
        self.model.prepare_speculative_target_commit(
            torch.tensor([2, 3], dtype=torch.int32)
        )

        self.model.finish_speculative_target_commit(True)
        self.assertFalse(torch.equal(self.state_base, state_before))
        self.assertFalse(torch.equal(self.ctx_base, context_before))

        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)
        self.assertIsNone(self.model._ple_target_transaction)
        self.model.finish_speculative_target_commit(False)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

    def test_target_prepare_rejects_invalid_accept_len_without_side_effects(self):
        cases = (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1, 4], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int32),
        )
        for accept_len in cases:
            with self.subTest(accept_len=accept_len):
                prefixes = (self._PAGE - 1, self._PAGE)
                self._seed_target_history(prefixes)
                inputs = self._target_inputs(prefixes, 3)
                state_before = self.state_base.clone()
                context_before = self.ctx_base.clone()
                self.model._apply_ple(
                    1,
                    torch.randn(6, _HC * _HIDDEN, dtype=torch.bfloat16),
                    torch.tensor([4, 5, 6, 8, 9, 10]),
                    inputs,
                )

                with self.assertRaisesRegex(RuntimeError, "accept_len"):
                    self.model.prepare_speculative_target_commit(accept_len)

                torch.testing.assert_close(self.state_base, state_before)
                torch.testing.assert_close(self.ctx_base, context_before)
                self.model.finish_speculative_target_commit(False)

    def test_target_prepare_rejects_missing_or_duplicate_destinations_before_write(
        self,
    ):
        prefixes = (2 * self._PAGE - 1, 2 * self._PAGE - 1)
        query_len = 2
        for failure in ("missing", "duplicate"):
            with self.subTest(failure=failure):
                self.state_base.zero_()
                self.ctx_base.zero_()
                self.model._ple_target_transaction = None
                initial_state, initial_context = self._seed_target_history(prefixes)
                self.assertEqual(initial_state.shape[0], 2)
                self.assertEqual(initial_context.shape[0], 2)
                inputs = self._target_inputs(prefixes, query_len)
                if failure == "missing":
                    inputs[PLE_STATE_TAG].kv_cache_block_id_device = (
                        self.state_blocks.clone()
                    )
                    inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 2] = 0
                else:
                    duplicate = self.state_blocks.clone()
                    duplicate[1, 2] = duplicate[0, 2]
                    inputs[PLE_STATE_TAG].kv_cache_block_id_device = duplicate
                state_before = self.state_base.clone()
                context_before = self.ctx_base.clone()
                self.model._apply_ple(
                    1,
                    torch.randn(
                        2 * query_len,
                        _HC * _HIDDEN,
                        dtype=torch.bfloat16,
                    ),
                    torch.tensor([4, 5, 8, 9]),
                    inputs,
                )

                expected = "unallocated" if failure == "missing" else "duplicate"
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.model.prepare_speculative_target_commit(
                        torch.tensor([2, 2], dtype=torch.int32)
                    )

                torch.testing.assert_close(self.state_base, state_before)
                torch.testing.assert_close(self.ctx_base, context_before)
                self.model.finish_speculative_target_commit(False)

    def test_target_prepare_requires_every_ple_layer_before_any_write(self):
        self.model.ple_layers["2"] = self.ple
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, 2)
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()
        self.model._apply_ple(
            1,
            torch.randn(4, _HC * _HIDDEN, dtype=torch.bfloat16),
            torch.tensor([4, 5, 8, 9]),
            inputs,
        )

        with self.assertRaisesRegex(RuntimeError, r"missing=\[2\]"):
            self.model.prepare_speculative_target_commit(
                torch.tensor([1, 2], dtype=torch.int32)
            )

        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)
        self.model.finish_speculative_target_commit(False)

    def test_target_prepare_syncs_before_publishing_and_abort_syncs_before_clear(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, 2)
        self.model._apply_ple(
            1,
            torch.randn(4, _HC * _HIDDEN, dtype=torch.bfloat16),
            torch.tensor([4, 5, 8, 9]),
            inputs,
        )
        transaction = self.model._ple_target_transaction
        self.assertIsNotNone(transaction)
        sync_observations = []

        def observe_sync(current):
            sync_observations.append(
                (
                    current.prepared_writes is None,
                    self.model._ple_target_transaction is current,
                )
            )

        self.model._synchronize_ple_transaction = observe_sync
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 2], dtype=torch.int32)
        )
        self.assertEqual(sync_observations, [(True, True)])
        self.assertIsNotNone(transaction.prepared_writes)

        self.model.finish_speculative_target_commit(False)
        self.assertEqual(sync_observations, [(True, True), (False, True)])
        self.assertIsNone(self.model._ple_target_transaction)

    def test_target_prepare_sync_failure_is_settled_by_abort(self):
        prefixes = (self._PAGE - 1, self._PAGE)
        self._seed_target_history(prefixes)
        inputs = self._target_inputs(prefixes, 2)
        self.model._apply_ple(
            1,
            torch.randn(4, _HC * _HIDDEN, dtype=torch.bfloat16),
            torch.tensor([4, 5, 8, 9]),
            inputs,
        )
        transaction = self.model._ple_target_transaction
        state_before = self.state_base.clone()
        context_before = self.ctx_base.clone()
        sync_calls = []

        def fail_prepare_sync(current):
            sync_calls.append(current)
            raise RuntimeError("injected prepare synchronize failure")

        self.model._synchronize_ple_transaction = fail_prepare_sync
        with self.assertRaisesRegex(RuntimeError, "injected prepare"):
            self.model.prepare_speculative_target_commit(
                torch.tensor([1, 2], dtype=torch.int32)
            )
        self.assertEqual(len(sync_calls), 1)
        self.assertIs(sync_calls[0], transaction)
        self.assertIsNone(transaction.prepared_writes)
        self.assertIs(self.model._ple_target_transaction, transaction)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

        self.model._synchronize_ple_transaction = sync_calls.append
        self.model.finish_speculative_target_commit(False)
        self.assertEqual(len(sync_calls), 2)
        self.assertIs(sync_calls[1], transaction)
        self.assertIsNone(self.model._ple_target_transaction)
        torch.testing.assert_close(self.state_base, state_before)
        torch.testing.assert_close(self.ctx_base, context_before)

    def test_target_prepare_without_staging_fails_when_ple_is_enabled(self):
        self.assertIsNone(getattr(self.model, "_ple_target_transaction", None))
        with self.assertRaisesRegex(RuntimeError, "no staged transaction"):
            self.model.prepare_speculative_target_commit(
                torch.tensor([1, 1], dtype=torch.int32)
            )

        self.model.ple_layers = nn.ModuleDict()
        self.model.prepare_speculative_target_commit(
            torch.tensor([1, 1], dtype=torch.int32)
        )

    def test_multimodal_placeholder_fails_fast(self):
        model_inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2, 99, 4], dtype=torch.long),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([True, True, False, True])
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "multimodal"):
            self.model._ple_input_ids(model_inputs)

        features_only = SimpleNamespace(
            input_ids=torch.tensor([1], dtype=torch.long),
            embedding_inputs=SimpleNamespace(text_tokens_mask=None),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[torch.ones(1, _HIDDEN)]
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "multimodal"):
            self.model._ple_input_ids(features_only)

    def test_ple_input_ids_rejects_mismatched_text_mask(self):
        model_inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.long),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([True], dtype=torch.bool)
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "mask size"):
            self.model._ple_input_ids(model_inputs)

    def test_decode_moves_state_across_a_page_boundary(self):
        ids = torch.arange(1, 2 * self._PAGE + 1, dtype=torch.long)
        hyper = torch.randn(2 * self._PAGE, _HC * _HIDDEN, dtype=torch.bfloat16)
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE, self._PAGE], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        self.model._apply_ple(1, hyper, ids, inputs)

        first_state = (
            self.state_base[[1, 2]].clone().view(2, self.ple.short_conv_state_len, -1)
        )
        first_ctx = self.ctx_base[[3, 4]].clone()
        decode_ids = torch.tensor([21, 22], dtype=torch.long)
        decode_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_out, expected_state = self.ple.decode_step(
            decode_hyper.view(2, 1, -1),
            torch.cat([first_ctx, decode_ids.view(2, 1)], dim=1),
            first_state,
        )
        decode_inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor([self._PAGE, self._PAGE], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([self._PAGE, self._PAGE], dtype=torch.int32),
        )

        got = self.model._apply_ple(1, decode_hyper, decode_ids, decode_inputs)

        torch.testing.assert_close(got, decode_hyper + expected_out.squeeze(1))
        torch.testing.assert_close(
            self.state_base[[5, 4]].view_as(expected_state), expected_state
        )
        torch.testing.assert_close(self.ctx_base[[1, 2]], decode_ids.view(2, 1))
        torch.testing.assert_close(
            self.state_base[[1, 2]].view_as(first_state), first_state
        )
        torch.testing.assert_close(self.ctx_base[[3, 4]], first_ctx)

        # The next step is wholly inside page 1: it must consume and update the
        # migrated state rather than falling back to page 0.
        next_ids = torch.tensor([23, 24], dtype=torch.long)
        next_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        expected_out, expected_state = self.ple.decode_step(
            next_hyper.view(2, 1, -1),
            torch.cat([decode_ids.view(2, 1), next_ids.view(2, 1)], dim=1),
            expected_state,
        )
        for tag in (PLE_STATE_TAG, PLE_NGRAM_CTX_TAG):
            decode_inputs[tag].sequence_lengths = torch.tensor(
                [self._PAGE + 1, self._PAGE + 1], dtype=torch.int32
            )
        got = self.model._apply_ple(1, next_hyper, next_ids, decode_inputs)
        torch.testing.assert_close(got, next_hyper + expected_out.squeeze(1))
        torch.testing.assert_close(
            self.state_base[[5, 4]].view_as(expected_state), expected_state
        )
        torch.testing.assert_close(self.ctx_base[[1, 2]], next_ids.view(2, 1))

    def test_decode_rejects_an_unallocated_current_page(self):
        inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor([self._PAGE, 1], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([self._PAGE, 1], dtype=torch.int32),
        )
        inputs[PLE_STATE_TAG].kv_cache_block_id_device = self.state_blocks.clone()
        inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 1] = 0

        with self.assertRaisesRegex(RuntimeError, "unallocated logical page"):
            self.model._apply_ple(
                1,
                torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(2, dtype=torch.long),
                inputs,
            )

    def test_multi_page_ragged_prefill_writes_only_terminal_then_decodes(self):
        lengths = (self._PAGE + 3, 2 * self._PAGE)
        token_count = sum(lengths)
        ids = torch.arange(1, token_count + 1, dtype=torch.long)
        hyper = torch.randn(token_count, _HC * _HIDDEN, dtype=torch.bfloat16)
        prefill_inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )

        got = self.model._apply_ple(1, hyper, ids, prefill_inputs)

        expected_outputs = []
        expected_states = []
        expected_contexts = []
        offset = 0
        for length in lengths:
            seq_ids = ids[offset : offset + length].view(1, length)
            history = torch.cat(
                [
                    seq_ids.new_full((1, self.ple.ple_embedding.context_len), 7),
                    seq_ids,
                ],
                dim=1,
            )
            output, state = self.ple.prefill(
                hyper[offset : offset + length].view(1, length, -1), history
            )
            expected_outputs.append(output.squeeze(0))
            expected_states.append(state.squeeze(0))
            expected_contexts.append(history[0, -self.ple.ple_embedding.context_len :])
            offset += length

        expected_states = torch.stack(expected_states)
        expected_contexts = torch.stack(expected_contexts)
        torch.testing.assert_close(got, hyper + torch.cat(expected_outputs, dim=0))

        # Both prompts end on logical page 1. Their page-0 and reserved page-2
        # rows must remain untouched even though the physical ids are sparse and
        # differ between the two typed pools.
        terminal_state_blocks = self.state_blocks[:, 1].to(torch.long)
        terminal_ctx_blocks = self.ctx_blocks[:, 1].to(torch.long)
        torch.testing.assert_close(
            self.state_base.index_select(0, terminal_state_blocks).view_as(
                expected_states
            ),
            expected_states,
        )
        torch.testing.assert_close(
            self.ctx_base.index_select(0, terminal_ctx_blocks), expected_contexts
        )
        untouched_state_blocks = self.state_blocks[:, [0, 2]].reshape(-1).to(torch.long)
        untouched_ctx_blocks = self.ctx_blocks[:, [0, 2]].reshape(-1).to(torch.long)
        self.assertEqual(
            int(
                torch.count_nonzero(
                    self.state_base.index_select(0, untouched_state_blocks)
                ).item()
            ),
            0,
        )
        self.assertEqual(
            int(
                torch.count_nonzero(
                    self.ctx_base.index_select(0, untouched_ctx_blocks)
                ).item()
            ),
            0,
        )

        decode_ids = torch.tensor([31, 32], dtype=torch.long)
        decode_hyper = torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16)
        decode_history = torch.cat([expected_contexts, decode_ids.view(2, 1)], dim=1)
        expected_output, expected_new_state = self.ple.decode_step(
            decode_hyper.view(2, 1, -1), decode_history, expected_states
        )
        decode_inputs = self._inputs_by_tag(
            is_prefill=False,
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor(lengths, dtype=torch.int32),
        )

        got = self.model._apply_ple(1, decode_hyper, decode_ids, decode_inputs)

        torch.testing.assert_close(got, decode_hyper + expected_output.squeeze(1))
        # P+3 stays within logical page 1; 2P crosses from page 1 to page 2.
        torch.testing.assert_close(
            self.state_base[5].view_as(expected_new_state[0]),
            expected_new_state[0],
        )
        torch.testing.assert_close(
            self.state_base[8].view_as(expected_new_state[1]),
            expected_new_state[1],
        )
        torch.testing.assert_close(
            self.ctx_base[1], decode_history[0, -self.ple.ple_embedding.context_len :]
        )
        torch.testing.assert_close(
            self.ctx_base[10], decode_history[1, -self.ple.ple_embedding.context_len :]
        )
        # The read-side checkpoint is not overwritten on the crossing row.
        torch.testing.assert_close(
            self.state_base[4].view_as(expected_states[1]), expected_states[1]
        )
        torch.testing.assert_close(self.ctx_base[2], expected_contexts[1])

    def test_prefill_rejects_an_unallocated_terminal_page(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE + 1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        inputs[PLE_STATE_TAG].kv_cache_block_id_device = self.state_blocks.clone()
        inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 1] = 0

        with self.assertRaisesRegex(RuntimeError, "unallocated logical page"):
            self.model._apply_ple(
                1,
                torch.randn(self._PAGE + 2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(self._PAGE + 2, dtype=torch.long),
                inputs,
            )

    def test_prefill_rejects_a_terminal_page_outside_the_block_table(self):
        length = 3 * self._PAGE + 1
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([length, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )

        with self.assertRaisesRegex(RuntimeError, "logical page exceeds"):
            self.model._apply_ple(
                1,
                torch.randn(length + 1, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(length + 1, dtype=torch.long),
                inputs,
            )

    def test_prefill_rejects_a_terminal_physical_block_outside_the_pool(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE + 1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        inputs[PLE_STATE_TAG].kv_cache_block_id_device = self.state_blocks.clone()
        inputs[PLE_STATE_TAG].kv_cache_block_id_device[0, 1] = int(
            self.state_base.shape[0]
        )

        with self.assertRaisesRegex(RuntimeError, "physical block id exceeds"):
            self.model._apply_ple(
                1,
                torch.randn(self._PAGE + 2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(self._PAGE + 2, dtype=torch.long),
                inputs,
            )

    def test_prefill_rejects_state_context_metadata_mismatch(self):
        inputs = self._inputs_by_tag(
            is_prefill=True,
            input_lengths=torch.tensor([self._PAGE + 1, 1], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
        )
        inputs[PLE_NGRAM_CTX_TAG].input_lengths = torch.tensor(
            [self._PAGE, 2], dtype=torch.int32
        )

        with self.assertRaisesRegex(RuntimeError, "metadata is inconsistent"):
            self.model._apply_ple(
                1,
                torch.randn(self._PAGE + 2, _HC * _HIDDEN, dtype=torch.bfloat16),
                torch.ones(self._PAGE + 2, dtype=torch.long),
                inputs,
            )

    def test_unsupported_prefix_cp_and_cuda_graph_fail_fast(self):
        for field, value, error in (
            ("prefix_lengths", torch.tensor([1, 0], dtype=torch.int32), "prefix"),
            ("context_parallel_info", object(), "context parallelism"),
            ("is_cuda_graph", True, "CUDA Graph"),
            ("is_s_padded", True, "padded"),
            ("cache_store_inputs", object(), "PD"),
        ):
            with self.subTest(field=field):
                inputs = self._inputs_by_tag(
                    is_prefill=True,
                    input_lengths=torch.tensor([1, 1], dtype=torch.int32),
                    prefix_lengths=torch.zeros(2, dtype=torch.int32),
                    sequence_lengths=torch.empty(0, dtype=torch.int32),
                )
                setattr(inputs[PLE_STATE_TAG], field, value)
                if field == "prefix_lengths":
                    setattr(inputs[PLE_NGRAM_CTX_TAG], field, value.clone())
                with self.assertRaisesRegex(RuntimeError, error):
                    self.model._apply_ple(
                        1,
                        torch.randn(2, _HC * _HIDDEN, dtype=torch.bfloat16),
                        torch.ones(2, dtype=torch.long),
                        inputs,
                    )


if __name__ == "__main__":
    main()
