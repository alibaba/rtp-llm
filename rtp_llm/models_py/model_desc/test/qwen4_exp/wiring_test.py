from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

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
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.ple_cache_views import PLECacheViews
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _EPS,
    _EXPERTS,
    _HC,
    _HIDDEN,
    _Attention,
    _bare_model,
    _hc_weights,
    _hybrid_types,
    _RoutedBackend,
    _select_topk,
    _spec_config,
)
from rtp_llm.models_py.modules.hybrid.gated_residual import (
    GatedResidual,
    inject_into_residual,
)
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


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

        self.assertIsInstance(layer.attn_hyper_connection, GatedResidual)
        self.assertIsInstance(layer.mlp_hyper_connection, GatedResidual)
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

        attn_unit = GatedResidual(*self.attn_hc, hc_mult=_HC, norm_eps=_EPS)
        mlp_unit = GatedResidual(*self.mlp_hc, hc_mult=_HC, norm_eps=_EPS)
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
    def test_private_hook_receives_actual_rank_and_phase(self):
        hook = Mock()
        stub = SimpleNamespace(
            parallelism_config=SimpleNamespace(get_attn_tp_rank=lambda: 3),
            _speculative_test_hook=hook,
        )
        call = qwen4_exp.Qwen4ExpModel._maybe_inject_speculative_fault
        call(stub, "prepare")
        hook.assert_called_once_with("prepare", 3)
        hook.side_effect = RuntimeError("commit test failure")
        with self.assertRaisesRegex(RuntimeError, "commit test failure"):
            call(stub, "commit")
        del stub._speculative_test_hook
        call(stub, "commit")


class Qwen4ExpMetadataTest(TestCase):
    def test_ple_region_alias_avoids_equality_sync_but_checks_distinct_values(self):
        values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        left = SimpleNamespace(input_lengths=values)
        alias = SimpleNamespace(input_lengths=values.view_as(values))
        different = SimpleNamespace(input_lengths=values + 1)
        with patch.object(torch, "equal", wraps=torch.equal) as equal:
            self.assertTrue(PLECacheViews._same_metadata(left, alias, "input_lengths"))
            equal.assert_not_called()
            self.assertFalse(
                PLECacheViews._same_metadata(left, different, "input_lengths")
            )
            equal.assert_called_once()


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
            self.global_weights = self._globals
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


class Qwen4ExpSharedHCStorageTest(TestCase):
    def test_real_residual_builder_reuses_owner_and_blocks_updates(self):
        import weakref

        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.model_loader.weight_manager import WeightManager

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        config = ModelConfig()
        config.hc_mult = 4
        config.layernorm_eps = 1e-6
        weights = ModelWeights(1, "cuda", torch.bfloat16)
        for norm, down, up, inject in (
            (
                W.qwen4_hc_attn_norm,
                W.qwen4_hc_attn_mix_down,
                W.qwen4_hc_attn_mix_up,
                W.qwen4_hc_attn_inject,
            ),
            (
                W.qwen4_hc_mlp_norm,
                W.qwen4_hc_mlp_mix_down,
                W.qwen4_hc_mlp_mix_up,
                W.qwen4_hc_mlp_inject,
            ),
        ):
            for key, shape in (
                (norm, (32,)),
                (down, (7, 32)),
                (up, (32, 7)),
                (inject, (4, 32)),
            ):
                weights.set_layer_weight(
                    0, key, torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                )
        old_refs = [
            weakref.ref(weights.weights[0][key])
            for key in (
                W.qwen4_hc_attn_mix_down,
                W.qwen4_hc_attn_inject,
                W.qwen4_hc_mlp_mix_down,
                W.qwen4_hc_mlp_inject,
            )
        ]
        layer = qwen4_exp.Qwen4ExpDecoderLayer.__new__(qwen4_exp.Qwen4ExpDecoderLayer)
        nn.Module.__init__(layer)
        layer._build_residual_modules(config, weights.weights[0])
        self.assertTrue(all(ref() is None for ref in old_refs))
        for unit, down, inject in (
            (
                layer.attn_hyper_connection,
                W.qwen4_hc_attn_mix_down,
                W.qwen4_hc_attn_inject,
            ),
            (
                layer.mlp_hyper_connection,
                W.qwen4_hc_mlp_mix_down,
                W.qwen4_hc_mlp_inject,
            ),
        ):
            self.assertEqual(
                weights.weights[0][down].untyped_storage().data_ptr(),
                unit.merged_down_inject.untyped_storage().data_ptr(),
            )
            self.assertIs(weights.weights[0][inject], unit.inject)
            before = unit.merged_down_inject.clone()
            with self.assertRaisesRegex(PermissionError, "restart"):
                weights.update_layer_weight(0, down, torch.zeros_like(unit.mix_down))
            torch.testing.assert_close(unit.merged_down_inject, before, rtol=0, atol=0)
        manager = WeightManager.__new__(WeightManager)
        manager._weights = weights
        # Invalid IPC data would fail decoding if the shared-view guard were late.
        with self.assertRaisesRegex(PermissionError, "restart"):
            manager.update(
                dict(
                    name="model.layers.0." + W.qwen4_hc_attn_inject,
                    method="shm",
                    desc="invalid-ipc-description",
                )
            )

    def test_unsupported_cpu_keeps_original_storage(self):
        fixture = Qwen4ExpDecoderLayerTest()
        fixture.setUp()
        layer = qwen4_exp.Qwen4ExpDecoderLayer.__new__(qwen4_exp.Qwen4ExpDecoderLayer)
        nn.Module.__init__(layer)
        original = fixture.weights[W.qwen4_hc_attn_mix_down]
        layer._build_residual_modules(fixture.config, fixture.weights)
        self.assertIs(fixture.weights[W.qwen4_hc_attn_mix_down], original)
        self.assertFalse(getattr(original, "_rtp_requires_restart_for_update", False))


if __name__ == "__main__":
    main()
