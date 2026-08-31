# Stage-aware model construction.
#
# Pins two things:
#   1. GptModelBase's PP stage view (pp_layer_ids / capability flags /
#      pp_rank fallback) — the partition lookup must stay equivalent to
#      LoadConfig.pp_layer_range (weight loading) and PPLayout::layerRangeOf
#      (C++), so loader / construction / cache always agree on stage
#      ownership. Fixtures materialize the partition like the startup
#      decision point does.
#   2. Qwen3Model as the exemplar refactor: each stage builds only its own
#      layers (global ids kept) plus the global components it owns
#      (embedding on first stage, final norm on last stage).

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from torch import nn

from rtp_llm.config.pp_layout import even_split_counts
from rtp_llm.models_py.model_desc.module_base import GptModelBase


def make_base(
    pp_size=1,
    pp_rank=None,
    world_rank=0,
    tp_size=1,
    dp_size=1,
    layer_num=32,
):
    """GptModelBase without running __init__ (needs full model config)."""
    model = object.__new__(GptModelBase)
    cfg = SimpleNamespace(
        pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, world_rank=world_rank
    )
    if pp_rank is not None:
        cfg.pp_rank = pp_rank
    if pp_size > 1:
        # Mirror the startup decision point: pp>1 carries materialized data.
        cfg.pp_stage_layer_counts = even_split_counts(layer_num, pp_size)
    model.parallelism_config = cfg
    model.layer_num = layer_num
    return model


class PPStageViewTest(unittest.TestCase):
    def test_pp_size_1_keeps_today_behavior(self):
        model = make_base(pp_size=1, pp_rank=0, layer_num=36)
        self.assertEqual(model.pp_layer_ids(), list(range(36)))
        # pp_size=1 is both first and last stage.
        self.assertTrue(model.pp_has_embedding)
        self.assertTrue(model.pp_has_lm_head)

    def test_even_split_two_stages(self):
        stage0 = make_base(pp_size=2, pp_rank=0, layer_num=32)
        stage1 = make_base(pp_size=2, pp_rank=1, layer_num=32)
        self.assertEqual(stage0.pp_layer_ids(), list(range(0, 16)))
        self.assertEqual(stage1.pp_layer_ids(), list(range(16, 32)))
        self.assertTrue(stage0.pp_has_embedding)
        self.assertFalse(stage0.pp_has_lm_head)
        self.assertFalse(stage1.pp_has_embedding)
        self.assertTrue(stage1.pp_has_lm_head)

    def test_remainder_goes_to_earlier_stages(self):
        # Mirrors the PPLayout::layerRangeOf example: 65 layers, pp=4 ->
        # 17/16/16/16.
        ranges = [
            make_base(pp_size=4, pp_rank=r, layer_num=65).pp_layer_ids()
            for r in range(4)
        ]
        self.assertEqual([len(ids) for ids in ranges], [17, 16, 16, 16])
        self.assertEqual(ranges[0], list(range(0, 17)))
        self.assertEqual(ranges[1], list(range(17, 33)))
        self.assertEqual(ranges[2], list(range(33, 49)))
        self.assertEqual(ranges[3], list(range(49, 65)))
        # Full coverage, no overlap.
        self.assertEqual(sorted(sum(ranges, [])), list(range(65)))

    def test_pp_rank_read_from_config_field(self):
        model = make_base(pp_size=2, pp_rank=1, world_rank=0)
        # Field wins even when world_rank would derive a different value.
        self.assertEqual(model.pp_rank, 1)

    def test_pp_rank_fallback_derives_pp_outermost(self):
        # Fake config without the pp_rank field: derive from world_rank
        # (pp_rank = world_rank // (dp_size * tp_size)).
        model = make_base(pp_size=2, pp_rank=None, world_rank=5, tp_size=2, dp_size=1)
        self.assertEqual(model.pp_rank, 2)


def make_qwen3(pp_size, pp_rank, num_layers=8):
    """Construct Qwen3Model with heavy building blocks mocked out.

    The mocks record their calls but return real (empty) nn.Modules, since
    nn.ModuleList rejects non-Module children.
    """
    from rtp_llm.models_py.model_desc import qwen3

    config = SimpleNamespace(num_layers=num_layers, vocab_size=100, layernorm_eps=1e-6)
    parallelism_config = SimpleNamespace(
        pp_size=pp_size,
        pp_rank=pp_rank,
        tp_size=1,
        dp_size=1,
        world_rank=pp_rank,
    )
    if pp_size > 1:
        parallelism_config.pp_stage_layer_counts = even_split_counts(
            num_layers, pp_size
        )
    weights = SimpleNamespace(
        weights={idx: f"layer_{idx}" for idx in range(num_layers)},
        get_global_weight=lambda name: f"global_{name}",
    )
    module_stub = lambda *args, **kwargs: nn.Module()  # noqa: E731
    with patch.object(
        qwen3, "Embedding", Mock(side_effect=module_stub)
    ) as mock_emb, patch.object(
        qwen3, "Qwen3DecoderLayer", Mock(side_effect=module_stub)
    ) as mock_layer, patch.object(
        qwen3, "RMSNorm", Mock(side_effect=module_stub)
    ) as mock_norm, patch(
        "rtp_llm.models_py.model_desc.module_base.get_device_type"
    ):
        model = qwen3.Qwen3Model(
            config, parallelism_config, weights, max_generate_batch_size=1
        )
    return model, mock_emb, mock_layer, mock_norm


class Qwen3StageConstructionTest(unittest.TestCase):
    def test_pp_size_1_builds_everything(self):
        model, mock_emb, mock_layer, mock_norm = make_qwen3(pp_size=1, pp_rank=0)
        self.assertIsNotNone(model.embed_tokens)
        self.assertIsNotNone(model.norm)
        self.assertEqual(len(model.layers), 8)
        self.assertEqual(model.pp_layer_ids_list, list(range(8)))
        mock_emb.assert_called_once()
        mock_norm.assert_called_once()
        self.assertEqual(mock_layer.call_count, 8)

    def test_first_stage_builds_embedding_and_front_layers(self):
        model, mock_emb, mock_layer, mock_norm = make_qwen3(pp_size=2, pp_rank=0)
        self.assertIsNotNone(model.embed_tokens)
        self.assertIsNone(model.norm)  # final norm belongs to the last stage
        self.assertEqual(model.pp_layer_ids_list, [0, 1, 2, 3])
        self.assertEqual(len(model.layers), 4)
        mock_emb.assert_called_once()
        mock_norm.assert_not_called()
        # Decoder layers are constructed with GLOBAL layer ids.
        built_ids = [call.args[2] for call in mock_layer.call_args_list]
        self.assertEqual(built_ids, [0, 1, 2, 3])

    def test_last_stage_builds_norm_and_back_layers(self):
        model, mock_emb, mock_layer, mock_norm = make_qwen3(pp_size=2, pp_rank=1)
        self.assertIsNone(model.embed_tokens)  # embedding on first stage only
        self.assertIsNotNone(model.norm)
        self.assertEqual(model.pp_layer_ids_list, [4, 5, 6, 7])
        self.assertEqual(len(model.layers), 4)
        mock_emb.assert_not_called()
        mock_norm.assert_called_once()
        built_ids = [call.args[2] for call in mock_layer.call_args_list]
        self.assertEqual(built_ids, [4, 5, 6, 7])

    def test_middle_stage_builds_neither_global_component(self):
        model, mock_emb, mock_layer, mock_norm = make_qwen3(pp_size=3, pp_rank=1)
        self.assertIsNone(model.embed_tokens)
        self.assertIsNone(model.norm)
        # 8 layers / 3 stages -> 3/3/2; middle stage owns [3, 6).
        self.assertEqual(model.pp_layer_ids_list, [3, 4, 5])
        mock_emb.assert_not_called()
        mock_norm.assert_not_called()


class Qwen3NextPPBoundaryTest(unittest.TestCase):
    """Stage-boundary tensors cross stages as a named map.

    qwen3_next uses the fused add-norm pattern, so the stream at a stage
    boundary is split into (branch output, accumulated residual):
      entry: non-first stages resume `residual` from inputs.pp_intermediates
             (first stage / pp_size=1 keeps zeros);
      exit:  non-last stages emit {"hidden_states", "residual"} (dropping
             the residual would corrupt the downstream stream); last stage
             norms and returns plain hidden states.
    """

    def _make_model(self, with_norm):
        import torch

        from rtp_llm.models_py.model_desc import qwen3_next as qn

        model = object.__new__(qn.Qwen3NextModel)
        model.embed_tokens = None  # exercise the non-first-stage entry
        model.kv_cache = None
        model.parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
        )
        layer = Mock()
        layer.layer_type = qn.HybridAttentionType.LINEAR
        # Echo layer: keeps (hidden, residual) flowing unchanged.
        layer.side_effect = lambda h, r, *a, **k: (h, r)
        model.layers = [layer]
        model.norm = (
            Mock(side_effect=lambda h, r: (h + r, h + r)) if with_norm else None
        )
        # Inherited from GptModelBase; needs real attention_inputs plumbing.
        model.prepare_fmha_impl = Mock(return_value=None)
        return model, qn

    def _make_inputs(self, torch_mod, intermediates=None):
        # Duck-typed PyModelInputs: forward() only accesses attributes, and the
        # compiled bindings module is not in this target's runfiles.
        # attention_inputs is read by block_map inside the layer loop; a
        # non-empty tag mapping passes validation and with kv_cache=None the
        # per-layer selection yields an empty list, which the echo layer
        # ignores.
        return SimpleNamespace(
            input_hiddens=torch_mod.zeros(2, 4),
            pp_intermediates=intermediates or {},
            attention_inputs={"full": SimpleNamespace()},
        )

    def _run_forward(self, model, qn, inputs):
        import torch

        attn = SimpleNamespace(
            is_target_verify=False, is_prefill=False, cu_seqlens_device=None
        )
        with patch.object(
            qn, "get_primary_attention_inputs", return_value=attn
        ), patch.object(qn, "prepare_causal_conv1d_metadata", return_value=None):
            return model.forward(inputs)

    def test_non_last_stage_emits_hidden_and_residual(self):
        import torch

        model, qn = self._make_model(with_norm=False)
        outputs = self._run_forward(model, qn, self._make_inputs(torch))
        # hidden = zeros echoed through; residual = zeros init (no upstream).
        self.assertEqual(set(outputs.pp_intermediates), {"hidden_states", "residual"})
        torch.testing.assert_close(
            outputs.pp_intermediates["hidden_states"], torch.zeros(2, 4)
        )
        torch.testing.assert_close(
            outputs.pp_intermediates["residual"], torch.zeros(2, 4)
        )

    def test_entry_resumes_upstream_hidden_and_residual(self):
        import torch

        model, qn = self._make_model(with_norm=False)
        upstream_hidden = torch.full((2, 4), 2.0)
        upstream_residual = torch.ones(2, 4)
        inputs = self._make_inputs(
            torch, {"hidden_states": upstream_hidden, "residual": upstream_residual}
        )
        outputs = self._run_forward(model, qn, inputs)
        # Echo layer forwards both resumed boundary tensors to the stage exit;
        # hidden must come from the map, NOT the (zeros) input_hiddens
        # fallback channel.
        torch.testing.assert_close(
            outputs.pp_intermediates["hidden_states"], upstream_hidden
        )
        torch.testing.assert_close(
            outputs.pp_intermediates["residual"], upstream_residual
        )

    def test_entry_falls_back_to_input_hiddens_without_map(self):
        import torch

        model, qn = self._make_model(with_norm=False)
        # No pp_intermediates (legacy/MTP-style handoff): input_hiddens is used.
        outputs = self._run_forward(model, qn, self._make_inputs(torch))
        torch.testing.assert_close(
            outputs.pp_intermediates["hidden_states"], torch.zeros(2, 4)
        )

    def test_last_stage_norms_and_emits_no_intermediates(self):
        import torch

        model, qn = self._make_model(with_norm=True)
        outputs = self._run_forward(model, qn, self._make_inputs(torch))
        self.assertFalse(outputs.pp_intermediates)
        # norm mock: hidden + residual = zeros + zeros
        torch.testing.assert_close(outputs.hidden_states, torch.zeros(2, 4))


class Qwen3PPBoundaryTest(unittest.TestCase):
    """Naive add-norm models carry a SINGLE boundary tensor.

    qwen3.py adds the residual inside each layer, so the stream at a stage
    boundary is already combined: the map holds only {"hidden_states"}.
    """

    def _make_model(self, with_norm):
        import torch

        from rtp_llm.models_py.model_desc import qwen3 as q3

        model = object.__new__(q3.Qwen3Model)
        model.embed_tokens = None  # non-first-stage entry
        model.kv_cache = None
        layer = Mock(side_effect=lambda h, *a, **k: h)  # echo layer
        model.layers = [layer]
        model.norm = Mock(side_effect=lambda h: h + 1.0) if with_norm else None
        model.prepare_fmha_impl = Mock(return_value=None)
        return model, q3

    def _make_inputs(self, torch_mod, intermediates=None):
        return SimpleNamespace(
            input_hiddens=torch_mod.zeros(2, 4),
            pp_intermediates=intermediates or {},
        )

    def test_non_last_stage_emits_single_hidden_key(self):
        import torch

        model, q3 = self._make_model(with_norm=False)
        upstream_hidden = torch.full((2, 4), 3.0)
        inputs = self._make_inputs(torch, {"hidden_states": upstream_hidden})
        outputs = model.forward(inputs)
        self.assertEqual(set(outputs.pp_intermediates), {"hidden_states"})
        torch.testing.assert_close(
            outputs.pp_intermediates["hidden_states"], upstream_hidden
        )

    def test_last_stage_norms_and_emits_no_intermediates(self):
        import torch

        model, q3 = self._make_model(with_norm=True)
        outputs = model.forward(inputs := self._make_inputs(torch))
        self.assertFalse(outputs.pp_intermediates)
        # norm mock adds 1.0 to the zeros input_hiddens fallback
        torch.testing.assert_close(outputs.hidden_states, torch.ones(2, 4))


if __name__ == "__main__":
    unittest.main()
