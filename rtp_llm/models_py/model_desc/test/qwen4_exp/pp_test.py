import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc import qwen4_exp
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.ops import HybridAttentionType
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs
from rtp_llm.utils.model_weight import W


class _EchoLayer(nn.Module):
    def __init__(self, global_layer_id: int):
        super().__init__()
        self.global_layer_id = global_layer_id
        self.layer_type = HybridAttentionType.LINEAR
        self.self_attn = nn.Identity()

    def forward(self, hidden_states, fmha_impl, **kwargs):
        return hidden_states + self.global_layer_id + 1


class _Mixer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.down_projection = nn.Identity()
        self.up_projection = nn.Identity()

    def forward(self, hidden_states):
        return hidden_states[:, : self.hidden_size], None, None


def _parallelism(counts, rank):
    return SimpleNamespace(
        pp_size=len(counts),
        pp_rank=rank,
        pp_stage_layer_counts=list(counts),
        role_type="PDFUSION",
    )


def _bare_stage(counts, rank, hidden_size=2, hc_mult=2):
    model = object.__new__(qwen4_exp.Qwen4ExpModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_layers=sum(counts),
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        is_mtp=False,
    )
    model.hc_mult = hc_mult
    model.layer_num = sum(counts)
    model.parallelism_config = _parallelism(counts, rank)
    model.layers = nn.ModuleList(
        [_EchoLayer(layer_id) for layer_id in model.layer_ids()]
    )
    model.ple_layers = nn.ModuleDict()
    model.kv_cache = None
    model._capture_mtp_target_hidden = False
    model._mtp_target_hidden_states = None
    model.hyper_connection_mixer = (
        _Mixer(hidden_size) if model.owns_model_output else None
    )
    model._try_fp8_hyper_embedding = lambda inputs: None
    model.prepare_fmha_impl = lambda inputs: None
    model._build_attn_meta = lambda *args, **kwargs: None
    return model


def _inputs(token_ids, intermediates=None):
    inputs = PyModelInputs()
    inputs.input_ids = token_ids
    inputs.pp_intermediates = intermediates or {}
    inputs.attention_inputs = PyAttentionInputs()
    inputs.attention_inputs.is_cuda_graph = False
    inputs.attention_inputs.is_prefill = True
    inputs.attention_inputs.is_target_verify = False
    return inputs


class Qwen4ExpPPBoundaryTest(unittest.TestCase):
    def test_warmup_boundary_is_one_materialized_wide_tensor(self):
        model = _bare_stage([2, 2], 1, hidden_size=3, hc_mult=4)
        tensors = model.make_empty_intermediate_tensors(torch.zeros(2, 3))
        self.assertEqual(set(tensors), {"hidden_states"})
        self.assertEqual(tuple(tensors["hidden_states"].shape), (2, 12))

    def test_partitioned_forward_matches_one_stage(self):
        counts = [2, 2, 2]
        token_ids = torch.tensor([1, 3], dtype=torch.int32)

        whole = _bare_stage([sum(counts)], 0)
        whole.word_embedding = lambda inputs: inputs.input_ids.float().unsqueeze(
            1
        ).repeat(1, 2)
        expected = whole._forward_with_ple_prefetch(_inputs(token_ids))

        intermediates = None
        for rank in range(len(counts)):
            stage = _bare_stage(counts, rank)
            if rank == 0:
                stage.word_embedding = (
                    lambda inputs: inputs.input_ids.float().unsqueeze(1).repeat(1, 2)
                )
            else:
                stage.word_embedding = MagicMock(
                    side_effect=AssertionError("non-first stage ran embedding")
                )
            output = stage._forward_with_ple_prefetch(
                _inputs(token_ids, intermediates)
            )
            if rank < len(counts) - 1:
                self.assertEqual(set(output.pp_intermediates), {"hidden_states"})
                self.assertEqual(
                    tuple(output.pp_intermediates["hidden_states"].shape), (2, 4)
                )
                intermediates = output.pp_intermediates
            else:
                self.assertFalse(output.pp_intermediates)

        torch.testing.assert_close(output.hidden_states, expected.hidden_states)

    def test_multimodal_embedding_is_injected_before_the_pp_boundary(self):
        tokens = torch.tensor([1, 3, 5], dtype=torch.int32)
        features = torch.tensor([[42.0, 43.0]])
        first = _bare_stage([1, 1], 0)
        first.embed_tokens = lambda ids, *_: ids.float().unsqueeze(1).repeat(1, 2)
        first.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        inputs = _inputs(tokens)
        mm_inputs = inputs.multimodal_inputs
        mm_inputs.multimodal_features = [features]
        mm_inputs.mm_features_locs = torch.tensor([1], dtype=torch.int32)
        inputs.multimodal_inputs = mm_inputs

        first_output = first._forward_with_ple_prefetch(inputs)
        expected = torch.tensor([[1.0, 1.0], [42.0, 43.0], [5.0, 5.0]])
        torch.testing.assert_close(
            first_output.pp_intermediates["hidden_states"], expected.repeat(1, 2) + 1
        )

        last = _bare_stage([1, 1], 1)
        result = last._forward_with_ple_prefetch(
            _inputs(tokens, first_output.pp_intermediates)
        )
        torch.testing.assert_close(result.hidden_states, expected + 3)


class Qwen4ExpPPConstructionTest(unittest.TestCase):
    def _output_model(self, counts, rank):
        model = object.__new__(qwen4_exp.Qwen4ExpModel)
        nn.Module.__init__(model)
        model.layer_num = sum(counts)
        model.config = SimpleNamespace(num_layers=sum(counts), is_mtp=False)
        model.parallelism_config = _parallelism(counts, rank)
        return model

    def test_hc_mixer_is_built_only_by_the_output_owner(self):
        weights = SimpleNamespace(
            global_weights={},
            get_global_weight=MagicMock(return_value=torch.ones(1)),
        )
        config = SimpleNamespace(hc_mult=2, layernorm_eps=1e-6)
        with patch.object(
            qwen4_exp, "GatedResidual", side_effect=lambda *a, **k: _Mixer(1)
        ), patch.object(qwen4_exp, "mark_projection_for_restart"):
            first = self._output_model([2, 2], 0)
            first._build_output_head(config, weights)
            self.assertIsNone(first.hyper_connection_mixer)
            weights.get_global_weight.assert_not_called()

            last = self._output_model([2, 2], 1)
            last._build_output_head(config, weights)
            self.assertIsInstance(last.hyper_connection_mixer, _Mixer)
            self.assertEqual(weights.get_global_weight.call_count, 3)


class Qwen4ExpPPPLEMappingTest(unittest.TestCase):
    @staticmethod
    def _model(counts, rank):
        model = object.__new__(qwen4_exp.Qwen4ExpModel)
        nn.Module.__init__(model)
        model.layer_num = sum(counts)
        model.config = SimpleNamespace(
            num_layers=sum(counts),
            is_mtp=False,
            hidden_size=2,
            hc_mult=2,
            layernorm_eps=1e-6,
            _qwen4_ple_layer_ids=[3],
            _qwen4_split_ngram_parts=1,
            _qwen4_ngram_size=3,
            _qwen4_ple_eos_token_id=0,
            qwen4_ple_prefetch_mb=1,
        )
        model.parallelism_config = SimpleNamespace(
            **vars(_parallelism(counts, rank)),
            get_attn_tp_size=lambda: 1,
            get_attn_tp_rank=lambda: 0,
        )
        prefix = W.qwen4_ple_ngram_shards + "."
        layer_weights = {
            prefix + "0": torch.ones(2, 2),
            W.qwen4_ple_ngram_vocab_sizes: torch.ones(1, dtype=torch.int32),
            W.qwen4_ple_ngram_offsets: torch.zeros(1, dtype=torch.int32),
            W.qwen4_ple_multipliers: torch.ones(1, dtype=torch.int32),
            W.qwen4_ple_key_proj_w: torch.ones(1),
            W.qwen4_ple_value_proj_w: torch.ones(1),
            W.qwen4_ple_conv_w: torch.ones(1, 1, 4),
            W.qwen4_ple_norm_key_gamma: torch.ones(1),
            W.qwen4_ple_norm_query_gamma: torch.ones(1),
            W.qwen4_ple_norm_conv_gamma: torch.ones(1),
        }
        model.weight = SimpleNamespace(
            weights=[{}, {}, layer_weights, {}, {}]
        )
        model.ple_layers = nn.ModuleDict()
        return model

    def test_global_ple_layer_maps_to_the_owner_local_layer(self):
        owner = self._model([2, 3], 1)
        shard = owner.weight.weights[2][W.qwen4_ple_ngram_shards + ".0"]
        embedding = SimpleNamespace(shards=[shard])
        ple_layer = nn.Module()
        ple_layer.key_projection = nn.Identity()
        ple_layer.value_projection = nn.Identity()
        with (
            patch.object(qwen4_exp, "Qwen4ExpNGramEmbedding", return_value=embedding),
            patch.object(qwen4_exp, "Qwen4ExpPLELayer", return_value=ple_layer),
        ):
            owner._build_ple_layers()

        self.assertEqual(set(owner.ple_layers), {"0"})

    def test_non_owner_does_not_require_the_ple_weights(self):
        non_owner = self._model([2, 3], 0)
        non_owner.weight.weights[2] = {}
        non_owner._build_ple_layers()
        self.assertFalse(non_owner.ple_layers)


if __name__ == "__main__":
    unittest.main()
