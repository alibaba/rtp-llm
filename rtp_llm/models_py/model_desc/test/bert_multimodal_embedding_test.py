from unittest import TestCase, main

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs
from rtp_llm.utils.model_weight import W

_FMHA_NOT_USED = object()


class BertMultimodalEmbeddingTest(TestCase):
    """Covers multimodal injection placement and values without decoder layers."""

    def _build_model(self) -> BertModel:
        config = ModelConfig()
        config.num_layers = 0
        config.hidden_size = 4
        config.vocab_size = 8

        weights = ModelWeights(0, "cuda", torch.float16)
        weights.set_global_weight(
            W.embedding,
            torch.arange(32, dtype=torch.float16, device="cuda").reshape(8, 4),
        )
        weights.set_global_weight(
            W.pre_decoder_ln_gamma,
            torch.ones(4, dtype=torch.float16, device="cuda"),
        )
        weights.set_global_weight(
            W.pre_decoder_ln_beta,
            torch.zeros(4, dtype=torch.float16, device="cuda"),
        )
        model = BertModel(config, ParallelismConfig(), weights, 1)
        self.assertEqual(len(model.layers), 0)
        return model

    def _build_inputs(self) -> PyModelInputs:
        inputs = PyModelInputs()
        inputs.input_ids = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
        inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
            [0, 1, 2], dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.position_encoding = torch.zeros(
            (3, 4), dtype=torch.float16, device="cuda"
        )
        inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.zeros(
            3, dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.token_type_embedding = torch.zeros(
            (1, 4), dtype=torch.float16, device="cuda"
        )
        return inputs

    def test_forward_splices_post_layernorm_features_and_preserves_text(self):
        model = self._build_model()
        baseline_inputs = self._build_inputs()
        baseline = model.forward(
            baseline_inputs, fmha_impl=_FMHA_NOT_USED
        ).hidden_states

        multimodal_inputs = self._build_inputs()
        first_feature = torch.tensor(
            [[9.0, 8.0, 7.0, 6.0]], dtype=torch.float16, device="cuda"
        )
        last_feature = torch.tensor(
            [[6.0, 7.0, 8.0, 9.0]], dtype=torch.float16, device="cuda"
        )
        multimodal_inputs.multimodal_inputs.multimodal_features = [
            first_feature,
            last_feature,
        ]
        multimodal_inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [0, 2], dtype=torch.int32
        )
        output = model.forward(
            multimodal_inputs, fmha_impl=_FMHA_NOT_USED
        ).hidden_states

        torch.testing.assert_close(output[0:1], first_feature)
        torch.testing.assert_close(output[2:3], last_feature)
        torch.testing.assert_close(output[1:2], baseline[1:2])

    def test_forward_moves_cpu_features(self):
        model = self._build_model()
        cpu_feature = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float16)
        inputs = self._build_inputs()
        inputs.multimodal_inputs.multimodal_features = [cpu_feature]
        inputs.multimodal_inputs.mm_features_locs = torch.tensor([1], dtype=torch.int32)

        output = model.forward(inputs, fmha_impl=_FMHA_NOT_USED).hidden_states
        torch.testing.assert_close(output[1:2], cpu_feature.cuda())

    def test_forward_rejects_dtype_mismatch(self):
        model = self._build_model()
        cpu_feature = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
        mismatched_inputs = self._build_inputs()
        mismatched_inputs.multimodal_inputs.multimodal_features = [cpu_feature]
        mismatched_inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [1], dtype=torch.int32
        )
        with self.assertRaisesRegex(TypeError, "dtype mismatch"):
            model.forward(mismatched_inputs, fmha_impl=_FMHA_NOT_USED)


if __name__ == "__main__":
    main()
