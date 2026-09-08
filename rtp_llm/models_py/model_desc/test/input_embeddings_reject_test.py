import types
import unittest

import torch

from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel
from rtp_llm.models_py.model_desc.disaggregate_qwen3 import DisaggregateModelBase


def _inputs(input_embeddings=None):
    return types.SimpleNamespace(
        input_embeddings=input_embeddings,
        input_embeddings_locs=torch.tensor([0]),
    )


class InputEmbeddingsRejectTest(unittest.TestCase):
    def test_bert_rejects_input_embeddings_with_runtime_error(self):
        model = BertModel.__new__(BertModel)
        inputs = _inputs([torch.zeros(1, 1)])

        with self.assertRaisesRegex(
            RuntimeError, "BertModel does not support input_embeddings"
        ):
            model.forward(inputs)

    def test_disaggregate_rejects_input_embeddings_with_runtime_error(self):
        model = DisaggregateModelBase.__new__(DisaggregateModelBase)
        inputs = _inputs([torch.zeros(1, 1)])

        with self.assertRaisesRegex(
            RuntimeError, "DisaggregateModelBase does not support input_embeddings"
        ):
            model._reject_input_embeddings(inputs)

    def test_disaggregate_rejects_input_embeddings_in_input_list(self):
        model = DisaggregateModelBase.__new__(DisaggregateModelBase)
        inputs = [_inputs([]), _inputs([torch.zeros(1, 1)])]

        with self.assertRaisesRegex(
            RuntimeError, "DisaggregateModelBase does not support input_embeddings"
        ):
            model._reject_input_embeddings(inputs)

    def test_disaggregate_allows_inputs_without_input_embeddings(self):
        model = DisaggregateModelBase.__new__(DisaggregateModelBase)

        model._reject_input_embeddings([_inputs(None), _inputs([])])

    def test_deepseek_v4_fixed_entrypoints_reject_input_embeddings(self):
        inputs = _inputs([torch.zeros(1, 1)])
        entrypoints = [
            (DeepSeekV4Model.__new__(DeepSeekV4Model).forward, "DeepSeekV4Model"),
            (
                DeepSeekV4MtpModel.__new__(DeepSeekV4MtpModel).forward,
                "DeepSeekV4MtpModel",
            ),
            (
                DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel).forward_propose,
                "DeepSeekV4DSparkModel",
            ),
            (
                DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel).forward_commit,
                "DeepSeekV4DSparkModel",
            ),
        ]

        for entrypoint, model_name in entrypoints:
            with self.subTest(entrypoint=entrypoint.__name__), self.assertRaisesRegex(
                RuntimeError, rf"{model_name} does not support input_embeddings"
            ):
                entrypoint(inputs)


if __name__ == "__main__":
    unittest.main()
