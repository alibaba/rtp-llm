import types
import unittest

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel


def _inputs(input_embeddings):
    return types.SimpleNamespace(input_embeddings=input_embeddings)


class Dsv4InputEmbeddingsRejectTest(unittest.TestCase):
    def test_dsv4_models_do_not_advertise_support(self):
        for model in (DeepSeekV4Model, DeepSeekV4MtpModel, DeepSeekV4DSparkModel):
            self.assertFalse(model.supports_input_embeddings)

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
