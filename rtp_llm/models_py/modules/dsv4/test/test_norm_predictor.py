import os
import tempfile
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.norm_predictor import (
    Dsv4NormTopP,
    NormPredictor,
    select_norm_top_p,
    validate_predictor_config,
)


class NormPredictorTopPTest(unittest.TestCase):
    def test_exclusive_prefix_keeps_native_slots_without_renormalizing(self):
        weights = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0, 0.0]])
        ids = torch.tensor([[4, 8, 12, 16, 20, 24]])
        norms = torch.zeros((1, 256))
        norms[0, ids[0]] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        kept_weights, kept_ids, kept_count = select_norm_top_p(
            weights, ids, norms, top_p=0.71
        )

        self.assertEqual(kept_count.tolist(), [3])
        self.assertTrue(torch.equal(kept_ids[0, :3], ids[0, :3]))
        self.assertEqual(kept_ids[0, 3:].tolist(), [256, 256, 256])
        self.assertTrue(torch.equal(kept_weights[0, :3], weights[0, :3]))
        self.assertEqual(kept_weights[0, 3:].tolist(), [0.0, 0.0, 0.0])

    def test_sorted_quality_is_scattered_back_to_native_slots(self):
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0]])
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        norms = torch.zeros((1, 256))
        norms[0, ids[0]] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        kept_weights, kept_ids, kept_count = select_norm_top_p(
            weights, ids, norms, top_p=0.55
        )

        self.assertEqual(kept_count.tolist(), [2])
        self.assertEqual(kept_ids[0].tolist(), [256, 256, 3, 4, 256, 256])
        self.assertTrue(
            torch.allclose(
                kept_weights[0],
                torch.tensor([0.0, 0.0, 0.3, 0.4, 0.0, 0.0]),
            )
        )

    def test_predictor_checkpoint_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "norm_predictor_layer_3.pt")
            torch.save(
                {
                    "fc1.weight": torch.ones((1024, 4)),
                    "fc1.bias": torch.zeros(1024),
                    "fc2.weight": torch.ones((256, 1024)),
                    "fc2.bias": torch.zeros(256),
                },
                path,
            )
            predictor = NormPredictor(directory, layer_id=3, hidden_size=4)
            output = predictor(torch.ones((2, 4)))
            self.assertEqual(tuple(output.shape), (2, 256))
            self.assertTrue(torch.isfinite(output).all())

    def test_predictor_config_validation(self):
        config = {
            "format": "dsv4-positive-norm-predictor-v3",
            "hidden_size": 4096,
            "bottleneck": 1024,
            "num_experts": 256,
            "layers": list(range(3, 43)),
            "output_transform": "softplus",
        }
        validate_predictor_config(config, hidden_size=4096, n_routed_experts=256)
        config["num_experts"] = 128
        with self.assertRaisesRegex(ValueError, "num_experts"):
            validate_predictor_config(
                config, hidden_size=4096, n_routed_experts=256
            )

    def test_hash_layer_is_not_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "RTP_DSV4_NORM_PREDICTOR_DIR": directory,
                    "RTP_DSV4_NORM_TOP_P": "0.8",
                },
                clear=False,
            ):
                module = Dsv4NormTopP(
                    layer_id=2,
                    dim=4096,
                    n_routed_experts=256,
                    n_hash_layers=3,
                )
        self.assertFalse(module.enabled)

    def test_top_p_one_is_an_exact_runtime_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "RTP_DSV4_NORM_PREDICTOR_DIR": directory,
                    "RTP_DSV4_NORM_TOP_P": "1.0",
                },
                clear=False,
            ):
                module = Dsv4NormTopP(
                    layer_id=3,
                    dim=4096,
                    n_routed_experts=256,
                    n_hash_layers=3,
                )

        weights = torch.rand(2, 6)
        ids = torch.arange(12).reshape(2, 6)
        output_weights, output_ids = module(torch.empty(2, 4096), weights, ids)
        self.assertFalse(module.enabled)
        self.assertIsNone(module.predictor)
        self.assertIs(output_weights, weights)
        self.assertIs(output_ids, ids)


if __name__ == "__main__":
    unittest.main()
