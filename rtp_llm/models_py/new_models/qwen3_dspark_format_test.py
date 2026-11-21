"""Checkpoint-format contracts. Deliberately independent of GPU/serving tests."""

import json
import os
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from rtp_llm.models_py.speculative.qwen3_dspark_checkpoint import (
    absolute_d2t,
    is_inference_weight,
    target_layer_ids,
)


class CheckpointContractTest(unittest.TestCase):
    def test_target_layer_encoding_is_explicit(self):
        self.assertEqual(
            target_layer_ids({"target_layer_ids": [1, 29, 56]}), [1, 29, 56]
        )
        self.assertEqual(
            target_layer_ids({"aux_hidden_state_layer_ids": [1, 3]}), [1, 3]
        )
        self.assertEqual(
            target_layer_ids(
                {
                    "speculators_model_type": "dspark",
                    "aux_hidden_state_layer_ids": [1, 3],
                }
            ),
            [0, 2],
        )
        for bad in (
            {"aux_hidden_state_layer_ids": [2, 1]},
            {"aux_hidden_state_layer_ids": [1], "target_layer_ids": [0]},
            {"speculators_model_type": "dspark", "aux_hidden_state_layer_ids": [0]},
        ):
            with self.assertRaises(ValueError):
                target_layer_ids(bad)

    def test_reduced_vocabulary_lifetime(self):
        for name, expected in (("d2t", [0, 2, 4, 6]), ("model.d2t", [0, 1, 2, 3])):
            src = torch.arange(4)
            value = absolute_d2t(name, src, 4, 8)
            self.assertNotEqual(value.data_ptr(), src.data_ptr())
            src.fill_(99)
            self.assertEqual(value.tolist(), expected)
        for value in (torch.ones(2), torch.ones((4, 1)), torch.tensor([0, 1, 2, 12])):
            with self.assertRaises(ValueError):
                absolute_d2t("d2t", value, 4, 8)

    def test_no_broad_weight_exclusions(self):
        self.assertFalse(is_inference_weight("confidence_head.proj.weight"))
        self.assertFalse(is_inference_weight("confidence_head.proj.bias"))
        for name in (
            "confidence_head.future_extension",
            "fc.weight",
            "markov_head.markov_w1.weight",
            "model.norm.weight",
        ):
            self.assertTrue(is_inference_weight(name))

    @unittest.skipUnless(
        os.environ.get("QWEN3_DSPARK_RELEASE"),
        "set QWEN3_DSPARK_RELEASE to check full release metadata",
    )
    def test_real_release_header(self):
        root = Path(os.environ["QWEN3_DSPARK_RELEASE"])
        cfg = json.loads((root / "config.json").read_text())
        ids = target_layer_ids(cfg)
        self.assertEqual(ids, [1, 29, 56])
        with safe_open(
            str(root / "model.safetensors"), framework="pt", device="cpu"
        ) as ckpt:
            self.assertEqual(
                ckpt.get_slice("fc.weight").get_shape(),
                [cfg["hidden_size"], len(ids) * cfg["hidden_size"]],
            )
            for which in ("w1", "w2"):
                self.assertEqual(
                    ckpt.get_slice(f"markov_head.markov_{which}.weight").get_shape(),
                    [cfg["vocab_size"], cfg["markov_rank"]],
                )
            self.assertEqual(
                {key for key in ckpt.keys() if not is_inference_weight(key)},
                {"confidence_head.proj.weight", "confidence_head.proj.bias"},
            )
            self.assertEqual(
                {int(k.split(".")[1]) for k in ckpt.keys() if k.startswith("layers.")},
                set(range(cfg["num_hidden_layers"])),
            )


if __name__ == "__main__":
    unittest.main()
