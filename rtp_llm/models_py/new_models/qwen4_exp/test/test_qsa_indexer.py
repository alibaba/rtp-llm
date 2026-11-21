"""Actual checkpoint-to-QSA CPU consumer, not sparse FMHA/GPU coverage."""

import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer
from rtp_llm.models_py.new_models.qwen4_exp.qsa_indexer import Qwen4QSAIndexer
from rtp_llm.models_py.registry import register_model


@register_model("qwen4_qsa_owner_fixture")
class _QSAFixture(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.indexer = Qwen4QSAIndexer(
            16,
            n_heads=3,
            head_dim=4,
            token_budget=4,
            compress_ratio=2,
            dtype=load_config.compute_dtype,
            fp8_storage=config.get("fp8_storage", False),
        )


class Qwen4QSAIndexerTest(unittest.TestCase):
    def test_real_checkpoint_project_pool_select_matches_existing_runtime(self):
        torch.manual_seed(13)
        weights = {
            "indexer.index_qk_proj.weight": (torch.randn(16, 16) * 0.02).to(
                torch.bfloat16
            ),
            "indexer.q_layernorm.weight": (torch.randn(4) * 0.02).to(torch.bfloat16),
            "indexer.k_layernorm.weight": (torch.randn(4) * 0.02).to(torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as path, mock.patch.dict(
            os.environ, {"LOAD_METHOD": ""}
        ):
            save_file(weights, os.path.join(path, "model.safetensors"))
            model = NewModelLoader(
                {"model_type": "qwen4_qsa_owner_fixture"},
                NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.bfloat16,
                    load_method=NewLoaderLoadMethod.SCRATCH,
                ),
                model_path=path,
            ).load()
            reference = Qwen4ExpQSAIndexer(
                *weights.values(),
                n_heads=3,
                kv_heads=1,
                head_dim=4,
                token_budget=4,
                compress_ratio=2,
                norm_eps=1e-6,
            )
            hidden = torch.randn(1, 9, 16, dtype=torch.bfloat16)
            angles = torch.randn(1, 9, 4)
            cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
            actual = model.indexer(hidden, cos, sin)
            expected = reference.forward_causal(hidden, cos, sin)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.shape, (1, 9, 5))
            self.assertIs(
                model.indexer.runtime.q_norm_gamma, model.indexer.q_layernorm.weight
            )
            torch.testing.assert_close(
                model.indexer.q_layernorm.weight,
                weights["indexer.q_layernorm.weight"],
                rtol=0,
                atol=0,
            )
            # Incremental query reuses raw (not prematurely normalized) keys.
            _, keys = reference.project(hidden[:, :-1])
            torch.testing.assert_close(
                model.indexer(hidden[:, -1:], cos, sin, keys),
                reference.forward_causal(hidden[:, -1:], cos, sin, keys),
                rtol=0,
                atol=0,
            )

    def test_invalid_geometry_fails_before_allocation(self):
        for kwargs in (
            dict(n_heads=0),
            dict(kv_heads=2),
            dict(token_budget=3),
            dict(compress_ratio=0),
        ):
            args = dict(n_heads=3, head_dim=4, token_budget=4, compress_ratio=2)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                Qwen4QSAIndexer(16, **args)


if __name__ == "__main__":
    unittest.main()
