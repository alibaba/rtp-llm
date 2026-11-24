"""Real CPU PLE checkpoint plus existing hash/gather/projection runtime."""

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
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from rtp_llm.models_py.new_models.qwen4_exp.ple_layer import Qwen4PLELayer
from rtp_llm.models_py.registry import register_model


@register_model("qwen4_ple_runtime_fixture")
class _PLEFixture(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.ple = Qwen4PLELayer(
            hidden_size=8,
            hc_mult=4,
            ngram_size=3,
            heads_per_ngram=2,
            shard_count=8,
            embedding_dim=4,
            conv_kernel_size=3,
            eos_token_id=2,
            dtype=load_config.compute_dtype,
            source_fp8=config.get("source_fp8", False),
            fp8_storage=config.get("fp8_storage", False),
        )

    def checkpoint_weight_name_filter(self):
        return self.ple.checkpoint_name_filter("ple.")


class Qwen4PLELayerTest(unittest.TestCase):
    def test_real_checkpoint_and_original_runtime_agree(self):
        torch.manual_seed(441)
        with tempfile.TemporaryDirectory() as path, mock.patch.dict(
            os.environ, {"LOAD_METHOD": ""}
        ):
            shards = [(torch.randn(4, 4) * 0.03).to(torch.bfloat16) for _ in range(8)]
            multipliers = torch.tensor(
                [2**44 + 123, 2**43 + 7, 2**42 + 13], dtype=torch.int64
            )
            offsets = torch.tensor([0, 3, 8, 15], dtype=torch.int64)
            sizes = torch.tensor([3, 5, 7, 11], dtype=torch.int64)
            weights = {
                "ple.ple_embedding.layer_multipliers": multipliers,
                "ple.ple_embedding.ngram_heads_offsets": offsets,
                "ple.ple_embedding.ngram_heads_vocab_sizes": sizes,
            }
            for index, tensor in enumerate(shards):
                weights[f"ple.ple_embedding.ngram_embedding.shard_{index}.weight"] = (
                    tensor
                )
            shapes = {
                "key_proj": (32, 16),
                "value_proj": (8, 16),
                "conv1d": (32, 1, 3),
                "norm_key": (32,),
                "norm_query": (32,),
                "norm_conv": (32,),
            }
            for name, shape in shapes.items():
                weights[f"ple.{name}.weight"] = (torch.randn(shape) * 0.03).to(
                    torch.bfloat16
                )
            save_file(weights, os.path.join(path, "model.safetensors"))
            model = NewModelLoader(
                {"model_type": "qwen4_ple_runtime_fixture"},
                NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.bfloat16,
                    load_method=NewLoaderLoadMethod.SCRATCH,
                ),
                model_path=path,
            ).load()
            reference_embedding = Qwen4ExpNGramEmbedding(
                shards, sizes, offsets, multipliers, ngram_size=3, eos_token_id=2
            )
            reference = Qwen4ExpPLELayer(
                reference_embedding,
                *[
                    weights[f"ple.{name}.weight"]
                    for name in (
                        "key_proj",
                        "value_proj",
                        "conv1d",
                        "norm_key",
                        "norm_query",
                        "norm_conv",
                    )
                ],
                hc_mult=4,
                hidden_size=8,
                conv_kernel_size=3,
                norm_eps=1e-6,
            )
            # Includes EOS boundaries and a 45-bit multiplier; neither metadata
            # nor the host table passes through a floating-point all-expert copy.
            history = torch.tensor([[0, 1, 2, 3, 4, 5], [2, 1, 4, 2, 7, 8]])
            hidden = torch.randn(2, 4, 32, dtype=torch.bfloat16)
            torch.testing.assert_close(
                model.ple(hidden, history), reference(hidden, history), rtol=0, atol=0
            )
            table = model.ple.ple_embedding.ngram_embedding
            for owner, borrowed in zip(
                table.runtime_shards(), model.ple.runtime.ple_embedding.shards
            ):
                self.assertEqual(owner.data_ptr(), borrowed.data_ptr())
                self.assertEqual(owner.device.type, "cpu")
            self.assertEqual(
                model.ple.ple_embedding.layer_multipliers.dtype, torch.int64
            )
            self.assertTrue(
                all(b._rtp_requires_restart_for_update for b in model.ple.buffers())
            )
            model.ple.close()
            model.ple.close()


if __name__ == "__main__":
    unittest.main()
