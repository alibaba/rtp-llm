import json
import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models.deepseek_v4 import DeepSeekV4DSpark, DeepSeekV4DSparkWeight
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
    DSparkMarkovHead,
    DeepSeekV4DSparkModel,
    DeepSeekV4DSparkParams,
)
from rtp_llm.models_py.modules.dsv4.decode.forward import forward_layers
from rtp_llm.utils.model_weight import W


CKPT = "/mnt/nas1/hf/DeepSeek-V4-Flash-0731"


class _AddLayer(nn.Module):
    def __init__(self, layer_id: int, increment: float):
        super().__init__()
        self.layer_id = layer_id
        self.increment = increment

    def forward_decode(self, hidden, metadata, input_ids, kv_cache=None):
        return hidden + self.increment


class DeepSeekV4DSparkTest(unittest.TestCase):
    def test_config_uses_target_layer_count_not_num_nextn(self):
        cfg = {
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [40, 41, 42],
            "dspark_markov_rank": 256,
            "num_nextn_predict_layers": 1,
        }
        params = DeepSeekV4DSparkParams.from_ckpt_config(cfg)
        self.assertEqual(params.target_layer_ids, [40, 41, 42])
        self.assertEqual(params.speculative_tokens, 4)
        self.assertEqual(params.block_width, 4)
        self.assertTrue(params.sample_from_anchor)

        real = DeepSeekV4DSpark._create_config(CKPT)
        self.assertEqual(real.num_layers, 3)
        self.assertEqual(real.attn_config.layer_compress_ratios, [0, 0, 0])

    def test_combine_and_markov_match_dense_oracle(self):
        torch.manual_seed(7)
        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        nn.Module.__init__(model)
        model.dspark_params = DeepSeekV4DSparkParams(
            target_layer_ids=[1, 2, 3],
            mask_token_id=31,
            speculative_tokens=4,
            block_size=5,
            markov_rank=3,
        )
        model._v4_args = SimpleNamespace(dim=2)
        model.main_proj = nn.Linear(6, 2, bias=False)
        model.main_norm = nn.Identity()

        aux = torch.randn(5, 3, 2)
        expected_combined = F.linear(aux.flatten(1), model.main_proj.weight)
        torch.testing.assert_close(model.combine_hidden_states(aux), expected_combined)

        vocab, rank, batch, width = 11, 3, 2, 4
        w1 = torch.randn(vocab, rank)
        w2 = torch.randn(vocab, rank)
        model.markov_head = DSparkMarkovHead(w1, w2)
        base = torch.randn(batch, width, vocab)
        anchor = torch.tensor([2, 7])
        tokens, corrected = model.markov_correct(base, anchor)

        oracle = base.float().clone()
        previous = anchor.clone()
        expected_tokens = []
        for step in range(width):
            oracle[:, step] += F.linear(w1[previous], w2).float()
            previous = oracle[:, step].argmax(-1)
            expected_tokens.append(previous)
        expected_tokens = torch.stack(expected_tokens, dim=1)
        torch.testing.assert_close(corrected, oracle)
        self.assertTrue(torch.equal(tokens, expected_tokens))
        # FP32 input must not be mutated by the correction loop.
        self.assertFalse(torch.equal(base, corrected))

    def test_target_aux_capture_mean_pools_mhc_lanes(self):
        v4 = SimpleNamespace()
        v4.capture_aux_hidden_layer_ids = (0, 2)
        v4._last_aux_hidden_states = None
        v4.embed = nn.Embedding(16, 3)
        with torch.no_grad():
            v4.embed.weight.copy_(torch.arange(48).view(16, 3).float())
        v4.hc_mult = 2
        v4.layers = nn.ModuleList(
            [_AddLayer(0, 1.0), _AddLayer(1, 2.0), _AddLayer(2, 3.0)]
        )
        v4._mtp_hidden_buffer = None
        v4._hc_head_reduce = lambda hidden: hidden.mean(dim=-2)
        v4.norm = nn.Identity()
        meta = SimpleNamespace(
            batch_size=1,
            q_len_per_req=2,
            is_cuda_graph=False,
        )
        token_ids = torch.tensor([1, 2])
        embedded = v4.embed(token_ids).view(1, 2, 3)
        forward_layers(v4, None, token_ids, meta)
        expected_l0 = embedded + 1.0
        expected_l2 = embedded + 1.0 + 2.0 + 3.0
        expected = torch.stack([expected_l0, expected_l2], dim=2).reshape(2, 2, 3)
        torch.testing.assert_close(v4._last_aux_hidden_states, expected)

    def test_weight_descriptor_and_checkpoint_inventory(self):
        descriptor = DeepSeekV4DSparkWeight.__new__(DeepSeekV4DSparkWeight)
        descriptor._num_layers = 3
        descriptor._compress_ratios = [0, 0, 0]
        descriptor._num_hash_layers = 0
        descriptor._hidden_size = 4096
        descriptor._size_per_head = 512
        descriptor._head_num = 64
        descriptor._head_num_kv = 1
        descriptor.expert_num_ = 256
        descriptor._moe_align_size = 64
        descriptor.enable_fp32_lm_head = False
        info = descriptor._get_weight_info()
        self.assertEqual(len(info.layer_weights), 3)
        globals_by_name = {weight.name: weight for weight in info.weights}
        for name in (
            W.v4_dspark_main_proj_w,
            W.v4_dspark_main_norm,
            W.v4_dspark_markov_w1,
            W.v4_dspark_markov_w2,
        ):
            self.assertIn(name, globals_by_name)

        quantized = info.to_quant_weight_info(
            Fp8BlockWiseQuantConfig(is_quanted=True)
        )
        qglobals = {weight.name: weight for weight in quantized.weights}
        main_proj = qglobals[W.v4_dspark_main_proj_w]
        self.assertEqual(main_proj.__class__.__name__, "V4PerBlockFp8Weight")
        kernel_sources = [w.tensor_name(None) for w in main_proj.kernel.weights]
        scale_sources = [w.tensor_name(None) for w in main_proj.scale.weights]
        self.assertIn("mtp.0.main_proj.weight", kernel_sources)
        self.assertIn("mtp.0.main_proj.scale", scale_sources)

        with open(os.path.join(CKPT, "model.safetensors.index.json")) as reader:
            checkpoint_keys = set(json.load(reader)["weight_map"])
        required = {
            "mtp.0.main_proj.weight",
            "mtp.0.main_proj.scale",
            "mtp.0.main_norm.weight",
            "mtp.2.norm.weight",
            "mtp.2.hc_head_fn",
            "mtp.2.hc_head_base",
            "mtp.2.hc_head_scale",
            "mtp.2.markov_head.markov_w1.weight",
            "mtp.2.markov_head.markov_w2.weight",
        }
        self.assertFalse(required - checkpoint_keys)
        for layer_id in range(3):
            self.assertIn(f"mtp.{layer_id}.attn.wkv.weight", checkpoint_keys)
            self.assertIn(f"mtp.{layer_id}.ffn.gate.weight", checkpoint_keys)


if __name__ == "__main__":
    unittest.main()
