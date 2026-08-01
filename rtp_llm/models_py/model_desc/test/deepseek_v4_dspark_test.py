import json
import os
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.models.deepseek_v4 import DeepSeekV4DSpark, DeepSeekV4DSparkWeight
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
    DSparkMarkovHead,
    DeepSeekV4DSparkModel,
    DeepSeekV4DSparkParams,
)
from rtp_llm.models_py.modules.dsv4.decode.forward import forward_layers
from rtp_llm.utils.model_weight import W
from rtp_llm.ops import TaskType


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
        self.assertEqual(params.speculative_tokens, 5)
        self.assertEqual(params.block_width, 5)
        self.assertTrue(params.sample_from_anchor)

        real = DeepSeekV4DSpark._create_config(CKPT)
        self.assertEqual(real.num_layers, 3)
        self.assertEqual(real.attn_config.layer_compress_ratios, [0, 0, 0])

    def test_config_rejects_non_greedy_proposal(self):
        cfg = {
            "dspark_block_size": 5,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [40, 41, 42],
            "dspark_markov_rank": 256,
            "dspark_proposal_type": "gumbel",
        }
        with self.assertRaisesRegex(ValueError, "supports only.*greedy"):
            DeepSeekV4DSparkParams.from_ckpt_config(cfg)

    def test_model_factory_preserves_frozen_anchor_layout(self):
        params = DeepSeekV4DSparkParams(
            target_layer_ids=[40, 41, 42],
            mask_token_id=128799,
            speculative_tokens=5,
            block_size=5,
            markov_rank=256,
        )
        sp_config = SimpleNamespace(
            sp_dspark_propose_num=3,
            gen_num_per_cycle=1,
            sp_dspark_mask_token_id=-1,
            sp_dspark_sample_from_anchor=False,
        )
        target_config = SimpleNamespace(
            num_layers=43, capture_aux_hidden_layer_ids=[]
        )
        propose_config = SimpleNamespace(dspark_config=params)

        ModelFactory._setup_dspark_configs(
            sp_config, target_config, propose_config
        )

        # The checkpoint config is frozen: setup replaces it instead of
        # mutating the instance retained by an earlier loader stage.
        self.assertEqual(params.speculative_tokens, 5)
        self.assertEqual(propose_config.dspark_config.speculative_tokens, 3)
        self.assertEqual(propose_config.dspark_config.block_width, 3)
        self.assertEqual(sp_config.gen_num_per_cycle, 3)
        self.assertEqual(sp_config.sp_dspark_mask_token_id, 128799)
        self.assertTrue(sp_config.sp_dspark_sample_from_anchor)
        self.assertEqual(target_config.capture_aux_hidden_layer_ids, [40, 41, 42])

    def test_swa_only_draft_skips_sparse_indexer_jit_prewarm(self):
        swa_layers = [
            SimpleNamespace(attn=SimpleNamespace(indexer=None)) for _ in range(3)
        ]
        self.assertIsNone(
            DeepSeekV4DSparkModel._first_indexer_for_jit_prewarm(swa_layers)
        )

        first_indexer = SimpleNamespace(n_heads=64)
        target_layers = [
            SimpleNamespace(attn=SimpleNamespace(indexer=None)),
            SimpleNamespace(attn=SimpleNamespace(indexer=first_indexer)),
        ]
        self.assertIs(
            DeepSeekV4DSparkModel._first_indexer_for_jit_prewarm(target_layers),
            first_indexer,
        )

    def test_loader_global_alias_is_non_owning_and_skips_descriptor(self):
        owner_tensor = torch.arange(12).reshape(3, 4)
        loader = ModelLoader.__new__(ModelLoader)
        loader._global_weight_aliases = {W.embedding: owner_tensor}
        loader._task_type = TaskType.LANGUAGE_MODEL
        loader._load_config = SimpleNamespace(num_layers=1, compute_dtype=torch.float32)

        self.assertTrue(loader._maybe_skip_weight(SimpleNamespace(name=W.embedding)))
        self.assertFalse(loader._maybe_skip_weight(SimpleNamespace(name=W.lm_head)))
        aliased = loader._create_model_weights("cpu")
        self.assertIs(aliased.get_global_weight(W.embedding), owner_tensor)

    def test_prefill_q_workspace_uses_rank_local_attention_heads(self):
        layers = [
            SimpleNamespace(attn=SimpleNamespace(n_heads=32)) for _ in range(3)
        ]
        self.assertEqual(
            DeepSeekV4DSparkModel._resolve_prefill_q_dim(layers, head_dim=512),
            32 * 512,
        )

        layers[-1].attn.n_heads = 64
        with self.assertRaisesRegex(RuntimeError, "rank-local Q-head width"):
            DeepSeekV4DSparkModel._resolve_prefill_q_dim(layers, head_dim=512)

    def test_combine_and_markov_match_dense_oracle(self):
        torch.manual_seed(7)
        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        nn.Module.__init__(model)
        model.dspark_params = DeepSeekV4DSparkParams(
            target_layer_ids=[1, 2, 3],
            mask_token_id=31,
            speculative_tokens=5,
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

    def test_draft_tail_anchor_accepts_native_and_verify_carriers(self):
        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        nn.Module.__init__(model)
        model.dspark_params = DeepSeekV4DSparkParams(
            target_layer_ids=[1, 2, 3],
            mask_token_id=31,
            speculative_tokens=5,
            block_size=5,
            markov_rank=3,
        )
        head_hidden = torch.zeros(10, 4)
        native = torch.tensor([11, 0, 0, 0, 0, 22, 0, 0, 0, 0])
        verify = torch.tensor([11, 1, 2, 3, 4, 5, 22, 6, 7, 8, 9, 10])
        self.assertTrue(
            torch.equal(
                model._anchor_ids_from_block_inputs(native, head_hidden),
                torch.tensor([11, 22]),
            )
        )
        self.assertTrue(
            torch.equal(
                model._anchor_ids_from_block_inputs(verify, head_hidden),
                torch.tensor([11, 22]),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "B\\*k or B\\*\\(k\\+1\\)"):
            model._anchor_ids_from_block_inputs(torch.zeros(11), head_hidden)

    def test_draft_tail_skips_softmax_when_probs_not_requested(self):
        model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
        nn.Module.__init__(model)
        model.compute_base_logits = lambda hidden: torch.tensor(
            [[[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]]]
        )
        model._anchor_ids_from_block_inputs = lambda input_ids, hidden: torch.tensor([0])
        model.markov_correct = lambda logits, anchors: (logits.argmax(-1), logits)

        outputs = SimpleNamespace(
            hidden_states=torch.zeros(2, 4), draft_tokens=None, draft_probs=None
        )
        inputs = SimpleNamespace(input_ids=torch.tensor([0, 1]), need_draft_probs=False)
        result = model.draft_tail(outputs, inputs)
        self.assertTrue(torch.equal(result.draft_tokens, torch.tensor([[1, 0]])))
        self.assertIsNone(result.draft_probs)

        outputs.draft_probs = None
        inputs.need_draft_probs = True
        result = model.draft_tail(outputs, inputs)
        torch.testing.assert_close(
            result.draft_probs,
            torch.softmax(torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]]]), -1),
        )

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
        embedding = globals_by_name[W.embedding]
        embedding_weight = torch.arange(24).view(6, 4)
        torch.testing.assert_close(
            embedding._get_split_func()(embedding_weight, tp=2, tp_rank=1),
            embedding_weight,
        )
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
