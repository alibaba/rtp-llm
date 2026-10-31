"""GLM NextN checkpoint, hidden-state and causal multi-query contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.deepseek_v2 import DeepSeekV2, Glm5Mtp, Glm5MtpWeight
from rtp_llm.models_py.model_desc.glm5_mtp import Glm5MtpModel
from rtp_llm.models_py.modules.base.cuda.indexer_op import IndexerOp
from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.utils.dsa_indexing import build_dsa_indexer_groups
from rtp_llm.utils.model_weight import W


class Glm5MtpTest(unittest.TestCase):
    def _config(self, layers=78):
        value = dict(
            intermediate_size=128,
            num_attention_heads=64,
            num_hidden_layers=layers,
            vocab_size=256,
            hidden_size=64,
            q_lora_rank=32,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=2,
            first_k_dense_replace=0,
            moe_intermediate_size=128,
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=4,
            index_topk_freq=4,
            index_skip_topk_offset=3,
            num_nextn_predict_layers=1,
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(value))
            return Glm5Mtp._create_config(directory)

    def test_registered_full_and_extracted_checkpoints(self):
        self.assertIs(ModelFactory.get_model_cls("glm_5_mtp"), Glm5Mtp)
        for main_layers, offset in ((78, 78), (1, 0)):
            config = self._config(main_layers)
            self.assertTrue(config.is_mtp)
            self.assertTrue(config.attn_config.is_sparse)
            self.assertEqual(config.num_layers, 1)
            self.assertEqual(config.mtp_layer_offset, offset)
            self.assertEqual(build_dsa_indexer_groups(config, 1), {0: (0,)})

    def test_full_checkpoint_weight_names_include_real_mtp_layer(self):
        weight = Glm5MtpWeight.__new__(Glm5MtpWeight)
        weight.model_config = self._config()
        weight._num_layers = 1
        weight._hidden_size = 64
        weight._get_hf_layer_weight_info = lambda _: []
        info = weight._get_weight_info()
        names = {
            module.name: [ckpt.name for ckpt in module.weights]
            for module in info.weights
        }
        self.assertEqual(names[W.embedding], ["model.embed_tokens.weight"])
        self.assertEqual(names[W.lm_head], ["lm_head.weight"])
        self.assertEqual(
            names[W.multi_tokens_predict_hnorm], ["model.layers.78.hnorm.weight"]
        )
        self.assertEqual(
            names[W.multi_tokens_predict_eh_proj], ["model.layers.78.eh_proj.weight"]
        )

    def test_offload_selection_covers_verify_queries(self):
        config = self._config()
        config.gen_num_per_cycle = 3
        config.attn_config.tokens_per_block = 256
        config.attn_config.kernel_tokens_per_block = 128
        Glm5Mtp._post_build_model_config(config)
        model = SimpleNamespace(model_config=config, max_generate_batch_size=2)
        DeepSeekV2._configure_sparse_kv_offload(model)
        main = next(
            desc for desc in config.kv_cache_spec_descs[0] if desc.tag == "default"
        )
        self.assertEqual(main.offload.max_selected_rows, 2 * 4 * 2048)

    def test_embedding_order_position_zero_and_draft_recurrent_hidden(self):
        model = Glm5MtpModel.__new__(Glm5MtpModel)
        torch.nn.Module.__init__(model)
        model.embed_tokens = lambda ids: torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        model.pre_fc_norm_embedding = lambda x: x * 2
        model.pre_fc_norm_hidden = lambda x: x * 3
        model.fc = lambda x: x
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            input_hiddens=torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        )
        fmha = {
            "default": SimpleNamespace(
                rope_params=SimpleNamespace(positions_d=torch.tensor([0, 12]))
            )
        }
        actual = model.embedding(inputs, fmha)
        torch.testing.assert_close(
            actual, torch.tensor([[0.0, 0.0, 21.0, 24.0], [8.0, 10.0, 27.0, 30.0]])
        )
        model.norm = lambda h, r: ((h + r) * 10, None)
        hidden, residual = torch.randn(2, 4), torch.randn(2, 4)
        logits_hidden = model._capture_canonical_final(hidden, residual)
        # Match the GLM reference: shared_head.norm output feeds both logits
        # and the next autoregressive draft step.
        torch.testing.assert_close(logits_hidden, (hidden + residual) * 10)

    def test_short_target_verify_and_draft_extend_keep_sparse_attention(self):
        class Sparse:
            support = staticmethod(lambda *args: True)
            support_parallelism_config = staticmethod(lambda *args: True)
            is_sparse = staticmethod(lambda: True)

            def __init__(self, *args, **kwargs):
                pass

        attn = SimpleNamespace(is_sparse=True, indexer_topk=2048)
        weights = SimpleNamespace(weights=[], get_global_weight_or_none=lambda _: None)
        for target, draft in ((True, False), (False, True)):
            inputs = SimpleNamespace(
                is_prefill=True, is_target_verify=target, is_draft_extend=draft
            )
            with patch.object(attn_factory, "PREFILL_MLA_IMPS", []), patch.object(
                attn_factory, "DECODE_MLA_IMPS", [Sparse]
            ):
                self.assertIsInstance(
                    attn_factory.get_mla_impl(attn, weights, inputs), Sparse
                )

    def test_graph_wrapper_initialization_keeps_captured_offload_runtime(self):
        model = Glm5MtpModel.__new__(Glm5MtpModel)
        torch.nn.Module.__init__(model)
        storage = SimpleNamespace(
            resident_planes=[torch.ones(2, 8)],
            generations=torch.zeros(2, dtype=torch.int64),
        )
        plan = SimpleNamespace(storage=storage)
        model.offload_plans = {(0, "default"): plan}
        cache = SimpleNamespace(
            layer_count=1,
            group_tags=["default"],
            get_offload_storage=lambda layer, tag: storage,
        )
        resource = SimpleNamespace(kv_cache=cache)
        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.GenericMoeModel.initialize"
        ) as rebuild:
            self.assertTrue(model.initialize(resource))
            self.assertTrue(model.initialize(resource))
            rebuild.assert_not_called()
        self.assertIs(model.offload_plans[0, "default"], plan)
        cache.get_offload_storage = lambda layer, tag: SimpleNamespace(
            resident_planes=[torch.zeros(2, 8)], generations=storage.generations
        )
        with self.assertRaisesRegex(ValueError, "share their KV storage"):
            model.initialize(resource)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_variable_verify_queries_respect_request_and_causal_prefix(self):
        torch.manual_seed(31)
        device = "cuda"
        pages, page, heads, topk = 84, 64, 4, 2048
        keys = torch.randn(pages, page, 128, device=device).to(torch.float8_e4m3fn)
        scales = torch.rand(pages, page, device=device) + 0.5
        pool = torch.empty(pages, page * 132, dtype=torch.uint8, device=device)
        pool[:, : page * 128] = keys.view(torch.uint8).reshape(pages, -1)
        pool[:, page * 128 :] = scales.view(torch.uint8).reshape(pages, -1)
        table = torch.arange(1, 81, dtype=torch.int32, device=device).reshape(2, 40)
        batch = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.int32, device=device)
        positions = torch.tensor(
            [2100, 2101, 2102, 2103, 2200, 2201], dtype=torch.int32, device=device
        )
        query = torch.randn(6, heads, 128, device=device).to(torch.float8_e4m3fn)
        weights = torch.rand(6, heads, 1, device=device)
        params = SimpleNamespace(
            batch_indice_d=batch,
            positions_d=positions,
            kvlen_d=None,
            expanded_seq_lens=positions + 1,
        )
        inputs = SimpleNamespace(
            is_target_verify=True,
            is_draft_extend=False,
            kv_cache_kernel_block_id_device=table,
            decode_cu_seqlens_device=None,
        )
        cache = SimpleNamespace(seq_size_per_block=page, kv_cache_base=pool)
        op = IndexerOp(heads, 128, topk, 64, blocksize=page)
        selected = op._get_topk_paged(query, weights, cache, params, inputs)
        for row, request in enumerate(batch.tolist()):
            length = int(positions[row]) + 1
            physical = table[request].long()
            k = keys[physical].float().reshape(-1, 128)[:length]
            s = scales[physical].reshape(-1)[:length]
            score = torch.einsum("hd,nd->hn", query[row].float(), k).relu()
            score = (score * weights[row].float()).sum(0) * s
            ids = selected[row].reshape(-1)
            valid = ids[ids >= 0].long()
            count = min(topk, length)
            self.assertEqual(valid.numel(), count)
            self.assertEqual(valid.unique().numel(), count)
            self.assertTrue(bool((valid < length).all()))
            threshold = score.topk(count).values[-1]
            self.assertTrue(bool((score[valid] >= threshold - 1e-3).all()))


if __name__ == "__main__":
    unittest.main()
