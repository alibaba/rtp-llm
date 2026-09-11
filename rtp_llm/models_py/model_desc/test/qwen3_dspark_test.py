import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.speculative.aux_hidden_capture import AuxHiddenCaptureMixin


class _Base(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _Capture(AuxHiddenCaptureMixin, _Base):
    def __init__(self, device="cpu"):
        config = SimpleNamespace(
            capture_aux_hidden_layer_ids=[0, 2],
            num_layers=3,
            hidden_size=4,
            max_seq_len=16,
            moe_prefill_max_tokens_per_rank=24,
            gen_num_per_cycle=3,
        )
        parallelism = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
        )
        weights = SimpleNamespace(
            get_global_weight=lambda _: torch.empty(1, device=device)
        )
        super().__init__(config, parallelism, weights, 2)


class _Layer(nn.Module):
    def forward(self, hidden, fmha_impl, kv_cache=None):
        return hidden + 1


class Qwen3DSparkTest(unittest.TestCase):
    def _target_fixture(self, cls):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        capture = _Capture()
        model._aux_columns = capture._aux_columns
        model._aux_hidden = capture._aux_hidden
        model.kv_cache = None
        return model

    def test_qwen_target_forward_exports_decoder_outputs_before_final_norm(self):
        from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model

        model = self._target_fixture(Qwen3Model)
        model.embed_tokens = nn.Embedding.from_pretrained(
            torch.arange(20).float().view(5, 4)
        )
        model.layers = nn.ModuleList([_Layer() for _ in range(3)])
        model.layer_num = 3
        model.norm = nn.LayerNorm(4)
        ids = torch.tensor([2, 0, 3])
        result = model(SimpleNamespace(input_ids=ids), object())
        embeddings = model.embed_tokens(ids)
        torch.testing.assert_close(result.hidden_states, model.norm(embeddings + 3))
        torch.testing.assert_close(
            model.get_mtp_target_hidden_states(3),
            torch.cat((embeddings + 1, embeddings + 3), dim=1),
        )

    def test_qwen_next_target_forward_exports_residual_features(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextModel

        model = self._target_fixture(Qwen3NextModel)
        model.parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
        )
        hidden = torch.ones(20, 4)
        model.word_embedding = Mock(return_value=hidden)
        model.layers = [
            Mock(layer_type=None, return_value=(hidden * i, hidden * (10 * i)))
            for i in (1, 2, 3)
        ]
        model.norm = Mock(
            side_effect=lambda hidden, residual: (hidden + residual, residual)
        )
        inputs = SimpleNamespace(
            attention_inputs={
                "full": SimpleNamespace(
                    is_prefill=True, is_target_verify=True, is_cuda_graph=False
                )
            }
        )
        model(inputs, object())
        torch.testing.assert_close(
            model.get_mtp_target_hidden_states(20),
            torch.tensor([[11.0] * 4 + [33.0] * 4]).expand(20, -1),
        )

        self.assertEqual(model.get_mtp_target_hidden_states(7).shape, (7, 8))
        for rows in (-1, 25):
            with self.assertRaises(ValueError):
                model.get_mtp_target_hidden_states(rows)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_graph_replay_after_eager_prompt_exports_current_features(self):
        model = _Capture("cuda")
        hidden = torch.ones(8, 4, device="cuda")
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            model.capture_aux_hidden(0, hidden)
            model.capture_aux_hidden(2, hidden + 2)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            model.capture_aux_hidden(0, hidden)
            model.capture_aux_hidden(2, hidden + 2)
        address = model._aux_hidden.data_ptr()
        model.capture_aux_hidden(0, torch.full((20, 4), -9.0, device="cuda"))
        hidden.fill_(7)
        graph.replay()
        output = model.get_mtp_target_hidden_states(3)
        torch.testing.assert_close(
            output, torch.tensor([[7.0] * 4 + [9.0] * 4], device="cuda").expand(3, -1)
        )
        self.assertEqual(output.data_ptr(), address)

    def test_commit_fused_projection_matches_separate_layer_kv(self):
        torch.manual_seed(3)
        model, layer_weights = self._draft_fixture(aux_width=12)
        self.assertEqual(model.cuda_graph_input_hidden_size(), 12)
        self.assertIsNone(model.get_mtp_target_hidden_states(2))
        self.assertEqual(model._aux_columns, {})
        caches = [object(), object()]
        model.kv_cache = SimpleNamespace(get_layer_cache=lambda i: caches[i])
        writer = Mock()
        features = torch.randn(6, 12)
        inputs = SimpleNamespace(input_ids=torch.zeros(6), input_hiddens=features)
        output = model.forward_commit(inputs, writer)
        torch.testing.assert_close(output.hidden_states, model.fc(features))
        self.assertEqual(model.context_kv_projection.call_count, 1)
        self.assertEqual(writer.forward.call_count, 2)
        context = model.hidden_norm(model.fc(features))
        for i, call in enumerate(writer.forward.call_args_list):
            key, value = (context @ layer_weights[i].T).chunk(2, dim=-1)
            expected = torch.cat(
                (torch.zeros(6, 4), model.context_k_norms[i](key), value), dim=1
            )
            torch.testing.assert_close(call.args[0], expected)
            self.assertIs(call.args[1], caches[i])

        writer.reset_mock()
        model.kv_cache.get_layer_cache_groups = lambda i: [SimpleNamespace(tag="draft")]
        unrelated = Mock()
        model.forward_commit(
            inputs,
            {"target": unrelated, "draft": writer},
        )
        self.assertEqual(writer.forward.call_count, 2)
        unrelated.forward.assert_not_called()
        inputs.input_ids = torch.zeros(5)
        with self.assertRaisesRegex(ValueError, "row-aligned"):
            model.forward_commit(inputs, writer)

    def test_commit_prepare_routes_only_model_cache_groups(self):
        from rtp_llm.models_py.model_desc import qwen3_dspark_model as module

        model, _ = self._draft_fixture()
        model.kv_cache = SimpleNamespace(
            get_layer_cache_groups=lambda i: [SimpleNamespace(tag=f"draft{i}")]
        )
        attention = {tag: object() for tag in ("target", "draft1", "draft0")}
        with patch.object(module, "ContextKVWriter") as factory:
            factory.side_effect = lambda *args: Mock()
            writers = model.prepare_forward_commit(
                SimpleNamespace(attention_inputs=attention), is_cuda_graph=True
            )
        self.assertEqual(list(writers), ["draft0", "draft1"])
        self.assertEqual(
            [call.args[1] for call in factory.call_args_list],
            [attention["draft0"], attention["draft1"]],
        )

    def _draft_fixture(self, aux_width=8):
        from rtp_llm.models_py.model_desc import qwen3_dspark_model as module
        from rtp_llm.utils.model_weight import W

        def init_backbone(model, config, *args, **kwargs):
            kwargs.pop("quant_config", None)
            AuxHiddenCaptureMixin.__init__(model, config, *args, **kwargs)

        attention = SimpleNamespace(
            head_num=2, kv_head_num=1, size_per_head=2, is_causal=False
        )
        config = SimpleNamespace(
            hidden_size=4,
            vocab_size=8,
            num_layers=2,
            capture_aux_hidden_layer_ids=[0, 2],
            layernorm_eps=1e-6,
            getAttentionConfigs=lambda _: attention,
        )
        parallelism = SimpleNamespace(
            get_attn_tp_size=lambda: 1,
            prefill_cp_config=SimpleNamespace(
                is_enabled=lambda: False, kv_cache_sharded=False
            ),
        )
        globals_ = {
            W.dspark_fc_w: torch.randn(aux_width, 4),
            W.dspark_hidden_norm_gamma: torch.ones(4),
        }
        layers = [
            {W.attn_qkv_w: torch.randn(4, 8), W.k_ln_gamma: torch.ones(2)}
            for _ in range(2)
        ]
        weights = SimpleNamespace(
            global_weights=globals_,
            get_global_weight=globals_.__getitem__,
            weights=layers,
        )
        with patch.object(module.Qwen3Model, "__init__", init_backbone), patch.object(
            module, "LinearFactory"
        ) as factory, patch.object(
            module, "RMSNorm", side_effect=lambda w, **kw: nn.LayerNorm(w.numel())
        ):
            factory.create_linear_from_weights.return_value = nn.Linear(
                aux_width, 4, bias=False
            )
            factory.create_linear.side_effect = lambda w, *args: Mock(
                side_effect=lambda x: x @ w
            )
            model = module.Qwen3DSparkModel(config, parallelism, weights, 2)
        return model, [w[W.attn_qkv_w][:, 4:].T for w in layers]

    def test_runtime_wiring_keeps_input_and_output_vocab_separate(self):
        from rtp_llm.model_factory import ModelFactory

        draft = SimpleNamespace(
            dspark_noise_token_id=7,
            input_vocab_size=8,
            vocab_size=3,
            dspark_target_layer_ids=[0, 2],
            dspark_markov_rank=2,
            dspark_sample_from_anchor=False,
        )
        target = SimpleNamespace(num_layers=3)
        sp = SimpleNamespace(gen_num_per_cycle=3)
        ModelFactory._setup_dspark_configs(sp, target, draft)
        self.assertEqual(sp.sp_dspark_mask_token_id, 7)
        self.assertFalse(sp.sp_dspark_sample_from_anchor)
        self.assertEqual(target.capture_aux_hidden_layer_ids, [0, 2])
        draft.dspark_target_layer_ids = [2, 0]
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            ModelFactory._setup_dspark_configs(sp, target, draft)

    def test_speculators_d2t_offsets_are_normalized_once(self):
        from rtp_llm.models.qwen_3_dspark import dspark_offset_d2t_to_absolute

        torch.testing.assert_close(
            dspark_offset_d2t_to_absolute([torch.tensor([2, 4, 5])]),
            torch.tensor([2, 5, 7]),
        )


if __name__ == "__main__":
    unittest.main()
