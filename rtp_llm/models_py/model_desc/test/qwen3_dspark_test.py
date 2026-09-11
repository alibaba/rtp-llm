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
        model._aux_graph_hidden = capture._aux_graph_hidden
        model._aux_graph_capacity = capture._aux_graph_capacity
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
        result = model(
            SimpleNamespace(
                input_ids=ids,
                attention_inputs={"full": SimpleNamespace(is_cuda_graph=False)},
            ),
            object(),
        )
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

    def test_capture_storage_follows_actual_eager_rows_and_graph_bound(self):
        model = _Capture()
        self.assertIsNone(model._aux_hidden)
        self.assertIsNone(model._aux_graph_hidden)
        for rows in (20, 3):
            hidden = torch.ones(rows, 4, dtype=torch.bfloat16)
            model.begin_aux_hidden_capture(hidden)
            self.assertEqual(model._aux_hidden.shape, (rows, 8))
            self.assertEqual(model._aux_hidden.dtype, hidden.dtype)
            self.assertIsNone(model._aux_graph_hidden)

        model.begin_aux_hidden_capture(torch.ones(4, 4), is_cuda_graph=True)
        address = model._aux_graph_hidden.data_ptr()
        self.assertEqual(model._aux_graph_hidden.shape, (8, 8))
        model.begin_aux_hidden_capture(torch.ones(20, 4))
        model.begin_aux_hidden_capture(torch.ones(8, 4), is_cuda_graph=True)
        self.assertEqual(model._aux_graph_hidden.data_ptr(), address)
        with self.assertRaisesRegex(ValueError, "graph token bound"):
            model.begin_aux_hidden_capture(torch.ones(9, 4), is_cuda_graph=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph replay")
    def test_graph_replay_after_eager_prompt_exports_current_features(self):
        model = _Capture("cuda")
        pool = torch.cuda.graph_pool_handle()
        graphs = {}
        stream = torch.cuda.Stream()
        for rows in (4, 8):
            hidden = torch.ones(rows, 4, device="cuda")
            residual = torch.full_like(hidden, 2)
            graph = torch.cuda.CUDAGraph()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                model.begin_aux_hidden_capture(hidden, is_cuda_graph=True)
                model.capture_aux_hidden(0, hidden)
                model.capture_aux_hidden(2, hidden, residual)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph, pool=pool):
                model.begin_aux_hidden_capture(hidden, is_cuda_graph=True)
                model.capture_aux_hidden(0, hidden)
                model.capture_aux_hidden(2, hidden, residual)
            graphs[rows] = (graph, hidden, residual)
        address = model._aux_graph_hidden.data_ptr()
        for rows, prompt_rows in ((4, 20), (8, 3), (4, 12)):
            prompt = torch.full((prompt_rows, 4), -9.0, device="cuda")
            model.begin_aux_hidden_capture(prompt)
            model.capture_aux_hidden(0, prompt)
            model.capture_aux_hidden(2, prompt)
            graph, hidden, residual = graphs[rows]
            hidden.fill_(rows)
            graph.replay()
            output = model.get_mtp_target_hidden_states_for_graph(rows - 1)
            expected = torch.tensor(
                [[float(rows)] * 4 + [float(rows + 2)] * 4], device="cuda"
            ).expand(rows - 1, -1)
            torch.testing.assert_close(output, expected)
            self.assertEqual(output.data_ptr(), address)
            torch.testing.assert_close(
                model.get_mtp_target_hidden_states(prompt_rows),
                torch.full((prompt_rows, 8), -9.0, device="cuda"),
            )

    def test_commit_fused_projection_matches_separate_layer_kv(self):
        torch.manual_seed(3)
        model, layer_weights = self._draft_fixture(aux_width=12)
        self.assertEqual(model.cuda_graph_input_hidden_size(), 12)
        self.assertIsNone(model.get_mtp_target_hidden_states(2))
        self.assertEqual(model._aux_columns, {})
        caches = [object(), object()]
        model.kv_cache = SimpleNamespace(
            get_layer_cache=lambda i: caches[i],
            get_layer_cache_groups=lambda i: [SimpleNamespace(tag=f"draft{i}")],
        )
        writer = Mock()
        features = torch.randn(6, 12)
        attention = {
            tag: SimpleNamespace(input_lengths=torch.tensor([2, 4]))
            for tag in ("target", "draft1", "draft0")
        }
        inputs = SimpleNamespace(
            input_ids=torch.zeros(6),
            input_hiddens=features.reshape(-1),
            attention_inputs=attention,
        )
        module = "rtp_llm.models_py.model_desc.qwen3_dspark_model"
        with patch(f"{module}.ContextKVWriter", return_value=writer) as factory:
            writers = model.prepare_forward_commit(inputs, is_cuda_graph=True)
        self.assertEqual(list(writers), ["draft0", "draft1"])
        self.assertEqual(
            [call.args[1] for call in factory.call_args_list],
            [attention["draft0"], attention["draft1"]],
        )
        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory"
        ) as factory:
            proposal_impls = model.prepare_fmha_impl(inputs, is_cuda_graph=True)
        self.assertEqual(list(proposal_impls), ["draft0", "draft1"])
        self.assertEqual(
            [call.args[3] for call in factory.get_fmha_impl.call_args_list],
            [attention["draft0"], attention["draft1"]],
        )
        output = model.forward_commit(inputs, writers)
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

        inputs.input_ids = torch.zeros(5)
        with self.assertRaisesRegex(ValueError, "row-aligned"):
            model.forward_commit(inputs, writer)

    def test_propose_reuses_framework_inputs_through_shared_flow(self):
        from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
        from rtp_llm.ops.compute_ops import PyModelOutputs

        model, _ = self._draft_fixture()
        model.kv_cache = object()
        for rows in (8, 0):
            inputs = SimpleNamespace(input_ids=torch.zeros(rows))
            metadata, hidden = object(), torch.randn(rows, 4, dtype=torch.float16)
            model.compute_draft_hidden_states = Mock(side_effect=lambda x: x * 2)
            with patch.object(
                Qwen3Model, "forward", return_value=PyModelOutputs(hidden)
            ) as backbone:
                output = model.forward_propose(inputs, metadata)
            backbone.assert_called_once_with(inputs, metadata)
            self.assertEqual(
                model.compute_draft_hidden_states.call_count, int(rows > 0)
            )
            torch.testing.assert_close(output.hidden_states, hidden * 2)

    def test_fused_projection_keeps_rocm_swizzled_layout(self):
        from rtp_llm.utils.swizzle_utils import swizzle_tensor

        def pack(weight):
            return swizzle_tensor(weight.T.to(torch.bfloat16)).T

        model, layer_kv = self._draft_fixture(hidden_size=64, pack=pack)
        expected = pack(torch.cat(layer_kv).T)
        actual = model.context_kv_projection.weight
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.stride(), expected.stride())

    def _draft_fixture(self, aux_width=8, hidden_size=4, pack=lambda weight: weight):
        from rtp_llm.models_py.model_desc import qwen3_dspark_model as module
        from rtp_llm.utils.model_weight import W

        attention = SimpleNamespace(
            head_num=2, kv_head_num=1, size_per_head=hidden_size // 2, is_causal=False
        )
        config = SimpleNamespace(
            hidden_size=hidden_size,
            gen_num_per_cycle=3,
            dspark_noise_token_id=7,
            dspark_sample_from_anchor=False,
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
            W.dspark_fc_w: torch.randn(aux_width, hidden_size),
            W.dspark_hidden_norm_gamma: torch.ones(hidden_size),
        }
        layers = [
            {
                W.attn_qkv_w: torch.randn(hidden_size, 2 * hidden_size),
                W.k_ln_gamma: torch.ones(hidden_size // 2),
            }
            for _ in range(2)
        ]
        layer_kv = [w[W.attn_qkv_w][:, hidden_size:].T for w in layers]
        for weights in layers:
            weights[W.attn_qkv_w] = pack(weights[W.attn_qkv_w])
        weights = SimpleNamespace(
            global_weights=globals_,
            get_global_weight=globals_.__getitem__,
            weights=layers,
        )
        with patch.object(
            module.Qwen3Model, "__init__", AuxHiddenCaptureMixin.__init__
        ), patch.object(module, "LinearFactory") as factory, patch.object(
            module, "RMSNorm", side_effect=lambda w, **kw: nn.LayerNorm(w.numel())
        ):
            factory.create_linear_from_weights.return_value = nn.Linear(
                aux_width, hidden_size, bias=False
            )
            factory.create_linear.side_effect = lambda w, *args: Mock(
                weight=w, side_effect=lambda x: x @ w
            )
            model = module.Qwen3DSparkModel(config, parallelism, weights, 2)
        return model, layer_kv

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
