import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.disaggregate_qwen3 import (
    BatchSplitInfo,
    Qwen3AttnModel,
    Qwen3GemmModel,
)
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeModel
from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.multimodal_generic import MultimodalGenericModel
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextModel
from rtp_llm.models_py.model_desc.qwen3vl import Qwen3VLModel
from rtp_llm.models_py.model_desc.qwen3vl_moe import Qwen3VLMoeModel


class _CaptureModel(GptModelBase):
    def __init__(self, layer_ids: list[int]) -> None:
        config = SimpleNamespace(
            num_layers=3,
            vocab_size=8,
            hidden_state_capture_layer_ids=layer_ids,
        )
        super().__init__(config, None, None, max_generate_batch_size=1)
        self.layer_hook_calls: list[int] = []
        self.final_hook_calls = 0
        self._init_capture_context(
            self._canonical_layer,
            self._canonical_final,
        )

    def _canonical_layer(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        self.layer_hook_calls.append(int(hidden_states[0, 0].item()))
        return hidden_states if residual is None else hidden_states + residual

    def _canonical_final(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        self.final_hook_calls += 1
        return hidden_states if residual is None else hidden_states + residual


class _Embedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = input_ids.float()
        return torch.stack((values, values * 10), dim=-1)


class _QwenLayer(nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, hidden_states, fmha_impl, kv_cache=None):
        return hidden_states + self.increment


class _MoeLayer(nn.Module):
    def __init__(self, hidden_increment: float, residual_increment: float) -> None:
        super().__init__()
        self.hidden_increment = hidden_increment
        self.residual_increment = residual_increment

    def forward(self, hidden_states, residual, fmha_impl, kv_cache=None):
        return DecodeLayerOutput(
            hidden_states + self.hidden_increment,
            residual + self.residual_increment,
        )


class _Qwen3NextLayer(_MoeLayer):
    layer_type = None

    def forward(
        self,
        hidden_states,
        residual,
        fmha_impl,
        kv_cache=None,
        attention_inputs=None,
        attn_meta=None,
    ):
        del attention_inputs, attn_meta
        output = super().forward(hidden_states, residual, fmha_impl, kv_cache)
        return output.hidden_states, output.residual


class _KimiLayer(_MoeLayer):
    layer_type = None

    def forward(
        self,
        hidden_states,
        residual,
        fmha_impl,
        kv_cache=None,
        attention_inputs=None,
        attn_meta=None,
    ):
        del attention_inputs, attn_meta
        return super().forward(hidden_states, residual, fmha_impl, kv_cache)


class _VLEmbedding(nn.Module):
    def forward(
        self,
        input_ids,
        position_ids,
        token_type_ids,
        text_tokens_mask,
    ):
        del position_ids, token_type_ids, text_tokens_mask
        return _Embedding()(input_ids)


class _MultimodalEmbeddingInjector(nn.Module):
    def forward(self, embeddings, multimodal_features, multimodal_locs):
        del multimodal_features, multimodal_locs
        return embeddings


class _DeepstackInjector(nn.Module):
    def forward(
        self,
        hidden_states,
        mm_deepstack_embeds,
        multimodal_locs,
        layer_id,
    ):
        injected = hidden_states.clone()
        for stack, loc in zip(mm_deepstack_embeds, multimodal_locs):
            layer_embed = stack[layer_id]
            injected.narrow(0, loc, layer_embed.shape[0]).add_(layer_embed)
        return injected


class _QwenNorm(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 10


class _MoeNorm(nn.Module):
    def forward(self, hidden_states, residual):
        logical_hidden = hidden_states + residual
        return logical_hidden * 10, logical_hidden


class _DisaggregatePreLayer(nn.Module):
    def forward(self, input_ids):
        residual = _Embedding()(input_ids)
        return torch.zeros_like(residual), residual


class _DisaggregateLayer(nn.Module):
    def __init__(self, residual_increment: float, is_last_layer: bool = False) -> None:
        super().__init__()
        self.residual_increment = residual_increment
        self.is_last_layer = is_last_layer

    def forward(self, residual, attention_output):
        next_residual = residual + self.residual_increment
        next_attention_input = (
            next_residual if self.is_last_layer else torch.zeros_like(next_residual)
        )
        return next_attention_input, next_residual


def _copy_recv_payloads(payloads: list[torch.Tensor]):
    payload_iter = iter(payloads)

    def fake_recv(tensor, rank, group):
        payload = next(payload_iter)
        tensor.copy_(payload.to(device=tensor.device, dtype=tensor.dtype))

    return fake_recv


def _make_disaggregate_control_receiver() -> Qwen3GemmModel:
    model = Qwen3GemmModel.__new__(Qwen3GemmModel)
    nn.Module.__init__(model)
    model.attn_dp_rank = [0, 1]
    model.micro_batch_size = 2
    model.device = "cpu"
    model.capture_hidden_states = False
    return model


def _run_disaggregate_final_output(capture_hidden_states: bool) -> torch.Tensor:
    model = Qwen3GemmModel.__new__(Qwen3GemmModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_state_capture_layer_ids=[1, 0])
    model.layer_num = 2
    model.capture_hidden_states = False
    model.capture_hidden_states_by_attn_rank = [False]
    model.attn_dp_rank = [0]
    model.micro_batch_size = 1
    model.device = "cpu"
    model.pre_layer = _DisaggregatePreLayer()
    model.layers = nn.ModuleList(
        [_DisaggregateLayer(1), _DisaggregateLayer(2, is_last_layer=True)]
    )
    model.norm = _QwenNorm()
    model._init_capture_context(
        model._capture_canonical_layer,
        model._capture_canonical_final,
    )
    object.__setattr__(
        model,
        "recv_from_attention",
        lambda micro_batch_sizes, total_tokens: torch.zeros((total_tokens, 2)),
    )
    sent = []
    object.__setattr__(
        model,
        "send_to_attention",
        lambda tensor, micro_batch_sizes, captured=None: sent.append(
            (captured if captured is not None else tensor).clone()
        ),
    )
    input_ids = torch.tensor([1, 2], dtype=torch.int64)
    with patch(
        "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
        side_effect=_copy_recv_payloads(
            [
                torch.tensor(
                    [input_ids.numel(), int(capture_hidden_states)],
                    dtype=torch.int64,
                ),
                input_ids,
            ]
        ),
    ):
        model.forward_micro_batch([])
    return sent[-1]


def _make_qwen_model(layer_ids: list[int]) -> Qwen3Model:
    model = Qwen3Model.__new__(Qwen3Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_state_capture_layer_ids=layer_ids)
    model.layer_num = 3
    model.kv_cache = None
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([_QwenLayer(1), _QwenLayer(2), _QwenLayer(3)])
    model.norm = _QwenNorm()
    model._init_capture_context(
        model._capture_canonical_layer,
        model._capture_canonical_final,
    )
    return model


def _make_moe_model(layer_ids: list[int]) -> GenericMoeModel:
    model = GenericMoeModel.__new__(GenericMoeModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_state_capture_layer_ids=layer_ids)
    model.layer_num = 2
    model.kv_cache = None
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([_MoeLayer(1, 10), _MoeLayer(2, 20)])
    model.norm = _MoeNorm()
    model._init_capture_context(
        model._capture_canonical_layer,
        model._capture_canonical_final,
    )
    return model


def _make_qwen3_next_model(layer_ids: list[int]) -> Qwen3NextModel:
    model = Qwen3NextModel.__new__(Qwen3NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_state_capture_layer_ids=layer_ids)
    model.kv_cache = None
    model.parallelism_config = SimpleNamespace(
        prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
    )
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([_Qwen3NextLayer(1, 10), _Qwen3NextLayer(2, 20)])
    model.norm = _MoeNorm()
    model._init_capture_context(
        model._capture_canonical_layer,
        model._capture_canonical_final,
    )
    return model


def _make_kimi_linear_model(layer_ids: list[int]) -> KimiLinearModel:
    model = KimiLinearModel.__new__(KimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_state_capture_layer_ids=layer_ids)
    model.kv_cache = None
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([_KimiLayer(1, 10), _KimiLayer(2, 20)])
    model.norm = _MoeNorm()
    model._init_capture_context(
        model._capture_canonical_layer,
        model._capture_canonical_final,
    )
    return model


def _configure_vl_model(model, layer_ids: list[int], layers: nn.ModuleList):
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_state_capture_layer_ids=layer_ids)
    model.layer_num = len(layers)
    model.kv_cache = None
    model.embed_tokens = _VLEmbedding()
    model.multimodal_embedding_injector = _MultimodalEmbeddingInjector()
    model.multimodal_deepstack_injector = _DeepstackInjector()
    model.layers = layers
    return model


def _make_qwen3vl_model(layer_ids: list[int]) -> Qwen3VLModel:
    model = Qwen3VLModel.__new__(Qwen3VLModel)
    _configure_vl_model(
        model,
        layer_ids,
        nn.ModuleList([_QwenLayer(1), _QwenLayer(2)]),
    )
    model.norm = _QwenNorm()
    model._init_capture_context(
        model._capture_canonical_layer,
        model._capture_canonical_final,
    )
    return model


def _make_qwen3vl_moe_model(layer_ids: list[int]) -> Qwen3VLMoeModel:
    model = Qwen3VLMoeModel.__new__(Qwen3VLMoeModel)
    _configure_vl_model(
        model,
        layer_ids,
        nn.ModuleList([_MoeLayer(1, 10), _MoeLayer(2, 20)]),
    )
    model.norm = _MoeNorm()
    model._init_capture_context(
        model._capture_canonical_layer,
        model._capture_canonical_final,
    )
    return model


def _make_vl_inputs(capture_hidden_states: bool) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor([1, 2], dtype=torch.int64),
        combo_position_ids=None,
        embedding_inputs=SimpleNamespace(
            combo_tokens_type_ids=None,
            text_tokens_mask=None,
        ),
        multimodal_inputs=SimpleNamespace(
            multimodal_features=[torch.zeros((1, 2))],
            mm_features_locs=torch.tensor([0]),
            mm_extra_input=[torch.tensor([100.0, 100.0, 200.0, 200.0])],
        ),
        capture_hidden_states=capture_hidden_states,
    )


def _deepstack_offsets(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    layer0 = torch.zeros_like(reference)
    layer1 = torch.zeros_like(reference)
    layer0[0] = 100
    layer1[0] = 200
    return layer0, layer1


def _forward_kimi(model: KimiLinearModel, inputs: SimpleNamespace) -> torch.Tensor:
    attention_inputs = SimpleNamespace(is_prefill=False, is_target_verify=False)
    with patch(
        "rtp_llm.models_py.model_desc.kimi_linear.get_primary_attention_inputs",
        return_value=attention_inputs,
    ), patch(
        "rtp_llm.models_py.model_desc.kimi_linear.select_attention_inputs_for_layer",
        return_value=attention_inputs,
    ):
        return model.forward(inputs, fmha_impl=object()).hidden_states


class HiddenStateCaptureTest(unittest.TestCase):
    def test_context_packs_layers_in_configured_order(self) -> None:
        model = _CaptureModel([2, 0])
        capture = model.capture_context(True)
        layer0 = torch.tensor([[1.0, 2.0]])
        layer1 = torch.tensor([[3.0, 4.0]])
        layer2 = torch.tensor([[5.0, 6.0]])
        final_hidden_states = torch.tensor([[7.0, 8.0]])

        capture.capture_layer(0, layer0)
        capture.capture_layer(1, layer1)
        capture.capture_layer(2, layer2)
        packed = capture.finalize(final_hidden_states).hidden_states

        torch.testing.assert_close(
            packed,
            torch.cat((layer2, layer0, final_hidden_states), dim=-1),
        )
        self.assertEqual(packed.shape[-1], 6)

    def test_target_models_register_instance_capture_capability(self) -> None:
        target_models = (
            _make_qwen_model([]),
            _make_moe_model([]),
            _make_qwen3_next_model([]),
            _make_kimi_linear_model([]),
            _make_qwen3vl_model([]),
            _make_qwen3vl_moe_model([]),
        )

        for model in target_models:
            with self.subTest(model=type(model).__name__):
                self.assertIsInstance(model, GptModelBase)
                self.assertTrue(model.supports_hidden_state_capture)

    def test_multimodal_generic_disables_inherited_capture_capability(self) -> None:
        def fake_parent_init(model, *args, **kwargs) -> None:
            del args, kwargs
            nn.Module.__init__(model)
            model.config = SimpleNamespace(hidden_state_capture_layer_ids=[0])
            model._init_capture_context(
                lambda hidden, residual: hidden, lambda hidden, residual: hidden
            )

        with patch.object(GenericMoeModel, "__init__", fake_parent_init):
            model = MultimodalGenericModel()

        self.assertFalse(model.supports_hidden_state_capture)

    def test_target_models_do_not_construct_or_pack_collectors(self) -> None:
        model_desc_dir = Path(__file__).resolve().parents[1]
        target_files = (
            "qwen3.py",
            "generic_moe.py",
            "qwen3_next.py",
            "kimi_linear.py",
            "qwen3vl.py",
            "qwen3vl_moe.py",
        )

        for target_file in target_files:
            model_path = model_desc_dir / target_file
            tree = ast.parse(model_path.read_text(), filename=str(model_path))
            direct_calls = {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            with self.subTest(model=target_file):
                self.assertNotIn("HiddenStateCaptureCollector", direct_calls)
                self.assertNotIn("PyModelOutputs", direct_calls)

    def test_inactive_context_is_reused_and_skips_layer_hook(self) -> None:
        model = _CaptureModel([0])
        first = model.capture_context(False)
        second = model.capture_context(False)

        self.assertIs(first, second)
        self.assertFalse(first.enabled)
        first.capture_layer(0, torch.tensor([[3.0, 4.0]]))
        self.assertEqual(model.layer_hook_calls, [])

        final_hidden_states = torch.tensor([[1.0, 2.0]])
        output = first.finalize(final_hidden_states)
        self.assertEqual(
            output.hidden_states.data_ptr(), final_hidden_states.data_ptr()
        )
        self.assertEqual(model.final_hook_calls, 1)

    def test_active_context_is_forward_local_and_canonicalizes_lazily(self) -> None:
        model = _CaptureModel([1])
        first = model.capture_context(True)
        second = model.capture_context(True)

        self.assertIsNot(first, second)
        self.assertTrue(first.enabled)
        first.capture_layer(0, torch.tensor([[9.0, 0.0]]))
        self.assertEqual(model.layer_hook_calls, [])

        hidden_states = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        first.capture_layer(1, hidden_states, residual)
        self.assertEqual(model.layer_hook_calls, [1])
        output = first.finalize(hidden_states, residual)
        torch.testing.assert_close(
            output.hidden_states,
            torch.cat((hidden_states + residual, hidden_states + residual), dim=-1),
        )

    def test_qwen_capture_uses_configured_order_and_post_final_norm(self) -> None:
        model = _make_qwen_model([2, 0])
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=True,
        )

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids)
        layer0 = embedded + 1
        layer2 = layer0 + 2 + 3
        expected = torch.cat((layer2, layer0, layer2 * 10), dim=-1)

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 6))

    def test_qwen_positive_path_keeps_ordinary_hidden_shape(self) -> None:
        model = _make_qwen_model([2, 0])
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=False,
        )

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        expected = (model.embed_tokens(inputs.input_ids) + 1 + 2 + 3) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_qwen3_next_capture_uses_complete_residual_stream(self) -> None:
        model = _make_qwen3_next_model([1, 0])
        self.assertTrue(model.supports_hidden_state_capture)
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=True,
        )
        attention_inputs = SimpleNamespace(is_prefill=False, is_target_verify=False)

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.get_primary_attention_inputs",
            return_value=attention_inputs,
        ), patch(
            "rtp_llm.models_py.model_desc.qwen3_next.select_attention_inputs_for_layer",
            return_value=attention_inputs,
        ):
            outputs = model.forward(inputs, fmha_impl=object()).hidden_states

        embedded = model.embed_tokens(inputs.input_ids)
        layer0_hidden = embedded + 1
        layer0_residual = torch.full_like(embedded, 10)
        layer0 = layer0_hidden + layer0_residual
        layer1_hidden = layer0_hidden + 2
        layer1_residual = layer0_residual + 20
        layer1 = layer1_hidden + layer1_residual
        expected = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 6))

    def test_qwen3_next_positive_path_keeps_ordinary_hidden_shape(self) -> None:
        model = _make_qwen3_next_model([1, 0])
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=False,
        )
        attention_inputs = SimpleNamespace(is_prefill=False, is_target_verify=False)

        with patch(
            "rtp_llm.models_py.model_desc.qwen3_next.get_primary_attention_inputs",
            return_value=attention_inputs,
        ), patch(
            "rtp_llm.models_py.model_desc.qwen3_next.select_attention_inputs_for_layer",
            return_value=attention_inputs,
        ):
            outputs = model.forward(inputs, fmha_impl=object()).hidden_states

        embedded = model.embed_tokens(inputs.input_ids)
        final_hidden = embedded + 1 + 2
        final_residual = torch.full_like(embedded, 30)
        expected = (final_hidden + final_residual) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_kimi_linear_capture_uses_hidden_plus_residual_and_post_norm(self) -> None:
        model = _make_kimi_linear_model([1, 0])
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=True,
        )

        outputs = _forward_kimi(model, inputs)
        embedded = model.embed_tokens(inputs.input_ids)
        layer0_hidden = embedded + 1
        layer0_residual = torch.full_like(embedded, 10)
        layer0 = layer0_hidden + layer0_residual
        layer1_hidden = layer0_hidden + 2
        layer1_residual = layer0_residual + 20
        layer1 = layer1_hidden + layer1_residual
        expected = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 6))

    def test_kimi_linear_capture_disabled_keeps_ordinary_width(self) -> None:
        model = _make_kimi_linear_model([1, 0])
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=False,
        )

        outputs = _forward_kimi(model, inputs)
        embedded = model.embed_tokens(inputs.input_ids)
        expected = (embedded + 1 + 2 + 30) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_qwen3vl_capture_is_after_deepstack_and_in_configured_order(self) -> None:
        model = _make_qwen3vl_model([1, 0])
        inputs = _make_vl_inputs(capture_hidden_states=True)

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids, None, None, None)
        deepstack0, deepstack1 = _deepstack_offsets(embedded)
        layer0 = embedded + 1 + deepstack0
        layer1 = layer0 + 2 + deepstack1
        expected = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 6))

    def test_qwen3vl_capture_disabled_keeps_ordinary_width(self) -> None:
        model = _make_qwen3vl_model([1, 0])
        inputs = _make_vl_inputs(capture_hidden_states=False)

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids, None, None, None)
        deepstack0, deepstack1 = _deepstack_offsets(embedded)
        expected = (embedded + 1 + deepstack0 + 2 + deepstack1) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_qwen3vl_moe_capture_includes_deepstack_and_residual(self) -> None:
        model = _make_qwen3vl_moe_model([1, 0])
        inputs = _make_vl_inputs(capture_hidden_states=True)

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids, None, None, None)
        deepstack0, deepstack1 = _deepstack_offsets(embedded)
        layer0_hidden = embedded + 1 + deepstack0
        layer0_residual = torch.full_like(embedded, 10)
        layer0 = layer0_hidden + layer0_residual
        layer1_hidden = layer0_hidden + 2 + deepstack1
        layer1_residual = layer0_residual + 20
        layer1 = layer1_hidden + layer1_residual
        expected = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 6))

    def test_qwen3vl_moe_capture_disabled_keeps_ordinary_width(self) -> None:
        model = _make_qwen3vl_moe_model([1, 0])
        inputs = _make_vl_inputs(capture_hidden_states=False)

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids, None, None, None)
        deepstack0, deepstack1 = _deepstack_offsets(embedded)
        final_hidden = embedded + 1 + deepstack0 + 2 + deepstack1
        final_residual = torch.full_like(embedded, 30)
        expected = (final_hidden + final_residual) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_disaggregate_attention_appends_capture_flag_to_split_info(self) -> None:
        for capture_hidden_states in (False, True):
            with self.subTest(capture_hidden_states=capture_hidden_states):
                model = Qwen3AttnModel.__new__(Qwen3AttnModel)
                nn.Module.__init__(model)
                model.device = "cpu"
                model.ffn_service_rank = 2
                inputs = [
                    SimpleNamespace(
                        input_ids=torch.tensor([1, 2], dtype=torch.int32),
                        capture_hidden_states=capture_hidden_states,
                    ),
                    SimpleNamespace(
                        input_ids=torch.tensor([3], dtype=torch.int32),
                        capture_hidden_states=capture_hidden_states,
                    ),
                ]

                with patch(
                    "rtp_llm.models_py.model_desc.disaggregate_qwen3.send"
                ) as send_mock:
                    model.send_mirco_batch_split_info(inputs)

                self.assertEqual(send_mock.call_count, 3)
                control_message = send_mock.call_args_list[0].args[0]
                torch.testing.assert_close(
                    control_message,
                    torch.tensor([2, 1, int(capture_hidden_states)], dtype=torch.int64),
                )
                self.assertIs(send_mock.call_args_list[1].args[0], inputs[0].input_ids)
                self.assertIs(send_mock.call_args_list[2].args[0], inputs[1].input_ids)

    def test_disaggregate_attention_skips_zero_token_real_and_fake_lanes(
        self,
    ) -> None:
        model = Qwen3AttnModel.__new__(Qwen3AttnModel)
        nn.Module.__init__(model)
        model.device = "cpu"
        model.ffn_service_rank = 2
        inputs = [
            SimpleNamespace(
                input_ids=torch.empty(0, dtype=torch.int32),
                capture_hidden_states=False,
            ),
            SimpleNamespace(
                input_ids=torch.tensor([10, 11], dtype=torch.int32),
                capture_hidden_states=False,
            ),
            SimpleNamespace(
                input_ids=torch.empty(0, dtype=torch.int32),
                capture_hidden_states=False,
            ),
            SimpleNamespace(
                input_ids=torch.tensor([20], dtype=torch.int32),
                capture_hidden_states=False,
            ),
        ]

        with patch("rtp_llm.models_py.model_desc.disaggregate_qwen3.send") as send_mock:
            model.send_mirco_batch_split_info(inputs)

        self.assertEqual(send_mock.call_count, 3)
        control_message = send_mock.call_args_list[0].args[0]
        torch.testing.assert_close(
            control_message,
            torch.tensor([0, 2, 0, 1, 0], dtype=torch.int64),
        )
        self.assertEqual(control_message.numel(), len(inputs) + 1)
        payloads = [call.args[0] for call in send_mock.call_args_list[1:]]
        self.assertTrue(all(payload.numel() > 0 for payload in payloads))
        self.assertIs(payloads[0], inputs[1].input_ids)
        self.assertIs(payloads[1], inputs[3].input_ids)

    def test_disaggregate_attention_rejects_inconsistent_micro_batch_flags(
        self,
    ) -> None:
        model = Qwen3AttnModel.__new__(Qwen3AttnModel)
        nn.Module.__init__(model)
        model.device = "cpu"
        model.ffn_service_rank = 2
        inputs = [
            SimpleNamespace(
                input_ids=torch.tensor([1, 2], dtype=torch.int32),
                capture_hidden_states=False,
            ),
            SimpleNamespace(
                input_ids=torch.tensor([3], dtype=torch.int32),
                capture_hidden_states=True,
            ),
        ]

        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.send"
        ) as send_mock, self.assertRaisesRegex(
            Exception,
            "capture_hidden_states must be consistent within an attention DP rank",
        ):
            model.send_mirco_batch_split_info(inputs)

        send_mock.assert_not_called()

    def test_disaggregate_attention_final_width_tracks_capture_flag(self) -> None:
        model = Qwen3AttnModel.__new__(Qwen3AttnModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            hidden_size=2,
            hidden_state_capture_layer_ids=[1, 0],
            compute_dtype=torch.float16,
        )
        model.device = "cpu"
        model.ffn_service_rank = 2

        for capture_hidden_states, expected_width in ((False, 2), (True, 6)):
            with self.subTest(capture_hidden_states=capture_hidden_states):
                with patch(
                    "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv"
                ) as recv_mock:
                    output = model.recv_final_from_ffn_service(
                        token_num=3, capture_hidden_states=capture_hidden_states
                    )

                self.assertEqual(tuple(output.shape), (3, expected_width))
                self.assertEqual(recv_mock.call_count, 1)

    def test_disaggregate_receive_buffers_use_configured_compute_dtype(self) -> None:
        for compute_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(compute_dtype=compute_dtype):
                gemm_model = Qwen3GemmModel.__new__(Qwen3GemmModel)
                nn.Module.__init__(gemm_model)
                gemm_model.config = SimpleNamespace(
                    attn_config=SimpleNamespace(head_num=2, size_per_head=4),
                    compute_dtype=compute_dtype,
                )
                gemm_model.device = "cpu"
                gemm_model.attn_dp_rank = [0, 1]

                attn_model = Qwen3AttnModel.__new__(Qwen3AttnModel)
                nn.Module.__init__(attn_model)
                attn_model.config = SimpleNamespace(
                    attn_config=SimpleNamespace(
                        head_num=2, kv_head_num=1, size_per_head=4
                    ),
                    hidden_size=8,
                    hidden_state_capture_layer_ids=[1, 0],
                    compute_dtype=compute_dtype,
                )
                attn_model.device = "cpu"
                attn_model.ffn_service_rank = 2

                with patch(
                    "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv"
                ) as recv_mock:
                    from_attention = gemm_model.recv_from_attention([1, 2], 3)
                    from_ffn = attn_model.recv_from_ffn_service(3)
                    final_from_ffn = attn_model.recv_final_from_ffn_service(3, True)

                self.assertEqual(from_attention.dtype, compute_dtype)
                self.assertEqual(from_ffn.dtype, compute_dtype)
                self.assertEqual(final_from_ffn.dtype, compute_dtype)
                self.assertEqual(recv_mock.call_count, 4)

    def test_disaggregate_ffn_applies_capture_flag_for_true_and_false(self) -> None:
        for capture_hidden_states in (False, True):
            with self.subTest(capture_hidden_states=capture_hidden_states):
                model = _make_disaggregate_control_receiver()
                capture_flag = int(capture_hidden_states)
                payloads = [
                    torch.tensor([2, 1, capture_flag], dtype=torch.int64),
                    torch.tensor([1, 2, capture_flag], dtype=torch.int64),
                    torch.tensor([10, 11], dtype=torch.int32),
                    torch.tensor([20], dtype=torch.int32),
                    torch.tensor([12], dtype=torch.int32),
                    torch.tensor([21, 22], dtype=torch.int32),
                ]

                with patch(
                    "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
                    side_effect=_copy_recv_payloads(payloads),
                ) as recv_mock:
                    input_ids_list, split_info = model.recv_micro_batch_split_info()

                self.assertEqual(recv_mock.call_count, len(payloads))
                self.assertEqual(model.capture_hidden_states, capture_hidden_states)
                self.assertEqual(
                    model.capture_hidden_states_by_attn_rank,
                    [capture_hidden_states, capture_hidden_states],
                )
                self.assertEqual(split_info, BatchSplitInfo([3, 3], [[2, 1], [1, 2]]))
                torch.testing.assert_close(
                    input_ids_list[0], torch.tensor([10, 11, 20], dtype=torch.int32)
                )
                torch.testing.assert_close(
                    input_ids_list[1], torch.tensor([12, 21, 22], dtype=torch.int32)
                )

    def test_disaggregate_ffn_skips_zero_token_real_and_fake_lanes(self) -> None:
        model = _make_disaggregate_control_receiver()
        model.micro_batch_size = 4
        payloads = [
            torch.tensor([0, 2, 0, 0, 0], dtype=torch.int64),
            torch.tensor([0, 0, 0, 1, 0], dtype=torch.int64),
            torch.tensor([10, 11], dtype=torch.int32),
            torch.tensor([20], dtype=torch.int32),
        ]

        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
            side_effect=_copy_recv_payloads(payloads),
        ) as recv_mock:
            input_ids_list, split_info = model.recv_micro_batch_split_info()

        self.assertEqual(recv_mock.call_count, len(payloads))
        self.assertEqual(
            split_info,
            BatchSplitInfo([0, 2, 0, 1], [[0, 0], [2, 0], [0, 0], [0, 1]]),
        )
        self.assertEqual(
            [input_ids.numel() for input_ids in input_ids_list], [0, 2, 0, 1]
        )
        torch.testing.assert_close(
            input_ids_list[1], torch.tensor([10, 11], dtype=torch.int32)
        )
        torch.testing.assert_close(
            input_ids_list[3], torch.tensor([20], dtype=torch.int32)
        )
        payload_recv_calls = recv_mock.call_args_list[2:]
        self.assertEqual(
            [call.args[0].numel() for call in payload_recv_calls],
            [2, 1],
        )
        self.assertEqual(
            [call.args[1] for call in payload_recv_calls],
            [0, 1],
        )

    def test_disaggregate_ffn_routes_per_rank_final_hidden_states(self) -> None:
        model = Qwen3GemmModel.__new__(Qwen3GemmModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_state_capture_layer_ids=[1, 0])
        model.layer_num = 2
        model.capture_hidden_states = False
        model.capture_hidden_states_by_attn_rank = [False, False]
        model.attn_dp_rank = [0, 1]
        model.micro_batch_size = 1
        model.device = "cpu"
        model.pre_layer = _DisaggregatePreLayer()
        model.layers = nn.ModuleList(
            [_DisaggregateLayer(1), _DisaggregateLayer(2, is_last_layer=True)]
        )
        model.norm = _QwenNorm()
        model._init_capture_context(
            model._capture_canonical_layer,
            model._capture_canonical_final,
        )
        object.__setattr__(
            model,
            "recv_from_attention",
            lambda micro_batch_sizes, total_tokens: torch.zeros((total_tokens, 2)),
        )
        sent = []

        def capture_send(tensor, rank, group):
            sent.append((rank, tensor.clone()))

        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
            side_effect=_copy_recv_payloads(
                [
                    torch.tensor([1, 0], dtype=torch.int64),
                    torch.tensor([1, 1], dtype=torch.int64),
                    torch.tensor([1], dtype=torch.int32),
                    torch.tensor([2], dtype=torch.int32),
                ]
            ),
        ), patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.send",
            side_effect=capture_send,
        ):
            model.forward_micro_batch([])

        self.assertTrue(model.capture_hidden_states)
        self.assertEqual(model.capture_hidden_states_by_attn_rank, [False, True])
        self.assertEqual(len(sent), 6)
        false_rank, false_payload = sent[-2]
        true_rank, true_payload = sent[-1]
        self.assertEqual((false_rank, true_rank), (0, 1))

        embedded = _Embedding()(torch.tensor([1, 2], dtype=torch.int64))
        layer0 = embedded + 1
        layer1 = layer0 + 2
        expected_false = (layer1 * 10)[:1]
        expected_true = torch.cat((layer1, layer0, layer1 * 10), dim=-1)[1:]
        torch.testing.assert_close(false_payload, expected_false)
        torch.testing.assert_close(true_payload, expected_true)

        attention_model = Qwen3AttnModel.__new__(Qwen3AttnModel)
        nn.Module.__init__(attention_model)
        attention_model.config = SimpleNamespace(
            hidden_size=2,
            hidden_state_capture_layer_ids=[1, 0],
            compute_dtype=torch.float16,
        )
        attention_model.device = "cpu"
        attention_model.ffn_service_rank = 2
        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
            side_effect=_copy_recv_payloads([false_payload, true_payload]),
        ):
            false_output = attention_model.recv_final_from_ffn_service(1, False)
            true_output = attention_model.recv_final_from_ffn_service(1, True)

        self.assertEqual(tuple(false_output.shape), (1, 2))
        self.assertEqual(tuple(true_output.shape), (1, 6))
        torch.testing.assert_close(false_output, expected_false.half())
        torch.testing.assert_close(true_output, expected_true.half())

    def test_disaggregate_final_hidden_numeric_parity_capture_on_off(self) -> None:
        capture_off = _run_disaggregate_final_output(False)
        capture_on = _run_disaggregate_final_output(True)
        capture_on_final = capture_on[..., -capture_off.shape[-1] :]

        embedded = _Embedding()(torch.tensor([1, 2], dtype=torch.int64))
        expected_once_normalized = (embedded + 1 + 2) * 10
        torch.testing.assert_close(capture_off, expected_once_normalized)
        torch.testing.assert_close(capture_on_final, expected_once_normalized)
        torch.testing.assert_close(capture_on_final, capture_off)

    def test_disaggregate_qwen_packs_complete_service_side_boundaries(self) -> None:
        model = Qwen3GemmModel.__new__(Qwen3GemmModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_state_capture_layer_ids=[1, 0])
        model.layer_num = 2
        model.capture_hidden_states = False
        model.attn_dp_rank = [0]
        model.micro_batch_size = 1
        model.device = "cpu"
        model.pre_layer = _DisaggregatePreLayer()
        model.layers = nn.ModuleList(
            [_DisaggregateLayer(1), _DisaggregateLayer(2, is_last_layer=True)]
        )
        model.norm = _QwenNorm()
        model._init_capture_context(
            model._capture_canonical_layer,
            model._capture_canonical_final,
        )
        input_ids = torch.tensor([1, 2], dtype=torch.int64)
        object.__setattr__(
            model,
            "recv_from_attention",
            lambda micro_batch_sizes, total_tokens: torch.zeros((total_tokens, 2)),
        )
        sent = []
        object.__setattr__(
            model,
            "send_to_attention",
            lambda tensor, micro_batch_sizes, captured=None: sent.append(
                (captured if captured is not None else tensor).clone()
            ),
        )

        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
            side_effect=_copy_recv_payloads(
                [torch.tensor([2, 1], dtype=torch.int64), input_ids]
            ),
        ):
            model.forward_micro_batch([])

        self.assertTrue(model.capture_hidden_states)
        self.assertEqual([tensor.shape[-1] for tensor in sent], [2, 2, 6])
        embedded = _Embedding()(input_ids)
        layer0 = embedded + 1
        layer1 = layer0 + 2
        expected = torch.cat((layer1, layer0, layer1 * 10), dim=-1)
        torch.testing.assert_close(sent[-1], expected)

    def test_generic_moe_positive_path_keeps_ordinary_hidden_shape(self) -> None:
        model = _make_moe_model([1, 0])
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=False,
        )

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids)
        layer0_hidden = embedded + 1
        layer0_residual = torch.full_like(embedded, 10)
        layer1_hidden = layer0_hidden + 2
        layer1_residual = layer0_residual + 20
        expected = (layer1_hidden + layer1_residual) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_generic_moe_captures_full_post_layer_state(self) -> None:
        model = _make_moe_model([1, 0])
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2], dtype=torch.int64),
            capture_hidden_states=True,
        )

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids)
        layer0_hidden = embedded + 1
        layer0_residual = torch.full_like(embedded, 10)
        layer0 = layer0_hidden + layer0_residual
        layer1_hidden = layer0_hidden + 2
        layer1_residual = layer0_residual + 20
        layer1 = layer1_hidden + layer1_residual
        expected = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
