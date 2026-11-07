import ast
import os
import subprocess
import sys
import textwrap
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeModel
from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext
from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.multimodal_generic import MultimodalGenericModel
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextModel
from rtp_llm.models_py.model_desc.qwen3vl import Qwen3VLModel
from rtp_llm.models_py.model_desc.qwen3vl_moe import Qwen3VLMoeModel
from rtp_llm.models_py.utils.profiler import (
    _NOOP_RECORD_FUNCTION_RANGE,
    disable_record_function_ranges,
)


class _CaptureModel(GptModelBase):
    def __init__(self, layer_ids: list[int]) -> None:
        config = SimpleNamespace(
            num_layers=3,
            vocab_size=8,
            hidden_state_capture_layer_ids=layer_ids,
            capture_aux_hidden_layer_ids=None,
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

    def forward(
        self, hidden_states, residual, fmha_impl, kv_cache=None, prev_topk_indices=None
    ):
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


def _make_inputs(
    input_ids: list[int],
    capture_hidden_states: bool,
    dtype: torch.dtype = torch.int64,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor(input_ids, dtype=dtype),
        capture_hidden_states=capture_hidden_states,
    )


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
    model.indexer_sources = {0: 0, 1: 1}
    model.offload_plans = {}
    model.config = SimpleNamespace(
        hidden_state_capture_layer_ids=layer_ids,
        attn_config=SimpleNamespace(use_mla=False, is_sparse=False),
    )
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


def _make_qwen3_next_model(
    layer_ids: list[int], aux_layer_ids: tuple[int, ...] = ()
) -> Qwen3NextModel:
    model = Qwen3NextModel.__new__(Qwen3NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_state_capture_layer_ids=layer_ids,
        capture_aux_hidden_layer_ids=aux_layer_ids,
    )
    model._mtp_aux_capture_layer_ids = aux_layer_ids
    model._mtp_aux_capture_layer_id_set = frozenset(aux_layer_ids)
    model._mtp_target_hidden_states = None
    model._mtp_target_graph_buffer = None
    model._mtp_target_prompt_buffer = None
    model._mtp_aux_capture_buffer = None
    model._mtp_aux_capture_rows = 0
    model._mtp_aux_capture_index = 0
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


def _forward_qwen3_next(model: Qwen3NextModel, inputs: SimpleNamespace):
    attention_inputs = SimpleNamespace(
        is_prefill=False, is_target_verify=False, is_cuda_graph=False
    )
    inputs.attention_inputs = {"default": attention_inputs}
    with patch(
        "rtp_llm.models_py.model_desc.qwen3_next.get_primary_attention_inputs",
        return_value=attention_inputs,
    ), patch(
        "rtp_llm.models_py.model_desc.qwen3_next.select_attention_inputs_for_layer",
        return_value=attention_inputs,
    ):
        return model.forward(inputs, fmha_impl=object())


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

    def test_capture_profiler_off_allocates_no_ranges(self) -> None:
        class UnformattableLayerId(int):
            def __format__(self, spec):
                raise AssertionError("profiler-off path formatted a dynamic range")

        model = _CaptureModel([2, 0])
        hidden = torch.tensor([[1.0, 2.0]])
        with patch.object(
            torch.autograd, "_profiler_enabled", return_value=False
        ), patch.object(torch._C._profiler, "_RecordFunctionFast") as record_range:
            capture = model.capture_context(True)
            self.assertIs(capture._profile_noop, _NOOP_RECORD_FUNCTION_RANGE)
            for layer_id in range(3):
                capture.capture_layer(UnformattableLayerId(layer_id), hidden)
            output = capture.finalize(hidden).hidden_states
        record_range.assert_not_called()
        torch.testing.assert_close(output, torch.cat([hidden] * 3, dim=-1))

    def test_capture_profiler_unavailable_apis_are_noop(self) -> None:
        class UnformattableLayerId(int):
            def __format__(self, spec):
                raise AssertionError("profiler-noop path formatted a dynamic range")

        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        for owner_name, attribute, missing, replacement in (
            ("torch_c", "_profiler", True, None),
            ("profiler", "_RecordFunctionFast", True, None),
            ("profiler", "_RecordFunctionFast", False, None),
            ("profiler", "_RecordFunctionFast", False, True),
            ("autograd", "_profiler_enabled", True, None),
            ("autograd", "_profiler_enabled", False, None),
            ("autograd", "_profiler_enabled", False, True),
        ):
            with self.subTest(
                owner=owner_name,
                attribute=attribute,
                missing=missing,
                replacement=replacement,
            ):
                record_range = Mock(
                    side_effect=AssertionError("unavailable API path created a range")
                )
                profiler = SimpleNamespace(_RecordFunctionFast=record_range)
                layer_hook = Mock(side_effect=lambda h, r: h + r)
                final_hook = Mock(side_effect=lambda h, r: h + r)
                plan = CaptureContext.configured([2, 0], layer_hook, final_hook)
                with patch.object(
                    torch._C, "_profiler", profiler, create=True
                ), patch.object(
                    torch.autograd, "_profiler_enabled", return_value=True, create=True
                ):
                    owner = {
                        "torch_c": torch._C,
                        "profiler": profiler,
                        "autograd": torch.autograd,
                    }[owner_name]
                    if missing:
                        delattr(owner, attribute)
                    else:
                        setattr(owner, attribute, replacement)
                    capture = plan.for_forward(True)
                    self.assertIs(capture._profile_enabled, False)
                    for layer_id in range(3):
                        capture.capture_layer(
                            UnformattableLayerId(layer_id), hidden + layer_id, residual
                        )
                    output = capture.finalize(hidden + 10, residual).hidden_states
                record_range.assert_not_called()
                self.assertEqual(layer_hook.call_count, 2)
                final_hook.assert_called_once()
                torch.testing.assert_close(
                    output,
                    torch.cat(
                        [
                            hidden + 2 + residual,
                            hidden + residual,
                            hidden + 10 + residual,
                        ],
                        dim=-1,
                    ),
                )

    def test_capture_profiler_snapshots_callable_and_enabled_state(self) -> None:
        hidden = torch.ones(1, 2)
        plan = CaptureContext.configured([0], lambda h, r: h, lambda h, r: h)
        record_range = Mock(side_effect=lambda name: nullcontext())
        profiler = SimpleNamespace(_RecordFunctionFast=record_range)
        with patch.object(torch._C, "_profiler", profiler, create=True), patch.object(
            torch.autograd, "_profiler_enabled", return_value=True, create=True
        ) as profiler_enabled:
            capture = plan.for_forward(True)
            self.assertIs(capture._profile_enabled, True)
            self.assertIs(capture._record_function_fast, record_range)
            profiler_enabled.assert_called_once_with()
            profiler_enabled.side_effect = AssertionError(
                "profiler state was rechecked"
            )
            with disable_record_function_ranges(), patch.object(
                profiler,
                "_RecordFunctionFast",
                side_effect=AssertionError("range factory was not cached"),
            ):
                capture.capture_layer(0, hidden)
                output = capture.finalize(hidden).hidden_states
        self.assertEqual(
            [call.args[0] for call in record_range.call_args_list],
            ["hidden_state_capture.layer(layer=0)", "hidden_state_capture.pack"],
        )
        torch.testing.assert_close(output, torch.cat([hidden, hidden], dim=-1))

    def test_capture_import_and_forward_do_not_import_fused_moe(self) -> None:
        source = textwrap.dedent(
            """\
            import importlib.abc
            import sys
            from contextlib import nullcontext

            import torch

            profile_enabled = sys.argv[1] == "1"
            missing_profiler_on_import = sys.argv[2] == "1"
            prefix = "rtp_llm.models_py.modules.factory.fused_moe"

            def is_moe(name):
                return name == prefix or name.startswith(prefix + ".")

            assert not any(is_moe(name) for name in sys.modules)

            class RejectMoE(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if is_moe(fullname):
                        raise AssertionError("forbidden MoE import: " + fullname)

            sys.meta_path.insert(0, RejectMoE())
            if missing_profiler_on_import:
                profiler_module = torch._C._profiler
                del torch._C._profiler
                try:
                    from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext
                finally:
                    torch._C._profiler = profiler_module
            else:
                from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext

            class UnformattableLayerId(int):
                def __format__(self, spec):
                    raise AssertionError("profiler-off path formatted a dynamic range")

            layer_calls = []
            final_calls = []

            def canonical_layer(hidden, residual):
                layer_calls.append(hidden)
                return hidden + residual

            def canonical_final(hidden, residual):
                final_calls.append(hidden)
                return hidden + residual

            hidden = torch.tensor([[1.0, 2.0]])
            residual = torch.tensor([[3.0, 4.0]])
            with (
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
                if profile_enabled else nullcontext()
            ) as prof:
                plan = CaptureContext.configured([2, 0], canonical_layer, canonical_final)
                assert not plan.enabled
                assert plan.for_forward(False) is plan
                capture = plan.for_forward(True)
                assert capture is not plan and capture.enabled
                assert capture._profile_enabled is profile_enabled
                for layer_id in range(3):
                    assert capture.wants_layer(layer_id) == (layer_id in (0, 2))
                    capture.capture_layer(
                        layer_id if profile_enabled else UnformattableLayerId(layer_id),
                        hidden + layer_id,
                        residual,
                    )
                output = capture.finalize(hidden + 10, residual).hidden_states

            assert len(layer_calls) == 2 and len(final_calls) == 1
            torch.testing.assert_close(
                output,
                torch.cat(
                    [hidden + 2 + residual, hidden + residual, hidden + 10 + residual],
                    dim=-1,
                ),
            )
            if profile_enabled:
                events = [
                    event for event in prof.events()
                    if event.name.startswith("hidden_state_capture.")
                ]
                assert [event.name for event in events] == [
                    "hidden_state_capture.layer(layer=0)",
                    "hidden_state_capture.layer(layer=2)",
                    "hidden_state_capture.pack",
                ]
                assert all(
                    event.scope == torch.profiler.RecordScope.FUNCTION.value
                    and not event.is_user_annotation
                    for event in events
                )
            assert not any(is_moe(name) for name in sys.modules)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for profile_enabled in (False, True):
            for missing_profiler_on_import in (False, True):
                with self.subTest(
                    profile_enabled=profile_enabled,
                    missing_profiler_on_import=missing_profiler_on_import,
                ):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            source,
                            str(int(profile_enabled)),
                            str(int(missing_profiler_on_import)),
                        ],
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=60,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )

    def test_capture_profiler_selected_layers_and_pack_preserve_values(self) -> None:
        model = _CaptureModel([2, 0])
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            capture = model.capture_context(True)
            for layer_id in range(3):
                capture.capture_layer(layer_id, hidden + layer_id, residual)
            output = capture.finalize(hidden, residual).hidden_states
        events = [
            e for e in prof.events() if e.name.startswith("hidden_state_capture.")
        ]
        self.assertEqual(
            [e.name for e in events],
            [
                "hidden_state_capture.layer(layer=0)",
                "hidden_state_capture.layer(layer=2)",
                "hidden_state_capture.pack",
            ],
        )
        self.assertTrue(all(not e.is_user_annotation for e in events))
        torch.testing.assert_close(
            output,
            torch.cat(
                [hidden + 2 + residual, hidden + residual, hidden + residual], dim=-1
            ),
        )

    def test_capture_profiler_respects_outer_disable_and_inactive_capture(self) -> None:
        model = _CaptureModel([0])
        hidden = torch.ones(1, 2)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            with disable_record_function_ranges():
                capture = model.capture_context(True)
            capture.capture_layer(0, hidden)
            capture.finalize(hidden)
            inactive = model.capture_context(False)
            inactive.capture_layer(0, hidden)
            output = inactive.finalize(hidden).hidden_states
        self.assertEqual(output.data_ptr(), hidden.data_ptr())
        self.assertFalse(
            any(e.name.startswith("hidden_state_capture.") for e in prof.events())
        )

    def test_capture_profiler_exception_closes_layer_and_pack_ranges(self) -> None:
        hidden = torch.ones(1, 2)
        model = _CaptureModel([0])
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            capture = model.capture_context(True)
            with patch.object(
                capture, "_canonical_layer", side_effect=RuntimeError("layer failure")
            ), self.assertRaisesRegex(RuntimeError, "layer failure"):
                capture.capture_layer(0, hidden)
            capture.capture_layer(0, hidden)
            with patch.object(
                torch, "cat", side_effect=RuntimeError("pack failure")
            ), self.assertRaisesRegex(RuntimeError, "pack failure"):
                capture.finalize(hidden)
            output = capture.finalize(hidden).hidden_states
        events = [
            e for e in prof.events() if e.name.startswith("hidden_state_capture.")
        ]
        self.assertEqual(len(events), 4)
        self.assertTrue(all(e.cpu_parent is None for e in events))
        torch.testing.assert_close(output, torch.cat([hidden, hidden], dim=-1))

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

    def test_active_capture_requires_final_hook(self) -> None:
        capture = CaptureContext(enabled=True)

        with self.assertRaisesRegex(RuntimeError, "missing canonical final hook"):
            capture.finalize(torch.ones((1, 2)))

    def test_passthrough_capture_context_preserves_hidden_states(self) -> None:
        hidden_states = torch.ones((1, 2))

        outputs = CaptureContext.passthrough().finalize(hidden_states)

        torch.testing.assert_close(outputs.hidden_states, hidden_states)

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

    def test_unsupported_context_finalize_returns_hidden_states_unchanged(self) -> None:
        hidden_states = torch.tensor([[1.0, 2.0]])
        output = CaptureContext.unsupported().finalize(hidden_states)

        self.assertEqual(output.hidden_states.data_ptr(), hidden_states.data_ptr())

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
        inputs = _make_inputs([1, 2], capture_hidden_states=True)

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        embedded = model.embed_tokens(inputs.input_ids)
        layer0 = embedded + 1
        layer2 = layer0 + 2 + 3
        expected = torch.cat((layer2, layer0, layer2 * 10), dim=-1)

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 6))

    def test_qwen_positive_path_keeps_ordinary_hidden_shape(self) -> None:
        model = _make_qwen_model([2, 0])
        inputs = _make_inputs([1, 2], capture_hidden_states=False)

        outputs = model.forward(inputs, fmha_impl=object()).hidden_states
        expected = (model.embed_tokens(inputs.input_ids) + 1 + 2 + 3) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_qwen3_next_capture_uses_complete_residual_stream(self) -> None:
        model = _make_qwen3_next_model([1, 0])
        self.assertTrue(model.supports_hidden_state_capture)
        inputs = _make_inputs([1, 2], capture_hidden_states=True)
        with patch.object(model.norm, "forward", wraps=model.norm.forward) as norm:
            outputs_with_aux = _forward_qwen3_next(model, inputs)
        norm.assert_called_once()

        outputs = outputs_with_aux.hidden_states
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

        # The new TorchSpec capture must not populate the legacy MTP channel.
        self.assertIsNone(outputs_with_aux.mtp_target_hidden_states)

    def test_qwen3_next_legacy_aux_capture_keeps_new_capture_disabled(self) -> None:
        model = _make_qwen3_next_model([], aux_layer_ids=(0, 1))
        inputs = _make_inputs([1, 2], capture_hidden_states=False)
        with patch.object(model.norm, "forward", wraps=model.norm.forward) as norm:
            outputs = _forward_qwen3_next(model, inputs)
        norm.assert_called_once()
        self.assertEqual(
            outputs.mtp_target_hidden_states.data_ptr(),
            model._mtp_target_prompt_buffer.data_ptr(),
        )

        embedded = model.embed_tokens(inputs.input_ids)
        layer0 = embedded + 1 + 10
        layer1 = embedded + 1 + 2 + 30
        torch.testing.assert_close(outputs.hidden_states, layer1 * 10)
        torch.testing.assert_close(
            outputs.mtp_target_hidden_states, torch.cat((layer0, layer1), dim=-1)
        )
        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))

    def test_qwen3_next_both_capture_transports_are_independent(self) -> None:
        model = _make_qwen3_next_model([1, 0], aux_layer_ids=(0, 1))
        inputs = _make_inputs([1, 2], capture_hidden_states=True)
        embedded = model.embed_tokens(inputs.input_ids)
        layer0 = embedded + 1 + 10
        layer1 = embedded + 1 + 2 + 30
        expected_capture = torch.cat((layer1, layer0, layer1 * 10), dim=-1)
        aux_buffer = None
        for enabled in (True, False, True):
            with self.subTest(capture=enabled):
                inputs.capture_hidden_states = enabled
                with patch.object(
                    model.norm, "forward", wraps=model.norm.forward
                ) as norm:
                    outputs = _forward_qwen3_next(model, inputs)
                norm.assert_called_once()
                torch.testing.assert_close(
                    outputs.hidden_states, expected_capture if enabled else layer1 * 10
                )
                torch.testing.assert_close(
                    outputs.mtp_target_hidden_states,
                    torch.cat((layer0, layer1), dim=-1),
                )
                if aux_buffer is None:
                    aux_buffer = model._mtp_target_prompt_buffer
                self.assertIs(model._mtp_target_prompt_buffer, aux_buffer)
                self.assertEqual(
                    outputs.mtp_target_hidden_states.data_ptr(), aux_buffer.data_ptr()
                )

    def test_qwen3_next_positive_path_keeps_ordinary_hidden_shape(self) -> None:
        model = _make_qwen3_next_model([1, 0])
        inputs = _make_inputs([1, 2], capture_hidden_states=False)
        with patch.object(model.norm, "forward", wraps=model.norm.forward) as norm:
            outputs = _forward_qwen3_next(model, inputs).hidden_states
        norm.assert_called_once()

        embedded = model.embed_tokens(inputs.input_ids)
        final_hidden = embedded + 1 + 2
        final_residual = torch.full_like(embedded, 30)
        expected = (final_hidden + final_residual) * 10

        torch.testing.assert_close(outputs, expected)
        self.assertEqual(tuple(outputs.shape), (2, 2))

    def test_kimi_linear_capture_uses_hidden_plus_residual_and_post_norm(self) -> None:
        model = _make_kimi_linear_model([1, 0])
        inputs = _make_inputs([1, 2], capture_hidden_states=True)

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
        inputs = _make_inputs([1, 2], capture_hidden_states=False)

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

    def test_generic_moe_positive_path_keeps_ordinary_hidden_shape(self) -> None:
        model = _make_moe_model([1, 0])
        inputs = _make_inputs([1, 2], capture_hidden_states=False)

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
        inputs = _make_inputs([1, 2], capture_hidden_states=True)

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

    def test_sparse_mla_tag_routing_preserves_capture_boundaries(self) -> None:
        for with_cache in (False, True):
            with self.subTest(with_cache=with_cache):
                model = _make_moe_model([1, 0])
                model.config.attn_config = SimpleNamespace(use_mla=True, is_sparse=True)
                layer_caches = [
                    [SimpleNamespace(tag=tag) for tag in tags]
                    for tags in (("indexer_kv", "default"), ("default", "indexer_kv"))
                ]
                if with_cache:
                    model.kv_cache = SimpleNamespace(
                        get_layer_cache_groups=layer_caches.__getitem__,
                        get_layer_cache=lambda layer, tag: next(
                            cache for cache in layer_caches[layer] if cache.tag == tag
                        ),
                    )
                fmha_impl = {"indexer_kv": object(), "default": object()}
                # Reuse the model across requests to check capture stays forward-local.
                for capture_hidden_states in (True, False):
                    with self.subTest(capture_hidden_states=capture_hidden_states):
                        inputs = _make_inputs([1, 2], capture_hidden_states)
                        with patch.object(
                            model.layers[0], "forward", wraps=model.layers[0].forward
                        ) as layer0, patch.object(
                            model.layers[1], "forward", wraps=model.layers[1].forward
                        ) as layer1, patch.object(
                            model.norm, "forward", wraps=model.norm.forward
                        ) as norm:
                            outputs = model.forward(inputs, fmha_impl).hidden_states

                        for layer_idx, layer in enumerate((layer0, layer1)):
                            layer.assert_called_once()
                            routed_impls = layer.call_args.args[2]
                            self.assertEqual(set(routed_impls), set(fmha_impl))
                            for tag, impl in fmha_impl.items():
                                self.assertIs(routed_impls[tag], impl)
                            routed_caches = layer.call_args.kwargs["kv_cache"]
                            if with_cache:
                                self.assertEqual(set(routed_caches), set(fmha_impl))
                                for cache in layer_caches[layer_idx]:
                                    self.assertIs(routed_caches[cache.tag], cache)
                            else:
                                self.assertIsNone(routed_caches)
                        norm.assert_called_once()
                        embedded = model.embed_tokens(inputs.input_ids)
                        boundary0 = embedded + 11
                        boundary1 = embedded + 33
                        expected = boundary1 * 10
                        if capture_hidden_states:
                            expected = torch.cat(
                                (boundary1, boundary0, expected), dim=-1
                            )
                        torch.testing.assert_close(outputs, expected)
                        self.assertEqual(
                            tuple(outputs.shape),
                            (2, 6 if capture_hidden_states else 2),
                        )

    def test_sparse_mla_capture_rejects_invalid_fmha_tags(self) -> None:
        model = _make_moe_model([1, 0])
        model.config.attn_config = SimpleNamespace(use_mla=True, is_sparse=True)
        inputs = _make_inputs([1, 2], capture_hidden_states=True)
        for fmha_impl in (
            object(),
            {"default": object()},
            {"default": object(), "wrong": object()},
            {"default": object(), "indexer_kv": object(), "extra": object()},
        ):
            with self.subTest(fmha_impl=fmha_impl), patch.object(
                model.layers[0], "forward", wraps=model.layers[0].forward
            ) as layer:
                with self.assertRaisesRegex(RuntimeError, "exactly FMHA tags"):
                    model.forward(inputs, fmha_impl)
                layer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
