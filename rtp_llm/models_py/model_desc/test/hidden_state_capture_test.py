import ast
import gc
import os
import subprocess
import sys
import textwrap
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.disaggregate_qwen3 import (
    BatchSplitInfo,
    Qwen3AttnModel,
    Qwen3GemmModel,
)
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeModel
from rtp_llm.models_py.model_desc.hidden_state_capture import (
    CaptureContext,
    is_fatal_capture_error,
)
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


def _make_inputs(
    input_ids: list[int],
    capture_hidden_states: bool,
    dtype: torch.dtype = torch.int64,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.tensor(input_ids, dtype=dtype),
        capture_hidden_states=capture_hidden_states,
    )


def _make_disaggregate_attention_model() -> Qwen3AttnModel:
    model = Qwen3AttnModel.__new__(Qwen3AttnModel)
    nn.Module.__init__(model)
    model.device = "cpu"
    model.ffn_service_rank = 2
    return model


def _make_disaggregate_control_receiver(
    attn_dp_rank: tuple[int, ...] = (0, 1), micro_batch_size: int = 2
) -> Qwen3GemmModel:
    model = Qwen3GemmModel.__new__(Qwen3GemmModel)
    nn.Module.__init__(model)
    model.attn_dp_rank = list(attn_dp_rank)
    model.micro_batch_size = micro_batch_size
    model.device = "cpu"
    model.capture_hidden_states = False
    return model


def _make_disaggregate_gemm_model(
    attn_dp_rank: tuple[int, ...] = (0,),
) -> Qwen3GemmModel:
    model = _make_disaggregate_control_receiver(attn_dp_rank, micro_batch_size=1)
    model.config = SimpleNamespace(hidden_state_capture_layer_ids=[1, 0])
    model.layer_num = 2
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
    return model


def _run_disaggregate_final_output(
    model: Qwen3GemmModel, input_ids: torch.Tensor, capture_hidden_states: bool
) -> list[torch.Tensor]:
    sent = []
    object.__setattr__(
        model,
        "send_to_attention",
        lambda tensor, micro_batch_sizes, captured=None, error=None, **kwargs: sent.append(
            (captured if captured is not None else tensor).clone()
        ),
    )
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
    return sent


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


def _forward_kimi(model: KimiLinearModel, inputs: SimpleNamespace):
    attention_inputs = SimpleNamespace(is_prefill=False, is_target_verify=False)
    with (
        patch(
            "rtp_llm.models_py.model_desc.kimi_linear.get_primary_attention_inputs",
            return_value=attention_inputs,
        ),
        patch(
            "rtp_llm.models_py.model_desc.kimi_linear.select_attention_inputs_for_layer",
            return_value=attention_inputs,
        ),
    ):
        return model.forward(inputs, fmha_impl=object())


class HiddenStateCaptureTest(unittest.TestCase):

    def test_capture_profiler_off_allocates_no_ranges(self) -> None:
        class UnformattableLayerId(int):
            def __format__(self, spec):
                raise AssertionError("profiler-off path formatted a dynamic range")

        model = _CaptureModel([2, 0])
        hidden = torch.tensor([[1.0, 2.0]])
        with (
            patch.object(torch.autograd, "_profiler_enabled", return_value=False),
            patch.object(torch._C._profiler, "_RecordFunctionFast") as record_range,
        ):
            capture = model.capture_context(True)
            self.assertIs(capture._profile_noop, _NOOP_RECORD_FUNCTION_RANGE)
            for layer_id in range(3):
                capture.capture_layer(UnformattableLayerId(layer_id), hidden)
            output = capture.finalize(hidden).captured_hidden_states
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
                with (
                    patch.object(torch._C, "_profiler", profiler, create=True),
                    patch.object(
                        torch.autograd,
                        "_profiler_enabled",
                        return_value=True,
                        create=True,
                    ),
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
                    output = capture.finalize(
                        hidden + 10, residual
                    ).captured_hidden_states
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
        with (
            patch.object(torch._C, "_profiler", profiler, create=True),
            patch.object(
                torch.autograd, "_profiler_enabled", return_value=True, create=True
            ) as profiler_enabled,
        ):
            capture = plan.for_forward(True)
            self.assertIs(capture._profile_enabled, True)
            self.assertIs(capture._record_function_fast, record_range)
            profiler_enabled.assert_called_once_with()
            profiler_enabled.side_effect = AssertionError(
                "profiler state was rechecked"
            )
            with (
                disable_record_function_ranges(),
                patch.object(
                    profiler,
                    "_RecordFunctionFast",
                    side_effect=AssertionError("range factory was not cached"),
                ),
            ):
                capture.capture_layer(0, hidden)
                output = capture.finalize(hidden).captured_hidden_states
        self.assertEqual(
            [call.args[0] for call in record_range.call_args_list],
            ["hidden_state_capture.layer(layer=0)", "hidden_state_capture.pack"],
        )
        torch.testing.assert_close(output, torch.cat([hidden, hidden], dim=-1))

    def test_capture_import_and_forward_do_not_import_fused_moe(self) -> None:
        source = textwrap.dedent("""\
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
                output = capture.finalize(hidden + 10, residual).captured_hidden_states

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
            """)
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
            output = capture.finalize(hidden, residual).captured_hidden_states
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
            layer_capture = model.capture_context(True)
            with patch.object(
                layer_capture,
                "_canonical_layer",
                side_effect=RuntimeError("layer failure"),
            ):
                layer_capture.capture_layer(0, hidden)
            layer_capture.capture_layer(0, hidden)
            layer_output = layer_capture.finalize(hidden)

            pack_model = _CaptureModel([0])
            pack_capture = pack_model.capture_context(True)
            pack_capture.capture_layer(0, torch.ones(2, 2))
            pack_output = pack_capture.finalize(hidden)
            self.assertEqual(pack_model.final_hook_calls, 1)
            repeated_pack_output = pack_capture.finalize(hidden)
            self.assertEqual(pack_model.final_hook_calls, 2)

            capture = model.capture_context(True)
            capture.capture_layer(0, hidden)
            output = capture.finalize(hidden)
        events = [
            e for e in prof.events() if e.name.startswith("hidden_state_capture.")
        ]
        self.assertEqual(
            [e.name for e in events],
            [
                "hidden_state_capture.layer(layer=0)",
                "hidden_state_capture.layer(layer=0)",
                "hidden_state_capture.pack",
                "hidden_state_capture.layer(layer=0)",
                "hidden_state_capture.pack",
            ],
        )
        self.assertTrue(all(e.cpu_parent is None for e in events))
        for failed_output, pattern in (
            (layer_output, r"^RuntimeError: layer failure$"),
            (pack_output, r"^RuntimeError: Sizes of tensors must match"),
            (repeated_pack_output, r"^RuntimeError: Sizes of tensors must match"),
        ):
            self.assertEqual(failed_output.hidden_states.data_ptr(), hidden.data_ptr())
            self.assertIsNone(failed_output.captured_hidden_states)
            self.assertRegex(failed_output.capture_error, pattern)
        torch.testing.assert_close(output.hidden_states, hidden)
        torch.testing.assert_close(
            output.captured_hidden_states, torch.cat([hidden, hidden], dim=-1)
        )
        self.assertIsNone(output.capture_error)

    def test_capture_pack_range_exit_failure_discards_payload(self) -> None:
        hidden = torch.ones(1, 2)
        capture = _CaptureModel([0]).capture_context(True)
        capture.capture_layer(0, hidden)
        record_range = MagicMock()
        record_range.__exit__.side_effect = RuntimeError("pack range close failure")
        with (
            patch.object(capture, "_record_function_fast", return_value=record_range),
            patch.object(torch, "cat", wraps=torch.cat) as pack,
        ):
            output = capture.finalize(hidden)
        pack.assert_called_once()
        record_range.__exit__.assert_called_once_with(None, None, None)
        self.assertEqual(output.hidden_states.data_ptr(), hidden.data_ptr())
        self.assertIsNone(output.captured_hidden_states)
        self.assertEqual(output.capture_error, "RuntimeError: pack range close failure")

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
        outputs = capture.finalize(final_hidden_states)
        torch.testing.assert_close(outputs.hidden_states, final_hidden_states)
        torch.testing.assert_close(
            outputs.captured_hidden_states,
            torch.cat((layer2, layer0, final_hidden_states), dim=-1),
        )
        self.assertEqual(outputs.captured_hidden_states.shape[-1], 6)

    def test_capture_layer_error_discards_partial_capture_and_is_forward_local(
        self,
    ) -> None:
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        for error in (ValueError("layer shape failure"), RuntimeError("layer failure")):
            with self.subTest(error=type(error).__name__):
                model = _CaptureModel([1, 0])
                capture = model.capture_context(True)
                capture.capture_layer(0, hidden, residual)
                with patch.object(
                    capture, "_canonical_layer", side_effect=error
                ) as layer_hook:
                    capture.capture_layer(1, hidden, residual)
                    capture.capture_layer(0, hidden, residual)
                    capture.capture_layer(1, hidden, residual)
                layer_hook.assert_called_once_with(hidden, residual)
                self.assertFalse(capture.wants_layer(0))
                self.assertFalse(capture.wants_layer(1))
                output = capture.finalize(hidden, residual)
                torch.testing.assert_close(output.hidden_states, hidden + residual)
                self.assertIsNone(output.captured_hidden_states)
                self.assertEqual(
                    output.capture_error, f"{type(error).__name__}: {error}"
                )
                self.assertIsNone(capture._captured_hidden_states)
                self.assertEqual(model.final_hook_calls, 1)

                next_capture = model.capture_context(True)
                for layer_id in (0, 1):
                    next_capture.capture_layer(layer_id, hidden, residual)
                next_output = next_capture.finalize(hidden, residual)
                torch.testing.assert_close(next_output.hidden_states, hidden + residual)
                torch.testing.assert_close(
                    next_output.captured_hidden_states,
                    torch.cat([hidden + residual] * 3, dim=-1),
                )
                self.assertIsNone(next_output.capture_error)

    def test_missing_configured_layer_only_fails_capture(self) -> None:
        model = _CaptureModel([2, 0])
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        capture = model.capture_context(True)
        capture.capture_layer(0, hidden, residual)
        with patch.object(torch, "cat") as pack:
            output = capture.finalize(hidden, residual)
        pack.assert_not_called()
        torch.testing.assert_close(output.hidden_states, hidden + residual)
        self.assertIsNone(output.captured_hidden_states)
        self.assertEqual(
            output.capture_error,
            "RuntimeError: hidden-state capture did not visit configured layers [2]",
        )
        self.assertEqual(model.final_hook_calls, 1)

    def test_capture_fatal_layer_and_pack_errors_propagate(self) -> None:
        hidden = torch.ones(1, 2)
        errors = [
            MemoryError("host allocation failed"),
            torch.OutOfMemoryError("device allocation failed"),
            RuntimeError("CUDA error: invalid configuration argument"),
            RuntimeError("CUDA driver error: invalid device context"),
            RuntimeError("CUDA runtime error: unknown error"),
            RuntimeError("HIP error: invalid device function"),
            RuntimeError("HIP runtime error: unknown error"),
            RuntimeError("out of memory"),
            RuntimeError("OutOfMemoryError: allocation failed"),
            RuntimeError("CuBlAs_Status_Alloc_Failed"),
            RuntimeError("CUSPARSE_STATUS_ALLOC_FAILED"),
            RuntimeError("CUDNN_STATUS_ALLOC_FAILED"),
            RuntimeError("HIPBLAS_STATUS_ALLOC_FAILED"),
            RuntimeError("rocblas_status_memory_error"),
            RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED"),
            RuntimeError("CUSPARSE_STATUS_EXECUTION_FAILED"),
            RuntimeError("CUDNN_STATUS_EXECUTION_FAILED"),
            RuntimeError("HIPBLAS_STATUS_EXECUTION_FAILED"),
            RuntimeError("rocblas_status_internal_error"),
            RuntimeError("illegal memory access"),
            RuntimeError("ILLEGAL ADDRESS"),
            RuntimeError("invalid device pointer"),
            RuntimeError("misaligned address"),
            RuntimeError("device-side assert"),
            RuntimeError("device side assert"),
            RuntimeError("launch failure"),
            RuntimeError("launch failed"),
            RuntimeError("launch timed out"),
            RuntimeError("launch_failure"),
            RuntimeError("launch_failed"),
            RuntimeError("cudaErrorIllegalAddress"),
            RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS"),
            RuntimeError("hipErrorIllegalAddress"),
            RuntimeError("cudaErrorMemoryAllocation"),
            RuntimeError("hipErrorOutOfMemory"),
            RuntimeError("NCCL error: watchdog failure"),
            RuntimeError("Collective failed"),
            RuntimeError("Collective failure"),
            RuntimeError("Collective operation timeout"),
            RuntimeError("Collective timed out"),
            KeyboardInterrupt("cancelled"),
            SystemExit("shutdown"),
        ]
        for error_type in (
            getattr(torch, "AcceleratorError", None),
            getattr(getattr(torch, "distributed", None), "DistError", None),
        ):
            if error_type is not None:
                errors.append(error_type("device execution failed"))
        for stage in ("layer", "pack"):
            for error in errors:
                with self.subTest(stage=stage, error=repr(error)):
                    capture = _CaptureModel([0]).capture_context(True)
                    if stage == "layer":
                        target, attribute = capture, "_canonical_layer"
                    else:
                        capture.capture_layer(0, hidden)
                        target, attribute = torch, "cat"
                    with patch.object(target, attribute, side_effect=error):
                        with self.assertRaises(type(error)) as raised:
                            if stage == "layer":
                                capture.capture_layer(0, hidden)
                            else:
                                capture.finalize(hidden)
                    self.assertIs(raised.exception, error)
                    self.assertIsNone(capture._capture_error)

    def test_capture_fatal_classifier_prefers_types_to_message_fallback(self) -> None:
        class UnformattableFatal(MemoryError):
            def __str__(self):
                raise AssertionError("typed fatal error must not format its message")

        self.assertTrue(is_fatal_capture_error(UnformattableFatal()))
        for error in (
            ValueError("capture shape mismatch"),
            RuntimeError("capture pack failure"),
            RuntimeError(
                "Expected all tensors to be on the same device, cpu and cuda:0"
            ),
            RuntimeError("unsupported dtype on CUDA"),
            RuntimeError("unsupported layout on HIP"),
            RuntimeError("CUBLAS_STATUS_NOT_SUPPORTED"),
            RuntimeError("CUSPARSE_STATUS_NOT_SUPPORTED"),
            RuntimeError("CUDNN_STATUS_NOT_SUPPORTED"),
            RuntimeError("HIPBLAS_STATUS_NOT_SUPPORTED"),
            RuntimeError("rocblas_status_not_implemented"),
        ):
            with self.subTest(error=repr(error)):
                self.assertFalse(is_fatal_capture_error(error))

    def test_canonical_final_errors_are_never_capture_errors(self) -> None:
        hidden = torch.ones(1, 2)
        residual = torch.full_like(hidden, 3)
        for requested, layer_error in ((False, False), (True, False), (True, True)):
            with self.subTest(requested=requested, layer_error=layer_error):
                error = ValueError("final norm failure")
                final_hook = Mock(side_effect=error)
                layer_hook = Mock(return_value=hidden + residual)
                if layer_error:
                    layer_hook.side_effect = RuntimeError("layer failure")
                plan = CaptureContext.configured([0], layer_hook, final_hook)
                capture = plan.for_forward(requested)
                capture.capture_layer(0, hidden, residual)
                with self.assertRaisesRegex(ValueError, "final norm failure") as raised:
                    capture.finalize(hidden, residual)
                self.assertIs(raised.exception, error)
                final_hook.assert_called_once_with(hidden, residual)

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
        self.assertIsNone(first._capture_sources)
        first.capture_layer(0, torch.tensor([[3.0, 4.0]]))
        self.assertEqual(model.layer_hook_calls, [])

        final_hidden_states = torch.tensor([[1.0, 2.0]])
        output = first.finalize(final_hidden_states)
        self.assertEqual(
            output.hidden_states.data_ptr(), final_hidden_states.data_ptr()
        )
        self.assertEqual(model.final_hook_calls, 1)
        self.assertIsNone(output.captured_hidden_states)
        self.assertIsNone(output.capture_error)

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
        torch.testing.assert_close(output.hidden_states, hidden_states + residual)
        torch.testing.assert_close(
            output.captured_hidden_states,
            torch.cat((hidden_states + residual, hidden_states + residual), dim=-1),
        )

    def test_active_context_retains_sources_until_forward_ends(self) -> None:
        model = _CaptureModel([0])
        capture = model.capture_context(True)
        sibling = model.capture_context(True)
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        sources = [weakref.ref(hidden), weakref.ref(residual)]
        capture.capture_layer(0, hidden, residual)
        del hidden, residual

        self.assertTrue(all(source() is not None for source in sources))
        self.assertIsNone(model._capture_context._capture_sources)
        self.assertEqual(sibling._capture_sources, [])
        final_hidden = torch.tensor([[5.0, 6.0]])
        final_residual = torch.tensor([[7.0, 8.0]])
        sources.extend((weakref.ref(final_hidden), weakref.ref(final_residual)))
        output = capture.finalize(final_hidden, final_residual)
        del final_hidden, final_residual

        torch.testing.assert_close(output.hidden_states, torch.tensor([[12.0, 14.0]]))
        torch.testing.assert_close(
            output.captured_hidden_states, torch.tensor([[4.0, 6.0, 12.0, 14.0]])
        )
        self.assertIsNone(output.capture_error)
        self.assertTrue(all(source() is not None for source in sources))
        del capture
        gc.collect()
        self.assertTrue(all(source() is None for source in sources))

    def test_capture_failure_retains_sources_only_for_its_forward(self) -> None:
        for stage in ("layer", "pack"):
            with self.subTest(stage=stage):
                model = _CaptureModel([1, 0])
                capture = model.capture_context(True)
                hidden = torch.tensor([[1.0, 2.0]])
                residual = torch.tensor([[3.0, 4.0]])
                sources = [weakref.ref(hidden), weakref.ref(residual)]
                capture.capture_layer(0, hidden, residual)
                del hidden, residual
                if stage == "layer":
                    with patch.object(
                        capture,
                        "_canonical_layer",
                        side_effect=ValueError("source keepalive failure"),
                    ):
                        capture.capture_layer(1, torch.ones(1, 2))
                    expected_error = r"^ValueError: source keepalive failure$"
                else:
                    capture.capture_layer(1, torch.ones(2, 2))
                    expected_error = r"^RuntimeError: Sizes of tensors must match"

                final_hidden = torch.full((1, 2), 9.0)
                output = capture.finalize(final_hidden)
                self.assertEqual(
                    output.hidden_states.data_ptr(), final_hidden.data_ptr()
                )
                self.assertIsNone(output.captured_hidden_states)
                self.assertRegex(output.capture_error, expected_error)
                self.assertIsNone(capture._captured_hidden_states)
                self.assertTrue(all(source() is not None for source in sources))
                self.assertEqual(model.capture_context(True)._capture_sources, [])
                self.assertIsNone(model.capture_context(False)._capture_sources)
                del capture
                gc.collect()
                self.assertTrue(all(source() is None for source in sources))

    def test_qwen_capture_uses_configured_order_and_post_final_norm(self) -> None:
        model = _make_qwen_model([2, 0])
        inputs = _make_inputs([1, 2], capture_hidden_states=True)

        outputs = model.forward(inputs, fmha_impl=object())
        embedded = model.embed_tokens(inputs.input_ids)
        layer0 = embedded + 1
        layer2 = layer0 + 2 + 3
        expected_capture = torch.cat((layer2, layer0, layer2 * 10), dim=-1)

        torch.testing.assert_close(outputs.hidden_states, layer2 * 10)
        torch.testing.assert_close(outputs.captured_hidden_states, expected_capture)
        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))

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

        embedded = model.embed_tokens(inputs.input_ids)
        layer0_hidden = embedded + 1
        layer0_residual = torch.full_like(embedded, 10)
        layer0 = layer0_hidden + layer0_residual
        layer1_hidden = layer0_hidden + 2
        layer1_residual = layer0_residual + 20
        layer1 = layer1_hidden + layer1_residual
        expected_capture = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs_with_aux.hidden_states, layer1 * 10)
        torch.testing.assert_close(
            outputs_with_aux.captured_hidden_states, expected_capture
        )
        self.assertEqual(tuple(outputs_with_aux.hidden_states.shape), (2, 2))

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
                torch.testing.assert_close(outputs.hidden_states, layer1 * 10)
                if enabled:
                    torch.testing.assert_close(
                        outputs.captured_hidden_states, expected_capture
                    )
                else:
                    self.assertIsNone(outputs.captured_hidden_states)
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
        expected_capture = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs.hidden_states, layer1 * 10)
        torch.testing.assert_close(outputs.captured_hidden_states, expected_capture)
        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))

    def test_kimi_linear_capture_disabled_keeps_ordinary_width(self) -> None:
        model = _make_kimi_linear_model([1, 0])
        inputs = _make_inputs([1, 2], capture_hidden_states=False)

        outputs = _forward_kimi(model, inputs)
        embedded = model.embed_tokens(inputs.input_ids)
        expected = (embedded + 1 + 2 + 30) * 10

        torch.testing.assert_close(outputs.hidden_states, expected)
        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))
        self.assertIsNone(outputs.captured_hidden_states)
        self.assertIsNone(outputs.capture_error)

    def test_qwen3vl_capture_is_after_deepstack_and_in_configured_order(self) -> None:
        model = _make_qwen3vl_model([1, 0])
        inputs = _make_vl_inputs(capture_hidden_states=True)

        outputs = model.forward(inputs, fmha_impl=object())
        embedded = model.embed_tokens(inputs.input_ids, None, None, None)
        deepstack0, deepstack1 = _deepstack_offsets(embedded)
        layer0 = embedded + 1 + deepstack0
        layer1 = layer0 + 2 + deepstack1
        expected_capture = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs.hidden_states, layer1 * 10)
        torch.testing.assert_close(outputs.captured_hidden_states, expected_capture)
        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))

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

        outputs = model.forward(inputs, fmha_impl=object())
        embedded = model.embed_tokens(inputs.input_ids, None, None, None)
        deepstack0, deepstack1 = _deepstack_offsets(embedded)
        layer0_hidden = embedded + 1 + deepstack0
        layer0_residual = torch.full_like(embedded, 10)
        layer0 = layer0_hidden + layer0_residual
        layer1_hidden = layer0_hidden + 2 + deepstack1
        layer1_residual = layer0_residual + 20
        layer1 = layer1_hidden + layer1_residual
        expected_capture = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs.hidden_states, layer1 * 10)
        torch.testing.assert_close(outputs.captured_hidden_states, expected_capture)
        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))

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
                model = _make_disaggregate_attention_model()
                inputs = [
                    _make_inputs(ids, capture_hidden_states, dtype=torch.int32)
                    for ids in ([1, 2], [3])
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
        model = _make_disaggregate_attention_model()
        inputs = [
            _make_inputs(ids, capture_hidden_states=False, dtype=torch.int32)
            for ids in ([], [10, 11], [], [20])
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
        model = _make_disaggregate_attention_model()
        inputs = [
            _make_inputs([1, 2], capture_hidden_states=False, dtype=torch.int32),
            _make_inputs([3], capture_hidden_states=True, dtype=torch.int32),
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
        model = _make_disaggregate_attention_model()
        model.config = SimpleNamespace(
            hidden_size=2,
            hidden_state_capture_layer_ids=[1, 0],
            compute_dtype=torch.float16,
        )

        for capture_hidden_states, expected_width in ((False, 2), (True, 6)):
            with self.subTest(capture_hidden_states=capture_hidden_states):
                with patch(
                    "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
                    side_effect=lambda t, *args: t.zero_(),
                ) as recv_mock:
                    output = model.recv_final_from_ffn_service(
                        token_num=3, capture_hidden_states=capture_hidden_states
                    )

                self.assertEqual(tuple(output.hidden_states.shape), (3, 2))
                self.assertIsNone(output.capture_error)
                if capture_hidden_states:
                    self.assertEqual(
                        tuple(output.captured_hidden_states.shape), (3, expected_width)
                    )
                else:
                    self.assertIsNone(output.captured_hidden_states)
                self.assertEqual(recv_mock.call_count, 1 + int(capture_hidden_states))

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

                attn_model = _make_disaggregate_attention_model()
                attn_model.config = SimpleNamespace(
                    attn_config=SimpleNamespace(
                        head_num=2, kv_head_num=1, size_per_head=4
                    ),
                    hidden_size=8,
                    hidden_state_capture_layer_ids=[1, 0],
                    compute_dtype=compute_dtype,
                )

                with patch(
                    "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
                    side_effect=lambda t, *args: t.zero_(),
                ) as recv_mock:
                    from_attention = gemm_model.recv_from_attention([1, 2], 3)
                    from_ffn = attn_model.recv_from_ffn_service(3)
                    final_from_ffn = attn_model.recv_final_from_ffn_service(3, True)

                self.assertEqual(from_attention.dtype, compute_dtype)
                self.assertEqual(from_ffn.dtype, compute_dtype)
                self.assertEqual(final_from_ffn.hidden_states.dtype, compute_dtype)
                self.assertEqual(
                    final_from_ffn.captured_hidden_states.dtype, compute_dtype
                )
                self.assertEqual(recv_mock.call_count, 5)

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

    def test_disaggregate_send_payload_layout_and_storage(self) -> None:
        model = Qwen3GemmModel.__new__(Qwen3GemmModel)
        nn.Module.__init__(model)
        model.attn_dp_rank = [0, 1]
        for capture_flags in (
            (False, True),
            (True, False),
            (True, True),
            (False, False),
        ):
            for sizes in (
                (2, 3),
                (3, 2),
                (1, 1),
                (1, 2),
                (2, 1),
                (0, 2),
                (2, 0),
                (0, 0),
            ):
                with self.subTest(capture_flags=capture_flags, sizes=sizes):
                    model.capture_hidden_states_by_attn_rank = list(capture_flags)
                    total_tokens = sum(sizes)
                    packed = torch.arange(
                        total_tokens * 6, dtype=torch.float32
                    ).reshape(total_tokens, 6)
                    captured = packed if any(capture_flags) else None
                    final_hidden = packed[:, -2:]
                    if captured is None:
                        final_hidden = final_hidden.contiguous()
                    offset = 0
                    expected_payloads = []
                    for capture, size in zip(capture_flags, sizes):
                        source = packed if capture else final_hidden
                        expected_payloads.append(source[offset : offset + size])
                        offset += size
                    sent = []

                    def check_send(tensor, rank, group):
                        expected = expected_payloads[len(sent)]
                        self.assertEqual(rank, model.attn_dp_rank[len(sent)])
                        self.assertTrue(
                            torch.ops.aten.is_non_overlapping_and_dense.default(tensor)
                        )
                        self.assertTrue(tensor.is_contiguous())
                        torch.testing.assert_close(tensor, expected)
                        if expected.is_contiguous():
                            self.assertEqual(
                                tensor.untyped_storage().data_ptr(),
                                expected.untyped_storage().data_ptr(),
                            )
                            self.assertEqual(
                                tensor.storage_offset(), expected.storage_offset()
                            )
                        else:
                            self.assertNotEqual(
                                tensor.untyped_storage().data_ptr(),
                                expected.untyped_storage().data_ptr(),
                            )
                            self.assertEqual(
                                tensor.untyped_storage().nbytes(),
                                tensor.numel() * tensor.element_size(),
                            )
                        sent.append(tensor)

                    with patch(
                        "rtp_llm.models_py.model_desc.disaggregate_qwen3.send",
                        side_effect=check_send,
                    ):
                        model.send_to_attention(final_hidden, list(sizes), captured)
                    self.assertEqual(len(sent), len(sizes))

    def test_disaggregate_ffn_routes_per_rank_final_hidden_states(self) -> None:
        model = _make_disaggregate_gemm_model(attn_dp_rank=(0, 1))
        model.capture_hidden_states_by_attn_rank = [False, False]
        sent = []

        def capture_send(tensor, rank, group):
            self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(tensor))
            sent.append((rank, tensor))

        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
            side_effect=_copy_recv_payloads(
                [
                    torch.tensor([2, 0], dtype=torch.int64),
                    torch.tensor([2, 1], dtype=torch.int64),
                    torch.tensor([1, 2], dtype=torch.int32),
                    torch.tensor([3, 4], dtype=torch.int32),
                ]
            ),
        ), patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.send",
            side_effect=capture_send,
        ):
            model.forward_micro_batch([])

        self.assertTrue(model.capture_hidden_states)
        self.assertEqual(model.capture_hidden_states_by_attn_rank, [False, True])
        self.assertEqual(len(sent), 7)
        false_rank, false_payload = sent[-3]
        header_rank, header = sent[-2]
        self.assertEqual(header_rank, 1)
        self.assertEqual(header.item(), 0)
        true_rank, true_payload = sent[-1]
        self.assertEqual((false_rank, true_rank), (0, 1))

        embedded = _Embedding()(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
        layer0 = embedded + 1
        layer1 = layer0 + 2
        expected_false = (layer1 * 10)[:2]
        expected_true = torch.cat((layer1, layer0, layer1 * 10), dim=-1)[2:]
        torch.testing.assert_close(false_payload, expected_false)
        torch.testing.assert_close(true_payload, expected_true)

        attention_model = _make_disaggregate_attention_model()
        attention_model.config = SimpleNamespace(
            hidden_size=2,
            hidden_state_capture_layer_ids=[1, 0],
            compute_dtype=torch.float16,
        )
        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
            side_effect=_copy_recv_payloads([false_payload, header, true_payload]),
        ):
            false_output = attention_model.recv_final_from_ffn_service(2, False)
            true_output = attention_model.recv_final_from_ffn_service(2, True)

        self.assertEqual(tuple(false_output.hidden_states.shape), (2, 2))
        self.assertEqual(tuple(true_output.hidden_states.shape), (2, 2))
        self.assertEqual(tuple(true_output.captured_hidden_states.shape), (2, 6))
        torch.testing.assert_close(false_output.hidden_states, expected_false.half())
        torch.testing.assert_close(
            true_output.hidden_states, expected_true[:, -2:].half()
        )
        torch.testing.assert_close(
            true_output.captured_hidden_states, expected_true.half()
        )

    def test_disaggregate_capture_failure_transports_diagnostic_and_canonical(self):
        gemm = _make_disaggregate_gemm_model(attn_dp_rank=(0, 1))
        gemm.capture_hidden_states_by_attn_rank = [False, True]
        gemm._capture_context._canonical_layer = Mock(
            side_effect=ValueError("snapshot")
        )
        sent = []
        with (
            patch(
                "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
                side_effect=_copy_recv_payloads(
                    [
                        torch.tensor([1, 0]),
                        torch.tensor([1, 1]),
                        torch.tensor([1], dtype=torch.int32),
                        torch.tensor([2], dtype=torch.int32),
                    ]
                ),
            ),
            patch(
                "rtp_llm.models_py.model_desc.disaggregate_qwen3.send",
                side_effect=lambda t, rank, group: sent.append((rank, t.clone())),
            ),
        ):
            gemm.forward_micro_batch([])
        attention = _make_disaggregate_attention_model()
        attention.config = SimpleNamespace(
            hidden_size=2,
            hidden_state_capture_layer_ids=[1, 0],
            compute_dtype=torch.float32,
        )
        # Two non-final sends per lane precede canonical/header/error/canonical.
        self.assertEqual(len(sent), 8)
        with patch(
            "rtp_llm.models_py.model_desc.disaggregate_qwen3.recv",
            side_effect=_copy_recv_payloads([t for rank, t in sent[-4:]]),
        ):
            ordinary = attention.recv_final_from_ffn_service(1, False)
            failed = attention.recv_final_from_ffn_service(1, True)
        self.assertIsNone(ordinary.capture_error)
        self.assertIsNone(failed.captured_hidden_states)
        self.assertEqual(failed.capture_error, "ValueError: snapshot")
        torch.testing.assert_close(
            failed.hidden_states, (_Embedding()(torch.tensor([2])) + 3) * 10
        )

    def test_disaggregate_final_hidden_numeric_parity_capture_on_off(self) -> None:
        outputs = []
        for capture_hidden_states in (False, True):
            with self.subTest(capture_hidden_states=capture_hidden_states):
                model = _make_disaggregate_gemm_model()
                model.capture_hidden_states_by_attn_rank = [False]
                input_ids = torch.tensor([1, 2], dtype=torch.int64)
                sent = _run_disaggregate_final_output(
                    model, input_ids, capture_hidden_states
                )
                outputs.append(sent[-1])
        capture_off, capture_on = outputs
        capture_on_final = capture_on[..., -capture_off.shape[-1] :]

        embedded = _Embedding()(torch.tensor([1, 2], dtype=torch.int64))
        expected_once_normalized = (embedded + 1 + 2) * 10
        torch.testing.assert_close(capture_off, expected_once_normalized)
        torch.testing.assert_close(capture_on_final, expected_once_normalized)
        torch.testing.assert_close(capture_on_final, capture_off)

    def test_disaggregate_qwen_packs_complete_service_side_boundaries(self) -> None:
        model = _make_disaggregate_gemm_model()
        input_ids = torch.tensor([1, 2], dtype=torch.int64)
        sent = _run_disaggregate_final_output(
            model, input_ids, capture_hidden_states=True
        )

        self.assertTrue(model.capture_hidden_states)
        self.assertEqual([tensor.shape[-1] for tensor in sent], [2, 2, 6])
        embedded = _Embedding()(input_ids)
        layer0 = embedded + 1
        layer1 = layer0 + 2
        expected = torch.cat((layer1, layer0, layer1 * 10), dim=-1)
        torch.testing.assert_close(sent[-1], expected)

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

        outputs = model.forward(inputs, fmha_impl=object())
        embedded = model.embed_tokens(inputs.input_ids)
        layer0_hidden = embedded + 1
        layer0_residual = torch.full_like(embedded, 10)
        layer0 = layer0_hidden + layer0_residual
        layer1_hidden = layer0_hidden + 2
        layer1_residual = layer0_residual + 20
        layer1 = layer1_hidden + layer1_residual
        expected_capture = torch.cat((layer1, layer0, layer1 * 10), dim=-1)

        torch.testing.assert_close(outputs.hidden_states, layer1 * 10)
        torch.testing.assert_close(outputs.captured_hidden_states, expected_capture)
        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))

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
                        get_layer_cache_groups=layer_caches.__getitem__
                    )
                fmha_impl = {"indexer_kv": object(), "default": object()}
                # Reuse the model across requests to check capture stays forward-local.
                for capture_hidden_states in (True, False):
                    with self.subTest(capture_hidden_states=capture_hidden_states):
                        inputs = _make_inputs([1, 2], capture_hidden_states)
                        with (
                            patch.object(
                                model.layers[0],
                                "forward",
                                wraps=model.layers[0].forward,
                            ) as layer0,
                            patch.object(
                                model.layers[1],
                                "forward",
                                wraps=model.layers[1].forward,
                            ) as layer1,
                            patch.object(
                                model.norm, "forward", wraps=model.norm.forward
                            ) as norm,
                        ):
                            outputs = model.forward(inputs, fmha_impl)

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
                        torch.testing.assert_close(outputs.hidden_states, expected)
                        if capture_hidden_states:
                            torch.testing.assert_close(
                                outputs.captured_hidden_states,
                                torch.cat((boundary1, boundary0, expected), dim=-1),
                            )
                        else:
                            self.assertIsNone(outputs.captured_hidden_states)
                        self.assertEqual(tuple(outputs.hidden_states.shape), (2, 2))

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
