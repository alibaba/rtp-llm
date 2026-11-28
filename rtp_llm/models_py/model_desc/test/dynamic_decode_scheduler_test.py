import importlib.metadata
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.cuda_graph import CudaGraphSelectionMode
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench import (
    _graph_timing_events,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector import (
    benchmark_sequence_cap,
    dynamic_backend_allowed,
    instantiate_decode_backend,
    select_decode_backend,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.selector import select_stable
from rtp_llm.ops.compute_ops import PyAttentionInputs


class DecodeSchedulerTest(unittest.TestCase):
    def test_native_modules_come_from_declared_runfiles(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        for name in ("libth_transformer_config", "librtp_compute_ops"):
            self.assertTrue(
                os.path.samefile(sys.modules[name].__file__, root / (name + ".so"))
            )
        # gpu_lock stages the declared torch wheel into stable JIT paths, so only
        # provenance is asserted here; library content belongs to the build.
        declared = importlib.metadata.distribution("torch")
        init = declared.locate_file("torch/__init__.py")
        self.assertTrue(str(init).startswith(os.environ["TEST_SRCDIR"] + os.sep))

    def test_dynamic_backend_arch_filter(self):
        self.assertTrue(dynamic_backend_allowed("XQADecodeImpl", 9))
        self.assertFalse(dynamic_backend_allowed("XQADecodeImpl", 10))
        self.assertTrue(dynamic_backend_allowed("FlashInferTRTLLMDecodeImpl", 10))
        self.assertTrue(dynamic_backend_allowed("PyFlashinferDecodeImpl", 10))

    def test_threshold_and_registry_tiebreak(self):
        self.assertEqual(select_stable({"A": [100], "B": [97]}, ["A", "B"], "A"), "A")
        self.assertEqual(
            select_stable({"A": [100], "B": [80], "C": [79]}, ["A", "B", "C"], "A"), "B"
        )

    def test_benchmark_cap_respects_physical_capacity_and_memory(self):
        self.assertEqual(benchmark_sequence_cap(2048, 2, 64, 8, 4096, 2**30), 255)
        self.assertLess(benchmark_sequence_cap(2048, 2, 64, 8, 4096, 0), 0)
        self.assertEqual(benchmark_sequence_cap(128, 2, 64, 8, 4096, 2**30), 128)

    def test_graph_timing_events_fall_back_when_external_is_unsupported(self):
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_bench"
        with patch(module + ".torch.cuda.Event", side_effect=TypeError):
            self.assertEqual(_graph_timing_events(), (None, None))


class DecodePlanTest(unittest.TestCase):
    def setUp(self):
        self.model = GptModelBase(
            SimpleNamespace(
                num_layers=1,
                vocab_size=10,
                max_seq_len=2048,
                capture_aux_hidden_layer_ids=[],
            ),
            SimpleNamespace(get_attn_tp_size=lambda: 1),
            object(),
            8,
        )
        from rtp_llm.models_py.modules.factory.attention.attn_factory import (
            DECODE_MHA_IMPS,
        )

        self.backend = DECODE_MHA_IMPS[-1]
        self.model.set_decode_backend_gate([self.backend.__name__])
        self.inputs = PyAttentionInputs()
        self.inputs.input_lengths = torch.ones(2, dtype=torch.int32)
        self.inputs.sequence_lengths = torch.zeros(2, dtype=torch.int32)
        self.inputs.is_prefill = False

    def test_enable_requires_gate(self):
        self.model.set_decode_backend_gate(None)
        with self.assertRaisesRegex(RuntimeError, "before precision"):
            self.model.configure_dynamic_decode(True)

    def test_disabled_dynamic_decode_does_not_parse_threshold_config(self):
        self.model.py_hw_kernel_config = SimpleNamespace(
            dyn_decode_gate_thresholds="invalid-json"
        )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder.parse_gate_thresholds",
            side_effect=AssertionError("disabled path parsed thresholds"),
        ):
            self.model.configure_dynamic_decode(False)
        self.assertFalse(self.model._dynamic_decode_enabled)

    def test_disabled_decode_graph_keeps_mainline_routing(self):
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"
        with patch(
            "rtp_llm.models_py.modules.AttnImplFactory.get_fmha_impl",
            return_value="static",
        ) as static, patch(module + ".select_decode_backend") as select, patch(
            module + ".instantiate_decode_backend"
        ) as instantiate:
            for explicit_disable in (False, True):
                if explicit_disable:
                    self.model.configure_dynamic_decode(False)
                self.assertEqual(
                    self.model.prepare_fmha_impl(
                        SimpleNamespace(attention_inputs=self.inputs),
                        True,
                        CudaGraphSelectionMode.DECODE_GRAPH,
                    ),
                    "static",
                )
        self.assertEqual(static.call_count, 2)
        select.assert_not_called()
        instantiate.assert_not_called()
        self.assertFalse(self.model.decode_backend_plan)

    def test_plan_selected_once_per_batch_and_applied(self):
        self.model.configure_dynamic_decode(True)
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"
        with patch(
            module + ".select_decode_backend", return_value=self.backend
        ) as select, patch(
            module + ".instantiate_decode_backend",
            return_value=SimpleNamespace(name="actual"),
        ) as instantiate:
            for _ in range(2):
                self.assertEqual(
                    self.model.prepare_fmha_impl(
                        SimpleNamespace(attention_inputs=self.inputs), True
                    ).name,
                    "actual",
                )
            self.inputs.input_lengths = torch.ones(4, dtype=torch.int32)
            self.model.prepare_fmha_impl(
                SimpleNamespace(attention_inputs=self.inputs), True
            )
            self.assertEqual(select.call_count, 2)
            self.assertEqual(instantiate.call_count, 3)
            self.assertEqual(set(self.model.decode_backend_plan), {2, 4})

    def test_selector_uses_fixed_five_percent_policy(self):
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"

        class A:
            support = staticmethod(lambda *args: True)
            support_parallelism_config = staticmethod(lambda *args: True)

        class B(A):
            pass

        self.model.config.getAttentionConfigs = lambda _: SimpleNamespace(
            max_seq_len=2048, kernel_tokens_per_block=64
        )
        self.model.parallelism_config = SimpleNamespace(
            get_attn_tp_size=lambda: 1, tp_size=1, tp_rank=0
        )
        cache = SimpleNamespace(kv_cache_base=torch.zeros(8, 64))
        self.model.kv_cache = SimpleNamespace(get_layer_cache_groups=lambda _: [cache])
        tensor = torch.tensor
        full = torch.full
        with patch(module + ".DECODE_MHA_IMPS", [A, B]), patch(
            module + ".get_sm", return_value=(9, 0)
        ), patch(
            module + "._implementation_allows_cuda_graph_selection_mode",
            return_value=True,
        ), patch(
            module + "._is_fmha_impl_disabled", return_value=False
        ), patch(
            module + ".torch.tensor",
            side_effect=lambda *a, **k: tensor(*a, **{**k, "device": "cpu"}),
        ), patch(
            module + ".torch.full",
            side_effect=lambda *a, **k: full(*a, **{**k, "device": "cpu"}),
        ), patch(
            module + ".torch.cuda.empty_cache"
        ), patch(
            module + ".torch.cuda.mem_get_info", return_value=(2**30, 2**30)
        ), patch(
            module + ".kv_grid", return_value=[64]
        ), patch(
            module + ".bench_backend_grid", side_effect=[[100], [97]]
        ), patch(
            module + "._fatal", side_effect=AssertionError("selector failed")
        ):
            self.assertIs(select_decode_backend(self.model, self.inputs, [A, B]), A)

    def test_headwise_with_gate_uses_static_graph_factory(self):
        self.model.configure_dynamic_decode(True)
        self.model.decode_backend_plan[2] = self.backend
        self.model.config.headwise_config = {}
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"
        with self.assertLogs(level="WARNING") as logs:
            self.model.configure_dynamic_decode(True)
        self.assertIn("does not qualify headwise attention", logs.output[0])
        self.assertFalse(self.model._dynamic_decode_enabled)
        self.assertFalse(self.model.decode_backend_plan)
        with patch(
            "rtp_llm.models_py.modules.AttnImplFactory.get_fmha_impl",
            return_value="static",
        ) as static, patch(module + ".select_decode_backend") as select, patch(
            module + ".instantiate_decode_backend"
        ) as instantiate:
            self.assertEqual(
                self.model.prepare_fmha_impl(
                    SimpleNamespace(attention_inputs=self.inputs),
                    True,
                    CudaGraphSelectionMode.DECODE_GRAPH,
                ),
                "static",
            )
        static.assert_called_once()
        select.assert_not_called()
        instantiate.assert_not_called()

    def test_long_kv_does_not_block_capture_selection(self):
        self.model.configure_dynamic_decode(True)
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"
        instance = SimpleNamespace(name="actual")
        with patch(
            module + ".select_decode_backend", return_value=self.backend
        ) as select, patch(
            module + ".instantiate_decode_backend",
            return_value=instance,
        ):
            # Both lengths are past the retired calibration caps; selection is
            # still per capture bucket, so each batch is planned once.
            for batch, sequence_length in ((2, 4095), (3, 4096)):
                self.inputs.input_lengths = torch.ones(batch, dtype=torch.int32)
                self.inputs.sequence_lengths = torch.full(
                    (batch,), sequence_length, dtype=torch.int32
                )
                with self.subTest(sequence_length=sequence_length):
                    self.assertIs(
                        self.model.prepare_fmha_impl(
                            SimpleNamespace(attention_inputs=self.inputs), True
                        ),
                        instance,
                    )
            self.assertEqual(select.call_count, 2)

    def test_no_backend_uses_static_graph_factory(self):
        self.model.configure_dynamic_decode(True)
        self.inputs.input_lengths = torch.ones(3, dtype=torch.int32)
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"
        with patch(
            module + ".select_decode_backend", return_value=None
        ) as select, patch(
            "rtp_llm.models_py.modules.AttnImplFactory.get_fmha_impl",
            return_value="static",
        ):
            self.assertEqual(
                self.model.prepare_fmha_impl(
                    SimpleNamespace(attention_inputs=self.inputs), True
                ),
                "static",
            )
            self.assertEqual(select.call_count, 1)

    def test_instantiate_passes_fmha_config_when_supported(self):
        class ConfiguredBackend:
            accepts_fmha_config = True

            def __init__(self, configs, inputs, parallelism, fmha_config=None):
                self.fmha_config = fmha_config

            def support_cuda_graph(self):
                return True

        self.model.config.getAttentionConfigs = lambda _size: object()
        self.model.fmha_config = object()
        with patch.object(
            ConfiguredBackend,
            "support_cuda_graph",
            return_value=True,
        ):
            instance = instantiate_decode_backend(
                self.model, self.inputs, ConfiguredBackend
            )
        self.assertIs(instance.fmha_config, self.model.fmha_config)

    def test_eager_and_prefill_keep_mainline_routing(self):
        self.model.configure_dynamic_decode(True)
        with patch(
            "rtp_llm.models_py.modules.AttnImplFactory.get_fmha_impl",
            return_value="static",
        ):
            self.assertEqual(
                self.model.prepare_fmha_impl(
                    SimpleNamespace(attention_inputs=self.inputs)
                ),
                "static",
            )
            self.inputs.is_prefill = True
            self.assertEqual(
                self.model.prepare_fmha_impl(
                    SimpleNamespace(attention_inputs=self.inputs),
                    True,
                    CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH,
                ),
                "static",
            )
        self.assertFalse(self.model.decode_backend_plan)


if __name__ == "__main__":
    unittest.main()
