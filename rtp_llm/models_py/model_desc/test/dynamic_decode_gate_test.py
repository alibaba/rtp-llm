import copy
import filecmp
import importlib.metadata
import os
import pickle
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.cuda_graph import CudaGraphSelectionMode
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
    AttentionLayerRecord,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.recording_wrapper import (
    RecordingWrapper,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
    build_decode_gate,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder import (
    GateRecorder,
    _GoldenFeedback,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    evaluate_precision,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.selector import select_stable
from rtp_llm.ops import HWKernelConfig, KvCacheDataType, RuntimeConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs


def records_fixture():
    q = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    layer = AttentionLayerRecord(
        0, q, q[:, :1].clone(), q[:, :1].clone(), q.reshape(2, 8).clone()
    )
    records = {
        "s": {
            "golden": [AttentionForwardRecord("golden", "decode_0", {0: layer})],
            "Good": [
                AttentionForwardRecord("Good", "decode_0", {0: copy.deepcopy(layer)})
            ],
        }
    }
    return records, {("s", "decode_0"): ((0, 2), ("Good",))}


class DecodeGateTest(unittest.TestCase):
    def test_native_modules_come_from_declared_runfiles(self):
        root = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"]
        for name in ("libth_transformer_config", "librtp_compute_ops"):
            self.assertTrue(
                os.path.samefile(sys.modules[name].__file__, root / (name + ".so"))
            )
        # gpu_lock stages declared wheel packages into stable JIT paths.
        # Check content identity rather than mistaking the copy for ambient torch.
        declared = importlib.metadata.distribution("torch")
        loaded = Path(torch.__file__).parent
        for relative in (
            "__init__.py",
            Path(torch._C.__file__).name,
            "lib/libtorch_cpu.so",
            "lib/libtorch_cuda.so",
        ):
            source = declared.locate_file("torch/" + str(relative))
            self.assertTrue(str(source).startswith(os.environ["TEST_SRCDIR"] + os.sep))
            self.assertTrue(filecmp.cmp(loaded / relative, source, shallow=False))

    def test_config_flags_survive_spawn_pickle(self):
        hw = HWKernelConfig()
        runtime = RuntimeConfig()
        self.assertFalse(hw.enable_dynamic_decode_backend)
        self.assertFalse(runtime.enable_accuracy_check)
        hw.enable_dynamic_decode_backend = True
        runtime.enable_accuracy_check = True
        self.assertTrue(pickle.loads(pickle.dumps(hw)).enable_dynamic_decode_backend)
        self.assertTrue(pickle.loads(pickle.dumps(runtime)).enable_accuracy_check)

    def test_complete_bf16_and_fp8(self):
        records, expected = records_fixture()
        for fp8 in (False, True):
            passed, details = build_decode_gate(records, expected, (0,), fp8)
            self.assertEqual(passed, frozenset({"Good"}))
            self.assertEqual(len(details["Good"]), 1)

    def test_missing_candidate_is_structural_failure(self):
        records, expected = records_fixture()
        records["s"]["Good"].clear()
        with self.assertRaisesRegex(ValueError, "incomplete candidate"):
            build_decode_gate(records, expected, (0,), False)

    def test_missing_layer_is_structural_failure(self):
        records, expected = records_fixture()
        with self.assertRaisesRegex(ValueError, "incomplete golden"):
            build_decode_gate(records, expected, (0, 1), False)

    def test_duplicate_phase_rejected(self):
        records, expected = records_fixture()
        records["s"]["Good"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_decode_gate(records, expected, (0,), False)

    def test_unaligned_inputs_rejected(self):
        records, expected = records_fixture()
        records["s"]["Good"][0].layer_records[0].q.zero_()
        with self.assertRaisesRegex(ValueError, "unaligned q"):
            build_decode_gate(records, expected, (0,), False)

    def test_nonfinite_never_passes(self):
        for value in (float("nan"), float("inf")):
            records, expected = records_fixture()
            records["s"]["Good"][0].layer_records[0].output[0, 0] = value
            passed, _ = build_decode_gate(records, expected, (0,), True)
            self.assertFalse(passed)

    def test_bad_candidate_filtered(self):
        records, expected = records_fixture()
        records["s"]["Good"][0].layer_records[0].output.mul_(2)
        for fp8 in (False, True):
            self.assertFalse(build_decode_gate(records, expected, (0,), fp8)[0])

    def test_empty_schedule_rejected(self):
        with self.assertRaisesRegex(ValueError, "no scenarios"):
            build_decode_gate({}, {}, (0,), False)

    def test_fp8_low_snr_retains_nrmse_gate(self):
        golden = torch.ones(128, dtype=torch.bfloat16)
        noise = torch.ones_like(golden) * 0.08
        noise[::2] *= -1
        fp8 = evaluate_precision(golden + noise, golden, True)
        self.assertTrue(fp8["low_snr"])
        self.assertTrue(fp8["overall_pass"])
        self.assertFalse(
            evaluate_precision(golden + noise, golden, False)["overall_pass"]
        )
        self.assertFalse(evaluate_precision(golden * 1.4, golden, True)["overall_pass"])

    def test_zero_output_is_valid(self):
        self.assertTrue(
            evaluate_precision(torch.zeros(8), torch.zeros(8), False)["overall_pass"]
        )

    def test_golden_feedback_records_actual_output(self):
        configs = SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=4)
        inner = SimpleNamespace(
            forward=lambda qkv, cache, layer: torch.full((2, 8), 9.0)
        )
        wrapper = RecordingWrapper(inner, configs, object())
        records, _ = records_fixture()
        feedback = _GoldenFeedback(wrapper, records["s"]["golden"][0])
        output = feedback.forward(torch.ones(2, 16), None)
        self.assertTrue(torch.equal(output, torch.ones(2, 8, dtype=torch.bfloat16)))
        self.assertTrue(
            torch.equal(feedback.layer_records[0].output, torch.full((2, 8), 9.0))
        )

    def test_threshold_and_registry_tiebreak(self):
        self.assertEqual(select_stable({"A": [100], "B": [97]}, ["A", "B"], "A"), "A")
        self.assertEqual(
            select_stable({"A": [100], "B": [80], "C": [79]}, ["A", "B", "C"], "A"), "B"
        )

    def test_benchmark_cap_respects_physical_capacity_and_memory(self):
        from rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector import (
            benchmark_sequence_cap,
        )

        self.assertEqual(benchmark_sequence_cap(2048, 2, 64, 8, 4096, 2**30), 255)
        self.assertLess(benchmark_sequence_cap(2048, 2, 64, 8, 4096, 0), 0)
        self.assertEqual(benchmark_sequence_cap(128, 2, 64, 8, 4096, 2**30), 128)


class DecodePlanTest(unittest.TestCase):
    def setUp(self):
        self.model = GptModelBase(
            SimpleNamespace(
                num_layers=1, vocab_size=10, capture_aux_hidden_layer_ids=[]
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
        self.inputs.is_prefill = False

    def test_enable_requires_gate(self):
        self.model.set_decode_backend_gate(None)
        with self.assertRaisesRegex(RuntimeError, "before precision"):
            self.model.configure_dynamic_decode(True)

    def test_plan_selected_once_per_batch_and_applied(self):
        self.model.configure_dynamic_decode(True)
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector"
        with patch(
            module + ".select_decode_backend", return_value=self.backend
        ) as select, patch(
            module + ".instantiate_decode_backend", return_value="actual"
        ) as instantiate:
            for _ in range(2):
                self.assertEqual(
                    self.model.prepare_fmha_impl(
                        SimpleNamespace(attention_inputs=self.inputs), True
                    ),
                    "actual",
                )
            self.inputs.input_lengths = torch.ones(4, dtype=torch.int32)
            self.model.prepare_fmha_impl(
                SimpleNamespace(attention_inputs=self.inputs), True
            )
            self.assertEqual(select.call_count, 2)
            self.assertEqual(instantiate.call_count, 3)
            self.assertEqual(set(self.model.decode_backend_plan), {2, 4})

    def test_eager_and_prefill_keep_mainline_routing(self):
        self.model.configure_dynamic_decode(True)
        with patch(
            "rtp_llm.models_py.model_desc.module_base.AttnImplFactory.get_fmha_impl",
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

    def test_world_gate_failure_does_not_install_names(self):
        class Good:
            pass

        records, expected = records_fixture()
        model = SimpleNamespace(
            prepare_fmha_impl=lambda: None,
            layer_num=1,
            config=SimpleNamespace(
                getAttentionConfigs=lambda size: SimpleNamespace(
                    dtype=torch.bfloat16,
                    kv_cache_dtype=KvCacheDataType.BASE,
                    rope_config=None,
                    max_seq_len=128,
                )
            ),
            parallelism_config=SimpleNamespace(
                world_size=2, get_attn_tp_size=lambda: 1
            ),
            set_decode_backend_gate=lambda names: None,
        )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ):
            recorder = GateRecorder(model)
        recorder.records, recorder.expected = records, expected
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        def other_rank_failed(mask, group):
            mask[0] += 0
            mask[1:] += 1

        with patch(module + ".DECODE_MHA_IMPS", [Good]), patch(
            module + ".torch.tensor", side_effect=cpu_tensor
        ), patch(module + ".all_reduce", side_effect=other_rank_failed):
            with self.assertRaisesRegex(RuntimeError, "at least one rank"):
                recorder.finalize()


if __name__ == "__main__":
    unittest.main()
