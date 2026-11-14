import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
    AttentionLayerRecord,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.golden_sdpa_impl import (
    GoldenSDPAImpl,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.recording_wrapper import (
    RecordingWrapper,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import PREFILL_MHA_IMPS
from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
    build_decode_gate,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate_calibration import (
    _grid,
    _report,
    calibrate,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder import (
    GateRecorder,
    _GoldenFeedback,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    DecodeGateConfig,
    FP8_GATE_CONFIG,
    evaluate_precision,
    gate_config,
    load_gate_profiles,
    parse_gate_thresholds,
    validate_calibration_target,
)
from rtp_llm.ops import KvCacheDataType


def records_fixture():
    q = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    layer = AttentionLayerRecord(
        0,
        q,
        q[:, :1].clone(),
        q[:, :1].clone(),
        q.reshape(2, 8).clone(),
        torch.ones(2, 2),
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


class DecodePrecisionGateTest(unittest.TestCase):
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

    def test_unaligned_candidate_is_discarded_without_structural_failure(self):
        records, expected = records_fixture()
        records["s"]["Good"][0].layer_records[0].q.zero_()
        passed, details = build_decode_gate(records, expected, (0,), False)
        self.assertFalse(passed)
        self.assertEqual(details["Good"][0][-1]["reason"], "unaligned q input")

    def test_missing_recorded_input_fails_closed(self):
        records, expected = records_fixture()
        records["s"]["Good"][0].layer_records[0].q = None
        with self.assertRaisesRegex(ValueError, "missing q input"):
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

    def test_low_signal_uses_contribution_budget(self):
        golden = torch.ones(1, 128, dtype=torch.bfloat16)
        noise = torch.ones_like(golden) * 0.08
        noise[:, ::2] *= -1
        # R=1 and S=4 select the low-signal branch independent of candidate error.
        contribution = torch.full((1, 2), 4.0)
        fp8 = evaluate_precision(golden + noise, golden, contribution, True)
        self.assertEqual(fp8["low_signal_sequences"], [0])
        self.assertTrue(fp8["overall_pass"])

    def test_reliable_signal_uses_classic_metrics(self):
        golden = torch.ones(1, 128, dtype=torch.bfloat16)
        noise = torch.ones_like(golden) * 0.08
        noise[:, ::2] *= -1
        contribution = torch.ones((1, 2))
        verdict = evaluate_precision(golden + noise, golden, contribution, False)
        self.assertEqual(verdict["low_signal_sequences"], [])
        self.assertFalse(verdict["overall_pass"])
        self.assertFalse(
            evaluate_precision(golden * 1.4, golden, contribution, False)[
                "overall_pass"
            ]
        )

    def test_fp8_profile_accepts_calibrated_quantization_loss(self):
        golden = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]],
            dtype=torch.bfloat16,
        )
        quantization_noise = torch.tensor(
            [[0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1]],
            dtype=torch.bfloat16,
        )
        candidate = golden + quantization_noise
        contribution = torch.ones((1, 2))
        fp8 = evaluate_precision(candidate, golden, contribution, True)
        bf16 = evaluate_precision(candidate, golden, contribution, False)
        self.assertTrue(fp8["overall_pass"])
        self.assertFalse(bf16["overall_pass"])

    def test_fp8_profile_rejects_scale_and_token_order_faults(self):
        golden = torch.tensor(
            [
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            ],
            dtype=torch.bfloat16,
        )
        contribution = torch.ones((2, 2))
        self.assertFalse(
            evaluate_precision(golden * 1.2, golden, contribution, True)["overall_pass"]
        )
        self.assertFalse(
            evaluate_precision(golden.flip(0), golden, contribution, True)[
                "overall_pass"
            ]
        )

    def test_fp8_and_bf16_use_independent_calibrations(self):
        self.assertEqual(gate_config(False).calibration_id, "h20-qwen3-20261030-v1")
        self.assertEqual(gate_config(True).calibration_id, "h20-qwen3-fp8-20261105-v2")
        self.assertEqual(gate_config(False).cosine_limit, 0.9995)
        self.assertEqual(gate_config(True).cosine_limit, FP8_GATE_CONFIG.cosine_limit)

    def test_zero_output_is_valid(self):
        self.assertTrue(
            evaluate_precision(
                torch.zeros(1, 8),
                torch.zeros(1, 8),
                torch.zeros(1, 2),
                False,
            )["overall_pass"]
        )

    def test_signal_class_does_not_depend_on_candidate_error(self):
        golden = torch.ones(1, 8, dtype=torch.bfloat16)
        contribution = torch.full((1, 2), 4.0)
        config = DecodeGateConfig(gain_limit=100.0, budget_coefficient=1.0)
        for candidate in (golden, golden * 100):
            verdict = evaluate_precision(candidate, golden, contribution, True, config)
            self.assertEqual(verdict["low_signal_sequences"], [0])

    def test_golden_contribution_matches_weighted_v_formula(self):
        impl = object.__new__(GoldenSDPAImpl)
        impl.head_num = 2
        impl.kv_head_num = 1
        impl.scale = 1.0
        q = torch.ones(1, 2, 4)
        k = torch.zeros(8, 1, 4)
        v = torch.ones(8, 1, 4)
        contribution = impl._attention_contribution(q, k, v)
        torch.testing.assert_close(contribution, torch.full((1, 2), 1.0 / 8**0.5))

    def test_golden_contribution_uses_rms_over_head_dim(self):
        impl = object.__new__(GoldenSDPAImpl)
        impl.head_num = 1
        impl.kv_head_num = 1
        impl.scale = 1.0
        q = torch.zeros(1, 1, 2)
        k = torch.zeros(1, 1, 2)
        v = torch.tensor([[[1.0, 3.0]]])
        contribution = impl._attention_contribution(q, k, v)
        torch.testing.assert_close(contribution, torch.tensor([[5.0**0.5]]))

    def test_golden_prefill_reads_device_cu_seqlens_and_host_fallbacks(self):
        device_only = SimpleNamespace(
            cu_seqlens_device=torch.tensor([0, 2, 5], dtype=torch.int32),
            cu_seqlens=torch.empty(0, dtype=torch.int32),
            input_lengths=torch.tensor([2, 3], dtype=torch.int32),
        )
        self.assertEqual(GoldenSDPAImpl._prefill_cu_seqlens(device_only), [0, 2, 5])
        host_only = SimpleNamespace(
            cu_seqlens_device=torch.empty(0, dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 4, 7], dtype=torch.int32),
            input_lengths=torch.empty(0, dtype=torch.int32),
        )
        self.assertEqual(GoldenSDPAImpl._prefill_cu_seqlens(host_only), [0, 4, 7])
        lengths_only = SimpleNamespace(
            cu_seqlens_device=torch.empty(0, dtype=torch.int32),
            cu_seqlens=torch.empty(0, dtype=torch.int32),
            input_lengths=torch.tensor([3, 2, 5], dtype=torch.int32),
        )
        self.assertEqual(
            GoldenSDPAImpl._prefill_cu_seqlens(lengths_only), [0, 3, 5, 10]
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            GoldenSDPAImpl._prefill_cu_seqlens(
                SimpleNamespace(
                    cu_seqlens_device=torch.empty(0, dtype=torch.int32),
                    cu_seqlens=torch.empty(0, dtype=torch.int32),
                    input_lengths=torch.empty(0, dtype=torch.int32),
                )
            )

    def test_recording_wrapper_skips_unneeded_tensor_copies(self):
        configs = SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=4)
        inner = SimpleNamespace(
            forward=lambda qkv, cache, layer: torch.ones(
                (2, 2, 4), dtype=torch.bfloat16
            ),
            last_attention_contribution=None,
        )
        skipped = RecordingWrapper(
            inner,
            configs,
            object(),
            record_inputs=False,
            record_output=False,
            record_reference=False,
        )
        skipped.forward(torch.ones(2, 16), None)
        self.assertEqual(skipped.layer_records, {})
        output_only = RecordingWrapper(
            inner,
            configs,
            object(),
            record_inputs=False,
            record_output=True,
            record_reference=False,
        )
        output_only.forward(torch.ones(2, 16), None)
        record = output_only.layer_records[0]
        self.assertIsNone(record.q)
        self.assertIsNone(record.k)
        self.assertIsNone(record.v)
        self.assertEqual(tuple(record.output.shape), (2, 8))

    def test_gate_profile_loader_and_calibration_target(self):
        config = DecodeGateConfig()
        payload = {
            "selected": {
                "bf16": {"config": config.to_dict()},
                "fp8": {"config": config.to_dict()},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            path.write_text(json.dumps(payload))
            profiles = load_gate_profiles(path)
        self.assertEqual(profiles[False], config)
        validate_calibration_target(config, 9, "Qwen3ForCausalLM")
        validate_calibration_target(config, 9, "qwen_3")
        self.assertIn(
            "runtime=SM10",
            validate_calibration_target(config, 10, "Qwen3ForCausalLM"),
        )

    def test_calibration_requires_normal_pass_and_fault_rejection(self):
        normal = torch.ones(1, 8, dtype=torch.bfloat16)
        records = [
            dict(
                candidate=normal,
                reference=normal,
                contribution=torch.ones(1, 2),
                fp8=fp8,
                kind="normal",
                cohort=f"normal-{fp8}",
            )
            for fp8 in (False, True)
        ] + [
            dict(
                candidate=normal * 2,
                reference=normal,
                contribution=torch.ones(1, 2),
                fp8=fp8,
                kind="scale",
                cohort=f"scale-{fp8}",
            )
            for fp8 in (False, True)
        ]
        selected, _ = calibrate(
            records,
            [DecodeGateConfig()],
            min_normal_pass=1.0,
            min_fault_detection=1.0,
        )
        self.assertIn("fp8", selected)

    def test_calibration_grid_carries_target_identity(self):
        grid = {
            "signal_split": [0.5],
            "budget_coefficient": [0.09],
            "gain_limit": [0.045],
            "cosine_limit": [0.998],
            "nrmse_limit": [0.2],
            "ulp_limit": [None],
        }
        configs = list(_grid(grid, "sm10-qwen3-test", 10, "qwen3"))
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].calibration_id, "sm10-qwen3-test")
        self.assertEqual(configs[0].calibration_sm_major, 10)
        self.assertEqual(configs[0].calibration_model_family, "qwen3")

    def test_calibration_report_carries_target_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "records.pt"
            grid_path = temp / "grid.json"
            input_path.write_text("records")
            grid_path.write_text("{}")
            report = _report(
                {},
                input_path,
                grid_path,
                "sm10-qwen3-test",
                10,
                "qwen3",
            )
        self.assertIn("calibration id: `sm10-qwen3-test`", report)
        self.assertIn("target SM major: `10`", report)
        self.assertIn("target model family: `qwen3`", report)

    def test_calibration_report_adds_qdq_diagnostics_without_admission(self):
        reference = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]],
            dtype=torch.bfloat16,
        )
        quantization_loss = torch.tensor(
            [[0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1]],
            dtype=torch.bfloat16,
        )
        qdq_reference = reference + quantization_loss
        candidate = qdq_reference + quantization_loss * 0.01
        contribution = torch.ones((1, 2))
        records = [
            {
                "candidate": candidate,
                "reference": reference,
                "qdq_reference": qdq_reference,
                "contribution": contribution,
                "fp8": True,
                "kind": "normal",
                "cohort": "fp8-normal",
            }
        ]
        selected = {
            "fp8": {
                "config": FP8_GATE_CONFIG.to_dict(),
                "normal_pass_rate": 1.0,
                "minimum_fault_detection_rate": 1.0,
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "records.pt"
            grid_path = temp / "grid.json"
            input_path.write_text("records")
            grid_path.write_text("{}")
            report = _report(
                selected,
                input_path,
                grid_path,
                "h20-qwen3-fp8-test",
                9,
                "qwen3",
                records,
            )
        self.assertIn("QDQ references are diagnostic only", report)
        self.assertIn("| fp8 | 1 | 1.0000 | 1.0000 | 1.0000 |", report)

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

    def test_first_golden_record_initializes_scenario_bucket(self):
        model = SimpleNamespace(
            prepare_fmha_impl=lambda: None,
            config=SimpleNamespace(
                getAttentionConfigs=lambda size: SimpleNamespace(
                    rope_config=None, max_seq_len=128
                )
            ),
            parallelism_config=SimpleNamespace(
                world_size=1, get_attn_tp_size=lambda: 1
            ),
            set_decode_backend_gate=lambda *args: None,
        )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ):
            recorder = GateRecorder(model)
        recorder.records = {"scenario": {}}
        recorder._current_bucket = recorder.records["scenario"]
        recorder._current_scenario_base_name = "scenario"
        recorder._impl_name = "golden"
        recorder._phase = "plain"
        recorder._wrapper = SimpleNamespace(
            layer_records={
                0: AttentionLayerRecord(
                    layer_idx=0,
                    q=None,
                    k=None,
                    v=None,
                    output=torch.ones(1, 8),
                )
            }
        )
        recorder.stop_run("scenario::golden::plain")
        self.assertEqual(len(recorder.records["scenario"]["golden"]), 1)

    def test_candidate_same_phase_keeps_golden_feedback(self):
        model = SimpleNamespace(
            prepare_fmha_impl=lambda: None,
            config=SimpleNamespace(
                getAttentionConfigs=lambda size: SimpleNamespace(
                    rope_config=None, max_seq_len=128
                )
            ),
            parallelism_config=SimpleNamespace(
                world_size=1, get_attn_tp_size=lambda: 1
            ),
            set_decode_backend_gate=lambda *args: None,
        )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ):
            recorder = GateRecorder(model)
        golden = AttentionForwardRecord(
            impl_name="golden",
            phase="plain",
            layer_records={
                0: AttentionLayerRecord(
                    layer_idx=0,
                    q=torch.ones(1, 2, 4),
                    k=torch.ones(1, 1, 4),
                    v=torch.ones(1, 1, 4),
                    output=torch.ones(1, 8),
                )
            },
        )
        recorder.records = {"scenario": {"golden": [golden]}}
        recorder._current_bucket = recorder.records["scenario"]
        recorder._current_scenario_base_name = "scenario"
        recorder._phase = "plain"
        recorder.start_run("scenario", PREFILL_MHA_IMPS[0].__name__, "plain", (0,))
        self.assertEqual(recorder.records["scenario"]["golden"], [golden])
        recorder.restore()

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
            from rtp_llm.models_py.distributed.collective_torch import Group

            self.assertEqual(group, Group.WORLD)
            mask[0] += 0
            mask[1:] += 1

        with patch(module + ".DECODE_MHA_IMPS", [Good]), patch(
            module + ".torch.tensor", side_effect=cpu_tensor
        ), patch(module + ".all_reduce", side_effect=other_rank_failed):
            with self.assertRaisesRegex(RuntimeError, "at least one rank"):
                recorder.finalize()

    def test_scenario_is_released_after_precision_consumption(self):
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
                ),
                model_type="Qwen3ForCausalLM",
            ),
            parallelism_config=SimpleNamespace(
                world_size=1, get_attn_tp_size=lambda: 1
            ),
            set_decode_backend_gate=lambda *args: None,
        )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ):
            recorder = GateRecorder(model)
        recorded_forwards = list(records["s"].values())
        recorder.records = records
        recorder.expected = expected
        recorder._finalize_scenario("s")
        self.assertEqual(recorder._passed, frozenset({"Good"}))
        self.assertNotIn("s", recorder.records)
        self.assertTrue(
            all(
                not record.layer_records
                for impl in recorded_forwards
                for record in impl
            )
        )

    def test_calibration_coverage_is_report_only(self):
        class NoCapacityAccess:
            def get_layer_cache_groups(self, layer):
                raise AssertionError(
                    "coverage must not inspect temporary pool capacity"
                )

        model = SimpleNamespace(
            kv_cache=NoCapacityAccess(),
            prepare_fmha_impl=lambda: None,
            config=SimpleNamespace(
                getAttentionConfigs=lambda size: SimpleNamespace(
                    rope_config=None, max_seq_len=128
                )
            ),
            parallelism_config=SimpleNamespace(
                world_size=1, get_attn_tp_size=lambda: 1
            ),
            set_decode_backend_gate=lambda *args: None,
        )
        with patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ):
            recorder = GateRecorder(model)
        recorder.expected = {
            ("b3__s0_p0c64d4__s1_p64c64d2__s2_p128c64d0", "decode_0"): (
                (0, 1, 2),
                ("A",),
            ),
            ("b3__s0_p0c64d4__s1_p64c64d2__s2_p128c64d0", "decode_3"): (
                (0,),
                ("A",),
            ),
        }
        coverage = recorder._calibration_coverage("cal")
        self.assertEqual(coverage["calibration_id"], "cal")
        self.assertEqual(coverage["tested_batches"], (1, 3))
        self.assertEqual(coverage["tested_kv_points"], (68, 193))
        self.assertEqual(coverage["observed_max_kv_per_batch"], ((1, 68), (3, 193)))
        self.assertNotIn("max_fit_kv_per_batch", coverage)


class DecodeGateThresholdInjectionTest(unittest.TestCase):
    def _thresholds(self):
        config = gate_config(False).to_dict()
        config.pop("calibration_sm_major")
        config.pop("calibration_model_family")
        config["target"] = {"sm_major": 9, "model_family": "qwen3"}
        return config

    def _recorder(self, *, fp8=False, environment=None):
        attention = SimpleNamespace(
            rope_config=None,
            max_seq_len=128,
            dtype=torch.bfloat16,
            kv_cache_dtype=KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE,
        )
        model = SimpleNamespace(
            prepare_fmha_impl=lambda: None,
            config=SimpleNamespace(
                model_type="qwen_3", getAttentionConfigs=lambda size: attention
            ),
            parallelism_config=SimpleNamespace(
                world_size=1, get_attn_tp_size=lambda: 1
            ),
            set_decode_backend_gate=lambda names: None,
        )
        with patch.dict("os.environ", environment or {}, clear=True), patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ), patch(
            "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder.get_sm",
            return_value=(9, 0),
        ):
            recorder = GateRecorder(model)
            recorder._gate_profile()
        return recorder

    def test_no_environment_keeps_builtin_and_legacy_file_profiles(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8), self.assertLogs(level="INFO") as logs:
                recorder = self._recorder(fp8=fp8)
            self.assertEqual(recorder._profile, (fp8, gate_config(fp8)))
            self.assertEqual(recorder.threshold_source, "builtin")
            self.assertIn("source=builtin", "\n".join(logs.output))
            self.assertRegex(recorder.threshold_sha256, r"^[0-9a-f]{8}$")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        name: {"config": gate_config(fp8).to_dict()}
                        for name, fp8 in (("bf16", False), ("fp8", True))
                    }
                )
            )
            for fp8 in (False, True):
                recorder = self._recorder(
                    fp8=fp8,
                    environment={"DYN_DECODE_GATE_PROFILE": str(path)},
                )
                self.assertEqual(recorder._profile, (fp8, gate_config(fp8)))
                self.assertEqual(recorder.threshold_source, "file")

    def test_inline_thresholds_override_file_and_apply_to_active_kv_profile(self):
        value = self._thresholds()
        value["calibration_id"] = "inline-equivalent"
        hashes = []
        for fp8 in (False, True):
            with self.subTest(fp8=fp8), self.assertLogs(level="INFO") as logs:
                recorder = self._recorder(
                    fp8=fp8,
                    environment={
                        "DYN_DECODE_GATE_THRESHOLDS": json.dumps(value),
                        "DYN_DECODE_GATE_PROFILE": "/must-not-be-read.json",
                    },
                )
            self.assertEqual(recorder._profile[0], fp8)
            self.assertEqual(
                recorder._profile[1], parse_gate_thresholds(json.dumps(value))
            )
            self.assertIn("source=env id=inline-equivalent", "\n".join(logs.output))
            recorder.expected = {("b1__s0_p0c1d1", "decode_0"): ((0,), ("Good",))}
            coverage = recorder._calibration_coverage("inline-equivalent")
            self.assertEqual(coverage["threshold_source"], "env")
            self.assertEqual(coverage["threshold_id"], "inline-equivalent")
            self.assertEqual(coverage["threshold_sha256"], recorder.threshold_sha256)
            hashes.append(recorder.threshold_sha256)
        self.assertEqual(hashes[0], hashes[1])
        value.pop("calibration_id")
        value.pop("target")
        config = parse_gate_thresholds(json.dumps(value))
        self.assertEqual(config.calibration_id, "operator-supplied")
        self.assertIsNone(config.calibration_sm_major)
        self.assertIsNone(validate_calibration_target(config, 10, "other_model"))

    def test_missing_thresholds_fail_without_inheriting_defaults(self):
        value = self._thresholds()
        for key in set(value) - {"calibration_id", "target"}:
            missing = dict(value)
            missing.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, f"missing keys:.*{key}"
            ):
                parse_gate_thresholds(json.dumps(missing))
        with self.assertRaisesRegex(ValueError, "budget_coefficient.*gain_limit"):
            parse_gate_thresholds(
                json.dumps(
                    {
                        key: item
                        for key, item in value.items()
                        if key not in ("budget_coefficient", "gain_limit")
                    }
                )
            )

    def test_unknown_keys_and_malformed_json_fail(self):
        for extra in ("unexpected", "bf16", "calibration_sm_major"):
            with self.subTest(extra=extra), self.assertRaisesRegex(
                ValueError, f"unknown keys:.*{extra}"
            ):
                parse_gate_thresholds(json.dumps({**self._thresholds(), extra: 1}))
        value = self._thresholds()
        value["target"]["unknown"] = True
        with self.assertRaisesRegex(ValueError, "target"):
            parse_gate_thresholds(json.dumps(value))
        for raw in ("[]", "null", '"path/to/profile.json"'):
            with self.subTest(raw=raw), self.assertRaisesRegex(
                ValueError, "JSON object"
            ):
                parse_gate_thresholds(raw)
        for raw in ("", "{broken"):
            with self.subTest(raw=raw), self.assertRaises(json.JSONDecodeError):
                self._recorder(
                    environment={
                        "DYN_DECODE_GATE_THRESHOLDS": raw,
                        "DYN_DECODE_GATE_PROFILE": "/must-not-be-read.json",
                    }
                )

    def test_invalid_ranges_and_nonfinite_values_fail(self):
        thresholds = set(self._thresholds()) - {"calibration_id", "target"}
        for key in thresholds:
            for invalid in (float("nan"), float("inf"), -float("inf"), True, "0.5"):
                value = {**self._thresholds(), key: invalid}
                with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(
                    ValueError, key
                ):
                    parse_gate_thresholds(json.dumps(value))
        for key, invalid in (
            ("signal_split", 0),
            ("signal_split", 1),
            ("cosine_limit", 0),
            ("cosine_limit", 1.01),
            ("budget_coefficient", -1),
            ("roundoff_coefficient", -1),
            ("gain_limit", -1),
            ("nrmse_limit", -1),
            ("ulp_limit", 0),
            ("ulp_limit", -1),
        ):
            with self.subTest(key=key, invalid=invalid), self.assertRaisesRegex(
                ValueError, key
            ):
                parse_gate_thresholds(json.dumps({**self._thresholds(), key: invalid}))
        valid = {
            **self._thresholds(),
            "cosine_limit": 1,
            "budget_coefficient": 0,
            "roundoff_coefficient": 0,
            "gain_limit": 0,
            "nrmse_limit": 0,
            "ulp_limit": None,
        }
        self.assertIsNone(parse_gate_thresholds(json.dumps(valid)).ulp_limit)

    def test_target_mismatch_only_warns_without_changing_verdict(self):
        value = self._thresholds()
        value["target"] = {"sm_major": 10, "model_family": "qwen3"}
        value["calibration_id"] = "sm100-qwen3-test"
        with self.assertLogs(level="WARNING") as logs:
            recorder = self._recorder(
                environment={"DYN_DECODE_GATE_THRESHOLDS": json.dumps(value)}
            )
        warning = "\n".join(logs.output)
        self.assertIn(
            "decode gate thresholds: source=env id=sm100-qwen3-test "
            "declared=SM10/qwen3 runtime=SM9/qwen_3;",
            warning,
        )
        self.assertIn(
            "verdicts are only as trustworthy as the supplied calibration", warning
        )
        golden = torch.ones(1, 8, dtype=torch.bfloat16)
        contribution = torch.ones(1, 2)
        for candidate in (golden, golden * 1.2):
            actual = evaluate_precision(
                candidate, golden, contribution, False, recorder._profile[1]
            )
            builtin = evaluate_precision(candidate, golden, contribution, False)
            actual.pop("calibration_id")
            builtin.pop("calibration_id")
            self.assertEqual(actual, builtin)


if __name__ == "__main__":
    unittest.main()
