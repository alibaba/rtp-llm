import copy
import json
import multiprocessing
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
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
from rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder import (
    GateRecorder,
    _GoldenFeedback,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    FP8_GATE_CONFIG,
    DecodeGateConfig,
    evaluate_precision,
    gate_config,
    parse_gate_thresholds,
    validate_calibration_target,
)
from rtp_llm.ops import KvCacheDataType


def _gate_world_worker(
    rank,
    rendezvous,
    result_dir,
    fail_rank,
    backend="gloo",
    fail_allocation=0,
    mixed_sm=False,
):
    import torch.distributed as dist

    module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
    device = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    dist.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    installed, operations = [], []
    status = torch.ones(1, dtype=torch.int32, device=device)
    original_empty = torch.empty
    allocations = 0
    model = SimpleNamespace(
        prepare_fmha_impl=lambda: None,
        layer_num=1,
        config=SimpleNamespace(
            getAttentionConfigs=lambda size: SimpleNamespace(
                rope_config=None, max_seq_len=128
            )
        ),
        parallelism_config=SimpleNamespace(world_size=2, get_attn_tp_size=lambda: 1),
        set_decode_backend_gate=installed.append,
    )

    def allocate(*args, **kwargs):
        nonlocal allocations
        current = allocations
        allocations += 1
        if rank == fail_rank and current == fail_allocation:
            raise MemoryError("injected candidate mask allocation failure")
        kwargs["device"] = device
        return original_empty(*args, **kwargs)

    def reduce(mask, group):
        operations.append(("SUM", mask.numel()))
        dist.all_reduce(mask, op=dist.ReduceOp.SUM)

    error = None
    try:
        registry = [type("Good", (), {})]
        if mixed_sm:
            registry.append(type("XQADecodeImpl", (), {}))
        with patch(module + ".DECODE_MHA_IMPS", registry), patch(
            module + ".PREFILL_MHA_IMPS", [type("Good", (), {})]
        ), patch(module + ".torch.empty", side_effect=allocate), patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ):
            try:
                recorder = GateRecorder(model)
            except MemoryError as exc:
                error = str(exc)
                status.zero_()
        # Mirror AccuracyChecker's existing recorder-init status convergence.
        operations.append(("MIN", 1))
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if status.item():
            recorder._profile = (False, gate_config(False))
            recorder._current_scenario_base_name = "s"
            recorder._candidates = {"decode_0": [cls.__name__ for cls in registry]}
            recorder.active_ids = {("s", "decode_0"): (0, 1)}
            recorder.records, _ = records_fixture()
            with patch(module + ".all_reduce", side_effect=reduce), patch(
                module + ".get_sm", return_value=(9 + rank if mixed_sm else 9, 0)
            ), patch(
                module + ".torch.empty",
                side_effect=AssertionError("collective storage allocated after init"),
            ), patch(
                module + ".torch.tensor",
                side_effect=AssertionError("collective tensor created after init"),
            ), patch.object(
                recorder, "_calibration_coverage", return_value={}
            ):
                assert recorder.list_candidates("decode_0") == ["Good"]
                assert recorder.expected[("s", "decode_0")] == ((0, 1), ("Good",))
                recorder.finalize()
            recorder.close()
        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    ready=bool(status.item()),
                    error=error,
                    installed=installed,
                    operations=operations,
                )
            )
        )
    finally:
        dist.destroy_process_group()


def _threshold_config_worker(hw_config, result_file):
    recorder = DecodeGateThresholdInjectionTest()._recorder(hw_config=hw_config)
    Path(result_file).write_text(
        json.dumps(
            {
                "profile": recorder._profile[1].to_dict(),
                "source": recorder.threshold_source,
            }
        )
    )
    recorder.close()


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


def gate_recorder(world_rank=0, world_size=1):
    """Recorder over the shared records fixture, with a configurable WORLD rank."""
    model = SimpleNamespace(
        prepare_fmha_impl=lambda: None,
        layer_num=1,
        config=SimpleNamespace(
            getAttentionConfigs=lambda size: SimpleNamespace(
                dtype=torch.bfloat16,
                kv_cache_dtype=KvCacheDataType.BASE,
                model_type="Qwen3ForCausalLM",
                rope_config=None,
                max_seq_len=128,
            )
        ),
        parallelism_config=SimpleNamespace(
            world_size=world_size,
            world_rank=world_rank,
            get_attn_tp_size=lambda: 1,
        ),
        set_decode_backend_gate=lambda *args: None,
    )
    module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
    original_empty = torch.empty
    with patch(
        "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
    ), patch(
        module + ".torch.empty",
        side_effect=lambda *args, **kwargs: original_empty(
            *args, **{**kwargs, "device": "cpu"}
        ),
    ), patch(
        module + ".DECODE_MHA_IMPS", [type("Good", (), {})]
    ):
        return GateRecorder(model)


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
        self.assertEqual(gate_config(True).calibration_id, "h20-qwen3-fp8-20261115-v3")
        self.assertEqual(gate_config(False).cosine_limit, 0.9995)
        self.assertEqual(gate_config(True).cosine_limit, FP8_GATE_CONFIG.cosine_limit)

    def test_fp8_total_error_budget_keeps_gain_and_direction_guards(self):
        golden = torch.ones(1, 128, dtype=torch.bfloat16)
        noise = golden.clone()
        noise[:, ::2] *= -1
        candidate = golden * 0.925 + noise * 0.29
        contribution = torch.ones(1, 2)
        old = replace(gate_config(True), cosine_limit=0.975, nrmse_limit=0.21)
        self.assertFalse(
            evaluate_precision(candidate, golden, contribution, True, old)[
                "overall_pass"
            ]
        )
        self.assertTrue(
            evaluate_precision(candidate, golden, contribution, True)["overall_pass"]
        )
        self.assertFalse(
            evaluate_precision(candidate, golden, contribution, False)["overall_pass"]
        )
        for bad in (golden * 1.2, -golden, golden + noise * 0.5):
            self.assertFalse(
                evaluate_precision(bad, golden, contribution, True)["overall_pass"]
            )

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

    def test_calibration_target_is_diagnostic(self):
        config = DecodeGateConfig()
        self.assertIsNone(validate_calibration_target(config, 9, "Qwen3ForCausalLM"))
        self.assertIsNone(validate_calibration_target(config, 9, "qwen_3"))
        self.assertIn(
            "runtime=SM10",
            validate_calibration_target(config, 10, "Qwen3ForCausalLM"),
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
        installed = []
        recorder = gate_recorder(0, 2)
        recorder.model.set_decode_backend_gate = installed.append
        recorder.records, recorder.expected = records_fixture()
        recorder._current_scenario_base_name = "s"
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"

        def remote_rank_failed(mask, group):
            from rtp_llm.models_py.distributed.collective_torch import Group

            self.assertEqual(group, Group.WORLD)
            # This rank passed locally; only the peer contributed a failed record.
            mask[1:] *= 2

        with patch(module + ".all_reduce", side_effect=remote_rank_failed):
            with self.assertRaisesRegex(RuntimeError, "at least one rank"):
                recorder.finalize()
        # This rank's own verdict passed; only the peer record was missing.
        self.assertEqual(recorder._passed, frozenset({"Good"}))
        self.assertEqual(installed, [])

    def test_failure_details_log_world_rank_and_backend(self):
        recorder = gate_recorder(3, 8)
        recorder.records, recorder.expected = records_fixture()
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
        verdict = (0, None, None, {"overall_pass": False, "reason": "injected"})
        with patch(
            module + ".build_decode_gate",
            return_value=(frozenset(), {"Good": [verdict]}),
        ), self.assertLogs(level="INFO") as captured:
            recorder._finalize_scenario("s")
        detail = next(line for line in captured.output if "scenario=s" in line)
        self.assertIn("rank=3/8", detail)
        self.assertIn("backend=Good", detail)
        self.assertIn("failures=[", detail)
        self.assertIn("injected", detail)

    def test_local_result_log_carries_rank_and_passed_names(self):
        recorder = gate_recorder(3, 8)
        recorder.records, recorder.expected = records_fixture()
        recorder._current_scenario_base_name = "s"
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"

        def all_ranks_passed(mask, group):
            mask[0] += 7
            mask[1:] *= 8

        with patch(module + ".all_reduce", side_effect=all_ranks_passed), patch.object(
            recorder, "_calibration_coverage", return_value={}
        ), self.assertLogs(level="INFO") as captured:
            recorder.finalize()
        local = next(line for line in captured.output if "decode gate local " in line)
        self.assertIn("rank=3/8", local)
        self.assertIn("records_ok=True", local)
        self.assertIn("passed=['Good']", local)
        world = next(line for line in captured.output if "decode gate WORLD" in line)
        self.assertIn("passed=['Good']", world)

    def test_structural_failure_log_carries_rank_and_still_raises(self):
        recorder = gate_recorder(1, 2)
        recorder.records = {"leaked": {}}
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
        with patch(
            module + ".all_reduce", side_effect=lambda mask, group: None
        ), self.assertLogs(level="ERROR") as captured:
            with self.assertRaisesRegex(RuntimeError, "at least one rank"):
                recorder.finalize()
        self.assertTrue(
            any(
                "rank=1/2" in line and "records are invalid" in line
                for line in captured.output
            ),
            captured.output,
        )

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
    def test_registered_thresholds_reach_model_and_recorder_after_spawn(self):
        from rtp_llm.server.server_args import server_args

        value = json.dumps(self._thresholds())
        with patch.dict(
            "os.environ", {"DYN_DECODE_GATE_THRESHOLDS": value}, clear=True
        ):
            config = server_args.setup_args(
                ["--enable_dynamic_decode_backend", "1"]
            ).py_hw_kernel_config
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmp:
            result_file = str(Path(tmp, "result.json"))
            worker = context.Process(
                target=_threshold_config_worker, args=(config, result_file)
            )
            try:
                worker.start()
                worker.join(timeout=60)
                self.assertFalse(worker.is_alive())
                self.assertEqual(worker.exitcode, 0)
                result = json.loads(Path(result_file).read_text())
                self.assertEqual(result["source"], "config")
                self.assertEqual(
                    result["profile"], parse_gate_thresholds(value).to_dict()
                )
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)

    def test_two_rank_mask_allocation_and_collective_order(self):
        context = multiprocessing.get_context("spawn")
        for fail_rank, allocation, mixed_sm in (
            (-1, 0, False),
            (1, 0, False),
            (1, 1, False),
            (1, 2, False),
            (-1, 0, True),
        ):
            with self.subTest(
                fail_rank=fail_rank, allocation=allocation, mixed_sm=mixed_sm
            ), tempfile.TemporaryDirectory() as tmp:
                workers = [
                    context.Process(
                        target=_gate_world_worker,
                        args=(
                            rank,
                            str(Path(tmp, "rendezvous")),
                            tmp,
                            fail_rank,
                            "gloo",
                            allocation,
                            mixed_sm,
                        ),
                    )
                    for rank in range(2)
                ]
                try:
                    for worker in workers:
                        worker.start()
                    for worker in workers:
                        worker.join(timeout=60)
                    self.assertTrue(all(not worker.is_alive() for worker in workers))
                    self.assertEqual([worker.exitcode for worker in workers], [0, 0])
                    results = [
                        json.loads(Path(tmp, f"rank-{rank}.json").read_text())
                        for rank in range(2)
                    ]
                    expected_ops = (
                        [
                            ["MIN", 1],
                            ["SUM", 2 if mixed_sm else 1],
                            ["SUM", 3 if mixed_sm else 2],
                        ]
                        if fail_rank == -1
                        else [["MIN", 1]]
                    )
                    for rank, result in enumerate(results):
                        self.assertEqual(result["operations"], expected_ops)
                        self.assertEqual(result["ready"], fail_rank == -1)
                        self.assertEqual(
                            result["installed"],
                            (
                                [None, ["Good"]]
                                if fail_rank == -1
                                else ([] if rank == fail_rank else [None])
                            ),
                        )
                    if fail_rank == 1:
                        self.assertIn("allocation failure", results[1]["error"])
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                            worker.join(timeout=5)

    def test_coverage_failure_does_not_block_admission(self):
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
        for report_error in (False, True):
            recorder = self._recorder()
            installed = []
            recorder.model.set_decode_backend_gate = installed.append
            recorder._passed = frozenset({"Good"})
            with patch.object(recorder, "_finalize_scenario"), patch.object(
                recorder,
                "_calibration_coverage",
                side_effect=ValueError("bad report") if report_error else None,
                return_value={"tested_batches": (1,)},
            ):
                if report_error:
                    with self.assertLogs(level="WARNING") as logs:
                        recorder.finalize()
                    self.assertIn("coverage report failed", "\n".join(logs.output))
                else:
                    recorder.finalize()
            self.assertEqual(installed, [["Good"]])

    def test_gate_errors_are_not_swallowed_by_report_protection(self):
        module = "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder"
        for failure in ("precision", "records", "world", "install"):
            recorder = self._recorder()
            installed = []
            recorder.model.set_decode_backend_gate = installed.append
            recorder._passed = (
                frozenset() if failure == "precision" else frozenset({"Good"})
            )
            if failure == "install":

                def fail_install(names):
                    raise RuntimeError("installation failed")

                recorder.model.set_decode_backend_gate = fail_install
            if failure == "world":
                recorder.model.parallelism_config.world_size = 2
            with patch.object(
                recorder,
                "_finalize_scenario",
                side_effect=ValueError("bad records") if failure == "records" else None,
            ), patch(
                module + ".all_reduce", side_effect=RuntimeError("WORLD failed")
            ), patch.object(
                recorder, "_calibration_coverage"
            ) as report:
                with self.assertRaises(RuntimeError):
                    recorder.finalize()
                report.assert_not_called()
            self.assertEqual(installed, [])

    def _thresholds(self):
        config = gate_config(False).to_dict()
        config.pop("calibration_sm_major")
        config.pop("calibration_model_family")
        config["target"] = {"sm_major": 9, "model_family": "qwen3"}
        return config

    def _recorder(self, *, fp8=False, thresholds=None, hw_config=None):
        from rtp_llm.models_py.model_desc.module_base import GptModelBase
        from rtp_llm.ops import HWKernelConfig

        attention = SimpleNamespace(
            rope_config=None,
            max_seq_len=128,
            dtype=torch.bfloat16,
            kv_cache_dtype=KvCacheDataType.FP8 if fp8 else KvCacheDataType.BASE,
        )
        if hw_config is None:
            hw_config = HWKernelConfig()
            hw_config.dyn_decode_gate_thresholds = thresholds
        model = GptModelBase(
            SimpleNamespace(
                num_layers=1,
                vocab_size=8,
                capture_aux_hidden_layer_ids=[],
                model_type="qwen_3",
                getAttentionConfigs=lambda size: attention,
            ),
            SimpleNamespace(world_size=1, get_attn_tp_size=lambda: 1),
            object(),
            8,
            py_hw_kernel_config=hw_config,
        )
        original_empty = torch.empty
        with patch.dict("os.environ", {}, clear=True), patch(
            "rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder.get_rope_cache_once"
        ), patch(
            "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder.get_sm",
            return_value=(9, 0),
        ), patch(
            "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder.DECODE_MHA_IMPS",
            [type("Good", (), {})],
        ), patch(
            "rtp_llm.models_py.modules.factory.attention.dispatch.gate_recorder.torch.empty",
            side_effect=lambda *args, **kwargs: original_empty(
                *args, **{**kwargs, "device": "cpu"}
            ),
        ):
            recorder = GateRecorder(model)
            recorder._gate_profile()
        return recorder

    def test_unconfigured_uses_builtin(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8), self.assertLogs(level="INFO") as logs:
                recorder = self._recorder(fp8=fp8)
            self.assertEqual(recorder._profile, (fp8, gate_config(fp8)))
            self.assertEqual(recorder.threshold_source, "builtin")
            self.assertIn("source=builtin", "\n".join(logs.output))
            self.assertRegex(recorder.threshold_sha256, r"^[0-9a-f]{8}$")

    def test_config_thresholds_apply_to_active_kv_profile(self):
        value = self._thresholds()
        value["calibration_id"] = "inline-equivalent"
        hashes = []
        for fp8 in (False, True):
            with self.subTest(fp8=fp8), self.assertLogs(level="INFO") as logs:
                recorder = self._recorder(
                    fp8=fp8,
                    thresholds=json.dumps(value),
                )
            self.assertEqual(recorder._profile[0], fp8)
            self.assertEqual(
                recorder._profile[1], parse_gate_thresholds(json.dumps(value))
            )
            self.assertIn("source=config id=inline-equivalent", "\n".join(logs.output))
            recorder.expected = {("b1__s0_p0c1d1", "decode_0"): ((0,), ("Good",))}
            coverage = recorder._calibration_coverage("inline-equivalent")
            self.assertEqual(coverage["threshold_source"], "config")
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
                self._recorder(thresholds=raw)

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
            recorder = self._recorder(thresholds=json.dumps(value))
        warning = "\n".join(logs.output)
        self.assertIn(
            "decode gate thresholds: source=config id=sm100-qwen3-test "
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
