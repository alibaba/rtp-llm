"""Precision gate built on top of the attention tensor recorder."""

import hashlib
import json
import logging
import re

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.recording_wrapper import (
    RecordingWrapper,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.tensor_recorder import (
    TensorRecorder,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    PREFILL_MHA_IMPS,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.backend_selector import (
    dynamic_backend_allowed,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
    build_decode_gate,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    gate_config,
    parse_gate_thresholds,
    validate_calibration_target,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops import KvCacheDataType


class _GoldenFeedback(RecordingWrapper):
    def __init__(self, wrapper, golden):
        super().__init__(
            wrapper.inner,
            wrapper.attn_configs,
            wrapper.attn_inputs,
            record_inputs=wrapper.record_inputs,
            record_output=wrapper.record_output,
            record_reference=wrapper.record_reference,
        )
        self.golden = golden

    def forward(self, qkv, kv_cache, layer_idx=0):
        actual = super().forward(qkv, kv_cache, layer_idx)
        # Record the candidate output, but isolate its error from later layers.
        return self.golden.layer_records[layer_idx].output.reshape_as(actual)


class GateRecorder(TensorRecorder):
    def __init__(self, model):
        inline_thresholds = getattr(
            getattr(model, "py_hw_kernel_config", None),
            "dyn_decode_gate_thresholds",
            None,
        )
        if inline_thresholds is not None:
            profile = parse_gate_thresholds(inline_thresholds)
            self.gate_profiles = {False: profile, True: profile}
            self.threshold_source = "config"
        else:
            self.gate_profiles = {False: gate_config(False), True: gate_config(True)}
            self.threshold_source = "builtin"
        self.threshold_sha256 = None
        # Allocate collective storage before the recorder-init status reduction.
        self._candidate_registries = {
            False: tuple(dict.fromkeys(PREFILL_MHA_IMPS)),
            True: tuple(dict.fromkeys(DECODE_MHA_IMPS)),
        }
        self._candidate_masks = {
            decode: torch.empty(len(registry), dtype=torch.int32, device="cuda")
            for decode, registry in self._candidate_registries.items()
        }
        self._world_mask = torch.empty(
            1 + len(self._candidate_registries[True]),
            dtype=torch.int32,
            device="cuda",
        )
        super().__init__(model)
        self.expected = {}
        self.active_ids = {}
        self._profile = None
        self._passed = None
        self.model.set_decode_backend_gate(None)

    def start_run(self, scenario_base_name, impl_name, phase, batch_sequence_ids):
        previous_scenario = self._current_scenario_base_name
        previous_phase = self._phase
        if previous_scenario != scenario_base_name or previous_phase != phase:
            self._release_non_decode_records(previous_scenario)
        if previous_scenario != scenario_base_name:
            self._finalize_scenario(previous_scenario)
        key = (scenario_base_name, phase)
        ids = tuple(batch_sequence_ids)
        if key in self.active_ids and self.active_ids[key] != ids:
            raise ValueError(f"active sequence order changed at {key}")
        self.active_ids[key] = ids
        super().start_run(scenario_base_name, impl_name, phase, batch_sequence_ids)

    def stop_run(self, scenario_name):
        self.restore()
        if self._wrapper is not None and self._wrapper.layer_records:
            if (
                self._current_bucket is None
                or self._impl_name is None
                or self._phase is None
            ):
                raise RuntimeError(f"recording state is incomplete for {scenario_name}")
            rec = self._current_bucket.setdefault(self._impl_name, [])
            rec.append(
                AttentionForwardRecord(
                    impl_name=self._impl_name,
                    phase=self._phase,
                    layer_records=self._wrapper.layer_records,
                )
            )
        elif self._impl_name == "golden" or (self._phase or "").startswith("decode_"):
            raise RuntimeError(f"no records for {scenario_name}")
        self._wrapper = None

    def _release_non_decode_records(self, scenario):
        if scenario is None:
            return
        bucket = self.records.get(scenario, {})
        for impl_name, record_list in bucket.items():
            keep = []
            for record in record_list:
                if record.phase.startswith("decode_"):
                    keep.append(record)
                else:
                    record.release()
            bucket[impl_name] = keep

    def _gate_profile(self):
        if self._profile is not None:
            return self._profile
        config = self.model.config.getAttentionConfigs(
            self.model.parallelism_config.get_attn_tp_size()
        )
        if config.dtype != torch.bfloat16 or config.kv_cache_dtype not in (
            KvCacheDataType.BASE,
            KvCacheDataType.FP8,
        ):
            raise ValueError(
                "decode gate supports BF16 activations with BASE or FP8 KV only"
            )
        fp8 = config.kv_cache_dtype == KvCacheDataType.FP8
        profile = self.gate_profiles[fp8]
        self.threshold_sha256 = hashlib.sha256(
            json.dumps(
                profile.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()[:8]
        runtime_sm = get_sm()[0]
        model_type = getattr(self.model.config, "model_type", "")
        logging.info(
            "decode gate thresholds source=%s id=%s sha256=%s "
            "declared=%s/%s runtime=SM%s/%s "
            "signal_split=%s budget=%s roundoff=%s gain=%s cosine=%s nrmse=%s ulp=%s",
            self.threshold_source,
            profile.calibration_id,
            self.threshold_sha256,
            (
                f"SM{profile.calibration_sm_major}"
                if profile.calibration_sm_major is not None
                else "unspecified"
            ),
            profile.calibration_model_family or "unspecified",
            runtime_sm,
            model_type or "unknown",
            profile.signal_split,
            profile.budget_coefficient,
            profile.roundoff_coefficient,
            profile.gain_limit,
            profile.cosine_limit,
            profile.nrmse_limit,
            profile.ulp_limit,
        )
        warning = validate_calibration_target(
            profile,
            runtime_sm,
            model_type,
            source=self.threshold_source,
        )
        if warning:
            logging.warning("%s", warning)
        self._profile = (fp8, profile)
        return self._profile

    def _rank_tag(self):
        # WORLD rank keeps ranks distinguishable across DP groups, where the
        # same tp_rank repeats; recorders without a rank stay readable as 0/1.
        parallelism_config = self.model.parallelism_config
        world_size = getattr(parallelism_config, "world_size", 1)
        world_rank = getattr(parallelism_config, "world_rank", 0)
        return f"rank={world_rank}/{world_size}"

    def _finalize_scenario(self, scenario):
        if scenario is None:
            return
        expected = {
            key: value for key, value in self.expected.items() if key[0] == scenario
        }
        bucket = self.records.get(scenario)
        if bucket is None:
            if expected:
                raise ValueError(f"scenario {scenario} has no records")
            return
        if not expected:
            for record_list in bucket.values():
                for record in record_list:
                    record.release()
            self.records.pop(scenario, None)
            return
        try:
            fp8, profile = self._gate_profile()
            passed, details = build_decode_gate(
                {scenario: bucket},
                expected,
                tuple(range(self.model.layer_num)),
                fp8,
                profile,
            )
            self._passed = (
                passed if self._passed is None else self._passed.intersection(passed)
            )
            for name, verdicts in details.items():
                failures = [
                    entry for entry in verdicts if not entry[-1]["overall_pass"]
                ]
                logging.info(
                    "decode gate %s scenario=%s backend=%s layers=%d failures=%s",
                    self._rank_tag(),
                    scenario,
                    name,
                    len(verdicts),
                    failures[:8],
                )
        finally:
            for record_list in bucket.values():
                for record in record_list:
                    record.release()
            self.records.pop(scenario, None)

    def _recording_options(self):
        decode = (self._phase or "").startswith("decode_")
        golden = self._impl_name == "golden"
        return {
            "record_inputs": decode,
            "record_output": decode or golden,
            "record_reference": decode and golden,
        }

    def list_candidates(self, phase):
        decode = phase.startswith("decode")
        registry = self._candidate_registries[decode]
        mask = self._candidate_masks[decode]
        names = set(self._candidates.get(phase, ()))
        sm_major = get_sm()[0] if decode else None
        for index, cls in enumerate(registry):
            mask[index] = int(
                cls.__name__ in names
                and (not decode or dynamic_backend_allowed(cls.__name__, sm_major))
            )
        world_size = self.model.parallelism_config.world_size
        if world_size > 1:
            all_reduce(mask, group=Group.WORLD)
        counts = mask.tolist()
        dropped = [
            cls.__name__
            for cls, count in zip(registry, counts)
            if 0 < count < world_size
        ]
        if dropped:
            logging.warning(
                "accuracy check [%s]: impls viable on only some ranks, "
                "dropped by world AND-reduce (no records): %s",
                phase,
                dropped,
            )
        candidates = [
            cls.__name__ for cls, count in zip(registry, counts) if count == world_size
        ]
        if phase.startswith("decode_"):
            key = (self._current_scenario_base_name, phase)
            if key in self.expected:
                raise ValueError(f"duplicate candidate schedule {key}")
            self.expected[key] = (self.active_ids[key], tuple(candidates))
        return candidates

    def _recording_prepare(self, inputs, is_cuda_graph=False):
        wrapper = super()._recording_prepare(inputs, is_cuda_graph)
        if self._impl_name != "golden":
            golden = [
                record
                for record in self._current_bucket["golden"]
                if record.phase == self._phase
            ]
            if len(golden) != 1:
                raise ValueError("candidate requires one matching golden forward")
            self._wrapper = _GoldenFeedback(wrapper, golden[0])
        return self._wrapper

    def finalize(self):
        registry = self._candidate_registries[True]
        passed, error = frozenset(), None
        try:
            self._finalize_scenario(self._current_scenario_base_name)
            if self.records:
                raise ValueError(
                    f"decode gate contains unconsumed scenarios: {sorted(self.records)}"
                )
            if self._passed is None:
                raise ValueError("decode gate contains no decode scenarios")
            passed = self._passed
            _, profile = self._gate_profile()
        except Exception as exc:
            error = exc
            logging.exception("decode gate %s records are invalid", self._rank_tag())
        logging.info(
            "decode gate local %s records_ok=%s passed=%s",
            self._rank_tag(),
            error is None,
            sorted(passed),
        )
        # Converge structural errors and precision verdicts in one fixed-shape
        # WORLD reduction. Benchmark selection later uses a separate TP group.
        mask = self._world_mask
        mask[0] = int(error is None)
        for index, cls in enumerate(registry, 1):
            mask[index] = int(cls.__name__ in passed)
        world_size = self.model.parallelism_config.world_size
        if world_size > 1:
            all_reduce(mask, group=Group.WORLD)
        values = mask.tolist()
        if values[0] != world_size:
            raise RuntimeError(
                "decode gate records failed on at least one rank"
            ) from error
        names = [
            cls.__name__
            for cls, count in zip(registry, values[1:])
            if count == world_size
        ]
        if not names:
            raise RuntimeError("no decode backend passed every scenario on every rank")
        self.model.set_decode_backend_gate(names)
        logging.info(
            "decode gate WORLD passed=%s threshold_source=%s threshold_id=%s "
            "threshold_sha256=%s",
            names,
            self.threshold_source,
            profile.calibration_id,
            self.threshold_sha256,
        )
        try:
            calibration_coverage = self._calibration_coverage(profile.calibration_id)
        except Exception:
            logging.warning(
                "decode gate calibration coverage report failed", exc_info=True
            )
        else:
            logging.info("decode gate calibration_coverage=%s", calibration_coverage)

    def _calibration_coverage(self, calibration_id):
        batches = set()
        batch_max_kv = {}
        tested_kv_points = set()
        for (scenario, phase), (active_ids, _) in self.expected.items():
            if not phase.startswith("decode_"):
                continue
            batch = len(active_ids)
            step = int(phase.rsplit("_", 1)[1])
            batches.add(batch)
            sequence_specs = {
                int(seq_id): (int(prefix), int(chunk))
                for seq_id, prefix, chunk, _ in re.findall(
                    r"_s(\d+)_p(\d+)c(\d+)d(\d+)", scenario
                )
            }
            missing = [seq_id for seq_id in active_ids if seq_id not in sequence_specs]
            if missing:
                raise RuntimeError(
                    f"decode calibration scenario {scenario} has no spec for "
                    f"sequence ids {missing}"
                )
            observed = max(
                (
                    sequence_specs.get(seq_id, (0, 0))[0]
                    + sequence_specs.get(seq_id, (0, 0))[1]
                    + step
                    + 1
                    for seq_id in active_ids
                ),
                default=max(1, step + 1),
            )
            batch_max_kv[batch] = max(batch_max_kv.get(batch, 0), observed)
            tested_kv_points.add(observed)
        if not batches:
            raise RuntimeError("decode gate contains no decode calibration batches")
        return {
            "calibration_id": calibration_id,
            "threshold_source": self.threshold_source,
            "threshold_id": calibration_id,
            "threshold_sha256": self.threshold_sha256,
            "tested_batches": tuple(sorted(batches)),
            "tested_kv_points": tuple(sorted(tested_kv_points)),
            "observed_max_kv_per_batch": tuple(sorted(batch_max_kv.items())),
        }
