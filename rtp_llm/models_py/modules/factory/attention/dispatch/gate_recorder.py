"""Precision consumer around the unchanged Hongmin TensorRecorder protocol."""

import logging
import os
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
from rtp_llm.models_py.modules.factory.attention.attn_factory import DECODE_MHA_IMPS
from rtp_llm.models_py.modules.factory.attention.dispatch.decode_gate import (
    build_decode_gate,
)
from rtp_llm.models_py.modules.factory.attention.dispatch.precision_metrics import (
    DecodeGateEnvelope,
    gate_config,
    load_gate_profiles,
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
        super().__init__(model)
        self.expected = {}
        self.active_ids = {}
        profile_path = os.environ.get("DYN_DECODE_GATE_PROFILE")
        self.gate_profiles = (
            load_gate_profiles(profile_path)
            if profile_path
            else {False: gate_config(False), True: gate_config(True)}
        )
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
        validate_calibration_target(
            profile,
            get_sm()[0],
            getattr(self.model.config, "model_type", ""),
            allow_uncalibrated=os.environ.get("DYN_DECODE_ALLOW_UNCALIBRATED_TARGET")
            == "1",
        )
        self._profile = (fp8, profile)
        return self._profile

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
                    "decode gate scenario=%s backend=%s layers=%d failures=%s",
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
        candidates = super().list_candidates(phase)
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
        registry = list(dict.fromkeys(DECODE_MHA_IMPS))
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
            logging.exception("decode gate records are invalid")
        # Converge structural errors and precision verdicts in one fixed-shape
        # WORLD reduction. Benchmark selection later uses a separate TP group.
        mask = torch.tensor(
            [int(error is None)] + [int(cls.__name__ in passed) for cls in registry],
            dtype=torch.int32,
            device="cuda",
        )
        world_size = self.model.parallelism_config.world_size
        if world_size > 1:
            all_reduce(mask, group=Group.DP_AND_TP)
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
        envelope = self._validated_envelope(profile.calibration_id)
        self.model.set_decode_backend_gate(names, envelope)
        logging.info("decode gate WORLD passed=%s", names)

    def _validated_envelope(self, calibration_id):
        batches = set()
        batch_max_kv = {}
        max_steps = 0
        for (scenario, phase), (active_ids, _) in self.expected.items():
            if not phase.startswith("decode_"):
                continue
            batch = len(active_ids)
            step = int(phase.rsplit("_", 1)[1])
            batches.add(batch)
            max_steps = max(max_steps, step + 1)
            sequence_specs = {
                int(seq_id): (int(prefix), int(chunk))
                for seq_id, prefix, chunk, _ in re.findall(
                    r"_s(\d+)_p(\d+)c(\d+)d(\d+)", scenario
                )
            }
            for seq_id in active_ids:
                if seq_id not in sequence_specs:
                    raise RuntimeError(
                        f"decode gate scenario {scenario} has no spec for sequence "
                        f"{seq_id}"
                    )
                prefix, chunk = sequence_specs[seq_id]
                # sequence_lengths contains history before the current token,
                # while the replay guard compares the KV length including it.
                kv_length = prefix + chunk + step + 1
                batch_max_kv[batch] = max(batch_max_kv.get(batch, 0), kv_length)
        if not batches or not batch_max_kv:
            raise RuntimeError("decode gate contains no validated decode envelope")
        return DecodeGateEnvelope(
            tuple(sorted(batches)),
            max(batch_max_kv.values()),
            max_steps,
            calibration_id,
            tuple(sorted(batch_max_kv.items())),
        )
