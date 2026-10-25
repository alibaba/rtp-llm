"""Precision consumer around the unchanged Hongmin TensorRecorder protocol."""

import logging

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
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
from rtp_llm.ops import KvCacheDataType


class _GoldenFeedback(RecordingWrapper):
    def __init__(self, wrapper, golden):
        super().__init__(wrapper.inner, wrapper.attn_configs, wrapper.attn_inputs)
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
        self.model.set_decode_backend_gate(None)

    def start_run(self, scenario_base_name, impl_name, phase, batch_sequence_ids):
        key = (scenario_base_name, phase)
        ids = tuple(batch_sequence_ids)
        if key in self.active_ids and self.active_ids[key] != ids:
            raise ValueError(f"active sequence order changed at {key}")
        self.active_ids[key] = ids
        super().start_run(scenario_base_name, impl_name, phase, batch_sequence_ids)

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
            passed, details = build_decode_gate(
                self.records,
                self.expected,
                tuple(range(self.model.layer_num)),
                config.kv_cache_dtype == KvCacheDataType.FP8,
            )
            for name, verdicts in details.items():
                failures = [
                    entry for entry in verdicts if not entry[-1]["overall_pass"]
                ]
                logging.info(
                    "decode gate backend=%s layers=%d failures=%s",
                    name,
                    len(verdicts),
                    failures[:8],
                )
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
        self.model.set_decode_backend_gate(names)
        logging.info("decode gate WORLD passed=%s", names)
