import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.attention.accuracy.attention_record import (
    AttentionForwardRecord,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.golden_kv_history import (
    GoldenKVHistory,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.golden_sdpa_impl import (
    GoldenSDPAImpl,
)
from rtp_llm.models_py.modules.factory.attention.accuracy.recording_wrapper import (
    RecordingWrapper,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    DECODE_MHA_IMPS,
    PREFILL_MHA_IMPS,
    get_all_supported_impls,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import PyModelInputs


class TensorRecorder:
    def __init__(self, model: Any) -> None:
        self.model = model
        # scenario -> impl -> forward records
        self.records: Dict[str, Dict[str, List[AttentionForwardRecord]]] = {}
        self._current_bucket: Optional[Dict[str, List[AttentionForwardRecord]]] = None
        self._current_scenario_base_name: Optional[str] = None
        self._golden_history: Optional[GoldenKVHistory] = None
        self._wrapper: Optional[RecordingWrapper] = None
        self._impl_name: Optional[str] = None
        self._phase: Optional[str] = None
        self._factory: Optional[Callable[..., FMHAImplBase]] = None
        self._candidates: Dict[str, List[str]] = {}
        self._orig_prepare: Callable[..., FMHAImplBase] = model.prepare_fmha_impl
        self._patched = False
        self._closed = False

    def list_candidates(self, phase: str) -> List[str]:
        # Keep only implementations supported on every rank
        raw_registry = (
            DECODE_MHA_IMPS if phase.startswith("decode") else PREFILL_MHA_IMPS
        )
        # Deduplicate while preserving registry order.
        registry = list(dict.fromkeys(raw_registry))
        names = set(self._candidates.get(phase, []))
        mask = torch.tensor(
            [1 if c.__name__ in names else 0 for c in registry],
            dtype=torch.int32,
            device="cuda",
        )
        world_size = self.model.parallelism_config.world_size
        if world_size > 1:
            all_reduce(mask, group=Group.DP_AND_TP)
            counts = mask.tolist()
            dropped = [
                c.__name__ for c, n in zip(registry, counts) if 0 < n < world_size
            ]
            if dropped:
                logging.warning(
                    f"accuracy check [{phase}]: impls viable on only some ranks, "
                    f"dropped by world AND-reduce (no records): {dropped}"
                )
            mask = (mask == world_size).to(torch.int32)
        return [c.__name__ for c, m in zip(registry, mask.tolist()) if m]

    def start_run(
        self,
        scenario_base_name: str,
        impl_name: str,
        phase: str,
        batch_sequence_ids: Sequence[int],
    ) -> None:
        if self._closed:
            raise RuntimeError("tensor recorder is closed")
        if scenario_base_name != self._current_scenario_base_name:
            self._golden_history = GoldenKVHistory()
            self._candidates = {}
            self._current_scenario_base_name = scenario_base_name
        self._current_bucket = self.records.setdefault(scenario_base_name, {})
        self._impl_name = impl_name
        self._phase = phase
        self._wrapper = None
        if impl_name == "golden":
            if self._golden_history is None:
                raise RuntimeError("golden K/V history is not initialized")
            history_view = self._golden_history.bind_batch(batch_sequence_ids)
            self._factory = lambda attn_configs, attention_inputs, _parallelism_config: GoldenSDPAImpl(
                attn_configs, attention_inputs, history_view
            )
        else:
            impls = DECODE_MHA_IMPS if phase.startswith("decode") else PREFILL_MHA_IMPS
            impl_cls = next(
                candidate_cls
                for candidate_cls in impls
                if candidate_cls.__name__ == impl_name
            )
            self._factory = (
                lambda attn_configs, attention_inputs, parallelism_config: impl_cls(
                    attn_configs, attention_inputs, parallelism_config
                )
            )
        self.model.prepare_fmha_impl = self._recording_prepare
        self._patched = True

    def stop_run(self, scenario_name: str) -> None:
        self.restore()
        if self._wrapper is None or not self._wrapper.layer_records:
            raise RuntimeError(f"no records for {scenario_name}")
        if (
            self._current_bucket is None
            or self._impl_name is None
            or self._phase is None
        ):
            raise RuntimeError(f"recording state is incomplete for {scenario_name}")
        rec = AttentionForwardRecord(
            impl_name=self._impl_name,
            phase=self._phase,
            layer_records=self._wrapper.layer_records,
        )
        self._current_bucket.setdefault(self._impl_name, []).append(rec)
        self._wrapper = None

    def restore(self) -> None:
        if self._patched:
            self.model.prepare_fmha_impl = self._orig_prepare
            self._patched = False

    def close(self) -> None:
        if self._closed:
            return

        self.restore()
        self._closed = True

        # Drop recorder-owned references before releasing records
        records, self.records = self.records, {}
        self._current_bucket = None
        self._current_scenario_base_name = None
        self._golden_history = None
        self._wrapper = None
        self._impl_name = None
        self._phase = None
        self._factory = None
        self._candidates.clear()

        for impl_map in records.values():
            for record_list in impl_map.values():
                for record in record_list:
                    record.release()

    def _recording_prepare(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> RecordingWrapper:
        if self._factory is None or self._phase is None:
            raise RuntimeError("accuracy recording has not been started")
        attn_configs = self.model.config.getAttentionConfigs(
            self.model.parallelism_config.get_attn_tp_size()
        )
        # Compare all implementations without RoPE or logn
        attn_configs.rope_config.style = RopeStyle.No
        attn_configs.use_logn_attn = False
        attention_inputs = inputs.attention_inputs
        attention_inputs.is_cuda_graph = False
        # Headwise attention is outside this check
        attention_inputs.headwise_config = None
        # Probe candidates for the current input shape
        parallelism_config = self.model.parallelism_config
        supported = get_all_supported_impls(
            attn_configs,
            attention_inputs,
            self.model.fmha_config,
            parallelism_config,
        )
        viable = []
        for impl_cls in supported:
            try:
                impl_cls(attn_configs, attention_inputs, parallelism_config)
                viable.append(impl_cls.__name__)
            except Exception as e:
                logging.error(
                    f"accuracy check: candidate {impl_cls.__name__} dry-instantiate failed, excluded: {e}"
                )
        self._candidates[self._phase] = viable
        self._wrapper = RecordingWrapper(
            self._factory(attn_configs, attention_inputs, parallelism_config),
            attn_configs,
            attention_inputs,
        )
        return self._wrapper
