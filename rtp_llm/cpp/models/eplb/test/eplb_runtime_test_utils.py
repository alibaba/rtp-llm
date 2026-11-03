"""Scripted runtime for C++ bridge/lifecycle tests; real policy has separate tests."""

import ctypes
import sys
import weakref
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class FakeModel:
    def __init__(self):
        self.state = SimpleNamespace(
            calls=[],
            samples=[],
            prompts=[],
            sample_pointers=[],
            balance_counts=[],
            apply_on_calls=set(),
            applied=0,
            destroyed=False,
        )

    def get_eplb_layers(self):
        self.state.calls.append("get_layers")
        return [self.state]


class FakeRuntime:
    def __init__(
        self,
        layers,
        config,
        ep_rank,
        ep_size,
        model_id,
        is_normal_engine=False,
        sync_prompt_phase=False,
    ):
        self.state = layers[0]
        self.state.options = vars(config).copy()
        self.state.identity = (ep_rank, ep_size, model_id)
        self.state.is_normal_engine = is_normal_engine
        self.state.sync_prompt_phase = sync_prompt_phase
        self.state.runtime = weakref.ref(self)
        self.mode = config.eplb_mode
        self.interval = config.eplb_update_time
        self.counts = None

    def step(self, counts, is_prompt=False):
        self.state.calls.append("step")
        self.state.prompts.append(is_prompt)
        self.counts = counts
        self.state.sample_pointers.append(counts.data_ptr())
        if self.mode != 0:
            self.state.samples.append(counts.clone())

        self.state.balance_counts.append(self.counts.clone())
        changed = (
            self.mode in (2, 3)
            and self.interval > 0
            and len(self.state.balance_counts) in self.state.apply_on_calls
        )
        self.state.applied += int(changed)
        return changed

    def __del__(self):
        self.state.destroyed = bool(ctypes.pythonapi.PyGILState_Check())


def fake_runtime_module():
    module = ModuleType("rtp_llm.eplb.runtime")
    module.RuntimeExpertBalancer = FakeRuntime
    return patch.dict(sys.modules, {module.__name__: module})
