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
    def __init__(self, layers, config, ep_rank, ep_size, model_id):
        self.state = layers[0]
        self.state.options = vars(config).copy()
        self.state.identity = (ep_rank, ep_size, model_id)
        self.state.runtime = weakref.ref(self)
        self.mode = config.eplb_mode
        self.counts = None

    def record_step(self, counts):
        self.state.calls.append("record")
        self.counts = counts
        self.state.sample_pointers.append(counts.data_ptr())
        if self.mode != 0:
            self.state.samples.append(counts.clone())

    def balance(self):
        self.state.calls.append("balance")
        self.state.balance_counts.append(self.counts.clone())
        changed = (
            self.mode in (2, 3)
            and len(self.state.balance_counts) in self.state.apply_on_calls
        )
        self.state.applied += int(changed)
        return changed

    def update_config(self, mode, update_time):
        self.state.calls.append(("update", mode, update_time))
        self.mode = mode

    def __del__(self):
        self.state.destroyed = bool(ctypes.pythonapi.PyGILState_Check())


def fake_runtime_module():
    module = ModuleType("rtp_llm.eplb.runtime")
    module.RuntimeExpertBalancer = FakeRuntime
    return patch.dict(sys.modules, {module.__name__: module})
