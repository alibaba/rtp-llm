"""CPU tests for the G5 device-side MoE segment plan.

Compares device_segment_plan against a host reference replicating the
production host-count loop (v41_moe.py contiguous path), fuzzed over expert
counts including zeros, single-expert piles, all-inactive routes, and a
parameterized alignment. Also checks the capacity/sink invariants.
"""

import importlib
import random
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
for suffix in [
    "",
    ".models_py",
    ".models_py.modules",
    ".models_py.modules.dsv4",
]:
    name = "rtp_llm" + suffix
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
    sys.modules.setdefault(name, module)

ds = importlib.import_module("rtp_llm.models_py.modules.dsv4._v41_device_segments")


def host_reference(keys, experts, align):
    """Replicates the host-count contiguous planning exactly."""
    route_count = len(keys)
    counts = [0] * (experts + 1)
    for k in keys:
        counts[k] += 1
    order = sorted(range(route_count), key=lambda j: (keys[j], j))  # stable
    sorted_e = [keys[j] for j in order]
    starts, c = [0] * (experts + 1), 0
    for e in range(experts + 1):
        starts[e] = c
        c += counts[e]
    offsets = [i - starts[sorted_e[i]] for i in range(route_count)]
    segments = [(counts[e] + align - 1) // align * align for e in range(experts)]
    seg_starts, cur = [], 0
    for s in segments:
        seg_starts.append(cur)
        cur += s
    allocated = cur
    sink_start = None
    capacity = ds.capacity_for(route_count, experts, align)
    sink = capacity - align
    dest = []
    for i, e in enumerate(sorted_e):
        dest.append(seg_starts[e] + offsets[i] if e < experts else sink)
    layout = []
    for bstart in range(0, capacity, align):
        owner, acc = -1, 0
        for e in range(experts):
            acc += segments[e]
            if bstart < acc:
                owner = e
                break
        layout.extend([owner] * align)
    return {
        "order": order,
        "dest": dest,
        "layout": layout,
        "allocated": allocated,
        "counts": counts,
        "sorted_e": sorted_e,
        "offsets": offsets,
        "sink": sink,
    }


class DeviceSegmentPlanTest(unittest.TestCase):
    def _check(self, keys, experts, align):
        ref = host_reference(keys, experts, align)
        route_count = len(keys)
        counts = torch.tensor(ref["counts"], dtype=torch.int64)
        sorted_experts = torch.tensor(ref["sorted_e"], dtype=torch.int64)
        offsets = torch.tensor(ref["offsets"], dtype=torch.int64)
        capacity = ds.capacity_for(route_count, experts, align)
        sink = capacity - align
        dest, layout, active = ds.device_segment_plan(
            counts, sorted_experts, offsets, experts, align, capacity, sink
        )
        self.assertEqual(dest.tolist(), ref["dest"])
        self.assertEqual(layout.tolist(), ref["layout"])
        self.assertEqual(active.tolist(), [e < experts for e in ref["sorted_e"]])
        # invariants
        self.assertLessEqual(ref["allocated"], sink)  # sink block is always tail
        self.assertEqual(layout[sink], -1)
        self.assertEqual(int(layout[sink : sink + align].min()), -1)
        self.assertEqual(len(layout), capacity)
        self.assertEqual(capacity % align, 0)
        # live destinations unique and inside their expert segment
        seen = set()
        for i, e in enumerate(ref["sorted_e"]):
            if e < experts:
                d = dest[i].item()
                self.assertNotIn(d, seen)
                seen.add(d)

    def test_fuzz(self):
        rng = random.Random(11)
        for align in (4, 8):
            for trial in range(120):
                experts = rng.randint(1, 6)
                route_count = rng.randint(0, 40)
                keys = [
                    rng.choice(list(range(experts)) + [experts, experts])
                    for _ in range(route_count)
                ]
                self._check(keys, experts, align)

    def test_all_inactive(self):
        self._check([3, 3, 3, 3], 3, 4)

    def test_single_pile(self):
        self._check([1] * 17 + [3] * 5, 4, 4)

    def test_exact_alignment_multiples(self):
        self._check([0] * 8 + [1] * 4, 2, 4)

    def test_capacity_bound_holds(self):
        for route_count in range(0, 64):
            for experts in (1, 4, 96):
                cap = ds.capacity_for(route_count, experts, 128)
                worst = route_count + experts * 127
                self.assertLessEqual(worst, cap - 128)
                self.assertEqual(cap % 128, 0)

    def test_flag_default_off(self):
        import os
        from unittest import mock

        env = dict(os.environ)
        env.pop("DSV41_MOE_DEVICE_SEGMENTS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(ds.device_segments_enabled())
        with mock.patch.dict(os.environ, {"DSV41_MOE_DEVICE_SEGMENTS": "1"}):
            self.assertTrue(ds.device_segments_enabled())


if __name__ == "__main__":
    unittest.main()
