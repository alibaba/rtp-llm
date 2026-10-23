"""Independent CP zigzag coverage and compact wire admission checks."""

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parents[1] / "_v41_compact_cp_plan.py"
spec = importlib.util.spec_from_file_location("v41_compact_plan", path)
plan = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plan
spec.loader.exec_module(plan)


class CompactPlanTest(unittest.TestCase):
    def test_every_owner_boundary_and_tail(self):
        for start in range(0, 32768, 4096):
            for layer, ratio in plan.PRODUCERS:
                rows = []
                for rank in range(4):
                    item = plan.CompactCPPlan(
                        "packed", -17, 2, 3, 9, 42, start, rank, layer, ratio
                    )
                    expected = [
                        start + p
                        for p in range(4096)
                        if min(p // 512, 7 - p // 512) == rank and (p + 1) % ratio == 0
                    ]
                    self.assertEqual(item.boundary_positions, tuple(expected))
                    self.assertEqual(item.rows, len(expected))
                    self.assertEqual(len(item.header), 16)
                    self.assertEqual(item.send_bytes, 194720 if ratio == 2 else 372864)
                    self.assertEqual(
                        item.tail_count, 2 if rank == 0 and ratio == 2 else 0
                    )
                    for p in expected:
                        if ratio == 2:
                            self.assertEqual(
                                min((p - start - 1) // 512, 7 - (p - start - 1) // 512),
                                rank,
                            )
                    rows.extend(expected)
                self.assertEqual(
                    sorted(rows), list(range(start + ratio - 1, start + 4096, ratio))
                )
                self.assertEqual(len(rows), len(set(rows)))

    def test_identity_header_and_rejection(self):
        item = plan.CompactCPPlan("packed", -17, 2, 3, 9, 42, 4096, 0, 2, 2)
        self.assertEqual(
            item.header, (1, 2, 0, 2, -17, 2, 3, 9, 42, 4096, 4096, 2, 0, 512, 2, 0)
        )
        for changes in (
            {"mode": "invalid"},
            {"mode": "off"},
            {"rank": 4},
            {"start": 1},
            {"start": 32768},
            {"layer": 21},
            {"ratio": 1},
            {"cache_incarnation": -1},
            {"generation": True},
            {"request_id": 1 << 63},
        ):
            with self.assertRaises(ValueError):
                replace(item, **changes)

    def test_uniform_host_fallback(self):
        cp = SimpleNamespace(
            cp_size=4,
            cp_rank=0,
            kv_cache_sharded=False,
            chunk_length=1024,
            padded_seq_len=4096,
            seq_len_full=4096,
            chunk_lengths_per_req=(1024,),
            input_lengths_global_host=(4096,),
            prefix_lengths_host=(0,),
        )
        span = SimpleNamespace(start=0, count=4096)
        kwargs = dict(
            stage=0, generation_count=0, device_type="cuda", capability=(12, 0)
        )
        self.assertEqual(
            plan.resolve_geometry("packed", cp, (span,), **kwargs),
            ("packed", "eligible"),
        )
        for field, value in (
            ("cp_size", 2),
            ("kv_cache_sharded", True),
            ("seq_len_full", 4095),
            ("chunk_lengths_per_req", (512, 512)),
            ("prefix_lengths_host", (1,)),
            ("swa_replay_start", 0),
            ("swa_replay_starts_host", (0,)),
            ("gather_restore_positions", object()),
        ):
            bad = SimpleNamespace(**{**vars(cp), field: value})
            self.assertEqual(
                plan.resolve_geometry("packed", bad, (span,), **kwargs)[0], "off"
            )
        for field, value in (
            ("stage", 1),
            ("generation_count", 1),
            ("capability", (10, 0)),
        ):
            self.assertEqual(
                plan.resolve_geometry(
                    "projected", cp, (span,), **{**kwargs, field: value}
                )[0],
                "off",
            )
        for spans in (
            (span, span),
            (SimpleNamespace(start=1, count=4096),),
            (SimpleNamespace(start=0, count=4095),),
        ):
            self.assertEqual(
                plan.resolve_geometry("projected", cp, spans, **kwargs)[0], "off"
            )
        with self.assertRaises(ValueError):
            plan.resolve_geometry("unknown", cp, (span,), **kwargs)


if __name__ == "__main__":
    unittest.main()
