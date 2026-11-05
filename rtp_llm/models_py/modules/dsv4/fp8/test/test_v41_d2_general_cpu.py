"""CPU tests for D2 generalized compact CP (Batch 5 #9): plan geometry at
4096/8192, admission matrix with the flag off/on, and the pack/unpack wire
roundtrip including multi-span concatenation. GPU arithmetic qualification
of the 2048-row projection shapes and e2e parity run on the node."""

import importlib.util
import os
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]

# Synthetic package so the modules' relative imports resolve when loaded
# directly from file (bare-python CPU battery, no bazel).
_PKG = "_r16_d2_pkg"
_pkg = types.ModuleType(_PKG)
_pkg.__path__ = [str(ROOT)]
import sys  # noqa: E402

sys.modules[_PKG] = _pkg


def load(relpath):
    name = relpath[:-3]
    spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plan = load("_v41_compact_cp_plan.py")
transport = load("_v41_compact_cp_transport.py")
# The producer module imports the Triton codec at module level; stub it so
# project_owner's tiling contract is CPU-testable without GPU deps.
sys.modules[f"{_PKG}._v41_fp4_triton"] = types.ModuleType(f"{_PKG}._v41_fp4_triton")
producer = load("_v41_compact_cp_producer.py")

RESOLVE_KW = dict(stage=0, generation_count=0, device_type="cuda", capability=(12, 0))


def make_cp(count, n_spans, starts):
    local = count // 4
    return SimpleNamespace(
        cp_size=4,
        cp_rank=1,
        kv_cache_sharded=False,
        chunk_length=n_spans * local,
        padded_seq_len=4 * n_spans * local,
        seq_len_full=n_spans * count,
        chunk_lengths_per_req=(local,) * n_spans,
        input_lengths_global_host=(count,) * n_spans,
        prefix_lengths_host=tuple(starts),
        swa_replay_start=None,
        swa_replay_starts_host=None,
        gather_restore_positions=None,
    )


def spans(count, n_spans, starts):
    return tuple(SimpleNamespace(start=s, count=count) for s in starts)


class FlagTest(unittest.TestCase):
    def test_default_off(self):
        clean = dict(os.environ)
        clean.pop("DSV41_D2_COMPACT_GENERAL", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertFalse(plan.d2_general_enabled())
        with mock.patch.dict(os.environ, {"DSV41_D2_COMPACT_GENERAL": "1"}):
            self.assertTrue(plan.d2_general_enabled())
        for value in ("0", "true", ""):
            with mock.patch.dict(os.environ, {"DSV41_D2_COMPACT_GENERAL": value}):
                self.assertFalse(plan.d2_general_enabled())


class GeometryTest(unittest.TestCase):
    def test_legacy_4096_unchanged(self):
        p = plan.CompactCPPlan("packed", 7, 0, 0, 3, 2, 8192, 1, 2, 2)
        self.assertEqual(
            p.intervals, ((8192 + 512, 8192 + 1024), (8192 + 3072, 8192 + 3584))
        )
        self.assertEqual(p.rows, 512)
        self.assertEqual(p.header[10], 4096)
        self.assertEqual(p.send_bytes, 128 + 512 * 364 + 2 * 4112)
        self.assertEqual(p.tail_count, 0)  # rank 1

    def test_8192_geometry(self):
        p0 = plan.CompactCPPlan("packed", 7, 0, 0, 3, 2, 8192, 0, 2, 2, 8192)
        p3 = plan.CompactCPPlan("packed", 7, 0, 0, 3, 2, 8192, 3, 20, 1, 8192)
        self.assertEqual(p0.half, 1024)
        self.assertEqual(p0.local_rows, 2048)
        self.assertEqual(p0.intervals, ((8192, 9216), (8192 + 7168, 8192 + 8192)))
        self.assertEqual(p0.rows, 1024)  # 2048 // 2
        self.assertEqual(p3.rows, 2048)  # 2048 // 1
        self.assertEqual(p0.header[10], 8192)
        self.assertEqual(p0.tail_count, 2)  # ratio 2 rank 0
        self.assertEqual(p3.tail_count, 0)
        self.assertEqual(p0.send_bytes, 128 + 1024 * 364 + 2 * 4112)

    def test_invalid_chunk_tokens(self):
        with self.assertRaises(ValueError):
            plan.CompactCPPlan("packed", 7, 0, 0, 3, 2, 0, 0, 2, 2, 2048)
        with self.assertRaises(ValueError):
            plan.CompactCPPlan("packed", 7, 0, 0, 3, 2, 4097, 0, 2, 2, 4096)
        with self.assertRaises(ValueError):  # start + count over the window
            plan.CompactCPPlan("packed", 7, 0, 0, 3, 2, 32768 - 4096, 0, 2, 2, 8192)


class AdmissionTest(unittest.TestCase):
    def resolve(self, mode, cp, sp, flag):
        env = {"DSV41_D2_COMPACT_GENERAL": "1"} if flag else {}
        clean = dict(os.environ)
        clean.pop("DSV41_D2_COMPACT_GENERAL", None)
        base = {**clean, **env}
        with mock.patch.dict(os.environ, base, clear=True):
            return plan.resolve_geometry(mode, cp, sp, **RESOLVE_KW)

    def test_flag_off_keeps_legacy_admission(self):
        cp = make_cp(4096, 1, [0])
        sp = spans(4096, 1, [0])
        self.assertEqual(self.resolve("packed", cp, sp, False), ("packed", "eligible"))
        self.assertEqual(
            self.resolve("packed", make_cp(8192, 1, [0]), spans(8192, 1, [0]), False),
            ("off", "chunk-geometry"),
        )
        self.assertEqual(
            self.resolve(
                "packed", make_cp(4096, 2, [0, 4096]), spans(4096, 2, [0, 4096]), False
            ),
            ("off", "request-geometry"),
        )

    def test_flag_on_generalizes(self):
        self.assertEqual(
            self.resolve("packed", make_cp(4096, 1, [0]), spans(4096, 1, [0]), True),
            ("packed", "eligible"),
        )
        self.assertEqual(
            self.resolve("packed", make_cp(8192, 1, [0]), spans(8192, 1, [0]), True),
            ("packed", "eligible"),
        )
        self.assertEqual(
            self.resolve(
                "packed", make_cp(8192, 1, [8192]), spans(8192, 1, [8192]), True
            ),
            ("packed", "eligible"),
        )
        self.assertEqual(
            self.resolve(
                "packed",
                make_cp(4096, 2, [0, 4096]),
                spans(4096, 2, [0, 4096]),
                True,
            ),
            ("packed", "eligible"),
        )
        self.assertEqual(
            self.resolve(
                "packed",
                make_cp(8192, 2, [0, 16384]),
                spans(8192, 2, [0, 16384]),
                True,
            ),
            ("packed", "eligible"),
        )

    def test_flag_on_rejections(self):
        cases = (
            (make_cp(2048, 1, [0]), spans(2048, 1, [0]), "chunk-geometry"),
            (make_cp(8192, 1, [4096]), spans(8192, 1, [4096]), "chunk-geometry"),
            (
                make_cp(8192, 1, [32768 - 4096]),
                spans(8192, 1, [32768 - 4096]),
                "chunk-geometry",
            ),
        )
        for cp, sp, reason in cases:
            self.assertEqual(self.resolve("packed", cp, sp, True), ("off", reason))
        # mixed counts
        mixed = (
            SimpleNamespace(start=0, count=4096),
            SimpleNamespace(start=4096, count=8192),
        )
        cp = make_cp(4096, 2, [0, 4096])
        self.assertEqual(
            self.resolve("packed", cp, mixed, True), ("off", "mixed-request-geometry")
        )
        # cp tuple disagreement (chunk_length wrong for B2)
        bad = make_cp(4096, 1, [0])
        self.assertEqual(
            self.resolve("packed", bad, spans(4096, 2, [0, 4096]), True),
            ("off", "cp-geometry"),
        )


class WireRoundtripTest(unittest.TestCase):
    def make_plans(self, count, starts, layer, ratio):
        return tuple(
            tuple(
                plan.CompactCPPlan(
                    "packed", 11, 0, 0, 5, 3, start, rank, layer, ratio, count
                )
                for rank in range(4)
            )
            for start in starts
        )

    def fake_payload(self, plans_by_span, device):
        g = torch.Generator().manual_seed(3)
        metas = transport.prepare_metadata_batch(
            tuple(p for group in plans_by_span for p in group), device
        )
        sent = []
        for span_index, group in enumerate(plans_by_span):
            p = group[0]
            main = torch.randint(0, 256, (p.rows, 288), generator=g, dtype=torch.uint8)
            idx = torch.randint(-128, 127, (p.rows, 64), generator=g, dtype=torch.int8)
            scales = torch.randint(
                -(2**31), 2**31 - 1, (p.rows,), generator=g, dtype=torch.int32
            )
            tail = torch.randn((2, 1024), generator=g).float() if p.tail_count else None
            meta = metas[span_index * 4]
            packet = transport.pack(p, meta, main, idx, scales, tail)
            sent.append((packet, main, idx, scales, tail))
        return metas, sent

    def test_single_span_8192_roundtrip(self):
        plans_by_span = self.make_plans(8192, [0], 2, 2)
        metas, sent = self.fake_payload(plans_by_span, torch.device("cpu"))
        validation = transport.prepare_validation_stacks(
            tuple(plans_by_span), torch.device("cpu")
        )
        # receiver sees the SAME packet from all four senders (rank-0 plan
        # shapes identical across ranks; rebuild per-rank packets)
        packets = []
        for rank in range(4):
            p = plans_by_span[0][rank]
            meta = metas[rank]
            g = torch.Generator().manual_seed(100 + rank)
            main = torch.randint(0, 256, (p.rows, 288), generator=g, dtype=torch.uint8)
            idx = torch.randint(-128, 127, (p.rows, 64), generator=g, dtype=torch.int8)
            scales = torch.randint(
                -(2**31), 2**31 - 1, (p.rows,), generator=g, dtype=torch.int32
            )
            tail = torch.randn((2, 1024), generator=g).float() if p.tail_count else None
            packets.append(transport.pack(p, meta, main, idx, scales, tail))
        received = torch.stack(packets)
        main, index, state = transport.unpack(
            plans_by_span[0], tuple(metas[:4]), received, validation[0]
        )
        self.assertEqual(main.shape, (4 * plans_by_span[0][0].rows, 288))
        self.assertEqual(index.shape, (4 * plans_by_span[0][0].rows, 68))
        for rank in range(4):
            rows = plans_by_span[0][0].rows
            self.assertTrue(
                torch.equal(
                    main[rank * rows : (rank + 1) * rows],
                    received[rank, 128 : 128 + rows * 364].view(rows, 364)[:, 8:296],
                )
            )
        self.assertIsNotNone(state)

    def test_two_span_concat_roundtrip(self):
        plans_by_span = self.make_plans(4096, [0, 4096], 8, 2)
        metas = transport.prepare_metadata_batch(
            tuple(p for group in plans_by_span for p in group), torch.device("cpu")
        )
        validation = transport.prepare_validation_stacks(
            tuple(plans_by_span), torch.device("cpu")
        )
        size = plans_by_span[0][0].send_bytes
        blocks = []
        truths = []
        for span_index, group in enumerate(plans_by_span):
            packets = []
            for rank in range(4):
                p = group[rank]
                meta = metas[span_index * 4 + rank]
                g = torch.Generator().manual_seed(span_index * 10 + rank)
                main = torch.randint(
                    0, 256, (p.rows, 288), generator=g, dtype=torch.uint8
                )
                idx = torch.randint(
                    -128, 127, (p.rows, 64), generator=g, dtype=torch.int8
                )
                scales = torch.randint(
                    -(2**31), 2**31 - 1, (p.rows,), generator=g, dtype=torch.int32
                )
                tail = (
                    torch.randn((2, 1024), generator=g).float()
                    if p.tail_count
                    else None
                )
                packets.append(transport.pack(p, meta, main, idx, scales, tail))
                if rank == 0:
                    truths.append((main, idx))
            blocks.append(torch.stack(packets))
        # runtime concatenates per-span packets in one buffer, receiver
        # slices view(4, total) columns per span
        total = torch.cat([b.reshape(4, size) for b in blocks], dim=1)
        for span_index, group in enumerate(plans_by_span):
            block = total[:, span_index * size : (span_index + 1) * size].contiguous()
            main, index, state = transport.unpack(
                group,
                tuple(metas[span_index * 4 : (span_index + 1) * 4]),
                block,
                validation[span_index],
            )
            rows = group[0].rows
            self.assertTrue(
                torch.equal(main[:rows, :288].view(rows, 288), truths[span_index][0])
            )

    def test_validation_rejects_corrupt_header(self):
        plans_by_span = self.make_plans(4096, [0], 2, 2)
        metas = transport.prepare_metadata_batch(
            tuple(p for group in plans_by_span for p in group), torch.device("cpu")
        )
        validation = transport.prepare_validation_stacks(
            tuple(plans_by_span), torch.device("cpu")
        )
        size = plans_by_span[0][0].send_bytes
        received = torch.zeros((4, size), dtype=torch.uint8)
        for rank in range(4):
            received[rank, :128] = metas[rank].header.contiguous().view(torch.uint8)
        received[2, 40] ^= 0xFF  # corrupt one header byte
        with self.assertRaises(ValueError):
            transport.unpack(
                plans_by_span[0], tuple(metas[:4]), received, validation[0]
            )


class StrictProjection:
    """Replicates the compiled provider's admission: exactly [1024,K] input."""

    def __init__(self, out_cols):
        self.out_cols = out_cols
        self.calls = []

    def forward(self, x, w):
        if x.shape[0] != 1024:
            raise ValueError(
                f"V41CompactCpProjection: expected [1024,{x.shape[1]}] input, got {tuple(x.shape)}"
            )
        self.calls.append(int(x.shape[0]))
        return x[:, : self.out_cols].float()


class ProjectOwnerTilingTest(unittest.TestCase):
    def make(self, rows, ratio):
        x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8) * 0.5
        owner = SimpleNamespace(
            global_wkv=torch.ones(4, 8), global_wgate=torch.ones(4, 8) * 2
        )
        return x, owner

    def test_1024_path_is_a_single_qualified_call(self):
        x, owner = self.make(1024, 2)
        kv, gate = StrictProjection(4), StrictProjection(4)

        class P:
            def forward(self, xs, w):
                return (kv if w is owner.global_wkv else gate).forward(xs, w)

        values, scores = producer.project_owner(P(), x, owner, 2)
        self.assertEqual(values.shape, (1024, 4))
        self.assertEqual(scores.shape, (1024, 4))
        self.assertEqual(kv.calls, [1024])
        self.assertEqual(gate.calls, [1024])

    def test_2048_tiles_into_two_qualified_calls_in_row_order(self):
        x, owner = self.make(2048, 2)
        kv, gate = StrictProjection(4), StrictProjection(4)

        class P:
            def forward(self, xs, w):
                return (kv if w is owner.global_wkv else gate).forward(xs, w)

        values, scores = producer.project_owner(P(), x, owner, 2)
        self.assertEqual(values.shape, (2048, 4))
        self.assertEqual(scores.shape, (2048, 4))
        self.assertEqual(kv.calls, [1024, 1024])
        self.assertEqual(gate.calls, [1024, 1024])
        # row order and per-row values preserved across the concatenation
        self.assertTrue(torch.equal(values, x[:, :4]))
        self.assertTrue(torch.equal(scores, x[:, :4]))

    def test_2048_ratio1_has_no_scores(self):
        x, owner = self.make(2048, 1)
        kv = StrictProjection(4)

        class P:
            def forward(self, xs, w):
                assert w is owner.global_wkv
                return kv.forward(xs, w)

        values, scores = producer.project_owner(P(), x, owner, 1)
        self.assertEqual(values.shape, (2048, 4))
        self.assertIsNone(scores)
        self.assertEqual(kv.calls, [1024, 1024])

    def test_other_row_counts_rejected(self):
        x, owner = self.make(512, 2)

        class P:
            def forward(self, xs, w):
                return xs[:, :4].float()

        with self.assertRaises(ValueError):
            producer.project_owner(P(), x, owner, 2)
        with self.assertRaises(ValueError):
            producer.project_owner(P(), self.make(1024, 3)[0], owner, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
