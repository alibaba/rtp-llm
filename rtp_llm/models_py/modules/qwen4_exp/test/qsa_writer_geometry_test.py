"""Byte-exact + sync-free verification for the step-frozen writer geometry.

Locks the G1 fusion item: the vectorized indexer writer consumes a per-step
:class:`IndexerWriterGeometry` (token/page resolution frozen once per model
step) and, when the geometry proves the destination slots unique, replaces the
per-call eager chain (``nonzero`` compression, masked-select gathers,
``argsort``/``unique`` dedup, implicit DtoH syncs) with a full-shape
masked-scatter write.

Checks:
* randomized and boundary geometries (B > 1, cross-page groups, prefix starts,
  ratio/page boundaries, capture_undo on/off, all three rope table styles,
  ``invalid_block_policy`` raise/skip): the geometry path writes pools that are
  byte-equal to the legacy eager path on identical inputs, returns identical
  result rows/counts, and its undo restores the exact pre-call bytes;
* stale/rebound/inapplicable geometry falls back to the legacy path instead of
  writing through a stale bundle;
* ring wrap (a request longer than the state ring) keeps the deduplicating
  legacy write (geometry present but provably not unique);
* the fast path launches no nonzero/unique/argsort kernel sequence and its
  kernel count stays far below the legacy chain;
* the step plans freeze a bundle their production writer call sites accept
  (the plan's own frozen tensors for the incremental fallback, the raw engine
  cu tensor for the cold writer), while strict validation and ring-spanning
  windows keep the bundle unset.
"""

import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
    _STEP_PLAN_CACHE,
    build_cold_step_plan,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa import indexer_compressor
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    build_indexer_writer_geometry,
    restore_indexer_cache,
    write_indexer_cache,
    writer_geometry_applies,
)

EPS = 1e-6
ROTARY = 64
D = 128


def _rotary_rows(positions, device):
    angles = (
        positions.to(torch.float32)[:, None]
        * torch.tensor([0.031, 0.017], device=device)
    )
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos.to(torch.bfloat16), sin.to(torch.bfloat16)


class IndexerWriterGeometryTest(unittest.TestCase):
    def setUp(self):
        # Some shared test helpers set the strict-validation flag process-wide
        # at import time; the fast path under test only exists in the default
        # (non-strict) mode, so pin it for this class regardless of the order
        # in which other test modules were loaded into the same process.
        patcher = patch.dict(os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build_case(self, rng, *, allow_holes=False, wrap=False):
        ratio = rng.choice((2, 4, 4, 8))
        ring = 2 * ratio
        kv_tokens_per_block = ratio * rng.choice((2, 4, 16))
        state_tokens_per_block = rng.choice((8, 16, 64))
        batch = rng.randint(1, 5)
        span = 2 * ring + 5 if wrap else ring
        lengths = [rng.randint(0, span) for _ in range(batch)]
        if not any(lengths):
            lengths[rng.randrange(batch)] = 1
        if wrap:
            lengths[0] = ring + 1 + rng.randint(0, 4)
        prefixes = [
            rng.choice((0, 1, ratio - 1, ratio, 2 * ratio + 1, 31, 64))
            for _ in range(batch)
        ]
        max_pos = max(
            prefix + length for prefix, length in zip(prefixes, lengths)
        )
        needed_pages = max(
            (prefix + length + tokens - 1) // tokens
            for prefix, length in zip(prefixes, lengths)
            for tokens in (kv_tokens_per_block, state_tokens_per_block)
        )
        width = needed_pages + 2

        # Distinct physical blocks per request (the allocator invariant); an
        # optional zero column exercises 'skip' semantics.
        hole_kv = rng.randrange(width) if allow_holes else -1
        hole_state = rng.randrange(width) if allow_holes else -1
        kv_rows = []
        state_rows = []
        next_kv_block = 1
        next_state_block = 1
        for _ in range(batch):
            kv_rows.append(
                [0 if column == hole_kv else next_kv_block + column
                 for column in range(width)]
            )
            state_rows.append(
                [0 if column == hole_state else next_state_block + column
                 for column in range(width)]
            )
            next_kv_block += width + 1
            next_state_block += width + 1
        return {
            "ratio": ratio,
            "ring": ring,
            "kv_tokens_per_block": kv_tokens_per_block,
            "state_tokens_per_block": state_tokens_per_block,
            "lengths": lengths,
            "prefixes": prefixes,
            "max_pos": max_pos + 2,
            "width": width,
            "kv_blocks": next_kv_block + width + 4,
            "state_blocks": next_state_block + width + 4,
            "kv_table": torch.tensor(kv_rows, dtype=torch.int32),
            "state_table": torch.tensor(state_rows, dtype=torch.int32),
            "rope_style": rng.choice(
                ("token_aligned", "shared", "request_specific")
            ),
            "capture_undo": rng.random() < 0.7,
        }

    def _scenario(self, case, device, seed, policy="raise"):
        """One frozen input set: identical copies for reference and fast runs."""
        lengths = case["lengths"]
        prefixes = case["prefixes"]
        token_count = sum(lengths)
        ratio = case["ratio"]
        kv_entries = case["kv_tokens_per_block"] // ratio
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kv_pool = torch.zeros(
            case["kv_blocks"], kv_entries, D, dtype=torch.bfloat16
        )
        state_pool = torch.zeros(
            case["state_blocks"], case["ring"], D, dtype=torch.float32
        )
        kv_pool.uniform_(-2, 2, generator=generator)
        state_pool.normal_(generator=generator)
        gamma = torch.rand(D, generator=generator)
        raw = torch.randn(
            token_count, D, dtype=torch.float32, generator=generator
        ).to(torch.bfloat16)

        positions = []
        for prefix, length in zip(prefixes, lengths):
            positions.extend(range(prefix, prefix + length))
        cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
        )
        starts = torch.tensor(prefixes, dtype=torch.int64)
        if case["rope_style"] == "token_aligned":
            if positions:
                rope_cos, rope_sin = _rotary_rows(
                    torch.tensor(positions, dtype=torch.long), "cpu"
                )
            else:
                rope_cos = torch.zeros(0, ROTARY, dtype=torch.bfloat16)
                rope_sin = torch.zeros(0, ROTARY, dtype=torch.bfloat16)
            rope_is_token_aligned = True
        else:
            rope_cos, rope_sin = _rotary_rows(
                torch.arange(case["max_pos"], dtype=torch.long), "cpu"
            )
            if case["rope_style"] == "request_specific":
                rope_cos = rope_cos.unsqueeze(0).expand(len(lengths), -1, -1)
                rope_sin = rope_sin.unsqueeze(0).expand(len(lengths), -1, -1)
            rope_is_token_aligned = False

        scenario = {
            "raw": raw.to(device),
            "cu": cu.to(device),
            "starts": starts.to(device),
            "rope_cos": rope_cos.to(device),
            "rope_sin": rope_sin.to(device),
            "gamma": gamma.to(device),
            "kv_table": case["kv_table"].to(device),
            "state_table": case["state_table"].to(device),
            "rope_is_token_aligned": rope_is_token_aligned,
            "token_count": token_count,
            "kv_before": kv_pool.to(device),
            "state_before": state_pool.to(device),
        }
        scenario["geometry"] = build_indexer_writer_geometry(
            scenario["cu"],
            scenario["starts"],
            scenario["kv_table"],
            scenario["state_table"],
            token_count=token_count,
            ratio=ratio,
            kv_tokens_per_block=case["kv_tokens_per_block"],
            state_tokens_per_block=case["state_tokens_per_block"],
            state_ring_entries=case["ring"],
            kv_pool_blocks=case["kv_blocks"],
            state_pool_blocks=case["state_blocks"],
            invalid_block_policy=policy,
            host_lengths=lengths,
            host_prefixes=prefixes,
        )
        return scenario

    def _write(self, case, scenario, *, geometry, policy="raise"):
        pools = {
            "kv": scenario["kv_before"].clone(),
            "state": scenario["state_before"].clone(),
        }
        result = write_indexer_cache(
            scenario["raw"],
            scenario["cu"],
            scenario["starts"],
            scenario["rope_cos"],
            scenario["rope_sin"],
            scenario["gamma"],
            pools["kv"],
            scenario["kv_table"],
            pools["state"],
            scenario["state_table"],
            ratio=case["ratio"],
            kv_tokens_per_block=case["kv_tokens_per_block"],
            state_tokens_per_block=case["state_tokens_per_block"],
            norm_eps=EPS,
            invalid_block_policy=policy,
            capture_undo=case["capture_undo"],
            rope_is_token_aligned=scenario["rope_is_token_aligned"],
            geometry=geometry,
        )
        return result, pools

    def _assert_geometry_matches_legacy(self, case, device, policy="raise"):
        scenario = self._scenario(case, device, seed=1234, policy=policy)
        if max(case["lengths"]) <= case["ring"]:
            self.assertTrue(
                scenario["geometry"].state_slots_unique,
                "short spans must prove state slots unique",
            )
        else:
            self.assertFalse(
                scenario["geometry"].state_slots_unique,
                "ring-wrapping spans cannot prove uniqueness",
            )
        reference, reference_pools = self._write(
            case, scenario, geometry=None, policy=policy
        )
        fast, fast_pools = self._write(
            case, scenario, geometry=scenario["geometry"], policy=policy
        )
        self.assertTrue(
            torch.equal(reference_pools["kv"], fast_pools["kv"]),
            "kv pool bytes differ from the legacy path",
        )
        self.assertTrue(
            torch.equal(reference_pools["state"], fast_pools["state"]),
            "state pool bytes differ from the legacy path",
        )
        for key in ("state_slots", "kv_slots", "completed"):
            self.assertTrue(
                torch.equal(reference[key], fast[key]), f"{key} rows differ"
            )
        self.assertEqual(
            int(reference["num_state_writes"]), int(fast["num_state_writes"])
        )
        self.assertEqual(
            int(reference["num_kv_writes"]), int(fast["num_kv_writes"])
        )
        if case["capture_undo"]:
            # The undo must restore the exact pre-call bytes.
            restore_indexer_cache(fast["undo"])
            self.assertTrue(
                torch.equal(fast_pools["kv"], scenario["kv_before"]),
                "undo did not restore the kv pool bytes",
            )
            self.assertTrue(
                torch.equal(fast_pools["state"], scenario["state_before"]),
                "undo did not restore the state pool bytes",
            )

    def test_geometry_path_matches_legacy_byte_exact_randomized(self):
        rng = random.Random(20261109)
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            for step in range(80):
                case = self._build_case(rng)
                if max(case["lengths"]) > case["ring"]:
                    continue
                with self.subTest(device=device, step=step):
                    self._assert_geometry_matches_legacy(case, device)

    def test_geometry_path_matches_legacy_for_skip_policy(self):
        rng = random.Random(4242)
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            for step in range(40):
                case = self._build_case(rng, allow_holes=True)
                if max(case["lengths"]) > case["ring"]:
                    continue
                with self.subTest(device=device, step=step):
                    self._assert_geometry_matches_legacy(case, device, policy="skip")

    def test_ring_wrap_still_matches_legacy(self):
        # A request longer than the state ring wraps the ring: the geometry is
        # present but not provably unique, so the write must take the
        # deduplicating legacy path and still match the reference bytes.
        rng = random.Random(777)
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            for step in range(20):
                case = self._build_case(rng, wrap=True)
                with self.subTest(device=device, step=step):
                    self._assert_geometry_matches_legacy(case, device)

    def test_boundary_geometries(self):
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        ratio = 4
        common = dict(
            ratio=ratio,
            ring=8,
            kv_tokens_per_block=16,
            state_tokens_per_block=8,
            width=4,
            kv_blocks=24,
            state_blocks=24,
        )
        cases = [
            # single token completing nothing
            dict(common, lengths=[1], prefixes=[0], max_pos=16,
                 kv_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
                 state_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
                 rope_style="shared", capture_undo=True),
            # group completing exactly at a page edge
            dict(common, lengths=[4], prefixes=[12], max_pos=24,
                 kv_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
                 state_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
                 rope_style="token_aligned", capture_undo=False),
            # ragged batch: one completes from current rows only, one reaches
            # back into committed state, one completes nothing
            dict(common, lengths=[4, 3, 2], prefixes=[16, 5, 7], max_pos=24,
                 kv_table=torch.tensor(
                     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                     dtype=torch.int32),
                 state_table=torch.tensor(
                     [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
                     dtype=torch.int32),
                 rope_style="request_specific", capture_undo=True),
            # empty window (all lengths zero)
            dict(common, lengths=[0, 0], prefixes=[3, 9], max_pos=16,
                 kv_table=torch.tensor([[1, 2, 3, 4]] * 2, dtype=torch.int32),
                 state_table=torch.tensor([[1, 2, 3, 4]] * 2, dtype=torch.int32),
                 rope_style="shared", capture_undo=True),
            # full ring span per request (the uniqueness boundary)
            dict(common, lengths=[8, 8], prefixes=[0, 64], max_pos=80,
                 width=12, kv_blocks=64, state_blocks=64,
                 kv_table=torch.tensor(
                     [list(range(1, 13)), list(range(13, 25))], dtype=torch.int32),
                 state_table=torch.tensor(
                     [list(range(25, 37)), list(range(37, 49))], dtype=torch.int32),
                 rope_style="token_aligned", capture_undo=True),
        ]
        cases.append(
            dict(
                ratio=2, ring=4, kv_tokens_per_block=4,
                state_tokens_per_block=8, width=4, kv_blocks=16,
                state_blocks=16, lengths=[2, 4], prefixes=[0, 3], max_pos=16,
                kv_table=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]],
                                      dtype=torch.int32),
                state_table=torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]],
                                         dtype=torch.int32),
                rope_style="shared", capture_undo=False,
            )
        )
        for case in cases:
            for device in devices:
                with self.subTest(device=device, lengths=case["lengths"]):
                    self._assert_geometry_matches_legacy(case, device)

    def test_wrap_geometry_does_not_use_full_shape_path(self):
        case = dict(
            ratio=4, ring=8, kv_tokens_per_block=16, state_tokens_per_block=8,
            lengths=[12], prefixes=[0], max_pos=20, width=4, kv_blocks=16,
            state_blocks=16,
            kv_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            state_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            rope_style="shared", capture_undo=True,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        scenario = self._scenario(case, device, seed=11)
        self.assertFalse(scenario["geometry"].state_slots_unique)
        with patch.object(
            indexer_compressor, "_write_indexer_cache_full_shape"
        ) as fast_path:
            self._write(case, scenario, geometry=scenario["geometry"])
        fast_path.assert_not_called()

    def test_stale_geometry_falls_back_to_legacy(self):
        case = dict(
            ratio=4, ring=8, kv_tokens_per_block=16, state_tokens_per_block=8,
            lengths=[4], prefixes=[0], max_pos=16, width=4, kv_blocks=16,
            state_blocks=16,
            kv_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            state_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            rope_style="token_aligned", capture_undo=True,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        scenario = self._scenario(case, device, seed=5)
        geometry = scenario["geometry"]
        # A rebound state table invalidates the bundle: the writer falls back.
        stale_scenario = dict(scenario)
        stale_scenario["state_table"] = torch.tensor(
            [[5, 6, 7, 8]], dtype=torch.int32, device=device
        )
        with patch.object(
            indexer_compressor, "_write_indexer_cache_full_shape"
        ) as fast_path:
            result, pools = self._write(
                case, stale_scenario, geometry=geometry
            )
        fast_path.assert_not_called()
        # The write used the rebound table's blocks (legacy semantics).
        reference, reference_pools = self._write(
            case, stale_scenario, geometry=None
        )
        self.assertTrue(torch.equal(pools["kv"], reference_pools["kv"]))
        self.assertTrue(torch.equal(pools["state"], reference_pools["state"]))
        _ = result

        # The applicability helper rejects mismatched cu/policy inputs.
        self.assertFalse(
            writer_geometry_applies(
                geometry,
                torch.zeros(2, dtype=torch.int32, device=device),
                scenario["starts"],
                scenario["kv_table"],
                scenario["state_table"],
                scenario["kv_before"],
                scenario["state_before"],
                token_count=4,
                ratio=4,
                kv_tokens_per_block=16,
                state_tokens_per_block=8,
                invalid_block_policy="raise",
            )
        )
        self.assertFalse(
            writer_geometry_applies(
                geometry,
                scenario["cu"],
                scenario["starts"],
                scenario["kv_table"],
                scenario["state_table"],
                scenario["kv_before"],
                scenario["state_before"],
                token_count=4,
                ratio=4,
                kv_tokens_per_block=16,
                state_tokens_per_block=8,
                invalid_block_policy="skip",
            )
        )

    def test_geometry_is_reusable_across_writes(self):
        # The frozen bundle is content-independent: repeated writes with fresh
        # pool copies consume one geometry object (the layer-loop pattern).
        case = dict(
            ratio=4, ring=8, kv_tokens_per_block=16, state_tokens_per_block=8,
            lengths=[4], prefixes=[0], max_pos=16, width=4, kv_blocks=16,
            state_blocks=16,
            kv_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            state_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            rope_style="token_aligned", capture_undo=False,
        )
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                scenario = self._scenario(case, device, seed=9)
                geometry = scenario["geometry"]
                first, first_pools = self._write(case, scenario, geometry=geometry)
                second, second_pools = self._write(
                    case, scenario, geometry=geometry
                )
                # Identical inputs (fresh pool copies) give identical bytes.
                self.assertTrue(
                    torch.equal(first_pools["kv"], second_pools["kv"])
                )
                self.assertTrue(
                    torch.equal(first_pools["state"], second_pools["state"])
                )
                self.assertTrue(
                    torch.equal(first["kv_slots"], second["kv_slots"])
                )

    def test_geometry_requires_valid_type(self):
        raw = torch.zeros(0, D, dtype=torch.bfloat16)
        cu = torch.zeros(1, dtype=torch.int32)
        starts = torch.zeros(0, dtype=torch.int64)
        cos = torch.zeros(0, ROTARY, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "geometry must be"):
            write_indexer_cache(
                raw,
                cu,
                starts,
                cos,
                cos,
                torch.rand(D),
                torch.zeros(2, 1, D, dtype=torch.bfloat16),
                torch.ones(1, 2, dtype=torch.int32),
                torch.zeros(2, 4, D),
                torch.ones(1, 2, dtype=torch.int32),
                ratio=4,
                kv_tokens_per_block=16,
                state_tokens_per_block=8,
                geometry=object(),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_geometry_fast_path_is_sync_free_and_bounded(self):
        from torch.profiler import ProfilerActivity, profile

        case = dict(
            ratio=4, ring=8, kv_tokens_per_block=64, state_tokens_per_block=64,
            lengths=[4, 3, 2, 1], prefixes=[64, 65, 127, 257], max_pos=300,
            width=8, kv_blocks=64, state_blocks=64,
            kv_table=torch.arange(1, 33, dtype=torch.int32).view(4, 8),
            state_table=torch.arange(33, 65, dtype=torch.int32).view(4, 8),
            rope_style="token_aligned", capture_undo=True,
        )
        device = "cuda:0"
        scenario = self._scenario(case, device, seed=3)
        self.assertTrue(scenario["geometry"].state_slots_unique)
        with patch.object(
            indexer_compressor,
            "_write_indexer_cache_full_shape",
            wraps=indexer_compressor._write_indexer_cache_full_shape,
        ) as fast_path:
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                self._write(case, scenario, geometry=scenario["geometry"])
                torch.cuda.synchronize()
            fast_path.assert_called_once()
        banned = ("nonzero", "unique", "argsort", "sort", "_local_scalar")
        kernels = []
        for event in prof.key_averages():
            lowered = event.key.lower()
            for marker in banned:
                self.assertNotIn(marker, lowered, f"fast path used {event.key}")
            if (
                event.device_type == torch.autograd.DeviceType.CUDA
                and "memcpy" not in lowered
            ):
                kernels.append((event.key, event.count))
        launches = sum(count for _, count in kernels)
        self.assertLessEqual(launches, 40, kernels)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_geometry_launch_count_beats_legacy(self):
        from torch.profiler import ProfilerActivity, profile

        case = dict(
            ratio=4, ring=8, kv_tokens_per_block=64, state_tokens_per_block=64,
            lengths=[4, 3, 2, 1], prefixes=[64, 65, 127, 257], max_pos=300,
            width=8, kv_blocks=64, state_blocks=64,
            kv_table=torch.arange(1, 33, dtype=torch.int32).view(4, 8),
            state_table=torch.arange(33, 65, dtype=torch.int32).view(4, 8),
            rope_style="token_aligned", capture_undo=True,
        )
        device = "cuda:0"
        scenario = self._scenario(case, device, seed=3)

        def count_launches(geometry):
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                self._write(case, scenario, geometry=geometry)
                torch.cuda.synchronize()
            return sum(
                event.count
                for event in prof.key_averages()
                if event.device_type == torch.autograd.DeviceType.CUDA
                and "memcpy" not in event.key.lower()
            )

        legacy = count_launches(None)
        fast = count_launches(scenario["geometry"])
        self.assertLess(fast, legacy * 0.6, (legacy, fast))


class StepPlanWriterGeometryIntegrationTest(unittest.TestCase):
    """Step plans freeze geometries their production writer calls accept.

    The draft-tail fallback writer receives the step plan's own frozen
    tensors (``plan.cu_seqlens`` / ``plan.prefixes``), and the cold writer
    receives the raw engine cu tensor plus the plan's ``prefixes_i64``; both
    must pass the bundle's identity + fingerprint revalidation, or every
    layer would silently pay the legacy chain again.  Strict validation and
    windows spanning past the state ring keep the bundle unset.
    """

    def _geometry_spy(self):
        return patch.object(
            indexer_compressor,
            "build_indexer_writer_geometry",
            wraps=indexer_compressor.build_indexer_writer_geometry,
        )

    def _writer_pools(self, context, plan, device):
        """Production-shaped typed pool views (``applies`` reads only shapes)."""
        kv_pool = torch.zeros(
            int(context.indexer_kv_cache.kv_cache_base.shape[0]),
            int(plan.kv_entries_per_block),
            int(plan.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        state_pool = torch.zeros(
            int(context.indexer_state_cache.kv_cache_base.shape[0]),
            2 * int(plan.ratio),
            int(plan.head_dim),
            dtype=torch.float32,
            device=device,
        )
        return kv_pool, state_pool

    def test_incremental_step_plan_freezes_applicable_writer_geometry(self):
        from qsa_draft_tail_vector_test import QsaDraftTailVectorTest

        fixtures = QsaDraftTailVectorTest.__new__(QsaDraftTailVectorTest)
        prefixes = [16, 24, 32]
        cases = (
            ("short", [3, 4, 2], "", True),
            ("strict", [3, 4, 2], "1", False),
            ("ring-wrap", [3, 9, 2], "", False),
        )
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            for case, lengths, strict, present in cases:
                with self.subTest(device=device, case=case):
                    _STEP_PLAN_CACHE.clear()
                    context = fixtures._incremental_context(
                        lengths=lengths,
                        prefixes=prefixes,
                        rows=fixtures._tail_rows(lengths, prefixes),
                        device=device,
                    )
                    with patch.dict(
                        os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": strict}
                    ), self._geometry_spy() as builder:
                        plan = fixtures._build_incremental(
                            context, sum(lengths), device
                        )
                    if not present:
                        self.assertIsNone(plan.writer_geometry)
                        builder.assert_not_called()
                        continue
                    builder.assert_called_once()
                    # The incremental build proved the cu partition and both
                    # tables, so the bundle is built without the redundant
                    # resolve asserts.
                    self.assertTrue(builder.call_args.kwargs["trusted_tables"])
                    geometry = plan.writer_geometry
                    self.assertEqual(geometry.lengths, tuple(lengths))
                    self.assertTrue(geometry.state_slots_unique)
                    self.assertIs(geometry.cu_seqlens, plan.cu_seqlens)
                    self.assertIs(geometry.start_positions, plan.prefixes_i64)
                    kv_pool, state_pool = self._writer_pools(context, plan, device)
                    self.assertTrue(
                        writer_geometry_applies(
                            geometry,
                            plan.cu_seqlens,
                            plan.prefixes_i64,
                            context.indexer_kv_inputs.kv_cache_kernel_block_id_device,
                            context.indexer_state_inputs.kv_cache_block_id_device,
                            kv_pool,
                            state_pool,
                            token_count=int(plan.token_count),
                            ratio=int(plan.ratio),
                            kv_tokens_per_block=int(plan.kv_tokens_per_block),
                            state_tokens_per_block=int(plan.state_tokens_per_block),
                            invalid_block_policy="raise",
                        )
                    )

    def _cold_context(self, lengths, device):
        from qsa_draft_tail_vector_test import D, F, PAGE, RATIO

        rows = []
        for length in lengths:
            for position in range(length):
                rows.extend([position] * F)
        position_ids = torch.tensor(rows, dtype=torch.int32).reshape(-1)
        input_lengths = torch.tensor(lengths, dtype=torch.int32)
        prefix_lengths = torch.zeros(len(lengths), dtype=torch.int32)
        boundaries = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
        device_cu = torch.tensor(boundaries, dtype=torch.int32, device=device)

        def make_inputs():
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=torch.empty(0, dtype=torch.int32),
                cu_seqlens_device=device_cu,
                cu_kv_seqlens_device=device_cu,
                combo_position_ids=position_ids.to(device),
                kv_cache_kernel_block_id_device=torch.tensor(
                    [[1, 2], [3, 4]], dtype=torch.int32, device=device
                ),
                kv_cache_block_id_device=torch.tensor(
                    [[5, 6], [7, 8]], dtype=torch.int32, device=device
                ),
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=make_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=torch.zeros(
                    16, PAGE // RATIO * D * 2, dtype=torch.uint8, device=device
                ),
                seq_size_per_block=PAGE,
            ),
            indexer_kv_inputs=make_inputs(),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=torch.zeros(
                    16, 2 * RATIO * D, dtype=torch.float32, device=device
                ),
                seq_size_per_block=PAGE,
            ),
            indexer_state_inputs=make_inputs(),
        )

    def test_cold_step_plan_freezes_applicable_writer_geometry(self):
        from qsa_draft_tail_vector_test import (
            QsaDraftTailVectorTest,
            _mrope_config,
        )

        fixtures = QsaDraftTailVectorTest.__new__(QsaDraftTailVectorTest)
        cases = (
            ("short", [3, 5], "", True),
            ("strict", [3, 5], "1", False),
            ("long", [9, 2], "", False),
        )
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            for case, lengths, strict, present in cases:
                with self.subTest(device=device, case=case):
                    _STEP_PLAN_CACHE.clear()
                    context = self._cold_context(lengths, device)
                    with patch.dict(
                        os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": strict}
                    ), self._geometry_spy() as builder:
                        plan = build_cold_step_plan(
                            context,
                            indexer=fixtures._incremental_indexer(device),
                            token_count=sum(lengths),
                            device=torch.device(device),
                            rope_config=_mrope_config(),
                            rope_dtype=torch.bfloat16,
                        )
                    if not present:
                        self.assertIsNone(plan.writer_geometry)
                        builder.assert_not_called()
                        continue
                    builder.assert_called_once()
                    # Direct cold builds keep the full resolve asserts.
                    self.assertFalse(builder.call_args.kwargs.get("trusted_tables"))
                    geometry = plan.writer_geometry
                    self.assertEqual(geometry.lengths, tuple(lengths))
                    self.assertTrue(geometry.state_slots_unique)
                    # The cold writer receives the raw engine cu tensor and
                    # the plan's prefix starts: the bundle must accept
                    # exactly those tensors, not a rebuilt copy.
                    self.assertIs(
                        geometry.cu_seqlens, context.main_inputs.cu_seqlens_device
                    )
                    self.assertIs(geometry.start_positions, plan.prefixes_i64)
                    kv_pool, state_pool = self._writer_pools(context, plan, device)
                    self.assertTrue(
                        writer_geometry_applies(
                            geometry,
                            context.main_inputs.cu_seqlens_device,
                            plan.prefixes_i64,
                            context.indexer_kv_inputs.kv_cache_kernel_block_id_device,
                            context.indexer_state_inputs.kv_cache_block_id_device,
                            kv_pool,
                            state_pool,
                            token_count=int(plan.token_count),
                            ratio=int(plan.ratio),
                            kv_tokens_per_block=int(plan.kv_tokens_per_block),
                            state_tokens_per_block=int(plan.state_tokens_per_block),
                            invalid_block_policy="raise",
                        )
                    )


if __name__ == "__main__":
    unittest.main()
