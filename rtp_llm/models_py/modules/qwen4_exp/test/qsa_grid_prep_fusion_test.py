"""Fused QSA step-plan grid/selection-padding prep tests.

``build_incremental_step_plan`` previously issued the per-step grid and
selection-padding arithmetic as a chain of small aten launches
(arange/add/where/floor_divide/remainder/fill/index_select per step-plan
window).  The fused prep replaces that chain with

* one host computation of the [B]/[B+1]-grade outputs (cu boundaries, cu_kv
  boundaries, required KV columns, required state columns, int32 prefix and
  length mirrors) uploaded as ONE pinned staging batch, and
* ONE Triton launch writing the [B, max_len]/[T]-grade tensors (visible
  lengths, compressed lengths, validity mask, selection row map, packed valid
  rows and both score-weight tables); block starts and MRoPE block sources
  ride the same pinned int64 batch as the logical positions.

These tests lock three properties:

1. every produced tensor equals the legacy eager reference exactly
   (``torch.equal``), across randomized batch/length/prefix geometries
   (B=1..8, lengths with and without ratio remainders, page-aligned
   prefixes, varied max_len) for the prefix-reuse, draft and CPU paths;
2. the batched prep launches a bounded number of device kernels and the
   arange/where/floor_divide/remainder/fill chain is gone;
3. dtype/shape invariants of every plan field.

The eager path stays as the fallback for unsupported geometry (CPU fixtures
and any geometry the Triton prep declines) and must stay byte-exact with
the fused plan.
"""

import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa import grid_prep_triton
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
)
from rtp_llm.ops import RopeStyle

D = 4
RATIO = 4
PAGE = 8


def _variant_total(fn):
    """Compiled variant count across devices (triton>=3.6 ``device_caches``)."""
    total = 0
    caches = getattr(fn, "device_caches", None) or {}
    for cache in caches.values():
        entry = cache[0] if isinstance(cache, tuple) else cache
        total += len(entry)
    return total
HEADS = 2
TOKEN_BUDGET = 64


def _rope_mrope():
    return SimpleNamespace(
        style=RopeStyle.Mrope,
        index_factor=3,
        dim=4,
        mrope_dim1=1,
        mrope_dim2=1,
        mrope_dim3=0,
        mrope_interleaved=True,
        indexer_is_neox_style=True,
        base=10_000.0,
    )


def _rope_base():
    return SimpleNamespace(
        style=RopeStyle.Base,
        index_factor=3,
        dim=4,
        indexer_is_neox_style=True,
        base=10_000.0,
        scale=1.0,
    )


def _indexer():
    return SimpleNamespace(
        head_dim=D,
        n_heads=HEADS,
        compress_ratio=RATIO,
        token_budget=TOKEN_BUDGET,
        k_norm_gamma=torch.rand(D),
        norm_eps=1e-6,
    )


def _cuda_device():
    """Indexed CUDA device: fixtures and the build must agree on ``cuda:i``."""
    return torch.device("cuda", torch.cuda.current_device())


def _context(prefixes, lengths, device, *, is_mtp_draft=False):
    """One per-layer context with device-resident metadata fixtures."""
    batch = len(lengths)
    input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device=device)
    sequence_lengths = torch.empty(0, dtype=torch.int32, device=device)
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu[1:] = input_lengths.cumsum(0)
    cu_kv = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_kv[1:] = (prefix_lengths + input_lengths).cumsum(0)
    positions = []
    for prefix, length in zip(prefixes, lengths):
        positions.extend([pos, pos, pos] for pos in range(prefix, prefix + length))
    position_ids = torch.tensor(positions, dtype=torch.int32, device=device)
    position_ids = position_ids.reshape(-1)
    kv_table = torch.arange(1, batch * 3 + 1, dtype=torch.int32, device=device)
    kv_table = kv_table.view(batch, 3)
    state_table = kv_table.clone()

    def _inputs():
        return SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            sequence_lengths=sequence_lengths,
            cu_seqlens_device=cu,
            cu_kv_seqlens_device=cu_kv,
            combo_position_ids=position_ids,
            kv_cache_kernel_block_id_device=kv_table,
            kv_cache_block_id_device=state_table,
        )

    return Qwen4ExpQSARuntimeContext(
        main_cache=SimpleNamespace(tag="full"),
        main_inputs=_inputs(),
        indexer_kv_cache=SimpleNamespace(
            tag=INDEXER_KV_TAG,
            kv_cache_base=torch.zeros(
                8, (PAGE // RATIO) * D * 2, dtype=torch.uint8, device=device
            ),
            seq_size_per_block=PAGE,
        ),
        indexer_kv_inputs=_inputs(),
        indexer_state_cache=SimpleNamespace(
            tag=INDEXER_STATE_TAG,
            kv_cache_base=torch.zeros(8, 2 * RATIO * D, dtype=torch.float32),
            seq_size_per_block=PAGE,
        ),
        indexer_state_inputs=_inputs(),
        is_mtp_draft=is_mtp_draft,
    )


def _build_plan(prefixes, lengths, device, *, draft=False, rope_config=None):
    context = _context(prefixes, lengths, device, is_mtp_draft=draft)
    return qsa_step_plan.build_incremental_step_plan(
        context,
        indexer=_indexer(),
        token_count=sum(lengths),
        device=device,
        rope_config=rope_config if rope_config is not None else _rope_mrope(),
        draft=draft,
        draft_prefix_reuse=False,
    )


def _legacy_reference(lengths, prefix_values, device, position_ids=None):
    """The pre-fusion aten chain, verbatim, as the ground truth."""
    batch = len(lengths)
    max_len = max(lengths)
    tokens = sum(lengths)
    kv_entries_per_block = PAGE // RATIO
    state_tokens_per_block = PAGE
    prefixes = torch.tensor(prefix_values, dtype=torch.int64, device=device)
    lengths_t = torch.tensor(lengths, dtype=torch.int64, device=device)
    visible_ends = prefixes + lengths_t
    expected_cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    expected_cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    expected_cu_kv = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    expected_cu_kv[1:] = visible_ends.to(torch.int32).cumsum(0)

    row_offsets = torch.arange(max_len, dtype=torch.int64, device=device)
    visible_lengths = prefixes[:, None] + row_offsets[None, :] + 1
    valid_rows = row_offsets[None, :] < lengths_t[:, None]
    visible_lengths = torch.where(valid_rows, visible_lengths, 0)
    visible_lengths_i32 = visible_lengths.reshape(-1).to(torch.int32)
    compressed_lengths = (visible_lengths // RATIO).to(torch.int32)
    final_compressed_lengths = (visible_ends // RATIO).to(torch.int32)
    required_kv_columns = (
        final_compressed_lengths + kv_entries_per_block - 1
    ) // kv_entries_per_block
    required_state_columns = torch.tensor(
        [
            (p + l + state_tokens_per_block - 1) // state_tokens_per_block
            for p, l in zip(prefix_values, lengths)
        ],
        dtype=torch.int32,
    )
    logical_positions = torch.cat(
        [
            torch.arange(p, p + l, device=device)
            for p, l in zip(prefix_values, lengths)
        ]
    ).to(torch.int64)
    block_starts = logical_positions - logical_positions.remainder(RATIO)
    mrope_sources = torch.arange(tokens, device=device) - logical_positions.remainder(
        RATIO
    )

    offsets = []
    valid_rows_list = []
    offset = 0
    for request_idx, length in enumerate(lengths):
        offsets.extend(offset + min(row, length - 1) for row in range(max_len))
        valid_rows_list.extend(request_idx * max_len + row for row in range(length))
        offset += length
    selection_row_map = torch.tensor(offsets, dtype=torch.int64, device=device)
    selection_valid_rows = torch.tensor(valid_rows_list, dtype=torch.int64, device=device)
    selection_valid = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    for request_idx, length in enumerate(lengths):
        selection_valid[request_idx, :length] = True

    score_weights = torch.full(
        (tokens, HEADS), 1.0 / math.sqrt(D), dtype=torch.float32, device=device
    )
    padded_score_weights = score_weights.index_select(0, selection_row_map)
    prefixes_i32 = torch.tensor(prefix_values, dtype=torch.int32, device=device)
    input_lengths_i32 = torch.tensor(lengths, dtype=torch.int32, device=device)
    padded_kv_lens = torch.where(
        selection_valid,
        prefixes_i32.unsqueeze(1)
        + torch.arange(1, max_len + 1, dtype=torch.int32, device=device),
        torch.zeros((), dtype=torch.int32, device=device),
    ).contiguous()
    kv_lens_rows = tuple(
        (
            prefixes_i32[i]
            + torch.arange(1, l + 1, dtype=torch.int32, device=device)
        ).unsqueeze(0).contiguous()
        for i, l in enumerate(lengths)
    )
    return {
        "prefixes_i64": prefixes,
        "prefixes_i32": prefixes_i32,
        "input_lengths_i32": input_lengths_i32,
        "cu_seqlens": expected_cu,
        "cu_kv_seqlens": expected_cu_kv,
        "visible_lengths": visible_lengths.to(torch.int32),
        "visible_lengths_i32": visible_lengths_i32,
        "compressed_lengths": compressed_lengths,
        "required_kv_columns": required_kv_columns,
        "required_state_columns": required_state_columns,
        "required_state_columns_device": required_state_columns.to(device),
        "logical_positions": logical_positions,
        "block_starts": block_starts,
        "mrope_sources": mrope_sources,
        "selection_row_map": selection_row_map,
        "selection_valid_rows": selection_valid_rows,
        "selection_valid": selection_valid,
        "score_weights": score_weights,
        "padded_score_weights": padded_score_weights,
        "padded_kv_lens": padded_kv_lens,
        "kv_lens_rows": kv_lens_rows,
        "cu_host": [0]
        + [total for total in torch.tensor(lengths).cumsum(0).tolist()],
        "position_ids": position_ids,
    }


def _random_cases(seed=2024):
    rng = random.Random(seed)
    cases = []
    for batch in range(1, 9):
        lengths = []
        for _ in range(batch):
            kind = rng.randrange(3)
            if kind == 0:
                length = RATIO * rng.randrange(1, 5)
            elif kind == 1:
                length = RATIO * rng.randrange(1, 4) + rng.randrange(1, RATIO)
            else:
                length = rng.randrange(1, RATIO + 1)
            lengths.append(max(1, length))
        if batch >= 2 and not any(length % RATIO for length in lengths):
            lengths[-1] = RATIO * 2 + 1
        prefixes = [PAGE * rng.randrange(0, 4) for _ in range(batch)]
        cases.append((prefixes, lengths))
    cases.append(([PAGE * 3], [1]))
    cases.append(([0, PAGE], [RATIO, RATIO]))
    cases.append(([PAGE, PAGE], [RATIO * 3, 1]))
    cases.append(([PAGE * 2, PAGE * 5, PAGE], [7, 13, 9]))
    return cases


class GridPrepFusionTest(unittest.TestCase):
    """Value fidelity of the fused prep against the legacy aten chain."""

    def setUp(self):
        qsa_step_plan.clear_step_plan_cache()
        strict_off = patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        )
        strict_off.start()
        self.addCleanup(strict_off.stop)

    def _assert_plan_matches_reference(self, plan, reference, *, rope_config):
        for name in (
            "prefixes_i64",
            "prefixes_i32",
            "input_lengths_i32",
            "cu_seqlens",
            "visible_lengths",
            "visible_lengths_i32",
            "compressed_lengths",
            "required_kv_columns",
            "required_state_columns",
            "required_state_columns_device",
            "logical_positions",
            "selection_row_map",
            "selection_valid_rows",
            "selection_valid",
            "score_weights",
            "padded_score_weights",
            "padded_kv_lens",
        ):
            self.assertTrue(
                torch.equal(getattr(plan, name), reference[name]),
                f"plan field {name} differs from the legacy reference",
            )
        for request_idx, (produced, expected) in enumerate(
            zip(plan.kv_lens_rows, reference["kv_lens_rows"])
        ):
            self.assertTrue(
                torch.equal(produced, expected),
                f"kv_lens_rows[{request_idx}] differs from the legacy reference",
            )
        self.assertEqual(plan.cu_host, reference["cu_host"])
        if rope_config["style"] is not None:
            if rope_config["style"] == RopeStyle.Mrope:
                sources = torch.arange(
                    plan.token_count, device=plan.device
                ) - reference["logical_positions"].remainder(RATIO)
                block_positions = (
                    reference["position_ids"]
                    .reshape(plan.token_count, 3)
                    .index_select(0, sources)
                )
                rope_cos, rope_sin = build_qsa_rope(
                    block_positions.reshape(-1),
                    rope_config["config"],
                    token_count=plan.token_count,
                    dtype=torch.bfloat16,
                    device=plan.device,
                )
            else:
                rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
                    reference["block_starts"],
                    rope_config["config"],
                    token_count=plan.token_count,
                    dtype=torch.bfloat16,
                    device=plan.device,
                )
            self.assertTrue(torch.equal(plan.rope_cos, rope_cos))
            self.assertTrue(torch.equal(plan.rope_sin, rope_sin))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_prefix_reuse_prep_matches_legacy_reference(self):
        device = _cuda_device()
        rope = _rope_mrope()
        for prefixes, lengths in _random_cases():
            with self.subTest(prefixes=prefixes, lengths=lengths):
                plan = _build_plan(prefixes, lengths, device)
                self.assertEqual(
                    plan.mode,
                    qsa_step_plan.MODE_PREFIX_REUSE,
                )
                reference = _legacy_reference(
                    lengths,
                    prefixes,
                    device,
                    position_ids=plan.main_inputs.combo_position_ids,
                )
                self._assert_plan_matches_reference(
                    plan, reference, rope_config={"style": RopeStyle.Mrope, "config": rope}
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_draft_prep_matches_legacy_reference(self):
        device = _cuda_device()
        rope = _rope_base()
        rng = random.Random(7)
        for prefixes, lengths in (
            ([PAGE, 1], [RATIO, 1]),
            ([PAGE * 2, 3], [1, RATIO]),
            ([5], [2]),
            ([PAGE, PAGE, 9], [1, 3, RATIO]),
            (
                [rng.randrange(1, 5 * PAGE) for _ in range(8)],
                [rng.randrange(1, RATIO + 1) for _ in range(8)],
            ),
        ):
            with self.subTest(prefixes=prefixes, lengths=lengths):
                plan = _build_plan(prefixes, lengths, device, draft=True, rope_config=rope)
                self.assertEqual(plan.mode, qsa_step_plan.MODE_DRAFT_INCREMENTAL)
                reference = _legacy_reference(
                    lengths,
                    prefixes,
                    device,
                    position_ids=plan.main_inputs.combo_position_ids,
                )
                self._assert_plan_matches_reference(
                    plan, reference, rope_config={"style": RopeStyle.Base, "config": rope}
                )

    def test_cpu_prep_matches_legacy_reference(self):
        device = torch.device("cpu")
        rope = _rope_mrope()
        for prefixes, lengths in _random_cases()[:6]:
            with self.subTest(prefixes=prefixes, lengths=lengths):
                plan = _build_plan(prefixes, lengths, device)
                reference = _legacy_reference(
                    lengths,
                    prefixes,
                    device,
                    position_ids=plan.main_inputs.combo_position_ids,
                )
                self._assert_plan_matches_reference(
                    plan, reference, rope_config={"style": RopeStyle.Mrope, "config": rope}
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_plan_matches_eager_fallback_bit_exact(self):
        device = _cuda_device()
        for prefixes, lengths in _random_cases():
            with self.subTest(prefixes=prefixes, lengths=lengths):
                fused = _build_plan(prefixes, lengths, device)
                with patch.object(
                    grid_prep_triton, "build_incremental_grid", return_value=None
                ):
                    eager = _build_plan(prefixes, lengths, device)
                for name in (
                    "prefixes_i64",
                    "prefixes_i32",
                    "input_lengths_i32",
                    "cu_seqlens",
                    "visible_lengths",
                    "visible_lengths_i32",
                    "compressed_lengths",
                    "required_kv_columns",
                    "required_state_columns",
                    "required_state_columns_device",
                    "logical_positions",
                    "selection_row_map",
                    "selection_valid_rows",
                    "selection_valid",
                    "score_weights",
                    "padded_score_weights",
                    "padded_kv_lens",
                    "rope_cos",
                    "rope_sin",
                    "current_cos",
                    "current_sin",
                ):
                    self.assertTrue(
                        torch.equal(getattr(fused, name), getattr(eager, name)),
                        f"fused/eager disagree on {name}",
                    )
                for produced, expected in zip(
                    fused.kv_lens_rows, eager.kv_lens_rows
                ):
                    self.assertTrue(torch.equal(produced, expected))
                self.assertEqual(fused.cu_host, eager.cu_host)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_grid_prep_reuses_one_variant_across_geometries(self):
        # Draft and prefill plan builds hit a new (batch, lengths, max_len)
        # geometry on most steps; specializing the int shape args pays a
        # ~100ms Triton JIT compile per new divisibility class.  One warmed
        # variant must serve every geometry.
        device = torch.device("cuda", torch.cuda.current_device())

        def build(lengths, prefixes):
            batch = len(lengths)
            prefixes_i32 = torch.tensor(prefixes, dtype=torch.int32, device=device)
            lengths_i32 = torch.tensor(lengths, dtype=torch.int32, device=device)
            cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
            cu[1:] = lengths_i32.cumsum(0)
            fused = grid_prep_triton.build_incremental_grid(
                prefixes_i32,
                lengths_i32,
                cu,
                max_len=max(lengths),
                token_count=sum(lengths),
                ratio=4,
                head_num=24,
                score_scale=1.0 / math.sqrt(256),
            )
            self.assertIsNotNone(fused, f"the fused grid prep declined {lengths}")

        build([8], [0])
        variants = _variant_total(grid_prep_triton._step_grid_prep_kernel)
        build([9], [128])
        build([5, 3], [128, 0])
        build([17], [0])
        build([129, 1], [0, 128])
        build([16], [16])
        self.assertEqual(
            _variant_total(grid_prep_triton._step_grid_prep_kernel),
            variants,
            "unseen geometries must reuse the warmed grid-prep variant",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_full_build_uses_the_fused_grid(self):
        device = _cuda_device()
        calls = []

        original = grid_prep_triton.build_incremental_grid

        def spy(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result is not None)
            return result

        with patch.object(grid_prep_triton, "build_incremental_grid", side_effect=spy):
            plan = _build_plan([PAGE, PAGE * 2], [5, 8], device)
        self.assertEqual(calls, [True])
        self.assertTrue(plan.visible_lengths.is_cuda)


class GridPrepLaunchBudgetTest(unittest.TestCase):
    """The prep cluster collapses to a bounded number of device launches."""

    def setUp(self):
        qsa_step_plan.clear_step_plan_cache()
        strict_off = patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        )
        strict_off.start()
        self.addCleanup(strict_off.stop)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_grid_prep_seam_tensors_match_reference(self):
        device = _cuda_device()
        for prefixes, lengths in _random_cases()[:5]:
            with self.subTest(prefixes=prefixes, lengths=lengths):
                prep = qsa_step_plan._build_step_grid_prep(
                    lengths=lengths,
                    prefix_values=prefixes,
                    device=device,
                    ratio=RATIO,
                    head_num=HEADS,
                    head_dim=D,
                    kv_entries_per_block=PAGE // RATIO,
                    state_tokens_per_block=PAGE,
                )
                reference = _legacy_reference(lengths, prefixes, device)
                for name in (
                    "prefixes_i64",
                    "prefixes_i32",
                    "input_lengths_i32",
                    "cu_seqlens",
                    "cu_kv_seqlens",
                    "visible_lengths",
                    "visible_lengths_i32",
                    "compressed_lengths",
                    "required_kv_columns",
                    "required_state_columns",
                    "required_state_columns_device",
                    "logical_positions",
                    "block_starts",
                    "mrope_sources",
                    "selection_row_map",
                    "selection_valid_rows",
                    "selection_valid",
                    "score_weights",
                    "padded_score_weights",
                ):
                    self.assertTrue(
                        torch.equal(getattr(prep, name), reference[name]),
                        f"prep field {name} differs from the legacy reference",
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_prep_launches_at_most_three_kernels_without_the_aten_chain(self):
        device = _cuda_device()
        lengths = [5, 8, 3, 9]
        prefixes = [PAGE, PAGE * 2, 0, PAGE]
        self.assertTrue(
            grid_prep_triton.incremental_grid_prep_supported(
                device, batch_size=len(lengths), max_len=max(lengths), head_num=HEADS
            )
        )
        kwargs = dict(
            lengths=lengths,
            prefix_values=prefixes,
            device=device,
            ratio=RATIO,
            head_num=HEADS,
            head_dim=D,
            kv_entries_per_block=PAGE // RATIO,
            state_tokens_per_block=PAGE,
        )
        prep = qsa_step_plan._build_step_grid_prep(**kwargs)
        self.assertTrue(prep.visible_lengths.is_cuda)
        torch.cuda.synchronize()

        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            prep = qsa_step_plan._build_step_grid_prep(**kwargs)
            torch.cuda.synchronize()

        kernel_rows = []
        for row in prof.key_averages():
            if getattr(row, "device_type", None) != DeviceType.CUDA:
                continue
            lowered = row.key.lower()
            if "memcpy" in lowered or "memset" in lowered:
                continue
            kernel_rows.append((row.key, row.count))
        total = sum(count for _, count in kernel_rows)
        self.assertLessEqual(
            total,
            3,
            f"fused prep must launch at most 3 kernels, got {kernel_rows}",
        )
        self.assertGreaterEqual(
            total, 1, "fused prep must launch its Triton kernel"
        )
        counts = {row.key: row.count for row in prof.key_averages()}
        for op in (
            "aten::arange",
            "aten::where",
            "aten::floor_divide",
            "aten::remainder",
            "aten::full",
            "aten::fill",
            "aten::index_select",
        ):
            self.assertEqual(
                counts.get(op, 0),
                0,
                f"{op} from the legacy chain must be absent from the fused prep",
            )


class GridPrepDtypeShapeTest(unittest.TestCase):
    """dtype/shape invariants of the fused prep outputs."""

    def setUp(self):
        qsa_step_plan.clear_step_plan_cache()
        strict_off = patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        )
        strict_off.start()
        self.addCleanup(strict_off.stop)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_plan_field_dtypes_and_shapes(self):
        device = _cuda_device()
        prefixes = [PAGE, PAGE * 2, 0]
        lengths = [5, 8, 3]
        batch = len(lengths)
        max_len = max(lengths)
        tokens = sum(lengths)
        plan = _build_plan(prefixes, lengths, device)

        self.assertEqual(plan.prefixes_i64.dtype, torch.int64)
        self.assertEqual(tuple(plan.prefixes_i64.shape), (batch,))
        self.assertEqual(plan.prefixes_i32.dtype, torch.int32)
        self.assertEqual(tuple(plan.prefixes_i32.shape), (batch,))
        self.assertEqual(plan.input_lengths_i32.dtype, torch.int32)
        self.assertEqual(tuple(plan.input_lengths_i32.shape), (batch,))
        self.assertEqual(plan.cu_seqlens.dtype, torch.int32)
        self.assertEqual(tuple(plan.cu_seqlens.shape), (batch + 1,))
        self.assertEqual(plan.logical_positions.dtype, torch.int64)
        self.assertEqual(tuple(plan.logical_positions.shape), (tokens,))
        self.assertEqual(plan.visible_lengths.dtype, torch.int32)
        self.assertEqual(tuple(plan.visible_lengths.shape), (batch, max_len))
        self.assertEqual(plan.visible_lengths_i32.dtype, torch.int32)
        self.assertEqual(
            tuple(plan.visible_lengths_i32.shape), (batch * max_len,)
        )
        self.assertEqual(plan.compressed_lengths.dtype, torch.int32)
        self.assertEqual(tuple(plan.compressed_lengths.shape), (batch, max_len))
        self.assertEqual(plan.required_kv_columns.dtype, torch.int32)
        self.assertEqual(tuple(plan.required_kv_columns.shape), (batch,))
        self.assertEqual(plan.required_state_columns.dtype, torch.int32)
        self.assertEqual(tuple(plan.required_state_columns.shape), (batch,))
        self.assertEqual(
            plan.required_state_columns_device.dtype, torch.int32
        )
        self.assertEqual(
            tuple(plan.required_state_columns_device.shape), (batch,)
        )
        self.assertEqual(plan.selection_row_map.dtype, torch.int64)
        self.assertEqual(
            tuple(plan.selection_row_map.shape), (batch * max_len,)
        )
        self.assertEqual(plan.selection_valid_rows.dtype, torch.int64)
        self.assertEqual(tuple(plan.selection_valid_rows.shape), (tokens,))
        self.assertEqual(plan.selection_valid.dtype, torch.bool)
        self.assertEqual(tuple(plan.selection_valid.shape), (batch, max_len))
        self.assertEqual(plan.score_weights.dtype, torch.float32)
        self.assertEqual(tuple(plan.score_weights.shape), (tokens, HEADS))
        self.assertEqual(plan.padded_score_weights.dtype, torch.float32)
        self.assertEqual(
            tuple(plan.padded_score_weights.shape), (batch * max_len, HEADS)
        )
        self.assertEqual(plan.padded_kv_lens.dtype, torch.int32)
        self.assertEqual(tuple(plan.padded_kv_lens.shape), (batch, max_len))
        for request_idx, length in enumerate(lengths):
            row = plan.kv_lens_rows[request_idx]
            self.assertEqual(row.dtype, torch.int32)
            self.assertEqual(tuple(row.shape), (1, length))
            self.assertTrue(row.is_contiguous())
        # The padded score weights stay the constant 1/sqrt(head_dim) fill.
        self.assertTrue(
            torch.equal(
                plan.padded_score_weights,
                torch.full_like(
                    plan.padded_score_weights, 1.0 / math.sqrt(D)
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                plan.score_weights,
                torch.full_like(plan.score_weights, 1.0 / math.sqrt(D)),
            )
        )


if __name__ == "__main__":
    unittest.main()
