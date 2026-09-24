"""Layout, allocation, and FP4 numeric contracts for grouped prefill scoring."""

import importlib.util
import os
import random
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_grouped_prefill_score as grouped


def key_views(widths, slab_counts, device="cpu"):
    result = []
    first = 0
    for count in slab_counts:
        sizes = widths[first : first + count]
        padded = [(n + 255) // 256 * 256 for n in sizes]
        quant = torch.empty(sum(padded), 64, dtype=torch.int8, device=device)
        scale = torch.empty(sum(padded), dtype=torch.int32, device=device)
        result.extend(
            SimpleNamespace(quant=q[:n], scale=s[:n])
            for q, s, n in zip(quant.split(padded), scale.split(padded), sizes)
        )
        first += count

    # _group_layout accepts the production key contract, including __len__.
    class Keys(SimpleNamespace):
        def __len__(self):
            return self.quant.shape[0]

    return [Keys(**vars(key)) for key in result]


class GroupedCEDDispatchCPU(unittest.TestCase):
    def test_compact_consumers_reuse_full_key_counts_and_invalidate_replaced_cp(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_batched_prefill_select as batched,
        )
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_sparse_prefill_indexer as sparse,
        )

        for batch in (1, 2, 32, 64):
            with self.subTest(batch=batch):
                lengths = torch.tensor([129 + 17 * i for i in range(batch)])
                prefixes = torch.tensor([512 * (i + 1) for i in range(batch)])
                local_rows = [2 * ((min(int(n), 128) + 7) // 8) for n in lengths]
                slices, start = [], 0
                for count in local_rows:
                    slices.append(slice(start, start + count))
                    start += count
                positions = torch.cat(
                    [
                        torch.arange(int(p + n) - count, int(p + n))
                        for p, n, count in zip(prefixes, lengths, local_rows)
                    ]
                )
                ids = torch.repeat_interleave(
                    torch.arange(batch), torch.tensor(local_rows)
                )
                projection = unittest.mock.Mock(
                    side_effect=AssertionError("reprojected compact weights")
                )
                shared = {
                    "ced_indexer_projection": projection,
                    "candidates": torch.zeros(start, 64, dtype=torch.int32),
                }
                owner = SimpleNamespace(
                    _shared_attention=shared,
                    layer_id=24,
                    compress_ratio=2,
                    index_topk=512,
                    _cp_ctx=SimpleNamespace(
                        prefix_lengths=prefixes, input_lengths_global=lengths
                    ),
                )
                q, sf, weights, output = (
                    SimpleNamespace(is_cuda=True),
                    object(),
                    object(),
                    object(),
                )
                calls = []

                def sparse_call(*args, **kwargs):
                    self.assertIs(args[0], q)
                    self.assertIs(args[1], sf)
                    self.assertIs(args[2], weights)
                    self.assertIs(args[5], positions)
                    self.assertIs(args[7], shared["candidates"])
                    self.assertIs(kwargs["req_ids"], ids)
                    torch.testing.assert_close(
                        kwargs["key_counts"],
                        (
                            (
                                owner._cp_ctx.prefix_lengths
                                + owner._cp_ctx.input_lengths_global
                            )
                            // 2
                        ).int(),
                    )
                    calls.append(kwargs["key_counts"])
                    return True

                def dispatch():
                    return batched.try_select_batched(
                        owner,
                        q,
                        sf,
                        weights,
                        [None] * batch,
                        slices,
                        positions,
                        output,
                        candidate_source=20,
                        publish_candidates=False,
                        candidate_size=8,
                        candidate_blocks=64,
                        req_ids=ids,
                    )

                with patch.object(
                    sparse, "try_batched_sparse", side_effect=sparse_call
                ):
                    for layer in (24, 28, 32, 36):
                        owner.layer_id = layer
                        self.assertEqual(dispatch(), batch > 1)
                    if batch > 1:
                        self.assertTrue(all(value is calls[0] for value in calls))
                        shared["prefill_sparse_plans"] = object()
                        owner._cp_ctx.prefix_lengths = prefixes + 512
                        self.assertTrue(dispatch())
                        self.assertIsNot(calls[0], calls[-1])
                        self.assertNotIn("prefill_sparse_plans", shared)
                        previous = calls[-1]
                        owner._cp_ctx.input_lengths_global = lengths + 7
                        self.assertTrue(dispatch())
                        self.assertIsNot(previous, calls[-1])
                projection.assert_not_called()

    def test_compact_l20_dispatches_grouped_publication_without_reprojection(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_batched_prefill_select as batched,
        )
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk

        for batch in (2, 32, 64):
            with self.subTest(batch=batch):
                counts = [
                    2 * ((min(n, 128) + 7) // 8)
                    for n in ([7, 31, 128, 257] * 16)[:batch]
                ]
                rows = sum(counts)
                projection = unittest.mock.Mock(
                    side_effect=AssertionError("unexpected projection")
                )
                shared = {
                    "ced_indexer_projection": projection,
                    "candidates": torch.full((rows, 64), -1, dtype=torch.int32),
                }
                owner = SimpleNamespace(
                    _shared_attention=shared,
                    layer_id=20,
                    compress_ratio=2,
                    index_topk=512,
                    _cp_ctx=SimpleNamespace(
                        prefix_lengths=torch.full((batch,), 4096),
                        input_lengths_global=torch.arange(batch) + 128,
                    ),
                )
                weights, positions, ids = object(), object(), object()
                output = torch.full((rows, 512), -777, dtype=torch.int32)
                logits, visible = torch.empty(rows, 512), torch.full(
                    (rows,), 512, dtype=torch.int32
                )
                bounds = (torch.zeros_like(visible), visible)

                def groups(*, mask_tail):
                    self.assertFalse(mask_tail)
                    yield slice(0, rows), logits, visible, bounds

                def publish(shared_arg, score, ends, span, bound, target, block, count):
                    self.assertIs(shared_arg, shared)
                    self.assertIs(score, logits)
                    target.copy_(torch.arange(512, dtype=torch.int32))
                    shared["candidates"][span].copy_(
                        torch.arange(64, dtype=torch.int32)
                    )
                    return True, target

                with patch.object(
                    grouped,
                    "try_grouped_scores",
                    return_value=SimpleNamespace(groups=groups),
                ) as score, patch.object(
                    batched, "_try_publish_with_tokens", side_effect=publish
                ) as published, patch.object(
                    topk,
                    "try_select_tokens",
                    side_effect=AssertionError("unexpected per-request fallback"),
                ):
                    self.assertTrue(
                        batched.try_select_batched(
                            owner,
                            SimpleNamespace(is_cuda=True),
                            None,
                            weights,
                            [None] * batch,
                            [slice(0, rows)],
                            positions,
                            output,
                            candidate_source=20,
                            publish_candidates=True,
                            candidate_size=8,
                            candidate_blocks=64,
                            req_ids=ids,
                        )
                    )
                    score.assert_called_once()
                    self.assertIs(score.call_args.args[2], weights)
                    self.assertIs(score.call_args.kwargs["req_ids"], ids)
                    published.assert_called_once()
                    self.assertTrue(
                        torch.equal(output, torch.arange(512).int().expand(rows, -1))
                    )
                    self.assertTrue(
                        torch.equal(
                            shared["candidates"],
                            torch.arange(64).int().expand(rows, -1),
                        )
                    )
                projection.assert_not_called()


class GroupedScoreEnvironmentTest(unittest.TestCase):
    def _load(self, value):
        name = grouped.__name__ + "_environment_test"
        spec = importlib.util.spec_from_file_location(name, grouped.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ), patch.dict(sys.modules, {name: module}):
            if value is None:
                os.environ.pop("DSV41_PREFILL_SCORE_MAX_BYTES", None)
            else:
                os.environ["DSV41_PREFILL_SCORE_MAX_BYTES"] = value
            spec.loader.exec_module(module)
        return module

    def test_default_and_positive_bounded_budget(self):
        for value, expected in (
            (None, 256 * 1024**2),
            ("1", 1),
            ("268435456", 256 * 1024**2),
            ("536870912", 512 * 1024**2),
            ("1073741824", 1024 * 1024**2),
        ):
            with self.subTest(value=value):
                module = self._load(value)
                self.assertEqual(module._MAX_LOGITS_BYTES, expected)
                self.assertEqual(module._MAX_GROUP_ROWS, 4096)
                self.assertEqual(module._MAX_LOGITS_INFLATION, 2)

    def test_invalid_budget_fails_import(self):
        for value in ("", "invalid", "1.5", "1GiB", "0", "-1", "1073741825"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, r"DSV41_PREFILL_SCORE_MAX_BYTES.*\[1, 1073741824\]"
            ):
                self._load(value)

    def test_environment_is_read_once(self):
        module = self._load(None)
        keys = key_views([65536] * 4, (4,))
        rows = (1500, 500, 1500, 500)
        original = module._group_layout(keys, rows)
        with patch.dict(os.environ, {"DSV41_PREFILL_SCORE_MAX_BYTES": "invalid"}):
            actual = module._group_layout(keys, rows)
        self.assertEqual(module._MAX_LOGITS_BYTES, 256 * 1024**2)
        self.assertEqual(
            [(g.first, g.stop, g.rows, g.width) for g in actual],
            [(g.first, g.stop, g.rows, g.width) for g in original],
        )

    def test_planner_obeys_configured_padded_byte_cap(self):
        keys = key_views([65536] * 4, (4,))
        rows = (1500, 500, 1500, 500)
        for mib, count in ((256, 6), (512, 2), (1024, 1)):
            with self.subTest(mib=mib):
                module = self._load(str(mib * 1024**2))
                layout = module._group_layout(keys, rows)
                self.assertEqual(len(layout), count)
                self.assertEqual(layout[0].rows.start, 0)
                self.assertEqual(layout[-1].rows.stop, sum(rows))
                for left, right in zip(layout, layout[1:]):
                    self.assertEqual(left.rows.stop, right.rows.start)
                for group in layout:
                    m = group.rows.stop - group.rows.start
                    padded_bytes = (
                        ((m + 3) // 4 * 4) * ((group.width + 255) // 256 * 256) * 4
                    )
                    self.assertLessEqual(m, 4096)
                    self.assertLessEqual(padded_bytes, mib * 1024**2)
        self.assertIsNone(self._load("1")._group_layout(keys, rows))


class GroupedScoreLayoutTest(unittest.TestCase):
    def test_bounds_descriptors_preserve_ragged_geometry_and_cap(self):
        layout, cursor = [], 0
        for i, count in enumerate((1, 127, 129, 4096)):
            layout.append(
                grouped._ScoreGroup(
                    i,
                    i + 1,
                    slice(cursor, cursor + count),
                    257 + i,
                    None,
                    torch.empty(257 + i),
                )
            )
            cursor += count
        descriptors, tiles, rows = grouped._bounds_descriptors(layout)
        self.assertEqual((tiles, rows), (36, 4353))
        self.assertEqual(descriptors[2], (4, 2, 3, 128, 129, 259, 259, 2))
        self.assertIsNone(grouped._bounds_descriptors(layout[:1]))
        self.assertIsNone(grouped._bounds_descriptors(layout * 65))
        self.assertIsNone(grouped._bounds_descriptors(layout[::-1]))

    def test_view_is_alias_even_with_nonzero_offset(self):
        slab = torch.empty(40, 64, dtype=torch.int8)
        got = grouped._slab_view((slab[4:12], slab[12:28]))
        self.assertEqual(got.data_ptr(), slab[4:].data_ptr())
        self.assertEqual(got.storage_offset(), 4 * 64)
        self.assertEqual(got.shape, (24, 64))
        for parts in (
            (slab[:4], slab[5:8]),
            (slab[:4], slab[4:8].clone()),
            (slab[:4, ::2], slab[4:8, ::2]),
            (),
        ):
            self.assertIsNone(grouped._slab_view(parts))

    def test_pool_boundary_19_13_and_byte_budget(self):
        keys = key_views([16384] * 32, (19, 13))
        layout = grouped._group_layout(keys, (512,) * 32)
        self.assertEqual(
            [(g.first, g.stop) for g in layout],
            [(0, 8), (8, 16), (16, 19), (19, 27), (27, 32)],
        )
        for g in layout:
            self.assertLessEqual(
                (g.rows.stop - g.rows.start) * g.width * 4, grouped._MAX_LOGITS_BYTES
            )
            self.assertEqual(g.payload.data_ptr(), keys[g.first].quant.data_ptr())
            self.assertEqual(g.scale.data_ptr(), keys[g.first].scale.data_ptr())

    def test_scale_boundary_alone_splits_group(self):
        keys = key_views([1024] * 4, (4,))
        scale = torch.empty(2048, dtype=torch.int32)
        keys[2].scale, keys[3].scale = scale[:1024], scale[1024:]
        layout = grouped._group_layout(keys, (4,) * 4)
        self.assertEqual([(g.first, g.stop) for g in layout], [(0, 2), (2, 4)])

    def test_ragged_rows_and_width_buckets(self):
        keys = key_views([1021, 1031, 2041, 2051, 3077], (5,))
        layout = grouped._group_layout(keys, (1, 7, 12, 17, 23))
        self.assertEqual(
            [(g.first, g.stop, g.rows, g.width) for g in layout],
            [(0, 5, slice(0, 60), 3077)],
        )

    def test_fresh_rows_respect_4096_cap(self):
        keys = key_views([8192] * 4, (4,))
        self.assertIsNone(grouped._group_layout(keys, (4096,) * 4))
        layout = grouped._group_layout(keys, (2048,) * 4)
        self.assertEqual([g.rows for g in layout], [slice(0, 4096), slice(4096, 8192)])

    def test_unsupported_metadata_falls_back(self):
        for widths, counts, slabs in [
            ([1024, 1024], (-1, 4), (2,)),
            ([1024, 1024], (0, 4), (2,)),
            ([1024, 1024], (4, 4), (1, 1)),
        ]:
            with self.subTest(widths=widths, counts=counts):
                self.assertIsNone(
                    grouped._group_layout(key_views(widths, slabs), counts)
                )
        keys = key_views([1024] * 2, (2,))
        backing = torch.empty(1025, dtype=torch.int32)
        keys[0].scale = backing[1:]
        self.assertIsNone(grouped._group_layout(keys, (4, 4)))

    def test_skew_inflation_keeps_long_key_request_separate(self):
        for widths, rows in (
            ([65536] + [1024] * 31, (2,) + (128,) * 31),
            ([1024] * 31 + [65536], (128,) * 31 + (2,)),
        ):
            layout = grouped._group_layout(key_views(widths, (32,)), rows)
            self.assertEqual(len(layout), 2)
            self.assertEqual(
                sorted(g.rows.stop - g.rows.start for g in layout), [2, 3968]
            )
            for g in layout:
                useful = sum(rows[i] * widths[i] for i in range(g.first, g.stop))
                self.assertLessEqual((g.rows.stop - g.rows.start) * g.width, 2 * useful)
                if g.width == 65536:
                    self.assertEqual(g.stop - g.first, 1)

    def test_oversized_request_splits_and_small_neighbors_still_group(self):
        counts = (7, 11, 770, 13, 17)
        keys = key_views([1021, 1031, 101377, 2041, 2051], (5,))
        layout = grouped._group_layout(keys, counts)
        self.assertEqual(
            [(g.first, g.stop) for g in layout], [(0, 2), (2, 3), (2, 3), (3, 5)]
        )
        self.assertEqual(
            [g.rows for g in layout],
            [slice(0, 18), slice(18, 678), slice(678, 788), slice(788, 818)],
        )
        for g in layout:
            size = (
                ((g.rows.stop - g.rows.start + 3) // 4 * 4)
                * ((g.width + 255) // 256 * 256)
                * 4
            )
            self.assertLessEqual(size, grouped._MAX_LOGITS_BYTES)
        self.assertEqual(layout[1].payload.data_ptr(), layout[2].payload.data_ptr())
        self.assertEqual(layout[1].scale.data_ptr(), layout[2].scale.data_ptr())
        large = grouped._group_layout(key_views([1024, 1024], (2,)), (4097, 4))
        self.assertEqual(
            [g.rows for g in large],
            [slice(0, 4096), slice(4096, 4097), slice(4097, 4101)],
        )

    def test_scheduler_order_preserves_inflation_and_coverage(self):
        rng = random.Random(711)
        workloads = [([65536, 1024] * 16, [2, 128] * 16)]
        base = [(4093 + 1873 * i, 257 + 17 * i) for i in range(32)]
        for _ in range(6):
            rng.shuffle(base)
            workloads.append(tuple(zip(*base)))
        for widths, rows in workloads:
            layout = grouped._group_layout(key_views(widths, (19, 13)), rows)
            if layout is None:
                continue
            self.assertEqual(layout[0].rows.start, 0)
            self.assertEqual(layout[-1].rows.stop, sum(rows))
            for left, right in zip(layout, layout[1:]):
                self.assertEqual(left.rows.stop, right.rows.start)
            for g in layout:
                useful = sum(rows[i] * widths[i] for i in range(g.first, g.stop))
                self.assertLessEqual((g.rows.stop - g.rows.start) * g.width, 2 * useful)
                size = (
                    ((g.rows.stop - g.rows.start + 3) // 4 * 4)
                    * ((g.width + 255) // 256 * 256)
                    * 4
                )
                self.assertLessEqual(size, grouped._MAX_LOGITS_BYTES)

    def test_cpu_factory_never_starts_device_work(self):
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("device access"),
        ):
            self.assertIsNone(
                grouped.try_grouped_scores(
                    torch.empty(8, 32, 64, dtype=torch.int8),
                    None,
                    None,
                    [(None, None)] * 2,
                    [slice(0, 4), slice(4, 8)],
                    None,
                    2,
                    {},
                )
            )


@unittest.skipUnless(
    os.environ.get("DSV41_GROUPED_SCORE_GPU_TEST") == "1",
    "GPU tests require explicit assigned GPU",
)
class GroupedScoreCudaTest(unittest.TestCase):
    def _bounds_plan(
        self,
        lengths,
        *,
        dtype=torch.int64,
        request_dtype=torch.int64,
        count_dtype=torch.int64,
        offset=0,
        ratio=1,
    ):
        counts = [-3, 0, 257, 509, 101377, 1024, 1025, 31]
        prefix = [0]
        for count in counts:
            prefix.append(prefix[-1] + (max(count, 0) + 255) // 256 * 256)
        layout, cursor = [], 0
        for i, length in enumerate(lengths):
            first = i % 7
            stop = min(8, first + 2)
            width = max(counts[first:stop])
            # Deliberately undersized slabs and widths exercise invalid bounds.
            span = prefix[stop - 1] - prefix[first] + max(counts[stop - 1], 0)
            if i % 5 == 3:
                span = max(0, span - 17)
            if i % 5 == 4:
                width = max(0, width - 17)
            layout.append(
                grouped._ScoreGroup(
                    first,
                    stop,
                    slice(cursor, cursor + length),
                    width,
                    None,
                    SimpleNamespace(numel=lambda n=span: n),
                )
            )
            cursor += length
        ids = torch.tensor(
            (
                [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 127, 128, 2147483647]
                * ((cursor + 12) // 13)
            )[:cursor],
            dtype=torch.int64,
        )
        if request_dtype == torch.int64:
            ids[::7] = 2**32 + 2  # Preserve original cast-before-range-check.
        pos = torch.tensor(
            ([-1, -9, 0, 1, 254, 256, 1000000, 2147483647] * ((cursor + 7) // 8))[
                :cursor
            ],
            dtype=dtype,
        )
        if dtype == torch.int64:
            pos[::11] = torch.iinfo(dtype).max

        def upload(value, dt):
            return torch.cat((torch.zeros(offset, dtype=dt), value.to(dt))).cuda()[
                offset:
            ]

        return grouped._GroupedScores(
            torch.empty(0, device="cuda"),
            None,
            None,
            upload(pos, dtype),
            ratio,
            {},
            tuple(lengths),
            tuple(layout),
            upload(ids, request_dtype),
            upload(torch.tensor(counts), count_dtype),
        )

    def _assert_bounds_reference(self, plan, actual):
        counts = plan.key_counts.cpu().to(torch.int64)
        padded = ((counts.clamp_min(0) + 255) // 256) * 256
        offsets = padded.cumsum(0) - padded
        for group, result in zip(plan.layout, actual):
            rows, *planes = result
            count = rows.stop - rows.start
            old = torch.empty((4, count), dtype=torch.int32, device="cuda")
            grouped._grouped_score_bounds_kernel[((count + 127) // 128,)](
                plan.positions[rows],
                plan.req_ids[rows],
                plan.key_counts,
                old,
                count,
                plan.key_counts.numel(),
                group.first,
                group.stop,
                group.scale.numel(),
                group.width,
                plan.ratio,
                num_warps=4,
            )
            self.assertTrue(torch.equal(torch.stack(planes), old))
            request = plan.req_ids[rows].cpu().to(torch.int32).long()
            safe = request.clamp(0, counts.numel() - 1)
            n = counts[safe]
            start = offsets[safe] - offsets[group.first]
            valid = (
                (request >= group.first)
                & (request < group.stop)
                & (request < len(counts))
                & (n >= 0)
                & (n <= group.width)
                & (start >= 0)
                & (start + n <= group.scale.numel())
            )
            visible = torch.minimum(
                (plan.positions[rows].cpu().long() + 1).clamp_min(0) // plan.ratio, n
            )
            visible = torch.where(valid, visible, 0).int()
            start = torch.where(valid, start, 0).int()
            expected = torch.stack(
                (start, start + visible, torch.zeros_like(start), visible)
            )
            self.assertTrue(torch.equal(old.cpu(), expected))
            self.assertEqual(planes[0].stride(), (1,))
            self.assertEqual(planes[0].data_ptr() % 16, 0)

    def test_all_group_bounds_exact_ragged_reorder_invalid_mixed_dtypes(self):
        for dtype in (torch.int32, torch.int64):
            for request_dtype in (torch.int32, torch.int64):
                for count_dtype in (torch.int32, torch.int64):
                    for offset in (0, 1):
                        for ratio in (1, 2):
                            plan = self._bounds_plan(
                                (1, 3, 127, 129, 263, 17, 4096),
                                dtype=dtype,
                                request_dtype=request_dtype,
                                count_dtype=count_dtype,
                                offset=offset,
                                ratio=ratio,
                            )
                            actual = plan._bounds()
                            self._assert_bounds_reference(plan, actual)
                            self.assertIs(plan._bounds(), actual)
                            self.assertEqual(
                                len(
                                    {v[1].untyped_storage().data_ptr() for v in actual}
                                ),
                                1,
                            )
                            plan.positions = plan.positions.clone()
                            self.assertIsNot(plan._bounds(), actual)

    def test_all_group_bounds_shapes_do_not_specialize(self):
        kernel = grouped._all_grouped_score_bounds_kernel
        for ratio in (1, 2):
            for offset in (0, 1):
                self._bounds_plan((4,) * 4, ratio=ratio, offset=offset)._bounds()
        before = len(kernel.device_caches[torch.cuda.current_device()][0])
        for lengths in ((1, 3, 5, 7), (129,) * 17, (3, 128, 4096, 17), (1,) * 256):
            for ratio in (1, 2):
                for offset in (0, 1):
                    plan = self._bounds_plan(lengths, ratio=ratio, offset=offset)
                    self._assert_bounds_reference(plan, plan._bounds())
        self.assertEqual(
            len(kernel.device_caches[torch.cuda.current_device()][0]), before
        )

    def test_all_group_bounds_capture_and_oversize_use_existing_path(self):
        plan = self._bounds_plan((3, 129, 1, 17))
        # Warm the old path, then force a cache miss inside capture.
        with patch.object(grouped, "_MAX_BOUNDS_GROUPS", 1):
            plan._bounds()
        plan.shared.clear()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with patch.object(
                grouped, "_bounds_descriptors", side_effect=AssertionError
            ):
                with torch.cuda.graph(graph, stream=stream):
                    actual = plan._bounds()
            plan.positions.fill_(31)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        self._assert_bounds_reference(plan, actual)
        large = self._bounds_plan((1,) * 257)
        with patch.object(
            grouped, "_all_grouped_score_bounds_kernel", side_effect=AssertionError
        ):
            actual = large._bounds()
        self._assert_bounds_reference(large, actual)

    def test_all_group_bounds_request_cap_and_independent_pointer_alignment(self):
        def plan_for(requests, ratio, mask):
            plan = self._bounds_plan((17, 129, 127, 263), ratio=ratio)
            count = torch.arange(requests, device="cuda") * 257
            plan.key_counts = count
            plan.req_ids = (
                torch.arange(plan.positions.numel(), device="cuda") % requests
            )
            # Visit first/last request and every request in between, including 127.
            plan.layout = tuple(
                grouped._ScoreGroup(
                    0,
                    requests,
                    g.rows,
                    32768,
                    None,
                    SimpleNamespace(numel=lambda: 4194304),
                )
                for g in plan.layout
            )
            for bit, name in enumerate(("positions", "req_ids", "key_counts")):
                if mask & (1 << bit):
                    value = getattr(plan, name)
                    storage = torch.empty(
                        value.numel() + 1, dtype=value.dtype, device="cuda"
                    )
                    storage[1:].copy_(value)
                    setattr(plan, name, storage[1:])
            return plan

        kernel = grouped._all_grouped_score_bounds_kernel
        for ratio in (1, 2):
            for mask in range(8):
                plan_for(4, ratio, mask)._bounds()
        before = len(kernel.device_caches[torch.cuda.current_device()][0])
        for requests in (1, 3, 17, 127, 128):
            for ratio in (1, 2):
                for mask in range(8):
                    plan = plan_for(requests, ratio, mask)
                    self._assert_bounds_reference(plan, plan._bounds())
        self.assertEqual(
            len(kernel.device_caches[torch.cuda.current_device()][0]), before
        )

    def test_few_groups_avoid_descriptor_upload(self):
        for lengths in ((17,), (3, 129), (3, 129, 4096)):
            plan = self._bounds_plan(lengths)
            with patch.object(
                grouped, "_bounds_descriptors", side_effect=AssertionError
            ):
                actual = plan._bounds()
            self._assert_bounds_reference(plan, actual)

    def test_bounds_lengths_and_group_ranges_do_not_specialize(self):
        kernel = grouped._grouped_score_bounds_kernel

        def launch(rows, requests, first, offset, ratio):
            counts = torch.full((requests,), 1024, device="cuda", dtype=torch.int32)
            ids = torch.full((rows + offset,), first, device="cuda", dtype=torch.int64)[
                offset:
            ]
            positions = torch.zeros(rows + offset, device="cuda", dtype=torch.int64)[
                offset:
            ]
            out = torch.empty((4, rows), device="cuda", dtype=torch.int32)
            kernel[((rows + 127) // 128,)](
                positions,
                ids,
                counts,
                out,
                rows,
                requests,
                first,
                requests,
                (requests - first) * 1024,
                1024,
                ratio,
                num_warps=4,
            )
            return out

        for ratio in (1, 2):
            for offset in (0, 1):
                launch(4, 2, 0, offset, ratio)
        before = len(kernel.device_caches[torch.cuda.current_device()][0])
        for rows, requests, first in (
            (1, 1, 0),
            (3, 3, 1),
            (129, 17, 5),
            (263, 127, 126),
            (4096, 128, 3),
        ):
            for ratio in (1, 2):
                for offset in (0, 1):
                    out = launch(rows, requests, first, offset, ratio)
                    self.assertTrue((out[0] == 0).all().item())
                    self.assertTrue((out[3] == 1 // ratio).all().item())
        self.assertEqual(
            len(kernel.device_caches[torch.cuda.current_device()][0]), before
        )

    def test_fused_bounds_dtype_offsets_and_invalid_requests(self):
        for dtype in (torch.int32, torch.int64):
            for request_dtype in (torch.int32, torch.int64):
                for count_dtype in (torch.int32, torch.int64):
                    for offset in (0, 1):
                        for ratio in (1, 2):
                            counts = torch.tensor(
                                [257, 509, 101377, 1024],
                                dtype=count_dtype,
                                device="cuda",
                            )
                            ids = torch.tensor(
                                [0] * offset + [1, 2, 3, 0, -1, 4, 1],
                                dtype=request_dtype,
                                device="cuda",
                            )[offset:]
                            pos = torch.tensor(
                                [0] * offset + [-1, 42, 90000, 5, 17, 31, 987],
                                dtype=dtype,
                                device="cuda",
                            )[offset:]
                            actual = torch.empty(
                                (4, 7), dtype=torch.int32, device="cuda"
                            )
                            span = 512 + 101632 + 1024
                            grouped._grouped_score_bounds_kernel[(1,)](
                                pos,
                                ids,
                                counts,
                                actual,
                                7,
                                4,
                                1,
                                4,
                                span,
                                101377,
                                ratio,
                                num_warps=4,
                            )
                            starts = torch.tensor(
                                [0, 512, 102144, 0, 0, 0, 0],
                                device="cuda",
                                dtype=torch.int32,
                            )
                            visible = torch.tensor(
                                [
                                    0,
                                    43 // ratio,
                                    min(90001 // ratio, 1024),
                                    0,
                                    0,
                                    0,
                                    min(988 // ratio, 509),
                                ],
                                device="cuda",
                                dtype=torch.int32,
                            )
                            expected = torch.stack(
                                (
                                    starts,
                                    starts + visible,
                                    torch.zeros_like(starts),
                                    visible,
                                )
                            )
                            self.assertTrue(torch.equal(actual, expected))

    def test_full_m_weights_and_all_logits_exact(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as indexer

        torch.manual_seed(728)
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        cases = [
            (
                (128, 129, 2, 131, 132, 3),
                [1021, 1033, 65536, 1049, 1061, 65533],
                (3, 3),
            ),
            ((2,) + (128,) * 31, [65536] + [1024] * 31, (32,)),
            ((7, 11, 770, 13, 17), [1021, 1031, 101377, 2041, 2051], (5,)),
            ((4097, 4), [1024, 1024], (2,)),
            (
                tuple(1 + i * 17 for i in range(32)),
                [997 + i * 1873 for i in range(32)],
                (19, 13),
            ),
            ((512,) * 32, [16384] * 32, (32,)),
            ((512,) * 32, [16384] * 32, (19, 13)),
            ((512,) * 32, [65536] * 32, (9, 9, 9, 5)),
            ((2048,) * 4, [8192] * 4, (4,)),
            ((1, 7, 12, 17, 23), [1021, 1031, 2041, 2051, 3077], (5,)),
        ]
        for counts, widths, slabs in cases:
            with self.subTest(counts=counts, widths=widths, slabs=slabs):
                ratio = 1 if slabs == (19, 13) else 2
                m = sum(counts)
                qp, sf = indexer.quantize_indexer_q(
                    torch.randn(m, 32, 128, device="cuda", dtype=torch.bfloat16)
                )
                # Exactly one original full-M BF16 head projection, before grouping.
                x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16) / 64
                weights = torch.nn.functional.linear(x, w).float() / 64
                del x, w
                keys, first = [], 0
                for count in slabs:
                    sizes = widths[first : first + count]
                    padded = [(n + 255) // 256 * 256 for n in sizes]
                    kp, ks = indexer.quantize_indexer_q(
                        torch.randn(
                            sum(padded), 128, device="cuda", dtype=torch.bfloat16
                        )
                    )
                    keys.extend(
                        (None, indexer.PrefillIndexerKeys(q[:n], s[:n]))
                        for q, s, n in zip(kp.split(padded), ks.split(padded), sizes)
                    )
                    first += count
                slices, offset, positions = [], 0, []
                for n, width in zip(counts, widths):
                    slices.append(slice(offset, offset + n))
                    offset += n
                    pos = torch.arange(
                        n, device="cuda", dtype=torch.int64
                    ) * ratio + ratio * (width - n)
                    pos[: min(4, n)] = torch.tensor(
                        [-1, 0, 1, ratio * width + 50], device="cuda"
                    )[: min(4, n)]
                    positions.append(pos)
                positions = torch.cat(positions)
                req_ids = torch.repeat_interleave(
                    torch.arange(len(counts), device="cuda"),
                    torch.tensor(counts, device="cuda"),
                    output_size=m,
                )
                key_counts = torch.tensor(widths, device="cuda")
                shared = {
                    "candidates": torch.empty(m, 1, device="cuda"),
                    "ced_indexer_projection": object(),
                }
                plan = grouped.try_grouped_scores(
                    qp,
                    sf,
                    weights,
                    keys,
                    slices,
                    positions,
                    ratio,
                    shared,
                    req_ids=req_ids,
                    key_counts=key_counts,
                )
                self.assertIsNotNone(plan)
                self.assertIs(plan._bounds(), plan._bounds())
                checked_rows = 0
                for row_slice, logits, visible, bounds in plan.groups():
                    self.assertLessEqual(
                        logits.untyped_storage().nbytes(), grouped._MAX_LOGITS_BYTES
                    )
                    self.assertEqual(
                        logits.shape,
                        (row_slice.stop - row_slice.start, logits.shape[1]),
                    )
                    self.assertTrue(torch.equal(bounds[0], torch.zeros_like(visible)))
                    for i, request in enumerate(slices):
                        start = max(request.start, row_slice.start)
                        stop = min(request.stop, row_slice.stop)
                        if start >= stop:
                            continue
                        request = slice(start, stop)
                        local = slice(start - row_slice.start, stop - row_slice.start)
                        checked_rows += stop - start
                        expected = indexer.score_indexer_chunk(
                            qp[request],
                            sf[request],
                            keys[i][1].quant,
                            keys[i][1].scale,
                            weights[request],
                            visible[local],
                            bounds=(bounds[0][local], visible[local]),
                        )
                        self.assertTrue(
                            torch.equal(expected, logits[local, : widths[i]])
                        )
                        self.assertTrue(
                            torch.isneginf(logits[local, widths[i] :]).all().item()
                        )
                        del expected
                    del logits
                self.assertEqual(checked_rows, m)
                # Same layout but replaced positions must not reuse old bounds.
                changed = positions + 2
                other = grouped.try_grouped_scores(
                    qp,
                    sf,
                    weights,
                    keys,
                    slices,
                    changed,
                    ratio,
                    shared,
                    req_ids=req_ids,
                    key_counts=key_counts,
                )
                self.assertIsNot(plan._bounds(), other._bounds())
                self.assertIsNone(
                    grouped.try_grouped_scores(
                        qp,
                        sf,
                        weights,
                        keys,
                        [slice(0, 1)] * len(keys),
                        positions,
                        2,
                        shared,
                    )
                )
                del plan, other, qp, sf, weights, positions, keys, shared

    def test_mask_overwrites_poison_and_slab_join_has_no_device_work(self):
        logits = torch.full((8, 1024), float("nan"), device="cuda")
        ends = torch.tensor(
            [0, 1, 255, 256, 257, 511, 1023, 1024], device="cuda", dtype=torch.int32
        )
        grouped._mask_tail_kernel[(8,)](logits, ends, 1024, logits.stride(0), 256)
        columns = torch.arange(1024, device="cuda")[None]
        self.assertTrue(torch.isneginf(logits[columns >= ends[:, None]]).all().item())
        self.assertTrue(torch.isnan(logits[columns < ends[:, None]]).all().item())
        slab = torch.empty(2048, 64, device="cuda", dtype=torch.int8)
        before = torch.cuda.memory_allocated()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            view = grouped._slab_view((slab[:1024], slab[1024:]))
        self.assertEqual(torch.cuda.memory_allocated(), before)
        self.assertEqual(view.data_ptr(), slab.data_ptr())
        self.assertFalse(
            [
                e
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
        )


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("DSV41_GROUPED_SCORE_GPU_TEST") == "1",
    "requires the grouped-score CUDA test environment",
)
class GroupedTailSelectionTest(unittest.TestCase):
    def _logits(self, width, *, strided=False, ties=False):
        ends = torch.tensor(
            [0, 1, min(511, width), min(512, width), width] * 2,
            device="cuda",
            dtype=torch.int32,
        )
        backing = torch.full((len(ends), width + 17), 123.0, device="cuda")
        logits = (
            backing[:, :width]
            if strided
            else torch.empty(len(ends), width, device="cuda")
        )
        values = torch.arange(width, device="cuda", dtype=torch.float32)
        if ties:
            values = values.div(7, rounding_mode="floor")
        logits.copy_(values)
        poison = torch.tensor(
            [float("nan"), float("inf"), 1e30, -float("inf")], device="cuda"
        )[torch.arange(width, device="cuda") % 4]
        tail = torch.arange(width, device="cuda")[None] >= ends[:, None]
        logits.copy_(torch.where(tail, poison, logits))
        clean = logits.masked_fill(tail, -torch.inf)
        return logits, clean, ends, tail

    def _select(
        self,
        logits,
        ends,
        *,
        publish=False,
        candidates=False,
        fallback=None,
        fuse=False,
        candidate_blocks=32,
        bitmap=True,
        retained=None,
        pool_reject=False,
    ):
        from rtp_llm.models_py.modules.dsv4.fp8 import (
            _v41_batched_prefill_select as batched,
        )
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as pool
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk
        from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention

        rows, width = logits.shape
        count = min(512, width)
        output_storage = torch.full(
            (rows, count + 1), -77, device="cuda", dtype=torch.int32
        )
        output = (
            output_storage[:, :count]
            if fallback == "out"
            else torch.full((rows, count), -77, device="cuda", dtype=torch.int32)
        )
        shared = {}
        if candidates:
            shared["candidates"] = torch.full(
                (rows, candidate_blocks), -99, device="cuda", dtype=torch.int32
            )
            shared["prefill_sparse_candidates"] = not bitmap
            if retained is not None:
                shared["ced_candidate_rows"] = retained
        attn = SimpleNamespace(
            _shared_attention=shared,
            _cp_ctx=SimpleNamespace(
                prefix_lengths=torch.zeros(2, device="cuda", dtype=torch.int64),
                input_lengths_global=torch.full(
                    (2,), width, device="cuda", dtype=torch.int32
                ),
            ),
            layer_id=14,
            compress_ratio=1,
            index_topk=512,
        )
        bounds = (torch.zeros_like(ends), ends)
        if fallback == "bounds":
            bounds = tuple(t.long() for t in bounds)

        def groups(*, mask_tail=True):
            self.assertFalse(mask_tail)
            yield slice(0, rows), logits, ends, bounds

        original_apply = attention._apply_prefill_candidates
        published = []

        def apply(*args):
            tail = torch.arange(width, device="cuda")[None] >= ends[:, None]
            self.assertTrue(torch.isneginf(args[1][tail]).all().item())
            published.append(True)
            return original_apply(*args)

        native = topk.try_select_tokens
        original_pool = pool.select_candidates

        def select_pool(*args, **kwargs):
            if pool_reject and kwargs.get("token_indices") is not None:
                return None
            return original_pool(*args, **kwargs)

        with ExitStack() as stack:
            if not fuse:
                stack.enter_context(
                    patch.object(
                        batched, "_try_publish_with_tokens", return_value=(False, None)
                    )
                )
            stack.enter_context(
                patch.object(pool, "select_candidates", side_effect=select_pool)
            )
            finish = stack.enter_context(
                patch.object(topk, "finish_tokens", wraps=topk.finish_tokens)
            )
            raw_native = stack.enter_context(
                patch.object(
                    topk.rtp_llm_ops, "topk_v3", wraps=topk.rtp_llm_ops.topk_v3
                )
            )
            finite_entry = getattr(topk.rtp_llm_ops, "dsv41_topk_v3_finite", None)
            finite_native = (
                stack.enter_context(
                    patch.object(
                        topk.rtp_llm_ops, "dsv41_topk_v3_finite", wraps=finite_entry
                    )
                )
                if finite_entry is not None
                else None
            )
            stack.enter_context(
                patch.object(
                    grouped,
                    "try_grouped_scores",
                    return_value=SimpleNamespace(groups=groups),
                )
            )
            stack.enter_context(
                patch.object(attention, "_apply_prefill_candidates", side_effect=apply)
            )
            select = stack.enter_context(
                patch.object(
                    topk,
                    "try_select_tokens",
                    side_effect=(
                        (lambda *a, **k: None) if fallback == "forced" else native
                    ),
                )
            )
            self.assertTrue(
                batched.try_select_batched(
                    attn,
                    torch.empty(rows, 1, device="cuda"),
                    None,
                    None,
                    [None, None],
                    [slice(0, rows // 2), slice(rows // 2, rows)],
                    torch.empty(rows, device="cuda", dtype=torch.int64),
                    output,
                    candidate_source=-1,
                    publish_candidates=publish,
                    candidate_size=8,
                    candidate_blocks=candidate_blocks,
                    req_ids=torch.zeros(rows, device="cuda", dtype=torch.int64),
                )
            )
            self.last_calls = dict(
                selector=select.call_count,
                native=raw_native.call_count,
                finite_native=(
                    finite_native.call_count if finite_native is not None else 0
                ),
                finish=finish.call_count,
                legacy_publication=len(published),
                raw_attempts=sum(
                    c.kwargs.get("filter_finite") is False
                    for c in select.call_args_list
                ),
            )
        if not fuse:
            self.assertEqual(self.last_calls["selector"], 1)
            self.assertEqual(len(published), int(publish and candidates))
        self.assertTrue((output_storage[:, -1] == -77).all().item())
        return output, shared

    def _assert_contract(self, actual, clean, ends, *, ties=False):
        values, ids = clean.topk(actual.shape[1], dim=-1)
        expected = torch.where(values.isfinite(), ids, -1).int()
        self.assertTrue(
            ((actual == -1) | ((actual >= 0) & (actual < ends[:, None]))).all().item()
        )
        if not ties:
            self.assertTrue(torch.equal(actual.sort(1).values, expected.sort(1).values))
        selected_values = clean.gather(1, actual.clamp_min(0).long())
        selected_values.masked_fill_(actual < 0, -torch.inf)
        self.assertTrue(
            torch.equal(
                selected_values.sort(1).values,
                values.masked_fill(~values.isfinite(), -torch.inf).sort(1).values,
            )
        )
        for row in actual:
            valid = row[row >= 0]
            self.assertEqual(valid.unique().numel(), valid.numel())

    def test_native_ignores_poisoned_tail_without_masking(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk

        for width in (512, 513, 8193):
            for strided in (False, True):
                for ties in (False, True):
                    with self.subTest(width=width, strided=strided, ties=ties):
                        logits, clean, ends, _ = self._logits(
                            width, strided=strided, ties=ties
                        )
                        before = logits.contiguous().view(torch.uint8).clone()
                        actual, _ = self._select(logits, ends)
                        old = topk.try_select_tokens(
                            clean, ends, bounds=(torch.zeros_like(ends), ends)
                        )
                        self.assertIsNotNone(old)
                        self.assertTrue(
                            torch.equal(actual.sort(1).values, old.sort(1).values)
                        )
                        self._assert_contract(actual, clean, ends, ties=ties)
                        self.assertTrue(
                            torch.equal(logits.contiguous().view(torch.uint8), before)
                        )

    def test_fallback_masks_before_original_torch_topk(self):
        for width in (257, 8193):
            for reason in ("forced", "out", "bounds"):
                with self.subTest(width=width, reason=reason):
                    logits, clean, ends, _ = self._logits(width, strided=True)
                    actual, _ = self._select(logits, ends, fallback=reason)
                    self.assertTrue(torch.equal(logits, clean))
                    values, ids = clean.topk(min(512, width), dim=-1)
                    expected = torch.where(values.isfinite(), ids, -1).int()
                    self.assertTrue(torch.equal(actual, expected))

    def test_publication_receives_masked_logits_and_keeps_candidates_exact(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention

        for fallback in (None, "forced"):
            logits, clean, ends, _ = self._logits(8193, strided=True)
            actual, shared = self._select(
                logits, ends, publish=True, candidates=True, fallback=fallback
            )
            expected_shared = {"candidates": torch.full_like(shared["candidates"], -99)}
            attention._apply_prefill_candidates(
                expected_shared, clean, ends, slice(0, len(ends)), 8, 32, True
            )
            self.assertTrue(
                torch.equal(shared["candidates"], expected_shared["candidates"])
            )
            self.assertTrue(torch.equal(logits, clean))
            self._assert_contract(actual, clean, ends)

    def test_nonpublishing_candidate_table_does_not_force_mask(self):
        for candidates, publish in ((True, False), (False, True)):
            logits, clean, ends, _ = self._logits(1024, strided=True)
            before = logits.contiguous().view(torch.uint8).clone()
            actual, _ = self._select(
                logits, ends, candidates=candidates, publish=publish
            )
            self._assert_contract(actual, clean, ends)
            self.assertTrue(torch.equal(logits.contiguous().view(torch.uint8), before))

    def test_fused_publication_filters_tokens_with_both_pool_modes(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention

        for width in (513, 16384, 32769):
            for bitmap in (False, True):
                with self.subTest(width=width, bitmap=bitmap):
                    logits, clean, ends, tail = self._logits(width, strided=True)
                    logits[4, :3] = torch.tensor(
                        [torch.nan, torch.inf, -torch.inf], device="cuda"
                    )
                    clean = logits.masked_fill(tail, -torch.inf)
                    before = logits.contiguous().view(torch.uint8).clone()
                    actual, shared = self._select(
                        logits,
                        ends,
                        publish=True,
                        candidates=True,
                        fuse=True,
                        candidate_blocks=2048,
                        bitmap=bitmap,
                    )
                    self.assertEqual(
                        self.last_calls,
                        dict(
                            selector=1,
                            native=1,
                            finish=0,
                            legacy_publication=0,
                            raw_attempts=1,
                        ),
                    )
                    expected = {
                        "candidates": torch.full_like(shared["candidates"], -99),
                        "prefill_sparse_candidates": not bitmap,
                    }
                    attention._apply_prefill_candidates(
                        expected, clean, ends, slice(0, len(ends)), 8, 2048, True
                    )
                    self.assertTrue(
                        torch.equal(
                            shared["candidates"].sort(1).values,
                            expected["candidates"].sort(1).values,
                        )
                    )
                    if bitmap:
                        self.assertTrue(
                            torch.equal(shared["candidates"], expected["candidates"])
                        )
                    self._assert_contract(actual, clean, ends, ties=True)
                    self.assertTrue(
                        torch.equal(logits.contiguous().view(torch.uint8), before)
                    )

    def test_fused_publication_skip_and_rejected_paths_keep_filter(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention

        for case in ("ced_skip", "pool_reject", "out", "bounds", "forced"):
            with self.subTest(case=case):
                logits, clean, ends, _ = self._logits(32769, strided=True)
                kwargs = dict(
                    fuse=True,
                    publish=True,
                    candidates=True,
                    candidate_blocks=2048,
                    retained=[] if case == "ced_skip" else None,
                    pool_reject=case == "pool_reject",
                )
                if case in ("out", "bounds", "forced"):
                    kwargs["fallback"] = case
                actual, shared = self._select(logits, ends, **kwargs)
                self.assertEqual(self.last_calls["legacy_publication"], 1)
                if case == "pool_reject":
                    self.assertEqual(
                        (
                            self.last_calls["raw_attempts"],
                            self.last_calls["native"],
                            self.last_calls["finish"],
                        ),
                        (1, 1, 1),
                    )
                elif case == "ced_skip":
                    self.assertEqual(
                        (
                            self.last_calls["raw_attempts"],
                            self.last_calls["finish"]
                            + self.last_calls["finite_native"],
                        ),
                        (0, 1),
                    )
                    self.assertTrue((shared["candidates"] == -99).all().item())
                elif case in ("out", "bounds"):
                    self.assertEqual(
                        (self.last_calls["raw_attempts"], self.last_calls["native"]),
                        (0, 0),
                    )
                self._assert_contract(actual, clean, ends)
                self.assertTrue(torch.equal(logits, clean))
                if case != "ced_skip":
                    expected = {
                        "candidates": torch.full_like(shared["candidates"], -99)
                    }
                    attention._apply_prefill_candidates(
                        expected, clean, ends, slice(0, len(ends)), 8, 2048, True
                    )
                    self.assertTrue(
                        torch.equal(shared["candidates"], expected["candidates"])
                    )

    def test_fused_all_nonfinite_tokens_preserve_candidate_contract(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention

        for width in (512, 32769):
            for value in (torch.nan, torch.inf, -torch.inf):
                with self.subTest(width=width, value=value):
                    logits, _, ends, tail = self._logits(width, strided=True)
                    logits.masked_fill_(~tail, value)
                    clean = logits.masked_fill(tail, -torch.inf)
                    actual, shared = self._select(
                        logits,
                        ends,
                        publish=True,
                        candidates=True,
                        fuse=True,
                        candidate_blocks=2048,
                        bitmap=False,
                    )
                    self.assertTrue((actual == -1).all().item())
                    self.assertEqual(self.last_calls["finish"], 0)
                    expected = {
                        "candidates": torch.full_like(shared["candidates"], -99),
                        "prefill_sparse_candidates": True,
                    }
                    attention._apply_prefill_candidates(
                        expected, clean, ends, slice(0, len(ends)), 8, 2048, True
                    )
                    self.assertTrue(
                        torch.equal(
                            shared["candidates"].sort(1).values,
                            expected["candidates"].sort(1).values,
                        )
                    )

    def test_candidate_preflight_rejects_alias_before_raw_selection(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as pool

        logits, _, ends, _ = self._logits(4096)
        out = torch.empty(len(ends), 512, device="cuda", dtype=torch.int32)
        self.assertFalse(
            pool.can_select_candidates(
                logits, ends, 8, 512, out=out, token_indices=out, token_ends=ends
            )
        )
        target = torch.empty_like(out)
        self.assertTrue(
            pool.can_select_candidates(
                logits, ends, 8, 512, out=out, token_indices=target, token_ends=ends
            )
        )
        self.assertFalse(
            pool.can_select_candidates(
                logits,
                ends,
                8,
                512,
                out=out,
                token_indices=target,
                token_ends=ends.long(),
            )
        )

    def test_groups_default_masks_and_opt_out_does_not_cache_logits(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as scorer

        logits, clean, ends, _ = self._logits(1025, strided=True)
        rows = slice(0, len(ends))
        plan = grouped._GroupedScores(
            logits,
            logits,
            logits,
            None,
            1,
            {},
            (len(ends),),
            (grouped._ScoreGroup(0, 1, rows, 1025, None, None),),
            None,
            None,
        )
        bounds = ((rows, torch.zeros_like(ends), ends, torch.zeros_like(ends), ends),)
        calls = []

        def score(*args, **kwargs):
            self.assertFalse(kwargs["clean_logits"])
            backing = torch.empty(len(ends), 1042, device="cuda")
            result = backing[:, :1025]
            result.copy_(logits)
            calls.append(result)
            return result

        with patch.object(plan, "_bounds", return_value=bounds), patch.object(
            scorer, "fp8_fp4_mqa_indexer_score", side_effect=score
        ):
            masked = next(plan.groups())[1]
            raw = next(plan.groups(mask_tail=False))[1]
            self.assertTrue(torch.equal(masked, clean))
            self.assertTrue(
                torch.equal(
                    raw.contiguous().view(torch.uint8),
                    logits.contiguous().view(torch.uint8),
                )
            )
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(masked.data_ptr(), raw.data_ptr())


if __name__ == "__main__":
    unittest.main()
