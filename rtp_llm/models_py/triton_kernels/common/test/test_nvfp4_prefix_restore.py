import math
import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_prefix_restore import (
    restore_prefix_planes,
)


def _reference(values, side, restore, dst, planes):
    """Old gather + six multidimensional index_copy_ operations."""
    dst = dst.to(torch.long)
    if restore is not None:
        values = values.index_select(0, restore)
        side = side.index_select(0, restore)
    else:
        values, side = values[: dst.numel()], side[: dst.numel()]
    m, _, s, _, i, t = [math.prod(p.shape[1:]) for p in planes]
    pieces = (
        values[:, :m],
        values[:, m : 2 * m],
        side[:, :s],
        side[:, s : 2 * s],
        side[:, 2 * s : 2 * s + i],
        side[:, 2 * s + i : 2 * s + i + t],
    )
    for plane, piece in zip(planes, pieces):
        # Keep HND/page dimensions: a flattened-only oracle can miss stride
        # mistakes in capacity-sliced multi-plane working pools.
        plane.view(torch.uint8).index_copy_(
            0, dst, piece.reshape(dst.numel(), *plane.shape[1:])
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestNVFP4PrefixRestore(unittest.TestCase):
    active_tiles = False

    def _restore(self, *args):
        restore_prefix_planes(*args, active_tiles=self.active_tiles)

    def _case(self, heads=2, page=7, dim=32, index_dim=64, capacity=13):
        shapes = (
            (heads, page, dim // 2),
            (heads, page, dim // 2),
            (heads, page, dim // 16),
            (heads, page, dim // 16),
            (page, index_dim // 2),
            (page, index_dim // 16),
        )
        sizes = [math.prod(shape) for shape in shapes]
        widths = (2 * sizes[0], 2 * sizes[2] + sizes[4] + sizes[5])
        # Exposed trailing bytes and hidden stride padding must both be ignored.
        sources = []
        for width in widths:
            base = torch.full((16, width + 81), 253, device="cuda", dtype=torch.uint8)
            base[:11, :width].random_(0, 240)
            sources.append(base[:, : width + 17])
        # Independent base pointers, nonzero offsets, differing backing
        # capacities, and FP8 scales cover the production pool-view contract.
        backing, planes = [], []
        for n, shape in enumerate(shapes):
            base = torch.full(
                (capacity + n + 4, *shape), 231, device="cuda", dtype=torch.uint8
            )
            backing.append(base)
            plane = base[1 : capacity + 1]
            if n in (2, 3, 5):
                plane = plane.view(torch.float8_e4m3fn)
            planes.append(plane)
        return *sources, tuple(planes), backing

    def _assert_bytes(self, actual, expected):
        for a, e in zip(actual, expected):
            self.assertTrue(torch.equal(a.view(torch.uint8), e.view(torch.uint8)))

    def test_multidimensional_restore_and_identity(self):
        for dtype in (torch.int32, torch.int64):
            for identity in (False, True):
                for geometry in ((2, 7, 32, 64), (4, 128, 128, 128), (1, 17, 528, 96)):
                    with self.subTest(
                        dtype=dtype, identity=identity, geometry=geometry
                    ):
                        values, side, planes, backing = self._case(*geometry)
                        src = (
                            None
                            if identity
                            else torch.tensor(
                                [3, 1, 3, 9, 0, 7, 2], device="cuda", dtype=dtype
                            )
                        )
                        dst = torch.tensor(
                            [8, 0, 9, 2, 7, 3, 6], device="cuda", dtype=dtype
                        )
                        expected = tuple(p.clone() for p in planes)
                        _reference(values, side, src, dst, expected)
                        self._restore(values, side, src, dst, planes)
                        self._assert_bytes(planes, expected)
                        for base in backing:
                            self.assertTrue(torch.all(base[0] == 231).item())
                            self.assertTrue(torch.all(base[14:] == 231).item())

    def test_empty_sources_and_zero_capacity(self):
        values, side, planes, _ = self._case()
        ids = torch.empty(0, device="cuda", dtype=torch.int64)
        for src in (None, ids):
            self._restore(
                values[:0], side[:0], src, ids, tuple(p[:0] for p in planes)
            )

    def test_invalid_indices_are_guarded_per_plane(self):
        values, side, planes, _ = self._case()
        planes = tuple(p[: 5 + n] for n, p in enumerate(planes))
        src = torch.tensor([-1, 16, 2, 3, 4, 5], device="cuda")
        dst = torch.tensor([0, 1, -1, 13, 6, 4], device="cuda")
        # Compute each plane from a common full-capacity oracle, then apply the
        # plane-specific destination bounds to model intentionally unequal views.
        full = tuple(
            torch.full((13, *p.shape[1:]), 231, device="cuda", dtype=torch.uint8)
            for p in planes
        )
        valid = (src >= 0) & (src < 16) & (dst >= 0) & (dst < 13)
        _reference(values, side, src[valid], dst[valid], full)
        self._restore(values, side, src, dst, planes)
        self._assert_bytes(planes, tuple(p[: a.shape[0]] for p, a in zip(full, planes)))

    def test_graph_replay_changes_indices_and_contents(self):
        for identity in (False, True):
            values, side, planes, _ = self._case()
            src = (
                None
                if identity
                else torch.tensor([0, 2, 3, 3, 8], device="cuda", dtype=torch.int32)
            )
            dst = torch.tensor([0, 4, 3, 6, 8], device="cuda", dtype=torch.int64)
            self._restore(values, side, src, dst, planes)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._restore(values, side, src, dst, planes)
            for turn in range(3):
                if src is not None:
                    src.copy_(src.roll(1))
                dst.copy_(dst.roll(2))
                values.random_(256)
                side.random_(256)
                for p in planes:
                    p.view(torch.uint8).fill_(231 + turn)
                expected = tuple(p.clone() for p in planes)
                _reference(values, side, src, dst, expected)
                graph.replay()
                self._assert_bytes(planes, expected)

    def test_fi_page_pitch_scale_permutation_and_graph_replay(self):
        values, side, mma, _ = self._case(4, 128, 128, 128)
        capacity = 13
        pool = torch.full((capacity + 3, 73728), 231, device="cuda", dtype=torch.uint8)
        data_shape, data_stride = (capacity, 4, 128, 64), (73728, 8192, 64, 1)
        scale_shape, scale_stride = (capacity, 4, 128, 8), (73728, 1024, 8, 1)
        planes = (
            torch.as_strided(pool, data_shape, data_stride, 0),
            torch.as_strided(pool, data_shape, data_stride, 36864),
            torch.as_strided(pool, scale_shape, scale_stride, 32768).view(torch.float8_e4m3fn),
            torch.as_strided(pool, scale_shape, scale_stride, 69632).view(torch.float8_e4m3fn),
            mma[4], mma[5],
        )
        src = torch.tensor([3, 1, 3, 9, 0, 7, 2], device="cuda", dtype=torch.int64)
        dst = torch.tensor([8, 0, 9, 2, 7, 3, 6], device="cuda", dtype=torch.int32)
        linear = torch.arange(1024, device="cuda")
        token, group = linear // 8, linear % 8
        mma_offset = (group // 4) * 512 + (token % 32) * 16 + (token // 32) * 4 + group % 4

        def restore():
            restore_prefix_planes(values, side, src, dst, planes,
                                  active_tiles=self.active_tiles, fi_working_layout=True)

        restore()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            restore()
        for turn in range(3):
            src.copy_(src.roll(1))
            dst.copy_(dst.roll(2))
            values.random_(256)
            side.random_(256)
            pool.fill_(231)
            for plane in planes[4:]:
                plane.view(torch.uint8).fill_(231)
            expected = tuple(torch.full_like(p.view(torch.uint8), 231) for p in mma)
            _reference(values, side, src, dst, expected)
            # Independent tensor reshape oracle: first undo MMA, then group
            # four V tokens together, rather than copying the kernel formula.
            expected = list(expected)
            for plane in (2, 3):
                linear_scale = expected[plane].reshape(capacity, 4, 1024)[:, :, mma_offset]
                if plane == 3:
                    linear_scale = linear_scale.reshape(capacity, 4, 32, 4, 8).transpose(3, 4)
                expected[plane] = linear_scale.contiguous().reshape_as(expected[plane])
            graph.replay()
            self._assert_bytes(planes, expected)
            self.assertEqual(int(torch.count_nonzero(pool[capacity:] != 231)), 0)

    def test_metadata_validation(self):
        values, side, planes, _ = self._case()
        ids = torch.tensor([0, 1], device="cuda")
        bad_cases = (
            (values.float(), side, ids, ids, planes),
            (values, side[:-1], ids, ids, planes),
            (values[:, :1], side, ids, ids, planes),
            (values, side, ids[:1], ids, planes),
            (values, side, ids.float(), ids, planes),
            (values, side, ids, ids, planes[:-1]),
            (values, side, ids, ids, (planes[0].transpose(1, 2), *planes[1:])),
            (values, side, ids.cpu(), ids, planes),
        )
        for args in bad_cases:
            with self.subTest(shapes=[tuple(x.shape) for x in args[:4]]):
                with self.assertRaises(ValueError):
                    self._restore(*args)


class TestNVFP4PrefixRestoreActiveTiles(TestNVFP4PrefixRestore):
    # Run the full byte/invalid-index/graph/state-reuse contract for the
    # experimental layout independently of the unchanged runtime default.
    active_tiles = True


if __name__ == "__main__":
    unittest.main()
