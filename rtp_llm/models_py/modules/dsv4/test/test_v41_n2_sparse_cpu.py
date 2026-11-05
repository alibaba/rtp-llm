"""CPU tests for the N2 sparse EP dispatch helpers: flag parsing,
destination mask semantics (invalid ids, multi-dest rows, all-invalid
rows), and the fixed-shape stable compaction order. The exchange itself
is validated by the parity sha gate + engagement on the node."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


n2 = load("_r16_n2", "_v41_n2_sparse.py")


class FlagTest(unittest.TestCase):
    def test_default_off_and_values(self):
        clean = dict(os.environ)
        clean.pop("DSV41_N2_SPARSE_DISPATCH", None)
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertFalse(n2.n2_sparse_dispatch_enabled())
        with mock.patch.dict(os.environ, {"DSV41_N2_SPARSE_DISPATCH": "1"}):
            self.assertTrue(n2.n2_sparse_dispatch_enabled())
        for value in ("0", "true", ""):
            with mock.patch.dict(os.environ, {"DSV41_N2_SPARSE_DISPATCH": value}):
                self.assertFalse(n2.n2_sparse_dispatch_enabled())


class DestinationMaskTest(unittest.TestCase):
    def brute(self, indices, world, per):
        rows = indices.shape[0]
        out = torch.zeros((rows, world), dtype=torch.bool)
        for i in range(rows):
            for v in indices[i].tolist():
                if v >= 0:
                    out[i, v // per] = True
        return out

    def test_matches_brute_force(self):
        g = torch.Generator().manual_seed(11)
        world, per = 4, 96
        for rows, topk in ((1, 8), (7, 2), (64, 8), (129, 1)):
            ids = torch.randint(0, world * per, (rows, topk), generator=g)
            # sprinkle invalid ids and duplicate destinations
            ids[ids % 7 == 0] = -1
            if topk >= 2:
                ids[:, 1] = ids[:, 0]
            got = n2.sparse_destination_mask(ids, world, per)
            self.assertTrue(torch.equal(got, self.brute(ids, world, per)))

    def test_all_invalid_row_goes_nowhere(self):
        ids = torch.full((3, 4), -1, dtype=torch.int64)
        ids[1, 0] = 5
        mask = n2.sparse_destination_mask(ids, 4, 96)
        self.assertEqual(mask[0].sum().item(), 0)
        self.assertEqual(mask[1].sum().item(), 1)
        self.assertEqual(mask[2].sum().item(), 0)


class CompactionOrderTest(unittest.TestCase):
    def reference(self, contrib):
        rows, world = contrib.shape
        picked = []
        for d in range(world):
            for i in range(rows):
                if contrib[i, d]:
                    picked.append(d * rows + i)
        total = rows * world
        return picked + [total] * (total - len(picked)), len(picked)

    def test_matches_reference(self):
        g = torch.Generator().manual_seed(5)
        for rows, world in ((1, 4), (5, 2), (64, 4), (128, 8)):
            contrib = torch.rand((rows, world), generator=g) < 0.3
            order = n2.sparse_compaction_order(contrib)
            ref, total = self.reference(contrib)
            self.assertEqual(order.numel(), rows * world)
            self.assertEqual(order[:total].tolist(), ref[:total])
            # fill tail is the sentinel
            self.assertTrue((order[total:] == rows * world).all())

    def test_edge_cases(self):
        rows, world = 8, 4
        none = torch.zeros((rows, world), dtype=torch.bool)
        order = n2.sparse_compaction_order(none)
        self.assertTrue((order == rows * world).all())
        allc = torch.ones((rows, world), dtype=torch.bool)
        order = n2.sparse_compaction_order(allc)
        self.assertEqual(order.tolist(), list(range(rows * world)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
