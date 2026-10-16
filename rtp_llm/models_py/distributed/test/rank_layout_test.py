"""Pure unit tests for the PP grid and mainline single-stage rank semantics."""

import itertools
import unittest
from types import SimpleNamespace

from rtp_llm.models_py.distributed.rank_layout import Coord, Group, RankLayout

_SWEEP = list(itertools.product((1, 2, 3), repeat=3))


# Grid-formula oracles: test references only, never import from production code.


def _grid_coord(world_rank, pp_size, dp_size, tp_size):
    tp_rank = world_rank % tp_size
    dp_rank = (world_rank // tp_size) % max(dp_size, 1)
    pp_rank = world_rank // (dp_size * tp_size)
    return (pp_rank, dp_rank, tp_rank)


def _legacy_tp_groups(world_size, pp_size, dp_size, tp_size):
    groups = []
    for pp_rank_val in range(pp_size):
        for dp_rank_val in range(dp_size):
            members = [
                r
                for r in range(world_size)
                if r // tp_size == pp_rank_val * dp_size + dp_rank_val
            ]
            if members:
                groups.append(members)
    return groups


def _legacy_dp_groups(world_size, pp_size, dp_size, tp_size):
    groups = []
    for pp_rank_val in range(pp_size):
        for tp_rank_val in range(tp_size):
            members = [
                r
                for r in range(world_size)
                if r % tp_size == tp_rank_val
                and r // (tp_size * dp_size) == pp_rank_val
            ]
            if members:
                groups.append(members)
    return groups


def _legacy_pp_groups(world_size, pp_size, dp_size, tp_size):
    groups = []
    for dp_rank_val in range(dp_size):
        for tp_rank_val in range(tp_size):
            members = [
                r
                for r in range(world_size)
                if r % tp_size == tp_rank_val
                and (r // tp_size) % dp_size == dp_rank_val
            ]
            if members:
                groups.append(members)
    return groups


class RankLayoutCoordTest(unittest.TestCase):
    def test_coord_rank_roundtrip_exhaustive(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            world_size = pp_size * dp_size * tp_size
            self.assertEqual(layout.world_size(), world_size)
            for r in range(world_size):
                coord = layout.coord_of(r)
                self.assertEqual(
                    layout.world_rank_of(coord), r, msg=f"layout={layout}, rank={r}"
                )

    def test_coord_matches_grid_formulas_exhaustive(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            for r in range(layout.world_size()):
                coord = layout.coord_of(r)
                grid_pp, grid_dp, grid_tp = _grid_coord(r, pp_size, dp_size, tp_size)
                self.assertEqual(
                    (coord.pp, coord.dp, coord.tp),
                    (grid_pp, grid_dp, grid_tp),
                    msg=f"layout={layout}, rank={r}",
                )

    def test_world_rank_of_layout_order(self):
        # world_rank = pp * (dp * tp) + dp * tp + tp: PP outermost, TP innermost.
        layout = RankLayout(world_size=24, pp_size=2, dp_size=3, tp_size=4)
        self.assertEqual(layout.world_rank_of(Coord(tp=0, dp=0, pp=0)), 0)
        self.assertEqual(layout.world_rank_of(Coord(tp=3, dp=0, pp=0)), 3)
        self.assertEqual(layout.world_rank_of(Coord(tp=0, dp=1, pp=0)), 4)
        self.assertEqual(layout.world_rank_of(Coord(tp=0, dp=0, pp=1)), 12)
        self.assertEqual(layout.world_rank_of(Coord(tp=3, dp=2, pp=1)), 23)

    def test_out_of_range_raises(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            for rank in (-1, layout.world_size()):
                with self.subTest(layout=layout, rank=rank):
                    with self.assertRaises(ValueError):
                        layout.coord_of(rank)

        layout = RankLayout(world_size=4, pp_size=2, dp_size=1, tp_size=2)
        with self.assertRaises(ValueError):
            layout.world_rank_of(Coord(tp=2, dp=0, pp=0))
        with self.assertRaises(ValueError):
            layout.world_rank_of(Coord(tp=0, dp=1, pp=0))
        with self.assertRaises(ValueError):
            layout.world_rank_of(Coord(tp=0, dp=0, pp=2))

    def test_invalid_sizes_raise(self):
        for bad in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 2, 2)):
            with self.assertRaises(ValueError):
                RankLayout(world_size=1, pp_size=bad[0], dp_size=bad[1], tp_size=bad[2])

    def test_basic_quantities(self):
        layout = RankLayout(world_size=24, pp_size=3, dp_size=2, tp_size=4)
        self.assertEqual(layout.size_of(Group.PP), 3)
        self.assertEqual(layout.size_of(Group.DP), 2)
        self.assertEqual(layout.size_of(Group.TP), 4)
        self.assertEqual(layout.lane_stride(), 8)

    def test_explicit_world_size_preserves_single_stage_rank_derivation(self):
        layout = RankLayout.from_parallelism_config(
            SimpleNamespace(world_size=4, pp_size=1, dp_size=1, tp_size=1)
        )
        self.assertEqual(layout.world_size(), 4)
        self.assertEqual(layout.lane_stride(), 4)
        for rank in range(4):
            self.assertEqual(layout.coord_of(rank), Coord(pp=0, dp=rank, tp=0))
            self.assertEqual(layout.ep_rank_of(rank, 1), 0)
        for rank in (-1, 4):
            with self.assertRaises(ValueError):
                layout.coord_of(rank)
            with self.assertRaises(ValueError):
                layout.ep_rank_of(rank, 1)

    def test_stage_boundaries_use_explicit_world_size(self):
        layout = RankLayout(world_size=8, pp_size=2, dp_size=1, tp_size=1)
        self.assertEqual(layout.lane_stride(), 4)
        self.assertEqual(layout.coord_of(3), Coord(pp=0, dp=3, tp=0))
        self.assertEqual(layout.coord_of(4), Coord(pp=1, dp=0, tp=0))
        self.assertEqual(layout.world_rank_of(Coord(pp=1)), 4)
        self.assertEqual(layout.groups(Group.WORLD), [list(range(8))])
        self.assertEqual(layout.groups(Group.STAGE), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(layout.size_of(Group.STAGE), 4)
        self.assertEqual(layout.rank_in_group(Group.STAGE, 7), 3)

    def test_world_size_must_be_divisible_by_pp_size(self):
        for world_size, pp_size in ((5, 2), (1, 2)):
            with self.subTest(world_size=world_size, pp_size=pp_size):
                with self.assertRaisesRegex(ValueError, "must be divisible"):
                    RankLayout(world_size=world_size, pp_size=pp_size)

    def test_world_size_is_part_of_layout_identity(self):
        layout = RankLayout(world_size=4)
        self.assertNotEqual(layout, RankLayout(world_size=1))
        self.assertIn("world_size=4", repr(layout))

    def test_from_parallelism_config_duck_typed(self):
        cfg = SimpleNamespace(world_size=24, pp_size=2, dp_size=3, tp_size=4)
        layout = RankLayout.from_parallelism_config(cfg)
        self.assertEqual(
            layout, RankLayout(world_size=24, pp_size=2, dp_size=3, tp_size=4)
        )

    def test_from_parallelism_config_rejects_missing_sizes(self):
        for field in ("world_size", "pp_size", "dp_size", "tp_size"):
            cfg = SimpleNamespace(world_size=1, pp_size=1, dp_size=1, tp_size=1)
            delattr(cfg, field)
            with self.subTest(field=field):
                with self.assertRaises(AttributeError):
                    RankLayout.from_parallelism_config(cfg)

    def test_from_parallelism_config_rejects_invalid_sizes(self):
        for field in ("world_size", "pp_size", "dp_size", "tp_size"):
            for value in (0, -1):
                cfg = SimpleNamespace(world_size=1, pp_size=1, dp_size=1, tp_size=1)
                setattr(cfg, field, value)
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        RankLayout.from_parallelism_config(cfg)


class RankLayoutGroupsTest(unittest.TestCase):
    def test_tp_groups_match_legacy_exhaustive(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            expected = _legacy_tp_groups(layout.world_size(), pp_size, dp_size, tp_size)
            self.assertEqual(layout.groups(Group.TP), expected, msg=f"layout={layout}")

    def test_dp_groups_match_legacy_exhaustive(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            expected = _legacy_dp_groups(layout.world_size(), pp_size, dp_size, tp_size)
            self.assertEqual(layout.groups(Group.DP), expected, msg=f"layout={layout}")

    def test_pp_groups_match_legacy_exhaustive(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            expected = _legacy_pp_groups(layout.world_size(), pp_size, dp_size, tp_size)
            self.assertEqual(layout.groups(Group.PP), expected, msg=f"layout={layout}")

    def test_world_group_spans_all_ranks_exhaustive(self):
        # WORLD is the whole lattice: one group of every rank.
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            all_ranks = list(range(layout.world_size()))
            self.assertEqual(layout.size_of(Group.WORLD), layout.world_size())
            self.assertEqual(layout.groups(Group.WORLD), [all_ranks])
            for r in all_ranks:
                self.assertEqual(layout.group_of(Group.WORLD, r), all_ranks)
                self.assertEqual(layout.rank_in_group(Group.WORLD, r), r)

    def test_stage_groups_are_per_stage_exhaustive(self):
        """STAGE: one group per pp stage holding all its dp*tp ranks; the
        in-group position is the lane-local index dp * tp_size + tp."""
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            stride = dp_size * tp_size
            expected = [
                list(range(p * stride, (p + 1) * stride)) for p in range(pp_size)
            ]
            self.assertEqual(layout.size_of(Group.STAGE), stride)
            self.assertEqual(
                layout.groups(Group.STAGE), expected, msg=f"layout={layout}"
            )
            for r in range(layout.world_size()):
                coord = layout.coord_of(r)
                self.assertEqual(
                    layout.group_of(Group.STAGE, r),
                    expected[coord.pp],
                    msg=f"layout={layout}, rank={r}",
                )
                self.assertEqual(
                    layout.rank_in_group(Group.STAGE, r),
                    coord.dp * tp_size + coord.tp,
                    msg=f"layout={layout}, rank={r}",
                )

    def test_groups_partition_world_exhaustive(self):
        # Every axis' groups must partition [0, world_size) exactly once.
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            for axis in (Group.TP, Group.DP, Group.PP, Group.STAGE):
                groups = layout.groups(axis)
                flat = sorted(r for g in groups for r in g)
                self.assertEqual(
                    flat,
                    list(range(layout.world_size())),
                    msg=f"layout={layout}, axis={axis}",
                )
                self.assertTrue(
                    all(len(g) == layout.size_of(axis) for g in groups),
                    msg=f"layout={layout}, axis={axis}",
                )

    def test_group_of_and_rank_in_group(self):
        layout = RankLayout(world_size=8, pp_size=2, dp_size=2, tp_size=2)
        # rank 5 = pp=1, dp=0, tp=1
        self.assertEqual(layout.coord_of(5), Coord(tp=1, dp=0, pp=1))
        self.assertEqual(layout.group_of(Group.TP, 5), [4, 5])
        self.assertEqual(layout.group_of(Group.DP, 5), [5, 7])
        self.assertEqual(layout.group_of(Group.PP, 5), [1, 5])
        self.assertEqual(layout.rank_in_group(Group.TP, 5), 1)
        self.assertEqual(layout.rank_in_group(Group.DP, 5), 0)
        self.assertEqual(layout.rank_in_group(Group.PP, 5), 1)

    def test_rank_in_group_equals_axis_coordinate_exhaustive(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            for r in range(layout.world_size()):
                coord = layout.coord_of(r)
                self.assertEqual(layout.rank_in_group(Group.TP, r), coord.tp)
                self.assertEqual(layout.rank_in_group(Group.DP, r), coord.dp)
                self.assertEqual(layout.rank_in_group(Group.PP, r), coord.pp)

    def test_degenerate_single_rank(self):
        layout = RankLayout(world_size=1)
        self.assertEqual(layout.world_size(), 1)
        self.assertEqual(layout.coord_of(0), Coord())
        for axis in (Group.TP, Group.DP, Group.PP):
            self.assertEqual(layout.groups(axis), [[0]])
            self.assertEqual(layout.group_of(axis, 0), [0])
            self.assertEqual(layout.rank_in_group(axis, 0), 0)

    def test_degenerate_pure_tp(self):
        # pp=dp=1 must reduce to the legacy pure-TP deployment exactly.
        layout = RankLayout(world_size=4, pp_size=1, dp_size=1, tp_size=4)
        self.assertEqual(layout.groups(Group.TP), [[0, 1, 2, 3]])
        self.assertEqual(layout.groups(Group.DP), [[0], [1], [2], [3]])
        self.assertEqual(layout.groups(Group.PP), [[0], [1], [2], [3]])


# Legacy ep_rank oracle: test reference only, never import from production code.
def _legacy_ep_rank(world_rank: int, ep_size: int) -> int:
    return world_rank % ep_size


class RankLayoutEpViewTest(unittest.TestCase):
    def test_ep_rank_matches_legacy_at_pp1_exhaustive(self):
        for dp_size, tp_size in itertools.product((1, 2, 3), repeat=2):
            layout = RankLayout(
                world_size=dp_size * tp_size,
                pp_size=1,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            for ep_size in (1, tp_size, dp_size, dp_size * tp_size):
                for r in range(layout.world_size()):
                    self.assertEqual(
                        layout.ep_rank_of(r, ep_size),
                        _legacy_ep_rank(r, ep_size),
                        msg=f"layout={layout}, ep_size={ep_size}, rank={r}",
                    )

    def test_ep_rank_is_lane_local_under_pp(self):
        layout = RankLayout(world_size=8, pp_size=2, dp_size=2, tp_size=2)
        stride = layout.lane_stride()
        for ep_size in (2, 4):
            for r in range(layout.world_size()):
                self.assertEqual(layout.ep_rank_of(r, ep_size), (r % stride) % ep_size)

    def test_ep_rank_rejects_non_divisible_ep_size(self):
        layout = RankLayout(world_size=4, pp_size=1, dp_size=2, tp_size=2)  # stride 4
        with self.assertRaises(ValueError):
            layout.ep_rank_of(0, 3)
        self.assertEqual(layout.ep_rank_of(3, 1), 0)

    def test_ep_rank_rejects_out_of_range(self):
        for pp_size, dp_size, tp_size in _SWEEP:
            layout = RankLayout(
                world_size=pp_size * dp_size * tp_size,
                pp_size=pp_size,
                dp_size=dp_size,
                tp_size=tp_size,
            )
            for ep_size in (1, layout.lane_stride()):
                for rank in (-1, layout.world_size()):
                    with self.subTest(layout=layout, ep_size=ep_size, rank=rank):
                        with self.assertRaises(ValueError):
                            layout.ep_rank_of(rank, ep_size)


if __name__ == "__main__":
    unittest.main()
