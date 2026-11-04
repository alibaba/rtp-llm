"""DP8 step-coordinator tests (L3, D/DP8 role, plan section 3.2).

All tests drive eight per-rank coordinators over the single-process
:class:`LocalCollectiveSimulator` (the ``LocalEnpmGroup`` pattern): one
busy rank + seven empty, skewed batches, a rank finishing early, and the
requirement that every rank executes the agreed collective order on every
step.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.dp_coord import (
    DPStepCoordinator,
    LocalCollectiveSimulator,
    OwnerRouter,
    PhaseTracker,
    RankStepState,
    TorchCollectiveBackend,
    collective_phase_order,
    expected_backend_calls,
)

NUM_LAYERS = 4  # small phase list; the order logic is layer-count agnostic
WORLD = 8


def _drive_rank(coordinator, plan, tracker, *, skip_phases=()):
    """Execute the agreed phases for one rank (the decode loop's shape).

    ENP query = gather ids + all-reduce over the backend; per-layer EP
    dispatch/combine ride deep_ep (tracked, not driven here); the release
    barrier runs inside ``finish_step``.
    """
    for phase in plan.collective_phases:
        if phase in skip_phases:
            continue  # simulate a rank that (incorrectly) skips a phase
        if phase == "step_exchange":
            continue  # already executed by begin_step
        if phase == "enp_query":
            coordinator.backend.all_gather_ints(torch.zeros(3, dtype=torch.int64))
            coordinator.backend.all_reduce_max_int(torch.zeros(1, dtype=torch.int64))
            tracker.enter(phase)
        elif phase == "step_barrier":
            continue  # executed by finish_step
        else:
            tracker.enter(phase)
    coordinator.finish_step(tracker, plan)


def _run_world(states, *, skip=None, num_layers=NUM_LAYERS):
    """Drive all 8 ranks through one coordinated step; return plans."""
    simulator = LocalCollectiveSimulator(size=WORLD)
    simulator.set_local_states(states)
    coordinators = [
        DPStepCoordinator(simulator.backend(rank), num_layers=num_layers)
        for rank in range(WORLD)
    ]
    plans = []
    for rank, coordinator in enumerate(coordinators):
        plan, tracker = coordinator.begin_step(states[rank])
        plans.append(plan)
        if plan.run:
            skip_phases = () if skip is None or rank not in skip else skip[rank]
            _drive_rank(coordinator, plan, tracker, skip_phases=skip_phases)
        else:
            coordinator.finish_step(tracker, plan)
    return simulator, coordinators, plans


class PhaseOrderTest(unittest.TestCase):
    def test_order_is_enp_then_ep_per_layer_then_barrier(self):
        phases = collective_phase_order(3)
        self.assertEqual(
            phases,
            (
                "step_exchange",
                "enp_query",
                "ep_dispatch_L0",
                "ep_combine_L0",
                "ep_dispatch_L1",
                "ep_combine_L1",
                "ep_dispatch_L2",
                "ep_combine_L2",
                "step_barrier",
            ),
        )

    def test_backend_collective_count(self):
        self.assertEqual(expected_backend_calls(collective_phase_order(40)), 4)


class StepExchangeTest(unittest.TestCase):
    def test_one_busy_rank_seven_empty(self):
        states = [RankStepState() for _ in range(WORLD)]
        states[3] = RankStepState(valid_requests=2, tokens=2)
        simulator, coordinators, plans = _run_world(states)
        # Identical world view on every rank (each plan was derived from its
        # own rank's exchange of the same collective).
        for rank in range(WORLD):
            self.assertEqual(plans[rank].world_tokens, 2)
            self.assertEqual(plans[rank].collective_phases, plans[0].collective_phases)
        self.assertEqual(plans[0].rank, 0)
        # Only the empty ranks synthesize placeholders.
        self.assertEqual(plans[3].local_tokens, 2)
        self.assertEqual(plans[3].placeholder_rows, 0)
        for rank in range(WORLD):
            if rank == 3:
                continue
            self.assertEqual(plans[rank].local_tokens, 0)
            self.assertEqual(plans[rank].placeholder_rows, 2)
            self.assertEqual(plans[rank].total_rows, 2)
        simulator.verify_all_ranks_participated()
        simulator.verify_collective_order(plans[0].collective_phases)

    def test_exchange_returns_the_full_world_tensor(self):
        simulator = LocalCollectiveSimulator(size=WORLD)
        states = [RankStepState() for _ in range(WORLD)]
        states[5] = RankStepState(valid_requests=2, tokens=3)
        simulator.set_local_states(states)
        coordinators = [
            DPStepCoordinator(simulator.backend(rank), num_layers=NUM_LAYERS)
            for rank in range(WORLD)
        ]
        views = [coordinators[r].exchange_step_state(states[r]) for r in range(WORLD)]
        for view in views:
            self.assertEqual(view.per_rank.shape, (WORLD, 3))
            self.assertEqual(int(view.per_rank[5][1]), 3)
            self.assertEqual(int(view.world_tokens), 3)
            self.assertTrue(view.run_step)
            self.assertEqual(int(view.max_rank_tokens), 3)

    def test_skewed_batches(self):
        tokens = [0, 1, 3, 0, 2, 5, 0, 1]
        states = [RankStepState(valid_requests=t, tokens=t) for t in tokens]
        simulator, _, plans = _run_world(states)
        for rank, token_count in enumerate(tokens):
            if token_count == 0:
                # Placeholder count follows the group maximum (5 here).
                self.assertEqual(plans[rank].placeholder_rows, 5)
            else:
                self.assertEqual(plans[rank].placeholder_rows, 0)
                self.assertEqual(plans[rank].total_rows, token_count)
            self.assertEqual(plans[rank].world_tokens, 12)
        simulator.verify_all_ranks_participated()
        simulator.verify_collective_order(plans[0].collective_phases)

    def test_drained_world_skips_compute_phases(self):
        states = [RankStepState() for _ in range(WORLD)]
        simulator, _, plans = _run_world(states)
        self.assertTrue(all(not plan.run for plan in plans))
        for plan in plans:
            self.assertEqual(plan.placeholder_rows, 0)
        simulator.verify_all_ranks_participated()
        # Drained steps still exchange and barrier, nothing else.
        simulator.verify_collective_order(("step_exchange", "step_barrier"))


class EarlyFinishTest(unittest.TestCase):
    def test_finished_rank_still_executes_all_collectives(self):
        states = [RankStepState() for _ in range(WORLD)]
        states[2] = RankStepState(valid_requests=1, finished_requests=1)
        states[5] = RankStepState(valid_requests=3, tokens=3)
        simulator, coordinators, plans = _run_world(states)
        self.assertTrue(plans[2].run)  # other ranks still have tokens
        self.assertEqual(plans[2].local_tokens, 0)
        self.assertEqual(plans[2].placeholder_rows, 3)
        # Finishing changes only that request's validity, never the schedule.
        coordinators[2].mark_finished(42)
        validity = coordinators[2].request_validity([1, 42])
        self.assertEqual(validity, {1: True, 42: False})
        simulator.verify_all_ranks_participated()
        simulator.verify_collective_order(plans[0].collective_phases)

    def test_mark_finished_does_not_skip_collectives(self):
        states = [RankStepState(valid_requests=1, tokens=1) for _ in range(WORLD)]
        simulator = LocalCollectiveSimulator(size=WORLD)
        simulator.set_local_states(states)
        coordinators = [
            DPStepCoordinator(simulator.backend(rank), num_layers=NUM_LAYERS)
            for rank in range(WORLD)
        ]
        # Every rank finishes its request before the step; the world still
        # agrees to run THIS step because the states carry one token each.
        for coordinator in coordinators:
            coordinator.mark_finished(7)
        plans = []
        for rank, coordinator in enumerate(coordinators):
            plan, tracker = coordinator.begin_step(states[rank])
            plans.append(plan)
            _drive_rank(coordinator, plan, tracker)
        self.assertTrue(all(plan.run for plan in plans))
        simulator.verify_all_ranks_participated()


class OrderViolationTest(unittest.TestCase):
    def test_skipping_a_phase_is_detected(self):
        states = [RankStepState() for _ in range(WORLD)]
        states[1] = RankStepState(valid_requests=1, tokens=1)
        with self.assertRaises(RuntimeError):
            _run_world(states, skip={4: ("enp_query",)})

    def test_tracker_detects_reordered_phases(self):
        tracker = PhaseTracker()
        phases = collective_phase_order(1)
        tracker.enter("step_exchange")
        tracker.enter("enp_query")
        tracker.enter("ep_dispatch_L0")
        tracker.enter("ep_combine_L0")
        tracker.enter("ep_combine_L0")  # duplicated, order broken
        with self.assertRaises(RuntimeError):
            tracker.finish_step(phases)


class PlaceholderBatchTest(unittest.TestCase):
    def test_placeholder_conventions(self):
        states = [RankStepState() for _ in range(WORLD)]
        states[0] = RankStepState(valid_requests=4, tokens=4)
        _, _, plans = _run_world(states)
        batch = plans[3].placeholder_batch(slots=8)
        self.assertTrue(bool((batch.expert_ids == -1).all()))
        self.assertTrue(bool((batch.expert_weights == 0).all()))
        self.assertFalse(bool(batch.valid_mask.any()))
        self.assertTrue(bool((batch.positions == -1).all()))
        self.assertTrue(bool((batch.request_ids == -1).all()))
        self.assertEqual(tuple(batch.expert_ids.shape), (4, 8))

    def test_busy_rank_has_no_placeholders(self):
        states = [RankStepState(valid_requests=1, tokens=1)] + [
            RankStepState() for _ in range(WORLD - 1)
        ]
        _, _, plans = _run_world(states)
        self.assertEqual(plans[0].placeholder_rows, 0)


class OwnerRouterTest(unittest.TestCase):
    def test_routing_is_stable_across_steps(self):
        router = OwnerRouter(dp_size=8)
        first = {rid: router.owner(rid) for rid in range(100)}
        for _ in range(3):
            for rid, owner in first.items():
                self.assertEqual(router.owner(rid), owner)
        self.assertTrue(all(0 <= owner < 8 for owner in first.values()))

    def test_sequential_requests_spread_round_robin(self):
        router = OwnerRouter(dp_size=8)
        owners = {router.owner(rid) for rid in range(80)}
        self.assertEqual(owners, set(range(8)))

    def test_is_local_matches_owner(self):
        router = OwnerRouter(dp_size=8)
        for rid in range(16):
            owner = router.owner(rid)
            self.assertTrue(router.is_local(rid, owner))
            self.assertFalse(router.is_local(rid, (owner + 1) % 8))

    def test_explicit_map_overrides(self):
        router = OwnerRouter(dp_size=8, explicit={100: 3})
        self.assertEqual(router.owner(100), 3)
        self.assertEqual(router.owner(101), 101 % 8)
        self.assertEqual(router.owners_of([100, 101]), {100: 3, 101: 101 % 8})


class TorchBackendTest(unittest.TestCase):
    def test_backend_surface_is_callable(self):
        # Construction only: no process group is initialized on a CPU host.
        backend = TorchCollectiveBackend.__new__(TorchCollectiveBackend)
        for name in ("all_gather_ints", "all_reduce_max_int", "barrier"):
            self.assertTrue(callable(getattr(backend, name)))
        for name in ("rank", "size"):
            self.assertTrue(isinstance(getattr(TorchCollectiveBackend, name), property))


if __name__ == "__main__":
    unittest.main()
