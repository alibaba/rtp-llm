import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.cpp.normal_engine.test.libdsv4_dp_decode_runtime_test_bindings import (
    Mode,
    Phase,
    Round,
    Runtime,
    StartupStage,
    Status,
    prepare_cache_capacity,
    prepare_model,
    prepare_speculative_input,
    prepare_startup,
    report_invalid_speculative_input,
)


def check_execution(test, execution, mode, phase, bucket, token_width):
    test.assertEqual(
        (execution.mode, execution.phase, execution.bucket, execution.token_width),
        (mode, phase, bucket, token_width),
    )


def schedule_round(runtime, requests):
    assert runtime.begin_round(False, False) == Round.EXECUTE
    return runtime.agree_scheduled_requests(requests)


def run_control_protocol(runtime, rank):
    test = unittest.TestCase()
    phase = Phase.TARGET
    test.assertEqual(schedule_round(runtime, 0), Round.IDLE)
    requests = (3, 0, 1, 0)[rank]
    test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
    test.assertEqual(schedule_round(runtime, -1 if rank == 1 else 0), Round.INVALID)
    for paused_rank in (1, 1, 3):
        test.assertEqual(runtime.begin_round(rank == paused_rank, False), Round.PAUSED)
    test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)

    for stage, peer_error in (
        (StartupStage.WARMUP_INPUTS, "warmup input preparation failed"),
        (StartupStage.CACHE_MANAGER, "KV cache manager initialization failed"),
        (StartupStage.ENGINE_READY, "engine initialization failed"),
    ):
        fail = rank == 1
        with test.assertRaisesRegex(
            RuntimeError, "injected local startup failure" if fail else peer_error
        ):
            prepare_startup(runtime, stage, fail)
        # A startup error already notified its peers. No second vote is sent.
        prepare_startup(runtime, stage, False)
    fail = rank == 1
    with test.assertRaisesRegex(
        RuntimeError,
        (
            "injected local cache capacity failure"
            if fail
            else "KV cache capacity preparation failed"
        ),
    ):
        prepare_cache_capacity(runtime, 5, fail)
    with test.assertRaisesRegex(RuntimeError, "KV cache capacity preparation failed"):
        prepare_cache_capacity(runtime, 2**31 if rank == 2 else 5, False)
    test.assertEqual(prepare_cache_capacity(runtime, 5 - rank, False), 2)
    with test.assertRaisesRegex(RuntimeError, "model initialization failed"):
        runtime.verify_model_initialization(rank != 2)
    runtime.verify_model_initialization(True)

    warmup = runtime.plan(phase, 1, requests, False).execution
    check_execution(test, warmup, Mode.EAGER, phase, 0, 1)
    check_execution(
        test, runtime.finish_preparation(warmup, Status.READY), Mode.EAGER, phase, 0, 1
    )
    with test.assertRaisesRegex(RuntimeError, "capture buckets differ"):
        runtime.verify_capture(phase, 1, [1, 2] if rank == 1 else [1, 4])
    runtime.verify_capture(phase, 1, [1, 4, 2] if rank == 1 else [1, 2, 4])
    with test.assertRaisesRegex(RuntimeError, "capture input preparation failed"):
        runtime.verify_capture_readiness(phase, 4, False, rank != 1)
    with test.assertRaisesRegex(RuntimeError, "warmup failed"):
        runtime.verify_capture_readiness(phase, 4, True, rank != 3)
    with test.assertRaisesRegex(RuntimeError, "control mismatch"):
        runtime.verify_capture_readiness(phase, 2 if rank == 2 else 4, False, True)
    runtime.verify_capture_readiness(phase, 4, False, True)
    runtime.verify_capture_readiness(phase, 4, True, True)
    runtime.finish_capture(phase)

    planned = runtime.plan(phase, 1, requests, True).execution
    check_execution(test, planned, Mode.GRAPH, phase, 4, 1)
    agreed = runtime.finish_preparation(planned, Status.READY)
    check_execution(test, agreed, Mode.GRAPH, phase, 4, 1)

    oversized_requests = 5 if rank == 0 else 0
    test.assertEqual(schedule_round(runtime, oversized_requests), Round.EXECUTE)
    oversized = runtime.plan(phase, 1, oversized_requests, True).execution
    check_execution(test, oversized, Mode.EAGER, phase, 0, 1)
    runtime.finish_preparation(oversized, Status.READY)

    test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
    planned = runtime.plan(phase, 1, requests, True).execution
    fallback = runtime.finish_preparation(
        planned, Status.GRAPH_UNAVAILABLE if rank == 3 else Status.READY
    )
    check_execution(test, fallback, Mode.EAGER, phase, 0, 1)
    agreed = runtime.finish_preparation(fallback, Status.READY)
    check_execution(test, agreed, Mode.EAGER, phase, 0, 1)

    planned = runtime.plan(phase, 1, requests, True).execution
    with test.assertRaisesRegex(RuntimeError, "preparation failed"):
        runtime.finish_preparation(
            planned, Status.INVALID if rank == 0 else Status.READY
        )
    for wrong_count in (-1, requests + 1):
        malformed = runtime.plan(phase, 1, wrong_count if rank == 2 else requests, True)
        test.assertEqual(malformed.valid_local_input, rank != 2)
        with test.assertRaisesRegex(RuntimeError, "preparation failed"):
            prepare_model(runtime, malformed, True, False, False)
    mismatched = runtime.plan(phase, 2 if rank == 3 else 1, requests, True)
    with test.assertRaisesRegex(RuntimeError, "control mismatch"):
        prepare_model(runtime, mismatched, True, False, False)
    # A rank diverging to a speculative phase fails on the same exchange.
    with test.assertRaisesRegex(RuntimeError, "control mismatch"):
        mismatch = runtime.plan(
            Phase.TARGET_VERIFY if rank == 3 else phase, 1, requests, True
        )
        prepare_model(runtime, mismatch, True, False, False)
    test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
    with test.assertRaisesRegex(RuntimeError, "preparation failed"):
        if rank == 1:
            runtime.reject_unprepared_round()
        else:
            planned = runtime.plan(phase, 1, requests, True)
            prepare_model(runtime, planned, True, False, False)
    # The one-shot failure guard cannot publish a second vote while unwinding.
    runtime.reject_unprepared_target()


def run_model_preparation(runtime, rank):
    test = unittest.TestCase()
    phase = Phase.TARGET
    requests = (3, 0, 1, 0)[rank]
    test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)

    planned = runtime.plan(phase, 1, requests, True)
    execution, stages, can_replay = prepare_model(runtime, planned, True, False, False)
    check_execution(test, execution, Mode.GRAPH, phase, 4, 1)
    test.assertEqual(stages, ["prepare_graph"])
    test.assertTrue(can_replay)

    planned = runtime.plan(phase, 1, requests, True)
    execution, stages, can_replay = prepare_model(
        runtime, planned, rank != 3, False, False
    )
    check_execution(test, execution, Mode.EAGER, phase, 0, 1)
    test.assertEqual(stages, ["prepare_graph", "discard_graph", "prepare_eager"])
    test.assertFalse(can_replay)

    for graph_available, fail_graph, fail_eager in (
        (True, rank == 2, False),
        (rank != 3, False, rank == 2),
    ):
        planned = runtime.plan(phase, 1, requests, True)
        with test.assertRaisesRegex(
            RuntimeError, "preparation failed on at least one rank"
        ):
            prepare_model(runtime, planned, graph_available, fail_graph, fail_eager)
        # A failed readiness vote must leave every peer ready for the next round.
        test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
    with test.assertRaisesRegex(RuntimeError, "preparation failed"):
        planned = runtime.plan(phase, 1, requests, True)
        prepare_model(runtime, planned, rank != 3, False, False, rank == 2)


def run_speculative_phases(runtime, rank):
    test = unittest.TestCase()
    requests = (2, 0, 1, 0)[rank]
    # Three draft tokens, with DSpARK sampling directly from the anchor.
    mtp_draft = (Phase.MTP_DRAFT, 1)
    target_verify = (Phase.TARGET_VERIFY, 4)
    mtp_commit = (Phase.MTP_COMMIT, 4)
    dspark_proposal = (Phase.DSPARK_PROPOSAL, 3)
    dspark_commit = (Phase.DSPARK_COMMIT, 4)
    for phase, width in (
        mtp_draft,
        target_verify,
        mtp_commit,
        dspark_proposal,
        dspark_commit,
    ):
        runtime.verify_capture(phase, width, [1, 2, 4])
        runtime.finish_capture(phase)

    for phases in (
        (mtp_draft, mtp_draft, target_verify, mtp_commit),
        (dspark_proposal, target_verify, dspark_commit),
    ):
        test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
        for phase, width in phases:
            planned = runtime.plan(phase, width, requests, True)
            check_execution(test, planned.execution, Mode.GRAPH, phase, 2, width)
            fallback = phase == Phase.DSPARK_PROPOSAL
            agreed, stages, can_replay = prepare_model(
                runtime, planned, not (fallback and rank == 1), False, False
            )
            if fallback:
                check_execution(test, agreed, Mode.EAGER, phase, 0, width)
                test.assertEqual(
                    stages, ["prepare_graph", "discard_graph", "prepare_eager"]
                )
            else:
                check_execution(test, agreed, Mode.GRAPH, phase, 2, width)
                test.assertEqual(stages, ["prepare_graph"])
            test.assertEqual(can_replay, not fallback)
            if phase == Phase.TARGET_VERIFY:
                test.assertTrue(runtime.agree_speculative_status(True))

    phase, width = target_verify
    with test.assertRaisesRegex(RuntimeError, "invalid input geometry"):
        if rank == 2:
            runtime.reject_speculative_input(phase)
        else:
            runtime.plan(phase, width, requests, True)
    planned = runtime.plan(phase, width, requests, True)
    check_execution(
        test,
        runtime.finish_preparation(planned.execution, Status.READY),
        Mode.GRAPH,
        phase,
        2,
        width,
    )
    test.assertFalse(runtime.agree_speculative_status(rank != 3))
    for i, (phase, width) in enumerate(
        (mtp_draft, target_verify, mtp_commit, dspark_proposal, dspark_commit)
    ):
        failing_rank = i % 4
        test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
        if rank == failing_rank:
            with test.assertRaisesRegex(
                RuntimeError, "injected rank-local preparation failure"
            ):
                prepare_speculative_input(runtime, phase, requests, True)
        else:
            prepared = prepare_speculative_input(runtime, phase, requests, False)
            with test.assertRaisesRegex(RuntimeError, "invalid input geometry"):
                runtime.plan(phase, width, prepared, True)

        # The error path must vote exactly once; a following round can plan normally.
        test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
        planned = runtime.plan(phase, width, requests, True)
        check_execution(
            test,
            runtime.finish_preparation(planned.execution, Status.READY),
            Mode.GRAPH,
            phase,
            2,
            width,
        )

    phase, width = mtp_commit
    if rank == 1:
        test.assertEqual(report_invalid_speculative_input(runtime, phase, 17), 17)
    else:
        with test.assertRaisesRegex(RuntimeError, "invalid input geometry"):
            runtime.plan(phase, width, requests, True)
    test.assertEqual(schedule_round(runtime, 0), Round.IDLE)


def run_first_phase_rejection(world_size, rank):
    test = unittest.TestCase()
    for phase, width in (
        (Phase.MTP_DRAFT, 1),
        (Phase.TARGET_VERIFY, 2),
        (Phase.DSPARK_PROPOSAL, 3),
    ):
        runtime = Runtime(world_size, first_phase=phase)
        try:
            runtime.verify_capture(phase, width, [1, 2])
            runtime.finish_capture(phase)
            requests = int(rank == 0)
            test.assertEqual(schedule_round(runtime, requests), Round.EXECUTE)
            with test.assertRaisesRegex(RuntimeError, "invalid input geometry"):
                if rank == 1:
                    runtime.reject_unprepared_round()
                else:
                    runtime.plan(phase, width, requests, True)
            if rank == 1:
                runtime.reject_unprepared_round()
            test.assertEqual(schedule_round(runtime, 0), Round.IDLE)
        finally:
            del runtime


def worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        runtime = Runtime(world_size)
        try:
            run_control_protocol(runtime, rank)
            run_model_preparation(runtime, rank)
            run_speculative_phases(runtime, rank)
            unittest.TestCase().assertEqual(
                runtime.begin_round(rank == 3, rank == 1), Round.STOP
            )
        finally:
            del runtime
        run_first_phase_rejection(world_size, rank)
        # The runtime must destroy its subgroups without destroying WORLD.
        dist.barrier()
    finally:
        dist.destroy_process_group()


class Dsv4DpDecodeRuntimeTest(unittest.TestCase):
    def test_gloo_control_protocol(self):
        with TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "store").as_uri()
            mp.spawn(worker, args=(4, rendezvous), nprocs=4)


if __name__ == "__main__":
    unittest.main()
