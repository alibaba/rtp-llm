package org.flexlb.balance.endpoint;

import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.ToLongFunction;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillStateSnapshotTest {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillActiveIndex waiting = PrefillActiveIndex.ordered(4,
            Comparator.comparingLong(ScheduledRequest::requestId));
    private final PrefillState state = new PrefillState(lock, waiting, clock::get, () -> { });
    private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });

    @Test
    void concurrentReadersShareACompleteImmutableMaterialization() throws Exception {
        var second = state.reserveUnqueuedRoute(item(2), 20, Long.MAX_VALUE).reservation();
        var first = state.reserveUnqueuedRoute(item(1), 10, Long.MAX_VALUE).reservation();
        var captured = capture();
        try (var executor = Executors.newFixedThreadPool(8)) {
            CountDownLatch start = new CountDownLatch(1);
            List<Future<WorkSnapshot>> readers = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                readers.add(executor.submit(() -> {
                    assertTrue(start.await(5, TimeUnit.SECONDS));
                    return captured.work().materialize();
                }));
            }
            start.countDown();
            WorkSnapshot shared = readers.getFirst().get(5, TimeUnit.SECONDS);
            for (var reader : readers) {
                assertSame(shared, reader.get(5, TimeUnit.SECONDS));
            }
            assertEquals(List.of(1L, 2L), shared.requests().stream()
                    .map(WorkSnapshot.RequestWork::requestId).toList());
            second.close();
            first.close();
            assertEquals(2, shared.requests().size());
            assertTrue(capture().work().materialize().requests().isEmpty());
        }
    }

    @Test
    void clockRollbackRecapturesWorkWithoutMutatingEarlierSnapshots() {
        state.reserveUnqueuedRoute(item(1), 10, Long.MAX_VALUE);
        var original = capture();
        clock.set(101);
        assertSame(original.work(), capture().work());
        clock.set(90);
        var rebased = capture();
        assertNotSame(original.work(), rebased.work());
        assertEquals(100, original.work().materialize().capturedAtMs());
        assertEquals(90, rebased.work().materialize().capturedAtMs());
        assertEquals(rebased.capturedAtMs(), rebased.work().materialize().capturedAtMs());
        assertEquals(original.work().materialize().requests(), rebased.work().materialize().requests());
    }

    @Test
    void queuedAndImmediateRoutesUseOneCommitAndTerminalOwnership() {
        ScheduledRequest queued = item(1), immediate = item(2);
        enqueue(queued);
        try (var queuedReservation = state.reserveRoute(queued, 30L).reservation();
             var immediateReservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation()) {
            assertEquals(List.of(queued), waitingItems());
            assertEquals(1, state.committedSnapshot().requests().size());
            try (var handoff = state.commitRouteGroup(List.of(queued, immediate),
                    List.of(queuedReservation, immediateReservation), generation.tryAcquireHandoff())) {
                assertTrue(waitingItems().isEmpty());
                assertEquals(0L, handoff.precedingWork().totalRemainingWorkMs().orElseThrow(),
                        "the selected group's own provisional work is excluded");
                assertEquals(List.of(1L, 2L), state.committedSnapshot().requests().stream()
                        .map(work -> work.requestId()).toList());
                assertEquals(70L, state.committedSnapshot().knownRemainingWorkMsAt(System.currentTimeMillis()));
            }
        }
        assertEquals(2, state.committedSnapshot().requests().size(), "closing committed capabilities cannot release Engine ownership");
        assertTrue(state.terminalizeCommittedItem(queued));
        assertTrue(state.terminalizeCommittedItem(immediate));
        assertTrue(state.committedSnapshot().requests().isEmpty());
    }

    @Test
    void immediatePreparationIsVisibleToLaterPredictionsAndRollbackRemovesIt() {
        ScheduledRequest immediate = item(1);
        try (var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation()) {
            assertTrue(waitingItems().isEmpty());
            assertEquals(40L, capture().work().materialize().knownRemainingWorkMsAt(System.currentTimeMillis()));
            reservation.updatePrediction(immediate, 80L);
            assertEquals(80L, capture().work().materialize().knownRemainingWorkMsAt(System.currentTimeMillis()));
        }
        assertTrue(capture().work().materialize().requests().isEmpty());
        assertTrue(state.committedSnapshot().requests().isEmpty());
    }

    @Test
    void leaseRollbackRetainsQueuedWorkButReleasesUnqueuedAdmission() {
        ScheduledRequest queued = item(1), immediate = item(2);
        enqueue(queued);
        state.reserveRoute(queued, 30L).reservation().close();
        state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation().close();
        assertEquals(List.of(queued), waitingItems());
        assertTrue(state.committedSnapshot().requests().isEmpty());
        assertTrue(state.reserveRoute(queued, 35L).reservation() != null);
    }

    @Test
    void routeCommitRejectsMissingQueueIndexBeforeCommittingAnyMember() {
        ScheduledRequest queued = item(1), immediate = item(2);
        enqueue(queued);
        try (var queuedReservation = state.reserveRoute(queued, 30L).reservation();
             var immediateReservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation()) {
            assertTrue(waiting.remove(queued));
            try (var handoff = generation.tryAcquireHandoff()) {
                assertThrows(IllegalStateException.class, () -> state.commitRouteGroup(
                        List.of(immediate, queued), List.of(immediateReservation, queuedReservation), handoff));
            }
            assertFalse(state.terminalizeCommittedItem(immediate));
            assertTrue(waiting.add(queued));
            try (var handoff = state.commitRouteGroup(List.of(immediate, queued),
                    List.of(immediateReservation, queuedReservation), generation.tryAcquireHandoff())) {
                assertTrue(waiting.isEmpty());
            }
        }
    }

    @Test
    void retirementProjectsQueuedPreparedAndCommittedItemsWithoutModeExceptions() {
        ScheduledRequest queued = item(1), prepared = item(2), committed = item(3);
        enqueue(queued);
        var preparedReservation = state.reserveUnqueuedRoute(prepared, 20L, Long.MAX_VALUE).reservation();
        var committedReservation = state.reserveUnqueuedRoute(committed, 30L, Long.MAX_VALUE).reservation();
        try (var handoff = state.commitRouteGroup(List.of(committed), List.of(committedReservation),
                generation.tryAcquireHandoff())) {
            assertEquals(2, state.committedSnapshot().requests().size());
        }
        var retired = state.retireGenerationOwnership();
        assertNull(retired.invariantFailure());
        assertEquals(3, retired.ownedItems().size());
        assertTrue(retired.ownedItems().containsAll(List.of(queued, prepared, committed)));
        preparedReservation.close();
        committedReservation.close();
        assertTrue(state.committedSnapshot().requests().isEmpty());
        assertTrue(waiting.isEmpty());
    }

    @Test
    void cancellationBeforeCommitReleasesAnUnqueuedAdmissionExactlyOnce() {
        ScheduledRequest immediate = item(1);
        var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation();
        lock.lock();
        try {
            assertTrue(state.terminalizeActiveRouteUnderLock(immediate, reservation));
            assertFalse(state.terminalizeActiveRouteUnderLock(immediate, reservation));
        } finally {
            lock.unlock();
        }
        reservation.close();
        assertTrue(state.committedSnapshot().requests().isEmpty());
        assertTrue(capture().work().materialize().requests().isEmpty());
    }

    @Test
    void interleavedImmediateCommitsCaptureOtherWorkAtTheActualCommitBoundary() {
        ScheduledRequest first = item(1), second = item(2);
        try (var firstReservation = state.reserveUnqueuedRoute(first, 30L, Long.MAX_VALUE).reservation();
             var secondReservation = state.reserveUnqueuedRoute(second, 40L, Long.MAX_VALUE).reservation()) {
            try (var secondHandoff = state.commitRouteGroup(List.of(second), List.of(secondReservation),
                    generation.tryAcquireHandoff())) {
                assertEquals(30L, secondHandoff.precedingWork().totalRemainingWorkMs().orElseThrow());
            }
            try (var firstHandoff = state.commitRouteGroup(List.of(first), List.of(firstReservation),
                    generation.tryAcquireHandoff())) {
                assertEquals(40L, firstHandoff.precedingWork().totalRemainingWorkMs().orElseThrow(),
                        "the earlier reservation must see work committed before its own commit");
            }
        }
    }

    @Test
    void batchAndIndividualCommitsCaptureTheSamePrecedingTimeline() {
        ScheduledRequest immediate = item(1), batchMember = item(2), later = item(3);
        try (var immediateReservation = state.reserveUnqueuedRoute(immediate, 30L, Long.MAX_VALUE).reservation();
             var firstHandoff = state.commitRouteGroup(List.of(immediate), List.of(immediateReservation),
                     generation.tryAcquireHandoff())) {
            assertEquals(0L, firstHandoff.precedingWork().totalRemainingWorkMs().orElseThrow());
            enqueue(batchMember);
            try (var batchReservation = state.reserveBatch(batchMember, 10L, 2, generation.tryAcquireHandoff()).reservation();
                 var batchHandoff = batchReservation.commit(List.of(batchMember), 40L)) {
                assertEquals(30L, batchHandoff.precedingWork().totalRemainingWorkMs().orElseThrow());
            }
            try (var laterReservation = state.reserveUnqueuedRoute(later, 50L, Long.MAX_VALUE).reservation();
                 var laterHandoff = state.commitRouteGroup(List.of(later), List.of(laterReservation),
                         generation.tryAcquireHandoff())) {
                assertEquals(70L, laterHandoff.precedingWork().totalRemainingWorkMs().orElseThrow());
            }
        }
    }

    @Test
    void localCleanupPreservesRunningBatchWorkAndItsClock() {
        ScheduledRequest a = item(1), b = item(2), c = item(3);
        commitBatch(List.of(a, b, c), 300L);
        reconcile(Map.of(), Map.of("1", task(1, TaskPhase.RUNNING, 0, 0),
                "2", task(2, TaskPhase.RUNNING, 0, 0),
                "3", task(3, TaskPhase.RUNNING, 0, 0)), unused -> {
                    throw new AssertionError("an unchanged batch needs no prediction");
                });

        clock.set(150);
        assertTrue(state.terminalizeCommittedItem(a));
        assertEquals(List.of(2L, 3L), state.committedSnapshot().batches().getFirst().requestIds());
        assertEquals(250L, remainingWork());
        clock.set(190);
        assertTrue(state.terminalizeCommittedItem(b));
        assertFalse(state.terminalizeCommittedItem(a));
        assertFalse(state.terminalizeCommittedItem(item(3)), "cleanup must match the exact item");
        assertEquals(210L, remainingWork());

        // Returning to QUEUED pauses elapsed-time accounting, but does not undo work already started.
        clock.set(230);
        reconcile(Map.of("1", task(1, null, 0, 100)),
                Map.of("3", task(3, TaskPhase.RECEIVED, 0, 0)), unused -> {
                    throw new AssertionError("local cancellation cannot shrink running work");
                });
        clock.set(250);
        assertEquals(170L, remainingWork());
        assertEquals(WorkSnapshot.Phase.ENGINE_QUEUED,
                state.committedSnapshot().batches().getFirst().phase());

        reconcile(Map.of("3", task(3, null, 0, 300)), Map.of(), unused -> {
            throw new AssertionError("a completed batch needs no prediction");
        });
        assertTrue(state.committedSnapshot().batches().isEmpty());
        assertEquals(0L, remainingWork());
    }

    @ParameterizedTest
    @CsvSource({"true,false,500,0", "false,true,500,0",
            "false,false,0,40", "false,false,500,40"})
    void executionEvidencePreservesBatchPrediction(
            boolean previouslyRunning, boolean currentlyRunning, long errorCode, long executionMs) {
        commitBatch(List.of(item(1), item(2), item(3)), 300L);
        if (previouslyRunning) {
            reconcile(Map.of(), Map.of("2", task(2, TaskPhase.RUNNING, 0, 0)), unused -> {
                throw new AssertionError("an unchanged batch needs no prediction");
            });
        }
        clock.set(150);
        Map<String, TaskInfo> active = currentlyRunning
                ? Map.of("2", task(2, TaskPhase.RUNNING, 0, 0)) : Map.of();
        reconcile(Map.of("1", task(1, null, errorCode, executionMs)), active, unused -> {
            throw new AssertionError("started Prefill must retain its batch prediction");
        });
        long remaining = previouslyRunning ? 250L : 300L;
        assertEquals(remaining, remainingWork());
        clock.set(190);
        long afterRunning = previouslyRunning || currentlyRunning ? remaining - 40L : remaining;
        assertEquals(afterRunning, remainingWork());
        assertFalse(state.committedSnapshot().hasUnknownWork());

        reconcile(Map.of("2", task(2, null, 500, 0)),
                Map.of("3", task(3, TaskPhase.RECEIVED, 0, 0)), unused -> {
                    throw new AssertionError("a queued observation cannot make started work repackable");
                });
        clock.set(230);
        assertEquals(afterRunning, remainingWork());
        reconcile(Map.of("3", task(3, null, 0, 300)), Map.of(), unused -> {
            throw new AssertionError("a completed batch needs no prediction");
        });
        assertTrue(state.committedSnapshot().batches().isEmpty());
    }

    @Test
    void onlyWorkerRemovalBeforeExecutionRepredictsTheQueuedBatch() {
        ScheduledRequest a = item(1), b = item(2);
        commitBatch(List.of(a, b), 300L);
        clock.set(1_000);
        reconcile(Map.of("1", task(1, null, 500, 0)),
                Map.of("2", task(2, TaskPhase.RECEIVED, 0, 0)), survivors -> {
                    assertEquals(List.of(b), survivors);
                    return 200L;
                });
        clock.set(1_200);
        assertEquals(200L, remainingWork(), "unstarted work does not age while queued");
        reconcile(Map.of(), Map.of("2", task(2, TaskPhase.RUNNING, 0, 0)), unused -> {
            throw new AssertionError("starting the batch uses its existing prediction");
        });
        clock.set(1_250);
        assertEquals(150L, remainingWork());
    }

    private void commitBatch(List<ScheduledRequest> members, long predictedMs) {
        members.forEach(this::enqueue);
        try (var reservation = state.reserveBatch(members.getFirst(), 10L, 2,
                generation.tryAcquireHandoff()).reservation();
             var handoff = reservation.commit(members, predictedMs)) {
            assertTrue(waitingItems().isEmpty());
        }
    }

    private void reconcile(Map<String, TaskInfo> finished, Map<String, TaskInfo> active,
                           ToLongFunction<List<ScheduledRequest>> repredictor) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(active);
        var observation = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090)
                .freezeStatusResponse(response);
        var outcome = state.reconcileWorkerStatus(observation,
                repredictor, () -> { }, () -> { });
        assertNull(outcome.publicationFailure());
    }

    private long remainingWork() {
        return state.committedSnapshot().totalRemainingWorkMsAt(clock.get()).orElseThrow();
    }

    private static TaskInfo task(long requestId, TaskPhase phase, long errorCode, long executionMs) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(10L);
        task.setPhase(phase);
        task.setErrorCode(errorCode);
        task.setExecutionTimeMs(executionMs);
        return task;
    }

    private static ScheduledRequest item(long id) {
        var item = mock(ScheduledRequest.class);
        when(item.requestId()).thenReturn(id);
        when(item.seqLen()).thenReturn(100L);
        return item;
    }

    private PrefillState.Snapshot capture() {
        lock.lock();
        try {
            return state.snapshotUnderLock();
        } finally {
            lock.unlock();
        }
    }

    private void enqueue(ScheduledRequest item) {
        lock.lock();
        try {
            assertTrue(state.enqueueActiveUnderLock(item, 0L));
        } finally {
            lock.unlock();
        }
    }

    private List<ScheduledRequest> waitingItems() {
        return capture().activeItems();
    }
}
