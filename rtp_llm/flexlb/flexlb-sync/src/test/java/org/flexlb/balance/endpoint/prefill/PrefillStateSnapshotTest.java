package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.estimate.WorkView;
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
            Comparator.comparing(ScheduledRequest::requestId));
    private final PrefillState state = new PrefillState(lock, waiting, clock::get, () -> { });
    private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });

    @Test
    void workerStatusUnconfirmedCountExcludesQueuedAndConfirmedRequests() {
        enqueue(item(9));
        assertEquals(0, state.stats().waitingForWorkerStatusRequests());
        try (var reservation = reserveQueued(item(8), 10)) {
            assertEquals(0, state.stats().waitingForWorkerStatusRequests());
        }
        var request = item(1);
        try (var reservation = reserveQueued(request, 10);
                var handoff = state.commitSingleGroup(List.of(request), List.of(reservation),
                        generation.tryAcquireHandoff())) {
            assertEquals(1, state.stats().waitingForWorkerStatusRequests());
        }
        reconcile(Map.of(), Map.of("1", task(1, TaskPhase.RUNNING, 0, 0)), unused -> 0L);
        assertEquals(0, state.stats().waitingForWorkerStatusRequests());
        assertEquals(1, state.stats().committedRequests());
    }

    @Test
    void concurrentReadersShareACompleteImmutableMaterialization() throws Exception {
        ScheduledRequest first = item(1), second = item(2);
        commitRoute(second, 20);
        commitRoute(first, 10);
        var captured = capture();
        try (var executor = Executors.newFixedThreadPool(8)) {
            CountDownLatch start = new CountDownLatch(1);
            List<Future<WorkView>> readers = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                readers.add(executor.submit(() -> {
                    assertTrue(start.await(5, TimeUnit.SECONDS));
                    return captured.work().materialize();
                }));
            }
            start.countDown();
            WorkView shared = readers.getFirst().get(5, TimeUnit.SECONDS);
            for (var reader : readers) {
                assertSame(shared, reader.get(5, TimeUnit.SECONDS));
            }
            assertEquals(List.of("1", "2"), shared.requests().stream()
                    .map(WorkView.RequestWork::requestId).toList());
            assertTrue(state.removeCommittedRequest(second));
            assertTrue(state.removeCommittedRequest(first));
            assertEquals(2, shared.requests().size());
            assertTrue(capture().work().materialize().requests().isEmpty());
        }
    }

    @Test
    void committedWorkChangesReuseActiveMembershipButCaptureBothAtOnePoint() {
        ScheduledRequest committed = item(2);
        commitRoute(committed, 20);
        var queued = item(1);
        enqueue(queued);
        var before = capture();
        assertTrue(state.removeCommittedRequest(committed));
        var after = capture();
        assertSame(before.active(), after.active());
        assertEquals(1, before.work().materialize().requests().size());
        assertTrue(after.work().materialize().requests().isEmpty());
        enqueue(item(3));
        var added = capture();
        assertNotSame(after.active(), added.active());
        assertSame(after.work(), added.work());
        assertEquals(List.of(queued), before.activeItems());
    }

    @Test
    void clockRollbackRecapturesWorkWithoutMutatingEarlierSnapshots() {
        commitRoute(item(1), 10);
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
    void queuedRoutesUseOneCommitAndTerminalOwnership() {
        ScheduledRequest first = item(1), second = item(2);
        try (var firstReservation = reserveQueued(first, 30L); var secondReservation = reserveQueued(second, 40L)) {
            assertEquals(List.of(first, second), waitingItems());
            assertTrue(state.committedSnapshot().requests().isEmpty());
            try (var handoff = state.commitSingleGroup(List.of(first, second),
                    List.of(firstReservation, secondReservation), generation.tryAcquireHandoff())) {
                assertTrue(waitingItems().isEmpty());
                assertEquals(0L, handoff.precedingWork().totalRemainingWorkMs().orElseThrow(),
                        "the selected group's own provisional work is excluded");
                assertEquals(List.of("1", "2"), state.committedSnapshot().requests().stream()
                        .map(work -> work.requestId()).toList());
                assertEquals(70L, state.committedSnapshot().knownRemainingWorkMsAt(System.currentTimeMillis()));
            }
        }
        assertEquals(2, state.committedSnapshot().requests().size(), "closing committed capabilities cannot release Engine ownership");
        assertTrue(state.removeCommittedRequest(first));
        assertTrue(state.removeCommittedRequest(second));
        assertTrue(state.committedSnapshot().requests().isEmpty());
    }

    @Test
    void routeCommitUsesTheUpdatedPrediction() {
        ScheduledRequest queued = item(1);
        try (var reservation = reserveQueued(queued, 40L)) {
            assertTrue(capture().work().materialize().requests().isEmpty(), "queued work is not committed work");
            reservation.updatePrediction(queued, 80L);
            try (var handoff =
                    state.commitSingleGroup(List.of(queued), List.of(reservation), generation.tryAcquireHandoff())) {
                assertEquals(80L, capture().work().materialize().knownRemainingWorkMsAt(System.currentTimeMillis()));
            }
        }
    }

    @Test
    void leaseRollbackKeepsTheRequestQueued() {
        ScheduledRequest queued = item(1);
        enqueue(queued);
        state.reserveSingle(queued, 30L).reservation().close();
        assertEquals(List.of(queued), waitingItems());
        assertTrue(state.committedSnapshot().requests().isEmpty());
        assertTrue(state.reserveSingle(queued, 35L).reservation() != null);
    }

    @Test
    void routeCommitRejectsMissingQueueIndexBeforeCommittingAnyMember() {
        ScheduledRequest removed = item(1), kept = item(2);
        try (var removedReservation = reserveQueued(removed, 30L); var keptReservation = reserveQueued(kept, 40L)) {
            assertTrue(waiting.remove(removed));
            try (var handoff = generation.tryAcquireHandoff()) {
                assertThrows(IllegalStateException.class, () -> state.commitSingleGroup(List.of(kept, removed),
                        List.of(keptReservation, removedReservation), handoff));
            }
            assertFalse(state.removeCommittedRequest(kept));
            assertTrue(waiting.add(removed));
            try (var handoff = state.commitSingleGroup(List.of(kept, removed),
                    List.of(keptReservation, removedReservation), generation.tryAcquireHandoff())) {
                assertTrue(waiting.isEmpty());
            }
        }
    }

    @Test
    void retirementHandlesQueuedPreparedAndCommittedItemsWithoutModeExceptions() {
        ScheduledRequest queued = item(1), prepared = item(2), committed = item(3);
        enqueue(queued);
        var preparedReservation = reserveQueued(prepared, 20L);
        var committedReservation = reserveQueued(committed, 30L);
        try (var handoff = state.commitSingleGroup(List.of(committed), List.of(committedReservation),
                generation.tryAcquireHandoff())) {
            assertEquals(1, state.committedSnapshot().requests().size());
        }
        var retired = state.shutdown();
        assertNull(retired.checkFailure());
        assertEquals(3, retired.removedItems().size());
        assertTrue(retired.removedItems().containsAll(List.of(queued, prepared, committed)));
        preparedReservation.close();
        committedReservation.close();
        assertTrue(state.committedSnapshot().requests().isEmpty());
        assertTrue(waiting.isEmpty());
    }

    @Test
    void cancellationBeforeCommitReleasesAQueuedRouteExactlyOnce() {
        ScheduledRequest queued = item(1);
        var reservation = reserveQueued(queued, 40L);
        lock.lock();
        try {
            assertTrue(state.removeQueuedSingleRequest(queued, reservation));
            assertFalse(state.removeQueuedSingleRequest(queued, reservation));
        } finally {
            lock.unlock();
        }
        reservation.close();
        assertTrue(waitingItems().isEmpty());
        assertTrue(state.committedSnapshot().requests().isEmpty());
        assertTrue(capture().work().materialize().requests().isEmpty());
    }

    @Test
    void interleavedRouteCommitsSeeOnlyWorkCommittedBeforeThem() {
        ScheduledRequest first = item(1), second = item(2);
        try (var firstReservation = reserveQueued(first, 30L); var secondReservation = reserveQueued(second, 40L)) {
            try (var secondHandoff = state.commitSingleGroup(List.of(second), List.of(secondReservation),
                    generation.tryAcquireHandoff())) {
                assertEquals(0L, secondHandoff.precedingWork().totalRemainingWorkMs().orElseThrow(),
                        "a request still waiting in the queue is not preceding work");
            }
            try (var firstHandoff = state.commitSingleGroup(List.of(first), List.of(firstReservation),
                    generation.tryAcquireHandoff())) {
                assertEquals(40L, firstHandoff.precedingWork().totalRemainingWorkMs().orElseThrow(),
                        "the earlier reservation must see work committed before its own commit");
            }
        }
    }

    @Test
    void batchAndIndividualCommitsCaptureTheSamePrecedingTimeline() {
        ScheduledRequest individual = item(1), batchMember = item(2), later = item(3);
        try (var individualReservation = reserveQueued(individual, 30L);
                var firstHandoff = state.commitSingleGroup(List.of(individual), List.of(individualReservation),
                        generation.tryAcquireHandoff())) {
            assertEquals(0L, firstHandoff.precedingWork().totalRemainingWorkMs().orElseThrow());
            enqueue(batchMember);
            try (var batchReservation = state.reserveBatch(batchMember, 10L, 2, generation.tryAcquireHandoff()).reservation();
                 var batchHandoff = batchReservation.commit(List.of(batchMember), 40L)) {
                assertEquals(30L, batchHandoff.precedingWork().totalRemainingWorkMs().orElseThrow());
            }
            try (var laterReservation = reserveQueued(later, 50L);
                    var laterHandoff = state.commitSingleGroup(List.of(later), List.of(laterReservation),
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
        assertTrue(state.removeCommittedRequest(a));
        assertEquals(List.of("2", "3"), state.committedSnapshot().batches().getFirst().requestIds());
        assertEquals(250L, remainingWork());
        clock.set(190);
        assertTrue(state.removeCommittedRequest(b));
        assertFalse(state.removeCommittedRequest(a));
        assertFalse(state.removeCommittedRequest(item(3)), "cleanup must match the exact item");
        assertEquals(210L, remainingWork());

        // Returning to QUEUED pauses elapsed-time accounting, but does not undo work already started.
        clock.set(230);
        reconcile(Map.of("1", task(1, null, 0, 100)),
                Map.of("3", task(3, TaskPhase.RECEIVED, 0, 0)), unused -> {
                    throw new AssertionError("local cancellation cannot shrink running work");
                });
        clock.set(250);
        assertEquals(170L, remainingWork());
        assertEquals(WorkView.Phase.ENGINE_QUEUED,
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

    private SingleReservation reserveQueued(ScheduledRequest item, long predictedMs) {
        enqueue(item);
        return state.reserveSingle(item, predictedMs).reservation();
    }

    private void commitRoute(ScheduledRequest item, long predictedMs) {
        try (var reservation = reserveQueued(item, predictedMs);
                var handoff =
                        state.commitSingleGroup(List.of(item), List.of(reservation), generation.tryAcquireHandoff())) {
            assertTrue(waitingItems().isEmpty());
        }
    }

    private void reconcile(Map<String, TaskInfo> finished,
                           Map<String, TaskInfo> active,
                           ToLongFunction<List<ScheduledRequest>> predictBatchTime) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(active);
        var observation = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090)
                .freezeStatusResponse(response);
        var outcome = state.updateFromWorkerStatus(observation, predictBatchTime, () -> {}, () -> {});
        assertNull(outcome.publicationFailure());
    }

    private long remainingWork() {
        return state.committedSnapshot().totalRemainingWorkMsAt(clock.get()).orElseThrow();
    }

    private static TaskInfo task(long requestId, TaskPhase phase, long errorCode, long executionMs) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(Long.toString(requestId));
        task.setBatchId(10L);
        task.setPhase(phase);
        task.setErrorCode(errorCode);
        task.setExecutionTimeMs(executionMs);
        return task;
    }

    private static ScheduledRequest item(long id) {
        var item = mock(ScheduledRequest.class);
        when(item.requestId()).thenReturn(Long.toString(id));
        when(item.seqLen()).thenReturn(100L);
        return item;
    }

    private Snapshot capture() {
        lock.lock();
        try {
            return state.snapshot();
        } finally {
            lock.unlock();
        }
    }

    private void enqueue(ScheduledRequest item) {
        lock.lock();
        try {
            assertTrue(state.enqueueActive(item, 0L));
        } finally {
            lock.unlock();
        }
    }

    private List<ScheduledRequest> waitingItems() {
        return capture().activeItems();
    }
}
