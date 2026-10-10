package org.flexlb.balance.endpoint;

import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.ToLongFunction;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
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
            Comparator.comparingLong(RequestRoute::requestId));
    private final PrefillState state = new PrefillState(lock, waiting, clock::get);
    private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void selectionCommitIncludesFailedBoundaryAndQueueDepth(boolean batch, boolean expiredBoundary) {
        RequestRoute selected = item(1), failed = item(2), waiting = item(3);
        enqueue(selected);
        enqueue(failed);
        enqueue(waiting);
        when(failed.requestExpired(clock.get())).thenReturn(expiredBoundary);
        var pin = generation.tryAcquireHandoff();
        var reservation = batch ? state.reserveBatch(selected, 9L, 1, pin).reservation() : null;
        lock.lock();
        try (var committed = batch
                ? reservation.commitLocked(List.of(selected), 10L, failed, clock.get())
                : state.commitQueuedRoutesLocked(List.of(selected), new long[]{10L}, pin, failed, clock.get())) {
            assertEquals(!expiredBoundary, committed.removedFailedMember());
            assertEquals(expiredBoundary ? 2 : 1, committed.remainingQueueDepth());
            assertEquals(expiredBoundary ? List.of(failed, waiting) : List.of(waiting), waitingItems());
            assertEquals(expiredBoundary ? 3 : 2, state.admissionSummary(0, 0L).occupiedRequests());
        } finally { lock.unlock(); }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedBoundaryValidationCannotPartiallyCommitSelection(boolean batch) {
        RequestRoute selected = item(1), failed = item(2);
        enqueue(selected);
        enqueue(failed);
        var failure = new IllegalStateException("boundary validation failed");
        when(failed.requestExpired(clock.get())).thenThrow(failure);
        var pin = generation.tryAcquireHandoff();
        var reservation = batch ? state.reserveBatch(selected, 9L, 1, pin).reservation() : null;
        lock.lock();
        try {
            assertSame(failure, assertThrows(IllegalStateException.class, () -> {
                if (batch) { reservation.commitLocked(List.of(selected), 10L, failed, clock.get()); }
                else { state.commitQueuedRoutesLocked(List.of(selected), new long[]{10L}, pin, failed, clock.get()); }
            }));
            assertEquals(List.of(selected, failed), waitingItems());
            assertEquals(2, state.admissionSummary(0, 0L).occupiedRequests());
            assertNoWork(state.committedSnapshot(), 1L, 2L);
        } finally {
            lock.unlock();
            if (reservation != null) { EndpointTestSupport.preparation(reservation).close(); }
            else { pin.close(); }
        }
        assertEquals(0, state.captureQueueCounters().batchSlots());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void unlockedCommitPreservesOwnershipAndCanBeRetriedWithTheLock(boolean batch) {
        RequestRoute request = item(1);
        if (batch) {
            enqueue(request);
            {
                var reservation = state.reserveBatch(request, 9L, 1,
                    generation.tryAcquireHandoff()).reservation();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertThrows(IllegalStateException.class,
                            () -> reservation.commitLocked(List.of(request), 10L, null, System.currentTimeMillis()));
                    assertEquals(List.of(request), state.captureQueue(1).items());
                    assertEquals(1, state.captureQueueCounters().batchSlots());
                    try (var handoff = EndpointTestSupport.commitBatch(state, reservation, List.of(request), 10L)) {
                        assertTrue(state.captureQueue(1).items().isEmpty());
                    }
                }
            }
        } else {
            {
                var reservation = state.reserveUnqueuedRoute(request, 10L, 1L).reservation();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                     var generationHandoff = generation.tryAcquireHandoff()) {
                    assertThrows(IllegalStateException.class,
                            () -> state.commitRouteGroupLocked(List.of(request), List.of(reservation), generationHandoff));
                    assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                    try (var handoff = EndpointTestSupport.commitRoutes(state, List.of(request),
                            List.of(reservation), generationHandoff)) {
                        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                    }
                }
            }
        }
        assertTrue(EndpointTestSupport.releaseRequest(state, request));
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(0, state.captureQueueCounters().batchSlots());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void concurrentCommittedCloseReleasesOnlyItsHandoffAndKeepsRequestOwnership(boolean batch) throws Exception {
        AtomicInteger drained = new AtomicInteger();
        EndpointGenerationLifecycle lifecycle = new EndpointGenerationLifecycle(drained::incrementAndGet);
        RequestRoute request = item(1);
        PrefillState.CommittedHandoff handoff;
        if (batch) {
            enqueue(request);
            {
                var lease = state.reserveBatch(request, 9L, 1, lifecycle.tryAcquireHandoff()).reservation();
                try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                    handoff = EndpointTestSupport.commitBatch(state, lease, List.of(request), 10L);
                }
            }
        } else {
            {
                var lease = state.reserveUnqueuedRoute(request, 10L, Long.MAX_VALUE).reservation();
                try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                    handoff = EndpointTestSupport.commitRoutes(state, List.of(request), List.of(lease), lifecycle.tryAcquireHandoff());
                }
            }
        }
        try (handoff; var otherHandoff = lifecycle.tryAcquireHandoff();
             var closers = Executors.newFixedThreadPool(2)) {
            lifecycle.beginRetirement();
            assertFalse(lifecycle.tryStartCleanup());
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> closes = new ArrayList<>();
            for (int i = 0; i < 2; i++) {
                closes.add(closers.submit(() -> {
                    assertTrue(start.await(5, TimeUnit.SECONDS));
                    handoff.close();
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> close : closes) { close.get(5, TimeUnit.SECONDS); }
            assertEquals(0, drained.get(), "the other handoff still prevents retirement");
            otherHandoff.close();
            handoff.close();
            assertEquals(1, drained.get());
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests(), "closing handoff cannot release committed capacity");
            assertTrue(EndpointTestSupport.releaseRequest(state, request));
            assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        }
    }

    @Test
    void duplicateBatchMembersFailBeforeQueueMutationAndKeepTheLeaseRetryable() {
        RequestRoute first = item(1), second = item(2);
        enqueue(first);
        enqueue(second);
        {
            var lease = state.reserveBatch(first, 9L, 1, generation.tryAcquireHandoff()).reservation();
            try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                var before = state.captureQueue(4);
                assertThrows(IllegalStateException.class,
                        () -> EndpointTestSupport.commitBatch(state, lease, List.of(first, first), 10L));
                assertEquals(before, state.captureQueue(4), "invalid input must not detach any ACTIVE member");
                assertEquals(1, state.captureQueueCounters().batchSlots());
                try (var committed = EndpointTestSupport.commitBatch(state, lease, List.of(first, second), 10L)) {
                    assertTrue(state.captureQueue(4).items().isEmpty());
                    assertWork(state.committedSnapshot(), 10L, 1L, 2L);
                    assertEquals(2, state.stats().locallyOwnedRequests());
                }
                assertTrue(EndpointTestSupport.releaseRequest(state, first));
                assertEquals(1, state.captureQueueCounters().batchSlots());
                assertTrue(EndpointTestSupport.releaseRequest(state, second));
                assertEquals(0, state.captureQueueCounters().batchSlots());
            }
        }
    }

    @Test
    void queueCaptureKeepsBoundedOrderAndVersionsAcrossChanges() {
        RequestRoute first = item(1), second = item(2), third = item(3), tail = item(4);
        enqueue(first);
        enqueue(second);
        enqueue(third);

        PrefillState.QueueSnapshot bounded = state.captureQueue(2);
        assertEquals(List.of(first, second), bounded.items());
        assertSame(first, bounded.head());
        assertEquals(List.of(first, second, third), state.captureQueue(4).items());
        assertEquals(waiting.version(), bounded.queueVersion());
        assertEquals(state.schedulingInputVersion(), bounded.schedulingInputVersion());

        lock.lock();
        try {
            state.schedulingInputsChangedLocked();
        } finally {
            lock.unlock();
        }
        PrefillState.QueueSnapshot changedInputs = state.captureQueue(2);
        assertEquals(bounded.queueVersion(), changedInputs.queueVersion());
        assertTrue(changedInputs.schedulingInputVersion() > bounded.schedulingInputVersion());
        enqueue(tail);
        PrefillState.QueueSnapshot changedQueue = state.captureQueue(2);
        assertEquals(List.of(first, second), changedQueue.items());
        assertTrue(changedQueue.queueVersion() > bounded.queueVersion());
        assertEquals(List.of(first, second), bounded.items(), "an earlier capture keeps its identities");
        assertThrows(UnsupportedOperationException.class, () -> bounded.items().clear());
    }

    @Test
    void batchCapacityRollbackRestoresAvailability() {
        PrefillState ledger = new PrefillState(lock, waiting, clock::get);
        var availability = (java.util.function.BooleanSupplier) () -> ledger.batchCapacityAvailable(1);
        var request = item(99L);
        lock.lock();
        try {
            assertTrue(ledger.enqueueActiveLocked(request, 0L));
        } finally {
            lock.unlock();
        }
        assertTrue(availability.getAsBoolean());
        {
            var lease = ledger.reserveBatch(request, 9L, 1, generation.tryAcquireHandoff()).reservation();
            try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                org.junit.jupiter.api.Assertions.assertNotNull(lease);
                assertFalse(availability.getAsBoolean());
            }
        }
        assertTrue(availability.getAsBoolean());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void openLeaseRollbackRestoresFactsEvenWhenHandoffFails(boolean batch) {
        PrefillState ledger = new PrefillState(lock, waiting, clock::get);
        RuntimeException failure = new IllegalStateException("generation drain failure");
        EndpointGenerationLifecycle retiring = new EndpointGenerationLifecycle(() -> { throw failure; });
        RequestRoute request = item(91L);
        if (batch) {
            lock.lock();
            try { assertTrue(ledger.enqueueActiveLocked(request, 10L)); }
            finally { lock.unlock(); }
        }
        long queuedVersion = waiting.version();
        PrefillState.Reservation lease = batch
                ? ledger.reserveBatch(request, 9L, 1, retiring.tryAcquireHandoff()).reservation()
                : ledger.reserveUnqueuedRoute(request, 10L, 10L).reservation();
        if (batch) {
            retiring.beginRetirement();
            assertFalse(retiring.tryStartCleanup());
            assertSame(failure, assertThrows(IllegalStateException.class, () -> EndpointTestSupport.rollback(lease)));
        } else {
            EndpointTestSupport.rollback(lease);
        }
        EndpointTestSupport.rollback(lease);
        lock.lock();
        try {
            assertEquals(0, ledger.captureQueueCounters().batchSlots());
            assertEquals(batch, waiting.contains(request), "batch rollback retains queue ownership; DIRECT rollback releases its seat");
            assertEquals(queuedVersion, waiting.version(), "lease rollback does not change queue membership");
        } finally {
            lock.unlock();
        }
    }

    @Test
    void batchIdRemainsReservedUntilItsLastMemberSettles() {
        RequestRoute first = item(1), sibling = item(2), next = item(3);
        enqueue(first);
        enqueue(sibling);
        enqueue(next);
        {
            var lease = state.reserveBatch(first, 10, 4, generation.tryAcquireHandoff()).reservation();
            try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                try (var probe = generation.tryAcquireHandoff()) {
                    assertEquals(PrefillState.CapacityStatus.BATCH_ID_ALREADY_RESERVED,
                            state.reserveBatch(next, 10, 4, probe).status());
                }
                try (var handoff = EndpointTestSupport.commitBatch(state, lease, List.of(first, sibling), 30)) {
                    assertEquals(1, state.stats().batchCount());
                }
                assertTrue(EndpointTestSupport.releaseRequest(state, first));
                try (var probe = generation.tryAcquireHandoff()) {
                    assertEquals(PrefillState.CapacityStatus.BATCH_ID_ALREADY_RESERVED,
                            state.reserveBatch(next, 10, 4, probe).status());
                }
                assertTrue(EndpointTestSupport.releaseRequest(state, sibling));
                {
                    var replacement = state.reserveBatch(next, 10, 4,
                        generation.tryAcquireHandoff()).reservation();
                    try (var preparationReplacement = EndpointTestSupport.preparation(replacement)) {
                        try (var handoff = EndpointTestSupport.commitBatch(state, replacement, List.of(next), 20)) {
                            assertEquals(List.of(10L), batchIds());
                            assertWork(state.committedSnapshot(), 20L, 3L);
                            assertEquals(1, state.stats().batchCount());
                        }
                        assertTrue(EndpointTestSupport.releaseRequest(state, next));
                    }
                }
            }
        }
        assertEquals(0, state.stats().locallyOwnedRequests());
        assertEquals(0, state.stats().batchCount());
    }

    @Test
    void orphanSweepSharesTtlAndKeepsEveryMemberOfARetainedBatch() {
        RequestRoute first = item(1), sibling = item(2), individual = item(3), queued = item(4);
        enqueue(first);
        enqueue(sibling);
        enqueue(queued);
        long beforeCommitVersion = waiting.version();
        {
            var batch = state.reserveBatch(first, 1, 4, generation.tryAcquireHandoff()).reservation();
            try (var preparationBatch = EndpointTestSupport.preparation(batch);
                 var handoff = EndpointTestSupport.commitBatch(state, batch, List.of(first, sibling), 30)) {
                assertEquals(2, state.stats().locallyOwnedRequests());
                assertTrue(waiting.version() > beforeCommitVersion, "batch commit invalidates the queue revision");
                assertEquals(List.of(queued), waitingItems());
            }
        }
        {
            var reservation = state.reserveUnqueuedRoute(individual, 20, Long.MAX_VALUE).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                 var handoff = EndpointTestSupport.commitRoutes(state, List.of(individual), List.of(reservation),
                     generation.tryAcquireHandoff())) {
                assertEquals(3, state.stats().locallyOwnedRequests());
            }
        }
        clock.set(105);
        RequestRoute fresh = item(5);
        {
            var reservation = state.reserveUnqueuedRoute(fresh, 20, Long.MAX_VALUE).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                 var handoff = EndpointTestSupport.commitRoutes(state, List.of(fresh), List.of(reservation),
                     generation.tryAcquireHandoff())) {
                assertEquals(4, state.stats().locallyOwnedRequests());
            }
        }
        clock.set(110);
        assertEquals(1, EndpointTestSupport.evictPrefill(state, 10, id -> id == first.requestId()));
        assertEquals(3, state.stats().locallyOwnedRequests(), "one retained member protects both batch members");
        assertEquals(1, state.stats().batchCount());
        assertEquals(1, state.stats().individuallyOwnedRequests(), "fresh individual survives");
        assertEquals(List.of(queued), waitingItems());
        assertEquals(1, EndpointTestSupport.evictPrefill(state, 10, ignored -> false), "a batch counts once");
        assertEquals(1, state.stats().locallyOwnedRequests());
        clock.set(115);
        assertEquals(1, EndpointTestSupport.evictPrefill(state, 10, ignored -> false));
        assertEquals(List.of(queued), waitingItems(), "orphan sweeps never remove queued ownership");
        assertNoWork(state.committedSnapshot(), 1L, 2L, 3L, 5L);
        assertEquals(0, state.stats().batchCount());
        assertEquals(0, state.stats().individuallyOwnedRequests());
    }

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
            assertWork(shared, 30L, 1L, 2L);
            EndpointTestSupport.rollback(second);
            EndpointTestSupport.rollback(first);
            assertWork(shared, 30L, 1L, 2L);
            assertNoWork(capture().work().materialize(), 1L, 2L);
        }
    }

    @Test
    void committedWorkChangesReuseActiveMembershipButCaptureBothAtOnePoint() {
        var queued = item(1);
        enqueue(queued);
        var before = capture();
        var reservation = state.reserveUnqueuedRoute(item(2), 20, Long.MAX_VALUE).reservation();
        var after = capture();
        assertSame(before.active(), after.active());
        assertNoWork(before.work().materialize(), 1L, 2L);
        assertWork(after.work().materialize(), 20L, 2L);
        assertFalse(after.work().materialize().containsRequest(1L));
        enqueue(item(3));
        var added = capture();
        assertNotSame(after.active(), added.active());
        assertSame(after.work(), added.work());
        assertEquals(List.of(queued.requestId()), before.active().projectedItems().stream()
                .map(org.flexlb.balance.planner.GroupPlanner.Item::requestId).toList());
        EndpointTestSupport.rollback(reservation);
        assertSame(added.active(), capture().active());
        assertNoWork(capture().work().materialize(), 2L);
        assertWork(after.work().materialize(), 20L, 2L);
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
        assertWork(original.work().materialize(), 10L, 1L);
        assertWork(rebased.work().materialize(), 10L, 1L);
        assertEquals(10L, original.work().materialize().knownRemainingWorkMsAt(110L));
        assertEquals(10L, rebased.work().materialize().knownRemainingWorkMsAt(110L));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queuedRouteCommitValidatesEveryCanonicalIdentityBeforeTransferringAny(boolean replaced) {
        RequestRoute first = routeItem(101), second = routeItem(102), replacement = routeItem(102);
        enqueue(first);
        enqueue(second);
        assertTrue(state.releaseRequest(second) != PrefillState.RequestRelease.NONE);
        if (replaced) { enqueue(replacement); }
        try (var permit = generation.tryAcquireHandoff()) {
            assertThrows(IllegalStateException.class, () -> EndpointTestSupport.commitQueuedRoutes(state,
                    List.of(first, second), new long[]{30L, 40L}, permit));
        }
        assertEquals(replaced ? List.of(first, replacement) : List.of(first), waitingItems());
        assertEquals(replaced ? 2L : 1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertNoWork(state.committedSnapshot(), 101L, 102L);
        RequestRoute validSecond = replaced ? replacement : second;
        if (!replaced) { enqueue(second); }
        try (var handoff = EndpointTestSupport.commitQueuedRoutes(state, List.of(first, validSecond),
                new long[]{30L, 40L}, generation.tryAcquireHandoff())) {
            assertTrue(waitingItems().isEmpty());
            assertEquals(0L, handoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow());
        }
        assertWork(state.committedSnapshot(), 70L, 101L, 102L);
        assertTrue(EndpointTestSupport.releaseRequest(state, first));
        assertTrue(EndpointTestSupport.releaseRequest(state, validSecond));
        assertNoWork(state.committedSnapshot(), 101L, 102L);
    }

    private static RequestRoute routeItem(long id) {
        var config = new FlexlbConfig();
        config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
        var context = new RequestContext(config);
        var request = new Request();
        request.setRequestId(id);
        request.setSeqLen(100L);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), null, null, null, null, null, null, 100L);
    }

    @Test
    void queuedRouteCommitCannotConsumeABatchPreparationOrPartiallyCommitThePrefix() {
        RequestRoute first = item(1), second = item(2);
        enqueue(first);
        enqueue(second);
        var batch = state.reserveBatch(second, 11L, 1, generation.tryAcquireHandoff()).reservation();
        try (var preparation = EndpointTestSupport.preparation(batch);
             var permit = generation.tryAcquireHandoff()) {
            assertThrows(IllegalStateException.class, () -> EndpointTestSupport.commitQueuedRoutes(state,
                    List.of(first, second), new long[]{30L, 40L}, permit));
            assertEquals(List.of(first, second), waitingItems());
            assertNoWork(state.committedSnapshot(), 1L, 2L);
            assertEquals(1, state.captureQueueCounters().batchSlots());
        }
        assertEquals(0, state.captureQueueCounters().batchSlots());
        try (var handoff = EndpointTestSupport.commitQueuedRoutes(state, List.of(first, second),
                new long[]{30L, 40L}, generation.tryAcquireHandoff())) {
            assertTrue(waitingItems().isEmpty());
        }
        assertEquals(70L, state.committedSnapshot().totalRemainingWorkMs().orElseThrow());
    }

    @Test
    void releasingQueuedBatchHeadLeavesPreparationToItsTransaction() {
        RequestRoute head = item(1);
        enqueue(head);
        var batch = state.reserveBatch(head, 11L, 1, generation.tryAcquireHandoff()).reservation();
        try (var preparation = EndpointTestSupport.preparation(batch)) {
            assertEquals(PrefillState.RequestRelease.QUEUED, state.releaseRequest(head));
            assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(1, state.captureQueueCounters().batchSlots());
            assertEquals(PrefillState.RequestRelease.NONE, state.releaseRequest(head));
        }
        assertEquals(0, state.captureQueueCounters().batchSlots());
    }

    @Test
    void releasingCommittedBatchMemberRetainsItsSiblingsAndSlot() {
        RequestRoute first = item(1), second = item(2);
        commitBatch(List.of(first, second), 100L);
        assertEquals(PrefillState.RequestRelease.COMMITTED, state.releaseRequest(first));
        assertEquals(PrefillState.RequestRelease.NONE, state.releaseRequest(first));
        assertWork(state.committedSnapshot(), 100L, 2L);
        assertFalse(state.committedSnapshot().containsRequest(1L));
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(1, state.captureQueueCounters().batchSlots());
        assertEquals(PrefillState.RequestRelease.COMMITTED, state.releaseRequest(second));
        assertEquals(0, state.captureQueueCounters().batchSlots());
    }

    @Test
    void queuedAndImmediateRoutesUseOneCommitAndTerminalOwnership() {
        RequestRoute queued = item(1), immediate = item(2);
        enqueue(queued);
        var immediateReservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation();
        try (var preparation = EndpointTestSupport.preparation(immediateReservation)) {
            assertEquals(List.of(queued), waitingItems());
            assertWork(state.committedSnapshot(), 40L, 2L);
            assertFalse(state.committedSnapshot().containsRequest(1L));
            try (var handoff = EndpointTestSupport.commitQueuedRoutes(state, List.of(queued),
                    new long[]{30L}, generation.tryAcquireHandoff())) {
                assertTrue(waitingItems().isEmpty());
                assertEquals(40L, handoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow(),
                        "uncommitted DIRECT work remains visible to the queued route");
            }
            try (var handoff = EndpointTestSupport.commitRoutes(state, List.of(immediate),
                    List.of(immediateReservation), generation.tryAcquireHandoff())) {
                assertEquals(30L, handoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow());
            }
        }
        assertEquals(70L, state.committedSnapshot().knownRemainingWorkMsAt(clock.get()));
        assertTrue(EndpointTestSupport.releaseRequest(state, queued));
        assertTrue(EndpointTestSupport.releaseRequest(state, immediate));
        assertNoWork(state.committedSnapshot(), 1L, 2L);
    }

    @Test
    void immediatePreparationIsVisibleToLaterPredictionsAndRollbackRemovesIt() {
        RequestRoute immediate = item(1);
        {
            var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                assertTrue(waitingItems().isEmpty());
                assertEquals(40L, capture().work().materialize().knownRemainingWorkMsAt(System.currentTimeMillis()));
            }
        }
        assertNoWork(capture().work().materialize(), 1L);
        assertNoWork(state.committedSnapshot(), 1L);
    }

    @Test
    void leaseRollbackRetainsQueuedWorkButReleasesUnqueuedAdmission() {
        RequestRoute queued = item(1), immediate = item(2);
        enqueue(queued);
        var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation();
        EndpointTestSupport.rollback(reservation);
        assertEquals(List.of(queued), waitingItems());
        assertNoWork(state.committedSnapshot(), 1L, 2L);
    }

    @Test
    void routeCommitRejectsMissingQueueIndexBeforeCommittingAnyMember() {
        RequestRoute first = item(1), second = item(2);
        enqueue(first);
        enqueue(second);
        assertTrue(waiting.remove(second));
        try (var permit = generation.tryAcquireHandoff()) {
            assertThrows(IllegalStateException.class, () -> EndpointTestSupport.commitQueuedRoutes(state,
                    List.of(first, second), new long[]{30L, 40L}, permit));
        }
        assertEquals(List.of(first), waitingItems());
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests(),
                "failed validation preserves both canonical request owners");
        assertNoWork(state.committedSnapshot(), 1L, 2L);
        assertTrue(waiting.add(second));
        try (var handoff = EndpointTestSupport.commitQueuedRoutes(state, List.of(first, second),
                new long[]{30L, 40L}, generation.tryAcquireHandoff())) {
            assertTrue(waiting.isEmpty());
        }
    }

    @Test
    void retirementProjectsQueuedPreparedAndCommittedItemsWithoutModeExceptions() {
        RequestRoute queued = item(1), prepared = item(2), committed = item(3);
        enqueue(queued);
        var preparedReservation = state.reserveUnqueuedRoute(prepared, 20L, Long.MAX_VALUE).reservation();
        var committedReservation = state.reserveUnqueuedRoute(committed, 30L, Long.MAX_VALUE).reservation();
        try (var handoff = EndpointTestSupport.commitRoutes(state, List.of(committed), List.of(committedReservation),
                generation.tryAcquireHandoff())) {
            assertWork(state.committedSnapshot(), 50L, 2L, 3L);
            assertFalse(state.committedSnapshot().containsRequest(1L));
        }
        var retired = state.retireGenerationOwnership();
        assertNull(retired.invariantFailure());
        assertEquals(3, retired.ownedItems().size());
        assertTrue(retired.ownedItems().containsAll(List.of(queued, prepared, committed)));
        EndpointTestSupport.rollback(preparedReservation);
        EndpointTestSupport.rollback(committedReservation);
        assertNoWork(state.committedSnapshot(), 1L, 2L, 3L);
        assertTrue(waiting.isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void retirementReleasesOpenBatchHandoffEvenWhenDrainCallbackFails(boolean callbackFails) {
        AtomicInteger drained = new AtomicInteger();
        var retiring = new EndpointGenerationLifecycle(() -> {
            assertFalse(lock.isHeldByCurrentThread());
            drained.incrementAndGet();
            if (callbackFails) { throw new IllegalStateException("drain callback failed"); }
        });
        RequestRoute request = item(1);
        enqueue(request);
        var lease = state.reserveBatch(request, 9L, 1, retiring.tryAcquireHandoff()).reservation();
        retiring.beginRetirement();
        assertFalse(retiring.tryStartCleanup());

        var retired = state.retireGenerationOwnership();
        assertEquals(0, drained.get(), "State returns cleanup capabilities without executing them");
        for (var handoff : retired.orphanedHandoffs()) {
            if (callbackFails) { assertThrows(IllegalStateException.class, handoff::close); } else { handoff.close(); }
        }
        assertEquals("retirement reached an OPEN Prefill batch lease", retired.invariantFailure().getMessage());
        assertEquals(List.of(request), retired.ownedItems());
        assertTrue(retired.batchCompletions().isEmpty());
        assertEquals(1, drained.get());
        assertEquals(0, state.captureQueueCounters().batchSlots());
        assertEquals(0, state.admissionSummary(0, 0L).occupiedRequests());
        assertTrue(waiting.isEmpty());
        EndpointTestSupport.rollback(lease);
        assertEquals(1, drained.get(), "late lease cleanup cannot release the generation twice");
        var repeated = state.retireGenerationOwnership();
        assertTrue(repeated.ownedItems().isEmpty());
        assertNull(repeated.invariantFailure());
    }

    @Test
    void retirementEmitsOneCompletionForSharedBatchAndRetainsDetachedStopOwner() {
        RequestRoute first = item(1), second = item(2), detached = item(3);
        commitBatch(List.of(first, second), 100L);
        enqueue(detached);
        assertSame(detached, state.detachNextActiveForStop());

        var retired = state.retireGenerationOwnership();
        assertNull(retired.invariantFailure());
        assertEquals(List.of(first, second, detached), retired.ownedItems());
        assertEquals(1, retired.batchCompletions().size());
        assertEquals(10L, retired.batchCompletions().getFirst().batchId());
        assertFalse(retired.batchCompletions().getFirst().learningEligible());
        assertEquals(0, state.captureQueueCounters().batchSlots());
        assertEquals(0, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(EndpointTestSupport.releaseRequest(state, first));
        assertFalse(EndpointTestSupport.releaseRequest(state, second));
        assertTrue(state.retireGenerationOwnership().batchCompletions().isEmpty());
    }

    @Test
    void cancellationBeforeCommitReleasesAnUnqueuedAdmissionExactlyOnce() {
        RequestRoute immediate = item(1);
        var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).reservation();
        assertEquals(PrefillState.RequestRelease.RESERVED, state.releaseRequest(immediate));
        assertEquals(PrefillState.RequestRelease.NONE, state.releaseRequest(immediate));
        EndpointTestSupport.rollback(reservation);
        assertNoWork(state.committedSnapshot(), 1L);
        assertNoWork(capture().work().materialize(), 1L);
    }

    @Test
    void foreignPreparationCannotCommitOrRollbackTheSameRequestId() {
        PrefillState other = new PrefillState(new ReentrantLock(), PrefillActiveIndex.disabled(), clock::get);
        RequestRoute localRequest = item(1), otherRequest = item(1);
        var localReservation = state.reserveUnqueuedRoute(localRequest, 10L, 1L).reservation();
        var otherReservation = other.reserveUnqueuedRoute(otherRequest, 20L, 1L).reservation();
        try (var localPreparation = EndpointTestSupport.preparation(localReservation);
             var otherPreparation = EndpointTestSupport.preparation(otherReservation);
             var permit = generation.tryAcquireHandoff()) {
            assertThrows(IllegalArgumentException.class, () -> other.rollbackPreparation(localReservation));
            assertThrows(IllegalArgumentException.class, () -> EndpointTestSupport.commitRoutes(other,
                    List.of(otherRequest), List.of(localReservation), permit));
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(1L, other.admissionSummary(0, 0L).occupiedRequests());
            try (var handoff = EndpointTestSupport.commitRoutes(other,
                    List.of(otherRequest), List.of(otherReservation), permit)) {
                assertTrue(other.releaseRequest(otherRequest) == PrefillState.RequestRelease.COMMITTED);
            }
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests(), "foreign settlement cannot release the local owner");
        }
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(0L, other.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void staleRouteCannotCommitOrReleaseReplacementWithTheSameRequestId() {
        RequestRoute first = item(1), replacement = item(1);
        enqueue(first);
        assertEquals(PrefillState.RequestRelease.QUEUED, state.releaseRequest(first));
        enqueue(replacement);
        assertEquals(PrefillState.RequestRelease.NONE, state.releaseRequest(first));
        try (var permit = generation.tryAcquireHandoff()) {
            assertThrows(IllegalStateException.class, () -> EndpointTestSupport.commitQueuedRoutes(state,
                    List.of(first), new long[]{999L}, permit));
        }
        assertEquals(List.of(replacement), waitingItems());
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(EndpointTestSupport.releaseRequest(state, first));
    }

    @Test
    void stopCallbackAcknowledgesOwnershipAlreadyReleasedByTerminalCleanup() {
        var ledger = new PrefillState(lock, waiting, clock::get);
        RequestRoute request = item(1);
        lock.lock();
        try { assertTrue(ledger.enqueueActiveLocked(request, 10L)); }
        finally { lock.unlock(); }
        assertSame(request, ledger.detachNextActiveForStop());
        assertTrue(waitingItems().isEmpty());
        assertEquals(1L, ledger.admissionSummary(0, 0L).occupiedRequests(), "stop callback retains a request seat until settlement");
        assertEquals(PrefillState.RequestRelease.QUEUED, ledger.releaseRequest(request));
        assertEquals(0L, ledger.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(PrefillState.RequestRelease.NONE, ledger.releaseRequest(request));
        lock.lock();
        try { assertTrue(ledger.acknowledgeStopTerminalLocked(request)); }
        finally { lock.unlock(); }
    }

    @Test
    void activeRemovalLeavesOpenBatchLeaseWithPreparingTransaction() {
        AtomicInteger drained = new AtomicInteger();
        var ledger = new PrefillState(lock, waiting, clock::get);
        var retiring = new EndpointGenerationLifecycle(drained::incrementAndGet);
        RequestRoute request = item(1);
        lock.lock();
        try {
            assertTrue(ledger.enqueueActiveLocked(request, 10L));
        } finally {
            lock.unlock();
        }
        var lease = ledger.reserveBatch(request, 9L, 1, retiring.tryAcquireHandoff()).reservation();
        retiring.beginRetirement();
        assertFalse(retiring.tryStartCleanup());
        lock.lock();
        try {
            assertTrue(ledger.removeQueuedLocked(request));
            assertTrue(waiting.isEmpty());
            assertEquals(0L, ledger.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(1, ledger.captureQueueCounters().batchSlots());
            assertEquals(0, drained.get());
        } finally {
            lock.unlock();
        }
        EndpointTestSupport.rollback(lease);
        EndpointTestSupport.rollback(lease);
        assertEquals(1, drained.get());
        lock.lock();
        try {
            assertEquals(0, ledger.captureQueueCounters().batchSlots());
        } finally {
            lock.unlock();
        }
    }

    @Test
    void interleavedImmediateCommitsCaptureOtherWorkAtTheActualCommitBoundary() {
        RequestRoute first = item(1), second = item(2);
        {
            var firstReservation = state.reserveUnqueuedRoute(first, 30L, Long.MAX_VALUE).reservation();
            var secondReservation = state.reserveUnqueuedRoute(second, 40L, Long.MAX_VALUE).reservation();
            try (var preparationFirstReservation = EndpointTestSupport.preparation(firstReservation);
                 var preparationSecondReservation = EndpointTestSupport.preparation(secondReservation)) {
                try (var secondHandoff = EndpointTestSupport.commitRoutes(state, List.of(second), List.of(secondReservation),
                        generation.tryAcquireHandoff())) {
                    assertEquals(30L, secondHandoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow());
                }
                try (var firstHandoff = EndpointTestSupport.commitRoutes(state, List.of(first), List.of(firstReservation),
                        generation.tryAcquireHandoff())) {
                    assertEquals(40L, firstHandoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow(),
                            "the earlier reservation must see work committed before its own commit");
                }
            }
        }
    }

    @Test
    void batchAndIndividualCommitsCaptureTheSamePrecedingTimeline() {
        RequestRoute immediate = item(1), batchMember = item(2), later = item(3);
        {
            var immediateReservation = state.reserveUnqueuedRoute(immediate, 30L, Long.MAX_VALUE).reservation();
            try (var preparationImmediateReservation = EndpointTestSupport.preparation(immediateReservation);
                 var firstHandoff = EndpointTestSupport.commitRoutes(state, List.of(immediate), List.of(immediateReservation),
                     generation.tryAcquireHandoff())) {
                assertEquals(0L, firstHandoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow());
                enqueue(batchMember);
                {
                    var batchReservation = state.reserveBatch(batchMember, 10L, 2, generation.tryAcquireHandoff()).reservation();
                    try (var preparationBatchReservation = EndpointTestSupport.preparation(batchReservation);
                         var batchHandoff = EndpointTestSupport.commitBatch(state, batchReservation, List.of(batchMember), 40L)) {
                        assertEquals(30L, batchHandoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow());
                    }
                }
                {
                    var laterReservation = state.reserveUnqueuedRoute(later, 50L, Long.MAX_VALUE).reservation();
                    try (var preparationLaterReservation = EndpointTestSupport.preparation(laterReservation);
                         var laterHandoff = EndpointTestSupport.commitRoutes(state, List.of(later), List.of(laterReservation),
                             generation.tryAcquireHandoff())) {
                        assertEquals(70L, laterHandoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow());
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void batchHandoffFreezesPrecedingWorkAcrossClockAdvanceAndTerminalCleanup(boolean cached) {
        RequestRoute preceding = item(81), incoming = item(82);
        commitBatch(List.of(preceding), 300L);
        reconcile(Map.of(), Map.of("81", task(81, TaskPhase.RUNNING, 0, 0)), unused -> {
            throw new AssertionError("an unchanged batch needs no prediction");
        });
        PrefillState.WorkCapture earlier = cached ? capture().work() : null;
        clock.set(150L);
        enqueue(incoming);
        {
            var reservation = state.reserveBatch(incoming, 82L, 2, generation.tryAcquireHandoff()).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                 var handoff = EndpointTestSupport.commitBatch(state, reservation, List.of(incoming), 50L)) {
                if (cached) { assertSame(earlier, handoff.precedingWork()); }
                assertTrue(EndpointTestSupport.releaseRequest(state, preceding));
                assertTrue(EndpointTestSupport.releaseRequest(state, incoming));
                clock.set(190L);
                WorkSnapshot frozen = handoff.precedingWork().materialize();
                assertTrue(frozen.containsRequest(81L));
                assertEquals(250L, frozen.knownRemainingWorkMsAt(150L));
                assertEquals(210L, frozen.knownRemainingWorkMsAt(190L));
                assertFalse(frozen.containsRequest(82L), "the incoming batch is not preceding work");
                assertNoWork(state.committedSnapshot(), 81L, 82L);
                assertEquals(0, state.stats().batchCount());
            }
        }
    }

    @Test
    void localCleanupPreservesRunningBatchWorkAndItsClock() {
        RequestRoute a = item(1), b = item(2), c = item(3);
        commitBatch(List.of(a, b, c), 300L);
        reconcile(Map.of(), Map.of("1", task(1, TaskPhase.RUNNING, 0, 0),
                "2", task(2, TaskPhase.RUNNING, 0, 0),
                "3", task(3, TaskPhase.RUNNING, 0, 0)), unused -> {
                    throw new AssertionError("an unchanged batch needs no prediction");
                });

        clock.set(150);
        assertTrue(EndpointTestSupport.releaseRequest(state, a));
        assertTrue(state.committedSnapshot().containsRequest(2L));
        assertTrue(state.committedSnapshot().containsRequest(3L));
        assertFalse(state.committedSnapshot().containsRequest(1L));
        assertEquals(250L, remainingWork());
        clock.set(190);
        assertTrue(EndpointTestSupport.releaseRequest(state, b));
        assertFalse(EndpointTestSupport.releaseRequest(state, a));
        assertFalse(EndpointTestSupport.releaseRequest(state, item(3)), "cleanup must match the exact item");
        assertEquals(210L, remainingWork());

        // Returning to QUEUED pauses elapsed-time accounting, but does not undo work already started.
        clock.set(230);
        reconcile(Map.of("1", task(1, null, 0, 100)),
                Map.of("3", task(3, TaskPhase.RECEIVED, 0, 0)), unused -> {
                    throw new AssertionError("local cancellation cannot shrink running work");
                });
        clock.set(250);
        assertEquals(170L, remainingWork());
        assertEquals(WorkSnapshot.Phase.ENGINE_QUEUED, batchPhase(10L));

        var completed = reconcile(Map.of("3", task(3, null, 0, 300)), Map.of(), unused -> {
            throw new AssertionError("a completed batch needs no prediction");
        });
        assertEquals(1, completed.batchCompletions().size());
        assertTrue(completed.batchCompletions().getFirst().successfulCompletion());
        assertFalse(completed.batchCompletions().getFirst().learningEligible(),
                "local cleanup disqualifies learning even when all remaining Worker results succeed");
        assertNoWork(state.committedSnapshot(), 1L, 2L, 3L);
        assertEquals(0, state.stats().batchCount());
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
        assertNoWork(state.committedSnapshot(), 1L, 2L, 3L);
        assertEquals(0, state.stats().batchCount());
    }

    @Test
    void onlyWorkerRemovalBeforeExecutionRepredictsTheQueuedBatch() {
        RequestRoute a = item(1), b = item(2);
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

    @Test
    void predictionFailureLeavesBatchAndWaitingOwnershipUnchanged() {
        RequestRoute a = item(1), b = item(2), queued = item(3);
        commitBatch(List.of(a, b), 300L);
        enqueue(queued);
        var before = capture();
        long version = state.mutationVersion();
        var failure = new IllegalStateException("prediction unavailable");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(Map.of("1", task(1, null, 500, 0)));
        response.setRunningTaskInfo(Map.of("2", task(2, TaskPhase.RECEIVED, 0, 0)));
        var observation = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090)
                .freezeStatusResponse(response);

        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> EndpointTestSupport.reconcile(state, observation, survivors -> {
                    assertEquals(List.of(b), survivors);
                    throw failure;
                })));

        assertEquals(version, state.mutationVersion());
        assertSame(before.work(), capture().work());
        assertWork(state.committedSnapshot(), 300L, 1L, 2L);
        assertEquals(300L, remainingWork());
        assertEquals(2, state.stats().locallyOwnedRequests());
        assertEquals(List.of(queued), waitingItems());
        var completed = reconcile(Map.of("1", task(1, null, 0L, 200L),
                "2", task(2, null, 0L, 300L)), Map.of(), unused -> {
                    throw new AssertionError("a completed batch needs no prediction");
                });
        assertEquals(1, completed.batchCompletions().size());
        assertTrue(completed.batchCompletions().getFirst().learningEligible(),
                "a failed prediction must not retain the speculative failure outcome");
        assertEquals(300L, completed.batchCompletions().getFirst().actualWorkMs());
    }

    @Test
    void invalidatedStatusReductionDoesNotPolluteBatchResults() {
        RequestRoute first = item(1), second = item(2), queued = item(3);
        commitBatch(List.of(first, second), 300L);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(Map.of("1", task(1, null, 500L, 0L)));
        response.setRunningTaskInfo(Map.of("2", task(2, TaskPhase.RECEIVED, 0L, 0L)));
        var observation = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090)
                .freezeStatusResponse(response);
        PrefillState.StatusReduction reduction;
        lock.lock();
        try {
            reduction = state.prepareStatusLocked(observation);
        } finally {
            lock.unlock();
        }
        assertEquals(Map.of(10L, List.of(second)), reduction.predictionInputs());
        enqueue(queued);
        lock.lock();
        try {
            assertNull(state.commitStatusLocked(reduction, Map.of(10L, 200L)));
        } finally {
            lock.unlock();
        }
        assertWork(state.committedSnapshot(), 300L, 1L, 2L);
        assertEquals(300L, remainingWork());
        var completed = reconcile(Map.of("1", task(1, null, 0L, 200L),
                "2", task(2, null, 0L, 300L)), Map.of(), unused -> {
                    throw new AssertionError("a completed batch needs no prediction");
                });
        assertEquals(1, completed.batchCompletions().size());
        assertTrue(completed.batchCompletions().getFirst().learningEligible(),
                "a stale reduction must not retain the speculative failure outcome");
        assertEquals(300L, completed.batchCompletions().getFirst().actualWorkMs());
        assertEquals(List.of(queued), waitingItems());
    }

    @Test
    void terminalBatchAndUnchangedBatchRetainTheirOwnFacts() {
        RequestRoute a = item(1), b = item(2), c = item(3), d = item(4);
        commitBatch(List.of(a, b), 300L);
        enqueue(c);
        enqueue(d);
        {
            var reservation = state.reserveBatch(c, 11L, 2,
                generation.tryAcquireHandoff()).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                 var handoff = EndpointTestSupport.commitBatch(state, reservation, List.of(c, d), 400L)) {
                assertTrue(waitingItems().isEmpty());
            }
        }
        clock.set(200L);
        TaskInfo activeC = task(3L, TaskPhase.RUNNING, 0L, 0L);
        TaskInfo activeD = task(4L, TaskPhase.RUNNING, 0L, 0L);
        activeC.setBatchId(11L);
        activeD.setBatchId(11L);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(Map.of("1", task(1, null, 0L, 300L),
                "2", task(2, null, 0L, 300L)));
        response.setRunningTaskInfo(Map.of("3", activeC, "4", activeD));
        var observation = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090)
                .freezeStatusResponse(response);
        var outcome = EndpointTestSupport.reconcile(state, observation, unused -> {
            throw new AssertionError("completed and unchanged batches need no new prediction");
        });

        assertEquals(4, outcome.requestStatuses().size());
        assertEquals(List.of(1L, 2L), outcome.requestStatuses().stream()
                .filter(requestStatus -> requestStatus.kind() == PrefillState.PrefillRequestStatus.Kind.COMPLETED)
                .map(requestStatus -> requestStatus.route().requestId()).sorted().toList());
        assertEquals(1, outcome.batchCompletions().size());
        var completion = outcome.batchCompletions().getFirst();
        assertEquals(10L, completion.batchId());
        assertEquals(300L, completion.actualWorkMs());
        assertTrue(completion.successfulCompletion());
        assertTrue(completion.learningEligible());
        assertEquals(1, state.stats().batchCount());
        assertEquals(2, state.stats().locallyOwnedRequests());
        clock.set(250L);
        assertEquals(350L, remainingWork(), "the batch without terminal events still advances to RUNNING");
        WorkSnapshot remaining = state.committedSnapshot();
        assertTrue(remaining.containsRequest(3L));
        assertTrue(remaining.containsRequest(4L));
        assertFalse(remaining.containsRequest(1L));
        assertFalse(remaining.containsRequest(2L));
    }

    @Test
    void localCleanupDoesNotInventExecutionEvidenceOrEnableLearning() {
        RequestRoute first = item(1), rejected = item(2), survivor = item(3);
        commitBatch(List.of(first, rejected, survivor), 300L);
        assertTrue(EndpointTestSupport.releaseRequest(state, first));
        var partial = reconcile(Map.of("2", task(2, null, 500L, 0L)), Map.of(), members -> {
            assertEquals(List.of(survivor), members, "local cleanup cannot imply execution started");
            return 90L;
        });
        assertTrue(partial.batchCompletions().isEmpty());
        assertEquals(90L, remainingWork());
        var result = reconcile(Map.of("3", task(3, null, 0L, 100L)), Map.of(), unused -> {
            throw new AssertionError("a completed batch needs no prediction");
        });
        var completion = result.batchCompletions().getFirst();
        assertEquals(100L, completion.actualWorkMs());
        assertTrue(completion.successfulCompletion());
        assertFalse(completion.learningEligible(), "local cleanup disqualifies the whole batch from learning");
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void partialSuccessThenFailureCompletesBatchOnceWithoutLearningOrRepacking() {
        commitBatch(List.of(item(1), item(2)), 400L);
        ToLongFunction<List<RequestRoute>> noRepacking = survivors -> {
            throw new AssertionError("execution evidence must prevent repacking");
        };
        var partial = reconcile(Map.of("1", task(1, null, 0L, 300L)),
                Map.of("2", task(2, TaskPhase.RUNNING, 0L, 0L)), noRepacking);
        assertTrue(partial.batchCompletions().isEmpty());
        assertEquals(1, state.stats().batchCount());
        assertEquals(1, state.stats().locallyOwnedRequests());

        var failed = Map.of("2", task(2, null, 1L, 200L));
        var settled = reconcile(failed, Map.of(), noRepacking);
        assertEquals(1, settled.batchCompletions().size());
        var completion = settled.batchCompletions().getFirst();
        assertEquals(10L, completion.batchId());
        assertEquals(300L, completion.actualWorkMs());
        assertTrue(completion.successfulCompletion());
        assertFalse(completion.learningEligible());
        assertEquals(0, state.stats().batchCount());
        assertEquals(0, state.stats().locallyOwnedRequests());
        assertTrue(reconcile(failed, Map.of(), noRepacking).batchCompletions().isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void oldBatchCompletionDoesNotSettleAReusedMemberId(boolean workerTerminal) {
        RequestRoute first = item(1), sibling = item(2);
        commitBatch(List.of(first, sibling), 300L);
        assertTrue(EndpointTestSupport.releaseRequest(state, first));

        RequestRoute replacement = item(1);
        enqueue(replacement);
        {
            var lease = state.reserveBatch(replacement, 11L, 2,
                generation.tryAcquireHandoff()).reservation();
            try (var preparationLease = EndpointTestSupport.preparation(lease);
                 var handoff = EndpointTestSupport.commitBatch(state, lease, List.of(replacement), 100L)) {
                assertEquals(2, state.captureQueueCounters().batchSlots());
            }
        }
        ToLongFunction<List<RequestRoute>> noRepacking = unused -> {
            throw new AssertionError("completed batches do not need reprediction");
        };
        if (workerTerminal) {
            var completion = reconcile(Map.of("2", task(2, null, 0L, 300L)), Map.of(), noRepacking);
            assertEquals(List.of(10L), completion.batchCompletions().stream()
                    .map(PrefillState.BatchCompletion::batchId).toList());
        } else {
            assertTrue(EndpointTestSupport.releaseRequest(state, sibling));
        }
        assertEquals(1, state.captureQueueCounters().batchSlots());
        assertEquals(1, state.stats().locallyOwnedRequests());
        assertEquals(List.of(11L), batchIds());
        assertWork(state.committedSnapshot(), 100L, 1L);
        assertFalse(state.committedSnapshot().containsRequest(2L));
        assertFalse(EndpointTestSupport.releaseRequest(state, first), "old item cannot settle the replacement");
        var stale = reconcile(Map.of("1", task(1, null, 0L, 300L)), Map.of(), noRepacking);
        assertTrue(stale.batchCompletions().isEmpty(), "old batch proof cannot settle the new batch");
        assertTrue(stale.requestStatuses().isEmpty());
        assertEquals(1, state.captureQueueCounters().batchSlots());

        TaskInfo terminal = task(1, null, 0L, 100L);
        terminal.setBatchId(11L);
        var completion = reconcile(Map.of("1", terminal), Map.of(), noRepacking);
        assertEquals(List.of(11L), completion.batchCompletions().stream()
                .map(PrefillState.BatchCompletion::batchId).toList());
        assertEquals(0, state.captureQueueCounters().batchSlots());
        assertEquals(0, state.stats().locallyOwnedRequests());
    }

    @Test
    void directGroupValidationPreservesArgumentAndOwnershipErrorPrecedence() {
        RequestRoute request = routeItem(101);
        var lease = state.reserveUnqueuedRoute(request, 30L, 10L).reservation();
        try (var preparation = EndpointTestSupport.preparation(lease);
             var permit = generation.tryAcquireHandoff()) {
            assertThrows(IllegalArgumentException.class, () -> EndpointTestSupport.commitRoutes(state,
                    List.of(), List.of(), permit));
            assertThrows(IllegalStateException.class, () -> EndpointTestSupport.commitRoutes(state,
                    List.of(request, routeItem(102)), List.of(lease), permit));
            assertThrows(IllegalArgumentException.class, () -> EndpointTestSupport.commitRoutes(state,
                    List.of(request), List.of(lease, lease), permit));
            assertEquals(30L, remainingWork());
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        }
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void directGroupLeaseMismatchLeavesEveryReservationRetryable() {
        RequestRoute first = routeItem(101), second = routeItem(102);
        var firstLease = state.reserveUnqueuedRoute(first, 30L, 10L).reservation();
        var secondLease = state.reserveUnqueuedRoute(second, 40L, 10L).reservation();
        try (var firstPreparation = EndpointTestSupport.preparation(firstLease);
             var secondPreparation = EndpointTestSupport.preparation(secondLease);
             var permit = generation.tryAcquireHandoff()) {
            assertThrows(IllegalStateException.class, () -> EndpointTestSupport.commitRoutes(state,
                    List.of(first, second), List.of(firstLease, firstLease), permit));
            assertEquals(70L, remainingWork());
            assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
            try (var handoff = EndpointTestSupport.commitRoutes(state, List.of(first, second),
                    List.of(firstLease, secondLease), permit)) {
                assertEquals(0L, handoff.precedingWork().materialize().totalRemainingWorkMs().orElseThrow());
            }
        }
        assertEquals(PrefillState.RequestRelease.COMMITTED, state.releaseRequest(first));
        assertEquals(PrefillState.RequestRelease.COMMITTED, state.releaseRequest(second));
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void duplicateTerminalReportsMergeBeforeBatchAggregationAndReleaseOnce() {
        commitBatch(List.of(item(1), item(2)), 300L);
        var finished = Map.of("first-success", task(1, null, 0L, 100L),
                "first-failure", task(1, null, 500L, 700L),
                "second-success", task(2, null, 0L, 200L));
        ToLongFunction<List<RequestRoute>> noRepacking = unused -> {
            throw new AssertionError("a completed batch needs no prediction");
        };
        var result = reconcile(finished, Map.of(), noRepacking);
        assertEquals(2, result.requestStatuses().size());
        assertEquals(1L, result.requestStatuses().stream()
                .filter(status -> status.kind() == PrefillState.PrefillRequestStatus.Kind.FAILED).count());
        assertEquals(1, result.batchCompletions().size());
        var completion = result.batchCompletions().getFirst();
        assertEquals(700L, completion.actualWorkMs());
        assertTrue(completion.successfulCompletion());
        assertFalse(completion.learningEligible());
        assertTrue(result.capacityReleased());
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        var repeated = reconcile(finished, Map.of(), noRepacking);
        assertTrue(repeated.batchCompletions().isEmpty());
        assertTrue(repeated.requestStatuses().isEmpty());
        assertFalse(repeated.capacityReleased());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preparedCapacityReleaseTracksUnknownOwnershipAndRejectsStaleCounts(boolean invalidate) {
        var worker = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningQueryLen(4L);
        state.reconcileHeartbeat(worker.freezeStatusResponse(response));
        assertEquals(4L, state.admissionSummary(0, 0L).occupiedRequests());
        response.setRunningQueryLen(2L);
        var observation = worker.freezeStatusResponse(response);
        lock.lock();
        try {
            var reduction = state.prepareStatusLocked(observation);
            assertEquals(4L, state.admissionSummary(0, 0L).occupiedRequests(), "preparation must not release capacity");
            if (invalidate) {
                response.setRunningQueryLen(1L);
                state.reconcileHeartbeat(worker.freezeStatusResponse(response));
                assertNull(state.commitStatusLocked(reduction, Map.of()));
                assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                reduction = state.prepareStatusLocked(observation);
            }
            var result = state.commitStatusLocked(reduction, Map.of());
            assertEquals(!invalidate, result.capacityReleased(),
                    "only a decrease from the current unknown ownership releases capacity");
            assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        } finally {
            lock.unlock();
        }
    }

    @Test
    void heartbeatRenewsExecutionWithoutConsumingTerminalFactsOrInvalidatingUnchangedInputs() {
        RequestRoute first = item(1), second = item(2);
        commitBatch(List.of(first, second), 300L);
        var worker = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        TaskInfo running = task(1, TaskPhase.RUNNING, 0L, 0L);
        running.setBatchId(10L);
        response.setRunningTaskInfo(Map.of("1", running));
        response.setFinishedTaskInfo(Map.of("1", task(1, null, 0L, 100L)));
        var observation = worker.freezeStatusResponse(response);
        var started = state.reconcileHeartbeat(observation);
        assertEquals(1, started.requestStatuses().size());
        assertEquals(PrefillState.PrefillRequestStatus.Kind.ACTIVE, started.requestStatuses().getFirst().kind());
        assertTrue(started.batchCompletions().isEmpty(), "a heartbeat cannot complete a batch");
        assertFalse(started.capacityReleased());
        assertTrue(started.schedulingInputsChanged());
        long version = state.mutationVersion();
        clock.set(200L);
        var renewed = state.reconcileHeartbeat(observation);
        assertFalse(renewed.schedulingInputsChanged(), "same-phase renewal leaves the projection revision stable");
        assertFalse(renewed.capacityReleased());
        assertEquals(version, state.mutationVersion());
        assertEquals(0L, state.stats().maxObservedAgeMs(), "same-phase renewal refreshes the activity timestamp");
        assertEquals(200L, remainingWork(), "same-phase renewal still advances the execution clock");
        assertWork(state.committedSnapshot(), 200L, 1L, 2L);
        assertEquals(2, state.stats().locallyOwnedRequests(),
                "the heartbeat's terminal report remains unconsumed");
        var completed = reconcile(Map.of("1", task(1, null, 0L, 100L), "2", task(2, null, 0L, 100L)),
                Map.of(), unused -> { throw new AssertionError("a completed batch needs no prediction"); });
        assertTrue(completed.schedulingInputsChanged());
        assertTrue(completed.capacityReleased());
        assertEquals(1, completed.batchCompletions().size());
        assertNoWork(state.committedSnapshot(), 1L, 2L);
        assertEquals(0, state.stats().batchCount());
    }

    private void commitBatch(List<RequestRoute> members, long predictedMs) {
        members.forEach(this::enqueue);
        {
            var reservation = state.reserveBatch(members.getFirst(), 10L, 2,
                generation.tryAcquireHandoff()).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                 var handoff = EndpointTestSupport.commitBatch(state, reservation, members, predictedMs)) {
                assertTrue(waitingItems().isEmpty());
            }
        }
    }

    @Test
    void unchangedWorkerObservationReusesWorkAndProjectionVersion() {
        RequestRoute request = item(1);
        commitBatch(List.of(request), 100L);
        var active = Map.of("1", task(1, TaskPhase.RUNNING, 0L, 0L));
        ToLongFunction<List<RequestRoute>> noRepacking = ignored -> {
            throw new AssertionError("unchanged membership needs no prediction");
        };
        reconcile(Map.of(), active, noRepacking);
        var before = capture();
        clock.addAndGet(20L);
        var result = reconcile(Map.of(), active, noRepacking);
        var after = capture();
        assertEquals(1, result.requestStatuses().size(), "activity must still reach the request owner");
        assertFalse(result.capacityReleased());
        assertEquals(before.version(), after.version());
        assertSame(before.work(), after.work());
        assertEquals(80L, after.work().materialize().totalRemainingWorkMsAt(clock.get()).orElseThrow());
    }

    private PrefillState.StatusReconciliation reconcile(Map<String, TaskInfo> finished, Map<String, TaskInfo> active,
                           ToLongFunction<List<RequestRoute>> repredictor) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(active);
        var observation = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090)
                .freezeStatusResponse(response);
        var outcome = EndpointTestSupport.reconcile(state, observation, repredictor);
        return outcome;
    }

    private static void assertWork(WorkSnapshot snapshot, long expectedMs, long... includedIds) {
        assertFalse(snapshot.hasUnknownWork());
        assertEquals(expectedMs, snapshot.totalRemainingWorkMs().orElseThrow());
        for (long id : includedIds) { assertTrue(snapshot.containsRequest(id), "missing request " + id); }
    }

    private static void assertNoWork(WorkSnapshot snapshot, long... excludedIds) {
        assertWork(snapshot, 0L);
        for (long id : excludedIds) { assertFalse(snapshot.containsRequest(id), "retained request " + id); }
    }

    private List<Long> batchIds() {
        lock.lock();
        try {
            Map<?, ?> batches = (Map<?, ?>) org.springframework.test.util.ReflectionTestUtils
                    .getField(state, "batches");
            return batches.keySet().stream().map(id -> (Long) id).sorted().toList();
        } finally { lock.unlock(); }
    }

    private WorkSnapshot.Phase batchPhase(long batchId) {
        lock.lock();
        try {
            Map<?, ?> batches = (Map<?, ?>) org.springframework.test.util.ReflectionTestUtils
                    .getField(state, "batches");
            Object batch = batches.get(batchId);
            org.junit.jupiter.api.Assertions.assertNotNull(batch);
            return (WorkSnapshot.Phase) org.springframework.test.util.ReflectionTestUtils
                    .getField(batch, "servicePhase");
        } finally { lock.unlock(); }
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

    private static RequestRoute item(long id) {
        var item = mock(RequestRoute.class);
        when(item.requestId()).thenReturn(id);
        when(item.seqLen()).thenReturn(100L);
        return item;
    }

    private PrefillState.Snapshot capture() {
        lock.lock();
        try {
            return state.snapshotLocked();
        } finally {
            lock.unlock();
        }
    }

    private void enqueue(RequestRoute item) {
        lock.lock();
        try {
            assertTrue(state.enqueueActiveLocked(item, 0L));
        } finally {
            lock.unlock();
        }
    }

    private List<RequestRoute> waitingItems() {
        return state.captureQueue(Integer.MAX_VALUE).items();
    }
}
