package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks shared Prefill request limits across waiting, prepared and committed work.
 */
class PrefillRequestCapacityTest {
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillEndpoint state =
            org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                    lock,
                    PrefillActiveIndex.ordered(
                            16,
                            Comparator.comparing(
                                    org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem
                                            ::requestId)),
                    System::currentTimeMillis);

    @Test
    void duplicateDirectAdmissionDoesNotReplaceOrReleaseTheOriginalOwner() {
        RequestRoute route = item(1);
        var original = reserve(route, 1L).value();
        assertNotNull(original);
        try {
            var duplicate = reserve(route, 1L);
            assertFalse(duplicate.accepted());
            assertSame(CapacityBoundary.OWNERSHIP_LOST, duplicate.boundary());
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(
                    CapacityBoundary.Resource.PREFILL_REQUEST,
                    reserve(item(2), 1L).boundary().resource());
        } finally {
            EndpointTestSupport.rollback(original);
        }
        assertTrue(state.requestCapacityAvailable(1, 1));
    }

    @Test
    void queuedOwnersDoNotConsumeImmediateDeliveryCapacity() {
        RequestRoute queued = item(1), immediate = item(2), another = item(3);
        assertTrue(enqueue(queued, 2L));
        var first = reserve(immediate, 2L).value();
        assertNotNull(first);
        var second = reserve(another, 2L).value();
        assertNotNull(second);
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(4), 2L).boundary().resource());
        try (var handoff = commit(immediate, first)) {}
        EndpointTestSupport.rollback(first);
        assertFalse(state.requestCapacityAvailable(2, 1), "committed execution retains its slot");
        assertTrue(EndpointTestSupport.releaseRequest(state, immediate));
        assertFalse(EndpointTestSupport.releaseRequest(state, immediate));
        assertTrue(state.requestCapacityAvailable(2, 1));
        EndpointTestSupport.rollback(second);
        assertTrue(state.requestCapacityAvailable(2, 2));
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void workerDetailsDeduplicateLocalRequestsAndForeignRequestIds() {
        RequestRoute local = item(1);
        var registration = reserve(local, 4L).value();
        var localObservation = observed(1);
        var foreign = observed(2);
        heartbeat(Map.of("local", localObservation, "foreign", foreign, "duplicate", foreign), 2L);
        assertEquals(
                2L,
                state.admissionSummary(0, 0L).occupiedRequests(),
                "early Engine observation and prepared local ownership are one request");
        try (var handoff = commit(local, registration)) {}
        EndpointTestSupport.rollback(registration);
        heartbeat(Map.of("local", localObservation, "foreign", foreign), 2L);
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < 2L);
        heartbeat(Map.of("local", localObservation), 1L);
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 2L);
    }

    @Test
    void scalarEngineWorkIsCountedWithoutInventingIdentity() {
        var registration = reserve(item(1), 4L).value();
        heartbeat(Map.of(), 2L);
        assertEquals(
                3L,
                state.admissionSummary(0, 0L).occupiedRequests(),
                "unseen local shadow cannot explain unidentified Engine work");
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(2), 3L).boundary().resource());
        EndpointTestSupport.rollback(registration);
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void changingLimitCannotReleaseExistingOwners() {
        var first = reserve(item(1), 2L).value();
        var second = reserve(item(2), 2L).value();
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(3), 1L).boundary().resource());
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < 1L);
        assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 5L);
        EndpointTestSupport.rollback(first);
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(3), 1L).boundary().resource());
        EndpointTestSupport.rollback(second);
        assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 1L);
        assertNotNull(reserve(item(3), 1L).value());
    }

    @Test
    void batchRequestCountDoesNotLimitWaitingAndBatchPermitWaitsForLastMember() {
        RequestRoute first = item(1), second = item(2);
        assertTrue(enqueue(first, 0L));
        assertTrue(enqueue(second, 0L));
        var generation = new EndpointRetirementTestSupport(() -> {});
        var reservation = state.reserveBatch(10L, 1, generation.tryPinGeneration()).value();
        try (var handoff =
                EndpointTestSupport.commitBatch(state, reservation, List.of(first, second), 0L)) {}
        assertFalse(state.batchCapacityAvailable(1));
        assertTrue(enqueue(item(3), 0L));
        assertTrue(EndpointTestSupport.releaseRequest(state, first));
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(state.batchCapacityAvailable(1));
        assertTrue(EndpointTestSupport.releaseRequest(state, second));
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertTrue(state.batchCapacityAvailable(1));
    }

    @Test
    void staleOwnerCannotReleaseAReusedRequestId() {
        RequestRoute previous = item(1);
        var old = reserve(previous, 1L).value();
        EndpointTestSupport.rollback(old);
        var current = reserve(item(1), 1L).value();
        EndpointTestSupport.rollback(old);
        assertFalse(EndpointTestSupport.releaseRequest(state, previous));
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        EndpointTestSupport.rollback(current);
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void immediateAdmissionChecksCurrentCapacityAfterOtherOwnershipChanges() {
        assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 2L);
        assertNotNull(reserve(item(1), 2L).value());
        assertNotNull(
                reserve(item(2), 2L).value(),
                "an intervening ownership mutation does not invalidate remaining capacity");
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(3), 2L).boundary().resource(),
                "the old available snapshot cannot authorize overselling current capacity");
    }

    @Test
    void progressReadersDoNotWaitForTheOwnershipLock() throws Exception {
        RequestRoute queued = item(1);
        assertTrue(enqueue(queued, 2L));
        try (var reader = Executors.newSingleThreadExecutor()) {
            lock.lock();
            try {
                assertFalse(
                        reader.submit(() -> state.progress().queueDepth() < 1L)
                                .get(5, TimeUnit.SECONDS));
                assertTrue(
                        reader.submit(() -> state.progress().queueDepth() < 2L)
                                .get(5, TimeUnit.SECONDS));
                assertSame(
                        queued,
                        org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(
                                state, queued));
                assertTrue(
                        reader.submit(() -> state.progress().queueDepth() < 1L)
                                .get(5, TimeUnit.SECONDS),
                        "removal publishes queue progress before the ownership lock is released");
            } finally {
                lock.unlock();
            }
        }
    }

    @Test
    void concurrentAvailableSnapshotsCannotOversellOneSeat() throws Exception {
        int contenders = 8;
        CountDownLatch selected = new CountDownLatch(contenders);
        CountDownLatch admit = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(contenders)) {
            List<Future<CapacityBoundary.Attempt<PrefillEndpoint.RouteReservation>>> attempts =
                    new ArrayList<>();
            for (int i = 0; i < contenders; i++) {
                RequestRoute request = item(i + 1);
                attempts.add(
                        executor.submit(
                                () -> {
                                    assertTrue(
                                            state.admissionSummary(0, 0L).occupiedRequests() < 1L);
                                    selected.countDown();
                                    assertTrue(admit.await(5, TimeUnit.SECONDS));
                                    return reserve(request, 1L);
                                }));
            }
            try {
                assertTrue(selected.await(5, TimeUnit.SECONDS));
            } finally {
                admit.countDown();
            }
            List<PrefillEndpoint.RouteReservation> acquired = new ArrayList<>();
            for (var attempt : attempts) {
                var result = attempt.get(5, TimeUnit.SECONDS);
                if (result.value() != null) {
                    acquired.add(result.value());
                } else {
                    assertEquals(
                            CapacityBoundary.Resource.PREFILL_REQUEST,
                            result.boundary().resource());
                }
            }
            assertEquals(1, acquired.size());
            assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < 1L);
            EndpointTestSupport.rollback(acquired.getFirst());
            assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 1L);
        }
    }

    @Test
    void summaryCountsUnknownWorkWithoutOverflowAndClearsOnRetirement() {
        var reservation = reserve(item(1), 2L).value();
        heartbeat(Map.of(), Long.MAX_VALUE);
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < Long.MAX_VALUE);
        EndpointTestSupport.rollback(reservation);
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < Long.MAX_VALUE);
        heartbeat(Map.of(), 0L);
        assertTrue(enqueue(item(2), 1L));
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < 1L);
        state.retireResources();
        assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 1L);
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < 0L);
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < -1L);
    }

    @Test
    void failedQueueInsertionDoesNotConsumePublishedCapacity() {
        var direct =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, PrefillActiveIndex.disabled(), System::currentTimeMillis);
        lock.lock();
        try {
            assertThrows(
                    IllegalStateException.class,
                    () -> org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(direct, item(1)));
            assertTrue(direct.admissionSummary(0, 0L).occupiedRequests() < 1L);
            assertEquals(0L, direct.admissionSummary(0, 0L).occupiedRequests());
        } finally {
            lock.unlock();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedReplacementInsertionKeepsVictimsAndCapacityForRetry(boolean fatal) {
        RequestRoute first = item(1), second = item(2), incoming = item(3);
        when(first.priority()).thenReturn(20);
        when(second.priority()).thenReturn(20);
        when(incoming.priority()).thenReturn(80);
        when(first.priority()).thenReturn(10);
        when(second.priority()).thenReturn(50);
        when(incoming.priority()).thenReturn(90);
        AtomicBoolean failInsertion = new AtomicBoolean(true);
        Comparator<org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem> ordering =
                (left, right) -> {
                    if (failInsertion.get() && (left == incoming || right == incoming)) {
                        if (fatal) {
                            throw new AssertionError("queue insertion failed");
                        }
                        throw new IllegalStateException("queue insertion failed");
                    }
                    return Long.compare(
                            Long.parseLong(left.requestId()), Long.parseLong(right.requestId()));
                };
        PrefillEndpoint ledger =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, PrefillActiveIndex.ordered(16, ordering), System::currentTimeMillis);
        lock.lock();
        try {
            assertSame(
                    first, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, first));
            assertSame(
                    second,
                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, second));
            Class<? extends Throwable> failureType =
                    fatal ? AssertionError.class : IllegalStateException.class;
            assertThrows(
                    failureType,
                    () ->
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    ledger, incoming));
            assertEquals(List.of(first, second), ledger.captureQueue(Integer.MAX_VALUE).items());
            assertEquals(2L, ledger.admissionSummary(0, 0L).occupiedRequests());

            failInsertion.set(false);
            assertSame(
                    incoming,
                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, incoming));
            assertEquals(
                    List.of(first, second, incoming),
                    ledger.captureQueue(Integer.MAX_VALUE).items());
            assertEquals(3L, ledger.admissionSummary(0, 0L).occupiedRequests());
            assertNull(
                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, item(3)),
                    "a reused id cannot replace its current queue owner");
            assertTrue(ledger.requestCapacityAvailable(1L, 1));
        } finally {
            lock.unlock();
        }
    }

    @Test
    void admissionSummaryKeepsOneRevisionAndReflectsReleaseAndUnknownEngineWork() {
        RequestRoute lower = item(1), same = item(2), higher = item(3);
        when(lower.priority()).thenReturn(10);
        when(same.priority()).thenReturn(50);
        when(higher.priority()).thenReturn(90);
        assertTrue(enqueue(lower, 3L));
        assertTrue(enqueue(same, 3L));
        assertTrue(enqueue(higher, 3L));
        var captured = state.admissionSummary(50, 2L);
        assertEquals(
                new org.flexlb.balance.strategy.PrefillSelectionStrategy.AdmissionSummary(
                        3, 2, 1, 1, 1, 0),
                captured);
        assertEquals(captured, state.admissionSummary(50, 2L));
        assertTrue(state.releaseRequest(lower));
        assertEquals(
                new org.flexlb.balance.strategy.PrefillSelectionStrategy.AdmissionSummary(
                        2, 1, 0, 1, 1, 0),
                state.admissionSummary(50, 2L));
        heartbeat(Map.of(), 2L);
        assertEquals(
                new org.flexlb.balance.strategy.PrefillSelectionStrategy.AdmissionSummary(
                        4, 3, 0, 1, 1, 2),
                state.admissionSummary(50, 2L));
        assertEquals(
                new org.flexlb.balance.strategy.PrefillSelectionStrategy.AdmissionSummary(
                        3, 2, 1, 1, 1, 0),
                captured,
                "a retained summary cannot change when the ledger advances");
    }

    @Test
    void fullStatusPublishesUnknownCapacityBeforeEndpointNotification() {
        var observation = mock(WorkerStatus.StatusObservation.class);
        var engine = mock(WorkerStatus.EngineObservation.class);
        when(engine.runningTaskList()).thenReturn(Map.of());
        when(engine.waitingQueryLen()).thenReturn(2L);
        when(observation.engine()).thenReturn(engine);
        when(observation.finishedTasks()).thenReturn(Map.of());
        var result = EndpointTestSupport.reconcile(state, observation, unused -> 0L);
        assertFalse(state.admissionSummary(0, 0L).occupiedRequests() < 2L);
        assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 3L);
        heartbeat(Map.of(), 1L);
        assertTrue(state.admissionSummary(0, 0L).occupiedRequests() < 2L);
    }

    private CapacityBoundary.Attempt<PrefillEndpoint.RouteReservation> reserve(
            RequestRoute item, long limit) {
        return state.reserveUnqueuedRoute(item, 0L, limit);
    }

    private org.flexlb.balance.scheduler.Delivery commit(
            RequestRoute item, PrefillEndpoint.RouteReservation registration) {
        return org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                state, item, registration);
    }

    private boolean enqueue(RequestRoute item, long limit) {
        lock.lock();
        try {
            return org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, item) == item;
        } finally {
            lock.unlock();
        }
    }

    private void heartbeat(Map<String, WorkerStatus.TaskObservation> tasks, long reportedActive) {
        var observation = mock(WorkerStatus.StatusObservation.class);
        var engine = mock(WorkerStatus.EngineObservation.class);
        when(engine.runningTaskList()).thenReturn(tasks);
        when(engine.waitingQueryLen()).thenReturn(reportedActive);
        when(observation.engine()).thenReturn(engine);
        when(observation.activeTasks()).thenReturn(tasks);
        lock.lock();
        try {
            state.commitStatusLocked(
                    state.prepareStatusLocked(observation, false), java.util.Map.of());
        } finally {
            lock.unlock();
        }
    }

    private static WorkerStatus.TaskObservation observed(long id) {
        return new WorkerStatus.TaskObservation(
                String.valueOf(id),
                0L,
                0L,
                0L,
                0L,
                0L,
                0L,
                0L,
                0L,
                "",
                0L,
                TaskPhase.RUNNING,
                0L,
                PriorityPreemptionProgress.NONE,
                null);
    }

    private static RequestRoute item(long id) {
        RequestRoute item = mock(RequestRoute.class);
        when(item.requestId()).thenReturn(String.valueOf(id));
        return item;
    }
}
