package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillRequestCapacityTest {
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillState state = new PrefillState(lock,
            PrefillActiveIndex.ordered(16, Comparator.comparingLong(ScheduledRequest::requestId)), () -> { });

    @Test
    void waitingPreparedAndCommittedRequestsShareOneCount() {
        ScheduledRequest queued = item(1), immediate = item(2);
        assertTrue(enqueue(queued, 2L));
        var registration = reserve(immediate, 2L).reservation();
        assertNotNull(registration);
        assertEquals(2L, state.observedRequestCount());
        assertFalse(enqueue(item(3), 2L));
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(3), 2L).status());
        try (var handoff = commit(immediate, registration)) {
            assertEquals(2L, state.observedRequestCount());
        }
        registration.close();
        assertEquals(2L, state.observedRequestCount(), "ACK and capability closure cannot release Engine ownership");
        assertTrue(state.terminalizeCommittedItem(immediate));
        assertFalse(state.terminalizeCommittedItem(immediate));
        assertTrue(enqueue(item(3), 2L));
        assertEquals(2L, state.observedRequestCount());
    }

    @Test
    void workerDetailsDeduplicateLocalRequestsAndForeignRequestIds() {
        ScheduledRequest local = item(1);
        var registration = reserve(local, 4L).reservation();
        var localObservation = observed(1);
        var foreign = observed(2);
        heartbeat(Map.of("local", localObservation, "foreign", foreign, "duplicate", foreign), 2L);
        assertEquals(2L, state.observedRequestCount(), "early Engine observation and prepared local ownership are one request");
        try (var handoff = commit(local, registration)) { }
        registration.close();
        heartbeat(Map.of("local", localObservation, "foreign", foreign), 2L);
        assertEquals(2L, state.observedRequestCount());
        assertFalse(state.canAcceptRequest(2L));
        heartbeat(Map.of("local", localObservation), 1L);
        assertEquals(1L, state.observedRequestCount());
        assertTrue(state.canAcceptRequest(2L));
    }

    @Test
    void scalarEngineWorkIsCountedWithoutInventingIdentity() {
        var registration = reserve(item(1), 4L).reservation();
        heartbeat(Map.of(), 2L);
        assertEquals(3L, state.observedRequestCount(), "unseen local shadow cannot explain unidentified Engine work");
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(2), 3L).status());
        registration.close();
        assertEquals(2L, state.observedRequestCount());
    }

    @Test
    void changingLimitCannotReleaseExistingOwners() {
        var first = reserve(item(1), 2L).reservation();
        var second = reserve(item(2), 2L).reservation();
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(3), 1L).status());
        assertEquals(2L, state.observedRequestCount());
        assertEquals(0L, state.availableRequestSlots(1L));
        assertEquals(3L, state.availableRequestSlots(5L));
        assertEquals(Long.MAX_VALUE, state.availableRequestSlots(0L));
        first.close();
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(3), 1L).status());
        second.close();
        assertEquals(1L, state.availableRequestSlots(1L));
        assertNotNull(reserve(item(3), 1L).reservation());
    }

    @Test
    void batchRequestCountDoesNotLimitWaitingAndBatchPermitWaitsForLastMember() {
        ScheduledRequest first = item(1), second = item(2);
        assertTrue(enqueue(first, 0L));
        assertTrue(enqueue(second, 0L));
        var generation = new EndpointGenerationLifecycle(() -> { });
        var reservation = state.reserveBatch(first, 10L, 1, generation.tryAcquireHandoff()).reservation();
        try (var handoff = reservation.commit(List.of(first, second), 0L)) { }
        assertFalse(state.batchAvailability(1).isAvailable());
        assertTrue(enqueue(item(3), 0L));
        assertTrue(state.terminalizeCommittedItem(first));
        assertEquals(2L, state.observedRequestCount());
        assertFalse(state.batchAvailability(1).isAvailable());
        assertTrue(state.terminalizeCommittedItem(second));
        assertEquals(1L, state.observedRequestCount());
        assertTrue(state.batchAvailability(1).isAvailable());
    }

    @Test
    void staleOwnerCannotReleaseAReusedRequestId() {
        ScheduledRequest previous = item(1);
        var old = reserve(previous, 1L).reservation();
        old.close();
        var current = reserve(item(1), 1L).reservation();
        old.close();
        assertFalse(state.terminalizeCommittedItem(previous));
        assertEquals(1L, state.observedRequestCount());
        current.close();
        assertEquals(0L, state.observedRequestCount());
    }

    @Test
    void immediateAdmissionChecksCurrentCapacityAfterOtherOwnershipChanges() {
        assertTrue(state.canAcceptRequest(2L));
        assertTrue(enqueue(item(1), 2L));
        assertNotNull(reserve(item(2), 2L).reservation(),
                "an intervening ownership mutation does not invalidate remaining capacity");
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(3), 2L).status(),
                "the old available snapshot cannot authorize overselling current capacity");
    }

    @Test
    void advisoryReadersDoNotWaitForTheOwnershipLock() throws Exception {
        ScheduledRequest queued = item(1);
        assertTrue(enqueue(queued, 2L));
        try (var reader = Executors.newSingleThreadExecutor()) {
            lock.lock();
            try {
                assertFalse(reader.submit(() -> state.canAcceptRequest(1L)).get(5, TimeUnit.SECONDS));
                assertTrue(reader.submit(() -> state.canAcceptRequest(2L)).get(5, TimeUnit.SECONDS));
                assertTrue(state.terminalizeActiveUnderLock(queued));
                assertTrue(reader.submit(() -> state.canAcceptRequest(1L)).get(5, TimeUnit.SECONDS),
                        "removal publishes capacity before the ownership lock is released");
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
            List<Future<PrefillState.ReservationResult<PrefillState.RouteReservation>>> attempts = new ArrayList<>();
            for (int i = 0; i < contenders; i++) {
                ScheduledRequest request = item(i + 1);
                attempts.add(executor.submit(() -> {
                    assertTrue(state.canAcceptRequest(1L));
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
            List<PrefillState.RouteReservation> acquired = new ArrayList<>();
            for (var attempt : attempts) {
                var result = attempt.get(5, TimeUnit.SECONDS);
                if (result.reservation() != null) {
                    acquired.add(result.reservation());
                } else {
                    assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, result.status());
                }
            }
            assertEquals(1, acquired.size());
            assertFalse(state.canAcceptRequest(1L));
            acquired.getFirst().close();
            assertTrue(state.canAcceptRequest(1L));
        }
    }

    @Test
    void summaryCountsUnknownWorkWithoutOverflowAndClearsOnRetirement() {
        var reservation = reserve(item(1), 2L).reservation();
        heartbeat(Map.of(), Long.MAX_VALUE);
        assertFalse(state.canAcceptRequest(Long.MAX_VALUE));
        reservation.close();
        assertFalse(state.canAcceptRequest(Long.MAX_VALUE));
        heartbeat(Map.of(), 0L);
        assertTrue(enqueue(item(2), 1L));
        assertFalse(state.canAcceptRequest(1L));
        state.retireGenerationOwnership();
        assertTrue(state.canAcceptRequest(1L));
        assertFalse(state.canAcceptRequest(0L));
        assertFalse(state.canAcceptRequest(-1L));
    }

    @Test
    void failedQueueInsertionDoesNotConsumePublishedCapacity() {
        var direct = new PrefillState(lock, PrefillActiveIndex.disabled(), () -> { });
        lock.lock();
        try {
            assertThrows(IllegalStateException.class, () -> direct.enqueueActiveUnderLock(item(1), 1L));
            assertTrue(direct.canAcceptRequest(1L));
            assertEquals(0L, direct.observedRequestCount());
        } finally {
            lock.unlock();
        }
    }

    @Test
    void fullStatusPublishesUnknownCapacityBeforeEndpointNotification() {
        var observation = mock(WorkerStatus.StatusObservation.class);
        var engine = mock(WorkerStatus.EngineObservation.class);
        when(engine.runningTaskList()).thenReturn(Map.of());
        when(engine.waitingQueryLen()).thenReturn(2L);
        when(observation.engine()).thenReturn(engine);
        when(observation.finishedTasks()).thenReturn(Map.of());
        var result = state.reconcileWorkerStatus(observation, unused -> 0L, () -> {
            assertFalse(state.canAcceptRequest(2L));
            assertTrue(state.canAcceptRequest(3L));
        }, () -> { });
        assertNull(result.publicationFailure());
        heartbeat(Map.of(), 1L);
        assertTrue(state.canAcceptRequest(2L));
    }

    private PrefillState.ReservationResult<PrefillState.RouteReservation> reserve(ScheduledRequest item, long limit) {
        return state.reserveUnqueuedRoute(item, 0L, limit);
    }

    private PrefillState.CommittedHandoff commit(ScheduledRequest item, PrefillState.RouteReservation registration) {
        return state.commitRouteGroup(List.of(item), List.of(registration),
                new EndpointGenerationLifecycle(() -> { }).tryAcquireHandoff());
    }

    private boolean enqueue(ScheduledRequest item, long limit) {
        lock.lock();
        try { return state.enqueueActiveUnderLock(item, limit); }
        finally { lock.unlock(); }
    }

    private void heartbeat(Map<String, WorkerStatus.TaskObservation> tasks, long reportedActive) {
        var observation = mock(WorkerStatus.StatusObservation.class);
        var engine = mock(WorkerStatus.EngineObservation.class);
        when(engine.runningTaskList()).thenReturn(tasks);
        when(engine.waitingQueryLen()).thenReturn(reportedActive);
        when(observation.engine()).thenReturn(engine);
        when(observation.runningTasks()).thenReturn(tasks);
        state.reconcileHeartbeat(observation);
    }

    private static WorkerStatus.TaskObservation observed(long id) {
        return new WorkerStatus.TaskObservation(id, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, "", 0L,
                TaskPhase.RUNNING, 0L, PriorityPreemptionProgress.NONE);
    }

    private static ScheduledRequest item(long id) {
        ScheduledRequest item = mock(ScheduledRequest.class);
        when(item.requestId()).thenReturn(id);
        return item;
    }
}
