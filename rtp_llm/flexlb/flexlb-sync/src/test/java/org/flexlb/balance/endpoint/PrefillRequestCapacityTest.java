package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.prefill.CommittedHandoff;
import org.flexlb.balance.endpoint.prefill.PrefillActiveIndex;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillRequestCapacityTest {
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillState state = new PrefillState(lock,
            PrefillActiveIndex.ordered(16, Comparator.comparing(ScheduledRequest::requestId)), () -> { });

    @Test
    void waitingPreparedAndCommittedRequestsShareOneCount() {
        ScheduledRequest queued = item(1), prepared = item(2);
        assertTrue(enqueue(queued, 2L));
        var registration = reserve(prepared, 2L).reservation();
        assertNotNull(registration);
        assertEquals(2L, state.observedRequestCount());
        assertFalse(enqueue(item(3), 2L));
        assertEquals(ReservationResult.Status.CAPACITY_FULL, reserve(item(3), 2L).status());
        try (var handoff = commit(prepared, registration)) {
            assertEquals(2L, state.observedRequestCount());
        }
        registration.close();
        assertEquals(2L, state.observedRequestCount(), "ACK and capability closure cannot release Engine ownership");
        assertTrue(state.removeCommittedRequest(prepared));
        assertFalse(state.removeCommittedRequest(prepared));
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
        ScheduledRequest local = item(1);
        var registration = reserve(local, 4L).reservation();
        heartbeat(Map.of(), 2L);
        assertEquals(3L, state.observedRequestCount(), "unseen local shadow cannot explain unidentified Engine work");
        assertEquals(ReservationResult.Status.CAPACITY_FULL, reserve(item(2), 3L).status());
        cancel(local, registration);
        assertEquals(2L, state.observedRequestCount());
    }

    @Test
    void changingLimitCannotReleaseExistingOwners() {
        ScheduledRequest firstItem = item(1), secondItem = item(2);
        var first = reserve(firstItem, 2L).reservation();
        var second = reserve(secondItem, 2L).reservation();
        assertEquals(ReservationResult.Status.CAPACITY_FULL, reserve(item(3), 1L).status());
        assertEquals(2L, state.observedRequestCount());
        assertFalse(state.canAcceptRequest(1L));
        assertTrue(state.canAcceptRequest(5L));
        cancel(firstItem, first);
        assertEquals(ReservationResult.Status.CAPACITY_FULL, reserve(item(3), 1L).status());
        cancel(secondItem, second);
        assertTrue(state.canAcceptRequest(1L));
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
        assertTrue(state.removeCommittedRequest(first));
        assertEquals(2L, state.observedRequestCount());
        assertFalse(state.batchAvailability(1).isAvailable());
        assertTrue(state.removeCommittedRequest(second));
        assertEquals(1L, state.observedRequestCount());
        assertTrue(state.batchAvailability(1).isAvailable());
    }

    @Test
    void staleOwnerCannotReleaseAReusedRequestId() {
        ScheduledRequest previous = item(1);
        var old = reserve(previous, 1L).reservation();
        cancel(previous, old);
        ScheduledRequest reused = item(1);
        var current = reserve(reused, 1L).reservation();
        old.close();
        assertFalse(state.removeCommittedRequest(previous));
        assertEquals(1L, state.observedRequestCount());
        cancel(reused, current);
        assertEquals(0L, state.observedRequestCount());
    }

    @Test
    void queuedAdmissionChecksCurrentCapacityAfterOtherOwnershipChanges() {
        assertTrue(state.canAcceptRequest(2L));
        assertTrue(enqueue(item(1), 2L));
        assertNotNull(reserve(item(2), 2L).reservation(),
                "an intervening ownership mutation does not invalidate remaining capacity");
        assertEquals(ReservationResult.Status.CAPACITY_FULL, reserve(item(3), 2L).status(),
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
                assertTrue(state.removeQueuedRequest(queued));
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
            List<ScheduledRequest> requests = new ArrayList<>();
            List<Future<ReservationResult<SingleReservation>>> attempts = new ArrayList<>();
            for (int i = 0; i < contenders; i++) {
                ScheduledRequest request = item(i + 1);
                requests.add(request);
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
            List<SingleReservation> acquired = new ArrayList<>();
            ScheduledRequest acquiredRequest = null;
            for (int i = 0; i < contenders; i++) {
                var result = attempts.get(i).get(5, TimeUnit.SECONDS);
                if (result.reservation() != null) {
                    acquired.add(result.reservation());
                    acquiredRequest = requests.get(i);
                } else {
                    assertEquals(ReservationResult.Status.CAPACITY_FULL, result.status());
                }
            }
            assertEquals(1, acquired.size());
            assertFalse(state.canAcceptRequest(1L));
            cancel(acquiredRequest, acquired.getFirst());
            assertTrue(state.canAcceptRequest(1L));
        }
    }

    @Test
    void summaryCountsUnknownWorkWithoutOverflowAndClearsOnRetirement() {
        ScheduledRequest queued = item(1);
        var reservation = reserve(queued, 2L).reservation();
        heartbeat(Map.of(), Long.MAX_VALUE);
        assertFalse(state.canAcceptRequest(Long.MAX_VALUE));
        cancel(queued, reservation);
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
    void fullStatusPublishesUnknownCapacityBeforeEndpointNotification() {
        var observation = mock(WorkerStatus.StatusObservation.class);
        var engine = mock(WorkerStatus.EngineObservation.class);
        when(engine.runningTaskList()).thenReturn(Map.of());
        when(engine.waitingQueryLen()).thenReturn(2L);
        when(observation.engine()).thenReturn(engine);
        when(observation.finishedTasks()).thenReturn(Map.of());
        var result = state.updateFromWorkerStatus(observation, unused -> 0L, () -> {
            assertFalse(state.canAcceptRequest(2L));
            assertTrue(state.canAcceptRequest(3L));
        }, () -> { });
        assertNull(result.publicationFailure());
        heartbeat(Map.of(), 1L);
        assertTrue(state.canAcceptRequest(2L));
    }

    private ReservationResult<SingleReservation> reserve(ScheduledRequest item, long limit) {
        if (!enqueue(item, limit)) {
            return ReservationResult.fail(ReservationResult.Status.CAPACITY_FULL);
        }
        return state.reserveSingle(item, 0L);
    }

    private void cancel(ScheduledRequest item, SingleReservation reservation) {
        lock.lock();
        try {
            assertTrue(state.removeQueuedSingleRequest(item, reservation));
        } finally {
            lock.unlock();
        }
        reservation.close();
    }

    private CommittedHandoff commit(ScheduledRequest item, SingleReservation registration) {
        return state.commitSingleGroup(List.of(item), List.of(registration),
                new EndpointGenerationLifecycle(() -> { }).tryAcquireHandoff());
    }

    private boolean enqueue(ScheduledRequest item, long limit) {
        lock.lock();
        try {
            return state.enqueueActive(item, limit);
        }
        finally { lock.unlock(); }
    }

    private void heartbeat(Map<String, WorkerStatus.TaskObservation> tasks, long reportedActive) {
        var observation = mock(WorkerStatus.StatusObservation.class);
        var engine = mock(WorkerStatus.EngineObservation.class);
        when(engine.runningTaskList()).thenReturn(tasks);
        when(engine.waitingQueryLen()).thenReturn(reportedActive);
        when(observation.engine()).thenReturn(engine);
        when(observation.activeTasks()).thenReturn(tasks);
        state.updateFromHeartbeat(observation);
    }

    private static WorkerStatus.TaskObservation observed(long id) {
        return new WorkerStatus.TaskObservation(Long.toString(id), 0L, 0L, 0L,
                0L, 0L, 0L, 0L, 0L, "", 0L, TaskPhase.RUNNING, 0L,
                PriorityPreemptionProgress.NONE, mock(WorkerStatus.TaskTelemetry.class));
    }

    private static ScheduledRequest item(long id) {
        ScheduledRequest item = mock(ScheduledRequest.class);
        when(item.requestId()).thenReturn(Long.toString(id));
        return item;
    }
}
