package org.flexlb.balance.endpoint;

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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillRequestCapacityTest {
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillState state = new PrefillState(lock,
            PrefillActiveIndex.ordered(16, Comparator.comparingLong(RequestRoute::requestId)));

    @Test
    void waitingPreparedAndCommittedRequestsShareOneCount() {
        RequestRoute queued = item(1), immediate = item(2);
        assertTrue(enqueue(queued, 2L));
        var registration = reserve(immediate, 2L).reservation();
        assertNotNull(registration);
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(enqueue(item(3), 2L));
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(3), 2L).status());
        try (var handoff = commit(immediate, registration)) {
            assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        }
        EndpointTestSupport.rollback(registration);
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests(), "ACK and capability closure cannot release Engine ownership");
        assertTrue(EndpointTestSupport.releaseRequest(state, immediate));
        assertFalse(EndpointTestSupport.releaseRequest(state, immediate));
        assertTrue(enqueue(item(3), 2L));
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void workerDetailsDeduplicateLocalRequestsAndForeignRequestIds() {
        RequestRoute local = item(1);
        var registration = reserve(local, 4L).reservation();
        var localObservation = observed(1);
        var foreign = observed(2);
        heartbeat(Map.of("local", localObservation, "foreign", foreign, "duplicate", foreign), 2L);
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests(), "early Engine observation and prepared local ownership are one request");
        try (var handoff = commit(local, registration)) { }
        EndpointTestSupport.rollback(registration);
        heartbeat(Map.of("local", localObservation, "foreign", foreign), 2L);
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(state.canAcceptRequest(2L));
        heartbeat(Map.of("local", localObservation), 1L);
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertTrue(state.canAcceptRequest(2L));
    }

    @Test
    void scalarEngineWorkIsCountedWithoutInventingIdentity() {
        var registration = reserve(item(1), 4L).reservation();
        heartbeat(Map.of(), 2L);
        assertEquals(3L, state.admissionSummary(0, 0L).occupiedRequests(), "unseen local shadow cannot explain unidentified Engine work");
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(2), 3L).status());
        EndpointTestSupport.rollback(registration);
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void changingLimitCannotReleaseExistingOwners() {
        var first = reserve(item(1), 2L).reservation();
        var second = reserve(item(2), 2L).reservation();
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(3), 1L).status());
        assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(state.canAcceptRequest(1L));
        assertTrue(state.canAcceptRequest(5L));
        EndpointTestSupport.rollback(first);
        assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, reserve(item(3), 1L).status());
        EndpointTestSupport.rollback(second);
        assertTrue(state.canAcceptRequest(1L));
        assertNotNull(reserve(item(3), 1L).reservation());
    }

    @Test
    void batchRequestCountDoesNotLimitWaitingAndBatchPermitWaitsForLastMember() {
        RequestRoute first = item(1), second = item(2);
        assertTrue(enqueue(first, 0L));
        assertTrue(enqueue(second, 0L));
        var generation = new EndpointGenerationLifecycle(() -> { });
        var reservation = state.reserveBatch(first, 10L, 1, generation.tryAcquireHandoff()).reservation();
        try (var handoff = EndpointTestSupport.commitBatch(state, reservation, List.of(first, second), 0L)) { }
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
        var old = reserve(previous, 1L).reservation();
        EndpointTestSupport.rollback(old);
        var current = reserve(item(1), 1L).reservation();
        EndpointTestSupport.rollback(old);
        assertFalse(EndpointTestSupport.releaseRequest(state, previous));
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        EndpointTestSupport.rollback(current);
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
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
        RequestRoute queued = item(1);
        assertTrue(enqueue(queued, 2L));
        try (var reader = Executors.newSingleThreadExecutor()) {
            lock.lock();
            try {
                assertFalse(reader.submit(() -> state.canAcceptRequest(1L)).get(5, TimeUnit.SECONDS));
                assertTrue(reader.submit(() -> state.canAcceptRequest(2L)).get(5, TimeUnit.SECONDS));
                assertTrue(state.removeQueuedLocked(queued));
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
                RequestRoute request = item(i + 1);
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
            EndpointTestSupport.rollback(acquired.getFirst());
            assertTrue(state.canAcceptRequest(1L));
        }
    }

    @Test
    void summaryCountsUnknownWorkWithoutOverflowAndClearsOnRetirement() {
        var reservation = reserve(item(1), 2L).reservation();
        heartbeat(Map.of(), Long.MAX_VALUE);
        assertFalse(state.canAcceptRequest(Long.MAX_VALUE));
        EndpointTestSupport.rollback(reservation);
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
        var direct = new PrefillState(lock, PrefillActiveIndex.disabled());
        lock.lock();
        try {
            assertThrows(IllegalStateException.class, () -> direct.enqueueActiveLocked(item(1), 1L));
            assertTrue(direct.canAcceptRequest(1L));
            assertEquals(0L, direct.admissionSummary(0, 0L).occupiedRequests());
        } finally {
            lock.unlock();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedReplacementInsertionKeepsVictimsAndCapacityForRetry(boolean fatal) {
        RequestRoute first = item(1), second = item(2), incoming = item(3);
        when(first.priority()).thenReturn(10);
        when(second.priority()).thenReturn(50);
        when(incoming.priority()).thenReturn(90);
        AtomicBoolean failInsertion = new AtomicBoolean(true);
        Comparator<RequestRoute> ordering = (left, right) -> {
            if (failInsertion.get() && (left == incoming || right == incoming)) {
                if (fatal) {
                    throw new AssertionError("queue insertion failed");
                }
                throw new IllegalStateException("queue insertion failed");
            }
            return Long.compare(left.requestId(), right.requestId());
        };
        PrefillState ledger = new PrefillState(lock, PrefillActiveIndex.ordered(16, ordering));
        lock.lock();
        try {
            assertTrue(ledger.enqueueActiveLocked(first, 2L));
            assertTrue(ledger.enqueueActiveLocked(second, 2L));
            assertFalse(ledger.enqueueActiveLocked(incoming, 2L));
            Class<? extends Throwable> failureType = fatal ? AssertionError.class : IllegalStateException.class;
            assertThrows(failureType,
                    () -> ledger.replaceQueuedRoutesLocked(incoming, 2L, List.of(first)));
            assertEquals(List.of(first, second), ledger.captureQueue(Integer.MAX_VALUE).items());
            assertEquals(2L, ledger.admissionSummary(0, 0L).occupiedRequests());

            failInsertion.set(false);
            assertTrue(ledger.replaceQueuedRoutesLocked(incoming, 2L, List.of(first)));
            assertEquals(List.of(second, incoming), ledger.captureQueue(Integer.MAX_VALUE).items());
            assertEquals(2L, ledger.admissionSummary(0, 0L).occupiedRequests());
            assertFalse(ledger.removeQueuedLocked(first), "a displaced owner cannot release the replacement");
            assertFalse(ledger.replaceQueuedRoutesLocked(item(3), 2L, List.of(second)),
                    "a reused request ID cannot replace its current queue owner");
            assertSame(incoming, ledger.captureQueue(Integer.MAX_VALUE).items().getLast());
            assertEquals(2L, ledger.admissionSummary(0, 0L).occupiedRequests());
        } finally {
            lock.unlock();
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"duplicate", "foreign", "missing"})
    void replacementRevalidatesEverySelectedOwnerBeforeAnyMutation(String mismatch) {
        RequestRoute first = item(1), second = item(2), incoming = item(3);
        PrefillState ledger = new PrefillState(lock,
                PrefillActiveIndex.ordered(16, Comparator.comparingLong(RequestRoute::requestId)));
        lock.lock();
        try {
            assertTrue(ledger.enqueueActiveLocked(first, 2L));
            assertTrue(ledger.enqueueActiveLocked(second, 2L));
            List<RequestRoute> victims = switch (mismatch) {
                case "duplicate" -> List.of(first, first);
                case "foreign" -> List.of(first, item(2));
                case "missing" -> List.of(first);
                default -> throw new AssertionError(mismatch);
            };

            assertFalse(ledger.replaceQueuedRoutesLocked(incoming, 1L, victims));
            assertEquals(List.of(first, second), ledger.captureQueue(Integer.MAX_VALUE).items());
            assertEquals(2L, ledger.admissionSummary(0, 0L).occupiedRequests());
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
        assertEquals(new PrefillState.AdmissionSummary(3, 2, 1, 1, 1, 0), captured);
        assertEquals(captured, state.admissionSummary(50, 2L));
        assertEquals(PrefillState.RequestRelease.QUEUED, state.releaseRequest(lower));
        assertEquals(new PrefillState.AdmissionSummary(2, 1, 0, 1, 1, 0), state.admissionSummary(50, 2L));
        heartbeat(Map.of(), 2L);
        assertEquals(new PrefillState.AdmissionSummary(4, 3, 0, 1, 1, 2), state.admissionSummary(50, 2L));
        assertEquals(new PrefillState.AdmissionSummary(3, 2, 1, 1, 1, 0), captured,
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
        assertFalse(state.canAcceptRequest(2L));
        assertTrue(state.canAcceptRequest(3L));
        heartbeat(Map.of(), 1L);
        assertTrue(state.canAcceptRequest(2L));
    }

    private PrefillState.ReservationResult<PrefillState.RouteReservation> reserve(RequestRoute item, long limit) {
        return state.reserveUnqueuedRoute(item, 0L, limit);
    }

    private PrefillState.CommittedHandoff commit(RequestRoute item, PrefillState.RouteReservation registration) {
        return EndpointTestSupport.commitRoutes(state, List.of(item), List.of(registration),
                new EndpointGenerationLifecycle(() -> { }).tryAcquireHandoff());
    }

    private boolean enqueue(RequestRoute item, long limit) {
        lock.lock();
        try { return state.enqueueActiveLocked(item, limit); }
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

    private static RequestRoute item(long id) {
        RequestRoute item = mock(RequestRoute.class);
        when(item.requestId()).thenReturn(id);
        return item;
    }
}
