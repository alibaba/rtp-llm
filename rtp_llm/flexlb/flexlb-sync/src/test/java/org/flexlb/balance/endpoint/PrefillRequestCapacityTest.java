package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.flexlb.balance.endpoint.EndpointTestSupport.withStateLock;
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
 * Checks Prefill capacity combines local requests with reported engine work without double counting.
 * Concurrent admission and exact release must preserve the configured limits.
 */
class PrefillRequestCapacityTest {
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillEndpoint state =
            org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                    lock, PrefillRequestQueue.fifo(16), System::currentTimeMillis);

    @Test
    void duplicateDirectAdmissionDoesNotReplaceOrReleaseTheOriginalOwner() {
        RouteResult route = item(1);
        var original = reserve(route, 1L).value();
        assertNotNull(original);
        try {
            var duplicate = reserve(route, 1L);
            assertFalse(duplicate.accepted());
            assertSame(CapacityBoundary.OWNERSHIP_LOST, duplicate.boundary());
            assertEquals(
                    1L,
                    org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
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
        RouteResult queued = item(1), immediate = item(2), another = item(3);
        assertTrue(enqueue(queued, 2L));
        var first = reserve(immediate, 2L).value();
        assertNotNull(first);
        var second = reserve(another, 2L).value();
        assertNotNull(second);
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(4), 2L).boundary().resource());
        {
            var handoff = commit(immediate, first);
        }
        EndpointTestSupport.rollback(first);
        assertFalse(state.requestCapacityAvailable(2, 1), "committed execution retains its slot");
        assertTrue(EndpointTestSupport.releaseRequest(state, immediate));
        assertFalse(EndpointTestSupport.releaseRequest(state, immediate));
        assertTrue(state.requestCapacityAvailable(2, 1));
        EndpointTestSupport.rollback(second);
        assertTrue(state.requestCapacityAvailable(2, 2));
        assertEquals(
                1L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
    }

    @Test
    void workerDetailsDeduplicateLocalRequestsAndForeignRequestIds() {
        RouteResult local = item(1);
        var registration = reserve(local, 4L).value();
        var localObservation = observed(1);
        var foreign = observed(2);
        heartbeat(Map.of("local", localObservation, "foreign", foreign, "duplicate", foreign), 2L);
        assertEquals(
                2L,
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state),
                "early Engine observation and prepared local ownership are one request");
        {
            var handoff = commit(local, registration);
        }
        EndpointTestSupport.rollback(registration);
        heartbeat(Map.of("local", localObservation, "foreign", foreign), 2L);
        assertEquals(
                2L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 2L);
        heartbeat(Map.of("local", localObservation), 1L);
        assertEquals(
                1L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 2L);
    }

    @Test
    void scalarEngineWorkIsCountedWithoutInventingIdentity() {
        var registration = reserve(item(1), 4L).value();
        heartbeat(Map.of(), 2L);
        assertEquals(
                3L,
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state),
                "unseen local shadow cannot explain unidentified Engine work");
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(2), 3L).boundary().resource());
        EndpointTestSupport.rollback(registration);
        assertEquals(
                2L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
    }

    @Test
    void changingLimitCannotReleaseExistingOwners() {
        var first = reserve(item(1), 2L).value();
        var second = reserve(item(2), 2L).value();
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(3), 1L).boundary().resource());
        assertEquals(
                2L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 1L);
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 5L);
        EndpointTestSupport.rollback(first);
        assertEquals(
                CapacityBoundary.Resource.PREFILL_REQUEST,
                reserve(item(3), 1L).boundary().resource());
        EndpointTestSupport.rollback(second);
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 1L);
        assertNotNull(reserve(item(3), 1L).value());
    }

    @Test
    void batchRequestCountDoesNotLimitWaitingAndBatchPermitWaitsForLastMember() {
        RouteResult first = item(1), second = item(2);
        assertTrue(enqueue(first, 0L));
        assertTrue(enqueue(second, 0L));
        var generation = new EndpointRetirementTestSupport(() -> {});
        var reservation = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(10L, 1);
        {
            var handoff =
                    EndpointTestSupport.commitBatch(state, reservation, List.of(first, second), 0L);
        }
        assertFalse(state.batchCapacityAvailable(1));
        assertTrue(enqueue(item(3), 0L));
        assertTrue(EndpointTestSupport.releaseRequest(state, first));
        assertEquals(
                2L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertFalse(state.batchCapacityAvailable(1));
        assertTrue(EndpointTestSupport.releaseRequest(state, second));
        assertEquals(
                1L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        assertTrue(state.batchCapacityAvailable(1));
    }

    @Test
    void staleOwnerCannotReleaseAReusedRequestId() {
        RouteResult previous = item(1);
        var old = reserve(previous, 1L).value();
        EndpointTestSupport.rollback(old);
        var current = reserve(item(1), 1L).value();
        EndpointTestSupport.rollback(old);
        assertFalse(EndpointTestSupport.releaseRequest(state, previous));
        assertEquals(
                1L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
        EndpointTestSupport.rollback(current);
        assertEquals(
                0L, org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state));
    }

    @Test
    void immediateAdmissionChecksCurrentCapacityAfterOtherOwnershipChanges() {
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 2L);
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
    void queueDepthReflectsRemovalAfterTheStateLockIsReleased() throws Exception {
        RouteResult queued = item(1);
        assertTrue(enqueue(queued, 2L));
        assertEquals(1, state.queueDepth());
        try (var reader = Executors.newSingleThreadExecutor()) {
            Future<Integer> depth;
            lock.lock();
            try {
                depth = reader.submit(state::queueDepth);
                assertSame(queued, EndpointTestSupport.removeQueued(state, queued));
            } finally {
                lock.unlock();
            }
            assertEquals(0, depth.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void concurrentAvailableSnapshotsCannotOversellOneSeat() throws Exception {
        int contenders = 8;
        CountDownLatch selected = new CountDownLatch(contenders);
        CountDownLatch admit = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(contenders)) {
            List<Future<CapacityBoundary.Attempt<PrefillEndpoint.RequestRecord>>> attempts =
                    new ArrayList<>();
            for (int i = 0; i < contenders; i++) {
                RouteResult request = item(i + 1);
                attempts.add(
                        executor.submit(
                                () -> {
                                    assertTrue(
                                            org.flexlb.balance.endpoint.EndpointTestSupport
                                                            .occupiedRequestCount(state)
                                                    < 1L);
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
            List<PrefillEndpoint.RequestRecord> acquired = new ArrayList<>();
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
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state)
                            < 1L);
            EndpointTestSupport.rollback(acquired.getFirst());
            assertTrue(
                    org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state)
                            < 1L);
        }
    }

    @Test
    void summaryCountsUnknownWorkWithoutOverflowAndClearsOnRetirement() {
        var reservation = reserve(item(1), 2L).value();
        heartbeat(Map.of(), Long.MAX_VALUE);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state)
                        < Long.MAX_VALUE);
        EndpointTestSupport.rollback(reservation);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state)
                        < Long.MAX_VALUE);
        heartbeat(Map.of(), 0L);
        assertTrue(enqueue(item(2), 1L));
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 1L);
        state.retireResources();
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 1L);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 0L);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < -1L);
    }

    @Test
    void failedQueueInsertionDoesNotConsumePublishedCapacity() {
        var direct =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, PrefillRequestQueue.disabled(), System::currentTimeMillis);
        runWithStateLock(
                lock,
                () -> {
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                            direct, item(1)));
                    assertTrue(
                            org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(
                                            direct)
                                    < 1L);
                    assertEquals(
                            0L,
                            org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(
                                    direct));
                });
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedReplacementInsertionKeepsVictimsAndCapacityForRetry(boolean fatal) {
        RouteResult first = item(1), second = item(2), incoming = item(3);
        when(first.priority()).thenReturn(20);
        when(second.priority()).thenReturn(20);
        when(incoming.priority()).thenReturn(80);
        when(first.priority()).thenReturn(10);
        when(second.priority()).thenReturn(50);
        when(incoming.priority()).thenReturn(90);
        AtomicBoolean failInsertion = new AtomicBoolean(true);
        var queue = org.mockito.Mockito.spy(PrefillRequestQueue.fifo(16));
        org.mockito.Mockito.doAnswer(
                        call -> {
                            if (failInsertion.get()) {
                                if (fatal) throw new AssertionError("queue insertion failed");
                                throw new IllegalStateException("queue insertion failed");
                            }
                            return call.callRealMethod();
                        })
                .when(queue)
                .add(incoming);
        PrefillEndpoint ledger =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, queue, System::currentTimeMillis);
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            first,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, first));
                    assertSame(
                            second,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    ledger, second));
                    Class<? extends Throwable> failureType =
                            fatal ? AssertionError.class : IllegalStateException.class;
                    assertThrows(
                            failureType,
                            () ->
                                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                            ledger, incoming));
                    assertEquals(
                            List.of(first, second), ledger.captureQueue(Integer.MAX_VALUE).items());
                    assertEquals(
                            2L,
                            org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(
                                    ledger));

                    failInsertion.set(false);
                    assertSame(
                            incoming,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    ledger, incoming));
                    assertEquals(
                            List.of(first, second, incoming),
                            ledger.captureQueue(Integer.MAX_VALUE).items());
                    assertEquals(
                            3L,
                            org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(
                                    ledger));
                    assertNull(
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    ledger, item(3)),
                            "a reused id cannot replace its current queue owner");
                    assertTrue(ledger.requestCapacityAvailable(1L, 1));
                });
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "1,0,0,1,0,HIGHER_PRIORITY_AHEAD",
        "1,0,1,0,0,SAME_PRIORITY_AHEAD",
        "1,0,0,0,1,UNSPECIFIED",
        "1,1,0,0,0,RESOURCE_EXHAUSTED",
        "1,0,1,0,1,UNSPECIFIED",
        "2,1,1,1,0,HIGHER_PRIORITY_AHEAD",
        "0,0,0,1,0,RESOURCE_EXHAUSTED"
    })
    void explainsRejectedAdmissionFromCurrentPriorities(
            long limit,
            int lower,
            int same,
            int higher,
            long unknown,
            org.flexlb.dao.loadbalance.AdmissionRejectReason expected) {
        org.springframework.test.util.ReflectionTestUtils.setField(state, "requestLimit", limit);
        int id = 1;
        for (int priority : new int[] {10, 50, 90}) {
            int count = priority == 10 ? lower : priority == 50 ? same : higher;
            for (int i = 0; i < count; i++) {
                RouteResult route = item(id++);
                when(route.priority()).thenReturn(priority);
                assertTrue(enqueue(route, 0L));
            }
        }
        heartbeat(Map.of(), unknown);
        assertEquals(expected, state.explainAdmissionRejection(50));
        assertEquals(
                lower + same + higher, state.queueDepth(), "explaining does not change the queue");
    }

    @Test
    void rejectionExplanationReflectsReleaseAndUnknownEngineWork() {
        org.springframework.test.util.ReflectionTestUtils.setField(state, "requestLimit", 2L);
        RouteResult lower = item(1), same = item(2), higher = item(3);
        when(lower.priority()).thenReturn(10);
        when(same.priority()).thenReturn(50);
        when(higher.priority()).thenReturn(90);
        assertTrue(enqueue(lower, 3L));
        assertTrue(enqueue(same, 3L));
        assertTrue(enqueue(higher, 3L));
        assertEquals(
                org.flexlb.dao.loadbalance.AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
                state.explainAdmissionRejection(50));
        assertTrue(state.releaseRequest(higher));
        assertEquals(
                org.flexlb.dao.loadbalance.AdmissionRejectReason.RESOURCE_EXHAUSTED,
                state.explainAdmissionRejection(50));
        heartbeat(Map.of(), 2L);
        assertEquals(
                org.flexlb.dao.loadbalance.AdmissionRejectReason.UNSPECIFIED,
                state.explainAdmissionRejection(50));
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
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 2L);
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 3L);
        heartbeat(Map.of(), 1L);
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.occupiedRequestCount(state) < 2L);
    }

    private CapacityBoundary.Attempt<PrefillEndpoint.RequestRecord> reserve(
            RouteResult item, long limit) {
        return org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                state, item, 0L, limit);
    }

    private WorkSnapshot commit(RouteResult item, PrefillEndpoint.RequestRecord registration) {
        return org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(state, item);
    }

    private boolean enqueue(RouteResult item, long limit) {
        return withStateLock(
                lock,
                () -> {
                    return org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, item)
                            == item;
                });
    }

    private void heartbeat(Map<String, WorkerStatus.TaskObservation> tasks, long reportedActive) {
        var observation = mock(WorkerStatus.StatusObservation.class);
        var engine = mock(WorkerStatus.EngineObservation.class);
        when(engine.runningTaskList()).thenReturn(tasks);
        when(engine.waitingQueryLen()).thenReturn(reportedActive);
        when(observation.engine()).thenReturn(engine);
        when(observation.activeTasks()).thenReturn(tasks);
        runWithStateLock(
                lock,
                () -> {
                    EndpointTestSupport.reconcile(state, observation, unused -> 0L);
                });
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

    private static RouteResult item(long id) {
        RouteResult item = mock(RouteResult.class);
        when(item.requestId()).thenReturn(String.valueOf(id));
        return item;
    }
}
