package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PrefillResourceSnapshotTest {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final PrefillActiveIndex waiting =
            PrefillActiveIndex.ordered(
                    4,
                    Comparator.comparing(
                            org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem::requestId));
    private final PrefillEndpoint state =
            org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                    lock, waiting, clock::get);
    private final EndpointRetirementTestSupport generation =
            new EndpointRetirementTestSupport(() -> {});

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void commitKeepsThePreparedDeliveryAndCloseOnlyReleasesTemporaryOwnership(boolean batch) {
        var route = item(9200L);
        enqueue(route);
        var reservation =
                batch ? state.reserveBatch(92L, 1, generation.tryPinGeneration()).value() : null;
        var delivery =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(route), reservation, new long[] {25L}, null, clock.get());
        try (delivery) {
            lock.lock();
            try {
                assertSame(delivery, state.commitLocked(delivery));
                assertThrows(IllegalStateException.class, () -> state.commitLocked(delivery));
            } finally {
                lock.unlock();
            }
            assertTrue(delivery.committed());
            assertTrue(delivery.generationPin().isOpen());
            assertTrue(state.captureQueue(1).items().isEmpty());
        }
        assertNull(delivery.generationPin());
        assertEquals(1, state.stats().locallyOwnedRequests());
        if (batch) assertFalse(state.batchCapacityAvailable(1));
        assertTrue(state.releaseRequest(route));
        assertTrue(state.batchCapacityAvailable(1));
    }

    @Test
    void cancelledPreparedMemberCannotCommitOrConsumeItsBatchSlot() {
        var route = item(9201L);
        org.mockito.Mockito.when(route.prefillEp()).thenReturn(state);
        enqueue(route);
        var pin = generation.tryPinGeneration();
        var reservation = state.reserveBatch(93L, 1, pin).value();
        var delivery =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(route), reservation, new long[] {25L}, null, clock.get());
        try (delivery) {
            assertTrue(state.releaseRequest(route));
            lock.lock();
            try {
                assertNull(state.commitLocked(delivery));
            } finally {
                lock.unlock();
            }
            assertFalse(delivery.committed());
            assertTrue(pin.isOpen(), "failed commit leaves preparation with its caller");
        }
        assertFalse(pin.isOpen());
        assertTrue(state.batchCapacityAvailable(1));
        assertEquals(0, state.stats().locallyOwnedRequests());
    }

    @Test
    void requestSettlementAndBatchRollbackReleaseTheirOwnResourcesExactlyOnce() {
        var request = item(9100L);
        enqueue(request);
        var pin = generation.tryPinGeneration();
        var reservation = state.reserveBatch(90L, 1, pin).value();
        assertNotNull(reservation);
        try {
            assertSame(request, state.closeQueue().getFirst());
            var released = state.releaseRequest(request);
            assertTrue(released);
            assertFalse(state.releaseRequest(request));
            assertFalse(state.batchCapacityAvailable(1));
            assertTrue(state.rollback(reservation));
            assertFalse(pin.isOpen());
            assertFalse(state.rollback(reservation));
            assertTrue(state.batchCapacityAvailable(1));
            assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        } finally {
            state.rollback(reservation);
            pin.close();
        }
    }

    @Test
    void cancellingPreparedHeadLeavesBatchSlotWithDelivery() {
        var head = item(9101L);
        enqueue(head);
        var pin = generation.tryPinGeneration();
        var reservation = state.reserveBatch(91L, 1, pin).value();
        assertNotNull(reservation);
        try {
            var released = state.releaseRequest(head);
            assertFalse(state.batchCapacityAvailable(1), "delivery still owns its preparation");
            assertTrue(state.rollback(reservation));
            assertFalse(pin.isOpen());
            assertFalse(state.rollback(reservation), "late owner close is idempotent");
            assertTrue(state.batchCapacityAvailable(1));
            assertTrue(waitingItems().isEmpty());
        } finally {
            state.rollback(reservation);
            pin.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationBeforeQueueCommitIsAnOwnershipConflict(boolean batch) {
        var member = item(9102L);
        enqueue(member);
        var pin = generation.tryPinGeneration();
        var reservation = batch ? state.reserveBatch(92L, 1, pin).value() : null;
        var plan =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(member), reservation, new long[] {10L}, null, clock.get());
        var released = state.releaseRequest(member);
        lock.lock();
        try {
            assertNull(state.commitLocked(plan));
            assertEquals(0, state.progress().queueDepth());
        } finally {
            lock.unlock();
            if (reservation != null) {
                state.rollback(reservation);
            }
            pin.close();
        }
    }

    @Test
    void admissionSummaryReflectsExactInputsWithoutChangingEarlierSnapshots() {
        var first = state.admissionSummary(50, 1L);
        assertEquals(first, state.admissionSummary(50, 1L));
        assertEquals(0L, first.occupiedRequests());
        var limited = state.admissionSummary(50, 2L);
        assertEquals(limited, state.admissionSummary(50, 2L));
        RequestRoute route = item(701);
        when(route.priority()).thenReturn(40);
        enqueue(route);
        var occupied = state.admissionSummary(50, 1L);
        assertEquals(1L, occupied.occupiedRequests());
        assertEquals(1L, occupied.requestsToRelease());
        assertEquals(1L, occupied.lowerPriorityRequests());
        assertEquals(occupied, state.admissionSummary(50, 1L));
        assertEquals(1L, state.admissionSummary(30, 1L).higherPriorityRequests());
        state.releaseRequest(route);
        assertEquals(0L, state.admissionSummary(50, 1L).occupiedRequests());
        assertEquals(
                0L,
                first.occupiedRequests(),
                "published summaries remain immutable after mutations");
    }

    @ParameterizedTest
    @CsvSource({"false,false", "false,true", "true,false", "true,true"})
    void selectionCommitIncludesFailedBoundaryAndQueueDepth(
            boolean batch, boolean expiredBoundary) {
        RequestRoute selected = item(1), failed = item(2), waiting = item(3);
        enqueue(selected);
        enqueue(failed);
        enqueue(waiting);
        when(failed.requestExpired(clock.get())).thenReturn(expiredBoundary);
        var pin = generation.tryPinGeneration();
        var reservation = batch ? state.reserveBatch(9L, 1, pin).value() : null;
        lock.lock();
        try (var committed =
                batch
                        ? reservation.owner.commitLocked(
                                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                        List.of(selected),
                                        reservation,
                                        new long[] {10L},
                                        failed,
                                        clock.get()))
                        : state.commitLocked(
                                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                        List.of(selected),
                                        null,
                                        new long[] {10L},
                                        failed,
                                        clock.get()))) {
            assertEquals(!expiredBoundary, committed.removedFailedMember());
            assertEquals(expiredBoundary ? 2 : 1, committed.remainingQueueDepth());
            assertEquals(
                    expiredBoundary ? List.of(failed, waiting) : List.of(waiting), waitingItems());
            assertEquals(
                    3,
                    state.admissionSummary(0, 0L).occupiedRequests(),
                    "removing a failed member from the queue does not settle its resources");
        } finally {
            lock.unlock();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedBoundaryValidationCannotPartiallyCommitSelection(boolean batch) {
        RequestRoute selected = item(1), failed = item(2);
        enqueue(selected);
        enqueue(failed);
        var failure = new IllegalStateException("boundary validation failed");
        when(failed.requestExpired(clock.get())).thenThrow(failure);
        var pin = generation.tryPinGeneration();
        var reservation = batch ? state.reserveBatch(9L, 1, pin).value() : null;
        lock.lock();
        try {
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () -> {
                                if (batch) {
                                    reservation.owner.commitLocked(
                                            org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                                    List.of(selected),
                                                    reservation,
                                                    new long[] {10L},
                                                    failed,
                                                    clock.get()));
                                } else {
                                    state.commitLocked(
                                            org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                                    List.of(selected),
                                                    null,
                                                    new long[] {10L},
                                                    failed,
                                                    clock.get()));
                                }
                            }));
            assertEquals(List.of(selected, failed), waitingItems());
            assertEquals(2, state.admissionSummary(0, 0L).occupiedRequests());
            assertNoWork(state.committedSnapshot(), 1L, 2L);
        } finally {
            lock.unlock();
            if (reservation != null) {
                EndpointTestSupport.preparation(reservation).close();
            } else {
                pin.close();
            }
        }
        assertEquals(0, state.queueCounters().batchSlots());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directCommitOwnsItsLockWhileBatchCommitUsesTheSelectionLock(boolean batch) {
        RequestRoute request = item(1);
        if (batch) {
            enqueue(request);
            {
                var reservation = state.reserveBatch(9L, 1, generation.tryPinGeneration()).value();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    reservation.owner.commitLocked(
                                            org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                                    List.of(request),
                                                    reservation,
                                                    new long[] {10L},
                                                    null,
                                                    System.currentTimeMillis())));
                    assertEquals(List.of(request), state.captureQueue(1).items());
                    assertEquals(1, state.queueCounters().batchSlots());
                    try (var handoff =
                            EndpointTestSupport.commitBatch(
                                    state, reservation, List.of(request), 10L)) {
                        assertTrue(state.captureQueue(1).items().isEmpty());
                    }
                }
            }
        } else {
            {
                var reservation = state.reserveUnqueuedRoute(request, 10L, 1L).value();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                        var handoff =
                                org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                        state, request, reservation)) {
                    assertNotNull(handoff);
                    assertSame(state, handoff.generationPin().endpoint());
                    assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                    assertFalse(lock.isHeldByCurrentThread());
                }
            }
        }
        assertTrue(EndpointTestSupport.releaseRequest(state, request));
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(0, state.queueCounters().batchSlots());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void concurrentCommittedCloseReleasesOnlyItsHandoffAndKeepsRequestOwnership(boolean batch)
            throws Exception {
        AtomicInteger drained = new AtomicInteger();
        PrefillEndpoint state = batch ? this.state : org.mockito.Mockito.spy(this.state);
        if (!batch) {
            org.mockito.Mockito.doAnswer(
                            call -> {
                                drained.incrementAndGet();
                                return null;
                            })
                    .when(state)
                    .continueRetirementAfterHandoff();
        }
        WorkerEndpoint lifecycle =
                batch ? new EndpointRetirementTestSupport(drained::incrementAndGet) : state;
        RequestRoute request = item(1);
        org.flexlb.balance.scheduler.Delivery handoff;
        if (batch) {
            enqueue(request);
            {
                var lease = state.reserveBatch(9L, 1, lifecycle.tryPinGeneration()).value();
                try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                    handoff = EndpointTestSupport.commitBatch(state, lease, List.of(request), 10L);
                }
            }
        } else {
            {
                var lease = state.reserveUnqueuedRoute(request, 10L, Long.MAX_VALUE).value();
                try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                    handoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, request, lease);
                }
            }
        }
        try (handoff;
                var otherHandoff = lifecycle.tryPinGeneration();
                var closers = Executors.newFixedThreadPool(2)) {
            lifecycle.beginRetirement();
            assertFalse(lifecycle.tryStartCleanup());
            CountDownLatch start = new CountDownLatch(1);
            List<Future<?>> closes = new ArrayList<>();
            for (int i = 0; i < 2; i++) {
                closes.add(
                        closers.submit(
                                () -> {
                                    assertTrue(start.await(5, TimeUnit.SECONDS));
                                    handoff.close();
                                    return null;
                                }));
            }
            start.countDown();
            for (Future<?> close : closes) {
                close.get(5, TimeUnit.SECONDS);
            }
            assertEquals(0, drained.get(), "the other handoff still prevents retirement");
            otherHandoff.close();
            handoff.close();
            assertEquals(1, drained.get());
            assertEquals(
                    1L,
                    state.admissionSummary(0, 0L).occupiedRequests(),
                    "closing handoff cannot release committed capacity");
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
            var lease = state.reserveBatch(9L, 1, generation.tryPinGeneration()).value();
            try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                var before = state.captureQueue(4);
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                EndpointTestSupport.commitBatch(
                                        state, lease, List.of(first, first), 10L));
                assertEquals(
                        before,
                        state.captureQueue(4),
                        "invalid input must not detach any ACTIVE member");
                assertEquals(1, state.queueCounters().batchSlots());
                try (var committed =
                        EndpointTestSupport.commitBatch(
                                state, lease, List.of(first, second), 10L)) {
                    assertTrue(state.captureQueue(4).items().isEmpty());
                    assertWork(state.committedSnapshot(), 10L, 1L, 2L);
                    assertEquals(2, state.stats().locallyOwnedRequests());
                }
                assertTrue(EndpointTestSupport.releaseRequest(state, first));
                assertEquals(1, state.queueCounters().batchSlots());
                assertTrue(EndpointTestSupport.releaseRequest(state, second));
                assertEquals(0, state.queueCounters().batchSlots());
            }
        }
    }

    @Test
    void queueCaptureKeepsBoundedOrderAndVersionsAcrossChanges() {
        RequestRoute first = item(1), second = item(2), third = item(3), tail = item(4);
        enqueue(first);
        enqueue(second);
        enqueue(third);

        PrefillEndpoint.QueueSnapshot bounded = state.captureQueue(2);
        assertEquals(List.of(first, second), bounded.items());
        assertSame(first, bounded.head());
        assertEquals(List.of(first, second, third), state.captureQueue(4).items());
        assertEquals(waiting.version(), bounded.queueVersion());
        assertEquals(state.progress().schedulingInputVersion(), bounded.schedulingInputVersion());

        lock.lock();
        try {
            state.schedulingInputsChangedLocked();
        } finally {
            lock.unlock();
        }
        PrefillEndpoint.QueueSnapshot changedInputs = state.captureQueue(2);
        assertEquals(bounded.queueVersion(), changedInputs.queueVersion());
        assertTrue(changedInputs.schedulingInputVersion() > bounded.schedulingInputVersion());
        enqueue(tail);
        PrefillEndpoint.QueueSnapshot changedQueue = state.captureQueue(2);
        assertEquals(List.of(first, second), changedQueue.items());
        assertTrue(changedQueue.queueVersion() > bounded.queueVersion());
        assertEquals(
                List.of(first, second), bounded.items(), "an earlier capture keeps its identities");
        assertThrows(UnsupportedOperationException.class, () -> bounded.items().clear());
    }

    @Test
    void batchCapacityRollbackRestoresAvailability() {
        PrefillEndpoint ledger =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, waiting, clock::get);
        var availability =
                (java.util.function.BooleanSupplier) () -> ledger.batchCapacityAvailable(1);
        var request = item(99L);
        lock.lock();
        try {
            assertSame(
                    request,
                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, request));
        } finally {
            lock.unlock();
        }
        assertTrue(availability.getAsBoolean());
        {
            var lease = ledger.reserveBatch(9L, 1, generation.tryPinGeneration()).value();
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
        PrefillEndpoint ledger =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, waiting, clock::get);
        RuntimeException failure = new IllegalStateException("generation drain failure");
        EndpointRetirementTestSupport retiring =
                new EndpointRetirementTestSupport(
                        () -> {
                            throw failure;
                        });
        RequestRoute request = item(91L);
        if (batch) {
            lock.lock();
            try {
                assertSame(
                        request,
                        org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, request));
            } finally {
                lock.unlock();
            }
        }
        long queuedVersion = waiting.version();
        PrefillEndpoint.Reservation lease =
                batch
                        ? ledger.reserveBatch(9L, 1, retiring.tryPinGeneration()).value()
                        : ledger.reserveUnqueuedRoute(request, 10L, 10L).value();
        if (batch) {
            retiring.beginRetirement();
            assertFalse(retiring.tryStartCleanup());
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () -> EndpointTestSupport.rollback(lease)));
        } else {
            EndpointTestSupport.rollback(lease);
        }
        EndpointTestSupport.rollback(lease);
        lock.lock();
        try {
            assertEquals(0, ledger.queueCounters().batchSlots());
            assertEquals(
                    batch,
                    waiting.contains(request),
                    "batch rollback retains queue ownership; DIRECT rollback releases its seat");
            assertEquals(
                    queuedVersion,
                    waiting.version(),
                    "lease rollback does not change queue membership");
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
            var lease = state.reserveBatch(10, 4, generation.tryPinGeneration()).value();
            try (var preparationLease = EndpointTestSupport.preparation(lease)) {
                try (var probe = generation.tryPinGeneration()) {
                    var rejected = state.reserveBatch(10, 4, probe);
                    assertFalse(rejected.accepted());
                    assertEquals(CapacityBoundary.Status.FAILED, rejected.boundary().status());
                    assertTrue(rejected.boundary().cause() instanceof IllegalStateException);
                    assertTrue(rejected.boundary().cause().getMessage().contains("batch_id=10"));
                }
                try (var handoff =
                        EndpointTestSupport.commitBatch(
                                state, lease, List.of(first, sibling), 30)) {
                    assertEquals(1, state.stats().batchCount());
                }
                assertTrue(EndpointTestSupport.releaseRequest(state, first));
                try (var probe = generation.tryPinGeneration()) {
                    assertEquals(
                            CapacityBoundary.Status.FAILED,
                            state.reserveBatch(10, 4, probe).boundary().status());
                }
                assertTrue(EndpointTestSupport.releaseRequest(state, sibling));
                {
                    var replacement =
                            state.reserveBatch(10, 4, generation.tryPinGeneration()).value();
                    try (var preparationReplacement =
                            EndpointTestSupport.preparation(replacement)) {
                        try (var handoff =
                                EndpointTestSupport.commitBatch(
                                        state, replacement, List.of(next), 20)) {
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
            var batch = state.reserveBatch(1, 4, generation.tryPinGeneration()).value();
            try (var preparationBatch = EndpointTestSupport.preparation(batch);
                    var handoff =
                            EndpointTestSupport.commitBatch(
                                    state, batch, List.of(first, sibling), 30)) {
                assertEquals(2, state.stats().locallyOwnedRequests());
                assertTrue(
                        waiting.version() > beforeCommitVersion,
                        "batch commit invalidates the queue revision");
                assertEquals(List.of(queued), waitingItems());
            }
        }
        {
            var reservation = state.reserveUnqueuedRoute(individual, 20, Long.MAX_VALUE).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                    var handoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, individual, reservation)) {
                assertEquals(3, state.stats().locallyOwnedRequests());
            }
        }
        clock.set(105);
        RequestRoute fresh = item(5);
        {
            var reservation = state.reserveUnqueuedRoute(fresh, 20, Long.MAX_VALUE).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                    var handoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, fresh, reservation)) {
                assertEquals(4, state.stats().locallyOwnedRequests());
            }
        }
        clock.set(110);
        assertEquals(
                1, EndpointTestSupport.evictPrefill(state, 10, id -> id.equals(first.requestId())));
        assertEquals(
                3,
                state.stats().locallyOwnedRequests(),
                "one retained member protects both batch members");
        assertEquals(1, state.stats().batchCount());
        assertEquals(1, state.stats().individuallyOwnedRequests(), "fresh individual survives");
        assertEquals(List.of(queued), waitingItems());
        assertEquals(
                1,
                EndpointTestSupport.evictPrefill(state, 10, ignored -> false),
                "a batch counts once");
        assertEquals(1, state.stats().locallyOwnedRequests());
        clock.set(115);
        assertEquals(1, EndpointTestSupport.evictPrefill(state, 10, ignored -> false));
        assertEquals(
                List.of(queued), waitingItems(), "orphan sweeps never remove queued ownership");
        assertNoWork(state.committedSnapshot(), 1L, 2L, 3L, 5L);
        assertEquals(0, state.stats().batchCount());
        assertEquals(0, state.stats().individuallyOwnedRequests());
    }

    @Test
    void concurrentReadersShareACompleteImmutableMaterialization() throws Exception {
        var second = state.reserveUnqueuedRoute(item(2), 20, Long.MAX_VALUE).value();
        var first = state.reserveUnqueuedRoute(item(1), 10, Long.MAX_VALUE).value();
        var captured = capture();
        try (var executor = Executors.newFixedThreadPool(8)) {
            CountDownLatch start = new CountDownLatch(1);
            List<Future<WorkSnapshot>> readers = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                readers.add(
                        executor.submit(
                                () -> {
                                    assertTrue(start.await(5, TimeUnit.SECONDS));
                                    return captured.work().calculate();
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
            assertNoWork(capture().work().calculate(), 1L, 2L);
        }
    }

    @Test
    void committedWorkChangesReuseActiveMembershipButCaptureBothAtOnePoint() {
        var queued = item(1);
        enqueue(queued);
        var before = capture();
        var reservation = state.reserveUnqueuedRoute(item(2), 20, Long.MAX_VALUE).value();
        var after = capture();
        assertSame(before.active(), after.active());
        assertNoWork(before.work().calculate(), 1L, 2L);
        assertWork(after.work().calculate(), 20L, 2L);
        assertFalse(after.work().calculate().containsRequest("1"));
        enqueue(item(3));
        var added = capture();
        assertNotSame(after.active(), added.active());
        assertSame(after.work(), added.work());
        assertEquals(
                List.of(queued.requestId()),
                before.active().items().stream()
                        .map(org.flexlb.balance.scheduler.RequestRoute::requestId)
                        .toList());
        EndpointTestSupport.rollback(reservation);
        assertSame(added.active(), capture().active());
        assertNoWork(capture().work().calculate(), 2L);
        assertWork(after.work().calculate(), 20L, 2L);
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
        assertEquals(100, original.work().calculate().capturedAtMs());
        assertEquals(90, rebased.work().calculate().capturedAtMs());
        assertEquals(rebased.capturedAtMs(), rebased.work().calculate().capturedAtMs());
        assertWork(original.work().calculate(), 10L, 1L);
        assertWork(rebased.work().calculate(), 10L, 1L);
        assertEquals(10L, original.work().calculate().knownRemainingWorkMsAt(110L));
        assertEquals(10L, rebased.work().calculate().knownRemainingWorkMsAt(110L));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void queuedRouteCommitValidatesEveryCanonicalIdentityBeforeTransferringAny(boolean replaced) {
        RequestRoute first = routeItem(101), second = routeItem(102), replacement = routeItem(102);
        enqueue(first);
        enqueue(second);
        assertTrue(state.releaseRequest(second));
        if (replaced) {
            enqueue(replacement);
        }
        try (var permit = generation.tryPinGeneration()) {
            assertNull(
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(first, second), new long[] {30L, 40L}));
        }
        assertEquals(replaced ? List.of(first, replacement) : List.of(first), waitingItems());
        assertEquals(replaced ? 2L : 1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertNoWork(state.committedSnapshot(), 101L, 102L);
        RequestRoute validSecond = replaced ? replacement : second;
        if (!replaced) {
            enqueue(second);
        }
        try (var handoff =
                EndpointTestSupport.commitQueuedRoutes(
                        state, List.of(first, validSecond), new long[] {30L, 40L})) {
            assertTrue(waitingItems().isEmpty());
            assertEquals(
                    0L, handoff.precedingWork().calculate().totalRemainingWorkMs().orElseThrow());
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
        var request = Request.builder().requestId(String.valueOf(id)).seqLen(100L).build();
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return SchedulingTestConfig.createRoute(context, null, null, null, null, null, null, 100L);
    }

    @Test
    void queuedRouteCommitCannotConsumeAnIndependentBatchReservation() {
        RequestRoute first = item(1), second = item(2);
        enqueue(first);
        enqueue(second);
        var batch = state.reserveBatch(11L, 1, generation.tryPinGeneration()).value();
        try (var preparation = EndpointTestSupport.preparation(batch);
                var committed =
                        EndpointTestSupport.commitQueuedRoutes(
                                state, List.of(first, second), new long[] {30L, 40L})) {
            assertNotNull(committed);
            assertTrue(waitingItems().isEmpty());
            assertEquals(70L, state.committedSnapshot().totalRemainingWorkMs().orElseThrow());
            assertEquals(1, state.queueCounters().batchSlots());
        }
        assertEquals(0, state.queueCounters().batchSlots());
        assertEquals(70L, state.committedSnapshot().totalRemainingWorkMs().orElseThrow());
    }

    @Test
    void releasingQueuedRequestDoesNotConsumeDeliveryReservation() {
        RequestRoute head = item(1);
        enqueue(head);
        var batch = state.reserveBatch(11L, 1, generation.tryPinGeneration()).value();
        try (var preparation = EndpointTestSupport.preparation(batch)) {
            var release = state.releaseRequest(head);
            assertTrue(release);
            assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(1, state.queueCounters().batchSlots());
            assertFalse(state.releaseRequest(head));
        }
        assertEquals(0, state.queueCounters().batchSlots());
    }

    @Test
    void releasingCommittedBatchMemberRetainsItsSiblingsAndSlot() {
        RequestRoute first = item(1), second = item(2);
        commitBatch(List.of(first, second), 100L);
        assertTrue(state.releaseRequest(first));
        assertFalse(state.releaseRequest(first));
        assertWork(state.committedSnapshot(), 100L, 2L);
        assertFalse(state.committedSnapshot().containsRequest("1"));
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(1, state.queueCounters().batchSlots());
        assertTrue(state.releaseRequest(second));
        assertEquals(0, state.queueCounters().batchSlots());
    }

    @Test
    void queuedAndImmediateRoutesUseOneCommitAndTerminalOwnership() {
        RequestRoute queued = item(1), immediate = item(2);
        enqueue(queued);
        var immediateReservation =
                state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).value();
        try (var preparation = EndpointTestSupport.preparation(immediateReservation)) {
            assertEquals(List.of(queued), waitingItems());
            assertWork(state.committedSnapshot(), 40L, 2L);
            assertFalse(state.committedSnapshot().containsRequest("1"));
            try (var handoff =
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(queued), new long[] {30L})) {
                assertTrue(waitingItems().isEmpty());
                assertEquals(
                        40L,
                        handoff.precedingWork().calculate().totalRemainingWorkMs().orElseThrow(),
                        "uncommitted DIRECT work remains visible to the queued route");
            }
            try (var handoff =
                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                            state, immediate, immediateReservation)) {
                assertEquals(
                        30L,
                        handoff.precedingWork().calculate().totalRemainingWorkMs().orElseThrow());
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
            var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                assertTrue(waitingItems().isEmpty());
                assertEquals(
                        40L,
                        capture()
                                .work()
                                .calculate()
                                .knownRemainingWorkMsAt(System.currentTimeMillis()));
            }
        }
        assertNoWork(capture().work().calculate(), 1L);
        assertNoWork(state.committedSnapshot(), 1L);
    }

    @Test
    void leaseRollbackRetainsQueuedWorkButReleasesUnqueuedAdmission() {
        RequestRoute queued = item(1), immediate = item(2);
        enqueue(queued);
        var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).value();
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
        try (var permit = generation.tryPinGeneration()) {
            assertNull(
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(first, second), new long[] {30L, 40L}),
                    "a withdrawn queue member invalidates the selection without committing resources");
        }
        assertEquals(List.of(first), waitingItems());
        assertEquals(
                2L,
                state.admissionSummary(0, 0L).occupiedRequests(),
                "failed validation preserves both canonical request owners");
        assertNoWork(state.committedSnapshot(), 1L, 2L);
        assertTrue(waiting.add(second));
        try (var handoff =
                EndpointTestSupport.commitQueuedRoutes(
                        state, List.of(first, second), new long[] {30L, 40L})) {
            assertTrue(waiting.isEmpty());
        }
    }

    @Test
    void retirementProjectsQueuedPreparedAndCommittedItemsWithoutModeExceptions() {
        RequestRoute queued = item(1), prepared = item(2), committed = item(3);
        enqueue(queued);
        var preparedReservation = state.reserveUnqueuedRoute(prepared, 20L, Long.MAX_VALUE).value();
        var committedReservation =
                state.reserveUnqueuedRoute(committed, 30L, Long.MAX_VALUE).value();
        try (var handoff =
                org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                        state, committed, committedReservation)) {
            assertWork(state.committedSnapshot(), 50L, 2L, 3L);
            assertFalse(state.committedSnapshot().containsRequest("1"));
        }
        var retired = state.retireResources();

        assertEquals(3, retired.ownedItems().size());
        assertTrue(retired.ownedItems().containsAll(List.of(queued, prepared, committed)));
        EndpointTestSupport.rollback(preparedReservation);
        EndpointTestSupport.rollback(committedReservation);
        assertNoWork(state.committedSnapshot(), 1L, 2L, 3L);
        assertTrue(waiting.isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void batchPreparationKeepsItsGenerationUntilRollback(boolean callbackFails) {
        var config = SchedulingTestConfig.newConfig();
        var cleanups = new AtomicInteger();
        var callbackFailure = new IllegalStateException("retirement callback failed");
        var endpoint =
                new PrefillEndpoint(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.8", 8080, 8090),
                        config,
                        config.getDispatcher().getType(),
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        new org.flexlb.balance.scheduler.PlacementAvailability(),
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config),
                        order -> PrefillActiveIndex.ordered(4, order)) {
                    @Override
                    protected void closeEndpoint() {
                        assertFalse(resourceLock().isHeldByCurrentThread());
                        cleanups.incrementAndGet();
                        super.closeEndpoint();
                        if (callbackFails) throw callbackFailure;
                    }
                };
        var reservation = endpoint.reserveBatch(99L, 1).value();
        assertNotNull(reservation);
        try {
            endpoint.close();
            assertEquals(0, cleanups.get(), "open preparation prevents generation cleanup");
            assertEquals(1, endpoint.queueCounters().batchSlots());
            assertTrue(
                    assertThrows(IllegalStateException.class, () -> endpoint.awaitRetirement(10L))
                            .getMessage()
                            .contains("timed out"));
            assertTrue(endpoint.rollback(reservation));
            if (callbackFails) {
                assertSame(
                        callbackFailure,
                        assertThrows(
                                IllegalStateException.class,
                                () -> endpoint.awaitRetirement(3_000L)));
            } else {
                endpoint.awaitRetirement(3_000L);
            }
            assertEquals(1, cleanups.get());
            assertEquals(0, endpoint.queueCounters().batchSlots());
            assertFalse(
                    endpoint.rollback(reservation), "late rollback cannot release the pin twice");
        } finally {
            endpoint.rollback(reservation);
            endpoint.close();
        }
    }

    @Test
    void bypassingGenerationDrainCannotDiscardAnOpenBatch() {
        var request = item(1);
        enqueue(request);
        var reservation = state.reserveBatch(9L, 1).value();
        assertNotNull(reservation);
        try {
            assertThrows(IllegalStateException.class, state::retireResources);
            assertEquals(1, state.queueCounters().batchSlots());
            assertEquals(List.of(request), waitingItems());
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        } finally {
            state.rollback(reservation);
        }
    }

    @Test
    void retirementEmitsOneCompletionForSharedBatchAndRetainsDetachedStopOwner() {
        RequestRoute first = item(1), second = item(2), detached = item(3);
        commitBatch(List.of(first, second), 100L);
        enqueue(detached);
        assertSame(detached, state.closeQueue().getFirst());

        var retired = state.retireResources();

        assertEquals(List.of(first, second, detached), retired.ownedItems());
        assertEquals(1, retired.batchCompletions().size());
        assertEquals(10L, retired.batchCompletions().getFirst().batchId());
        assertFalse(retired.batchCompletions().getFirst().learningEligible());
        assertEquals(0, state.queueCounters().batchSlots());
        assertEquals(0, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(EndpointTestSupport.releaseRequest(state, first));
        assertFalse(EndpointTestSupport.releaseRequest(state, second));
        assertTrue(state.retireResources().batchCompletions().isEmpty());
    }

    @Test
    void cancellationBeforeCommitReleasesAnUnqueuedAdmissionExactlyOnce() {
        RequestRoute immediate = item(1);
        var reservation = state.reserveUnqueuedRoute(immediate, 40L, Long.MAX_VALUE).value();
        assertTrue(state.releaseRequest(immediate));
        assertFalse(state.releaseRequest(immediate));
        EndpointTestSupport.rollback(reservation);
        assertNoWork(state.committedSnapshot(), 1L);
        assertNoWork(capture().work().calculate(), 1L);
    }

    @Test
    void failedDirectCommitDoesNotRetainAGenerationHandoff() {
        var request = item(9001L);
        var other = item(9002L);
        var reservation = state.reserveUnqueuedRoute(request, 10L, 1L).value();
        var before = state.progress();

        assertThrows(
                IllegalStateException.class,
                () ->
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                state, other, reservation));

        assertEquals(before, state.progress());
        assertFalse(lock.isHeldByCurrentThread());
        assertTrue(state.rollback(reservation));
        state.beginRetirement();
        assertTrue(state.tryStartCleanup(), "failed commit must release its private handoff");
    }

    @Test
    void retiredGenerationRejectsDirectCommitWithoutConsumingTheReservation() {
        var request = item(9003L);
        var reservation = state.reserveUnqueuedRoute(request, 10L, 1L).value();
        var before = state.progress();
        state.beginRetirement();

        assertNull(
                org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                        state, request, reservation));

        assertEquals(before, state.progress());
        assertTrue(state.rollback(reservation));
        assertTrue(state.tryStartCleanup());
    }

    @Test
    void foreignPreparationCannotCommitOrRollbackTheSameRequestId() {
        PrefillEndpoint other =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new ReentrantLock(), PrefillActiveIndex.disabled(), clock::get);
        RequestRoute localRequest = item(1), otherRequest = item(1);
        var localReservation = state.reserveUnqueuedRoute(localRequest, 10L, 1L).value();
        var otherReservation = other.reserveUnqueuedRoute(otherRequest, 20L, 1L).value();
        try (var localPreparation = EndpointTestSupport.preparation(localReservation);
                var otherPreparation = EndpointTestSupport.preparation(otherReservation);
                var permit = generation.tryPinGeneration()) {
            assertThrows(IllegalArgumentException.class, () -> other.rollback(localReservation));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    other, otherRequest, localReservation));
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(1L, other.admissionSummary(0, 0L).occupiedRequests());
            try (var handoff =
                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                            other, otherRequest, otherReservation)) {
                assertTrue(other.releaseRequest(otherRequest));
            }
            assertEquals(
                    1L,
                    state.admissionSummary(0, 0L).occupiedRequests(),
                    "foreign settlement cannot release the local owner");
        }
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(0L, other.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void staleRouteCannotCommitOrReleaseReplacementWithTheSameRequestId() {
        RequestRoute first = item(1), replacement = item(1);
        enqueue(first);
        assertTrue(state.releaseRequest(first));
        enqueue(replacement);
        assertFalse(state.releaseRequest(first));
        try (var permit = generation.tryPinGeneration()) {
            assertNull(
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(first), new long[] {999L}));
        }
        assertEquals(List.of(replacement), waitingItems());
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(EndpointTestSupport.releaseRequest(state, first));
    }

    @Test
    void stopCallbackAcknowledgesOwnershipAlreadyReleasedByTerminalCleanup() {
        var ledger =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, waiting, clock::get);
        RequestRoute request = item(1);
        lock.lock();
        try {
            assertSame(
                    request,
                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, request));
        } finally {
            lock.unlock();
        }
        assertSame(request, ledger.closeQueue().getFirst());
        assertTrue(waitingItems().isEmpty());
        assertEquals(
                1L,
                ledger.admissionSummary(0, 0L).occupiedRequests(),
                "stop callback retains a request seat until settlement");
        assertTrue(ledger.releaseRequest(request));
        assertEquals(0L, ledger.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(ledger.releaseRequest(request));
        lock.lock();
        try {
            assertFalse(ledger.releaseRequest(request));
        } finally {
            lock.unlock();
        }
    }

    @Test
    void activeRemovalLeavesOpenBatchLeaseWithPreparingTransaction() {
        AtomicInteger drained = new AtomicInteger();
        var ledger =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, waiting, clock::get);
        var retiring = new EndpointRetirementTestSupport(drained::incrementAndGet);
        RequestRoute request = item(1);
        lock.lock();
        try {
            assertSame(
                    request,
                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(ledger, request));
        } finally {
            lock.unlock();
        }
        var lease = ledger.reserveBatch(9L, 1, retiring.tryPinGeneration()).value();
        retiring.beginRetirement();
        assertFalse(retiring.tryStartCleanup());
        lock.lock();
        try {
            assertSame(
                    request,
                    org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(ledger, request));
            assertTrue(waiting.isEmpty());
            assertEquals(1L, ledger.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(1, ledger.queueCounters().batchSlots());
            assertEquals(0, drained.get());
        } finally {
            lock.unlock();
        }
        assertTrue(ledger.releaseRequest(request));
        assertFalse(ledger.batchCapacityAvailable(1));
        EndpointTestSupport.rollback(lease);
        EndpointTestSupport.rollback(lease);
        assertEquals(1, drained.get());
        lock.lock();
        try {
            assertEquals(0, ledger.queueCounters().batchSlots());
        } finally {
            lock.unlock();
        }
    }

    @Test
    void interleavedImmediateCommitsCaptureOtherWorkAtTheActualCommitBoundary() {
        RequestRoute first = item(1), second = item(2);
        {
            var firstReservation = state.reserveUnqueuedRoute(first, 30L, Long.MAX_VALUE).value();
            var secondReservation = state.reserveUnqueuedRoute(second, 40L, Long.MAX_VALUE).value();
            try (var preparationFirstReservation =
                            EndpointTestSupport.preparation(firstReservation);
                    var preparationSecondReservation =
                            EndpointTestSupport.preparation(secondReservation)) {
                try (var secondHandoff =
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                state, second, secondReservation)) {
                    assertEquals(
                            30L,
                            secondHandoff
                                    .precedingWork()
                                    .calculate()
                                    .totalRemainingWorkMs()
                                    .orElseThrow());
                }
                try (var firstHandoff =
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                state, first, firstReservation)) {
                    assertEquals(
                            40L,
                            firstHandoff
                                    .precedingWork()
                                    .calculate()
                                    .totalRemainingWorkMs()
                                    .orElseThrow(),
                            "the earlier reservation must see work committed before its own commit");
                }
            }
        }
    }

    @Test
    void batchAndIndividualCommitsCaptureTheSamePrecedingTimeline() {
        RequestRoute immediate = item(1), batchMember = item(2), later = item(3);
        {
            var immediateReservation =
                    state.reserveUnqueuedRoute(immediate, 30L, Long.MAX_VALUE).value();
            try (var preparationImmediateReservation =
                            EndpointTestSupport.preparation(immediateReservation);
                    var firstHandoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, immediate, immediateReservation)) {
                assertEquals(
                        0L,
                        firstHandoff
                                .precedingWork()
                                .calculate()
                                .totalRemainingWorkMs()
                                .orElseThrow());
                enqueue(batchMember);
                {
                    var batchReservation =
                            state.reserveBatch(10L, 2, generation.tryPinGeneration()).value();
                    try (var preparationBatchReservation =
                                    EndpointTestSupport.preparation(batchReservation);
                            var batchHandoff =
                                    EndpointTestSupport.commitBatch(
                                            state, batchReservation, List.of(batchMember), 40L)) {
                        assertEquals(
                                30L,
                                batchHandoff
                                        .precedingWork()
                                        .calculate()
                                        .totalRemainingWorkMs()
                                        .orElseThrow());
                    }
                }
                {
                    var laterReservation =
                            state.reserveUnqueuedRoute(later, 50L, Long.MAX_VALUE).value();
                    try (var preparationLaterReservation =
                                    EndpointTestSupport.preparation(laterReservation);
                            var laterHandoff =
                                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                            state, later, laterReservation)) {
                        assertEquals(
                                70L,
                                laterHandoff
                                        .precedingWork()
                                        .calculate()
                                        .totalRemainingWorkMs()
                                        .orElseThrow());
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
        reconcile(
                Map.of(),
                Map.of("81", task(81, TaskPhase.RUNNING, 0, 0)),
                unused -> {
                    throw new AssertionError("an unchanged batch needs no prediction");
                });
        org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot earlier =
                cached ? capture().work() : null;
        clock.set(150L);
        enqueue(incoming);
        {
            var reservation = state.reserveBatch(82L, 2, generation.tryPinGeneration()).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                    var handoff =
                            EndpointTestSupport.commitBatch(
                                    state, reservation, List.of(incoming), 50L)) {
                if (cached) {
                    assertSame(earlier, handoff.precedingWork());
                }
                assertTrue(EndpointTestSupport.releaseRequest(state, preceding));
                assertTrue(EndpointTestSupport.releaseRequest(state, incoming));
                clock.set(190L);
                WorkSnapshot frozen = handoff.precedingWork().calculate();
                assertTrue(frozen.containsRequest("81"));
                assertEquals(250L, frozen.knownRemainingWorkMsAt(150L));
                assertEquals(210L, frozen.knownRemainingWorkMsAt(190L));
                assertFalse(
                        frozen.containsRequest("82"), "the incoming batch is not preceding work");
                assertNoWork(state.committedSnapshot(), 81L, 82L);
                assertEquals(0, state.stats().batchCount());
            }
        }
    }

    @Test
    void localCleanupPreservesRunningBatchWorkAndItsClock() {
        RequestRoute a = item(1), b = item(2), c = item(3);
        commitBatch(List.of(a, b, c), 300L);
        reconcile(
                Map.of(),
                Map.of(
                        "1",
                        task(1, TaskPhase.RUNNING, 0, 0),
                        "2",
                        task(2, TaskPhase.RUNNING, 0, 0),
                        "3",
                        task(3, TaskPhase.RUNNING, 0, 0)),
                unused -> {
                    throw new AssertionError("an unchanged batch needs no prediction");
                });

        clock.set(150);
        assertTrue(EndpointTestSupport.releaseRequest(state, a));
        assertTrue(state.committedSnapshot().containsRequest("2"));
        assertTrue(state.committedSnapshot().containsRequest("3"));
        assertFalse(state.committedSnapshot().containsRequest("1"));
        assertEquals(250L, remainingWork());
        clock.set(190);
        assertTrue(EndpointTestSupport.releaseRequest(state, b));
        assertFalse(EndpointTestSupport.releaseRequest(state, a));
        assertFalse(
                EndpointTestSupport.releaseRequest(state, item(3)),
                "cleanup must match the exact item");
        assertEquals(210L, remainingWork());

        // Returning to QUEUED pauses elapsed-time accounting, but does not undo work already
        // started.
        clock.set(230);
        reconcile(
                Map.of("1", task(1, null, 0, 100)),
                Map.of("3", task(3, TaskPhase.RECEIVED, 0, 0)),
                unused -> {
                    throw new AssertionError("local cancellation cannot shrink running work");
                });
        clock.set(250);
        assertEquals(170L, remainingWork());
        assertEquals(WorkSnapshot.Phase.ENGINE_QUEUED, batchPhase(10L));

        var completed =
                reconcile(
                        Map.of("3", task(3, null, 0, 300)),
                        Map.of(),
                        unused -> {
                            throw new AssertionError("a completed batch needs no prediction");
                        });
        assertEquals(1, completed.batchCompletions().size());
        assertTrue(completed.batchCompletions().getFirst().successfulCompletion());
        assertFalse(
                completed.batchCompletions().getFirst().learningEligible(),
                "local cleanup disqualifies learning even when all remaining Worker results succeed");
        assertNoWork(state.committedSnapshot(), 1L, 2L, 3L);
        assertEquals(0, state.stats().batchCount());
        assertEquals(0L, remainingWork());
    }

    @ParameterizedTest
    @CsvSource({"true,false,500,0", "false,true,500,0", "false,false,0,40", "false,false,500,40"})
    void executionEvidencePreservesBatchPrediction(
            boolean previouslyRunning, boolean currentlyRunning, long errorCode, long executionMs) {
        commitBatch(List.of(item(1), item(2), item(3)), 300L);
        if (previouslyRunning) {
            reconcile(
                    Map.of(),
                    Map.of("2", task(2, TaskPhase.RUNNING, 0, 0)),
                    unused -> {
                        throw new AssertionError("an unchanged batch needs no prediction");
                    });
        }
        clock.set(150);
        Map<String, TaskInfo> active =
                currentlyRunning ? Map.of("2", task(2, TaskPhase.RUNNING, 0, 0)) : Map.of();
        reconcile(
                Map.of("1", task(1, null, errorCode, executionMs)),
                active,
                unused -> {
                    throw new AssertionError("started Prefill must retain its batch prediction");
                });
        long remaining = previouslyRunning ? 250L : 300L;
        assertEquals(remaining, remainingWork());
        clock.set(190);
        long afterRunning = previouslyRunning || currentlyRunning ? remaining - 40L : remaining;
        assertEquals(afterRunning, remainingWork());
        assertFalse(state.committedSnapshot().hasUnknownWork());

        reconcile(
                Map.of("2", task(2, null, 500, 0)),
                Map.of("3", task(3, TaskPhase.RECEIVED, 0, 0)),
                unused -> {
                    throw new AssertionError(
                            "a queued observation cannot make started work repackable");
                });
        clock.set(230);
        assertEquals(afterRunning, remainingWork());
        reconcile(
                Map.of("3", task(3, null, 0, 300)),
                Map.of(),
                unused -> {
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
        reconcile(
                Map.of("1", task(1, null, 500, 0)),
                Map.of("2", task(2, TaskPhase.RECEIVED, 0, 0)),
                survivors -> {
                    assertEquals(List.of(b), survivors);
                    return 200L;
                });
        clock.set(1_200);
        assertEquals(200L, remainingWork(), "unstarted work does not age while queued");
        reconcile(
                Map.of(),
                Map.of("2", task(2, TaskPhase.RUNNING, 0, 0)),
                unused -> {
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
        long version = state.progress().ownershipVersion();
        var failure = new IllegalStateException("prediction unavailable");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(Map.of("1", task(1, null, 500, 0)));
        response.setRunningTaskInfo(Map.of("2", task(2, TaskPhase.RECEIVED, 0, 0)));
        var observation =
                WorkerStatusTestSupport.freezeStatusResponse(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        response);

        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                EndpointTestSupport.reconcile(
                                        state,
                                        observation,
                                        survivors -> {
                                            assertEquals(List.of(b), survivors);
                                            throw failure;
                                        })));

        assertEquals(version, state.progress().ownershipVersion());
        assertSame(before.work(), capture().work());
        assertWork(state.committedSnapshot(), 300L, 1L, 2L);
        assertEquals(300L, remainingWork());
        assertEquals(2, state.stats().locallyOwnedRequests());
        assertEquals(List.of(queued), waitingItems());
        var completed =
                reconcile(
                        Map.of("1", task(1, null, 0L, 200L), "2", task(2, null, 0L, 300L)),
                        Map.of(),
                        unused -> {
                            throw new AssertionError("a completed batch needs no prediction");
                        });
        assertEquals(1, completed.batchCompletions().size());
        assertTrue(
                completed.batchCompletions().getFirst().learningEligible(),
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
        var observation =
                WorkerStatusTestSupport.freezeStatusResponse(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        response);
        PrefillEndpoint.StatusReduction reduction;
        lock.lock();
        try {
            reduction = state.prepareStatusLocked(observation, true);
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
        var completed =
                reconcile(
                        Map.of("1", task(1, null, 0L, 200L), "2", task(2, null, 0L, 300L)),
                        Map.of(),
                        unused -> {
                            throw new AssertionError("a completed batch needs no prediction");
                        });
        assertEquals(1, completed.batchCompletions().size());
        assertTrue(
                completed.batchCompletions().getFirst().learningEligible(),
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
            var reservation = state.reserveBatch(11L, 2, generation.tryPinGeneration()).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                    var handoff =
                            EndpointTestSupport.commitBatch(
                                    state, reservation, List.of(c, d), 400L)) {
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
        response.setFinishedTaskInfo(
                Map.of("1", task(1, null, 0L, 300L), "2", task(2, null, 0L, 300L)));
        response.setRunningTaskInfo(Map.of("3", activeC, "4", activeD));
        var observation =
                WorkerStatusTestSupport.freezeStatusResponse(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        response);
        var outcome =
                EndpointTestSupport.reconcile(
                        state,
                        observation,
                        unused -> {
                            throw new AssertionError(
                                    "completed and unchanged batches need no new prediction");
                        });

        assertEquals(4, outcome.requestStatuses().size());
        assertEquals(
                List.of(1L, 2L),
                outcome.requestStatuses().stream()
                        .filter(
                                requestStatus ->
                                        requestStatus.kind()
                                                == PrefillEndpoint.PrefillRequestStatus.Kind
                                                        .COMPLETED)
                        .map(requestStatus -> Long.parseLong(requestStatus.route().requestId()))
                        .sorted()
                        .toList());
        assertEquals(1, outcome.batchCompletions().size());
        var completion = outcome.batchCompletions().getFirst();
        assertEquals(10L, completion.batchId());
        assertEquals(300L, completion.actualWorkMs());
        assertTrue(completion.successfulCompletion());
        assertTrue(completion.learningEligible());
        assertEquals(1, state.stats().batchCount());
        assertEquals(2, state.stats().locallyOwnedRequests());
        clock.set(250L);
        assertEquals(
                350L,
                remainingWork(),
                "the batch without terminal events still advances to RUNNING");
        WorkSnapshot remaining = state.committedSnapshot();
        assertTrue(remaining.containsRequest("3"));
        assertTrue(remaining.containsRequest("4"));
        assertFalse(remaining.containsRequest("1"));
        assertFalse(remaining.containsRequest("2"));
    }

    @Test
    void localCleanupDoesNotInventExecutionEvidenceOrEnableLearning() {
        RequestRoute first = item(1), rejected = item(2), survivor = item(3);
        commitBatch(List.of(first, rejected, survivor), 300L);
        assertTrue(EndpointTestSupport.releaseRequest(state, first));
        var partial =
                reconcile(
                        Map.of("2", task(2, null, 500L, 0L)),
                        Map.of(),
                        members -> {
                            assertEquals(
                                    List.of(survivor),
                                    members,
                                    "local cleanup cannot imply execution started");
                            return 90L;
                        });
        assertTrue(partial.batchCompletions().isEmpty());
        assertEquals(90L, remainingWork());
        var result =
                reconcile(
                        Map.of("3", task(3, null, 0L, 100L)),
                        Map.of(),
                        unused -> {
                            throw new AssertionError("a completed batch needs no prediction");
                        });
        var completion = result.batchCompletions().getFirst();
        assertEquals(100L, completion.actualWorkMs());
        assertTrue(completion.successfulCompletion());
        assertFalse(
                completion.learningEligible(),
                "local cleanup disqualifies the whole batch from learning");
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void partialSuccessThenFailureCompletesBatchOnceWithoutLearningOrRepacking() {
        commitBatch(List.of(item(1), item(2)), 400L);
        ToLongFunction<List<RequestRoute>> noRepacking =
                survivors -> {
                    throw new AssertionError("execution evidence must prevent repacking");
                };
        var partial =
                reconcile(
                        Map.of("1", task(1, null, 0L, 300L)),
                        Map.of("2", task(2, TaskPhase.RUNNING, 0L, 0L)),
                        noRepacking);
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
            var lease = state.reserveBatch(11L, 2, generation.tryPinGeneration()).value();
            try (var preparationLease = EndpointTestSupport.preparation(lease);
                    var handoff =
                            EndpointTestSupport.commitBatch(
                                    state, lease, List.of(replacement), 100L)) {
                assertEquals(2, state.queueCounters().batchSlots());
            }
        }
        ToLongFunction<List<RequestRoute>> noRepacking =
                unused -> {
                    throw new AssertionError("completed batches do not need reprediction");
                };
        if (workerTerminal) {
            var completion = reconcile(Map.of("2", task(2, null, 0L, 300L)), Map.of(), noRepacking);
            assertEquals(
                    List.of(10L),
                    completion.batchCompletions().stream()
                            .map(PrefillEndpoint.BatchCompletion::batchId)
                            .toList());
        } else {
            assertTrue(EndpointTestSupport.releaseRequest(state, sibling));
        }
        assertEquals(1, state.queueCounters().batchSlots());
        assertEquals(1, state.stats().locallyOwnedRequests());
        assertEquals(List.of(11L), batchIds());
        assertWork(state.committedSnapshot(), 100L, 1L);
        assertFalse(state.committedSnapshot().containsRequest("2"));
        assertFalse(
                EndpointTestSupport.releaseRequest(state, first),
                "old item cannot settle the replacement");
        var stale = reconcile(Map.of("1", task(1, null, 0L, 300L)), Map.of(), noRepacking);
        assertTrue(
                stale.batchCompletions().isEmpty(), "old batch proof cannot settle the new batch");
        assertTrue(stale.requestStatuses().isEmpty());
        assertEquals(1, state.queueCounters().batchSlots());

        TaskInfo terminal = task(1, null, 0L, 100L);
        terminal.setBatchId(11L);
        var completion = reconcile(Map.of("1", terminal), Map.of(), noRepacking);
        assertEquals(
                List.of(11L),
                completion.batchCompletions().stream()
                        .map(PrefillEndpoint.BatchCompletion::batchId)
                        .toList());
        assertEquals(0, state.queueCounters().batchSlots());
        assertEquals(0, state.stats().locallyOwnedRequests());
    }

    @Test
    void staleDirectReservationCannotCommitOrReleaseItsReplacement() {
        RequestRoute old = routeItem(101), replacement = routeItem(101);
        var stale = state.reserveUnqueuedRoute(old, 30L, 10L).value();
        assertTrue(state.rollback(stale));
        var current = state.reserveUnqueuedRoute(replacement, 40L, 10L).value();
        try (var cleanup = EndpointTestSupport.preparation(current);
                var pin = generation.tryPinGeneration()) {
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, replacement, stale));
            assertFalse(state.rollback(stale));
            assertEquals(40L, remainingWork());
            assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
            try (var committed =
                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                            state, replacement, current)) {
                assertEquals(
                        0L,
                        committed.precedingWork().calculate().totalRemainingWorkMs().orElseThrow());
            }
        }
        assertTrue(state.releaseRequest(replacement));
    }

    @Test
    void mismatchedDirectReservationLeavesBothRequestsRetryable() {
        RequestRoute first = routeItem(101), second = routeItem(102);
        var firstReservation = state.reserveUnqueuedRoute(first, 30L, 10L).value();
        var secondReservation = state.reserveUnqueuedRoute(second, 40L, 10L).value();
        try (var firstCleanup = EndpointTestSupport.preparation(firstReservation);
                var secondCleanup = EndpointTestSupport.preparation(secondReservation);
                var firstPin = generation.tryPinGeneration();
                var secondPin = generation.tryPinGeneration()) {
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, second, firstReservation));
            assertEquals(70L, remainingWork());
            try (var committed =
                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                            state, first, firstReservation)) {
                assertEquals(
                        40L,
                        committed.precedingWork().calculate().totalRemainingWorkMs().orElseThrow());
            }
            try (var committed =
                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                            state, second, secondReservation)) {
                assertEquals(
                        30L,
                        committed.precedingWork().calculate().totalRemainingWorkMs().orElseThrow());
            }
        }
        assertTrue(state.releaseRequest(first));
        assertTrue(state.releaseRequest(second));
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void duplicateTerminalReportsMergeBeforeBatchAggregationAndReleaseOnce() {
        commitBatch(List.of(item(1), item(2)), 300L);
        var finished =
                Map.of(
                        "first-success",
                        task(1, null, 0L, 100L),
                        "first-failure",
                        task(1, null, 500L, 700L),
                        "second-success",
                        task(2, null, 0L, 200L));
        ToLongFunction<List<RequestRoute>> noRepacking =
                unused -> {
                    throw new AssertionError("a completed batch needs no prediction");
                };
        var result = reconcile(finished, Map.of(), noRepacking);
        assertEquals(2, result.requestStatuses().size());
        assertEquals(
                1L,
                result.requestStatuses().stream()
                        .filter(
                                status ->
                                        status.kind()
                                                == PrefillEndpoint.PrefillRequestStatus.Kind.FAILED)
                        .count());
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

    @Test
    void saturatedUnknownOccupancyStillRequiresCapacityForANewRequest() {
        var worker = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningQueryLen(Long.MAX_VALUE);
        lock.lock();
        try {
            var prepared =
                    state.prepareStatusLocked(
                            WorkerStatusTestSupport.freezeStatusResponse(worker, response), false);
            assertNotNull(state.commitStatusLocked(prepared, Map.of()));
            assertEquals(Long.MAX_VALUE, state.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(Long.MAX_VALUE, state.admissionSummary(0, 1L).requestsToRelease());
            assertEquals(1L, state.admissionSummary(0, Long.MAX_VALUE).requestsToRelease());
            assertNotNull(org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, item(9L)));
            assertFalse(state.requestCapacityAvailable(Long.MAX_VALUE, 1));
            assertEquals(Long.MAX_VALUE, state.admissionSummary(0, 0L).occupiedRequests());
        } finally {
            lock.unlock();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preparedCapacityReleaseTracksUnknownOwnershipAndRejectsStaleCounts(boolean invalidate) {
        var worker = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningQueryLen(4L);
        lock.lock();
        try {
            state.commitStatusLocked(
                    state.prepareStatusLocked(
                            WorkerStatusTestSupport.freezeStatusResponse(worker, response), false),
                    Map.of());
        } finally {
            lock.unlock();
        }
        assertEquals(4L, state.admissionSummary(0, 0L).occupiedRequests());
        response.setRunningQueryLen(2L);
        var observation = WorkerStatusTestSupport.freezeStatusResponse(worker, response);
        lock.lock();
        try {
            var reduction = state.prepareStatusLocked(observation, true);
            assertEquals(
                    4L,
                    state.admissionSummary(0, 0L).occupiedRequests(),
                    "preparation must not release capacity");
            if (invalidate) {
                response.setRunningQueryLen(1L);
                lock.lock();
                try {
                    state.commitStatusLocked(
                            state.prepareStatusLocked(
                                    WorkerStatusTestSupport.freezeStatusResponse(worker, response),
                                    false),
                            Map.of());
                } finally {
                    lock.unlock();
                }
                assertNull(state.commitStatusLocked(reduction, Map.of()));
                assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                reduction = state.prepareStatusLocked(observation, true);
            }
            var result = state.commitStatusLocked(reduction, Map.of());
            assertEquals(
                    !invalidate,
                    result.capacityReleased(),
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
        var observation = WorkerStatusTestSupport.freezeStatusResponse(worker, response);
        WorkerEndpoint.StatusChange.Prefill started;
        lock.lock();
        try {
            started =
                    state.commitStatusLocked(
                            state.prepareStatusLocked(observation, false), java.util.Map.of());
        } finally {
            lock.unlock();
        }
        assertEquals(1, started.requestStatuses().size());
        assertEquals(
                PrefillEndpoint.PrefillRequestStatus.Kind.ACTIVE,
                started.requestStatuses().getFirst().kind());
        assertTrue(started.batchCompletions().isEmpty(), "a heartbeat cannot complete a batch");
        assertFalse(started.capacityReleased());
        assertTrue(started.schedulingInputsChanged());
        long version = state.progress().ownershipVersion();
        clock.set(200L);
        WorkerEndpoint.StatusChange.Prefill renewed;
        lock.lock();
        try {
            renewed =
                    state.commitStatusLocked(
                            state.prepareStatusLocked(observation, false), java.util.Map.of());
        } finally {
            lock.unlock();
        }
        assertFalse(
                renewed.schedulingInputsChanged(),
                "same-phase renewal leaves the projection revision stable");
        assertFalse(renewed.capacityReleased());
        assertEquals(version, state.progress().ownershipVersion());
        assertEquals(
                0L,
                state.stats().maxObservedAgeMs(),
                "same-phase renewal refreshes the activity timestamp");
        assertEquals(
                200L, remainingWork(), "same-phase renewal still advances the execution clock");
        assertWork(state.committedSnapshot(), 200L, 1L, 2L);
        assertEquals(
                2,
                state.stats().locallyOwnedRequests(),
                "the heartbeat's terminal report remains unconsumed");
        var completed =
                reconcile(
                        Map.of("1", task(1, null, 0L, 100L), "2", task(2, null, 0L, 100L)),
                        Map.of(),
                        unused -> {
                            throw new AssertionError("a completed batch needs no prediction");
                        });
        assertTrue(completed.schedulingInputsChanged());
        assertTrue(completed.capacityReleased());
        assertEquals(1, completed.batchCompletions().size());
        assertNoWork(state.committedSnapshot(), 1L, 2L);
        assertEquals(0, state.stats().batchCount());
    }

    private void commitBatch(List<RequestRoute> members, long predictedMs) {
        members.forEach(this::enqueue);
        {
            var reservation = state.reserveBatch(10L, 2, generation.tryPinGeneration()).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                    var handoff =
                            EndpointTestSupport.commitBatch(
                                    state, reservation, members, predictedMs)) {
                assertTrue(waitingItems().isEmpty());
            }
        }
    }

    @Test
    void unchangedWorkerObservationReusesWorkAndProjectionVersion() {
        RequestRoute request = item(1);
        commitBatch(List.of(request), 100L);
        var active = Map.of("1", task(1, TaskPhase.RUNNING, 0L, 0L));
        ToLongFunction<List<RequestRoute>> noRepacking =
                ignored -> {
                    throw new AssertionError("unchanged membership needs no prediction");
                };
        reconcile(Map.of(), active, noRepacking);
        var before = capture();
        clock.addAndGet(20L);
        var result = reconcile(Map.of(), active, noRepacking);
        var after = capture();
        assertEquals(
                1, result.requestStatuses().size(), "activity must still reach the request owner");
        assertFalse(result.capacityReleased());
        assertEquals(before.version(), after.version());
        assertSame(before.work(), after.work());
        assertEquals(
                80L, after.work().calculate().totalRemainingWorkMsAt(clock.get()).orElseThrow());
    }

    private WorkerEndpoint.StatusChange.Prefill reconcile(
            Map<String, TaskInfo> finished,
            Map<String, TaskInfo> active,
            ToLongFunction<List<RequestRoute>> repredictor) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(active);
        var observation =
                WorkerStatusTestSupport.freezeStatusResponse(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        response);
        var outcome = EndpointTestSupport.reconcile(state, observation, repredictor);
        return outcome;
    }

    private static void assertWork(WorkSnapshot snapshot, long expectedMs, long... includedIds) {
        assertFalse(snapshot.hasUnknownWork());
        assertEquals(expectedMs, snapshot.totalRemainingWorkMs().orElseThrow());
        for (long id : includedIds) {
            assertTrue(snapshot.containsRequest(String.valueOf(id)), "missing request " + id);
        }
    }

    private static void assertNoWork(WorkSnapshot snapshot, long... excludedIds) {
        assertWork(snapshot, 0L);
        for (long id : excludedIds) {
            assertFalse(snapshot.containsRequest(String.valueOf(id)), "retained request " + id);
        }
    }

    private List<Long> batchIds() {
        lock.lock();
        try {
            Map<?, ?> batches =
                    (Map<?, ?>)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    state, "batches");
            return batches.keySet().stream().map(id -> (Long) id).sorted().toList();
        } finally {
            lock.unlock();
        }
    }

    private WorkSnapshot.Phase batchPhase(long batchId) {
        lock.lock();
        try {
            Map<?, ?> batches =
                    (Map<?, ?>)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    state, "batches");
            Object batch = batches.get(batchId);
            org.junit.jupiter.api.Assertions.assertNotNull(batch);
            return (WorkSnapshot.Phase)
                    org.springframework.test.util.ReflectionTestUtils.getField(
                            batch, "servicePhase");
        } finally {
            lock.unlock();
        }
    }

    private long remainingWork() {
        return state.committedSnapshot().totalRemainingWorkMsAt(clock.get()).orElseThrow();
    }

    private static TaskInfo task(
            long requestId, TaskPhase phase, long errorCode, long executionMs) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setBatchId(10L);
        task.setPhase(phase);
        task.setErrorCode(errorCode);
        task.setExecutionTimeMs(executionMs);
        return task;
    }

    private static RequestRoute item(long id) {
        var item = mock(RequestRoute.class);
        when(item.requestId()).thenReturn(String.valueOf(id));
        when(item.seqLen()).thenReturn(100L);
        return item;
    }

    private record Snapshot(
            PrefillEndpoint.Progress version,
            long capturedAtMs,
            PrefillRequestQueue.Snapshot active,
            org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot work) {}

    private Snapshot capture() {
        lock.lock();
        try {
            return new Snapshot(
                    state.progress(),
                    clock.get(),
                    waiting.capture(),
                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                            state, "captureWorkLocked", clock.get(), true, null));
        } finally {
            lock.unlock();
        }
    }

    private void enqueue(RequestRoute item) {
        lock.lock();
        try {
            assertSame(item, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, item));
        } finally {
            lock.unlock();
        }
    }

    private List<RequestRoute> waitingItems() {
        return state.captureQueue(Integer.MAX_VALUE).items();
    }
}
