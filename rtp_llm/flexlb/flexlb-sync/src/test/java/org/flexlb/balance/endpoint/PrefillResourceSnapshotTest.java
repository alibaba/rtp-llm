package org.flexlb.balance.endpoint;

import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RouteResult;
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

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.flexlb.balance.endpoint.EndpointTestSupport.withStateLock;
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

/**
 * Checks immutable Prefill snapshots and atomic delivery commits under cancellation, worker feedback and retirement.
 * Stale routes must not commit or release replacement resource records.
 */
class PrefillResourceSnapshotTest {
    private final AtomicLong clock = new AtomicLong(100);
    private final ReentrantLock lock = new ReentrantLock();
    private final OrderedPrefillRequestQueue waiting = OrderedPrefillRequestQueue.fifo(4);
    private final PrefillEndpoint state =
            org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                    lock, waiting, clock::get);
    private final EndpointRetirementTestSupport generation =
            new EndpointRetirementTestSupport(() -> {});

    @Test
    void wholePrefixCapacityFailureDoesNotBecomeSuccessBecauseOneRequestFits() {
        org.springframework.test.util.ReflectionTestUtils.setField(state, "requestLimit", 2L);
        var active = item(9401L);
        var first = item(9402L);
        var second = item(9403L);
        enqueue(active);
        enqueue(first);
        enqueue(second);
        assertTrue(
                state.dequeueForDelivery(
                                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                        List.of(active), null, new long[] {10L}))
                        .accepted());
        var result =
                state.dequeueForDelivery(
                        org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                List.of(first, second), null, new long[] {10L, 10L}));
        assertFalse(result.accepted());
        assertEquals(
                org.flexlb.balance.delivery.CapacityBoundary.Resource.PREFILL_REQUEST,
                result.boundary().resource());
        assertTrue(state.requestCapacityAvailable(2L, 1));
        assertEquals(List.of(first, second), waitingItems());
        assertTrue(state.releaseRequest(active));
        assertEquals(
                org.flexlb.balance.delivery.CapacityBoundary.Resource.PREFILL_REQUEST,
                result.boundary().resource(),
                "later capacity changes cannot rewrite the rejection");
    }

    @Test
    void rollingBackQueuedSelectionKeepsUnchangedCommittedWorkSnapshot() {
        var config = SchedulingTestConfig.newConfig();
        state.enableQueue(
                org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                        config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
        var before = state.captureRouteProjectionInputs();
        var item =
                new org.flexlb.balance.planner.GroupPlanner.Item(
                        "rollback-queued", 50, 1L, 100L, Long.MAX_VALUE, 32L, 0L);
        var registration = state.tryRegisterSelection(item, before, 10L, true);
        assertNotNull(registration.record());
        assertSame(before.work(), state.captureRouteProjectionInputs().work());

        assertTrue(state.rollback(registration.record()));

        var after = state.captureRouteProjectionInputs();
        assertSame(before.work(), after.work());
        assertNoWork(after.work());
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void changingCallerInputsCannotChangeTheSelectedPrefixOrItsWork() {
        var head = item(9331L);
        var tail = item(9332L);
        enqueue(head);
        enqueue(tail);
        var items = new ArrayList<>(List.of(head));
        var predictions = new long[] {25L};
        var selection =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(items, null, predictions);
        items.set(0, tail);
        predictions[0] = 99L;
        selection.predictions()[0] = 100L;

        assertNotNull(state.dequeueForDelivery(selection).value());
        assertFalse(
                state.releaseQueuedRequest(selection.items().getFirst()),
                "late queue cleanup must not release committed delivery");

        assertEquals(List.of(tail), waitingItems());
        assertWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 25L, 9331L);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("9332"));
    }

    @Test
    void deadlineIsCheckedWhenTheEndpointDequeuesRatherThanWhenSelectionWasPrepared() {
        var route = item(9321L);
        enqueue(route);
        var selection =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(route),
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(97L, 1),
                        new long[] {25L});
        clock.incrementAndGet();
        when(route.requestExpired(clock.get())).thenReturn(true);

        assertNull(state.dequeueForDelivery(selection).value());

        assertEquals(List.of(route), waitingItems());
        assertEquals(0, state.queueCounters().batchSlots());
        assertNoWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 9321L);
    }

    @Test
    void rejectingNonPrefixLeavesQueueCapacityAndPredictionsUnchanged() {
        var head = item(9301L);
        var tail = item(9302L);
        enqueue(head);
        enqueue(tail);
        var skippedHead =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(tail),
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(94L, 1),
                        new long[] {25L});

        assertNull(state.dequeueForDelivery(skippedHead).value());

        assertEquals(List.of(head, tail), waitingItems());
        assertEquals(0, state.queueCounters().batchSlots());
        assertNoWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 9301L, 9302L);
        assertEquals(2, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void occupiedBatchSlotRejectsNextPrefixWithoutRemovingItsFailedBoundary() {
        var first = item(9311L);
        var next = item(9312L);
        var failed = item(9313L);
        enqueue(first);
        enqueue(next);
        enqueue(failed);
        assertNotNull(
                state.dequeueForDelivery(
                                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                        List.of(first),
                                        new org.flexlb.balance.scheduler.DeliveryTestSupport
                                                .BatchInput(95L, 1),
                                        new long[] {25L}))
                        .value());
        var nextSelection =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(next),
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(96L, 1),
                        new long[] {30L});

        var rejected = state.dequeueForDelivery(nextSelection);
        assertFalse(rejected.accepted());
        assertEquals(
                org.flexlb.balance.delivery.CapacityBoundary.Resource.PREFILL_BATCH,
                rejected.boundary().resource());
        assertEquals(List.of(next, failed), waitingItems());
        assertEquals(1, state.queueCounters().batchSlots());
        assertWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 25L, 9311L);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("9312"));
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("9313"));

        assertTrue(state.releaseRequest(first));
        assertNotNull(state.dequeueForDelivery(nextSelection).value());
        assertEquals(List.of(failed), waitingItems());
        assertEquals(1, state.queueCounters().batchSlots());
        assertTrue(state.releaseRequest(next));
        assertTrue(state.releaseRequest(failed));
        assertEquals(0, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void registrationUsesSourceVersionsInsteadOfTheMaterializedObjectIdentity() {
        var captured = state.captureRouteProjectionInputs();
        var rebuilt =
                new org.flexlb.balance.projection.RequestTimeEstimation.Inputs(
                        captured.queue(),
                        captured.work(),
                        captured.selectionVersion(),
                        captured.evaluator(),
                        captured.deliveryMode());
        assertNotSame(captured, rebuilt);
        var input =
                new org.flexlb.balance.planner.GroupPlanner.Item(
                        "registration-copy", 50, 1L, 100L, Long.MAX_VALUE, 32L, 0L);
        var registered = state.tryRegisterSelection(input, rebuilt, 10L, true);
        assertEquals(org.flexlb.balance.RoutingResult.Registration.REGISTERED, registered.status());
        assertTrue(state.rollback(registered.record()));
    }

    @Test
    void registrationRejectsChangedSchedulingInputsEvenWithTheSameOwnershipVersion() {
        var captured = state.captureRouteProjectionInputs();
        state.signal(PrefillEndpoint.Signal.INPUT);
        assertEquals(
                captured.ownershipVersion(),
                org.flexlb.balance.endpoint.EndpointTestSupport.versions(state).ownershipVersion());
        var input =
                new org.flexlb.balance.planner.GroupPlanner.Item(
                        "registration-stale", 50, 1L, 100L, Long.MAX_VALUE, 32L, 0L);
        assertEquals(
                org.flexlb.balance.RoutingResult.Registration.CONFLICT,
                state.tryRegisterSelection(input, captured, 10L, true).status());
    }

    @Test
    void retirementSortsRequestsAfterReleasingEndpointLock() {
        var first = item(9901);
        var second = item(9902);
        enqueue(first);
        enqueue(second);
        when(first.enqueueSeq())
                .thenAnswer(
                        call -> {
                            assertFalse(lock.isHeldByCurrentThread());
                            return 2L;
                        });
        when(second.enqueueSeq())
                .thenAnswer(
                        call -> {
                            assertFalse(lock.isHeldByCurrentThread());
                            return 1L;
                        });
        assertEquals(List.of(second, first), state.retireResources().ownedItems());
        assertTrue(state.captureQueue(1).items().isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void dequeueRegistersWorkOnceAndReleaseReturnsCapacity(boolean batch) {
        var route = item(9200L);
        enqueue(route);
        var reservation =
                batch
                        ? new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(92L, 1)
                        : null;
        var delivery =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(route), reservation, new long[] {25L});
        {
            runWithStateLock(
                    lock,
                    () -> {
                        assertNotNull(state.dequeueForDelivery(delivery).value());
                        assertNull(state.dequeueForDelivery(delivery).value());
                    });

            assertFalse(state.isRetired());
            assertTrue(state.captureQueue(1).items().isEmpty());
        }
        assertFalse(state.isRetired());
        assertEquals(1, state.stats().locallyOwnedRequests());
        if (batch) assertFalse(state.batchCapacityAvailable(1));
        assertTrue(state.releaseRequest(route));
        assertTrue(state.batchCapacityAvailable(1));
    }

    @Test
    void cancelledPreparedMemberCannotCommitOrConsumeItsBatchSlot() {
        var route = item(9201L);
        org.mockito.Mockito.when(route.prefillEndpoint()).thenReturn(state);
        enqueue(route);
        var selectedEndpoint = generation;
        var reservation = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(93L, 1);
        var delivery =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(route), reservation, new long[] {25L});
        {
            assertTrue(state.releaseRequest(route));
            runWithStateLock(
                    lock,
                    () -> {
                        assertNull(state.dequeueForDelivery(delivery).value());
                    });

            assertTrue(
                    !selectedEndpoint.isRetired(),
                    "failed commit leaves preparation with its caller");
        }
        assertFalse(selectedEndpoint.isRetired());
        assertTrue(state.batchCapacityAvailable(1));
        assertEquals(0, state.stats().locallyOwnedRequests());
    }

    @Test
    void cancelledBatchCandidateConsumesNoBatchCapacity() {
        var request = item(9100L);
        enqueue(request);
        var plan =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(request),
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(90L, 1),
                        new long[] {10L});
        assertSame(request, state.closeQueue().getFirst());
        assertTrue(state.releaseRequest(request));
        assertFalse(state.releaseRequest(request));
        assertTrue(state.batchCapacityAvailable(1));
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void cancellingPreparedHeadRejectsCommitWithoutOccupyingBatchSlot() {
        var head = item(9101L);
        enqueue(head);
        var plan =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(head),
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(91L, 1),
                        new long[] {10L});
        assertTrue(state.batchCapacityAvailable(1));
        assertTrue(state.releaseRequest(head));
        assertNull(state.dequeueForDelivery(plan).value());
        assertTrue(state.batchCapacityAvailable(1));
        assertTrue(waitingItems().isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancellationBeforeQueueCommitIsAnOwnershipConflict(boolean batch) {
        var member = item(9102L);
        enqueue(member);
        var selectedEndpoint = generation;
        var reservation =
                batch
                        ? new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(92L, 1)
                        : null;
        var plan =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(member), reservation, new long[] {10L});
        var released = state.releaseRequest(member);
        {
            runWithStateLock(
                    lock,
                    () -> {
                        assertNull(state.dequeueForDelivery(plan).value());
                        assertEquals(0, state.queueDepth());
                    });
        }
    }

    @Test
    void admissionSummaryReflectsExactInputsWithoutChangingEarlierSnapshots() {
        var first = state.admissionSummary(50, 1L);
        assertEquals(first, state.admissionSummary(50, 1L));
        assertEquals(0L, first.occupiedRequests());
        var limited = state.admissionSummary(50, 2L);
        assertEquals(limited, state.admissionSummary(50, 2L));
        RouteResult route = item(701);
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
    void dequeueOnlyRemovesSelectedPrefixRegardlessOfUnselectedRequestExpiry(
            boolean batch, boolean expiredBoundary) {
        RouteResult selected = item(1), failed = item(2), waiting = item(3);
        enqueue(selected);
        enqueue(failed);
        enqueue(waiting);
        when(failed.requestExpired(clock.get())).thenReturn(expiredBoundary);
        var selectedEndpoint = generation;
        var reservation =
                batch
                        ? new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(9L, 1)
                        : null;
        {
            var committed =
                    withStateLock(
                            lock,
                            () ->
                                    batch
                                            ? state.dequeueForDelivery(
                                                            org.flexlb.balance.scheduler
                                                                    .DeliveryTestSupport.queued(
                                                                    List.of(selected),
                                                                    reservation,
                                                                    new long[] {10L}))
                                                    .value()
                                            : state.dequeueForDelivery(
                                                            org.flexlb.balance.scheduler
                                                                    .DeliveryTestSupport.queued(
                                                                    List.of(selected),
                                                                    null,
                                                                    new long[] {10L}))
                                                    .value());

            assertEquals(2, state.queueDepth());
            assertEquals(List.of(failed, waiting), waitingItems());
            assertEquals(
                    3,
                    state.admissionSummary(0, 0L).occupiedRequests(),
                    "dequeue registers selected work and retains unselected request registrations");
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void unselectedRequestValidationCannotPreventDeliveringSelectedPrefix(boolean batch) {
        RouteResult selected = item(1), failed = item(2);
        enqueue(selected);
        enqueue(failed);
        when(failed.requestExpired(org.mockito.ArgumentMatchers.anyLong()))
                .thenThrow(new IllegalStateException("unselected request must not be checked"));
        var selection =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(selected),
                        batch
                                ? new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(
                                        9L, 1)
                                : null,
                        new long[] {10L});
        assertNotNull(state.dequeueForDelivery(selection).value());
        assertEquals(List.of(failed), waitingItems());
        assertWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 10L, 1L);
        org.mockito.Mockito.verify(failed, org.mockito.Mockito.never())
                .requestExpired(org.mockito.ArgumentMatchers.anyLong());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directAndQueuedDeliveryAcquireTheirOwnEndpointLock(boolean batch) {
        RouteResult request = item(1);
        if (batch) {
            enqueue(request);
            {
                var reservation =
                        new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(9L, 1);
                {
                    {
                        var handoff =
                                EndpointTestSupport.commitBatch(
                                        state, reservation, List.of(request), 10L);

                        assertTrue(state.captureQueue(1).items().isEmpty());
                    }
                }
            }
        } else {
            {
                var reservation =
                        org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                        state, request, 10L, 1L)
                                .value();
                {
                    var handoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, request);
                    try (var preparationReservation =
                            EndpointTestSupport.preparation(reservation)) {
                        assertNotNull(handoff);
                        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                        assertFalse(lock.isHeldByCurrentThread());
                    }
                }
            }
        }
        assertTrue(EndpointTestSupport.releaseRequest(state, request));
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(0, state.queueCounters().batchSlots());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void concurrentDequeueRegistersThePrefixOnlyOnce(boolean batch) throws Exception {
        RouteResult request = item(1);
        enqueue(request);
        var selection =
                org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                        List.of(request),
                        batch
                                ? new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(
                                        9L, 1)
                                : null,
                        new long[] {10L});
        var start = new CountDownLatch(1);
        try (var callers = Executors.newVirtualThreadPerTaskExecutor()) {
            java.util.concurrent.Callable<WorkSnapshot> dequeue =
                    () -> {
                        start.await();
                        return state.dequeueForDelivery(selection).value();
                    };
            var first = callers.submit(dequeue);
            var second = callers.submit(dequeue);
            start.countDown();
            assertEquals(
                    1,
                    (first.get(5, TimeUnit.SECONDS) == null ? 0 : 1)
                            + (second.get(5, TimeUnit.SECONDS) == null ? 0 : 1));
        }
        assertTrue(waitingItems().isEmpty());
        assertWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 10L, 1L);
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertTrue(state.releaseRequest(request));
        assertFalse(state.releaseRequest(request));
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void duplicateBatchMembersFailBeforeQueueMutationAndAllowCorrectedCandidate() {
        RouteResult first = item(1), second = item(2);
        enqueue(first);
        enqueue(second);
        {
            var lease = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(9L, 1);
            {
                var before = state.captureQueue(4);
                assertNull(
                        EndpointTestSupport.commitBatch(state, lease, List.of(first, first), 10L));
                assertEquals(
                        before,
                        state.captureQueue(4),
                        "invalid input must not detach any ACTIVE member");
                assertEquals(0, state.queueCounters().batchSlots());
                {
                    var committed =
                            EndpointTestSupport.commitBatch(
                                    state, lease, List.of(first, second), 10L);
                    assertTrue(state.captureQueue(4).items().isEmpty());
                    assertWork(
                            org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state),
                            10L,
                            1L,
                            2L);
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
        RouteResult first = item(1), second = item(2), third = item(3), tail = item(4);
        enqueue(first);
        enqueue(second);
        enqueue(third);

        PrefillEndpoint.QueueSnapshot bounded = state.captureQueue(2);
        assertEquals(List.of(first, second), bounded.items());
        assertSame(first, bounded.head());
        assertEquals(List.of(first, second, third), state.captureQueue(4).items());
        assertEquals(waiting.version(), bounded.queueVersion());
        assertEquals(
                org.flexlb.balance.endpoint.EndpointTestSupport.versions(state)
                        .schedulingInputVersion(),
                bounded.schedulingInputVersion());

        state.signal(PrefillEndpoint.Signal.INPUT);
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
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            request,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    ledger, request));
                });
        assertTrue(availability.getAsBoolean());
        {
            var lease = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(9L, 1);
            {
                org.junit.jupiter.api.Assertions.assertNotNull(lease);
                assertTrue(availability.getAsBoolean());
            }
        }
        assertTrue(availability.getAsBoolean());
    }

    @Test
    void rollbackOfDirectRegistrationIsIdempotent() {
        RouteResult request = item(91L);
        var record =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, request, 10L, 10L)
                        .value();
        assertTrue(state.rollback(record));
        assertFalse(state.rollback(record));
        assertEquals(0, state.queueCounters().batchSlots());
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertTrue(waitingItems().isEmpty());
    }

    @ValueSource(booleans = {false, true})
    void rollbackIsIdempotentAndPreservesWaitingBatchMembers(boolean batch) {
        RouteResult request = item(91L);
        Runnable rollback;
        if (batch) {
            enqueue(request);
            var delivery =
                    org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                            List.of(request),
                            new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(9L, 1),
                            new long[] {10L});
            rollback = () -> {};
        } else {
            var record =
                    org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    state, request, 10L, 10L)
                            .value();
            rollback = () -> state.rollback(record);
        }
        rollback.run();
        rollback.run();
        assertEquals(0, state.queueCounters().batchSlots());
        assertEquals(batch ? 1L : 0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(batch ? List.of(request) : List.of(), waitingItems());
    }

    @Test
    void committedBatchIdRemainsInUseUntilItsLastMemberSettles() {
        RouteResult first = item(1), sibling = item(2), next = item(3);
        enqueue(first);
        enqueue(sibling);
        enqueue(next);
        var input = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(10, 4);
        assertNotNull(EndpointTestSupport.commitBatch(state, input, List.of(first, sibling), 30));
        assertThrows(
                IllegalStateException.class,
                () -> EndpointTestSupport.commitBatch(state, input, List.of(next), 20));
        assertTrue(state.releaseRequest(first));
        assertThrows(
                IllegalStateException.class,
                () -> EndpointTestSupport.commitBatch(state, input, List.of(next), 20));
        assertTrue(state.releaseRequest(sibling));
        assertNotNull(EndpointTestSupport.commitBatch(state, input, List.of(next), 20));
        assertEquals(List.of(10L), batchIds());
    }

    @Test
    void orphanSweepSharesTtlAndKeepsEveryMemberOfARetainedBatch() {
        RouteResult first = item(1), sibling = item(2), individual = item(3), queued = item(4);
        enqueue(first);
        enqueue(sibling);
        enqueue(queued);
        long beforeCommitVersion = waiting.version();
        {
            var batch = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(1, 4);
            {
                var handoff =
                        EndpointTestSupport.commitBatch(state, batch, List.of(first, sibling), 30);
                {
                    assertEquals(2, state.stats().locallyOwnedRequests());
                    assertTrue(
                            waiting.version() > beforeCommitVersion,
                            "batch commit invalidates the queue revision");
                    assertEquals(List.of(queued), waitingItems());
                }
            }
        }
        {
            var reservation =
                    org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    state, individual, 20, Long.MAX_VALUE)
                            .value();
            {
                var handoff =
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                state, individual);
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertEquals(3, state.stats().locallyOwnedRequests());
                }
            }
        }
        clock.set(105);
        RouteResult fresh = item(5);
        {
            var reservation =
                    org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    state, fresh, 20, Long.MAX_VALUE)
                            .value();
            {
                var handoff =
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(state, fresh);
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertEquals(4, state.stats().locallyOwnedRequests());
                }
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
        assertNoWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state),
                1L,
                2L,
                3L,
                5L);
        assertEquals(0, state.stats().batchCount());
        assertEquals(0, state.stats().individuallyOwnedRequests());
    }

    @Test
    void concurrentReadersShareACompleteImmutableMaterialization() throws Exception {
        var second =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, item(2), 20, Long.MAX_VALUE)
                        .value();
        var first =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, item(1), 10, Long.MAX_VALUE)
                        .value();
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
        var reservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, item(2), 20, Long.MAX_VALUE)
                        .value();
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
                        .map(org.flexlb.balance.scheduler.RouteResult::requestId)
                        .toList());
        EndpointTestSupport.rollback(reservation);
        assertSame(added.active(), capture().active());
        assertNoWork(capture().work().calculate(), 2L);
        assertWork(after.work().calculate(), 20L, 2L);
    }

    @Test
    void clockRollbackRecapturesWorkWithoutMutatingEarlierSnapshots() {
        org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                state, item(1), 10, Long.MAX_VALUE);
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
        RouteResult first = routeItem(101), second = routeItem(102), replacement = routeItem(102);
        enqueue(first);
        enqueue(second);
        assertTrue(state.releaseRequest(second));
        if (replaced) {
            enqueue(replacement);
        }
        {
            var permit = generation;

            assertNull(
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(first, second), new long[] {30L, 40L}));
        }
        assertEquals(replaced ? List.of(first, replacement) : List.of(first), waitingItems());
        assertEquals(replaced ? 2L : 1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertNoWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 101L, 102L);
        RouteResult validSecond = replaced ? replacement : second;
        if (!replaced) {
            enqueue(second);
        }
        {
            var handoff =
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(first, validSecond), new long[] {30L, 40L});

            assertTrue(waitingItems().isEmpty());
            assertEquals(0L, handoff.calculate().totalRemainingWorkMs().orElseThrow());
        }
        assertWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state),
                70L,
                101L,
                102L);
        assertTrue(EndpointTestSupport.releaseRequest(state, first));
        assertTrue(EndpointTestSupport.releaseRequest(state, validSecond));
        assertNoWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 101L, 102L);
    }

    @Test
    void routeCommitDoesNotConsumeAnUncommittedBatchCandidate() {
        RouteResult first = item(1), second = item(2);
        enqueue(first);
        enqueue(second);
        var batch = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(11L, 1);
        {
            var committed =
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(first, second), new long[] {30L, 40L});
            assertNotNull(committed);
            assertTrue(waitingItems().isEmpty());
            assertEquals(
                    70L,
                    org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                            .totalRemainingWorkMs()
                            .orElseThrow());
            assertEquals(0, state.queueCounters().batchSlots());
        }
        assertEquals(0, state.queueCounters().batchSlots());
        assertEquals(
                70L,
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .totalRemainingWorkMs()
                        .orElseThrow());
    }

    @Test
    void releasingQueuedRequestLeavesNoBatchCapacityOccupied() {
        RouteResult head = item(1);
        enqueue(head);
        var batch = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(11L, 1);
        {
            var release = state.releaseRequest(head);
            assertTrue(release);
            assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
            assertEquals(0, state.queueCounters().batchSlots());
            assertFalse(state.releaseRequest(head));
        }
        assertEquals(0, state.queueCounters().batchSlots());
    }

    @Test
    void releasingCommittedBatchMemberRetainsItsSiblingsAndSlot() {
        RouteResult first = item(1), second = item(2);
        commitBatch(List.of(first, second), 100L);
        assertTrue(state.releaseRequest(first));
        assertFalse(state.releaseRequest(first));
        assertWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 100L, 2L);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("1"));
        assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(1, state.queueCounters().batchSlots());
        assertTrue(state.releaseRequest(second));
        assertEquals(0, state.queueCounters().batchSlots());
    }

    @Test
    void queuedAndImmediateRoutesUseOneCommitAndTerminalOwnership() {
        RouteResult queued = item(1), immediate = item(2);
        enqueue(queued);
        var immediateReservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, immediate, 40L, Long.MAX_VALUE)
                        .value();
        try (var preparation = EndpointTestSupport.preparation(immediateReservation)) {
            assertEquals(List.of(queued), waitingItems());
            assertWork(
                    org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 40L, 2L);
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                            .containsRequest("1"));
            {
                var handoff =
                        EndpointTestSupport.commitQueuedRoutes(
                                state, List.of(queued), new long[] {30L});

                assertTrue(waitingItems().isEmpty());
                assertEquals(
                        40L,
                        handoff.calculate().totalRemainingWorkMs().orElseThrow(),
                        "uncommitted DIRECT work remains visible to the queued route");
            }
            {
                var handoff =
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                state, immediate);

                assertEquals(30L, handoff.calculate().totalRemainingWorkMs().orElseThrow());
            }
        }
        assertEquals(
                70L,
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .knownRemainingWorkMsAt(clock.get()));
        assertTrue(EndpointTestSupport.releaseRequest(state, queued));
        assertTrue(EndpointTestSupport.releaseRequest(state, immediate));
        assertNoWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L, 2L);
    }

    @Test
    void immediatePreparationIsVisibleToLaterPredictionsAndRollbackRemovesIt() {
        RouteResult immediate = item(1);
        {
            var reservation =
                    org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    state, immediate, 40L, Long.MAX_VALUE)
                            .value();
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
        assertNoWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L);
    }

    @Test
    void leaseRollbackRetainsQueuedWorkButReleasesUnqueuedAdmission() {
        RouteResult queued = item(1), immediate = item(2);
        enqueue(queued);
        var reservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, immediate, 40L, Long.MAX_VALUE)
                        .value();
        EndpointTestSupport.rollback(reservation);
        assertEquals(List.of(queued), waitingItems());
        assertNoWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L, 2L);
    }

    @Test
    void routeCommitRejectsMissingQueueIndexBeforeCommittingAnyMember() {
        RouteResult first = item(1), second = item(2);
        enqueue(first);
        enqueue(second);
        assertTrue(waiting.remove(second));
        {
            var permit = generation;

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
        assertNoWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L, 2L);
        assertTrue(waiting.add(second));
        {
            var handoff =
                    EndpointTestSupport.commitQueuedRoutes(
                            state, List.of(first, second), new long[] {30L, 40L});

            assertTrue(waiting.isEmpty());
        }
    }

    @Test
    void retirementProjectsQueuedPreparedAndCommittedItemsWithoutModeExceptions() {
        RouteResult queued = item(1), prepared = item(2), committed = item(3);
        enqueue(queued);
        var preparedReservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, prepared, 20L, Long.MAX_VALUE)
                        .value();
        var committedReservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, committed, 30L, Long.MAX_VALUE)
                        .value();
        {
            var handoff =
                    org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(state, committed);

            assertWork(
                    org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state),
                    50L,
                    2L,
                    3L);
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                            .containsRequest("1"));
        }
        var retired = state.retireResources();

        assertEquals(3, retired.ownedItems().size());
        assertTrue(retired.ownedItems().containsAll(List.of(queued, prepared, committed)));
        EndpointTestSupport.rollback(preparedReservation);
        EndpointTestSupport.rollback(committedReservation);
        assertNoWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L, 2L, 3L);
        assertTrue(waiting.isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void retirementDiscardsUncommittedBatchAndLateRollbackIsHarmless(boolean callbackFails) {
        var config = SchedulingTestConfig.newConfig();
        var cleanups = new AtomicInteger();
        var callbackFailure = new IllegalStateException("retirement callback failed");
        var endpoint =
                new PrefillEndpoint(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.8", 8080, 8090),
                        config,
                        config.getDispatcher().getType(),
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        new org.flexlb.balance.scheduler.CapacityNotifier(),
                        config.isQueue()
                                ? org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                        config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config))
                                : null,
                        () -> OrderedPrefillRequestQueue.fifo(4)) {
                    @Override
                    protected void closeEndpoint() {
                        assertFalse(stateLock().isHeldByCurrentThread());
                        cleanups.incrementAndGet();
                        super.closeEndpoint();
                        if (callbackFails) throw callbackFailure;
                    }
                };
        var reservation = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(99L, 1);
        assertNotNull(reservation);

        if (callbackFails) {
            assertSame(callbackFailure, assertThrows(IllegalStateException.class, endpoint::close));
            assertSame(
                    callbackFailure,
                    assertThrows(
                            IllegalStateException.class, () -> endpoint.awaitRetirement(3000)));
        } else {
            endpoint.close();
            endpoint.awaitRetirement(3000);
        }
        assertEquals(1, cleanups.get());
        assertEquals(0, endpoint.queueCounters().batchSlots());
    }

    @Test
    void retirementDoesNotReportAnUncommittedBatchAsCompleted() {
        var request = item(1);
        enqueue(request);
        var record = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(9L, 1);
        var retired = state.retireResources();
        assertEquals(List.of(request), retired.ownedItems());
        assertTrue(retired.batchCompletions().isEmpty());
        assertEquals(0, state.queueCounters().batchSlots());
        assertTrue(waitingItems().isEmpty());
    }

    @Test
    void retirementEmitsOneCompletionForSharedBatchAndRetainsDetachedStopOwner() {
        RouteResult first = item(1), second = item(2), detached = item(3);
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
        RouteResult immediate = item(1);
        var reservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, immediate, 40L, Long.MAX_VALUE)
                        .value();
        assertTrue(state.releaseRequest(immediate));
        assertFalse(state.releaseRequest(immediate));
        EndpointTestSupport.rollback(reservation);
        assertNoWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L);
        assertNoWork(capture().work().calculate(), 1L);
    }

    @Test
    void failedDirectCommitDoesNotRetainAGenerationHandoff() {
        var request = item(9001L);
        var other = item(9002L);
        var reservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, request, 10L, 1L)
                        .value();
        var before = org.flexlb.balance.endpoint.EndpointTestSupport.versions(state);

        assertNull(state.commitDirect(other));

        assertEquals(before, org.flexlb.balance.endpoint.EndpointTestSupport.versions(state));
        assertFalse(lock.isHeldByCurrentThread());
        assertTrue(state.rollback(reservation));
        state.beginRetirement();
        state.close();
        state.awaitRetirement(2000);
    }

    @Test
    void retiredGenerationRejectsDirectCommitWithoutConsumingTheReservation() {
        var request = item(9003L);
        var reservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, request, 10L, 1L)
                        .value();
        var before = org.flexlb.balance.endpoint.EndpointTestSupport.versions(state);
        state.beginRetirement();

        assertNull(org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(state, request));

        assertEquals(before, org.flexlb.balance.endpoint.EndpointTestSupport.versions(state));
        assertTrue(state.rollback(reservation));
        state.close();
        state.awaitRetirement(2000);
    }

    @Test
    void foreignPreparationCannotCommitOrRollbackTheSameRequestId() {
        PrefillEndpoint other =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        new ReentrantLock(), OrderedPrefillRequestQueue.disabled(), clock::get);
        RouteResult localRequest = item(1), otherRequest = item(1);
        var localReservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, localRequest, 10L, 1L)
                        .value();
        var otherReservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                other, otherRequest, 20L, 1L)
                        .value();
        {
            var permit = generation;
            try (var localPreparation = EndpointTestSupport.preparation(localReservation);
                    var otherPreparation = EndpointTestSupport.preparation(otherReservation)) {
                assertThrows(
                        IllegalArgumentException.class, () -> other.rollback(localReservation));
                assertNull(other.commitDirect(localRequest));
                assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                assertEquals(1L, other.admissionSummary(0, 0L).occupiedRequests());
                {
                    var handoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    other, otherRequest);

                    assertTrue(other.releaseRequest(otherRequest));
                }
                assertEquals(
                        1L,
                        state.admissionSummary(0, 0L).occupiedRequests(),
                        "foreign settlement cannot release the local owner");
            }
        }
        assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
        assertEquals(0L, other.admissionSummary(0, 0L).occupiedRequests());
    }

    @Test
    void staleRouteCannotCommitOrReleaseReplacementWithTheSameRequestId() {
        RouteResult first = item(1), replacement = item(1);
        enqueue(first);
        assertTrue(state.releaseRequest(first));
        enqueue(replacement);
        assertFalse(state.releaseRequest(first));
        {
            var permit = generation;

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
        RouteResult request = item(1);
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            request,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    ledger, request));
                });
        assertSame(request, ledger.closeQueue().getFirst());
        assertTrue(waitingItems().isEmpty());
        assertEquals(
                1L,
                ledger.admissionSummary(0, 0L).occupiedRequests(),
                "stop callback retains a request seat until settlement");
        assertTrue(ledger.releaseRequest(request));
        assertEquals(0L, ledger.admissionSummary(0, 0L).occupiedRequests());
        assertFalse(ledger.releaseRequest(request));
        runWithStateLock(
                lock,
                () -> {
                    assertFalse(ledger.releaseRequest(request));
                });
    }

    @Test
    void removingUncommittedCandidateLeavesNoBatchCapacityOccupied() {
        var ledger =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, waiting, clock::get);
        RouteResult request = item(1);
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            request,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    ledger, request));
                });
        var lease = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(9L, 1);
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            request,
                            org.flexlb.balance.endpoint.EndpointTestSupport.removeQueued(
                                    ledger, request));
                    assertTrue(waiting.isEmpty());
                    assertEquals(1L, ledger.admissionSummary(0, 0L).occupiedRequests());
                    assertEquals(0, ledger.queueCounters().batchSlots());
                });
        assertTrue(ledger.releaseRequest(request));
        assertTrue(ledger.batchCapacityAvailable(1));

        runWithStateLock(
                lock,
                () -> {
                    assertEquals(0, ledger.queueCounters().batchSlots());
                });
    }

    @Test
    void interleavedImmediateCommitsCaptureOtherWorkAtTheActualCommitBoundary() {
        RouteResult first = item(1), second = item(2);
        {
            var firstReservation =
                    org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    state, first, 30L, Long.MAX_VALUE)
                            .value();
            var secondReservation =
                    org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    state, second, 40L, Long.MAX_VALUE)
                            .value();
            try (var preparationFirstReservation =
                            EndpointTestSupport.preparation(firstReservation);
                    var preparationSecondReservation =
                            EndpointTestSupport.preparation(secondReservation)) {
                {
                    var secondHandoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, second);
                    assertEquals(
                            30L, secondHandoff.calculate().totalRemainingWorkMs().orElseThrow());
                }
                {
                    var firstHandoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, first);
                    assertEquals(
                            40L,
                            firstHandoff.calculate().totalRemainingWorkMs().orElseThrow(),
                            "the earlier reservation must see work committed before its own commit");
                }
            }
        }
    }

    @Test
    void batchAndIndividualCommitsCaptureTheSamePrecedingTimeline() {
        RouteResult immediate = item(1), batchMember = item(2), later = item(3);
        {
            var immediateReservation =
                    org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    state, immediate, 30L, Long.MAX_VALUE)
                            .value();
            try (var preparationImmediateReservation =
                    EndpointTestSupport.preparation(immediateReservation)) {
                var firstHandoff =
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                state, immediate);
                assertEquals(0L, firstHandoff.calculate().totalRemainingWorkMs().orElseThrow());

                enqueue(batchMember);
                {
                    var batchReservation =
                            new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(10L, 2);
                    {
                        var batchHandoff =
                                EndpointTestSupport.commitBatch(
                                        state, batchReservation, List.of(batchMember), 40L);
                        assertEquals(
                                30L, batchHandoff.calculate().totalRemainingWorkMs().orElseThrow());
                    }
                }
                {
                    var laterReservation =
                            org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                            state, later, 50L, Long.MAX_VALUE)
                                    .value();
                    try (var preparationLaterReservation =
                            EndpointTestSupport.preparation(laterReservation)) {
                        var laterHandoff =
                                org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                        state, later);
                        assertEquals(
                                70L, laterHandoff.calculate().totalRemainingWorkMs().orElseThrow());
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void batchHandoffFreezesPrecedingWorkAcrossClockAdvanceAndTerminalCleanup(boolean cached) {
        RouteResult preceding = item(81), incoming = item(82);
        commitBatch(List.of(preceding), 300L);
        reconcile(
                Map.of(),
                Map.of("81", task(81, TaskPhase.RUNNING, 0, 0)),
                unused -> {
                    throw new AssertionError("an unchanged batch needs no prediction");
                });
        WorkSnapshot earlier = cached ? capture().work() : null;
        clock.set(150L);
        enqueue(incoming);
        {
            var reservation =
                    new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(82L, 2);
            {
                var handoff =
                        EndpointTestSupport.commitBatch(state, reservation, List.of(incoming), 50L);
                {
                    if (cached) {
                        assertSame(earlier, handoff);
                    }
                    assertTrue(EndpointTestSupport.releaseRequest(state, preceding));
                    assertTrue(EndpointTestSupport.releaseRequest(state, incoming));
                    clock.set(190L);
                    WorkSnapshot frozen = handoff.calculate();
                    assertTrue(frozen.containsRequest("81"));
                    assertEquals(250L, frozen.knownRemainingWorkMsAt(150L));
                    assertEquals(210L, frozen.knownRemainingWorkMsAt(190L));
                    assertFalse(
                            frozen.containsRequest("82"),
                            "the incoming batch is not preceding work");
                    assertNoWork(
                            org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state),
                            81L,
                            82L);
                    assertEquals(0, state.stats().batchCount());
                }
            }
        }
    }

    @Test
    void localCleanupPreservesRunningBatchWorkAndItsClock() {
        RouteResult a = item(1), b = item(2), c = item(3);
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
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("2"));
        assertTrue(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("3"));
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("1"));
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
        assertNoWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L, 2L, 3L);
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
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .hasUnknownWork());

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
        assertNoWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L, 2L, 3L);
        assertEquals(0, state.stats().batchCount());
    }

    @Test
    void onlyWorkerRemovalBeforeExecutionRepredictsTheQueuedBatch() {
        RouteResult a = item(1), b = item(2);
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
        RouteResult a = item(1), b = item(2), queued = item(3);
        commitBatch(List.of(a, b), 300L);
        enqueue(queued);
        var before = capture();
        long version =
                org.flexlb.balance.endpoint.EndpointTestSupport.versions(state).ownershipVersion();
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

        assertEquals(
                version,
                org.flexlb.balance.endpoint.EndpointTestSupport.versions(state).ownershipVersion());
        assertSame(before.work(), capture().work());
        assertWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 300L, 1L, 2L);
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
        RouteResult first = item(1), second = item(2), queued = item(3);
        commitBatch(List.of(first, second), 300L);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setFinishedTaskInfo(Map.of("1", task(1, null, 500L, 0L)));
        response.setRunningTaskInfo(Map.of("2", task(2, TaskPhase.RECEIVED, 0L, 0L)));
        var observation =
                WorkerStatusTestSupport.freezeStatusResponse(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        response);
        var reduction = withStateLock(lock, () -> state.prepareStatusLocked(observation, true));
        assertEquals(Map.of(10L, List.of(second)), reduction.predictionInputs());
        enqueue(queued);
        runWithStateLock(
                lock,
                () -> {
                    assertNull(state.commitStatusLocked(reduction, Map.of(10L, 200L)));
                });
        assertWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 300L, 1L, 2L);
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
        RouteResult a = item(1), b = item(2), c = item(3), d = item(4);
        commitBatch(List.of(a, b), 300L);
        enqueue(c);
        enqueue(d);
        {
            var reservation =
                    new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(11L, 2);
            {
                var handoff =
                        EndpointTestSupport.commitBatch(state, reservation, List.of(c, d), 400L);
                {
                    assertTrue(waitingItems().isEmpty());
                }
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
        WorkSnapshot remaining =
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state);
        assertTrue(remaining.containsRequest("3"));
        assertTrue(remaining.containsRequest("4"));
        assertFalse(remaining.containsRequest("1"));
        assertFalse(remaining.containsRequest("2"));
    }

    @Test
    void localCleanupDoesNotInventExecutionEvidenceOrEnableLearning() {
        RouteResult first = item(1), rejected = item(2), survivor = item(3);
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
        ToLongFunction<List<RouteResult>> noRepacking =
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
        RouteResult first = item(1), sibling = item(2);
        commitBatch(List.of(first, sibling), 300L);
        assertTrue(EndpointTestSupport.releaseRequest(state, first));

        RouteResult replacement = item(1);
        enqueue(replacement);
        {
            var lease = new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(11L, 2);
            {
                var handoff =
                        EndpointTestSupport.commitBatch(state, lease, List.of(replacement), 100L);
                {
                    assertEquals(2, state.queueCounters().batchSlots());
                }
            }
        }
        ToLongFunction<List<RouteResult>> noRepacking =
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
        assertWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 100L, 1L);
        assertFalse(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                        .containsRequest("2"));
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
        RouteResult old = routeItem(101), replacement = routeItem(101);
        var stale =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, old, 30L, 10L)
                        .value();
        assertTrue(state.rollback(stale));
        var current =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, replacement, 40L, 10L)
                        .value();
        {
            var selectedEndpoint = generation;
            try (var cleanup = EndpointTestSupport.preparation(current)) {
                assertNull(state.commitDirect(old));
                assertFalse(state.rollback(stale));
                assertEquals(40L, remainingWork());
                assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                {
                    var committed =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, replacement);
                    assertEquals(0L, committed.calculate().totalRemainingWorkMs().orElseThrow());
                }
            }
        }
        assertTrue(state.releaseRequest(replacement));
    }

    @Test
    void mismatchedDirectReservationLeavesBothRequestsRetryable() {
        RouteResult first = routeItem(101), second = routeItem(102);
        var firstReservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, first, 30L, 10L)
                        .value();
        var secondReservation =
                org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                state, second, 40L, 10L)
                        .value();
        {
            var firstWorker = generation;
            var secondWorker = generation;
            try (var firstCleanup = EndpointTestSupport.preparation(firstReservation);
                    var secondCleanup = EndpointTestSupport.preparation(secondReservation)) {
                assertThrows(IllegalArgumentException.class, () -> firstReservation.bind(second));
                assertEquals(70L, remainingWork());
                {
                    var committed =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, first);
                    assertEquals(40L, committed.calculate().totalRemainingWorkMs().orElseThrow());
                }
                {
                    var committed =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    state, second);
                    assertEquals(30L, committed.calculate().totalRemainingWorkMs().orElseThrow());
                }
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
        ToLongFunction<List<RouteResult>> noRepacking =
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
        runWithStateLock(
                lock,
                () -> {
                    var prepared =
                            state.prepareStatusLocked(
                                    WorkerStatusTestSupport.freezeStatusResponse(worker, response),
                                    false);
                    assertNotNull(state.commitStatusLocked(prepared, Map.of()));
                    assertEquals(Long.MAX_VALUE, state.admissionSummary(0, 0L).occupiedRequests());
                    assertEquals(Long.MAX_VALUE, state.admissionSummary(0, 1L).requestsToRelease());
                    assertEquals(1L, state.admissionSummary(0, Long.MAX_VALUE).requestsToRelease());
                    assertNotNull(
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                    state, item(9L)));
                    assertFalse(state.requestCapacityAvailable(Long.MAX_VALUE, 1));
                    assertEquals(Long.MAX_VALUE, state.admissionSummary(0, 0L).occupiedRequests());
                });
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preparedCapacityReleaseTracksUnknownOwnershipAndRejectsStaleCounts(boolean invalidate) {
        var worker = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningQueryLen(4L);
        runWithStateLock(
                lock,
                () -> {
                    state.commitStatusLocked(
                            state.prepareStatusLocked(
                                    WorkerStatusTestSupport.freezeStatusResponse(worker, response),
                                    false),
                            Map.of());
                });
        assertEquals(4L, state.admissionSummary(0, 0L).occupiedRequests());
        response.setRunningQueryLen(2L);
        var observation = WorkerStatusTestSupport.freezeStatusResponse(worker, response);
        runWithStateLock(
                lock,
                () -> {
                    var reduction = state.prepareStatusLocked(observation, true);
                    assertEquals(
                            4L,
                            state.admissionSummary(0, 0L).occupiedRequests(),
                            "preparation must not release capacity");
                    if (invalidate) {
                        response.setRunningQueryLen(1L);
                        runWithStateLock(
                                lock,
                                () -> {
                                    state.commitStatusLocked(
                                            state.prepareStatusLocked(
                                                    WorkerStatusTestSupport.freezeStatusResponse(
                                                            worker, response),
                                                    false),
                                            Map.of());
                                });
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
                });
    }

    @Test
    void heartbeatRenewsExecutionWithoutConsumingTerminalFactsOrInvalidatingUnchangedInputs() {
        RouteResult first = item(1), second = item(2);
        commitBatch(List.of(first, second), 300L);
        var worker = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        TaskInfo running = task(1, TaskPhase.RUNNING, 0L, 0L);
        running.setBatchId(10L);
        response.setRunningTaskInfo(Map.of("1", running));
        response.setFinishedTaskInfo(Map.of("1", task(1, null, 0L, 100L)));
        var observation = WorkerStatusTestSupport.freezeStatusResponse(worker, response);
        var started =
                withStateLock(
                        lock,
                        () ->
                                state.commitStatusLocked(
                                        state.prepareStatusLocked(observation, false),
                                        java.util.Map.of()));
        assertEquals(1, started.requestStatuses().size());
        assertEquals(
                PrefillEndpoint.PrefillRequestStatus.Kind.ACTIVE,
                started.requestStatuses().getFirst().kind());
        assertTrue(started.batchCompletions().isEmpty(), "a heartbeat cannot complete a batch");
        assertFalse(started.capacityReleased());
        assertTrue(started.schedulingInputsChanged());
        long version =
                org.flexlb.balance.endpoint.EndpointTestSupport.versions(state).ownershipVersion();
        clock.set(200L);
        var renewed =
                withStateLock(
                        lock,
                        () ->
                                state.commitStatusLocked(
                                        state.prepareStatusLocked(observation, false),
                                        java.util.Map.of()));
        assertFalse(
                renewed.schedulingInputsChanged(),
                "same-phase renewal leaves the projection revision stable");
        assertFalse(renewed.capacityReleased());
        assertEquals(
                version,
                org.flexlb.balance.endpoint.EndpointTestSupport.versions(state).ownershipVersion());
        assertEquals(
                0L,
                state.stats().maxObservedAgeMs(),
                "same-phase renewal refreshes the activity timestamp");
        assertEquals(
                200L, remainingWork(), "same-phase renewal still advances the execution clock");
        assertWork(
                org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 200L, 1L, 2L);
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
        assertNoWork(org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state), 1L, 2L);
        assertEquals(0, state.stats().batchCount());
    }

    @Test
    void unchangedWorkerObservationReusesWorkAndProjectionVersion() {
        RouteResult request = item(1);
        commitBatch(List.of(request), 100L);
        var active = Map.of("1", task(1, TaskPhase.RUNNING, 0L, 0L));
        ToLongFunction<List<RouteResult>> noRepacking =
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

    private static RouteResult routeItem(long id) {
        var config = new FlexlbConfig();
        config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
        var context = new RequestContext(config);
        var request = Request.builder().requestId(String.valueOf(id)).seqLen(100L).build();
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, Long.MAX_VALUE));
        return org.flexlb.balance.scheduler.RequestExecutionTestSupport.mockProtocolRoute(
                SchedulingTestConfig.createRoute(
                        context, null, null, null, null, null, null, 100L));
    }

    private void commitBatch(List<RouteResult> members, long predictedMs) {
        members.forEach(this::enqueue);
        {
            var reservation =
                    new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(10L, 2);
            {
                var handoff =
                        EndpointTestSupport.commitBatch(state, reservation, members, predictedMs);
                {
                    assertTrue(waitingItems().isEmpty());
                }
            }
        }
    }

    private List<RouteResult> waitingItems() {
        return state.captureQueue(Integer.MAX_VALUE).items();
    }

    private WorkerEndpoint.StatusChange.Prefill reconcile(
            Map<String, TaskInfo> finished,
            Map<String, TaskInfo> active,
            ToLongFunction<List<RouteResult>> repredictor) {
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

    private static void assertNoWork(WorkSnapshot snapshot, long... excludedIds) {
        assertWork(snapshot, 0L);
        for (long id : excludedIds) {
            assertFalse(snapshot.containsRequest(String.valueOf(id)), "retained request " + id);
        }
    }

    private static void assertWork(WorkSnapshot snapshot, long expectedMs, long... includedIds) {
        assertFalse(snapshot.hasUnknownWork());
        assertEquals(expectedMs, snapshot.totalRemainingWorkMs().orElseThrow());
        for (long id : includedIds) {
            assertTrue(snapshot.containsRequest(String.valueOf(id)), "missing request " + id);
        }
    }

    private List<Long> batchIds() {
        return withStateLock(
                lock,
                () -> {
                    Map<?, ?> batches =
                            (Map<?, ?>)
                                    org.springframework.test.util.ReflectionTestUtils.getField(
                                            state, "batches");
                    return batches.keySet().stream().map(id -> (Long) id).sorted().toList();
                });
    }

    private WorkSnapshot.Phase batchPhase(long batchId) {
        return withStateLock(
                lock,
                () -> {
                    Map<?, ?> batches =
                            (Map<?, ?>)
                                    org.springframework.test.util.ReflectionTestUtils.getField(
                                            state, "batches");
                    Object batch = batches.get(batchId);
                    org.junit.jupiter.api.Assertions.assertNotNull(batch);
                    return (WorkSnapshot.Phase)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    batch, "servicePhase");
                });
    }

    private long remainingWork() {
        return org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                .totalRemainingWorkMsAt(clock.get())
                .orElseThrow();
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

    private static RouteResult item(long id) {
        var item = mock(RouteResult.class);
        when(item.requestId()).thenReturn(String.valueOf(id));
        when(item.seqLen()).thenReturn(100L);
        return item;
    }

    private Snapshot capture() {
        return withStateLock(
                lock,
                () -> {
                    return new Snapshot(
                            org.flexlb.balance.endpoint.EndpointTestSupport.versions(state),
                            clock.get(),
                            waiting.capture(),
                            org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                                    state, "captureWorkLocked", clock.get(), true, null));
                });
    }

    private void enqueue(RouteResult item) {
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            item,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(state, item));
                });
    }

    private record Snapshot(
            org.flexlb.balance.projection.RequestTimeEstimation.SelectionVersion version,
            long capturedAtMs,
            PrefillRequestQueue.Snapshot active,
            WorkSnapshot work) {}
}
