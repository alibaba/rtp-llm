package org.flexlb.balance.scheduler;

import static org.flexlb.balance.scheduler.DeliveryStrategy.failUnsentDelivery;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestEndpointCapabilities;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestRequestScheduler;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestTelemetry;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atMostOnce;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Exact ordered-prefix and per-request completion contract for route delivery. */
class RouteDeliveryStrategyTest {

    @Test
    void selectionRejectsMissingPrefillBeforeDeliveryResourceAcquisition() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 1L);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var assignment = mock(org.flexlb.balance.strategy.WorkerAssignment.class);
        var status = new org.flexlb.dao.loadbalance.ServerStatus();
        status.setRequestId(1L);
        status.setRole(org.flexlb.dao.route.RoleType.DECODE);
        status.setSuccess(true);
        when(assignment.serverStatus()).thenReturn(status);
        when(assignment.requestId()).thenReturn(1L);
        when(assignment.role()).thenReturn(org.flexlb.dao.route.RoleType.DECODE);
        when(assignment.endpoint()).thenReturn(decode);

        var failure = assertThrows(IllegalStateException.class,
                () -> RequestRoute.prepare(context, List.of(assignment)));

        assertTrue(failure.getMessage().contains("no Prefill"));
        verify(assignment, never()).assignToRequest();
        verify(assignment, never()).close();
        verify(decode, never()).acquireDispatchPermit(any(), any());
    }

    @Test
    void decodeCapacityWaitersSubscribeIndependentlyAndCanResubscribe() {
        var status = org.flexlb.dao.master.WorkerStatus.createDiscovered(
                org.flexlb.dao.route.RoleType.DECODE, "test", "127.0.0.1", 8000, 8001, "test");
        var endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(mock(AbstractRequestScheduler.class)));
        var capacity = new DecodeResources.AdmissionCapacity(1, 100);
        var counts = List.of(new AtomicInteger(), new AtomicInteger());
        var listeners = List.<Runnable>of(counts.get(0)::incrementAndGet, counts.get(1)::incrementAndGet);
        var waiters = new java.util.ArrayList<org.flexlb.balance.delivery.CapacityBoundary.Availability>();
        try {
            DecodeResources.ReservationHandle occupying;
            try (var pin = endpoint.tryPinGeneration()) {
                occupying = endpoint.tryReserveQueuedRequest(pin, 1L, 0L, 0L, 50, null);
            }
            var permit = endpoint.acquireDispatchPermit(occupying, capacity).permit();
            for (int i = 0; i < 2; i++) {
                long requestId = i + 2L;
                var item = mock(RequestRoute.class);
                try (var pin = endpoint.tryPinGeneration()) {
                    when(item.decodeReservation()).thenReturn(endpoint.tryReserveQueuedRequest(pin, requestId, 0L, 0L, 50, null));
                }
                when(item.requestId()).thenReturn(requestId);
                when(item.decodeEp()).thenReturn(endpoint);
                when(item.requirements()).thenReturn(SchedulingTestConfig.decodeRequirements(50, 0L, 0L, capacity));
                var boundary = DeliveryTransaction.prepareMember(item).boundary();
                assertTrue(boundary.unavailable());
                waiters.add(boundary.availability());
                org.junit.jupiter.api.Assertions.assertFalse(boundary.availability().isAvailable());
                boundary.availability().addListener(listeners.get(i));
                boundary.availability().addListener(listeners.get(i));
            }
            assertTrue(permit.release());
            assertEquals(List.of(1, 1), counts.stream().map(AtomicInteger::get).toList());
            assertTrue(waiters.get(0).isAvailable());
            waiters.get(0).removeListener(listeners.get(0));
            waiters.get(0).removeListener(listeners.get(0));
            assertTrue(endpoint.acquireDispatchPermit(occupying, capacity).permit().release());
            assertEquals(List.of(1, 2), counts.stream().map(AtomicInteger::get).toList());
            waiters.get(0).addListener(listeners.get(0));
            assertTrue(endpoint.acquireDispatchPermit(occupying, capacity).permit().release());
            assertEquals(List.of(2, 3), counts.stream().map(AtomicInteger::get).toList());
        } finally {
            try {
                for (int i = 0; i < waiters.size(); i++) {
                    waiters.get(i).removeListener(listeners.get(i));
                }
            } finally {
                endpoint.close();
                endpoint.awaitRetirement();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void rejectedHandoffSettlesEveryCommittedOwnerEvenWhenOneCleanupFails(boolean cleanupFails) {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L), sibling = fixture.item(2L);
        var cleanup = new IllegalStateException("first member cleanup failed");
        try (var transaction = fixture.strategy.prepare(List.of(first, sibling),
                DeliveryStrategyTestSupport.EVALUATOR, OptionalLong.empty())) {
            transaction.commitSelectionLocked(System.currentTimeMillis());
            Throwable failure = assertThrows(NullPointerException.class,
                    () -> fixture.strategy.deliver(transaction, "invalid-work", 0, null, DeliveryStrategyTestSupport.EVALUATOR));
            if (cleanupFails) {
                doThrow(cleanup).when(fixture.schedulerFixture.scheduler()).failDeliveryPreparation(first, failure);
                assertSame(cleanup, assertThrows(IllegalStateException.class, () -> failUnsentDelivery(transaction, failure, false)));
            } else {
                failUnsentDelivery(transaction, failure, false);
            }
            failUnsentDelivery(transaction, failure, false);
            for (RequestRoute item : List.of(first, sibling)) {
                verify(fixture.schedulerFixture.scheduler()).failDeliveryPreparation(item, failure);
                verify(fixture.capabilities.permit(item), never()).dispatch();
                verify(fixture.capabilities.permit(item)).release();
            }
        }
        assertEquals(1, fixture.capabilities.handoffs().size());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
        assertTrue(fixture.schedulerFixture.completions().isEmpty());
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    @ParameterizedTest
    @ValueSource(longs = {4, 5, 10, 15, 100})
    void routePredictionVisitsEachTentativeMemberOnce(long budgetMs) {
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        var strategy = new RouteDeliveryStrategy(mock(DeliveryMetricsReporter.class));
        var items = LongStream.rangeClosed(1, 3).mapToObj(id -> {
            RequestRoute item = mock(RequestRoute.class);
            when(item.seqLen()).thenReturn(id);
            when(evaluator.estimateMs(id, 0L)).thenReturn(id * 5L);
            return item;
        }).toList();
        var constraints = new GroupPlanner.Constraints(
                3, Long.MAX_VALUE, Long.MAX_VALUE, budgetMs, 0L);
        var expected = GroupPlanner.selectWithPrediction(items, constraints,
                (added, prefix) -> prefix.stream().mapToDouble(item -> item.seqLen() * 5.0).sum());
        var actual = GroupPlanner.selectWithPrediction(items, constraints,
                strategy.newGroupPredictor(evaluator));
        assertEquals(expected, actual);
        verify(evaluator).estimateMs(1L, 0L);
        verify(evaluator, atMostOnce()).estimateMs(2L, 0L);
        verify(evaluator, atMostOnce()).estimateMs(3L, 0L);
        // A new decision must start at zero even if the previous tentative tail exceeded its budget.
        assertEquals(5.0, strategy.newGroupPredictor(evaluator).append(items.get(0), List.of(items.get(0))));
    }

    @Test
    void failedPermitReleaseStillClosesSiblingsBeforeGenerationHandoff() {
        var first = mock(DecodeEndpoint.EngineDispatchPermit.class);
        var sibling = mock(DecodeEndpoint.EngineDispatchPermit.class);
        var handoff = mock(PrefillState.CommittedHandoff.class);
        var members = List.of(
                new DeliveryTransaction.Member(mock(RequestRoute.class), first),
                new DeliveryTransaction.Member(mock(RequestRoute.class), sibling));
        doThrow(new IllegalStateException("release failed")).when(first).release();

        DeliveryTransaction.closeCommitted(members, handoff);

        var order = org.mockito.Mockito.inOrder(first, sibling, handoff);
        order.verify(first).release();
        order.verify(sibling).release();
        order.verify(handoff).close();
        order.verifyNoMoreInteractions();
    }

    @ParameterizedTest
    @ValueSource(strings = {"TRANSFERRED", "OWNERSHIP_LOST", "ENDPOINT_RETIRED", "THROW", "NO_DECODE"})
    void committedCleanupDelegatesToEveryExactPermit(String result) {
        var scheduler = org.mockito.Mockito.mock(AbstractRequestScheduler.class, org.mockito.Mockito.CALLS_REAL_METHODS);
        var context = org.mockito.Mockito.mock(RequestContext.class);
        var first = org.mockito.Mockito.mock(RequestRoute.class);
        when(first.ctx()).thenReturn(context);
        when(first.decodeEp()).thenReturn(org.mockito.Mockito.mock(DecodeEndpoint.class));
        when(first.decodeReservation()).thenReturn(org.mockito.Mockito.mock(DecodeResources.ReservationHandle.class));
        when(scheduler.ownsPreparedDeliveryLocked(context, first)).thenReturn(true);
        var claim = org.mockito.Mockito.mock(RequestContext.DeliveryClaim.class);
        when(context.beginDelivery(eq(first), eq(DeliveryClaimKind.ROUTE_DECISION), eq(0L), org.mockito.ArgumentMatchers.anyLong())).thenReturn(claim);
        org.springframework.test.util.ReflectionTestUtils.setField(scheduler, "runtime", org.mockito.Mockito.mock(SchedulerRuntime.class));
        RequestRoute sibling = mock(RequestRoute.class);
        var firstPermit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        when(firstPermit.belongsTo(first.decodeEp(), first.decodeReservation())).thenReturn(true);
        var siblingPermit = mock(DecodeEndpoint.EngineDispatchPermit.class);
        var handoff = mock(PrefillState.CommittedHandoff.class);
        var failure = new IllegalStateException("dispatch failed");
        if ("THROW".equals(result)) {
            when(firstPermit.dispatch()).thenThrow(failure);
        } else if (!"NO_DECODE".equals(result)) {
            when(firstPermit.dispatch()).thenReturn(DecodeResources.EngineDispatchPermitTransferStatus.valueOf(result));
        }
        var firstMember = new DeliveryTransaction.Member(
                first, "NO_DECODE".equals(result) ? null : firstPermit);
        var siblingMember = new DeliveryTransaction.Member(sibling, siblingPermit);
        var members = List.of(firstMember, siblingMember);
        try {
            if ("THROW".equals(result)) {
                assertSame(failure, assertThrows(IllegalStateException.class,
                        () -> scheduler.claimDelivery(first, DeliveryClaimKind.ROUTE_DECISION, 0L, firstMember.decode())));
            } else if ("ENDPOINT_RETIRED".equals(result) || "OWNERSHIP_LOST".equals(result)) {
                assertThrows(IllegalStateException.class,
                        () -> scheduler.claimDelivery(first, DeliveryClaimKind.ROUTE_DECISION, 0L, firstMember.decode()));
            } else {
                assertSame(claim, scheduler.claimDelivery(first, DeliveryClaimKind.ROUTE_DECISION, 0L, firstMember.decode()));
            }
        } finally {
            DeliveryTransaction.closeCommitted(members, handoff);
        }
        if ("NO_DECODE".equals(result)) {
            verify(firstPermit, never()).release();
        } else {
            verify(firstPermit).release();
        }
        verify(siblingPermit, never()).dispatch();
        verify(siblingPermit).release();
        verify(handoff).close();
    }

    @ParameterizedTest
    @ValueSource(strings = {"TRANSFERRED", "ABANDONED", "REPLACED", "RETIRED", "RELEASE_FAILURE"})
    void memberCleanupUsesRealPermitResolutionWithoutReleasingAnotherOwner(String outcome) {
        var status = org.mockito.Mockito.spy(org.flexlb.dao.master.WorkerStatus.createDiscovered(
                org.flexlb.dao.route.RoleType.DECODE, "test", "127.0.0.1", 8000, 8001, "test"));
        var endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(mock(AbstractRequestScheduler.class)));
        var capacity = new DecodeResources.AdmissionCapacity(2, 100);
        AtomicInteger notifications = new AtomicInteger();
        endpoint.addEngineDispatchCapacityListener(notifications::incrementAndGet);
        try {
            DecodeResources.ReservationHandle reservation;
            try (var pin = endpoint.tryPinGeneration()) {
                reservation = endpoint.tryReserveQueuedRequest(pin, 1L, 100L, 200L, 50, null);
            }
            var acquired = endpoint.acquireDispatchPermit(reservation, capacity);
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, acquired.status());
            var member = new DeliveryTransaction.Member(mock(RequestRoute.class), acquired.permit());
            DecodeEndpoint.EngineDispatchPermit replacement = null;
            if ("TRANSFERRED".equals(outcome)) {
                assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED, member.decode().dispatch());
                assertEquals(0, endpoint.resourceSnapshot().activeDispatchPermits());
                assertEquals(0, endpoint.resourceSnapshot().queuedCount());
            } else if ("RETIRED".equals(outcome)) {
                endpoint.close();
                endpoint.awaitRetirement();
                assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.ENDPOINT_RETIRED, member.decode().dispatch());
            } else {
                if ("ABANDONED".equals(outcome)) {
                    member.close();
                } else if ("RELEASE_FAILURE".equals(outcome)) {
                    var failure = new IllegalStateException("capacity publication failed after release");
                    doThrow(failure).when(status).topologySnapshot();
                    try {
                        assertSame(failure, assertThrows(IllegalStateException.class, member::close));
                    } finally {
                        org.mockito.Mockito.doCallRealMethod().when(status).topologySnapshot();
                    }
                    assertEquals(0, endpoint.resourceSnapshot().activeDispatchPermits());
                    assertEquals(1, notifications.get());
                } else {
                    assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
                    try (var pin = endpoint.tryPinGeneration()) {
                        reservation = endpoint.tryReserveQueuedRequest(pin, 1L, 100L, 200L, 50, null);
                    }
                }
                replacement = endpoint.acquireDispatchPermit(reservation, capacity).permit();
                assertEquals(1, endpoint.resourceSnapshot().activeDispatchPermits());
                assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, member.decode().dispatch());
            }
            var before = endpoint.resourceSnapshot();
            long version = endpoint.placementVersion();
            int notified = notifications.get();
            member.close();
            member.close();
            assertEquals(before, endpoint.resourceSnapshot());
            assertEquals(version, endpoint.placementVersion());
            assertEquals(notified, notifications.get());
            if (replacement != null) {
                assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED, replacement.dispatch());
            }
        } finally {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void removedMemberDoesNotContributeToLaterRouteCompletionTime(boolean claimThrows) {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute cancelled = fixture.item(2L);
        RequestRoute last = fixture.item(3L);
        fixture.capabilities.precedingWork(new WorkSnapshot(1_000L, List.of(new WorkSnapshot.RequestWork(99L, WorkSnapshot.Phase.COMMITTED, 25L)), List.of(), 0L));
        when(first.seqLen()).thenReturn(20L);
        when(cancelled.seqLen()).thenReturn(30L);
        when(last.seqLen()).thenReturn(40L);
        if (claimThrows) {
            fixture.schedulerFixture.throwCommitFor(cancelled);
        } else {
            fixture.schedulerFixture.commitLostFor(cancelled);
        }
        fixture.schedulerFixture.beforeCompletion(() ->
                assertEquals(List.of(first, cancelled, last), fixture.schedulerFixture.committed()));

        fixture.context.deliver(fixture.strategy, List.of(first, cancelled, last),
                "cancelled-middle", 0, OptionalLong.empty());

        assertEquals(Map.of(first, 10L,
                last, 40L), fixture.schedulerFixture.unstartedWorkMs());
        assertEquals(35L, fixture.schedulerFixture.remainingWorkMsAt(first, 1_000L).orElseThrow());
        assertEquals(65L, fixture.schedulerFixture.remainingWorkMsAt(last, 1_000L).orElseThrow());
        assertEquals(List.of(List.of(first, last)), fixture.telemetry.routes());
        verify(fixture.capabilities.permit(cancelled)).release();
        verify(fixture.capabilities.permit(cancelled), never()).dispatch();
        verify(fixture.capabilities.permit(first)).dispatch();
        verify(fixture.capabilities.permit(last)).dispatch();
    }

    @Test
    void zeroPredictionStillDeliversTheRoute() {
        Fixture fixture = new Fixture();
        RequestRoute item = fixture.item(1L);
        org.mockito.Mockito.when(item.seqLen()).thenReturn(0L);
        org.mockito.Mockito.when(item.hitCache()).thenReturn(0L);

        fixture.context.deliver(fixture.strategy, List.of(item),
                "zero", 0, OptionalLong.empty());

        assertEquals(Map.of(item, 0L), fixture.schedulerFixture.unstartedWorkMs());
        assertEquals(List.of(List.of(item)), fixture.telemetry.routes());
    }

    @Test
    void unknownPrecedingWorkStillDeliversEveryRoute() {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        fixture.capabilities.precedingWork(new WorkSnapshot(2_000L, List.of(), List.of(), 1L));

        fixture.context.deliver(fixture.strategy, List.of(first, second),
                "unknown", 0, OptionalLong.empty());

        assertEquals(Map.of(first, 90L,
                second, 180L), fixture.schedulerFixture.unstartedWorkMs());
        assertTrue(fixture.schedulerFixture.remainingWorkMsAt(first, 2_000L).isEmpty());
        assertSame(fixture.schedulerFixture.precedingWork(first), fixture.schedulerFixture.precedingWork(second));
        assertEquals(List.of(List.of(first, second)), fixture.telemetry.routes());
    }

    @Test
    void slowFirstRoutePublicationDoesNotAgeTheSecondMembersUnstartedWork() {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        fixture.capabilities.precedingWork(new WorkSnapshot(1_000L, List.of(
                        new WorkSnapshot.RequestWork(3L, WorkSnapshot.Phase.ENGINE_RUNNING, 1_000L),
                        new WorkSnapshot.RequestWork(4L, WorkSnapshot.Phase.ENGINE_QUEUED, 300L)), List.of(), 0L));
        AtomicInteger published = new AtomicInteger();
        AtomicLong deliveryClock = new AtomicLong(1_000L);
        fixture.schedulerFixture.beforeCompletion(() -> {
            if (published.getAndIncrement() == 0) {
                assertEquals(1_390L, fixture.schedulerFixture.remainingWorkMsAt(first, deliveryClock.get()).orElseThrow());
                deliveryClock.set(3_000L);
            } else {
                assertEquals(480L, fixture.schedulerFixture.remainingWorkMsAt(second, deliveryClock.get()).orElseThrow());
                assertEquals(180L, fixture.schedulerFixture.unstartedWorkMs().get(second));
            }
        });

        fixture.context.deliver(fixture.strategy, List.of(first, second),
                "slow-first-route", 0, OptionalLong.empty());

        assertEquals(2, published.get());
    }

    @Test
    void commitsAndDeliversEveryExactRouteInOrder() {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second), "route", 7,
                OptionalLong.empty());

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first, second), fixture.schedulerFixture.committed());
        assertTrue(fixture.schedulerFixture.identities().stream().allMatch(identity ->
                identity.kind() == DeliveryClaimKind.ROUTE_DECISION
                        && identity.correlationId() == 0L));
        assertEquals(List.of(
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                first,
                                DeliveryResult.delivered()),
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                second,
                                DeliveryResult.delivered())),
                fixture.schedulerFixture.completions());
        assertEquals(List.of(List.of(first, second)),
                fixture.telemetry.routes());
        verify(fixture.capabilities.prefill()).commitQueuedRoutesLocked(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.AdditionalMatchers.aryEq(new long[]{90L, 90L}), org.mockito.ArgumentMatchers.isNull(), org.mockito.ArgumentMatchers.anyLong());
        verify(fixture.capabilities.permit(first)).dispatch();
        verify(fixture.capabilities.permit(second)).dispatch();
        assertEquals(1, fixture.capabilities.handoffs().size());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
    }

    @Test
    void unavailableHeadReturnsExactBoundaryWithoutPublishing() {
        Fixture fixture = new Fixture();
        RequestRoute head = fixture.item(1L);
        fixture.capabilities.rejectPermitAt(0);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "blocked", 0, OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                fixture.context.emptyBoundary().result().status());
        assertEquals(List.of(head), fixture.schedulerFixture.prepared());
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    @Test
    void lostHeadOwnershipMaterializesOwnershipBoundary() {
        Fixture fixture = new Fixture();
        RequestRoute head = fixture.item(1L);
        fixture.schedulerFixture.preparationLostFor(head);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "lost", 0, OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(CapacityBoundary.OWNERSHIP_LOST,
                fixture.context.emptyBoundary().result());
        assertTrue(fixture.schedulerFixture.prepared().isEmpty());
    }

    @Test
    void unavailableSuffixCommitsOnlyLargestOrderedPrefix() {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        fixture.capabilities.rejectPermitAt(1);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "prefix", 1, OptionalLong.empty());

        assertEquals("COMMITTED", result);
        assertEquals(
                List.of(first), fixture.context.preparedSelection().items());
        assertSame(second, fixture.context.committedBoundary().item());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                fixture.context.committedBoundary().result().status());
        assertEquals(List.of(first), fixture.schedulerFixture.committed());
        assertEquals(List.of(List.of(first)), fixture.telemetry.routes());
    }

    @Test
    void capacityBoundaryStopsAfterPredictingTheBlockedMember() {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L), blocked = fixture.item(2L), later = fixture.item(3L);
        when(first.seqLen()).thenReturn(20L);
        when(blocked.seqLen()).thenReturn(40L);
        fixture.capabilities.rejectPermitAt(1);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.estimateMs(20L, 10L)).thenReturn(5L);
        when(evaluator.estimateMs(40L, 10L)).thenAnswer(invocation -> {
            assertTrue(Thread.holdsLock(blocked.ctx()));
            return 6L;
        });
        try (var transaction = fixture.strategy.prepare(List.of(first, blocked, later), evaluator, OptionalLong.empty())) {
            assertEquals(List.of(first), transaction.items());
            assertSame(blocked, transaction.blockedItem());
            assertTrue(transaction.blockedResult().unavailable());
            assertEquals(5L, transaction.routePredictions[0]);
        }
        verify(evaluator).estimateMs(20L, 10L);
        verify(evaluator).estimateMs(40L, 10L);
        verify(evaluator, never()).estimateMs(100L, 10L);
        verify(fixture.capabilities.permit(first)).release();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void predictionFailurePrecedesBlockedCapacityAndReleasesPreparedPrefix(boolean cleanupFails) {
        Fixture fixture = new Fixture();
        List<RequestRoute> items = List.of(fixture.item(1L), fixture.item(2L), fixture.item(3L));
        when(items.get(0).seqLen()).thenReturn(20L);
        when(items.get(1).seqLen()).thenReturn(40L);
        fixture.capabilities.rejectPermitAt(1);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        var primary = new IllegalStateException("route prediction failed");
        var cleanup = new IllegalStateException("permit cleanup failed");
        when(evaluator.estimateMs(40L, 10L)).thenAnswer(invocation -> {
            assertTrue(Thread.holdsLock(items.get(1).ctx()));
            if (cleanupFails) { doThrow(cleanup).when(fixture.capabilities.permit(items.get(0))).release(); }
            throw primary;
        });

        assertSame(primary, assertThrows(IllegalStateException.class,
                () -> fixture.strategy.prepare(items, evaluator, OptionalLong.empty())));

        verify(fixture.capabilities.permit(items.getFirst())).release();
        verify(fixture.capabilities.permit(items.getFirst()), never()).dispatch();
        for (RequestRoute item : items.subList(1, items.size())) {
            verify(item.decodeEp(), never()).acquireDispatchPermit(eq(item.decodeReservation()), any());
        }
        verify(evaluator, never()).estimateMs(100L, 10L);
        assertEquals(cleanupFails ? List.of(cleanup) : List.of(), List.of(primary.getSuppressed()));
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    @Test
    void negativePredictionPrecedesUnavailableHeadCapacity() {
        Fixture fixture = new Fixture();
        RequestRoute head = fixture.item(1L);
        fixture.capabilities.rejectPermitAt(0);
        var evaluator = mock(PrefillTimePredictor.Evaluator.class);
        when(evaluator.estimateMs(100L, 10L)).thenReturn(-1L);

        assertThrows(org.flexlb.balance.prediction.InvalidPrefillPredictionException.class,
                () -> fixture.strategy.prepare(List.of(head), evaluator, OptionalLong.empty()));

        verify(head.decodeEp(), never()).acquireDispatchPermit(any(), any());
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    @Test
    void contextCommitFailureTerminalizesExactItemAndContinuesLaterRoutes() {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        RequestRoute third = fixture.item(3L);
        fixture.schedulerFixture.throwCommitFor(first);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second, third),
                "isolate-commit", 0,
                OptionalLong.empty());

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first, second, third), fixture.schedulerFixture.committed());
        assertEquals(List.of(first), fixture.schedulerFixture.failedPrepared());
        assertInstanceOf(IllegalStateException.class,
                fixture.schedulerFixture.preparedFailures().getFirst());
        assertEquals(List.of(List.of(second, third)),
                fixture.telemetry.routes());
    }

    @Test
    void completionFailureIsAggregatedOnlyAfterLaterRoutesComplete() {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        when(first.seqLen()).thenReturn(20L);
        when(second.seqLen()).thenReturn(40L);
        fixture.schedulerFixture.throwCompletionFor(first);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> fixture.context.deliver(
                        fixture.strategy, List.of(first, second),
                        "isolate-completion", 0,
                        OptionalLong.empty()));

        assertTrue(failure.getMessage().contains("completion failure 1"));
        assertEquals(List.of(first, second), fixture.schedulerFixture.committed());
        assertEquals(2, fixture.schedulerFixture.completions().size());
        assertEquals(Map.of(first, 10L, second, 40L), fixture.schedulerFixture.unstartedWorkMs());
        assertEquals(List.of(List.of(second)), fixture.telemetry.routes());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
    }

    @Test
    void failedQueueCommitReleasesPreparedAdmissionWithoutPublishing() {
        Fixture fixture = new Fixture();
        fixture.context.commit(false);
        RequestRoute head = fixture.item(1L);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "lost-commit", 0, OptionalLong.empty());

        assertEquals("NOT_COMMITTED", result);
        verify(fixture.capabilities.permit(head)).release();
        verify(fixture.capabilities.permit(head), never())
                .dispatch();
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preparationFailureReleasesAllOwnedResourcesAndPreservesCause(boolean cleanupFails) {
        Fixture fixture = new Fixture();
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        RequestRoute last = fixture.item(3L);
        IllegalStateException primary = new IllegalStateException("preparation failed");
        IllegalStateException cleanup = new IllegalStateException("permit release failed");
        doAnswer(invocation -> {
            if (cleanupFails) {
                doThrow(cleanup).when(fixture.capabilities.permit(first)).release();
            }
            throw primary;
        }).when(fixture.schedulerFixture.scheduler()).ownsPreparedDeliveryLocked(any(), eq(last));

        assertSame(primary, assertThrows(IllegalStateException.class,
                () -> fixture.strategy.prepare(List.of(first, second, last),
                        DeliveryStrategyTestSupport.EVALUATOR, OptionalLong.empty())));

        for (RequestRoute item : List.of(first, second)) {
            verify(fixture.capabilities.permit(item)).release();
            verify(fixture.capabilities.permit(item), never()).dispatch();
        }
        assertEquals(cleanupFails ? List.of(cleanup) : List.of(), List.of(primary.getSuppressed()));
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
    }

    private static final class Fixture {
        private final TestEndpointCapabilities capabilities =
                new TestEndpointCapabilities();
        private final TestRequestScheduler schedulerFixture = new TestRequestScheduler();
        private final TestTelemetry telemetry = new TestTelemetry();
        private final TestContext context = new TestContext();
        private final RouteDeliveryStrategy strategy =
                new RouteDeliveryStrategy(telemetry.metrics());

        private RequestRoute item(long requestId) {
            RequestRoute item = DeliveryStrategyTestSupport.item(requestId);
            var request = org.mockito.Mockito.mock(RequestContext.class);
            org.mockito.Mockito.when(request.scheduler()).thenReturn(schedulerFixture.scheduler());
            org.mockito.Mockito.when(item.ctx()).thenReturn(request);
            capabilities.bind(item);
            return item;
        }
    }
}
