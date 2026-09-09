package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestEndpointCapabilities;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestRequestRegistry;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestTelemetry;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

/** Exact ordered-prefix and per-request completion contract for route delivery. */
class RouteDeliveryStrategyTest {

    @Test
    void removedMemberDoesNotContributeToLaterRouteCompletionTime() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest cancelled = fixture.item(2L);
        ScheduledRequest last = fixture.item(3L);
        fixture.capabilities.precedingWork(new WorkSnapshot(1_000L, List.of(new WorkSnapshot.RequestWork(99L, WorkSnapshot.Phase.COMMITTED, 25L)), List.of(), 0L));
        fixture.slots.commitLostFor(cancelled);

        fixture.context.deliver(fixture.strategy, List.of(first, cancelled, last),
                "cancelled-middle", 0, OptionalLong.empty());

        assertEquals(Map.of(first, 90L,
                last, 180L), fixture.slots.unstartedWorkMs());
        assertEquals(115L, fixture.slots.remainingWorkMsAt(first, 1_000L).orElseThrow());
        assertEquals(205L, fixture.slots.remainingWorkMsAt(last, 1_000L).orElseThrow());
        assertEquals(List.of(List.of(first, last)), fixture.telemetry.routes());
    }

    @Test
    void zeroPredictionStillDeliversTheRoute() {
        Fixture fixture = new Fixture();
        ScheduledRequest item = fixture.item(1L);
        org.mockito.Mockito.when(item.seqLen()).thenReturn(0L);
        org.mockito.Mockito.when(item.hitCache()).thenReturn(0L);

        fixture.context.deliver(fixture.strategy, List.of(item),
                "zero", 0, OptionalLong.empty());

        assertEquals(Map.of(item, 0L), fixture.slots.unstartedWorkMs());
        assertEquals(List.of(List.of(item)), fixture.telemetry.routes());
    }

    @Test
    void unknownPrecedingWorkStillDeliversEveryRoute() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.capabilities.precedingWork(new WorkSnapshot(2_000L, List.of(), List.of(), 1L));

        fixture.context.deliver(fixture.strategy, List.of(first, second),
                "unknown", 0, OptionalLong.empty());

        assertEquals(Map.of(first, 90L,
                second, 180L), fixture.slots.unstartedWorkMs());
        assertTrue(fixture.slots.remainingWorkMsAt(first, 2_000L).isEmpty());
        assertSame(fixture.slots.precedingWork(first), fixture.slots.precedingWork(second));
        assertEquals(List.of(List.of(first, second)), fixture.telemetry.routes());
    }

    @Test
    void slowFirstRoutePublicationDoesNotAgeTheSecondMembersUnstartedWork() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.capabilities.precedingWork(new WorkSnapshot(1_000L, List.of(
                        new WorkSnapshot.RequestWork(3L, WorkSnapshot.Phase.ENGINE_RUNNING, 1_000L),
                        new WorkSnapshot.RequestWork(4L, WorkSnapshot.Phase.ENGINE_QUEUED, 300L)), List.of(), 0L));
        AtomicInteger published = new AtomicInteger();
        AtomicLong deliveryClock = new AtomicLong(1_000L);
        fixture.slots.beforeCompletion(() -> {
            if (published.getAndIncrement() == 0) {
                assertEquals(1_390L, fixture.slots.remainingWorkMsAt(first, deliveryClock.get()).orElseThrow());
                deliveryClock.set(3_000L);
            } else {
                assertEquals(480L, fixture.slots.remainingWorkMsAt(second, deliveryClock.get()).orElseThrow());
                assertEquals(180L, fixture.slots.unstartedWorkMs().get(second));
            }
        });

        fixture.context.deliver(fixture.strategy, List.of(first, second),
                "slow-first-route", 0, OptionalLong.empty());

        assertEquals(2, published.get());
    }

    @Test
    void commitsAndDeliversEveryExactRouteInOrder() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second), "route", 7,
                OptionalLong.empty());

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first, second), fixture.slots.committed());
        assertTrue(fixture.slots.identities().stream().allMatch(identity ->
                identity.kind() == DeliveryClaimKind.ROUTE_DECISION
                        && identity.correlationId() == 0L));
        assertEquals(List.of(
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                first,
                                DeliveryResult.delivered()),
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                second,
                                DeliveryResult.delivered())),
                fixture.slots.completions());
        assertEquals(List.of(List.of(first, second)),
                fixture.telemetry.routes());
        verify(fixture.capabilities.routeReservation(first))
                .updatePrediction(first, 90L);
        verify(fixture.capabilities.routeReservation(second))
                .updatePrediction(second, 90L);
        verify(fixture.capabilities.routeCommit()).commit(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(List.of(
                        fixture.capabilities.routeReservation(first),
                        fixture.capabilities.routeReservation(second))));
        verify(fixture.capabilities.permit(first)).transferToEngineLifecycle();
        verify(fixture.capabilities.permit(second)).transferToEngineLifecycle();
        assertEquals(1, fixture.capabilities.handoffs().size());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
    }

    @Test
    void unavailableHeadReturnsExactBoundaryWithoutPublishing() {
        Fixture fixture = new Fixture();
        ScheduledRequest head = fixture.item(1L);
        fixture.capabilities.rejectPermitAt(0);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "blocked", 0, OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                fixture.context.emptyBoundary().result().status());
        assertEquals(List.of(head), fixture.slots.prepared());
        assertTrue(fixture.slots.committed().isEmpty());
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    @Test
    void lostHeadOwnershipMaterializesOwnershipBoundary() {
        Fixture fixture = new Fixture();
        ScheduledRequest head = fixture.item(1L);
        fixture.slots.preparationLostFor(head);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "lost", 0, OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(CapacityBoundary.OWNERSHIP_LOST,
                fixture.context.emptyBoundary().result());
        assertTrue(fixture.slots.prepared().isEmpty());
    }

    @Test
    void unavailableSuffixCommitsOnlyLargestOrderedPrefix() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
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
        assertEquals(List.of(first), fixture.slots.committed());
        assertEquals(List.of(List.of(first)), fixture.telemetry.routes());
    }

    @Test
    void slotCommitFailureTerminalizesExactItemAndContinuesLaterRoutes() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        ScheduledRequest third = fixture.item(3L);
        fixture.slots.throwCommitFor(first);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second, third),
                "isolate-commit", 0,
                OptionalLong.empty());

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first, second, third), fixture.slots.committed());
        assertEquals(List.of(first), fixture.slots.failedPrepared());
        assertInstanceOf(IllegalStateException.class,
                fixture.slots.preparedFailures().getFirst());
        assertEquals(List.of(List.of(second, third)),
                fixture.telemetry.routes());
    }

    @Test
    void completionFailureIsAggregatedOnlyAfterLaterRoutesComplete() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.slots.throwCompletionFor(first);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> fixture.context.deliver(
                        fixture.strategy, List.of(first, second),
                        "isolate-completion", 0,
                        OptionalLong.empty()));

        assertTrue(failure.getMessage().contains("completion failure 1"));
        assertEquals(List.of(first, second), fixture.slots.committed());
        assertEquals(2, fixture.slots.completions().size());
        assertEquals(List.of(List.of(second)), fixture.telemetry.routes());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
    }

    @Test
    void failedQueueCommitReleasesPreparedAdmissionWithoutPublishing() {
        Fixture fixture = new Fixture();
        fixture.context.commit(false);
        ScheduledRequest head = fixture.item(1L);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "lost-commit", 0, OptionalLong.empty());

        assertEquals("NOT_COMMITTED", result);
        verify(fixture.capabilities.routeReservation(head), never()).close();
        verify(fixture.capabilities.permit(head)).release();
        verify(fixture.capabilities.permit(head), never())
                .transferToEngineLifecycle();
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    private static final class Fixture {
        private final TestEndpointCapabilities capabilities =
                new TestEndpointCapabilities();
        private final TestRequestRegistry slots = new TestRequestRegistry();
        private final TestTelemetry telemetry = new TestTelemetry();
        private final TestContext context = new TestContext();
        private final RouteDeliveryStrategy strategy =
                new RouteDeliveryStrategy(
                        slots.requests(), telemetry.metrics());

        private ScheduledRequest item(long requestId) {
            ScheduledRequest item = DeliveryStrategyTestSupport.item(requestId);
            capabilities.bind(item);
            return item;
        }
    }
}
