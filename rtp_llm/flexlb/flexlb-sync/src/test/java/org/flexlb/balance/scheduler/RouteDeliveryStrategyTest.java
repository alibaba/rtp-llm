package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestEndpointCapabilities;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestRequestRegistry;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestTelemetry;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;

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
        verify(fixture.capabilities.prefill()).reserveRoute(
                org.mockito.ArgumentMatchers.same(first),
                org.mockito.ArgumentMatchers.eq(90L),
                org.mockito.ArgumentMatchers.anyInt());
        verify(fixture.capabilities.prefill()).reserveRoute(
                org.mockito.ArgumentMatchers.same(second),
                org.mockito.ArgumentMatchers.eq(90L),
                org.mockito.ArgumentMatchers.anyInt());
        verify(fixture.capabilities.routeReservation(first)).commitGroup(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(List.of(
                        fixture.capabilities.routeReservation(first),
                        fixture.capabilities.routeReservation(second))));
        verify(fixture.capabilities.permit(first)).transferToEngineLifecycle();
        verify(fixture.capabilities.permit(second)).transferToEngineLifecycle();
        assertEquals(2, fixture.capabilities.handoffs().size());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
    }

    @Test
    void unavailableHeadReturnsExactBoundaryWithoutPublishing() {
        Fixture fixture = new Fixture();
        ScheduledRequest head = fixture.item(1L);
        fixture.capabilities.rejectRouteAt(0);

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
        fixture.capabilities.rejectRouteAt(1);

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
        verify(fixture.capabilities.routeReservation(head)).close();
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
