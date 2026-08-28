package org.flexlb.balance.delivery;

import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestAdmissionPort;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestSlotPort;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestTelemetry;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;

import static org.flexlb.balance.delivery.DeliveryStrategyTestSupport.item;
import static org.flexlb.balance.delivery.DeliveryStrategyTestSupport.unavailable;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Exact ordered-prefix and per-request completion contract for route delivery. */
class RouteDeliveryStrategyTest {

    @Test
    void commitsAndDeliversEveryExactRouteInOrder() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = item(1L);
        ScheduledRequest second = item(2L);
        DeliveryMetadata metadata = new DeliveryMetadata("route", 7);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second), metadata,
                OptionalLong.empty());

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first, second),
                fixture.admission.preparedItems());
        assertEquals(List.of(90L, 90L),
                fixture.admission.preparedPredictions());
        assertEquals(List.of(first, second),
                fixture.admission.committedItems());
        assertEquals(0L, fixture.admission.committedPrediction());
        assertEquals(List.of(first, second),
                fixture.admission.transferred());
        assertEquals(List.of(first, second), fixture.slots.committed());
        assertTrue(fixture.slots.identities().stream().allMatch(identity ->
                identity.boundary()
                        == SlotDeliveryPort.Identity.ConfirmationBoundary
                        .COMMIT_CONFIRMED
                        && identity.correlationId().isEmpty()));
        assertEquals(List.of(
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                first,
                                SlotDeliveryPort.Completion.delivered()),
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                second,
                                SlotDeliveryPort.Completion.delivered())),
                fixture.slots.completions());
        assertEquals(List.of(List.of(first, second)),
                fixture.telemetry.routes());
        assertSame(metadata, fixture.context.publishedMetadata());
        assertEquals(1, fixture.admission.committedCloseCount());
    }

    @Test
    void unavailableHeadReturnsExactBoundaryWithoutPublishing() {
        Fixture fixture = new Fixture();
        ScheduledRequest head = item(1L);
        CapacityBoundary unavailable = unavailable();
        fixture.admission.prepareBoundary(unavailable);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                new DeliveryMetadata("blocked", 0), OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(unavailable, fixture.context.emptyBoundary().result());
        assertTrue(fixture.slots.prepared().isEmpty());
        assertTrue(fixture.slots.committed().isEmpty());
        assertTrue(fixture.telemetry.routes().isEmpty());
    }

    @Test
    void lostHeadOwnershipMaterializesOwnershipBoundary() {
        Fixture fixture = new Fixture();
        ScheduledRequest head = item(1L);
        fixture.slots.preparationLostFor(head);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                new DeliveryMetadata("lost", 0), OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(CapacityBoundary.OWNERSHIP_LOST,
                fixture.context.emptyBoundary().result());
        assertTrue(fixture.admission.preparedItems().isEmpty());
    }

    @Test
    void unavailableSuffixCommitsOnlyLargestOrderedPrefix() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = item(1L);
        ScheduledRequest second = item(2L);
        CapacityBoundary unavailable = unavailable();
        fixture.admission.rejectAppendAt(1, unavailable);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                new DeliveryMetadata("prefix", 1), OptionalLong.empty());

        assertEquals("COMMITTED", result);
        assertEquals(
                List.of(first), fixture.context.preparedSelection().items());
        assertSame(second, fixture.context.committedBoundary().item());
        assertSame(unavailable, fixture.context.committedBoundary().result());
        assertEquals(List.of(first), fixture.admission.committedItems());
        assertEquals(List.of(first), fixture.slots.committed());
        assertEquals(List.of(List.of(first)), fixture.telemetry.routes());
    }

    @Test
    void slotCommitFailureTerminalizesExactItemAndContinuesLaterRoutes() {
        Fixture fixture = new Fixture();
        ScheduledRequest first = item(1L);
        ScheduledRequest second = item(2L);
        ScheduledRequest third = item(3L);
        fixture.slots.throwCommitFor(first);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second, third),
                new DeliveryMetadata("isolate-commit", 0),
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
        ScheduledRequest first = item(1L);
        ScheduledRequest second = item(2L);
        fixture.slots.throwCompletionFor(first);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> fixture.context.deliver(
                        fixture.strategy, List.of(first, second),
                        new DeliveryMetadata("isolate-completion", 0),
                        OptionalLong.empty()));

        assertTrue(failure.getMessage().contains("completion failure 1"));
        assertEquals(List.of(first, second), fixture.slots.committed());
        assertEquals(2, fixture.slots.completions().size());
        assertEquals(List.of(List.of(second)), fixture.telemetry.routes());
        assertEquals(1, fixture.admission.committedCloseCount());
    }

    @Test
    void failedQueueCommitReleasesPreparedAdmissionWithoutPublishing() {
        Fixture fixture = new Fixture();
        fixture.context.commit(false);
        ScheduledRequest head = item(1L);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                new DeliveryMetadata("lost-commit", 0), OptionalLong.empty());

        assertEquals("NOT_COMMITTED", result);
        assertEquals(1, fixture.admission.preparedCloseCount());
        assertTrue(fixture.admission.committedItems().isEmpty());
        assertTrue(fixture.telemetry.routes().isEmpty());
        assertSame(null, fixture.context.publishedMetadata());
    }

    private static final class Fixture {
        private final TestAdmissionPort admission = new TestAdmissionPort();
        private final TestSlotPort slots = new TestSlotPort();
        private final TestTelemetry telemetry = new TestTelemetry();
        private final TestContext context = new TestContext();
        private final RouteDeliveryStrategy strategy =
                new RouteDeliveryStrategy(admission, slots, telemetry.metrics());
    }
}
