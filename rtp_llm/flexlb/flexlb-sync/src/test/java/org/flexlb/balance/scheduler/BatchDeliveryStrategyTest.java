package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestBatchSubmission;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestEndpointCapabilities;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestRequestRegistry;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestTelemetry;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;

import static org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.unavailable;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;

/** Final batch admission, transport handoff, and completion-correlation contract. */
class BatchDeliveryStrategyTest {

    @Test
    void preparedPredictionAndExactBatchReachTransportOnce() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second), "fixed_window", 3,
                OptionalLong.of(83L));

        assertEquals("COMMITTED", result);
        DeliveryStrategyTestSupport.SubmittedBatch command =
                fixture.submission.command();
        assertEquals(List.of(first, second), command.exactItems());
        assertEquals(701L, command.batchId());
        assertEquals(83L, command.predictedMs());
        assertEquals("fixed_window", command.decisionReason());
        assertTrue(fixture.slots.identities().stream().allMatch(identity ->
                identity.kind() == DeliveryClaimKind.BATCH_ENQUEUE
                        && identity.correlationId() == 701L));
        assertEquals(1, fixture.telemetry.batches().size());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry =
                fixture.telemetry.batches().getFirst();
        assertEquals(701L, telemetry.batchId());
        assertEquals(List.of(first, second), telemetry.dispatched());
        assertEquals(83L, telemetry.predictedMs());
        assertEquals(3, telemetry.remainingQueueDepth());
        verify(fixture.capabilities.batchReservation()).commit(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(83L));
        verify(fixture.capabilities.permit(first)).transferToEngineLifecycle();
        verify(fixture.capabilities.permit(second)).transferToEngineLifecycle();
        assertEquals(1, fixture.capabilities.handoffs().size());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
        assertEquals(1, fixture.submission.totalCloseCount());
    }

    @Test
    void missingPlannedPredictionUsesFrozenEvaluatorForCommittedBatch() {
        Fixture fixture = new Fixture(702L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);

        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "predict", 0, OptionalLong.empty());

        assertEquals(200L, fixture.submission.command().predictedMs());
        assertEquals(200L, fixture.telemetry.batches()
                .getFirst().predictedMs());
        verify(fixture.capabilities.batchReservation()).commit(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(200L));
    }

    @Test
    void nonPositiveBatchIdClosesSubmissionBeforeEndpointOwnership() {
        Fixture fixture = new Fixture(0L);
        ScheduledRequest item = fixture.item(1L);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(item),
                "bad-id", 0, OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(item, fixture.context.emptyBoundary().item());
        RuntimeException failure = assertInstanceOf(
                RuntimeException.class,
                fixture.context.emptyBoundary().result().cause());
        assertTrue(failure.getMessage().contains("batch id supplier"));
        assertEquals(1, fixture.submission.closeCount());
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableSubmissionReturnsExactHeadBoundaryBeforeAdmission() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest head = fixture.item(1L);
        CapacityBoundary unavailable = unavailable();
        fixture.submission.prepareBoundary(unavailable);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "submission-full", 0,
                OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(unavailable, fixture.context.emptyBoundary().result());
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableAdmissionClosesPreparedSubmissionAndReturnsBoundary() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest head = fixture.item(1L);
        fixture.capabilities.rejectPermitAt(0);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "admission-full", 0,
                OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                fixture.context.emptyBoundary().result().status());
        assertEquals(1, fixture.submission.closeCount());
        verify(fixture.capabilities.batchReservation()).close();
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableSuffixSubmitsLargestAdmittedPrefixAndRepredictsIt() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.capabilities.rejectPermitAt(1);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "prefix", 1, OptionalLong.of(999L));

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first),
                fixture.submission.command().exactItems());
        assertEquals(100L, fixture.submission.command().predictedMs());
        assertSame(second, fixture.context.committedBoundary().item());
        assertEquals(CapacityBoundary.Status.UNAVAILABLE,
                fixture.context.committedBoundary().result().status());
        assertEquals(List.of(first), fixture.slots.committed());
    }

    @Test
    void synchronousTransportCompletionWaitsForCapabilityHandoffClose() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.submission.completeSynchronously(
                first, DeliveryResult.delivered());
        fixture.submission.completeSynchronously(
                second, DeliveryResult.delivered());
        fixture.slots.beforeCompletion(() -> {
            fixture.capabilities.handoffs()
                    .forEach(handoff -> verify(handoff).close());
            assertEquals(1, fixture.submission.totalCloseCount(),
                    "transport preparation must close before callbacks open");
        });

        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "gate", 0, OptionalLong.of(20L));

        assertEquals(List.of(
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                first,
                                DeliveryResult.delivered()),
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                second,
                                DeliveryResult.delivered())),
                fixture.slots.completions());
    }

    @Test
    void callbackForUnsubmittedIdentityFailsClosed() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest canonical = fixture.item(1L);
        ScheduledRequest lookalike = DeliveryStrategyTestSupport.item(
                canonical.requestId(), canonical.priority(),
                canonical.enqueuedAtMs(), canonical.seqLen(),
                canonical.hitCache());
        fixture.context.deliver(
                fixture.strategy, List.of(canonical),
                "identity-fence", 0,
                OptionalLong.empty());

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> fixture.submission.complete(
                        lookalike,
                        DeliveryResult.delivered()));

        assertTrue(failure.getMessage().contains("unsubmitted identity"));
        assertTrue(fixture.slots.completions().isEmpty());
    }

    @Test
    void timeoutAndUncertainTransportOutcomesReachExactClaims() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "outcomes", 0, OptionalLong.empty());
        RuntimeException timeout = new RuntimeException("timeout");
        RuntimeException uncertain = new RuntimeException("uncertain");

        fixture.submission.complete(
                first, DeliveryResult.timedOut(timeout));
        fixture.submission.complete(
                second, DeliveryResult.uncertain(uncertain));

        assertEquals(2, fixture.slots.completions().size());
        DeliveryResult timedOut =
                fixture.slots.completions().get(0).completion();
        DeliveryResult unresolved =
                fixture.slots.completions().get(1).completion();
        assertEquals(DeliveryResult.Status.TIMED_OUT,
                timedOut.status());
        assertEquals(DeliveryResult.Status.UNCERTAIN,
                unresolved.status());
        assertSame(timeout, timedOut.cause());
        assertSame(uncertain, unresolved.cause());
    }

    @Test
    void lostClaimExcludesOnlyThatMemberFromSubmittedBatch() {
        Fixture fixture = new Fixture(701L);
        ScheduledRequest first = fixture.item(1L);
        ScheduledRequest second = fixture.item(2L);
        fixture.slots.commitLostFor(first);

        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "claim-race", 0,
                OptionalLong.of(999L));

        assertEquals(List.of(second),
                fixture.submission.command().exactItems());
        assertEquals(100L, fixture.submission.command().predictedMs());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry =
                fixture.telemetry.batches().getFirst();
        assertEquals(List.of(second), telemetry.dispatched());
        assertEquals(100L, telemetry.predictedMs());
    }

    private static final class Fixture {
        private final TestBatchSubmission submission =
                new TestBatchSubmission();
        private final TestEndpointCapabilities capabilities =
                new TestEndpointCapabilities();
        private final TestRequestRegistry slots = new TestRequestRegistry();
        private final TestTelemetry telemetry = new TestTelemetry();
        private final TestContext context = new TestContext();
        private final long correlationId;
        private final BatchDeliveryStrategy strategy;

        private Fixture(long correlationId) {
            this.correlationId = correlationId;
            this.strategy = new BatchDeliveryStrategy(
                    submission::tryPrepareSubmission,
                    () -> this.correlationId,
                    slots.requests(),
                    telemetry.metrics());
        }

        private ScheduledRequest item(long requestId) {
            ScheduledRequest item = DeliveryStrategyTestSupport.item(requestId);
            capabilities.bind(item);
            return item;
        }
    }
}
