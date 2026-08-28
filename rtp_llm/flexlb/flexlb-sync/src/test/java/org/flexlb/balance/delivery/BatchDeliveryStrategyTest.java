package org.flexlb.balance.delivery;

import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestAdmissionPort;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestBatchSubmissionPort;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestItem;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestSlotPort;
import org.flexlb.balance.delivery.DeliveryStrategyTestSupport.TestTelemetry;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;

import static org.flexlb.balance.delivery.DeliveryStrategyTestSupport.EVALUATOR;
import static org.flexlb.balance.delivery.DeliveryStrategyTestSupport.item;
import static org.flexlb.balance.delivery.DeliveryStrategyTestSupport.unavailable;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Final batch admission, transport handoff, and completion-correlation contract. */
class BatchDeliveryStrategyTest {

    @Test
    void preparedPredictionAndExactBatchReachTransportOnce() {
        Fixture fixture = new Fixture(701L);
        TestItem first = item(1L);
        TestItem second = item(2L);
        DeliveryMetadata metadata = new DeliveryMetadata("fixed_window", 3);

        String result = fixture.strategy.admitAndDeliver(
                List.of(first, second), metadata, EVALUATOR,
                OptionalLong.of(83L), fixture.context);

        assertEquals("COMMITTED", result);
        BatchSubmissionPort.Command command = fixture.submission.command();
        assertEquals(List.of(first, second), command.exactItems());
        assertEquals(701L, command.batchId());
        assertEquals(83L, command.predictedMs());
        assertSame(metadata, command.metadata());
        assertEquals(List.of(first, second),
                fixture.admission.committedItems());
        assertEquals(83L, fixture.admission.committedPrediction());
        assertEquals(List.of(first, second),
                fixture.admission.transferred());
        assertTrue(fixture.slots.identities().stream().allMatch(identity ->
                identity.boundary()
                        == SlotDeliveryPort.Identity.ConfirmationBoundary
                        .EXTERNAL_ACK
                        && identity.requiredCorrelationId() == 701L));
        assertEquals(1, fixture.telemetry.batches().size());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry =
                fixture.telemetry.batches().getFirst();
        assertEquals(701L, telemetry.batchId());
        assertEquals(List.of(first, second), telemetry.dispatched());
        assertEquals(83L, telemetry.predictedMs());
        assertEquals(1, fixture.admission.committedCloseCount());
        assertEquals(1, fixture.submission.totalCloseCount());
    }

    @Test
    void missingPlannedPredictionUsesFrozenEvaluatorForCommittedBatch() {
        Fixture fixture = new Fixture(702L);
        TestItem first = item(1L);
        TestItem second = item(2L);

        fixture.strategy.admitAndDeliver(
                List.of(first, second),
                new DeliveryMetadata("predict", 0),
                EVALUATOR, OptionalLong.empty(), fixture.context);

        assertEquals(200L, fixture.submission.command().predictedMs());
        assertEquals(200L, fixture.admission.committedPrediction());
        assertEquals(200L, fixture.telemetry.batches()
                .getFirst().predictedMs());
    }

    @Test
    void admissionMustProvidePositiveTransportCorrelation() {
        for (OptionalLong invalid : List.of(
                OptionalLong.empty(), OptionalLong.of(0L),
                OptionalLong.of(-1L))) {
            Fixture fixture = new Fixture(invalid);
            TestItem item = item(1L);

            String result = fixture.strategy.admitAndDeliver(
                    List.of(item),
                    new DeliveryMetadata("bad-id", 0),
                    EVALUATOR, OptionalLong.empty(), fixture.context);

            assertEquals("BOUNDARY", result);
            assertSame(item, fixture.context.emptyBoundary().item());
            CapacityBoundary.Failed failed = assertInstanceOf(
                    CapacityBoundary.Failed.class,
                    fixture.context.emptyBoundary().result());
            RuntimeException failure = assertInstanceOf(
                    RuntimeException.class, failed.cause());
            assertTrue(failure.getMessage().contains("correlation id"));
            assertEquals(1, fixture.submission.closeCount());
            assertEquals(1, fixture.admission.preparedCloseCount());
            assertTrue(fixture.slots.committed().isEmpty());
        }
    }

    @Test
    void unavailableSubmissionReturnsExactHeadBoundaryBeforeAdmission() {
        Fixture fixture = new Fixture(701L);
        TestItem head = item(1L);
        CapacityBoundary.Unavailable unavailable = unavailable();
        fixture.submission.prepareBoundary(unavailable);

        String result = fixture.strategy.admitAndDeliver(
                List.of(head), new DeliveryMetadata("submission-full", 0),
                EVALUATOR, OptionalLong.empty(), fixture.context);

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(unavailable, fixture.context.emptyBoundary().result());
        assertTrue(fixture.admission.preparedItems().isEmpty());
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableAdmissionClosesPreparedSubmissionAndReturnsBoundary() {
        Fixture fixture = new Fixture(701L);
        TestItem head = item(1L);
        CapacityBoundary.Unavailable unavailable = unavailable();
        fixture.admission.prepareBoundary(unavailable);

        String result = fixture.strategy.admitAndDeliver(
                List.of(head), new DeliveryMetadata("admission-full", 0),
                EVALUATOR, OptionalLong.empty(), fixture.context);

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(unavailable, fixture.context.emptyBoundary().result());
        assertEquals(1, fixture.submission.closeCount());
        assertTrue(fixture.slots.committed().isEmpty());
    }

    @Test
    void unavailableSuffixSubmitsLargestAdmittedPrefixAndRepredictsIt() {
        Fixture fixture = new Fixture(701L);
        TestItem first = item(1L);
        TestItem second = item(2L);
        CapacityBoundary.Unavailable unavailable = unavailable();
        fixture.admission.rejectAppendAt(1, unavailable);

        String result = fixture.strategy.admitAndDeliver(
                List.of(first, second),
                new DeliveryMetadata("prefix", 1),
                EVALUATOR, OptionalLong.of(999L), fixture.context);

        assertEquals("COMMITTED", result);
        assertEquals(List.of(first),
                fixture.submission.command().exactItems());
        assertEquals(100L, fixture.submission.command().predictedMs());
        assertSame(second, fixture.context.committedBoundary().item());
        assertSame(unavailable, fixture.context.committedBoundary().result());
        assertEquals(List.of(first), fixture.admission.committedItems());
        assertEquals(List.of(first), fixture.slots.committed());
    }

    @Test
    void synchronousTransportCompletionWaitsForCapabilityHandoffClose() {
        Fixture fixture = new Fixture(701L);
        TestItem first = item(1L);
        TestItem second = item(2L);
        fixture.submission.completeSynchronously(
                first, SlotDeliveryPort.Completion.Delivered.INSTANCE);
        fixture.submission.completeSynchronously(
                second, SlotDeliveryPort.Completion.Delivered.INSTANCE);
        fixture.slots.beforeCompletion(() -> {
            assertEquals(1, fixture.admission.committedCloseCount(),
                    "endpoint handoff must close before callbacks open");
            assertEquals(1, fixture.submission.totalCloseCount(),
                    "transport preparation must close before callbacks open");
        });

        fixture.strategy.admitAndDeliver(
                List.of(first, second), new DeliveryMetadata("gate", 0),
                EVALUATOR, OptionalLong.of(20L), fixture.context);

        assertEquals(List.of(
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                first,
                                SlotDeliveryPort.Completion.Delivered.INSTANCE),
                        new DeliveryStrategyTestSupport.CompletionEvent(
                                second,
                                SlotDeliveryPort.Completion.Delivered.INSTANCE)),
                fixture.slots.completions());
    }

    @Test
    void callbackForUnsubmittedIdentityFailsClosed() {
        Fixture fixture = new Fixture(701L);
        TestItem canonical = item(1L);
        TestItem lookalike = new TestItem(
                canonical.requestId(), canonical.priority(),
                canonical.enqueuedAtMs(), canonical.seqLen(),
                canonical.hitCache());
        fixture.strategy.admitAndDeliver(
                List.of(canonical),
                new DeliveryMetadata("identity-fence", 0),
                EVALUATOR, OptionalLong.empty(), fixture.context);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> fixture.submission.complete(
                        lookalike,
                        SlotDeliveryPort.Completion.Delivered.INSTANCE));

        assertTrue(failure.getMessage().contains("unsubmitted identity"));
        assertTrue(fixture.slots.completions().isEmpty());
    }

    @Test
    void timeoutAndUncertainTransportOutcomesReachExactClaims() {
        Fixture fixture = new Fixture(701L);
        TestItem first = item(1L);
        TestItem second = item(2L);
        fixture.strategy.admitAndDeliver(
                List.of(first, second),
                new DeliveryMetadata("outcomes", 0),
                EVALUATOR, OptionalLong.empty(), fixture.context);
        RuntimeException timeout = new RuntimeException("timeout");
        RuntimeException uncertain = new RuntimeException("uncertain");

        fixture.submission.complete(
                first, new SlotDeliveryPort.Completion.TimedOut(timeout));
        fixture.submission.complete(
                second, new SlotDeliveryPort.Completion.Uncertain(uncertain));

        assertEquals(2, fixture.slots.completions().size());
        SlotDeliveryPort.Completion.TimedOut timedOut = assertInstanceOf(
                SlotDeliveryPort.Completion.TimedOut.class,
                fixture.slots.completions().get(0).completion());
        SlotDeliveryPort.Completion.Uncertain unresolved = assertInstanceOf(
                SlotDeliveryPort.Completion.Uncertain.class,
                fixture.slots.completions().get(1).completion());
        assertSame(timeout, timedOut.cause());
        assertSame(uncertain, unresolved.cause());
    }

    @Test
    void lostClaimExcludesOnlyThatMemberFromSubmittedBatch() {
        Fixture fixture = new Fixture(701L);
        TestItem first = item(1L);
        TestItem second = item(2L);
        fixture.slots.commitLostFor(first);

        fixture.strategy.admitAndDeliver(
                List.of(first, second),
                new DeliveryMetadata("claim-race", 0),
                EVALUATOR, OptionalLong.of(999L), fixture.context);

        assertEquals(List.of(second),
                fixture.submission.command().exactItems());
        assertEquals(100L, fixture.submission.command().predictedMs());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry =
                fixture.telemetry.batches().getFirst();
        assertEquals(List.of(second), telemetry.dispatched());
        assertEquals(100L, telemetry.predictedMs());
    }

    private static final class Fixture {
        private final TestBatchSubmissionPort submission =
                new TestBatchSubmissionPort();
        private final TestAdmissionPort admission = new TestAdmissionPort();
        private final TestSlotPort slots = new TestSlotPort();
        private final TestTelemetry telemetry = new TestTelemetry();
        private final TestContext context = new TestContext();
        private final BatchDeliveryStrategy strategy =
                new BatchDeliveryStrategy(
                        submission, admission, slots, telemetry);

        private Fixture(long correlationId) {
            this(OptionalLong.of(correlationId));
        }

        private Fixture(OptionalLong correlationId) {
            admission.correlationId(correlationId);
        }
    }
}
