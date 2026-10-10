package org.flexlb.balance.scheduler;

import static org.flexlb.balance.scheduler.DeliveryStrategy.failUnsentDelivery;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestBatchSubmission;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestContext;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestEndpointCapabilities;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestRequestScheduler;
import org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.TestTelemetry;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import static org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.unavailable;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.mock;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;

/** Final batch admission, transport handoff, and completion-correlation contract. */
class BatchDeliveryStrategyTest {

    @Test
    void preparedPredictionAndExactBatchReachTransportOnce() {
        Fixture fixture = new Fixture(701L);
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
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
        assertTrue(fixture.schedulerFixture.identities().stream().allMatch(identity ->
                identity.kind() == DeliveryClaimKind.BATCH_ENQUEUE
                        && identity.correlationId() == 701L));
        assertEquals(1, fixture.telemetry.batches().size());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry =
                fixture.telemetry.batches().getFirst();
        assertEquals(701L, telemetry.batchId());
        assertEquals(List.of(first, second), telemetry.dispatched());
        assertEquals(83L, telemetry.predictedMs());
        assertEquals(3, telemetry.remainingQueueDepth());
        verify(fixture.capabilities.batchReservation()).commitLocked(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(83L), org.mockito.ArgumentMatchers.isNull(), org.mockito.ArgumentMatchers.anyLong());
        verify(fixture.capabilities.permit(first)).dispatch();
        verify(fixture.capabilities.permit(second)).dispatch();
        assertEquals(1, fixture.capabilities.handoffs().size());
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
        assertEquals(1, fixture.submission.totalCloseCount());
    }

    @Test
    void missingPlannedPredictionUsesFrozenEvaluatorForCommittedBatch() {
        Fixture fixture = new Fixture(702L);
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);

        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "predict", 0, OptionalLong.empty());

        assertEquals(200L, fixture.submission.command().predictedMs());
        assertEquals(200L, fixture.telemetry.batches()
                .getFirst().predictedMs());
        verify(fixture.capabilities.batchReservation()).commitLocked(
                org.mockito.ArgumentMatchers.eq(List.of(first, second)),
                org.mockito.ArgumentMatchers.eq(200L), org.mockito.ArgumentMatchers.isNull(), org.mockito.ArgumentMatchers.anyLong());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void predictionFailureRollsBackAllPreparedResources(boolean cleanupFails) {
        Fixture fixture = new Fixture(701L);
        RequestRoute first = fixture.item(1L), second = fixture.item(2L);
        var evaluator = mock(org.flexlb.balance.prediction.PrefillTimePredictor.Evaluator.class);
        var primary = new IllegalStateException("batch prediction failed");
        var cleanup = new IllegalStateException("permit cleanup failed");
        org.mockito.Mockito.when(evaluator.predictBatchMs(any())).thenAnswer(invocation -> {
            // Permits are acquired lazily by prepare, before evaluating the complete batch.
            if (cleanupFails) { doThrow(cleanup).when(fixture.capabilities.permit(first)).release(); }
            throw primary;
        });

        assertSame(primary, assertThrows(IllegalStateException.class,
                () -> fixture.strategy.prepare(List.of(first, second), evaluator, OptionalLong.empty())));

        for (var item : List.of(first, second)) {
            verify(fixture.capabilities.permit(item)).release();
            verify(fixture.capabilities.permit(item), never()).dispatch();
        }
        verify(fixture.capabilities.prefill()).rollbackReservation(fixture.capabilities.batchReservation());
        assertEquals(1, fixture.submission.totalCloseCount());
        assertEquals(cleanupFails ? List.of(cleanup) : List.of(), List.of(primary.getSuppressed()));
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
    }

    @Test
    void nonPositiveBatchIdClosesSubmissionBeforeEndpointOwnership() {
        Fixture fixture = new Fixture(0L);
        RequestRoute item = fixture.item(1L);

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
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedPreparationClosesSubmissionOnceAndPreservesCause(boolean cleanupFails) {
        Fixture fixture = new Fixture(701L);
        var submission = mock(DefaultBatchDispatcher.PreparedSubmission.class);
        var primary = new IllegalStateException("batch id allocation failed");
        var cleanup = new IllegalStateException("submission close failed");
        if (cleanupFails) {
            doThrow(cleanup).when(submission).close();
        }
        var strategy = new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.accepted(submission), () -> { throw primary; }, fixture.telemetry.metrics());

        var transaction = strategy.prepare(List.of(fixture.item(1L)),
                DeliveryStrategyTestSupport.EVALUATOR, OptionalLong.empty());

        assertThrows(IllegalStateException.class, () -> transaction.commitSelectionLocked(System.currentTimeMillis()));
        transaction.close();
        assertEquals(CapacityBoundary.Status.FAILED, transaction.blockedResult().status());
        assertSame(primary, transaction.blockedResult().cause());
        assertEquals(cleanupFails ? List.of(cleanup) : List.of(), List.of(primary.getSuppressed()));
        verify(submission).close();
        verify(submission, never()).submit(any());
        assertTrue(fixture.capabilities.handoffs().isEmpty());
    }

    @Test
    void unavailableSubmissionReturnsExactHeadBoundaryBeforeAdmission() {
        Fixture fixture = new Fixture(701L);
        RequestRoute head = fixture.item(1L);
        CapacityBoundary unavailable = unavailable();
        fixture.submission.prepareBoundary(unavailable);

        String result = fixture.context.deliver(
                fixture.strategy, List.of(head),
                "submission-full", 0,
                OptionalLong.empty());

        assertEquals("BOUNDARY", result);
        assertSame(head, fixture.context.emptyBoundary().item());
        assertSame(unavailable, fixture.context.emptyBoundary().result());
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
    }

    @Test
    void unavailableAdmissionClosesPreparedSubmissionAndReturnsBoundary() {
        Fixture fixture = new Fixture(701L);
        RequestRoute head = fixture.item(1L);
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
        verify(fixture.capabilities.prefill()).rollbackReservation(fixture.capabilities.batchReservation());
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
    }

    @Test
    void unavailableSuffixSubmitsLargestAdmittedPrefixAndRepredictsIt() {
        Fixture fixture = new Fixture(701L);
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
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
        assertEquals(List.of(first), fixture.schedulerFixture.committed());
    }

    @Test
    void synchronousTransportCompletionWaitsForCapabilityHandoffClose() {
        Fixture fixture = new Fixture(701L);
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        fixture.submission.completeSynchronously(
                first, DeliveryResult.delivered());
        fixture.submission.completeSynchronously(
                second, DeliveryResult.delivered());
        fixture.schedulerFixture.beforeCompletion(() -> {
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
                fixture.schedulerFixture.completions());
    }

    @Test
    void callbackForUnsubmittedIdentityFailsClosed() {
        Fixture fixture = new Fixture(701L);
        RequestRoute canonical = fixture.item(1L);
        RequestRoute lookalike = DeliveryStrategyTestSupport.item(
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
        assertTrue(fixture.schedulerFixture.completions().isEmpty());
    }

    @Test
    void timeoutAndUncertainTransportOutcomesReachExactClaims() {
        Fixture fixture = new Fixture(701L);
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        fixture.context.deliver(
                fixture.strategy, List.of(first, second),
                "outcomes", 0, OptionalLong.empty());
        var timeout = new java.util.concurrent.TimeoutException("timeout");
        RuntimeException uncertain = new RuntimeException("uncertain");

        fixture.submission.complete(
                first, DeliveryResult.uncertain(timeout));
        fixture.submission.complete(
                second, DeliveryResult.uncertain(uncertain));

        assertEquals(2, fixture.schedulerFixture.completions().size());
        DeliveryResult timedOut =
                fixture.schedulerFixture.completions().get(0).completion();
        DeliveryResult unresolved =
                fixture.schedulerFixture.completions().get(1).completion();
        assertEquals(DeliveryResult.Status.UNCERTAIN,
                timedOut.status());
        assertEquals(DeliveryResult.Status.UNCERTAIN,
                unresolved.status());
        assertSame(timeout, timedOut.cause());
        assertSame(uncertain, unresolved.cause());
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 1})
    void lostClaimExcludesOnlyThatMemberFromSubmittedBatch(int lostIndex) {
        Fixture fixture = new Fixture(701L);
        List<RequestRoute> items = List.of(fixture.item(1L), fixture.item(2L), fixture.item(3L));
        RequestRoute lost = items.get(lostIndex);
        List<RequestRoute> submitted = items.stream().filter(item -> item != lost).toList();
        fixture.schedulerFixture.commitLostFor(lost);
        fixture.capabilities.precedingWork(new WorkSnapshot(1_000L, List.of(new WorkSnapshot.RequestWork(99L, WorkSnapshot.Phase.COMMITTED, 25L)), List.of(), 0L));

        fixture.context.deliver(fixture.strategy, items, "claim-race", 0, OptionalLong.of(999L));

        assertEquals(submitted, fixture.submission.command().exactItems());
        assertEquals(200L, fixture.submission.command().predictedMs());
        DeliveryStrategyTestSupport.BatchTelemetry telemetry = fixture.telemetry.batches().getFirst();
        assertEquals(submitted, telemetry.dispatched());
        assertEquals(200L, telemetry.predictedMs());
        assertEquals(Map.of(submitted.get(0), 200L, submitted.get(1), 200L),
                fixture.schedulerFixture.unstartedWorkMs(),
                "cancelled members must not inflate the delivered batch lifetime");
        for (RequestRoute item : submitted) {
            assertEquals(225L, fixture.schedulerFixture.remainingWorkMsAt(item, 1_000L).orElseThrow());
            verify(fixture.capabilities.permit(item)).dispatch();
        }
        verify(fixture.capabilities.permit(lost), never()).dispatch();
        verify(fixture.capabilities.permit(lost)).release();
    }

    @Test
    void deferredExecutorRepredictsCancelledMembersUsingTheDecisionEvaluator() {
        Fixture fixture = new Fixture(701L);
        List<RequestRoute> items = List.of(fixture.item(1L), fixture.item(2L));
        var scheduled = new java.util.concurrent.atomic.AtomicReference<DefaultBatchDispatcher.Delivery>();
        var submission = mock(DefaultBatchDispatcher.PreparedSubmission.class);
        doAnswer(invocation -> { scheduled.set(invocation.getArgument(0)); return null; })
                .when(submission).submit(any());
        var strategy = new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.accepted(submission),
                () -> 701L, fixture.telemetry.metrics());
        var evaluator = mock(org.flexlb.balance.prediction.PrefillTimePredictor.Evaluator.class);
        org.mockito.Mockito.when(evaluator.predictBatchMs(any())).thenAnswer(invocation ->
                invocation.getArgument(0, org.flexlb.balance.prediction.PrefillBatchFeatures.class).items().size() * 37.0);

        try (var transaction = strategy.prepare(items, evaluator, OptionalLong.empty())) {
            assertEquals(74L, transaction.batch.predictedMs);
            var preceding = transaction.commitSelectionLocked(System.currentTimeMillis()).precedingWork().materialize();
            strategy.deliver(transaction, "deferred-cancellation", 0, preceding, evaluator);
            fixture.schedulerFixture.commitLostFor(items.get(1));
            scheduled.get().run((submitted, batchId, predictedMs, reason, completion) -> {
                assertEquals(List.of(items.getFirst()), submitted);
                assertEquals(701L, batchId);
                assertEquals(37L, predictedMs);
                completion.accept(submitted.getFirst(), DeliveryResult.delivered());
                assertTrue(fixture.schedulerFixture.completions().isEmpty(), "early ACK must wait for handoff cleanup");
            });
        }

        verify(evaluator, org.mockito.Mockito.times(2)).predictBatchMs(any());
        assertEquals(Map.of(items.getFirst(), 37L), fixture.schedulerFixture.unstartedWorkMs());
        assertEquals(1, fixture.schedulerFixture.completions().size());
        verify(fixture.capabilities.permit(items.get(1)), never()).dispatch();
        verify(fixture.capabilities.permit(items.get(1))).release();
        verify(submission).close();
    }

    @Test
    void zeroPredictionStillSubmitsTheBatch() {
        Fixture fixture = new Fixture(701L);
        RequestRoute item = fixture.item(1L);

        fixture.context.deliver(fixture.strategy, List.of(item),
                "zero", 0, OptionalLong.of(0L));

        assertEquals(List.of(item), fixture.submission.command().exactItems());
        assertEquals(Map.of(item, 0L), fixture.schedulerFixture.unstartedWorkMs());
    }

    @Test
    void unknownPrecedingWorkStillSubmitsWithUnknownRemainingTime() {
        Fixture fixture = new Fixture(701L);
        RequestRoute item = fixture.item(1L);
        fixture.capabilities.precedingWork(new WorkSnapshot(2_000L, List.of(), List.of(), 1L));

        fixture.context.deliver(fixture.strategy, List.of(item),
                "unknown", 0, OptionalLong.of(100L));

        assertEquals(List.of(item), fixture.submission.command().exactItems());
        assertEquals(Map.of(item, 100L),
                fixture.schedulerFixture.unstartedWorkMs());
        assertTrue(fixture.schedulerFixture.remainingWorkMsAt(item, 2_000L).isEmpty());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preparationFailureReleasesAllOwnedResourcesAndPreservesCause(boolean cleanupFails) {
        Fixture fixture = new Fixture(701L);
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
        verify(fixture.capabilities.prefill()).rollbackReservation(fixture.capabilities.batchReservation());
        assertEquals(1, fixture.submission.totalCloseCount());
        assertEquals(cleanupFails ? List.of(cleanup) : List.of(), List.of(primary.getSuppressed()));
        assertTrue(fixture.schedulerFixture.committed().isEmpty());
    }

    @Test
    void preparedMembersAreFrozenUntilCommitOrClose() {
        Fixture fixture = new Fixture(701L);
        RequestRoute first = fixture.item(1L);
        RequestRoute late = fixture.item(2L);
        try (var transaction = (DeliveryTransaction) fixture.strategy.prepare(
                List.of(first), DeliveryStrategyTestSupport.EVALUATOR, OptionalLong.of(10L))) {
            assertEquals(List.of(first), transaction.items());
            assertThrows(UnsupportedOperationException.class, () -> transaction.items().set(0, late));
            assertThrows(IllegalStateException.class, () -> transaction.append(late, 0L, null, null));
        }
        verify(fixture.capabilities.permit(first)).release();
        verify(fixture.capabilities.prefill()).rollbackReservation(fixture.capabilities.batchReservation());
        assertEquals(1, fixture.submission.closeCount());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedMemberCaptureReleasesItsPermitAndOtherPreparedResources(boolean cleanupFails) throws Exception {
        Fixture fixture = new Fixture(701L);
        RequestRoute item = fixture.item(1L);
        var primary = new IllegalStateException("member capture failed");
        var cleanup = new IllegalStateException("permit cleanup failed");
        try (var transaction = new DeliveryTransaction(List.of(item), true)) {
            @SuppressWarnings("unchecked")
            var members = org.mockito.Mockito.spy((java.util.ArrayList<DeliveryTransaction.Member>)
                    org.springframework.test.util.ReflectionTestUtils.getField(transaction, "members"));
            doAnswer(capture -> {
                if (cleanupFails) { doThrow(cleanup).when(fixture.capabilities.permit(item)).release(); }
                throw primary;
            }).when(members).add(any());
            org.springframework.test.util.ReflectionTestUtils.setField(transaction, "members", members);
            var boundary = transaction.append(item, 0L, fixture.submission::tryPrepareSubmission, () -> 701L);
            assertTrue(transaction.items().isEmpty());
            assertSame(primary, boundary.cause());
            assertEquals(cleanupFails ? List.of(cleanup) : List.of(), List.of(primary.getSuppressed()));
        }
        verify(fixture.capabilities.permit(item)).release();
        verify(fixture.capabilities.permit(item), never()).dispatch();
        verify(fixture.capabilities.prefill()).rollbackReservation(fixture.capabilities.batchReservation());
        assertEquals(1, fixture.submission.totalCloseCount());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedCommittedDeliverySettlesEveryMemberDespiteCleanupFailures(boolean submitRejected) {
        Fixture fixture = new Fixture(701L);
        RequestRoute first = fixture.item(1L);
        RequestRoute second = fixture.item(2L);
        var submission = mock(DefaultBatchDispatcher.PreparedSubmission.class);
        var cause = new IllegalStateException("delivery abandoned");
        var submissionFailure = new IllegalStateException("submission cleanup failed");
        var memberFailure = new IllegalStateException("first member cleanup failed");
        doThrow(cause).when(submission).submit(any());
        doThrow(submissionFailure).when(submission).close();
        doThrow(memberFailure).when(fixture.schedulerFixture.scheduler()).failDeliveryPreparation(first, cause);
        var strategy = new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.accepted(submission), () -> 701L, fixture.telemetry.metrics());

        try (var transaction = strategy.prepare(List.of(first, second),
                DeliveryStrategyTestSupport.EVALUATOR, OptionalLong.of(20L))) {
            var preceding = transaction.commitSelectionLocked(System.currentTimeMillis()).precedingWork().materialize();
            if (submitRejected) {
                assertSame(cause, assertThrows(IllegalStateException.class,
                        () -> strategy.deliver(transaction, "rejected", 0, preceding, DeliveryStrategyTestSupport.EVALUATOR)));
                assertEquals(List.of(submissionFailure), List.of(cause.getSuppressed()));
            } else {
                assertSame(submissionFailure, assertThrows(IllegalStateException.class,
                        () -> failUnsentDelivery(transaction, cause, false)));
            }
            failUnsentDelivery(transaction, cause, false);
        }

        assertEquals(List.of(memberFailure), List.of(submissionFailure.getSuppressed()));
        verify(submission).close();
        verify(fixture.schedulerFixture.scheduler()).failDeliveryPreparation(first, cause);
        verify(fixture.schedulerFixture.scheduler()).failDeliveryPreparation(second, cause);
        verify(fixture.capabilities.permit(first)).release();
        verify(fixture.capabilities.permit(second)).release();
        fixture.capabilities.handoffs().forEach(handoff -> verify(handoff).close());
        assertTrue(fixture.schedulerFixture.completions().isEmpty());
    }

    private static final class Fixture {
        private final TestBatchSubmission submission =
                new TestBatchSubmission();
        private final TestEndpointCapabilities capabilities =
                new TestEndpointCapabilities();
        private final TestRequestScheduler schedulerFixture = new TestRequestScheduler();
        private final TestTelemetry telemetry = new TestTelemetry();
        private final TestContext context = new TestContext();
        private final long correlationId;
        private final BatchDeliveryStrategy strategy;

        private Fixture(long correlationId) {
            this.correlationId = correlationId;
            this.strategy = new BatchDeliveryStrategy(submission::tryPrepareSubmission, () -> this.correlationId, telemetry.metrics());
        }

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
