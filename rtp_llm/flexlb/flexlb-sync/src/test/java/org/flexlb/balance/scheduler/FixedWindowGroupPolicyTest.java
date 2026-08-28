package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.flexlb.balance.scheduler.GroupPolicyTestSupport.NOW_MS;
import static org.flexlb.balance.scheduler.GroupPolicyTestSupport.capacity;
import static org.flexlb.balance.scheduler.GroupPolicyTestSupport.fixed;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Final fixed-window grouping, scheduling-clock, and resource-boundary contract. */
class FixedWindowGroupPolicyTest {

    private final FixedWindowGroupPolicy policy = new FixedWindowGroupPolicy();

    @Test
    void emptyQueueHasNoActionAndNeverConsultsDelivery() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 10_000L, 500L);

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, result);
        assertTrue(fixture.delivery().attempts().isEmpty());
        assertTrue(fixture.delivery().projectedGroups().isEmpty());
    }

    @Test
    void incompleteWindowReturnsExactEventDrivenWait() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 10_000L, 0L);
        BatchItem head = fixture.add(
                1L, 50, 10L, NOW_MS - 5_000L, NOW_MS + 20_000L);
        fixture.bumpSchedulingInputVersion();

        BatcherCycleResult.AwaitingSchedulingChange waiting =
                assertInstanceOf(
                        BatcherCycleResult.AwaitingSchedulingChange.class,
                        policy.processQueue(fixture.context()));

        assertSame(head, waiting.head());
        assertEquals(1L, waiting.queueVersion());
        assertEquals(1L, waiting.schedulingInputVersion());
        assertEquals(NOW_MS + 5_000L, waiting.wakeAtMs());
        assertSame(BatcherCycleResult.SchedulingWaitReason.COLLECTION_WINDOW,
                waiting.reason());
        assertEquals(List.of(head), fixture.activeItems());
        assertTrue(fixture.delivery().attempts().isEmpty());
        assertTrue(fixture.delivery().projectedGroups().isEmpty());
    }

    @Test
    void collectionWaitIsCappedByAbsoluteHeadExpiry() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 10_000L, 0L);
        BatchItem head = fixture.add(
                1L, 50, 10L, NOW_MS - 1_000L, NOW_MS + 2_000L);

        BatcherCycleResult.AwaitingSchedulingChange waiting =
                assertInstanceOf(
                        BatcherCycleResult.AwaitingSchedulingChange.class,
                        policy.processQueue(fixture.context()));

        assertSame(head, waiting.head());
        assertEquals(NOW_MS + 2_000L, waiting.wakeAtMs());
        assertSame(BatcherCycleResult.SchedulingWaitReason.COLLECTION_WINDOW,
                waiting.reason());
    }

    @Test
    void zeroWindowAdmitsEveryCurrentlyFeasibleMember() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 0L);
        fixture.add(1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 10L, NOW_MS + 1L, Long.MAX_VALUE);
        fixture.add(3L, 50, 10L, NOW_MS + 2L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L, 3L), ids(admitted.items()));
        assertEquals(
                "fixed_window_timeout",
                admitted.metadata().decisionReason());
        assertEquals(List.of(List.of(1L, 2L, 3L)),
                fixture.delivery().committedGroups());
        assertTrue(fixture.activeItems().isEmpty());
    }

    @Test
    void predictionGrowthBeyondBudgetKeepsSuffixActive() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 60_000L, 500L);
        fixture.delivery().projection(items ->
                items.size() == 1 ? 499.0 : 501.0);
        fixture.add(1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        BatchItem suffix = fixture.add(
                2L, 50, 10L, NOW_MS + 1L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L), ids(admitted.items()));
        assertEquals(
                "predicted_execution_cap",
                admitted.metadata().decisionReason());
        assertEquals(List.of(List.of(1L), List.of(1L, 2L)),
                fixture.delivery().projectedGroups());
        assertEquals(List.of(OptionalLong.of(499L)),
                fixture.delivery().plannedPredictions());
        assertEquals(List.of(suffix), fixture.activeItems());
    }

    @Test
    void belowBudgetPredictionDispatchesOnlyWhenWindowElapsed() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 100L, 500L);
        fixture.delivery().projection(items -> 499.75);
        fixture.add(1L, 50, 10L, NOW_MS - 100L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(
                "fixed_window_timeout",
                admitted.metadata().decisionReason());
        assertEquals(List.of(OptionalLong.of(499L)),
                fixture.delivery().plannedPredictions());
        assertEquals(List.of(List.of(1L)),
                fixture.delivery().projectedGroups());
    }

    @Test
    void predictionExactlyAtBudgetKeepsEqualMemberAndDispatches() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 3, 60_000L, 500L);
        fixture.delivery().projection(items ->
                items.size() == 1 ? 100.0 : 500.0);
        fixture.add(1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 10L, NOW_MS + 1L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L), ids(admitted.items()));
        assertEquals(
                "predicted_execution_cap",
                admitted.metadata().decisionReason());
        assertEquals(List.of(OptionalLong.of(500L)),
                fixture.delivery().plannedPredictions());
    }

    @Test
    void singletonExactlyAtBudgetDispatchesBeforeWindow() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 60_000L, 500L);
        fixture.delivery().projection(items -> 500.0);
        fixture.add(1L, 50, 10L, NOW_MS, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L), ids(admitted.items()));
        assertEquals(
                "predicted_execution_cap",
                admitted.metadata().decisionReason());
    }

    @Test
    void maxRequestCountDispatchesLargestFeasibleGroup() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 60_000L, 500L);
        fixture.delivery().projection(items -> 499.0);
        for (int index = 1; index <= 4; index++) {
            fixture.add(index, 50, 10L,
                    NOW_MS + index, Long.MAX_VALUE);
        }

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L, 3L, 4L), ids(admitted.items()));
        assertEquals("batch_full", admitted.metadata().decisionReason());
        assertEquals(4, fixture.delivery().projectedGroups().size());
        assertTrue(fixture.activeItems().isEmpty());
    }

    @Test
    void nonMonotonicPredictionsStopAtFirstOverBudgetPrefix() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 5, 60_000L, 500L);
        fixture.delivery().projection(items -> switch (items.size()) {
            case 1 -> 100.0;
            case 2 -> 200.0;
            case 3 -> 600.0;
            default -> 50.0;
        });
        for (int index = 1; index <= 4; index++) {
            fixture.add(index, 50, 10L,
                    NOW_MS + index, Long.MAX_VALUE);
        }

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L), ids(admitted.items()));
        assertEquals(List.of(
                        List.of(1L),
                        List.of(1L, 2L),
                        List.of(1L, 2L, 3L)),
                fixture.delivery().projectedGroups());
        assertEquals(List.of(3L, 4L), ids(fixture.activeItems()));
    }

    @Test
    void computeLimitedPrefixWaitsUntilCollectionWindow() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 10_000L, 1_000L);
        fixture.status(capacity(250L, 0L, 0L, 0L));
        fixture.delivery().projection(items -> 100.0);
        fixture.add(1L, 50, 100L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 100L, NOW_MS + 1L, Long.MAX_VALUE);
        fixture.add(3L, 50, 100L, NOW_MS + 2L, Long.MAX_VALUE);

        BatcherCycleResult.AwaitingSchedulingChange waiting =
                assertInstanceOf(
                        BatcherCycleResult.AwaitingSchedulingChange.class,
                        policy.processQueue(fixture.context()));

        assertSame(BatcherCycleResult.SchedulingWaitReason.COLLECTION_WINDOW,
                waiting.reason());
        assertEquals(NOW_MS + 10_000L, waiting.wakeAtMs());
        assertTrue(fixture.delivery().attempts().isEmpty());
        assertEquals(List.of(1L, 2L, 3L), ids(fixture.activeItems()));
    }

    @Test
    void elapsedWindowDispatchesLargestComputeFeasiblePrefix() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 1_000L);
        fixture.status(capacity(250L, 0L, 0L, 0L));
        fixture.delivery().projection(items -> 100.0);
        fixture.add(1L, 50, 100L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 100L, NOW_MS + 1L, Long.MAX_VALUE);
        BatchItem suffix = fixture.add(
                3L, 50, 100L, NOW_MS + 2L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L), ids(admitted.items()));
        assertEquals(List.of(suffix), fixture.activeItems());
    }

    @Test
    void deliveryCapacityBlockLeavesExactPredictedHeadActive() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 1, 0L, 500L);
        fixture.delivery().projection(items -> 125.0);
        fixture.delivery().block();
        BatchItem head = fixture.add(
                1L, 50, 10L, NOW_MS, Long.MAX_VALUE);

        BatcherCycleResult.CapacityBlocked blocked = assertInstanceOf(
                BatcherCycleResult.CapacityBlocked.class,
                policy.processQueue(fixture.context()));

        assertSame(head, blocked.item());
        assertEquals(List.of(List.of(1L)), fixture.delivery().attempts());
        assertEquals(List.of(List.of(1L)),
                fixture.delivery().projectedGroups());
        assertEquals(List.of(head), fixture.activeItems());
    }

    @Test
    void invalidPredictionTerminalizesOnlyItsExactHead() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 500L);
        fixture.delivery().projection(items -> Double.NaN);
        BatchItem head = fixture.add(
                1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        BatchItem suffix = fixture.add(
                2L, 50, 10L, NOW_MS + 1L, Long.MAX_VALUE);

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.QUEUE_CHANGED, result);
        assertEquals(List.of(head),
                fixture.lifecycle().deliveryFailureItems());
        assertInstanceOf(InvalidPrefillPredictionException.class,
                fixture.lifecycle().deliveryFailures().getFirst());
        assertEquals(List.of(suffix), fixture.activeItems());
        assertTrue(fixture.delivery().attempts().isEmpty());
    }

    @Test
    void enginePaddedTokenShapeIsStrict() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 3, 0L, 1_000L);
        fixture.status(capacity(250L, 0L, 0L, 0L));
        fixture.delivery().projection(items -> 100.0);
        fixture.add(1L, 50, 100L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 100L, NOW_MS + 1L, Long.MAX_VALUE);
        BatchItem third = fixture.add(
                3L, 50, 100L, NOW_MS + 2L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L), ids(admitted.items()));
        assertEquals(List.of(third), fixture.activeItems());
    }

    @Test
    void largeRequestDispatchesAloneWhenPairWouldOverflow() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 1_000L);
        fixture.status(capacity(300L, 0L, 0L, 0L));
        fixture.delivery().projection(items -> 100.0);
        fixture.add(1L, 50, 200L, NOW_MS, Long.MAX_VALUE);
        BatchItem second = fixture.add(
                2L, 50, 200L, NOW_MS + 1L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L), ids(admitted.items()));
        assertEquals(List.of(second), fixture.activeItems());
    }

    @Test
    void kvShapeLimitsGroupPrefixWithoutRejectingSuffix() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 1_000L);
        fixture.status(capacity(1_000L, 0L, 250L, 250L));
        fixture.delivery().projection(items -> 100.0);
        fixture.add(1L, 50, 100L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 100L, NOW_MS + 1L, Long.MAX_VALUE);
        BatchItem third = fixture.add(
                3L, 50, 100L, NOW_MS + 2L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L), ids(admitted.items()));
        assertEquals(List.of(third), fixture.activeItems());
        assertTrue(fixture.lifecycle().offerFailureItems().isEmpty());
    }

    @Test
    void maxSeqLenIsFallbackWhenBatchTokenLimitIsUnpublished() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 1, 0L, 0L);
        fixture.status(capacity(0L, 150L, 0L, 0L));
        fixture.add(1L, 50, 100L, NOW_MS, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L), ids(admitted.items()));
        assertTrue(fixture.lifecycle().offerFailureItems().isEmpty());
    }

    @Test
    void requestAtStrictEngineTokenLimitIsRejectedBeforePrediction() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 500L);
        fixture.status(capacity(100L, 0L, 0L, 0L));
        BatchItem head = fixture.add(
                1L, 50, 100L, NOW_MS, Long.MAX_VALUE);

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.QUEUE_CHANGED, result);
        assertEquals(List.of(head),
                fixture.lifecycle().offerFailureItems());
        assertInstanceOf(BatchTokenCapacityExceededException.class,
                fixture.lifecycle().offerFailures().getFirst());
        assertTrue(fixture.delivery().projectedGroups().isEmpty());
        assertTrue(fixture.activeItems().isEmpty());
    }

    @Test
    void offerDuringPredictionBelongsToNextStableSnapshot() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                true, 2, 60_000L, 500L);
        fixture.add(1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 40, 10L, NOW_MS + 1L, Long.MAX_VALUE);
        AtomicBoolean offered = new AtomicBoolean();
        BatchItem urgent = fixture.item(
                3L, 100, 10L, NOW_MS + 2L, Long.MAX_VALUE);
        fixture.delivery().projection(items -> {
            if (offered.compareAndSet(false, true)) {
                fixture.add(urgent);
            }
            return 100.0;
        });

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(1L, 2L), ids(admitted.items()));
        assertEquals(List.of(urgent), fixture.activeItems());
    }

    @Test
    void removalDuringPredictionInvalidatesWholeCapturedGroup() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 2, 60_000L, 500L);
        BatchItem first = fixture.add(
                1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        BatchItem second = fixture.add(
                2L, 50, 10L, NOW_MS + 1L, Long.MAX_VALUE);
        AtomicBoolean removed = new AtomicBoolean();
        fixture.delivery().projection(items -> {
            if (items.size() == 2 && removed.compareAndSet(false, true)) {
                assertTrue(fixture.remove(second));
            }
            return 100.0;
        });

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, result);
        assertEquals(List.of(first), fixture.activeItems());
        assertTrue(fixture.delivery().committedGroups().isEmpty());
        assertTrue(fixture.lifecycle().deliveryFailureItems().isEmpty());
        assertEquals(List.of(List.of(1L, 2L)),
                fixture.delivery().attempts());
    }

    @Test
    void prioritySnapshotAdmitsStrictOrderedPrefixWithoutStarvation() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                true, 2, 0L, 0L);
        BatchItem low = fixture.add(
                1L, 10, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 100, 10L, NOW_MS + 1L, Long.MAX_VALUE);
        fixture.add(3L, 50, 10L, NOW_MS + 2L, Long.MAX_VALUE);

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(List.of(2L, 3L), ids(admitted.items()));
        assertEquals(List.of(low), fixture.activeItems());
    }

    @Test
    void expiredMemberInsideCapturedPrefixIsRemovedBeforePrediction() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 3, 0L, 500L);
        BatchItem head = fixture.add(
                1L, 50, 10L, NOW_MS - 10L, Long.MAX_VALUE);
        BatchItem expired = fixture.add(
                2L, 50, 10L, NOW_MS - 9L, NOW_MS);

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.QUEUE_CHANGED, result);
        assertEquals(List.of(expired), fixture.lifecycle().expired());
        assertEquals(List.of(head), fixture.activeItems());
        assertTrue(fixture.delivery().projectedGroups().isEmpty());
    }

    @Test
    void requestExpiringDuringPredictionCannotCommit() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 1, 0L, 500L);
        BatchItem expiring = fixture.add(
                1L, 50, 10L, NOW_MS, NOW_MS + 1L);
        fixture.delivery().projection(items -> {
            fixture.advanceTo(NOW_MS + 1L);
            return 500.0;
        });

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, result);
        assertEquals(List.of(expiring), fixture.activeItems());
        assertTrue(fixture.delivery().committedGroups().isEmpty());
    }

    @Test
    void windowElapsedDuringPredictionDispatchesInSamePass() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 2, 20L, 500L);
        fixture.add(1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.delivery().projection(items -> {
            fixture.advanceTo(NOW_MS + 25L);
            return 100.0;
        });

        BatcherCycleResult.Admitted admitted = assertInstanceOf(
                BatcherCycleResult.Admitted.class,
                policy.processQueue(fixture.context()));

        assertEquals(
                "fixed_window_timeout",
                admitted.metadata().decisionReason());
        assertEquals(List.of(1L), ids(admitted.items()));
    }

    @Test
    void decisionKeepsEvaluatorSnapshotWhenEndpointPublishesReplacement() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 1, 0L, 500L);
        PrefillTimePredictor.Evaluator captured = fixture.evaluator();
        PrefillTimePredictor.Evaluator replacement =
                new PrefillTimePredictor.Evaluator() {
                    @Override
                    public long estimateMs(long totalTokens, long hitTokens) {
                        return 999L;
                    }

                    @Override
                    public double predictBatchMs(
                            org.flexlb.balance.prediction.PrefillBatchFeatures
                                    features) {
                        return 999.0;
                    }
                };
        fixture.add(1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.delivery().projection(items -> {
            fixture.evaluator(replacement);
            return 500.0;
        });

        policy.processQueue(fixture.context());

        assertEquals(1, fixture.delivery().evaluators().size());
        assertSame(captured, fixture.delivery().evaluators().getFirst());
    }

    @Test
    void capacityBecomingUnavailableDuringPredictionBlocksAtDeliveryGate() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 1, 0L, 500L);
        BatchItem head = fixture.add(
                1L, 50, 10L, NOW_MS, Long.MAX_VALUE);
        fixture.delivery().projection(items -> {
            fixture.delivery().block();
            return 500.0;
        });

        BatcherCycleResult.CapacityBlocked result = assertInstanceOf(
                BatcherCycleResult.CapacityBlocked.class,
                policy.processQueue(fixture.context()));

        assertSame(head, result.item());
        assertEquals(List.of(List.of(1L)), fixture.delivery().attempts());
        assertEquals(List.of(head), fixture.activeItems());
    }

    @Test
    void computeCapacityDropDuringPredictionPreventsAdmission() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 2, 0L, 500L);
        WorkerStatus.EngineObservation reduced =
                capacity(200L, 0L, 0L, 0L);
        fixture.status(capacity(1_000L, 0L, 0L, 0L));
        fixture.add(1L, 50, 100L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 100L, NOW_MS + 1L, Long.MAX_VALUE);
        fixture.delivery().projection(items -> {
            fixture.status(reduced);
            return 100.0;
        });

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, result);
        assertTrue(fixture.delivery().attempts().isEmpty());
        assertEquals(List.of(1L, 2L), ids(fixture.activeItems()));
    }

    @Test
    void kvCapacityDropDuringPredictionPreventsAdmission() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 2, 0L, 500L);
        fixture.status(capacity(1_000L, 0L, 1_000L, 1_000L));
        fixture.add(1L, 50, 100L, NOW_MS, Long.MAX_VALUE);
        fixture.add(2L, 50, 100L, NOW_MS + 1L, Long.MAX_VALUE);
        fixture.delivery().projection(items -> {
            fixture.status(capacity(1_000L, 0L, 1_000L, 199L));
            return 100.0;
        });

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.NO_ACTION, result);
        assertTrue(fixture.delivery().attempts().isEmpty());
        assertEquals(List.of(1L, 2L), ids(fixture.activeItems()));
    }

    @Test
    void singletonKvBlockWaitsWithoutPredictionOrPolling() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 500L);
        fixture.status(capacity(1_000L, 0L, 1_000L, 99L));
        BatchItem head = fixture.add(
                1L, 50, 100L, NOW_MS, NOW_MS + 10_000L);

        BatcherCycleResult.AwaitingSchedulingChange waiting =
                assertInstanceOf(
                        BatcherCycleResult.AwaitingSchedulingChange.class,
                        policy.processQueue(fixture.context()));

        assertSame(head, waiting.head());
        assertEquals(NOW_MS + 10_000L, waiting.wakeAtMs());
        assertSame(BatcherCycleResult.SchedulingWaitReason.PREFILL_KV_CAPACITY,
                waiting.reason());
        assertEquals(1, fixture.statusReads());
        assertTrue(fixture.delivery().projectedGroups().isEmpty());
        assertTrue(fixture.delivery().attempts().isEmpty());
    }

    @Test
    void expiredHeadIsTerminalizedAfterOneAdvisoryResourceRead() {
        GroupPolicyTestSupport.Fixture fixture = fixed(
                false, 4, 0L, 500L);
        BatchItem expired = fixture.add(
                1L, 50, 10L, NOW_MS - 10L, NOW_MS);

        BatcherCycleResult result = policy.processQueue(fixture.context());

        assertSame(BatcherCycleResult.Outcome.QUEUE_CHANGED, result);
        assertEquals(List.of(expired), fixture.lifecycle().expired());
        assertEquals(1, fixture.statusReads());
        assertTrue(fixture.delivery().projectedGroups().isEmpty());
        assertTrue(fixture.activeItems().isEmpty());
    }

    private static List<Long> ids(List<? extends DeliveryItem> items) {
        return items.stream().map(DeliveryItem::requestId).toList();
    }
}
