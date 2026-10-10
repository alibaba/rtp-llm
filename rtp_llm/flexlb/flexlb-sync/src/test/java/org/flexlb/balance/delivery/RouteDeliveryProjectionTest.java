package org.flexlb.balance.delivery;

import org.flexlb.service.monitor.DeliveryMetricsReporter;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.BatchDeliveryStrategy;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

/**
 * Exact-value behavior contracts for the frozen
 * {@link RouteProjection.DeliveryProjection} SPI as implemented by
 * {@link RouteDeliveryStrategy} and {@link BatchDeliveryStrategy}.
 *
 * <p>Each selected group asks once for either the probe prefix or the complete
 * group. Completion may reuse the exact prefix evaluated by the planning cursor.
 */
@DisplayName("Delivery projection contracts")
class RouteDeliveryProjectionTest {

    private static GroupPlanner.Item item(long id, long seqLen) {
        return new GroupPlanner.Item(
                id, 0, id, 1000L, 1_000_000L, seqLen, 0L);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void growingPlanningPrefixReusesOnlyEvaluatedWorkAndResetsForNextQueue(boolean batch) {
        var requests = mock(AbstractRequestScheduler.class);
        var metrics = mock(DeliveryMetricsReporter.class);
        RouteProjection.DeliveryProjection projection = batch
                ? new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.OWNERSHIP_LOST), () -> 1L, metrics).projectionPolicy()
                : new RouteDeliveryStrategy(metrics).projectionPolicy();
        var predictions = new CountingPredictions();
        var first = item(1L, 100L);
        var second = item(2L, 200L);
        var cursor = projection.planning(predictions);
        assertEquals(batch ? 777.0 : 100.0, cursor.durationMs(List.of(first), 0));
        assertEquals(batch ? 777.0 : 100.0, cursor.durationMs(List.of(first, second), 0));
        assertEquals(1, batch ? predictions.batchPlanningCalls : predictions.itemCalls(1L));
        assertEquals(0, predictions.itemCalls(2L));
        assertEquals(batch ? 777.0 : 300.0, cursor.durationMs(List.of(first, second), 1));
        assertEquals(2, batch ? predictions.batchPlanningCalls
                : predictions.itemCalls(1L) + predictions.itemCalls(2L));

        var nextPredictions = new CountingPredictions();
        var next = projection.planning(nextPredictions);
        assertEquals(batch ? 777.0 : 900.0, next.durationMs(List.of(item(3L, 900L)), 0));
        assertEquals(1, batch ? nextPredictions.batchPlanningCalls : nextPredictions.itemCalls(3L));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void eachCompletionEvaluatesOnlyItsRequiredPrefix(boolean batch) {
        var projection = projection(batch);
        var items = List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
        long[] expected = batch ? new long[]{100L, 200L, 300L} : new long[]{100L, 300L, 350L};
        for (int index = 0; index < items.size(); index++) {
            var predictions = new CountingPredictions();
            assertEquals(expected[index], projection.completionOffsetMs(items, index, predictions, null));
            if (batch) {
                assertEquals(1, predictions.batchDurationCalls());
            } else {
                for (int member = 0; member < items.size(); member++) {
                    assertEquals(member <= index ? 1 : 0, predictions.itemCalls(member + 1L));
                }
            }
        }
    }

    @Test
    void probePrefixSurvivesASuffixPredictionFailure() {
        var projection = projection(false);
        var items = List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
        var predictions = new CountingPredictions();
        predictions.failOn(3L);
        assertEquals(300L, projection.completionOffsetMs(items, 1, predictions, null));
        assertEquals(0, predictions.itemCalls(3L));
        assertThrows(RuntimeException.class,
                () -> projection.completionOffsetMs(items, 2, predictions, null));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void memberIndexIsBoundsChecked(boolean batch) {
        var projection = projection(batch);
        var items = List.of(item(1L, 100L));
        var predictions = new CountingPredictions();
        assertThrows(IndexOutOfBoundsException.class,
                () -> projection.completionOffsetMs(items, -1, predictions, null));
        assertThrows(IndexOutOfBoundsException.class,
                () -> projection.completionOffsetMs(items, 1, predictions, null));
        assertEquals(0, predictions.itemCalls(1L));
        assertEquals(0, predictions.batchDurationCalls());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void planningEvaluatesOnlyThroughTheRequiredMember(boolean batch) {
        var projection = projection(batch);
        var items = List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
        var predictions = new CountingPredictions();
        assertEquals(batch ? 777.0 : 300.0, projection.planning(predictions).durationMs(items, 1));
        assertEquals(0, predictions.itemCalls(3L));
        assertEquals(batch ? 2 : 0, predictions.batchPlanningCalls);
    }

    @Test
    void routeCompletionReusesThePlannedProbePrefix() {
        var projection = projection(false);
        var items = List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
        var predictions = new CountingPredictions();
        var cursor = projection.planning(predictions);
        var plan = GroupPlanner.selectWithPrediction(items, new GroupPlanner.Constraints(3, 1_000L, 1_000L, 5_000L, 0L),
                (added, prefix) -> cursor.durationMs(prefix, Math.min(1, prefix.size() - 1)));

        assertEquals(items, plan.items());
        assertEquals(300L, projection.completionOffsetMs(plan.items(), 1, predictions, cursor));
        assertEquals(1, predictions.itemCalls(1L));
        assertEquals(1, predictions.itemCalls(2L));
        assertEquals(0, predictions.itemCalls(3L));
    }

    @Test
    void routeCompletionReusesTheAcceptedPrefixAfterAnOverBudgetAppend() {
        var projection = projection(false);
        var first = item(1L, 100L);
        var predictions = new CountingPredictions();
        var cursor = projection.planning(predictions);
        var plan = GroupPlanner.selectWithPrediction(List.of(first, item(2L, 200L), item(3L, 50L)),
                new GroupPlanner.Constraints(3, 1_000L, 1_000L, 250L, 0L),
                (added, prefix) -> cursor.durationMs(prefix, prefix.size() - 1));

        assertEquals(List.of(first), plan.items());
        assertEquals(100L, projection.completionOffsetMs(plan.items(), 0, predictions, cursor));
        assertEquals(1, predictions.itemCalls(1L));
        assertEquals(1, predictions.itemCalls(2L));
        assertEquals(0, predictions.itemCalls(3L));
    }

    @Test
    void routePlanningCacheKeepsExactLongValuesAndSaturates() {
        var projection = projection(false);
        long large = (1L << 53) + 3L;
        var items = List.of(item(1L, large), item(2L, 5L), item(3L, Long.MAX_VALUE));
        var predictions = new CountingPredictions();
        var cursor = projection.planning(predictions);

        cursor.durationMs(items, 1);
        assertEquals(large, projection.completionOffsetMs(items, 0, predictions, cursor));
        assertEquals(large + 5L, projection.completionOffsetMs(items, 1, predictions, cursor));
        cursor.durationMs(items, 2);
        assertEquals(large + 5L, projection.completionOffsetMs(items, 1, predictions, cursor));
        assertEquals(Long.MAX_VALUE, projection.completionOffsetMs(items, 2, predictions, cursor));
        for (long requestId = 1L; requestId <= 3L; requestId++) {
            assertEquals(1, predictions.itemCalls(requestId));
        }
    }

    @Test
    void routePlanningRetainsCompletedPrefixAcrossFailureAndSaturates() {
        var predictions = new CountingPredictions();
        var cursor = projection(false).planning(predictions);
        var items = List.of(item(1L, Long.MAX_VALUE - 10L), item(2L, 20L));
        predictions.failOn(2L);
        assertThrows(RuntimeException.class, () -> cursor.durationMs(items, 1));
        predictions.failOn();
        assertEquals((double) Long.MAX_VALUE, cursor.durationMs(items, 1));
        assertEquals(1, predictions.itemCalls(1L));
        assertEquals(2, predictions.itemCalls(2L));
        assertThrows(IllegalArgumentException.class, () -> cursor.durationMs(items, 0));
    }

    private static RouteProjection.DeliveryProjection projection(boolean batch) {
        var requests = mock(AbstractRequestScheduler.class);
        var metrics = mock(DeliveryMetricsReporter.class);
        return batch
                ? new BatchDeliveryStrategy(() -> CapacityBoundary.Attempt.rejected(
                        CapacityBoundary.OWNERSHIP_LOST), () -> 1L, metrics).projectionPolicy()
                : new RouteDeliveryStrategy(metrics).projectionPolicy();
    }

    private static final class CountingPredictions
            implements RouteProjection.Predictions {

        private final Map<Long, Integer> itemCalls = new HashMap<>();
        private Set<Long> failingIds = Set.of();
        private int batchDurationCalls;
        private int batchPlanningCalls;

        void failOn(long... ids) {
            var set = new java.util.HashSet<Long>();
            for (long id : ids) set.add(id);
            this.failingIds = set;
        }

        int itemCalls(long id) {
            return itemCalls.getOrDefault(id, 0);
        }

        int batchDurationCalls() {
            return batchDurationCalls;
        }

        @Override
        public long itemDurationMs(GroupPlanner.Item item) {
            itemCalls.merge(item.requestId(), 1, Integer::sum);
            if (failingIds.contains(item.requestId())) {
                throw new RuntimeException("fail " + item.requestId());
            }
            return item.seqLen(); // deterministic: seqLen as duration
        }

        @Override
        public long itemDurationMs(long seqLen, long hitCache) {
            return itemDurationMs(new GroupPlanner.Item(0L, 0, 0L, 0L, Long.MAX_VALUE, seqLen, hitCache));
        }

        @Override
        public long singletonBatchDurationMs(long seqLen, long hitCache) {
            return batchDurationMs(List.of(new GroupPlanner.Item(0L, 0, 0L, 0L, Long.MAX_VALUE, seqLen, hitCache)));
        }

        @Override
        public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
            return (seqLen, hitCache) -> {
                batchPlanningCalls++;
                return 777.0;
            };
        }

        @Override
        public long batchDurationMs(List<GroupPlanner.Item> items) {
            batchDurationCalls++;
            return 100L * items.size(); // discriminating: prefix-size dependent
        }

    }
}
