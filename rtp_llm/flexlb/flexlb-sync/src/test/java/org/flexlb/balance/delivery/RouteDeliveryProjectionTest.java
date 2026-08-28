package org.flexlb.balance.delivery;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.RouteProjection;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

/**
 * Exact-value behavior contracts for the frozen
 * {@link RouteProjection.DeliveryProjection} SPI as implemented by
 * {@link RouteDeliveryStrategy} and {@link BatchDeliveryStrategy}.
 *
 * <p>Route service is a lazy cumulative prefix sum (asking for member i
 * evaluates only items 0..i). Batch service independently predicts each
 * prefix length via {@code batchDurationMs(items[0..i+1])}.
 */
@DisplayName("Delivery projection contracts")
class RouteDeliveryProjectionTest {

    private static GroupPlanner.Item item(long id, long seqLen) {
        return new GroupPlanner.Item(
                id, 0, id, 1000L, 1_000_000L, seqLen, 0L);
    }

    private static GroupPlanner.Plan<GroupPlanner.Item> plan(
            List<GroupPlanner.Item> items, OptionalDouble selectedPredictionMs) {
        GroupPlanner.Shape shape = GroupPlanner.Shape.empty();
        for (GroupPlanner.Item it : items) shape = shape.add(it.seqLen());
        return new GroupPlanner.Plan<>(items, shape, 1000L, 1300L, false,
                selectedPredictionMs, GroupPlanner.WindowReadiness.READY,
                GroupPlanner.BATCH_FULL);
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Route projection: lazy cumulative cursor")
    class RouteService {

        private final RouteProjection.DeliveryProjection projection =
                new RouteDeliveryStrategy(
                        mock(PrefillAdmissionPort.class),
                        mock(SlotDeliveryPort.class),
                        mock(DeliveryMetrics.class)).projectionPolicy();

        @Test
        void completionOffsetIsTheCumulativePrefixSum() {
            List<GroupPlanner.Item> items =
                    List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
            CountingPredictions pred = new CountingPredictions();
            RouteProjection.GroupService svc =
                    projection.service(plan(items, OptionalDouble.empty()), pred);
            // offset(0)=100, offset(1)=100+200=300, offset(2)=300+50=350
            assertEquals(100L, svc.completionOffsetMs(0));
            assertEquals(300L, svc.completionOffsetMs(1));
            assertEquals(350L, svc.completionOffsetMs(2));
            assertEquals(350L, svc.totalDurationMs());
        }

        @Test
        void probePrefixDoesNotEvaluateTheSuffix() {
            List<GroupPlanner.Item> items =
                    List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
            CountingPredictions pred = new CountingPredictions();
            RouteProjection.GroupService svc =
                    projection.service(plan(items, OptionalDouble.empty()), pred);
            svc.completionOffsetMs(1);
            assertEquals(1, pred.itemCalls(1L));
            assertEquals(1, pred.itemCalls(2L));
            assertEquals(0, pred.itemCalls(3L),
                    "suffix must not be predicted for a prefix query");
        }

        @Test
        void prefixTtftSurvivesASuffixPredictionFailure() {
            List<GroupPlanner.Item> items =
                    List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
            CountingPredictions pred = new CountingPredictions();
            pred.failOn(3L);
            RouteProjection.GroupService svc =
                    projection.service(plan(items, OptionalDouble.empty()), pred);
            assertEquals(300L, svc.completionOffsetMs(1),
                    "probe TTFT preserved even when suffix will fail");
            assertThrows(RuntimeException.class, svc::totalDurationMs,
                    "drain requires the failing suffix → unknown");
        }

        @Test
        void offsetsAreMemoizedAndEvaluatedAtMostOnce() {
            List<GroupPlanner.Item> items = List.of(item(1L, 100L), item(2L, 200L));
            CountingPredictions pred = new CountingPredictions();
            RouteProjection.GroupService svc =
                    projection.service(plan(items, OptionalDouble.empty()), pred);
            svc.completionOffsetMs(1);
            svc.completionOffsetMs(0);
            svc.totalDurationMs();
            assertEquals(1, pred.itemCalls(1L));
            assertEquals(1, pred.itemCalls(2L));
        }

        @Test
        void planningDurationIsThePrefixSumThroughRequiredIndex() {
            List<GroupPlanner.Item> items =
                    List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
            CountingPredictions pred = new CountingPredictions();
            RouteProjection.GroupPlanning planning = projection.planning(pred);
            // Through index 1: items[0]+items[1] = 100+200 = 300
            assertEquals(300.0, planning.durationMs(items, 1));
            assertEquals(0, pred.itemCalls(3L), "suffix beyond index not evaluated");
        }

        @Test
        void memberIndexIsBoundsChecked() {
            RouteProjection.GroupService svc = projection.service(
                    plan(List.of(item(1L, 100L)), OptionalDouble.empty()),
                    new CountingPredictions());
            assertThrows(IndexOutOfBoundsException.class, () -> svc.completionOffsetMs(1));
            assertThrows(IndexOutOfBoundsException.class, () -> svc.completionOffsetMs(-1));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    @Nested
    @DisplayName("Batch projection: per-prefix independent prediction")
    class BatchService {

        private final RouteProjection.DeliveryProjection projection =
                new BatchDeliveryStrategy(
                        mock(BatchSubmissionPort.class),
                        mock(PrefillAdmissionPort.class),
                        mock(SlotDeliveryPort.class),
                        mock(DeliveryMetrics.class)).projectionPolicy();

        @Test
        void eachMemberGetsBatchDurationOfItsExactPrefix() {
            // batchDurationMs returns 100*items.size() for discrimination.
            List<GroupPlanner.Item> items =
                    List.of(item(1L, 100L), item(2L, 200L), item(3L, 50L));
            CountingPredictions pred = new CountingPredictions();
            RouteProjection.GroupService svc =
                    projection.service(plan(items, OptionalDouble.empty()), pred);
            // offset(0) = batchDurationMs([item1]) = 100*1 = 100
            assertEquals(100L, svc.completionOffsetMs(0));
            // offset(1) = batchDurationMs([item1,item2]) = 100*2 = 200
            assertEquals(200L, svc.completionOffsetMs(1));
            // offset(2) = batchDurationMs([item1,item2,item3]) = 100*3 = 300
            assertEquals(300L, svc.completionOffsetMs(2));
            assertEquals(300L, svc.totalDurationMs());
        }

        @Test
        void eachPrefixIsComputedIndependentlyAndMemoized() {
            List<GroupPlanner.Item> items = List.of(item(1L, 100L), item(2L, 200L));
            CountingPredictions pred = new CountingPredictions();
            RouteProjection.GroupService svc =
                    projection.service(plan(items, OptionalDouble.empty()), pred);
            svc.completionOffsetMs(0);
            svc.completionOffsetMs(0); // repeated
            svc.completionOffsetMs(1);
            svc.completionOffsetMs(1); // repeated
            // Each prefix evaluated exactly once.
            assertEquals(2, pred.batchDurationCalls());
        }

        @Test
        void planningDurationDelegates() {
            List<GroupPlanner.Item> items = List.of(item(1L, 100L), item(2L, 200L));
            CountingPredictions pred = new CountingPredictions();
            RouteProjection.GroupPlanning planning = projection.planning(pred);
            // Batch planning returns batchPlanningDurationMs = 777 always.
            assertEquals(777.0, planning.durationMs(items, items.size() - 1));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    private static final class CountingPredictions
            implements RouteProjection.Predictions {

        private final Map<Long, Integer> itemCalls = new HashMap<>();
        private Set<Long> failingIds = Set.of();
        private int batchDurationCalls;

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
        public double batchPlanningDurationMs(List<GroupPlanner.Item> items) {
            return 777.0;
        }

        @Override
        public long batchDurationMs(List<GroupPlanner.Item> items) {
            batchDurationCalls++;
            return 100L * items.size(); // discriminating: prefix-size dependent
        }

        @Override
        public long committedGroupDurationMs(double plannedDurationMs) {
            return (long) plannedDurationMs;
        }
    }
}
