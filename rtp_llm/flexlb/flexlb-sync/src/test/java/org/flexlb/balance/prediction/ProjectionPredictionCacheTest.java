package org.flexlb.balance.prediction;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.junit.jupiter.api.Test;

import java.util.AbstractList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;

/**
 * Verifies Route projection reuses the offsets from the selected planning prefix.
 */
class ProjectionPredictionCacheTest {
    private final RouteProjection.DeliveryProjection policy =
            new RouteDeliveryStrategy(mock(RequestRegistry.class), mock(DeliveryMetrics.class))
                    .projectionPolicy();

    @Test
    void routePlanningAndServiceReuseTheSelectedPrefix() {
        AtomicInteger reads = new AtomicInteger();
        AtomicInteger calls = new AtomicInteger();
        var items =
                new AbstractList<GroupPlanner.Item>() {
                    @Override
                    public int size() {
                        return 512;
                    }

                    @Override
                    public GroupPlanner.Item get(int index) {
                        reads.incrementAndGet();
                        return new GroupPlanner.Item(
                                "request-" + index, 0, index, 1000, 100_000, 100, 0);
                    }
                };
        var predictions =
                new RouteProjection.Predictions() {
                    @Override
                    public long itemDurationMs(GroupPlanner.Item item) {
                        calls.incrementAndGet();
                        return 10L;
                    }

                    @Override
                    public double batchPlanningDurationMs(List<GroupPlanner.Item> items) {
                        throw new AssertionError();
                    }

                    @Override
                    public long batchDurationMs(List<GroupPlanner.Item> items) {
                        throw new AssertionError();
                    }
                };
        var planning = policy.planning(predictions);
        for (int size = 1; size <= items.size(); size++) {
            assertEquals(size * 10.0, planning.durationMs(items.subList(0, size), size - 1));
        }
        assertEquals(512, reads.get());
        assertEquals(512, calls.get());

        var plan =
                GroupPlanner.plan(
                        items.subList(0, 511),
                        GroupPlanner.itemAccess(),
                        new GroupPlanner.Constraints(511, 1_000_000, 1_000_000, 0, 0),
                        1000,
                        null);
        assertEquals(5110L, policy.service(plan, predictions, planning).totalDurationMs());
        assertEquals(512, calls.get());

        assertEquals(10.0, policy.planning(predictions).durationMs(items.subList(0, 1), 0));
        assertEquals(513, calls.get());
    }
}
