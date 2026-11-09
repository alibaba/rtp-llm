package org.flexlb.balance.prediction;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;

/**
 * Verifies route-prefix predictions reuse reads and offsets within one projection.
 */
class ProjectionPredictionCacheTest {
    private final RouteProjection.DeliveryProjection policy =
            new RouteDeliveryStrategy(mock(RequestRegistry.class), mock(DeliveryMetrics.class))
                    .projectionPolicy();

    @Test
    void routePrefixesReadEachMemberOnceAndReuseSelectedOffsets() {
        AtomicInteger reads = new AtomicInteger();
        AtomicInteger predictions = new AtomicInteger();
        var items =
                new java.util.AbstractList<GroupPlanner.Item>() {
                    public int size() {
                        return 512;
                    }

                    public GroupPlanner.Item get(int index) {
                        reads.incrementAndGet();
                        return new GroupPlanner.Item(
                                String.valueOf(index), 0, index, 1000, 100_000, 100, 0);
                    }
                };
        var boundary =
                new RouteProjection.Predictions() {
                    public long itemDurationMs(GroupPlanner.Item item) {
                        predictions.incrementAndGet();
                        return 10;
                    }

                    public double batchPlanningDurationMs(List<GroupPlanner.Item> items) {
                        throw new AssertionError();
                    }

                    public long batchDurationMs(List<GroupPlanner.Item> items) {
                        throw new AssertionError();
                    }
                };
        var planning = policy.planning(boundary);
        for (int i = 1; i <= items.size(); i++) {
            assertEquals(i * 10, planning.durationMs(items.subList(0, i), i - 1));
        }
        assertEquals(512, reads.get());
        assertEquals(512, predictions.get());
        // A rejected tentative last member must not contaminate the selected prefix.
        var selected =
                GroupPlanner.plan(
                        items.subList(0, 511),
                        GroupPlanner.itemAccess(),
                        new GroupPlanner.Constraints(511, 1_000_000, 1_000_000, 0, 0),
                        1000,
                        null);
        assertEquals(5110, policy.service(selected, boundary, planning).totalDurationMs());
        assertEquals(512, predictions.get());
        // Resetting for a different group must discard the cached offsets.
        var next = policy.planning(boundary);
        assertEquals(10, next.durationMs(items.subList(0, 1), 0));
        assertEquals(513, predictions.get());
    }
}
