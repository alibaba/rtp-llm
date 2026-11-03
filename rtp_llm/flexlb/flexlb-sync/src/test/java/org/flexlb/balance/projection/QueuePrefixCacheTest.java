package org.flexlb.balance.projection;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.BatchDeliveryStrategy;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class QueuePrefixCacheTest {
    private final RouteProjection.DeliveryProjection route = new RouteDeliveryStrategy(
            mock(RequestRegistry.class), mock(DeliveryMetrics.class)).projectionPolicy();
    private final RouteProjection.DeliveryProjection batch = new BatchDeliveryStrategy(
            () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
            () -> 1L, mock(RequestRegistry.class), mock(DeliveryMetrics.class)).projectionPolicy();
    private static final Comparator<GroupPlanner.Item> ORDER = Comparator.comparingInt(GroupPlanner.Item::priority)
            .thenComparingLong(GroupPlanner.Item::enqueueSeq);

    private static RouteProjection.Inputs input(int limit, long expiry, long window) {
        var items = new ArrayList<GroupPlanner.Item>();
        for (int i = 0; i < 24; i++) items.add(new GroupPlanner.Item(Integer.toString(i), i / 8, i * 2, 1000,
                i % 3 == 0 ? expiry : Long.MAX_VALUE, 100 + i * 10, i));
        return new RouteProjection.Inputs(new QueueSnapshot(1000, true, ORDER,
                new GroupPlanner.Constraints(limit, 10_000, 100_000, 700, window), items, null),
                new WorkSnapshot(1000, List.of(), List.of(), 0), 1);
    }

    private static PrefillTimePredictor.Evaluator model(AtomicInteger calls) {
        return new PrefillTimePredictor.Evaluator() {
            public long estimateMs(long input, long hit) { calls.incrementAndGet(); return input - hit; }
            public double predictBatchMs(PrefillBatchFeatures f) {
                calls.incrementAndGet();
                return f.items().stream().mapToLong(i -> i.seqLen() - i.hitCache()).sum();
            }
        };
    }

    @Test
    void warmPrefixAvoidsPredictionsForBothDeliveryModes() {
        for (var policy : List.of(route, batch)) {
            var inputs = input(2, 10_000, 0);
            var calls = new AtomicInteger();
            var model = model(calls);
            var probe = new RouteProjection.Probe("probe", 3, 1000, 100_000, 100, 0, 0);
            var first = RouteProjection.project(inputs, probe, model, policy);
            int cold = calls.get();
            calls.set(0);
            assertEquals(first, RouteProjection.project(inputs, probe, model, policy));
            assertTrue(calls.get() < cold / 2, "warm prefix must skip actual prediction calls");
            assertNotNull(inputs.prefixCache().read(model.snapshotIdentity(), policy, 1001));
            // A different model cannot read this model's memo.
            assertNull(inputs.prefixCache().read(new Object(), policy, 1001));
            assertNull(inputs.prefixCache().read(model.snapshotIdentity(), policy, 10_000));
            var replacement = new RouteProjection.Inputs(inputs.queue(), inputs.work(), 2);
            assertNull(replacement.prefixCache().read(model.snapshotIdentity(), policy, 1001));
        }
    }

    @Test
    void warmedProjectionsMatchUncachedOracleAcrossInsertionTimeAndPolicyChanges() {
        Random random = new Random(1729);
        var model = model(new AtomicInteger());
        for (int trial = 0; trial < 80; trial++) {
            var inputs = input(1 + trial % 6, 1002 + trial % 7, trial % 3 == 0 ? 5 : 0);
            var oracle = new RouteProjection.Inputs(inputs.queue(), inputs.work(), 1, null);
            // Warm the longest unaffected prefix first, then try inserting anywhere.
            RouteProjection.project(inputs, new RouteProjection.Probe("probe", 9, 1000, 100_000, 100, 0, 0),
                    model, batch, 1000);
            for (int i = 0; i < 50; i++) {
                var policy = (i / 10) % 2 == 0 ? batch : route;
                var probe = new RouteProjection.Probe("probe", random.nextInt(5), random.nextInt(50),
                        1000, i % 7 == 0 ? 1003 : 100_000, 100 + random.nextInt(1000), 0, 0);
                long now = 1000 + i % 12;
                assertEquals(RouteProjection.project(oracle, probe, model, policy, now),
                        RouteProjection.project(inputs, probe, model, policy, now), "trial=" + trial + " probe=" + i);
            }
        }
    }

    @Test
    void publishedGroupsAreImmutableAndWarmHitsDoNotReplaceTheEntry() {
        var cache = new QueuePrefixCache();
        Object model = new Object();
        var groups = new ArrayList<QueuePrefixCache.Group>();
        groups.add(new QueuePrefixCache.Group(2, 100));
        groups.add(new QueuePrefixCache.Group(2, 200));
        cache.publish(model, batch, 1000, 2000, groups);
        var published = cache.read(model, batch, 1000);
        groups.clear();
        assertEquals(2, published.groups().size());
        cache.publish(model, batch, 1001, 2000, List.of(new QueuePrefixCache.Group(2, 100)));
        assertSame(published, cache.read(model, batch, 1001));
        assertNull(cache.read(model, batch, 2000));
        cache.publish(model, batch, 2000, 3000, List.of(new QueuePrefixCache.Group(1, 42)));
        assertEquals(42, cache.read(model, batch, 2000).groups().getFirst().durationMs());
    }

    @Test
    void concurrentReadersAndPublishersMatchUncachedProjection() throws Exception {
        var inputs = input(3, 10_000, 0);
        var oracle = new RouteProjection.Inputs(inputs.queue(), inputs.work(), 1, null);
        var model = model(new AtomicInteger());
        try (var pool = java.util.concurrent.Executors.newFixedThreadPool(8)) {
            var tasks = new ArrayList<java.util.concurrent.Callable<Void>>();
            for (int i = 0; i < 200; i++) {
                int p = i;
                tasks.add(() -> {
                    var probe = new RouteProjection.Probe("probe", p % 5, p % 50, 1000, 100_000, 100 + p, 0, 0);
                    assertEquals(RouteProjection.project(oracle, probe, model, batch),
                            RouteProjection.project(inputs, probe, model, batch));
                    return null;
                });
            }
            for (var future : pool.invokeAll(tasks)) future.get();
        }
    }
}
