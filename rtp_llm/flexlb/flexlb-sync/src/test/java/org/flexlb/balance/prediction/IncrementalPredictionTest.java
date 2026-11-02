package org.flexlb.balance.prediction;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.BatchDeliveryStrategy;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;

class IncrementalPredictionTest {
    static String formula() throws Exception {
        try (var stream = IncrementalPredictionTest.class.getResourceAsStream("/prediction/deepseek-v4.txt")) {
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    void everyPrefixIsBitExactForRealFormulaAndNumericEdges() throws Exception {
        for (String expression : List.of(formula(),
                "sum(computeTokens) + totalComputeTokens^2 + maxInputTokens + maxComputeTokens + batchSize",
                "sum(sum(computeTokens / 3)) + sum(max(hitCacheTokens - 4, 0))",
                "inputTokens + computeTokens + hasHitCache + hitCacheTokens", // scalar bindings remain zero in batches
                "-0.0", "sum(-0.0)", "sum(computeTokens / 0)",
                "sum(1e308 * computeTokens) - sum(1e308 * computeTokens)")) {
            var model = new FormulaPredictor(expression);
            Random random = new Random(41);
            for (int trial = 0; trial < 100; trial++) {
                var incremental = model.newBatchPrediction();
                List<PrefillBatchFeatures.Item> prefix = new ArrayList<>();
                for (int i = 0; i < 64; i++) {
                    long input = trial == 0 ? new long[]{0, 1, (1L << 53) + 1, Long.MAX_VALUE}[i % 4]
                            : random.nextInt(131072);
                    long hit = input == 0 ? 0 : Math.floorMod(random.nextLong(), input);
                    prefix.add(new PrefillBatchFeatures.Item(input, hit));
                    assertEquals(Double.doubleToLongBits(model.predictBatchMs(new PrefillBatchFeatures(prefix))),
                            Double.doubleToLongBits(incremental.append(input, hit)),
                            "expression=" + expression + " prefix=" + prefix.size());
                }
            }
        }
    }

    @Test
    void freshSessionsHandleRemovalInsertionReorderAndModelReplacement() throws Exception {
        var random = new Random(73);
        var source = new ArrayList<GroupPlanner.Item>();
        var model = new FormulaPredictor(formula());
        for (int i = 0; i < 64; i++) source.add(item(i, 100 + random.nextInt(32768), i % 4, 100_000));
        for (int round = 0; round < 120; round++) {
            if (round % 3 == 0 && !source.isEmpty()) source.remove(random.nextInt(source.size()));
            else source.add(random.nextInt(source.size() + 1), item(1000 + round, 1000, 0, 100_000));
            if (round % 5 == 0) java.util.Collections.reverse(source);
            var constraints = new GroupPlanner.Constraints(64, 200_000, 300_000, 700, 700);
            var full = GroupPlanner.select(source, GroupPlanner.itemAccess(), constraints,
                    items -> model.predictBatchMs(PrefillBatchFeatures.from(items,
                            GroupPlanner.Item::seqLen, GroupPlanner.Item::hitCache)));
            var batch = model.newBatchPrediction();
            var incremental = GroupPlanner.selectWithPrediction(source, GroupPlanner.itemAccess(), constraints,
                    (added, prefix) -> batch.append(added.seqLen(), added.hitCache()));
            assertEquals(full, incremental);
            assertEquals(GroupPlanner.evaluateReadiness(full, constraints, 2000),
                    GroupPlanner.evaluateReadiness(incremental, constraints, 2000));
        }
        var oldSession = model.newBatchPrediction();
        var replacement = new FormulaPredictor("42");
        assertNotEquals(oldSession.append(100, 0), replacement.newBatchPrediction().append(100, 0));
    }

    @Test
    void projectionMatchesFullEvaluationAcrossExpiryPriorityAndProbePositions() throws Exception {
        var model = new FormulaPredictor(formula());
        var policy = new BatchDeliveryStrategy(
                () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, mock(RequestRegistry.class), mock(DeliveryMetrics.class)).projectionPolicy();
        var full = new PrefillTimePredictor.Evaluator() {
            public long estimateMs(long t, long h) { return model.estimateMs(t, h); }
            public double predictBatchMs(PrefillBatchFeatures f) { return model.predictBatchMs(f); }
        };
        var random = new Random(17);
        Comparator<GroupPlanner.Item> order = Comparator.comparingInt(GroupPlanner.Item::priority)
                .thenComparingLong(GroupPlanner.Item::enqueueSeq);
        for (int trial = 0; trial < 150; trial++) {
            var items = new ArrayList<GroupPlanner.Item>();
            for (int i = 0; i < trial % 80; i++) {
                items.add(item(i, 100 + random.nextInt(32768), i % 4,
                        i % 7 == 0 ? 500 : i % 11 == 0 ? 1100 : 100_000));
            }
            items.sort(order);
            var queue = new QueueSnapshot(1000, true, order,
                    new GroupPlanner.Constraints(64, 200_000, 300_000, 700, 700), items, null);
            var inputs = new RouteProjection.Inputs(queue, new WorkSnapshot(1000, List.of(), List.of(), 0));
            var probe = new RouteProjection.Probe(999, trial % 5, 1000, 100_000,
                    1 + random.nextInt(32768), 0, 0);
            assertEquals(RouteProjection.project(inputs, probe, full, policy),
                    RouteProjection.project(inputs, probe, model, policy));
        }
    }

    @Test
    void concurrentSessionsDoNotShareMutableAccumulators() throws Exception {
        var model = new FormulaPredictor(formula());
        try (var pool = Executors.newFixedThreadPool(8)) {
            var tasks = new ArrayList<java.util.concurrent.Callable<Double>>();
            for (int k = 0; k < 32; k++) {
                int count = k + 1;
                tasks.add(() -> {
                    var batch = model.newBatchPrediction();
                    double actual = 0;
                    var items = new ArrayList<PrefillBatchFeatures.Item>();
                    for (int i = 0; i < count; i++) {
                        items.add(new PrefillBatchFeatures.Item(1000, 333));
                        actual = batch.append(1000, 333);
                    }
                    assertEquals(model.predictBatchMs(new PrefillBatchFeatures(items)), actual);
                    return actual;
                });
            }
            for (var result : pool.invokeAll(tasks)) result.get();
        }
    }

    @Test
    void liveBatchStrategyUsesAppendAndKeepsPreviousPredictionOnOvershoot() {
        var evaluator = new PrefillTimePredictor.Evaluator() {
            public long estimateMs(long t, long h) { throw new AssertionError("unexpected scalar prediction"); }
            public double predictBatchMs(PrefillBatchFeatures f) { throw new AssertionError("unexpected full prediction"); }
            public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                int[] count = {0};
                return (t, h) -> ++count[0] * 250.25;
            }
        };
        var strategy = new BatchDeliveryStrategy(
                () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, mock(RequestRegistry.class), mock(DeliveryMetrics.class));
        var items = new ArrayList<org.flexlb.balance.scheduler.ScheduledRequest>();
        for (int i = 0; i < 4; i++) {
            var request = mock(org.flexlb.balance.scheduler.ScheduledRequest.class);
            org.mockito.Mockito.when(request.seqLen()).thenReturn(100L);
            org.mockito.Mockito.when(request.hitCache()).thenReturn(0L);
            items.add(request);
        }
        var access = new GroupPlanner.ItemAccess<org.flexlb.balance.scheduler.ScheduledRequest>() {
            public long seqLen(org.flexlb.balance.scheduler.ScheduledRequest r) { return r.seqLen(); }
            public long enqueuedAtMs(org.flexlb.balance.scheduler.ScheduledRequest r) { return 0; }
        };
        var selected = GroupPlanner.selectWithPrediction(items, access,
                new GroupPlanner.Constraints(64, 100_000, 100_000, 700, 700),
                strategy.newGroupPredictor(evaluator));
        assertEquals(items.subList(0, 2), selected.items());
        assertEquals(500.5, selected.selectedPredictionMs().orElseThrow());
        assertEquals(501L, PrefillPredictionBoundary.committedDecisionGroupMs(
                selected.selectedPredictionMs().orElseThrow()));
        assertTrue(selected.predictionBoundaryTriggered());
        // A second snapshot must receive fresh accumulation state.
        assertEquals(selected, GroupPlanner.selectWithPrediction(items, access,
                new GroupPlanner.Constraints(64, 100_000, 100_000, 700, 700),
                strategy.newGroupPredictor(evaluator)));
    }

    @Test
    void serviceReusesOnlyEvaluatedPrefixesFromItsOwnPlanningCursor() {
        var strategy = new BatchDeliveryStrategy(
                () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, mock(RequestRegistry.class), mock(DeliveryMetrics.class));
        int[] fullCalls = {0};
        var predictions = new RouteProjection.Predictions() {
            public long itemDurationMs(GroupPlanner.Item i) { return 100; }
            public double batchPlanningDurationMs(List<GroupPlanner.Item> items) { return items.size() * 100.25; }
            public long batchDurationMs(List<GroupPlanner.Item> items) {
                fullCalls[0]++;
                return PrefillPredictionBoundary.committedDecisionGroupMs(batchPlanningDurationMs(items));
            }
            public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                int[] size = {0};
                return (t, h) -> ++size[0] * 100.25;
            }
        };
        var items = List.of(item(1, 100, 0, 100_000), item(2, 100, 0, 100_000), item(3, 100, 0, 100_000));
        var policy = strategy.projectionPolicy();
        var cursor = policy.planning(predictions);
        assertEquals(200.5, cursor.durationMs(items, 1)); // only through probe, not suffix
        var plan = new GroupPlanner.Plan<>(items, new GroupPlanner.Shape(3, 300, 100, 100),
                1000, 1700, false, java.util.OptionalDouble.of(200.5), GroupPlanner.BATCH_FULL);
        var service = policy.service(plan, predictions, cursor);
        assertEquals(201, service.completionOffsetMs(1));
        assertEquals(0, fullCalls[0]);
        assertEquals(301, service.totalDurationMs()); // must not mistake P2 for P3
        assertEquals(1, fullCalls[0]);
        var independent = policy.service(plan, predictions);
        assertEquals(301, independent.totalDurationMs());
        assertEquals(2, fullCalls[0]);
    }

    @Test
    void appendCursorKeepsLatestAndOvershootPredecessorAndRecomputesOlderPrefixes() {
        var policy = new BatchDeliveryStrategy(
                () -> CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST),
                () -> 1L, mock(RequestRegistry.class), mock(DeliveryMetrics.class)).projectionPolicy();
        int[] appended = {0};
        int[] recomputed = {0};
        var predictions = new RouteProjection.Predictions() {
            public long itemDurationMs(GroupPlanner.Item item) { return 11; }
            public double batchPlanningDurationMs(List<GroupPlanner.Item> items) { return items.size() * 10.25; }
            public long batchDurationMs(List<GroupPlanner.Item> items) {
                recomputed[0]++;
                return (long) Math.ceil(batchPlanningDurationMs(items));
            }
            public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                return (tokens, hit) -> ++appended[0] * 10.25;
            }
        };
        var items = List.of(item(1, 100, 0, 100_000), item(2, 100, 0, 100_000),
                item(3, 100, 0, 100_000), item(4, 100, 0, 100_000));
        var cursor = policy.planning(predictions);
        assertEquals(41, cursor.durationMs(items, 3)); // Jump directly over several prefixes.
        assertEquals(41, cursor.durationMs(items, 3)); // Probe boundary reached: no further appends.
        assertEquals(4, appended[0]);
        assertEquals(41, cursor.predictedPrefixMs(4).orElseThrow());
        assertEquals(30.75, cursor.predictedPrefixMs(3).orElseThrow());
        assertTrue(cursor.predictedPrefixMs(2).isEmpty());
        assertTrue(cursor.predictedPrefixMs(0).isEmpty());
        assertTrue(cursor.predictedPrefixMs(5).isEmpty());
        var selected = items.subList(0, 2);
        var plan = new GroupPlanner.Plan<>(selected, new GroupPlanner.Shape(2, 100, 200, 200),
                1000, 1000, false, java.util.OptionalDouble.empty(), GroupPlanner.BATCH_FULL);
        assertEquals(21, policy.service(plan, predictions, cursor).totalDurationMs());
        assertEquals(1, recomputed[0]);
    }

    private static GroupPlanner.Item item(long id, long input, int priority, long expires) {
        return new GroupPlanner.Item(id, priority, id, 1000, expires, input, input / 3);
    }
}
