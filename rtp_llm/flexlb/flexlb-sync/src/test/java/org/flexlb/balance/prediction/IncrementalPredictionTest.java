package org.flexlb.balance.prediction;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class IncrementalPredictionTest {
    static String formula() throws Exception {
        try (var stream =
                IncrementalPredictionTest.class.getResourceAsStream(
                        "/prediction/deepseek-v4.txt")) {
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    void everyPrefixIsBitExactForRealFormulaAndNumericEdges() throws Exception {
        for (String expression :
                List.of(
                        formula(),
                        "sum(computeTokens) + totalComputeTokens^2 + maxInputTokens + maxComputeTokens + batchSize",
                        "sum(sum(computeTokens / 3)) + sum(max(hitCacheTokens - 4, 0))",
                        "inputTokens + computeTokens + hasHitCache + hitCacheTokens", // scalar
                        // bindings
                        // remain zero
                        // in batches
                        "-0.0",
                        "sum(-0.0)",
                        "sum(computeTokens / 0)",
                        "sum(1e308 * computeTokens) - sum(1e308 * computeTokens)")) {
            var model = new FormulaPredictor(expression);
            Random random = new Random(41);
            for (int trial = 0; trial < 100; trial++) {
                var incremental = model.newBatchPrediction();
                List<PrefillBatchFeatures.Item> prefix = new ArrayList<>();
                for (int i = 0; i < 64; i++) {
                    long input =
                            trial == 0
                                    ? new long[] {0, 1, (1L << 53) + 1, Long.MAX_VALUE}[i % 4]
                                    : random.nextInt(131072);
                    long hit = input == 0 ? 0 : Math.floorMod(random.nextLong(), input);
                    prefix.add(new PrefillBatchFeatures.Item(input, hit));
                    assertEquals(
                            Double.doubleToLongBits(
                                    model.predictBatchMs(new PrefillBatchFeatures(prefix))),
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
        for (int i = 0; i < 64; i++)
            source.add(item(i, 100 + random.nextInt(32768), i % 4, 100_000));
        for (int round = 0; round < 120; round++) {
            if (round % 3 == 0 && !source.isEmpty()) source.remove(random.nextInt(source.size()));
            else
                source.add(random.nextInt(source.size() + 1), item(1000 + round, 1000, 0, 100_000));
            if (round % 5 == 0) java.util.Collections.reverse(source);
            var constraints = new GroupPlanner.Constraints(64, 200_000, 300_000, 700, 700);
            var full =
                    GroupPlanner.FIXED_WINDOW.select(
                            source,
                            constraints,
                            (added, items) ->
                                    model.predictBatchMs(
                                            PrefillBatchFeatures.from(
                                                    items,
                                                    GroupPlanner.Item::seqLen,
                                                    GroupPlanner.Item::hitCache)));
            var batch = model.newBatchPrediction();
            var incremental =
                    GroupPlanner.FIXED_WINDOW.select(
                            source,
                            constraints,
                            (added, prefix) -> batch.append(added.seqLen(), added.hitCache()));
            assertEquals(full, incremental);
            assertEquals(
                    GroupPlanner.FIXED_WINDOW.dispatchReason(full, constraints, 2000),
                    GroupPlanner.FIXED_WINDOW.dispatchReason(incremental, constraints, 2000));
        }
        var oldSession = model.newBatchPrediction();
        var replacement = new FormulaPredictor("42");
        assertNotEquals(oldSession.append(100, 0), replacement.newBatchPrediction().append(100, 0));
    }

    @Test
    void projectionMatchesFullEvaluationAcrossExpiryPriorityAndProbePositions() throws Exception {
        var model = new FormulaPredictor(formula());
        var policy = org.flexlb.config.DispatcherConfig.Type.BATCH;
        var full =
                new PrefillTimePredictor.Evaluator() {
                    public long estimateMs(long t, long h) {
                        return model.estimateMs(t, h);
                    }

                    public double predictBatchMs(PrefillBatchFeatures f) {
                        return model.predictBatchMs(f);
                    }
                };
        var random = new Random(17);
        Comparator<GroupPlanner.Item> order =
                Comparator.comparingInt(GroupPlanner.Item::priority)
                        .thenComparingLong(GroupPlanner.Item::enqueueSeq);
        for (int trial = 0; trial < 150; trial++) {
            var items = new ArrayList<GroupPlanner.Item>();
            for (int i = 0; i < trial % 80; i++) {
                items.add(
                        item(
                                i,
                                100 + random.nextInt(32768),
                                i % 4,
                                i % 7 == 0 ? 500 : i % 11 == 0 ? 1100 : 100_000));
            }
            items.sort(order);
            var queue =
                    new QueueSnapshot(
                            1000,
                            true,
                            org.flexlb.balance.planner.GroupPlanner.FIXED_WINDOW,
                            order,
                            new GroupPlanner.Constraints(64, 200_000, 300_000, 700, 700),
                            items,
                            null);
            var inputs =
                    new RequestTimeEstimation.Inputs(
                            queue, new WorkSnapshot(1000, List.of(), List.of(), 0), 0L);
            var probe =
                    new RequestTimeEstimationTestSupport.Probe(
                            999, trial % 5, 1000, 100_000, 1 + random.nextInt(32768), 0, 0);
            assertEquals(
                    RequestTimeEstimationTestSupport.project(inputs, probe, full, policy),
                    RequestTimeEstimationTestSupport.project(inputs, probe, model, policy));
        }
    }

    @Test
    void concurrentSessionsDoNotShareMutableAccumulators() throws Exception {
        var model = new FormulaPredictor(formula());
        try (var pool = Executors.newFixedThreadPool(8)) {
            var tasks = new ArrayList<java.util.concurrent.Callable<Double>>();
            for (int k = 0; k < 32; k++) {
                int count = k + 1;
                tasks.add(
                        () -> {
                            var batch = model.newBatchPrediction();
                            double actual = 0;
                            var items = new ArrayList<PrefillBatchFeatures.Item>();
                            for (int i = 0; i < count; i++) {
                                items.add(new PrefillBatchFeatures.Item(1000, 333));
                                actual = batch.append(1000, 333);
                            }
                            assertEquals(
                                    model.predictBatchMs(new PrefillBatchFeatures(items)), actual);
                            return actual;
                        });
            }
            for (var result : pool.invokeAll(tasks)) result.get();
        }
    }

    @Test
    void liveBatchStrategyUsesAppendAndKeepsPreviousPredictionOnOvershoot() {
        var evaluator =
                new PrefillTimePredictor.Evaluator() {
                    public long estimateMs(long t, long h) {
                        throw new AssertionError("unexpected scalar prediction");
                    }

                    public double predictBatchMs(PrefillBatchFeatures f) {
                        throw new AssertionError("unexpected full prediction");
                    }

                    public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                        int[] count = {0};
                        return (t, h) -> ++count[0] * 250.25;
                    }
                };
        var mode = org.flexlb.config.DispatcherConfig.Type.BATCH;
        var items = new ArrayList<org.flexlb.balance.scheduler.RequestRoute>();
        for (int i = 0; i < 4; i++) {
            var request = mock(org.flexlb.balance.scheduler.RequestRoute.class);
            org.mockito.Mockito.when(request.seqLen()).thenReturn(100L);
            org.mockito.Mockito.when(request.hitCache()).thenReturn(0L);
            items.add(request);
        }
        var selected =
                GroupPlanner.FIXED_WINDOW.select(
                        items,
                        new GroupPlanner.Constraints(64, 100_000, 100_000, 700, 700),
                        RequestTimeEstimation.newGroupPredictor(mode, evaluator));
        assertEquals(items.subList(0, 2), selected.items());
        assertEquals(500.5, selected.selectedPredictionMs().orElseThrow());
        assertEquals(
                501L,
                PrefillPredictionBoundary.committedDecisionGroupMs(
                        selected.selectedPredictionMs().orElseThrow()));
        assertTrue(selected.predictionBoundaryTriggered());
        // A second snapshot must receive fresh accumulation state.
        assertEquals(
                selected,
                GroupPlanner.FIXED_WINDOW.select(
                        items,
                        new GroupPlanner.Constraints(64, 100_000, 100_000, 700, 700),
                        RequestTimeEstimation.newGroupPredictor(mode, evaluator)));
    }

    @Test
    void serviceReusesOnlyEvaluatedPrefixesFromItsOwnPlanningCursor() {
        int[] fullCalls = {0};
        var evaluator =
                new PrefillTimePredictor.Evaluator() {
                    public long estimateMs(long seqLen, long hitCache) {
                        return 100;
                    }

                    public double predictBatchMs(PrefillBatchFeatures features) {
                        fullCalls[0]++;
                        return features.batchSize() * 100.25;
                    }

                    public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                        int[] size = {0};
                        return (t, h) -> ++size[0] * 100.25;
                    }
                };
        assertEquals(201, project(evaluator, 2, 1000).projectedTtftMsValue());
        assertEquals(0, fullCalls[0]);
        assertEquals(401, project(evaluator, 4, 1000).projectedTtftMsValue());
        assertEquals(0, fullCalls[0], "a new group must not reuse the earlier probe's prefix");
        assertEquals(401, project(evaluator, 4, 0).projectedTtftMsValue());
        assertEquals(1, fullCalls[0]);
    }

    @Test
    void selectedPredictionKeepsTheOvershootPredecessorAndResetsForTheNextGroup() {
        int[] appended = {0};
        int[] recomputed = {0};
        var evaluator =
                new PrefillTimePredictor.Evaluator() {
                    public long estimateMs(long seqLen, long hitCache) {
                        return 11;
                    }

                    public double predictBatchMs(PrefillBatchFeatures features) {
                        recomputed[0]++;
                        return features.batchSize() * 10.25;
                    }

                    public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                        int[] size = {0};
                        return (tokens, hit) -> {
                            appended[0]++;
                            return ++size[0] * 10.25;
                        };
                    }
                };
        assertEquals(42, project(evaluator, 4, 31).projectedTtftMsValue());
        assertEquals(5, appended[0]);
        assertEquals(0, recomputed[0]);
        assertEquals(41, project(evaluator, 4, 0).projectedTtftMsValue());
        assertEquals(1, recomputed[0]);
    }

    private static RequestTimeEstimation.Candidate project(
            PrefillTimePredictor.Evaluator evaluator, int probeId, long budget) {
        var active = new ArrayList<GroupPlanner.Item>();
        for (int id = 1; id <= 4; id++) {
            if (id != probeId) active.add(item(id, 100, 0, 100_000));
        }
        var queue =
                new QueueSnapshot(
                        1000,
                        true,
                        org.flexlb.balance.planner.GroupPlanner.FIXED_WINDOW,
                        Comparator.comparingLong(GroupPlanner.Item::requestId),
                        new GroupPlanner.Constraints(64, 100_000, 100_000, budget, 0),
                        active,
                        null);
        return RequestTimeEstimationTestSupport.project(
                new RequestTimeEstimation.Inputs(
                        queue, new WorkSnapshot(1000, List.of(), List.of(), 0), 0),
                new RequestTimeEstimationTestSupport.Probe(probeId, 0, 1000, 100_000, 100, 0, 0),
                evaluator,
                org.flexlb.config.DispatcherConfig.Type.BATCH);
    }

    private static GroupPlanner.Item item(long id, long input, int priority, long expires) {
        return new GroupPlanner.Item(id, priority, id, 1000, expires, input, input / 3);
    }
}
