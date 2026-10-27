package org.flexlb.balance.delivery;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport;
import org.flexlb.config.DispatcherConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Delivery timing contracts exercised through the complete projection entry point. */
class RouteDeliveryProjectionTest {

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void eachCompletionEvaluatesOnlyItsRequiredPrefix(boolean batch) {
        for (int index = 0; index < 3; index++) {
            var model = new CountingEvaluator();
            var result = project(model, batch, 0, index, 100, 200, 50);
            assertEquals(
                    batch ? (index + 1) * 100L : new long[] {100, 300, 350}[index],
                    result.projectedTtftMsValue());
            if (batch) {
                assertEquals(1, model.fullCalls);
                assertEquals(index + 1, model.lastBatchSize);
            } else {
                assertEquals(index == 0 ? 1 : 2, model.calls(new long[] {100, 200, 50}[index]));
                for (int member = index + 1; member < 3; member++) {
                    assertEquals(0, model.calls(new long[] {100, 200, 50}[member]));
                }
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void growingPlanningPrefixReusesOnlyEvaluatedWorkAndResetsForNextQueue(boolean batch) {
        var model = new CountingEvaluator();
        var first = project(model, batch, 1000, 1, 100, 200, 50);
        assertEquals(batch ? 200 : 300, first.projectedTtftMsValue());
        assertEquals(0, model.calls(50));
        assertEquals(batch ? 2 : 0, model.appends);
        assertEquals(0, model.fullCalls);
        assertEquals(batch ? 1 : 0, model.sessions);

        var next = project(model, batch, 1000, 1, 400, 500, 60);
        assertEquals(batch ? 200 : 900, next.projectedTtftMsValue());
        assertEquals(0, model.calls(60));
        assertEquals(batch ? 4 : 0, model.appends);
        assertEquals(batch ? 2 : 0, model.sessions);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void probePrefixSurvivesASuffixPredictionFailure(boolean batch) {
        var model = new CountingEvaluator();
        model.failingTokens = 50;
        for (long budget : new long[] {0, 1000}) {
            assertEquals(
                    batch ? 200 : 300,
                    project(model, batch, budget, 1, 100, 200, 50).projectedTtftMsValue());
            assertEquals(0, model.calls(50));
            assertEquals(
                    RequestTimeEstimation.Candidate.State.UNAVAILABLE,
                    project(model, batch, budget, 1, 50, 200, 100).state());
            model.calls.clear();
        }
    }

    @Test
    void routeCompletionReusesTheAcceptedPrefixAfterAnOverBudgetAppend() {
        var model = new CountingEvaluator();
        var result = project(model, false, 250, 1, 100, 200, 50);
        assertEquals(300L, result.projectedTtftMsValue());
        assertEquals(1, model.calls(100));
        assertEquals(2, model.calls(200), "one incoming estimate and one tentative append");
        assertEquals(0, model.calls(50));
    }

    @Test
    void batchCompletionUsesTheAcceptedFractionalPrefixAfterAnOverBudgetAppend() {
        var model = new CountingEvaluator();
        model.batchPerItem = 100.25;
        var result = project(model, true, 250, 2, 100, 200, 50, 60);
        assertEquals(302L, result.projectedTtftMsValue(), "ceil(200.5) + ceil(100.25)");
        assertEquals(4, model.appends, "three tentative members then the next group head");
        assertEquals(2, model.sessions);
        assertEquals(0, model.fullCalls, "accepted group predictions are reused");
    }

    @Test
    void routePlanningCacheKeepsExactLongValuesAndSaturates() {
        var model = new CountingEvaluator();
        long large = (1L << 53) + 3;
        model.durations.put(100L, large);
        model.durations.put(200L, 5L);
        model.durations.put(50L, Long.MAX_VALUE);
        for (long budget : new long[] {0, Long.MAX_VALUE}) {
            assertEquals(
                    large, project(model, false, budget, 0, 100, 200, 50).projectedTtftMsValue());
            assertEquals(
                    large + 5,
                    project(model, false, budget, 1, 100, 200, 50).projectedTtftMsValue());
            assertEquals(
                    Long.MAX_VALUE,
                    project(model, false, budget, 2, 100, 200, 50).projectedTtftMsValue());
        }
        var overshoot = new CountingEvaluator();
        overshoot.durations.put(100L, large);
        overshoot.durations.put(200L, 100L);
        overshoot.durations.put(50L, 5L);
        assertEquals(
                large + 105,
                project(overshoot, false, large + 4, 2, 100, 200, 50).projectedTtftMsValue(),
                "overshoot must not round the accepted predecessor through double");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void failedProjectionCannotPoisonTheNextGroup(boolean batch) {
        var model = new CountingEvaluator();
        model.failingTokens = 100;
        assertEquals(
                RequestTimeEstimation.Candidate.State.UNAVAILABLE,
                project(model, batch, 1000, 1, 100, 200).state());
        model.failingTokens = -1;
        assertEquals(
                batch ? 200 : 300, project(model, batch, 1000, 1, 100, 200).projectedTtftMsValue());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void nestedProjectionDuringGroupPredictionKeepsTheOuterPrefix(boolean batch) {
        var model =
                new CountingEvaluator() {
                    @Override
                    public long estimateMs(long tokens, long hit) {
                        if (!batch && tokens == 100) {
                            assertEquals(
                                    800,
                                    project(new CountingEvaluator(), false, 1000, 0, 800, 900)
                                            .projectedTtftMsValue());
                        }
                        return super.estimateMs(tokens, hit);
                    }

                    @Override
                    public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                        var delegate = super.newBatchPrediction();
                        return (tokens, hit) -> {
                            if (tokens == 100) {
                                assertEquals(
                                        800,
                                        project(new CountingEvaluator(), false, 1000, 0, 800, 900)
                                                .projectedTtftMsValue());
                            }
                            return delegate.append(tokens, hit);
                        };
                    }
                };
        assertEquals(
                batch ? 200 : 300,
                project(model, batch, 1000, 1, 100, 200, 50).projectedTtftMsValue());
        assertEquals(0, model.calls(50));
    }

    private static RequestTimeEstimation.Candidate project(
            CountingEvaluator model, boolean batch, long budget, int probeIndex, long... tokens) {
        var active = new ArrayList<GroupPlanner.Item>();
        for (int i = 0; i < tokens.length; i++) {
            if (i != probeIndex) {
                active.add(
                        new GroupPlanner.Item(i + 1, 0, i + 1, 1000, Long.MAX_VALUE, tokens[i], 0));
            }
        }
        var queue =
                new QueueSnapshot(
                        1000,
                        true,
                        GroupPlanner.FIXED_WINDOW,
                        Comparator.comparingLong(GroupPlanner.Item::requestId),
                        new GroupPlanner.Constraints(64, Long.MAX_VALUE, Long.MAX_VALUE, budget, 0),
                        active,
                        null);
        return RequestTimeEstimationTestSupport.project(
                new RequestTimeEstimation.Inputs(
                        queue, new WorkSnapshot(1000, List.of(), List.of(), 0), 0),
                new RequestTimeEstimationTestSupport.Probe(
                        probeIndex + 1, 0, 1000, Long.MAX_VALUE, tokens[probeIndex], 0, 0),
                model,
                batch ? DispatcherConfig.Type.BATCH : DispatcherConfig.Type.NON_BATCH);
    }

    private static class CountingEvaluator implements PrefillTimePredictor.Evaluator {
        private final Map<Long, Integer> calls = new HashMap<>();
        private final Map<Long, Long> durations = new HashMap<>();
        private long failingTokens = -1;
        private int fullCalls, lastBatchSize, sessions, appends;
        private double batchPerItem = 100;

        int calls(long tokens) {
            return calls.getOrDefault(tokens, 0);
        }

        public long estimateMs(long tokens, long hit) {
            calls.merge(tokens, 1, Integer::sum);
            validate(tokens);
            return durations.getOrDefault(tokens, tokens);
        }

        public double predictBatchMs(PrefillBatchFeatures features) {
            fullCalls++;
            lastBatchSize = features.batchSize();
            features.items().forEach(i -> validate(i.seqLen()));
            return batchPerItem * features.batchSize();
        }

        public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
            sessions++;
            int[] count = {0};
            return (tokens, hit) -> {
                validate(tokens);
                appends++;
                return ++count[0] * batchPerItem;
            };
        }

        private void validate(long tokens) {
            if (tokens == failingTokens) {
                throw new IllegalStateException("prediction failed");
            }
        }
    }
}
