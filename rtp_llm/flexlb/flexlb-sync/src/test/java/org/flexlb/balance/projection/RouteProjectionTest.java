package org.flexlb.balance.projection;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Comparator;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Contract tests for the canonical route-projection value boundary. */
class RouteProjectionTest {

    @Test
    void unchangedRunningWorkCanReuseItsEarlierClockBase() {
        WorkSnapshot cached =
                new WorkSnapshot(
                        13L,
                        List.of(
                                new WorkSnapshot.RequestWork(
                                        "1", WorkSnapshot.Phase.ENGINE_RUNNING, 1_000L)),
                        List.of(),
                        0L);
        WorkSnapshot current =
                new WorkSnapshot(
                        20L,
                        List.of(
                                new WorkSnapshot.RequestWork(
                                        "1", WorkSnapshot.Phase.ENGINE_RUNNING, 993L)),
                        List.of(),
                        0L);
        assertEquals(
                RouteProjection.project(
                        new RouteProjection.Inputs(emptyQueue(20L), current),
                        probe(),
                        new CountingEvaluator(),
                        routeProjection()),
                RouteProjection.project(
                        new RouteProjection.Inputs(emptyQueue(20L), cached),
                        probe(),
                        new CountingEvaluator(),
                        routeProjection()));
    }

    @Test
    void queueAndWorkComeFromOneCanonicalInput() {
        RouteProjection.Inputs inputs = inputs(13L);
        CountingEvaluator evaluator = new CountingEvaluator();

        RouteProjection.Candidate result =
                RouteProjection.project(inputs, probe(), evaluator, routeProjection());

        assertTrue(result.hasPrediction());
        assertEquals(20L, result.incomingPrefillMs());
        assertTrue(evaluator.invocations() > 0);

        assertThrows(
                IllegalArgumentException.class,
                () -> new RouteProjection.Inputs(emptyQueue(13L), emptyWork(14L)));
    }

    @Test
    void oneCapturedEvaluatorRemainsStableAfterLearningPublishesReplacement() {
        LearningPredictor predictor = new LearningPredictor();
        PrefillTimePredictor.Evaluator captured = predictor.evaluator();
        PrefillBatchFeatures sample =
                new PrefillBatchFeatures(List.of(new PrefillBatchFeatures.Item(100L, 0L)));
        long before = captured.estimateMs(20L, 0L);

        for (int sampleIndex = 0; sampleIndex < 4; sampleIndex++) {
            predictor.learn(sample, 1L, 100L);
        }

        assertNotSame(captured, predictor.evaluator());
        RouteProjection.Candidate result =
                RouteProjection.project(inputs(31L), probe(), captured, routeProjection());
        assertTrue(result.hasPrediction());
        assertEquals(
                before,
                captured.estimateMs(20L, 0L),
                "a projection-owned evaluator cannot change mid-call");
    }

    @Test
    void predictionAvailabilityIsDerivedFromItsValue() {
        assertThrows(IllegalArgumentException.class, () -> candidate(-2L));
        RouteProjection.Candidate predicted = candidate(10L);
        assertTrue(predicted.hasPrediction());
        assertEquals(10L, predicted.requiredProjectedTtftMs());
        RouteProjection.Candidate unavailable = candidate(RouteProjection.Candidate.UNKNOWN);
        assertFalse(unavailable.hasPrediction());
        assertEquals(OptionalLong.empty(), unavailable.projectedTtftMs());
        assertThrows(IllegalStateException.class, unavailable::requiredProjectedTtftMs);
    }

    private static RouteProjection.Candidate candidate(long projectedTtftMs) {
        return new RouteProjection.Candidate(
                projectedTtftMs,
                0L,
                RouteProjection.Candidate.InitialHeadDisposition.NONE,
                "",
                0L,
                0L);
    }

    private static RouteProjection.Inputs inputs(long capturedAtMs) {
        return new RouteProjection.Inputs(emptyQueue(capturedAtMs), emptyWork(capturedAtMs));
    }

    private static QueueSnapshot emptyQueue(long capturedAtMs) {
        return new QueueSnapshot(
                capturedAtMs,
                true,
                Comparator.comparing(GroupPlanner.Item::requestId),
                new GroupPlanner.Constraints(1, 1_000_000L, 1_000_000L, 0L, 30L),
                List.of(),
                null);
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 0L);
    }

    private static RouteProjection.Probe probe() {
        return new RouteProjection.Probe("99", 50, 13L, Long.MAX_VALUE, 20L, 0L, 0L);
    }

    private static RouteProjection.DeliveryProjection routeProjection() {
        return new RouteDeliveryStrategy(
                        Mockito.mock(RequestRegistry.class), Mockito.mock(DeliveryMetrics.class))
                .projectionPolicy();
    }

    private static final class CountingEvaluator implements PrefillTimePredictor.Evaluator {
        private final AtomicInteger invocations = new AtomicInteger();

        @Override
        public long estimateMs(long totalTokens, long hitTokens) {
            invocations.incrementAndGet();
            return Math.max(0L, totalTokens - hitTokens);
        }

        @Override
        public double predictBatchMs(PrefillBatchFeatures features) {
            invocations.incrementAndGet();
            return features.items().stream()
                    .mapToLong(item -> item.seqLen() - item.hitCache())
                    .sum();
        }

        int invocations() {
            return invocations.get();
        }
    }
}
