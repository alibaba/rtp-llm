package org.flexlb.balance.projection;

import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Comparator;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Contract tests for the canonical route-projection value boundary. */
class RouteProjectionTest {

    @Test
    void queueWorkAndPendingCountComeFromOneCanonicalInput() {
        RouteProjection.Inputs inputs = inputs(13L, 7L);
        CountingEvaluator evaluator = new CountingEvaluator();

        RouteProjection.Candidate result = RouteProjection.project(
                inputs,
                probe(RouteProjection.Demand.TTFT_AND_DRAIN),
                evaluator,
                routeProjection());

        assertTrue(result.selectable());
        assertEquals(7L, result.requiredPendingCount());
        assertEquals(20L, result.incomingPrefillMs());
        assertTrue(evaluator.invocations() > 0);

        assertThrows(IllegalArgumentException.class,
                () -> new RouteProjection.Inputs(
                        emptyQueue(13L), emptyWork(14L), 7L));
    }

    @Test
    void oneCapturedEvaluatorRemainsStableAfterLearningPublishesReplacement() {
        LearningPredictor predictor = new LearningPredictor();
        PrefillTimePredictor.Evaluator captured = predictor.evaluator();
        PrefillBatchFeatures sample = new PrefillBatchFeatures(List.of(
                new PrefillBatchFeatures.Item(100L, 0L)));
        long before = captured.estimateMs(20L, 0L);

        for (int sampleIndex = 0; sampleIndex < 4; sampleIndex++) {
            predictor.learn(sample, 1L, 100L);
        }

        assertNotSame(captured, predictor.evaluator());
        RouteProjection.Candidate result = RouteProjection.project(
                inputs(31L, 0L),
                probe(RouteProjection.Demand.TTFT_AND_DRAIN),
                captured,
                routeProjection());
        assertTrue(result.selectable());
        assertEquals(before, captured.estimateMs(20L, 0L),
                "a projection-owned evaluator cannot change mid-call");
    }

    @Test
    void candidatePendingCountExistsExactlyForUsableProjection() {
        assertThrows(IllegalArgumentException.class,
                () -> candidate(RouteProjection.Candidate.State.MODELED,
                        OptionalLong.of(10L), OptionalLong.empty()));
        assertThrows(IllegalArgumentException.class,
                () -> candidate(RouteProjection.Candidate.State.UNAVAILABLE,
                        OptionalLong.empty(), OptionalLong.of(1L)));
        assertThrows(IllegalArgumentException.class,
                () -> new RouteProjection.Candidate(
                        RouteProjection.Candidate.State.MODELED,
                        OptionalLong.of(-1L), OptionalLong.empty(), 0L,
                        RouteProjection.Candidate.InitialHeadDisposition.NONE,
                        "invalid", null, 0L, 0L, OptionalLong.of(1L)));

        RouteProjection.Candidate modeledCandidate =
                candidate(RouteProjection.Candidate.State.MODELED,
                        OptionalLong.of(10L), OptionalLong.of(20L));
        RouteProjection.Candidate unmodeledCandidate =
                candidate(RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                        OptionalLong.empty(), OptionalLong.of(2L));
        assertEquals(20L, modeledCandidate.requiredPendingCount());
        assertEquals(2L, unmodeledCandidate.requiredPendingCount());
        assertTrue(unmodeledCandidate.engineWorkUnmodeled());
    }

    private static RouteProjection.Candidate candidate(
            RouteProjection.Candidate.State state,
            OptionalLong projectedTtftMs,
            OptionalLong pendingCount) {
        return new RouteProjection.Candidate(
                state, projectedTtftMs, OptionalLong.empty(), 0L,
                RouteProjection.Candidate.InitialHeadDisposition.NONE,
                state.name(), null, 0L, 0L, pendingCount);
    }

    private static RouteProjection.Inputs inputs(
            long capturedAtMs, long pendingCount) {
        return new RouteProjection.Inputs(
                emptyQueue(capturedAtMs),
                emptyWork(capturedAtMs),
                pendingCount);
    }

    private static QueueSnapshot emptyQueue(long capturedAtMs) {
        return new QueueSnapshot(
                capturedAtMs,
                true,
                Comparator.comparingLong(GroupPlanner.Item::requestId),
                new GroupPlanner.Constraints(
                        1, 1_000_000L, 1_000_000L, 0L, 30L),
                List.of(),
                null);
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(
                capturedAtMs, List.of(), List.of(), 0L);
    }

    private static RouteProjection.Probe probe(
            RouteProjection.Demand demand) {
        return new RouteProjection.Probe(
                99L, 50, 13L, Long.MAX_VALUE,
                20L, 0L, 0L, demand);
    }

    private static RouteProjection.DeliveryProjection routeProjection() {
        return new RouteDeliveryStrategy(
                Mockito.mock(RequestRegistry.class),
                Mockito.mock(DeliveryMetrics.class))
                .projectionPolicy();
    }

    private static final class CountingEvaluator
            implements PrefillTimePredictor.Evaluator {
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
