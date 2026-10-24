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
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Contract tests for the canonical route-projection value boundary. */
class RouteProjectionTest {

    @Test
    void unchangedRunningWorkCanReuseItsEarlierClockBase() {
        WorkSnapshot cached = new WorkSnapshot(13L, List.of(new WorkSnapshot.RequestWork(
                "1", WorkSnapshot.Phase.ENGINE_RUNNING, 1_000L)), List.of(), 0L);
        WorkSnapshot current = new WorkSnapshot(20L, List.of(new WorkSnapshot.RequestWork(
                "1", WorkSnapshot.Phase.ENGINE_RUNNING, 993L)), List.of(), 0L);
        assertEquals(
                PrefillTtftEstimate.project(new PrefillTtftEstimate.Inputs(emptyQueue(20L), current),
                        probe(), new CountingEvaluator(), routeProjection()),
                PrefillTtftEstimate.project(new PrefillTtftEstimate.Inputs(emptyQueue(20L), cached),
                        probe(), new CountingEvaluator(), routeProjection()));
    }

    @Test
    void queueAndWorkComeFromOneCanonicalInput() {
        PrefillTtftEstimate.Inputs inputs = inputs(13L);
        CountingEvaluator evaluator = new CountingEvaluator();

        PrefillTtftEstimate.Candidate result = PrefillTtftEstimate.project(
                inputs,
                probe(),
                evaluator,
                routeProjection());

        assertTrue(result.selectable());
        assertEquals(20L, result.incomingPrefillMs());
        assertTrue(evaluator.invocations() > 0);

        assertThrows(IllegalArgumentException.class,
                () -> new PrefillTtftEstimate.Inputs(
                        emptyQueue(13L), emptyWork(14L)));
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
        PrefillTtftEstimate.Candidate result = PrefillTtftEstimate.project(
                inputs(31L),
                probe(),
                captured,
                routeProjection());
        assertTrue(result.selectable());
        assertEquals(before, captured.estimateMs(20L, 0L),
                "a projection-owned evaluator cannot change mid-call");
    }

    @Test
    void onlyModeledCandidateCarriesKnownTtft() {
        assertThrows(IllegalArgumentException.class,
                () -> candidate(PrefillTtftEstimate.Candidate.State.MODELED,
                        OptionalLong.empty()));
        assertThrows(IllegalArgumentException.class,
                () -> candidate(PrefillTtftEstimate.Candidate.State.UNAVAILABLE,
                        OptionalLong.of(1L)));
        assertThrows(IllegalArgumentException.class,
                () -> candidate(PrefillTtftEstimate.Candidate.State.MODELED,
                        OptionalLong.of(-2L)));

        PrefillTtftEstimate.Candidate modeled = candidate(
                PrefillTtftEstimate.Candidate.State.MODELED, OptionalLong.of(10L));
        PrefillTtftEstimate.Candidate unmodeled = candidate(
                PrefillTtftEstimate.Candidate.State.UNMODELED_ENGINE_WORK, OptionalLong.empty());
        assertEquals(10L, modeled.requiredProjectedTtftMs());
        assertTrue(unmodeled.engineWorkUnmodeled());
    }

    private static PrefillTtftEstimate.Candidate candidate(
            PrefillTtftEstimate.Candidate.State state,
            OptionalLong projectedTtftMs) {
        return new PrefillTtftEstimate.Candidate(
                state,
                projectedTtftMs.orElse(PrefillTtftEstimate.Candidate.UNKNOWN),
                0L,
                PrefillTtftEstimate.Candidate.InitialHeadDisposition.NONE,
                state.name(), null, 0L, 0L);
    }

    private static PrefillTtftEstimate.Inputs inputs(long capturedAtMs) {
        return new PrefillTtftEstimate.Inputs(
                emptyQueue(capturedAtMs), emptyWork(capturedAtMs));
    }

    private static QueueSnapshot emptyQueue(long capturedAtMs) {
        return new QueueSnapshot(
                capturedAtMs,
                Comparator.comparing(GroupPlanner.Item::requestId),
                new GroupPlanner.Constraints(
                        1, 1_000_000L, 1_000_000L, 0L, 30L),
                List.of(),
                null);
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(
                capturedAtMs, List.of(), List.of(), 0L);
    }

    private static PrefillTtftEstimate.Probe probe() {
        return new PrefillTtftEstimate.Probe(
                "99", 50, 13L, Long.MAX_VALUE,
                20L, 0L, 0L);
    }

    private static PrefillTtftEstimate.DeliveryProjection routeProjection() {
        return new RouteDeliveryStrategy(
                Mockito.mock(RequestRegistry.class),
                Mockito.mock(DeliveryMetrics.class))
                .ttftEstimationRule();
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
