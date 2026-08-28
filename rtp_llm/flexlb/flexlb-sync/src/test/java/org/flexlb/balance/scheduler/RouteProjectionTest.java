package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.BATCH;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.NOW_MS;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.ROUTE;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.TOKEN_EVALUATOR;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.candidate;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.constraints;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.item;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.noCommittedWork;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.probe;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.project;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.queue;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.work;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Canonical frozen-snapshot route timeline and candidate contract. */
class RouteProjectionTest {

    @Test
    void committedWorkOverlapsCollectionWindow() {
        WorkSnapshot committed = work(
                List.of(new WorkSnapshot.RequestWork(
                        1L, WorkSnapshot.Phase.ENGINE_RUNNING, 100L)),
                List.of(), 0L);

        RouteProjection.Candidate result = project(
                queue(false, constraints(4, 30L), List.of()),
                committed,
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertModeled(result, 120L, 120L);
        assertEquals("SERIAL_FROZEN_QUEUE", result.detail());
    }

    @Test
    void singleRequestGroupNeverPaysCollectionWindow() {
        RouteProjection.Candidate result = project(
                queue(false, constraints(1, 30L), List.of()),
                noCommittedWork(),
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertModeled(result, 20L, 20L);
    }

    @Test
    void successiveDecisionGroupsUseOneSerialCursor() {
        List<GroupPlanner.Item> active = List.of(
                item(1L, 50, 1L, 10L),
                item(2L, 50, 2L, 10L));

        RouteProjection.Candidate result = project(
                queue(false, constraints(2, 30L), active),
                noCommittedWork(),
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertModeled(result, 50L, 50L);
    }

    @Test
    void coherentWorkPhasesContributeTheirFrozenRemainingDuration() {
        WorkSnapshot committed = work(
                List.of(
                        new WorkSnapshot.RequestWork(
                                1L, WorkSnapshot.Phase.COMMITTED, 10L),
                        new WorkSnapshot.RequestWork(
                                2L, WorkSnapshot.Phase.ENGINE_QUEUED, 20L),
                        new WorkSnapshot.RequestWork(
                                3L, WorkSnapshot.Phase.ENGINE_RUNNING, 30L)),
                List.of(new WorkSnapshot.BatchWork(
                        7L, List.of(4L, 5L),
                        WorkSnapshot.Phase.ENGINE_RUNNING, 40L)),
                0L);

        RouteProjection.Candidate result = project(
                queue(false, false, constraints(1, 0L), List.of(), null),
                committed,
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertModeled(result, 120L, 120L);
        assertEquals("SERIAL_FROZEN_DIRECT", result.detail());
    }

    @Test
    void inputsRejectSnapshotsFromDifferentLinearizationPoints() {
        QueueSnapshot queue = queue(false, constraints(1, 0L), List.of());
        WorkSnapshot later = new WorkSnapshot(
                NOW_MS + 1L, List.of(), List.of(), 0L);

        assertThrows(IllegalArgumentException.class,
                () -> new RouteProjection.Inputs(queue, later, 0L));
    }

    @Test
    void fifoAndPriorityPlaceProbeInDifferentDecisionGroups() {
        List<GroupPlanner.Item> active = List.of(
                item(1L, 100, 1L, 10L),
                item(2L, 10, 2L, 100L));
        RouteProjection.Probe probe = probe(
                99L, 90, 20L, 0L,
                RouteProjection.Demand.TTFT_AND_DRAIN);

        RouteProjection.Candidate fifo = project(
                queue(false, constraints(2, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR, probe, BATCH);
        RouteProjection.Candidate priority = project(
                queue(true, constraints(2, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR, probe, BATCH);

        assertModeled(fifo, 130L, 130L);
        assertModeled(priority, 30L, 130L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .BEFORE_PROBE, fifo.initialHeadDisposition());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .BEFORE_PROBE, priority.initialHeadDisposition());
    }

    @Test
    void batchCompletionIncludesLowerPrioritySuffixInSameBatch() {
        List<GroupPlanner.Item> active = List.of(
                item(1L, 100, 1L, 10L),
                item(2L, 10, 2L, 100L));

        RouteProjection.Candidate result = project(
                queue(true, constraints(3, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertModeled(result, 30L, 130L);
    }

    @Test
    void routeCompletionStopsAtProbeWhileDrainIncludesSuffix() {
        List<GroupPlanner.Item> active = List.of(
                item(1L, 100, 1L, 10L),
                item(2L, 10, 2L, 100L));

        RouteProjection.Candidate result = project(
                queue(true, constraints(3, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertModeled(result, 30L, 130L);
    }

    @Test
    void endpointCacheHitChangesServiceAndCandidateMetadata() {
        RouteProjection.Probe coldProbe = new RouteProjection.Probe(
                99L, 50, NOW_MS, Long.MAX_VALUE,
                1_000L, 0L, 123L,
                RouteProjection.Demand.TTFT_AND_DRAIN);
        RouteProjection.Probe warmProbe = new RouteProjection.Probe(
                100L, 50, NOW_MS, Long.MAX_VALUE,
                1_000L, 800L, 900L,
                RouteProjection.Demand.TTFT_AND_DRAIN);

        RouteProjection.Candidate cold = candidate(
                queue(false, constraints(1, 0L), List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                coldProbe, ROUTE, 7L);
        RouteProjection.Candidate warm = candidate(
                queue(false, constraints(1, 0L), List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                warmProbe, ROUTE, 8L);

        assertEquals(1_000L, cold.incomingPrefillMs());
        assertEquals(OptionalLong.of(1_000L), cold.projectedTtftMs());
        assertEquals(0L, cold.cacheHitTokens());
        assertEquals(123L, cold.routingCacheMatchTokens());
        assertEquals(7L, cold.requiredPendingCount());
        assertEquals(440L, warm.incomingPrefillMs());
        assertEquals(OptionalLong.of(440L), warm.projectedTtftMs());
        assertEquals(800L, warm.cacheHitTokens());
        assertEquals(900L, warm.routingCacheMatchTokens());
        assertEquals(8L, warm.requiredPendingCount());
    }

    @Test
    void negativeSinglePredictionIsUnavailableInsteadOfClamped() {
        PrefillTimePredictor.Evaluator invalid = evaluator(
                (tokens, hits) -> tokens == 20L ? -1L : tokens,
                items -> items.stream()
                        .mapToLong(PrefillBatchFeatures.Item::seqLen)
                        .sum());

        RouteProjection.Candidate result = project(
                queue(true, constraints(1, 0L), List.of()),
                noCommittedWork(), invalid,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertInvalidPrediction(result);
    }

    @Test
    void invalidBatchServicePredictionsAreUnavailable() {
        for (double invalidValue : List.of(
                -1.0,
                Double.NaN,
                Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY)) {
            PrefillTimePredictor.Evaluator invalid = evaluator(
                    (tokens, hits) -> tokens,
                    ignored -> invalidValue);

            RouteProjection.Candidate result = project(
                    queue(true, constraints(1, 0L), List.of()),
                    noCommittedWork(), invalid,
                    probe(99L, 50, 20L, 0L,
                            RouteProjection.Demand.TTFT_AND_DRAIN),
                    BATCH);

            assertInvalidPrediction(result);
        }
    }

    @Test
    void invalidPlannerBoundaryPredictionIsUnavailable() {
        GroupPlanner.Constraints predictionBounded =
                new GroupPlanner.Constraints(
                        2, 1_000_000L, 1_000_000L, 100L, 0L);
        PrefillTimePredictor.Evaluator invalid = evaluator(
                (tokens, hits) -> tokens,
                ignored -> Double.NaN);

        RouteProjection.Candidate result = project(
                queue(true, predictionBounded, List.of()),
                noCommittedWork(), invalid,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertInvalidPrediction(result);
    }

    @Test
    void unknownRequestWorkIsExplicitlyUnmodeledAndKeepsPendingCount() {
        RouteProjection.Candidate candidate = candidate(
                queue(true, constraints(1, 0L), List.of()),
                work(List.of(), List.of(), 1L),
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE,
                9L);

        assertEquals(RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                candidate.state());
        assertEquals("ENGINE_WORK_UNOBSERVABLE",
                candidate.detail());
        assertFalse(candidate.selectable());
        assertTrue(candidate.engineWorkUnmodeled());
        assertEquals(9L, candidate.requiredPendingCount());
        assertEquals(20L, candidate.incomingPrefillMs());
    }

    @Test
    void unknownRepackedBatchWorkIsExplicitlyUnmodeled() {
        WorkSnapshot unknown = work(
                List.of(),
                List.of(new WorkSnapshot.BatchWork(
                        7L,
                        List.of(1L),
                        WorkSnapshot.Phase.ENGINE_RUNNING,
                        OptionalLong.empty())),
                0L);

        RouteProjection.Candidate result = project(
                queue(true, constraints(1, 0L), List.of()),
                unknown, TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                result.state());
        assertEquals("ENGINE_WORK_UNOBSERVABLE", result.detail());
        assertFalse(result.selectable());
    }

    @Test
    void nonPositiveExistingExpiryIsTerminalAndDoesNotBlockProbe() {
        RouteProjection.Candidate result = project(
                queue(false, constraints(1, 0L), List.of(
                        item(1L, 50, 1L, 100L, 0L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertModeled(result, 20L, 20L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .TERMINAL_PRUNED, result.initialHeadDisposition());
    }

    @Test
    void probeExpiringExactlyAtCollectionDeadlineIsNotDispatched() {
        RouteProjection.Candidate result = project(
                queue(false, constraints(4, 30L), List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, NOW_MS, NOW_MS + 30L,
                        20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, result.state());
        assertEquals("INCOMING_EXPIRED_BEFORE_DISPATCH", result.detail());
        assertFalse(result.selectable());
    }

    @Test
    void existingMemberExpiringInsideWindowIsRemovedBeforeService() {
        RouteProjection.Candidate result = project(
                queue(false, constraints(4, 30L), List.of(
                        item(1L, 50, 1L, 10L, NOW_MS + 10L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertModeled(result, 50L, 50L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .TERMINAL_PRUNED, result.initialHeadDisposition());
    }

    @Test
    void batchTokenCapacityDoesNotRejectIndivisibleRequests() {
        GroupPlanner.Constraints strict = new GroupPlanner.Constraints(
                1, 100L, 1_000_000L, 0L, 0L);

        RouteProjection.Candidate afterInvalidHead = project(
                queue(false, strict, List.of(item(1L, 50, 1L, 100L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);
        RouteProjection.Candidate invalidProbe = project(
                queue(false, strict, List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(100L, 50, 100L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertModeled(afterInvalidHead, 120L, 120L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .BEFORE_PROBE, afterInvalidHead.initialHeadDisposition());
        assertModeled(invalidProbe, 100L, 100L);
    }

    @Test
    void routeSuffixFailureDoesNotEraseEstablishedProbeTtft() {
        RouteProjection.Candidate result = project(
                queue(true, constraints(3, 0L), List.of(
                        item(1L, 100, 1L, 10L),
                        item(2L, 10, 2L, 999L))),
                noCommittedWork(), suffixFailingSingleEvaluator(),
                probe(99L, 90, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(30L), result.projectedTtftMs());
        assertEquals(OptionalLong.empty(), result.projectedDrainMs());
        assertEquals("DRAIN_PREDICTION_UNAVAILABLE", result.detail());
    }

    @Test
    void ttftOnlyNeverTouchesRouteSuffixAfterProbeCompletion() {
        RouteProjection.Candidate result = project(
                queue(true, constraints(3, 0L), List.of(
                        item(1L, 100, 1L, 10L),
                        item(2L, 10, 2L, 999L))),
                noCommittedWork(), suffixFailingSingleEvaluator(),
                probe(99L, 90, 20L, 0L,
                        RouteProjection.Demand.TTFT_ONLY),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(30L), result.projectedTtftMs());
        assertEquals(OptionalLong.empty(), result.projectedDrainMs());
        assertEquals("SERIAL_FROZEN_QUEUE_TTFT_ONLY", result.detail());
    }

    @Test
    void ttftOnlyNeverTouchesLaterBatchGroup() {
        RouteProjection.Candidate result = project(
                queue(true, constraints(1, 0L), List.of(
                        item(1L, 10, 1L, 999L))),
                noCommittedWork(), suffixFailingBatchEvaluator(),
                probe(99L, 90, 20L, 0L,
                        RouteProjection.Demand.TTFT_ONLY),
                BATCH);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(OptionalLong.empty(), result.projectedDrainMs());
    }

    @Test
    void invalidLaterBatchPredictionKeepsTtftButMakesDrainUnknown() {
        RouteProjection.Candidate result = project(
                queue(true, constraints(1, 0L), List.of(
                        item(1L, 10, 1L, 999L))),
                noCommittedWork(), suffixFailingBatchEvaluator(),
                probe(99L, 90, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                BATCH);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(OptionalLong.empty(), result.projectedDrainMs());
        assertEquals("DRAIN_PREDICTION_UNAVAILABLE", result.detail());
    }

    @Test
    void suffixExpiryAffectsDrainButNotEstablishedProbeTtft() {
        GroupPlanner.Item expiringSuffix = item(
                1L, 10, 1L, 100L, NOW_MS + 10L);
        GroupPlanner.Constraints splitByComputeShape =
                new GroupPlanner.Constraints(
                        2, 150L, 1_000_000L, 0L, 30L);
        QueueSnapshot snapshot = queue(
                true, splitByComputeShape, List.of(expiringSuffix));
        RouteProjection.Probe ttftProbe = probe(
                99L, 90, NOW_MS - 30L, Long.MAX_VALUE,
                20L, 0L, RouteProjection.Demand.TTFT_ONLY);
        RouteProjection.Probe drainProbe = probe(
                100L, 90, NOW_MS - 30L, Long.MAX_VALUE,
                20L, 0L, RouteProjection.Demand.TTFT_AND_DRAIN);

        RouteProjection.Candidate ttftOnly = project(
                snapshot, noCommittedWork(), TOKEN_EVALUATOR,
                ttftProbe, ROUTE);
        RouteProjection.Candidate withDrain = project(
                snapshot, noCommittedWork(), TOKEN_EVALUATOR,
                drainProbe, ROUTE);

        assertEquals(OptionalLong.of(20L), ttftOnly.projectedTtftMs());
        assertEquals(OptionalLong.empty(), ttftOnly.projectedDrainMs());
        assertModeled(withDrain, 20L, 20L);
    }

    @Test
    void suffixKvBlockDoesNotEraseEstablishedProbeTtft() {
        GroupPlanner.Constraints kvLimited = new GroupPlanner.Constraints(
                1, 1_000_000L, 50L, 0L, 30L);

        RouteProjection.Candidate result = project(
                queue(true, kvLimited, List.of(
                        item(1L, 10, 1L, 100L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 100, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(OptionalLong.empty(), result.projectedDrainMs());
        assertEquals("DRAIN_BLOCKED_PREFILL_KV_CAPACITY", result.detail());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition.AFTER_PROBE,
                result.initialHeadDisposition());
    }

    @Test
    void duplicateIncomingIdentityIsRejectedAtActiveAndCommittedBoundaries() {
        RouteProjection.Candidate active = project(
                queue(false, constraints(1, 0L), List.of(
                        item(99L, 50, 1L, 20L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);
        RouteProjection.Candidate committed = project(
                queue(false, constraints(1, 0L), List.of()),
                work(List.of(new WorkSnapshot.RequestWork(
                                99L,
                                WorkSnapshot.Phase.ENGINE_RUNNING,
                                20L)),
                        List.of(), 0L),
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_AND_DRAIN),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, active.state());
        assertEquals("INCOMING_ALREADY_ACTIVE", active.detail());
        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE,
                committed.state());
        assertEquals("INCOMING_ALREADY_COMMITTED", committed.detail());
    }

    @Test
    void oneFrozenEvaluatorSuppliesPlanningAndServicePredictions() {
        AtomicInteger singleCalls = new AtomicInteger();
        AtomicInteger batchCalls = new AtomicInteger();
        PrefillTimePredictor.Evaluator evaluator = evaluator(
                (tokens, hits) -> {
                    singleCalls.incrementAndGet();
                    return tokens - hits;
                },
                items -> {
                    batchCalls.incrementAndGet();
                    return items.stream()
                            .mapToLong(PrefillBatchFeatures.Item::seqLen)
                            .sum();
                });

        RouteProjection.Candidate result = project(
                queue(false,
                        new GroupPlanner.Constraints(
                                1, 1_000_000L, 1_000_000L,
                                500L, 0L),
                        List.of()),
                noCommittedWork(), evaluator,
                probe(99L, 50, 20L, 0L,
                        RouteProjection.Demand.TTFT_ONLY),
                BATCH);

        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(1, singleCalls.get());
        assertEquals(2, batchCalls.get(),
                "planner and committed-service boundaries use the same snapshot evaluator");
    }

    private static void assertModeled(
            RouteProjection.Candidate result,
            long ttftMs,
            long drainMs) {
        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(ttftMs), result.projectedTtftMs());
        assertEquals(OptionalLong.of(drainMs), result.projectedDrainMs());
        assertTrue(result.selectable());
    }

    private static void assertInvalidPrediction(RouteProjection.Candidate result) {
        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, result.state());
        assertEquals("PREDICTOR_RETURNED_INVALID_VALUE", result.detail());
        assertEquals(OptionalLong.empty(), result.projectedTtftMs());
        assertEquals(OptionalLong.empty(), result.projectedDrainMs());
        assertFalse(result.selectable());
    }

    private static PrefillTimePredictor.Evaluator suffixFailingSingleEvaluator() {
        return evaluator(
                (tokens, hits) -> {
                    if (tokens == 999L) {
                        throw new IllegalStateException(
                                "suffix prediction unavailable");
                    }
                    return tokens;
                },
                items -> items.stream()
                        .mapToLong(PrefillBatchFeatures.Item::seqLen)
                        .sum());
    }

    private static PrefillTimePredictor.Evaluator suffixFailingBatchEvaluator() {
        return evaluator(
                (tokens, hits) -> tokens,
                items -> items.stream()
                        .anyMatch(item -> item.seqLen() == 999L)
                                ? Double.NaN
                                : items.stream()
                                        .mapToLong(
                                                PrefillBatchFeatures.Item::seqLen)
                                        .sum());
    }

    private static PrefillTimePredictor.Evaluator evaluator(
            SinglePrediction single,
            BatchPrediction batch) {
        return new PrefillTimePredictor.Evaluator() {
            @Override
            public long estimateMs(long totalTokens, long hitTokens) {
                return single.predict(totalTokens, hitTokens);
            }

            @Override
            public double predictBatchMs(PrefillBatchFeatures features) {
                return batch.predict(features.items());
            }
        };
    }

    @FunctionalInterface
    private interface SinglePrediction {
        long predict(long totalTokens, long hitTokens);
    }

    @FunctionalInterface
    private interface BatchPrediction {
        double predict(List<PrefillBatchFeatures.Item> items);
    }
}
