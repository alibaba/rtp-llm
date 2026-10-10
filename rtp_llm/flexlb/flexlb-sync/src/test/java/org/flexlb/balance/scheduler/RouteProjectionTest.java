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
    void disabledExecutionBudgetDoesNotConstructAnUnusedPrefixPredictor() {
        PrefillTimePredictor.Evaluator evaluator = new PrefillTimePredictor.Evaluator() {
            @Override
            public long estimateMs(long seqLen, long hitCache) {
                return TOKEN_EVALUATOR.estimateMs(seqLen, hitCache);
            }

            @Override
            public double predictBatchMs(PrefillBatchFeatures features) {
                return TOKEN_EVALUATOR.predictBatchMs(features);
            }

            @Override
            public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                throw new AssertionError("disabled execution budget must not construct a prefix predictor");
            }
        };
        QueueSnapshot snapshot = queue(false, constraints(2, 0L), List.of(
                item(1L, 50, 1L, 10L), item(2L, 50, 2L, 10L), item(3L, 50, 3L, 10L)));
        for (RouteProjection.DeliveryProjection delivery : List.of(BATCH, ROUTE)) {
            assertModeled(project(snapshot, noCommittedWork(), evaluator,
                    probe(99L, 50, 20L, 0L), delivery), 50L);
        }
    }

    @Test
    void probeCompletionMatchesEveryPriorityPositionAcrossConsumedGroups() {
        // Exercise full groups and tails rejected by prediction, compute or KV capacity.
        for (var limits : List.of(
                new GroupPlanner.Constraints(1, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L),
                new GroupPlanner.Constraints(3, Long.MAX_VALUE, Long.MAX_VALUE, 0L, 0L),
                new GroupPlanner.Constraints(3, Long.MAX_VALUE, Long.MAX_VALUE, 25L, 0L),
                new GroupPlanner.Constraints(3, 25L, Long.MAX_VALUE, 0L, 0L),
                new GroupPlanner.Constraints(3, Long.MAX_VALUE, 25L, 0L, 0L))) {
            for (int position = 0; position <= 12; position++) {
                var active = new java.util.ArrayList<GroupPlanner.Item>();
                long precedingMs = 0L;
                for (int index = 0; index < 12; index++) {
                    boolean expired = index % 4 == 1;
                    active.add(item(index + 1L, 100 - 2 * index, index + 1L,
                            10L, expired ? NOW_MS : Long.MAX_VALUE));
                    if (index < position && !expired) {
                        precedingMs += 10L;
                    }
                }
                for (RouteProjection.DeliveryProjection delivery : List.of(BATCH, ROUTE)) {
                    var result = project(queue(true, limits, active), noCommittedWork(),
                            TOKEN_EVALUATOR, probe(99L, 101 - 2 * position, 7L, 0L), delivery);
                    assertModeled(result, precedingMs + 7L);
                }
            }
        }
    }

    @Test
    void expiryOnBothSidesOfProbeDuringCollectionKeepsItsLogicalPosition() {
        QueueSnapshot snapshot = queue(true, constraints(5, 30L), List.of(
                item(1L, 100, 1L, 1L, NOW_MS + 5L),
                item(2L, 100, 2L, 2L, NOW_MS + 5L),
                item(3L, 100, 3L, 3L, NOW_MS + 5L),
                item(4L, 100, 4L, 4L, NOW_MS + 5L),
                item(5L, 100, 5L, 5L, NOW_MS + 5L),
                item(6L, 90, 6L, 600L, NOW_MS + 10L),
                item(7L, 80, 7L, 50L),
                item(8L, 60, 8L, 700L, NOW_MS + 5L)));
        for (RouteProjection.DeliveryProjection delivery : List.of(BATCH, ROUTE)) {
            assertModeled(project(snapshot, noCommittedWork(), TOKEN_EVALUATOR,
                    probe(99L, 70, 80L, 0L), delivery), 159L);
        }
    }

    @Test
    void emptyQueueUsesTheSameWindowAndBudgetReadinessAsLiveGrouping() {
        for (RouteProjection.DeliveryProjection delivery : List.of(BATCH, ROUTE)) {
            var waiting = project(queue(false, constraints(4, 30L), List.of()), noCommittedWork(),
                    TOKEN_EVALUATOR, probe(99L, 50, 20L, 0L), delivery);
            assertModeled(waiting, 50L);
            var capped = project(queue(false, new GroupPlanner.Constraints(4, Long.MAX_VALUE,
                    Long.MAX_VALUE, 20L, 30L), List.of()), noCommittedWork(),
                    TOKEN_EVALUATOR, probe(99L, 50, 20L, 0L), delivery);
            assertModeled(capped, 20L);
        }
    }

    @Test
    void emptyQueueCannotProjectDeliveryBeyondAvailableKv() {
        var limits = new GroupPlanner.Constraints(4, Long.MAX_VALUE, 19L, 0L, 30L);
        for (RouteProjection.DeliveryProjection delivery : List.of(BATCH, ROUTE)) {
            var result = project(queue(false, limits, List.of()), noCommittedWork(),
                    TOKEN_EVALUATOR, probe(99L, 50, 20L, 0L), delivery);
            assertEquals(RouteProjection.Candidate.State.BLOCKED, result.state());
            assertEquals("PREFILL_KV_CAPACITY", result.detail());
            assertEquals(RouteProjection.Candidate.InitialHeadDisposition.NONE, result.initialHeadDisposition());
        }
    }

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
                probe(99L, 50, 20L, 0L),
                BATCH);

        assertModeled(result, 120L);
        assertEquals("EMPTY_ACTIVE_QUEUE_SINGLETON", result.detail());
    }

    @Test
    void singleRequestGroupNeverPaysCollectionWindow() {
        RouteProjection.Candidate result = project(
                queue(false, constraints(1, 30L), List.of()),
                noCommittedWork(),
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                BATCH);

        assertModeled(result, 20L);
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
                probe(99L, 50, 20L, 0L),
                BATCH);

        assertModeled(result, 50L);
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
                        List.of(4L, 5L),
                        WorkSnapshot.Phase.ENGINE_RUNNING, OptionalLong.of(40L))),
                0L);

        RouteProjection.Candidate result = project(
                queue(false, false, constraints(1, 0L), List.of(), null),
                committed,
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertModeled(result, 120L);
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
        RouteProjectionTestSupport.Probe probe = probe(
                99L, 90, 20L, 0L);

        RouteProjection.Candidate fifo = project(
                queue(false, constraints(2, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR, probe, BATCH);
        RouteProjection.Candidate priority = project(
                queue(true, constraints(2, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR, probe, BATCH);

        assertModeled(fifo, 130L);
        assertModeled(priority, 30L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .BEFORE_PROBE, fifo.initialHeadDisposition());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .BEFORE_PROBE, priority.initialHeadDisposition());
    }

    @Test
    void batchProjectionStopsAtProbeCompletion() {
        List<GroupPlanner.Item> active = List.of(
                item(1L, 100, 1L, 10L),
                item(2L, 10, 2L, 100L));

        RouteProjection.Candidate result = project(
                queue(true, constraints(3, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L),
                BATCH);

        assertModeled(result, 30L);
    }

    @Test
    void routeProjectionStopsAtProbeCompletion() {
        List<GroupPlanner.Item> active = List.of(
                item(1L, 100, 1L, 10L),
                item(2L, 10, 2L, 100L));

        RouteProjection.Candidate result = project(
                queue(true, constraints(3, 0L), active),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L),
                ROUTE);

        assertModeled(result, 30L);
    }

    @Test
    void endpointCacheHitChangesServiceAndCandidateMetadata() {
        RouteProjectionTestSupport.Probe coldProbe = new RouteProjectionTestSupport.Probe(
                99L, 50, NOW_MS, Long.MAX_VALUE,
                1_000L, 0L, 123L);
        RouteProjectionTestSupport.Probe warmProbe = new RouteProjectionTestSupport.Probe(
                100L, 50, NOW_MS, Long.MAX_VALUE,
                1_000L, 800L, 900L);

        RouteProjection.Candidate cold = project(
                queue(false, constraints(1, 0L), List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                coldProbe, ROUTE);
        RouteProjection.Candidate warm = project(
                queue(false, constraints(1, 0L), List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                warmProbe, ROUTE);

        assertEquals(1_000L, cold.incomingPrefillMs());
        assertEquals(1_000L, cold.projectedTtftMsValue());
        assertEquals(0L, cold.cacheHitTokens());
        assertEquals(123L, cold.routingCacheMatchTokens());
        assertEquals(440L, warm.incomingPrefillMs());
        assertEquals(440L, warm.projectedTtftMsValue());
        assertEquals(800L, warm.cacheHitTokens());
        assertEquals(900L, warm.routingCacheMatchTokens());
    }

    @Test
    void singlePredictionExecutionFailureRetainsItsStageDiagnosis() {
        PrefillTimePredictor.Evaluator failing = evaluator(
                (tokens, hits) -> { throw new IllegalStateException("predictor failed"); },
                items -> 1.0);

        RouteProjection.Candidate result = project(
                queue(true, constraints(1, 0L), List.of()),
                noCommittedWork(), failing, probe(99L, 50, 20L, 0L), ROUTE);

        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, result.state());
        assertEquals("SINGLE_PREDICTION_FAILED", result.detail());
        assertEquals(RouteProjection.Candidate.UNKNOWN, result.projectedTtftMsValue());
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
                probe(99L, 50, 20L, 0L),
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
                    probe(99L, 50, 20L, 0L),
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
                probe(99L, 50, 20L, 0L),
                BATCH);

        assertInvalidPrediction(result);
    }

    @Test
    void unknownRequestWorkIsExplicitlyUnmodeled() {
        RouteProjection.Candidate candidate = project(
                queue(true, constraints(1, 0L), List.of()),
                work(List.of(), List.of(), 1L),
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                candidate.state());
        assertEquals("ENGINE_WORK_UNOBSERVABLE",
                candidate.detail());
        assertFalse(candidate.selectable());
        assertTrue(candidate.engineWorkUnmodeled());
        assertEquals(20L, candidate.incomingPrefillMs());
    }

    @Test
    void unknownRepackedBatchWorkIsExplicitlyUnmodeled() {
        WorkSnapshot unknown = work(
                List.of(),
                List.of(new WorkSnapshot.BatchWork(
                        List.of(1L),
                        WorkSnapshot.Phase.ENGINE_RUNNING,
                        OptionalLong.empty())),
                0L);

        RouteProjection.Candidate result = project(
                queue(true, constraints(1, 0L), List.of()),
                unknown, TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
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
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertModeled(result, 20L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .TERMINAL_PRUNED, result.initialHeadDisposition());
    }

    @Test
    void probeExpiringExactlyAtCollectionDeadlineIsNotDispatched() {
        RouteProjection.Candidate result = project(
                queue(false, constraints(4, 30L), List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, NOW_MS, NOW_MS + 30L,
                        20L, 0L),
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
                probe(99L, 50, 20L, 0L),
                BATCH);

        assertModeled(result, 50L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .TERMINAL_PRUNED, result.initialHeadDisposition());
    }

    @Test
    void collectionWaitReusesPredictionUntilExpiryOrBudgetChangesTheDecision() {
        AtomicInteger planningCalls = new AtomicInteger();
        AtomicInteger predictionCalls = new AtomicInteger();
        RouteProjection.DeliveryProjection counted = new RouteProjection.DeliveryProjection() {
            @Override
            public long singletonCompletionOffsetMs(long seqLen, long hitCache,
                    RouteProjection.Predictions predictions) {
                return ROUTE.singletonCompletionOffsetMs(seqLen, hitCache, predictions);
            }

            @Override
            public RouteProjection.GroupPlanning planning(RouteProjection.Predictions predictions) {
                planningCalls.incrementAndGet();
                RouteProjection.GroupPlanning delegate = ROUTE.planning(predictions);
                return (items, index) -> {
                    predictionCalls.incrementAndGet();
                    return delegate.durationMs(items, index);
                };
            }

            @Override
            public long completionOffsetMs(List<GroupPlanner.Item> items, int memberIndex,
                    RouteProjection.Predictions predictions, RouteProjection.GroupPlanning planning) {
                return ROUTE.completionOffsetMs(items, memberIndex, predictions, planning);
            }
        };
        List<GroupPlanner.Item> active = List.of(item(1L, 50, 1L, 10L));

        RouteProjection.Candidate waited = project(
                queue(false, new GroupPlanner.Constraints(3, 1_000_000L, 1_000_000L, 31L, 30L), active),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, NOW_MS, NOW_MS + 30L, 20L, 0L), counted);
        assertModeled(waited, 59L);
        assertEquals(1, planningCalls.get());
        assertEquals(2, predictionCalls.get(), "the frozen group is predicted once across the wait");

        planningCalls.set(0);
        predictionCalls.set(0);
        RouteProjection.Candidate expiredPrefix = project(
                queue(false, new GroupPlanner.Constraints(3, 1_000_000L, 1_000_000L, 31L, 30L),
                        List.of(item(0L, 50, 0L, 10L, NOW_MS), active.getFirst())),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, NOW_MS, NOW_MS + 30L, 20L, 0L), counted);
        assertModeled(expiredPrefix, 59L);
        assertEquals(1, planningCalls.get(), "an already expired prefix does not change the waiting group");
        assertEquals(2, predictionCalls.get(), "the surviving frozen group is predicted once");

        planningCalls.set(0);
        predictionCalls.set(0);
        RouteProjection.Candidate expired = project(
                queue(false, new GroupPlanner.Constraints(3, 1_000_000L, 1_000_000L, 31L, 30L), active),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, NOW_MS, NOW_MS + 29L, 20L, 0L), counted);
        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, expired.state());
        assertEquals("INCOMING_EXPIRED_BEFORE_DISPATCH", expired.detail());
        assertEquals(1, planningCalls.get());

        planningCalls.set(0);
        RouteProjection.Candidate budgetReady = project(
                queue(false, new GroupPlanner.Constraints(3, 1_000_000L, 1_000_000L, 30L, 30L), active),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, NOW_MS, NOW_MS + 29L, 20L, 0L), counted);
        assertModeled(budgetReady, 30L);
        assertEquals(1, planningCalls.get());
    }

    @Test
    void expirySkipsDeliveredPrefixAndRemovesInterleavedWaitingMembers() {
        QueueSnapshot snapshot = queue(false, constraints(5, 30L), List.of(
                item(1L, 50, 1L, 1L, NOW_MS + 5L),
                item(2L, 50, 2L, 2L, NOW_MS + 5L),
                item(3L, 50, 3L, 3L, NOW_MS + 5L),
                item(4L, 50, 4L, 4L, NOW_MS + 5L),
                item(5L, 50, 5L, 5L, NOW_MS + 5L),
                item(6L, 50, 6L, 600L, NOW_MS + 10L),
                item(7L, 50, 7L, 50L),
                item(8L, 50, 8L, 700L, NOW_MS + 5L)));

        for (RouteProjection.DeliveryProjection delivery : List.of(BATCH, ROUTE)) {
            RouteProjection.Candidate result = project(
                    snapshot, noCommittedWork(), TOKEN_EVALUATOR,
                    probe(99L, 50, 80L, 0L), delivery);

            // The full first group takes 15 ms. The remaining group waits until
            // 29 ms, drops requests 6 and 8, then executes requests 7 and 99.
            assertModeled(result, 159L);
            assertEquals(RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                    result.initialHeadDisposition());
        }
    }

    @Test
    void batchTokenCapacityDoesNotRejectIndivisibleRequests() {
        GroupPlanner.Constraints strict = new GroupPlanner.Constraints(
                1, 100L, 1_000_000L, 0L, 0L);

        RouteProjection.Candidate afterInvalidHead = project(
                queue(false, strict, List.of(item(1L, 50, 1L, 100L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);
        RouteProjection.Candidate invalidProbe = project(
                queue(false, strict, List.of()),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(100L, 50, 100L, 0L),
                ROUTE);

        assertModeled(afterInvalidHead, 120L);
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                .BEFORE_PROBE, afterInvalidHead.initialHeadDisposition());
        assertModeled(invalidProbe, 100L);
    }

    @Test
    void projectionNeverTouchesRouteSuffixAfterProbeCompletion() {
        RouteProjection.Candidate result = project(
                queue(true, constraints(3, 0L), List.of(
                        item(1L, 100, 1L, 10L),
                        item(2L, 10, 2L, 999L))),
                noCommittedWork(), suffixFailingSingleEvaluator(),
                probe(99L, 90, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(30L, result.projectedTtftMsValue());
        assertEquals("SERIAL_FROZEN_QUEUE", result.detail());
    }

    @Test
    void projectionNeverTouchesLaterBatchGroup() {
        RouteProjection.Candidate result = project(
                queue(true, constraints(1, 0L), List.of(
                        item(1L, 10, 1L, 999L))),
                noCommittedWork(), suffixFailingBatchEvaluator(),
                probe(99L, 90, 20L, 0L),
                BATCH);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
    }

    @Test
    void suffixExpiryDoesNotChangeProbeTtft() {
        GroupPlanner.Item expiringSuffix = item(
                1L, 10, 1L, 100L, NOW_MS + 10L);
        GroupPlanner.Constraints splitByComputeShape =
                new GroupPlanner.Constraints(
                        2, 150L, 1_000_000L, 0L, 30L);
        QueueSnapshot snapshot = queue(
                true, splitByComputeShape, List.of(expiringSuffix));
        RouteProjectionTestSupport.Probe probe = probe(
                99L, 90, NOW_MS - 30L, Long.MAX_VALUE, 20L, 0L);
        RouteProjection.Candidate result = project(
                snapshot, noCommittedWork(), TOKEN_EVALUATOR, probe, ROUTE);

        assertModeled(result, 20L);
    }

    @Test
    void projectionNeverEvaluatesKvBlockAfterProbe() {
        GroupPlanner.Constraints kvLimited = new GroupPlanner.Constraints(
                1, 1_000_000L, 50L, 0L, 30L);

        RouteProjection.Candidate result = project(
                queue(true, kvLimited, List.of(
                        item(1L, 10, 1L, 100L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 100, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition.AFTER_PROBE,
                result.initialHeadDisposition());
    }

    @Test
    void duplicateIncomingIdentityIsRejectedAtActiveAndCommittedBoundaries() {
        RouteProjection.Candidate active = project(
                queue(false, constraints(1, 0L), List.of(
                        item(99L, 50, 1L, 20L))),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);
        RouteProjection.Candidate committed = project(
                queue(false, constraints(1, 0L), List.of()),
                work(List.of(new WorkSnapshot.RequestWork(
                                99L,
                                WorkSnapshot.Phase.ENGINE_RUNNING,
                                20L)),
                        List.of(), 0L),
                TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, active.state());
        assertEquals("INCOMING_ALREADY_ACTIVE", active.detail());
        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE,
                committed.state());
        assertEquals("INCOMING_ALREADY_COMMITTED", committed.detail());
    }

    @Test
    void singletonProjectionEvaluatesOneFrozenPredictorOnce() {
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
                probe(99L, 50, 20L, 0L),
                BATCH);

        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(1, singleCalls.get());
        assertEquals(1, batchCalls.get(),
                "singleton projection evaluates one frozen predictor exactly once");
    }

    @Test
    void singletonProjectionReusesAnIdenticalModelAcrossEndpoints() {
        Object sharedSnapshot = new Object();
        AtomicInteger singleCalls = new AtomicInteger();
        AtomicInteger batchCalls = new AtomicInteger();
        PrefillTimePredictor.Evaluator first = sharedEvaluator(
                sharedSnapshot, singleCalls, batchCalls);
        PrefillTimePredictor.Evaluator second = sharedEvaluator(
                sharedSnapshot, singleCalls, batchCalls);

        QueueSnapshot empty = queue(false,
                new GroupPlanner.Constraints(
                        1, 1_000_000L, 1_000_000L,
                        500L, 0L),
                List.of());
        RouteProjectionTestSupport.Probe probe = probe(
                100L, 50, 20L, 0L);

        assertTrue(project(empty, noCommittedWork(), first, probe, BATCH)
                .selectable());
        assertTrue(project(empty, noCommittedWork(), second, probe, BATCH)
                .selectable());
        assertEquals(1, singleCalls.get());
        assertEquals(1, batchCalls.get());
    }

    private static PrefillTimePredictor.Evaluator sharedEvaluator(
            Object snapshot,
            AtomicInteger singleCalls,
            AtomicInteger batchCalls) {
        return new PrefillTimePredictor.Evaluator() {
            @Override
            public Object snapshotIdentity() {
                return snapshot;
            }

            @Override
            public long estimateMs(long totalTokens, long hitTokens) {
                singleCalls.incrementAndGet();
                return totalTokens - hitTokens;
            }

            @Override
            public double predictBatchMs(PrefillBatchFeatures features) {
                batchCalls.incrementAndGet();
                return features.items().stream()
                        .mapToLong(PrefillBatchFeatures.Item::seqLen)
                        .sum();
            }
        };
    }

    private static void assertModeled(
            RouteProjection.Candidate result,
            long ttftMs) {
        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(ttftMs, result.projectedTtftMsValue());
        assertTrue(result.selectable());
    }

    private static void assertInvalidPrediction(RouteProjection.Candidate result) {
        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, result.state());
        assertEquals("PREDICTOR_RETURNED_INVALID_VALUE", result.detail());
        assertEquals(RouteProjection.Candidate.UNKNOWN, result.projectedTtftMsValue());
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
