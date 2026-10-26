package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation.Candidate;
import org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.project;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract tests for the canonical route-projection value boundary.
 */
class RequestTimeEstimationTest {

    @Test
    void runningWorkRetainsSaturationWhenClockDifferenceWraps() {
        WorkSnapshot snapshot =
                new WorkSnapshot(
                        Long.MIN_VALUE,
                        List.of(
                                new WorkSnapshot.RequestWork(
                                        "1", WorkSnapshot.Phase.ENGINE_RUNNING, Long.MAX_VALUE)),
                        List.of(),
                        0L);

        assertEquals(Long.MAX_VALUE, snapshot.knownRemainingWorkMsAt(Long.MAX_VALUE));
        assertEquals(Long.MAX_VALUE, snapshot.totalRemainingWorkMsAt(Long.MAX_VALUE).orElseThrow());
    }

    @Test
    void workSnapshotRetainsOnlyFrozenIdentitiesAndDurationAccounting() {
        var requests =
                new ArrayList<>(
                        List.of(
                                new WorkSnapshot.RequestWork(
                                        "5", WorkSnapshot.Phase.ENGINE_RUNNING, 30L),
                                new WorkSnapshot.RequestWork(
                                        "1", WorkSnapshot.Phase.COMMITTED, 20L)));
        var members = new ArrayList<>(List.of("3", "2"));
        var batches =
                new ArrayList<>(
                        List.of(
                                new WorkSnapshot.BatchWork(
                                        members,
                                        WorkSnapshot.Phase.ENGINE_QUEUED,
                                        OptionalLong.of(40L)),
                                new WorkSnapshot.BatchWork(
                                        List.of("4"),
                                        WorkSnapshot.Phase.ENGINE_RUNNING,
                                        OptionalLong.of(10L))));
        WorkSnapshot snapshot = new WorkSnapshot(100L, requests, batches, 0L);
        requests.clear();
        batches.clear();
        members.clear();

        for (long id = 1L; id <= 5L; id++) {
            assertTrue(snapshot.containsRequest(String.valueOf(id)));
        }
        assertFalse(snapshot.containsRequest("90"), "an uncaptured request identity is absent");
        assertFalse(snapshot.hasUnknownWork());
        assertEquals(100L, snapshot.capturedAtMs());
        assertEquals(100L, snapshot.totalRemainingWorkMs().orElseThrow());
        assertEquals(100L, snapshot.totalRemainingWorkMsAt(90L).orElseThrow());
        assertEquals(90L, snapshot.totalRemainingWorkMsAt(105L).orElseThrow());
        assertEquals(60L, snapshot.totalRemainingWorkMsAt(150L).orElseThrow());
    }

    @Test
    void unknownWorkPreservesMemberIdentityAndIndependentKnownDuration() {
        WorkSnapshot snapshot =
                new WorkSnapshot(
                        100L,
                        List.of(
                                new WorkSnapshot.RequestWork(
                                        "1", WorkSnapshot.Phase.ENGINE_RUNNING, 20L)),
                        List.of(
                                new WorkSnapshot.BatchWork(
                                        List.of("3", "2"),
                                        WorkSnapshot.Phase.ENGINE_QUEUED,
                                        OptionalLong.empty())),
                        0L);
        assertTrue(snapshot.hasUnknownWork());
        assertTrue(snapshot.containsRequest("1"));
        assertTrue(snapshot.containsRequest("2"));
        assertTrue(snapshot.containsRequest("3"));
        assertTrue(snapshot.totalRemainingWorkMs().isEmpty());
        assertEquals(15L, snapshot.knownRemainingWorkMsAt(105L));
        assertThrows(
                NullPointerException.class,
                () ->
                        new WorkSnapshot(
                                100L,
                                java.util.Arrays.asList((WorkSnapshot.RequestWork) null),
                                List.of(),
                                -1L));
        assertThrows(
                NullPointerException.class,
                () ->
                        new WorkSnapshot(
                                100L,
                                List.of(),
                                java.util.Arrays.asList((WorkSnapshot.BatchWork) null),
                                -1L));
        assertThrows(
                IllegalArgumentException.class,
                () -> new WorkSnapshot(100L, List.of(), List.of(), -1L));
    }

    @Test
    void emptyFixedWindowMatchesTheGenericPlannerWithoutBuildingAGroup() {
        var expired = List.of(new GroupPlanner.Item("1", 0, 1L, 0L, 99L, 1L, 0L));
        var work =
                List.of(
                        emptyWork(100L),
                        new WorkSnapshot(
                                100L,
                                List.of(
                                        new WorkSnapshot.RequestWork(
                                                "1", WorkSnapshot.Phase.COMMITTED, 200L)),
                                List.of(),
                                0L),
                        new WorkSnapshot(
                                100L,
                                List.of(
                                        new WorkSnapshot.RequestWork(
                                                "1",
                                                WorkSnapshot.Phase.ENGINE_RUNNING,
                                                Long.MAX_VALUE)),
                                List.of(),
                                0L),
                        new WorkSnapshot(100L, List.of(), List.of(), 1L));
        for (boolean batch : new boolean[] {false, true}) {
            var delivery = projection(batch);
            for (int maxRequests : new int[] {1, 16}) {
                for (long window : new long[] {0L, 10L, Long.MAX_VALUE}) {
                    for (long kvCapacity : new long[] {19L, 20L, Long.MAX_VALUE}) {
                        for (long enqueuedAt : new long[] {90L, 100L, 110L}) {
                            for (long expiresAt :
                                    new long[] {100L, 101L, 110L, 111L, Long.MAX_VALUE}) {
                                var incoming =
                                        new RequestTimeEstimationTestSupport.Probe(
                                                "99", 0, enqueuedAt, expiresAt, 20L, 3L, 3L);
                                var constraints =
                                        new GroupPlanner.Constraints(
                                                maxRequests, 0L, kvCapacity, 0L, window);
                                for (WorkSnapshot committed : work) {
                                    var fastEvaluator = new CountingEvaluator();
                                    var genericEvaluator = new CountingEvaluator();
                                    var fast =
                                            project(
                                                    new RequestTimeEstimation.Inputs(
                                                            fixedWindow(constraints, List.of()),
                                                            committed,
                                                            0L),
                                                    incoming,
                                                    fastEvaluator,
                                                    delivery,
                                                    100L);
                                    // An expired member forces the existing merge/planner path
                                    // without adding live work.
                                    var generic =
                                            project(
                                                    new RequestTimeEstimation.Inputs(
                                                            fixedWindow(constraints, expired),
                                                            committed,
                                                            0L),
                                                    incoming,
                                                    genericEvaluator,
                                                    delivery,
                                                    100L);
                                    assertEquals(generic.state(), fast.state());
                                    assertEquals(
                                            generic.projectedTtftMsValue(),
                                            fast.projectedTtftMsValue());
                                    assertEquals(
                                            generic.incomingPrefillMs(), fast.incomingPrefillMs());
                                    assertEquals(
                                            genericEvaluator.invocations(),
                                            fastEvaluator.invocations());
                                    if (!fast.selectable()) {
                                        assertEquals(generic.detail(), fast.detail());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void emptyWindowBatchPredictionFailuresMatchTheGenericPlanner() {
        var constraints = new GroupPlanner.Constraints(16, 1_000L, 1_000L, 0L, 10L);
        var incoming =
                new RequestTimeEstimationTestSupport.Probe(
                        "99", 0, 100L, Long.MAX_VALUE, 20L, 0L, 0L);
        for (String failure : List.of("exception", "nan", "negative", "infinity")) {
            var fastEvaluator = failingBatchEvaluator(failure);
            var genericEvaluator = failingBatchEvaluator(failure);
            var fast =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    fixedWindow(constraints, List.of()), emptyWork(100L), 0L),
                            incoming,
                            fastEvaluator,
                            projection(true),
                            100L);
            var generic =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    fixedWindow(
                                            constraints,
                                            List.of(
                                                    new GroupPlanner.Item(
                                                            "1", 0, 1L, 0L, 99L, 1L, 0L))),
                                    emptyWork(100L),
                                    0L),
                            incoming,
                            genericEvaluator,
                            projection(true),
                            100L);
            assertEquals(generic.state(), fast.state());
            assertEquals(generic.detail(), fast.detail());
            assertEquals(generic.projectedTtftMsValue(), fast.projectedTtftMsValue());
        }
    }

    @Test
    void emptySingleQueueMatchesTheGenericPathAndKeepsPolicyValidation() {
        var constraints = new GroupPlanner.Constraints(1, 1_000L, 1_000L, 0L, 0L);
        for (boolean batch : new boolean[] {false, true}) {
            var fastEvaluator = new CountingEvaluator();
            var genericEvaluator = new CountingEvaluator();
            var fast =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    new QueueSnapshot(
                                            100L,
                                            true,
                                            org.flexlb.balance.planner.GroupingPolicy.SINGLE,
                                            Comparator.comparing(GroupPlanner.Item::requestId),
                                            constraints,
                                            List.of(),
                                            null),
                                    emptyWork(100L),
                                    0L),
                            probe(),
                            fastEvaluator,
                            projection(batch),
                            100L);
            var generic =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    new QueueSnapshot(
                                            100L,
                                            true,
                                            org.flexlb.balance.planner.GroupingPolicy.SINGLE,
                                            Comparator.comparing(GroupPlanner.Item::requestId),
                                            constraints,
                                            List.of(
                                                    new GroupPlanner.Item(
                                                            "1", 0, 1L, 0L, 99L, 1L, 0L)),
                                            null),
                                    emptyWork(100L),
                                    0L),
                            probe(),
                            genericEvaluator,
                            projection(batch),
                            100L);
            assertEquals(generic.state(), fast.state());
            assertEquals(generic.projectedTtftMsValue(), fast.projectedTtftMsValue());
            assertEquals(genericEvaluator.invocations(), fastEvaluator.invocations());
        }
        var invalid =
                new QueueSnapshot(
                        100L,
                        true,
                        org.flexlb.balance.planner.GroupingPolicy.SINGLE,
                        Comparator.comparing(GroupPlanner.Item::requestId),
                        new GroupPlanner.Constraints(16, 1_000L, 1_000L, 0L, 0L),
                        List.of(),
                        null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        project(
                                new RequestTimeEstimation.Inputs(invalid, emptyWork(100L), 0L),
                                probe(),
                                new CountingEvaluator(),
                                routeProjection(),
                                100L));
    }

    private static QueueSnapshot fixedWindow(
            GroupPlanner.Constraints constraints, List<GroupPlanner.Item> active) {
        return new QueueSnapshot(
                100L,
                true,
                org.flexlb.balance.planner.GroupingPolicy.FIXED_WINDOW,
                Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                constraints,
                active,
                null);
    }

    private static RequestTimeEstimation.DeliveryProjection projection(boolean batch) {
        return batch
                ? RequestTimeEstimation.forDelivery(org.flexlb.config.DispatcherConfig.Type.BATCH)
                : routeProjection();
    }

    private static PrefillTimePredictor.Evaluator failingBatchEvaluator(String failure) {
        return new PrefillTimePredictor.Evaluator() {
            @Override
            public long estimateMs(long totalTokens, long hitTokens) {
                return 20L;
            }

            @Override
            public double predictBatchMs(PrefillBatchFeatures features) {
                return switch (failure) {
                    case "exception" -> throw new IllegalStateException("predictor unavailable");
                    case "nan" -> Double.NaN;
                    case "negative" -> -1.0d;
                    case "infinity" -> Double.POSITIVE_INFINITY;
                    default -> throw new AssertionError(failure);
                };
            }
        };
    }

    @Test
    void sortedQueueInsertionKeepsEqualItemsAheadAndSkipsExpiredItems() {
        var items = new java.util.ArrayList<GroupPlanner.Item>();
        for (int i = 0; i < 1024; i++) {
            items.add(
                    new GroupPlanner.Item(
                            String.valueOf(i),
                            i / 256,
                            i,
                            0L,
                            i % 2 == 0 ? 13L : Long.MAX_VALUE,
                            1L,
                            0L));
        }
        Comparator<GroupPlanner.Item> order = Comparator.comparingInt(GroupPlanner.Item::priority);
        var queue =
                new QueueSnapshot(
                        13L,
                        true,
                        org.flexlb.balance.planner.GroupingPolicy.FIXED_WINDOW,
                        order,
                        new GroupPlanner.Constraints(1, 1_000_000L, 1_000_000L, 0L, 0L),
                        items,
                        null);
        var incoming =
                new RequestTimeEstimationTestSupport.Probe(
                        "9999", 1, 0L, Long.MAX_VALUE, 20L, 0L, 0L);
        var result =
                project(
                        new RequestTimeEstimation.Inputs(queue, emptyWork(13L), 0L),
                        incoming,
                        new CountingEvaluator(),
                        routeProjection());
        assertTrue(result.selectable());
        // 256 live items of priority 0 or 1 must complete before this probe.
        assertEquals(276L, result.projectedTtftMsValue());
    }

    @Test
    void duplicateIdentityUsesLiveMembershipAndKeepsUnknownWorkConservative() {
        var policy = routeProjection();
        for (long expiry : new long[] {-1L, 0L, 12L, 13L, 14L, Long.MAX_VALUE}) {
            var items =
                    List.of(
                            new GroupPlanner.Item("99", 0, 1L, 0L, 13L, 1L, 0L),
                            new GroupPlanner.Item("99", 0, 2L, 0L, expiry, 1L, 0L));
            var queue =
                    new QueueSnapshot(
                            13L,
                            true,
                            org.flexlb.balance.planner.GroupingPolicy.FIXED_WINDOW,
                            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                            new GroupPlanner.Constraints(1, 1_000L, 1_000L, 0L, 0L),
                            items,
                            null);
            var known =
                    project(
                            new RequestTimeEstimation.Inputs(queue, emptyWork(13L), 0L),
                            probe(),
                            new CountingEvaluator(),
                            policy);
            if (expiry > 13L) {
                assertEquals("INCOMING_ALREADY_ACTIVE", known.detail());
            } else {
                assertTrue(known.selectable());
                assertEquals(20L, known.projectedTtftMsValue());
            }
            var unknown =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    queue, new WorkSnapshot(13L, List.of(), List.of(), 1L), 0L),
                            probe(),
                            new CountingEvaluator(),
                            policy);
            assertEquals("INCOMING_ALREADY_ACTIVE", unknown.detail());
        }
    }

    @Test
    void sameFrozenMembershipExpiresAtTheLaterPlanningClock() {
        var items = new java.util.ArrayList<GroupPlanner.Item>();
        items.add(new GroupPlanner.Item("99", 0, 1L, 0L, 14L, 1L, 0L));
        var queue =
                new QueueSnapshot(
                        13L,
                        true,
                        org.flexlb.balance.planner.GroupingPolicy.FIXED_WINDOW,
                        Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                        new GroupPlanner.Constraints(1, 1_000L, 1_000L, 0L, 0L),
                        items,
                        null);
        items.clear();
        var inputs = new RequestTimeEstimation.Inputs(queue, emptyWork(13L), 0L);
        var policy = routeProjection();
        assertEquals(
                "INCOMING_ALREADY_ACTIVE",
                project(inputs, probe(), new CountingEvaluator(), policy, 13L).detail());
        var expired = project(inputs, probe(), new CountingEvaluator(), policy, 14L);
        assertTrue(expired.selectable());
        assertEquals(20L, expired.projectedTtftMsValue());
        assertEquals(1, queue.activeItems().size());
        assertThrows(UnsupportedOperationException.class, () -> queue.activeItems().clear());
    }

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
                project(
                        new RequestTimeEstimation.Inputs(emptyQueue(20L), current, 0L),
                        probe(),
                        new CountingEvaluator(),
                        routeProjection()),
                project(
                        new RequestTimeEstimation.Inputs(emptyQueue(20L), cached, 0L),
                        probe(),
                        new CountingEvaluator(),
                        routeProjection()));
    }

    @Test
    void queueAndWorkComeFromOneCanonicalInput() {
        RequestTimeEstimation.Inputs inputs = inputs(13L);
        CountingEvaluator evaluator = new CountingEvaluator();

        Candidate result = project(inputs, probe(), evaluator, routeProjection());

        assertTrue(result.selectable());
        assertEquals(20L, result.incomingPrefillMs());
        assertTrue(evaluator.invocations() > 0);

        assertThrows(
                IllegalArgumentException.class,
                () -> new RequestTimeEstimation.Inputs(emptyQueue(13L), emptyWork(14L), 0L));
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
        Candidate result = project(inputs(31L), probe(), captured, routeProjection());
        assertTrue(result.selectable());
        assertEquals(
                before,
                captured.estimateMs(20L, 0L),
                "a projection-owned evaluator cannot change mid-call");
    }

    @Test
    void onlyModeledCandidateCarriesKnownTtft() {
        assertThrows(
                IllegalArgumentException.class,
                () -> candidate(Candidate.State.MODELED, OptionalLong.empty()));
        assertThrows(
                IllegalArgumentException.class,
                () -> candidate(Candidate.State.UNAVAILABLE, OptionalLong.of(1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> candidate(Candidate.State.MODELED, OptionalLong.of(-2L)));

        Candidate modeled = candidate(Candidate.State.MODELED, OptionalLong.of(10L));
        Candidate unmodeled =
                candidate(Candidate.State.UNMODELED_ENGINE_WORK, OptionalLong.empty());
        assertEquals(10L, modeled.projectedTtftMsValue());
        assertTrue(unmodeled.engineWorkUnmodeled());
    }

    private static Candidate candidate(Candidate.State state, OptionalLong projectedTtftMs) {
        return new Candidate(
                state,
                projectedTtftMs.orElse(Candidate.UNKNOWN),
                0L,
                Candidate.InitialHeadDisposition.NONE,
                state.name(),
                null,
                0L,
                0L);
    }

    private static RequestTimeEstimation.Inputs inputs(long capturedAtMs) {
        return new RequestTimeEstimation.Inputs(
                emptyQueue(capturedAtMs), emptyWork(capturedAtMs), 0L);
    }

    private static QueueSnapshot emptyQueue(long capturedAtMs) {
        return new QueueSnapshot(
                capturedAtMs,
                true,
                org.flexlb.balance.planner.GroupingPolicy.FIXED_WINDOW,
                Comparator.comparing(GroupPlanner.Item::requestId),
                new GroupPlanner.Constraints(1, 1_000_000L, 1_000_000L, 0L, 30L),
                List.of(),
                null);
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 0L);
    }

    private static RequestTimeEstimationTestSupport.Probe probe() {
        return new RequestTimeEstimationTestSupport.Probe(
                "99", 50, 13L, Long.MAX_VALUE, 20L, 0L, 0L);
    }

    private static RequestTimeEstimation.DeliveryProjection routeProjection() {
        return RequestTimeEstimation.forDelivery(org.flexlb.config.DispatcherConfig.Type.NON_BATCH);
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
