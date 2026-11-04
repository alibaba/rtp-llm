package org.flexlb.balance.projection;

import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.LearningPredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.projection.RequestTimeEstimation.Candidate;
import org.flexlb.balance.projection.RequestTimeEstimation.QueueSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.Probe;
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
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Contract tests for the canonical route-projection value boundary. */
class RequestTimeEstimationTest {

    @Test
    void frozenWorkCanBeReadConcurrentlyWithoutAnExtraCaptureObject() throws Exception {
        var requests = new java.util.ArrayList<WorkSnapshot.RequestWork>();
        requests.add(new WorkSnapshot.RequestWork(7L, WorkSnapshot.Phase.ENGINE_RUNNING, 30L));
        var batches = new java.util.ArrayList<WorkSnapshot.BatchWork>();
        batches.add(
                new WorkSnapshot.BatchWork(
                        List.of(9L, 8L),
                        WorkSnapshot.Phase.COMMITTED,
                        java.util.OptionalLong.of(40L)));
        var snapshot = new WorkSnapshot(100L, requests, batches, 0L);
        requests.clear();
        batches.clear();
        try (var readers = java.util.concurrent.Executors.newFixedThreadPool(8)) {
            var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int i = 0; i < 32; i++) {
                futures.add(
                        readers.submit(
                                () -> {
                                    assertEquals(60L, snapshot.knownRemainingWorkMsAt(110L));
                                    assertTrue(snapshot.containsRequest(7L));
                                    assertTrue(snapshot.containsRequest(8L));
                                    assertTrue(snapshot.containsRequest(9L));
                                    assertFalse(snapshot.containsRequest(10L));
                                    assertFalse(snapshot.hasUnknownWork());
                                    assertSame(snapshot, snapshot.calculate());
                                }));
            }
            for (var future : futures) future.get(5, java.util.concurrent.TimeUnit.SECONDS);
        }
    }

    @Test
    void customPlannerIsUsedForEveryProjectedPrefix() {
        var planner = new org.flexlb.balance.planner.FirstRequestPlanner();
        var queue =
                new QueueSnapshot(
                        31,
                        true,
                        planner,
                        Comparator.comparingLong(GroupPlanner.Item::enqueuedAtMs),
                        new GroupPlanner.Constraints(16, 1000, 1000, 0, 0),
                        List.of(new GroupPlanner.Item(1, 0, 1, 10, Long.MAX_VALUE, 10, 0)),
                        null);
        var inputs =
                new RequestTimeEstimation.Inputs(
                        queue,
                        emptyWork(31),
                        0,
                        new CountingEvaluator(),
                        org.flexlb.config.DispatcherConfig.Type.BATCH);
        var result = project(inputs, new Probe(99, 0, 13, Long.MAX_VALUE, 20, 0, 0));
        assertEquals(30L, result.projectedTtftMsValue());
        assertEquals(2, planner.selections.get());
        assertEquals(1, queue.activeItems().size());
    }

    @Test
    void reusedProjectionReadsEveryIncomingSchedulingAndPredictionField() {
        var queue =
                new QueueSnapshot(
                        31L,
                        true,
                        BuiltinGroupPlanner.SINGLE,
                        Comparator.comparingInt(GroupPlanner.Item::priority)
                                .reversed()
                                .thenComparingLong(GroupPlanner.Item::enqueuedAtMs),
                        new GroupPlanner.Constraints(1, 1000, 1000, 0, 0),
                        List.of(new GroupPlanner.Item(1, 0, 1, 10, Long.MAX_VALUE, 10, 0)),
                        null);
        var inputs =
                new RequestTimeEstimation.Inputs(
                        queue, emptyWork(31), 0, new CountingEvaluator(), routeProjection());
        var original = new Probe(99, 0, 13, Long.MAX_VALUE, 20, 0, 0);
        var changed =
                List.of(
                        new Probe(99, 1, 13, Long.MAX_VALUE, 20, 0, 0),
                        new Probe(99, 0, 5, Long.MAX_VALUE, 20, 0, 0),
                        new Probe(99, 0, 13, Long.MAX_VALUE, 50, 0, 0),
                        new Probe(99, 0, 13, Long.MAX_VALUE, 20, 7, 7));
        long[] expected = {20, 20, 60, 23};
        for (int index = 0; index < changed.size(); index++) {
            assertEquals(30L, project(inputs, original).projectedTtftMsValue());
            assertEquals(
                    expected[index], project(inputs, changed.get(index)).projectedTtftMsValue());
        }
        assertEquals(30L, project(inputs, original).projectedTtftMsValue());
        assertEquals(
                "INCOMING_ALREADY_ACTIVE",
                project(inputs, new Probe(1, 0, 13, Long.MAX_VALUE, 20, 0, 0)).detail());
        assertEquals(
                "INCOMING_EXPIRED", project(inputs, new Probe(99, 0, 13, 30, 20, 0, 0)).detail());
        assertEquals(30L, project(inputs, original).projectedTtftMsValue());
        assertEquals(
                30L, project(inputs, new Probe(100, 0, 13, 100, 20, 0, 0)).projectedTtftMsValue());
        var laterQueue =
                new QueueSnapshot(
                        150,
                        true,
                        BuiltinGroupPlanner.SINGLE,
                        queue.ordering(),
                        queue.constraints(),
                        queue.activeItems(),
                        null);
        var later =
                new RequestTimeEstimation.Inputs(
                        laterQueue, emptyWork(150), 0, inputs.evaluator(), inputs.deliveryMode());
        assertEquals(
                30L, project(later, new Probe(100, 0, 13, 200, 20, 0, 0)).projectedTtftMsValue());
    }

    @Test
    void recursiveBatchPredictionCannotOverwriteTheOuterQueueCursor() {
        var queue =
                new QueueSnapshot(
                        31L,
                        true,
                        BuiltinGroupPlanner.FIXED_WINDOW,
                        Comparator.comparingLong(GroupPlanner.Item::requestId),
                        new GroupPlanner.Constraints(1, 1000, 1000, 0, 30),
                        List.of(new GroupPlanner.Item(1, 0, 1, 13, Long.MAX_VALUE, 10, 0)),
                        null);
        var evaluator =
                new PrefillTimePredictor.Evaluator() {
                    public long estimateMs(long tokens, long hit) {
                        return tokens - hit;
                    }

                    public double predictBatchMs(PrefillBatchFeatures features) {
                        var nested =
                                project(
                                        new RequestTimeEstimation.Inputs(
                                                queue,
                                                emptyWork(31),
                                                0,
                                                new CountingEvaluator(),
                                                routeProjection()),
                                        new Probe(100L, 0, 13L, Long.MAX_VALUE, 700L, 100L, 200L));
                        assertEquals(610L, nested.projectedTtftMsValue());
                        return features.items().stream()
                                .mapToLong(i -> i.seqLen() - i.hitCache())
                                .sum();
                    }
                };
        for (int repeat = 0; repeat < 3; repeat++) {
            var result =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    queue,
                                    emptyWork(31),
                                    0,
                                    evaluator,
                                    org.flexlb.config.DispatcherConfig.Type.BATCH),
                            probe());
            assertEquals(Candidate.State.MODELED, result.state());
            assertEquals(30L, result.projectedTtftMsValue());
            assertEquals(1, queue.activeItems().size());
        }
    }

    @Test
    void nestedProjectionCannotReplaceTheOuterRequestOrPredictor() {
        var nestedModel = new CountingEvaluator();
        var outerModel =
                new PrefillTimePredictor.Evaluator() {
                    @Override
                    public long estimateMs(long tokens, long hit) {
                        var nested =
                                project(
                                        inputs(31L, nestedModel),
                                        new Probe(100L, 0, 13L, Long.MAX_VALUE, 700L, 100L, 200L));
                        assertEquals(600L, nested.projectedTtftMsValue());
                        return tokens - hit;
                    }

                    @Override
                    public double predictBatchMs(PrefillBatchFeatures features) {
                        return features.items().stream()
                                .mapToLong(i -> i.seqLen() - i.hitCache())
                                .sum();
                    }
                };
        var result = project(inputs(31L, outerModel), probe());
        assertEquals(Candidate.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(20L, result.incomingPrefillMs());
        assertEquals(0L, result.cacheHitTokens());
        assertEquals(0L, result.routingCacheMatchTokens());
    }

    @Test
    void batchSessionCreationFailureBecomesAnUnavailableProjection() {
        var evaluator =
                new PrefillTimePredictor.Evaluator() {
                    public long estimateMs(long tokens, long hit) {
                        return tokens;
                    }

                    public double predictBatchMs(PrefillBatchFeatures features) {
                        return 1;
                    }

                    public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
                        throw new IllegalStateException("session creation failed");
                    }
                };
        var queue =
                new QueueSnapshot(
                        31L,
                        true,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.FIXED_WINDOW,
                        Comparator.comparingLong(GroupPlanner.Item::requestId),
                        new GroupPlanner.Constraints(4, 1000, 1000, 100, 30),
                        List.of(new GroupPlanner.Item(1, 0, 1, 13, Long.MAX_VALUE, 10, 0)),
                        null);
        var result =
                project(
                        new RequestTimeEstimation.Inputs(
                                queue,
                                emptyWork(31),
                                0,
                                evaluator,
                                org.flexlb.config.DispatcherConfig.Type.BATCH),
                        probe());
        assertEquals(Candidate.State.UNAVAILABLE, result.state());
        assertEquals("BATCH_PREDICTION_FAILED", result.detail());
    }

    @Test
    void runningWorkRetainsSaturationWhenClockDifferenceWraps() {
        WorkSnapshot snapshot =
                new WorkSnapshot(
                        Long.MIN_VALUE,
                        List.of(
                                new WorkSnapshot.RequestWork(
                                        1L, WorkSnapshot.Phase.ENGINE_RUNNING, Long.MAX_VALUE)),
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
                                        5L, WorkSnapshot.Phase.ENGINE_RUNNING, 30L),
                                new WorkSnapshot.RequestWork(
                                        1L, WorkSnapshot.Phase.COMMITTED, 20L)));
        var members = new ArrayList<>(List.of(3L, 2L));
        var batches =
                new ArrayList<>(
                        List.of(
                                new WorkSnapshot.BatchWork(
                                        members,
                                        WorkSnapshot.Phase.ENGINE_QUEUED,
                                        OptionalLong.of(40L)),
                                new WorkSnapshot.BatchWork(
                                        List.of(4L),
                                        WorkSnapshot.Phase.ENGINE_RUNNING,
                                        OptionalLong.of(10L))));
        WorkSnapshot snapshot = new WorkSnapshot(100L, requests, batches, 0L);
        requests.clear();
        batches.clear();
        members.clear();

        for (long id = 1L; id <= 5L; id++) {
            assertTrue(snapshot.containsRequest(id));
        }
        assertFalse(snapshot.containsRequest(90L), "an uncaptured request identity is absent");
        assertFalse(snapshot.containsRequest(Long.MIN_VALUE));
        assertFalse(snapshot.containsRequest(Long.MAX_VALUE));
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
                                        1L, WorkSnapshot.Phase.ENGINE_RUNNING, 20L)),
                        List.of(
                                new WorkSnapshot.BatchWork(
                                        List.of(3L, 2L),
                                        WorkSnapshot.Phase.ENGINE_QUEUED,
                                        OptionalLong.empty())),
                        0L);
        assertTrue(snapshot.hasUnknownWork());
        assertTrue(snapshot.containsRequest(1L));
        assertTrue(snapshot.containsRequest(2L));
        assertTrue(snapshot.containsRequest(3L));
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
        var expired = List.of(new GroupPlanner.Item(1L, 0, 1L, 0L, 99L, 1L, 0L));
        var work =
                List.of(
                        emptyWork(100L),
                        new WorkSnapshot(
                                100L,
                                List.of(
                                        new WorkSnapshot.RequestWork(
                                                1L, WorkSnapshot.Phase.COMMITTED, 200L)),
                                List.of(),
                                0L),
                        new WorkSnapshot(
                                100L,
                                List.of(
                                        new WorkSnapshot.RequestWork(
                                                1L,
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
                                        new Probe(99L, 0, enqueuedAt, expiresAt, 20L, 3L, 3L);
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
                                                            0L,
                                                            fastEvaluator,
                                                            delivery),
                                                    incoming,
                                                    100L);
                                    // An expired member forces the existing merge/planner path
                                    // without adding live work.
                                    var generic =
                                            project(
                                                    new RequestTimeEstimation.Inputs(
                                                            fixedWindow(constraints, expired),
                                                            committed,
                                                            0L,
                                                            genericEvaluator,
                                                            delivery),
                                                    incoming,
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
        var incoming = new Probe(99L, 0, 100L, Long.MAX_VALUE, 20L, 0L, 0L);
        for (String failure : List.of("exception", "nan", "negative", "infinity")) {
            var fastEvaluator = failingBatchEvaluator(failure);
            var genericEvaluator = failingBatchEvaluator(failure);
            var fast =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    fixedWindow(constraints, List.of()),
                                    emptyWork(100L),
                                    0L,
                                    fastEvaluator,
                                    projection(true)),
                            incoming,
                            100L);
            var generic =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    fixedWindow(
                                            constraints,
                                            List.of(
                                                    new GroupPlanner.Item(
                                                            1L, 0, 1L, 0L, 99L, 1L, 0L))),
                                    emptyWork(100L),
                                    0L,
                                    genericEvaluator,
                                    projection(true)),
                            incoming,
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
                                            org.flexlb.balance.planner.BuiltinGroupPlanner.SINGLE,
                                            Comparator.comparingLong(GroupPlanner.Item::requestId),
                                            constraints,
                                            List.of(),
                                            null),
                                    emptyWork(100L),
                                    0L,
                                    fastEvaluator,
                                    projection(batch)),
                            probe(),
                            100L);
            var generic =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    new QueueSnapshot(
                                            100L,
                                            true,
                                            org.flexlb.balance.planner.BuiltinGroupPlanner.SINGLE,
                                            Comparator.comparingLong(GroupPlanner.Item::requestId),
                                            constraints,
                                            List.of(
                                                    new GroupPlanner.Item(
                                                            1L, 0, 1L, 0L, 99L, 1L, 0L)),
                                            null),
                                    emptyWork(100L),
                                    0L,
                                    genericEvaluator,
                                    projection(batch)),
                            probe(),
                            100L);
            assertEquals(generic.state(), fast.state());
            assertEquals(generic.projectedTtftMsValue(), fast.projectedTtftMsValue());
            assertEquals(genericEvaluator.invocations(), fastEvaluator.invocations());
        }
        var invalid =
                new QueueSnapshot(
                        100L,
                        true,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.SINGLE,
                        Comparator.comparingLong(GroupPlanner.Item::requestId),
                        new GroupPlanner.Constraints(16, 1_000L, 1_000L, 0L, 0L),
                        List.of(),
                        null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        project(
                                new RequestTimeEstimation.Inputs(
                                        invalid,
                                        emptyWork(100L),
                                        0L,
                                        new CountingEvaluator(),
                                        routeProjection()),
                                probe(),
                                100L));
    }

    private static QueueSnapshot fixedWindow(
            GroupPlanner.Constraints constraints, List<GroupPlanner.Item> active) {
        return new QueueSnapshot(
                100L,
                true,
                org.flexlb.balance.planner.BuiltinGroupPlanner.FIXED_WINDOW,
                Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                constraints,
                active,
                null);
    }

    private static org.flexlb.config.DispatcherConfig.Type projection(boolean batch) {
        return batch ? org.flexlb.config.DispatcherConfig.Type.BATCH : routeProjection();
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
                            i, i / 256, i, 0L, i % 2 == 0 ? 13L : Long.MAX_VALUE, 1L, 0L));
        }
        Comparator<GroupPlanner.Item> order = Comparator.comparingInt(GroupPlanner.Item::priority);
        var queue =
                new QueueSnapshot(
                        13L,
                        true,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.FIXED_WINDOW,
                        order,
                        new GroupPlanner.Constraints(1, 1_000_000L, 1_000_000L, 0L, 0L),
                        items,
                        null);
        var incoming = new Probe(9999L, 1, 0L, Long.MAX_VALUE, 20L, 0L, 0L);
        var result =
                project(
                        new RequestTimeEstimation.Inputs(
                                queue,
                                emptyWork(13L),
                                0L,
                                new CountingEvaluator(),
                                routeProjection()),
                        incoming);
        assertTrue(result.selectable());
        // 256 live items of priority 0 or 1 must complete before this probe.
        assertEquals(276L, result.projectedTtftMsValue());
    }

    @Test
    void duplicateIdentityUsesLiveMembershipRegardlessOfUnknownWork() {
        var policy = routeProjection();
        for (long expiry : new long[] {-1L, 0L, 12L, 13L, 14L, Long.MAX_VALUE}) {
            var items =
                    List.of(
                            new GroupPlanner.Item(99L, 0, 1L, 0L, 13L, 1L, 0L),
                            new GroupPlanner.Item(99L, 0, 2L, 0L, expiry, 1L, 0L));
            var queue =
                    new QueueSnapshot(
                            13L,
                            true,
                            org.flexlb.balance.planner.BuiltinGroupPlanner.FIXED_WINDOW,
                            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                            new GroupPlanner.Constraints(1, 1_000L, 1_000L, 0L, 0L),
                            items,
                            null);
            var known =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    queue, emptyWork(13L), 0L, new CountingEvaluator(), policy),
                            probe());
            if (expiry > 13L) {
                assertEquals("INCOMING_ALREADY_ACTIVE", known.detail());
            } else {
                assertTrue(known.selectable());
                assertEquals(20L, known.projectedTtftMsValue());
            }
            var unknown =
                    project(
                            new RequestTimeEstimation.Inputs(
                                    queue,
                                    new WorkSnapshot(13L, List.of(), List.of(), 1L),
                                    0L,
                                    new CountingEvaluator(),
                                    policy),
                            probe());
            assertEquals(known.state(), unknown.state());
            assertEquals(known.detail(), unknown.detail());
            assertEquals(known.projectedTtftMsValue(), unknown.projectedTtftMsValue());
        }
    }

    @Test
    void sameFrozenMembershipExpiresAtTheLaterPlanningClock() {
        var items = new java.util.ArrayList<GroupPlanner.Item>();
        items.add(new GroupPlanner.Item(99L, 0, 1L, 0L, 14L, 1L, 0L));
        var queue =
                new QueueSnapshot(
                        13L,
                        true,
                        org.flexlb.balance.planner.BuiltinGroupPlanner.FIXED_WINDOW,
                        Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                        new GroupPlanner.Constraints(1, 1_000L, 1_000L, 0L, 0L),
                        items,
                        null);
        items.clear();
        var inputs =
                new RequestTimeEstimation.Inputs(
                        queue, emptyWork(13L), 0L, new CountingEvaluator(), routeProjection());
        assertEquals("INCOMING_ALREADY_ACTIVE", project(inputs, probe(), 13L).detail());
        var expired = project(inputs, probe(), 14L);
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
                                        1L, WorkSnapshot.Phase.ENGINE_RUNNING, 1_000L)),
                        List.of(),
                        0L);
        WorkSnapshot current =
                new WorkSnapshot(
                        20L,
                        List.of(
                                new WorkSnapshot.RequestWork(
                                        1L, WorkSnapshot.Phase.ENGINE_RUNNING, 993L)),
                        List.of(),
                        0L);
        assertEquals(
                project(
                        new RequestTimeEstimation.Inputs(
                                emptyQueue(20L),
                                current,
                                0L,
                                new CountingEvaluator(),
                                routeProjection()),
                        probe()),
                project(
                        new RequestTimeEstimation.Inputs(
                                emptyQueue(20L),
                                cached,
                                0L,
                                new CountingEvaluator(),
                                routeProjection()),
                        probe()));
    }

    @Test
    void queueAndWorkComeFromOneCanonicalInput() {
        CountingEvaluator evaluator = new CountingEvaluator();
        RequestTimeEstimation.Inputs inputs = inputs(13L, evaluator);

        Candidate result = project(inputs, probe());

        assertTrue(result.selectable());
        assertEquals(20L, result.incomingPrefillMs());
        assertTrue(evaluator.invocations() > 0);

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new RequestTimeEstimation.Inputs(
                                emptyQueue(13L), emptyWork(14L), 0L, evaluator, routeProjection()));
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
        Candidate result = project(inputs(31L, captured), probe());
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
                0L,
                0L);
    }

    private static RequestTimeEstimation.Inputs inputs(
            long capturedAtMs, PrefillTimePredictor.Evaluator evaluator) {
        return new RequestTimeEstimation.Inputs(
                emptyQueue(capturedAtMs),
                emptyWork(capturedAtMs),
                0L,
                evaluator,
                routeProjection());
    }

    private static QueueSnapshot emptyQueue(long capturedAtMs) {
        return new QueueSnapshot(
                capturedAtMs,
                true,
                org.flexlb.balance.planner.BuiltinGroupPlanner.FIXED_WINDOW,
                Comparator.comparingLong(GroupPlanner.Item::requestId),
                new GroupPlanner.Constraints(1, 1_000_000L, 1_000_000L, 0L, 30L),
                List.of(),
                null);
    }

    private static WorkSnapshot emptyWork(long capturedAtMs) {
        return new WorkSnapshot(capturedAtMs, List.of(), List.of(), 0L);
    }

    private static Probe probe() {
        return new Probe(99L, 50, 13L, Long.MAX_VALUE, 20L, 0L, 0L);
    }

    private static org.flexlb.config.DispatcherConfig.Type routeProjection() {
        return org.flexlb.config.DispatcherConfig.Type.NON_BATCH;
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
