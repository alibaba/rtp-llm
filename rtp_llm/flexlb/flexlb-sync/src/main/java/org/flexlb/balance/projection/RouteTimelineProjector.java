package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;

/** Pure frozen-snapshot TTFT projection shared by endpoint selection policies. */
final class RouteTimelineProjector {

    private static final String INVALID_PREDICTION_DETAIL =
            "PREDICTOR_RETURNED_INVALID_VALUE";
    private final PredictionBoundary predictions = new PredictionBoundary();
    private long requestId;
    private int priority;
    private long enqueuedAtMs;
    private long expiresAtMs;
    private long seqLen;
    private long hitCache;
    private long routingCacheMatchTokens;
    private final ProjectedCandidate result = new ProjectedCandidate();

    RouteTimelineProjector() {
    }

    void reset(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens) {
        this.requestId = requestId;
        this.priority = priority;
        this.enqueuedAtMs = enqueuedAtMs;
        this.expiresAtMs = expiresAtMs;
        this.seqLen = seqLen;
        this.hitCache = hitCache;
        this.routingCacheMatchTokens = routingCacheMatchTokens;
    }

    /**
     * Project one endpoint under a frozen serial-work model.
     *
     * <p>Already committed work forms the engine cursor. Collection deadlines
     * and that cursor overlap via {@code max(cursor, readyAt)}. The snapshot's
     * frozen delivery projection defines each group's completion shape.
     */
    RouteProjection.CandidateView project(
            QueueSnapshot queue,
            WorkSnapshot committed,
            PrefillTimePredictor.Evaluator evaluator,
            RouteProjection.DeliveryProjection deliveryProjection,
            long planningAtMs) {
        long projectionAtMs = Math.max(queue.capturedAtMs(), planningAtMs);
        if (evaluator == null) {
            return unavailable("PREDICTOR_MISSING");
        }
        if (expiresAtMs <= 0L || projectionAtMs >= expiresAtMs) {
            return unavailable("INCOMING_EXPIRED");
        }

        predictions.reset(evaluator);
        try {
            return projectWithPredictions(
                    queue, committed, deliveryProjection,
                    projectionAtMs, predictions);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("PREDICTION_FAILED"));
        } finally {
            predictions.reset(null);
        }
    }

    private RouteProjection.CandidateView projectWithPredictions(
            QueueSnapshot queue,
            WorkSnapshot committed,
            RouteProjection.DeliveryProjection deliveryProjection,
            long projectionAtMs,
            PredictionBoundary predictions) {
        if (containsCommittedRequest(committed, requestId)) {
            return unavailable("INCOMING_ALREADY_COMMITTED");
        }

        final long incomingPrefillMs;
        try {
            incomingPrefillMs = predictions.singleMs(
                    seqLen, hitCache);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(
                    predictionFailure.detail("SINGLE_PREDICTION_FAILED"));
        }

        if (committed.hasUnknownWork()) {
            if (containsActiveRequest(queue, requestId)) {
                return unavailable("INCOMING_ALREADY_ACTIVE");
            }
            if (queue.queueScheduling()) {
                if (seqLen > queue.constraints().batchKvCapacity()) {
                    return blocked(
                            incomingPrefillMs,
                            RouteProjection.Candidate.InitialHeadDisposition.NONE,
                            "PREFILL_KV_CAPACITY");
                }
            }
            return unmodeledEngineWork(incomingPrefillMs);
        }

        long committedMs = knownRemainingWorkMsAt(committed, projectionAtMs);

        /*
         * With no active endpoint queue, known committed work only advances
         * the serial engine cursor. The incoming request is still a singleton
         * decision, so the generic ProjectedQueue/readiness machinery is
         * exactly equivalent to committedMs + singleton completion offset.
         */
        if (queue.queueScheduling() && queue.activeItems().isEmpty()) {
            long collectionDeadline = GroupPlanner.collectionDeadlineMs(
                    enqueuedAtMs,
                    queue.constraints().collectionWindowMs());
            if (expiresAtMs <= collectionDeadline) {
                return unavailable("INCOMING_EXPIRED_BEFORE_DISPATCH");
            }
            return projectIdleSingleton(
                    deliveryProjection, predictions,
                    incomingPrefillMs, committedMs);
        }

        if (!queue.queueScheduling()) {
            long completionMs = saturatedAdd(committedMs, incomingPrefillMs);
            return candidate(
                    RouteProjection.Candidate.State.MODELED,
                    completionMs,
                    incomingPrefillMs,
                    RouteProjection.Candidate.InitialHeadDisposition.NONE,
                    "SERIAL_FROZEN_DIRECT");
        }
        GroupPlanner.Item probe = new GroupPlanner.Item(
                requestId, priority, Long.MAX_VALUE, enqueuedAtMs,
                expiresAtMs, seqLen, hitCache);
        GroupPlanner.Item initialActiveHead =
                queue.activeItems().isEmpty() ? null : queue.activeItems().getFirst();
        RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition =
                initialActiveHead == null
                        ? RouteProjection.Candidate.InitialHeadDisposition.NONE
                        : null;
        List<GroupPlanner.Item> eligibleActive = new ArrayList<>(
                queue.activeItems().size());
        for (GroupPlanner.Item item : queue.activeItems()) {
            // Expired work is a terminal queue mutation, not work ahead of a
            // new request in this snapshot projection.
            if (expiredAt(item, projectionAtMs)) {
                if (item == initialActiveHead) {
                    initialHeadDisposition =
                            RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
                }
                continue;
            }
            if (item.requestId() == requestId) {
                return unavailable("INCOMING_ALREADY_ACTIVE");
            }
            eligibleActive.add(item);
        }
        ProjectedQueue ordered = ProjectedQueue.create(
                eligibleActive,
                probe,
                initialActiveHead,
                queue.ordering());

        if (initialHeadDisposition == null) {
            initialHeadDisposition = ordered.initialHeadDisposition();
        }
        long cursorMs = committedMs;
        long decisionNowMs = projectionAtMs;

        while (!ordered.isEmpty()) {
            ExpirationPrune expiration = ordered.pruneExpired(decisionNowMs);
            if (expiration.initialHeadExpired()) {
                initialHeadDisposition =
                        RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
            }
            if (expiration.probeExpired()) {
                return unavailable("INCOMING_EXPIRED_BEFORE_DISPATCH");
            }
            if (ordered.isEmpty()) {
                break;
            }

            GroupPlanner.Item head = ordered.head();
            if (head.seqLen() > queue.constraints().batchKvCapacity()) {
                return blocked(
                        incomingPrefillMs,
                        initialHeadDisposition,
                        "PREFILL_KV_CAPACITY");
            }

            final GroupPlanner.Plan<GroupPlanner.Item> plan;
            try {
                RouteProjection.GroupPlanning planning =
                        deliveryProjection.planning(predictions);
                plan = GroupPlanner.plan(
                        ordered,
                        GroupPlanner.itemAccess(),
                        queue.constraints(),
                        decisionNowMs,
                        items -> {
                            int projectedProbeIndex =
                                    identityIndexOf(items, probe);
                            int requiredThroughIndex = projectedProbeIndex >= 0
                                    ? projectedProbeIndex
                                    : items.size() - 1;
                            return planning.durationMs(
                                    items, requiredThroughIndex);
                        });
            } catch (PredictionFailure predictionFailure) {
                return unavailable(
                        predictionFailure.detail("BATCH_PREDICTION_FAILED"));
            }
            if (plan.items().isEmpty()) {
                throw new IllegalStateException(
                        "non-empty projected queue produced an empty group");
            }

            if (!plan.ready()) {
                // The production worker wakes for the collection deadline or
                // the current head's absolute expiry, then validates every
                // candidate again. Advancing this explicit scheduling clock
                // and replanning reproduces that terminal cleanup without
                // pretending expired work consumes engine service.
                decisionNowMs = Math.min(
                        plan.collectionDeadlineMs(), head.expiresAtMs());
                continue;
            }

            long readyInMs = elapsedFromNow(projectionAtMs, decisionNowMs);
            long startMs = Math.max(cursorMs, readyInMs);

            int probeIndex = identityIndexOf(plan.items(), probe);
            try {
                RouteProjection.GroupService service =
                        deliveryProjection.service(plan, predictions);
                if (probeIndex >= 0) {
                    return candidate(
                            RouteProjection.Candidate.State.MODELED,
                            saturatedAdd(startMs, service.completionOffsetMs(probeIndex)),
                            incomingPrefillMs,
                            initialHeadDisposition,
                            "SERIAL_FROZEN_QUEUE");
                }
                cursorMs = saturatedAdd(startMs, service.totalDurationMs());
            } catch (PredictionFailure predictionFailure) {
                return unavailable(
                        predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
            }
            ordered.removePlannedPrefix(plan.items().size());
        }
        throw new IllegalStateException(
                "projected queue exhausted before planning the probe");
    }

    private RouteProjection.CandidateView projectIdleSingleton(
            RouteProjection.DeliveryProjection deliveryProjection,
            PredictionBoundary predictions,
            long incomingPrefillMs,
            long committedMs) {
        final long completionMs;
        try {
            completionMs = saturatedAdd(
                    committedMs,
                    deliveryProjection.singletonCompletionOffsetMs(
                            seqLen, hitCache, predictions));
        } catch (PredictionFailure predictionFailure) {
            return unavailable(
                    predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
        }
        return candidate(
                RouteProjection.Candidate.State.MODELED,
                completionMs,
                incomingPrefillMs,
                RouteProjection.Candidate.InitialHeadDisposition.NONE,
                "EMPTY_ACTIVE_QUEUE_SINGLETON");
    }

    private static boolean containsCommittedRequest(
            WorkSnapshot committed, long requestId) {
        return committed.containsRequest(requestId);
    }

    private static boolean containsActiveRequest(
            QueueSnapshot queue, long requestId) {
        for (GroupPlanner.Item item : queue.activeItems()) {
            if (item.requestId() == requestId) {
                return true;
            }
        }
        return false;
    }

    /** Remaining committed work normalized to the projection's common clock. */
    private static long knownRemainingWorkMsAt(
            WorkSnapshot committed, long projectionAtMs) {
        return committed.knownRemainingWorkMsAt(projectionAtMs);
    }

    private static boolean expiredAt(
            GroupPlanner.Item item, long nowMs) {
        return item.expiresAtMs() <= 0L || nowMs >= item.expiresAtMs();
    }

    private record ExpirationPrune(
            boolean probeExpired,
            boolean initialHeadExpired) {
    }

    private static int identityIndexOf(
            List<GroupPlanner.Item> items,
            GroupPlanner.Item target) {
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i) == target) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Ordered snapshot plus probe. Prefix consumption advances an index; expiry
     * clears individual slots. The heap visits each deadline once, including
     * deadlines belonging to already consumed items.
     */
    private static final class ProjectedQueue implements Iterable<GroupPlanner.Item> {

        private final GroupPlanner.Item[] itemsInQueueOrder;
        private final PriorityQueue<Integer> expiryIndexes;
        private final int probeIndex;
        private final int initialHeadIndex;
        private int headIndex;

        private ProjectedQueue(
                GroupPlanner.Item[] itemsInQueueOrder,
                int probeIndex,
                int initialHeadIndex) {
            this.itemsInQueueOrder = itemsInQueueOrder;
            this.expiryIndexes = new PriorityQueue<>(itemsInQueueOrder.length,
                    Comparator.comparingLong((Integer index) -> itemsInQueueOrder[index].expiresAtMs())
                            .thenComparingInt(Integer::intValue));
            for (int index = 0; index < itemsInQueueOrder.length; index++) {
                expiryIndexes.add(index);
            }
            this.probeIndex = probeIndex;
            this.initialHeadIndex = initialHeadIndex;
        }

        private static ProjectedQueue create(
                List<GroupPlanner.Item> eligibleActive,
                GroupPlanner.Item probe,
                GroupPlanner.Item initialActiveHead,
                Comparator<GroupPlanner.Item> order) {
            GroupPlanner.Item[] itemsInQueueOrder = new GroupPlanner.Item[eligibleActive.size() + 1];
            int probeIndex = -1;
            int initialHeadIndex = -1;
            int index = 0;
            for (GroupPlanner.Item item : eligibleActive) {
                if (probeIndex < 0 && order.compare(probe, item) < 0) {
                    probeIndex = index;
                    itemsInQueueOrder[index++] = probe;
                }
                if (item == initialActiveHead) {
                    initialHeadIndex = index;
                }
                itemsInQueueOrder[index++] = item;
            }
            if (probeIndex < 0) {
                probeIndex = index;
                itemsInQueueOrder[index] = probe;
            }
            return new ProjectedQueue(itemsInQueueOrder, probeIndex, initialHeadIndex);
        }

        private boolean isEmpty() {
            return headIndex == itemsInQueueOrder.length;
        }

        private RouteProjection.Candidate.InitialHeadDisposition
                initialHeadDisposition() {
            if (initialHeadIndex < 0) {
                return RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
            }
            return initialHeadIndex < probeIndex
                    ? RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE
                    : RouteProjection.Candidate.InitialHeadDisposition.AFTER_PROBE;
        }

        private GroupPlanner.Item head() {
            if (isEmpty()) {
                throw new IllegalStateException("projected queue is empty");
            }
            return itemsInQueueOrder[headIndex];
        }

        @Override
        public Iterator<GroupPlanner.Item> iterator() {
            return new Iterator<>() {
                private int index = headIndex;

                @Override
                public boolean hasNext() {
                    while (index < itemsInQueueOrder.length && itemsInQueueOrder[index] == null) {
                        index++;
                    }
                    return index < itemsInQueueOrder.length;
                }

                @Override
                public GroupPlanner.Item next() {
                    if (!hasNext()) {
                        throw new NoSuchElementException();
                    }
                    return itemsInQueueOrder[index++];
                }
            };
        }

        private void removePlannedPrefix(int count) {
            if (count <= 0) {
                throw new IllegalArgumentException(
                        "planned prefix must contain at least one item");
            }
            for (int removed = 0; removed < count; removed++) {
                if (isEmpty()) {
                    throw new IllegalStateException(
                            "planner selected beyond the projected queue prefix");
                }
                // Keep consumed items readable while their indexes remain in the expiry heap.
                headIndex++;
                while (headIndex < itemsInQueueOrder.length && itemsInQueueOrder[headIndex] == null) {
                    headIndex++;
                }
            }
        }

        private ExpirationPrune pruneExpired(long nowMs) {
            boolean probeExpired = false;
            boolean initialHeadExpired = false;
            while (!expiryIndexes.isEmpty()) {
                int expiredIndex = expiryIndexes.peek();
                if (!expiredAt(itemsInQueueOrder[expiredIndex], nowMs)) {
                    break;
                }
                // Remove from the heap before clearing a slot used by its comparator.
                expiryIndexes.remove();
                if (expiredIndex < headIndex) {
                    continue;
                }
                itemsInQueueOrder[expiredIndex] = null;
                if (expiredIndex == probeIndex) {
                    probeExpired = true;
                }
                if (expiredIndex == initialHeadIndex) {
                    initialHeadExpired = true;
                }
            }
            while (headIndex < itemsInQueueOrder.length && itemsInQueueOrder[headIndex] == null) {
                headIndex++;
            }
            return new ExpirationPrune(probeExpired, initialHeadExpired);
        }
    }

    private static long elapsedFromNow(long nowMs, long deadlineMs) {
        return deadlineMs <= nowMs ? 0L : deadlineMs - nowMs;
    }

    private RouteProjection.CandidateView unavailable(String detail) {
        return candidate(
                RouteProjection.Candidate.State.UNAVAILABLE,
                RouteProjection.Candidate.UNKNOWN,
                0L,
                RouteProjection.Candidate.InitialHeadDisposition.NONE, detail);
    }

    private RouteProjection.CandidateView unmodeledEngineWork(
            long incomingPrefillMs) {
        return candidate(
                RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                RouteProjection.Candidate.UNKNOWN,
                incomingPrefillMs,
                RouteProjection.Candidate.InitialHeadDisposition.NONE,
                "ENGINE_WORK_UNOBSERVABLE");
    }

    private RouteProjection.CandidateView blocked(
            long incomingPrefillMs,
            RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition,
            String detail) {
        return candidate(
                RouteProjection.Candidate.State.BLOCKED,
                RouteProjection.Candidate.UNKNOWN,
                incomingPrefillMs,
                initialHeadDisposition, detail);
    }

    private RouteProjection.CandidateView candidate(
            RouteProjection.Candidate.State state,
            long projectedTtftMs,
            long incomingPrefillMs,
            RouteProjection.Candidate.InitialHeadDisposition headDisposition,
            String detail) {
        result.reset(
                state, projectedTtftMs, incomingPrefillMs,
                headDisposition, detail, null,
                hitCache, routingCacheMatchTokens);
        return result;
    }

    /** Reused only by the owning projection thread. */
    private static final class ProjectedCandidate
            implements RouteProjection.CandidateView {
        private RouteProjection.Candidate.State state;
        private long projectedTtftMsValue;
        private long incomingPrefillMs;
        private RouteProjection.Candidate.InitialHeadDisposition headDisposition;
        private String detail;
        private org.flexlb.dao.route.RoleType blockerRole;
        private long cacheHitTokens;
        private long routingCacheMatchTokens;

        private void reset(
                RouteProjection.Candidate.State exactState,
                long exactProjectedTtftMs,
                long exactIncomingPrefillMs,
                RouteProjection.Candidate.InitialHeadDisposition exactHeadDisposition,
                String exactDetail,
                org.flexlb.dao.route.RoleType exactBlockerRole,
                long exactCacheHitTokens,
                long exactRoutingCacheMatchTokens) {
            state = exactState;
            projectedTtftMsValue = exactProjectedTtftMs;
            incomingPrefillMs = exactIncomingPrefillMs;
            headDisposition = exactHeadDisposition;
            detail = exactDetail;
            blockerRole = exactBlockerRole;
            cacheHitTokens = exactCacheHitTokens;
            routingCacheMatchTokens = exactRoutingCacheMatchTokens;
        }

        @Override
        public RouteProjection.Candidate.State state() {
            return state;
        }

        @Override
        public long projectedTtftMsValue() {
            return projectedTtftMsValue;
        }

        @Override
        public long incomingPrefillMs() {
            return incomingPrefillMs;
        }

        @Override
        public RouteProjection.Candidate.InitialHeadDisposition
                initialHeadDisposition() {
            return headDisposition;
        }

        @Override
        public String detail() {
            return detail;
        }

        @Override
        public org.flexlb.dao.route.RoleType blockerRole() {
            return blockerRole;
        }

        @Override
        public long cacheHitTokens() {
            return cacheHitTokens;
        }

        @Override
        public long routingCacheMatchTokens() {
            return routingCacheMatchTokens;
        }

    }

    private static long saturatedAdd(long left, long right) {
        return right > 0L && left > Long.MAX_VALUE - right
                ? Long.MAX_VALUE : left + right;
    }

    /**
     * The sole boundary between projection math and an external predictor.
     * Predictor failures remain distinguishable from invalid returned values,
     * while every numeric output is validated before it reaches scheduling.
     */
    private static final class PredictionBoundary
            implements RouteProjection.Predictions {

        private PrefillTimePredictor.Evaluator evaluator;
        private Object cachedSingleSnapshot;
        private long cachedSeqLen = -1L;
        private long cachedHitCache = -1L;
        private long cachedSingleMs;
        private Object cachedBatchSnapshot;
        private long cachedBatchSeqLen = -1L;
        private long cachedBatchHitCache = -1L;
        private double cachedBatchMs;
        private PrefillBatchFeatures cachedSingletonBatch;

        private PredictionBoundary() {
        }

        private void reset(PrefillTimePredictor.Evaluator evaluator) {
            this.evaluator = evaluator;
            // Prediction caches are keyed by immutable model snapshot and
            // request shape, so they remain valid across equal-model endpoint
            // probes in one full-fleet selection (and across later calls).
        }

        private long singleMs(long seqLen, long hitCache) {
            Object snapshot = evaluator.snapshotIdentity();
            if (cachedSingleSnapshot == snapshot
                    && cachedSeqLen == seqLen
                    && cachedHitCache == hitCache) {
                return cachedSingleMs;
            }
            try {
                long predicted = PrefillPredictionBoundary.predictSingleRequestMs(
                        evaluator, seqLen, hitCache);
                cachedSingleSnapshot = snapshot;
                cachedSeqLen = seqLen;
                cachedHitCache = hitCache;
                cachedSingleMs = predicted;
                return predicted;
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            } catch (RuntimeException predictionFailure) {
                throw PredictionFailure.execution(predictionFailure);
            }
        }

        @Override
        public long itemDurationMs(GroupPlanner.Item item) {
            return singleMs(item.seqLen(), item.hitCache());
        }

        @Override
        public long itemDurationMs(long seqLen, long hitCache) {
            return singleMs(seqLen, hitCache);
        }

        @Override
        public double batchPlanningDurationMs(
                List<GroupPlanner.Item> items) {
            try {
                return PrefillPredictionBoundary.predictDecisionGroupMs(
                        evaluator, batchFeatures(items));
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            } catch (RuntimeException predictionFailure) {
                throw PredictionFailure.execution(predictionFailure);
            }
        }

        @Override
        public double singletonBatchPlanningDurationMs(
                long seqLen, long hitCache) {
            Object snapshot = evaluator.snapshotIdentity();
            if (cachedBatchSnapshot == snapshot
                    && cachedBatchSeqLen == seqLen
                    && cachedBatchHitCache == hitCache) {
                return cachedBatchMs;
            }
            try {
                double predicted = PrefillPredictionBoundary.predictDecisionGroupMs(
                        evaluator, singletonBatch(seqLen, hitCache));
                cachedBatchSnapshot = snapshot;
                cachedBatchSeqLen = seqLen;
                cachedBatchHitCache = hitCache;
                cachedBatchMs = predicted;
                return predicted;
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            } catch (RuntimeException predictionFailure) {
                throw PredictionFailure.execution(predictionFailure);
            }
        }

        @Override
        public long batchDurationMs(
                List<GroupPlanner.Item> items) {
            try {
                return PrefillPredictionBoundary.predictCommittedBatchMs(
                        evaluator, batchFeatures(items));
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            } catch (RuntimeException predictionFailure) {
                throw PredictionFailure.execution(predictionFailure);
            }
        }

        @Override
        public long singletonBatchDurationMs(
                long seqLen, long hitCache) {
            return committedGroupDurationMs(
                    singletonBatchPlanningDurationMs(seqLen, hitCache));
        }

        private PrefillBatchFeatures singletonBatch(
                long seqLen, long hitCache) {
            PrefillBatchFeatures features = cachedSingletonBatch;
            if (features == null
                    || features.items().getFirst().seqLen() != seqLen
                    || features.items().getFirst().hitCache() != hitCache) {
                features = new PrefillBatchFeatures(List.of(
                        new PrefillBatchFeatures.Item(seqLen, hitCache)));
                cachedSingletonBatch = features;
            }
            return features;
        }

        @Override
        public long committedGroupDurationMs(double predictedMs) {
            try {
                return PrefillPredictionBoundary.committedDecisionGroupMs(predictedMs);
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            }
        }

        private static PrefillBatchFeatures batchFeatures(
                List<GroupPlanner.Item> items) {
            return PrefillBatchFeatures.from(
                    items,
                    GroupPlanner.Item::seqLen,
                    GroupPlanner.Item::hitCache);
        }
    }

    /** Predictor failures preserve the distinction between execution and invalid values. */
    private static final class PredictionFailure extends RuntimeException {
        private final boolean invalidValue;

        private PredictionFailure(
                RuntimeException cause,
                boolean invalidValue) {
            super(cause);
            this.invalidValue = invalidValue;
        }

        private static PredictionFailure execution(RuntimeException cause) {
            return new PredictionFailure(cause, false);
        }

        private static PredictionFailure invalid(RuntimeException cause) {
            return new PredictionFailure(cause, true);
        }

        private String detail(String executionDetail) {
            return invalidValue ? INVALID_PREDICTION_DETAIL : executionDetail;
        }
    }
}
