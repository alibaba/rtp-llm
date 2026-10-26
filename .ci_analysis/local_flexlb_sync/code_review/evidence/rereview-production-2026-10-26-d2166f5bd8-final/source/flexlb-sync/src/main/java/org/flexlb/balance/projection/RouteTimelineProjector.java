package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.planner.GroupingPolicy;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation.Candidate;

import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/** Frozen-work cost through the incoming request; collection waits are not predicted. */
public final class RouteTimelineProjector implements RequestTimeEstimation.CandidateView {

    private static final ThreadLocal<RouteTimelineProjector> PROJECTORS =
            ThreadLocal.withInitial(RouteTimelineProjector::new);

    private static final String INVALID_PREDICTION_DETAIL = "PREDICTOR_RETURNED_INVALID_VALUE";
    private final PredictionBoundary predictions = new PredictionBoundary();
    private long requestId;
    private int priority;
    private long enqueuedAtMs;
    private long expiresAtMs;
    private long seqLen;
    private long hitCache;
    private long routingCacheMatchTokens;
    private Candidate.State state;
    private long projectedTtftMsValue;
    private long incomingPrefillMs;
    private Candidate.InitialHeadDisposition headDisposition;
    private String detail;
    private GroupPlanner.Constraints rankingConstraints;

    private RouteTimelineProjector() {}

    /** Reusable projector owned by the current planner thread. */
    public static RouteTimelineProjector current() {
        return PROJECTORS.get();
    }

    /** Borrowed result: read or copy it before this thread's next projectView call. */
    public RequestTimeEstimation.CandidateView projectView(
            RequestTimeEstimation.Inputs inputs,
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            PrefillTimePredictor.Evaluator evaluator,
            RequestTimeEstimation.DeliveryProjection deliveryProjection,
            long planningAtMs) {
        return projectView(
                inputs,
                requestId,
                priority,
                enqueuedAtMs,
                expiresAtMs,
                seqLen,
                hitCache,
                routingCacheMatchTokens,
                evaluator,
                deliveryProjection,
                planningAtMs,
                false);
    }

    /** Rank known serial work without converting a transient KV wait into a cache miss. */
    public RequestTimeEstimation.CandidateView projectRankingView(
            RequestTimeEstimation.Inputs inputs,
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            PrefillTimePredictor.Evaluator evaluator,
            RequestTimeEstimation.DeliveryProjection deliveryProjection,
            long planningAtMs) {
        return projectView(
                inputs,
                requestId,
                priority,
                enqueuedAtMs,
                expiresAtMs,
                seqLen,
                hitCache,
                routingCacheMatchTokens,
                evaluator,
                deliveryProjection,
                planningAtMs,
                true);
    }

    private RequestTimeEstimation.CandidateView projectView(
            RequestTimeEstimation.Inputs inputs,
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            PrefillTimePredictor.Evaluator evaluator,
            RequestTimeEstimation.DeliveryProjection deliveryProjection,
            long planningAtMs,
            boolean rankOnly) {
        this.requestId = requestId;
        this.priority = priority;
        this.enqueuedAtMs = enqueuedAtMs;
        this.expiresAtMs = expiresAtMs;
        this.seqLen = seqLen;
        this.hitCache = hitCache;
        this.routingCacheMatchTokens = routingCacheMatchTokens;
        QueueSnapshot queue = inputs.queue();
        var projected =
                project(
                        queue,
                        inputs.work(),
                        evaluator,
                        deliveryProjection,
                        planningAtMs,
                        rankOnly);
        return rankOnly ? projected : RequestTimeEstimation.applyAdmissionPolicy(queue, projected);
    }

    /**
     * Project one endpoint under a frozen serial-work model.
     *
     * <p>Already committed work forms the engine cursor. Ordered queued groups add only their
     * execution cost. Future arrivals and collection deadlines do not advance the projection clock.
     */
    private RequestTimeEstimation.CandidateView project(
            QueueSnapshot queue,
            WorkSnapshot committed,
            PrefillTimePredictor.Evaluator evaluator,
            RequestTimeEstimation.DeliveryProjection deliveryProjection,
            long planningAtMs,
            boolean rankOnly) {
        long projectionAtMs = Math.max(queue.capturedAtMs(), planningAtMs);
        if (evaluator == null) {
            return unavailable("PREDICTOR_MISSING");
        }
        if (expiresAtMs <= 0L || projectionAtMs >= expiresAtMs) {
            return unavailable("INCOMING_EXPIRED");
        }

        predictions.evaluator = evaluator;
        try {
            return projectWithPredictions(
                    queue, committed, deliveryProjection, projectionAtMs, predictions, rankOnly);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("PREDICTION_FAILED"));
        } finally {
            predictions.evaluator = null;
        }
    }

    private RequestTimeEstimation.CandidateView projectWithPredictions(
            QueueSnapshot queue,
            WorkSnapshot committed,
            RequestTimeEstimation.DeliveryProjection deliveryProjection,
            long projectionAtMs,
            PredictionBoundary predictions,
            boolean rankOnly) {
        var constraints = rankOnly ? rankingConstraints(queue.constraints()) : queue.constraints();
        if (committed.containsRequest(requestId)) {
            return unavailable("INCOMING_ALREADY_COMMITTED");
        }

        final long incomingPrefillMs;
        try {
            incomingPrefillMs = predictions.itemDurationMs(seqLen, hitCache);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("SINGLE_PREDICTION_FAILED"));
        }

        if (queue.queueScheduling() && seqLen > constraints.batchKvCapacity()) {
            return blocked(
                    incomingPrefillMs,
                    Candidate.InitialHeadDisposition.NONE,
                    "PREFILL_KV_CAPACITY");
        }

        if (!rankOnly && committed.hasUnknownWork()) {
            if (containsActiveRequest(queue, requestId)) {
                return unavailable("INCOMING_ALREADY_ACTIVE");
            }
            return unmodeledEngineWork(incomingPrefillMs);
        }

        long committedMs = committed.knownRemainingWorkMsAt(projectionAtMs);

        // An empty queue has no membership to merge or decision group to build.
        // Invalid SINGLE constraints still go through the grouping policy's validation.
        if (!queue.queueScheduling()
                || queue.activeItems().isEmpty()
                        && (queue.grouping() == GroupingPolicy.FIXED_WINDOW
                                || constraints.maxRequests() == 1
                                        && constraints.collectionWindowMs() == 0L
                                        && constraints.predictedExecutionBudgetMs() == 0L)) {
            return projectSingleton(
                    deliveryProjection, committedMs, incomingPrefillMs, queue.queueScheduling());
        }
        return projectQueue(
                queue,
                constraints,
                deliveryProjection,
                projectionAtMs,
                committedMs,
                incomingPrefillMs);
    }

    private GroupPlanner.Constraints rankingConstraints(GroupPlanner.Constraints source) {
        var cached = rankingConstraints;
        if (cached == null
                || cached.maxRequests() != source.maxRequests()
                || cached.batchTokenCapacity() != source.batchTokenCapacity()
                || cached.predictedExecutionBudgetMs() != source.predictedExecutionBudgetMs()) {
            // Fleet nodes usually share these immutable policy values. Do not allocate
            // a second queue snapshot and constraint record for every worker/request.
            cached =
                    new GroupPlanner.Constraints(
                            source.maxRequests(),
                            source.batchTokenCapacity(),
                            Long.MAX_VALUE,
                            source.predictedExecutionBudgetMs(),
                            0L);
            rankingConstraints = cached;
        }
        return cached;
    }

    private RequestTimeEstimation.CandidateView projectSingleton(
            RequestTimeEstimation.DeliveryProjection deliveryProjection,
            long committedMs,
            long incomingPrefillMs,
            boolean queueScheduling) {
        try {
            long durationMs =
                    deliveryProjection.singletonCompletionOffsetMs(seqLen, hitCache, predictions);
            return candidate(
                    Candidate.State.MODELED,
                    saturatedAdd(committedMs, durationMs),
                    incomingPrefillMs,
                    Candidate.InitialHeadDisposition.NONE,
                    queueScheduling ? "EMPTY_ACTIVE_QUEUE_SINGLETON" : "SERIAL_FROZEN_DIRECT");
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
        }
    }

    private RequestTimeEstimation.CandidateView projectQueue(
            QueueSnapshot queue,
            GroupPlanner.Constraints constraints,
            RequestTimeEstimation.DeliveryProjection deliveryProjection,
            long projectionAtMs,
            long committedMs,
            long incomingPrefillMs) {
        GroupPlanner.Item probe =
                new GroupPlanner.Item(
                        requestId,
                        priority,
                        Long.MAX_VALUE,
                        enqueuedAtMs,
                        expiresAtMs,
                        seqLen,
                        hitCache);
        ProjectedQueue ordered =
                ProjectedQueue.create(queue.activeItems(), probe, queue.ordering(), projectionAtMs);
        if (ordered == null) {
            return unavailable("INCOMING_ALREADY_ACTIVE");
        }
        long cursorMs = committedMs;

        while (true) {
            ordered.skipExpiredHead();
            Candidate.InitialHeadDisposition initialHeadDisposition =
                    ordered.initialHeadDisposition();

            GroupPlanner.Item head = ordered.head();
            if (head.seqLen() > constraints.batchKvCapacity()) {
                return blocked(incomingPrefillMs, initialHeadDisposition, "PREFILL_KV_CAPACITY");
            }

            final GroupPlanner.Selection<GroupPlanner.Item> selection;
            final RequestTimeEstimation.GroupPlanning planning;
            try {
                planning =
                        constraints.predictedExecutionBudgetMs() > 0L
                                ? deliveryProjection.planning(predictions)
                                : null;
                selection =
                        queue.grouping()
                                .select(
                                        ordered,
                                        constraints,
                                        planning == null
                                                ? null
                                                : (added, items) ->
                                                        planning.durationMs(
                                                                items,
                                                                Math.min(
                                                                        ordered.probePosition,
                                                                        items.size() - 1)));
            } catch (PredictionFailure predictionFailure) {
                return unavailable(predictionFailure.detail("BATCH_PREDICTION_FAILED"));
            }
            int probePosition = ordered.probePosition;
            checkState(
                    !selection.items().isEmpty(),
                    "non-empty projected queue produced an empty group");

            try {
                long durationMs =
                        deliveryProjection.completionOffsetMs(
                                selection.items(),
                                Math.min(probePosition, selection.items().size() - 1),
                                predictions,
                                planning);
                long completionMs = saturatedAdd(cursorMs, durationMs);
                if (probePosition < selection.items().size()) {
                    return candidate(
                            Candidate.State.MODELED,
                            completionMs,
                            incomingPrefillMs,
                            initialHeadDisposition,
                            "SERIAL_FROZEN_QUEUE");
                }
                cursorMs = completionMs;
            } catch (PredictionFailure predictionFailure) {
                return unavailable(predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
            }
            ordered.removePlannedPrefix(selection.items().size());
        }
    }

    private static boolean containsActiveRequest(QueueSnapshot queue, long requestId) {
        for (GroupPlanner.Item item : queue.activeItems()) {
            if (item.requestId() == requestId) {
                return true;
            }
        }
        return false;
    }

    private static boolean expiredAt(GroupPlanner.Item item, long nowMs) {
        return item.expiresAtMs() <= 0L || nowMs >= item.expiresAtMs();
    }

    /**
     * Merge a probe into the immutable snapshot without copying or mutating it.
     * Only groups before the probe are consumed; projection returns when a group includes it.
     * Expired members are skipped by the planner's iterator at the current decision time.
     */
    private static final class ProjectedQueue
            implements Iterable<GroupPlanner.Item>, Iterator<GroupPlanner.Item> {
        private final List<GroupPlanner.Item> active;
        private final GroupPlanner.Item probe;
        private final int probeIndex;
        private final long nowMs;
        private int headIndex;
        private final boolean initialHeadPruned;
        // Set by the selection iterator when it reaches the probe, including a rejected tail.
        private int probePosition;
        private int index;
        private int visited;

        private ProjectedQueue(
                List<GroupPlanner.Item> active,
                GroupPlanner.Item probe,
                int probeIndex,
                long nowMs) {
            this.active = active;
            this.probe = probe;
            this.probeIndex = probeIndex;
            this.nowMs = nowMs;
            this.initialHeadPruned = !active.isEmpty() && expiredAt(active.getFirst(), nowMs);
        }

        private static ProjectedQueue create(
                List<GroupPlanner.Item> active,
                GroupPlanner.Item probe,
                Comparator<GroupPlanner.Item> order,
                long nowMs) {
            for (GroupPlanner.Item item : active) {
                if (item.requestId() == probe.requestId() && !expiredAt(item, nowMs)) {
                    return null;
                }
            }
            // Upper bound keeps equal scheduling keys ahead of the new probe.
            int low = 0, high = active.size();
            while (low < high) {
                int middle = (low + high) >>> 1;
                if (order.compare(probe, active.get(middle)) < 0) {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            return new ProjectedQueue(active, probe, low, nowMs);
        }

        private Candidate.InitialHeadDisposition initialHeadDisposition() {
            if (active.isEmpty()) {
                return Candidate.InitialHeadDisposition.NONE;
            }
            if (initialHeadPruned) {
                return Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
            }
            return probeIndex == 0
                    ? Candidate.InitialHeadDisposition.AFTER_PROBE
                    : Candidate.InitialHeadDisposition.BEFORE_PROBE;
        }

        private GroupPlanner.Item head() {
            return itemAt(headIndex);
        }

        private GroupPlanner.Item itemAt(int position) {
            return position == probeIndex
                    ? probe
                    : active.get(position < probeIndex ? position : position - 1);
        }

        /** The planner consumes one cursor at a time; no iterator escapes this projection. */
        @Override
        public Iterator<GroupPlanner.Item> iterator() {
            probePosition = Integer.MAX_VALUE;
            index = headIndex;
            visited = 0;
            return this;
        }

        @Override
        public boolean hasNext() {
            while (index <= active.size()
                    && index != probeIndex
                    && expiredAt(itemAt(index), nowMs)) {
                index++;
            }
            return index <= active.size();
        }

        @Override
        public GroupPlanner.Item next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            if (index == probeIndex) {
                probePosition = visited;
            }
            visited++;
            GroupPlanner.Item item = itemAt(index++);
            return item;
        }

        private void removePlannedPrefix(int count) {
            checkState(count <= probePosition, "cannot consume the projected probe");
            // Every member occupies one virtual position, including a rejected probe.
            headIndex = visited > count ? index - 1 : index;
        }

        private void skipExpiredHead() {
            while (headIndex < probeIndex && expiredAt(active.get(headIndex), nowMs)) {
                headIndex++;
            }
        }
    }

    private RequestTimeEstimation.CandidateView unavailable(String detail) {
        return candidate(
                Candidate.State.UNAVAILABLE,
                Candidate.UNKNOWN,
                0L,
                Candidate.InitialHeadDisposition.NONE,
                detail);
    }

    private RequestTimeEstimation.CandidateView unmodeledEngineWork(long incomingPrefillMs) {
        return candidate(
                Candidate.State.UNMODELED_ENGINE_WORK,
                Candidate.UNKNOWN,
                incomingPrefillMs,
                Candidate.InitialHeadDisposition.NONE,
                "ENGINE_WORK_UNOBSERVABLE");
    }

    private RequestTimeEstimation.CandidateView blocked(
            long incomingPrefillMs,
            Candidate.InitialHeadDisposition initialHeadDisposition,
            String detail) {
        return candidate(
                Candidate.State.BLOCKED,
                Candidate.UNKNOWN,
                incomingPrefillMs,
                initialHeadDisposition,
                detail);
    }

    private RequestTimeEstimation.CandidateView candidate(
            Candidate.State state,
            long projectedTtftMs,
            long incomingPrefillMs,
            Candidate.InitialHeadDisposition headDisposition,
            String detail) {
        this.state = state;
        this.projectedTtftMsValue = projectedTtftMs;
        this.incomingPrefillMs = incomingPrefillMs;
        this.headDisposition = headDisposition;
        this.detail = detail;
        return this;
    }

    @Override
    public Candidate.State state() {
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
    public Candidate.InitialHeadDisposition initialHeadDisposition() {
        return headDisposition;
    }

    @Override
    public String detail() {
        return detail;
    }

    @Override
    public org.flexlb.dao.route.RoleType blockerRole() {
        return null;
    }

    @Override
    public long cacheHitTokens() {
        return hitCache;
    }

    @Override
    public long routingCacheMatchTokens() {
        return routingCacheMatchTokens;
    }

    /**
     * The sole boundary between projection math and an external predictor.
     * Predictor failures remain distinguishable from invalid returned values,
     * while every numeric output is validated before it reaches scheduling.
     */
    private static final class PredictionBoundary implements RequestTimeEstimation.Predictions {

        private PrefillTimePredictor.Evaluator evaluator;
        // Equal immutable models and request shapes share both singleton predictions across calls.
        private Object cachedSnapshot;
        private long cachedSeqLen = -1L;
        private long cachedHitCache = -1L;
        private long cachedSingleMs = -1L;
        private long cachedBatchMs = -1L;

        private long singletonMs(long seqLen, long hitCache, boolean batch) {
            Object snapshot = evaluator.snapshotIdentity();
            boolean sameInput =
                    cachedSnapshot == snapshot
                            && cachedSeqLen == seqLen
                            && cachedHitCache == hitCache;
            long cached = batch ? cachedBatchMs : cachedSingleMs;
            if (sameInput && cached >= 0L) {
                return cached;
            }
            try {
                long predicted =
                        batch
                                ? PrefillPredictionBoundary.predictCommittedBatchMs(
                                        evaluator,
                                        new PrefillBatchFeatures(
                                                List.of(
                                                        new PrefillBatchFeatures.Item(
                                                                seqLen, hitCache))))
                                : PrefillPredictionBoundary.predictSingleRequestMs(
                                        evaluator, seqLen, hitCache);
                if (!sameInput) {
                    cachedSnapshot = snapshot;
                    cachedSeqLen = seqLen;
                    cachedHitCache = hitCache;
                    cachedSingleMs = cachedBatchMs = -1L;
                }
                if (batch) {
                    cachedBatchMs = predicted;
                } else {
                    cachedSingleMs = predicted;
                }
                return predicted;
            } catch (RuntimeException predictionFailure) {
                throw new PredictionFailure(predictionFailure);
            }
        }

        @Override
        public long itemDurationMs(GroupPlanner.Item item) {
            return singletonMs(item.seqLen(), item.hitCache(), false);
        }

        @Override
        public long itemDurationMs(long seqLen, long hitCache) {
            return singletonMs(seqLen, hitCache, false);
        }

        @Override
        public long batchDurationMs(List<GroupPlanner.Item> items) {
            if (items.size() == 1) {
                var item = items.getFirst();
                return singletonMs(item.seqLen(), item.hitCache(), true);
            }
            try {
                if (items.size() == 2) {
                    return PrefillPredictionBoundary.predictCommittedBatchMs(
                            evaluator,
                            new PrefillBatchFeatures(
                                    List.of(items.get(0).features(), items.get(1).features())));
                }
                var features = new PrefillBatchFeatures.Item[items.size()];
                for (int i = 0; i < features.length; i++) {
                    features[i] = items.get(i).features();
                }
                return PrefillPredictionBoundary.predictCommittedBatchMs(
                        evaluator, new PrefillBatchFeatures(List.of(features)));
            } catch (RuntimeException predictionFailure) {
                throw new PredictionFailure(predictionFailure);
            }
        }

        @Override
        public PrefillTimePredictor.BatchPrediction newBatchPrediction() {
            PrefillTimePredictor.BatchPrediction batch = evaluator.newBatchPrediction();
            return (seqLen, hitCache) -> {
                try {
                    return PrefillPredictionBoundary.requireValidDecisionGroupMs(
                            batch.append(seqLen, hitCache));
                } catch (RuntimeException predictionFailure) {
                    throw new PredictionFailure(predictionFailure);
                }
            };
        }

        @Override
        public long singletonBatchDurationMs(long seqLen, long hitCache) {
            return singletonMs(seqLen, hitCache, true);
        }
    }

    /** Predictor failures preserve the distinction between execution and invalid values. */
    private static final class PredictionFailure extends RuntimeException {
        private PredictionFailure(RuntimeException cause) {
            super(cause);
        }

        private String detail(String executionDetail) {
            return getCause() instanceof InvalidPrefillPredictionException
                    ? INVALID_PREDICTION_DETAIL
                    : executionDetail;
        }
    }
}
