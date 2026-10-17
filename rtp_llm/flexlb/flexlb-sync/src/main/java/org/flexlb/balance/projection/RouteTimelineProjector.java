package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.planner.GroupingPolicy;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;

import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/** Thread-confined frozen-snapshot TTFT projector and invocation-scoped result view. */
public final class RouteTimelineProjector implements RouteProjection.CandidateView {

    private static final ThreadLocal<RouteTimelineProjector> PROJECTORS =
            ThreadLocal.withInitial(RouteTimelineProjector::new);

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
    private RouteProjection.Candidate.State state;
    private long projectedTtftMsValue;
    private long incomingPrefillMs;
    private RouteProjection.Candidate.InitialHeadDisposition headDisposition;
    private String detail;

    private RouteTimelineProjector() {
    }

    /** Reusable projector owned by the current planner thread. */
    public static RouteTimelineProjector current() {
        return PROJECTORS.get();
    }

    /** Borrowed result: read or copy it before this thread's next projectView call. */
    public RouteProjection.CandidateView projectView(
            RouteProjection.Inputs inputs,
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            PrefillTimePredictor.Evaluator evaluator,
            RouteProjection.DeliveryProjection deliveryProjection,
            long planningAtMs) {
        this.requestId = requestId;
        this.priority = priority;
        this.enqueuedAtMs = enqueuedAtMs;
        this.expiresAtMs = expiresAtMs;
        this.seqLen = seqLen;
        this.hitCache = hitCache;
        this.routingCacheMatchTokens = routingCacheMatchTokens;
        return RouteProjection.applyAdmissionPolicy(inputs.queue(), project(
                inputs.queue(), inputs.work(), evaluator, deliveryProjection, planningAtMs));
    }

    /**
     * Project one endpoint under a frozen serial-work model.
     *
     * <p>Already committed work forms the engine cursor. Collection deadlines
     * and that cursor overlap via {@code max(cursor, readyAt)}. The snapshot's
     * frozen delivery projection defines each group's completion shape.
     */
    private RouteProjection.CandidateView project(
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

        predictions.evaluator = evaluator;
        try {
            return projectWithPredictions(
                    queue, committed, deliveryProjection,
                    projectionAtMs, predictions);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("PREDICTION_FAILED"));
        } finally {
            predictions.evaluator = null;
        }
    }

    private RouteProjection.CandidateView projectWithPredictions(
            QueueSnapshot queue,
            WorkSnapshot committed,
            RouteProjection.DeliveryProjection deliveryProjection,
            long projectionAtMs,
            PredictionBoundary predictions) {
        if (committed.containsRequest(requestId)) {
            return unavailable("INCOMING_ALREADY_COMMITTED");
        }

        final long incomingPrefillMs;
        try {
            incomingPrefillMs = predictions.itemDurationMs(
                    seqLen, hitCache);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(
                    predictionFailure.detail("SINGLE_PREDICTION_FAILED"));
        }

        if (committed.hasUnknownWork()) {
            if (containsActiveRequest(queue, requestId)) {
                return unavailable("INCOMING_ALREADY_ACTIVE");
            }
            if (queue.queueScheduling() && seqLen > queue.constraints().batchKvCapacity()) {
                return blocked(incomingPrefillMs, RouteProjection.Candidate.InitialHeadDisposition.NONE,
                        "PREFILL_KV_CAPACITY");
            }
            return unmodeledEngineWork(incomingPrefillMs);
        }

        long committedMs = committed.knownRemainingWorkMsAt(projectionAtMs);

        if (!queue.queueScheduling()) {
            long completionMs = saturatedAdd(committedMs, incomingPrefillMs);
            return candidate(
                    RouteProjection.Candidate.State.MODELED,
                    completionMs,
                    incomingPrefillMs,
                    RouteProjection.Candidate.InitialHeadDisposition.NONE,
                    "SERIAL_FROZEN_DIRECT");
        }
        // An empty queue has no membership to merge or decision group to build.
        // Invalid SINGLE constraints still go through the grouping policy's validation.
        if (queue.activeItems().isEmpty() && queue.constraints().predictedExecutionBudgetMs() <= 0L
                && (queue.grouping() == GroupingPolicy.FIXED_WINDOW
                    || queue.constraints().maxRequests() == 1 && queue.constraints().collectionWindowMs() == 0L
                        && queue.constraints().predictedExecutionBudgetMs() == 0L)) {
            return projectSingleton(queue.constraints(), deliveryProjection, projectionAtMs, committedMs, incomingPrefillMs);
        }
        return projectQueue(queue, deliveryProjection, projectionAtMs, committedMs, incomingPrefillMs);
    }

    private RouteProjection.CandidateView projectSingleton(
            GroupPlanner.Constraints constraints, RouteProjection.DeliveryProjection deliveryProjection,
            long projectionAtMs, long committedMs, long incomingPrefillMs) {
        if (seqLen > constraints.batchKvCapacity()) {
            return blocked(incomingPrefillMs, RouteProjection.Candidate.InitialHeadDisposition.NONE,
                    "PREFILL_KV_CAPACITY");
        }
        long readyAtMs = projectionAtMs;
        if (constraints.maxRequests() > 1 && !GroupPlanner.windowElapsed(enqueuedAtMs, projectionAtMs,
                constraints.collectionWindowMs())) {
            readyAtMs = GroupPlanner.collectionDeadlineMs(enqueuedAtMs, constraints.collectionWindowMs());
            checkArgument(readyAtMs >= 0L, "collection deadline must be non-negative");
            if (readyAtMs >= expiresAtMs) { return unavailable("INCOMING_EXPIRED_BEFORE_DISPATCH"); }
        }
        try {
            long durationMs = deliveryProjection.singletonCompletionOffsetMs(seqLen, hitCache, predictions);
            return candidate(RouteProjection.Candidate.State.MODELED,
                    saturatedAdd(Math.max(committedMs, elapsedFromNow(projectionAtMs, readyAtMs)), durationMs),
                    incomingPrefillMs, RouteProjection.Candidate.InitialHeadDisposition.NONE,
                    "EMPTY_ACTIVE_QUEUE_SINGLETON");
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
        }
    }

    private RouteProjection.CandidateView projectQueue(
            QueueSnapshot queue, RouteProjection.DeliveryProjection deliveryProjection,
            long projectionAtMs, long committedMs, long incomingPrefillMs) {
        GroupPlanner.Item probe = new GroupPlanner.Item(
                requestId, priority, Long.MAX_VALUE, enqueuedAtMs,
                expiresAtMs, seqLen, hitCache);
        ProjectedQueue ordered = ProjectedQueue.create(
                queue.activeItems(), probe, queue.ordering(), projectionAtMs);
        if (ordered == null) {
            return unavailable("INCOMING_ALREADY_ACTIVE");
        }
        boolean singleton = queue.activeItems().isEmpty();
        long cursorMs = committedMs;
        long decisionNowMs = projectionAtMs;

        while (true) {
            if (ordered.pruneExpired(decisionNowMs)) {
                return unavailable("INCOMING_EXPIRED_BEFORE_DISPATCH");
            }
            RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition =
                    ordered.initialHeadDisposition();

            GroupPlanner.Item head = ordered.head();
            if (head.seqLen() > queue.constraints().batchKvCapacity()) {
                return blocked(
                        incomingPrefillMs,
                        initialHeadDisposition,
                        "PREFILL_KV_CAPACITY");
            }

            final GroupPlanner.Selection<GroupPlanner.Item> selection;
            final RouteProjection.GroupPlanning planning;
            try {
                // An idle singleton needs no readiness prediction when already full or due.
                boolean needsPrediction = queue.constraints().predictedExecutionBudgetMs() > 0L
                        && (!singleton || queue.constraints().maxRequests() > 1
                            && !GroupPlanner.windowElapsed(enqueuedAtMs, decisionNowMs,
                                    queue.constraints().collectionWindowMs()));
                planning = needsPrediction ? deliveryProjection.planning(predictions) : null;
                selection = queue.grouping().select(ordered, queue.constraints(),
                        planning == null ? null : (added, items) -> planning.durationMs(
                                items, Math.min(ordered.probePosition, items.size() - 1)));
            } catch (PredictionFailure predictionFailure) {
                return unavailable(
                        predictionFailure.detail("BATCH_PREDICTION_FAILED"));
            }
            int probePosition = ordered.probePosition;
            checkState(!selection.items().isEmpty(), "non-empty projected queue produced an empty group");

            if (queue.grouping().dispatchReason(selection, queue.constraints(), decisionNowMs) == null) {
                decisionNowMs = Math.min(
                        GroupPlanner.collectionDeadlineMs(selection.windowOpenedAtMs(),
                                queue.constraints().collectionWindowMs()), head.expiresAtMs());
                // Only visited members (including a rejected tail) can change this frozen prefix.
                // Consumed prefixes and unvisited suffixes cannot change the pending group.
                if (decisionNowMs >= ordered.earliestExpiryMs) {
                    continue;
                }
            }

            long readyInMs = elapsedFromNow(projectionAtMs, decisionNowMs);
            long startMs = Math.max(cursorMs, readyInMs);

            try {
                long durationMs = singleton && planning == null
                        ? deliveryProjection.singletonCompletionOffsetMs(seqLen, hitCache, predictions)
                        : deliveryProjection.completionOffsetMs(selection.items(),
                                Math.min(probePosition, selection.items().size() - 1), predictions, planning);
                long completionMs = saturatedAdd(startMs, durationMs);
                if (probePosition < selection.items().size()) {
                    return candidate(
                            RouteProjection.Candidate.State.MODELED,
                            completionMs,
                            incomingPrefillMs,
                            initialHeadDisposition,
                            singleton ? "EMPTY_ACTIVE_QUEUE_SINGLETON" : "SERIAL_FROZEN_QUEUE");
                }
                cursorMs = completionMs;
            } catch (PredictionFailure predictionFailure) {
                return unavailable(
                        predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
            }
            ordered.removePlannedPrefix(selection.items().size());
        }
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

    private static boolean expiredAt(
            GroupPlanner.Item item, long nowMs) {
        return item.expiresAtMs() <= 0L || nowMs >= item.expiresAtMs();
    }

    /**
     * Merge a probe into the immutable snapshot without copying or mutating it.
     * Only groups before the probe are consumed; projection returns when a group includes it.
     * Expired members are skipped by the planner's iterator at the current decision time.
     */
    private static final class ProjectedQueue implements Iterable<GroupPlanner.Item>, Iterator<GroupPlanner.Item> {
        private final List<GroupPlanner.Item> active;
        private final GroupPlanner.Item probe;
        private final int probeIndex;
        private long earliestExpiryMs;
        private long nowMs;
        private int headIndex;
        private boolean initialHeadPruned;
        // Set by the selection iterator when it reaches the probe, including a rejected tail.
        private int probePosition;
        private int index;
        private int visited;

        private ProjectedQueue(List<GroupPlanner.Item> active, GroupPlanner.Item probe,
                               int probeIndex, long nowMs) {
            this.active = active;
            this.probe = probe;
            this.probeIndex = probeIndex;
            this.nowMs = nowMs;
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
                if (order.compare(probe, active.get(middle)) < 0) { high = middle; }
                else { low = middle + 1; }
            }
            return new ProjectedQueue(active, probe, low, nowMs);
        }

        private RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition() {
            if (active.isEmpty()) {
                return RouteProjection.Candidate.InitialHeadDisposition.NONE;
            }
            if (initialHeadPruned) {
                return RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
            }
            return probeIndex == 0
                    ? RouteProjection.Candidate.InitialHeadDisposition.AFTER_PROBE
                    : RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE;
        }

        private GroupPlanner.Item head() {
            return itemAt(headIndex);
        }

        private GroupPlanner.Item itemAt(int position) {
            return position == probeIndex ? probe
                    : active.get(position < probeIndex ? position : position - 1);
        }

        /** The planner consumes one cursor at a time; no iterator escapes this projection. */
        @Override
        public Iterator<GroupPlanner.Item> iterator() {
            probePosition = Integer.MAX_VALUE;
            earliestExpiryMs = probe.expiresAtMs();
            index = headIndex;
            visited = 0;
            return this;
        }

        @Override
        public boolean hasNext() {
            while (index <= active.size() && index != probeIndex
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
            // Include a tentative tail rejected by the planner: its expiry can unblock later members.
            earliestExpiryMs = Math.min(earliestExpiryMs, item.expiresAtMs());
            return item;
        }

        private void removePlannedPrefix(int count) {
            checkState(count <= probePosition, "cannot consume the projected probe");
            // Every member occupies one virtual position, including a rejected probe.
            headIndex = visited > count ? index - 1 : index;
        }

        private boolean pruneExpired(long nowMs) {
            this.nowMs = nowMs;
            // A consumed head remains BEFORE_PROBE even if its deadline later elapses.
            if (headIndex == 0 && !active.isEmpty() && expiredAt(active.getFirst(), nowMs)) {
                initialHeadPruned = true;
            }
            skipExpiredHead();
            return expiredAt(probe, nowMs);
        }

        private void skipExpiredHead() {
            while (headIndex < probeIndex && expiredAt(active.get(headIndex), nowMs)) {
                headIndex++;
            }
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
        this.state = state;
        this.projectedTtftMsValue = projectedTtftMs;
        this.incomingPrefillMs = incomingPrefillMs;
        this.headDisposition = headDisposition;
        this.detail = detail;
        return this;
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
    public RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition() {
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
    private static final class PredictionBoundary
            implements RouteProjection.Predictions {

        private PrefillTimePredictor.Evaluator evaluator;
        // Equal immutable models and request shapes share both singleton predictions across calls.
        private Object cachedSnapshot;
        private long cachedSeqLen = -1L;
        private long cachedHitCache = -1L;
        private long cachedSingleMs = -1L;
        private long cachedBatchMs = -1L;

        private long singletonMs(long seqLen, long hitCache, boolean batch) {
            Object snapshot = evaluator.snapshotIdentity();
            boolean sameInput = cachedSnapshot == snapshot && cachedSeqLen == seqLen && cachedHitCache == hitCache;
            long cached = batch ? cachedBatchMs : cachedSingleMs;
            if (sameInput && cached >= 0L) { return cached; }
            try {
                long predicted = batch
                        ? PrefillPredictionBoundary.predictCommittedBatchMs(evaluator,
                                new PrefillBatchFeatures(List.of(new PrefillBatchFeatures.Item(seqLen, hitCache))))
                        : PrefillPredictionBoundary.predictSingleRequestMs(evaluator, seqLen, hitCache);
                if (!sameInput) {
                    cachedSnapshot = snapshot;
                    cachedSeqLen = seqLen;
                    cachedHitCache = hitCache;
                    cachedSingleMs = cachedBatchMs = -1L;
                }
                if (batch) { cachedBatchMs = predicted; }
                else { cachedSingleMs = predicted; }
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
                    return PrefillPredictionBoundary.predictCommittedBatchMs(evaluator,
                            new PrefillBatchFeatures(List.of(items.get(0).features(), items.get(1).features())));
                }
                var features = new PrefillBatchFeatures.Item[items.size()];
                for (int i = 0; i < features.length; i++) { features[i] = items.get(i).features(); }
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
                    return PrefillPredictionBoundary.requireValidDecisionGroupMs(batch.append(seqLen, hitCache));
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
                    ? INVALID_PREDICTION_DETAIL : executionDetail;
        }
    }
}
