package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.config.DispatcherConfig;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.OptionalLong;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Estimates known execution work through the incoming request from frozen queue and model inputs.
 * The selecting strategy combines this estimate with independently observed delivery capacity.
 */
public final class RequestTimeEstimation {
    private static final ThreadLocal<RequestTimeEstimation> ESTIMATORS =
            ThreadLocal.withInitial(RequestTimeEstimation::new);
    private static final String INVALID_PREDICTION_DETAIL = "PREDICTOR_RETURNED_INVALID_VALUE";

    public static final long UNKNOWN = -1L;

    private final Prediction predictions = new Prediction();
    private final ProjectedQueue ordered = new ProjectedQueue();

    private GroupPlanner.Item incoming;

    private boolean inUse;

    private String requestId;

    private int priority;
    private long enqueuedAtMs;
    private long expiresAtMs;
    private long seqLen;

    private GroupPlanner.Constraints rankingConstraints;

    private RequestTimeEstimation() {}

    /** Reusable estimator owned by the current planner thread. */
    public static RequestTimeEstimation current() {
        return ESTIMATORS.get();
    }

    /**
     * Borrowed ranking and admission result, valid until the next call on this thread.
     * Ranking cost remains available when current delivery capacity blocks admission.
     */
    public CandidateView projectCandidate(
            Inputs inputs,
            String requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            long planningAtMs) {
        // A predictor may invoke this estimator recursively; isolate its unfinished calculation.
        if (inUse) {
            return new RequestTimeEstimation()
                    .projectCandidate(
                            inputs,
                            requestId,
                            priority,
                            enqueuedAtMs,
                            expiresAtMs,
                            seqLen,
                            hitCache,
                            routingCacheMatchTokens,
                            planningAtMs);
        }
        this.requestId = requestId;
        this.priority = priority;
        this.enqueuedAtMs = enqueuedAtMs;
        this.expiresAtMs = expiresAtMs;
        this.seqLen = seqLen;
        predictions.hitCache = hitCache;
        predictions.routingCacheMatchTokens = routingCacheMatchTokens;
        inUse = true;
        try {
            return project(inputs, planningAtMs);
        } finally {
            predictions.evaluator = null;
            predictions.batchPrediction = null;
            inUse = false;
        }
    }

    /** One incremental predictor for one live queue selection. */
    public static GroupPlanner.PrefixPrediction<RouteResult> newGroupPredictor(
            DispatcherConfig.Type mode, PrefillTimePredictor.Evaluator evaluator) {
        if (mode == DispatcherConfig.Type.BATCH) {
            var prediction = evaluator.newBatchPrediction();
            return (added, items) ->
                    PrefillTimePredictor.requireValidDecisionGroupMs(
                            prediction.append(added.seqLen(), added.hitCache()));
        }
        return new GroupPlanner.PrefixPrediction<>() {
            private double totalMs;

            @Override
            public double append(RouteResult added, List<RouteResult> items) {
                try {
                    totalMs +=
                            PrefillTimePredictor.predictSingleRequestMs(
                                    evaluator, added.seqLen(), added.hitCache());
                } catch (PrefillTimePredictor.InvalidPrefillPredictionException failure) {
                    throw failure.atMember(items.size() - 1);
                }
                return PrefillTimePredictor.requireValidDecisionGroupMs(totalMs);
            }
        };
    }

    /**
     * Project one endpoint under a frozen serial-work model.
     *
     * <p>Already committed work forms the engine cursor. Ordered queued groups add only their
     * execution cost. Future arrivals and collection deadlines do not advance the projection clock.
     */
    private CandidateView project(Inputs inputs, long planningAtMs) {
        var queue = inputs.queue();
        var evaluator = inputs.evaluator();
        long projectionAtMs = Math.max(queue.capturedAtMs(), planningAtMs);
        if (evaluator == null) {
            return unavailable("PREDICTOR_MISSING");
        }
        if (expiresAtMs <= 0L || projectionAtMs >= expiresAtMs) {
            return unavailable("INCOMING_EXPIRED");
        }

        predictions.evaluator = evaluator;
        predictions.batch = inputs.deliveryMode() == DispatcherConfig.Type.BATCH;
        try {
            return projectWithPredictions(queue, inputs.work(), projectionAtMs);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("PREDICTION_FAILED"));
        }
    }

    private CandidateView unavailable(String detail) {
        return candidate(State.UNAVAILABLE, UNKNOWN, 0L, InitialHeadDisposition.NONE, detail);
    }

    private CandidateView candidate(
            State state,
            long projectedTtftMs,
            long incomingPrefillMs,
            InitialHeadDisposition headDisposition,
            String detail) {
        predictions.state = state;
        predictions.projectedTtftMsValue = projectedTtftMs;
        predictions.incomingPrefillMs = incomingPrefillMs;
        predictions.headDisposition = headDisposition;
        predictions.detail = detail;
        return predictions;
    }

    private CandidateView projectWithPredictions(
            EstimationSnapshot queue, WorkSnapshot committed, long projectionAtMs) {
        var constraints = rankingConstraints(queue.constraints());
        if (committed.containsRequest(requestId)) {
            return unavailable("INCOMING_ALREADY_COMMITTED");
        }

        final long incomingPrefillMs;
        try {
            incomingPrefillMs = predictions.singletonMs(seqLen, predictions.hitCache, false);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("SINGLE_PREDICTION_FAILED"));
        }

        if (queue.queueScheduling() && seqLen > constraints.batchKvCapacity()) {
            return blocked(incomingPrefillMs, InitialHeadDisposition.NONE, "PREFILL_KV_CAPACITY");
        }

        long committedMs = committed.knownRemainingWorkMsAt(projectionAtMs);

        // An empty queue has no membership to merge or decision group to build.
        if (!queue.queueScheduling() || queue.activeItems().isEmpty()) {
            if (queue.queueScheduling()) {
                queue.grouping().validateConstraints(constraints);
            }
            return projectSingleton(committedMs, incomingPrefillMs, queue.queueScheduling());
        }
        try {
            return projectQueue(queue, constraints, projectionAtMs, committedMs, incomingPrefillMs);
        } finally {
            ordered.active = null;
            ordered.probe = null;
        }
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

    private CandidateView blocked(
            long incomingPrefillMs, InitialHeadDisposition initialHeadDisposition, String detail) {
        return candidate(State.BLOCKED, UNKNOWN, incomingPrefillMs, initialHeadDisposition, detail);
    }

    private CandidateView projectSingleton(
            long committedMs, long incomingPrefillMs, boolean queueScheduling) {
        try {
            long durationMs =
                    predictions.singletonMs(seqLen, predictions.hitCache, predictions.batch);
            return candidate(
                    State.MODELED,
                    saturatedAdd(committedMs, durationMs),
                    incomingPrefillMs,
                    InitialHeadDisposition.NONE,
                    queueScheduling ? "EMPTY_ACTIVE_QUEUE_SINGLETON" : "SERIAL_FROZEN_DIRECT");
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
        }
    }

    private CandidateView projectQueue(
            EstimationSnapshot queue,
            GroupPlanner.Constraints constraints,
            long projectionAtMs,
            long committedMs,
            long incomingPrefillMs) {
        if (incoming == null
                || !java.util.Objects.equals(incoming.requestId(), requestId)
                || incoming.priority() != priority
                || incoming.enqueuedAtMs() != enqueuedAtMs
                || incoming.expiresAtMs() != expiresAtMs
                || incoming.seqLen() != seqLen
                || incoming.hitCache() != predictions.hitCache) {
            incoming =
                    new GroupPlanner.Item(
                            requestId,
                            priority,
                            Long.MAX_VALUE,
                            enqueuedAtMs,
                            expiresAtMs,
                            seqLen,
                            predictions.hitCache);
        }
        if (!ordered.reset(queue.activeItems(), incoming, queue.ordering(), projectionAtMs)) {
            return unavailable("INCOMING_ALREADY_ACTIVE");
        }
        long cursorMs = committedMs;

        while (true) {
            ordered.skipExpiredHead();
            InitialHeadDisposition initialHeadDisposition = ordered.initialHeadDisposition();

            GroupPlanner.Item head = ordered.head();
            if (head.seqLen() > constraints.batchKvCapacity()) {
                return blocked(incomingPrefillMs, initialHeadDisposition, "PREFILL_KV_CAPACITY");
            }

            final GroupPlanner.Selection<GroupPlanner.Item> selection;
            try {
                boolean predict = constraints.predictedExecutionBudgetMs() > 0L;
                if (predict) {
                    predictions.startGroup(requestId);
                }
                selection =
                        queue.grouping().select(ordered, constraints, predict ? predictions : null);
            } catch (PredictionFailure predictionFailure) {
                return unavailable(predictionFailure.detail("BATCH_PREDICTION_FAILED"));
            }
            int probePosition = ordered.probePosition;
            checkState(
                    !selection.items().isEmpty(),
                    "non-empty projected queue produced an empty group");

            try {
                long durationMs = predictions.completionMs(selection, probePosition);
                long completionMs = saturatedAdd(cursorMs, durationMs);
                if (probePosition < selection.items().size()) {
                    return candidate(
                            State.MODELED,
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
        private List<GroupPlanner.Item> active;
        private GroupPlanner.Item probe;

        private int probeIndex;
        private long nowMs;
        private int headIndex;
        private boolean initialHeadPruned;
        // Set by the selection iterator when it reaches the probe, including a rejected tail.
        private int probePosition;
        private int index;
        private int visited;

        /** The planner consumes one cursor at a time; no iterator escapes this projection. */
        @Override
        public Iterator<GroupPlanner.Item> iterator() {
            probePosition = Integer.MAX_VALUE;
            index = headIndex;
            visited = 0;
            return this;
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

        @Override
        public boolean hasNext() {
            while (index <= active.size()
                    && index != probeIndex
                    && expiredAt(itemAt(index), nowMs)) {
                index++;
            }
            return index <= active.size();
        }

        private boolean reset(
                List<GroupPlanner.Item> active,
                GroupPlanner.Item probe,
                Comparator<? super GroupPlanner.Item> order,
                long nowMs) {
            for (int index = 0; index < active.size(); index++) {
                var item = active.get(index);
                if (java.util.Objects.equals(item.requestId(), probe.requestId())
                        && !expiredAt(item, nowMs)) {
                    return false;
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
            this.active = active;
            this.probe = probe;
            this.probeIndex = low;
            this.nowMs = nowMs;
            this.headIndex = 0;
            this.initialHeadPruned = !active.isEmpty() && expiredAt(active.getFirst(), nowMs);
            return true;
        }

        private InitialHeadDisposition initialHeadDisposition() {
            if (active.isEmpty()) {
                return InitialHeadDisposition.NONE;
            }
            if (initialHeadPruned) {
                return InitialHeadDisposition.TERMINAL_PRUNED;
            }
            return probeIndex == 0
                    ? InitialHeadDisposition.AFTER_PROBE
                    : InitialHeadDisposition.BEFORE_PROBE;
        }

        private GroupPlanner.Item head() {
            return itemAt(headIndex);
        }

        private GroupPlanner.Item itemAt(int position) {
            return position == probeIndex
                    ? probe
                    : active.get(position < probeIndex ? position : position - 1);
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

    /**
     * Read-only projection result. Invocation-scoped views must not be retained.
     */
    public interface CandidateView {
        String detail();

        long incomingPrefillMs();

        InitialHeadDisposition initialHeadDisposition();

        long cacheHitTokens();

        long routingCacheMatchTokens();

        default boolean selectable() {
            return state() == State.MODELED && projectedTtftMsValue() != UNKNOWN;
        }

        State state();

        /** Known execution cost through this request; excludes future collection waits. */
        long projectedTtftMsValue();
    }

    public enum InitialHeadDisposition {
        NONE,
        BEFORE_PROBE,
        AFTER_PROBE,
        TERMINAL_PRUNED
    }

    public enum State {
        MODELED,
        BLOCKED,
        UNAVAILABLE
    }

    /**
     * Queue/work captured under one ownership lock, with an immutable model and source revisions.
     */
    public record Inputs(
            EstimationSnapshot queue,
            WorkSnapshot work,
            SelectionVersion selectionVersion,
            PrefillTimePredictor.Evaluator evaluator,
            DispatcherConfig.Type deliveryMode) {

        /**
         * Builds projection-only inputs without an Endpoint registration identity.
         */
        public Inputs(
                EstimationSnapshot queue,
                WorkSnapshot work,
                long ownershipVersion,
                PrefillTimePredictor.Evaluator evaluator,
                DispatcherConfig.Type deliveryMode) {
            this(
                    queue,
                    work,
                    new SelectionVersion(-1L, ownershipVersion, -1L),
                    evaluator,
                    deliveryMode);
        }

        public Inputs {
            // Committed work may reuse an older clock base while its ownership
            // is unchanged. The projector rebases running duration to queue time.
            checkArgument(
                    queue.capturedAtMs() >= work.capturedAtMs(),
                    "work snapshot cannot be newer than its queue capture");
            java.util.Objects.requireNonNull(selectionVersion, "selectionVersion");
            checkArgument(
                    selectionVersion.ownershipVersion() >= 0L,
                    "ownershipVersion must be non-negative");
        }

        public long ownershipVersion() {
            return selectionVersion.ownershipVersion();
        }
    }

    /**
     * Endpoint generation and resource/scheduling revisions captured for atomic registration.
     * Projection-only inputs use -1 for generation and scheduling revision and cannot register resources.
     */
    public record SelectionVersion(
            long generationId, long ownershipVersion, long schedulingInputVersion) {}

    /**
     * The sole boundary between projection math and an external predictor.
     * Predictor failures remain distinguishable from invalid returned values,
     * while every numeric output is validated before it reaches scheduling.
     */
    private static final class Prediction
            implements GroupPlanner.PrefixPrediction<GroupPlanner.Item>, CandidateView {
        private State state;

        private long projectedTtftMsValue;
        private long incomingPrefillMs;

        private InitialHeadDisposition headDisposition;
        private String detail;

        private long hitCache;
        private long routingCacheMatchTokens;

        private PrefillTimePredictor.Evaluator evaluator;

        private boolean batch;

        private PrefillTimePredictor.BatchPrediction batchPrediction;
        private String probeId;

        private boolean probeReached;
        private double batchMs;
        private long totalMs;
        private long previousMs;
        private int computedSize;

        // Equal immutable models and request shapes share both singleton predictions across calls.
        private Object cachedSnapshot;

        private long cachedSeqLen = -1L;
        private long cachedHitCache = -1L;
        private long cachedSingleMs = -1L;
        private long cachedBatchMs = -1L;

        @Override
        public State state() {
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
        public InitialHeadDisposition initialHeadDisposition() {
            return headDisposition;
        }

        @Override
        public String detail() {
            return detail;
        }

        @Override
        public long cacheHitTokens() {
            return hitCache;
        }

        @Override
        public long routingCacheMatchTokens() {
            return routingCacheMatchTokens;
        }

        @Override
        public double append(GroupPlanner.Item added, List<GroupPlanner.Item> prefix) {
            if (!probeReached) {
                if (batch) {
                    try {
                        batchMs =
                                PrefillTimePredictor.requireValidDecisionGroupMs(
                                        batchPrediction.append(added.seqLen(), added.hitCache()));
                    } catch (RuntimeException failure) {
                        throw new PredictionFailure(failure);
                    }
                } else {
                    long itemMs = singletonMs(added.seqLen(), added.hitCache(), false);
                    previousMs = totalMs;
                    totalMs = saturatedAdd(totalMs, itemMs);
                }
                computedSize++;
                probeReached = java.util.Objects.equals(added.requestId(), probeId);
            }
            return batch ? batchMs : totalMs;
        }

        private long completionMs(
                GroupPlanner.Selection<GroupPlanner.Item> selection, int probePosition) {
            var items = selection.items();
            int size = (int) Math.min(probePosition + 1L, items.size());
            if (selection.selectedPredictionMs().isPresent()) {
                if (batch) {
                    return PrefillTimePredictor.committedDecisionGroupMs(
                            selection.selectedPredictionMs().getAsDouble());
                }
                // Keep integer precision beyond 2^53, including the rejected-tail predecessor.
                checkState(
                        size == computedSize || size == computedSize - 1,
                        "selected prefix must belong to this group");
                return size == computedSize ? totalMs : previousMs;
            }
            if (size == 1) {
                var item = items.getFirst();
                return singletonMs(item.seqLen(), item.hitCache(), batch);
            }
            if (!batch) {
                long total = 0L;
                for (int i = 0; i < size; i++) {
                    var item = items.get(i);
                    total = saturatedAdd(total, singletonMs(item.seqLen(), item.hitCache(), false));
                }
                return total;
            }
            try {
                if (size == 2) {
                    return PrefillTimePredictor.predictCommittedBatchMs(
                            evaluator,
                            new PrefillBatchFeatures(
                                    List.of(items.get(0).features(), items.get(1).features())));
                }
                var features = new PrefillBatchFeatures.Item[size];
                for (int i = 0; i < features.length; i++) {
                    features[i] = items.get(i).features();
                }
                return PrefillTimePredictor.predictCommittedBatchMs(
                        evaluator, new PrefillBatchFeatures(List.of(features)));
            } catch (RuntimeException predictionFailure) {
                throw new PredictionFailure(predictionFailure);
            }
        }

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
                                ? PrefillTimePredictor.predictCommittedBatchMs(
                                        evaluator,
                                        new PrefillBatchFeatures(
                                                List.of(
                                                        new PrefillBatchFeatures.Item(
                                                                seqLen, hitCache))))
                                : PrefillTimePredictor.predictSingleRequestMs(
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

        private void startGroup(String probeId) {
            this.probeId = probeId;
            probeReached = false;
            batchMs = 0.0;
            totalMs = previousMs = 0L;
            computedSize = 0;
            try {
                batchPrediction = batch ? evaluator.newBatchPrediction() : null;
            } catch (RuntimeException failure) {
                throw new PredictionFailure(failure);
            }
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

    /**
     * Frozen request identities and Prefill work accounting after endpoint admission.
     * Each running request or batch consumes elapsed time independently; a batch is counted once.
     * Unknown work preserves member identities and known durations, but makes the total unavailable.
     */
    public static final class WorkSnapshot {
        private static final long[] EMPTY_LONGS = new long[0];

        private final long capturedAtMs;

        private List<RequestWork> requests;
        private List<BatchWork> batches;

        private long knownNonRunningWorkMs;

        private long[] runningWorkMs = EMPTY_LONGS;
        private String[] requestIds = new String[0];

        private boolean unknownWork;
        private volatile boolean calculated;

        /** Freeze resource facts cheaply; aggregate and sort when read outside the endpoint lock. */
        public WorkSnapshot(
                long capturedAtMs,
                List<RequestWork> requests,
                List<BatchWork> batches,
                long unknownRequestCount) {
            this.capturedAtMs = capturedAtMs;
            this.requests = List.copyOf(requests);
            this.batches = List.copyOf(batches);
            checkArgument(unknownRequestCount >= 0L, "unknownRequestCount must be non-negative");
            unknownWork = unknownRequestCount > 0L;
        }

        public long capturedAtMs() {
            return capturedAtMs;
        }

        public boolean containsRequest(String requestId) {
            calculate();
            return Arrays.binarySearch(requestIds, requestId) >= 0;
        }

        public WorkSnapshot calculate() {
            if (!calculated) {
                synchronized (this) {
                    if (!calculated) {
                        calculateWork();
                        requests = null;
                        batches = null;
                        calculated = true;
                    }
                }
            }
            return this;
        }

        /**
         * Complete committed duration, absent when any work unit lacks a duration.
         */
        public OptionalLong totalRemainingWorkMs() {
            return totalRemainingWorkMsAt(capturedAtMs);
        }

        /** Complete preceding work at a later clock; only observed running work consumes time. */
        public OptionalLong totalRemainingWorkMsAt(long observedAtMs) {
            return hasUnknownWork()
                    ? OptionalLong.empty()
                    : OptionalLong.of(knownRemainingWorkMsAt(observedAtMs));
        }

        public boolean hasUnknownWork() {
            calculate();
            return unknownWork;
        }

        /** Known work rebased to a later planning clock without copying the snapshot. */
        public long knownRemainingWorkMsAt(long planningAtMs) {
            calculate();
            long elapsedMs = planningAtMs <= capturedAtMs ? 0L : planningAtMs - capturedAtMs;
            long total = knownNonRunningWorkMs;
            for (long runningMs : runningWorkMs) {
                long remaining = elapsedMs >= runningMs ? 0L : runningMs - elapsedMs;
                total = saturatedAdd(total, remaining);
            }
            return total;
        }

        private void calculateWork() {
            int runningCount = 0;
            int requestIdCount = requests.size();
            long nonRunningMs = 0L;
            boolean hasUnknown = unknownWork;
            for (RequestWork request : requests) {
                if (request.phase() == Phase.ENGINE_RUNNING) {
                    runningCount++;
                } else {
                    nonRunningMs = saturatedAdd(nonRunningMs, request.remainingWorkMs());
                }
            }
            for (BatchWork batch : batches) {
                requestIdCount = Math.addExact(requestIdCount, batch.requestIds().size());
                if (batch.remainingWorkMs().isEmpty()) {
                    hasUnknown = true;
                } else if (batch.phase() == Phase.ENGINE_RUNNING) {
                    runningCount++;
                } else {
                    nonRunningMs = saturatedAdd(nonRunningMs, batch.remainingWorkMs().getAsLong());
                }
            }

            this.knownNonRunningWorkMs = nonRunningMs;
            this.runningWorkMs = runningCount == 0 ? EMPTY_LONGS : new long[runningCount];
            this.requestIds = new String[requestIdCount];
            this.unknownWork = hasUnknown;
            indexRequests();
        }

        private static long saturatedAdd(long left, long right) {
            return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
        }

        private void indexRequests() {
            int runningIndex = 0;
            int requestIdIndex = 0;
            for (RequestWork request : requests) {
                requestIds[requestIdIndex++] = request.requestId();
                if (request.phase() == Phase.ENGINE_RUNNING) {
                    runningWorkMs[runningIndex++] = request.remainingWorkMs();
                }
            }
            for (BatchWork batch : batches) {
                for (String requestId : batch.requestIds()) {
                    requestIds[requestIdIndex++] = requestId;
                }
                if (batch.phase() == Phase.ENGINE_RUNNING && batch.remainingWorkMs().isPresent()) {
                    runningWorkMs[runningIndex++] = batch.remainingWorkMs().getAsLong();
                }
            }
            Arrays.sort(requestIds);
        }

        /** Lifecycle phase visible to a projection. Only ENGINE_RUNNING consumes time. */
        public enum Phase {
            COMMITTED,
            ENGINE_QUEUED,
            ENGINE_RUNNING
        }

        /** One individually delivered request, identified by request id. */
        public record RequestWork(String requestId, Phase phase, long remainingWorkMs) {

            public RequestWork {
                checkArgument(remainingWorkMs >= 0L, "remaining request work must be non-negative");
            }
        }

        /**
         * Frozen batch members and execution time. Resolves its shared estimate only when read.
         */
        public record BatchWork(
                List<String> requestIds,
                Phase phase,
                BatchEstimate estimate,
                long elapsedRunningMs) {
            public BatchWork {
                requestIds = List.copyOf(requestIds);
                checkArgument(elapsedRunningMs >= 0L, "elapsed running time must be non-negative");
            }

            public BatchWork(List<String> requestIds, Phase phase, OptionalLong remainingWorkMs) {
                this(
                        requestIds,
                        phase,
                        remainingWorkMs.isPresent()
                                ? new BatchEstimate(remainingWorkMs.getAsLong())
                                : null,
                        0L);
            }

            public OptionalLong remainingWorkMs() {
                return estimate == null
                        ? OptionalLong.empty()
                        : OptionalLong.of(Math.max(0L, estimate.durationMs() - elapsedRunningMs));
            }
        }

        /**
         * One frozen batch's prediction inputs and lazily cached duration, shared by work snapshots.
         * Replacing members before execution creates a new estimate; reads never change Endpoint records.
         */
        public static final class BatchEstimate {
            private static final PrefillTimePredictor.Evaluator FALLBACK =
                    new org.flexlb.balance.prediction.FormulaPredictor(
                            new org.flexlb.config.RoutingConfig.ExecutionTimeEstimatorConfig()
                                    .getExpression());
            private List<GroupPlanner.Item> members;
            private PrefillTimePredictor predictor;
            private volatile long durationMs = -1L;

            public BatchEstimate(long durationMs) {
                checkArgument(durationMs >= 0L, "batch duration must be non-negative");
                this.durationMs = durationMs;
            }

            public BatchEstimate(List<GroupPlanner.Item> members, PrefillTimePredictor predictor) {
                this.members = List.copyOf(members);
                this.predictor = predictor;
            }

            public long durationMs() {
                if (durationMs >= 0L) return durationMs;
                synchronized (this) {
                    if (durationMs >= 0L) return durationMs;
                    var ordered = new java.util.ArrayList<>(members);
                    ordered.sort(
                            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                                    .thenComparing(GroupPlanner.Item::requestId));
                    var features =
                            new PrefillBatchFeatures(
                                    ordered.stream().map(GroupPlanner.Item::features).toList());
                    long predicted;
                    try {
                        predicted =
                                PrefillTimePredictor.predictCommittedBatchMs(
                                        predictor.evaluator(), features);
                    } catch (RuntimeException failure) {
                        try {
                            org.slf4j.LoggerFactory.getLogger("syncLogger")
                                    .warn(
                                            "Prefill remaining batch prediction failed; using default formula",
                                            failure);
                        } catch (RuntimeException ignoredLoggingFailure) {
                            // Logging must not prevent the fallback estimate.
                        }
                        predicted =
                                PrefillTimePredictor.predictCommittedBatchMs(FALLBACK, features);
                    }
                    members = null;
                    predictor = null;
                    durationMs = predicted;
                    return predicted;
                }
            }
        }
    }

    /**
     * Freezes queue rules and membership for request-time estimation.
     * Converts waiting routes to planning inputs once on first read, then releases route references.
     */
    public static final class EstimationSnapshot {
        private final long capturedAtMs;
        private final boolean queueScheduling;

        private final GroupPlanner grouping;
        private final Comparator<? super GroupPlanner.Item> ordering;
        private final GroupPlanner.Constraints constraints;

        private List<RouteResult> waitingRequests;
        private List<GroupPlanner.Item> selectedRequests;
        private volatile List<GroupPlanner.Item> activeItems;

        public EstimationSnapshot(
                long capturedAtMs,
                boolean queueScheduling,
                GroupPlanner grouping,
                Comparator<? super GroupPlanner.Item> ordering,
                GroupPlanner.Constraints constraints,
                List<GroupPlanner.Item> activeItems) {
            this(
                    capturedAtMs,
                    queueScheduling,
                    grouping,
                    ordering,
                    constraints,
                    List.of(),
                    List.of());
            this.activeItems = List.copyOf(activeItems);
        }

        public EstimationSnapshot(
                long capturedAtMs,
                boolean queueScheduling,
                GroupPlanner grouping,
                Comparator<? super GroupPlanner.Item> ordering,
                GroupPlanner.Constraints constraints,
                List<RouteResult> waitingRequests,
                List<GroupPlanner.Item> selectedRequests) {
            if (queueScheduling) {
                Objects.requireNonNull(grouping, "grouping");
            }
            this.capturedAtMs = capturedAtMs;
            this.queueScheduling = queueScheduling;
            this.grouping = grouping;
            this.ordering = ordering;
            this.constraints = constraints;
            this.waitingRequests = List.copyOf(waitingRequests);
            this.selectedRequests = List.copyOf(selectedRequests);
        }

        public long capturedAtMs() {
            return capturedAtMs;
        }

        public boolean queueScheduling() {
            return queueScheduling;
        }

        public GroupPlanner grouping() {
            return grouping;
        }

        public Comparator<? super GroupPlanner.Item> ordering() {
            return ordering;
        }

        public GroupPlanner.Constraints constraints() {
            return constraints;
        }

        public List<GroupPlanner.Item> activeItems() {
            var items = activeItems;
            if (items != null) return items;
            synchronized (this) {
                if (activeItems == null) {
                    var converted =
                            new java.util.ArrayList<GroupPlanner.Item>(
                                    waitingRequests.size() + selectedRequests.size());
                    for (var route : waitingRequests) {
                        converted.add(route.planningInput());
                    }
                    converted.addAll(selectedRequests);
                    activeItems = List.copyOf(converted);
                    waitingRequests = null;
                    selectedRequests = null;
                }
                return activeItems;
            }
        }
    }
}
