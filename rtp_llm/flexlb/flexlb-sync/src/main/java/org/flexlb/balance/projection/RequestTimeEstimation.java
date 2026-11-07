package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.prediction.PrefillTimePredictor.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.flexlb.balance.scheduler.RouteExecution;
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

/** Frozen-work cost through the incoming request; collection waits are not predicted. */
public final class RequestTimeEstimation {

    private static final ThreadLocal<RequestTimeEstimation> ESTIMATORS =
            ThreadLocal.withInitial(RequestTimeEstimation::new);

    private static final String INVALID_PREDICTION_DETAIL = "PREDICTOR_RETURNED_INVALID_VALUE";
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
            project(inputs, planningAtMs);
            applyAdmissionPolicy(inputs.queue());
            return predictions;
        } finally {
            predictions.evaluator = null;
            predictions.batchPrediction = null;
            inUse = false;
        }
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

    private CandidateView projectWithPredictions(
            ProjectionQueueSnapshot queue, WorkSnapshot committed, long projectionAtMs) {
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
            ProjectionQueueSnapshot queue,
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

    private CandidateView unavailable(String detail) {
        return candidate(State.UNAVAILABLE, UNKNOWN, 0L, InitialHeadDisposition.NONE, detail);
    }

    private CandidateView blocked(
            long incomingPrefillMs, InitialHeadDisposition initialHeadDisposition, String detail) {
        return candidate(State.BLOCKED, UNKNOWN, incomingPrefillMs, initialHeadDisposition, detail);
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

    /**
     * Admission restriction imposed by the captured head, including after a probe overtakes it.
     */
    public record AdmissionBlockSemantics(
            String blockedDetail, State afterProbe, String afterProbeDetail) {
        public AdmissionBlockSemantics {
            checkArgument(
                    afterProbe == State.BLOCKED || afterProbe == State.UNAVAILABLE,
                    "an admission block must remain blocked or unavailable after the probe");
        }
    }

    /** Read-only projection result. Invocation-scoped views must not be retained. */
    public interface CandidateView {
        State state();

        default State admissionState() {
            return state();
        }

        default String admissionDetail() {
            return detail();
        }

        /** Known execution cost through this request; excludes future collection waits. */
        long projectedTtftMsValue();

        long incomingPrefillMs();

        InitialHeadDisposition initialHeadDisposition();

        String detail();

        long cacheHitTokens();

        long routingCacheMatchTokens();

        default boolean engineWorkUnmodeled() {
            return state() == State.UNMODELED_ENGINE_WORK;
        }

        default boolean selectable() {
            return state() == State.MODELED && projectedTtftMsValue() != UNKNOWN;
        }
    }

    public static final long UNKNOWN = -1L;

    public enum InitialHeadDisposition {
        NONE,
        BEFORE_PROBE,
        AFTER_PROBE,
        TERMINAL_PRUNED
    }

    public enum State {
        MODELED,
        UNMODELED_ENGINE_WORK,
        BLOCKED,
        UNAVAILABLE
    }

    /** Queue/work captured under one ownership lock, with an immutable model for this projection. */
    public record Inputs(
            ProjectionQueueSnapshot queue,
            WorkSnapshot work,
            long ownershipVersion,
            PrefillTimePredictor.Evaluator evaluator,
            DispatcherConfig.Type deliveryMode) {

        public Inputs {
            // Committed work may reuse an older clock base while its ownership
            // is unchanged. The projector rebases running duration to queue time.
            checkArgument(
                    queue.capturedAtMs() >= work.capturedAtMs(),
                    "work snapshot cannot be newer than its queue capture");
            checkArgument(ownershipVersion >= 0L, "ownershipVersion must be non-negative");
        }
    }

    /**
     * Combine the captured capacity wait with the projection without changing its ranking cost.
     */
    private void applyAdmissionPolicy(ProjectionQueueSnapshot queue) {
        var candidate = predictions;
        candidate.admissionState = candidate.state;
        candidate.admissionDetail = candidate.detail;
        var observation = queue.admissionBlock();
        if (!queue.queueScheduling()
                || observation == null
                || observation.semantics() == null
                || (candidate.state != State.MODELED && !candidate.engineWorkUnmodeled())) {
            return;
        }
        var semantics = observation.semantics();
        if (!candidate.engineWorkUnmodeled()) {
            switch (candidate.initialHeadDisposition()) {
                case TERMINAL_PRUNED -> {
                    return;
                }
                case NONE ->
                        throw new IllegalStateException(
                                "admission-blocked ACTIVE head was not projected");
                case BEFORE_PROBE -> {}
                case AFTER_PROBE -> {
                    candidate.admissionState = semantics.afterProbe();
                    candidate.admissionDetail = semantics.afterProbeDetail();
                    return;
                }
            }
        }
        candidate.admissionState = State.BLOCKED;
        candidate.admissionDetail = semantics.blockedDetail();
    }

    /** One incremental predictor for one live queue selection. */
    public static GroupPlanner.PrefixPrediction<RouteExecution> newGroupPredictor(
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
            public double append(RouteExecution added, List<RouteExecution> items) {
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
     * The sole boundary between projection math and an external predictor.
     * Predictor failures remain distinguishable from invalid returned values,
     * while every numeric output is validated before it reaches scheduling.
     */
    private static final class Prediction
            implements GroupPlanner.PrefixPrediction<GroupPlanner.Item>, CandidateView {

        private State state;
        private State admissionState;
        private String admissionDetail;

        @Override
        public State admissionState() {
            return admissionState;
        }

        @Override
        public String admissionDetail() {
            return admissionDetail;
        }

        private long projectedTtftMsValue;
        private long incomingPrefillMs;
        private InitialHeadDisposition headDisposition;
        private String detail;
        private long hitCache;
        private long routingCacheMatchTokens;

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

        public long capturedAtMs() {
            return capturedAtMs;
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

        /** One shared EnqueueBatch work unit and its live request identities. */
        public record BatchWork(
                List<String> requestIds, Phase phase, OptionalLong remainingWorkMs) {

            public BatchWork {
                requestIds = List.copyOf(requestIds);
                checkArgument(
                        !remainingWorkMs.isPresent() || remainingWorkMs.getAsLong() >= 0L,
                        "remaining batch work must be non-negative");
            }
        }

        public boolean hasUnknownWork() {
            calculate();
            return unknownWork;
        }

        public boolean containsRequest(String requestId) {
            calculate();
            return Arrays.binarySearch(requestIds, requestId) >= 0;
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

        private static long saturatedAdd(long left, long right) {
            return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
        }
    }

    /**
     * Immutable scheduling inputs captured for a route-time what-if projection.
     *
     * <p>The active items are already in production queue order. This object is
     * materialized from the endpoint's PrefillEndpoint at one linearization
     * point together with committed work and pending count.
     */
    public record ProjectionQueueSnapshot(
            long capturedAtMs,
            boolean queueScheduling,
            GroupPlanner grouping,
            Comparator<? super GroupPlanner.Item> ordering,
            GroupPlanner.Constraints constraints,
            List<GroupPlanner.Item> activeItems,
            AdmissionBlock admissionBlock) {

        public ProjectionQueueSnapshot {
            if (queueScheduling) {
                Objects.requireNonNull(grouping, "grouping");
            }
            activeItems = List.copyOf(activeItems);
            if (admissionBlock != null) {
                checkArgument(!activeItems.isEmpty(), "admission block requires an ACTIVE head");
                GroupPlanner.Item head = activeItems.getFirst();
                checkArgument(
                        java.util.Objects.equals(head.requestId(), admissionBlock.requestId())
                                && head.enqueueSeq() == admissionBlock.enqueueSeq(),
                        "admission block must identify the exact ACTIVE head");
            }
        }

        /**
         * Exact ACTIVE head whose current capacity rejection parks the worker.
         * Null semantics mean the wait only limits delivery, not queue publication.
         */
        public record AdmissionBlock(
                String requestId,
                long enqueueSeq,
                RequestTimeEstimation.AdmissionBlockSemantics semantics) {}
    }
}
