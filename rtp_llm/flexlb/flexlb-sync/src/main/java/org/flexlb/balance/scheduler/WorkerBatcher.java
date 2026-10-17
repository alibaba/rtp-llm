package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.DeliveryDispatcher.Member;
import java.util.ArrayList;
import java.util.Arrays;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.planner.GroupingPolicy;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityOrdering;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/** Queue execution for one P generation; endpoint state owns identities and capacity. */
public final class WorkerBatcher {

    /** Frozen decision evidence; the log representation is materialized only on failure. */
    private static final class QueueWaitSnapshot {
        private final String endpoint;
        private final String cause;
        private final long capturedAtMs;
        private final PrefillState.QueueCounters counters;
        private final DecodeRoutingView decode;
        private volatile Map<String, Object> snapshotValues;

        private QueueWaitSnapshot(String endpoint, String cause, long capturedAtMs,
                                  PrefillState.QueueCounters counters, DecodeRoutingView decode) {
            this.endpoint = endpoint;
            this.cause = cause;
            this.capturedAtMs = capturedAtMs;
            this.counters = counters;
            this.decode = decode;
        }

        private Map<String, Object> asMap() {
            Map<String, Object> cached = snapshotValues;
            if (cached != null) { return cached; }
            Map<Integer, Integer> priorityCounts = new HashMap<>();
            int depth = 0;
            int[] requestCountsByPriority = counters.byPriority();
            for (int priority = 0; priority < requestCountsByPriority.length; priority++) {
                if (requestCountsByPriority[priority] > 0) { priorityCounts.put(priority, requestCountsByPriority[priority]); }
                depth += requestCountsByPriority[priority];
            }
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("cause", cause);
            details.put("capturedAtMs", capturedAtMs);
            details.put("endpoint", endpoint);
            details.put("queueDepth", depth);
            details.put("queueVersion", counters.version());
            details.put("priorityCounts", Collections.unmodifiableMap(priorityCounts));
            details.put("prefillRequests", counters.outstandingRequests());
            details.put("prefillBatchSlots", counters.batchSlots());
            if (decode != null) {
                details.put("decode", Map.of("endpoint", decode.address(), "version", decode.admissionVersion(),
                        "engineLoad", decode.engineLoad(), "totalLoad", decode.totalLoad(),
                        "kvTotal", decode.totalKv(), "kvAvailable", decode.placementUsage().hardKvAvailable()));
            }
            cached = Collections.unmodifiableMap(details);
            snapshotValues = cached;
            return cached;
        }
    }

    private enum RuntimeState {
        NEW,
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED
    }

    /** Exact predicate captured by one scheduling cycle. */
    private record BatcherCycleResult(
            RequestRoute request,
            CapacityBoundary unavailable,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs,
            String waitReason) {

        private static final BatcherCycleResult NO_ACTION =
                new BatcherCycleResult(null, null, 0L, 0L, 0L, null);

        private static BatcherCycleResult capacityBlocked(RequestRoute item, CapacityBoundary unavailable) {
            return new BatcherCycleResult(Objects.requireNonNull(item), Objects.requireNonNull(unavailable), 0L, 0L, 0L,
                    unavailable.projectionSemantics() == null ? "Prefill batch capacity exhausted" : unavailable.projectionSemantics().blockedDetail());
        }

        private static BatcherCycleResult awaitingSchedulingChange(RequestRoute head, long queueVersion,
                long schedulingInputVersion, long wakeAtMs, String waitReason) {
            return new BatcherCycleResult(Objects.requireNonNull(head), null, queueVersion, schedulingInputVersion, wakeAtMs, waitReason);
        }

    }

    /** Priority descending, then enqueue order and request identity. */
    public static final Comparator<RequestRoute> PRIORITY_QUEUE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());

    /** FIFO order: unique monotonic enqueue sequence. */
    public static final Comparator<RequestRoute> FIFO_QUEUE_ORDER =
            Comparator.comparingLong(RequestRoute::enqueueSeq);

    private static final Map<String, Object> INITIAL_QUEUE_WAIT_SNAPSHOT =
            Map.of("cause", "waiting for Prefill decision");

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueExecutionSettings settings;
    private final GroupingPolicy grouping;
    private final DeliveryDispatcher dispatcher;
    private Runnable removeCapacityListener;
    /** Shared P ownership lock serializes queue changes, wait predicates and subscriptions. */
    private final ReentrantLock queueLock;
    /** Canonical request ownership guarded exclusively by {@link #queueLock}. */
    private final PrefillState prefillState;
    private final Condition stateChanged;
    /** Exact queued identities whose stop facts must run before normal grouping. */
    private final ArrayDeque<RequestRoute> controlInbox = new ArrayDeque<>();
    private final Thread workerThread;
    private final CancellationException normalStopFailure;
    private final IllegalStateException stopAcknowledgementFailure;
    private final CompletableFuture<Throwable> stopCompletion = new CompletableFuture<>();
    private Thread terminationOwner;
    private volatile RuntimeState runtimeState = RuntimeState.NEW;
    private volatile boolean drainClaimed;
    public final Runnable capacityAvailableSignal;
    /** Exact active head for which this worker is waiting on a capacity event. */
    private BatcherCycleResult capacityBlockedHead;

    /** Last waiting decision; timeout readers never acquire queueLock. */
    private volatile QueueWaitSnapshot latestQueueWaitSnapshot;

    public WorkerBatcher(
            String key,
            PrefillEndpoint prefillEp,
            QueueExecutionSettings settings,
            DeliveryDispatcher dispatcher,
            PrefillState prefillState) {
        this.key = key;
        this.prefillEndpoint = Objects.requireNonNull(prefillEp, "prefillEndpoint");
        this.settings = Objects.requireNonNull(settings);
        this.prefillState = Objects.requireNonNull(prefillState, "prefillState");
        this.queueLock = prefillState.lock;
        this.capacityAvailableSignal = () -> {
            signal(true);
            prefillEndpoint.signal(PrefillEndpoint.Signal.PLACEMENT);
        };
        this.stateChanged = queueLock.newCondition();
        this.grouping = settings.grouping() == DecisionPolicyConfig.Type.SINGLE ? GroupingPolicy.SINGLE : GroupingPolicy.FIXED_WINDOW;
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        this.normalStopFailure = new CancellationException("FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure = new IllegalStateException("FlexLB worker stop callback lost its exact retained owner: " + key);
        this.workerThread = Thread.ofVirtual()
                .name("flexlb-batcher-" + key)
                .uncaughtExceptionHandler((thread, failure) -> Logger.error(
                        "WorkerBatcher[{}] thread died unexpectedly", key, failure))
                .unstarted(this::runLoop);
    }

    public synchronized void start() {
        if (runtimeState != RuntimeState.NEW) {
            throw new IllegalStateException(
                    "Worker batcher cannot start from "
                            + runtimeState);
        }
        runtimeState = RuntimeState.STARTING;
        try {
            workerThread.start();
            runtimeState = RuntimeState.RUNNING;
        } catch (RuntimeException | Error startFailure) {
            Throwable cleanupFailure = stopAndDrain(
                    startFailure,
                    false);
            runtimeState = RuntimeState.STOPPED;
            Failures.append(startFailure, cleanupFailure);
            throw startFailure;
        }
    }

    public boolean offer(RequestRoute item) {
        checkArgument(item.prefillEp() == prefillEndpoint, "incoming item belongs to another Prefill generation");
        if (runtimeState != RuntimeState.RUNNING || drainClaimed) {
            return false;
        }
        boolean accepted;
        List<RequestRoute> victims = List.of();
        queueLock.lock();
        try {
            if (drainClaimed || settings.dispatcherType() == DispatcherConfig.Type.NON_BATCH && prefillEndpoint.isGenerationRetiringOrRetired()) {
                return false;
            }
            accepted = prefillState.updateQueue(PrefillState.QueueChange.ENQUEUE, item, settings.maxOutstandingRequests(), System.currentTimeMillis()) != null;
            if (!accepted && !drainClaimed
                    && settings.preemptQueued()) {
                victims = prefillEndpoint.ownership.replaceQueuedRoutesLocked(item, settings.maxOutstandingRequests());
                accepted = !victims.isEmpty();
            }
            if (accepted) {
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
        if (accepted) {
            for (RequestRoute victim : victims) {
                var owner = victim.ctx().data().getScheduler();
                try { owner.onQueueEvent(victim, AbstractRequestScheduler.QueueKind.PREEMPTED, item, null); }
                catch (Throwable failure) {
                    owner.runtime.recordFailure(failure);
                    try { Logger.error("Queued victim notification failed after replacement: request_id={}", victim.requestId(), failure); }
                    catch (Throwable ignored) { }
                }
            }
        }
        return accepted;
    }

    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        int[] counts = prefillState.captureQueueCounters().byPriority();
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] > 0) { sizeByPriority.put(priority, counts[priority]); }
        }
        return sizeByPriority;
    }

    /** Caller holds P lock; availability only probes the already subscribed resource. */
    public PlanningView planningViewLocked() {
        checkState(queueLock.isHeldByCurrentThread(), "planning view requires queueLock");
        BatcherCycleResult blocked = capacityBlockedHead;
        AdmissionBlock block = blocked == null
                || !prefillState.queueWaitCurrentLocked(blocked.request(), 0L, 0L, true, System.currentTimeMillis())
                || capacityAvailable(blocked.request(), blocked.unavailable()) ? null
                : new AdmissionBlock(blocked.request().requestId(), blocked.request().enqueueSeq(), blocked.unavailable().projectionSemantics());
        return new PlanningView(grouping, schedulingConstraints(), block);
    }

    public Throwable stopAndAwait() {
        checkState(Thread.currentThread() != workerThread, "Prefill runtime cannot await its own worker thread");
        Throwable failure = stopAndDrain(
                normalStopFailure,
                true);
        boolean interrupted = false;
        while (true) {
            try {
                workerThread.join();
                break;
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        synchronized (this) {
            runtimeState = RuntimeState.STOPPED;
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        return failure;
    }

    private Throwable stopAndDrain(
            Throwable terminalFailure,
            boolean interruptWorker) {
        boolean alreadyStopping;
        synchronized (this) {
            alreadyStopping = drainClaimed;
            if (alreadyStopping) {
                checkState(stopCompletion.isDone() || terminationOwner != Thread.currentThread(),
                        "Prefill runtime cannot await its active stop transaction");
            } else {
                drainClaimed = true;
                terminationOwner = Thread.currentThread();
                runtimeState = runtimeState == RuntimeState.NEW
                        ? RuntimeState.STOPPED : RuntimeState.STOPPING;
            }
        }
        if (alreadyStopping) {
            // join is uninterruptible and preserves the caller's interrupt flag.
            return stopCompletion.join();
        }

        Throwable cleanupFailure = null;
        try {
            try {
                queueLock.lock();
                try {
                    controlInbox.clear();
                    cleanupFailure = Failures.run(cleanupFailure, () -> setCapacityBlockedHeadLocked(null));
                    stateChanged.signalAll();
                } catch (Throwable wakeFailure) {
                    cleanupFailure = Failures.append(cleanupFailure, wakeFailure);
                } finally {
                    cleanupFailure = Failures.run(cleanupFailure, queueLock::unlock);
                }
            } catch (Throwable lockFailure) {
                cleanupFailure = Failures.append(cleanupFailure, lockFailure);
            }

            if (interruptWorker && Thread.currentThread() != workerThread) {
                cleanupFailure = Failures.run(cleanupFailure, workerThread::interrupt);
            }

            while (true) {
                RequestRoute item;
                try {
                    item = prefillState.updateQueue(PrefillState.QueueChange.STOP, null, 0L, System.currentTimeMillis());
                } catch (Throwable claimFailure) {
                    cleanupFailure = Failures.append(cleanupFailure, claimFailure);
                    break;
                }
                if (item == null) {
                    break;
                }
                cleanupFailure = Failures.run(cleanupFailure, capacityAvailableSignal);
                try {
                    item.ctx().data().getScheduler().onQueueEvent(item, AbstractRequestScheduler.QueueKind.REJECTED, null, terminalFailure);
                    boolean acknowledged = prefillState.updateQueue(PrefillState.QueueChange.ACK_STOP, item, 0L, System.currentTimeMillis()) != null;
                    if (!acknowledged) {
                        cleanupFailure = Failures.append(cleanupFailure, stopAcknowledgementFailure);
                    }
                } catch (Throwable settlementFailure) {
                    // A failed callback is never acknowledged. A failed acknowledgement
                    // also leaves the exact owner available to generation retirement.
                    cleanupFailure = Failures.append(cleanupFailure, settlementFailure);
                    try {
                        Logger.error("WorkerBatcher[{}] shutdown settlement failed request_id={}",
                                key, item.requestId(), settlementFailure);
                    } catch (Throwable ignoredLoggingFailure) {
                        // Cleanup ownership cannot depend on diagnostics.
                    }
                }
            }
        } catch (Throwable unexpectedCleanupFailure) {
            cleanupFailure = Failures.append(cleanupFailure, unexpectedCleanupFailure);
        } finally {
            synchronized (this) {
                terminationOwner = null;
                stopCompletion.complete(cleanupFailure);
            }
        }
        return cleanupFailure;
    }

    // ==================== endpoint queue operations ====================

    public boolean control(RequestRoute item, Control command) {
        checkArgument(item.prefillEp() == prefillEndpoint, "control belongs to another Prefill generation");
        boolean accepted;
        queueLock.lock();
        try {
            if (drainClaimed || runtimeState != RuntimeState.RUNNING) { return false; }
            if (command == Control.WAKE) { controlInbox.addLast(item); accepted = true; }
            else { accepted = prefillState.updateQueue(PrefillState.QueueChange.REMOVE, item, 0L, System.currentTimeMillis()) != null; }
            if (accepted) { stateChanged.signal(); }
        } finally { queueLock.unlock(); }
        if (accepted && command == Control.REMOVE) { capacityAvailableSignal.run(); }
        return accepted;
    }

    /** Read the last captured queue wait state without waiting or acquiring the queue lock. */
    public Map<String, Object> getLatestQueueWaitSnapshot() {
        QueueWaitSnapshot snapshot = latestQueueWaitSnapshot;
        return snapshot == null ? INITIAL_QUEUE_WAIT_SNAPSHOT : snapshot.asMap();
    }

    /** Capture fixed-size counters without formatting the PV record on the scheduling loop. */
    private void recordQueueWait(RequestRoute head, String reason) {
        PrefillState.QueueCounters counters = prefillState.captureQueueCounters();
        DecodeRoutingView decode = head.decodeEp() == null ? null : head.decodeEp().routingView();
        latestQueueWaitSnapshot = new QueueWaitSnapshot(key, reason, System.currentTimeMillis(), counters, decode);
    }

    // ==================== Queue ownership and projection ====================

    private GroupPlanner.Constraints schedulingConstraints() {
        WorkerStatus status = prefillEndpoint.getStatus();
        var engine = status == null ? null : status.committedEngineObservation();
        return new GroupPlanner.Constraints(settings.maxRequests(), schedulingCapacity(engine, PlanningResource.TOKENS),
                schedulingCapacity(engine, PlanningResource.KV), settings.executionBudgetMs(), settings.collectionWaitMs());
    }

    public enum PlanningResource { TOKENS, KV }

    /** The same effective limits drive worker selection and status input invalidation. */
    public static long schedulingCapacity(WorkerStatus.EngineObservation engine, PlanningResource resource) {
        Objects.requireNonNull(resource, "resource");
        if (engine == null) { return Long.MAX_VALUE; }
        long maximum = resource == PlanningResource.KV ? engine.totalKvCacheTokens()
                : engine.maxBatchTokensSize() > 0L ? engine.maxBatchTokensSize() : engine.maxSeqLen();
        return maximum <= 0L ? Long.MAX_VALUE : resource == PlanningResource.KV
                ? Math.clamp(engine.availableKvCacheTokens(), 0L, maximum) : maximum;
    }

    // ==================== Delivery ownership ====================

    /** Prepare an ordered prefix, commit once, and hand off outside all owner locks. */
    CapacityBoundary processSelection(List<RequestRoute> candidates, String reason,
            PrefillTimePredictor.Evaluator evaluator, OptionalLong plannedPrediction) {
        queueLock.lock();
        try {
            if (candidates.isEmpty() || drainClaimed || !prefillState.ownsSelectionLocked(candidates, System.currentTimeMillis())) { return null; }
        } finally { queueLock.unlock(); }
        Preparation prep = new Preparation(candidates.getFirst(), candidates.size());
        CapacityBoundary boundary = null;
        RequestRoute blocked = null;
        long payloadBytes = 16L;
        try (prep) {
            try {
                for (RequestRoute item : candidates) {
                    long itemBytes = prep.batch ? item.batchPayloadSizeUpperBound() : 0L;
                    if (prep.batch && !prep.members.isEmpty()
                            && itemBytes > org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE - payloadBytes) { break; }
                    synchronized (item.ctx()) {
                        var state = item.ctx().snapshot();
                        boundary = state.preparedDelivery() && state.route() == item
                                ? prep.acquire(item, evaluator) : CapacityBoundary.OWNERSHIP_LOST;
                        if (boundary != null) { blocked = item; break; }
                        payloadBytes += itemBytes;
                    }
                }
                if (prep.members.isEmpty()) { return boundary; }
                List<RequestRoute> items = prep.members.stream().map(Member::route).toList();
                long prediction = !prep.batch ? 0L : plannedPrediction.isPresent() && items.size() == candidates.size()
                        ? plannedPrediction.getAsLong() : PrefillPredictionBoundary.predictCommittedBatchMs(evaluator,
                                PrefillBatchFeatures.from(items, RequestRoute::seqLen, RequestRoute::hitCache));
                RequestRoute failed = boundary != null && boundary.status() == CapacityBoundary.Status.FAILED ? blocked : null;
                long[] predictions = prep.batch ? new long[]{prediction} : prep.predictions.length == items.size()
                        ? prep.predictions : Arrays.copyOf(prep.predictions, items.size());
                prep.committed = commitSelection(now -> prefillEndpoint.commitLocked(new PrefillState.CommitPlan(
                        prep.batch ? PrefillState.CommitKind.BATCH : PrefillState.CommitKind.QUEUED, items, List.of(), prep.reservation, predictions, failed, now)));
                if (prep.committed == null) { return null; }
                prep.reservation = null;
                boolean removedFailedMember = prep.committed.removedFailedMember();
                if (removedFailedMember) { removeAndNotify(blocked, boundary, true); }
                WorkSnapshot precedingWork = prep.committed.precedingWork().materialize();
                if (prep.batch) {
                    var command = new DeliveryDispatcher.Batch(prep.batchId, prep.members, precedingWork, prediction, evaluator,
                            boundary != null && boundary.unavailable() ? "delivery_capacity_prefix" : reason,
                            prep.committed.remainingQueueDepth(), prep.committed, prep.executionPermit);
                    prep.transferred = true;
                    dispatcher.deliverBatch(command);
                } else {
                    deliverRoutes(prep.members, prep.predictions, precedingWork, prep.runtime, prep.committed);
                    prep.committed = null;
                }
                if (!removedFailedMember) { prefillEndpoint.signal(PrefillEndpoint.Signal.PLACEMENT); }
                return null;
            } catch (Throwable failure) {
                if (prep.committed != null && !prep.transferred) {
                    Throwable cause = failure;
                    for (Member member : prep.members) {
                        failure = Failures.run(failure, () -> member.route().ctx().data().getScheduler().onDeliveryEvidence(null,
                                new ContextResult.PreparationFailure(null, member.route(), String.valueOf(cause.getMessage()), System.currentTimeMillis())));
                    }
                }
                if (prep.committed != null) { prep.primaryFailure = failure; Logger.error("Committed delivery failed", failure); return null; }
                throw Failures.propagate(failure, "delivery preparation failed");
            }
        }
    }

    /** Route publication and its committed queue owner finish before telemetry. */
    private void deliverRoutes(List<Member> members, long[] predictions, WorkSnapshot precedingWork,
            SchedulerRuntime runtime, PrefillState.CommittedHandoff committed) {
        DeliveryClaim[] claims = new DeliveryClaim[members.size()];
        for (int index = 0; index < members.size(); index++) {
            Member member = members.get(index);
            try { claims[index] = member.route().ctx().data().getScheduler().claimDelivery(member.route(),
                    DeliveryClaimKind.ROUTE_DECISION, 0L, member.decode()); }
            catch (Throwable failure) {
                try { member.route().ctx().data().getScheduler().onDeliveryEvidence(null,
                            new ContextResult.PreparationFailure(null, member.route(), String.valueOf(failure.getMessage()), System.currentTimeMillis())); }
                catch (Throwable ownerFailure) { runtime.recordFailure(ownerFailure); }
            }
        }
        long unstarted = 0L;
        List<RequestRoute> delivered = new ArrayList<>(claims.length);
        Throwable failure = null;
        for (int index = 0; index < claims.length; index++) {
            DeliveryClaim claim = claims[index];
            if (claim == null) { continue; }
            unstarted = saturatedAdd(unstarted, predictions[index]);
            try {
                long nowMs = System.currentTimeMillis();
                claim.owner().data().getScheduler().onDeliveryEvidence(claim,
                        new ContextResult.RouteDelivered(claim, PrefillPredictionBoundary.predictedVisibilityDeadlineMs(precedingWork, unstarted,
                                claim.owner().data().getConfig().getRequestLifecycle().getDecision().getLifetime(), nowMs), nowMs));
                delivered.add(claim.item());
            } catch (Throwable problem) { failure = Failures.append(failure, problem); }
        }
        failure = Failures.append(failure, DeliveryDispatcher.closeCommitted(members, committed));
        members.clear();
        if (failure != null) {
            runtime.recordFailure(failure);
            try { Logger.error("Route publication failed", failure); }
            catch (Throwable ignored) { }
        }
        if (!delivered.isEmpty()) {
            try { runtime.deliveryReporter().reportDelivery(0L, null, committed.remainingQueueDepth(), delivered, 0L); }
            catch (Throwable monitoringFailure) { Logger.error("Route delivery telemetry failed", monitoringFailure); }
        }
    }

    boolean capacityAvailable(RequestRoute item, CapacityBoundary boundary) {
        var runtime = item.ctx().data().getScheduler().runtime;
        return switch (boundary.resource()) {
            case DISPATCH_EXECUTOR -> runtime.deliveries().deliveryCapacityAvailable();
            case PREFILL_BATCH -> prefillEndpoint.ownership.batchCapacityAvailable(runtime.deliveries().maxInflightBatches);
            case DECODE_ENGINE -> item.decodeEp().shouldRetryDispatch(item.decodeReservation(), runtime.deliveries().decodeDeliveryCapacity);
        };
    }

    Runnable watchCapacity(RequestRoute item, CapacityBoundary boundary, Runnable wake) {
        return switch (boundary.resource()) {
            case DISPATCH_EXECUTOR -> item.ctx().data().getScheduler().runtime.deliveries().watchDeliveryCapacity(wake);
            case DECODE_ENGINE -> {
                yield item.decodeEp().watchEngineDispatchCapacity(wake);
            }
            case PREFILL_BATCH -> () -> { };
        };
    }

    /** Serialize a delivery commit with queue drain and exact generation ownership. */
    PrefillState.CommittedHandoff commitSelection(java.util.function.LongFunction<PrefillState.CommittedHandoff> commit) {
        queueLock.lock();
        try { return drainClaimed ? null : commit.apply(System.currentTimeMillis()); }
        finally { queueLock.unlock(); }
    }

    private BatcherCycleResult commitBoundary(
            RequestRoute blockedItem,
            CapacityBoundary blockedResult) {
        if (blockedItem == null || blockedResult == null) { return BatcherCycleResult.NO_ACTION; }
        boolean capacityBlocked = blockedResult.unavailable();
        boolean failed = blockedResult.status() == CapacityBoundary.Status.FAILED;
        boolean removed = false;
        queueLock.lock();
        try {
            if (drainClaimed) { return BatcherCycleResult.NO_ACTION; }
            long nowMs = System.currentTimeMillis();
            if (capacityBlocked) {
                if (!prefillState.queueWaitCurrentLocked(blockedItem, 0L, 0L, true, nowMs)) {
                    return BatcherCycleResult.NO_ACTION;
                }
            } else if (failed) {
                removed = prefillState.updateQueue(PrefillState.QueueChange.REMOVE_UNEXPIRED, blockedItem, 0L, nowMs) != null;
            }
        } finally {
            queueLock.unlock();
        }
        if (capacityBlocked) { return BatcherCycleResult.capacityBlocked(blockedItem, blockedResult); }
        if (removed) { removeAndNotify(blockedItem, blockedResult, true); }
        return BatcherCycleResult.NO_ACTION;
    }

    /** Once removed, the request must be notified even if a capacity listener fails. */
    private void removeAndNotify(RequestRoute item, CapacityBoundary failure, boolean alreadyRemoved) {
        if (!alreadyRemoved && prefillState.updateQueue(PrefillState.QueueChange.REMOVE, item, 0L, System.currentTimeMillis()) == null) { return; }
        Throwable problem = Failures.run(null, capacityAvailableSignal);
        try {
            if (failure == null) {
                item.ctx().data().getScheduler().onQueueEvent(item, AbstractRequestScheduler.QueueKind.EXPIRED, null, null);
            } else {
                item.ctx().data().getScheduler().onDeliveryEvidence(null, new ContextResult.PreparationFailure(null, item, String.valueOf(failure.cause().getMessage()), System.currentTimeMillis()));
            }
        } catch (Throwable callbackFailure) {
            item.ctx().data().getScheduler().runtime.recordFailure(callbackFailure);
            problem = Failures.run(problem, () -> Logger.error("WorkerBatcher[{}] removed request callback failed request_id={}",
                    key, item.requestId(), callbackFailure));
        }
        Failures.rethrow(problem, "queue removal notification failed");
    }

    // ==================== Group decisions ====================

    private BatcherCycleResult processQueue() {
        int maxRequests = settings.maxRequests();
        long predictionBudgetMs = settings.executionBudgetMs();
        long windowMs = settings.collectionWaitMs();
        PrefillState.QueueSnapshot snapshot = prefillState.captureQueue(maxRequests);
        RequestRoute head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.NO_ACTION;
        }
        long nowMs = System.currentTimeMillis();
        var requestState = head.ctx().snapshot();
        if (requestState.placement() != null && !requestState.placement().published()) {
            return BatcherCycleResult.awaitingSchedulingChange(head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(), Long.MAX_VALUE, "Route commit in progress");
        }
        if (head.requestExpired(nowMs)) {
            removeAndNotify(head, null, false);
            return BatcherCycleResult.NO_ACTION;
        }
        GroupPlanner.Constraints constraints = schedulingConstraints();
        if (Math.max(0L, head.seqLen()) > constraints.batchKvCapacity()) {
            return BatcherCycleResult.awaitingSchedulingChange(head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(), head.expiresAtMs(), "Prefill KV capacity exhausted");
        }
        for (int index = 1; index < Math.min(maxRequests, snapshot.items().size()); index++) {
            RequestRoute member = snapshot.items().get(index);
            if (member.requestExpired(nowMs)) {
                removeAndNotify(member, null, false);
                return BatcherCycleResult.NO_ACTION;
            }
        }
        try {
            PrefillTimePredictor.Evaluator evaluator = prefillEndpoint.predictor.evaluator();
            var predictor = predictionBudgetMs > 0L ? RouteProjection.newGroupPredictor(settings.dispatcherType(), evaluator) : null;
            long planningAtMs = System.currentTimeMillis();
            var selection = grouping.select(snapshot.items(), constraints, predictor);
            if (selection.items().isEmpty()) {
                return BatcherCycleResult.NO_ACTION;
            }
            String reason = grouping.dispatchReason(selection, constraints, planningAtMs);
            if (reason == null) {
                return BatcherCycleResult.awaitingSchedulingChange(head, snapshot.queueVersion(),
                        snapshot.schedulingInputVersion(),
                        Math.min(GroupPlanner.collectionDeadlineMs(selection.windowOpenedAtMs(), windowMs),
                                head.expiresAtMs()), "Prefill collection window");
            }
            // Recheck advisory capacity after prediction, before acquiring hard-capacity leases.
            constraints = schedulingConstraints();
            if ((selection.items().size() > 1 && !selection.fitsCompute(constraints.batchTokenCapacity()))
                    || !selection.fitsKv(constraints.batchKvCapacity())) {
                return BatcherCycleResult.NO_ACTION;
            }
            OptionalLong prediction = selection.selectedPredictionMs().isEmpty() ? OptionalLong.empty()
                    : OptionalLong.of(PrefillPredictionBoundary.committedDecisionGroupMs(selection.selectedPredictionMs().getAsDouble()));
            return commitBoundary(head, processSelection(selection.items(), reason, evaluator, prediction));
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(head, CapacityBoundary.failed(failure));
        }
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        try {
            while (!drainClaimed && !Thread.currentThread().isInterrupted()) {
                try {
                    awaitWork(null);
                    if (drainClaimed) { return; }
                    RequestRoute control;
                    queueLock.lock();
                    try { control = controlInbox.pollFirst(); }
                    finally { queueLock.unlock(); }
                    if (control != null) {
                        control.ctx().data().getScheduler().onQueueEvent(control, AbstractRequestScheduler.QueueKind.CONTROL, null, null);
                        continue;
                    }
                    BatcherCycleResult result = processQueue();
                    if (result.waitReason() != null) { recordQueueWait(result.request(), result.waitReason()); }
                    if (result.request() != null) { awaitWork(result); }
                } catch (InterruptedException interruption) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Throwable failure) {
                    try { Logger.error("WorkerBatcher[{}] stopped after an unexpected loop failure", key, failure); }
                    catch (Throwable ignoredLoggingFailure) { }
                    Throwable cleanup = stopAndDrain(failure, false);
                    if (cleanup != null) {
                        try { Logger.error("WorkerBatcher[{}] failure cleanup exposed invariants", key, cleanup); }
                        catch (Throwable ignoredLoggingFailure) { }
                    }
                    return;
                }
            }
        } finally {
            synchronized (this) { runtimeState = RuntimeState.STOPPED; }
        }
    }

    /**
     * All waits share the queue lock and control-inbox check. Capacity waits
     * subscribe before testing availability, so release-before-await is safe.
     * A null decision waits for the first item; other decisions retain their
     * exact head and either resource or queue/input-version predicate.
     */
    private void awaitWork(BatcherCycleResult waiting) throws InterruptedException {
        boolean capacityWait = waiting != null && waiting.unavailable() != null;
        queueLock.lockInterruptibly();
        try {
            if (capacityWait && !drainClaimed) {
                setCapacityBlockedHeadLocked(waiting);
            }
            while (!drainClaimed && controlInbox.isEmpty() && prefillState.queueWaitCurrentLocked(
                    waiting == null ? null : waiting.request(),
                    waiting == null ? 0L : waiting.queueVersion(),
                    waiting == null ? 0L : waiting.schedulingInputVersion(),
                    capacityWait, System.currentTimeMillis())
                    && (!capacityWait || !capacityAvailable(waiting.request(), waiting.unavailable()))) {
                if (waiting == null) {
                    setCapacityBlockedHeadLocked(null);
                }
                long wakeAtMs = waiting == null ? Long.MAX_VALUE
                        : capacityWait ? waiting.request().expiresAtMs() : waiting.wakeAtMs();
                long nowMs = System.currentTimeMillis();
                if (wakeAtMs <= nowMs) {
                    return;
                }
                if (wakeAtMs == Long.MAX_VALUE) {
                    stateChanged.await();
                } else {
                    stateChanged.awaitNanos(TimeUnit.MILLISECONDS.toNanos(wakeAtMs - nowMs));
                }
            }
        } finally {
            try {
                if (capacityWait) {
                    setCapacityBlockedHeadLocked(null);
                }
            } finally {
                queueLock.unlock();
            }
        }
    }

    /** Called after Prefill or Decode capacity changes, outside endpoint locks. */

    /** Own the blocked head, its listener and projection invalidation under queueLock. */
    private void setCapacityBlockedHeadLocked(
            BatcherCycleResult blocked) {
        checkState(queueLock.isHeldByCurrentThread(), "capacity block update requires queueLock");
        if (capacityBlockedHead == blocked) {
            return;
        }
        Runnable previous = removeCapacityListener;
        removeCapacityListener = null;
        capacityBlockedHead = blocked;
        prefillState.schedulingInputsChangedLocked();
        try { if (previous != null) { previous.run(); } }
        finally {
            if (blocked != null) {
                removeCapacityListener = watchCapacity(blocked.request(), blocked.unavailable(), capacityAvailableSignal);
            }
        }
    }

    /** Wake after the status transaction has already committed its planning revision. */
    public void wake() {
        queueLock.lock();
        try { stateChanged.signal(); }
        finally { queueLock.unlock(); }
    }

    /** Wake decisions whose advisory worker-status or predictor input changed. */
    public void signal(boolean capacityOnly) {
        queueLock.lock();
        try {
            if (!capacityOnly || capacityBlockedHead != null) {
                prefillState.schedulingInputsChangedLocked();
                stateChanged.signal();
            }
        } finally { queueLock.unlock(); }
    }

    public enum Control { REMOVE, WAKE }
    public record PlanningView(GroupingPolicy grouping, GroupPlanner.Constraints constraints, AdmissionBlock block) { }

    private final class Preparation implements AutoCloseable {
        final boolean batch = settings.dispatcherType() == DispatcherConfig.Type.BATCH;
        final SchedulerRuntime runtime;
        final ArrayList<Member> members;
        final long[] predictions;
        DeliveryExecution.DeliveryPermit executionPermit;
        PrefillState.BatchReservation reservation;
        PrefillState.CommittedHandoff committed;
        long batchId;
        boolean transferred;
        Throwable primaryFailure;

        Preparation(RequestRoute first, int size) {
            runtime = first.ctx().data().getScheduler().runtime;
            members = new ArrayList<>(size);
            predictions = batch ? null : new long[size];
        }

        CapacityBoundary acquire(RequestRoute item, PrefillTimePredictor.Evaluator evaluator) {
            if (batch && members.isEmpty()) {
                try {
                    var execution = runtime.deliveries().tryReserveDelivery();
                    if (!execution.accepted()) { return execution.boundary(); }
                    executionPermit = execution.value();
                    batchId = runtime.deliveries().nextBatchId();
                    checkState(batchId > 0L, "batch id supplier returned a non-positive id");
                    var acquired = prefillEndpoint.reserveBatch(item, batchId, runtime.deliveries().maxInflightBatches);
                    if (acquired.status() != PrefillState.CapacityStatus.ACQUIRED) {
                        return switch (acquired.status()) {
                            case CAPACITY_FULL -> CapacityBoundary.deliveryUnavailable(CapacityBoundary.Resource.PREFILL_BATCH);
                            case REQUEST_NOT_ACTIVE -> CapacityBoundary.OWNERSHIP_LOST;
                            default -> CapacityBoundary.failed(new IllegalStateException("Prefill batch admission failed: request_id="
                                    + item.requestId() + " status=" + acquired.status()));
                        };
                    }
                    reservation = acquired.reservation();
                } catch (Throwable failure) { return CapacityBoundary.failed(failure); }
            }
            long prediction = batch ? 0L : PrefillPredictionBoundary.predictSingleRequestMs(evaluator, item.seqLen(), item.hitCache());
            var attempt = Member.prepare(item, runtime.deliveries().decodeDeliveryCapacity);
            if (!attempt.accepted()) { return attempt.boundary(); }
            Member member = attempt.value();
            try {
                if (!batch) { predictions[members.size()] = prediction; }
                members.add(member);
                return null;
            } catch (Throwable failure) { return CapacityBoundary.failed(Failures.run(failure, member::close)); }
        }

        @Override public void close() {
            if (transferred) { return; }
            Throwable failure = null;
            for (Member member : members) { failure = Failures.run(failure, member::close); }
            if (reservation != null) { failure = Failures.run(failure, () -> prefillEndpoint.release(reservation, null)); }
            if (committed != null) { failure = Failures.append(failure, Failures.close(committed)); }
            if (executionPermit != null) { failure = Failures.append(failure, Failures.close(executionPermit)); }
            if (primaryFailure != null) { Failures.append(primaryFailure, failure); }
            else { Failures.rethrow(failure, "temporary delivery resource cleanup failed"); }
        }
    }

}
