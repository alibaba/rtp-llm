package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.planner.GroupingPolicy;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.DecisionPolicyConfig;
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

/**
 * Runs grouping and delivery for one Prefill endpoint generation.
 * PrefillState owns queued identities, capacity reservations and queue revisions;
 * this runtime owns the worker thread, control inbox and wait/wakeup protocol.
 * Queue decisions and subscriptions use the State ownership lock. Delivery and
 * scheduler callbacks run after unlocking. Groups may use EnqueueBatch or
 * individual route decisions, independently of their grouping policy.
 */
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

        private static BatcherCycleResult capacityBlocked(
                RequestRoute item,
                CapacityBoundary unavailable) {
            return new BatcherCycleResult(
                    Objects.requireNonNull(item, "item"),
                    Objects.requireNonNull(unavailable, "unavailable"),
                    0L, 0L, 0L, unavailable.projectionSemantics() == null
                            ? "Prefill batch capacity exhausted" : unavailable.projectionSemantics().blockedDetail());
        }

        private static BatcherCycleResult awaitingSchedulingChange(
                RequestRoute head,
                long queueVersion,
                long schedulingInputVersion,
                long wakeAtMs,
                String waitReason) {
            return new BatcherCycleResult(
                    Objects.requireNonNull(head, "head"), null,
                    queueVersion, schedulingInputVersion, wakeAtMs, waitReason);
        }

        private boolean capacityBlocked() {
            return unavailable != null;
        }
    }

    /**
     * PRIORITY queue order: delegates to
     * {@link PriorityOrdering} (priority desc → enqueue-seq asc for
     * same-priority FIFO) with {@code requestId} as the final deterministic
     * tie-break.
     *
     * <p>{@link #FIFO_QUEUE_ORDER} preserves enqueue order.
     */
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
    private final DeliveryStrategy deliveryStrategy;
    /**
     * Guards queue mutations and exact ownership publication.
     *
     * <p>The lock and generation stay active for both FIFO and PRIORITY so
     * every ordering mode has the same mutation guarantees.
     */
    private final ReentrantLock queueLock;
    /** Canonical request ownership guarded exclusively by {@link #queueLock}. */
    private final PrefillState prefillState;
    /**
     * One per-worker condition for new queue work and endpoint-capacity release.
     * Every predicate transition and signal is serialized by queueLock, so a
     * release cannot race between the cap re-check and await.
     */
    private final Condition stateChanged;
    /** Exact queued identities whose stop facts must run before normal grouping. */
    private final ArrayDeque<RequestRoute> controlInbox = new ArrayDeque<>();
    /** QUEUE-only serialized runtime; DIRECT owns no queue execution thread. */
    private final Thread workerThread;
    /** Constructed before this endpoint generation can begin retirement. */
    private final CancellationException normalStopFailure;
    /** Fixed diagnostic for an impossible exact stop acknowledgement loss. */
    private final IllegalStateException stopAcknowledgementFailure;
    private final CompletableFuture<Throwable> stopCompletion = new CompletableFuture<>();
    private Thread terminationOwner;
    private volatile RuntimeState runtimeState = RuntimeState.NEW;
    private volatile boolean drainClaimed;
    private final Runnable capacityAvailableSignal;
    /** Exact active head for which this worker is waiting on a capacity event. */
    private BatcherCycleResult capacityBlockedHead;

    /** Last waiting decision; timeout readers never acquire queueLock. */
    private volatile QueueWaitSnapshot latestQueueWaitSnapshot;

    public WorkerBatcher(
            String key,
            PrefillEndpoint prefillEp,
            QueueExecutionSettings settings,
            DeliveryStrategy deliveryStrategy,
            PrefillState prefillState) {
        this.key = key;
        this.prefillEndpoint = Objects.requireNonNull(prefillEp, "prefillEndpoint");
        this.settings = Objects.requireNonNull(settings);
        this.prefillState = Objects.requireNonNull(prefillState, "prefillState");
        this.queueLock = prefillState.ownershipLock();
        this.capacityAvailableSignal = () -> {
            signalDeliveryCapacityAvailable();
            prefillEndpoint.signalPlacementCapacityChanged();
        };
        this.stateChanged = queueLock.newCondition();
        this.grouping = settings.grouping() == DecisionPolicyConfig.Type.SINGLE ? GroupingPolicy.SINGLE : GroupingPolicy.FIXED_WINDOW;
        this.deliveryStrategy = Objects.requireNonNull(
                deliveryStrategy, "deliveryStrategy");
        this.normalStopFailure = new CancellationException(
                "FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure = new IllegalStateException(
                "FlexLB worker stop callback lost its exact retained owner: "
                        + key);
        this.workerThread = Thread.ofVirtual()
                .name("flexlb-batcher-" + key)
                .uncaughtExceptionHandler((thread, failure) -> Logger.error(
                        "WorkerBatcher[{}] thread died unexpectedly", key, failure))
                .unstarted(this::runLoop);
    }

    /** The permanently bound wake identity used by this generation's capacity source. */
    public Runnable capacityAvailableSignal() { return capacityAvailableSignal; }

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
        requireExactEndpoint(item, "incoming item");
        if (runtimeState != RuntimeState.RUNNING || drainClaimed) {
            return false;
        }
        boolean accepted;
        List<RequestRoute> victims = List.of();
        queueLock.lock();
        try {
            if (drainClaimed || item.requiresRouteReservation() && prefillEndpoint.isGenerationRetiringOrRetired()) {
                return false;
            }
            accepted = prefillState.enqueueActiveLocked(item, settings.maxOutstandingRequests());
            if (!accepted && !drainClaimed
                    && settings.preemptQueued()) {
                victims = prefillEndpoint.replaceQueuedRoutesLocked(item, settings.maxOutstandingRequests());
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
                victim.ctx().scheduler().onQueuedItemPreempted(victim, item);
            }
        }
        return accepted;
    }

    /**
     * Snapshot of the current active queue depth bucketed by normalized
     * scheduling priority. Only priorities present in the queue appear in the
     * result, matching the tagged queue-depth metric behavior.
     */
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        int[] counts = prefillState.captureQueueCounters().byPriority();
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] > 0) { sizeByPriority.put(priority, counts[priority]); }
        }
        return sizeByPriority;
    }

    /**
     * Capture the exact active head whose worker wait predicate still holds.
     * Delivery-only waits retain null semantics so publication can continue.
     *
     * <p>The availability read is the already-subscribed wait predicate; it
     * neither previews nor reserves capacity. Caller holds {@link #queueLock}.
     */
    public AdmissionBlock admissionBlockLocked() {
        checkState(queueLock.isHeldByCurrentThread(), "capacity block snapshot requires queueLock");
        BatcherCycleResult blocked = capacityBlockedHead;
        if (blocked == null
                || !prefillState.queueWaitCurrentLocked(blocked.request(), 0L, 0L,
                        true, now())
                || blocked.unavailable().availability().isAvailable()) {
            return null;
        }
        return new AdmissionBlock(
                blocked.request().requestId(),
                blocked.request().enqueueSeq(),
                blocked.unavailable().projectionSemantics());
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

    private void stopAfterUnexpectedLoopFailure(Throwable loopFailure) {
        try {
            Logger.error("WorkerBatcher[{}] stopped after an unexpected loop failure",
                    key, loopFailure);
        } catch (Throwable ignoredLoggingFailure) {
            // The exact stop transaction must still run.
        }
        Throwable cleanupFailure = stopAndDrain(
                loopFailure,
                false);
        if (cleanupFailure != null) {
            try {
                Logger.error("WorkerBatcher[{}] failure cleanup exposed invariants",
                        key, cleanupFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Cleanup is already complete.
            }
        }
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
                    item = prefillState.detachNextActiveForStop();
                } catch (Throwable claimFailure) {
                    cleanupFailure = Failures.append(cleanupFailure, claimFailure);
                    break;
                }
                if (item == null) {
                    break;
                }
                cleanupFailure = Failures.run(cleanupFailure, capacityAvailableSignal);
                try {
                    item.ctx().scheduler().onQueueOfferFailure(item, terminalFailure);
                    if (!acknowledgeStoppedItem(item)) {
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

    /** Acknowledge only the retained owner whose terminal callback returned. */
    private boolean acknowledgeStoppedItem(RequestRoute item) {
        queueLock.lock();
        try {
            return prefillState.acknowledgeStopTerminalLocked(item);
        } finally {
            queueLock.unlock();
        }
    }

    // ==================== endpoint queue operations ====================

    public boolean removeQueued(
            RequestRoute exactItem,
            String reason) {
        RequestRoute item = exactItem;
        requireExactEndpoint(item, "queued item");
        boolean removed;
        queueLock.lock();
        try {
            removed = runtimeState == RuntimeState.RUNNING
                    && !drainClaimed
                    && prefillState.removeQueuedLocked(item);
            if (removed) {
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
        if (removed) {
            capacityAvailableSignal.run();
            try {
                Logger.debug(
                        "[request-scheduler] exact queue remove: worker={} reason={} request_id={}",
                        key, reason, item.requestId());
            } catch (Throwable ignoredLoggingFailure) {
                // Diagnostics cannot turn a committed exact removal into failure.
            }
        }
        return removed;
    }

    /** A control producer only wakes this generation; the worker settles the exact context. */
    public boolean signalControl(RequestRoute exact) {
        requireExactEndpoint(exact, "control item");
        queueLock.lock();
        try {
            if (!drainClaimed && runtimeState == RuntimeState.RUNNING) {
                controlInbox.addLast(exact);
                stateChanged.signal();
                return true;
            }
            return false;
        } finally {
            queueLock.unlock();
        }
    }

    private void requireExactEndpoint(
            RequestRoute item, String operation) {
        checkArgument(item.prefillEp() == prefillEndpoint, "%s belongs to another Prefill generation", operation);
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
        latestQueueWaitSnapshot = new QueueWaitSnapshot(key, reason, now(), counters, decode);
    }

    // ==================== Queue ownership and projection ====================

    private static long now() {
        return System.currentTimeMillis();
    }

    /** Grouping semantics fixed for this endpoint generation. */
    public GroupingPolicy groupingPolicy() { return grouping; }

    /** Capture scheduling constraints while holding the shared Prefill ownership lock. */
    public GroupPlanner.Constraints projectionConstraintsLocked() {
        checkState(queueLock.isHeldByCurrentThread(), "projection constraints require queueLock");
        return schedulingConstraints(settings.maxRequests(), settings.executionBudgetMs(), settings.collectionWaitMs());
    }

    private GroupPlanner.Constraints schedulingConstraints(int maxRequests, long predictionBudgetMs, long windowMs) {
        long tokenCapacity = Long.MAX_VALUE;
        long kvCapacity = Long.MAX_VALUE;
        WorkerStatus status = prefillEndpoint.getStatus();
        if (status != null) {
            WorkerStatus.EngineObservation engine = status.committedEngineObservation();
            long configuredTokens = engine.maxBatchTokensSize() > 0L
                    ? engine.maxBatchTokensSize() : engine.maxSeqLen();
            if (configuredTokens > 0L) {
                tokenCapacity = configuredTokens;
            }
            if (engine.totalKvCacheTokens() > 0L) {
                kvCapacity = Math.clamp(engine.availableKvCacheTokens(), 0L, engine.totalKvCacheTokens());
            }
        }
        return new GroupPlanner.Constraints(maxRequests, tokenCapacity, kvCapacity, predictionBudgetMs, windowMs);
    }

    // ==================== Delivery ownership ====================

    private BatcherCycleResult admitAndDeliverCapacityFeasiblePrefix(
            List<RequestRoute> candidates,
            String decisionReason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedCommittedPredictionMs) {
        if (candidates.isEmpty() || !selectionStillOwned(candidates)) {
            return BatcherCycleResult.NO_ACTION;
        }
        try (DeliveryTransaction transaction =
                deliveryStrategy.prepare(
                        candidates, evaluator,
                        plannedCommittedPredictionMs)) {
            if (transaction.items().isEmpty()) {
                return commitBoundary(
                        transaction.blockedItem(),
                        transaction.blockedResult());
            }
            return commitPreparedSelection(transaction, decisionReason, evaluator);
        }
    }

    private void handoff(DeliveryTransaction transaction, String decisionReason,
                         int remainingQueueDepth, WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator) {
        Throwable failure = null;
        try {
            deliveryStrategy.deliver(transaction, decisionReason, remainingQueueDepth, precedingWork, evaluator);
        } catch (Throwable deliveryFailure) {
            failure = deliveryFailure;
        } finally {
            Throwable deliveryFailure = failure;
            failure = Failures.run(failure, () -> DeliveryStrategy.failUnsentDelivery(transaction, deliveryFailure, false));
        }
        if (failure != null) {
            Logger.error("WorkerBatcher[{}] committed delivery failed", key, failure);
        }
    }

    private boolean selectionStillOwned(List<RequestRoute> candidates) {
        queueLock.lock();
        try {
            return !drainClaimed && prefillState.ownsSelectionLocked(candidates, now());
        } finally {
            queueLock.unlock();
        }
    }

    private BatcherCycleResult commitPreparedSelection(DeliveryTransaction transaction,
                                                       String decisionReason, PrefillTimePredictor.Evaluator evaluator) {
        boolean removedBoundary;
        try {
            PrefillState.CommittedHandoff committed;
            queueLock.lock();
            try {
                long nowMs = now();
                if (drainClaimed) {
                    return BatcherCycleResult.NO_ACTION;
                }
                committed = transaction.commitSelectionLocked(nowMs);
                if (committed == null) { return BatcherCycleResult.NO_ACTION; }
                removedBoundary = committed.removedFailedMember();
            } finally {
                queueLock.unlock();
            }
            if (removedBoundary) {
                notifyTerminalAdmissionFailure(transaction.blockedItem(), transaction.blockedResult());
            }
            String reason = transaction.blockedResult() != null && transaction.blockedResult().unavailable()
                    ? "delivery_capacity_prefix" : decisionReason;
            Objects.requireNonNull(reason);
            // handoff always resolves or aborts the transaction, including its failure path.
            handoff(transaction, reason, committed.remainingQueueDepth(), committed.precedingWork().materialize(), evaluator);
        } catch (Throwable commitFailure) {
            Throwable failure = Failures.run(commitFailure, () -> DeliveryStrategy.failUnsentDelivery(transaction, commitFailure, false));
            throw Failures.propagate(failure, "delivery selection failed after ownership commit");
        }
        if (!removedBoundary) { prefillEndpoint.signalPlacementCapacityChanged(); }
        return BatcherCycleResult.NO_ACTION;
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
            long nowMs = now();
            if (capacityBlocked) {
                if (!prefillState.queueWaitCurrentLocked(blockedItem, 0L, 0L, true, nowMs)) {
                    return BatcherCycleResult.NO_ACTION;
                }
            } else if (failed) {
                removed = prefillState.removeQueuedIfUnexpiredLocked(blockedItem, nowMs);
            }
        } finally {
            queueLock.unlock();
        }
        if (capacityBlocked) { return BatcherCycleResult.capacityBlocked(blockedItem, blockedResult); }
        if (removed) { notifyTerminalAdmissionFailure(blockedItem, blockedResult); }
        return BatcherCycleResult.NO_ACTION;
    }

    private void notifyTerminalAdmissionFailure(RequestRoute item, CapacityBoundary boundary) {
        notifyQueueRemoval(item, () -> item.ctx().scheduler().failDeliveryPreparation(item, boundary.cause()));
    }

    /** Once removed, the request must be notified even if a capacity listener fails. */
    private void notifyQueueRemoval(RequestRoute item, Runnable terminal) {
        Throwable failure = Failures.run(null, capacityAvailableSignal);
        try {
            terminal.run();
        } catch (Throwable callbackFailure) {
            failure = Failures.run(failure, () -> Logger.error(
                    "WorkerBatcher[{}] removed request callback failed request_id={}",
                    key, item.requestId(), callbackFailure));
        }
        Failures.rethrow(failure, "queue removal notification failed");
    }

    private void dropHead(RequestRoute head) {
        Logger.debug("flexlb_{}_drop request_id={} reason=request_expired expires_at_ms={} now_ms={}",
                "queue", head.requestId(), head.expiresAtMs(), now());
        boolean removed;
        queueLock.lock();
        try {
            removed = prefillState.removeQueuedLocked(head);
        } finally {
            queueLock.unlock();
        }
        if (!removed) {
            return;
        }
        notifyQueueRemoval(head, () -> head.ctx().scheduler().onQueuedItemExpired(head));
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
        long nowMs = now();
        if (head.ctx().stage() == RequestContext.RequestStage.ROUTING) {
            return BatcherCycleResult.awaitingSchedulingChange(head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(), Long.MAX_VALUE, "Route commit in progress");
        }
        if (head.requestExpired(nowMs)) {
            dropHead(head);
            return BatcherCycleResult.NO_ACTION;
        }
        GroupPlanner.Constraints constraints = schedulingConstraints(maxRequests, predictionBudgetMs, windowMs);
        if (Math.max(0L, head.seqLen()) > constraints.batchKvCapacity()) {
            return BatcherCycleResult.awaitingSchedulingChange(head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(), head.expiresAtMs(), "Prefill KV capacity exhausted");
        }
        for (int index = 1; index < Math.min(maxRequests, snapshot.items().size()); index++) {
            RequestRoute member = snapshot.items().get(index);
            if (member.requestExpired(nowMs)) {
                dropHead(member);
                return BatcherCycleResult.NO_ACTION;
            }
        }
        try {
            PrefillTimePredictor.Evaluator evaluator = prefillEndpoint.getPredictor().evaluator();
            var predictor = predictionBudgetMs > 0L ? deliveryStrategy.newGroupPredictor(evaluator) : null;
            long planningAtMs = now();
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
            constraints = schedulingConstraints(maxRequests, predictionBudgetMs, windowMs);
            if ((selection.items().size() > 1 && !selection.fitsCompute(constraints.batchTokenCapacity()))
                    || !selection.fitsKv(constraints.batchKvCapacity())) {
                return BatcherCycleResult.NO_ACTION;
            }
            return admitAndDeliverCapacityFeasiblePrefix(selection.items(), reason, evaluator,
                    committedPrediction(selection));
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(head, CapacityBoundary.failed(failure));
        }
    }

    private static OptionalLong committedPrediction(
            GroupPlanner.Selection<RequestRoute> selection) {
        if (selection.selectedPredictionMs().isEmpty()) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(
                PrefillPredictionBoundary.committedDecisionGroupMs(
                        selection.selectedPredictionMs().getAsDouble()));
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        try {
            while (!drainClaimed && !Thread.currentThread().isInterrupted()) {
                try {
                    runOneCycle();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Throwable t) {
                    // Every expected delivery failure is terminalized inside the
                    // typed cycle. An escaping Throwable is therefore an invariant
                    // failure; retrying the same ACTIVE state can only spin.
                    stopAfterUnexpectedLoopFailure(t);
                    return;
                }
            }
        } finally {
            synchronized (this) {
                runtimeState = RuntimeState.STOPPED;
            }
        }
    }

    private void runOneCycle() throws InterruptedException {
        awaitWork(null);
        if (drainClaimed) {
            return;
        }

        RequestRoute control;
        queueLock.lock();
        try {
            control = controlInbox.pollFirst();
        } finally {
            queueLock.unlock();
        }
        if (control != null) {
            control.ctx().scheduler().onQueuedItemControl(control);
            return;
        }

        BatcherCycleResult result = processQueue();
        if (result.waitReason() != null) {
            recordQueueWait(result.request(), result.waitReason());
        }
        if (result.request() != null) {
            awaitWork(result);
        }
    }

    /**
     * All waits share the queue lock and control-inbox check. Capacity waits
     * subscribe before testing availability, so release-before-await is safe.
     * A null decision waits for the first item; other decisions retain their
     * exact head and either resource or queue/input-version predicate.
     */
    private void awaitWork(BatcherCycleResult waiting) throws InterruptedException {
        boolean capacityWait = waiting != null && waiting.capacityBlocked();
        queueLock.lockInterruptibly();
        try {
            if (capacityWait && !drainClaimed) {
                setCapacityBlockedHeadLocked(waiting);
            }
            while (!drainClaimed && controlInbox.isEmpty() && prefillState.queueWaitCurrentLocked(
                    waiting == null ? null : waiting.request(),
                    waiting == null ? 0L : waiting.queueVersion(),
                    waiting == null ? 0L : waiting.schedulingInputVersion(),
                    capacityWait, now())
                    && (!capacityWait || !waiting.unavailable().availability().isAvailable())) {
                if (waiting == null) {
                    setCapacityBlockedHeadLocked(null);
                }
                long wakeAtMs = waiting == null ? Long.MAX_VALUE
                        : capacityWait ? waiting.request().expiresAtMs() : waiting.wakeAtMs();
                long nowMs = now();
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
    public void signalDeliveryCapacityAvailable() {
        queueLock.lock();
        try {
            if (capacityBlockedHead != null) {
                prefillState.schedulingInputsChangedLocked();
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /** Own the blocked head, its listener and projection invalidation under queueLock. */
    private void setCapacityBlockedHeadLocked(
            BatcherCycleResult blocked) {
        checkState(queueLock.isHeldByCurrentThread(), "capacity block update requires queueLock");
        if (capacityBlockedHead == blocked) {
            return;
        }
        CapacityBoundary.Availability previousSource = capacityBlockedHead == null
                ? null : capacityBlockedHead.unavailable().availability();
        CapacityBoundary.Availability nextSource = blocked == null
                ? null : blocked.unavailable().availability();
        capacityBlockedHead = blocked;
        prefillState.schedulingInputsChangedLocked();
        if (previousSource != nextSource) {
            try {
                if (previousSource != null) {
                    previousSource.removeListener(capacityAvailableSignal);
                }
            } finally {
                if (nextSource != null) {
                    nextSource.addListener(capacityAvailableSignal);
                }
            }
        }
    }

    /** Wake decisions whose advisory worker-status or predictor input changed. */
    public void signalSchedulingInputsChanged() {
        queueLock.lock();
        try {
            prefillState.schedulingInputsChangedLocked();
            stateChanged.signal();
        } finally {
            queueLock.unlock();
        }
    }

}
