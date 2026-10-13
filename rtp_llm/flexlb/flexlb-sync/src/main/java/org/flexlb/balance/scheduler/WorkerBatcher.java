package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.util.PriorityOrdering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * Per-worker scheduling queue that owns queue state, grouping, hard-capacity
 * admission and delivery handoff.
 *
 * <p>One instance belongs to each Prefill worker. Exact requests enter this
 * concrete runtime and are grouped by the configured decision mode. A group
 * may independently be delivered through
 * EnqueueBatch or as individual route decisions.
 *
 * <p>The active queue is the single source of truth for queued requests.
 * PRIORITY ordering uses
 * the explicit priority comparator ({@link #PRIORITY_QUEUE_ORDER}); FIFO uses
 * a unique monotonic enqueue sequence. A monotonic mutation generation lets
 * snapshots and diagnostics identify queue-state changes.
 */
public final class WorkerBatcher {

    /** Immutable queue state materialized under this runtime's queue lock. */
    public record QueueSnapshot(
            String endpointId,
            long queueVersion,
            List<ScheduledRequest> items) {
        public QueueSnapshot {
            if (queueVersion < 0L) {
                throw new IllegalArgumentException(
                        "queueVersion must be non-negative");
            }
            items = List.copyOf(items);
        }
    }

    /** Frozen decision evidence; the log representation is materialized only on failure. */
    private static final class QueueWaitSnapshot {
        private final String endpoint;
        private final String cause;
        private final long capturedAtMs;
        private final long queueVersion;
        private final int[] requestCountsByPriority;
        private final long prefillRequests;
        private final int prefillBatchSlots;
        private final DecodeRoutingView decode;
        private volatile Map<String, Object> diagnostics;

        private QueueWaitSnapshot(String endpoint, String cause, long capturedAtMs, long queueVersion,
                                  int[] requestCountsByPriority, long prefillRequests, int prefillBatchSlots, DecodeRoutingView decode) {
            this.endpoint = endpoint;
            this.cause = cause;
            this.capturedAtMs = capturedAtMs;
            this.queueVersion = queueVersion;
            this.requestCountsByPriority = requestCountsByPriority;
            this.prefillRequests = prefillRequests;
            this.prefillBatchSlots = prefillBatchSlots;
            this.decode = decode;
        }

        private Map<String, Object> diagnostics() {
            Map<String, Object> cached = diagnostics;
            if (cached != null) { return cached; }
            Map<Integer, Integer> priorityCounts = new HashMap<>();
            int depth = 0;
            for (int priority = 0; priority < requestCountsByPriority.length; priority++) {
                if (requestCountsByPriority[priority] > 0) { priorityCounts.put(priority, requestCountsByPriority[priority]); }
                depth += requestCountsByPriority[priority];
            }
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("cause", cause);
            details.put("capturedAtMs", capturedAtMs);
            details.put("endpoint", endpoint);
            details.put("queueDepth", depth);
            details.put("queueVersion", queueVersion);
            details.put("priorityCounts", Collections.unmodifiableMap(priorityCounts));
            details.put("prefillRequests", prefillRequests);
            details.put("prefillBatchSlots", prefillBatchSlots);
            if (decode != null) {
                details.put("decode", Map.of("endpoint", decode.address(), "version", decode.admissionVersion(),
                        "engineLoad", decode.engineLoad(), "totalLoad", decode.totalLoad(),
                        "kvTotal", decode.totalKv(), "kvAvailable", decode.placementUsage().availableKvAfterReservations()));
            }
            cached = Collections.unmodifiableMap(details);
            diagnostics = cached;
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

    /** Exact wait predicate; windowCandidateCount is zero outside collection-window waits. */
    private record BatcherCycleResult(
            boolean placementCapacityChanged,
            ScheduledRequest request,
            CapacityBoundary unavailable,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs,
            int windowCandidateCount,
            String waitReason) {

        private static final BatcherCycleResult NO_ACTION = simple(false);
        private static final BatcherCycleResult CAPACITY_CHANGED = simple(true);

        private BatcherCycleResult {
            boolean waiting = request != null;
            if ((placementCapacityChanged && waiting)
                    || (!waiting && unavailable != null)) {
                throw new IllegalArgumentException(
                        "worker cycle requires either a capacity change or an exact wait predicate");
            }
        }

        private static BatcherCycleResult capacityBlocked(
                ScheduledRequest item,
                CapacityBoundary unavailable) {
            return new BatcherCycleResult(false,
                    Objects.requireNonNull(item, "item"),
                    Objects.requireNonNull(unavailable, "unavailable"),
                    0L, 0L, 0L, 0, unavailable.projectionSemantics() == null
                            ? "Prefill batch capacity exhausted" : unavailable.projectionSemantics().blockedDetail());
        }

        private static BatcherCycleResult awaitingSchedulingChange(
                ScheduledRequest head,
                long queueVersion,
                long schedulingInputVersion,
                long wakeAtMs,
                int windowCandidateCount,
                String waitReason) {
            return new BatcherCycleResult(false,
                    Objects.requireNonNull(head, "head"), null,
                    queueVersion, schedulingInputVersion, wakeAtMs, windowCandidateCount, waitReason);
        }

        private static BatcherCycleResult simple(boolean capacityChanged) {
            return new BatcherCycleResult(
                    capacityChanged, null, null, 0L, 0L, 0L, 0, null);
        }

        private boolean capacityBlocked() {
            return unavailable != null;
        }

        private boolean awaitingSchedulingChange() {
            return request != null && unavailable == null;
        }
    }

    /**
     * PRIORITY queue order: delegates to
     * {@link PriorityOrdering#STRICT} (priority desc → enqueue-seq asc for
     * same-priority FIFO) with {@code requestId} as the final deterministic
     * tie-break.
     *
     * <p>{@link #FIFO_QUEUE_ORDER} preserves enqueue order.
     */
    public static final Comparator<ScheduledRequest> PRIORITY_QUEUE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());

    /** FIFO order: unique monotonic enqueue sequence. */
    public static final Comparator<ScheduledRequest> FIFO_QUEUE_ORDER =
            Comparator.comparingLong(ScheduledRequest::enqueueSeq);

    private static final Comparator<GroupPlanner.Item> PRIORITY_PROJECTION_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_PROJECTION_ORDER =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparing(GroupPlanner.Item::requestId);
    private static final GroupPlanner.ItemAccess<ScheduledRequest>
            PLANNER_ITEM_ACCESS = new GroupPlanner.ItemAccess<>() {
                @Override
                public long enqueuedAtMs(ScheduledRequest item) {
                    return item.enqueuedAtMs();
                }

                @Override
                public long seqLen(ScheduledRequest item) {
                    return item.seqLen();
                }
            };
    private static final String SINGLE_DECISION_REASON = "single_request";
    /** Initial queue index allocation; unrelated to admission limits. */
    private static final int INITIAL_QUEUE_ALLOCATION = 16;
    private static final Map<String, Object> INITIAL_WAIT_DIAGNOSTICS =
            Map.of("cause", "waiting for Prefill decision");

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final EndpointEventProjector endpointEvents;
    private final FlexlbConfig config;
    private final DecisionPolicyConfig fixedWindowDecision;
    private final boolean singleDecision;
    private final boolean queueScheduling;
    private final DeliveryStrategy deliveryStrategy;
    private final PrefillActiveIndex activeIndex;
    private final Comparator<GroupPlanner.Item> projectionOrder;
    /**
     * Monotonic queue mutation generation, bumped on enqueue, removal,
     * delivery and drain. It is exposed in diagnostic snapshots.
     */
    private final AtomicLong queueVersion = new AtomicLong();
    /**
     * Generation of non-queue scheduling inputs: worker status, prediction and
     * delivery-capacity blocks. Invalidates both decisions and route projections.
     * Writes hold queueLock; volatile supports the lock-free cache check.
     * Updated only through schedulingInputsChangedUnderLock().
     */
    private volatile long schedulingInputVersion;
    /**
     * Immutable projection inputs reused while this endpoint's queue and
     * scheduling inputs are unchanged. Route decisions still evaluate every
     * live endpoint, but do not rebuild identical queue snapshots for each
     * request in the global planner.
     */
    private volatile ProjectionSource projectionSource;
    /**
     * Guards queue mutations and exact ownership publication.
     *
     * <p>The lock and generation stay active for both FIFO and PRIORITY so
     * every ordering mode has the same mutation guarantees.
     */
    private final ReentrantLock queueLock = new ReentrantLock();
    /** Canonical request ownership guarded exclusively by {@link #queueLock}. */
    private final PrefillState prefillState;
    /**
     * One per-worker condition for new queue work and endpoint-capacity release.
     * Every predicate transition and signal is serialized by queueLock, so a
     * release cannot race between the cap re-check and await.
     */
    private final Condition stateChanged = queueLock.newCondition();
    /** QUEUE-only serialized runtime; DIRECT owns no queue execution thread. */
    private final Thread workerThread;
    /** Constructed before this endpoint generation can begin retirement. */
    private final CancellationException normalStopFailure;
    /** Fixed diagnostic for an impossible exact stop acknowledgement loss. */
    private final IllegalStateException stopAcknowledgementFailure;
    private boolean terminationStarted;
    private boolean terminationFinished;
    private Thread terminationOwner;
    private volatile RuntimeState runtimeState = RuntimeState.NEW;
    private Throwable stopFailure;
    private volatile boolean stopped;
    private final Runnable capacityAvailableSignal =
            this::signalCapacityAvailable;
    /** Exact decision whose capacity, KV or collection-window predicate is being awaited. */
    private volatile BatcherCycleResult waitingDecision;

    /** Last waiting decision; timeout readers never acquire queueLock. */
    private volatile QueueWaitSnapshot waitSnapshot;

    public WorkerBatcher(
            String key,
            PrefillEndpoint prefillEp,
            FlexlbConfig config,
            DeliveryStrategy deliveryStrategy,
            EndpointEventProjector endpointEvents) {
        this(key, prefillEp, () -> config, deliveryStrategy, endpointEvents);
    }

    public WorkerBatcher(
            String key,
            PrefillEndpoint prefillEp,
            Supplier<FlexlbConfig> configSupplier,
            DeliveryStrategy deliveryStrategy,
            EndpointEventProjector endpointEvents) {
        this.configSupplier = Objects.requireNonNull(configSupplier, "configSupplier");
        FlexlbConfig config = Objects.requireNonNull(
                configSupplier.get(), "loadBalanceConfig");
        this.key = key;
        this.prefillEndpoint = prefillEp;
        this.config = config;
        this.queueScheduling = config.isQueue();
        this.singleDecision = config.isSingleDecision();
        DecisionPolicyConfig resolvedDecision = queueScheduling
                ? config.decisionPolicy() : null;
        this.fixedWindowDecision = resolvedDecision != null
                && resolvedDecision.getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW
                ? resolvedDecision : null;
        boolean priorityOrdering = config.isPriorityOrdering();
        this.endpointEvents = Objects.requireNonNull(
                endpointEvents, "endpointEvents");
        this.deliveryStrategy = Objects.requireNonNull(
                deliveryStrategy, "deliveryStrategy");
        Comparator<ScheduledRequest> queueOrder = priorityOrdering
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        this.projectionOrder =
                priorityOrdering
                        ? PRIORITY_PROJECTION_ORDER : FIFO_PROJECTION_ORDER;
        this.activeIndex = queueScheduling
                ? PrefillActiveIndex.ordered(INITIAL_QUEUE_ALLOCATION, queueOrder,
                        this::signalQueuedDecodeCapacityAvailable)
                : PrefillActiveIndex.disabled();
        this.prefillState = new PrefillState(queueLock, activeIndex, capacityAvailableSignal);
        this.normalStopFailure = new CancellationException(
                "FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure = new IllegalStateException(
                "FlexLB worker stop callback lost its exact retained owner: "
                        + key);
        // Waiting QUEUE runtimes use virtual threads so a large endpoint fleet
        // does not consume one OS thread per worker. DIRECT never constructs a
        // scheduler thread.
        this.workerThread = queueScheduling
                ? Thread.ofVirtual()
                        .name("flexlb-batcher-" + key)
                        .uncaughtExceptionHandler((thread, failure) -> Logger.error(
                                "WorkerBatcher[{}] thread died unexpectedly",
                                key, failure))
                        .unstarted(this::runLoop)
                : null;
    }

    public synchronized void start() {
        if (runtimeState != RuntimeState.NEW) {
            throw new IllegalStateException(
                    "Worker batcher cannot start from "
                            + runtimeState);
        }
        runtimeState = RuntimeState.STARTING;
        if (!queueScheduling) {
            // DIRECT uses only the shared Prefill ownership/projection ledger.
            // It has no admission queue and therefore owns no scheduler thread.
            runtimeState = RuntimeState.RUNNING;
            return;
        }
        try {
            Objects.requireNonNull(
                    workerThread, "QUEUE worker thread").start();
            runtimeState = RuntimeState.RUNNING;
        } catch (RuntimeException | Error startFailure) {
            Throwable cleanupFailure = stopAndDrain(
                    startFailure,
                    false);
            runtimeState = RuntimeState.STOPPED;
            notifyAll();
            addSuppressedNoFail(startFailure, cleanupFailure);
            throw startFailure;
        }
    }

    public boolean offer(ScheduledRequest exactItem) {
        if (!queueScheduling) {
            throw new IllegalStateException(
                    "DIRECT Prefill generation cannot accept queued work");
        }
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "incoming item");
        if (runtimeState != RuntimeState.RUNNING || stopped) {
            return false;
        }
        return enqueue(item);
    }

    private boolean enqueue(ScheduledRequest item) {
        queueLock.lock();
        try {
            if (stopped || !publishActiveIndexUnderLock(item)) {
                return false;
            }
            stateChanged.signal();
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    public int queueSize() {
        queueLock.lock();
        try {
            return activeIndex.size();
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Snapshot of the current active queue depth bucketed by normalized
     * scheduling priority. Only priorities present in the queue appear in the
     * result, matching the tagged queue-depth metric behavior.
     */
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        queueLock.lock();
        try {
            for (int priority = 0; priority <= PriorityNormalizer.MAX_PRIORITY; priority++) {
                int count = activeIndex.size(priority);
                if (count > 0) { sizeByPriority.put(priority, count); }
            }
        } finally {
            queueLock.unlock();
        }
        return sizeByPriority;
    }

    public PrefillState ownedState() {
        return prefillState;
    }

    /** Immutable delivery semantics used by a pure route projection. */
    public RouteProjection.DeliveryProjection deliveryProjection() {
        return deliveryStrategy.projectionPolicy();
    }

    /**
     * Anchor the current worker wait to the canonical queue head. The waiting
     * request may be an eligible successor after Decode bypass.
     * Delivery-only waits retain null semantics so publication can continue.
     *
     * <p>The availability read is the already-subscribed wait predicate; it
     * neither previews nor reserves capacity. Caller holds {@link #queueLock}.
     */
    private AdmissionBlock admissionBlockUnderLock() {
        if (!queueLock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "capacity block snapshot requires queueLock");
        }
        BatcherCycleResult blocked = waitingDecision;
        if (blocked == null
                || !blocked.capacityBlocked()
                || activeIndex.isEmpty()
                || dispatchHeadUnderLock() != blocked.request()
                || blocked.request().requestExpired(now())
                || blocked.unavailable().availability().isAvailable()) {
            return null;
        }
        return new AdmissionBlock(
                activeIndex.peek().requestId(),
                activeIndex.peek().enqueueSeq(),
                blocked.unavailable().projectionSemantics());
    }

    public Throwable stopAndAwait() {
        if (workerThread != null && Thread.currentThread() == workerThread) {
            throw new IllegalStateException(
                    "Prefill runtime cannot await its own worker thread");
        }
        Throwable failure = stopAndDrain(
                normalStopFailure,
                true);
        boolean interrupted = false;
        if (workerThread != null) {
            while (true) {
                try {
                    workerThread.join();
                    break;
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }
        synchronized (this) {
            runtimeState = RuntimeState.STOPPED;
            notifyAll();
            failure = stopFailure;
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
        boolean interrupted = false;
        synchronized (this) {
            stopped = true;
            if (terminationStarted) {
                if (!terminationFinished
                        && terminationOwner == Thread.currentThread()) {
                    throw new IllegalStateException(
                            "Prefill runtime cannot await its active stop transaction");
                }
                while (!terminationFinished) {
                    try {
                        wait();
                    } catch (InterruptedException interruption) {
                        interrupted = true;
                    }
                }
                Throwable completedFailure = stopFailure;
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                return completedFailure;
            }
            terminationStarted = true;
            terminationOwner = Thread.currentThread();
            runtimeState = runtimeState == RuntimeState.NEW
                    ? RuntimeState.STOPPED : RuntimeState.STOPPING;
        }

        Throwable cleanupFailure = null;
        try {
            boolean queueLocked = false;
            try {
                queueLock.lock();
                queueLocked = true;
                stopAcceptingUnderLock();
                try {
                    setWaitingDecisionUnderLock(null);
                } catch (Throwable subscriptionFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, subscriptionFailure);
                }
                stateChanged.signalAll();
            } catch (Throwable wakeFailure) {
                cleanupFailure = appendCleanupFailure(
                        cleanupFailure, wakeFailure);
            } finally {
                if (queueLocked) {
                    try {
                        queueLock.unlock();
                    } catch (Throwable unlockFailure) {
                        cleanupFailure = appendCleanupFailure(
                                cleanupFailure, unlockFailure);
                    }
                }
            }

            if (interruptWorker && workerThread != null
                    && Thread.currentThread() != workerThread) {
                try {
                    workerThread.interrupt();
                } catch (Throwable interruptFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, interruptFailure);
                }
            }

            while (true) {
                ScheduledRequest item;
                try {
                    item = detachNextStoppedItem();
                } catch (Throwable claimFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, claimFailure);
                    break;
                }
                if (item == null) {
                    break;
                }
                try {
                    endpointEvents.onQueueOfferFailure(
                            item, terminalFailure);
                } catch (Throwable callbackFailure) {
                    try {
                        Logger.error("WorkerBatcher[{}] shutdown callback failed request_id={}",
                                key, item.requestId(), callbackFailure);
                    } catch (Throwable ignoredLoggingFailure) {
                        // Cleanup ownership cannot depend on diagnostics.
                    }
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, callbackFailure);
                    // Do not acknowledge: generation retirement still owns
                    // this exact pending identity and will project it again.
                    continue;
                }
                try {
                    if (!acknowledgeStoppedItem(item)) {
                        cleanupFailure = appendCleanupFailure(
                                cleanupFailure,
                                stopAcknowledgementFailure);
                    }
                } catch (Throwable acknowledgementFailure) {
                    // The terminal fact was already delivered. If the exact
                    // entry remains, generation retirement converges it as a
                    // stale/idempotent replay.
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, acknowledgementFailure);
                }
            }
        } catch (Throwable unexpectedCleanupFailure) {
            cleanupFailure = appendCleanupFailure(
                    cleanupFailure, unexpectedCleanupFailure);
        } finally {
            synchronized (this) {
                stopFailure = cleanupFailure;
                terminationFinished = true;
                terminationOwner = null;
                notifyAll();
            }
        }
        return cleanupFailure;
    }

    /**
     * Detach one exact ACTIVE queue index after the stop gate is closed while
     * retaining its canonical entry. The callback result decides whether that
     * entry is acknowledged or carried into generation retirement.
     */
    private ScheduledRequest detachNextStoppedItem() {
        ScheduledRequest item;
        queueLock.lock();
        try {
            item = detachNextStopTerminalUnderLock();
            stateChanged.signalAll();
        } finally {
            queueLock.unlock();
        }
        return item;
    }

    /** Acknowledge only the retained owner whose terminal callback returned. */
    private boolean acknowledgeStoppedItem(ScheduledRequest item) {
        queueLock.lock();
        try {
            return acknowledgeStopTerminalUnderLock(item);
        } finally {
            queueLock.unlock();
        }
    }

    private static Throwable appendCleanupFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        // Preserve cleanup totality under allocation failure. The first causal
        // failure is sufficient; later leaves were still all attempted.
        return first;
    }

    private static void addSuppressedNoFail(
            Throwable primary,
            Throwable leaf) {
        if (primary == null || leaf == null || primary == leaf) {
            return;
        }
        try {
            primary.addSuppressed(leaf);
        } catch (Throwable ignoredAggregationFailure) {
            // The start transaction already drained every acquired leaf.
        }
    }

    // ==================== endpoint queue operations ====================

    public boolean removeQueued(
            ScheduledRequest exactItem,
            String reason) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "queued item");
        boolean removed;
        queueLock.lock();
        try {
            removed = runtimeState == RuntimeState.RUNNING
                    && !stopped
                    && removeUnderLock(item);
            if (removed) {
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
        if (removed) {
            signalCapacityAvailable();
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

    private void requireExactEndpoint(
            ScheduledRequest item, String operation) {
        if (item.prefillEp() != prefillEndpoint) {
            throw new IllegalArgumentException(
                    operation + " belongs to another Prefill generation");
        }
    }

    public Map<String, Object> waitDiagnostics() {
        QueueWaitSnapshot snapshot = waitSnapshot;
        return snapshot == null ? INITIAL_WAIT_DIAGNOSTICS : snapshot.diagnostics();
    }

    /** Capture fixed-size counters without formatting the PV record on the scheduling loop. */
    private void recordQueueWait(ScheduledRequest head, String reason) {
        int[] requestCountsByPriority = new int[PriorityNormalizer.MAX_PRIORITY + 1];
        long version;
        long prefillRequests;
        int prefillBatchSlots;
        queueLock.lock();
        try {
            version = queueVersion.get();
            prefillRequests = prefillState.observedRequestCount();
            prefillBatchSlots = prefillState.batchLeasesInUseUnderLock();
            for (int priority = 0; priority < requestCountsByPriority.length; priority++) {
                requestCountsByPriority[priority] = activeIndex.size(priority);
            }
        } finally {
            queueLock.unlock();
        }
        DecodeRoutingView decode = head.decodeEp() == null ? null : head.decodeEp().routingView();
        waitSnapshot = new QueueWaitSnapshot(key, reason, now(), version, requestCountsByPriority,
                prefillRequests, prefillBatchSlots, decode);
    }

    public QueueSnapshot captureQueueSnapshot() {
        queueLock.lock();
        try {
            return new QueueSnapshot(
                    key,
                    queueVersion.get(),
                    activeItemsInSchedulingOrder());
        } finally {
            queueLock.unlock();
        }
    }

    // ==================== Queue ownership and projection ====================

    private int maxDecisionRequests() {
        return fixedWindowDecision == null ? 1 : fixedWindowDecision.resolveMaxRequests();
    }

    private long collectionWindowMs() {
        return fixedWindowDecision == null
                ? 0L : Math.max(0L,
                configSupplier.get().decisionPolicy().getMaxCollectionWaitMs());
    }

    private long predictedExecutionBudgetMs() {
        if (fixedWindowDecision == null) {
            return 0L;
        }
        Long configured = fixedWindowDecision.getMaxPredictedExecutionMs();
        return configured == null ? 0L : configured;
    }

    private static long now() {
        return System.currentTimeMillis();
    }

    /** Caller holds {@link #queueLock}. */
    private boolean publishActiveIndexUnderLock(ScheduledRequest item) {
        boolean published = prefillState.enqueueActiveUnderLock(item, 0L);
        if (published) {
            queueVersion.incrementAndGet();
        }
        return published;
    }

    /** Caller holds {@link #queueLock}. */
    private void stopAcceptingUnderLock() {
        if (!queueLock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "queue mutation requires queueLock");
        }
        stopped = true;
    }

    /** Caller holds {@link #queueLock}. */
    private ScheduledRequest detachNextStopTerminalUnderLock() {
        ScheduledRequest item =
                prefillState.detachNextActiveForStopUnderLock();
        if (item != null) {
            queueVersion.incrementAndGet();
        }
        return item;
    }

    /** Caller holds {@link #queueLock}. */
    private boolean acknowledgeStopTerminalUnderLock(
            ScheduledRequest item) {
        return prefillState.acknowledgeStopTerminalUnderLock(item);
    }

    /** Caller holds {@link #queueLock}. */
    private boolean removeUnderLock(ScheduledRequest item) {
        boolean removed = prefillState.terminalizeActiveUnderLock(item);
        if (removed) {
            queueVersion.incrementAndGet();
        }
        return removed;
    }

    private List<ScheduledRequest> activeItemsInSchedulingOrder() {
        return activeIndex.capture().items();
    }

    private ActiveQueueSnapshot snapshotActiveQueueHead() {
        queueLock.lock();
        try {
            ScheduledRequest head = dispatchHeadUnderLock();
            return new ActiveQueueSnapshot(
                    queueVersion.get(), schedulingInputVersion,
                    head == null ? List.of() : List.of(head));
        } finally {
            queueLock.unlock();
        }
    }

    private ActiveQueueSnapshot snapshotActiveQueue() {
        long version;
        long inputVersion;
        List<ScheduledRequest> items;
        queueLock.lock();
        try {
            version = queueVersion.get();
            inputVersion = schedulingInputVersion;
            if (activeIndex.isEmpty()) {
                return new ActiveQueueSnapshot(
                        version, inputVersion, List.of());
            }
            items = dispatchItemsUnderLock();
        } finally {
            queueLock.unlock();
        }
        return new ActiveQueueSnapshot(version, inputVersion, items);
    }

    /**
     * Preserve the canonical order. Decode-unready predecessors establish a
     * priority barrier. Keep looking past requests held by that barrier so a
     * later equal/higher-priority ready request can still run.
     * The common path reuses the membership snapshot; bypass creates only a
     * subList view, never a filtered copy or a second queue.
     */
    private List<ScheduledRequest> dispatchItemsUnderLock() {
        List<ScheduledRequest> items = activeIndex.capture().items();
        int priorityBarrier = -1;
        long nowMs = now();
        for (int start = 0; start < items.size(); start++) {
            ScheduledRequest item = items.get(start);
            if (item.priority() >= priorityBarrier
                    && (item.requestExpired(nowMs) || item.decodeReadyForDispatch())) {
                if (start == 0) { return items; }
                int end = start + 1;
                int limit = start + Math.min(maxDecisionRequests(), items.size() - start);
                while (end < limit && items.get(end).priority() >= priorityBarrier
                        && items.get(end).decodeReadyForDispatch()) {
                    end++;
                }
                return items.subList(start, end);
            }
            priorityBarrier = Math.max(priorityBarrier, item.priority());
        }
        // No eligible successor: attempt the original head to obtain its
        // exact capacity wait predicate (or terminal ownership failure).
        return items;
    }

    private ScheduledRequest dispatchHeadUnderLock() {
        ScheduledRequest head = activeIndex.peek();
        if (head == null || head.requestExpired(now()) || head.decodeReadyForDispatch()) { return head; }
        // SINGLE and wait predicates need only one identity, not a full snapshot.
        int priorityBarrier = head.priority();
        long nowMs = now();
        for (ScheduledRequest item : activeIndex) {
            if (item == head) { continue; }
            if (item.priority() >= priorityBarrier
                    && (item.requestExpired(nowMs) || item.decodeReadyForDispatch())) {
                return item;
            }
            priorityBarrier = Math.max(priorityBarrier, item.priority());
        }
        return head;
    }

    /** Recheck head and window inputs after the caller has validated queue/input versions. */
    private boolean waitingSelectionUnchangedUnderLock(BatcherCycleResult waiting) {
        ScheduledRequest head = activeIndex.peek();
        if (head == null) { return false; }
        int expectedCount = waiting.windowCandidateCount();
        int limit = maxDecisionRequests();
        long nowMs = now();
        if (head.requestExpired(nowMs) || head.decodeReadyForDispatch()) {
            return head == waiting.request()
                    && (expectedCount == 0 || Math.min(activeIndex.size(), limit) == expectedCount);
        }
        int priorityBarrier = head.priority();
        Iterator<ScheduledRequest> ordered = activeIndex.iterator();
        ordered.next();
        while (ordered.hasNext()) {
            ScheduledRequest item = ordered.next();
            if (item.priority() >= priorityBarrier
                    && (item.requestExpired(nowMs) || item.decodeReadyForDispatch())) {
                if (item != waiting.request()) { return false; }
                if (expectedCount == 0) { return true; }
                int count = 1;
                while (count < limit && ordered.hasNext()) {
                    ScheduledRequest next = ordered.next();
                    if (next.priority() < priorityBarrier || !next.decodeReadyForDispatch()) { break; }
                    count++;
                }
                return count == expectedCount;
            }
            priorityBarrier = Math.max(priorityBarrier, item.priority());
        }
        // As in dispatchItemsUnderLock, an unready queue falls back to its raw head.
        return head == waiting.request()
                && (expectedCount == 0 || Math.min(activeIndex.size(), limit) == expectedCount);
    }

    private boolean isDispatchPrefixUnderLock(List<ScheduledRequest> selected) {
        if (activeIndex.peek() == selected.getFirst()) { return true; }
        List<ScheduledRequest> eligible = dispatchItemsUnderLock();
        if (eligible.size() < selected.size()) { return false; }
        for (int i = 0; i < selected.size(); i++) {
            if (eligible.get(i) != selected.get(i)) { return false; }
        }
        return true;
    }

    private record ActiveQueueSnapshot(
            long queueVersion,
            long schedulingInputVersion,
            List<ScheduledRequest> items) {
        ScheduledRequest head() {
            return items.isEmpty() ? null : items.get(0);
        }
    }

    private record ProjectionVersion(
            long queue,
            long schedulingInputs,
            long ownership,
            long collectionWindowMs) {
    }

    /** Captured under queueLock; the shared result is built without that lock. */
    private final class ProjectionSource {
        private final ProjectionVersion version;
        private PrefillState.Snapshot ownership;
        private final GroupPlanner.Constraints constraints;
        private final AdmissionBlock admissionBlock;
        private volatile RouteProjection.Inputs materialized;

        private ProjectionSource(ProjectionVersion version, PrefillState.Snapshot ownership,
                                 GroupPlanner.Constraints constraints, AdmissionBlock admissionBlock) {
            this.version = version;
            this.ownership = ownership;
            this.constraints = constraints;
            this.admissionBlock = admissionBlock;
        }

        private RouteProjection.Inputs materialize() {
            RouteProjection.Inputs result = materialized;
            if (result != null) {
                return result;
            }
            synchronized (this) {
                if (materialized == null) {
                    var queueSnapshot = new org.flexlb.balance.projection.QueueSnapshot(
                            ownership.capturedAtMs(), queueScheduling, projectionOrder,
                            constraints, ownership.active().projectedItems(), admissionBlock);
                    materialized = new RouteProjection.Inputs(
                            queueSnapshot, ownership.work().materialize(), version.ownership());
                    // The cached projection must not retain completed request contexts.
                    ownership = null;
                }
                return materialized;
            }
        }
    }

    public RouteProjection.Inputs captureRouteProjectionInputs() {
        ProjectionSource source = projectionSource;
        if (source == null || !isCurrentProjection(source.version)) {
            queueLock.lock();
            try {
                source = projectionSource;
                if (source == null || !isCurrentProjection(source.version)) {
                    source = captureProjectionSourceUnderLock();
                    projectionSource = source;
                }
            } finally {
                queueLock.unlock();
            }
        }
        // Concurrent callers share one source per version. A late build only
        // fills its own source; it can never overwrite a newer capture.
        return source.materialize();
    }

    private boolean isCurrentProjection(ProjectionVersion version) {
        return version.queue() == queueVersion.get()
                && version.schedulingInputs() == schedulingInputVersion
                && version.ownership() == prefillState.mutationVersion()
                && version.collectionWindowMs() == collectionWindowMs();
    }

    /** Caller holds queueLock. */
    private ProjectionSource captureProjectionSourceUnderLock() {
        long fixedWaitMs = collectionWindowMs();
        ProjectionVersion version = new ProjectionVersion(
                queueVersion.get(), schedulingInputVersion,
                prefillState.mutationVersionUnderLock(), fixedWaitMs);
        BatchCapacitySnapshot capacity = batchCapacitySnapshot();
        PrefillState.Snapshot ownership = prefillState.snapshotUnderLock();
        return new ProjectionSource(
                version, ownership,
                new GroupPlanner.Constraints(
                        maxDecisionRequests(), capacity.batchTokenCapacity(),
                        capacity.batchKvCapacity(), predictedExecutionBudgetMs(),
                        fixedWaitMs),
                ownership.activeItems().isEmpty() ? null : admissionBlockUnderLock());
    }

    private BatchCapacitySnapshot batchCapacitySnapshot() {
        WorkerStatus status = prefillEndpoint != null
                ? prefillEndpoint.getStatus() : null;
        if (status == null) {
            return new BatchCapacitySnapshot(Long.MAX_VALUE, Long.MAX_VALUE);
        }
        WorkerStatus.EngineObservation engineStatus =
                status.committedEngineObservation();
        long engineCapacity = engineStatus.maxBatchTokensSize();
        if (engineCapacity <= 0) {
            engineCapacity = engineStatus.maxSeqLen();
        }
        if (engineCapacity <= 0) {
            engineCapacity = config.getFallbackBatchTokenCapacity();
        }
        long batchTokenCapacity = positiveOrUnlimited(engineCapacity);
        long total = engineStatus.totalKvCacheTokens();
        if (total <= 0) {
            return new BatchCapacitySnapshot(
                    batchTokenCapacity, Long.MAX_VALUE);
        }
        long available = Math.max(
                0, engineStatus.availableKvCacheTokens());
        return new BatchCapacitySnapshot(
                batchTokenCapacity, Math.min(total, available));
    }

    private record BatchCapacitySnapshot(
            long batchTokenCapacity,
            long batchKvCapacity) {
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }

    // ==================== Delivery ownership ====================

    private BatcherCycleResult admitAndDeliverCapacityFeasiblePrefix(
            List<ScheduledRequest> candidates,
            String decisionReason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedCommittedPredictionMs) {
        if (candidates.isEmpty() || !selectionStillOwned(candidates)) {
            return BatcherCycleResult.NO_ACTION;
        }
        try (DeliveryStrategy.Transaction transaction =
                deliveryStrategy.prepare(
                        candidates, evaluator,
                        plannedCommittedPredictionMs)) {
            if (transaction.items().isEmpty()) {
                return commitBoundary(
                        transaction.blockedItem(),
                        transaction.blockedResult());
            }
            BatcherCycleResult admitted = commitPreparedSelection(
                    transaction, decisionReason);
            return admitted == null
                    ? BatcherCycleResult.NO_ACTION : admitted;
        }
    }

    private void handoff(
            DeliveryStrategy.Transaction transaction,
            String decisionReason,
            int remainingQueueDepth,
            WorkSnapshot precedingWork) {
        Throwable deliveryFailure = null;
        try {
            String groupId = java.util.UUID.randomUUID().toString();
            long committedAtMs = System.currentTimeMillis();
            for (ScheduledRequest item : transaction.items()) {
                var context = item.ctx();
                context.setDecisionGroup(new org.flexlb.dao.pv.DecisionGroup(
                        groupId,
                        context.getConfig().decisionPolicy().getType().name(),
                        context.getConfig().getDispatcher().getType().name(),
                        item.prefillEp().getStatus().getMetricIpPort(),
                        transaction.items().size(),
                        decisionReason,
                        committedAtMs,
                        Math.max(0L, committedAtMs - item.enqueuedAtMs())));
            }
            transaction.handoff(decisionReason, remainingQueueDepth, precedingWork);
        } catch (Throwable failure) {
            deliveryFailure = failure;
        }
        Throwable unresolved = deliveryFailure != null
                ? deliveryFailure
                : new IllegalStateException(
                        "delivery returned without resolving owner");
        try {
            transaction.abort(unresolved);
        } catch (Throwable cleanupFailure) {
            if (deliveryFailure == null) {
                deliveryFailure = cleanupFailure;
            } else if (deliveryFailure != cleanupFailure) {
                deliveryFailure.addSuppressed(cleanupFailure);
            }
        }
        if (deliveryFailure != null) {
            Logger.error("WorkerBatcher[{}] committed delivery failed",
                    key, deliveryFailure);
        }
    }

    private BatcherCycleResult runPredictionBound(
            ScheduledRequest exactHead,
            Supplier<BatcherCycleResult> operation) {
        try {
            return operation.get();
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(
                    exactHead, CapacityBoundary.failed(failure));
        }
    }

    private boolean selectionStillOwned(
            List<ScheduledRequest> candidates) {
        queueLock.lock();
        try {
            if (stopped) {
                return false;
            }
            long nowMs = now();
            for (ScheduledRequest item : candidates) {
                if (!containsIdentity(item)
                        || item.requestExpired(nowMs)) {
                    return false;
                }
            }
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    private BatcherCycleResult commitPreparedSelection(
            DeliveryStrategy.Transaction transaction,
            String decisionReason) {
        List<ScheduledRequest> items = transaction.items();
        if (items.isEmpty()) {
            throw new IllegalStateException(
                    "prepared selection commit requires a non-empty selection");
        }
        String committedReason = null;
        int remainingQueueDepth = 0;
        boolean removedTerminalBoundary = false;
        Throwable postCommitFailure = null;
        WorkSnapshot precedingWork;
        queueLock.lock();
        try {
            if (stopped || !isDispatchPrefixUnderLock(items)) {
                return null;
            }
            long nowMs = now();
            for (ScheduledRequest item : items) {
                if (!containsIdentity(item)
                        || item.requestExpired(nowMs)) {
                    return null;
                }
            }
            precedingWork = transaction.commitUnderLock();
            try {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        transaction.blockedItem(),
                        transaction.blockedResult(), nowMs);
                queueVersion.incrementAndGet();
                committedReason = transaction.blockedResult() != null
                        && transaction.blockedResult().unavailable()
                        ? "delivery_capacity_prefix" : decisionReason;
                remainingQueueDepth = activeIndex.size();
            } catch (Throwable failure) {
                postCommitFailure = failure;
            }
        } finally {
            queueLock.unlock();
        }

        if (postCommitFailure != null) {
            Throwable failure = postCommitFailure;
            if (removedTerminalBoundary) {
                signalCapacityAvailable();
            }
            try {
                notifyTerminalAdmissionFailure(
                        removedTerminalBoundary,
                        transaction.blockedItem(),
                        transaction.blockedResult());
            } catch (Throwable notificationFailure) {
                failure = appendFailure(failure, notificationFailure);
            }
            try {
                transaction.abort(failure);
            } catch (Throwable ownerFailure) {
                failure = appendFailure(failure, ownerFailure);
            }
            throw propagateCommitFailure(failure);
        }
        if (removedTerminalBoundary) {
            signalCapacityAvailable();
        }
        notifyTerminalAdmissionFailure(
                removedTerminalBoundary,
                transaction.blockedItem(),
                transaction.blockedResult());
        handoff(transaction, Objects.requireNonNull(committedReason),
                remainingQueueDepth, precedingWork);
        return BatcherCycleResult.CAPACITY_CHANGED;
    }

    private BatcherCycleResult commitBoundary(
            ScheduledRequest blockedItem,
            CapacityBoundary blockedResult) {
        BatcherCycleResult result = BatcherCycleResult.NO_ACTION;
        boolean removedTerminalBoundary = false;
        queueLock.lock();
        try {
            if (stopped) {
                return result;
            }
            long nowMs = now();
            if (blockedResult != null && blockedResult.unavailable()) {
                result = resolveEmptyCapacityUnderLock(
                        blockedItem, blockedResult, nowMs);
            } else {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        blockedItem, blockedResult, nowMs);
                if (removedTerminalBoundary) {
                    queueVersion.incrementAndGet();
                    result = BatcherCycleResult.CAPACITY_CHANGED;
                }
            }
        } finally {
            queueLock.unlock();
        }
        if (removedTerminalBoundary) {
            signalCapacityAvailable();
        }
        notifyTerminalAdmissionFailure(
                removedTerminalBoundary, blockedItem, blockedResult);
        return result;
    }

    /** Caller holds {@link #queueLock}. */
    private BatcherCycleResult resolveEmptyCapacityUnderLock(
            ScheduledRequest item,
            CapacityBoundary unavailable,
            long nowMs) {
        if (dispatchHeadUnderLock() != item
                || item.requestExpired(nowMs)
                || !containsIdentity(item)) {
            return BatcherCycleResult.NO_ACTION;
        }
        return BatcherCycleResult.capacityBlocked(item, unavailable);
    }

    /** Caller holds {@link #queueLock}. */
    private boolean removeSelectionBoundaryUnderLock(
            ScheduledRequest blockedItem,
            CapacityBoundary blockedResult,
            long nowMs) {
        if (blockedItem == null
                || blockedResult.unavailable()
                || blockedResult == CapacityBoundary.OWNERSHIP_LOST) {
            return false;
        }
        if (!containsIdentity(blockedItem)
                || blockedItem.requestExpired(nowMs)
                || !prefillState.terminalizeActiveUnderLock(blockedItem)) {
            return false;
        }
        return true;
    }

    private void notifyTerminalAdmissionFailure(
            boolean removed,
            ScheduledRequest item,
            CapacityBoundary boundary) {
        if (removed && boundary.status() == CapacityBoundary.Status.FAILED) {
            notifyAdmissionFailure(item, boundary.cause());
        }
    }

    private boolean containsIdentity(ScheduledRequest expected) {
        for (ScheduledRequest item : activeIndex) {
            if (item == expected) {
                return true;
            }
        }
        return false;
    }

    private void notifyAdmissionFailure(
            ScheduledRequest item,
            Throwable cause) {
        try {
            endpointEvents.onPreparedDeliveryFailure(item, cause);
        } catch (Throwable callbackFailure) {
            Logger.error("WorkerBatcher[{}] delivery-failure callback failed "
                            + "request_id={}",
                    key, item.requestId(), callbackFailure);
        }
    }

    private void dropHead(ScheduledRequest head) {
        boolean removed;
        queueLock.lock();
        try {
            removed = removeUnderLock(head);
        } finally {
            queueLock.unlock();
        }
        if (!removed) {
            return;
        }
        signalCapacityAvailable();
        try {
            endpointEvents.onQueuedItemExpired(head);
        } catch (Throwable callbackFailure) {
            Logger.error("WorkerBatcher[{}] ACTIVE terminal callback failed "
                            + "request_id={} reason={}",
                    key, head.requestId(), "request_expired",
                    callbackFailure);
        }
    }

    private static Throwable appendFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static RuntimeException propagateCommitFailure(
            Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "delivery selection failed after ownership commit", failure);
    }

    // ==================== Group decisions ====================

    private BatcherCycleResult processQueue() {
        return singleDecision
                ? processSingleRequest()
                : processFixedWindow();
    }

    private BatcherCycleResult processSingleRequest() {
        ActiveQueueSnapshot snapshot = snapshotActiveQueueHead();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        long nowMs = now();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_single_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            dropHead(head);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }

        GroupPlanner.Shape shape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        BatchCapacitySnapshot capacity = batchCapacitySnapshot();
        if (!shape.fitsKv(capacity.batchKvCapacity())) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        // Worker status and delivery ledgers advance independently. Re-read
        // the advisory gate immediately before hard-capacity admission.
        capacity = batchCapacitySnapshot();
        if (!shape.fitsKv(capacity.batchKvCapacity())) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        return runPredictionBound(head, () -> {
            PrefillTimePredictor predictor =
                    prefillEndpoint.getPredictor();
            PrefillTimePredictor.Evaluator evaluator =
                    predictor.evaluator();
            return admitAndDeliverCapacityFeasiblePrefix(
                    List.of(head), SINGLE_DECISION_REASON,
                    evaluator, OptionalLong.empty());
        });
    }

    private BatcherCycleResult processFixedWindow() {
        if (activeIndex.isEmpty()) {
            return BatcherCycleResult.NO_ACTION;
        }

        // Cheap advisory gates avoid sorting while the worker cannot dispatch.
        ScheduledRequest observedHead;
        int observedSize;
        long observedOldestEnqueuedAtMs = Long.MAX_VALUE;
        long observedQueueVersion;
        long observedSchedulingInputVersion;
        queueLock.lock();
        try {
            observedHead = activeIndex.peek();
            List<ScheduledRequest> eligible = observedHead != null
                    && !observedHead.requestExpired(now()) && !observedHead.decodeReadyForDispatch()
                    ? dispatchItemsUnderLock() : null;
            if (eligible != null) { observedHead = eligible.getFirst(); }
            observedSize = eligible == null ? activeIndex.size() : eligible.size();
            observedQueueVersion = queueVersion.get();
            observedSchedulingInputVersion = schedulingInputVersion;
            for (ScheduledRequest item : eligible == null ? activeIndex : eligible) {
                observedOldestEnqueuedAtMs = Math.min(
                        observedOldestEnqueuedAtMs, item.enqueuedAtMs());
            }
        } finally {
            queueLock.unlock();
        }
        if (observedHead == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        long nowMs = now();
        long fixedWaitMs = collectionWindowMs();
        BatchCapacitySnapshot capacity = batchCapacitySnapshot();
        int batchMaxCount = maxDecisionRequests();
        long predictThresholdMs = predictedExecutionBudgetMs();
        long batchMaxTokens = capacity.batchTokenCapacity();

        if (observedHead.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    observedHead.requestId(),
                    observedHead.expiresAtMs(), nowMs);
            dropHead(observedHead);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }

        boolean fullCandidate = observedSize >= batchMaxCount;
        if (predictThresholdMs <= 0 && !fullCandidate
                && !GroupPlanner.windowElapsed(
                observedOldestEnqueuedAtMs, nowMs, fixedWaitMs)) {
            return awaitCollectionWindow(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion,
                    observedOldestEnqueuedAtMs, fixedWaitMs,
                    Math.min(observedSize, batchMaxCount));
        }

        long batchKvTokens = capacity.batchKvCapacity();
        if (!GroupPlanner.Shape.empty().add(observedHead.seqLen())
                .fitsKv(batchKvTokens)) {
            return awaitPrefillKvCapacity(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion);
        }

        // The ordered snapshot is the selection linearization point;
        // prediction intentionally runs after releasing queueLock.
        ActiveQueueSnapshot snapshot = snapshotActiveQueue();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        nowMs = now();
        capacity = batchCapacitySnapshot();
        batchMaxCount = maxDecisionRequests();
        predictThresholdMs = predictedExecutionBudgetMs();
        batchMaxTokens = capacity.batchTokenCapacity();
        if (head.requestExpired(nowMs)) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    head.requestId(), head.expiresAtMs(), nowMs);
            dropHead(head);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }
        ScheduledRequest expiredMember = firstExpiredMember(
                snapshot.items(), batchMaxCount, nowMs);
        if (expiredMember != null) {
            Logger.debug("flexlb_batch_drop request_id={} "
                            + "reason=request_expired expires_at_ms={} now_ms={}",
                    expiredMember.requestId(),
                    expiredMember.expiresAtMs(), nowMs);
            dropHead(expiredMember);
            return BatcherCycleResult.CAPACITY_CHANGED;
        }
        GroupPlanner.Shape headShape =
                GroupPlanner.Shape.empty().add(head.seqLen());
        batchKvTokens = capacity.batchKvCapacity();
        if (!headShape.fitsKv(batchKvTokens)) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        long exactBatchMaxTokens = batchMaxTokens;
        long exactBatchKvTokens = batchKvTokens;
        int exactBatchMaxCount = batchMaxCount;
        long exactPredictThresholdMs = predictThresholdMs;
        long exactFixedWaitMs = fixedWaitMs;
        return runPredictionBound(head, () -> {
            PrefillTimePredictor predictor =
                    prefillEndpoint.getPredictor();
            PrefillTimePredictor.Evaluator evaluator = predictor == null
                    ? null : predictor.evaluator();
            PrefillTimePredictor.Evaluator planningEvaluator =
                    exactPredictThresholdMs > 0 ? evaluator : null;
            GroupPlanner.Constraints plannerConstraints =
                    new GroupPlanner.Constraints(
                            exactBatchMaxCount, exactBatchMaxTokens,
                            exactBatchKvTokens, exactPredictThresholdMs,
                            exactFixedWaitMs);
            GroupPlanner.Selection<ScheduledRequest> selection =
                    GroupPlanner.selectWithPrediction(
                            snapshot.items(), PLANNER_ITEM_ACCESS,
                            plannerConstraints,
                            planningEvaluator == null ? null :
                                    deliveryStrategy.newGroupPredictor(planningEvaluator));
            GroupPlanner.Plan<ScheduledRequest> plan =
                    GroupPlanner.evaluateReadiness(
                            selection, plannerConstraints, now());
            if (plan.ready()) {
                return admitDecisionGroup(
                        plan.items(), plan.shape(), plan.reason(),
                        evaluator, committedPrediction(plan));
            }
            return awaitCollectionWindow(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion(),
                    plan.windowOpenedAtMs(), exactFixedWaitMs,
                    Math.min(snapshot.items().size(), exactBatchMaxCount));
        });
    }

    private static BatcherCycleResult awaitCollectionWindow(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion,
            long windowOpenedAtMs,
            long collectionWindowMs,
            int windowCandidateCount) {
        long collectionDeadline = GroupPlanner.collectionDeadlineMs(
                windowOpenedAtMs, collectionWindowMs);
        return BatcherCycleResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                Math.min(collectionDeadline, head.expiresAtMs()), windowCandidateCount,
                "Prefill collection window");
    }

    private static BatcherCycleResult awaitPrefillKvCapacity(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion) {
        return BatcherCycleResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                head.expiresAtMs(), 0, "Prefill KV capacity exhausted");
    }

    private static ScheduledRequest firstExpiredMember(
            List<ScheduledRequest> orderedItems,
            int maxCount,
            long nowMs) {
        int inspected = Math.min(
                Math.max(1, maxCount), orderedItems.size());
        for (int index = 0; index < inspected; index++) {
            ScheduledRequest item = orderedItems.get(index);
            if (item.requestExpired(nowMs)) {
                return item;
            }
        }
        return null;
    }

    private BatcherCycleResult admitDecisionGroup(
            List<ScheduledRequest> picked,
            GroupPlanner.Shape shape,
            String reason,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedCommittedPredictionMs) {
        BatchCapacitySnapshot capacity = batchCapacitySnapshot();
        if ((picked.size() > 1 && !shape.fitsCompute(capacity.batchTokenCapacity()))
                || !shape.fitsKv(capacity.batchKvCapacity())) {
            return BatcherCycleResult.NO_ACTION;
        }
        return admitAndDeliverCapacityFeasiblePrefix(
                picked, reason, evaluator,
                plannedCommittedPredictionMs);
    }

    private static OptionalLong committedPrediction(
            GroupPlanner.Plan<ScheduledRequest> plan) {
        if (plan.selectedPredictionMs().isEmpty()) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(
                PrefillPredictionBoundary.committedDecisionGroupMs(
                        plan.selectedPredictionMs().getAsDouble()));
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        try {
            while (!stopped && !Thread.currentThread().isInterrupted()) {
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
                notifyAll();
            }
        }
    }

    private void runOneCycle() throws InterruptedException {
        waitForNonEmpty();
        if (stopped) {
            return;
        }

        BatcherCycleResult result = processQueue();
        if (result.waitReason() != null) {
            recordQueueWait(result.request(), result.waitReason());
        }
        if (result.placementCapacityChanged()) {
            // Only a committed removal creates a new queue seat. Advisory
            // waits and delivery-capacity misses must not feed placement back
            // into itself.
            prefillEndpoint.signalPlacementCapacityChanged();
        }
        if (result.capacityBlocked()) {
            awaitBlockedHeadCapacity(result);
        } else if (result.awaitingSchedulingChange()) {
            awaitSchedulingChange(result);
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped && activeIndex.isEmpty()) {
                setWaitingDecisionUnderLock(null);
                awaitStateChange();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Wait while the same eligible request is blocked by the rejecting resource.
     * Offers or Decode changes may expose an eligible successor; shared Prefill
     * pressure still keeps the same request waiting. Capacity is checked after
     * listener installation while queueLock is held, closing
     * the release-before-await race without polling.
     */
    private void awaitBlockedHeadCapacity(
            BatcherCycleResult blocked)
            throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            if (stopped) {
                return;
            }
            setWaitingDecisionUnderLock(blocked);
            while (!stopped
                    && !activeIndex.isEmpty()
                    && dispatchHeadUnderLock() == blocked.request()
                    && !blocked.request().requestExpired(System.currentTimeMillis())
                    && !blocked.unavailable().availability().isAvailable()) {
                long expiresAtMs = blocked.request().expiresAtMs();
                long nowMs = System.currentTimeMillis();
                if (expiresAtMs <= nowMs) {
                    return;
                }
                if (expiresAtMs == Long.MAX_VALUE) {
                    awaitStateChange();
                } else {
                    awaitStateChangeUntil(expiresAtMs - nowMs);
                }
            }
        } finally {
            try {
                if (waitingDecision == blocked) { setWaitingDecisionUnderLock(null); }
            } finally {
                queueLock.unlock();
            }
        }
    }

    /**
     * Wait for the exact queue/input generation captured by the policy or
     * for its deadline. Every predicate and signal is serialized by queueLock.
     */
    private void awaitSchedulingChange(
            BatcherCycleResult waiting)
            throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            if (stopped) { return; }
            setWaitingDecisionUnderLock(waiting);
            while (!stopped
                    && queueVersion.get() == waiting.queueVersion()
                    && schedulingInputVersion == waiting.schedulingInputVersion()
                    && waitingSelectionUnchangedUnderLock(waiting)) {
                long nowMs = System.currentTimeMillis();
                if (waiting.wakeAtMs() <= nowMs) {
                    return;
                }
                if (waiting.wakeAtMs() == Long.MAX_VALUE) {
                    awaitStateChange();
                } else {
                    awaitStateChangeUntil(waiting.wakeAtMs() - nowMs);
                }
            }
        } finally {
            try {
                if (waitingDecision == waiting) { setWaitingDecisionUnderLock(null); }
            } finally {
                queueLock.unlock();
            }
        }
    }

    private void awaitStateChange() throws InterruptedException {
        stateChanged.await();
    }

    private void awaitStateChangeUntil(long remainingMs) throws InterruptedException {
        stateChanged.awaitNanos(TimeUnit.MILLISECONDS.toNanos(
                Math.max(1L, remainingMs)));
    }

    /** Decode readiness can change a strategy wait or an independent Decode-capacity wait. */
    private static boolean decodeCanChangeWait(BatcherCycleResult waiting) {
        if (waiting == null) { return false; }
        if (!waiting.capacityBlocked()) { return true; }
        var semantics = waiting.unavailable().projectionSemantics();
        return semantics != null
                && semantics.afterProbe() == RouteProjection.AfterProbeAdmission.INDEPENDENT;
    }

    private void signalQueuedDecodeCapacityAvailable() {
        if (!decodeCanChangeWait(waitingDecision)) { return; }
        queueLock.lock();
        try {
            if (decodeCanChangeWait(waitingDecision)) { stateChanged.signal(); }
        } finally {
            queueLock.unlock();
        }
    }

    /** Called after Prefill or Decode capacity changes, outside endpoint locks. */
    public void signalDeliveryCapacityAvailable() {
        // Exact capacity notifications do not affect strategy waits.
        BatcherCycleResult waiting = waitingDecision;
        if (waiting == null || !waiting.capacityBlocked()) { return; }
        queueLock.lock();
        try {
            if (waitingDecision != null && waitingDecision.capacityBlocked()) {
                schedulingInputsChangedUnderLock();
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /** Publish a real Prefill ownership or batch capacity change to local and global waiters. */
    private void signalCapacityAvailable() {
        signalDeliveryCapacityAvailable();
        prefillEndpoint.signalPlacementCapacityChanged();
    }

    /** Only capacity waits own an admission observation and an exact resource subscription. */
    private void setWaitingDecisionUnderLock(BatcherCycleResult next) {
        if (!queueLock.isHeldByCurrentThread()) {
            throw new IllegalStateException("wait update requires queueLock");
        }
        BatcherCycleResult previous = waitingDecision;
        if (previous == next) { return; }
        boolean previousCapacity = previous != null && previous.capacityBlocked();
        boolean nextCapacity = next != null && next.capacityBlocked();
        CapacityBoundary.Availability previousSource = previousCapacity
                ? previous.unavailable().availability() : null;
        CapacityBoundary.Availability nextSource = nextCapacity
                ? next.unavailable().availability() : null;
        waitingDecision = next;
        if (previousCapacity || nextCapacity) { schedulingInputsChangedUnderLock(); }
        if (previousSource != nextSource) {
            try {
                if (previousSource != null) { previousSource.removeListener(capacityAvailableSignal); }
            } finally {
                if (nextSource != null) { nextSource.addListener(capacityAvailableSignal); }
            }
        }
    }

    /**
     * Record an input change independently of waking a thread. The generation
     * rejects cached and unfinished projections as well as stale decision waits.
     */
    private void schedulingInputsChangedUnderLock() {
        if (!queueLock.isHeldByCurrentThread()) {
            throw new IllegalStateException("scheduling input update requires queueLock");
        }
        schedulingInputVersion++;
    }

    /** Wake decisions whose advisory worker-status or predictor input changed. */
    public void signalSchedulingInputsChanged() {
        queueLock.lock();
        try {
            schedulingInputsChangedUnderLock();
            stateChanged.signal();
        } finally {
            queueLock.unlock();
        }
    }

}
