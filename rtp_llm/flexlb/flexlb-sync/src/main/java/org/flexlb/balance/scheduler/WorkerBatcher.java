package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.EndpointGenerationRetiredException;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityOrdering;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-worker scheduling queue that owns request grouping and hard-capacity admission,
 * delegating grouping decisions to a {@link GroupPolicy}.
 *
 * <p>One instance per Prefill worker. Exact requests enter through the neutral
 * {@link PrefillGenerationRuntime} boundary and are grouped by the configured
 * decision policy. A
 * group may independently be delivered through
 * EnqueueBatch or as individual route decisions.
 *
 * <p>The context-owned active queue is the single source of truth for queued
 * requests. PRIORITY ordering uses
 * the explicit priority comparator ({@link #PRIORITY_QUEUE_ORDER}); FIFO uses
 * a unique monotonic enqueue sequence. All mutations go through
 * {@link BatcherContext} (or the
 * lock-holding methods here). A monotonic mutation generation lets snapshots
 * and diagnostics identify queue-state changes.
 */
final class WorkerBatcher implements PrefillGenerationRuntime {

    private enum RuntimeState {
        NEW,
        STARTING,
        RUNNING,
        STOPPING,
        STOPPED
    }

    private enum PreparedOfferState {
        OPEN,
        SEALED,
        REVOKED,
        COMMITTED,
        CLOSED,
        RETIRED
    }

    /** Exact generation-local pending/waiting seat. */
    private final class PreparedOfferImpl implements PreparedOffer {
        private final long requestId;
        private final int priority;
        private final long sequence;
        /** Guarded by queueLock. */
        private PreparedOfferState state = PreparedOfferState.OPEN;

        private PreparedOfferImpl(long requestId, int priority, long sequence) {
            this.requestId = requestId;
            this.priority = priority;
            this.sequence = sequence;
        }

        @Override
        public boolean seal() {
            queueLock.lock();
            try {
                if (state == PreparedOfferState.OPEN) {
                    requireOwnedPreparedOffer(this);
                    state = PreparedOfferState.SEALED;
                    return true;
                }
                if (state == PreparedOfferState.SEALED) {
                    requireOwnedPreparedOffer(this);
                    return true;
                }
                return false;
            } finally {
                queueLock.unlock();
            }
        }

        @Override
        public void commit(DeliveryItem exactItem) {
            if (!(exactItem instanceof BatchItem item)) {
                throw new IllegalArgumentException(
                        "prepared Prefill offer requires a BatchItem");
            }
            if (item.requestId() != requestId
                    || item.prefillEp() != prefillEndpoint
                    || item.priority() != priority) {
                throw new IllegalArgumentException(
                        "prepared Prefill offer does not match exact request/generation");
            }

            boolean capacityFreed = false;
            RuntimeException failure = null;
            queueLock.lock();
            try {
                requireSealedPreparedOffer(this);
                if (runtimeState != RuntimeState.RUNNING || stopped) {
                    releasePreparedOfferUnderLock(
                            this, PreparedOfferState.RETIRED);
                    capacityFreed = refreshOfferCapacityUnderLock();
                    failure = retiredGeneration();
                } else if (!ctx.publishActiveIndexUnderLock(item)) {
                    releasePreparedOfferUnderLock(
                            this, PreparedOfferState.CLOSED);
                    capacityFreed = refreshOfferCapacityUnderLock();
                    failure = new IllegalStateException(
                            "prepared Prefill request conflicts with canonical owner request_id="
                                    + requestId);
                } else {
                    PreparedOfferImpl removed = preparedOffers.remove(requestId);
                    if (removed != this) {
                        throw new IllegalStateException(
                                "prepared Prefill offer lost its exact hold request_id="
                                        + requestId);
                    }
                    state = PreparedOfferState.COMMITTED;
                    // hold -> ACTIVE keeps both capacity counts unchanged.
                    refreshOfferCapacityUnderLock();
                    stateChanged.signal();
                }
            } finally {
                queueLock.unlock();
            }
            notifyOfferCapacity(capacityFreed);
            if (failure != null) {
                throw failure;
            }
        }

        @Override
        public void close() {
            boolean capacityFreed = false;
            queueLock.lock();
            try {
                if (state != PreparedOfferState.OPEN
                        && state != PreparedOfferState.SEALED) {
                    return;
                }
                releasePreparedOfferUnderLock(
                        this, PreparedOfferState.CLOSED);
                capacityFreed = refreshOfferCapacityUnderLock();
            } finally {
                queueLock.unlock();
            }
            notifyOfferCapacity(capacityFreed);
        }
    }

    /** Stable listener identity for the lifetime of this endpoint generation. */
    private final class OfferAvailability
            implements CapacityBoundary.Availability {
        private final CopyOnWriteArraySet<Runnable> listeners =
                new CopyOnWriteArraySet<>();

        @Override
        public boolean isAvailable() {
            return offerCapacityAvailable;
        }

        @Override
        public void addListener(Runnable listener) {
            listeners.add(java.util.Objects.requireNonNull(listener, "listener"));
        }

        @Override
        public void removeListener(Runnable listener) {
            listeners.remove(listener);
        }

        private void signal() {
            for (Runnable listener : listeners) {
                try {
                    listener.run();
                } catch (Throwable failure) {
                    try {
                        Logger.error("WorkerBatcher[{}] offer-capacity listener failed",
                                key, failure);
                    } catch (Throwable ignoredLoggingFailure) {
                        // Capacity has already been released.
                    }
                }
            }
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
    public static final Comparator<BatchItem> PRIORITY_QUEUE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());

    /** FIFO order: unique monotonic enqueue sequence. */
    public static final Comparator<BatchItem> FIFO_QUEUE_ORDER =
            Comparator.comparingLong(BatchItem::enqueueSeq);

    private static final Comparator<GroupPlanner.Item> PRIORITY_PROJECTION_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_PROJECTION_ORDER =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparingLong(GroupPlanner.Item::requestId);

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final DeliveryStrategy deliveryStrategy;
    private final DeliveryLifecyclePort deliveryLifecycle;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final boolean priorityOrdering;
    private final long maximumPendingRequests;
    private final int maximumWaitingRequests;
    /**
     * Monotonic queue mutation generation, bumped on enqueue, removal,
     * delivery and drain. It is exposed in diagnostic snapshots.
     */
    private final AtomicLong queueVersion = new AtomicLong();
    /**
     * Guards queue mutations and atomic victim replacement.
     *
     * <p>The lock and generation stay active for both FIFO and PRIORITY so
     * every ordering mode has the same mutation guarantees.
     */
    private final ReentrantLock queueLock = new ReentrantLock();
    /** Live OPEN/SEALED cross-role offer holds, guarded by {@link #queueLock}. */
    private final Map<Long, PreparedOfferImpl> preparedOffers = new HashMap<>();
    /** Unique hold order used only to choose the latest equal-priority OPEN hold. */
    private long nextPreparedOfferSequence;
    private final AtomicLong offerCapacityEpoch = new AtomicLong();
    private final OfferAvailability offerAvailability = new OfferAvailability();
    private volatile boolean offerCapacityAvailable;
    /** Last canonical seat use observed under {@link #queueLock}. */
    private long observedPendingSeatUse;
    private long observedWaitingSeatUse;
    /** Canonical request ownership guarded exclusively by {@link #queueLock}. */
    private final PrefillWorkRegistry workRegistry;
    /**
     * One per-worker condition for new queue work and endpoint-capacity release.
     * Every predicate transition and signal is serialized by queueLock, so a
     * release cannot race between the cap re-check and await.
     */
    private final Condition stateChanged = queueLock.newCondition();
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
    private final GroupPolicy groupPolicy;
    private final BatcherContext ctx;
    private final Runnable capacityAvailableSignal =
            this::signalDeliveryCapacityAvailable;
    /** Exact resource event source for the currently blocked active head. */
    private CapacityBoundary.Availability
            subscribedCapacityAvailability;
    /** Exact active head for which this worker is waiting on a capacity event. */
    private BatcherCycleResult.CapacityBlocked capacityBlockedHead;

    WorkerBatcher(
            String key,
            PrefillEndpoint prefillEp,
            FlexlbConfig config,
            DeliveryStrategy deliveryStrategy,
            DeliveryLifecyclePort deliveryLifecycle) {
        this.key = key;
        this.prefillEndpoint = prefillEp;
        boolean queueScheduling = config.isQueue();
        this.priorityOrdering = config.isPriorityOrdering();
        this.deliveryStrategy = deliveryStrategy;
        this.deliveryLifecycle = deliveryLifecycle;
        this.maximumPendingRequests = config.getRouter().getRoles()
                .getPrefill().getAvailability().getMaxPendingRequests();
        this.maximumWaitingRequests = config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker();
        Comparator<BatchItem> queueOrder = priorityOrdering
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        Comparator<GroupPlanner.Item> projectionOrder =
                priorityOrdering
                        ? PRIORITY_PROJECTION_ORDER : FIFO_PROJECTION_ORDER;
        this.queue = new PriorityBlockingQueue<>(11, queueOrder);
        this.workRegistry = new PrefillWorkRegistry(
                queueLock, queue, capacityAvailableSignal);
        this.groupPolicy = config.isSingleDecision()
                ? new SingleRequestGroupPolicy()
                : new FixedWindowGroupPolicy();
        this.ctx = new BatcherContext(
                key, prefillEp, config, deliveryLifecycle,
                queue, queueVersion, queueLock, queueOrder,
                projectionOrder, queueScheduling, deliveryStrategy,
                workRegistry);
        this.normalStopFailure = new CancellationException(
                "FlexLB worker scheduling queue stopped: " + key);
        this.stopAcknowledgementFailure = new IllegalStateException(
                "FlexLB worker stop callback lost its exact retained owner: "
                        + key);
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
    }

    @Override
    public synchronized void start() {
        if (runtimeState != RuntimeState.NEW) {
            throw new IllegalStateException(
                    "Prefill generation runtime cannot start from "
                            + runtimeState);
        }
        runtimeState = RuntimeState.STARTING;
        try {
            workerThread.start();
            runtimeState = RuntimeState.RUNNING;
            queueLock.lock();
            try {
                refreshOfferCapacityUnderLock();
            } finally {
                queueLock.unlock();
            }
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

    @Override
    public boolean offer(DeliveryItem exactItem) {
        BatchItem item = (BatchItem) exactItem;
        assert item.prefillEp() == prefillEndpoint
                : "incoming item belongs to another Prefill generation";
        if (runtimeState != RuntimeState.RUNNING || stopped) {
            return false;
        }
        return enqueue(item) == OfferResult.OFFERED;
    }

    @Override
    public PreparedOffer prepareOffer(long requestId, int priority) {
        if (requestId < 0L) {
            throw new IllegalArgumentException("requestId must be non-negative");
        }
        queueLock.lock();
        try {
            if (runtimeState != RuntimeState.RUNNING || stopped) {
                throw retiredGeneration();
            }
            if (preparedOffers.containsKey(requestId)
                    || workRegistry.ownsRequestUnderLock(requestId)) {
                throw new IllegalStateException(
                        "Prefill request already has a canonical owner request_id="
                                + requestId);
            }
            if (!hasOfferCapacityUnderLock()) {
                PreparedOffer replacement =
                        replaceOpenPreparedOfferUnderLock(requestId, priority);
                // Pending ownership may also grow through endpoint status or
                // DIRECT ledger reconciliation.  A failed/full probe is the
                // QUEUE boundary which must publish that pressure before the
                // corresponding release can advance the epoch and wake a lane.
                refreshOfferCapacityUnderLock();
                return replacement;
            }
            PreparedOfferImpl prepared = newPreparedOffer(requestId, priority);
            preparedOffers.put(requestId, prepared);
            refreshOfferCapacityUnderLock();
            return prepared;
        } finally {
            queueLock.unlock();
        }
    }

    /** A full PRIORITY worker may transfer exactly one revocable hold seat. */
    private PreparedOfferImpl replaceOpenPreparedOfferUnderLock(
            long requestId,
            int priority) {
        assert queueLock.isHeldByCurrentThread()
                : "prepared offer replacement requires queueLock";
        if (!priorityOrdering) {
            return null;
        }
        PreparedOfferImpl victim = null;
        for (PreparedOfferImpl candidate : preparedOffers.values()) {
            if (candidate.state != PreparedOfferState.OPEN) {
                continue;
            }
            if (victim == null
                    || candidate.priority < victim.priority
                    || candidate.priority == victim.priority
                    && candidate.sequence > victim.sequence) {
                victim = candidate;
            }
        }
        if (victim == null || priority <= victim.priority) {
            return null;
        }
        if (!preparedOffers.remove(victim.requestId, victim)) {
            throw new IllegalStateException(
                    "prepared Prefill replacement lost its exact victim");
        }
        victim.state = PreparedOfferState.REVOKED;
        PreparedOfferImpl incoming = newPreparedOffer(requestId, priority);
        preparedOffers.put(requestId, incoming);
        // One live hold replaced one live hold: no capacity epoch or signal.
        return incoming;
    }

    private PreparedOfferImpl newPreparedOffer(long requestId, int priority) {
        assert queueLock.isHeldByCurrentThread()
                : "prepared offer creation requires queueLock";
        return new PreparedOfferImpl(
                requestId, priority, nextPreparedOfferSequence++);
    }

    @Override
    public CapacityBoundary.Availability offerAvailability() {
        return offerAvailability;
    }

    @Override
    public long offerCapacityEpoch() {
        return offerCapacityEpoch.get();
    }

    private enum OfferResult {
        OFFERED,
        FULL,
        STOPPED
    }

    private OfferResult enqueue(BatchItem item) {
        queueLock.lock();
        try {
            return enqueueUnderLock(item);
        } finally {
            queueLock.unlock();
        }
    }

    /** Caller holds {@link #queueLock}. */
    private OfferResult enqueueUnderLock(BatchItem item) {
        if (stopped) {
            return OfferResult.STOPPED;
        }
        if (!hasOfferCapacityUnderLock()) {
            refreshOfferCapacityUnderLock();
            return OfferResult.FULL;
        }
        if (preparedOffers.containsKey(item.requestId())
                || !ctx.publishActiveIndexUnderLock(item)) {
            return OfferResult.STOPPED;
        }
        refreshOfferCapacityUnderLock();
        stateChanged.signal();
        return OfferResult.OFFERED;
    }

    @Override
    public int queueSize() {
        return queue.size();
    }

    @Override
    public long pendingRequestCount() {
        queueLock.lock();
        try {
            return effectivePendingSeatUseUnderLock();
        } finally {
            queueLock.unlock();
        }
    }

    private boolean hasOfferCapacityUnderLock() {
        assert queueLock.isHeldByCurrentThread()
                : "offer capacity requires queueLock";
        return effectivePendingSeatUseUnderLock() < maximumPendingRequests
                && (maximumWaitingRequests <= 0
                || effectiveWaitingSeatUseUnderLock() < maximumWaitingRequests);
    }

    private long effectivePendingSeatUseUnderLock() {
        assert queueLock.isHeldByCurrentThread()
                : "pending projection requires queueLock";
        return saturatingAdd(
                workRegistry.pendingRequestCount(), preparedOffers.size());
    }

    private long effectiveWaitingSeatUseUnderLock() {
        assert queueLock.isHeldByCurrentThread()
                : "waiting projection requires queueLock";
        return saturatingAdd(queue.size(), preparedOffers.size());
    }

    /** Refresh the lock-free availability view and detect an actual seat release. */
    private boolean refreshOfferCapacityUnderLock() {
        assert queueLock.isHeldByCurrentThread()
                : "offer capacity refresh requires queueLock";
        long pending = effectivePendingSeatUseUnderLock();
        long waiting = effectiveWaitingSeatUseUnderLock();
        boolean capacityFreed = pending < observedPendingSeatUse
                || waiting < observedWaitingSeatUse;
        observedPendingSeatUse = pending;
        observedWaitingSeatUse = waiting;
        boolean available = runtimeState == RuntimeState.RUNNING
                && !stopped
                && pending < maximumPendingRequests
                && (maximumWaitingRequests <= 0
                || waiting < maximumWaitingRequests);
        offerCapacityAvailable = available;
        if (capacityFreed) {
            offerCapacityEpoch.updateAndGet(
                    epoch -> epoch == Long.MAX_VALUE ? epoch : epoch + 1L);
        }
        return capacityFreed && available;
    }

    private void notifyOfferCapacity(boolean capacityFreedAndAvailable) {
        if (capacityFreedAndAvailable) {
            offerAvailability.signal();
        }
    }

    private void requireOwnedPreparedOffer(PreparedOfferImpl prepared) {
        assert queueLock.isHeldByCurrentThread()
                : "prepared offer validation requires queueLock";
        if (preparedOffers.get(prepared.requestId) != prepared) {
            throw new IllegalStateException(
                    "prepared Prefill offer lost its exact hold request_id="
                            + prepared.requestId);
        }
    }

    private void requireSealedPreparedOffer(PreparedOfferImpl prepared) {
        assert queueLock.isHeldByCurrentThread()
                : "prepared offer validation requires queueLock";
        if (prepared.state == PreparedOfferState.RETIRED) {
            throw retiredGeneration();
        }
        if (prepared.state != PreparedOfferState.SEALED) {
            throw new IllegalStateException(
                    "prepared Prefill offer must be sealed before commit request_id="
                            + prepared.requestId);
        }
        requireOwnedPreparedOffer(prepared);
    }

    private void releasePreparedOfferUnderLock(
            PreparedOfferImpl prepared,
            PreparedOfferState terminalState) {
        requireOwnedPreparedOffer(prepared);
        preparedOffers.remove(prepared.requestId, prepared);
        prepared.state = terminalState;
    }

    private EndpointGenerationRetiredException retiredGeneration() {
        return new EndpointGenerationRetiredException(
                "Prefill generation retired before offer endpoint=" + key);
    }

    private static long saturatingAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    /**
     * Snapshot of the current active queue depth bucketed by normalized
     * scheduling priority. Only priorities present in the queue appear in the
     * result, matching the tagged queue-depth metric behavior.
     */
    @Override
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        for (BatchItem item : queue) {
            sizeByPriority.merge(item.priority(), 1, Integer::sum);
        }
        return sizeByPriority;
    }

    /** Current active queue mutation generation. */
    public long queueVersion() {
        return queueVersion.get();
    }

    /** Current generation of worker-status and prediction-model inputs. */
    public long schedulingInputVersion() {
        return ctx.schedulingInputVersionValue();
    }

    PrefillWorkLedger ownedLedger() {
        return workRegistry;
    }

    @Override
    public RouteProjection.Inputs captureRouteProjectionInputs() {
        queueLock.lock();
        try {
            RouteProjection.Inputs inputs =
                    ctx.captureRouteProjectionInputs(this::admissionBlockUnderLock);
            return new RouteProjection.Inputs(
                    inputs.queue(),
                    inputs.work(),
                    saturatingAdd(
                            inputs.pendingRequestCount(), preparedOffers.size()));
        } finally {
            queueLock.unlock();
        }
    }

    /** Immutable delivery semantics used by a pure route projection. */
    @Override
    public RouteProjection.DeliveryProjection deliveryProjection() {
        return deliveryStrategy.projectionPolicy();
    }

    /**
     * Capture only an admission rejection whose worker wait predicate still
     * holds. The availability read is the already-subscribed wait predicate;
     * it neither previews nor reserves capacity.
     *
     * <p>Caller holds {@link #queueLock}.
     */
    private AdmissionBlock admissionBlockUnderLock() {
        assert queueLock.isHeldByCurrentThread()
                : "capacity block snapshot requires queueLock";
        BatcherCycleResult.CapacityBlocked blocked = capacityBlockedHead;
        if (blocked == null
                || ctx.peek() != blocked.item()
                || blocked.item().requestExpired(ctx.now())
                || blocked.unavailable().availability().isAvailable()) {
            return null;
        }
        return new AdmissionBlock(
                blocked.item().requestId(),
                blocked.item().enqueueSeq(),
                blocked.unavailable().projectionSemantics());
    }

    @Override
    public Throwable stopAndAwait() {
        if (Thread.currentThread() == workerThread) {
            throw new IllegalStateException(
                    "Prefill runtime cannot await its own worker thread");
        }
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
                ctx.stopAcceptingUnderLock();
                for (PreparedOfferImpl prepared : preparedOffers.values()) {
                    prepared.state = PreparedOfferState.RETIRED;
                }
                preparedOffers.clear();
                refreshOfferCapacityUnderLock();
                try {
                    unsubscribeFromBlockedCapacity();
                } catch (Throwable subscriptionFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, subscriptionFailure);
                }
                capacityBlockedHead = null;
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

            if (interruptWorker && Thread.currentThread() != workerThread) {
                try {
                    workerThread.interrupt();
                } catch (Throwable interruptFailure) {
                    cleanupFailure = appendCleanupFailure(
                            cleanupFailure, interruptFailure);
                }
            }

            while (true) {
                BatchItem item;
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
                    deliveryLifecycle.onOfferFailure(item, terminalFailure);
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
    private BatchItem detachNextStoppedItem() {
        queueLock.lock();
        try {
            BatchItem item = ctx.detachNextStopTerminalUnderLock();
            stateChanged.signalAll();
            return item;
        } finally {
            queueLock.unlock();
        }
    }

    /** Acknowledge only the retained owner whose terminal callback returned. */
    private boolean acknowledgeStoppedItem(BatchItem item) {
        queueLock.lock();
        try {
            return ctx.acknowledgeStopTerminalUnderLock(item);
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

    // ==================== priority scheduling queue operations ====================

    @Override
    public boolean removeQueued(
            DeliveryItem exactItem,
            String reason) {
        BatchItem item = (BatchItem) exactItem;
        assert item.prefillEp() == prefillEndpoint
                : "queued item belongs to another Prefill generation";
        boolean removed;
        boolean capacityFreed = false;
        queueLock.lock();
        try {
            removed = runtimeState == RuntimeState.RUNNING
                    && !stopped
                    && ctx.removeUnderLock(item);
            if (removed) {
                capacityFreed = refreshOfferCapacityUnderLock();
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
        notifyOfferCapacity(capacityFreed);
        if (removed) {
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

    @Override
    @SuppressWarnings("unchecked")
    public QueueReplacement replaceQueued(
            List<DeliveryItem> exactVictims,
            DeliveryItem incoming) {
        List<BatchItem> victims =
                (List<BatchItem>) (List<?>) exactVictims;
        BatchItem incomingItem = (BatchItem) incoming;
        QueueReplacement committed = new QueueReplacement(
                QueueReplacementStatus.SUCCESS);
        queueLock.lock();
        try {
            if (runtimeState != RuntimeState.RUNNING || stopped) {
                return new QueueReplacement(
                        QueueReplacementStatus.DECLINED);
            }
            long pending = effectivePendingSeatUseUnderLock();
            long waiting = effectiveWaitingSeatUseUnderLock();
            if (pending > maximumPendingRequests
                    || maximumWaitingRequests > 0
                    && waiting > maximumWaitingRequests) {
                return new QueueReplacement(
                        QueueReplacementStatus.DECLINED);
            }
            boolean saturated = pending == maximumPendingRequests
                    || maximumWaitingRequests > 0
                    && waiting == maximumWaitingRequests;
            BatchItem worst = worstActiveUnderLock();
            if (!saturated
                    || victims.size() != 1
                    || worst == null
                    || victims.get(0) != worst
                    || incomingItem.priority() <= worst.priority()) {
                return new QueueReplacement(
                        QueueReplacementStatus.DECLINED);
            }
            if (preparedOffers.containsKey(incomingItem.requestId())) {
                return new QueueReplacement(
                        QueueReplacementStatus.CONFLICT);
            }
            PrefillWorkRegistry.ActiveReplaceStatus replacement =
                    workRegistry.replaceActiveExact(victims, incomingItem);
            if (replacement
                    == PrefillWorkRegistry.ActiveReplaceStatus.CONFLICT) {
                return new QueueReplacement(
                        QueueReplacementStatus.CONFLICT);
            }
            queueVersion.incrementAndGet();
            stateChanged.signal();
            return committed;
        } finally {
            queueLock.unlock();
        }
    }

    private BatchItem worstActiveUnderLock() {
        BatchItem worst = null;
        for (BatchItem item : queue) {
            if (worst == null || PRIORITY_QUEUE_ORDER.compare(item, worst) > 0) {
                worst = item;
            }
        }
        return worst;
    }

    @Override
    @SuppressWarnings("unchecked")
    public QueueSnapshot captureQueueSnapshot() {
        queueLock.lock();
        try {
            List<BatchItem> queued = ctx.activeItemsInSchedulingOrder();
            List<DeliveryItem> items =
                    (List<DeliveryItem>) (List<?>) queued;
            return new QueueSnapshot(
                    key,
                    queueVersion.get(),
                    ctx.maxQueueCapacity(),
                    effectiveWaitingSeatUseUnderLock(),
                    effectivePendingSeatUseUnderLock(),
                    maximumPendingRequests,
                    items);
        } finally {
            queueLock.unlock();
        }
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

        BatcherCycleResult result = groupPolicy.processQueue(ctx);
        // The policy may terminalize an ACTIVE item without a ledger lease.
        // Reconcile offer seats before the worker enters its next wait.
        boolean offerCapacityFreed;
        queueLock.lock();
        try {
            offerCapacityFreed = refreshOfferCapacityUnderLock();
        } finally {
            queueLock.unlock();
        }
        notifyOfferCapacity(offerCapacityFreed);
        if (result instanceof BatcherCycleResult.CapacityBlocked blocked) {
            awaitBlockedHeadCapacity(blocked);
        } else if (result
                instanceof BatcherCycleResult.AwaitingSchedulingChange waiting) {
            awaitSchedulingChange(waiting);
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped && !ctx.hasProcessableWork()) {
                capacityBlockedHead = null;
                unsubscribeFromBlockedCapacity();
                awaitStateChange();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Subscribe to the exact resource which rejected the active head.
     * Caller holds {@link #queueLock}.
     */
    private void subscribeToBlockedCapacity(
            BatcherCycleResult.CapacityBlocked blocked) {
        CapacityBoundary.Availability nextSource =
                blocked.unavailable().availability();
        if (nextSource != subscribedCapacityAvailability) {
            unsubscribeFromBlockedCapacity();
            subscribedCapacityAvailability = nextSource;
            nextSource.addListener(capacityAvailableSignal);
        }
    }

    /** Caller holds queueLock. */
    private void unsubscribeFromBlockedCapacity() {
        CapacityBoundary.Availability source =
                subscribedCapacityAvailability;
        if (source == null) {
            return;
        }
        subscribedCapacityAvailability = null;
        source.removeListener(capacityAvailableSignal);
    }

    /**
     * Wait while the exact active head is blocked by the exact rejecting
     * resource. Offers wake the condition: a new higher-priority head exits the
     * wait immediately, while a tail offer leaves the predicate true. Capacity
     * is checked after listener installation while queueLock is held, closing
     * the release-before-await race without polling.
     */
    private void awaitBlockedHeadCapacity(
            BatcherCycleResult.CapacityBlocked blocked)
            throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            if (stopped) {
                return;
            }
            capacityBlockedHead = blocked;
            subscribeToBlockedCapacity(blocked);
            while (!stopped
                    && ctx.peek() == blocked.item()
                    && !blocked.item().requestExpired(System.currentTimeMillis())
                    && !blocked.unavailable().availability().isAvailable()) {
                long expiresAtMs = blocked.item().expiresAtMs();
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
            capacityBlockedHead = null;
            unsubscribeFromBlockedCapacity();
            queueLock.unlock();
        }
    }

    /**
     * Wait for the exact queue/input generation captured by the policy or
     * for its deadline. Every predicate and signal is serialized by queueLock.
     */
    private void awaitSchedulingChange(
            BatcherCycleResult.AwaitingSchedulingChange waiting)
            throws InterruptedException {
        queueLock.lockInterruptibly();
        try {
            while (!stopped
                    && ctx.peek() == waiting.head()
                    && queueVersion.get() == waiting.queueVersion()
                    && ctx.schedulingInputVersionValue()
                    == waiting.schedulingInputVersion()) {
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
            queueLock.unlock();
        }
    }

    private void awaitStateChange() throws InterruptedException {
        stateChanged.await();
    }

    private void awaitStateChangeUntil(long remainingMs) throws InterruptedException {
        stateChanged.awaitNanos(TimeUnit.MILLISECONDS.toNanos(
                Math.max(1L, remainingMs)));
    }

    /** Called after Prefill or Decode capacity changes, outside endpoint locks. */
    public void signalDeliveryCapacityAvailable() {
        boolean offerCapacityFreed;
        queueLock.lock();
        try {
            offerCapacityFreed = refreshOfferCapacityUnderLock();
            if (capacityBlockedHead != null) {
                stateChanged.signal();
            }
        } finally {
            queueLock.unlock();
        }
        notifyOfferCapacity(offerCapacityFreed);
    }

    /** Wake decisions whose advisory worker-status or predictor input changed. */
    @Override
    public void signalSchedulingInputsChanged() {
        queueLock.lock();
        try {
            ctx.incrementSchedulingInputVersion();
            stateChanged.signal();
        } finally {
            queueLock.unlock();
        }
    }

}
