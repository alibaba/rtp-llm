package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.BlockedRequest;
import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.delivery.DeliveryGroup;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryTransaction;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;
import org.flexlb.balance.endpoint.PrefillActiveIndex;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.estimate.QueueView.AdmissionBlock;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.queue.BatchCapacityView;
import org.flexlb.balance.queue.GroupingLimits;
import org.flexlb.balance.queue.LoadViewCache;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.queue.QueueWaitSnapshot;
import org.flexlb.balance.queue.QueuedRequestsSnapshot;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.PrefillQueueEnqueueResult;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.util.PriorityOrdering;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.TimeUnit;
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
public final class PrefillQueue {

    /** Exact predicate captured by one scheduling cycle. */
    private record BatcherCycleResult(
            boolean placementCapacityChanged,
            ScheduledRequest request,
            DeliveryBlockReason unavailable,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs,
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

        private static BatcherCycleResult capacityBlocked(ScheduledRequest item, DeliveryBlockReason unavailable) {
            return new BatcherCycleResult(false, Objects.requireNonNull(item, "item"), unavailable, 0L, 0L, 0L,
                    unavailable.estimateSemantics() == null
                            ? "Prefill delivery capacity exhausted"
                            : unavailable.estimateSemantics().blockedDetail());
        }

        private static BatcherCycleResult awaitingSchedulingChange(
                ScheduledRequest head,
                long queueVersion,
                long schedulingInputVersion,
                long wakeAtMs,
                String waitReason) {
            return new BatcherCycleResult(false,
                    Objects.requireNonNull(head, "head"), null,
                    queueVersion, schedulingInputVersion, wakeAtMs, waitReason);
        }

        private static BatcherCycleResult simple(boolean capacityChanged) {
            return new BatcherCycleResult(
                    capacityChanged, null, null, 0L, 0L, 0L, null);
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

    private static final Comparator<GroupPlanner.Item> PRIORITY_ESTIMATE_ORDER =
            (left, right) -> PriorityOrdering.compareWithRequestId(
                    left.priority(), left.enqueueSeq(), left.requestId(),
                    right.priority(), right.enqueueSeq(), right.requestId());
    private static final Comparator<GroupPlanner.Item> FIFO_ESTIMATE_ORDER =
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

    private final String key;
    private final PrefillEndpoint prefillEndpoint;
    private final QueueCapacity queueCapacity;
    private final EndpointEventHandler endpointEvents;
    private final FlexlbConfig config;
    private final GroupingLimits groupingLimits;
    private final boolean singleDecision;
    private final DeliveryStrategy deliveryStrategy;
    private final PrefillTtftEstimator ttftEstimator;
    private final QueueState queueState;
    private final LoadViewCache loadViewCache;
    /** Canonical request ownership guarded exclusively by the queue lock. */
    private final PrefillState prefillState;
    private final PrefillQueueLifecycle lifecycle;
    private final PrefillQueueEnqueuer enqueuer;
    private final Runnable capacityAvailableSignal =
            this::signalCapacityAvailable;
    /** Exact active head for which this worker is waiting on a capacity event. */
    private BatcherCycleResult capacityBlockedHead;

    /** Last waiting decision; timeout readers never acquire the queue lock. */
    private volatile QueueWaitSnapshot waitSnapshot;

    public PrefillQueue(
            String key,
            PrefillEndpoint prefillEp,
            FlexlbConfig config,
            DeliveryStrategy deliveryStrategy,
            PrefillTtftEstimator ttftEstimator,
            EndpointEventHandler endpointEvents,
            QueueCapacity queueCapacity) {
        this(key, prefillEp, () -> config, deliveryStrategy, ttftEstimator, endpointEvents, queueCapacity);
    }

    public PrefillQueue(String key,
                         PrefillEndpoint prefillEp,
                         Supplier<FlexlbConfig> configSupplier,
                         DeliveryStrategy deliveryStrategy,
                         PrefillTtftEstimator ttftEstimator,
                         EndpointEventHandler endpointEvents,
                         QueueCapacity queueCapacity) {
        FlexlbConfig config = configSupplier.get();
        this.key = key;
        this.prefillEndpoint = prefillEp;
        this.queueCapacity = queueCapacity;
        this.config = config;
        this.singleDecision = config.isSingleDecision();
        this.groupingLimits = new GroupingLimits(config, configSupplier);
        boolean priorityOrdering = config.isPriorityOrdering();
        this.endpointEvents = endpointEvents;
        this.deliveryStrategy = deliveryStrategy;
        this.ttftEstimator = ttftEstimator;
        Comparator<ScheduledRequest> queueOrder = priorityOrdering
                ? PRIORITY_QUEUE_ORDER : FIFO_QUEUE_ORDER;
        Comparator<GroupPlanner.Item> estimateOrder = priorityOrdering ? PRIORITY_ESTIMATE_ORDER : FIFO_ESTIMATE_ORDER;
        this.queueState = new QueueState(PrefillActiveIndex.ordered(INITIAL_QUEUE_ALLOCATION, queueOrder));
        this.prefillState = new PrefillState(queueState.reentrantLock(),
                                             queueState.activeIndex(),
                                             capacityAvailableSignal);
        this.loadViewCache = new LoadViewCache(queueState,
                                               prefillState,
                                               prefillEp,
                                               config,
                                               groupingLimits,
                                               estimateOrder,
                                               this::admissionBlockUnderLock);
        this.lifecycle = new PrefillQueueLifecycle(key,
                                                   queueState,
                                                   prefillState,
                                                   prefillEp,
                                                   queueCapacity,
                                                   endpointEvents,
                                                   () -> setCapacityBlockedHeadUnderLock(null),
                                                   this::signalCapacityAvailable,
                                                   this::runLoop);
        this.enqueuer = new PrefillQueueEnqueuer(queueState,
                                                 prefillState,
                                                 prefillEp,
                                                 queueCapacity,
                                                 endpointEvents,
                                                 config);
    }

    public void start() {
        lifecycle.start();
    }

    /**
     * 停止调度线程，把还在排队的请求逐个交还给请求调度器，并等调度线程退出。
     * 返回清理过程中遇到的第一个异常，没有则返回 null。
     */
    public Throwable stopAndAwait() {
        return lifecycle.stopAndAwait();
    }

    public PrefillQueueEnqueueResult offerWithResult(ScheduledRequest exactItem) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "incoming item");
        if (!lifecycle.isRunning() || queueState.isStopped()) {
            return PrefillQueueEnqueueResult.REJECTED;
        }
        return enqueuer.enqueue(item);
    }

    public int queueSize() {
        queueState.lock();
        try {
            return queueState.activeIndex().size();
        } finally {
            queueState.unlock();
        }
    }

    public long oldestSubQueueWaitMs() {
        queueState.lock();
        try {
            long oldest = Long.MAX_VALUE;
            for (ScheduledRequest item : queueState.activeIndex()) {
                long enteredAt = item.subQueueEnteredAtNanos();
                if (enteredAt != 0L) {
                    oldest = Math.min(oldest, enteredAt);
                }
            }
            return oldest == Long.MAX_VALUE ? 0L
                    : TimeUnit.NANOSECONDS.toMillis(Math.max(0L, System.nanoTime() - oldest));
        } finally {
            queueState.unlock();
        }
    }

    /**
     * Snapshot of the current active queue depth bucketed by normalized
     * scheduling priority. Only priorities present in the queue appear in the
     * result, matching the tagged queue-depth metric behavior.
     */
    public Map<Integer, Integer> queueSizeByPriority() {
        Map<Integer, Integer> sizeByPriority = new HashMap<>();
        queueState.lock();
        try {
            for (int priority = 0; priority <= PriorityNormalizer.MAX_PRIORITY; priority++) {
                int count = queueState.activeIndex().size(priority);
                if (count > 0) { sizeByPriority.put(priority, count); }
            }
        } finally {
            queueState.unlock();
        }
        return sizeByPriority;
    }

    public PrefillState ownedState() {
        return prefillState;
    }

    /** Immutable delivery semantics used by a pure route estimate. */
    public PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule() {
        return ttftEstimator.ttftEstimationRule();
    }

    /**
     * Capture the exact active head whose worker wait predicate still holds.
     * Delivery-only waits retain null semantics so publication can continue.
     *
     * <p>The availability read is the already-subscribed wait predicate; it
     * neither previews nor reserves capacity. Caller holds the queue lock.
     */
    private AdmissionBlock admissionBlockUnderLock() {
        if (!queueState.isLockedByCurrentThread()) {
            throw new IllegalStateException(
                    "capacity block snapshot requires the queue lock");
        }
        BatcherCycleResult blocked = capacityBlockedHead;
        if (blocked == null
                || queueState.activeIndex().peek() != blocked.request()
                || blocked.request().requestExpired(now())
                || blocked.unavailable().availability().isAvailable()) {
            return null;
        }
        return new AdmissionBlock(
                blocked.request().requestId(),
                blocked.request().enqueueSeq(),
                blocked.unavailable().estimateSemantics());
    }

    // ==================== endpoint queue operations ====================

    public boolean removeQueued(
            ScheduledRequest exactItem,
            String reason) {
        ScheduledRequest item = exactItem;
        requireExactEndpoint(item, "queued item");
        boolean removed;
        queueState.lock();
        try {
            removed = lifecycle.isRunning()
                    && !queueState.isStopped()
                    && removeUnderLock(item);
            if (removed) {
                queueState.signalChange();
            }
        } finally {
            queueState.unlock();
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
        return snapshot == null ? QueueWaitSnapshot.INITIAL_DIAGNOSTICS : snapshot.diagnostics();
    }

    /** Capture fixed-size counters without formatting the PV record on the scheduling loop. */
    private void recordQueueWait(ScheduledRequest head, String reason) {
        int[] requestCountsByPriority = new int[PriorityNormalizer.MAX_PRIORITY + 1];
        long version;
        long prefillRequests;
        int prefillBatchSlots;
        queueState.lock();
        try {
            version = queueState.queueVersion();
            prefillRequests = prefillState.observedRequestCount();
            prefillBatchSlots = prefillState.batchLeasesInUseUnderLock();
            for (int priority = 0; priority < requestCountsByPriority.length; priority++) {
                requestCountsByPriority[priority] = queueState.activeIndex().size(priority);
            }
        } finally {
            queueState.unlock();
        }
        DecodeRoutingView decode = head.decodeEp() == null ? null : head.decodeEp().routingView();
        waitSnapshot = new QueueWaitSnapshot(key, reason, now(), version, requestCountsByPriority,
                prefillRequests, prefillBatchSlots, decode);
    }

    public QueuedRequestsSnapshot captureQueueSnapshot() {
        queueState.lock();
        try {
            return new QueuedRequestsSnapshot(
                    key,
                    queueState.queueVersion(),
                    activeItemsInSchedulingOrder());
        } finally {
            queueState.unlock();
        }
    }

    // ==================== Queue ownership and load view ====================

    private static long now() {
        return System.currentTimeMillis();
    }

    /** Caller holds the queue lock. */
    private boolean removeTerminalActiveUnderLock(ScheduledRequest item) {
        boolean removed;
        if (!item.requiresRouteReservation()) {
            removed = prefillState.terminalizeActiveUnderLock(item);
        } else {
            PrefillState.RouteReservation reservation =
                    item.takePublishedRouteReservation();
            if (reservation == null) {
                return false;
            }
            try {
                removed = prefillState.terminalizeActiveRouteUnderLock(
                        item, reservation);
            } catch (Throwable failure) {
                restoreRouteReservation(item, reservation);
                throw failure;
            }
            if (!removed) {
                restoreRouteReservation(item, reservation);
            }
        }
        if (removed) {
            prefillEndpoint.reportSubQueueResidence(item);
            queueCapacity.releaseEngineSubQueueSlot(prefillEndpoint.getStatus().getRole());
        }
        return removed;
    }

    static void restoreRouteReservation(
            ScheduledRequest item,
            PrefillState.RouteReservation reservation) {
        if (!item.restorePublishedRouteReservation(reservation)) {
            throw new IllegalStateException(
                    "ACTIVE route reservation could not be restored request_id="
                            + item.requestId());
        }
    }

    /** Caller holds the queue lock. */
    private boolean removeUnderLock(ScheduledRequest item) {
        boolean removed = removeTerminalActiveUnderLock(item);
        if (removed) {
            queueState.incrementQueueVersion();
        }
        return removed;
    }

    private List<ScheduledRequest> activeItemsInSchedulingOrder() {
        return queueState.activeIndex().capture().items();
    }

    private ActiveQueueSnapshot snapshotActiveQueueHead() {
        queueState.lock();
        try {
            ScheduledRequest head = queueState.activeIndex().peek();
            return new ActiveQueueSnapshot(
                    queueState.queueVersion(), queueState.schedulingInputVersion(),
                    head == null ? List.of() : List.of(head));
        } finally {
            queueState.unlock();
        }
    }

    private ActiveQueueSnapshot snapshotActiveQueue() {
        long version;
        long inputVersion;
        List<ScheduledRequest> items;
        queueState.lock();
        try {
            version = queueState.queueVersion();
            inputVersion = queueState.schedulingInputVersion();
            if (queueState.activeIndex().isEmpty()) {
                return new ActiveQueueSnapshot(
                        version, inputVersion, List.of());
            }
            items = queueState.activeIndex().capture().items();
        } finally {
            queueState.unlock();
        }
        return new ActiveQueueSnapshot(version, inputVersion, items);
    }

    private record ActiveQueueSnapshot(
            long queueVersion,
            long schedulingInputVersion,
            List<ScheduledRequest> items) {
        ScheduledRequest head() {
            return items.isEmpty() ? null : items.get(0);
        }
    }

    public PrefillTtftEstimate.LoadView captureLoadView() {
        return loadViewCache.capture();
    }

    // ==================== Delivery ownership ====================

    private BatcherCycleResult admitAndDeliverCapacityFeasiblePrefix(List<ScheduledRequest> candidates,
                                                                     String decisionReason,
                                                                     PrefillTimePredictor.Evaluator evaluator,
                                                                     OptionalLong plannedCommittedPredictionMs) {
        if (candidates.isEmpty() || !selectionStillOwned(candidates)) {
            return BatcherCycleResult.NO_ACTION;
        }
        try (DeliveryTransaction transaction =
                deliveryStrategy.prepare(new DeliveryGroup(candidates, evaluator, plannedCommittedPredictionMs))) {
            if (transaction.requests().isEmpty()) {
                return commitBoundary(transaction.blocked());
            }
            BatcherCycleResult admitted = commitPreparedSelection(
                    transaction, decisionReason);
            return admitted == null
                    ? BatcherCycleResult.NO_ACTION : admitted;
        } catch (EndpointRetiredException retired) {
            // The resources are closed, no handoff happened and the head is still queued: requeue it
            // instead of letting the exception stop the loop and fail every queued request.
            return commitBoundary(new BlockedRequest(
                    candidates.get(0), DeliveryBlockReason.failed(retired)));
        }
    }

    private void deliver(DeliveryTransaction transaction,
                         String decisionReason,
                         int remainingQueueDepth,
                         WorkView precedingWork) {
        Throwable deliveryFailure = null;
        try {
            String groupId = java.util.UUID.randomUUID().toString();
            long committedAtMs = System.currentTimeMillis();
            for (ScheduledRequest item : transaction.requests()) {
                var context = item.ctx();
                context.setDecisionGroup(new org.flexlb.dao.pv.DecisionGroup(groupId,
                        context.getConfig().decisionPolicy().getType().name(),
                        context.getConfig().getDispatcher().getType().name(),
                        item.prefillEp().getStatus().getMetricIpPort(), transaction.requests().size(), decisionReason,
                        committedAtMs, Math.max(0L, committedAtMs - item.enqueuedAtMs())));
            }
            transaction.deliver(decisionReason, remainingQueueDepth, precedingWork);
        } catch (Throwable failure) {
            deliveryFailure = failure;
        }
        Throwable unresolved = deliveryFailure != null
                ? deliveryFailure
                : new IllegalStateException(
                        "delivery returned without resolving owner");
        try {
            transaction.rollbackCommitted(unresolved);
        } catch (Throwable cleanupFailure) {
            if (deliveryFailure == null) {
                deliveryFailure = cleanupFailure;
            } else if (deliveryFailure != cleanupFailure) {
                deliveryFailure.addSuppressed(cleanupFailure);
            }
        }
        if (deliveryFailure != null) {
            Logger.error("PrefillQueue[{}] committed delivery failed",
                    key, deliveryFailure);
        }
    }

    private BatcherCycleResult runPredictionBound(
            ScheduledRequest exactHead,
            Supplier<BatcherCycleResult> operation) {
        try {
            return operation.get();
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(new BlockedRequest(
                    exactHead, DeliveryBlockReason.failed(failure)));
        }
    }

    private boolean selectionStillOwned(
            List<ScheduledRequest> candidates) {
        queueState.lock();
        try {
            if (queueState.isStopped()) {
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
            queueState.unlock();
        }
    }

    private BatcherCycleResult commitPreparedSelection(DeliveryTransaction transaction, String decisionReason) {
        List<ScheduledRequest> items = transaction.requests();
        if (items.isEmpty()) {
            throw new IllegalStateException(
                    "prepared selection commit requires a non-empty selection");
        }
        String committedReason = null;
        int remainingQueueDepth = 0;
        boolean removedTerminalBoundary = false;
        Throwable postCommitFailure = null;
        WorkView precedingWork;
        queueState.lock();
        try {
            if (queueState.isStopped()) {
                return null;
            }
            long nowMs = now();
            for (ScheduledRequest item : items) {
                if (!containsIdentity(item)
                        || item.requestExpired(nowMs)) {
                    return null;
                }
            }
            precedingWork = transaction.commit();
            // Committed requests no longer count toward either waiting limit.
            var routingRole = prefillEndpoint.getStatus().getRole();
            for (ScheduledRequest item : items) {
                prefillEndpoint.reportSubQueueResidence(item);
                queueCapacity.releaseEngineSubQueueSlot(routingRole);
                queueCapacity.decreaseTotalQueueCount(item.future());
            }
            try {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        transaction.blocked(), nowMs);
                queueState.incrementQueueVersion();
                committedReason = transaction.blocked() != null
                        && transaction.blocked().reason().unavailable()
                        ? "delivery_capacity_prefix" : decisionReason;
                remainingQueueDepth = queueState.activeIndex().size();
            } catch (Throwable failure) {
                postCommitFailure = failure;
            }
        } finally {
            queueState.unlock();
        }

        if (postCommitFailure != null) {
            Throwable failure = postCommitFailure;
            if (removedTerminalBoundary) {
                signalCapacityAvailable();
            }
            try {
                notifyTerminalAdmissionFailure(
                        removedTerminalBoundary, transaction.blocked());
            } catch (Throwable notificationFailure) {
                failure = appendFailure(failure, notificationFailure);
            }
            try {
                transaction.rollbackCommitted(failure);
            } catch (Throwable ownerFailure) {
                failure = appendFailure(failure, ownerFailure);
            }
            throw propagateCommitFailure(failure);
        }
        if (removedTerminalBoundary) {
            signalCapacityAvailable();
        }
        notifyTerminalAdmissionFailure(
                removedTerminalBoundary, transaction.blocked());
        deliver(transaction, committedReason, remainingQueueDepth, precedingWork);
        return BatcherCycleResult.CAPACITY_CHANGED;
    }

    private BatcherCycleResult commitBoundary(BlockedRequest blocked) {
        BatcherCycleResult result = BatcherCycleResult.NO_ACTION;
        boolean removedTerminalBoundary = false;
        queueState.lock();
        try {
            if (queueState.isStopped()) {
                return result;
            }
            long nowMs = now();
            if (blocked != null && blocked.reason().unavailable()) {
                result = resolveEmptyCapacityUnderLock(
                        blocked.item(), blocked.reason(), nowMs);
            } else {
                removedTerminalBoundary = removeSelectionBoundaryUnderLock(
                        blocked, nowMs);
                if (removedTerminalBoundary) {
                    queueState.incrementQueueVersion();
                    result = BatcherCycleResult.CAPACITY_CHANGED;
                }
            }
        } finally {
            queueState.unlock();
        }
        if (removedTerminalBoundary) {
            signalCapacityAvailable();
        }
        notifyTerminalAdmissionFailure(removedTerminalBoundary, blocked);
        return result;
    }

    /** Caller holds the queue lock. */
    private BatcherCycleResult resolveEmptyCapacityUnderLock(
            ScheduledRequest item,
            DeliveryBlockReason unavailable,
            long nowMs) {
        if (queueState.activeIndex().peek() != item
                || item.requestExpired(nowMs)
                || !containsIdentity(item)) {
            return BatcherCycleResult.NO_ACTION;
        }
        return BatcherCycleResult.capacityBlocked(item, unavailable);
    }

    /** Caller holds the queue lock. */
    private boolean removeSelectionBoundaryUnderLock(
            BlockedRequest blocked,
            long nowMs) {
        if (blocked == null
                || blocked.reason().unavailable()
                || blocked.reason() == DeliveryBlockReason.OWNERSHIP_LOST) {
            return false;
        }
        ScheduledRequest blockedItem = blocked.item();
        if (!containsIdentity(blockedItem)
                || blockedItem.requestExpired(nowMs)
                || !removeTerminalActiveUnderLock(blockedItem)) {
            return false;
        }
        return true;
    }

    private void notifyTerminalAdmissionFailure(
            boolean removed,
            BlockedRequest blocked) {
        if (removed && blocked.reason().status() == DeliveryBlockReason.Status.FAILED) {
            notifyAdmissionFailure(blocked.item(), blocked.reason().cause());
        }
    }

    private boolean containsIdentity(ScheduledRequest expected) {
        for (ScheduledRequest item : queueState.activeIndex()) {
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
            Logger.error("PrefillQueue[{}] delivery-failure callback failed "
                            + "request_id={}",
                    key, item.requestId(), callbackFailure);
        }
    }

    private void dropHead(ScheduledRequest head) {
        boolean removed;
        queueState.lock();
        try {
            removed = removeUnderLock(head);
        } finally {
            queueState.unlock();
        }
        if (!removed) {
            return;
        }
        signalCapacityAvailable();
        try {
            endpointEvents.onQueuedItemExpired(head);
        } catch (Throwable callbackFailure) {
            Logger.error("PrefillQueue[{}] ACTIVE terminal callback failed "
                            + "request_id={} reason={}",
                    key, head.requestId(), "request_expired",
                    callbackFailure);
        }
    }

    static Throwable appendFailure(
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

    /** Release the publish-time reservation only while it is still item-owned. */
    static RuntimeException propagateCommitFailure(
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
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        if (!shape.fitsKv(capacity.kvCapacityFor(shape))) {
            return awaitPrefillKvCapacity(
                    head, snapshot.queueVersion(),
                    snapshot.schedulingInputVersion());
        }

        // Worker status and delivery ledgers advance independently. Re-read
        // the advisory gate immediately before hard-capacity admission.
        capacity = BatchCapacityView.of(prefillEndpoint, config);
        if (!shape.fitsKv(capacity.kvCapacityFor(shape))) {
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
        if (queueState.activeIndex().isEmpty()) {
            return BatcherCycleResult.NO_ACTION;
        }

        // Cheap advisory gates avoid sorting while the worker cannot dispatch.
        ScheduledRequest observedHead;
        int observedSize;
        long observedOldestEnqueuedAtMs = Long.MAX_VALUE;
        long observedQueueVersion;
        long observedSchedulingInputVersion;
        queueState.lock();
        try {
            observedHead = queueState.activeIndex().peek();
            observedSize = queueState.activeIndex().size();
            observedQueueVersion = queueState.queueVersion();
            observedSchedulingInputVersion = queueState.schedulingInputVersion();
            for (ScheduledRequest item : queueState.activeIndex()) {
                observedOldestEnqueuedAtMs = Math.min(
                        observedOldestEnqueuedAtMs, item.enqueuedAtMs());
            }
        } finally {
            queueState.unlock();
        }
        if (observedHead == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        long nowMs = now();
        long fixedWaitMs = groupingLimits.collectionWindowMs();
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        int batchMaxCount = groupingLimits.maxRequests();
        long predictThresholdMs = groupingLimits.predictedExecutionBudgetMs();
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
                    observedOldestEnqueuedAtMs, fixedWaitMs);
        }

        GroupPlanner.Shape observedHeadShape = GroupPlanner.Shape.empty().add(observedHead.seqLen());
        long batchKvTokens = capacity.kvCapacityFor(observedHeadShape);
        if (!observedHeadShape.fitsKv(batchKvTokens)) {
            return awaitPrefillKvCapacity(
                    observedHead,
                    observedQueueVersion,
                    observedSchedulingInputVersion);
        }

        // The ordered snapshot is the selection linearization point;
        // prediction intentionally runs after releasing the queue lock.
        ActiveQueueSnapshot snapshot = snapshotActiveQueue();
        ScheduledRequest head = snapshot.head();
        if (head == null) {
            return BatcherCycleResult.NO_ACTION;
        }

        nowMs = now();
        capacity = BatchCapacityView.of(prefillEndpoint, config);
        batchMaxCount = groupingLimits.maxRequests();
        predictThresholdMs = groupingLimits.predictedExecutionBudgetMs();
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
        batchKvTokens = capacity.kvCapacityFor(headShape);
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
            PrefillTimePredictor.Evaluator evaluator = prefillEndpoint.getPredictor().evaluator();
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
                                    ttftEstimator.newGroupPredictor(planningEvaluator));
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
                    plan.windowOpenedAtMs(), exactFixedWaitMs);
        });
    }

    private static BatcherCycleResult awaitCollectionWindow(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion,
            long windowOpenedAtMs,
            long collectionWindowMs) {
        long collectionDeadline = GroupPlanner.collectionDeadlineMs(
                windowOpenedAtMs, collectionWindowMs);
        return BatcherCycleResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                Math.min(collectionDeadline, head.expiresAtMs()), "Prefill collection window");
    }

    private static BatcherCycleResult awaitPrefillKvCapacity(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion) {
        return BatcherCycleResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                head.expiresAtMs(), "Prefill KV capacity exhausted");
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

    private BatcherCycleResult admitDecisionGroup(List<ScheduledRequest> picked,
                                                  GroupPlanner.Shape shape,
                                                  String reason,
                                                  PrefillTimePredictor.Evaluator evaluator,
                                                  OptionalLong plannedCommittedPredictionMs) {
        BatchCapacityView capacity = BatchCapacityView.of(prefillEndpoint, config);
        if ((picked.size() > 1 && !shape.fitsCompute(capacity.batchTokenCapacity()))
                || !shape.fitsKv(capacity.kvCapacityFor(shape))) {
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
            while (!queueState.isStopped() && !Thread.currentThread().isInterrupted()) {
                try {
                    runOneCycle();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Throwable t) {
                    // Every expected delivery failure is terminalized inside the
                    // typed cycle. An escaping Throwable is therefore an invariant
                    // failure; retrying the same ACTIVE state can only spin.
                    lifecycle.stopAfterUnexpectedLoopFailure(t);
                    return;
                }
            }
        } finally {
            lifecycle.markLoopFinished();
        }
    }

    private void runOneCycle() throws InterruptedException {
        waitForNonEmpty();
        if (queueState.isStopped()) {
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
        queueState.lockInterruptibly();
        try {
            while (!queueState.isStopped() && queueState.activeIndex().isEmpty()) {
                setCapacityBlockedHeadUnderLock(null);
                queueState.awaitChange();
            }
        } finally {
            queueState.unlock();
        }
    }

    /**
     * Wait while the exact active head is blocked by the exact rejecting
     * resource. Offers wake the condition: a new higher-priority head exits the
     * wait immediately, while a tail offer leaves the predicate true. Capacity
     * is checked after listener installation while the queue lock is held, closing
     * the release-before-await race without polling.
     */
    private void awaitBlockedHeadCapacity(
            BatcherCycleResult blocked)
            throws InterruptedException {
        queueState.lockInterruptibly();
        try {
            if (queueState.isStopped()) {
                return;
            }
            setCapacityBlockedHeadUnderLock(blocked);
            while (!queueState.isStopped()
                    && queueState.activeIndex().peek() == blocked.request()
                    && !blocked.request().requestExpired(System.currentTimeMillis())
                    && !blocked.unavailable().availability().isAvailable()) {
                long expiresAtMs = blocked.request().expiresAtMs();
                long nowMs = System.currentTimeMillis();
                if (expiresAtMs <= nowMs) {
                    return;
                }
                if (expiresAtMs == Long.MAX_VALUE) {
                    queueState.awaitChange();
                } else {
                    queueState.awaitChangeFor(expiresAtMs - nowMs);
                }
            }
        } finally {
            try {
                setCapacityBlockedHeadUnderLock(null);
            } finally {
                queueState.unlock();
            }
        }
    }

    /**
     * Wait for the exact queue/input generation captured by the policy or
     * for its deadline. Every predicate and signal is serialized by the queue lock.
     */
    private void awaitSchedulingChange(
            BatcherCycleResult waiting)
            throws InterruptedException {
        queueState.lockInterruptibly();
        try {
            while (!queueState.isStopped()
                    && queueState.activeIndex().peek() == waiting.request()
                    && queueState.queueVersion() == waiting.queueVersion()
                    && queueState.schedulingInputVersion()
                    == waiting.schedulingInputVersion()) {
                long nowMs = System.currentTimeMillis();
                if (waiting.wakeAtMs() <= nowMs) {
                    return;
                }
                if (waiting.wakeAtMs() == Long.MAX_VALUE) {
                    queueState.awaitChange();
                } else {
                    queueState.awaitChangeFor(waiting.wakeAtMs() - nowMs);
                }
            }
        } finally {
            queueState.unlock();
        }
    }

    /** Called after Prefill or Decode capacity changes, outside endpoint locks. */
    public void signalDeliveryCapacityAvailable() {
        queueState.lock();
        try {
            if (capacityBlockedHead != null) {
                queueState.incrementSchedulingInputVersionUnderLock();
                queueState.signalChange();
            }
        } finally {
            queueState.unlock();
        }
    }

    /** Publish a real Prefill ownership or batch capacity change to local and global waiters. */
    private void signalCapacityAvailable() {
        signalDeliveryCapacityAvailable();
        prefillEndpoint.signalPlacementCapacityChanged();
    }

    /** Own the blocked head, its listener and estimate invalidation under the queue lock. */
    private void setCapacityBlockedHeadUnderLock(
            BatcherCycleResult blocked) {
        if (!queueState.isLockedByCurrentThread()) {
            throw new IllegalStateException(
                    "capacity block update requires the queue lock");
        }
        if (capacityBlockedHead == blocked) {
            return;
        }
        DeliveryBlockReason.Availability previousSource = capacityBlockedHead == null
                ? null : capacityBlockedHead.unavailable().availability();
        DeliveryBlockReason.Availability nextSource = blocked == null
                ? null : blocked.unavailable().availability();
        capacityBlockedHead = blocked;
        queueState.incrementSchedulingInputVersionUnderLock();
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
        queueState.lock();
        try {
            queueState.incrementSchedulingInputVersionUnderLock();
            queueState.signalChange();
        } finally {
            queueState.unlock();
        }
    }

}
