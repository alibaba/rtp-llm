package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillState;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.QueueSnapshot.AdmissionBlock;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * Controlled access to shared {@link WorkerBatcher} infrastructure.
 *
 * <p>Passed to {@link GroupPolicy} methods so policies can
 * inspect and mutate the queue, read config, and invoke callbacks
 * without directly depending on WorkerBatcher internals.
 *
 * <p>Every queue mutation is performed under the shared queue lock and bumps
 * the queue version, keeping the priority scheduling invariant "version unchanged ⇒
 * queue content unchanged" (optimistic plan validation).
 */
class BatcherContext {

    /** Compute and KV limits derived from one WorkerStatus publication. */
    record BatchCapacitySnapshot(
            long batchTokenCapacity,
            long batchKvCapacity) {
    }

    /*
     * Queue ownership state machine:
     *
     * ACTIVE (queue)
     *   -> CALLBACK_OWNED (one nested delivery owner after capacity reservation)
     *   -> terminal endpoint lifecycle (no batcher container)
     *
     * Capacity failure leaves the ordered head ACTIVE. A capacity-feasible
     * prefix moves directly to callback ownership as the final logical
     * decision; callback failure is terminal.
     */

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig config;
    private final DecisionPolicyConfig fixedWindowDecision;
    private final EndpointEventSink deliveryLifecycle;
    private final PriorityBlockingQueue<ScheduledRequest> queue;
    private final AtomicLong queueVersion;
    private final AtomicLong schedulingInputVersion = new AtomicLong();
    private final ReentrantLock queueLock;
    private final Comparator<ScheduledRequest> queueOrder;
    private final Comparator<GroupPlanner.Item> projectionOrder;
    private final boolean queueScheduling;
    private final DeliveryCoordinator delivery;
    private final PrefillState prefillState;

    private boolean stopped;

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig config,
                   EndpointEventSink deliveryLifecycle,
                   PriorityBlockingQueue<ScheduledRequest> queue,
                   AtomicLong queueVersion,
                   ReentrantLock queueLock,
                   Comparator<ScheduledRequest> queueOrder,
                   Comparator<GroupPlanner.Item> projectionOrder,
                   boolean queueScheduling,
                   DeliveryStrategy deliveryStrategy,
                   PrefillState prefillState) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.config = config;
        DecisionPolicyConfig resolvedDecision = config.isQueue()
                ? config.decisionPolicy() : null;
        this.fixedWindowDecision = resolvedDecision != null
                && resolvedDecision.getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW
                ? resolvedDecision : null;
        this.deliveryLifecycle = Objects.requireNonNull(
                deliveryLifecycle, "deliveryLifecycle");
        this.queue = queue;
        this.queueVersion = queueVersion;
        this.queueLock = queueLock;
        this.queueOrder = queueOrder;
        this.projectionOrder = Objects.requireNonNull(
                projectionOrder, "projectionOrder");
        this.queueScheduling = queueScheduling;
        this.delivery = new DeliveryCoordinator(key, deliveryStrategy);
        this.prefillState = Objects.requireNonNull(prefillState, "prefillState");
    }

    // ---- accessors ----

    String key() {
        return key;
    }

    PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    int maxQueueCapacity() {
        return config.queueScheduler().getCapacity()
                .getMaxWaitingRequestsPerPrefillWorker();
    }

    /**
     * Largest group the queue's decision policy may release at once. Delivery
     * ownership is deliberately not consulted here.
     */
    int maxDecisionRequests() {
        return fixedWindowDecision == null
                ? 1 : Math.max(1, fixedWindowDecision.getMaxRequests());
    }

    /**
     * How long an incomplete group may wait for the arrivals that would
     * complete it. A single-request group is never incomplete, so it has no
     * window to spend.
     */
    long collectionWindowMs() {
        return fixedWindowDecision == null
                ? 0L : Math.max(0L, fixedWindowDecision.getMaxCollectionWaitMs());
    }

    /**
     * Predicted-execution budget capping group growth, or {@code 0} when no
     * budget applies. The SINGLE policy has no prediction budget; a FIXED_WINDOW
     * singleton remains indivisible but reaching the budget still releases it.
     */
    long predictedExecutionBudgetMs() {
        if (fixedWindowDecision == null) {
            return 0L;
        }
        Long configured = fixedWindowDecision.getMaxPredictedExecutionMs();
        return configured == null ? 0L : configured;
    }

    long now() {
        return System.currentTimeMillis();
    }

    ReentrantLock queueLock() {
        return queueLock;
    }

    long schedulingInputVersionValue() {
        return schedulingInputVersion.get();
    }

    /** Caller holds the shared queue lock. */
    void incrementSchedulingInputVersion() {
        schedulingInputVersion.incrementAndGet();
    }

    // ---- queue inspection ----

    ScheduledRequest peek() {
        return queue.peek();
    }

    /**
     * Whether the active queue still has work requiring a logical decision.
     * A zero charged depth already implies a physically empty queue, so the
     * container is only consulted once something is charged.
     */
    boolean isActiveEmpty() {
        return queue.isEmpty();
    }

    int size() {
        return queue.size();
    }

    boolean hasProcessableWork() {
        return !queue.isEmpty();
    }

    // ---- queue mutation ----

    /** Caller owns one previously reserved queue-depth unit and holds queueLock. */
    boolean publishActiveIndexUnderLock(ScheduledRequest item) {
        boolean published = prefillState.enqueueActiveUnderLock(item);
        if (published) {
            queueVersion.incrementAndGet();
        }
        return published;
    }

    /** Claim one ACTIVE request for a terminal reducer. */
    private boolean removeTerminalActiveUnderLock(ScheduledRequest item) {
        return prefillState.terminalizeActiveUnderLock(item);
    }

    /**
     * Close this generation's ACTIVE admission gate before shutdown starts
     * claiming exact queue identities. Caller holds {@code queueLock}.
     */
    void stopAcceptingUnderLock() {
        if (!queueLock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "queue mutation requires queueLock");
        }
        stopped = true;
    }

    /** Detach one exact queue head while retaining its stop-terminal owner. */
    ScheduledRequest detachNextStopTerminalUnderLock() {
        ScheduledRequest item = prefillState.detachNextActiveForStopUnderLock();
        if (item != null) {
            queueVersion.incrementAndGet();
        }
        return item;
    }

    /** Acknowledge only the exact retained owner whose callback completed. */
    boolean acknowledgeStopTerminalUnderLock(ScheduledRequest item) {
        return prefillState.acknowledgeStopTerminalUnderLock(item);
    }

    /** Caller holds queueLock. */
    boolean removeUnderLock(ScheduledRequest item) {
        boolean removed = removeTerminalActiveUnderLock(item);
        if (removed) {
            queueVersion.incrementAndGet();
        }
        return removed;
    }

    /**
     * Items in active queue order (FIFO: {@link ScheduledRequest#enqueueSeq()};
     * PRIORITY: {@link WorkerBatcher#PRIORITY_QUEUE_ORDER}, which delegates
     * to {@link PriorityOrdering#STRICT}), suitable for greedy-fill iteration
     * in grouping algorithms.
     */
    List<ScheduledRequest> activeItemsInSchedulingOrder() {
        List<ScheduledRequest> candidates = new ArrayList<>(queue);
        candidates.sort(queueOrder);
        return candidates;
    }

    /**
     * Capture the active-queue head and its versions for a single-request
     * decision. The queue already exposes its least element, so this needs
     * neither a copy nor a sort.
     */
    ActiveQueueSnapshot snapshotActiveQueueHead() {
        queueLock.lock();
        try {
            long version = queueVersion.get();
            long inputVersion = schedulingInputVersion.get();
            ScheduledRequest head = queue.peek();
            return new ActiveQueueSnapshot(
                    version, inputVersion,
                    head == null ? List.of() : List.of(head));
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Capture one stable, fully ordered active-queue snapshot for a batching
     * decision. The version and identities are linearized under the same lock;
     * prediction intentionally runs after the lock is released.
     */
    ActiveQueueSnapshot snapshotActiveQueue() {
        long version;
        long inputVersion;
        List<ScheduledRequest> items;
        queueLock.lock();
        try {
            version = queueVersion.get();
            inputVersion = schedulingInputVersion.get();
            if (queue.isEmpty()) {
                return new ActiveQueueSnapshot(version, inputVersion, List.of());
            }
            items = new ArrayList<>(queue);
        } finally {
            queueLock.unlock();
        }
        // Ordering keys are immutable after offer. Sorting the frozen identity
        // copy outside queueLock preserves the snapshot while keeping O(N logN)
        // comparator work out of the offer/remove critical section.
        items.sort(queueOrder);
        return new ActiveQueueSnapshot(version, inputVersion, items);
    }

    record ActiveQueueSnapshot(long queueVersion,
                               long schedulingInputVersion,
                               List<ScheduledRequest> items) {
        ScheduledRequest head() {
            return items.isEmpty() ? null : items.get(0);
        }
    }

    /** Capture queue, committed work and pending count under the ownership lock. */
    RouteProjection.Inputs captureRouteProjectionInputs(
            Supplier<AdmissionBlock> admissionBlockSnapshot) {
        queueLock.lock();
        try {
            BatchCapacitySnapshot capacity = batchCapacitySnapshot();
            PrefillState.Snapshot ownership =
                    prefillState.snapshotUnderLock(queueOrder);
            List<GroupPlanner.Item> items =
                    ownership.activeItems().stream()
                            .map(BatcherContext::projectionItem)
                            .toList();
            QueueSnapshot queueSnapshot = new QueueSnapshot(
                    ownership.capturedAtMs(),
                    queueScheduling,
                    projectionOrder,
                    new GroupPlanner.Constraints(
                            maxDecisionRequests(),
                            capacity.batchTokenCapacity(),
                            capacity.batchKvCapacity(),
                            predictedExecutionBudgetMs(),
                            collectionWindowMs()),
                    items,
                    items.isEmpty() ? null : admissionBlockSnapshot.get());
            return new RouteProjection.Inputs(
                    queueSnapshot,
                    ownership.committedWork(),
                    ownership.pendingRequestCount());
        } finally {
            queueLock.unlock();
        }
    }

    private static GroupPlanner.Item projectionItem(ScheduledRequest item) {
        return new GroupPlanner.Item(
                item.requestId(),
                item.priority(),
                item.enqueueSeq(),
                item.enqueuedAtMs(),
                item.expiresAtMs(),
                item.seqLen(),
                item.hitCache());
    }

    /** Stable low-cost state used before a full ordered group snapshot is needed. */
    ActiveQueueState activeQueueState() {
        queueLock.lock();
        try {
            ScheduledRequest head = queue.peek();
            long oldestEnqueuedAtMs = Long.MAX_VALUE;
            for (ScheduledRequest item : queue) {
                oldestEnqueuedAtMs = Math.min(
                        oldestEnqueuedAtMs, item.enqueuedAtMs());
            }
            return new ActiveQueueState(
                    queueVersion.get(), schedulingInputVersion.get(),
                    head, queue.size(), oldestEnqueuedAtMs);
        } finally {
            queueLock.unlock();
        }
    }

    record ActiveQueueState(long queueVersion,
                            long schedulingInputVersion,
                            ScheduledRequest head,
                            int activeSize,
                            long oldestEnqueuedAtMs) {
    }

    BatcherCycleResult awaitingSchedulingChange(
            ScheduledRequest head,
            long observedQueueVersion,
            long observedSchedulingInputVersion,
            long wakeAtMs,
            BatcherCycleResult.SchedulingWaitReason reason) {
        return BatcherCycleResult.awaitingSchedulingChange(
                head, observedQueueVersion, observedSchedulingInputVersion,
                wakeAtMs, reason);
    }

    /**
     * Compute and KV limits from one atomically published worker status.
     *
     * <p>After admitting the first request as an indivisible candidate, the
     * Engine's FIFO scheduler rejects additions whose padded context shape
     * ({@code maxSeqLen * batchSize}) is greater than or equal to
     * {@code max_batch_tokens_size}. Prefer that exact worker-reported limit;
     * {@code max_seq_len} is a conservative fallback for workers that have not
     * populated the newer field yet. An internal safety ceiling covers the
     * interval before either value arrives.
     * A zero KV total means the worker has not published that capacity yet, so
     * the returned KV limit is unbounded.
     */
    BatchCapacitySnapshot batchCapacitySnapshot() {
        long capacity = positiveOrUnlimited(
                config.getInternalRuntime().getFallbackBatchTokenCapacity());
        WorkerStatus status = prefillEp != null ? prefillEp.getStatus() : null;
        if (status == null) {
            return new BatchCapacitySnapshot(capacity, Long.MAX_VALUE);
        }

        WorkerStatus.EngineObservation engineStatus =
                status.committedEngineObservation();
        long engineCapacity = engineStatus.maxBatchTokensSize();
        if (engineCapacity <= 0) {
            engineCapacity = engineStatus.maxSeqLen();
        }
        long batchTokenCapacity = Math.min(
                capacity, positiveOrUnlimited(engineCapacity));

        // A zero total means the worker has not published KV capacity yet, so
        // batching remains compute-bound only.
        long total = engineStatus.totalKvCacheTokens();
        if (total <= 0) {
            return new BatchCapacitySnapshot(
                    batchTokenCapacity, Long.MAX_VALUE);
        }
        long available = Math.max(0, engineStatus.availableKvCacheTokens());
        return new BatchCapacitySnapshot(
                batchTokenCapacity, Math.min(total, available));
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }

    // ---- capacity-aware decision publication ----

    /**
     * Reserve capacity for the largest ordered prefix and publish exactly that
     * prefix as the final logical decision. If the head cannot reserve capacity,
     * it remains ACTIVE and the worker waits on the rejecting resource event.
     *
     * <p>The ordered snapshot is the selection linearization point. Offers that
     * arrive after that snapshot belong to the next decision and do not revoke
     * this one. Removal or expiration of a selected member does revoke it, and
     * every provisional reservation is then released. The immutable evaluator
     * captured for this decision remains valid if online learning publishes a
     * replacement concurrently.
     */
    BatcherCycleResult admitAndDeliverCapacityFeasiblePrefix(
            List<ScheduledRequest> candidates,
            DeliveryMetadata metadata,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedCommittedPredictionMs) {
        if (candidates.isEmpty()) {
            return BatcherCycleResult.NO_ACTION;
        }
        return delivery.deliver(
                this, candidates, metadata, evaluator,
                plannedCommittedPredictionMs);
    }

    double projectGroupDurationMs(
            List<ScheduledRequest> items,
            PrefillTimePredictor.Evaluator evaluator) {
        return delivery.projectGroupDurationMs(items, evaluator);
    }

    RouteProjection.DeliveryProjection deliveryProjection() {
        return delivery.projectionPolicy();
    }

    /**
     * Reduce an invalid prediction at the exact ACTIVE request that anchored
     * the prediction. The operation owns all prediction work for one policy
     * pass, so malformed model output cannot escape into the worker-loop
     * failure path and drain unrelated requests.
     */
    BatcherCycleResult runPredictionBound(
            ScheduledRequest exactHead,
            Supplier<BatcherCycleResult> operation) {
        try {
            return operation.get();
        } catch (InvalidPrefillPredictionException failure) {
            return commitBoundary(
                    exactHead, CapacityBoundary.failed(failure));
        }
    }

    boolean selectionStillOwned(List<ScheduledRequest> candidates) {
        return candidateSelectionStillOwned(candidates);
    }

    BatcherCycleResult commitPreparedSelection(
            DeliveryStrategy.Transaction transaction,
            String decisionReason) {
        List<ScheduledRequest> items = batchItems(transaction.items());
        if (items.isEmpty()) {
            throw new IllegalStateException(
                    "prepared selection commit requires a non-empty selection");
        }
        BatcherCycleResult admittedResult = null;
        RemovedTerminalBoundary removedBoundary = RemovedTerminalBoundary.NONE;
        Throwable postCommitFailure = null;

        queueLock.lock();
        try {
            if (stopped) {
                return null;
            }
            long nowMs = now();
            for (ScheduledRequest item : items) {
                if (!containsIdentity(item) || item.requestExpired(nowMs)) {
                    return null;
                }
            }
            transaction.commitUnderLock();
            try {
                removedBoundary = removeSelectionBoundaryUnderLock(
                        transaction.blockedItem(),
                        transaction.blockedResult(),
                        nowMs);
                queueVersion.incrementAndGet();
                String committedReason = transaction.blockedResult() != null
                        && transaction.blockedResult().unavailable()
                        ? "delivery_capacity_prefix" : decisionReason;
                admittedResult = BatcherCycleResult.admitted(
                        items,
                        new DeliveryMetadata(
                                committedReason, queue.size()));
            } catch (Throwable failure) {
                postCommitFailure = failure;
            }
        } finally {
            queueLock.unlock();
        }

        if (postCommitFailure != null) {
            Throwable failure = postCommitFailure;
            try {
                notifyTerminalAdmissionFailure(removedBoundary);
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
        notifyTerminalAdmissionFailure(removedBoundary);
        return Objects.requireNonNull(admittedResult);
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

    private static RuntimeException propagateCommitFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "delivery selection failed after ownership commit", failure);
    }

    BatcherCycleResult commitBoundary(
            ScheduledRequest blockedItem,
            CapacityBoundary blockedResult) {
        BatcherCycleResult result = BatcherCycleResult.NO_ACTION;
        RemovedTerminalBoundary removedBoundary = RemovedTerminalBoundary.NONE;
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
                removedBoundary = removeSelectionBoundaryUnderLock(
                        blockedItem, blockedResult, nowMs);
                if (removedBoundary.wasRemoved()) {
                    queueVersion.incrementAndGet();
                    result = BatcherCycleResult.QUEUE_CHANGED;
                }
            }
        } finally {
            queueLock.unlock();
        }
        notifyTerminalAdmissionFailure(removedBoundary);
        return result;
    }

    /** Caller holds queueLock. */
    private BatcherCycleResult resolveEmptyCapacityUnderLock(
            ScheduledRequest item,
            CapacityBoundary unavailable,
            long nowMs) {
        if (queue.peek() != item
                || item.requestExpired(nowMs)
                || !containsIdentity(item)) {
            return BatcherCycleResult.NO_ACTION;
        }
        return BatcherCycleResult.capacityBlocked(item, unavailable);
    }

    /** Caller holds queueLock. */
    private RemovedTerminalBoundary removeSelectionBoundaryUnderLock(
            ScheduledRequest blockedItem,
            CapacityBoundary blockedResult,
            long nowMs) {
        // OwnershipLost is not a terminal fact. In particular, the request's
        // admission mutation may still be closing after publishing the exact
        // queue item. Removing it here would leave RequestState QUEUED
        // without either a queue owner or a terminal callback.
        if (blockedItem == null
                || blockedResult.unavailable()
                || blockedResult == CapacityBoundary.OWNERSHIP_LOST) {
            return RemovedTerminalBoundary.NONE;
        }
        ScheduledRequest item = blockedItem;
        if (!containsIdentity(item)
                || item.requestExpired(nowMs)
                || !removeTerminalActiveUnderLock(item)) {
            return RemovedTerminalBoundary.NONE;
        }
        Throwable failure = blockedResult.status()
                == CapacityBoundary.Status.FAILED
                ? blockedResult.cause() : null;
        return new RemovedTerminalBoundary(item, failure);
    }

    private void notifyTerminalAdmissionFailure(
            RemovedTerminalBoundary removedTerminalBoundary) {
        if (removedTerminalBoundary.failure() != null) {
            notifyAdmissionFailure(
                    removedTerminalBoundary.item(),
                    removedTerminalBoundary.failure());
        }
    }

    private record RemovedTerminalBoundary(ScheduledRequest item, Throwable failure) {
        private static final RemovedTerminalBoundary NONE =
                new RemovedTerminalBoundary(null, null);

        boolean wasRemoved() {
            return item != null;
        }
    }

    /** Delivery items are the scheduler-owned ScheduledRequest identities. */
    @SuppressWarnings("unchecked")
    private static List<ScheduledRequest> batchItems(List<ScheduledRequest> items) {
        return (List<ScheduledRequest>) (List<?>) items;
    }

    private boolean candidateSelectionStillOwned(List<ScheduledRequest> candidates) {
        queueLock.lock();
        try {
            if (stopped) {
                return false;
            }
            long nowMs = now();
            for (ScheduledRequest item : candidates) {
                if (!containsIdentity(item) || item.requestExpired(nowMs)) {
                    return false;
                }
            }
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    /** Identity membership avoids conflating different request generations. */
    private boolean containsIdentity(ScheduledRequest expected) {
        for (ScheduledRequest item : queue) {
            if (item == expected) {
                return true;
            }
        }
        return false;
    }

    private void notifyAdmissionFailure(ScheduledRequest item, Throwable cause) {
        try {
            deliveryLifecycle.onPreparedDeliveryFailure(item, cause);
        } catch (Throwable callbackFailure) {
            Logger.error("WorkerBatcher[{}] delivery-failure callback failed "
                            + "request_id={}",
                    key, item.requestId(), callbackFailure);
        }
    }

    /** Terminate the head whose absolute request expiration has been reached. */
    void dropHead(ScheduledRequest head) {
        terminateActiveItem(head, ActiveQueueExpired.INSTANCE);
    }

    /**
     * Single reducer for normal terminal exits from ACTIVE. Queue removal is
     * the ownership claim; only the winner invokes the item-scoped terminal
     * callback. Callback failure is logged and never retried or promoted to a
     * worker-wide invariant failure.
     */
    private void terminateActiveItem(
            ScheduledRequest item,
            ActiveQueueTermination termination) {
        boolean removed;
        queueLock.lock();
        try {
            removed = removeUnderLock(item);
        } finally {
            queueLock.unlock();
        }
        if (!removed) {
            return;
        }
        try {
            if (termination == ActiveQueueExpired.INSTANCE) {
                deliveryLifecycle.onQueuedItemExpired(item);
            } else {
                ActiveQueueRejected rejected = (ActiveQueueRejected) termination;
                deliveryLifecycle.onQueueOfferFailure(
                        item, rejected.cause());
            }
        } catch (Throwable callbackFailure) {
            Logger.error("WorkerBatcher[{}] ACTIVE terminal callback failed "
                            + "request_id={} reason={}",
                    key, item.requestId(), termination.reason(), callbackFailure);
        }
    }

    private sealed interface ActiveQueueTermination
            permits ActiveQueueExpired, ActiveQueueRejected {
        String reason();
    }

    private enum ActiveQueueExpired implements ActiveQueueTermination {
        INSTANCE;

        @Override
        public String reason() {
            return "request_expired";
        }
    }

    private record ActiveQueueRejected(Throwable cause)
            implements ActiveQueueTermination {
        @Override
        public String reason() {
            return "batch_token_capacity_exceeded";
        }
    }

}
