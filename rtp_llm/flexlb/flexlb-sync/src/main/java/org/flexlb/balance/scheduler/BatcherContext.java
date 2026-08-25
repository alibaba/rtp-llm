package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.FixedWindowDecisionConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Controlled access to shared {@link WorkerBatcher} infrastructure.
 *
 * <p>Passed to {@link BatcherAlgorithm} methods so algorithms can
 * inspect and mutate the queue, read config, and invoke callbacks
 * without directly depending on WorkerBatcher internals.
 *
 * <p>Every queue mutation is performed under the shared queue lock and bumps
 * the queue version, keeping the priority scheduling invariant "version unchanged ⇒
 * queue content unchanged" (optimistic plan validation).
 */
public class BatcherContext {

    /*
     * Queue ownership state machine:
     *
     * ACTIVE (queue)
     *   -> CALLBACK_OWNED (AdmittedDecisionGroup, only after capacity reservation)
     *   -> terminal endpoint lifecycle (no batcher container)
     *
     * Capacity failure leaves the ordered head ACTIVE. A capacity-feasible
     * prefix moves directly to callback ownership as the final logical
     * decision; callback failure is terminal.
     */

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig config;
    private final FixedWindowDecisionConfig fixedWindowDecision;
    private final DecisionGroupHandler decisionHandler;
    private final DeliveryCapacityAdmission capacityAdmission;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final AtomicInteger queueDepth;
    private final AtomicLong queueVersion;
    private final AtomicLong schedulingInputVersion = new AtomicLong();
    private final ReentrantLock queueLock;
    private final Comparator<BatchItem> queueOrder;
    private final BatchSchedulerReporter reporter;

    /** Requests synchronously owned by the admitted-delivery callback. */
    private final AtomicInteger callbackOwnedRequestCount = new AtomicInteger();
    private boolean stopped;

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig config,
                   DecisionGroupHandler decisionHandler,
                   DeliveryCapacityAdmission capacityAdmission,
                   PriorityBlockingQueue<BatchItem> queue,
                   AtomicInteger queueDepth,
                   AtomicLong queueVersion,
                   ReentrantLock queueLock,
                   Comparator<BatchItem> queueOrder,
                   BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.config = config;
        DecisionPolicyConfig resolvedDecision = config.isQueue()
                ? config.decisionPolicy() : null;
        this.fixedWindowDecision = resolvedDecision instanceof FixedWindowDecisionConfig fixed
                ? fixed : null;
        this.decisionHandler = decisionHandler;
        this.capacityAdmission = capacityAdmission;
        this.queue = queue;
        this.queueDepth = queueDepth;
        this.queueVersion = queueVersion;
        this.queueLock = queueLock;
        this.queueOrder = queueOrder;
        this.reporter = reporter;
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

    BatchSchedulerReporter reporter() {
        return reporter;
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

    Comparator<BatchItem> queueOrder() {
        return queueOrder;
    }

    // ---- queue inspection ----

    BatchItem peek() {
        return queue.peek();
    }

    /**
     * Whether the active queue still has work requiring a logical decision.
     * A zero charged depth already implies a physically empty queue, so the
     * container is only consulted once something is charged.
     */
    boolean isActiveEmpty() {
        return queueDepth.get() == 0 || queue.isEmpty();
    }

    /**
     * Physical active decision-queue depth. Capacity-blocked work remains in
     * this queue until an ordered prefix can reserve everything it needs.
     */
    int activeSize() {
        return queue.size();
    }

    int size() {
        return queueDepth.get();
    }

    boolean hasProcessableWork() {
        return !queue.isEmpty();
    }

    // ---- queue mutation ----

    boolean remove(BatchItem item) {
        queueLock.lock();
        try {
            boolean removed = queue.remove(item);
            if (removed) {
                queueDepth.decrementAndGet();
                queueVersion.incrementAndGet();
            }
            return removed;
        } finally {
            queueLock.unlock();
        }
    }

    void drainTo(List<BatchItem> dst) {
        queueLock.lock();
        try {
            int drained = queue.drainTo(dst);
            if (drained > 0) {
                queueDepth.addAndGet(-drained);
                queueVersion.incrementAndGet();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Thread-confined unordered copy of the active queue. The caller must
     * hold {@link #queueLock()} so the copy and any version capture taken
     * under the same lock hold linearize with queue mutations. Sorting the
     * returned copy is safe outside the lock because item ordering keys are
     * frozen once the item is constructed ({@code enqueueSequence} is
     * assigned at construction, before the item can enter the queue) —
     * the same invariant {@link #snapshotActiveQueue(int)} relies on.
     */
    List<BatchItem> copiedItems() {
        return new ArrayList<>(queue);
    }

    /**
     * Items in active queue order (FIFO: {@link BatchItem#enqueueSeq()};
     * PRIORITY: {@link WorkerBatcher#PRIORITY_QUEUE_ORDER}, which delegates
     * to {@link PriorityOrdering#STRICT}), suitable for greedy-fill iteration
     * in grouping algorithms.
     */
    List<BatchItem> activeItemsInSchedulingOrder() {
        List<BatchItem> candidates = new ArrayList<>(queue);
        candidates.sort(queueOrder);
        return candidates;
    }

    /**
     * Capture one stable, ordered active-queue snapshot for a batching
     * decision. The version and identities are linearized under the same lock;
     * prediction intentionally runs after the lock is released.
     *
     * <p>{@code maxItems} is the largest group the decision can release. A
     * single-request group needs only the ordering head, which the queue
     * already exposes without a copy.
     */
    ActiveQueueSnapshot snapshotActiveQueue(int maxItems) {
        long version;
        long inputVersion;
        List<BatchItem> items;
        queueLock.lock();
        try {
            version = queueVersion.get();
            inputVersion = schedulingInputVersion.get();
            if (maxItems <= 1) {
                // The ordering head is the queue's own least element, so a
                // single-request decision needs neither a copy nor a sort.
                BatchItem head = queue.peek();
                return new ActiveQueueSnapshot(
                        version, inputVersion,
                        head == null ? List.of() : List.of(head));
            }
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
                               List<BatchItem> items) {
        BatchItem head() {
            return items.isEmpty() ? null : items.get(0);
        }
    }

    /** Stable low-cost state used before a full ordered group snapshot is needed. */
    ActiveQueueState activeQueueState() {
        queueLock.lock();
        try {
            BatchItem head = queue.peek();
            long oldestEnqueuedAtMs = Long.MAX_VALUE;
            for (BatchItem item : queue) {
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
                            BatchItem head,
                            int activeSize,
                            long oldestEnqueuedAtMs) {
    }

    BatcherCycleResult awaitingSchedulingChange(
            BatchItem head,
            long observedQueueVersion,
            long observedSchedulingInputVersion,
            long wakeAtMs,
            BatcherCycleResult.SchedulingWaitReason reason) {
        return new BatcherCycleResult.AwaitingSchedulingChange(
                head, observedQueueVersion, observedSchedulingInputVersion,
                wakeAtMs, reason);
    }

    /**
     * Compact priority-position cache for the current queue version. It keeps
     * only primitive counts, never request objects or futures.
     */
    private record QueuePositionView(long version,
                                     int activeSize,
                                     int[] activeCountByPriority) {
    }

    private volatile QueuePositionView queuePositionView;

    private QueuePositionView queuePositionView() {
        QueuePositionView view = queuePositionView;
        if (view != null && view.version() == queueVersion.get()) {
            return view;
        }
        queueLock.lock();
        try {
            long version = queueVersion.get();
            view = queuePositionView;
            if (view == null || view.version() != version) {
                int[] counts = new int[101];
                int activeSize = 0;
                for (BatchItem item : queue) {
                    int normalizedPriority = Math.max(1, Math.min(100, item.priority()));
                    counts[normalizedPriority]++;
                    activeSize++;
                }
                view = new QueuePositionView(version, activeSize, counts);
                queuePositionView = view;
            }
            return view;
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Estimate only the additional fixed-window collection delay for an
     * incoming priority request. Engine execution and request-slot turnover are
     * already accounted by the endpoint ledgers and must not be fabricated from
     * the wall-clock interval between unrelated logical decisions.
     *
     * <p>The estimate is zero when the incoming request's ordered group can be
     * filled by members already active behind it; otherwise it is the configured
     * window. Resource-shape filtering can still make the actual delay larger;
     * no request-shape information exists at this probe boundary.
     */
    long estimateIncomingWaitMs(int priority, long arrivalMs, long requestId) {
        int maxRequests = maxDecisionRequests();
        if (maxRequests <= 1) {
            return 0L;
        }
        QueuePositionView view = queuePositionView();
        int normalizedPriority = Math.max(1, Math.min(100, priority));
        int activeItemsAhead = 0;
        for (int existingPriority = normalizedPriority;
             existingPriority <= 100; existingPriority++) {
            activeItemsAhead += view.activeCountByPriority()[existingPriority];
        }
        int membersAtOrBehindProbe = view.activeSize() - activeItemsAhead + 1;
        int openSlots = maxRequests - activeItemsAhead % maxRequests;
        return membersAtOrBehindProbe >= openSlots ? 0L : collectionWindowMs();
    }

    /** Same collection-delay estimate for a FIFO probe appended at the tail. */
    long estimateFifoWaitMs() {
        int maxRequests = maxDecisionRequests();
        if (maxRequests <= 1) {
            return 0L;
        }
        int activeItemsAhead = activeSize();
        int openSlots = maxRequests - activeItemsAhead % maxRequests;
        return openSlots == 1 ? 0L : collectionWindowMs();
    }

    BatchItem findQueued(long requestId) {
        for (BatchItem item : queue) {
            if (item.requestId() == requestId) {
                return item;
            }
        }
        return null;
    }

    /**
     * Effective strict padded-token limit for one FlexLB batch.
     *
     * <p>The Engine's FIFO scheduler rejects a group when its padded context
     * shape ({@code maxSeqLen * batchSize}) is greater than or equal to
     * {@code max_batch_tokens_size}. Prefer
     * that exact worker-reported limit; {@code max_seq_len} is a conservative
     * fallback for workers that have not populated the newer field yet. An
     * internal safety ceiling covers the interval before either value arrives.
     */
    long batchTokenCapacity() {
        long capacity = positiveOrUnlimited(
                config.getInternalRuntime().getFallbackBatchTokenCapacity());
        WorkerStatus status = prefillEp != null ? prefillEp.getStatus() : null;
        if (status == null) {
            return capacity;
        }

        long engineCapacity = status.getMaxBatchTokensSize();
        if (engineCapacity <= 0) {
            engineCapacity = status.getMaxSeqLen();
        }
        return Math.min(capacity, positiveOrUnlimited(engineCapacity));
    }

    /**
     * Latest worker-reported KV budget. A zero total means the worker has not
     * published KV capacity yet, so batching remains compute-bound only.
     */
    long batchKvCapacity() {
        WorkerStatus status = prefillEp != null ? prefillEp.getStatus() : null;
        long total = status == null ? 0 : status.getTotalKvCacheTokens().get();
        if (total <= 0) {
            return Long.MAX_VALUE;
        }
        long available = Math.max(0, status.getAvailableKvCacheTokens().get());
        return Math.min(total, available);
    }

    void rejectForBatchTokenCapacity(BatchItem item, long capacity) {
        terminateActiveItem(item, new ActiveQueueRejected(
                new BatchTokenCapacityExceededException(
                        "request seq_len=" + item.seqLen()
                                + " cannot fit strict padded batch token capacity="
                                + capacity)));
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
     * this one. Removal, expiration, or predictor replacement of a selected
     * member does revoke it, and every provisional reservation is then released.
     */
    BatcherCycleResult admitAndDeliverCapacityFeasiblePrefix(
            List<BatchItem> candidates,
            DecisionGroupMetadata metadata,
            PrefillTimePredictor expectedPredictor,
            long expectedPredictorGeneration) {
        if (candidates == null || candidates.isEmpty()) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }
        Objects.requireNonNull(metadata, "decision metadata");
        validateCandidateGroup(candidates);
        if (!candidateSelectionStillOwned(
                candidates, expectedPredictor, expectedPredictorGeneration)) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }

        CapacityPrefix capacityPrefix = reserveCapacityPrefix(candidates);
        AdmittedDecisionGroup admittedGroup = prepareAdmittedGroup(capacityPrefix);
        DeliveryCapacityAdmission.BatchLoadPublication batchLoadPublication = null;
        if (admittedGroup != null && capacityPrefix.batchReservation() != null) {
            DeliveryCapacityAdmission.BatchLoadPublicationResult publicationResult =
                    admittedGroup.establishBatchLoadPublication();
            if (publicationResult
                    instanceof DeliveryCapacityAdmission.BatchLoadPublicationFailed failed) {
                return terminalizePublicationFailure(capacityPrefix, failed.cause());
            }
            batchLoadPublication =
                    ((DeliveryCapacityAdmission.BatchLoadPublicationEstablished)
                            publicationResult).publication();
        }
        boolean ownershipTransferred = false;
        RemovedTerminalBoundary removedTerminalBoundary =
                RemovedTerminalBoundary.NONE;
        DecisionGroupMetadata admittedMetadata = null;
        queueLock.lock();
        try {
            if (stopped || !predictorStillCurrent(
                    expectedPredictor, expectedPredictorGeneration)) {
                return BatcherCycleResult.Outcome.NO_ACTION;
            }

            if (capacityPrefix.items().isEmpty()
                    && capacityPrefix.firstUnreservedResult()
                    instanceof DeliveryCapacityAdmission.CapacityUnavailable unavailable) {
                BatchItem blockedHead = capacityPrefix.firstUnreservedItem();
                if (queue.peek() != blockedHead
                        || blockedHead.ctx().requestExpired(now())
                        || !containsIdentity(blockedHead)) {
                    return BatcherCycleResult.Outcome.NO_ACTION;
                }
                return new BatcherCycleResult.CapacityBlocked(
                        blockedHead, unavailable);
            }

            long nowMs = now();
            for (BatchItem item : capacityPrefix.items()) {
                if (!containsIdentity(item) || item.ctx().requestExpired(nowMs)) {
                    return BatcherCycleResult.Outcome.NO_ACTION;
                }
            }

            for (BatchItem item : capacityPrefix.items()) {
                if (!queue.remove(item)) {
                    throw new IllegalStateException(
                            "validated decision member disappeared request_id="
                                    + item.requestId());
                }
            }

            removedTerminalBoundary = removeTerminalAdmissionBoundary(
                    capacityPrefix, nowMs);

            int removedCount = capacityPrefix.items().size()
                    + removedTerminalBoundary.count();
            if (removedCount > 0) {
                queueDepth.addAndGet(-removedCount);
                queueVersion.incrementAndGet();
            }
            ownershipTransferred = !capacityPrefix.items().isEmpty();
            if (ownershipTransferred) {
                String reason = capacityPrefix.wasCapacityLimited()
                        ? "delivery_capacity_prefix" : metadata.reason();
                admittedMetadata = new DecisionGroupMetadata(reason, queue.size());
            }
        } finally {
            queueLock.unlock();
            if (!ownershipTransferred) {
                try {
                    releaseCapacityPrefix(capacityPrefix);
                } finally {
                    closeBatchLoadPublication(batchLoadPublication);
                }
            }
        }

        notifyTerminalAdmissionFailure(removedTerminalBoundary);
        if (!ownershipTransferred) {
            return removedTerminalBoundary.wasRemoved()
                    ? BatcherCycleResult.Outcome.QUEUE_CHANGED
                    : BatcherCycleResult.Outcome.NO_ACTION;
        }

        try {
            BatcherCycleResult.Admitted admitted = new BatcherCycleResult.Admitted(
                    capacityPrefix.items(), admittedMetadata);
            deliverAdmittedGroup(admittedGroup, admittedMetadata);
            return admitted;
        } finally {
            closeBatchLoadPublication(batchLoadPublication);
        }
    }

    private void closeBatchLoadPublication(
            DeliveryCapacityAdmission.BatchLoadPublication publication) {
        if (publication == null) {
            return;
        }
        try {
            publication.close();
        } catch (Throwable cleanupFailure) {
            Logger.error("WorkerBatcher[{}] batch load publication cleanup failed",
                    key, cleanupFailure);
        }
    }

    /**
     * A publication invariant failure is terminal for the exact prefix which
     * already reserved capacity. Returning it to ACTIVE would only repeat the
     * same deterministic failure. The first non-reserved member is consumed by
     * the same terminal-boundary rule as a successful publication: an explicit
     * admission failure is removed and notified once, ownership loss is only
     * removed, and capacity unavailability remains ACTIVE.
     *
     * <p>Members concurrently claimed by another terminal owner are left to
     * that owner and their exact reservations are still released below.
     */
    private BatcherCycleResult terminalizePublicationFailure(
            CapacityPrefix capacityPrefix,
            Throwable publicationFailure) {
        List<BatchItem> terminalItems = new ArrayList<>(capacityPrefix.items().size());
        RemovedTerminalBoundary removedTerminalBoundary =
                RemovedTerminalBoundary.NONE;
        queueLock.lock();
        try {
            if (!stopped) {
                long nowMs = now();
                for (BatchItem item : capacityPrefix.items()) {
                    if (containsIdentity(item)
                            && !item.ctx().requestExpired(nowMs)
                            && queue.remove(item)) {
                        terminalItems.add(item);
                    }
                }
                removedTerminalBoundary = removeTerminalAdmissionBoundary(
                        capacityPrefix, nowMs);
                int removedCount = terminalItems.size()
                        + removedTerminalBoundary.count();
                if (removedCount > 0) {
                    queueDepth.addAndGet(-removedCount);
                    queueVersion.incrementAndGet();
                }
            }
        } finally {
            queueLock.unlock();
            releaseCapacityPrefix(capacityPrefix);
        }
        for (BatchItem item : terminalItems) {
            notifyAdmissionFailure(item, publicationFailure);
        }
        notifyTerminalAdmissionFailure(removedTerminalBoundary);
        return terminalItems.isEmpty() && !removedTerminalBoundary.wasRemoved()
                ? BatcherCycleResult.Outcome.NO_ACTION
                : BatcherCycleResult.Outcome.QUEUE_CHANGED;
    }

    /**
     * Consume the one terminal admission boundary captured while reserving an
     * ordered prefix. Capacity pressure is deliberately excluded: it is the
     * only admission result allowed to remain ACTIVE.
     *
     * <p>Caller holds {@link #queueLock}.
     */
    private RemovedTerminalBoundary removeTerminalAdmissionBoundary(
            CapacityPrefix capacityPrefix,
            long nowMs) {
        DeliveryCapacityAdmission.AdmissionResult result =
                capacityPrefix.firstUnreservedResult();
        if (result != DeliveryCapacityAdmission.OwnershipLost.INSTANCE
                && !(result instanceof DeliveryCapacityAdmission.AdmissionFailed)) {
            return RemovedTerminalBoundary.NONE;
        }

        BatchItem item = capacityPrefix.firstUnreservedItem();
        if (item == null
                || !containsIdentity(item)
                || item.ctx().requestExpired(nowMs)
                || !queue.remove(item)) {
            return RemovedTerminalBoundary.NONE;
        }
        Throwable failure = result
                instanceof DeliveryCapacityAdmission.AdmissionFailed admissionFailed
                ? admissionFailed.cause()
                : null;
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

    private record RemovedTerminalBoundary(BatchItem item, Throwable failure) {
        private static final RemovedTerminalBoundary NONE =
                new RemovedTerminalBoundary(null, null);

        int count() {
            return wasRemoved() ? 1 : 0;
        }

        boolean wasRemoved() {
            return item != null;
        }
    }

    /**
     * Finish validation and allocate the callback payload before queue ownership
     * changes. A construction failure releases every provisional reservation
     * while all requests are still ACTIVE.
     */
    private AdmittedDecisionGroup prepareAdmittedGroup(CapacityPrefix capacityPrefix) {
        if (capacityPrefix.items().isEmpty()) {
            return null;
        }
        try {
            return AdmittedDecisionGroup.create(
                    capacityPrefix.items(), capacityPrefix.reservations(),
                    capacityPrefix.batchReservation());
        } catch (RuntimeException | Error payloadFailure) {
            releaseCapacityPrefix(capacityPrefix);
            throw payloadFailure;
        }
    }

    private boolean candidateSelectionStillOwned(
            List<BatchItem> candidates,
            PrefillTimePredictor expectedPredictor,
            long expectedPredictorGeneration) {
        queueLock.lock();
        try {
            if (stopped || !predictorStillCurrent(
                    expectedPredictor, expectedPredictorGeneration)) {
                return false;
            }
            long nowMs = now();
            for (BatchItem item : candidates) {
                if (!containsIdentity(item) || item.ctx().requestExpired(nowMs)) {
                    return false;
                }
            }
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    /** Caller holds {@link #queueLock}. */
    private boolean predictorStillCurrent(
            PrefillTimePredictor expectedPredictor,
            long expectedPredictorGeneration) {
        return expectedPredictor == null
                || (prefillEp.getPredictor() == expectedPredictor
                && expectedPredictor.generation() == expectedPredictorGeneration);
    }

    /** Identity membership avoids conflating different request generations. */
    private boolean containsIdentity(BatchItem expected) {
        for (BatchItem item : queue) {
            if (item == expected) {
                return true;
            }
        }
        return false;
    }

    private CapacityPrefix reserveCapacityPrefix(List<BatchItem> candidates) {
        BatchItem head = candidates.get(0);
        DeliveryCapacityAdmission.BatchCapacityReservation batchReservation = null;
        if (head.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            DeliveryCapacityAdmission.BatchCapacityResult batchResult =
                    tryReserveBatchCapacity(head);
            if (batchResult
                    instanceof DeliveryCapacityAdmission.BatchCapacityReserved reserved) {
                batchReservation = reserved.reservation();
            } else if (batchResult
                    instanceof DeliveryCapacityAdmission.BatchCapacityUnavailable unavailable) {
                return CapacityPrefix.blocked(head,
                        new DeliveryCapacityAdmission.CapacityUnavailable(
                                unavailable.resource(),
                                unavailable.availability()));
            } else if (batchResult
                    == DeliveryCapacityAdmission.BatchOwnershipLost.INSTANCE) {
                return CapacityPrefix.stopped(
                        head, DeliveryCapacityAdmission.OwnershipLost.INSTANCE);
            } else {
                DeliveryCapacityAdmission.BatchAdmissionFailed failure =
                        (DeliveryCapacityAdmission.BatchAdmissionFailed) batchResult;
                return CapacityPrefix.stopped(head,
                        new DeliveryCapacityAdmission.AdmissionFailed(failure.cause()));
            }
        }

        List<BatchItem> admittedItems = new ArrayList<>(candidates.size());
        List<DeliveryCapacityAdmission.ItemCapacityReservation> reservations =
                new ArrayList<>(candidates.size());
        for (BatchItem item : candidates) {
            DeliveryCapacityAdmission.AdmissionResult result =
                    tryReserveExactItemCapacity(item);
            if (result instanceof DeliveryCapacityAdmission.CapacityReserved reserved) {
                admittedItems.add(item);
                reservations.add(reserved.reservation());
                continue;
            }
            return new CapacityPrefix(
                    admittedItems, reservations, batchReservation, item, result);
        }
        return new CapacityPrefix(
                admittedItems, reservations, batchReservation, null, null);
    }

    private DeliveryCapacityAdmission.BatchCapacityResult tryReserveBatchCapacity(
            BatchItem head) {
        DeliveryCapacityAdmission.BatchCapacityResult result;
        try {
            result = capacityAdmission.tryReserveBatchCapacity(head);
            if (result == null) {
                return new DeliveryCapacityAdmission.BatchAdmissionFailed(
                        new IllegalStateException(
                                "batch capacity admission returned no result"));
            }
        } catch (Throwable failure) {
            return new DeliveryCapacityAdmission.BatchAdmissionFailed(failure);
        }
        if (result instanceof DeliveryCapacityAdmission.BatchCapacityReserved reserved) {
            DeliveryCapacityAdmission.BatchCapacityReservation reservation =
                    reserved.reservation();
            BatchItem reservationHead;
            try {
                reservationHead = reservation.head();
            } catch (Throwable validationFailure) {
                releaseBatchCapacityReservation(reservation);
                return new DeliveryCapacityAdmission.BatchAdmissionFailed(
                        validationFailure);
            }
            if (reservationHead != head) {
                releaseBatchCapacityReservation(reservation);
                return new DeliveryCapacityAdmission.BatchAdmissionFailed(
                        new IllegalStateException(
                                "batch capacity reservation belongs to another head"));
            }
        }
        return result;
    }

    private DeliveryCapacityAdmission.AdmissionResult tryReserveExactItemCapacity(
            BatchItem item) {
        DeliveryCapacityAdmission.AdmissionResult result;
        try {
            result = capacityAdmission.tryReserveItemCapacity(item);
            if (result == null) {
                return new DeliveryCapacityAdmission.AdmissionFailed(
                        new IllegalStateException("capacity admission returned no result"));
            }
        } catch (Throwable failure) {
            return new DeliveryCapacityAdmission.AdmissionFailed(failure);
        }
        if (result instanceof DeliveryCapacityAdmission.CapacityReserved reserved) {
            DeliveryCapacityAdmission.ItemCapacityReservation reservation =
                    reserved.reservation();
            BatchItem reservationOwner;
            try {
                reservationOwner = reservation.item();
            } catch (Throwable validationFailure) {
                releaseCapacityReservation(reservation);
                return new DeliveryCapacityAdmission.AdmissionFailed(
                        validationFailure);
            }
            if (reservationOwner != item) {
                releaseCapacityReservation(reservation);
                return new DeliveryCapacityAdmission.AdmissionFailed(
                        new IllegalStateException(
                                "capacity reservation belongs to another request"));
            }
        }
        return result;
    }

    private record CapacityPrefix(
            List<BatchItem> items,
            List<DeliveryCapacityAdmission.ItemCapacityReservation> reservations,
            DeliveryCapacityAdmission.BatchCapacityReservation batchReservation,
            BatchItem firstUnreservedItem,
            DeliveryCapacityAdmission.AdmissionResult firstUnreservedResult) {

        private CapacityPrefix {
            items = List.copyOf(items);
            reservations = List.copyOf(reservations);
            if (items.size() != reservations.size()) {
                throw new IllegalArgumentException(
                        "capacity prefix requires one reservation per item");
            }
        }

        static CapacityPrefix blocked(
                BatchItem item,
                DeliveryCapacityAdmission.CapacityUnavailable unavailable) {
            return new CapacityPrefix(
                    List.of(), List.of(), null, item, unavailable);
        }

        static CapacityPrefix stopped(
                BatchItem item,
                DeliveryCapacityAdmission.AdmissionResult firstUnreservedResult) {
            return new CapacityPrefix(
                    List.of(), List.of(), null, item, firstUnreservedResult);
        }

        boolean wasCapacityLimited() {
            return firstUnreservedResult
                    instanceof DeliveryCapacityAdmission.CapacityUnavailable;
        }
    }

    /** Reject malformed groups before the first endpoint reservation is acquired. */
    private void validateCandidateGroup(List<BatchItem> candidates) {
        for (int index = 0; index < candidates.size(); index++) {
            BatchItem item = candidates.get(index);
            if (item == null) {
                throw new IllegalArgumentException(
                        "decision group contains a null item at index " + index);
            }
            for (int earlier = 0; earlier < index; earlier++) {
                if (candidates.get(earlier) == item) {
                    throw new IllegalArgumentException(
                            "decision group contains the same item more than once"
                                    + " request_id=" + item.requestId());
                }
            }
            if (index > 0
                    && queueOrder.compare(candidates.get(index - 1), item) >= 0) {
                throw new IllegalArgumentException(
                        "decision candidates are not in queue order request_id="
                                + item.requestId());
            }
        }
        DeliveryMode deliveryMode = candidates.get(0).deliveryMode();
        for (int index = 1; index < candidates.size(); index++) {
            if (candidates.get(index).deliveryMode() != deliveryMode) {
                throw new IllegalArgumentException(
                        "decision group contains mixed delivery modes");
            }
        }
    }

    private void releaseCapacityReservations(
            List<DeliveryCapacityAdmission.ItemCapacityReservation> reservations) {
        for (DeliveryCapacityAdmission.ItemCapacityReservation reservation : reservations) {
            releaseCapacityReservation(reservation);
        }
    }

    private void releaseCapacityPrefix(CapacityPrefix capacityPrefix) {
        releaseCapacityReservations(capacityPrefix.reservations());
        if (capacityPrefix.batchReservation() != null) {
            releaseBatchCapacityReservation(capacityPrefix.batchReservation());
        }
    }

    private void releaseBatchCapacityReservation(
            DeliveryCapacityAdmission.BatchCapacityReservation reservation) {
        try {
            reservation.release();
        } catch (Throwable cleanupFailure) {
            Logger.error("WorkerBatcher[{}] batch capacity cleanup failed",
                    key, cleanupFailure);
        }
    }

    /**
     * Capacity cleanup must never prevent ownership cleanup for another item.
     * Endpoint reservations attempt all of their own resources; this boundary
     * additionally isolates custom admission implementations item by item.
     */
    private void releaseCapacityReservation(
            DeliveryCapacityAdmission.ItemCapacityReservation reservation) {
        try {
            reservation.release();
        } catch (Throwable cleanupFailure) {
            long requestId = -1L;
            try {
                BatchItem item = reservation.item();
                if (item != null) {
                    requestId = item.requestId();
                }
            } catch (Throwable ignored) {
                // Preserve the capacity cleanup failure as the diagnostic owner.
            }
            Logger.error("WorkerBatcher[{}] capacity cleanup failed request_id={}",
                    key, requestId, cleanupFailure);
        }
    }

    private void notifyAdmissionFailure(BatchItem item, Throwable cause) {
        try {
            decisionHandler.onDeliveryFailure(item, cause);
        } catch (Throwable callbackFailure) {
            Logger.error("WorkerBatcher[{}] delivery-failure callback failed "
                            + "request_id={}",
                    key, item.requestId(), callbackFailure);
        }
    }

    private void deliverAdmittedGroup(
            AdmittedDecisionGroup admittedGroup,
            DecisionGroupMetadata metadata) {
        int admittedItemCount = admittedGroup.members().size();
        callbackOwnedRequestCount.addAndGet(admittedItemCount);
        Throwable callbackFailure = null;
        try {
            decisionHandler.onDecisionGroupAdmitted(admittedGroup, metadata);
        } catch (Throwable t) {
            callbackFailure = t;
        } finally {
            Throwable batchHandoffFailure = callbackFailure != null
                    ? callbackFailure
                    : new IllegalStateException(
                            "batch callback did not establish a batch lifecycle");
            AdmittedDecisionGroup.BatchCapacityCleanup batchCleanup =
                    admittedGroup.terminateUntransferredBatchCapacity(
                            batchHandoffFailure);
            try {
                admittedGroup.completeTransferredBatchHandoff();
            } catch (Throwable handoffCompletionFailure) {
                if (callbackFailure == null) {
                    callbackFailure = handoffCompletionFailure;
                } else if (callbackFailure != handoffCompletionFailure) {
                    callbackFailure.addSuppressed(handoffCompletionFailure);
                }
            }
            Throwable unresolvedMemberDefault = callbackFailure != null
                    ? callbackFailure
                    : batchCleanup.untransferred()
                            ? batchCleanup.failure()
                            : new IllegalStateException(
                                    "delivery callback left admitted request unresolved");
            Map<BatchItem, Throwable> terminalItems = null;
            for (AdmittedDecisionGroup.AdmittedItem admittedItem
                    : admittedGroup.members()) {
                Throwable unresolvedFailure = admittedItem.terminateIfUnresolved(
                        unresolvedMemberDefault);
                if (unresolvedFailure != null) {
                    if (terminalItems == null) {
                        terminalItems = new java.util.LinkedHashMap<>();
                    }
                    terminalItems.put(admittedItem.request(), unresolvedFailure);
                }
            }
            callbackOwnedRequestCount.addAndGet(-admittedItemCount);
            if (terminalItems != null) {
                for (Map.Entry<BatchItem, Throwable> failure
                        : terminalItems.entrySet()) {
                    try {
                        decisionHandler.onDeliveryFailure(failure.getKey(), failure.getValue());
                    } catch (Throwable terminalCallbackFailure) {
                        Logger.error("WorkerBatcher[{}] terminal delivery callback failed "
                                        + "request_id={}",
                                key, failure.getKey().requestId(), terminalCallbackFailure);
                        if (callbackFailure == null) {
                            callbackFailure = terminalCallbackFailure;
                        } else if (callbackFailure != terminalCallbackFailure) {
                            callbackFailure.addSuppressed(terminalCallbackFailure);
                        }
                    }
                }
            }
        }
        if (callbackFailure != null) {
            Logger.error("WorkerBatcher[{}] decision-group callback failed after"
                            + " terminal ownership cleanup",
                    key, callbackFailure);
        }
    }

    /**
     * Linearize shutdown with the ACTIVE-to-callback ownership transfer. The
     * queue lock decides the race: ACTIVE requests are drained; requests already
     * removed into an {@link AdmittedDecisionGroup} remain callback-owned.
     */
    void stopAndDrainTo(List<BatchItem> dst) {
        queueLock.lock();
        try {
            stopped = true;
            int drained = queue.drainTo(dst);
            if (drained > 0) {
                queueDepth.addAndGet(-drained);
                queueVersion.incrementAndGet();
            }
        } finally {
            queueLock.unlock();
        }
    }

    int callbackOwnedRequestCount() {
        return callbackOwnedRequestCount.get();
    }

    /** Terminate the head whose absolute request expiration has been reached. */
    void dropHead(BatchItem head) {
        terminateActiveItem(head, ActiveQueueExpired.INSTANCE);
    }

    /**
     * Single reducer for normal terminal exits from ACTIVE. Queue removal is
     * the ownership claim; only the winner invokes the item-scoped terminal
     * callback. Callback failure is logged and never retried or promoted to a
     * worker-wide invariant failure.
     */
    private void terminateActiveItem(
            BatchItem item,
            ActiveQueueTermination termination) {
        if (!remove(item)) {
            return;
        }
        try {
            if (termination == ActiveQueueExpired.INSTANCE) {
                decisionHandler.onExpired(item);
            } else {
                ActiveQueueRejected rejected = (ActiveQueueRejected) termination;
                decisionHandler.onOfferFailure(item, rejected.cause());
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
        private ActiveQueueRejected {
            Objects.requireNonNull(cause, "rejection cause");
        }

        @Override
        public String reason() {
            return "batch_token_capacity_exceeded";
        }
    }

}
