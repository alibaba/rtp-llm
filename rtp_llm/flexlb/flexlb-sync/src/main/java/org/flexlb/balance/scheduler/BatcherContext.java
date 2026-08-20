package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.PriorityOrdering;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.CancellationException;
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

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig cfg;
    private final DecisionGroupHandler decisionHandler;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final AtomicInteger queueDepth;
    private final AtomicLong queueVersion;
    private final ReentrantLock queueLock;
    private final Comparator<BatchItem> queueOrder;
    private final BatchSchedulerReporter reporter;

    /**
     * Route decisions whose logical dispatch group is already ready, but
     * whose delivery is waiting for a request-mode inflight slot.
     *
     * <p>The backlog is guarded by the existing per-worker {@link #queueLock}
     * and shares {@link #queueDepth} with the active decision queue. It cannot
     * grow beyond requests already admitted into that bounded queue and does
     * not allocate a second per-request wrapper.
     */
    private final PriorityQueue<BatchItem> readyDeliveryQueue;
    private volatile int readyDeliveryCount;

    /**
     * Items removed from the priority queue for a delivery callback but not
     * yet classified as delivered, restored, or terminal. Guarded by
     * {@link #queueLock}. Their queue slots remain charged in
     * {@link #queueDepth} until the callback resolves ownership.
     */
    private final Map<BatchItem, PendingDelivery> pendingDeliveries = new IdentityHashMap<>();
    private boolean stopped;

    /**
     * Compact ownership state stored directly as the identity-map value.
     * Enum singletons avoid allocating a wrapper every time an item is staged.
     */
    private enum PendingDelivery {
        STAGED_ACTIVE(false, false),
        CLAIMED_ACTIVE(true, false),
        STAGED_READY(false, true),
        CLAIMED_READY(true, true);

        private final boolean claimed;
        private final boolean restoreToReadyQueue;

        PendingDelivery(boolean claimed, boolean restoreToReadyQueue) {
            this.claimed = claimed;
            this.restoreToReadyQueue = restoreToReadyQueue;
        }

        boolean isStaged() {
            return !claimed;
        }

        boolean isClaimed() {
            return claimed;
        }

        boolean restoresToReadyQueue() {
            return restoreToReadyQueue;
        }

        PendingDelivery claimedState() {
            return restoreToReadyQueue ? CLAIMED_READY : CLAIMED_ACTIVE;
        }
    }

    enum PendingRestoreResult { RESTORED, STOPPED, NOT_PENDING }
    enum PendingClaimResult { CLAIMED, STOPPED, NOT_PENDING }
    enum ReadyDeliveryResult { EMPTY, CAPACITY_BLOCKED, DELIVERED }

    /** Decision-interval sliding average for the queue wait estimate. */
    private volatile long lastDecisionAtMs;
    private volatile double decisionIntervalEmaMs;

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   DecisionGroupHandler decisionHandler,
                   PriorityBlockingQueue<BatchItem> queue,
                   BatchSchedulerReporter reporter) {
        this(key, prefillEp, cfg, decisionHandler, queue, new AtomicInteger(queue.size()), reporter);
    }

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   DecisionGroupHandler decisionHandler,
                   PriorityBlockingQueue<BatchItem> queue,
                   AtomicInteger queueDepth,
                   BatchSchedulerReporter reporter) {
        this(key, prefillEp, cfg, decisionHandler, queue, queueDepth, new AtomicLong(),
                new ReentrantLock(), WorkerBatcher.FIFO_QUEUE_ORDER, reporter);
    }

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   DecisionGroupHandler decisionHandler,
                   PriorityBlockingQueue<BatchItem> queue,
                   AtomicInteger queueDepth,
                   AtomicLong queueVersion,
                   ReentrantLock queueLock,
                   Comparator<BatchItem> queueOrder,
                   BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.decisionHandler = decisionHandler;
        this.queue = queue;
        this.queueDepth = queueDepth;
        this.queueVersion = queueVersion;
        this.queueLock = queueLock;
        this.queueOrder = queueOrder;
        this.reporter = reporter;
        this.readyDeliveryQueue = new PriorityQueue<>(11, queueOrder);
    }

    // ---- accessors ----

    String key() {
        return key;
    }

    PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    FlexlbConfig cfg() {
        return cfg;
    }

    int maxQueueCapacity() {
        if (cfg.getDispatcher() instanceof BatchDispatcherConfig batch) {
            return batch.getMaxWaitingRequestsPerPrefillWorker();
        }
        return cfg.getInternalRuntime().getNonBatchWaitingRequestsPerPrefillWorker();
    }

    int maxDecisionRequests() {
        return cfg.getDispatcher() instanceof BatchDispatcherConfig batch
                ? Math.max(1, batch.getMaxRequests())
                : 1;
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

    long queueVersionValue() {
        return queueVersion.get();
    }

    Comparator<BatchItem> queueOrder() {
        return queueOrder;
    }

    // ---- queue inspection ----

    BatchItem peek() {
        return queue.peek();
    }

    /** Whether the active queue still has work requiring a logical decision. */
    boolean isActiveEmpty() {
        return queue.isEmpty();
    }

    boolean isEmpty() {
        return queueDepth.get() == 0;
    }

    /**
     * Active decision-queue depth. The common no-backlog path preserves the
     * existing charged-depth read exactly; only a live ready backlog needs the
     * physical queue size to exclude already-decided requests.
     */
    int activeSize() {
        return readyDeliveryCount == 0 ? queueDepth.get() : queue.size();
    }

    int size() {
        return queueDepth.get();
    }

    int readyDeliveryCount() {
        return readyDeliveryCount;
    }

    boolean hasProcessableWork() {
        return !queue.isEmpty() || readyDeliveryCount > 0;
    }

    // ---- queue mutation ----

    boolean remove(BatchItem item) {
        queueLock.lock();
        try {
            boolean removed = queue.remove(item);
            if (!removed && readyDeliveryQueue.remove(item)) {
                readyDeliveryCount--;
                item.clearRouteDecisionReady();
                removed = true;
            }
            if (removed) {
                item.clearParkTrace();
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
            for (int i = dst.size() - drained; i < dst.size(); i++) {
                dst.get(i).clearParkTrace();
            }
            while (!readyDeliveryQueue.isEmpty()) {
                BatchItem ready = readyDeliveryQueue.poll();
                ready.clearRouteDecisionReady();
                ready.clearParkTrace();
                dst.add(ready);
                drained++;
            }
            readyDeliveryCount = 0;
            if (drained > 0) {
                queueDepth.addAndGet(-drained);
                queueVersion.incrementAndGet();
            }
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Items in active queue order (FIFO: {@link BatchItem#enqueueSeq()};
     * PRIORITY: {@link WorkerBatcher#PRIORITY_QUEUE_ORDER}, which delegates
     * to {@link PriorityOrdering#STRICT}), suitable for greedy-fill iteration
     * in grouping algorithms.
     */
    List<BatchItem> sortedItems() {
        List<BatchItem> candidates = new ArrayList<>(queue);
        candidates.sort(queueOrder);
        return candidates;
    }

    /**
     * All removable, not-yet-delivered requests in priority order. Used by
     * admission snapshots so a ready route decision remains preemptible and
     * capacity-charged until its delivery actually claims ownership.
     * Caller holds {@link #queueLock}.
     */
    List<BatchItem> sortedQueuedItems() {
        if (readyDeliveryCount == 0) {
            return sortedItems();
        }
        List<BatchItem> candidates = new ArrayList<>(queue.size() + readyDeliveryCount);
        candidates.addAll(queue);
        candidates.addAll(readyDeliveryQueue);
        candidates.sort(queueOrder);
        return candidates;
    }

    BatchItem findQueued(long requestId) {
        for (BatchItem item : queue) {
            if (item.requestId() == requestId) {
                return item;
            }
        }
        for (BatchItem item : readyDeliveryQueue) {
            if (item.requestId() == requestId) {
                return item;
            }
        }
        return null;
    }

    void addReadyQueueSizeByPriority(Map<Integer, Integer> sizeByPriority) {
        if (readyDeliveryCount == 0) {
            return;
        }
        queueLock.lock();
        try {
            for (BatchItem item : readyDeliveryQueue) {
                sizeByPriority.merge(item.priority(), 1, Integer::sum);
            }
        } finally {
            queueLock.unlock();
        }
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
                cfg.getInternalRuntime().getFallbackBatchTokenCapacity());
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

    /**
     * Request-mode capacity visible to the current decision.
     *
     * <p>This value must not be used as the logical batch target: priority scheduling
     * still decides <em>when</em> work is ready with the existing batch policy.
     * The request cap only limits the subset delivered to the caller after that
     * decision has been made. {@link PrefillEndpoint#tryCommitRequest} remains
     * the authoritative concurrent hard gate.
     */
    int availableDeliverySlots() {
        if (prefillEp == null) {
            return Integer.MAX_VALUE;
        }
        Integer maximum = cfg.getDispatcher() instanceof NonBatchDispatcherConfig nonBatch
                ? nonBatch.getMaxInflightRequestsPerPrefillWorker()
                : null;
        return prefillEp.availableRequestSlots(maximum == null ? 0 : maximum);
    }

    /** Delivery-unit inflight count used only for decision diagnostics. */
    int deliveryInflightCount(BatchItem head) {
        return head != null && head.deliveryMode() == DeliveryMode.ROUTE_DECISION
                ? prefillEp.getInflightRouteRequestCount()
                : prefillEp.getInflightBatchCount();
    }

    void rejectForBatchTokenCapacity(BatchItem item, long capacity) {
        if (remove(item)) {
            decisionHandler.onOfferFailure(item, new BatchTokenCapacityExceededException(
                    "request seq_len=" + item.seqLen()
                            + " cannot fit strict padded batch token capacity=" + capacity));
        }
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }

    // ---- delivery staging (shared infrastructure) ----

    /**
     * Remove items from the queue and notify the decision handler.
     * Caller is responsible for algorithm-specific logging and state cleanup
     * before calling this.
     */
    void stageForDelivery(List<BatchItem> items, DecisionGroupMetadata metadata) {
        // The decision-interval EMA only feeds the PRIORITY queue-wait
        // estimate (PrefillQueueManager.estimateWaitMs); FIFO does not need
        // this synchronized bookkeeping.
        if (cfg.isPriorityOrdering()) {
            recordDecisionInterval(now());
        }
        deliverStaged(stageRequests(items), metadata);
    }

    /**
     * Publish one logical dispatch group.
     *
     * <p>The batch-only fast path is the established delivery protocol. If a
     * route-decision member is present, every such member becomes delivery
     * ready atomically: the available prefix is staged for the callback and
     * the remainder moves to {@link #readyDeliveryQueue}. Request capacity never
     * feeds back into the logical batching decision.
     */
    void stageDecisionGroup(List<BatchItem> logicalGroup, DecisionGroupMetadata metadata) {
        if (logicalGroup == null || logicalGroup.isEmpty()) {
            return;
        }
        boolean containsRouteDecision = false;
        for (BatchItem item : logicalGroup) {
            if (item.deliveryMode() == DeliveryMode.ROUTE_DECISION) {
                containsRouteDecision = true;
                break;
            }
        }
        if (!containsRouteDecision) {
            stageForDelivery(logicalGroup, metadata);
            return;
        }

        if (cfg.isPriorityOrdering()) {
            recordDecisionInterval(now());
        }
        List<BatchItem> staged = stageDecisionGroup(
                logicalGroup, availableDeliverySlots(), metadata.reason());
        deliverStaged(staged,
                new DecisionGroupMetadata(metadata.reason(), liveQueuedDepth()));
    }

    /** Drain a previously-decided route backlog before making a new decision. */
    ReadyDeliveryResult deliverReadyRequests() {
        if (readyDeliveryCount == 0) {
            return ReadyDeliveryResult.EMPTY;
        }
        int availableSlots = availableDeliverySlots();
        if (availableSlots == 0) {
            return ReadyDeliveryResult.CAPACITY_BLOCKED;
        }
        int maxDelivery = 1;
        int deliveryLimit = availableSlots == Integer.MAX_VALUE
                ? maxDelivery : Math.min(maxDelivery, availableSlots);
        ReadyStage readyStage = stageReadyRequests(deliveryLimit);
        if (readyStage.items().isEmpty()) {
            return readyDeliveryCount == 0
                    ? ReadyDeliveryResult.EMPTY : ReadyDeliveryResult.CAPACITY_BLOCKED;
        }
        deliverStaged(readyStage.items(),
                new DecisionGroupMetadata(readyStage.reason(), readyStage.queueDepth()));
        return ReadyDeliveryResult.DELIVERED;
    }

    private void deliverStaged(List<BatchItem> staged, DecisionGroupMetadata metadata) {
        if (staged.isEmpty()) {
            return;
        }
        Throwable callbackFailure = null;
        try {
            decisionHandler.onDecisionGroupReady(staged, metadata);
        } catch (Throwable t) {
            callbackFailure = t;
        } finally {
            // Preserve the original DecisionGroupHandler contract: a normal
            // return consumes every member the handler did not explicitly
            // resolve. Only a failed callback restores still-STAGED members.
            // CLAIMED ownership is never safe to restore, even on failure.
            Map<BatchItem, Throwable> failedItems = null;
            for (BatchItem item : staged) {
                boolean stagedResolved;
                if (callbackFailure == null) {
                    stagedResolved = completeStagedPendingDelivery(item);
                } else {
                    PendingRestoreResult restore = restoreStagedPendingDelivery(item);
                    stagedResolved = restore != PendingRestoreResult.NOT_PENDING;
                    if (restore == PendingRestoreResult.STOPPED) {
                        if (failedItems == null) {
                            failedItems = new java.util.LinkedHashMap<>();
                        }
                        failedItems.put(item,
                                new CancellationException(
                                        "FlexLB worker scheduling queue stopped: " + key));
                    }
                }
                if (!stagedResolved && completeClaimedPendingDelivery(item)) {
                    // A callback which claimed ownership but escaped without
                    // resolving the item must not leave a charged orphan. It
                    // is no longer safe to requeue (Decode may be visible), so
                    // hand it to the terminal failure callback exactly once.
                    if (failedItems == null) {
                        failedItems = new java.util.LinkedHashMap<>();
                    }
                    failedItems.put(item, callbackFailure != null
                            ? callbackFailure
                            : new IllegalStateException(
                                    "delivery callback left claimed item unresolved"));
                }
            }
            if (failedItems != null) {
                for (Map.Entry<BatchItem, Throwable> failure : failedItems.entrySet()) {
                    try {
                        decisionHandler.onDeliveryFailure(failure.getKey(), failure.getValue());
                    } catch (Throwable ignored) {
                        // The queue slot and pending ownership are already
                        // resolved. Preserve the original callback failure.
                    }
                }
            }
        }
        if (callbackFailure instanceof RuntimeException runtimeException) {
            throw runtimeException;
        }
        if (callbackFailure instanceof Error error) {
            throw error;
        }
        if (callbackFailure != null) {
            throw new IllegalStateException("decision-group callback failed", callbackFailure);
        }
    }

    private List<BatchItem> stageRequests(List<BatchItem> items) {
        queueLock.lock();
        try {
            List<BatchItem> staged = new ArrayList<>(items.size());
            for (BatchItem item : items) {
                if (!queue.remove(item)) {
                    continue;
                }
                PendingDelivery previous = pendingDeliveries.putIfAbsent(
                        item, PendingDelivery.STAGED_ACTIVE);
                if (previous != null) {
                    // Defensive only: request IDs are unique in one batcher.
                    queue.add(item);
                    throw new IllegalStateException(
                            "duplicate pending-delivery item request_id=" + item.requestId());
                }
                // Removing the item invalidates queue snapshots even though
                // its capacity slot remains charged until resolution.
                queueVersion.incrementAndGet();
                staged.add(item);
            }
            return staged;
        } finally {
            queueLock.unlock();
        }
    }

    private List<BatchItem> stageDecisionGroup(List<BatchItem> logicalGroup,
                                               int availableRouteSlots,
                                               String reason) {
        int routeSlots = Math.max(0, availableRouteSlots);
        queueLock.lock();
        try {
            int initialCapacity = routeSlots == Integer.MAX_VALUE
                    ? logicalGroup.size()
                    : Math.min(logicalGroup.size(), routeSlots + 1);
            List<BatchItem> staged = new ArrayList<>(initialCapacity);
            for (BatchItem item : logicalGroup) {
                if (!queue.remove(item)) {
                    continue;
                }

                boolean restoreToReadyQueue = false;
                boolean stageNow = true;
                if (item.deliveryMode() == DeliveryMode.ROUTE_DECISION) {
                    item.markRouteDecisionReady(reason);
                    restoreToReadyQueue = true;
                    stageNow = routeSlots > 0;
                    if (stageNow && routeSlots != Integer.MAX_VALUE) {
                        routeSlots--;
                    }
                }

                if (stageNow) {
                    PendingDelivery previous = pendingDeliveries.putIfAbsent(
                            item, restoreToReadyQueue
                                    ? PendingDelivery.STAGED_READY
                                    : PendingDelivery.STAGED_ACTIVE);
                    if (previous != null) {
                        restoreAfterDuplicatePending(item, restoreToReadyQueue);
                        throw new IllegalStateException(
                                "duplicate pending-delivery item request_id=" + item.requestId());
                    }
                    staged.add(item);
                } else {
                    readyDeliveryQueue.add(item);
                    readyDeliveryCount++;
                }
                // Active -> ready/pending changes the actionable queue state,
                // but the original capacity charge remains held.
                queueVersion.incrementAndGet();
            }
            return staged;
        } finally {
            queueLock.unlock();
        }
    }

    private ReadyStage stageReadyRequests(int deliveryLimit) {
        queueLock.lock();
        try {
            if (stopped || readyDeliveryCount == 0 || deliveryLimit <= 0) {
                return ReadyStage.EMPTY;
            }
            int count = Math.min(deliveryLimit, readyDeliveryCount);
            List<BatchItem> staged = new ArrayList<>(count);
            String reason = null;
            while (staged.size() < count) {
                BatchItem item = readyDeliveryQueue.peek();
                if (item == null) {
                    readyDeliveryCount = 0;
                    break;
                }
                String itemReason = item.readyDeliveryReason();
                if (reason != null && !Objects.equals(reason, itemReason)) {
                    break;
                }
                if (reason == null) {
                    reason = itemReason;
                }
                readyDeliveryQueue.poll();
                readyDeliveryCount--;
                PendingDelivery previous = pendingDeliveries.putIfAbsent(
                        item, PendingDelivery.STAGED_READY);
                if (previous != null) {
                    readyDeliveryQueue.add(item);
                    readyDeliveryCount++;
                    throw new IllegalStateException(
                            "duplicate ready pending-delivery item request_id=" + item.requestId());
                }
                queueVersion.incrementAndGet();
                staged.add(item);
            }
            return staged.isEmpty()
                    ? ReadyStage.EMPTY
                    : new ReadyStage(staged,
                            reason == null ? "route_decision_ready" : reason,
                            liveQueuedDepth());
        } finally {
            queueLock.unlock();
        }
    }

    private void restoreAfterDuplicatePending(BatchItem item,
                                              boolean restoreToReadyQueue) {
        if (restoreToReadyQueue) {
            readyDeliveryQueue.add(item);
            readyDeliveryCount++;
        } else {
            queue.add(item);
        }
    }

    private int liveQueuedDepth() {
        return queue.size() + readyDeliveryCount;
    }

    private record ReadyStage(List<BatchItem> items, String reason, int queueDepth) {
        private static final ReadyStage EMPTY =
                new ReadyStage(List.of(), "route_decision_ready", 0);
    }

    /** Claim a staged item for the scheduler callback, fenced against shutdown. */
    PendingClaimResult claimPendingDelivery(BatchItem item) {
        queueLock.lock();
        try {
            if (stopped) {
                return PendingClaimResult.STOPPED;
            }
            PendingDelivery pending = pendingDeliveries.get(item);
            if (pending == null || !pending.isStaged()) {
                return PendingClaimResult.NOT_PENDING;
            }
            pendingDeliveries.put(item, pending.claimedState());
            // Fence the queue-to-delivery ownership transition so a concurrent
            // queue-side admission cannot commit across it.
            queueVersion.incrementAndGet();
            return PendingClaimResult.CLAIMED;
        } finally {
            queueLock.unlock();
        }
    }

    /** Resolve a staged/claimed item as delivered or terminal, releasing its queue slot. */
    boolean completePendingDelivery(BatchItem item) {
        queueLock.lock();
        try {
            PendingDelivery pending = pendingDeliveries.get(item);
            if (pending == null) {
                return false;
            }
            pendingDeliveries.remove(item);
            if (pending.restoresToReadyQueue()) {
                item.clearRouteDecisionReady();
            }
            queueDepth.decrementAndGet();
            queueVersion.incrementAndGet();
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    /** Consume an unclaimed member after a successful legacy callback. */
    private boolean completeStagedPendingDelivery(BatchItem item) {
        queueLock.lock();
        try {
            PendingDelivery pending = pendingDeliveries.get(item);
            if (pending == null || !pending.isStaged()) {
                return false;
            }
            pendingDeliveries.remove(item);
            if (pending.restoresToReadyQueue()) {
                item.clearRouteDecisionReady();
            }
            queueDepth.decrementAndGet();
            queueVersion.incrementAndGet();
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    /** Terminal fallback for a callback that escaped while owning CLAIMED. */
    private boolean completeClaimedPendingDelivery(BatchItem item) {
        queueLock.lock();
        try {
            PendingDelivery pending = pendingDeliveries.get(item);
            if (pending == null || !pending.isClaimed()) {
                return false;
            }
            pendingDeliveries.remove(item);
            if (pending.restoresToReadyQueue()) {
                item.clearRouteDecisionReady();
            }
            queueDepth.decrementAndGet();
            queueVersion.incrementAndGet();
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Put a capacity-blocked staged item back into the same priority queue.
     * The original sort key, enqueue timestamp, priority, and charged queue
     * slot are retained; no offer statistics are recorded a second time.
     */
    PendingRestoreResult restorePendingDelivery(BatchItem item) {
        return restorePendingDelivery(item, false);
    }

    /** Callback-finally fallback: never restore a request already claimed by the scheduler. */
    private PendingRestoreResult restoreStagedPendingDelivery(BatchItem item) {
        return restorePendingDelivery(item, true);
    }

    private PendingRestoreResult restorePendingDelivery(BatchItem item,
                                                         boolean stagedOnly) {
        queueLock.lock();
        try {
            PendingDelivery pending = pendingDeliveries.get(item);
            if (pending == null || (stagedOnly && !pending.isStaged())) {
                return PendingRestoreResult.NOT_PENDING;
            }
            pendingDeliveries.remove(item);
            if (stopped) {
                if (pending.restoresToReadyQueue()) {
                    item.clearRouteDecisionReady();
                }
                queueDepth.decrementAndGet();
                queueVersion.incrementAndGet();
                return PendingRestoreResult.STOPPED;
            }
            if (pending.restoresToReadyQueue()) {
                readyDeliveryQueue.add(item);
                readyDeliveryCount++;
            } else {
                queue.add(item);
            }
            queueVersion.incrementAndGet();
            return PendingRestoreResult.RESTORED;
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Linearize shutdown with queue and pending-delivery ownership. Staged
     * items remain engine-unseen and are drained; a callback that already
     * claimed an item owns finishing or restoring it.
     */
    void stopAndDrainTo(List<BatchItem> dst) {
        queueLock.lock();
        try {
            stopped = true;
            int drained = queue.drainTo(dst);
            for (int i = dst.size() - drained; i < dst.size(); i++) {
                dst.get(i).clearParkTrace();
            }
            while (!readyDeliveryQueue.isEmpty()) {
                BatchItem ready = readyDeliveryQueue.poll();
                ready.clearRouteDecisionReady();
                ready.clearParkTrace();
                dst.add(ready);
                drained++;
            }
            readyDeliveryCount = 0;
            if (drained > 0) {
                queueDepth.addAndGet(-drained);
            }
            boolean stagedDrained = false;
            java.util.Iterator<Map.Entry<BatchItem, PendingDelivery>> iterator =
                    pendingDeliveries.entrySet().iterator();
            while (iterator.hasNext()) {
                Map.Entry<BatchItem, PendingDelivery> entry = iterator.next();
                BatchItem item = entry.getKey();
                PendingDelivery pending = entry.getValue();
                if (pending.isStaged()) {
                    if (pending.restoresToReadyQueue()) {
                        item.clearRouteDecisionReady();
                    }
                    item.clearParkTrace();
                    dst.add(item);
                    iterator.remove();
                    queueDepth.decrementAndGet();
                    stagedDrained = true;
                }
            }
            if (drained > 0 || stagedDrained) {
                queueVersion.incrementAndGet();
            }
        } finally {
            queueLock.unlock();
        }
    }

    int pendingDeliveryCount() {
        queueLock.lock();
        try {
            return pendingDeliveries.size();
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Remove the head from the queue and notify the decision handler that the
     * request's absolute expiration has been reached.
     * Caller is responsible for algorithm-specific logging and state cleanup.
     */
    void dropHead(BatchItem head) {
        remove(head);
        decisionHandler.onExpired(head);
    }

    // ---- decision interval estimation (design doc 8.4) ----

    private synchronized void recordDecisionInterval(long nowMs) {
        if (lastDecisionAtMs > 0 && nowMs > lastDecisionAtMs) {
            long intervalMs = nowMs - lastDecisionAtMs;
            decisionIntervalEmaMs = decisionIntervalEmaMs <= 0
                    ? intervalMs
                    : 0.3 * intervalMs + 0.7 * decisionIntervalEmaMs;
        }
        lastDecisionAtMs = nowMs;
    }

    /**
     * Sliding-average interval between logical decision releases; before any
     * release is observed, falls back to the fixed grouping window.
     */
    long avgDecisionIntervalMs() {
        double ema = decisionIntervalEmaMs;
        if (ema > 0) {
            return Math.max(1, Math.round(ema));
        }
        return cfg.getDispatcher() instanceof BatchDispatcherConfig batch
                ? Math.max(1, batch.getMaxCollectionWaitMs())
                : 1;
    }
}
