package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.PriorityOrdering;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
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
 * the queue version, keeping the Auto-TPM invariant "version unchanged ⇒
 * queue content unchanged" (optimistic plan validation).
 */
public class BatcherContext {

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig cfg;
    private final BatchDecisionHandler handler;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final AtomicInteger queueDepth;
    private final AtomicLong queueVersion;
    private final ReentrantLock queueLock;
    private final Comparator<BatchItem> queueOrder;
    private final BatchSchedulerReporter reporter;

    /**
     * Items removed from the priority queue for a dispatch callback but not
     * yet classified as dispatched, restored, or terminal. Guarded by
     * {@link #queueLock}. Their queue slots remain charged in
     * {@link #queueDepth} until the callback resolves ownership.
     */
    private final Map<BatchItem, PendingDispatch> dispatchPending = new IdentityHashMap<>();
    private boolean stopped;

    private enum PendingDispatchState { STAGED, CLAIMED }

    private record PendingDispatch(BatchItem item, PendingDispatchState state) {
    }

    enum PendingRestoreResult { RESTORED, STOPPED, NOT_PENDING }
    enum PendingClaimResult { CLAIMED, STOPPED, NOT_PENDING }

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchDecisionHandler handler,
                   PriorityBlockingQueue<BatchItem> queue,
                   BatchSchedulerReporter reporter) {
        this(key, prefillEp, cfg, handler, queue, new AtomicInteger(queue.size()), reporter);
    }

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchDecisionHandler handler,
                   PriorityBlockingQueue<BatchItem> queue,
                   AtomicInteger queueDepth,
                   BatchSchedulerReporter reporter) {
        this(key, prefillEp, cfg, handler, queue, queueDepth, new AtomicLong(),
                new ReentrantLock(), Comparator.comparingLong(BatchItem::sortKey), reporter);
    }

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchDecisionHandler handler,
                   PriorityBlockingQueue<BatchItem> queue,
                   AtomicInteger queueDepth,
                   AtomicLong queueVersion,
                   ReentrantLock queueLock,
                   Comparator<BatchItem> queueOrder,
                   BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.handler = handler;
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

    FlexlbConfig cfg() {
        return cfg;
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

    boolean isEmpty() {
        return queueDepth.get() == 0;
    }

    int size() {
        return queueDepth.get();
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
     * Items in active queue order (legacy: {@link BatchItem#sortKey()};
     * Auto-TPM: {@link WorkerBatcher#AUTO_TPM_QUEUE_ORDER}, which delegates
     * to {@link PriorityOrdering#STRICT}), suitable for greedy-fill iteration
     * in dispatch algorithms.
     */
    List<BatchItem> sortedItems() {
        List<BatchItem> candidates = new ArrayList<>(queue);
        candidates.sort(queueOrder);
        return candidates;
    }

    /**
     * Unordered mutable copy of the live queue members. Callers holding
     * {@link #queueLock()} get a copy consistent with the current queue
     * version that they may sort outside the lock.
     */
    List<BatchItem> copiedItems() {
        return new ArrayList<>(queue);
    }

    /**
     * Effective strict padded-token limit for one FlexLB batch.
     *
     * <p>The Engine's FIFO scheduler rejects a group when its padded context
     * shape ({@code maxSeqLen * batchSize}) is greater than or equal to
     * {@code max_batch_tokens_size}. Prefer
     * that exact worker-reported limit; {@code max_seq_len} is a conservative
     * fallback for workers that have not populated the newer field yet. The
     * FlexLB setting remains an operator-controlled upper bound.
     */
    long batchTokenCapacity() {
        long capacity = positiveOrUnlimited(cfg.getFlexlbBatchMaxCapacity());
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
        if (remove(item)) {
            handler.onOfferFailure(item, new BatchTokenCapacityExceededException(
                    "request seq_len=" + item.seqLen()
                            + " cannot fit strict padded batch token capacity=" + capacity));
        }
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }

    // ---- dispatch helpers (shared infrastructure) ----

    /**
     * Remove items from queue and notify handler.
     * Caller is responsible for algorithm-specific logging and state cleanup
     * (e.g. {@code lastParkByRequest.remove()}) before calling this.
     */
    void dispatch(List<BatchItem> items, DispatchMeta meta) {
        List<BatchItem> staged = stageForDispatch(items);
        if (staged.isEmpty()) {
            return;
        }
        Throwable callbackFailure = null;
        try {
            handler.onBatchReady(staged, meta);
        } catch (Throwable t) {
            callbackFailure = t;
        } finally {
            // Preserve the original BatchDecisionHandler contract: a normal
            // return consumes every member the handler did not explicitly
            // resolve. Only a failed callback restores still-STAGED members.
            // CLAIMED ownership is never safe to restore, even on failure.
            Map<BatchItem, Throwable> failedItems = new java.util.LinkedHashMap<>();
            for (BatchItem item : staged) {
                boolean stagedResolved;
                if (callbackFailure == null) {
                    stagedResolved = completeStagedPendingDispatch(item);
                } else {
                    PendingRestoreResult restore = restoreStagedPendingDispatch(item);
                    stagedResolved = restore != PendingRestoreResult.NOT_PENDING;
                    if (restore == PendingRestoreResult.STOPPED) {
                        failedItems.put(item,
                                new CancellationException("FlexLB batcher stopped: " + key));
                    }
                }
                if (!stagedResolved && completeClaimedPendingDispatch(item)) {
                    // A callback which claimed ownership but escaped without
                    // resolving the item must not leave a charged orphan. It
                    // is no longer safe to requeue (Decode may be visible), so
                    // hand it to the terminal failure callback exactly once.
                    failedItems.put(item, callbackFailure != null
                            ? callbackFailure
                            : new IllegalStateException(
                                    "batch callback left claimed item unresolved"));
                }
            }
            for (Map.Entry<BatchItem, Throwable> failure : failedItems.entrySet()) {
                try {
                    handler.onOfferFailure(failure.getKey(), failure.getValue());
                } catch (Throwable ignored) {
                    // The queue slot and pending ownership are already
                    // resolved. Preserve the original callback failure.
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
            throw new IllegalStateException("batch dispatch callback failed", callbackFailure);
        }
    }

    private List<BatchItem> stageForDispatch(List<BatchItem> items) {
        queueLock.lock();
        try {
            List<BatchItem> staged = new ArrayList<>(items.size());
            for (BatchItem item : items) {
                if (!queue.remove(item)) {
                    continue;
                }
                PendingDispatch previous = dispatchPending.putIfAbsent(item,
                        new PendingDispatch(item, PendingDispatchState.STAGED));
                if (previous != null) {
                    // Defensive only: request IDs are unique in one batcher.
                    queue.add(item);
                    throw new IllegalStateException(
                            "duplicate dispatch-pending item request_id=" + item.requestId());
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

    /** Claim a staged item for the scheduler callback, fenced against shutdown. */
    PendingClaimResult claimPendingDispatch(BatchItem item) {
        queueLock.lock();
        try {
            if (stopped) {
                return PendingClaimResult.STOPPED;
            }
            PendingDispatch pending = dispatchPending.get(item);
            if (pending == null || pending.item() != item
                    || pending.state() != PendingDispatchState.STAGED) {
                return PendingClaimResult.NOT_PENDING;
            }
            dispatchPending.put(item,
                    new PendingDispatch(item, PendingDispatchState.CLAIMED));
            // Fence the queue-to-dispatch ownership transition so a versioned
            // queue-side plan cannot commit across it.
            queueVersion.incrementAndGet();
            return PendingClaimResult.CLAIMED;
        } finally {
            queueLock.unlock();
        }
    }

    /** Resolve a staged/claimed item as dispatched or terminal, releasing its queue slot. */
    boolean completePendingDispatch(BatchItem item) {
        queueLock.lock();
        try {
            PendingDispatch pending = dispatchPending.get(item);
            if (pending == null || pending.item() != item) {
                return false;
            }
            dispatchPending.remove(item);
            queueDepth.decrementAndGet();
            queueVersion.incrementAndGet();
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    /** Consume an unclaimed member after a successful legacy callback. */
    private boolean completeStagedPendingDispatch(BatchItem item) {
        queueLock.lock();
        try {
            PendingDispatch pending = dispatchPending.get(item);
            if (pending == null || pending.item() != item
                    || pending.state() != PendingDispatchState.STAGED) {
                return false;
            }
            dispatchPending.remove(item);
            queueDepth.decrementAndGet();
            queueVersion.incrementAndGet();
            return true;
        } finally {
            queueLock.unlock();
        }
    }

    /** Terminal fallback for a callback that escaped while owning CLAIMED. */
    private boolean completeClaimedPendingDispatch(BatchItem item) {
        queueLock.lock();
        try {
            PendingDispatch pending = dispatchPending.get(item);
            if (pending == null || pending.item() != item
                    || pending.state() != PendingDispatchState.CLAIMED) {
                return false;
            }
            dispatchPending.remove(item);
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
    PendingRestoreResult restorePendingDispatch(BatchItem item) {
        return restorePendingDispatch(item, null);
    }

    /** Callback-finally fallback: never restore a request already claimed by the scheduler. */
    private PendingRestoreResult restoreStagedPendingDispatch(BatchItem item) {
        return restorePendingDispatch(item, PendingDispatchState.STAGED);
    }

    private PendingRestoreResult restorePendingDispatch(BatchItem item,
                                                         PendingDispatchState requiredState) {
        queueLock.lock();
        try {
            PendingDispatch pending = dispatchPending.get(item);
            if (pending == null || pending.item() != item
                    || (requiredState != null && pending.state() != requiredState)) {
                return PendingRestoreResult.NOT_PENDING;
            }
            dispatchPending.remove(item);
            if (stopped) {
                queueDepth.decrementAndGet();
                queueVersion.incrementAndGet();
                return PendingRestoreResult.STOPPED;
            }
            queue.add(item);
            queueVersion.incrementAndGet();
            return PendingRestoreResult.RESTORED;
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Linearize shutdown with queue and dispatch-pending ownership. Staged
     * items remain engine-unseen and are drained; a callback that already
     * claimed an item owns finishing or restoring it.
     */
    void stopAndDrainTo(List<BatchItem> dst) {
        queueLock.lock();
        try {
            stopped = true;
            int drained = queue.drainTo(dst);
            if (drained > 0) {
                queueDepth.addAndGet(-drained);
            }
            boolean stagedDrained = false;
            java.util.Iterator<Map.Entry<BatchItem, PendingDispatch>> iterator =
                    dispatchPending.entrySet().iterator();
            while (iterator.hasNext()) {
                PendingDispatch pending = iterator.next().getValue();
                if (pending.state() == PendingDispatchState.STAGED) {
                    dst.add(pending.item());
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

    int dispatchPendingSize() {
        queueLock.lock();
        try {
            return dispatchPending.size();
        } finally {
            queueLock.unlock();
        }
    }

    /**
     * Remove head from queue and notify handler of expiry.
     * Only called by algorithms that support deadline-based expiry.
     * Caller is responsible for algorithm-specific logging and state cleanup.
     */
    void dropHead(BatchItem head) {
        remove(head);
        handler.onExpired(head);
    }
}
