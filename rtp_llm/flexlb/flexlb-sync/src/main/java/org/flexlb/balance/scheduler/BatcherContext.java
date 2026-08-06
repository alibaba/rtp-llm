package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
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

    /** Dispatch-interval sliding average for the 8.4 queue wait estimate. */
    private volatile long lastDispatchAtMs;
    private volatile double dispatchIntervalEmaMs;

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
     * Auto-TPM: {@link WorkerBatcher#AUTO_TPM_QUEUE_ORDER}), suitable for
     * greedy-fill iteration in dispatch algorithms.
     */
    List<BatchItem> sortedItems() {
        List<BatchItem> candidates = new ArrayList<>(queue);
        candidates.sort(queueOrder);
        return candidates;
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
        // The dispatch-interval EMA only feeds the Auto-TPM queue-wait
        // estimate (PrefillQueueManager.estimateWaitMs); skip the synchronized
        // bookkeeping entirely on the legacy path (task10 P2-9).
        if (cfg.isAutoTpmEnabled()) {
            recordDispatchInterval(now());
        }
        for (BatchItem item : items) {
            remove(item);
        }
        handler.onBatchReady(items, meta);
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

    // ---- dispatch interval estimation (design doc 8.4) ----

    private synchronized void recordDispatchInterval(long nowMs) {
        if (lastDispatchAtMs > 0 && nowMs > lastDispatchAtMs) {
            long intervalMs = nowMs - lastDispatchAtMs;
            dispatchIntervalEmaMs = dispatchIntervalEmaMs <= 0
                    ? intervalMs
                    : 0.3 * intervalMs + 0.7 * dispatchIntervalEmaMs;
        }
        lastDispatchAtMs = nowMs;
    }

    /**
     * Sliding-average interval between batch dispatches; before any dispatch
     * is observed, falls back to the algorithm's batching window.
     */
    long avgDispatchIntervalMs() {
        double ema = dispatchIntervalEmaMs;
        if (ema > 0) {
            return Math.max(1, Math.round(ema));
        }
        long windowMs = "fixed_window".equalsIgnoreCase(cfg.getFlexlbBatchAlgorithm())
                ? cfg.getFlexlbBatchFixedWaitMs()
                : cfg.getFlexlbBatchWindowMs();
        return Math.max(1, windowMs);
    }
}
