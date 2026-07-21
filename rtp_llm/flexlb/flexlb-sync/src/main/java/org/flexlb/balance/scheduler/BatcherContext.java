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

/**
 * Controlled access to shared {@link WorkerBatcher} infrastructure.
 *
 * <p>Passed to {@link BatcherAlgorithm} methods so algorithms can
 * inspect and mutate the queue, read config, and invoke callbacks
 * without directly depending on WorkerBatcher internals.
 */
public class BatcherContext {

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig cfg;
    private final BatchDecisionHandler handler;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final AtomicInteger queueDepth;
    private final BatchSchedulerReporter reporter;

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
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.handler = handler;
        this.queue = queue;
        this.queueDepth = queueDepth;
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
        boolean removed = queue.remove(item);
        if (removed) {
            queueDepth.decrementAndGet();
        }
        return removed;
    }

    void drainTo(List<BatchItem> dst) {
        int drained = queue.drainTo(dst);
        if (drained > 0) {
            queueDepth.addAndGet(-drained);
        }
    }

    /**
     * Items sorted by {@link BatchItem#sortKey()}, suitable for
     * greedy-fill iteration in dispatch algorithms.
     */
    List<BatchItem> sortedItems() {
        List<BatchItem> candidates = new ArrayList<>(queue);
        candidates.sort(Comparator.comparingLong(BatchItem::sortKey));
        return candidates;
    }

    /**
     * Effective strict token limit for one FlexLB batch.
     *
     * <p>The Engine's FIFO scheduler rejects a group when the aggregate context
     * length is greater than or equal to {@code max_batch_tokens_size}. Prefer
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

    /** Engine admission uses a strict {@code total < capacity} comparison. */
    static boolean fitsBatchTokenCapacity(long currentTokens, long itemTokens, long capacity) {
        if (currentTokens < 0 || itemTokens < 0 || capacity <= 0 || currentTokens >= capacity) {
            return false;
        }
        return itemTokens < capacity - currentTokens;
    }

    void rejectForBatchTokenCapacity(BatchItem item, long capacity) {
        if (remove(item)) {
            handler.onOfferFailure(item, new IllegalArgumentException(
                    "request seq_len=" + item.seqLen()
                            + " cannot fit strict batch token capacity=" + capacity));
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
}
