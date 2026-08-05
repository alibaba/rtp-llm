package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
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
 * <p>The read-only subset is exposed to {@link FixedWindowBatcherAlgorithm}
 * through the {@link BatcherReadView} interface so the algorithm can inspect
 * the queue, config, and engine state without producing side effects. The
 * mutating execution methods ({@link #dispatch}, {@link #dropHead},
 * {@link #rejectForBatchTokenCapacity}, {@link #remove}, {@link #drainTo})
 * remain package-private for {@link WorkerBatcher} to execute decisions.
 * Ready batches go straight to {@link PrefillEndpoint#submitBatch}; rejected
 * or expired items settle themselves through {@link BatchItem} terminal
 * transitions.
 */
public class BatcherContext implements BatcherReadView {

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig cfg;
    private final PriorityBlockingQueue<BatchItem> queue;
    private final AtomicInteger queueDepth;
    private final BatchSchedulerReporter reporter;

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   PriorityBlockingQueue<BatchItem> queue,
                   BatchSchedulerReporter reporter) {
        this(key, prefillEp, cfg, queue, new AtomicInteger(queue.size()), reporter);
    }

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   PriorityBlockingQueue<BatchItem> queue,
                   AtomicInteger queueDepth,
                   BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.queue = queue;
        this.queueDepth = queueDepth;
        this.reporter = reporter;
    }

    // ---- accessors ----

    @Override
    public String key() {
        return key;
    }

    PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    @Override
    public FlexlbConfig cfg() {
        return cfg;
    }

    BatchSchedulerReporter reporter() {
        return reporter;
    }

    @Override
    public long now() {
        return System.currentTimeMillis();
    }

    /** Current inflight batch count on the prefill worker, for backpressure. */
    @Override
    public int currentInflightCount() {
        return prefillEp.prefillInflightCount() + prefillEp.prefillEngineTaskCount();
    }

    /** Prefill-time predictor for predictor-based early dispatch (read-only). */
    @Override
    public PrefillTimePredictor predictor() {
        return prefillEp.getPredictor();
    }

    // ---- queue inspection ----

    @Override
    public BatchItem peek() {
        return queue.peek();
    }

    @Override
    public boolean isEmpty() {
        return queueDepth.get() == 0;
    }

    @Override
    public int size() {
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
    @Override
    public List<BatchItem> sortedItems() {
        List<BatchItem> candidates = new ArrayList<>(queue);
        candidates.sort(Comparator.comparingLong(BatchItem::sortKey));
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
    @Override
    public long batchTokenCapacity() {
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
    @Override
    public long batchKvCapacity() {
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
            item.failOffer(new IllegalArgumentException(
                    "request seq_len=" + item.seqLen()
                            + " cannot fit strict padded batch token capacity=" + capacity));
        }
    }

    private static long positiveOrUnlimited(long value) {
        return value > 0 ? value : Long.MAX_VALUE;
    }

    // ---- dispatch helpers (shared infrastructure) ----

    /**
     * Remove items from queue and hand the ready batch to the endpoint.
     * Called by {@link WorkerBatcher} when executing a
     * {@link BatchDecision.Dispatch}; the batcher is responsible for metric
     * reporting and decision logging before calling this.
     */
    void dispatch(List<BatchItem> items, DispatchMeta meta) {
        for (BatchItem item : items) {
            remove(item);
        }
        prefillEp.submitBatch(items, meta);
    }

    /**
     * Remove head from queue and settle it as expired.
     * Called by {@link WorkerBatcher} when executing a
     * {@link BatchDecision.Drop} with cause
     * {@link BatchDecision.DropCause#QUEUE_DEADLINE_EXCEEDED}; the batcher
     * is responsible for drop logging before calling this.
     */
    void dropHead(BatchItem head) {
        remove(head);
        head.failExpired();
    }
}
