package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.service.monitor.BatchSchedulerReporter;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

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
    private final Map<BatchItem, WorkerBatcher.QueueHandle> handles;
    private final Object queueMutex;
    private final AtomicInteger publishedQueueDepth;
    private final AtomicLong publishedHeadSortKey;
    private final BatchSchedulerReporter reporter;
    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchDecisionHandler handler,
                   PriorityBlockingQueue<BatchItem> queue,
                   Map<BatchItem, WorkerBatcher.QueueHandle> handles,
                   Object queueMutex,
                   AtomicInteger publishedQueueDepth,
                   AtomicLong publishedHeadSortKey,
                   BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.handler = handler;
        this.queue = queue;
        this.handles = handles;
        this.queueMutex = queueMutex;
        this.publishedQueueDepth = publishedQueueDepth;
        this.publishedHeadSortKey = publishedHeadSortKey;
        this.reporter = reporter;
    }

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

    BatchItem peek() {
        synchronized (queueMutex) {
            return queue.peek();
        }
    }

    boolean isEmpty() {
        return publishedQueueDepth.get() == 0;
    }

    int size() {
        return publishedQueueDepth.get();
    }

    boolean remove(BatchItem item) {
        synchronized (queueMutex) {
            boolean removed = queue.remove(item);
            if (removed) {
                WorkerBatcher.QueueHandle handle = handles.remove(item);
                if (handle == null) {
                    throw new IllegalStateException("queued item has no ownership handle");
                }
                handle.state = WorkerBatcher.RemoveResult.CLAIMED;
                publishedQueueDepth.decrementAndGet();
                publishHead();
            }
            return removed;
        }
    }

    boolean runIfQueued(BatchItem item, Runnable action) {
        synchronized (queueMutex) {
            if (!handles.containsKey(item)) {
                return false;
            }
            action.run();
            return true;
        }
    }

    void drainTo(List<BatchItem> dst) {
        synchronized (queueMutex) {
            int drained = queue.drainTo(dst);
            if (drained > 0) {
                publishedQueueDepth.addAndGet(-drained);
                for (BatchItem item : dst) {
                    WorkerBatcher.QueueHandle handle = handles.remove(item);
                    if (handle != null) {
                        handle.state = WorkerBatcher.RemoveResult.REMOVED;
                    }
                }
                publishHead();
            }
        }
    }

    void publishHead() {
        BatchItem head = queue.peek();
        publishedHeadSortKey.set(head == null ? 0 : head.sortKey());
    }

    List<BatchItem> sortedItems() {
        List<BatchItem> candidates;
        synchronized (queueMutex) {
            candidates = new ArrayList<>(queue);
        }
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

    void dispatch(List<BatchItem> items, DispatchMeta meta) {
        List<BatchItem> claimed = new ArrayList<>(items.size());
        for (BatchItem item : items) {
            if (remove(item)) {
                claimed.add(item);
            }
        }
        if (!claimed.isEmpty()) {
            handler.onBatchReady(claimed,
                    new DispatchMeta(meta.reason(), size()));
        }
    }

    void dropHead(BatchItem head) {
        if (remove(head)) {
            handler.onExpired(head);
        }
    }
}
