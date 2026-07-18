package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
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
    private final AtomicInteger queueDepth;
    private final AtomicLong headSortKey;
    private final BatchSchedulerReporter reporter;
    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchDecisionHandler handler,
                   PriorityBlockingQueue<BatchItem> queue,
                   BatchSchedulerReporter reporter) {
        this(key, prefillEp, cfg, handler, queue, new AtomicInteger(queue.size()),
                new AtomicLong(initialHeadSortKey(queue)), reporter);
    }

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchDecisionHandler handler,
                   PriorityBlockingQueue<BatchItem> queue,
                   AtomicInteger queueDepth,
                   AtomicLong headSortKey,
                   BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.handler = handler;
        this.queue = queue;
        this.queueDepth = queueDepth;
        this.headSortKey = headSortKey;
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

    BatchDecisionHandler handler() {
        return handler;
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

    BatchItem poll() {
        BatchItem item = queue.poll();
        if (item != null) {
            queueDepth.decrementAndGet();
            refreshHeadSortKey();
        }
        return item;
    }

    boolean remove(BatchItem item) {
        boolean removed = queue.remove(item);
        if (removed) {
            queueDepth.decrementAndGet();
            refreshHeadSortKey();
        }
        return removed;
    }

    void drainTo(List<BatchItem> dst) {
        int drained = queue.drainTo(dst);
        if (drained > 0) {
            queueDepth.addAndGet(-drained);
            refreshHeadSortKey();
        }
    }

    void refreshHeadSortKey() {
        headSortKey.set(initialHeadSortKey(queue));
    }

    private static long initialHeadSortKey(PriorityBlockingQueue<BatchItem> queue) {
        BatchItem head = queue.peek();
        return head != null ? head.sortKey() : 0;
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
