package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;

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
    private volatile boolean sortedCacheDirty = true;
    private List<BatchItem> sortedCache = null;

    BatcherContext(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                   BatchDecisionHandler handler,
                   PriorityBlockingQueue<BatchItem> queue) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.handler = handler;
        this.queue = queue;
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

    long now() {
        return System.currentTimeMillis();
    }

    // ---- queue inspection ----

    BatchItem peek() {
        return queue.peek();
    }

    boolean isEmpty() {
        return queue.isEmpty();
    }

    int size() {
        return queue.size();
    }

    // ---- queue mutation ----

    BatchItem poll() {
        sortedCacheDirty = true;
        return queue.poll();
    }

    boolean remove(BatchItem item) {
        sortedCacheDirty = true;
        return queue.remove(item);
    }

    void drainTo(List<BatchItem> dst) {
        sortedCacheDirty = true;
        queue.drainTo(dst);
    }

    /**
     * Items sorted by {@link BatchItem#sortKey()}, suitable for
     * greedy-fill iteration in dispatch algorithms.
     * Results are cached until the queue is mutated.
     */
    List<BatchItem> sortedItems() {
        if (!sortedCacheDirty && sortedCache != null) {
            return sortedCache;
        }
        List<BatchItem> candidates = new ArrayList<>(queue);
        candidates.sort(Comparator.comparingLong(BatchItem::sortKey));
        sortedCache = candidates;
        sortedCacheDirty = false;
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
