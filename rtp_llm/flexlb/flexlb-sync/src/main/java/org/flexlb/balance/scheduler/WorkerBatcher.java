package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-worker request batcher that owns the queue and lifecycle, delegating
 * dispatch decision logic to a pluggable {@link BatcherAlgorithm}.
 *
 * <p>One instance per Prefill worker. Requests are submitted via
 * {@link #offer(BatchItem)} and batched by the configured algorithm.
 */
public class WorkerBatcher {

    private final String key;
    private final PrefillEndpoint prefillEp;
    private final FlexlbConfig cfg;
    private final BatchDecisionHandler handler;
    private final PriorityBlockingQueue<BatchItem> queue =
            new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));
    private final AtomicInteger queueDepth = new AtomicInteger();
    private final AtomicLong headSortKey = new AtomicLong();
    private final Thread workerThread;
    private volatile boolean stopped;
    private final BatcherAlgorithm algorithm;
    private final BatcherContext ctx;

    public WorkerBatcher(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                         BatchDecisionHandler handler,
                         BatchSchedulerReporter reporter) {
        this.key = key;
        this.prefillEp = prefillEp;
        this.cfg = cfg;
        this.handler = handler;
        this.algorithm = createAlgorithm(cfg);
        this.ctx = new BatcherContext(
                key, prefillEp, cfg, handler, queue, queueDepth, headSortKey, reporter);
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
    }

    private static BatcherAlgorithm createAlgorithm(FlexlbConfig config) {
        String algoName = config.getFlexlbBatchAlgorithm();
        if ("slo_budget".equalsIgnoreCase(algoName)) {
            return new SloBudgetBatcherAlgorithm();
        }
        // Fallback: fixed_window for any unrecognized value (safer default)
        return new FixedWindowBatcherAlgorithm();
    }

    public void start() {
        workerThread.start();
    }

    public void offer(BatchItem item) {
        if (stopped) {
            handler.onOfferFailure(item, new IllegalStateException("FlexLB batcher stopped"));
            return;
        }
        int maxSize = cfg.getFlexlbBatchQueueMaxSize();
        if (!reserveQueueSlot(maxSize)) {
            handler.onOfferFailure(item,
                    new IllegalStateException("FlexLB batcher queue full, maxSize=" + maxSize));
            return;
        }
        try {
            long sortKey = algorithm.computeSortKey(ctx, item);
            item.setSortKey(sortKey);
            algorithm.onOffer(ctx, item, System.currentTimeMillis());
            queue.add(item);
            ctx.refreshHeadSortKey();
        } catch (RuntimeException | Error e) {
            queueDepth.decrementAndGet();
            ctx.refreshHeadSortKey();
            throw e;
        }
    }

    public int queueSize() {
        return queueDepth.get();
    }

    public long headSortKey() {
        return headSortKey.get();
    }

    /**
     * Estimated remaining wait time of the head request.
     * Delegates to the algorithm-specific {@link BatcherAlgorithm#headWaitMs}.
     */
    public long headWaitMs() {
        long currentHeadSortKey = headSortKey.get();
        if (queueDepth.get() == 0 || currentHeadSortKey == 0) {
            return 0;
        }
        long now = System.currentTimeMillis();
        if (algorithm instanceof FixedWindowBatcherAlgorithm) {
            long elapsedMs = now - currentHeadSortKey;
            return Math.max(0, cfg.getFlexlbBatchFixedWaitMs() - elapsedMs);
        }
        return Math.max(0, currentHeadSortKey - now);
    }

    /**
     * Estimated time a new request would wait in the queue before dispatch.
     * Delegates to the algorithm-specific {@link BatcherAlgorithm#queueWaitMs}.
     */
    public long queueWaitMs() {
        return algorithm.queueWaitMs(ctx);
    }

    /**
     * Return queue items that would be in the same batch as a new request.
     * Uses remaining = queueSize % batchMaxCount to determine which items
     * the new request would join after full-batch dispatches.
     *
     * @return the last 'remaining' sorted items, or empty list if remaining == 0
     */
    public List<BatchItem> peekBatchItems() {
        int queueSize = queueDepth.get();
        if (queueSize == 0) {
            return List.of();
        }
        int batchMaxCount = Math.max(1, cfg.getFlexlbBatchSizeMax());
        int remaining = queueSize % batchMaxCount;
        if (remaining == 0) {
            return List.of();
        }
        List<BatchItem> sorted = ctx.sortedItems();
        return new ArrayList<>(sorted.subList(sorted.size() - remaining, sorted.size()));
    }

    public void shutdown() {
        stopped = true;
        workerThread.interrupt();
        algorithm.onShutdown(ctx);
        List<BatchItem> remaining = new ArrayList<>();
        ctx.drainTo(remaining);
        for (BatchItem item : remaining) {
            handler.onOfferFailure(item,
                    new CancellationException("FlexLB batcher stopped: " + key));
        }
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        while (!stopped && !Thread.currentThread().isInterrupted()) {
            try {
                waitForNonEmpty();
                algorithm.processQueue(ctx);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                Logger.error("WorkerBatcher[{}] loop failed", key, t);
            }
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        BatchItem item = queue.take();
        queue.put(item);
    }

    private boolean reserveQueueSlot(int maxSize) {
        if (maxSize <= 0) {
            queueDepth.incrementAndGet();
            return true;
        }
        while (true) {
            int current = queueDepth.get();
            if (current >= maxSize) {
                return false;
            }
            if (queueDepth.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }
}
