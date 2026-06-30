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
        this.ctx = new BatcherContext(key, prefillEp, cfg, handler, queue, reporter);
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
    }

    private static BatcherAlgorithm createAlgorithm(FlexlbConfig config) {
        String algoName = config.getFlexlbBatchAlgorithm();
        if ("fixed_window".equalsIgnoreCase(algoName)) {
            return new FixedWindowBatcherAlgorithm();
        }
        // Fallback: slo_budget for any unrecognized value
        return new SloBudgetBatcherAlgorithm();
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
        if (maxSize > 0 && queue.size() >= maxSize) {
            handler.onOfferFailure(item,
                    new IllegalStateException("FlexLB batcher queue full, maxSize=" + maxSize));
            return;
        }
        long sortKey = algorithm.computeSortKey(ctx, item);
        item.setSortKey(sortKey);
        algorithm.onOffer(ctx, item, System.currentTimeMillis());
        queue.add(item);
    }

    public int queueSize() {
        return queue.size();
    }

    public long headSortKey() {
        BatchItem head = queue.peek();
        return head != null ? head.sortKey() : 0;
    }

    /**
     * Estimated remaining wait time of the head request.
     * Delegates to the algorithm-specific {@link BatcherAlgorithm#headWaitMs}.
     */
    public long headWaitMs() {
        return algorithm.headWaitMs(ctx);
    }

    /**
     * Estimated time a new request would wait in the queue before dispatch.
     * Delegates to the algorithm-specific {@link BatcherAlgorithm#queueWaitMs}.
     */
    public long queueWaitMs() {
        return algorithm.queueWaitMs(ctx);
    }

    public void shutdown() {
        stopped = true;
        workerThread.interrupt();
        algorithm.onShutdown(ctx);
        List<BatchItem> remaining = new ArrayList<>();
        queue.drainTo(remaining);
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
}
