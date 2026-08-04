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
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Per-worker request batcher that owns the queue and lifecycle, delegating
 * dispatch decision logic to {@link FixedWindowBatcherAlgorithm}.
 *
 * <p>One instance per Prefill worker. Requests are submitted via
 * {@link #offer(BatchItem)} and batched by the algorithm. Batching decisions
 * are reported through the three functional callbacks supplied at
 * construction ({@code onExpired}, {@code onBatchReady}, {@code onOfferFailure}).
 */
public class WorkerBatcher {

    private final String key;
    private final FlexlbConfig cfg;
    private final BiConsumer<BatchItem, Throwable> onOfferFailure;
    private final PriorityBlockingQueue<BatchItem> queue =
            new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));
    private final AtomicInteger queueDepth = new AtomicInteger();
    private final Thread workerThread;
    private volatile boolean stopped;
    private final FixedWindowBatcherAlgorithm algorithm;
    private final BatcherContext ctx;

    public WorkerBatcher(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                         Consumer<BatchItem> onExpired,
                         BiConsumer<List<BatchItem>, DispatchMeta> onBatchReady,
                         BiConsumer<BatchItem, Throwable> onOfferFailure,
                         BatchSchedulerReporter reporter) {
        this.key = key;
        this.cfg = cfg;
        this.onOfferFailure = onOfferFailure;
        this.algorithm = new FixedWindowBatcherAlgorithm();
        this.ctx = new BatcherContext(
                key, prefillEp, cfg, onExpired, onBatchReady, onOfferFailure,
                queue, queueDepth, reporter);
        this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
        this.workerThread.setDaemon(true);
        this.workerThread.setUncaughtExceptionHandler((t, e) ->
                Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
    }

    public void start() {
        workerThread.start();
    }

    public void offer(BatchItem item) {
        if (stopped) {
            onOfferFailure.accept(item, new IllegalStateException("FlexLB batcher stopped"));
            return;
        }
        int maxSize = cfg.getFlexlbBatchQueueMaxSize();
        if (!reserveQueueSlot(maxSize)) {
            onOfferFailure.accept(item,
                    new IllegalStateException("FlexLB batcher queue full, maxSize=" + maxSize));
            return;
        }
        try {
            long sortKey = algorithm.computeSortKey(ctx, item);
            item.setSortKey(sortKey);
            algorithm.onOffer(ctx, item, System.currentTimeMillis());
            queue.add(item);
        } catch (RuntimeException | Error e) {
            queueDepth.decrementAndGet();
            throw e;
        }
    }

    public int queueSize() {
        return queueDepth.get();
    }

    /**
     * Estimated time a new request would wait in the queue before dispatch.
     * Delegates to {@link FixedWindowBatcherAlgorithm#queueWaitMs}.
     */
    public long queueWaitMs() {
        return algorithm.queueWaitMs(ctx);
    }

    public void shutdown() {
        stopped = true;
        workerThread.interrupt();
        algorithm.onShutdown(ctx);
        List<BatchItem> remaining = new ArrayList<>();
        ctx.drainTo(remaining);
        for (BatchItem item : remaining) {
            onOfferFailure.accept(item,
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
