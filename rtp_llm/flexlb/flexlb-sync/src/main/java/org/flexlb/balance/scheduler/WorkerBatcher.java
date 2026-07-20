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

    /**
     * Estimated remaining wait time of the head request.
     * Uses deadline semantics for SLO batching and elapsed-window semantics for fixed-window batching.
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
     * Uses remaining = sortedSize % batchMaxCount to determine which items
     * the new request would join after full-batch dispatches.
     *
     * <p>Performance short-circuit: {@code queueDepth} is checked first as
     * an upper bound on the actual queue size. Because {@code offer}
     * increments {@code queueDepth} before enqueuing and {@code poll}
     * dequeues before decrementing, a value of 0 guarantees the queue is
     * empty, allowing an early return without the O(n) snapshot copy and
     * sort. A non-zero {@code queueDepth} does not guarantee items are
     * present (the counter may lag the actual queue), so the subsequent
     * {@code sortedItems} snapshot is still the source of truth for size.
     *
     * <p>{@code remaining} and the {@code subList} bounds are both derived
     * from the single {@link BatcherContext#sortedItems()} snapshot, whose
     * size is stable for the duration of this call. This avoids the race
     * condition where reading {@code queueDepth} (an AtomicInteger that the
     * batcher run-loop thread decretes concurrently) and then reading the
     * queue snapshot non-atomically could produce {@code remaining > snapshot
     * .size()}, making {@code subList}'s first argument negative and throwing
     * {@link IndexOutOfBoundsException}.
     *
     * @return the last 'remaining' sorted items, or empty list if remaining == 0
     */
    public List<BatchItem> peekBatchItems() {
        // 快速短路：queueDepth 是队列大小的上界（offer 先增计数后入队，
        // poll 先出队后减计数），为 0 时队列必然为空，无需 O(n) 拷贝+排序。
        if (queueDepth.get() == 0) {
            return List.of();
        }
        List<BatchItem> sorted = ctx.sortedItems();
        int sortedSize = sorted.size();
        if (sortedSize == 0) {
            return List.of();
        }
        int batchMaxCount = Math.max(1, cfg.getFlexlbBatchSizeMax());
        int remaining = sortedSize % batchMaxCount;
        if (remaining == 0) {
            return List.of();
        }
        return new ArrayList<>(sorted.subList(sortedSize - remaining, sortedSize));
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
