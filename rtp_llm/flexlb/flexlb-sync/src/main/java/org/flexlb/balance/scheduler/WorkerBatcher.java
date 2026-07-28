package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
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
    private final Object queueMutex = new Object();
    private final IdentityHashMap<BatchItem, QueueHandle> handles = new IdentityHashMap<>();
    private final AtomicInteger publishedQueueDepth = new AtomicInteger();
    private final AtomicLong publishedHeadSortKey = new AtomicLong();
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
                key, prefillEp, cfg, handler, queue, handles, queueMutex,
                publishedQueueDepth, publishedHeadSortKey, reporter);
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

    public QueueHandle offer(BatchItem item) {
        int maxSize = cfg.getFlexlbBatchQueueMaxSize();
        rejectIfUnavailable(maxSize);
        long sortKey = algorithm.computeSortKey(ctx, item);
        item.setSortKey(sortKey);
        synchronized (queueMutex) {
            rejectIfUnavailable(maxSize);
            algorithm.onOffer(ctx, item, System.currentTimeMillis());
            QueueHandle handle = new QueueHandle(this, item);
            handles.put(item, handle);
            queue.add(item);
            publishedQueueDepth.incrementAndGet();
            ctx.publishHead();
            queueMutex.notifyAll();
            return handle;
        }
    }

    private void rejectIfUnavailable(int maxSize) {
        if (stopped) {
            throw new RejectedExecutionException("FlexLB batcher stopped");
        }
        if (maxSize > 0 && publishedQueueDepth.get() >= maxSize) {
            throw new RejectedExecutionException(
                    "FlexLB batcher queue full, maxSize=" + maxSize);
        }
    }

    public RemoveResult remove(QueueHandle handle) {
        if (handle == null || handle.owner != this) {
            return RemoveResult.FOREIGN;
        }
        synchronized (queueMutex) {
            if (handle.state == RemoveResult.CLAIMED) {
                return RemoveResult.CLAIMED;
            }
            if (handle.state == RemoveResult.REMOVED) {
                return RemoveResult.REMOVED;
            }
            if (!queue.remove(handle.item)) {
                throw new IllegalStateException("queued handle has no queue owner");
            }
            handles.remove(handle.item);
            handle.state = RemoveResult.REMOVED;
            publishedQueueDepth.decrementAndGet();
            ctx.publishHead();
        }
        algorithm.onExternalRemove(ctx, handle.item);
        return RemoveResult.REMOVED;
    }

    public int queueSize() {
        return publishedQueueDepth.get();
    }

    /**
     * Estimated remaining wait time of the head request.
     * Uses deadline semantics for SLO batching and elapsed-window semantics for fixed-window batching.
     */
    public long headWaitMs() {
        long currentHeadSortKey = publishedHeadSortKey.get();
        if (publishedQueueDepth.get() == 0 || currentHeadSortKey == 0) {
            return 0;
        }
        long now = System.currentTimeMillis();
        if (algorithm instanceof FixedWindowBatcherAlgorithm) {
            long elapsedMs = now - currentHeadSortKey;
            return Math.max(0, cfg.getFlexlbBatchFixedWaitMs() - elapsedMs);
        }
        return Math.max(0, currentHeadSortKey - now);
    }

    public void shutdown() {
        synchronized (queueMutex) {
            if (stopped) {
                return;
            }
            stopped = true;
            queueMutex.notifyAll();
        }
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
        synchronized (queueMutex) {
            while (!stopped && publishedQueueDepth.get() == 0) {
                queueMutex.wait();
            }
            if (stopped) {
                throw new InterruptedException("batcher stopped");
            }
        }
    }

    public enum RemoveResult {
        QUEUED,
        REMOVED,
        CLAIMED,
        FOREIGN
    }

    public static final class QueueHandle {
        private final WorkerBatcher owner;
        private final BatchItem item;
        RemoveResult state = RemoveResult.QUEUED;

        private QueueHandle(WorkerBatcher owner, BatchItem item) {
            this.owner = owner;
            this.item = item;
        }

    }
}
