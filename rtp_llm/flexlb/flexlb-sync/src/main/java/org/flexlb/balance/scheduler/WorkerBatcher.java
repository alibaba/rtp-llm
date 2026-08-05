package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-worker request batcher that owns the queue, lifecycle, and all side
 * effects (dispatch execution, metric reporting, logging, parking),
 * delegating the pure dispatch decision to {@link FixedWindowBatcherAlgorithm}.
 *
 * <p>One instance per Prefill worker. Requests are submitted via
 * {@link #offer(BatchItem)} and batched by the algorithm. The run loop
 * interprets each {@link BatchDecision}: ready batches go to
 * {@link PrefillEndpoint#submitBatch} via the {@link BatcherContext};
 * rejected or expired items settle themselves through {@link BatchItem}
 * terminal transitions.
 */
public class WorkerBatcher {

    private final String key;
    private final FlexlbConfig cfg;
    private final PriorityBlockingQueue<BatchItem> queue =
            new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));
    private final AtomicInteger queueDepth = new AtomicInteger();
    private final Thread workerThread;
    private volatile boolean stopped;
    private final FixedWindowBatcherAlgorithm algorithm;
    private final BatcherContext ctx;

    /** Guard + signal for the run-loop's blocking wait when the queue is empty. */
    private final ReentrantLock waitLock = new ReentrantLock();
    private final Condition notEmpty = waitLock.newCondition();

    public WorkerBatcher(String key, PrefillEndpoint prefillEp, FlexlbConfig cfg,
                         BatchSchedulerReporter reporter) {
        this.key = key;
        this.cfg = cfg;
        this.algorithm = new FixedWindowBatcherAlgorithm();
        this.ctx = new BatcherContext(key, prefillEp, cfg, queue, queueDepth, reporter);
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
            item.failOffer(new IllegalStateException("FlexLB batcher stopped"));
            return;
        }
        int maxSize = cfg.getFlexlbBatchQueueMaxSize();
        if (!reserveQueueSlot(maxSize)) {
            item.failOffer(
                    new IllegalStateException("FlexLB batcher queue full, maxSize=" + maxSize));
            return;
        }
        try {
            long sortKey = algorithm.computeSortKey(ctx, item);
            item.setSortKey(sortKey);
            queue.add(item);
            signalNotEmpty();
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
        List<BatchItem> remaining = new ArrayList<>();
        ctx.drainTo(remaining);
        for (BatchItem item : remaining) {
            item.failOffer(
                    new CancellationException("FlexLB batcher stopped: " + key));
        }
    }

    // ==================== Internal: Run loop ====================

    private void runLoop() {
        while (!stopped && !Thread.currentThread().isInterrupted()) {
            try {
                waitForNonEmpty();
                BatchDecision decision = algorithm.decide(ctx);
                if (decision == null) {
                    // No action this cycle (park / engine backpressure)
                    TimeUnit.MILLISECONDS.sleep(1);
                    continue;
                }
                switch (decision) {
                    case BatchDecision.Dispatch d -> executeDispatch(d);
                    case BatchDecision.Drop d -> executeDrop(d);
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable t) {
                Logger.error("WorkerBatcher[{}] loop failed", key, t);
            }
        }
    }

    /**
     * Execute a {@link BatchDecision.Dispatch}: report metrics, log the
     * decision, then remove the items from the queue and hand the batch to
     * the endpoint.
     */
    private void executeDispatch(BatchDecision.Dispatch d) {
        String role = RoleType.PREFILL.name();
        String ip = ctx.prefillEp().getIp();
        ctx.reporter().reportDispatchReason(role, ip, d.reason());
        ctx.reporter().reportBatchSize(role, ip, d.reason(), d.items().size());

        // Compute batch-aggregated cache hit ratio
        long totalSeqLen = 0;
        long totalHitCache = 0;
        for (BatchItem item : d.items()) {
            totalSeqLen += item.seqLen();
            totalHitCache += item.hitCache();
        }
        ctx.reporter().reportBatchCacheHitMetrics(role, ip, totalHitCache, totalSeqLen);
        ctx.reporter().reportBatchTotalTokens(role, ip, d.reason(), totalSeqLen);

        Logger.debug("flexlb_batch_decision reason={} picked_size={} "
                        + "wait_ms={} queue_before={} worker={} head_req_id={}",
                d.reason(), d.items().size(), d.headWaitMs(),
                d.queueSizeBefore(), ctx.key(), d.items().get(0).requestId());

        ctx.dispatch(d.items(),
                new DispatchMeta(d.reason()));
    }

    /**
     * Execute a {@link BatchDecision.Drop}: log (deadline expiry only), then
     * remove the head item from the queue and settle it through the matching
     * {@link BatchItem} terminal transition.
     */
    private void executeDrop(BatchDecision.Drop d) {
        switch (d.cause()) {
            case QUEUE_DEADLINE_EXCEEDED -> {
                Logger.warn("flexlb_batch_drop request_id={} reason=queue_deadline_exceeded {}",
                        d.item().requestId(), d.detail());
                ctx.dropHead(d.item());
            }
            case EXCEEDS_BATCH_TOKEN_CAPACITY ->
                    ctx.rejectForBatchTokenCapacity(d.item(), ctx.batchTokenCapacity());
        }
    }

    /**
     * Block until the queue is non-empty, using {@link Condition#await()}
     * instead of the previous {@code take()+put()} round-trip which caused
     * invalid re-sorting of the priority queue and wasted operations.
     *
     * <p>The fast path checks {@link PriorityBlockingQueue#peek()} without
     * holding the lock. Only when the queue is empty does the thread
     * acquire {@link #waitLock} and await on {@link #notEmpty}, which is
     * signalled by {@link #offer(BatchItem)} after each successful add.
     */
    private void waitForNonEmpty() throws InterruptedException {
        if (queue.peek() != null) {
            return;
        }
        waitLock.lock();
        try {
            while (queue.peek() == null) {
                notEmpty.await();
            }
        } finally {
            waitLock.unlock();
        }
    }

    /** Signal the run-loop thread that an item has been added. */
    private void signalNotEmpty() {
        waitLock.lock();
        try {
            notEmpty.signal();
        } finally {
            waitLock.unlock();
        }
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
