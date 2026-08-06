package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Per-worker request batcher: a thin shell that handles thread coordination
 * and side-effect execution (metric reporting, dispatch to the engine,
 * settlement), delegating all queue management and dispatch decisions to
 * {@link FixedWindowBatcherAlgorithm}.
 *
 * <p>One instance per Prefill worker. Requests are submitted via
 * {@link #offer(BatchItem)} and batched by the algorithm. The run loop
 * interprets each {@link BatchDecision}: ready batches go to
 * {@link PrefillEndpoint#submitBatch}; rejected or expired items settle
 * themselves through {@link BatchItem} terminal transitions. The algorithm
 * removes items from its internal queue inside {@link FixedWindowBatcherAlgorithm#decide};
 * this class never touches the queue directly.
 */
public class WorkerBatcher {

    private final String key;
    private final FlexlbConfig cfg;
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
        this.algorithm = new FixedWindowBatcherAlgorithm(cfg, prefillEp);
        this.ctx = new BatcherContext(key, prefillEp, cfg, reporter);
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
        if (maxSize > 0 && algorithm.size() >= maxSize) {
            item.failOffer(
                    new IllegalStateException("FlexLB batcher queue full, maxSize=" + maxSize));
            return;
        }
        algorithm.offer(item);
        signalNotEmpty();
    }

    public int queueSize() {
        return algorithm.size();
    }

    /**
     * Current queue depth per Auto-TPM priority level, for periodic gauge
     * reporting. Delegates to {@link FixedWindowBatcherAlgorithm#depthByPriority}.
     */
    public Map<Integer, Integer> depthByPriority() {
        return algorithm.depthByPriority();
    }

    /**
     * Estimated time a new request would wait in the queue before dispatch.
     * Delegates to {@link FixedWindowBatcherAlgorithm#queueWaitMs}.
     */
    public long queueWaitMs() {
        return algorithm.queueWaitMs();
    }

    public void shutdown() {
        stopped = true;
        workerThread.interrupt();
        List<BatchItem> remaining = new ArrayList<>();
        algorithm.drainTo(remaining);
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
                BatchDecision decision = algorithm.decide();
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
     * decision, then hand the batch to the endpoint. The algorithm has
     * already removed the picked items from the queue.
     */
    private void executeDispatch(BatchDecision.Dispatch d) {
        String role = RoleType.PREFILL.name();
        String ip = ctx.prefillEp().getIp();
        ctx.reporter().reportDispatchReason(role, ip, d.reason());
        ctx.reporter().reportBatchSize(role, ip, d.reason(), d.items().size());

        // Compute batch-aggregated cache hit ratio
        long totalSeqLen = 0;
        long totalHitCache = 0;
        long now = System.currentTimeMillis();
        for (BatchItem item : d.items()) {
            totalSeqLen += item.seqLen();
            totalHitCache += item.hitCache();
            // Auto-TPM: per-item queue wait by priority
            ctx.reporter().reportAutoTpmQueueWaitTimeMs(
                    item.priority(), ip, Math.max(0, now - item.enqueuedAtMs()));
        }
        ctx.reporter().reportBatchCacheHitMetrics(role, ip, totalHitCache, totalSeqLen);
        ctx.reporter().reportBatchTotalTokens(role, ip, d.reason(), totalSeqLen);

        Logger.debug("flexlb_batch_decision reason={} picked_size={} "
                        + "wait_ms={} queue_before={} worker={} head_req_id={} head_priority={}",
                d.reason(), d.items().size(), d.headWaitMs(),
                d.queueSizeBefore(), ctx.key(), d.items().get(0).requestId(),
                d.items().get(0).priority());

        ctx.prefillEp().submitBatch(d.items(),
                new DispatchMeta(d.reason()));
    }

    /**
     * Execute a {@link BatchDecision.Drop}: log (deadline expiry only), then
     * settle the item through the matching {@link BatchItem} terminal
     * transition. The algorithm has already removed the item from the queue.
     */
    private void executeDrop(BatchDecision.Drop d) {
        switch (d.cause()) {
            case QUEUE_DEADLINE_EXCEEDED -> {
                Logger.warn("flexlb_batch_drop request_id={} priority={} reason=queue_deadline_exceeded {}",
                        d.item().requestId(), d.item().priority(), d.detail());
                // Auto-TPM: starvation observation — expired count by priority
                ctx.reporter().reportAutoTpmExpiredCount(d.item().priority());
                d.item().failExpired();
            }
            case EXCEEDS_BATCH_TOKEN_CAPACITY ->
                    d.item().failOffer(new IllegalArgumentException(
                            "request cannot fit strict padded batch token capacity: " + d.detail()));
        }
    }

    /**
     * Block until the queue is non-empty, using {@link Condition#await()}.
     *
     * <p>The fast path checks {@link FixedWindowBatcherAlgorithm#size()}
     * without holding the lock. Only when the queue is empty does the thread
     * acquire {@link #waitLock} and await on {@link #notEmpty}, which is
     * signalled by {@link #offer(BatchItem)} after each successful enqueue.
     */
    private void waitForNonEmpty() throws InterruptedException {
        if (algorithm.size() > 0) {
            return;
        }
        waitLock.lock();
        try {
            while (algorithm.size() == 0) {
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
}
