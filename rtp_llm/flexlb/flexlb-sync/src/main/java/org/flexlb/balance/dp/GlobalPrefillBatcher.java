package org.flexlb.balance.dp;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DpBatchReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Per-model global queue for DP batching with SLO-aware deadline and
 * compute-length bucketing.
 *
 * <h3>Bucket structure</h3>
 * Requests are grouped by {@code computeTokenLength / dpBucketIntervalTokens}
 * so that a single batch contains requests with similar effective compute,
 * reducing padding waste at the DP barrier. When bucketing is disabled
 * ({@code dpBucketIntervalTokens == 0}), all requests share bucket 0 and the
 * behavior degenerates to the original FIFO global queue.
 *
 * <h3>Triggers</h3>
 * <ol>
 *   <li><b>Bucket full:</b> a single bucket reaches {@code dpSize} — flush
 *       that bucket immediately.</li>
 *   <li><b>Per-request timeout:</b> the head request waited longer than
 *       {@code dpBatchTimeoutMs} — force-flush with bucket merging.</li>
 *   <li><b>SLO deadline:</b> the earliest deadline across all queued requests
 *       is reached — flush with bucket merging.</li>
 *   <li><b>Window timer:</b> fallback safety net using the configured
 *       {@code dpBatchWindowMs}.</li>
 * </ol>
 */
public class GlobalPrefillBatcher {

    private final String model;
    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;
    private final DispatchPlanner planner;
    private final Consumer<PrefillBatch> dispatchCallback;
    private final ScheduledExecutorService timerExecutor;
    private final CacheAwareService cacheAwareService;
    private final DpBatchReporter dpBatchReporter;

    // --- guarded by synchronized(this) ---
    private final HashMap<Integer, ArrayDeque<QueuedRequest>> buckets = new HashMap<>();
    private int totalSize = 0;
    private long earliestDeadlineMicros = Long.MAX_VALUE;
    private long avgQueueTimeMicros = 0;
    private volatile ScheduledFuture<?> windowTimer;

    private static final double EWMA_ALPHA = 0.1;

    public GlobalPrefillBatcher(String model,
                                ConfigService configService,
                                EngineWorkerStatus engineWorkerStatus,
                                DispatchPlanner planner,
                                Consumer<PrefillBatch> dispatchCallback,
                                ScheduledExecutorService timerExecutor,
                                CacheAwareService cacheAwareService,
                                DpBatchReporter dpBatchReporter) {
        this.model = model;
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
        this.planner = planner;
        this.dispatchCallback = dispatchCallback;
        this.timerExecutor = timerExecutor;
        this.cacheAwareService = cacheAwareService;
        this.dpBatchReporter = dpBatchReporter;
    }

    public GlobalPrefillBatcher(String model,
                                ConfigService configService,
                                EngineWorkerStatus engineWorkerStatus,
                                DispatchPlanner planner,
                                Consumer<PrefillBatch> dispatchCallback,
                                ScheduledExecutorService timerExecutor,
                                CacheAwareService cacheAwareService) {
        this(model, configService, engineWorkerStatus, planner, dispatchCallback, timerExecutor,
                cacheAwareService, null);
    }

    public GlobalPrefillBatcher(String model,
                                ConfigService configService,
                                EngineWorkerStatus engineWorkerStatus,
                                DispatchPlanner planner,
                                Consumer<PrefillBatch> dispatchCallback,
                                ScheduledExecutorService timerExecutor) {
        this(model, configService, engineWorkerStatus, planner, dispatchCallback, timerExecutor, null, null);
    }

    public void offer(QueuedRequest req) {
        FlexlbConfig cfg = configService.loadBalanceConfig();
        int dpSize = currentDpSize(cfg);
        long batchWindowMs = cfg.getDpBatchWindowMs();
        long requestTimeoutMs = cfg.getDpBatchTimeoutMs();
        int bucketInterval = cfg.getDpBucketIntervalTokens();

        QueuedRequest enriched = enrichRequest(req, cfg, dpSize);

        List<QueuedRequest> drained = null;
        DpBatchReporter.FlushReason flushReason = null;
        synchronized (this) {
            int bIdx = enriched.bucketIndex();
            buckets.computeIfAbsent(bIdx, k -> new ArrayDeque<>()).add(enriched);
            totalSize++;
            earliestDeadlineMicros = Math.min(earliestDeadlineMicros, enriched.sloDeadlineMicros());

            long nowMicros = System.nanoTime() / 1000;

            if (bucketSize(bIdx) >= dpSize) {
                drained = drainBucketLocked(bIdx, dpSize, batchWindowMs);
                flushReason = DpBatchReporter.FlushReason.BUCKET_FULL;
            } else if (headWaitedTooLong(requestTimeoutMs)) {
                drained = drainWithMergeLocked(dpSize, batchWindowMs);
                flushReason = DpBatchReporter.FlushReason.PER_REQUEST_TIMEOUT;
            } else if (earliestDeadlineMicros <= nowMicros) {
                drained = drainWithMergeLocked(dpSize, batchWindowMs);
                flushReason = DpBatchReporter.FlushReason.DEADLINE;
            } else {
                armTimerLocked(batchWindowMs, nowMicros);
            }
        }
        if (drained != null) {
            planAndDispatch(drained, dpSize, cfg, flushReason);
        }
    }

    public boolean cancelInQueue(long requestId) {
        synchronized (this) {
            for (Map.Entry<Integer, ArrayDeque<QueuedRequest>> entry : buckets.entrySet()) {
                Iterator<QueuedRequest> it = entry.getValue().iterator();
                while (it.hasNext()) {
                    QueuedRequest qr = it.next();
                    if (qr.requestId() == requestId) {
                        it.remove();
                        totalSize--;
                        qr.future().completeExceptionally(
                                new CancellationException("Cancelled while queued in DP batcher"));
                        recalcEarliestDeadlineLocked();
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public int queueSize() {
        synchronized (this) {
            return totalSize;
        }
    }

    // ============== internal ==============

    private QueuedRequest enrichRequest(QueuedRequest raw, FlexlbConfig cfg, int dpSize) {
        if (raw.ctx() == null || raw.ctx().getRequest() == null) {
            return raw;
        }

        long seqLen = raw.ctx().getRequest().getSeqLen();

        long cacheMatchedTokens = 0;
        if (cfg.isCacheAwareSchedulingEnabled() && cacheAwareService != null) {
            List<Long> cacheKeys = raw.ctx().getRequest().getBlockCacheKeys();
            if (cacheKeys != null && !cacheKeys.isEmpty()) {
                long blockSize = guessBlockSize();
                String bestPrefillIp = guessPrefillIpPort();
                if (bestPrefillIp != null) {
                    int prefixBlocks = cacheAwareService.findMatchingPrefixLength(bestPrefillIp, cacheKeys);
                    cacheMatchedTokens = prefixBlocks * blockSize;
                }
            }
        }

        int computeTokenLength = (int) Math.max(0, seqLen - cacheMatchedTokens);
        int bucketInterval = cfg.getDpBucketIntervalTokens();
        int bucketIndex = bucketInterval > 0 ? computeTokenLength / bucketInterval : 0;

        long avgQueueTimeMs;
        synchronized (this) {
            avgQueueTimeMs = avgQueueTimeMicros / 1000;
        }
        long sloDeadlineMicros = BatchDeadlineEstimator.computeDeadlineMicros(
                System.nanoTime() / 1000, seqLen, cacheMatchedTokens,
                avgQueueTimeMs,
                cfg.getDpTtftSloMs(), cfg.getDpSafeMarginMs(),
                cfg.getDpMinBatchIntervalMs(), cfg.getDpMaxBatchIntervalMs());

        raw.ctx().setCacheMatchedTokens(cacheMatchedTokens);
        return QueuedRequest.of(raw.ctx(), raw.future(), computeTokenLength, sloDeadlineMicros, bucketIndex);
    }

    private void flushOnTimeout() {
        FlexlbConfig cfg = configService.loadBalanceConfig();
        int dpSize = currentDpSize(cfg);
        long batchWindowMs = cfg.getDpBatchWindowMs();

        List<QueuedRequest> drained = null;
        synchronized (this) {
            windowTimer = null;
            if (totalSize > 0) {
                drained = drainWithMergeLocked(dpSize, batchWindowMs);
            }
        }
        if (drained != null) {
            planAndDispatch(drained, dpSize, cfg, DpBatchReporter.FlushReason.WINDOW_TIMER);
        }
    }

    /** Drain up to dpSize from a single bucket. Caller MUST hold synchronized(this). */
    private List<QueuedRequest> drainBucketLocked(int bucketIdx, int dpSize, long batchWindowMs) {
        cancelTimerLocked();
        ArrayDeque<QueuedRequest> deque = buckets.get(bucketIdx);
        if (deque == null || deque.isEmpty()) {
            return null;
        }

        int count = Math.min(deque.size(), dpSize);
        List<QueuedRequest> chunk = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            chunk.add(deque.poll());
        }
        totalSize -= chunk.size();
        if (deque.isEmpty()) {
            buckets.remove(bucketIdx);
        }

        updateEwmaLocked(chunk);
        recalcEarliestDeadlineLocked();
        rearmIfNeededLocked(batchWindowMs);
        return chunk.isEmpty() ? null : chunk;
    }

    /** Drain up to dpSize with greedy bucket merging. Caller MUST hold synchronized(this). */
    private List<QueuedRequest> drainWithMergeLocked(int dpSize, long batchWindowMs) {
        cancelTimerLocked();
        if (totalSize == 0) {
            return null;
        }

        int targetBucket = findMostUrgentBucket();
        List<QueuedRequest> chunk = new ArrayList<>(dpSize);

        // Phase 1: drain from target bucket
        drainFromBucket(targetBucket, dpSize, chunk);

        // Phase 2: expand to adjacent buckets by distance
        if (chunk.size() < dpSize) {
            TreeMap<Integer, ArrayDeque<QueuedRequest>> sorted = new TreeMap<>(buckets);
            for (int distance = 1; chunk.size() < dpSize; distance++) {
                boolean found = false;
                int lo = targetBucket - distance;
                int hi = targetBucket + distance;
                if (sorted.containsKey(lo)) {
                    drainFromBucket(lo, dpSize - chunk.size(), chunk);
                    found = true;
                }
                if (chunk.size() < dpSize && sorted.containsKey(hi)) {
                    drainFromBucket(hi, dpSize - chunk.size(), chunk);
                    found = true;
                }
                if (!found && lo < sorted.firstKey() && hi > sorted.lastKey()) {
                    break;
                }
            }
        }

        totalSize -= chunk.size();
        updateEwmaLocked(chunk);
        recalcEarliestDeadlineLocked();
        cleanEmptyBucketsLocked();
        rearmIfNeededLocked(batchWindowMs);
        return chunk.isEmpty() ? null : chunk;
    }

    private void drainFromBucket(int bucketIdx, int maxCount, List<QueuedRequest> out) {
        ArrayDeque<QueuedRequest> deque = buckets.get(bucketIdx);
        if (deque == null) return;
        int count = Math.min(deque.size(), maxCount);
        for (int i = 0; i < count; i++) {
            out.add(deque.poll());
        }
    }

    private int findMostUrgentBucket() {
        int urgentBucket = 0;
        long urgentDeadline = Long.MAX_VALUE;
        for (Map.Entry<Integer, ArrayDeque<QueuedRequest>> entry : buckets.entrySet()) {
            ArrayDeque<QueuedRequest> deque = entry.getValue();
            if (!deque.isEmpty()) {
                QueuedRequest head = deque.peek();
                if (head.sloDeadlineMicros() < urgentDeadline) {
                    urgentDeadline = head.sloDeadlineMicros();
                    urgentBucket = entry.getKey();
                }
            }
        }
        return urgentBucket;
    }

    private void cancelTimerLocked() {
        ScheduledFuture<?> t = windowTimer;
        if (t != null) {
            t.cancel(false);
            windowTimer = null;
        }
    }

    private void armTimerLocked(long batchWindowMs, long nowMicros) {
        if (windowTimer != null) return;
        long deadlineDelayMs = earliestDeadlineMicros < Long.MAX_VALUE
                ? Math.max(1, (earliestDeadlineMicros - nowMicros) / 1000)
                : batchWindowMs;
        long delayMs = Math.min(batchWindowMs, deadlineDelayMs);
        windowTimer = timerExecutor.schedule(this::flushOnTimeout, delayMs, TimeUnit.MILLISECONDS);
    }

    private void rearmIfNeededLocked(long batchWindowMs) {
        if (totalSize > 0 && windowTimer == null) {
            long nowMicros = System.nanoTime() / 1000;
            armTimerLocked(batchWindowMs, nowMicros);
        }
    }

    private void recalcEarliestDeadlineLocked() {
        long min = Long.MAX_VALUE;
        for (ArrayDeque<QueuedRequest> deque : buckets.values()) {
            for (QueuedRequest qr : deque) {
                if (qr.sloDeadlineMicros() < min) {
                    min = qr.sloDeadlineMicros();
                }
            }
        }
        earliestDeadlineMicros = min;
    }

    private void cleanEmptyBucketsLocked() {
        buckets.entrySet().removeIf(e -> e.getValue().isEmpty());
    }

    private void updateEwmaLocked(List<QueuedRequest> drained) {
        if (drained.isEmpty()) return;
        long nowMicros = System.nanoTime() / 1000;
        long totalWait = 0;
        for (QueuedRequest qr : drained) {
            totalWait += (nowMicros - qr.enqueuedAtMicros());
        }
        long avgWait = totalWait / drained.size();
        avgQueueTimeMicros = (long) (EWMA_ALPHA * avgWait + (1 - EWMA_ALPHA) * avgQueueTimeMicros);
    }

    private boolean headWaitedTooLong(long requestTimeoutMs) {
        for (ArrayDeque<QueuedRequest> deque : buckets.values()) {
            if (!deque.isEmpty()) {
                QueuedRequest head = deque.peek();
                if (head.waitMicros() > requestTimeoutMs * 1000L) {
                    return true;
                }
            }
        }
        return false;
    }

    private void planAndDispatch(List<QueuedRequest> drained, int dpSize, FlexlbConfig cfg,
                                  DpBatchReporter.FlushReason flushReason) {
        if (dpBatchReporter != null && flushReason != null) {
            dpBatchReporter.reportSloBatchFlush(model, flushReason, drained.size());
            long nowMicros = System.nanoTime() / 1000;
            for (QueuedRequest qr : drained) {
                long waitMs = (nowMicros - qr.enqueuedAtMicros()) / 1000;
                dpBatchReporter.reportSloQueueWait(model, flushReason, waitMs);
            }
        }
        DispatchPlan result;
        try {
            result = planner.plan(drained, new DispatchContext(model, dpSize, cfg, drained));
        } catch (Throwable t) {
            Logger.error("DispatchPlanner threw on {} requests; failing all", drained.size(), t);
            failAll(drained, t);
            return;
        }

        for (FailedRequest fr : result.failures()) {
            fr.request().future().complete(failureResponse(fr.reason(), fr.message()));
        }
        for (PrefillBatch batch : result.batches()) {
            try {
                dispatchCallback.accept(batch);
            } catch (Throwable t) {
                Logger.error("dispatch callback threw on batch of {}; failing all in batch",
                        batch.size(), t);
                for (PendingRequest pr : batch.requests()) {
                    pr.future().completeExceptionally(t);
                }
            }
        }
    }

    private static void failAll(List<QueuedRequest> chunk, Throwable cause) {
        for (QueuedRequest qr : chunk) {
            qr.future().completeExceptionally(cause);
        }
    }

    private static Response failureResponse(StrategyErrorType type, String detail) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(type.getErrorCode());
        r.setErrorMessage(type.getErrorMsg() + (detail != null ? ": " + detail : ""));
        return r;
    }

    private int currentDpSize(FlexlbConfig cfg) {
        int configured = cfg.getDpBatchSizeMax();
        if (configured > 0) {
            return configured;
        }
        Map<String, WorkerStatus> workers = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (workers == null || workers.isEmpty()) {
            return 1;
        }
        for (WorkerStatus w : workers.values()) {
            if (w != null && w.getDpSize() > 1) {
                return (int) w.getDpSize();
            }
        }
        return 1;
    }

    private int bucketSize(int bucketIdx) {
        ArrayDeque<QueuedRequest> deque = buckets.get(bucketIdx);
        return deque == null ? 0 : deque.size();
    }

    private long guessBlockSize() {
        Map<String, WorkerStatus> workers = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (workers != null) {
            for (WorkerStatus w : workers.values()) {
                if (w != null && w.getCacheStatus() != null && w.getCacheStatus().getBlockSize() > 0) {
                    return w.getCacheStatus().getBlockSize();
                }
            }
        }
        return 1;
    }

    private String guessPrefillIpPort() {
        Map<String, WorkerStatus> workers = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (workers != null) {
            for (WorkerStatus w : workers.values()) {
                if (w != null && w.isAlive() && w.getDpSize() > 1) {
                    return w.getIpPort();
                }
            }
        }
        return null;
    }
}
