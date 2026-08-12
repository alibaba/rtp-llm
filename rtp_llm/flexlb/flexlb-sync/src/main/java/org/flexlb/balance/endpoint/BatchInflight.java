package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

final class BatchInflight implements InflightEvictor.TtlTracked {

    private final long predictTimeMs;
    private final List<BatchItem> requests;
    private final long originalPredictTimeMs;
    private final List<BatchItem> originalRequests;
    private final AtomicLong progressBaseMs;
    /** Last worker-status observation for any member still owned by this batch. */
    private final AtomicLong lastObservedAtMs;
    /** Largest engine execution time observed while members finish incrementally. */
    private final AtomicLong maxExecutionTimeMs;
    private volatile boolean running;

    BatchInflight(long predictTimeMs, List<BatchItem> requests) {
        this(predictTimeMs, requests, System.currentTimeMillis());
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests, long nowMs) {
        this(predictTimeMs, requests, predictTimeMs, requests,
                nowMs, nowMs, 0, false);
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests,
                          long originalPredictTimeMs,
                          List<BatchItem> originalRequests,
                          long progressBaseMs,
                          long lastObservedAtMs,
                          long maxExecutionTimeMs,
                          boolean running) {
        this.predictTimeMs = predictTimeMs;
        this.requests = requests;
        this.originalPredictTimeMs = originalPredictTimeMs;
        this.originalRequests = originalRequests;
        this.progressBaseMs = new AtomicLong(progressBaseMs);
        this.lastObservedAtMs = new AtomicLong(lastObservedAtMs);
        this.maxExecutionTimeMs = new AtomicLong(maxExecutionTimeMs);
        this.running = running;
    }

    long predictTimeMs() {
        return predictTimeMs;
    }

    List<BatchItem> requests() {
        return requests;
    }

    @Override
    public long createdAtMs() {
        // TTL is inactivity based.  A long-running Prefill batch must not be
        // evicted merely because its wall-clock execution exceeds the TTL.
        return lastObservedAtMs.get();
    }

    long progressBaseMs() {
        return progressBaseMs.get();
    }

    void markQueued(long statusMs) {
        touch(statusMs);
        if (!running) {
            progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
        }
    }

    void markRunning(long statusMs) {
        touch(statusMs);
        if (!running) {
            progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
            running = true;
        }
    }

    void touch(long statusMs) {
        lastObservedAtMs.updateAndGet(last -> Math.max(last, statusMs));
    }

    void observeExecutionTime(long executionTimeMs) {
        if (executionTimeMs > 0) {
            maxExecutionTimeMs.updateAndGet(current -> Math.max(current, executionTimeMs));
        }
    }

    long maxExecutionTimeMs() {
        return maxExecutionTimeMs.get();
    }

    long originalPredictTimeMs() {
        return originalPredictTimeMs;
    }

    List<BatchItem> originalRequests() {
        return originalRequests;
    }

    BatchInflight repack(long newPredictTimeMs, List<BatchItem> newRequests) {
        return new BatchInflight(newPredictTimeMs, newRequests,
                originalPredictTimeMs, originalRequests,
                progressBaseMs(), lastObservedAtMs.get(),
                maxExecutionTimeMs.get(), running);
    }
}
