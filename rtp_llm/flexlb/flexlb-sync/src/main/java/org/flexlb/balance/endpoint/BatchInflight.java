package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public final class BatchInflight implements InflightEvictor.TtlTracked {

    private final long batchId;
    private final long predictTimeMs;
    private final List<BatchItem> requests;
    private final long createdAtMs;
    private final AtomicLong progressBaseMs;
    private volatile boolean running;
    private volatile long lastSeenMs;

    public BatchInflight(long batchId, long predictTimeMs, List<BatchItem> requests) {
        this(batchId, predictTimeMs, requests, System.currentTimeMillis());
    }

    private BatchInflight(long batchId, long predictTimeMs,
                          List<BatchItem> requests, long nowMs) {
        this(batchId, predictTimeMs, requests, nowMs, nowMs, false, nowMs);
    }

    private BatchInflight(long batchId,
                          long predictTimeMs,
                          List<BatchItem> requests,
                          long createdAtMs,
                          long progressBaseMs,
                          boolean running,
                          long lastSeenMs) {
        this.batchId = batchId;
        this.predictTimeMs = predictTimeMs;
        this.requests = requests;
        this.createdAtMs = createdAtMs;
        this.progressBaseMs = new AtomicLong(progressBaseMs);
        this.running = running;
        this.lastSeenMs = lastSeenMs;
    }

    public long predictTimeMs() {
        return predictTimeMs;
    }

    public List<BatchItem> requests() {
        return requests;
    }

    @Override
    public long createdAtMs() {
        return createdAtMs;
    }

    public long progressBaseMs() {
        return progressBaseMs.get();
    }

    public void markQueued(long statusMs) {
        if (!running) {
            progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
        }
        lastSeenMs = Math.max(lastSeenMs, statusMs);
    }

    public void markRunning(long statusMs) {
        if (!running) {
            progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
            running = true;
        }
        lastSeenMs = Math.max(lastSeenMs, statusMs);
    }

    public BatchInflight repack(long newPredictTimeMs, List<BatchItem> newRequests) {
        return new BatchInflight(batchId, newPredictTimeMs, newRequests,
                createdAtMs, progressBaseMs(), running, lastSeenMs);
    }
}
