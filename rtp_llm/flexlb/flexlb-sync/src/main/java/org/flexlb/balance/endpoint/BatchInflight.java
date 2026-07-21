package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

final class BatchInflight implements InflightEvictor.TtlTracked {

    private final long predictTimeMs;
    private final List<BatchItem> requests;
    private final long createdAtMs;
    private final AtomicLong progressBaseMs;
    private volatile boolean running;

    BatchInflight(long predictTimeMs, List<BatchItem> requests) {
        this(predictTimeMs, requests, System.currentTimeMillis());
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests, long nowMs) {
        this(predictTimeMs, requests, nowMs, nowMs, false);
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests,
                          long createdAtMs,
                          long progressBaseMs,
                          boolean running) {
        this.predictTimeMs = predictTimeMs;
        this.requests = requests;
        this.createdAtMs = createdAtMs;
        this.progressBaseMs = new AtomicLong(progressBaseMs);
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
        return createdAtMs;
    }

    long progressBaseMs() {
        return progressBaseMs.get();
    }

    void markQueued(long statusMs) {
        if (!running) {
            progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
        }
    }

    void markRunning(long statusMs) {
        if (!running) {
            progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
            running = true;
        }
    }

    BatchInflight repack(long newPredictTimeMs, List<BatchItem> newRequests) {
        return new BatchInflight(newPredictTimeMs, newRequests,
                createdAtMs, progressBaseMs(), running);
    }
}
