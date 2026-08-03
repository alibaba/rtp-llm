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
    private final boolean partiallyCompleted;

    BatchInflight(long predictTimeMs, List<BatchItem> requests) {
        this(predictTimeMs, List.copyOf(requests), System.currentTimeMillis());
    }

    private BatchInflight(long predictTimeMs, List<BatchItem> requests, long nowMs) {
        this(predictTimeMs, requests, nowMs, nowMs, false, false);
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests,
                          long createdAtMs,
                          long progressBaseMs,
                          boolean running,
                          boolean partiallyCompleted) {
        this.predictTimeMs = predictTimeMs;
        this.requests = List.copyOf(requests);
        this.createdAtMs = createdAtMs;
        this.progressBaseMs = new AtomicLong(progressBaseMs);
        this.running = running;
        this.partiallyCompleted = partiallyCompleted;
    }

    long predictTimeMs() {
        return predictTimeMs;
    }

    List<BatchItem> requests() {
        return requests;
    }

    boolean partiallyCompleted() {
        return partiallyCompleted;
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
                createdAtMs, progressBaseMs(), running, true);
    }
}
