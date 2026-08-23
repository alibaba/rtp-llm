package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.strategy.PrefillBatchFeatures;

import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

final class BatchInflight implements InflightEvictor.TtlTracked {

    private final long predictTimeMs;
    private final List<BatchItem> requests;
    private final Runnable batchCapacityRelease;
    private final long originalPredictTimeMs;
    private final Set<Long> originalRequestIds;
    private final PrefillBatchFeatures originalFeatures;
    private final long createdAtMs;
    private final AtomicLong progressBaseMs;
    private final AtomicLong lastObservedAtMs;
    private final AtomicLong maxExecutionTimeMs;
    private final AtomicBoolean successfulCompletionObserved;
    private final AtomicBoolean learningEligible;
    private volatile boolean running;

    BatchInflight(long predictTimeMs,
                  List<BatchItem> requests,
                  Runnable batchCapacityRelease) {
        List<BatchItem> batchRequests = requireBatchRequests(requests);
        long nowMs = System.currentTimeMillis();
        this.predictTimeMs = predictTimeMs;
        this.requests = batchRequests;
        this.batchCapacityRelease = Objects.requireNonNull(
                batchCapacityRelease, "batchCapacityRelease");
        this.originalPredictTimeMs = predictTimeMs;
        this.originalRequestIds = batchRequests.stream()
                .map(BatchItem::requestId)
                .collect(Collectors.toUnmodifiableSet());
        this.originalFeatures = PrefillBatchFeatures.from(batchRequests);
        this.createdAtMs = nowMs;
        this.progressBaseMs = new AtomicLong(nowMs);
        this.lastObservedAtMs = new AtomicLong(nowMs);
        this.maxExecutionTimeMs = new AtomicLong();
        this.successfulCompletionObserved = new AtomicBoolean();
        this.learningEligible = new AtomicBoolean(true);
        this.running = false;
    }

    private static List<BatchItem> requireBatchRequests(List<BatchItem> requests) {
        List<BatchItem> batchRequests = List.copyOf(
                Objects.requireNonNull(requests, "requests"));
        if (batchRequests.isEmpty()) {
            throw new IllegalArgumentException(
                    "an inflight batch must contain at least one request");
        }
        return batchRequests;
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests,
                          long originalPredictTimeMs,
                          Set<Long> originalRequestIds,
                          PrefillBatchFeatures originalFeatures,
                          Runnable batchCapacityRelease,
                          long createdAtMs,
                          long progressBaseMs,
                          long lastObservedAtMs,
                          long maxExecutionTimeMs,
                          boolean successfulCompletionObserved,
                          boolean learningEligible,
                          boolean running) {
        this.predictTimeMs = predictTimeMs;
        this.requests = requireBatchRequests(requests);
        this.originalPredictTimeMs = originalPredictTimeMs;
        this.originalRequestIds = originalRequestIds;
        this.originalFeatures = originalFeatures;
        this.batchCapacityRelease = Objects.requireNonNull(
                batchCapacityRelease, "batchCapacityRelease");
        this.createdAtMs = createdAtMs;
        this.progressBaseMs = new AtomicLong(progressBaseMs);
        this.lastObservedAtMs = new AtomicLong(lastObservedAtMs);
        this.maxExecutionTimeMs = new AtomicLong(maxExecutionTimeMs);
        this.successfulCompletionObserved = new AtomicBoolean(successfulCompletionObserved);
        this.learningEligible = new AtomicBoolean(learningEligible);
        this.running = running;
    }

    long predictTimeMs() {
        return predictTimeMs;
    }

    List<BatchItem> requests() {
        return requests;
    }

    void releaseCapacitySlot() {
        batchCapacityRelease.run();
    }

    @Override
    public long createdAtMs() {
        return createdAtMs;
    }

    long lastObservedAtMs() {
        return lastObservedAtMs.get();
    }

    long progressBaseMs() {
        return progressBaseMs.get();
    }

    void markQueued(long statusMs) {
        touch(statusMs);
        progressBaseMs.updateAndGet(base -> Math.max(base, statusMs));
        running = false;
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

    Set<Long> originalRequestIds() {
        return originalRequestIds;
    }

    PrefillBatchFeatures originalFeatures() {
        return originalFeatures;
    }

    void observeSuccessfulCompletion() {
        successfulCompletionObserved.set(true);
    }

    boolean successfulCompletionObserved() {
        return successfulCompletionObserved.get();
    }

    void observeFailure() {
        learningEligible.set(false);
    }

    boolean learningEligible() {
        return learningEligible.get();
    }

    BatchInflight repack(long newPredictTimeMs, List<BatchItem> newRequests) {
        return new BatchInflight(newPredictTimeMs, newRequests,
                originalPredictTimeMs, originalRequestIds, originalFeatures,
                batchCapacityRelease,
                createdAtMs, progressBaseMs(), lastObservedAtMs.get(),
                maxExecutionTimeMs.get(), successfulCompletionObserved.get(),
                learningEligible.get(), running);
    }
}
