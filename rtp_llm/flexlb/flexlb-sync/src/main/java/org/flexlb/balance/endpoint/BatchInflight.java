package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.strategy.PrefillBatchFeatures;

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

final class BatchInflight implements InflightEvictor.TtlTracked {

    private final long predictTimeMs;
    private final List<BatchItem> requests;
    private final long originalPredictTimeMs;
    private final Set<Long> originalRequestIds;
    private final PrefillBatchFeatures originalFeatures;
    private final long createdAtMs;
    private final AtomicLong progressBaseMs;
    private final AtomicLong lastObservedAtMs;
    private final AtomicLong maxExecutionTimeMs;
    private final AtomicBoolean successfulCompletionObserved;
    private final AtomicBoolean learningEligible;
    private final AtomicBoolean cancelOverlayObserved;
    /**
     * Consecutive calibrate rounds (each backed by a real, version-advanced
     * engine report) in which no member of this batch was mentioned in
     * either finished or running task info (Fix A, ACKNOWLEDGED-lost
     * detection). Reset to zero the moment any member is observed. A
     * {@link #repack} starts a fresh instance at zero — the settle that
     * triggered the repack is itself an observation.
     */
    private final AtomicInteger observationMisses = new AtomicInteger(0);
    private volatile boolean running;

    BatchInflight(long predictTimeMs, List<BatchItem> requests) {
        this(predictTimeMs, requests, System.currentTimeMillis());
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests, long nowMs) {
        this(predictTimeMs, requests, predictTimeMs,
                requests.stream().map(BatchItem::requestId).collect(Collectors.toUnmodifiableSet()),
                PrefillBatchFeatures.from(requests), nowMs,
                nowMs, nowMs, 0, false, true, false, false);
    }

    private BatchInflight(long predictTimeMs,
                          List<BatchItem> requests,
                          long originalPredictTimeMs,
                          Set<Long> originalRequestIds,
                          PrefillBatchFeatures originalFeatures,
                          long createdAtMs,
                          long progressBaseMs,
                          long lastObservedAtMs,
                          long maxExecutionTimeMs,
                          boolean successfulCompletionObserved,
                          boolean learningEligible,
                          boolean cancelOverlayObserved,
                          boolean running) {
        this.predictTimeMs = predictTimeMs;
        this.requests = requests;
        this.originalPredictTimeMs = originalPredictTimeMs;
        this.originalRequestIds = originalRequestIds;
        this.originalFeatures = originalFeatures;
        this.createdAtMs = createdAtMs;
        this.progressBaseMs = new AtomicLong(progressBaseMs);
        this.lastObservedAtMs = new AtomicLong(lastObservedAtMs);
        this.maxExecutionTimeMs = new AtomicLong(maxExecutionTimeMs);
        this.successfulCompletionObserved = new AtomicBoolean(successfulCompletionObserved);
        this.learningEligible = new AtomicBoolean(learningEligible);
        this.cancelOverlayObserved = new AtomicBoolean(cancelOverlayObserved);
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

    /**
     * Record that a WorkerStatus round reported a member of this batch in a
     * priority-cancel overlay (CANCELING/CANCELED + PENDING, never executed).
     * Kept as forensic evidence for hard-age eviction warnings.
     */
    void observeCancelOverlay() {
        cancelOverlayObserved.set(true);
    }

    boolean cancelOverlayObserved() {
        return cancelOverlayObserved.get();
    }

    /** One more observed calibrate round without any member of this batch. */
    int recordObservationMiss() {
        return observationMisses.incrementAndGet();
    }

    /** Some member of this batch was mentioned by the engine report. */
    void resetObservationMisses() {
        observationMisses.set(0);
    }

    int observationMisses() {
        return observationMisses.get();
    }

    /** Last wall-clock time a settle-deferred WARN was emitted for this batch. */
    private volatile long lastDeferWarnAtMs;

    /**
     * Anti-spam gate for the settle-deferred audit WARN: allows at most one
     * line per {@code rateLimitMs} window (calibrate runs every ~20ms and
     * finished snapshots repeat members, so an unresolved fence would
     * otherwise flood). Race-tolerant by design — a concurrent pass may
     * emit one extra line, never fewer than one per window per instance.
     */
    boolean shouldWarnSettleDeferred(long nowMs, long rateLimitMs) {
        long last = lastDeferWarnAtMs;
        if (nowMs - last < rateLimitMs) {
            return false;
        }
        lastDeferWarnAtMs = nowMs;
        return true;
    }

    boolean running() {
        return running;
    }

    BatchInflight repack(long newPredictTimeMs, List<BatchItem> newRequests) {
        return new BatchInflight(newPredictTimeMs, newRequests,
                originalPredictTimeMs, originalRequestIds, originalFeatures,
                createdAtMs, progressBaseMs(), lastObservedAtMs.get(),
                maxExecutionTimeMs.get(), successfulCompletionObserved.get(),
                learningEligible.get(), cancelOverlayObserved.get(), running);
    }
}
