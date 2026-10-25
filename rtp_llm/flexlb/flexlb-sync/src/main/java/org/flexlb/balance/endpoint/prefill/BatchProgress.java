package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.estimate.WorkView.Phase;
import org.flexlb.balance.prediction.PrefillBatchFeatures;

/**
 * BATCH 提交给 Prefill 之后整批共享的执行进度。同一批的成员引用同一份，由 PrefillState 的锁保护。
 */
final class BatchProgress {
    private final BatchReservation lease;
    private final long predictedWorkMs;
    private final PrefillBatchFeatures featuresAtCommit;
    private Phase phase = Phase.COMMITTED;
    /**
     * 在 remainingWorkSinceMs 时刻还剩的预测耗时。
     */
    private long remainingWorkMs;
    /**
     * remainingWorkMs 最近一次算出的时间。处在 ENGINE_RUNNING 时，从这个时间起扣减剩余耗时。
     */
    private long remainingWorkSinceMs;
    /**
     * 提交时间或最近一次被引擎上报的时间，用来判断本地记录是否过期。
     */
    private long lastObservedAtMs;
    /**
     * 引擎是否已经开始跑这一批。开始之后即使再上报排队，也不再为剩下的成员重新预测耗时。
     */
    private boolean engineStarted;
    /**
     * 已结束成员上报的执行耗时中的最大值。
     */
    private long maxExecutionTimeMs;
    /**
     * 已结束成员中是否至少有一个由引擎上报成功。
     */
    private boolean anyMemberSucceeded;
    /**
     * 已结束成员是否都由引擎上报成功。
     */
    private boolean usableForLearning = true;

    BatchProgress(BatchReservation lease, long predictedWorkMs, PrefillBatchFeatures featuresAtCommit, long nowMs) {
        if (lease.batchId() < 0L) {
            throw new IllegalArgumentException("batchId must be non-negative");
        }
        if (predictedWorkMs < 0L) {
            throw new IllegalArgumentException("predicted batch work must be non-negative");
        }
        this.lease = lease;
        this.predictedWorkMs = predictedWorkMs;
        this.featuresAtCommit = featuresAtCommit;
        this.remainingWorkMs = predictedWorkMs;
        this.remainingWorkSinceMs = nowMs;
        this.lastObservedAtMs = nowMs;
    }

    BatchReservation lease() {
        return lease;
    }

    long batchId() {
        return lease.batchId();
    }

    long predictedWorkMs() {
        return predictedWorkMs;
    }

    PrefillBatchFeatures featuresAtCommit() {
        return featuresAtCommit;
    }

    Phase phase() {
        return phase;
    }

    long lastObservedAtMs() {
        return lastObservedAtMs;
    }

    boolean engineStarted() {
        return engineStarted;
    }

    long maxExecutionTimeMs() {
        return maxExecutionTimeMs;
    }

    boolean anyMemberSucceeded() {
        return anyMemberSucceeded;
    }

    boolean usableForLearning() {
        return usableForLearning;
    }

    long remainingWorkMsAt(long nowMs) {
        if (phase != Phase.ENGINE_RUNNING) {
            return remainingWorkMs;
        }
        return Math.max(0L, remainingWorkMs - Math.max(0L, nowMs - remainingWorkSinceMs));
    }

    /**
     * 按 WorkerStatus 上报的阶段更新进度。离开 ENGINE_RUNNING 时先扣掉已经跑过的耗时。
     */
    void updateProgressFromWorkerStatus(Phase next, long nowMs) {
        remainingWorkMs = remainingWorkMsAt(nowMs);
        remainingWorkSinceMs = Math.max(remainingWorkSinceMs, nowMs);
        lastObservedAtMs = Math.max(lastObservedAtMs, nowMs);
        engineStarted |= next == Phase.ENGINE_RUNNING;
        phase = next;
    }

    /**
     * 按一个成员的结束上报更新进度。workerObserved 为 false 表示本地清理，不是引擎上报。
     */
    void updateProgressFromFinishedTask(boolean workerObserved, long errorCode, long executionTimeMs, long nowMs) {
        lastObservedAtMs = Math.max(lastObservedAtMs, nowMs);
        engineStarted |= workerObserved && (errorCode == 0L || executionTimeMs > 0L);
        if (executionTimeMs >= 0L) {
            maxExecutionTimeMs = Math.max(maxExecutionTimeMs, executionTimeMs);
        }
        anyMemberSucceeded |= workerObserved && errorCode == 0L;
        usableForLearning &= workerObserved && errorCode == 0L;
    }

    /**
     * 用剩下成员重新预测出的耗时替换剩余耗时，从 nowMs 起重新计时。
     */
    void resetRemainingWork(long remainingWorkMs, long nowMs) {
        this.remainingWorkMs = remainingWorkMs;
        this.remainingWorkSinceMs = nowMs;
        lastObservedAtMs = Math.max(lastObservedAtMs, nowMs);
    }

    FinishedBatch finishedBatch() {
        return new FinishedBatch(batchId(), featuresAtCommit, predictedWorkMs, maxExecutionTimeMs, anyMemberSucceeded,
                usableForLearning);
    }

    /**
     * 节点退役是外部终止，这时的汇总不能拿去校准预测。
     */
    FinishedBatch finishedBatchOnRetirement() {
        return new FinishedBatch(batchId(), featuresAtCommit, predictedWorkMs, maxExecutionTimeMs, anyMemberSucceeded,
                false);
    }
}
