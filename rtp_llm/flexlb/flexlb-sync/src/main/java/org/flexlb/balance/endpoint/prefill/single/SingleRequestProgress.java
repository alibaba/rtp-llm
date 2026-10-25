package org.flexlb.balance.endpoint.prefill.single;

import org.flexlb.balance.estimate.WorkView.Phase;

/**
 * NON_BATCH 请求提交给 Prefill 之后的执行进度：处在哪个阶段、还剩多少预测耗时、最近一次被引擎上报的时间。
 * 每个请求一份，由 PrefillState 的锁保护。
 */
public final class SingleRequestProgress {
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

    public SingleRequestProgress(long predictedWorkMs, long nowMs) {
        this.remainingWorkMs = predictedWorkMs;
        this.remainingWorkSinceMs = nowMs;
        this.lastObservedAtMs = nowMs;
    }

    public Phase phase() {
        return phase;
    }

    public long lastObservedAtMs() {
        return lastObservedAtMs;
    }

    public long remainingWorkMsAt(long nowMs) {
        if (phase != Phase.ENGINE_RUNNING) {
            return remainingWorkMs;
        }
        return Math.max(0L, remainingWorkMs - Math.max(0L, nowMs - remainingWorkSinceMs));
    }

    /**
     * 按 WorkerStatus 上报的阶段更新进度。离开 ENGINE_RUNNING 时先扣掉已经跑过的耗时。
     */
    public void updateProgressFromWorkerStatus(Phase next, long nowMs) {
        if (next != Phase.ENGINE_QUEUED && next != Phase.ENGINE_RUNNING) {
            throw new IllegalArgumentException("invalid Engine phase " + next);
        }
        remainingWorkMs = remainingWorkMsAt(nowMs);
        remainingWorkSinceMs = Math.max(remainingWorkSinceMs, nowMs);
        lastObservedAtMs = Math.max(lastObservedAtMs, nowMs);
        phase = next;
    }
}
