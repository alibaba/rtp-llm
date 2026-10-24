package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.estimate.QueueView.AdmissionBlock;
import org.flexlb.balance.queue.QueueState;

/**
 * 记录队首请求因投递容量不足而交付不了，以及容量空出来时唤醒调度线程。
 * <p>
 * 队首被挡住时，调度线程在拒绝它的那份容量上挂一个监听，然后等待。容量释放时监听被调用，
 * 推进调度输入版本并唤醒调度线程重试。选路预估 TTFT 时也会读这份记录，知道队首正被挡住。
 * 所有读写都在队列锁内进行。
 */
final class CapacityBlockedHead {

    private final QueueState queueState;
    private final PrefillEndpoint prefillEndpoint;
    private final Runnable capacityAvailableSignal = this::signalCapacityAvailable;
    /**
     * 正在等容量的队首请求和挡住它的容量；没有被挡住时为 null。
     */
    private ProcessQueueResult blockedHead;

    CapacityBlockedHead(QueueState queueState, PrefillEndpoint prefillEndpoint) {
        this.queueState = queueState;
        this.prefillEndpoint = prefillEndpoint;
    }

    /**
     * 容量空出来时要调用的回调。PrefillState 只接受这同一个实例作为容量监听。
     */
    Runnable capacityAvailableSignal() {
        return capacityAvailableSignal;
    }

    /**
     * 队首仍被同一份容量挡住时，返回它的请求 ID、入队序号和预估 TTFT 用的阻塞说明；否则返回 null。
     * <p>
     * 只读取已挂监听的那份容量当前是否可用，不预占容量。调用方持有队列锁。
     */
    AdmissionBlock admissionBlockUnderLock() {
        if (!queueState.isLockedByCurrentThread()) {
            throw new IllegalStateException(
                    "capacity block snapshot requires the queue lock");
        }
        ProcessQueueResult blocked = blockedHead;
        if (blocked == null
                || queueState.activeIndex().peek() != blocked.request()
                || blocked.request().requestExpired(System.currentTimeMillis())
                || blocked.unavailable().availability().isAvailable()) {
            return null;
        }
        return new AdmissionBlock(
                blocked.request().requestId(),
                blocked.request().enqueueSeq(),
                blocked.unavailable().estimateSemantics());
    }

    /**
     * Prefill 或 Decode 容量变化后调用，调用方不持有 endpoint 的锁。队首正被挡住时推进调度输入版本并唤醒调度线程。
     */
    void signalDeliveryCapacityAvailable() {
        queueState.lock();
        try {
            if (blockedHead != null) {
                queueState.incrementSchedulingInputVersionUnderLock();
                queueState.signalChange();
            }
        } finally {
            queueState.unlock();
        }
    }

    /**
     * Prefill 资源预留或 batch 容量真的释放后调用：唤醒本队列被挡住的队首，并通知全局选路有容量空出来。
     */
    void signalCapacityAvailable() {
        signalDeliveryCapacityAvailable();
        prefillEndpoint.signalPlacementCapacityChanged();
    }

    /**
     * 记下新的被挡队首，传 null 表示清除。挡住它的容量变了时，把监听从旧容量挪到新容量上；
     * 同时推进调度输入版本，让缓存的负载视图失效。调用方持有队列锁。
     */
    void setUnderLock(ProcessQueueResult blocked) {
        if (!queueState.isLockedByCurrentThread()) {
            throw new IllegalStateException(
                    "capacity block update requires the queue lock");
        }
        if (blockedHead == blocked) {
            return;
        }
        DeliveryBlockReason.Availability previousSource = blockedHead == null
                ? null : blockedHead.unavailable().availability();
        DeliveryBlockReason.Availability nextSource = blocked == null
                ? null : blocked.unavailable().availability();
        blockedHead = blocked;
        queueState.incrementSchedulingInputVersionUnderLock();
        if (previousSource != nextSource) {
            try {
                if (previousSource != null) {
                    previousSource.removeListener(capacityAvailableSignal);
                }
            } finally {
                if (nextSource != null) {
                    nextSource.addListener(capacityAvailableSignal);
                }
            }
        }
    }
}
