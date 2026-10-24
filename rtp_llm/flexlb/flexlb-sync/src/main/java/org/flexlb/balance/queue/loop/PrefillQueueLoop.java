package org.flexlb.balance.queue.loop;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.queue.CapacityBlockedHead;
import org.flexlb.balance.queue.ProcessQueueResult;
import org.flexlb.balance.queue.QueueState;
import org.flexlb.balance.queue.dispatch.QueuePicker;

/**
 * 一台 Prefill 排队队列的调度线程主循环。
 * <p>
 * 每一轮先等队列里有请求，再调用 {@link QueuePicker#processQueue} 选出请求并交付，然后按返回的
 * {@link ProcessQueueResult} 决定下一轮之前等什么：队首被投递容量挡住时等容量空出来，
 * 在等凑组或 Prefill KV 容量时等队列或调度输入变化、或者到点。
 * 循环中抛出意外异常时停止整个队列。
 */
public final class PrefillQueueLoop {

    private final QueueState queueState;
    private final PrefillEndpoint prefillEndpoint;
    private final CapacityBlockedHead capacityBlockedHead;
    private final QueueWaitRecorder queueWaitRecorder;
    private final PrefillQueueLifecycle lifecycle;
    private final QueuePicker queuePicker;

    public PrefillQueueLoop(QueueState queueState,
                     PrefillEndpoint prefillEndpoint,
                     CapacityBlockedHead capacityBlockedHead,
                     QueueWaitRecorder queueWaitRecorder,
                     PrefillQueueLifecycle lifecycle,
                     QueuePicker queuePicker) {
        this.queueState = queueState;
        this.prefillEndpoint = prefillEndpoint;
        this.capacityBlockedHead = capacityBlockedHead;
        this.queueWaitRecorder = queueWaitRecorder;
        this.lifecycle = lifecycle;
        this.queuePicker = queuePicker;
    }

    public void run() {
        try {
            while (!queueState.isStopped() && !Thread.currentThread().isInterrupted()) {
                try {
                    runOneCycle();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Throwable t) {
                    // 预期内的交付失败在每一轮里都已经处理掉了，能抛到这里说明状态出了问题；
                    // 重试只会对着同样的排队状态空转，所以直接停掉整个队列。
                    lifecycle.stopAfterUnexpectedLoopFailure(t);
                    return;
                }
            }
        } finally {
            lifecycle.markLoopFinished();
        }
    }

    private void runOneCycle() throws InterruptedException {
        waitForNonEmpty();
        if (queueState.isStopped()) {
            return;
        }

        ProcessQueueResult result = queuePicker.processQueue();
        if (result.waitReason() != null) {
            queueWaitRecorder.record(result.request(), result.waitReason());
        }
        if (result.placementCapacityChanged()) {
            // 只有请求真正离开队列才会空出排队位置。等待中或投递容量不够时不通知选路，
            // 否则选路会反复放请求进来又等。
            prefillEndpoint.signalPlacementCapacityChanged();
        }
        if (result.capacityBlocked()) {
            awaitBlockedHeadCapacity(result);
        } else if (result.awaitingSchedulingChange()) {
            awaitSchedulingChange(result);
        }
    }

    private void waitForNonEmpty() throws InterruptedException {
        queueState.lockInterruptibly();
        try {
            while (!queueState.isStopped() && queueState.activeIndex().isEmpty()) {
                capacityBlockedHead.setUnderLock(null);
                queueState.awaitChange();
            }
        } finally {
            queueState.unlock();
        }
    }

    /**
     * 队首被某项投递容量挡住时，等这项容量空出来、队首变化或请求超时。
     * <p>
     * 新请求入队也会唤醒等待：新请求优先级更高、成了新队首时立刻结束等待；排在后面的不影响。
     * 先在持有队列锁时登记容量释放的通知，再检查容量，这样不会漏掉开始等待前发生的释放，也不用轮询。
     */
    private void awaitBlockedHeadCapacity(
            ProcessQueueResult blocked)
            throws InterruptedException {
        queueState.lockInterruptibly();
        try {
            if (queueState.isStopped()) {
                return;
            }
            capacityBlockedHead.setUnderLock(blocked);
            while (!queueState.isStopped()
                    && queueState.activeIndex().peek() == blocked.request()
                    && !blocked.request().requestExpired(System.currentTimeMillis())
                    && !blocked.unavailable().availability().isAvailable()) {
                long expiresAtMs = blocked.request().expiresAtMs();
                long nowMs = System.currentTimeMillis();
                if (expiresAtMs <= nowMs) {
                    return;
                }
                if (expiresAtMs == Long.MAX_VALUE) {
                    queueState.awaitChange();
                } else {
                    queueState.awaitChangeFor(expiresAtMs - nowMs);
                }
            }
        } finally {
            try {
                capacityBlockedHead.setUnderLock(null);
            } finally {
                queueState.unlock();
            }
        }
    }

    /**
     * 等到队列版本或调度输入版本和拿快照时不同，或者到了唤醒时间。所有检查和唤醒都在队列锁内进行。
     */
    private void awaitSchedulingChange(
            ProcessQueueResult waiting)
            throws InterruptedException {
        queueState.lockInterruptibly();
        try {
            while (!queueState.isStopped()
                    && queueState.activeIndex().peek() == waiting.request()
                    && queueState.queueVersion() == waiting.queueVersion()
                    && queueState.schedulingInputVersion()
                    == waiting.schedulingInputVersion()) {
                long nowMs = System.currentTimeMillis();
                if (waiting.wakeAtMs() <= nowMs) {
                    return;
                }
                if (waiting.wakeAtMs() == Long.MAX_VALUE) {
                    queueState.awaitChange();
                } else {
                    queueState.awaitChangeFor(waiting.wakeAtMs() - nowMs);
                }
            }
        } finally {
            queueState.unlock();
        }
    }
}
