package org.flexlb.balance.queue;

import org.flexlb.balance.endpoint.PrefillEndpoint;

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
                    // Every expected delivery failure is terminalized inside the
                    // typed cycle. An escaping Throwable is therefore an invariant
                    // failure; retrying the same ACTIVE state can only spin.
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
            // Only a committed removal creates a new queue seat. Advisory
            // waits and delivery-capacity misses must not feed placement back
            // into itself.
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
     * Wait while the exact active head is blocked by the exact rejecting
     * resource. Offers wake the condition: a new higher-priority head exits the
     * wait immediately, while a tail offer leaves the predicate true. Capacity
     * is checked after listener installation while the queue lock is held, closing
     * the release-before-await race without polling.
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
     * Wait for the exact queue/input generation captured by the policy or
     * for its deadline. Every predicate and signal is serialized by the queue lock.
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
