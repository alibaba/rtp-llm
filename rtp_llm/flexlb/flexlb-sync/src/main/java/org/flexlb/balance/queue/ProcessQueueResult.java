package org.flexlb.balance.queue;

import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.scheduler.ScheduledRequest;

/**
 * processQueue() 的返回值：调度线程跑完一轮后的结果，决定下一轮之前做什么：
 * <ul>
 *   <li>{@link #NO_ACTION}：这一轮什么也没交付，直接进下一轮。</li>
 *   <li>{@link #CAPACITY_CHANGED}：有请求离开了队列，通知选路队列空出了位置。</li>
 *   <li>{@link #capacityBlocked}：队首因投递容量不足交付不了，等容量空出来、队首变化或请求超时。</li>
 *   <li>{@link #awaitingSchedulingChange}：队首在等凑组或 Prefill KV 容量，等队列、调度输入变化或到点。</li>
 * </ul>
 */
public record ProcessQueueResult(
        boolean placementCapacityChanged,
        ScheduledRequest request,
        DeliveryBlockReason unavailable,
        long queueVersion,
        long schedulingInputVersion,
        long wakeAtMs,
        String waitReason) {

    public static final ProcessQueueResult NO_ACTION = simple(false);
    public static final ProcessQueueResult CAPACITY_CHANGED = simple(true);

    public ProcessQueueResult {
        boolean waiting = request != null;
        if ((placementCapacityChanged && waiting)
                || (!waiting && unavailable != null)) {
            throw new IllegalArgumentException(
                    "worker cycle requires either a capacity change or an exact wait predicate");
        }
    }

    public static ProcessQueueResult capacityBlocked(ScheduledRequest item, DeliveryBlockReason unavailable) {
        return new ProcessQueueResult(false, item, unavailable, 0L, 0L, 0L,
                unavailable.estimateSemantics() == null
                        ? "Prefill delivery capacity exhausted"
                        : unavailable.estimateSemantics().blockedDetail());
    }

    public static ProcessQueueResult awaitingSchedulingChange(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion,
            long wakeAtMs,
            String waitReason) {
        return new ProcessQueueResult(false,
                head, null,
                queueVersion, schedulingInputVersion, wakeAtMs, waitReason);
    }

    /**
     * 队首在等 Prefill 的 KV 容量，等到队列或调度输入变化、或请求超时再进下一轮。
     */
    public static ProcessQueueResult awaitingPrefillKvCapacity(
            ScheduledRequest head,
            long queueVersion,
            long schedulingInputVersion) {
        return ProcessQueueResult.awaitingSchedulingChange(
                head, queueVersion, schedulingInputVersion,
                head.expiresAtMs(), "Prefill KV capacity exhausted");
    }

    static ProcessQueueResult simple(boolean capacityChanged) {
        return new ProcessQueueResult(
                capacityChanged, null, null, 0L, 0L, 0L, null);
    }

    public boolean capacityBlocked() {
        return unavailable != null;
    }

    public boolean awaitingSchedulingChange() {
        return request != null && unavailable == null;
    }
}
