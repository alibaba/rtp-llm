package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * 持锁时拿到的一份 Prefill 状态：等待队列和已提交的工作，LoadViewCache 用它算负载。
 */
public record PrefillLoadView(long takenAtMs, WaitingQueueView waiting, WorkCapture work) {

    public List<ScheduledRequest> waitingItems() {
        return waiting.items();
    }
}
