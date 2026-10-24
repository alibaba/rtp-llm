package org.flexlb.balance.queue;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * 在队列锁内复制下来的排队请求，以及复制时的队列版本和调度输入版本。
 * 选请求在锁外进行，等待时用这两个版本号判断队列有没有变。
 *
 * @param items 按调度顺序排好的排队请求，可能只含队首
 */
record ActiveQueueSnapshot(long queueVersion,
                           long schedulingInputVersion,
                           List<ScheduledRequest> items) {

    /**
     * 队首请求；队列为空时为 null。
     */
    ScheduledRequest head() {
        return items.isEmpty() ? null : items.get(0);
    }
}
