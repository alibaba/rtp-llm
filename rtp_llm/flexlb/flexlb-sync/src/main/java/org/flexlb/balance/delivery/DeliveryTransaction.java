package org.flexlb.balance.delivery;

import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * 一次投递事务，贯穿 prepare、入队提交和 handoff 三个阶段。
 */
public interface DeliveryTransaction extends AutoCloseable {

    List<ScheduledRequest> items();

    /**
     * 本事务没有覆盖到的第一个候选请求及其原因，没有则为 null。
     */
    BlockedRequest blocked();

    /**
     * 在同一把 endpoint 锁内提交归属，并记录排在前面的工作量。
     */
    WorkSnapshot commitUnderLock();

    /**
     * 把已提交的归属交给配置的投递模式。
     */
    void handoff(String decisionReason, int remainingQueueDepth, WorkSnapshot precedingWork);

    /**
     * 处理 handoff 没有转交出去的、已提交的归属。
     */
    void abort(Throwable cause);

    @Override
    void close();
}
