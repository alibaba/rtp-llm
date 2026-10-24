package org.flexlb.balance.delivery;

import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * 一次投递事务，贯穿 prepare、入队提交和 deliver 三个阶段。
 */
public interface DeliveryTransaction extends AutoCloseable {

    List<ScheduledRequest> requests();

    /**
     * 本事务没有覆盖到的第一个候选请求及其原因，没有则为 null。
     */
    BlockedRequest blocked();

    /**
     * 提交归属并返回排在前面的工作量。必须在 endpoint 锁内调用。
     */
    WorkView commit();

    /**
     * 把已提交的归属交给配置的投递模式去发送。
     */
    void deliver(String decisionReason, int remainingQueueDepth, WorkView precedingWork);

    /**
     * 回滚已提交但没有被 deliver 接走的归属。deliver 之后无论成败都会再调一次，
     * 已接走时不做任何事。
     */
    void rollbackCommitted(Throwable cause);

    /**
     * 事务还没提交时回滚预留；已提交的归属由 rollbackCommitted 处理。
     */
    @Override
    void close();
}
