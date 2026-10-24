package org.flexlb.balance.delivery;

import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;
import java.util.OptionalLong;

/**
 * 投递方式中与模式相关的部分。分组负责选出一个有序的候选组；
 * 调度器负责协调 prepare、入队提交和 handoff 三个阶段；
 * 具体的投递模式只管理自己的精确容量和传输资源。
 */
public interface DeliveryStrategy {

    /**
     * 预留最大的可行前缀，此时不改变队列的归属。
     */
    Transaction prepare(List<ScheduledRequest> candidates,
                        PrefillTimePredictor.Evaluator evaluator,
                        OptionalLong plannedPredictionMs);

    /**
     * 一次投递事务，贯穿 prepare、入队提交和 handoff 三个阶段。
     */
    interface Transaction extends AutoCloseable {

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
}
