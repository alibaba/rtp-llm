package org.flexlb.balance.delivery;

import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;
import java.util.OptionalLong;

/**
 * 投递方式中与模式相关的部分。分组负责选出一个有序的候选组；
 * 调度器负责协调 prepare、入队提交和 deliver 三个阶段；
 * 具体的投递模式只管理自己的精确容量和传输资源。
 */
public interface DeliveryStrategy {

    /**
     * 预留最大的可行前缀，此时不改变队列的归属。
     */
    DeliveryTransaction prepare(List<ScheduledRequest> candidates,
                                PrefillTimePredictor.Evaluator evaluator,
                                OptionalLong plannedPredictionMs);
}
