package org.flexlb.balance.estimate.estimator;

import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.ScheduledRequest;

/**
 * Prefill 耗时的预估，纯计算、不持有任何资源。
 * 不同的投递模式（BATCH、NON_BATCH）对排队耗时的折算方式不同，各自提供自己的实现。
 */
public interface PrefillTtftEstimator {

    /**
     * 为一次 select 调用中逐步变长的前缀，创建新的耗时预测回调，供 GroupPlanner 使用。
     */
    GroupPlanner.PrefixPrediction<ScheduledRequest> newGroupPredictor(PrefillTimePredictor.Evaluator evaluator);

    /**
     * 预估 TTFT 的规则：本投递模式下，请求前面排着的工作量怎么折算成等待时间。
     */
    PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule();
}
