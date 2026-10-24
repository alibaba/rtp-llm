package org.flexlb.balance.delivery;

import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;
import java.util.OptionalLong;

/**
 * 调度器选出、准备一起投递给同一台 Prefill 的一组请求。
 *
 * @param requests 按顺序排列的请求，至少有一个；投递策略从第一个开始尽可能多地预留。
 * @param evaluator 这台 Prefill 的耗时预测器，用来估算请求或整批的处理时间。
 * @param plannedPredictionMs 分组阶段已经算出的整组耗时（毫秒），没有算过则为空。
 *                            只有 BATCH 模式使用：最终提交的请求和整组完全一致时直接采用这个值。
 */
public record DeliveryGroup(List<ScheduledRequest> requests,
                            PrefillTimePredictor.Evaluator evaluator,
                            OptionalLong plannedPredictionMs) {

    public DeliveryGroup {
        if (requests.isEmpty()) {
            throw new IllegalArgumentException("delivery group requires at least one request");
        }
    }
}
