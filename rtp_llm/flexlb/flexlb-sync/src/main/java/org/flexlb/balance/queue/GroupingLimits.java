package org.flexlb.balance.queue;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;

import java.util.function.Supplier;

/**
 * 一次从队列里取出一组请求时的上限：最多取几个、队首最多等多久凑齐、这一组的预测耗时最多多长。
 * <p>
 * 只有 FIXED_WINDOW 决策策略会凑组；其他策略一次只取一个请求，不等待，也不限制预测耗时。
 */
public final class GroupingLimits {

    /**
     * FIXED_WINDOW 策略的配置；其他策略为 null。
     */
    private final DecisionPolicyConfig fixedWindowDecision;
    private final Supplier<FlexlbConfig> configSupplier;

    public GroupingLimits(FlexlbConfig config, Supplier<FlexlbConfig> configSupplier) {
        DecisionPolicyConfig decision = config.decisionPolicy();
        this.fixedWindowDecision =
                decision.getType() == DecisionPolicyConfig.Type.FIXED_WINDOW ? decision : null;
        this.configSupplier = configSupplier;
    }

    public int maxRequests() {
        return fixedWindowDecision == null ? 1 : fixedWindowDecision.resolveMaxRequests();
    }

    /**
     * 队首请求最多等多久让后面的请求凑进同一组。每次都读最新配置，改配置不用重启队列。
     */
    public long collectionWindowMs() {
        return fixedWindowDecision == null
                ? 0L : Math.max(0L,
                configSupplier.get().decisionPolicy().getMaxCollectionWaitMs());
    }

    /**
     * 一组请求的预测耗时上限，0 表示不限制。
     */
    public long predictedExecutionBudgetMs() {
        if (fixedWindowDecision == null) {
            return 0L;
        }
        Long configured = fixedWindowDecision.getMaxPredictedExecutionMs();
        return configured == null ? 0L : configured;
    }
}
