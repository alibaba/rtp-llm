package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.LoadBalanceStrategyEnum;

/**
 * @author zjw
 * description:
 * date: 2025/3/10
 */
@Getter
@Setter
public class WhaleMasterConfig {

    /**
     * 负载均衡策略
     */
    private LoadBalanceStrategyEnum loadBalanceStrategy = LoadBalanceStrategyEnum.SHORTEST_TTFT;
    /**
     * 权重衰减因子，控制权重差异程度
     * 值越小权重差异越小，值越大权重差异越明显
     * 建议范围：0.001-0.01 (针对缓存使用量数值范围优化)
     */
    private double weightedCacheDecayFactor = 0.001;
}
