package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.LoadBalanceStrategyEnum;

/**
 * 支持环境变量覆盖配置
 * 环境变量命名规则: {FIELD_NAME_UPPER_SNAKE_CASE}
 * 例如: enableQueueing -> ENABLE_QUEUEING
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

    // ========== 排队配置 ==========

    /**
     * 是否启用排队
     */
    private boolean enableQueueing = false;

    /**
     * 每个模型的最大队列长度
     */
    private int maxQueueSize = 1000000;

    /**
     * Prefill角色的排队阈值
     * 当小于此阈值时,认为该Worker可用
     */
    private long prefillQueueSizeThreshold = 3;

    /**
     * Decode角色的KV cache可用阈值(百分比)
     * 当Worker的KV cache使用率小于此阈值时,认为该Worker可用
     * 范围: 1-100, 默认90表示当使用率超过90%时Worker不可用
     */
    private long decodeAvailableMemoryThreshold = 90;

    /**
     * 资源可用性迟滞偏置(百分比)
     * 用于防止资源可用性在阈值附近频繁切换
     * 范围: 0-100, 默认15表示迟滞范围为15%
     */
    private long hysteresisBiasPercent = 15;

    // ========== Worker 线程池配置 ==========

    /**
     * Worker线程池核心线程数(默认CPU核心数)
     */
    private int workerThreadPoolCoreSize = Runtime.getRuntime().availableProcessors();

    /**
     * 资源可用性检查间隔(毫秒,默认10ms)
     */
    private long resourceCheckIntervalMs = 10;

    // ========== Netty 线程池配置 ==========

    /**
     * Netty select线程数倍数(默认1)
     * 实际select线程数 = availableProcessors * nettySelectThreadMultiplier
     */
    private int nettySelectThreadMultiplier = 1;

    /**
     * Netty worker线程数倍数(默认2)
     * 实际worker线程数 = availableProcessors * nettyWorkerThreadMultiplier
     */
    private int nettyWorkerThreadMultiplier = 2;
}
