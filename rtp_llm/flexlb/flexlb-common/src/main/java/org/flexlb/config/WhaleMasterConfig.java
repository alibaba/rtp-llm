package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.LoadBalanceStrategyEnum;

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
    private int maxQueueSize = 100000;

    /**
     * 最大重试次数
     */
    private int maxRetryCount = 300;

    /**
     * 最大排队等待时间(毫秒)
     */
    private long maxQueueWaitTimeMs = 5 * 60 * 1000;

    /**
     * 调度间隔(毫秒)
     */
    private long queueScheduleIntervalMs = 50;

    /**
     * Prefill角色的排队阈值
     * 当小于此阈值时,认为该Worker可用
     */
    private long prefillQueueSizeThreshold = 3;

    /**
     * Decode角色的可用显存阈值(tokens)
     * 当Worker的availableKvCacheTokens大于此阈值时,认为该Worker可用
     */
    private long decodeAvailableMemoryThreshold = 1000;

    // ========== Worker 线程池配置 ==========

    /**
     * Worker线程池核心线程数(默认16,建议设为CPU核心数/2)
     */
    private int workerThreadPoolCoreSize = 16;

    /**
     * Worker线程池最大线程数(默认32,建议设为CPU核心数)
     */
    private int workerThreadPoolMaxSize = 32;

    /**
     * Worker线程池队列容量(默认10000)
     */
    private int workerThreadPoolQueueSize = 10000;

    /**
     * 资源可用性检查间隔(毫秒,默认10ms)
     */
    private long resourceCheckIntervalMs = 10;
}
