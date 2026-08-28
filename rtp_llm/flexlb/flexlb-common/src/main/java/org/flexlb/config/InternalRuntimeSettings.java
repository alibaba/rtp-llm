package org.flexlb.config;

import lombok.Getter;

/**
 * Implementation sizing defaults. These values are intentionally not part of
 * the public FLEXLB_CONFIG schema.
 */
@Getter
public final class InternalRuntimeSettings {

    private final int prefillSaturatedAtPendingRequests = 20;
    private final long decodeFullSpeedBelowKvUsagePercent = 40;
    private final long decodeSaturatedAtKvUsagePercent = 80;

    private final int nettySelectThreadMultiplier = 1;
    private final int nettyWorkerThreadMultiplier = 2;
    private final int grpcClientExecutorThreads = 32;
    private final int grpcClientExecutorQueueCapacity = 10_000;
    private final int grpcClientEventLoopThreads = 8;
    private final int grpcServerWorkerEventLoopThreads = 4;
    private final int httpNettyEventLoopThreads = 4;
    private final int httpNettyEventExecutorThreads = 16;
    private final int httpNettyEventExecutorQueueCapacity = 1000;
    private final int httpRequestExecutorThreads = 32;
    private final int httpRequestExecutorQueueCapacity = 10_000;
    private final int engineSyncExecutorThreads = 32;
    private final int statusCheckExecutorThreads = 32;
    private final int serviceDiscoveryMaxThreads = 32;
    private final int batchDispatchThreads = 64;
    private final int batchDispatchQueueCapacity = 256;
    private final int fallbackBatchTokenCapacity = 1_048_576;
    private final long masterForwardRpcTimeoutMs = 5000;
}
