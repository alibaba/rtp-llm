package org.flexlb.config;

import lombok.Getter;

/**
 * Implementation sizing defaults. These values are intentionally not part of
 * the public FLEXLB_CONFIG schema.
 */
@Getter
public final class InternalRuntimeSettings {

    private static final String QUEUE_PLANNER_THREADS_PROPERTY =
            "flexlb.queue.planner.threads";

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
    /** Number of parallel route planners owned by the model queue by default. */
    private final int queuePlannerThreads = resolveQueuePlannerThreads();
    private final long queueDecisionThreadJoinTimeoutMs = 5_000L;
    /** Completion callbacks must not contend with the dispatch handoff pool. */
    private final int batchDispatchCompletionThreads = 8;
    private final long masterForwardRpcTimeoutMs = 5000;

    /**
     * Upper bound on the total wire-size (bytes) of concurrently in-flight
     * EnqueueBatch payloads. The admission semaphore bounds in-flight batch
     * COUNT; this bounds the serialized direct-buffer footprint those RPCs
     * pin while awaiting the engine ACK. Default 4 GiB, override via
     * -Dflexlb.dispatch.max-inflight-bytes.
     */
    private final long batchDispatchMaxInflightBytes =
            resolveBatchDispatchMaxInflightBytes();

    private static long resolveBatchDispatchMaxInflightBytes() {
        long configured = Long.getLong(
                "flexlb.dispatch.max-inflight-bytes", 0L);
        return configured > 0L ? configured : 4L * 1024 * 1024 * 1024;
    }

    private static int resolveQueuePlannerThreads() {
        int processors = Math.max(1, Runtime.getRuntime().availableProcessors());
        return Math.max(1, Integer.getInteger(
                QUEUE_PLANNER_THREADS_PROPERTY, processors));
    }
}
