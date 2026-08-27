package org.flexlb.config;

import lombok.Getter;

import java.util.Map;

/**
 * Implementation sizing defaults. These values are intentionally not part of
 * the public FLEXLB_CONFIG schema.
 *
 * <p>The batch dispatch sizing is additionally overridable through process
 * environment variables ({@code FLEXLB_BATCH_DISPATCH_THREADS} and
 * {@code FLEXLB_BATCH_DISPATCH_QUEUE_CAPACITY}) so overload benchmarks can
 * widen the dispatcher admission gate without editing code. Defaults keep the
 * upstream production sizing (64 threads + 256 queue slots = 320 admission
 * permits). Invalid or missing values fall back to the defaults.</p>
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
    private final int batchDispatchThreads;
    private final int batchDispatchQueueCapacity;
    private final int fallbackBatchTokenCapacity = 1_048_576;
    private final long masterForwardRpcTimeoutMs = 5000;

    static final int DEFAULT_BATCH_DISPATCH_THREADS = 64;
    static final int DEFAULT_BATCH_DISPATCH_QUEUE_CAPACITY = 256;
    static final String BATCH_DISPATCH_THREADS_ENV = "FLEXLB_BATCH_DISPATCH_THREADS";
    static final String BATCH_DISPATCH_QUEUE_CAPACITY_ENV = "FLEXLB_BATCH_DISPATCH_QUEUE_CAPACITY";

    public InternalRuntimeSettings() {
        this(System.getenv());
    }

    /** Package-visible env injection keeps unit tests deterministic. */
    InternalRuntimeSettings(Map<String, String> env) {
        this.batchDispatchThreads = positiveIntOrDefault(
                env.get(BATCH_DISPATCH_THREADS_ENV), DEFAULT_BATCH_DISPATCH_THREADS);
        this.batchDispatchQueueCapacity = positiveIntOrDefault(
                env.get(BATCH_DISPATCH_QUEUE_CAPACITY_ENV), DEFAULT_BATCH_DISPATCH_QUEUE_CAPACITY);
    }

    private static int positiveIntOrDefault(String raw, int fallback) {
        if (raw == null || raw.isEmpty()) {
            return fallback;
        }
        try {
            int parsed = Integer.parseInt(raw.trim());
            return parsed > 0 ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }
}
