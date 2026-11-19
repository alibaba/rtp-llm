package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * Groups worker liveness polling and cache-status refresh policy.
 */
@Getter
@Setter
public final class WorkerRegistryConfig {

    private HealthConfig health = new HealthConfig();
    private CacheStatusConfig cacheStatus = new CacheStatusConfig();
    private ExecutorsConfig executors = new ExecutorsConfig();

    /**
     * Thread counts are captured when the owning executors start.
     */
    @Getter
    @Setter
    public static final class ExecutorsConfig {
        private int cacheStatusThreads = 32;

        /**
         * CPU-bound diff work is also capped by available processors.
         */
        private int cacheDiffParallelism = 8;
    }

    /**
     * Defines status polling, RPC timeout, stale-status and cleanup intervals in milliseconds.
     */
    @Getter
    @Setter
    public static final class HealthConfig {
        private long statusPollIntervalMs = 20;
        private long statusRpcTimeoutMs = 5000;
        private long cleanupIntervalMs = 3000;
    }

    /**
     * Bounds adaptive cache refresh intervals and the target size of metadata diffs.
     */
    @Getter
    @Setter
    public static final class CacheStatusConfig {
        private int targetDiffSize = 30;
        private long minRefreshIntervalMs = 50;
        private long maxRefreshIntervalMs = 3000;
        private boolean fullSnapshotDebugMode;
    }
}
