package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class WorkerRegistryConfig {

    private HealthConfig health = new HealthConfig();
    private CacheStatusConfig cacheStatus = new CacheStatusConfig();

    @Getter
    @Setter
    public static final class HealthConfig {
        public static final long DEFAULT_VIT_STATUS_RPC_TIMEOUT_MS = 2000;
        public static final boolean DEFAULT_RETAIN_VIT_ALIVE_ON_TIMEOUT = true;

        private long statusPollIntervalMs = 20;
        private long statusRpcTimeoutMs = 5000;
        private long statusStaleAfterMs = 10_000;
        private long vitStatusRpcTimeoutMs = DEFAULT_VIT_STATUS_RPC_TIMEOUT_MS;
        private boolean retainVitAliveOnTimeout = DEFAULT_RETAIN_VIT_ALIVE_ON_TIMEOUT;
    }

    @Getter
    @Setter
    public static final class CacheStatusConfig {
        private int targetDiffSize = 30;
        private long minRefreshIntervalMs = 50;
        private long maxRefreshIntervalMs = 3000;
        private boolean fullSnapshotDebugMode;
    }
}
