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
        private long statusPollIntervalMs = 20;
        private long statusRpcTimeoutMs = 5000;
        private long statusStaleAfterMs = 10_000;
        private long taskConfirmationTimeoutMs = 300_000;
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
