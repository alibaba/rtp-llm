package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Selects the configured cache source from manual failover state and KVCM client health.
 */
@Slf4j
@Component
public class CacheMatchFailoverManager {

    private final boolean autoSwitchEnabled;
    private final boolean kvcmEnabled;
    private final KvcmGrpcClient kvcmGrpcClient;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final AtomicBoolean manualFallbackActive = new AtomicBoolean();
    private final AtomicReference<CacheMatchSource> activeSource;
    private final AtomicLong lastFailoverTimeMs = new AtomicLong();
    private final AtomicReference<String> lastFailoverReason = new AtomicReference<>("initial");

    public CacheMatchFailoverManager(
            CacheMatchConfiguration configuration,
            KvcmGrpcClient kvcmGrpcClient,
            CacheMetricsReporter cacheMetricsReporter) {
        this.autoSwitchEnabled = configuration.isAutoSwitchEnabled();
        this.kvcmEnabled = configuration.isKvcmEnabled();
        this.kvcmGrpcClient = kvcmGrpcClient;
        this.cacheMetricsReporter = cacheMetricsReporter;
        CacheMatchSource initialSource = kvcmEnabled
                ? CacheMatchSource.KVCM
                : CacheMatchSource.LOCAL_SYNC;
        this.activeSource = new AtomicReference<>(initialSource);
        this.cacheMetricsReporter.reportActiveCacheMatchSource(initialSource);
        if (kvcmEnabled) {
            this.kvcmGrpcClient.setHealthSnapshotListener(this::updateFromKvcmHealth);
            updateFromKvcmHealth(this.kvcmGrpcClient.healthSnapshot());
        }
    }

    public CacheMatchSource activeSource() {
        return activeSource.get();
    }

    /**
     * Reconciles the active cache source with the latest KVCM health snapshot.
     * This method is called after every scheduled heartbeat and must remain idempotent.
     */
    void updateFromKvcmHealth(KvcmHealthSnapshot health) {
        if (!kvcmEnabled) {
            return;
        }

        // A manual fallback is an operator override and has higher priority than health updates.
        if (manualFallbackActive.get()) {
            cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
            return;
        }

        // Healthy snapshots converge the cache source back to KVCM.
        if (health.isHealthy()) {
            updateActiveSource(CacheMatchSource.KVCM, "KVCM heartbeat recovered");
            cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
            return;
        }

        // Unhealthy snapshots activate Local Standby only when automatic failover is enabled.
        if (autoSwitchEnabled) {
            updateActiveSource(CacheMatchSource.LOCAL_STANDBY, health.lastStateChangeReason());
            cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
            return;
        }

        // Keep the current source unchanged and wait for an explicit manual fallback.
        cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
        log.warn("KVCM is unavailable but automatic failover is disabled; manual failover is required, reason={}, "
                        + "consecutiveQueryFailures={}, consecutiveHeartbeatFailures={}",
                health.lastStateChangeReason(),
                health.consecutiveQueryFailures(),
                health.consecutiveHeartbeatFailures());
    }

    public void activateFallbackManually() {
        manualFallbackActive.set(true);
        updateActiveSource(CacheMatchSource.LOCAL_STANDBY, "manual failover activated");
        log.info("Manual cache failover activated; Local Standby is the active cache source");
    }

    public void recoverPrimaryManually() {
        KvcmHealthSnapshot health = kvcmGrpcClient.healthSnapshot();
        if (!health.isHealthy()) {
            throw new IllegalStateException("cannot recover KVCM primary while KVCM is unhealthy");
        }
        manualFallbackActive.set(false);
        updateFromKvcmHealth(health);
        log.info("Manual cache failover cleared; active cache source follows KVCM health, source={}", activeSource());
    }

    public long lastFailoverTimeMs() {
        return lastFailoverTimeMs.get();
    }

    public String lastFailoverReason() {
        return lastFailoverReason.get();
    }

    public KvcmHealthSnapshot healthSnapshot() {
        return kvcmGrpcClient.healthSnapshot();
    }

    private void updateActiveSource(CacheMatchSource desiredSource, String reason) {
        while (true) {
            CacheMatchSource currentSource = activeSource.get();
            if (currentSource == desiredSource) {
                return;
            }
            if (activeSource.compareAndSet(currentSource, desiredSource)) {
                lastFailoverReason.set(reason);
                lastFailoverTimeMs.set(System.currentTimeMillis());
                cacheMetricsReporter.reportCacheMatchSourceChange(currentSource, desiredSource);
                if (desiredSource == CacheMatchSource.LOCAL_STANDBY) {
                    log.warn("Local Standby cache fallback activated, reason={}", reason);
                } else {
                    log.info("KVCM cache matching restored, reason={}", reason);
                }
                return;
            }
        }
    }

}
