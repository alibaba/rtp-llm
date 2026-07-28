package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchSource;
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
    private final KvcmGrpcClient kvcmGrpcClient;
    private final AtomicBoolean manualFallbackActive = new AtomicBoolean();
    private final AtomicReference<CacheMatchSource> activeSource =
            new AtomicReference<>(CacheMatchSource.KVCM);
    private final AtomicLong lastFailoverTimeMs = new AtomicLong();
    private final AtomicReference<String> lastFailoverReason = new AtomicReference<>("initial");

    public CacheMatchFailoverManager(CacheMatchConfiguration configuration, KvcmGrpcClient kvcmGrpcClient) {
        this.autoSwitchEnabled = configuration.isAutoSwitchEnabled();
        this.kvcmGrpcClient = kvcmGrpcClient;
        this.kvcmGrpcClient.setHealthSnapshotListener(this::updateFromKvcmHealth);
        updateFromKvcmHealth(this.kvcmGrpcClient.healthSnapshot());
    }

    public CacheMatchSource activeSource() {
        return activeSource.get();
    }

    /**
     * Reconciles the active cache source with the latest KVCM health snapshot.
     * This method is called after every scheduled heartbeat and must remain idempotent.
     */
    void updateFromKvcmHealth(KvcmHealthSnapshot health) {
        // A manual fallback is an operator override and has higher priority than health updates.
        if (manualFallbackActive.get()) {
            return;
        }

        // Healthy snapshots converge the cache source back to KVCM.
        if (health.isHealthy()) {
            updateActiveSource(CacheMatchSource.KVCM, "KVCM heartbeat recovered");
            return;
        }

        // Unhealthy snapshots activate Local Standby only when automatic failover is enabled.
        if (autoSwitchEnabled) {
            updateActiveSource(CacheMatchSource.LOCAL_STANDBY, health.lastStateChangeReason());
            return;
        }

        // Keep the current source unchanged and wait for an explicit manual fallback.
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
        if (!health.isHealthy() && !autoSwitchEnabled) {
            throw new IllegalStateException("cannot recover KVCM primary while KVCM is unhealthy and automatic failover is disabled");
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
