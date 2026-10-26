package org.flexlb.cache.match;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.CacheMatchMode;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Selects the cache source from cacheMatching.type and KVCM client health.
 */
@Slf4j
@Component
public class CacheMatchFailoverManager {

    private final boolean autoSwitchEnabled;
    private final boolean kvcmEnabled;
    private CacheMatchMode configuredMode;
    private final KvcmGrpcClient kvcmGrpcClient;
    private final CacheMetricsReporter cacheMetricsReporter;
    private final AtomicBoolean autoSwitchDisabledWarningLogged = new AtomicBoolean();
    private final AtomicReference<CacheMatchSource> activeSource;
    private final AtomicLong lastFailoverTimeMs = new AtomicLong();
    private final AtomicReference<String> lastFailoverReason = new AtomicReference<>("initial");

    public CacheMatchFailoverManager(CacheMatchConfiguration configuration,
                                     KvcmGrpcClient kvcmGrpcClient,
                                     CacheMetricsReporter cacheMetricsReporter,
                                     ConfigService configService) {
        this.autoSwitchEnabled = configuration.isAutoSwitchEnabled();
        this.kvcmEnabled = configuration.isKvcmEnabled();
        this.configuredMode = configuration.getConfiguredMode();
        this.kvcmGrpcClient = kvcmGrpcClient;
        this.cacheMetricsReporter = cacheMetricsReporter;
        CacheMatchSource initialSource = CacheMatchSource.valueOf(configuredMode.name());
        this.activeSource = new AtomicReference<>(initialSource);
        this.cacheMetricsReporter.reportActiveCacheMatchSource(initialSource);
        if (kvcmEnabled) {
            this.kvcmGrpcClient.setHealthSnapshotListener(ignored -> updateFromKvcmHealth());
            updateFromKvcmHealth();
            configService.addUpdateListener(this::updateConfiguredMode);
        }
    }

    private synchronized void updateConfiguredMode(FlexlbConfig config) {
        CacheMatchMode type = config.getCacheMatching().getType();
        if (type == CacheMatchMode.LOCAL_SYNC) {
            // LOCAL_SYNC uses components selected at startup, so switching to it requires a restart.
            return;
        }
        if (type == configuredMode) {
            return;
        }
        configuredMode = type;
        autoSwitchDisabledWarningLogged.set(false);
        KvcmHealthSnapshot health = kvcmGrpcClient.healthSnapshot();
        CacheMatchSource source;
        if (type == CacheMatchMode.LOCAL_STANDBY) {
            // Explicit standby selection takes precedence over KVCM health.
            source = CacheMatchSource.LOCAL_STANDBY;
            log.info("Using Local Standby as explicitly configured: type={}, kvcmHealthy={}, autoSwitchEnabled={}",
                    type, health.isHealthy(), autoSwitchEnabled);
        } else if (autoSwitchEnabled && !health.isHealthy()) {
            // In KVCM mode, automatic failover uses standby while KVCM is unhealthy.
            source = CacheMatchSource.LOCAL_STANDBY;
            log.info("Using Local Standby because KVCM is unhealthy and automatic failover is enabled: "
                    + "type={}, kvcmHealthy={}, autoSwitchEnabled={}", type, false, true);
        } else {
            // Use KVCM when it is healthy or automatic failover is disabled.
            source = CacheMatchSource.KVCM;
            log.info(
                    "Using KVCM because it is healthy or automatic failover is disabled: "
                            + "type={}, kvcmHealthy={}, autoSwitchEnabled={}",
                    type, health.isHealthy(), autoSwitchEnabled);
        }
        String reason = "configured cacheMatching.type=" + type;
        if (!updateActiveSource(source, reason)) {
            lastFailoverReason.set(reason);
            lastFailoverTimeMs.set(System.currentTimeMillis());
        }
        cacheMetricsReporter.reportActiveCacheMatchSource(source);
        log.info("Cache matching configuration changed: configuredMode={}, activeSource={}", type, source);
    }

    public CacheMatchSource activeSource() {
        return activeSource.get();
    }

    /**
     * Reconciles the active cache source with the latest KVCM health snapshot.
     * This method is called after every scheduled heartbeat and must remain idempotent.
     */
    synchronized void updateFromKvcmHealth() {
        if (!kvcmEnabled) {
            return;
        }
        updateFromKvcmHealth(kvcmGrpcClient.healthSnapshot());
    }

    private void updateFromKvcmHealth(KvcmHealthSnapshot health) {
        if (configuredMode == CacheMatchMode.LOCAL_STANDBY) {
            cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
            return;
        }
        // Healthy snapshots converge the cache source back to KVCM.
        if (health.isHealthy()) {
            autoSwitchDisabledWarningLogged.set(false);
            updateActiveSource(CacheMatchSource.KVCM, "KVCM health recovered");
            cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
            return;
        }

        // Unhealthy snapshots activate Local Standby only when automatic failover is enabled.
        if (autoSwitchEnabled) {
            updateActiveSource(CacheMatchSource.LOCAL_STANDBY, health.lastStateChangeReason());
            cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
            return;
        }

        // Keep KVCM selected until a configuration update selects Local Standby.
        cacheMetricsReporter.reportActiveCacheMatchSource(activeSource());
        if (autoSwitchDisabledWarningLogged.compareAndSet(false, true)) {
            log.warn(
                    "KVCM is unavailable but automatic failover is disabled; "
                            + "set cacheMatching.type=LOCAL_STANDBY to select standby, reason={}, "
                            + "consecutiveQueryFailures={}, consecutiveHeartbeatFailures={}",
                    health.lastStateChangeReason(), health.consecutiveQueryFailures(),
                    health.consecutiveHeartbeatFailures());
        }
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

    private boolean updateActiveSource(CacheMatchSource desiredSource, String reason) {
        while (true) {
            CacheMatchSource currentSource = activeSource.get();
            if (currentSource == desiredSource) {
                return false;
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
                return true;
            }
        }
    }

}
