package org.flexlb.config;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Centralized cache-matching configuration derived from the parsed model service routes.
 */
@Getter
@Slf4j
@Component
public class CacheMatchConfiguration {

    private final List<ServiceRoute> serviceRoutes;
    private final ServiceRoute kvcmServiceRoute;
    private final KvcmConfig kvcmConfig;
    private final LocalStandbyConfig localStandbyConfig;
    private final boolean kvcmEnabled;
    private final boolean localSyncEnabled;
    private final boolean localStandbyEnabled;
    private final boolean autoSwitchEnabled;
    private final CacheMatchMode configuredMode;

    public CacheMatchConfiguration(ModelMetaConfig modelMetaConfig) {
        this.serviceRoutes = List.copyOf(modelMetaConfig.getServiceRoutes());
        this.kvcmServiceRoute = resolveKvcmServiceRoute(serviceRoutes);
        this.kvcmConfig = kvcmServiceRoute == null ? null : kvcmServiceRoute.getKvcm();
        this.kvcmEnabled = kvcmConfig != null && kvcmConfig.isEnabled();
        this.localSyncEnabled = !kvcmEnabled;
        this.localStandbyEnabled = kvcmEnabled;
        this.localStandbyConfig = resolveLocalStandbyConfig();
        this.autoSwitchEnabled = localStandbyEnabled && (localStandbyConfig != null && localStandbyConfig.isAutoSwitch());
        this.configuredMode = kvcmEnabled ? CacheMatchMode.KVCM : CacheMatchMode.LOCAL_SYNC;
        logInitialization();
    }

    private ServiceRoute resolveKvcmServiceRoute(List<ServiceRoute> routes) {
        for (ServiceRoute route : routes) {
            if (route != null && route.isKvcmEnabled()) {
                return route;
            }
        }
        return null;
    }

    private LocalStandbyConfig resolveLocalStandbyConfig() {
        if (!kvcmEnabled) {
            return null;
        }
        LocalStandbyConfig configured = kvcmConfig.getLocalStandby();
        return configured == null ? new LocalStandbyConfig() : configured;
    }

    private void logInitialization() {
        log.info("Cache match configuration initialized: configuredMode={}, kvcmEnabled={}, "
                        + "localSyncEnabled={}, localStandbyEnabled={}, autoSwitchEnabled={}",
                configuredMode,
                kvcmEnabled,
                localSyncEnabled,
                localStandbyEnabled,
                autoSwitchEnabled);
        if (kvcmEnabled) {
            log.info("KVCM cache matching configuration: serviceId={}, address={}, namespace={}, "
                            + "requestTimeoutMs={}, leaderRefreshIntervalMs={}, "
                            + "heartbeatFailureThreshold={}, queryFailureThreshold={}, maxQueryRetryCount={}, "
                            + "recoverySuccessThreshold={}, p2pHostCount={}",
                    kvcmServiceRoute.getServiceId(),
                    kvcmConfig.getAddress(),
                    kvcmConfig.getNamespace(),
                    kvcmConfig.getRequestTimeoutMs(),
                    kvcmConfig.getLeaderRefreshIntervalMs(),
                    kvcmConfig.getHeartbeatFailureThreshold(),
                    kvcmConfig.getQueryFailureThreshold(),
                    kvcmConfig.getMaxQueryRetryCount(),
                    kvcmConfig.getRecoverySuccessThreshold(),
                    kvcmConfig.getP2pHostCount());
        }
        if (localStandbyEnabled) {
            log.info("Local standby cache configuration: autoSwitch={}, blockSize={}, "
                            + "ttlMs={}, minimumTtlMs={}, ttlReductionStartRatio={}, "
                            + "maximumEntries={}, capacityMultiplier={}, asyncQueueCapacity={}, "
                            + "hashThreadCount={}, hashQueueCapacity={}",
                    localStandbyConfig.isAutoSwitch(),
                    localStandbyConfig.getBlockSize(),
                    localStandbyConfig.getTtlMs(),
                    localStandbyConfig.getMinimumTtlMs(),
                    localStandbyConfig.getTtlReductionStartRatio(),
                    localStandbyConfig.getMaximumEntries(),
                    localStandbyConfig.getCapacityMultiplier(),
                    localStandbyConfig.getAsyncQueueCapacity(),
                    localStandbyConfig.getHashThreadCount(),
                    localStandbyConfig.getHashQueueCapacity());
        }
    }
}
