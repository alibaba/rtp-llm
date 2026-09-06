package org.flexlb.config;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.springframework.beans.factory.annotation.Autowired;
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
    private final KvcmCacheMatchingConfig kvcmRuntimeConfig;
    private final LocalStandbyConfig localStandbyConfig;
    private final boolean kvcmEnabled;
    private final boolean localSyncEnabled;
    private final boolean localStandbyEnabled;
    private final boolean autoSwitchEnabled;
    private final CacheMatchMode configuredMode;

    @Autowired
    public CacheMatchConfiguration(
            ModelMetaConfig modelMetaConfig,
            ConfigService configService) {
        this(modelMetaConfig, configService.loadBalanceConfig());
    }

    public CacheMatchConfiguration(
            ModelMetaConfig modelMetaConfig,
            FlexlbConfig flexlbConfig) {
        this.serviceRoutes = List.copyOf(modelMetaConfig.getServiceRoutes());
        this.kvcmServiceRoute = resolveKvcmServiceRoute(serviceRoutes);
        this.kvcmConfig = kvcmServiceRoute == null ? null : kvcmServiceRoute.getKvcm();
        this.kvcmEnabled = flexlbConfig.isKvcmCacheMatching();
        if (kvcmEnabled && kvcmConfig == null) {
            throw new IllegalStateException(
                    "FLEXLB_CONFIG cacheMatching.type=KVCM requires MODEL_SERVICE_CONFIG kvcm topology");
        }
        this.kvcmRuntimeConfig = kvcmEnabled
                ? flexlbConfig.kvcmCacheMatching()
                : null;
        this.localSyncEnabled = !kvcmEnabled;
        this.localStandbyEnabled = kvcmEnabled;
        this.localStandbyConfig = kvcmEnabled
                ? kvcmRuntimeConfig.getLocalStandby()
                : null;
        this.autoSwitchEnabled = localStandbyEnabled
                && localStandbyConfig.isAutoSwitch();
        this.configuredMode = kvcmEnabled ? CacheMatchMode.KVCM : CacheMatchMode.LOCAL_SYNC;
        logInitialization();
    }

    private ServiceRoute resolveKvcmServiceRoute(List<ServiceRoute> routes) {
        for (ServiceRoute route : routes) {
            if (route != null && route.getKvcm() != null) {
                return route;
            }
        }
        return null;
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
                    kvcmRuntimeConfig.getRequestTimeoutMs(),
                    kvcmRuntimeConfig.getLeaderRefreshIntervalMs(),
                    kvcmRuntimeConfig.getHeartbeatFailureThreshold(),
                    kvcmRuntimeConfig.getQueryFailureThreshold(),
                    kvcmRuntimeConfig.getMaxQueryRetryCount(),
                    kvcmRuntimeConfig.getRecoverySuccessThreshold(),
                    kvcmRuntimeConfig.getP2pHostCount());
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
