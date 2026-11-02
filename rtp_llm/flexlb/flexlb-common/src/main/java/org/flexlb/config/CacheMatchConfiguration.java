package org.flexlb.config;

import lombok.Getter;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.springframework.stereotype.Component;

/** One startup cache mode; runtime tuning never changes lifecycle ownership. */
@Getter
@Component
public final class CacheMatchConfiguration {
    private final KvcmConfig kvcmConfig;
    private final boolean kvcmEnabled;
    private final KvcmCacheMatchingConfig kvcmRuntimeConfig;

    @org.springframework.beans.factory.annotation.Autowired
    public CacheMatchConfiguration(ModelMetaConfig model, ConfigService service) {
        this(model.getServiceRoute(), service.loadBalanceConfig());
    }

    public CacheMatchConfiguration(ServiceRoute route, FlexlbConfig config) {
        kvcmEnabled = config.getCacheMatching() instanceof KvcmCacheMatchingConfig;
        kvcmRuntimeConfig = kvcmEnabled ? (KvcmCacheMatchingConfig) config.getCacheMatching() : null;
        kvcmConfig = route.getKvcm();
        if (kvcmEnabled && (kvcmConfig == null || kvcmConfig.getAddress() == null || kvcmConfig.getAddress().isBlank()
                || kvcmConfig.getNamespace() == null || kvcmConfig.getNamespace().isBlank()
                || kvcmConfig.getPort() < 1 || kvcmConfig.getPort() > 65535)) {
            throw new IllegalArgumentException("KVCM requires MODEL_SERVICE_CONFIG.kvcm address, port and namespace");
        }
    }

    public boolean isAutoSwitchEnabled() { return kvcmEnabled && getLocalStandbyConfig().isAutoSwitch(); }
    public CacheMatchMode getConfiguredMode() { return kvcmEnabled ? CacheMatchMode.KVCM : CacheMatchMode.LOCAL_SYNC; }
    public boolean isLocalSyncEnabled() { return !kvcmEnabled; }
    public boolean isLocalStandbyEnabled() { return kvcmEnabled; }
    public LocalStandbyConfig getLocalStandbyConfig() { return kvcmEnabled ? kvcmRuntimeConfig.getLocalStandby() : null; }
}
