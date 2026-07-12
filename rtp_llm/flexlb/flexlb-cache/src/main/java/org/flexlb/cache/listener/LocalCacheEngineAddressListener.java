package org.flexlb.cache.listener;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Removes local cache entries for engines that are no longer discoverable.
 */
@Slf4j
@Component
public class LocalCacheEngineAddressListener implements EngineAddressResolver.Listener {

    private final KvCacheManager kvCacheManager;

    public LocalCacheEngineAddressListener(
            EngineAddressResolver addressResolver,
            KvCacheManager kvCacheManager,
            ModelMetaConfig modelMetaConfig) {
        this.kvCacheManager = kvCacheManager;

        boolean kvcmEnabled = modelMetaConfig.getServiceRoutes().stream()
                .anyMatch(ServiceRoute::isKvcmEnabled);
        if (kvcmEnabled) {
            log.info("KVCM is enabled; local cache engine address cleanup is disabled");
            return;
        }
        addressResolver.subscribe(this);
    }

    @Override
    public void onAddressUpdate(List<String> ipPortList) {
        if (ipPortList == null) {
            log.warn("Ignoring null engine address update for local cache cleanup");
            return;
        }
        kvCacheManager.removeStaleEngineCaches(ipPortList);
    }
}
