package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.CacheMatchProvider;
import org.flexlb.cache.service.CacheMatchSource;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Default implementation of cache-aware service
 * Provides unified cache management through the configured cache metadata provider.
 *
 * @author FlexLB
 */
@Slf4j
@Service
public class DefaultCacheAwareService implements CacheAwareService {

    private final CacheMetricsReporter cacheMetricsReporter;
    private final CacheMatchProvider cacheMatchProvider;

    public DefaultCacheAwareService(
            List<CacheMatchProvider> cacheMatchProviders,
            ModelMetaConfig modelMetaConfig,
            CacheMetricsReporter cacheMetricsReporter) {
        this.cacheMetricsReporter = cacheMetricsReporter;
        Map<CacheMatchSource, CacheMatchProvider> providers = new EnumMap<>(CacheMatchSource.class);
        for (CacheMatchProvider provider : cacheMatchProviders) {
            CacheMatchProvider previous = providers.putIfAbsent(provider.source(), provider);
            if (previous != null) {
                throw new IllegalStateException("Multiple cache match providers registered for " + provider.source());
            }
        }

        boolean kvcmEnabled = modelMetaConfig.getServiceRoutes().stream()
                .anyMatch(ServiceRoute::isKvcmEnabled);
        CacheMatchSource source = kvcmEnabled ? CacheMatchSource.KVCM : CacheMatchSource.LOCAL;
        this.cacheMatchProvider = providers.get(source);
        if (cacheMatchProvider == null) {
            throw new IllegalStateException("No cache match provider registered for " + source);
        }
        log.info("Using cache match provider: {}", source);
    }
    
    @Override
    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, String group) {

        long startTime = System.nanoTime() / 1000;

        try {
            if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
                return Collections.emptyMap();
            }

            Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> resultMap
                = cacheMatchProvider.findMatchingEngines(blockCacheKeys, roleType, group);

            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "0");

            return resultMap;
        } catch (Exception e) {
            cacheMetricsReporter.reportFindMatchingEnginesRT(roleType, startTime, "1");
            log.error("Error finding matching engines for role: {}", roleType, e);
            return Collections.emptyMap();
        }
    }
    
    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
        return cacheMatchProvider.updateEngineBlockCache(workerStatus);
    }
}
