package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.core.KvCacheManager;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 缓存感知服务默认实现
 * 提供统一的缓存管理服务，封装底层的KvCacheManager
 * 
 * @author FlexLB
 */
@Slf4j
@Service
public class DefaultCacheAwareService implements CacheAwareService {
    
    @Autowired
    private KvCacheManager kvCacheManager;
    
    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;
    
    @Override
    public Map<String, Integer> findMatchingEngines(List<Long> blockCacheKeys,
        RoleType roleType, String group) {

        long startTime = System.nanoTime() / 1000;

        try {
            if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
                return Collections.emptyMap();
            }

            Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> resultMap
                = kvCacheManager.findMatchingEngines(blockCacheKeys, roleType, group);

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
        long startTime = System.nanoTime() / 1000;
        String engineIpPort = workerStatus.getIpPort();
        String role = workerStatus.getRole();

        try {
            if (workerStatus.getCacheStatus() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cache Status is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
                return result;
            }

            String ipPort = workerStatus.getIpPort();
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus.getCachedKeys() == null) {
                WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, "Worker Cached Keys is null");
                cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
                return result;
            }

            Set<Long> cachedKeys = cacheStatus.getCachedKeys();
            
            // 更新缓存
            kvCacheManager.updateEngineCache(ipPort, role, cachedKeys);
            
            WorkerCacheUpdateResult result = buildSuccessResult(workerStatus, cacheStatus);

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(ipPort, role, startTime, "1");
            
            return result;
                
        } catch (Throwable e) {
            log.error("Error updating worker cache for: {}", engineIpPort, e);
            
            WorkerCacheUpdateResult result = buildFailureResult(engineIpPort, e.getMessage());

            cacheMetricsReporter.reportUpdateEngineBlockCacheRT(engineIpPort, role, startTime, "0");
            
            return result;
        }
    }

    /**
     * 构建成功结果
     */
    private WorkerCacheUpdateResult buildSuccessResult(WorkerStatus workerStatus, CacheStatus cacheStatus) {
        return WorkerCacheUpdateResult.builder()
            .success(true)
            .engineIpPort(workerStatus.getIpPort())
            .cacheBlockCount(cacheStatus.getCachedKeys() != null ? cacheStatus.getCachedKeys().size() : 0)
            .availableKvCache(cacheStatus.getAvailableKvCache())
            .totalKvCache(cacheStatus.getTotalKvCache())
            .cacheVersion(cacheStatus.getVersion())
            .build();
    }
    
    /**
     * 构建失败结果
     */
    private WorkerCacheUpdateResult buildFailureResult(String engineIpPort, String errorMessage) {
        return WorkerCacheUpdateResult.builder()
            .success(false)
            .engineIpPort(engineIpPort)
            .errorMessage(errorMessage)
            .build();
    }
}