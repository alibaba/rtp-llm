package org.flexlb.cache.service;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/**
 * 缓存感知服务接口
 * 为外部模块提供统一的缓存管理接口
 * 
 * @author FlexLB
 */
public interface CacheAwareService {
    
    /**
     * 查找匹配的引擎
     *
     * @param blockCacheKeys List of cache block IDs to query
     * @param roleType       Engine role to query
     * @param group          Engine group to query
     * @return Engine matching result map, key: engineIpPort, value: prefixMatchLength
     */
    Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(List<Long> blockCacheKeys, RoleType roleType, String group);
    
    /**
     * 更新Engine的Block KvCache的缓存状态
     *
     * @param workerStatus worker状态信息
     * @return 更新结果
     */
    WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus);
}