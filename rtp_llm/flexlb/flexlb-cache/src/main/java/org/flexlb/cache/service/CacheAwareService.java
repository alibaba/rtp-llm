package org.flexlb.cache.service;

import java.util.List;
import java.util.Map;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

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
     * @param blockCacheKeys 查询的缓存块ID列表
     * @param modelName      模型名称
     * @param roleType       查询的引擎角色
     * @param group          查询的引擎组
     * @return 引擎匹配结果映射，key: engineIpPort，value: prefixMatchLength
     */
    Map<String/*engineIpPort*/, Integer/*prefixMatchLength*/> findMatchingEngines(List<Long> blockCacheKeys, String modelName, RoleType roleType, String group);
    
    /**
     * 更新Engine的Block KvCache的缓存状态
     *
     * @param workerStatus worker状态信息
     * @return 更新结果
     */
    WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus);
}