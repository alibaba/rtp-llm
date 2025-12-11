package org.flexlb.cache.core;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.DiffResult;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

/**
 * 引擎本地视图 (小Hash表)
 * 管理每个引擎的本地缓存状态和元数据
 * 存储结构: EngineIpPort -> HashMap<Long>
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class EngineLocalView {

    /**
     * 核心存储结构: EngineIpPort -> Set<Long>
     */
    private final ConcurrentHashMap<String, Set<Long>> engineViews = new ConcurrentHashMap<>();

    /**
     * 自定义ForkJoin线程池用于并行计算
     */
    private final ForkJoinPool customPool = new ForkJoinPool(Math.min(Runtime.getRuntime().availableProcessors(), 8));

    /**
     * 缓存监控指标上报器
     */
    @Autowired
    private CacheMetricsReporter cacheMetricsReporter;
    
    /**
     * 动态同步间隔管理器
     */
    @Autowired
    private DynamicCacheIntervalService dynamicIntervalManager;

    /**
     * 计算Diff结果
     *
     * @param engineIPort       引擎IP
     * @param newCacheBlocks 新的缓存块集合
     * @param role           引擎角色
     * @return Diff计算结果
     */
    public DiffResult calculateDiff(String engineIPort, Set<Long> newCacheBlocks, String role) {
        if (engineIPort == null || newCacheBlocks == null) {
            return DiffResult.empty(engineIPort);
        }

        Set<Long> oldCacheBlocks = getEngineCacheBlocks(engineIPort);

        // 高效计算差异
        Set<Long> addedBlocks = ConcurrentHashMap.newKeySet(128);
        Set<Long> removedBlocks = ConcurrentHashMap.newKeySet(128);

        // 使用自定义ForkJoin线程池并行计算新增和删除的缓存块
        ForkJoinTask<?> addedTask = customPool.submit(() ->
            newCacheBlocks.parallelStream()
                 .filter(blockCacheKey -> !oldCacheBlocks.contains(blockCacheKey))
                 .forEach(addedBlocks::add)
        );

        ForkJoinTask<?> removedTask = customPool.submit(() ->
            oldCacheBlocks.parallelStream()
                .filter(blockCacheKey -> !newCacheBlocks.contains(blockCacheKey))
                .forEach(removedBlocks::add)
        );

        addedTask.join();
        removedTask.join();

        cacheMetricsReporter.reportCacheDiffMetrics(engineIPort, role, addedBlocks.size(), removedBlocks.size());

        // 更新动态同步间隔管理器的统计信息
        int diffSize = addedBlocks.size() + removedBlocks.size();
        dynamicIntervalManager.updateDiffStatistics(diffSize);

        return DiffResult.builder()
            .engineIp(engineIPort)
            .addedBlocks(addedBlocks)
            .removedBlocks(removedBlocks)
            .version("1.0.0")
            .build();

    }

    /**
     * 添加或更新引擎缓存块
     *
     * @param engineIPort     引擎IP
     * @param blockCacheKey   缓存块哈希值
     */
    public void addOrUpdateCacheBlock(String engineIPort, Long blockCacheKey) {
        if (engineIPort == null || blockCacheKey == null) {
            log.warn("Invalid parameters: engineIPort={}, blockCacheKey={}", engineIPort, blockCacheKey);
            return;
        }
        Set<Long> engineCache = engineViews.computeIfAbsent(engineIPort, k -> ConcurrentHashMap.newKeySet());

        engineCache.add(blockCacheKey);
    }

    /**
     * 移除引擎缓存块
     *
     * @param engineIPort   引擎IP
     * @param blockCacheKey 缓存块哈希值
     */
    public void removeCacheBlock(String engineIPort, Long blockCacheKey) {
        if (engineIPort == null || blockCacheKey == null) {
            return;
        }

        Set<Long> engineCache = engineViews.get(engineIPort);
        if (engineCache == null) {
            return;
        }

        boolean isRemoveSuccess = engineCache.remove(blockCacheKey);

        if (isRemoveSuccess) {

            // 如果引擎缓存为空，可以选择移除整个引擎条目
            if (engineCache.isEmpty()) {
                engineViews.remove(engineIPort);
            }
        }
    }

    /**
     * 移除某个引擎的所有缓存块
     *
     * @param engineIPort 引擎 IP
     */
    public void removeAllCacheBlockOfEngine(String engineIPort) {
        if (engineIPort == null) {
            return;
        }

        Set<Long> removed = engineViews.remove(engineIPort);
        // 如果移除失败则告警
        if (removed == null) {
            log.warn("Remove failed, the engine: {} not exist.", engineIPort);
        }
    }

    /**
     * 获取引擎的所有缓存块ID
     *
     * @param engineIPort 引擎IP
     * @return 缓存块ID集合
     */
    public Set<Long> getEngineCacheBlocks(String engineIPort) {
        if (engineIPort == null) {
            return Collections.emptySet();
        }
        Set<Long> engineCache = engineViews.get(engineIPort);
        return engineCache == null ? Collections.emptySet() : engineCache;
    }

    /**
     * 清空所有数据
     */
    public void clear() {

        engineViews.clear();
        log.info("Cleared engine local view");

    }

    public int size(String engineIpPort) {
        Set<Long> engineCache = engineViews.get(engineIpPort);
        return engineCache == null ? 0 : engineCache.size();
    }

    /**
     * 获取引擎视图Map的大小（即当前有多少个引擎）
     *
     * @return engineViews Map的大小
     */
    public int getEngineViewsMapSize() {
        return engineViews.size();
    }

    /**
     * 获取所有引擎的IP:Port集合
     *
     * @return 所有引擎的IP:Port集合
     */
    public Set<String> getAllEngineIpPorts() {
        return engineViews.keySet();
    }
}