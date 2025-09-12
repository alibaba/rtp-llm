package org.flexlb.cache.core;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;

import com.google.common.collect.Sets;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 全局缓存索引 (大Hash表)
 * 管理 block_hash_id -> Set<EngineIP:EnginePort> 的映射关系
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class GlobalCacheIndex {

    /**
     * 核心存储结构: block_hash_id -> Set<engine_ip:engine_port>
     */
    private final ConcurrentHashMap<Long, Set<String>> blockToEnginesMap = new ConcurrentHashMap<>();

    /**
     * 读写锁保证数据一致性
     */
    private final ReentrantLock lock = new ReentrantLock();
    
    /**
     * 统计信息
     */
    private final LongAdder totalBlocks = new LongAdder();
    private final LongAdder totalMappings = new LongAdder();

    /**
     * 添加缓存块到指定引擎
     *
     * @param blockCacheKey 缓存块哈希值
     * @param engineIpPort  引擎IP:Port
     */
    public void addCacheBlock(Long blockCacheKey, String engineIpPort) {
        if (blockCacheKey == null || engineIpPort == null) {
            log.warn("Invalid parameters: blockCacheKey={}, engineIpPort={}", blockCacheKey, engineIpPort);
            return;
        }
        
        lock.lock();
        try {
            Set<String> engines = blockToEnginesMap.computeIfAbsent(blockCacheKey, k -> {
                totalBlocks.increment();
                return Sets.newConcurrentHashSet();
            });

            boolean added = engines.add(engineIpPort);
            if (added) {
                totalMappings.increment();
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * 从指定引擎移除缓存块
     *
     * @param engineIp      引擎IP
     * @param blockCacheKey 缓存块哈希值
     */
    public void removeCacheBlock(String engineIp, Long blockCacheKey) {
        if (blockCacheKey == null || engineIp == null) {
            return;
        }
        
        lock.lock();
        try {
            Set<String> engines = blockToEnginesMap.get(blockCacheKey);
            if (engines == null) {
                return;
            }

            boolean removed = engines.remove(engineIp);
            if (removed) {
                totalMappings.decrement();

                // 如果没有引擎拥有该缓存块，则移除整个条目
                if (engines.isEmpty()) {
                    blockToEnginesMap.remove(blockCacheKey);
                    totalBlocks.decrement();
                }
            }
        } finally {
            lock.unlock();
        }
    }

    /**
     * 移除某个引擎
     *
     * @param engineIp 引擎 IP
     */
    public void removeAllCacheBlockOfEngine(String engineIp) {
        if (engineIp == null) {
            return;
        }

        lock.lock();
        try {
            blockToEnginesMap.forEach((blockCacheKey, engines) -> {
                boolean removed = engines.remove(engineIp);
                if (removed) {
                    totalMappings.decrement();

                    // 如果没有引擎拥有该缓存块，则移除整个条目
                    if (engines.isEmpty()) {
                        blockToEnginesMap.remove(blockCacheKey);
                        totalBlocks.decrement();
                    }
                }
            });
        } finally {
            lock.unlock();
        }
    }

    /**
     * 基于前缀匹配计算引擎的前缀匹配长度
     *
     * @param engineIpPorts  引擎IP:Port列表
     * @param blockCacheKeys 按顺序的缓存块哈希值列表
     * @return Map<EngineIP:EnginePort, PrefixMatchLength>
     */
    public Map<String, Integer> batchCalculatePrefixMatchLength(List<String> engineIpPorts,
                                                                List<Long> blockCacheKeys) {

        if (isEmpty(engineIpPorts) || isEmpty(blockCacheKeys)) {
            return Collections.emptyMap();
        }
        return calculatePrefixMatchLength(engineIpPorts, blockCacheKeys);
    }

    /**
     * 前缀匹配计算
     *
     * @param engineIpPorts  引擎IP:Port列表
     * @param blockCacheKeys 按顺序的缓存块哈希值列表
     * @return Map<EngineIP:EnginePort, PrefixMatchLength>
     */
    private Map<String, Integer> calculatePrefixMatchLength(List<String> engineIpPorts,
                                                            List<Long> blockCacheKeys) {

        Map<String, Integer> result = new HashMap<>(engineIpPorts.size());

        // 初始化所有引擎为候选，还未确定前缀长度的引擎集合
        Set<String> candidateEngines = Sets.newHashSet(engineIpPorts);

        // 遍历每个block，逐步过滤候选引擎
        for (int i = 0; i < blockCacheKeys.size(); i++) {
            Long blockCacheKey = blockCacheKeys.get(i);
            Set<String> blockOwners = getEnginesForBlock(blockCacheKey);

            // 已经确认匹配长度的引擎集合
            Set<String> confirmedEngines = Sets.newHashSet();
            // 过滤候选引擎：只保留在当前block中存在的引擎
            for (String candidateEngine : candidateEngines) {
                if (blockOwners.isEmpty() || !blockOwners.contains(candidateEngine)) {
                    // 此引擎在当前block中不存在，前缀匹配中断
                    result.put(candidateEngine, i);
                    confirmedEngines.add(candidateEngine);
                }
            }

            // 从候选集中移除已确定前缀长度的引擎
            candidateEngines.removeAll(confirmedEngines);

            // 如果没有候选引擎了，可以提前结束
            if (candidateEngines.isEmpty()) {
                break;
            }
        }

        // 处理剩余的候选引擎（它们匹配了所有block）
        for (String remainingEngine : candidateEngines) {
            result.put(remainingEngine, blockCacheKeys.size());
        }

        return result;
    }

    /**
     * 检查集合是否为空
     */
    private boolean isEmpty(List<?> list) {
        return list == null || list.isEmpty();
    }

    /**
     * 获取指定缓存块的引擎集合
     */
    private Set<String> getEnginesForBlock(Long blockCacheKey) {
        if (blockCacheKey == null) {
            return Collections.emptySet();
        }
        Set<String> engines = blockToEnginesMap.get(blockCacheKey);
        return engines != null ? engines : Collections.emptySet();
    }

    /**
     * 清空所有数据
     */
    public void clear() {

        blockToEnginesMap.clear();
        totalBlocks.reset();
        totalMappings.reset();
        log.info("Cleared global cache index");
    }

    public long totalBlocks() {
        return totalBlocks.sum();
    }

    public long totalMappings() {
        return totalMappings.sum();
    }
}