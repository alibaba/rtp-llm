package org.flexlb.cache.monitor;

import javax.annotation.PostConstruct;

import com.taobao.kmonitor.ImmutableMetricTags;
import com.taobao.kmonitor.KMonitor;
import com.taobao.kmonitor.MetricType;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.RoleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import static org.flexlb.constant.MetricConstant.CACHE_DIFF_ADDED_BLOCKS_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_DIFF_REMOVED_BLOCKS_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_BYTES;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_LOCAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_ENGINE_VIEWS_MAP_SIZE;
import static org.flexlb.constant.MetricConstant.CACHE_FIND_MATCHING_ENGINES_RT;
import static org.flexlb.constant.MetricConstant.CACHE_GLOBAL_BYTES;
import static org.flexlb.constant.MetricConstant.CACHE_GLOBAL_TOTAL_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL;
import static org.flexlb.constant.MetricConstant.CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT;

/**
 * 缓存监控指标上报器
 * 统一管理所有缓存相关的监控指标注册和上报逻辑
 *
 * <p>内存估算基于以下JVM参数假设：</p>
 * <ul>
 *   <li>64位JVM</li>
 *   <li>压缩指针启用 (CompressedOops=true)</li>
 *   <li>8字节内存对齐</li>
 *   <li>对象头结构: Mark Word(8字节) + Class Pointer(4字节) = 12字节</li>
 * </ul>
 *
 * <p>关键对象内存占用：</p>
 * <ul>
 *   <li>Long包装对象: 24字节 (对象头12 + long值8 + 对齐4)</li>
 *   <li>String对象(15字符): 72字节 (String对象24 + char[]数组48)</li>
 *   <li>ConcurrentHashMap.Node: 32字节 (对象头12 + 4个字段16 + 对齐4)</li>
 * </ul>
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class CacheMetricsReporter {

    /**
     * Long 类型占用的 bytes（header 12 + data 8 + padding 4）
     */
    private static final long LONG_OBJECT_BYTES = 24;

    @Autowired
    private KMonitor kMonitor;

    @PostConstruct
    public void init() {
        registerCacheMetrics();
        log.info("CacheMetricsReporter initialized and metrics registered");
    }

    /**
     * 注册所有缓存相关的监控指标
     */
    private void registerCacheMetrics() {
        // 引擎本地缓存指标
        kMonitor.register(CACHE_ENGINE_LOCAL_COUNT, MetricType.GAUGE);
        kMonitor.register(CACHE_ENGINE_LOCAL_BYTES, MetricType.GAUGE);

        // 全局缓存指标
        kMonitor.register(CACHE_GLOBAL_TOTAL_COUNT, MetricType.GAUGE);
        kMonitor.register(CACHE_GLOBAL_BYTES, MetricType.GAUGE);

        // 缓存命中率指标
        kMonitor.register(CACHE_HIT_COUNT, MetricType.GAUGE);
        kMonitor.register(CACHE_HIT_RATIO, MetricType.GAUGE);
        kMonitor.register(CACHE_REQUEST_TOTAL, MetricType.QPS);

        // 缓存服务响应时间指标
        kMonitor.register(CACHE_FIND_MATCHING_ENGINES_RT, MetricType.GAUGE);
        kMonitor.register(CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT, MetricType.GAUGE);
        
        // 缓存diff相关指标
        kMonitor.register(CACHE_DIFF_ADDED_BLOCKS_SIZE, MetricType.GAUGE);
        kMonitor.register(CACHE_DIFF_REMOVED_BLOCKS_SIZE, MetricType.GAUGE);
        
        // 引擎视图Map大小指标
        kMonitor.register(CACHE_ENGINE_VIEWS_MAP_SIZE, MetricType.GAUGE);
    }


    /**
     * 上报单个引擎的本地缓存指标
     *
     * @param engineIp   引擎IP
     * @param role       引擎角色
     * @param cacheCount 缓存数量
     */
    public void reportEngineLocalMetrics(String engineIp, String role, int cacheCount) {
        if (engineIp == null) {
            return;
        }

        // 计算缓存数量和字节数
        long cacheBytes = calculateEngineCacheBytes(cacheCount);

        ImmutableMetricTags tags = new ImmutableMetricTags(
                "engineIp", engineIp,
                "role", role
        );

        kMonitor.report(CACHE_ENGINE_LOCAL_COUNT, tags, cacheCount);
        kMonitor.report(CACHE_ENGINE_LOCAL_BYTES, tags, cacheBytes);
    }

    /**
     * 上报全局缓存指标
     *
     * @param totalBlocks   全局缓存块总数
     * @param totalMappings 全局缓存映射关系总数
     */
    public void reportGlobalCacheMetrics(long totalBlocks, long totalMappings) {

        // 使用基于统计数字的估算
        long totalBytes = calculateGlobalCacheBytes(totalBlocks, totalMappings);

        kMonitor.report(CACHE_GLOBAL_TOTAL_COUNT, totalBlocks);
        kMonitor.report(CACHE_GLOBAL_BYTES, totalBytes);
    }

    /**
     * 上报缓存命中率指标
     *
     * @param modelName 模型名称
     * @param roleType  角色类型
     * @param engineIp  引擎IP
     * @param hitTokens 命中的tokens数量
     * @param hitRatio  命中百分比
     */
    public void reportCacheHitMetrics(String modelName, RoleType roleType, String engineIp, long hitTokens, double hitRatio) {

        ImmutableMetricTags baseTags = new ImmutableMetricTags(
                "model", modelName,
                "role", roleType.name(),
                "engineIp", engineIp
        );

        // 上报命中tokens数量和命中百分比
        kMonitor.report(CACHE_HIT_COUNT, baseTags, hitTokens);
        kMonitor.report(CACHE_HIT_RATIO, baseTags, hitRatio);
        kMonitor.report(CACHE_REQUEST_TOTAL, baseTags, 1.0);
    }

    /**
     * 内存对齐到8字节边界 (64位JVM标准对齐)
     *
     * @param size 原始大小
     * @return 对齐后的大小
     */
    private static long alignTo8Bytes(long size) {
        return (size + 7) & ~7L;
    }

    /**
     * 计算String对象的内存占用 (基于字符数)
     * String对象 + char数组的总开销
     *
     * @return String对象的内存占用字节数
     */
    private static long getStringObjectSize() {
        // String对象: 对象头(12) + 字段(8) + 对齐(4) = 24字节
        long stringObjectSize = 24L;

        // char[]数组: 数组头(16) + 数据(charCount*2) + 对齐
        long charArraySize = alignTo8Bytes(16L + 15 * 2L);

        return stringObjectSize + charArraySize;
    }

    /**
     * 计算引擎缓存占用的字节数
     * 精确估算：KeySetView<Long, Boolean>
     * 基于64位JVM + 压缩指针的内存模型
     *
     * @param cacheCount 缓存数量
     * @return 估算的字节数
     */
    public long calculateEngineCacheBytes(int cacheCount) {
        if (cacheCount <= 0) {
            return 0L;
        }

        // ConcurrentHashMap 基础结构开销 (对象头12 + 字段44 = 56字节对齐)
        long baseStructureSize = 56L;

        // KeySetView 实例开销 (对象头12 + 引用字段8 + 对齐4 = 24字节)
        long keySetViewOverhead = 24L;

        // table数组开销 (基于75%负载因子计算容量)
        int initialCapacity = Math.max(16, (cacheCount * 4 + 3) / 3);
        long tableArraySize = alignTo8Bytes(16L + initialCapacity * 4L); // 压缩指针下引用4字节

        // 每个ConcurrentHashMap.Node的开销 (对象头12 + 4个字段16 + 对齐4 = 32字节)
        long nodeSize = 32L;

        // Boolean.TRUE 不计入

        return baseStructureSize + keySetViewOverhead + tableArraySize +
                cacheCount * (nodeSize + LONG_OBJECT_BYTES);
    }

    /**
     * 计算全局缓存占用的字节数
     * 精确估算：ConcurrentHashMap<Long, Set<String>>
     * 基于64位JVM + 压缩指针的完整内存模型
     *
     * @param totalBlocks   缓存块总数
     * @param totalMappings 映射关系总数
     * @return 估算的字节数
     */
    public long calculateGlobalCacheBytes(long totalBlocks, long totalMappings) {
        if (totalBlocks <= 0) {
            return 0L;
        }

        // ConcurrentHashMap 基础结构开销 (对象头12 + 字段44 = 56字节对齐)
        long chmBaseSize = 56L;

        // table数组开销 (基于75%负载因子计算容量，压缩指针下引用4字节)
        int initialCapacity = Math.max(16, (int) Math.ceil(totalBlocks / 0.75));
        long tableArraySize = alignTo8Bytes(16L + initialCapacity * 4L);

        // 每个ConcurrentHashMap.Node的开销 (对象头12 + 4个字段16 + 对齐4 = 32字节)
        long nodeSize = 32L;

        // 每个Set<String>的开销 (ConcurrentHashMap.KeySetView)
        long setWrapperSize = 24L; // Set wrapper对象开销
        long setStructureSize = 40L; // 内部ConcurrentHashMap结构开销

        // 每个String value的开销 (引擎IP:Port)
        // 假设平均IP:Port长度为15字符 (e.g., "192.168.1.1:8080")
        long stringSize = getStringObjectSize();

        // 总字节数 = 基础结构 + 数组 + 所有节点 + 所有键 + 所有值集合 + 所有字符串
        return chmBaseSize + tableArraySize +
                totalBlocks * (nodeSize + LONG_OBJECT_BYTES) +
                totalBlocks * (setWrapperSize + setStructureSize) +
                totalMappings * stringSize;
    }

    /**
     * 上报查找匹配引擎的响应时间
     *
     * @param modelName 模型名称
     * @param roleType  角色类型
     * @param startTime 开启时间（毫秒）
     * @param success   是否成功
     */
    public void reportFindMatchingEnginesRT(String modelName, RoleType roleType, long startTime, String success) {
        ImmutableMetricTags tags = new ImmutableMetricTags(
                "model", modelName,
                "role", roleType.name(),
                "success", success
        );

        kMonitor.report(CACHE_FIND_MATCHING_ENGINES_RT, tags, System.currentTimeMillis() - startTime);
    }

    /**
     * 上报更Engine缓存的响应时间
     *
     * @param engineIpPort 引擎IP:Port
     * @param role         引擎角色
     * @param startTime    开启时间（毫秒）
     * @param success      是否成功
     */
    public void reportUpdateEngineBlockCacheRT(String engineIpPort, String role, long startTime, String success) {
        ImmutableMetricTags tags = new ImmutableMetricTags(
                "engineIpPort", engineIpPort,
                "role", role,
                "success", success
        );

        kMonitor.report(CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT, tags, System.currentTimeMillis() - startTime);
    }

    /**
     * 上报缓存Diff计算的指标
     *
     * @param engineIp          引擎IP
     * @param role              角色
     * @param addedBlocksSize   新增块数量
     * @param removedBlocksSize 移除块数量
     */
    public void reportCacheDiffMetrics(String engineIp, String role, int addedBlocksSize, int removedBlocksSize) {
        if (engineIp == null) {
            return;
        }

        ImmutableMetricTags tags = new ImmutableMetricTags(
                "engineIp", engineIp,
                "role", role != null ? role : "unknown"
        );

        kMonitor.report(CACHE_DIFF_ADDED_BLOCKS_SIZE, tags, addedBlocksSize);
        kMonitor.report(CACHE_DIFF_REMOVED_BLOCKS_SIZE, tags, removedBlocksSize);
    }

    /**
     * 上报引擎视图Map的大小
     *
     * @param mapSize 引擎视图Map的大小（即当前有多少个引擎）
     */
    public void reportEngineViewsMapSize(int mapSize) {
        kMonitor.report(CACHE_ENGINE_VIEWS_MAP_SIZE, mapSize);
    }
}