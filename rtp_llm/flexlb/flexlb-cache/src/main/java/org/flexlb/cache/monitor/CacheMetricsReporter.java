package org.flexlb.cache.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

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
 * Cache metrics reporter
 * Manages registration and reporting of all cache-related monitoring metrics
 *
 * <p>Memory estimation based on the following JVM assumptions:</p>
 * <ul>
 *   <li>64-bit JVM</li>
 *   <li>Compressed Oops enabled (CompressedOops=true)</li>
 *   <li>8-byte memory alignment</li>
 *   <li>Object header structure: Mark Word(8 bytes) + Class Pointer(4 bytes) = 12 bytes</li>
 * </ul>
 *
 * <p>Key object memory footprint:</p>
 * <ul>
 *   <li>Long wrapper object: 24 bytes (object header 12 + long value 8 + padding 4)</li>
 *   <li>String object (15 chars): 72 bytes (String object 24 + char[] array 48)</li>
 *   <li>ConcurrentHashMap.Node: 32 bytes (object header 12 + 4 fields 16 + padding 4)</li>
 * </ul>
 *
 * @author FlexLB
 */
@Slf4j
@Component
public class CacheMetricsReporter {

    /**
     * Bytes occupied by Long type (header 12 + data 8 + padding 4)
     */
    private static final long LONG_OBJECT_BYTES = 24;

    @Autowired
    private FlexMonitor monitor;

    @PostConstruct
    public void init() {
        registerCacheMetrics();
        log.info("CacheMetricsReporter initialized and metrics registered");
    }

    /**
     * Register all cache-related monitoring metrics
     */
    private void registerCacheMetrics() {
        // Engine local cache metrics
        monitor.register(CACHE_ENGINE_LOCAL_COUNT, FlexMetricType.GAUGE);
        monitor.register(CACHE_ENGINE_LOCAL_BYTES, FlexMetricType.GAUGE);

        // Global cache metrics
        monitor.register(CACHE_GLOBAL_TOTAL_COUNT, FlexMetricType.GAUGE);
        monitor.register(CACHE_GLOBAL_BYTES, FlexMetricType.GAUGE);

        // Cache hit rate metrics
        monitor.register(CACHE_HIT_COUNT, FlexMetricType.GAUGE);
        monitor.register(CACHE_HIT_RATIO, FlexMetricType.GAUGE);
        monitor.register(CACHE_REQUEST_TOTAL, FlexMetricType.QPS);

        // Cache service response time metrics
        monitor.register(CACHE_FIND_MATCHING_ENGINES_RT, FlexMetricType.GAUGE);
        monitor.register(CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT, FlexMetricType.GAUGE);

        // Cache diff related metrics
        monitor.register(CACHE_DIFF_ADDED_BLOCKS_SIZE, FlexMetricType.GAUGE);
        monitor.register(CACHE_DIFF_REMOVED_BLOCKS_SIZE, FlexMetricType.GAUGE);

        // Engine view map size metrics
        monitor.register(CACHE_ENGINE_VIEWS_MAP_SIZE, FlexMetricType.GAUGE);
    }

    /**
     * Report local cache metrics for a single engine
     *
     * @param engineIp   Engine IP
     * @param role       Engine role
     * @param cacheCount Cache count
     */
    public void reportEngineLocalMetrics(String engineIp, String role, int cacheCount) {
        if (engineIp == null) {
            return;
        }

        // Calculate cache count and bytes
        long cacheBytes = calculateEngineCacheBytes(cacheCount);

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", engineIp,
                "role", role
        );

        monitor.report(CACHE_ENGINE_LOCAL_COUNT, tags, cacheCount);
        monitor.report(CACHE_ENGINE_LOCAL_BYTES, tags, cacheBytes);
    }

    /**
     * Report global cache metrics
     *
     * @param totalBlocks   Total global cache block count
     * @param totalMappings Total global cache mapping count
     */
    public void reportGlobalCacheMetrics(long totalBlocks, long totalMappings) {

        // Use statistics-based estimation
        long totalBytes = calculateGlobalCacheBytes(totalBlocks, totalMappings);

        monitor.report(CACHE_GLOBAL_TOTAL_COUNT, totalBlocks);
        monitor.report(CACHE_GLOBAL_BYTES, totalBytes);
    }

    /**
     * Report cache hit rate metrics
     *
     * @param roleType  Role type
     * @param engineIp  Engine IP
     * @param hitTokens Number of hit tokens
     * @param hitRatio  Hit percentage
     */
    public void reportCacheHitMetrics(RoleType roleType, String engineIp, long hitTokens, double hitRatio) {

        FlexMetricTags baseTags = FlexMetricTags.of(
                "role", roleType.name(),
                "engineIp", engineIp
        );

        // Report hit token count and hit percentage
        monitor.report(CACHE_HIT_COUNT, baseTags, hitTokens);
        monitor.report(CACHE_HIT_RATIO, baseTags, hitRatio);
        monitor.report(CACHE_REQUEST_TOTAL, baseTags, 1.0);
    }

    /**
     * Align memory to 8-byte boundary (64-bit JVM standard alignment)
     *
     * @param size Original size
     * @return Aligned size
     */
    private static long alignTo8Bytes(long size) {
        return (size + 7) & ~7L;
    }

    /**
     * Calculate String object memory footprint (based on character count)
     * Total overhead of String object + char array
     *
     * @return Memory footprint in bytes of String object
     */
    private static long getStringObjectSize() {
        // String object: object header(12) + fields(8) + padding(4) = 24 bytes
        long stringObjectSize = 24L;

        // char[] array: array header(16) + data(charCount*2) + alignment
        long charArraySize = alignTo8Bytes(16L + 15 * 2L);

        return stringObjectSize + charArraySize;
    }

    /**
     * Calculate memory footprint of engine cache
     * Precise estimation: KeySetView<Long, Boolean>
     * Based on 64-bit JVM + Compressed Oops memory model
     *
     * @param cacheCount Cache count
     * @return Estimated bytes
     */
    public long calculateEngineCacheBytes(int cacheCount) {
        if (cacheCount <= 0) {
            return 0L;
        }

        // ConcurrentHashMap base structure overhead (object header 12 + fields 44 = 56 bytes aligned)
        long baseStructureSize = 56L;

        // KeySetView instance overhead (object header 12 + reference field 8 + padding 4 = 24 bytes)
        long keySetViewOverhead = 24L;

        // table array overhead (capacity calculated based on 75% load factor)
        int initialCapacity = Math.max(16, (cacheCount * 4 + 3) / 3);
        long tableArraySize = alignTo8Bytes(16L + initialCapacity * 4L); // Reference is 4 bytes with compressed oops

        // Each ConcurrentHashMap.Node overhead (object header 12 + 4 fields 16 + padding 4 = 32 bytes)
        long nodeSize = 32L;

        // Boolean.TRUE not counted

        return baseStructureSize + keySetViewOverhead + tableArraySize +
                cacheCount * (nodeSize + LONG_OBJECT_BYTES);
    }

    /**
     * Calculate memory footprint of global cache
     * Precise estimation: ConcurrentHashMap<Long, Set<String>>
     * Based on 64-bit JVM + Compressed Oops complete memory model
     *
     * @param totalBlocks   Total cache block count
     * @param totalMappings Total mapping count
     * @return Estimated bytes
     */
    public long calculateGlobalCacheBytes(long totalBlocks, long totalMappings) {
        if (totalBlocks <= 0) {
            return 0L;
        }

        // ConcurrentHashMap base structure overhead (object header 12 + fields 44 = 56 bytes aligned)
        long chmBaseSize = 56L;

        // table array overhead (capacity calculated based on 75% load factor, reference is 4 bytes with compressed oops)
        int initialCapacity = Math.max(16, (int) Math.ceil(totalBlocks / 0.75));
        long tableArraySize = alignTo8Bytes(16L + initialCapacity * 4L);

        // Each ConcurrentHashMap.Node overhead (object header 12 + 4 fields 16 + padding 4 = 32 bytes)
        long nodeSize = 32L;

        // Each Set<String> overhead (ConcurrentHashMap.KeySetView)
        long setWrapperSize = 24L; // Set wrapper object overhead
        long setStructureSize = 40L; // Internal ConcurrentHashMap structure overhead

        // Each String value overhead (engine IP:Port)
        // Assume average IP:Port length is 15 characters (e.g., "192.168.1.1:8080")
        long stringSize = getStringObjectSize();

        // Total bytes = base structure + array + all nodes + all keys + all value sets + all strings
        return chmBaseSize + tableArraySize +
                totalBlocks * (nodeSize + LONG_OBJECT_BYTES) +
                totalBlocks * (setWrapperSize + setStructureSize) +
                totalMappings * stringSize;
    }

    /**
     * Report response time for finding matching engines
     *
     * @param roleType  Role type
     * @param startTime Start time in microseconds
     * @param success   Whether successful
     */
    public void reportFindMatchingEnginesRT(RoleType roleType, long startTime, String success) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", roleType.name(),
                "success", success
        );

        monitor.report(CACHE_FIND_MATCHING_ENGINES_RT, tags, ((double) System.nanoTime() / 1000) - startTime);
    }

    /**
     * Report response time for updating engine cache
     *
     * @param engineIpPort Engine IP:Port
     * @param role         Engine role
     * @param startTime    Start time in microseconds
     * @param success      Whether successful
     */
    public void reportUpdateEngineBlockCacheRT(String engineIpPort, String role, long startTime, String success) {
        FlexMetricTags tags = FlexMetricTags.of(
                "engineIpPort", engineIpPort,
                "role", role,
                "success", success
        );

        monitor.report(CACHE_UPDATE_ENGINE_BLOCK_CACHE_RT, tags, ((double) System.nanoTime() / 1000) - startTime);
    }

    /**
     * Report cache diff calculation metrics
     *
     * @param engineIp          Engine IP
     * @param role              Role
     * @param addedBlocksSize   Number of added blocks
     * @param removedBlocksSize Number of removed blocks
     */
    public void reportCacheDiffMetrics(String engineIp, String role, int addedBlocksSize, int removedBlocksSize) {
        if (engineIp == null) {
            return;
        }

        FlexMetricTags tags = FlexMetricTags.of(
                "engineIp", engineIp,
                "role", role != null ? role : "unknown"
        );

        monitor.report(CACHE_DIFF_ADDED_BLOCKS_SIZE, tags, addedBlocksSize);
        monitor.report(CACHE_DIFF_REMOVED_BLOCKS_SIZE, tags, removedBlocksSize);
    }

    /**
     * Report engine view map size
     *
     * @param mapSize Engine view map size (number of current engines)
     */
    public void reportEngineViewsMapSize(int mapSize) {
        monitor.report(CACHE_ENGINE_VIEWS_MAP_SIZE, mapSize);
    }
}