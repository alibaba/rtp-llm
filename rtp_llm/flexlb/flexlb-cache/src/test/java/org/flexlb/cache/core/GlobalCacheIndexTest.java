package org.flexlb.cache.core;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.MockitoAnnotations;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * GlobalCacheIndex 单元测试
 */
public class GlobalCacheIndexTest {

    private GlobalCacheIndex globalCacheIndex;

    @Before
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        globalCacheIndex = new GlobalCacheIndex();
    }

    @Test
    public void testEmptyInput() {
        // 测试空输入情况
        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(
                Collections.emptyList(), Arrays.asList(1L, 2L, 3L));
        assertTrue("Empty engines should return empty result", result.isEmpty());

        result = globalCacheIndex.batchCalculatePrefixMatchLength(
                Arrays.asList("engine1", "engine2"), Collections.emptyList());
        assertTrue("Empty blocks should return empty result", result.isEmpty());

        result = globalCacheIndex.batchCalculatePrefixMatchLength(null, null);
        assertTrue("Null inputs should return empty result", result.isEmpty());
    }

    @Test
    public void testBasicPrefixMatching() {
        // 设置测试数据
        // Block 1L: engine1, engine2, engine3
        // Block 2L: engine1, engine3  
        // Block 3L: engine1
        globalCacheIndex.addCacheBlock(1L, "engine1");
        globalCacheIndex.addCacheBlock(1L, "engine2");
        globalCacheIndex.addCacheBlock(1L, "engine3");
        globalCacheIndex.addCacheBlock(2L, "engine1");
        globalCacheIndex.addCacheBlock(2L, "engine3");
        globalCacheIndex.addCacheBlock(3L, "engine1");

        List<String> engines = Arrays.asList("engine1", "engine2", "engine3");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        // 验证结果
        assertEquals("Engine1 should match all 3 blocks", 3, result.get("engine1").intValue());
        assertEquals("Engine2 should match only first block", 1, result.get("engine2").intValue());
        assertEquals("Engine3 should match first 2 blocks", 2, result.get("engine3").intValue());
    }

    @Test
    public void testNoMatchingEngines() {
        // 测试没有引擎匹配任何block的情况
        globalCacheIndex.addCacheBlock(1L, "other_engine");
        globalCacheIndex.addCacheBlock(2L, "another_engine");

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals("Engine1 should have 0 prefix match", 0, result.get("engine1").intValue());
        assertEquals("Engine2 should have 0 prefix match", 0, result.get("engine2").intValue());
    }

    @Test
    public void testAllEnginesMatchAllBlocks() {
        // 测试所有引擎都匹配所有block的情况
        List<String> engines = Arrays.asList("engine1", "engine2", "engine3");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        // 为所有block添加所有引擎
        for (Long block : blocks) {
            for (String engine : engines) {
                globalCacheIndex.addCacheBlock(block, engine);
            }
        }

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        for (String engine : engines) {
            assertEquals("All engines should match all 4 blocks", 4, result.get(engine).intValue());
        }
    }

    @Test
    public void testEarlyTermination() {
        // 测试提前终止的情况
        // 设置数据使得在第2个block后没有候选引擎了
        globalCacheIndex.addCacheBlock(1L, "engine1");
        globalCacheIndex.addCacheBlock(1L, "engine2");
        globalCacheIndex.addCacheBlock(2L, "engine1");
        // block 3L 和 4L 没有任何引擎
        
        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals("Engine1 should match 2 blocks before termination", 2, result.get("engine1").intValue());
        assertEquals("Engine2 should match 1 block before termination", 1, result.get("engine2").intValue());
    }

    @Test
    public void testPartialMatching() {
        // 测试部分匹配的复杂场景
        globalCacheIndex.addCacheBlock(1L, "engine1");
        globalCacheIndex.addCacheBlock(1L, "engine2");
        globalCacheIndex.addCacheBlock(1L, "engine3");
        globalCacheIndex.addCacheBlock(1L, "engine4");

        globalCacheIndex.addCacheBlock(2L, "engine1");
        globalCacheIndex.addCacheBlock(2L, "engine2");
        globalCacheIndex.addCacheBlock(2L, "engine3");

        globalCacheIndex.addCacheBlock(3L, "engine1");
        globalCacheIndex.addCacheBlock(3L, "engine2");

        globalCacheIndex.addCacheBlock(4L, "engine1");

        List<String> engines = Arrays.asList("engine1", "engine2", "engine3", "engine4");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals("Engine1 should match all blocks", 4, result.get("engine1").intValue());
        assertEquals("Engine2 should match first 3 blocks", 3, result.get("engine2").intValue());
        assertEquals("Engine3 should match first 2 blocks", 2, result.get("engine3").intValue());
        assertEquals("Engine4 should match first block only", 1, result.get("engine4").intValue());
    }

    @Test
    public void testSingleBlockSingleEngine() {
        // 测试单个block单个引擎的情况
        globalCacheIndex.addCacheBlock(1L, "engine1");

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(
            Collections.singletonList("engine1"), Collections.singletonList(1L));

        assertEquals("Single engine should match single block", 1, result.get("engine1").intValue());
    }

    @Test
    public void testNonExistentBlocks() {
        // 测试不存在的block
        globalCacheIndex.addCacheBlock(1L, "engine1");

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 999L); // 999L 不存在

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals("Engine1 should match first block only", 1, result.get("engine1").intValue());
        assertEquals("Engine2 should have no matches", 0, result.get("engine2").intValue());
    }

    @Test
    public void testEngineRemoval() {
        // 测试引擎移除后的匹配情况
        globalCacheIndex.addCacheBlock(1L, "engine1");
        globalCacheIndex.addCacheBlock(1L, "engine2");
        globalCacheIndex.addCacheBlock(2L, "engine1");
        globalCacheIndex.addCacheBlock(2L, "engine2");

        // 移除engine2从block 2L
        globalCacheIndex.removeCacheBlock("engine2", 2L);

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals("Engine1 should still match both blocks", 2, result.get("engine1").intValue());
        assertEquals("Engine2 should match only first block after removal", 1, result.get("engine2").intValue());
    }

    @Test
    public void testLargeScaleScenario() {
        // 测试大规模场景以验证性能优化
        List<String> engines = Arrays.asList("engine1", "engine2", "engine3", "engine4", "engine5");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L);

        // 设置渐进式匹配：engine1匹配所有，engine2匹配前9个，以此类推
        for (int i = 0; i < engines.size(); i++) {
            String engine = engines.get(i);
            for (int j = 0; j <= blocks.size() - 1 - i; j++) {
                globalCacheIndex.addCacheBlock(blocks.get(j), engine);
            }
        }

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals("Engine1 should match 10 blocks", 10, result.get("engine1").intValue());
        assertEquals("Engine2 should match 9 blocks", 9, result.get("engine2").intValue());
        assertEquals("Engine3 should match 8 blocks", 8, result.get("engine3").intValue());
        assertEquals("Engine4 should match 7 blocks", 7, result.get("engine4").intValue());
        assertEquals("Engine5 should match 6 blocks", 6, result.get("engine5").intValue());
    }
}