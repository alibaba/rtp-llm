package org.flexlb.cache.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GlobalCacheIndex unit tests
 */
@ExtendWith(MockitoExtension.class)
class GlobalCacheIndexTest {

    private GlobalCacheIndex globalCacheIndex;

    @BeforeEach
    void setUp() {
        globalCacheIndex = new GlobalCacheIndex();
    }

    @Test
    void testEmptyInput() {
        // Test empty input cases
        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(
                Collections.emptyList(), Arrays.asList(1L, 2L, 3L));
        assertTrue(result.isEmpty(), "Empty engines should return empty result");

        result = globalCacheIndex.batchCalculatePrefixMatchLength(
                Arrays.asList("engine1", "engine2"), Collections.emptyList());
        assertTrue(result.isEmpty(), "Empty blocks should return empty result");

        result = globalCacheIndex.batchCalculatePrefixMatchLength(null, null);
        assertTrue(result.isEmpty(), "Null inputs should return empty result");
    }

    @Test
    void testBasicPrefixMatching() {
        // Set up test data
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

        // Verify results
        assertEquals(3, result.get("engine1").intValue(), "Engine1 should match all 3 blocks");
        assertEquals(1, result.get("engine2").intValue(), "Engine2 should match only first block");
        assertEquals(2, result.get("engine3").intValue(), "Engine3 should match first 2 blocks");
    }

    @Test
    void testNoMatchingEngines() {
        // Test case where no engine matches any block
        globalCacheIndex.addCacheBlock(1L, "other_engine");
        globalCacheIndex.addCacheBlock(2L, "another_engine");

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(0, result.get("engine1").intValue(), "Engine1 should have 0 prefix match");
        assertEquals(0, result.get("engine2").intValue(), "Engine2 should have 0 prefix match");
    }

    @Test
    void testAllEnginesMatchAllBlocks() {
        // Test case where all engines match all blocks
        List<String> engines = Arrays.asList("engine1", "engine2", "engine3");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        // Add all engines to all blocks
        for (Long block : blocks) {
            for (String engine : engines) {
                globalCacheIndex.addCacheBlock(block, engine);
            }
        }

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        for (String engine : engines) {
            assertEquals(4, result.get(engine).intValue(), "All engines should match all 4 blocks");
        }
    }

    @Test
    void testEarlyTermination() {
        // Test early termination case
        // Set up data so there are no candidate engines after the 2nd block
        globalCacheIndex.addCacheBlock(1L, "engine1");
        globalCacheIndex.addCacheBlock(1L, "engine2");
        globalCacheIndex.addCacheBlock(2L, "engine1");
        // block 3L and 4L have no engines

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(2, result.get("engine1").intValue(), "Engine1 should match 2 blocks before termination");
        assertEquals(1, result.get("engine2").intValue(), "Engine2 should match 1 block before termination");
    }

    @Test
    void testPartialMatching() {
        // Test complex partial matching scenario
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

        assertEquals(4, result.get("engine1").intValue(), "Engine1 should match all blocks");
        assertEquals(3, result.get("engine2").intValue(), "Engine2 should match first 3 blocks");
        assertEquals(2, result.get("engine3").intValue(), "Engine3 should match first 2 blocks");
        assertEquals(1, result.get("engine4").intValue(), "Engine4 should match first block only");
    }

    @Test
    void testSingleBlockSingleEngine() {
        // Test single block single engine case
        globalCacheIndex.addCacheBlock(1L, "engine1");

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(
                List.of("engine1"), List.of(1L));

        assertEquals(1, result.get("engine1").intValue(), "Single engine should match single block");
    }

    @Test
    void testNonExistentBlocks() {
        // Test non-existent blocks
        globalCacheIndex.addCacheBlock(1L, "engine1");

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 999L); // 999L does not exist

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(1, result.get("engine1").intValue(), "Engine1 should match first block only");
        assertEquals(0, result.get("engine2").intValue(), "Engine2 should have no matches");
    }

    @Test
    void testEngineRemoval() {
        // Test matching after engine removal
        globalCacheIndex.addCacheBlock(1L, "engine1");
        globalCacheIndex.addCacheBlock(1L, "engine2");
        globalCacheIndex.addCacheBlock(2L, "engine1");
        globalCacheIndex.addCacheBlock(2L, "engine2");

        // Remove engine2 from block 2L
        globalCacheIndex.removeCacheBlock("engine2", 2L);

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L);

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(2, result.get("engine1").intValue(), "Engine1 should still match both blocks");
        assertEquals(1, result.get("engine2").intValue(), "Engine2 should match only first block after removal");
    }

    @Test
    void testLargeScaleScenario() {
        // Test large-scale scenario to verify performance optimization
        List<String> engines = Arrays.asList("engine1", "engine2", "engine3", "engine4", "engine5");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L);

        // Set up progressive matching: engine1 matches all, engine2 matches first 9, and so on
        for (int i = 0; i < engines.size(); i++) {
            String engine = engines.get(i);
            for (int j = 0; j <= blocks.size() - 1 - i; j++) {
                globalCacheIndex.addCacheBlock(blocks.get(j), engine);
            }
        }

        Map<String, Integer> result = globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(10, result.get("engine1").intValue(), "Engine1 should match 10 blocks");
        assertEquals(9, result.get("engine2").intValue(), "Engine2 should match 9 blocks");
        assertEquals(8, result.get("engine3").intValue(), "Engine3 should match 8 blocks");
        assertEquals(7, result.get("engine4").intValue(), "Engine4 should match 7 blocks");
        assertEquals(6, result.get("engine5").intValue(), "Engine5 should match 6 blocks");
    }
}