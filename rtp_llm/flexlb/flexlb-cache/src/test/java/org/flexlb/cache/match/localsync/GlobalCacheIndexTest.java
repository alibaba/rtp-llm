package org.flexlb.cache.match.localsync;

import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * GlobalCacheIndex unit tests
 */
@ExtendWith(MockitoExtension.class)
class GlobalCacheIndexTest {

    private GlobalCacheIndex globalCacheIndex;

    @Test
    void retirementPreservesSharedOwnership() {
        EngineLocalView local = new EngineLocalView();
        for (long key = 0; key < 1025; key++) {
            local.addOrUpdateCacheBlock("retired", key);
            globalCacheIndex.addCacheBlock(key, "retired", null);
            if (key % 2 == 0) globalCacheIndex.addCacheBlock(key, "survivor", null);
        }
        globalCacheIndex.addCacheBlock(2000L, "survivor", null);
        Set<Long> detached = local.getEngineCacheBlocks("retired");
        assertEquals(1025, detached.size());
        globalCacheIndex.removeAllCacheBlockOfEngine("retired", detached, null);
        local.removeAllCacheBlockOfEngine("retired");
        assertEquals(0, local.size("retired"));
        assertEquals(514L, globalCacheIndex.totalBlocks());
        assertEquals(514L, globalCacheIndex.totalMappings());
        assertEquals(Map.of("survivor", 1), globalCacheIndex.batchCalculatePrefixMatchLength(
                List.of("retired", "survivor"), List.of(0L, 1L)));
        globalCacheIndex.removeAllCacheBlockOfEngine("retired", detached, null);
        assertEquals(514L, globalCacheIndex.totalMappings());
    }

    @BeforeEach
    void setUp() {
        globalCacheIndex = new GlobalCacheIndex();
    }

    @Test
    void uncontendedMutationsDoNotRecordLockTimings() {
        var trace = new EndpointTrace();
        var pollTrace = trace.beginPoll("CACHE", 1L);
        globalCacheIndex.addCacheBlock(1L, "worker", pollTrace);
        globalCacheIndex.removeCacheBlock("worker", 1L, pollTrace);

        var snapshot = trace.snapshot().cachePoll();
        assertEquals(0L, snapshot.indexWait().count());
        assertEquals("LEASE_ACQUIRED", snapshot.stage());
    }

    @Test
    void cleanupRecordsActualLockSpanAndScanCountsWithoutChangingOtherOwners() {
        globalCacheIndex.addCacheBlock(1L, "retiring", null);
        globalCacheIndex.addCacheBlock(1L, "survivor", null);
        globalCacheIndex.addCacheBlock(2L, "retiring", null);
        var history = new EndpointTrace();
        globalCacheIndex.removeAllCacheBlockOfEngine("retiring", Set.of(1L, 2L), history);
        var snapshot = history.snapshot();
        assertEquals("CACHE_INDEX_LOCK_RELEASED", snapshot.phase());
        assertEquals(2L, snapshot.scannedKeys());
        assertEquals(2L, snapshot.removedMappings());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    @Timeout(10)
    @SuppressWarnings("unchecked")
    void waitingCachePollRecordsContentionUntilGlobalLockIsAcquired(boolean remove) throws Exception {
        String retiring = "10.43.7.113:28200";
        globalCacheIndex.addCacheBlock(1L, retiring, null);
        globalCacheIndex.addCacheBlock(2L, "worker-B", null);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var mapField = GlobalCacheIndex.class.getDeclaredField("blockToEnginesMap");
        mapField.setAccessible(true);
        var map = (Map<Long, Set<String>>) mapField.get(globalCacheIndex);
        var blockingOwners = new java.util.HashSet<String>() {
            @Override
            public boolean remove(Object value) {
                entered.countDown();
                try {
                    if (!release.await(5, TimeUnit.SECONDS)) {
                        throw new AssertionError("cleanup was not released");
                    }
                } catch (InterruptedException interruption) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(interruption);
                }
                return super.remove(value);
            }
        };
        blockingOwners.add(retiring);
        map.put(1L, blockingOwners);
        var cleanupHistory = new EndpointTrace();
        var waiterHistory = new EndpointTrace();
        var poll = waiterHistory.beginPoll("CACHE", 2L);
        var failure = new AtomicReference<Throwable>();
        Thread cleaning = new Thread(() -> {
            try {
                globalCacheIndex.removeAllCacheBlockOfEngine(retiring, Set.of(1L), cleanupHistory);
            } catch (Throwable thrown) {
                failure.set(thrown);
            }
        });
        Thread waiting = new Thread(() -> {
            try {
                if (remove) {
                    globalCacheIndex.removeCacheBlock("worker-B", 2L, poll);
                } else {
                    globalCacheIndex.addCacheBlock(2L, "worker-B", poll);
                }
            } catch (Throwable thrown) {
                failure.set(thrown);
            }
        });
        try {
            cleaning.start();
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            waiting.start();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while (waiting.getState() != Thread.State.WAITING && System.nanoTime() < deadline) {
                Thread.onSpinWait();
            }
            assertEquals(Thread.State.WAITING, waiting.getState());
            var snapshot = waiterHistory.snapshot();
            var indexWait = snapshot.cachePoll().indexWait();
            assertEquals("CACHE_INDEX_LOCK_ACQUIRED", cleanupHistory.snapshot().phase());
            assertEquals(1L, indexWait.count());
            assertTrue(indexWait.ageUs() > 0L);
        } finally {
            release.countDown();
            cleaning.join(3000L);
            waiting.join(3000L);
            poll.record(Stage.LEASE_RELEASED);
        }
        assertFalse(cleaning.isAlive());
        assertFalse(waiting.isAlive());
        assertEquals(null, failure.get());
        var waited = waiterHistory.snapshot().cachePoll().indexWait();
        assertEquals(0L, waited.ageUs());
        assertTrue(waited.totalWaitUs() > 0L);
        assertEquals(remove ? Map.of() : Map.of("worker-B", 1), globalCacheIndex.batchCalculatePrefixMatchLength(
                List.of("worker-B"), List.of(2L)));
    }

    @Test
    void countsFollowSharedBlockOwnershipAndClear() {
        globalCacheIndex.addCacheBlock(1L, "engine1", null);
        globalCacheIndex.addCacheBlock(1L, "engine1", null);
        globalCacheIndex.addCacheBlock(1L, "engine2", null);
        globalCacheIndex.addCacheBlock(2L, "engine1", null);
        assertEquals(2L, globalCacheIndex.totalBlocks());
        assertEquals(3L, globalCacheIndex.totalMappings());

        globalCacheIndex.removeCacheBlock("engine1", 1L, null);
        assertEquals(2L, globalCacheIndex.totalBlocks());
        assertEquals(2L, globalCacheIndex.totalMappings());
        globalCacheIndex.removeAllCacheBlockOfEngine("engine2", Set.of(1L), null);
        assertEquals(1L, globalCacheIndex.totalBlocks());
        assertEquals(1L, globalCacheIndex.totalMappings());
        globalCacheIndex.removeCacheBlock("missing", 2L, null);
        globalCacheIndex.clear();
        assertEquals(0L, globalCacheIndex.totalBlocks());
        assertEquals(0L, globalCacheIndex.totalMappings());
        assertTrue(
                globalCacheIndex
                        .batchCalculatePrefixMatchLength(
                                List.of("engine1", "engine2"), List.of(1L, 2L))
                        .isEmpty());

        globalCacheIndex.addCacheBlock(3L, "engine1", null);
        assertEquals(1L, globalCacheIndex.totalBlocks());
        assertEquals(1L, globalCacheIndex.totalMappings());
    }

    @Test
    void testEmptyInput() {
        // Test empty input cases
        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(
                        Collections.emptyList(), Arrays.asList(1L, 2L, 3L));
        assertTrue(result.isEmpty(), "Empty engines should return empty result");

        result =
                globalCacheIndex.batchCalculatePrefixMatchLength(
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
        globalCacheIndex.addCacheBlock(1L, "engine1", null);
        globalCacheIndex.addCacheBlock(1L, "engine2", null);
        globalCacheIndex.addCacheBlock(1L, "engine3", null);
        globalCacheIndex.addCacheBlock(2L, "engine1", null);
        globalCacheIndex.addCacheBlock(2L, "engine3", null);
        globalCacheIndex.addCacheBlock(3L, "engine1", null);

        List<String> engines = Arrays.asList("engine1", "engine2", "engine3");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L);

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        // Verify results
        assertEquals(3, result.get("engine1").intValue(), "Engine1 should match all 3 blocks");
        assertEquals(1, result.get("engine2").intValue(), "Engine2 should match only first block");
        assertEquals(2, result.get("engine3").intValue(), "Engine3 should match first 2 blocks");
    }

    @Test
    void testNoMatchingEngines() {
        // Test case where no engine matches any block
        globalCacheIndex.addCacheBlock(1L, "other_engine", null);
        globalCacheIndex.addCacheBlock(2L, "another_engine", null);

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L);

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertTrue(
                result.isEmpty(),
                "engines with zero prefix match are omitted from the sparse result");
    }

    @Test
    void testAllEnginesMatchAllBlocks() {
        // Test case where all engines match all blocks
        List<String> engines = Arrays.asList("engine1", "engine2", "engine3");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        // Add all engines to all blocks
        for (Long block : blocks) {
            for (String engine : engines) {
                globalCacheIndex.addCacheBlock(block, engine, null);
            }
        }

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        for (String engine : engines) {
            assertEquals(4, result.get(engine).intValue(), "All engines should match all 4 blocks");
        }
    }

    @Test
    void testEarlyTermination() {
        // Test early termination case
        // Set up data so there are no candidate engines after the 2nd block
        globalCacheIndex.addCacheBlock(1L, "engine1", null);
        globalCacheIndex.addCacheBlock(1L, "engine2", null);
        globalCacheIndex.addCacheBlock(2L, "engine1", null);
        // block 3L and 4L have no engines

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(
                2,
                result.get("engine1").intValue(),
                "Engine1 should match 2 blocks before termination");
        assertEquals(
                1,
                result.get("engine2").intValue(),
                "Engine2 should match 1 block before termination");
    }

    @Test
    void testPartialMatching() {
        // Test complex partial matching scenario
        globalCacheIndex.addCacheBlock(1L, "engine1", null);
        globalCacheIndex.addCacheBlock(1L, "engine2", null);
        globalCacheIndex.addCacheBlock(1L, "engine3", null);
        globalCacheIndex.addCacheBlock(1L, "engine4", null);

        globalCacheIndex.addCacheBlock(2L, "engine1", null);
        globalCacheIndex.addCacheBlock(2L, "engine2", null);
        globalCacheIndex.addCacheBlock(2L, "engine3", null);

        globalCacheIndex.addCacheBlock(3L, "engine1", null);
        globalCacheIndex.addCacheBlock(3L, "engine2", null);

        globalCacheIndex.addCacheBlock(4L, "engine1", null);

        List<String> engines = Arrays.asList("engine1", "engine2", "engine3", "engine4");
        List<Long> blocks = Arrays.asList(1L, 2L, 3L, 4L);

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(4, result.get("engine1").intValue(), "Engine1 should match all blocks");
        assertEquals(3, result.get("engine2").intValue(), "Engine2 should match first 3 blocks");
        assertEquals(2, result.get("engine3").intValue(), "Engine3 should match first 2 blocks");
        assertEquals(1, result.get("engine4").intValue(), "Engine4 should match first block only");
    }

    @Test
    void testSingleBlockSingleEngine() {
        // Test single block single engine case
        globalCacheIndex.addCacheBlock(1L, "engine1", null);

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(List.of("engine1"), List.of(1L));

        assertEquals(
                1, result.get("engine1").intValue(), "Single engine should match single block");
    }

    @Test
    void reusedCompactionBufferNeverLeaksCandidatesFromALargerQuery() {
        globalCacheIndex.addCacheBlock(1L, "engine1", null);
        globalCacheIndex.addCacheBlock(1L, "engine2", null);

        assertEquals(
                Map.of("engine1", 1, "engine2", 1),
                globalCacheIndex.batchCalculatePrefixMatchLength(
                        List.of("engine1", "engine2"), List.of(1L)));
        assertEquals(
                Map.of("engine1", 1),
                globalCacheIndex.batchCalculatePrefixMatchLength(List.of("engine1"), List.of(1L)));
    }

    @Test
    void testNonExistentBlocks() {
        // Test non-existent blocks
        globalCacheIndex.addCacheBlock(1L, "engine1", null);

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 999L); // 999L does not exist

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(1, result.get("engine1").intValue(), "Engine1 should match first block only");
        assertTrue(
                !result.containsKey("engine2"), "zero-prefix engines are represented by absence");
    }

    @Test
    void testEngineRemoval() {
        // Test matching after engine removal
        globalCacheIndex.addCacheBlock(1L, "engine1", null);
        globalCacheIndex.addCacheBlock(1L, "engine2", null);
        globalCacheIndex.addCacheBlock(2L, "engine1", null);
        globalCacheIndex.addCacheBlock(2L, "engine2", null);

        // Remove engine2 from block 2L
        globalCacheIndex.removeCacheBlock("engine2", 2L, null);

        List<String> engines = Arrays.asList("engine1", "engine2");
        List<Long> blocks = Arrays.asList(1L, 2L);

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(2, result.get("engine1").intValue(), "Engine1 should still match both blocks");
        assertEquals(
                1,
                result.get("engine2").intValue(),
                "Engine2 should match only first block after removal");
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
                globalCacheIndex.addCacheBlock(blocks.get(j), engine, null);
            }
        }

        Map<String, Integer> result =
                globalCacheIndex.batchCalculatePrefixMatchLength(engines, blocks);

        assertEquals(10, result.get("engine1").intValue(), "Engine1 should match 10 blocks");
        assertEquals(9, result.get("engine2").intValue(), "Engine2 should match 9 blocks");
        assertEquals(8, result.get("engine3").intValue(), "Engine3 should match 8 blocks");
        assertEquals(7, result.get("engine4").intValue(), "Engine4 should match 7 blocks");
        assertEquals(6, result.get("engine5").intValue(), "Engine5 should match 6 blocks");
    }
}
