package org.flexlb.cache.match.localstandby;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class LocalStandbyCacheIndexTest {

    @Test
    void backgroundCleanupRemovesExpiredMappingsAndEmptyBlocks() throws InterruptedException {
        LocalStandbyCacheIndex cacheIndex = new LocalStandbyCacheIndex(1, 10, false);
        cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L));

        Thread.sleep(10);
        cacheIndex.removeExpiredMappingsBatch();

        assertEquals(0, cacheIndex.mappingCount());
        assertNull(cacheIndex.getUnexpiredEnginesForBlock(11L, System.nanoTime()));
        cacheIndex.shutdown();
    }

    @Test
    void concurrentRefreshOfExistingMappingKeepsSingleEntry() {
        LocalStandbyCacheIndex cacheIndex = new LocalStandbyCacheIndex(60_000, 10, false);
        cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L));

        IntStream.range(0, 1_000)
                .parallel()
                .forEach(ignored -> cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L)));

        assertEquals(1, cacheIndex.mappingCount());
        assertEquals(1, cacheIndex.getUnexpiredEnginesForBlock(11L, System.nanoTime()).size());
        cacheIndex.shutdown();
    }

    @Test
    void existingMappingCanBeRefreshedAtCapacity() {
        LocalStandbyCacheIndex cacheIndex = new LocalStandbyCacheIndex(60_000, 1, false);
        assertEquals(0, cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L)));

        assertEquals(0, cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L)));
        assertEquals(1, cacheIndex.addWorkerBlockMappings("10.0.0.2:8080", List.of(11L)));
        assertEquals(1, cacheIndex.mappingCount());
        cacheIndex.shutdown();
    }
}
