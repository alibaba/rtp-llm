package org.flexlb.cache.match.localstandby;

import org.junit.jupiter.api.Test;

import java.util.List;

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
}
