package org.flexlb.cache.match.localstandby;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LocalStandbyCacheIndexTest {

    @Test
    void backgroundCleanupRemovesExpiredMappingsAndEmptyBlocks() throws InterruptedException {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(1, 1, 0.8, 10);
        cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L));

        Thread.sleep(10);
        cacheIndex.removeExpiredMappingsBatch();

        assertEquals(0, cacheIndex.mappingCount());
        assertNull(cacheIndex.getUnexpiredEnginesForBlock(11L, System.nanoTime()));
        cacheIndex.shutdown();
    }

    @Test
    void concurrentRefreshOfExistingMappingKeepsSingleEntry() {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(60_000, 20_000, 0.8, 10);
        cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L));

        IntStream.range(0, 1_000)
                .parallel()
                .forEach(ignored ->
                        cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L)));

        assertEquals(1, cacheIndex.mappingCount());
        assertEquals(1, cacheIndex.getUnexpiredEnginesForBlock(11L, System.nanoTime()).size());
        cacheIndex.shutdown();
    }

    @Test
    void acceptsNewMappingsBeyondEstimatedCapacity() {
        String worker = "10.0.0.1:8080";
        LocalStandbyCacheIndex cacheIndex = cacheIndex(60_000, 20_000, 0.8, 2);
        cacheIndex.updateMaximumEntries(2);

        cacheIndex.addWorkerBlockMappings(worker, List.of(11L, 22L, 33L));

        assertEquals(3, cacheIndex.mappingCount());
        assertEquals(Set.of(worker), owners(cacheIndex, 11L));
        assertEquals(Set.of(worker), owners(cacheIndex, 22L));
        assertEquals(Set.of(worker), owners(cacheIndex, 33L));
        cacheIndex.shutdown();
    }

    @Test
    void reducesTtlLinearlyAsGlobalCapacityFills() {
        String worker = "10.0.0.1:8080";
        LocalStandbyCacheIndex cacheIndex = cacheIndex(300_000, 100_000, 0.8, 10);
        cacheIndex.updateMaximumEntries(10);

        cacheIndex.addWorkerBlockMappings(
                worker, IntStream.range(0, 8).mapToObj(value -> (long) value).toList());
        assertEquals(
                TimeUnit.MILLISECONDS.toNanos(300_000),
                cacheIndex.effectiveEntryTtlNanos());

        cacheIndex.addWorkerBlockMappings(worker, List.of(8L));
        long pressureTtlMs =
                TimeUnit.NANOSECONDS.toMillis(cacheIndex.effectiveEntryTtlNanos());
        assertTrue(pressureTtlMs >= 199_999 && pressureTtlMs <= 200_001);

        cacheIndex.addWorkerBlockMappings(worker, List.of(9L));
        assertEquals(
                TimeUnit.MILLISECONDS.toNanos(100_000),
                cacheIndex.effectiveEntryTtlNanos());
        cacheIndex.shutdown();
    }

    @Test
    void increasesCleanupFrequencyAsCapacityFills() {
        String worker = "10.0.0.1:8080";
        LocalStandbyCacheIndex cacheIndex = cacheIndex(300_000, 100_000, 0.8, 10);
        cacheIndex.updateMaximumEntries(10);

        cacheIndex.addWorkerBlockMappings(
                worker, IntStream.range(0, 7).mapToObj(value -> (long) value).toList());
        assertEquals(3, cacheIndex.checksBeforeCleanup());

        cacheIndex.addWorkerBlockMappings(worker, List.of(7L));
        assertEquals(2, cacheIndex.checksBeforeCleanup());

        cacheIndex.addWorkerBlockMappings(worker, List.of(8L, 9L));
        assertEquals(1, cacheIndex.checksBeforeCleanup());
        cacheIndex.shutdown();
    }

    @Test
    void appliesReducedTtlToExistingMappingsUnderCapacityPressure()
            throws InterruptedException {
        String worker = "10.0.0.1:8080";
        LocalStandbyCacheIndex cacheIndex = cacheIndex(100, 20, 0.8, 10);
        cacheIndex.updateMaximumEntries(10);
        cacheIndex.addWorkerBlockMappings(
                worker, IntStream.range(0, 10).mapToObj(value -> (long) value).toList());

        Thread.sleep(30);

        assertNull(cacheIndex.getUnexpiredEnginesForBlock(0L, System.nanoTime()));
        assertEquals(9, cacheIndex.mappingCount());
        cacheIndex.shutdown();
    }

    @Test
    void concurrentUpdatesBeyondCapacityKeepIndexAndCountersConsistent() {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(60_000, 20_000, 0.8, 100);
        cacheIndex.updateMaximumEntries(100);

        IntStream.range(0, 1_000).parallel().forEach(index -> {
            String worker = "10.0.0." + (index % 4 + 1) + ":8080";
            cacheIndex.addWorkerBlockMappings(worker, List.of((long) index));
        });

        long indexedMappings = IntStream.range(0, 1_000)
                .mapToLong(index -> {
                    Map<String, Long> owners =
                            cacheIndex.getUnexpiredEnginesForBlock(
                                    (long) index, System.nanoTime());
                    return owners == null ? 0 : owners.size();
                })
                .sum();
        assertEquals(1_000, cacheIndex.mappingCount());
        assertEquals(cacheIndex.mappingCount(), indexedMappings);
        cacheIndex.shutdown();
    }

    private static LocalStandbyCacheIndex cacheIndex(
            long entryTtlMs,
            long minimumEntryTtlMs,
            double ttlReductionStartRatio,
            long maximumEntries) {
        return new LocalStandbyCacheIndex(
                entryTtlMs,
                minimumEntryTtlMs,
                ttlReductionStartRatio,
                maximumEntries,
                false);
    }

    private static Set<String> owners(LocalStandbyCacheIndex cacheIndex, long blockCacheKey) {
        return Set.copyOf(
                cacheIndex.getUnexpiredEnginesForBlock(blockCacheKey, System.nanoTime()).keySet());
    }
}
