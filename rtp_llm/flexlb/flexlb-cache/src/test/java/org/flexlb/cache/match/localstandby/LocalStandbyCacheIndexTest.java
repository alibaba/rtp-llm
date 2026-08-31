package org.flexlb.cache.match.localstandby;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
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
    void appliesUpdatedTtlToExistingMappings() {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(300_000, 100_000, 0.8, 10);
        cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L));

        cacheIndex.updateExpirationSettings(1, 1, 0.8);

        assertNull(cacheIndex.getUnexpiredEnginesForBlock(
                11L, System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(2)));
        assertEquals(0, cacheIndex.mappingCount());
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
    void rejectsNewMappingsAtHardLimitAndResumesAfterCleanup() throws InterruptedException {
        String worker = "10.0.0.1:8080";
        LocalStandbyCacheIndex cacheIndex = cacheIndex(1, 1, 0.8, 2);
        cacheIndex.updateMaximumEntries(2);

        assertEquals(
                1, cacheIndex.addWorkerBlockMappings(worker, List.of(11L, 22L, 33L)));
        assertEquals(0, cacheIndex.addWorkerBlockMappings(worker, List.of(11L)));

        assertEquals(2, cacheIndex.mappingCount());
        assertNull(cacheIndex.getUnexpiredEnginesForBlock(33L, System.nanoTime()));

        Thread.sleep(5);
        cacheIndex.runHighWatermarkFullScan();
        assertEquals(0, cacheIndex.mappingCount());
        assertEquals(0, cacheIndex.addWorkerBlockMappings(worker, List.of(33L)));
        assertEquals(1, cacheIndex.mappingCount());
        cacheIndex.shutdown();
    }

    @Test
    void highWatermarkCleanupDoesNotEvictUnexpiredMappings() {
        String worker = "10.0.0.1:8080";
        LocalStandbyCacheIndex cacheIndex = cacheIndex(60_000, 20_000, 0.8, 10);
        cacheIndex.addWorkerBlockMappings(
                worker, IntStream.range(0, 10).mapToObj(value -> (long) value).toList());

        cacheIndex.runHighWatermarkFullScan();

        assertEquals(10, cacheIndex.mappingCount());
        cacheIndex.shutdown();
    }

    @Test
    void highWatermarkCleanupScansEntireIndexForExpiredMappings() throws InterruptedException {
        String worker = "10.0.0.1:8080";
        LocalStandbyCacheIndex cacheIndex = cacheIndex(1, 1, 0.8, 10);
        cacheIndex.addWorkerBlockMappings(
                worker, IntStream.range(0, 10).mapToObj(value -> (long) value).toList());

        Thread.sleep(5);
        cacheIndex.runHighWatermarkFullScan();

        assertEquals(0, cacheIndex.mappingCount());
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
                cacheIndex.effectiveTtlNanos());

        cacheIndex.addWorkerBlockMappings(worker, List.of(8L));
        long pressureTtlMs =
                TimeUnit.NANOSECONDS.toMillis(cacheIndex.effectiveTtlNanos());
        assertTrue(pressureTtlMs >= 199_999 && pressureTtlMs <= 200_001);

        cacheIndex.addWorkerBlockMappings(worker, List.of(9L));
        assertEquals(
                TimeUnit.MILLISECONDS.toNanos(100_000),
                cacheIndex.effectiveTtlNanos());
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
        cacheIndex.shutdown();
    }

    @Test
    void normalCleanupScansTenPercentOfBlockHashes() throws InterruptedException {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(1, 1, 0.8, 100);
        cacheIndex.addWorkerBlockMappings(
                "10.0.0.1:8080",
                IntStream.range(0, 70).mapToObj(value -> (long) value).toList());
        Thread.sleep(5);

        cacheIndex.runCleanupCheck();
        cacheIndex.runCleanupCheck();
        cacheIndex.runCleanupCheck();

        assertEquals(63, cacheIndex.mappingCount());
        cacheIndex.shutdown();
    }

    @Test
    void pressureCleanupScansTwentyPercentOfBlockHashes() throws InterruptedException {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(1, 1, 0.8, 100);
        cacheIndex.addWorkerBlockMappings(
                "10.0.0.1:8080",
                IntStream.range(0, 80).mapToObj(value -> (long) value).toList());
        Thread.sleep(5);

        cacheIndex.runCleanupCheck();
        cacheIndex.runCleanupCheck();

        assertEquals(64, cacheIndex.mappingCount());
        cacheIndex.shutdown();
    }

    @Test
    void highWatermarkCleanupScansAllBlockHashes() throws InterruptedException {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(1, 1, 0.8, 10);
        cacheIndex.addWorkerBlockMappings(
                "10.0.0.1:8080",
                IntStream.range(0, 9).mapToObj(value -> (long) value).toList());
        Thread.sleep(5);

        cacheIndex.runCleanupCheck();

        assertEquals(0, cacheIndex.mappingCount());
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
        AtomicInteger rejectedMappings = new AtomicInteger();

        IntStream.range(0, 1_000).parallel().forEach(index -> {
            String worker = "10.0.0." + (index % 4 + 1) + ":8080";
            rejectedMappings.addAndGet(
                    cacheIndex.addWorkerBlockMappings(worker, List.of((long) index)));
        });

        long indexedMappings = IntStream.range(0, 1_000)
                .mapToLong(index -> {
                    Map<String, Long> owners =
                            cacheIndex.getUnexpiredEnginesForBlock(
                                    (long) index, System.nanoTime());
                    return owners == null ? 0 : owners.size();
                })
                .sum();
        assertTrue(cacheIndex.mappingCount() >= 100);
        assertEquals(1_000, cacheIndex.mappingCount() + rejectedMappings.get());
        assertEquals(cacheIndex.mappingCount(), indexedMappings);
        cacheIndex.shutdown();
    }

    private static LocalStandbyCacheIndex cacheIndex(
            long ttlMs,
            long minimumTtlMs,
            double ttlReductionStartRatio,
            long maximumEntries) {
        return new LocalStandbyCacheIndex(
                ttlMs,
                minimumTtlMs,
                ttlReductionStartRatio,
                maximumEntries,
                false);
    }

}
