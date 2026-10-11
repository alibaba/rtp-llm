package org.flexlb.cache.match.localstandby;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LocalStandbyCacheIndexTest {

    @Test
    void rejectsInvalidTtlConfigurationBeforeStartingCleanup() {
        assertThrows(IllegalArgumentException.class, () -> cacheIndex(10, 20, 0.8, 10));
        assertThrows(IllegalArgumentException.class, () -> cacheIndex(0, 0, 0.8, 10));
        assertThrows(IllegalArgumentException.class, () -> cacheIndex(10, 0, 0.8, 10));
        for (double ratio : List.of(-0.1, 1.1, Double.NaN, Double.POSITIVE_INFINITY)) {
            assertThrows(IllegalArgumentException.class, () -> cacheIndex(20, 10, ratio, 10));
        }
    }

    @Test
    void appliesMinimumTtlAtCapacityWhenReductionStartsAtOne() {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(60_000, 20_000, 1.0, 10);
        try {
            cacheIndex.addWorkerBlockMappings("10.0.0.1:8080",
                    IntStream.range(0, 9).mapToObj(value -> (long) value).toList());
            assertEquals(TimeUnit.MILLISECONDS.toNanos(60_000), cacheIndex.effectiveTtlNanos());

            cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(9L));

            assertEquals(TimeUnit.MILLISECONDS.toNanos(20_000), cacheIndex.effectiveTtlNanos());
        } finally {
            cacheIndex.shutdown();
        }
    }

    @Test
    void highWatermarkRefreshesDoNotQueueFullScansDuringCooldown() throws Exception {
        AtomicInteger cleanupChecks = new AtomicInteger();
        LocalStandbyCacheIndex cacheIndex = new LocalStandbyCacheIndex(60_000, 20_000, 0.8, 10, true) {
            @Override
            void runCleanupCheck() {
                cleanupChecks.incrementAndGet();
                super.runCleanupCheck();
            }
        };
        ScheduledExecutorService cleanupExecutor = cleanupExecutor(cacheIndex);
        try {
            cacheIndex.addWorkerBlockMappings("10.0.0.1:8080",
                    IntStream.range(0, 9).mapToObj(value -> (long) value).toList());
            cleanupExecutor.submit(() -> { }).get(2, TimeUnit.SECONDS);
            assertEquals(1, cleanupChecks.get());

            for (int request = 0; request < 1_000; request++) {
                cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(0L));
            }
            cleanupExecutor.submit(() -> { }).get(2, TimeUnit.SECONDS);
            assertEquals(1, cleanupChecks.get());
            assertEquals(9, cacheIndex.mappingCount());

            ReflectionTestUtils.setField(cacheIndex, "nextHighWatermarkScanTimeNanos", System.nanoTime() - 1);
            cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(0L));
            cleanupExecutor.submit(() -> { }).get(2, TimeUnit.SECONDS);
            assertEquals(2, cleanupChecks.get());
        } finally {
            cacheIndex.shutdown();
        }
    }

    @Test
    void scheduledChecksAlsoRespectFullScanCooldown() {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(60_000, 20_000, 0.8, 10);
        try {
            cacheIndex.addWorkerBlockMappings("10.0.0.1:8080",
                    IntStream.range(0, 9).mapToObj(value -> (long) value).toList());
            cacheIndex.runCleanupCheck();
            long now = System.nanoTime();
            for (long block = 0; block < 9; block++) {
                Map<String, Long> workers = cacheIndex.getUnexpiredEnginesForBlock(block, now);
                workers.replaceAll((worker, timestamp) -> now - TimeUnit.MINUTES.toNanos(1));
            }

            cacheIndex.runCleanupCheck();

            assertEquals(9, cacheIndex.mappingCount());
            ReflectionTestUtils.setField(cacheIndex, "nextHighWatermarkScanTimeNanos", System.nanoTime() - 1);
            cacheIndex.runCleanupCheck();
            assertEquals(0, cacheIndex.mappingCount());
        } finally {
            cacheIndex.shutdown();
        }
    }

    @Test
    void shutdownInterruptsActiveBackgroundCleanup() throws Exception {
        CountDownLatch cleanupStarted = new CountDownLatch(1);
        CountDownLatch cleanupStopped = new CountDownLatch(1);
        AtomicBoolean interrupted = new AtomicBoolean();
        LocalStandbyCacheIndex cacheIndex = new LocalStandbyCacheIndex(60_000, 20_000, 0.8, 10, true) {
            @Override
            void runCleanupCheck() {
                cleanupStarted.countDown();
                try {
                    new CountDownLatch(1).await();
                } catch (InterruptedException exception) {
                    interrupted.set(true);
                    Thread.currentThread().interrupt();
                } finally {
                    cleanupStopped.countDown();
                }
            }
        };
        try {
            cacheIndex.addWorkerBlockMappings("10.0.0.1:8080",
                    IntStream.range(0, 9).mapToObj(value -> (long) value).toList());
            assertTrue(cleanupStarted.await(2, TimeUnit.SECONDS));

            cacheIndex.shutdown();

            assertTrue(cleanupStopped.await(2, TimeUnit.SECONDS));
            assertTrue(interrupted.get());
            assertTrue(cleanupExecutor(cacheIndex).awaitTermination(2, TimeUnit.SECONDS));
        } finally {
            cacheIndex.shutdown();
        }
    }

    @Test
    void cleanupStopsWhenInterruptedOrClosed() throws InterruptedException {
        LocalStandbyCacheIndex cacheIndex = cacheIndex(1, 1, 0.8, 10);
        cacheIndex.addWorkerBlockMappings("10.0.0.1:8080", List.of(11L));
        Thread.sleep(5);
        try {
            Thread.currentThread().interrupt();
            cacheIndex.runHighWatermarkFullScan();
            cacheIndex.removeExpiredMappingsBatch();
        } finally {
            Thread.interrupted();
        }
        assertEquals(1, cacheIndex.mappingCount());

        cacheIndex.shutdown();
        cacheIndex.runCleanupCheck();
        cacheIndex.runHighWatermarkFullScan();
        cacheIndex.removeExpiredMappingsBatch();

        assertEquals(1, cacheIndex.mappingCount());
    }

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
        assertEquals(100, cacheIndex.mappingCount());
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

    private static ScheduledExecutorService cleanupExecutor(LocalStandbyCacheIndex cacheIndex) {
        return (ScheduledExecutorService) ReflectionTestUtils.getField(cacheIndex, "cleanupExecutor");
    }

}
