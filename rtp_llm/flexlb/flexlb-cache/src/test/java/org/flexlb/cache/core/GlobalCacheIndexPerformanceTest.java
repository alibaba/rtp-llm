package org.flexlb.cache.core;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Four worker snapshots apply to the same index, including overlapping ownership. */
@Tag("performance-regression")
class GlobalCacheIndexPerformanceTest {
    @ParameterizedTest
    @ValueSource(ints = {1_000_000, 2_000_000, 3_000_000, 4_000_000})
    @Timeout(240)
    void concurrentMillionBlockUpdates(int blocks) throws Exception {
        var index = new GlobalCacheIndex();
        var workers = List.of("worker-A", "worker-B", "worker-C", "worker-D");
        try (var executor = Executors.newFixedThreadPool(workers.size())) {
            for (int round = 0; round < 3; round++) {
                int iteration = round;
                var ready = new CountDownLatch(workers.size());
                var start = new CountDownLatch(1);
                var tasks = new ArrayList<Callable<Void>>();
                for (String worker : workers) {
                    tasks.add(() -> {
                        ready.countDown();
                        start.await();
                        if (iteration == 0) {
                            for (long key = 0; key < blocks; key++) {
                                index.addCacheBlock(hash(key), worker, null);
                            }
                        } else {
                            long from = iteration == 1 ? 0 : blocks;
                            long to = iteration == 1 ? blocks : 0;
                            for (long key = 0; key < blocks / 2; key++) {
                                index.removeCacheBlock(worker, hash(from + key), null);
                                index.addCacheBlock(hash(to + key), worker, null);
                            }
                        }
                        return null;
                    });
                }
                var futures = tasks.stream().map(executor::submit).toList();
                ready.await();
                long before = System.nanoTime();
                start.countDown();
                for (var future : futures) {
                    future.get();
                }
                long elapsed = System.nanoTime() - before;
                assertEquals(blocks, index.totalBlocks());
                assertEquals((long) blocks * workers.size(), index.totalMappings());
                assertEquals(workers.size(), index.batchCalculatePrefixMatchLength(workers,
                        List.of(hash((long) blocks / 2))).size());
                System.out.printf("Concurrent index: blocksPerWorker=%d workers=%d round=%d elapsedMs=%.3f mutationsPerSecond=%.0f%n",
                        blocks, workers.size(), iteration, elapsed / 1_000_000.0,
                        (double) blocks * workers.size() / (elapsed / 1_000_000_000.0));
            }
        } finally {
            index.clear();
        }
    }

    /** Full resident 400M ownership mappings; opt in explicitly on a sufficiently large heap. */
    @Test
    @EnabledIfSystemProperty(named = "flexlb.perf.fleet", matches = "true")
    @Timeout(1200)
    void hundredWorkersWithFourMillionBlocks() throws Exception {
        int blocks = 4_000_000;
        int workerCount = 100;
        int threads = Integer.getInteger("flexlb.perf.threads", 32);
        var index = new GlobalCacheIndex();
        var workers = java.util.stream.IntStream.range(0, workerCount).mapToObj(i -> "worker-" + i).toList();
        // Populate the real mutable index directly; cold construction is outside the measured workload.
        var mapField = GlobalCacheIndex.class.getDeclaredField("blockToEnginesMap");
        mapField.setAccessible(true);
        @SuppressWarnings("unchecked")
        var mappings = (java.util.concurrent.ConcurrentHashMap<Long, java.util.Set<String>>) mapField.get(index);
        long warmupStart = System.nanoTime();
        for (long key = 0; key < blocks; key++) {
            var owners = java.util.concurrent.ConcurrentHashMap.<String>newKeySet(workerCount);
            owners.addAll(workers);
            mappings.put(hash(key), owners);
        }
        var countField = GlobalCacheIndex.class.getDeclaredField("totalMappings");
        countField.setAccessible(true);
        ((java.util.concurrent.atomic.LongAdder) countField.get(index)).add((long) blocks * workerCount);
        System.out.printf("FLEET warmupCompleteMs=%.3f mappings=%d%n", (System.nanoTime() - warmupStart) / 1e6, index.totalMappings());
        try (var executor = Executors.newFixedThreadPool(threads)) {
            for (int round = 1; round <= 3; round++) {
                int iteration = round;
                int changed = blocks * round / 100;
                var ready = new CountDownLatch(Math.min(threads, workerCount));
                var start = new CountDownLatch(1);
                var futures = new ArrayList<java.util.concurrent.Future<Long>>();
                long[] started = new long[workerCount];
                long[] finished = new long[workerCount];
                for (int worker = 0; worker < workerCount; worker++) {
                    int workerId = worker;
                    String address = workers.get(worker);
                    futures.add(executor.submit(() -> {
                        ready.countDown();
                        start.await();
                        started[workerId] = System.nanoTime();
                        for (long key = 0; key < changed; key++) {
                            long oldKey = key < blocks * (iteration - 1L) / 100
                                    ? (long) blocks * (iteration - 1) + key : key;
                            index.removeCacheBlock(address, hash(oldKey), null);
                            long newKey = (long) blocks * iteration + key;
                            index.addCacheBlock(hash(newKey), address, null);
                        }
                        finished[workerId] = System.nanoTime();
                        return finished[workerId] - started[workerId];
                    }));
                }
                ready.await();
                long before = System.nanoTime();
                start.countDown();
                var durations = new ArrayList<Long>();
                for (var future : futures) {durations.add(future.get());}
                long elapsed = System.nanoTime() - before;
                durations.sort(Long::compare);
                assertEquals(blocks, index.totalBlocks());
                assertEquals((long) blocks * workerCount, index.totalMappings());
                assertEquals(workerCount, index.batchCalculatePrefixMatchLength(
                        workers, List.of(hash(3_000_000))).size());
                long used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                System.out.printf("FLEET blocksPerWorker=%d workers=%d threads=%d change=%s elapsedMs=%.3f serviceP50Ms=%.3f serviceP95Ms=%.3f serviceMaxMs=%.3f lastQueueMs=%.3f heapUsedMiB=%.1f mappings=%d%n",
                        blocks, workerCount, threads, iteration + "pct",
                        elapsed / 1e6, durations.get(49) / 1e6, durations.get(94) / 1e6,
                        durations.getLast() / 1e6,
                        (java.util.Arrays.stream(started).max().orElseThrow() - before) / 1e6,
                        used / 1048576.0, index.totalMappings());
                System.out.flush();
            }
        } finally {
            index.clear();
        }
    }

    private static long hash(long value) {
        long mixed = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        mixed = (mixed ^ (mixed >>> 27)) * 0x94d049bb133111ebL;
        return mixed ^ (mixed >>> 31);
    }
}
