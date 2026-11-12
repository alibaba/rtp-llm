package org.flexlb.cache.core;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.lang.management.ManagementFactory;
import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Index-only fleet benchmark. Every worker owns a disjoint, random-looking key range. */
@Tag("performance-regression")
class CacheFleetChurnPerformanceTest {
    private static final int WORKERS = 100;
    private static final int RETIRE = 30;
    private static final int KEYS = Integer.getInteger("fleet.keysPerWorker", 250_000);
    private static final int CLEANERS = Integer.getInteger("fleet.cleaners", 4);

    @Test
    @Timeout(900)
    void initializeFleetThenRetireThirtyWhileSeventyReplaceRandomTwoPercent() throws Exception {
        assertTrue(KEYS > 0 && KEYS % 50 == 0);
        GlobalCacheIndex index = new GlobalCacheIndex();
        long total = (long) WORKERS * KEYS;
        int changed = KEYS / 50;
        int[][] selected = new int[WORKERS][];
        for (int worker = RETIRE; worker < WORKERS; worker++) {
            selected[worker] = new int[changed];
            BitSet seen = new BitSet(KEYS);
            SplittableRandom random = new SplittableRandom(1729L + worker);
            for (int n = 0; n < changed;) {
                int position = random.nextInt(KEYS);
                if (!seen.get(position)) {
                    seen.set(position);
                    selected[worker][n++] = position;
                }
            }
        }
        try (var updates = Executors.newFixedThreadPool(32);
             var cleaners = Executors.newFixedThreadPool(CLEANERS)) {
            CountDownLatch initialize = new CountDownLatch(1);
            var initial = new ArrayList<Future<Long>>();
            for (int worker = 0; worker < WORKERS; worker++) {
                final int id = worker;
                initial.add(updates.submit(() -> {
                    initialize.await();
                    long start = System.nanoTime();
                    String owner = "worker-" + id;
                    for (int n = 0; n < KEYS; n++) index.addCacheBlock(oldKey(id, n), owner, null);
                    return System.nanoTime() - start;
                }));
            }
            long initStart = System.nanoTime();
            initialize.countDown();
            long[] initLatency = collect(initial);
            double initMs = ms(System.nanoTime() - initStart);
            assertEquals(total, index.totalBlocks());
            assertEquals(total, index.totalMappings());
            long heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
            System.out.printf(Locale.ROOT, "FLEET_INIT workers=100 keysPerWorker=%d totalKeys=%d threads=32 elapsedMs=%.3f workerServiceP50Ms=%.3f workerServiceP99Ms=%.3f heapUsedMiB=%.1f%n",
                    KEYS, total, initMs, percentile(initLatency, .5), percentile(initLatency, .99), heap / 1048576.0);

            CountDownLatch begin = new CountDownLatch(1);
            var retired = new ArrayList<Future<Long>>();
            var updated = new ArrayList<Future<Long>>();
            long[] cleanupEnd = new long[RETIRE];
            long[] updateEnd = new long[WORKERS - RETIRE];
            for (int worker = 0; worker < RETIRE; worker++) {
                final int id = worker;
                retired.add(cleaners.submit(() -> {
                    begin.await();
                    long start = System.nanoTime();
                    index.removeAllCacheBlockOfEngine("worker-" + id, new AbstractSet<Long>() {
                        @Override public int size() { return KEYS; }
                        @Override public Iterator<Long> iterator() {
                            return new Iterator<>() {
                                int n;
                                @Override public boolean hasNext() { return n < KEYS; }
                                @Override public Long next() {
                                    if (!hasNext()) throw new NoSuchElementException();
                                    return oldKey(id, n++);
                                }
                            };
                        }
                    }, null);
                    cleanupEnd[id] = System.nanoTime();
                    return cleanupEnd[id] - start;
                }));
            }
            for (int worker = RETIRE; worker < WORKERS; worker++) {
                final int id = worker;
                updated.add(updates.submit(() -> {
                    begin.await();
                    long start = System.nanoTime();
                    String owner = "worker-" + id;
                    for (int n = 0; n < changed; n++) {
                        index.removeCacheBlock(owner, oldKey(id, selected[id][n]), null);
                        index.addCacheBlock(newKey(id, n, changed), owner, null);
                    }
                    updateEnd[id - RETIRE] = System.nanoTime();
                    return updateEnd[id - RETIRE] - start;
                }));
            }
            long mixedStart = System.nanoTime();
            begin.countDown();
            long[] updateService = collect(updated);
            long[] cleanupService = collect(retired);
            long lastCleanup = Arrays.stream(cleanupEnd).max().orElseThrow();
            long lastUpdate = Arrays.stream(updateEnd).max().orElseThrow();
            long progressed = Arrays.stream(updateEnd).filter(t -> t < lastCleanup).count();
            long[] completion = Arrays.stream(updateEnd).map(t -> t - mixedStart).sorted().toArray();
            System.out.printf(Locale.ROOT, "FLEET_MIX keysPerWorker=%d retiredKeys=%d changedKeys=%d updateOps=%d cleanerThreads=%d updateThreads=32 allMs=%.3f retire30Ms=%.3f update70Ms=%.3f updateServiceP50Ms=%.3f updateServiceP99Ms=%.3f updateCompletionP99Ms=%.3f cleanupServiceP99Ms=%.3f updatesBeforeLastCleanup=%d/70%n",
                    KEYS, (long) RETIRE * KEYS, (long) 70 * changed, (long) 140 * changed, CLEANERS,
                    ms(Math.max(lastCleanup, lastUpdate) - mixedStart), ms(lastCleanup - mixedStart),
                    ms(lastUpdate - mixedStart), percentile(updateService, .5), percentile(updateService, .99),
                    percentile(completion, .99), percentile(cleanupService, .99), progressed);
            assertEquals((long) 70 * KEYS, index.totalBlocks());
            assertEquals((long) 70 * KEYS, index.totalMappings());
            for (int worker = 0; worker < RETIRE; worker++) {
                assertTrue(index.batchCalculatePrefixMatchLength(List.of("worker-" + worker),
                        List.of(oldKey(worker, 0))).isEmpty());
            }
            for (int worker = RETIRE; worker < WORKERS; worker++) {
                String owner = "worker-" + worker;
                for (int n = 0; n < changed; n++) {
                    assertTrue(index.batchCalculatePrefixMatchLength(List.of(owner),
                            List.of(oldKey(worker, selected[worker][n]))).isEmpty());
                    assertEquals(Map.of(owner, 1), index.batchCalculatePrefixMatchLength(List.of(owner),
                            List.of(newKey(worker, n, changed))));
                }
            }
        }
    }
    private static long[] collect(List<Future<Long>> jobs) throws Exception {
        long[] result = new long[jobs.size()];
        for (int i = 0; i < result.length; i++) result[i] = jobs.get(i).get(800, TimeUnit.SECONDS);
        Arrays.sort(result);
        return result;
    }
    private static double percentile(long[] sorted, double p) {
        return ms(sorted[(int) Math.ceil(sorted.length * p) - 1]);
    }
    private static double ms(long nanos) { return nanos / 1_000_000.0; }
    private static long oldKey(int worker, int offset) { return mix((long) worker * KEYS + offset); }
    private static long newKey(int worker, int offset, int changed) {
        return mix((long) WORKERS * KEYS + (long) worker * changed + offset);
    }
    private static long mix(long value) {
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }
}
