package org.flexlb.cache.core;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.AbstractSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Real-map regression: retiring one worker must not monopolize unrelated cache updates.
 *
 * <p>Run from rtp_llm/flexlb:
 * ./mvnw -pl flexlb-cache -am -Pcache-performance-regression
 * -Dtest=CacheRetirementPerformanceTest -Dsurefire.failIfNoSpecifiedTests=false test
 *
 * <p>Optional: -Dretirement.blocks=4000000 -Dretirement.updateBudgetMs=1000.
 * The progress assertion is independent of machine speed; the latency budget is secondary.
 * This isolates index retirement versus unrelated writes. It does not model RPC parsing,
 * 100-worker full snapshots, startup discovery or same-address generation replacement.
 */
@Tag("performance-regression")
class CacheRetirementPerformanceTest {
    private static final int BLOCKS = Integer.getInteger("retirement.blocks", 4_000_000);
    private static final int WRITERS = 32;
    // A local regression budget, not a claim about the production SLA.
    private static final long BUDGET_MS = Long.getLong("retirement.updateBudgetMs", 1000L);

    @Test
    @Timeout(180)
    void unrelatedUpdatesWithoutRetirement() throws Exception {
        run(false);
    }

    @Test
    @Timeout(180)
    void retirementMustAllowUnrelatedUpdatesBeforeFinishingWholeSnapshot() throws Exception {
        run(true);
    }

    @Test
    @Timeout(180)
    void largeDisjointConcurrentAddsAndRemoves() throws Exception {
        GlobalCacheIndex index = new GlobalCacheIndex();
        int workers = 100;
        try (var executor = Executors.newFixedThreadPool(32)) {
            for (boolean remove : new boolean[]{false, true}) {
                var jobs = new ArrayList<java.util.concurrent.Future<?>>();
                CountDownLatch start = new CountDownLatch(1);
                long begin = System.nanoTime();
                for (int worker = 0; worker < workers; worker++) {
                    final int id = worker;
                    jobs.add(executor.submit(() -> {
                        assertTrue(start.await(10, TimeUnit.SECONDS));
                        String owner = "bulk-" + id;
                        for (long n = id; n < BLOCKS; n += workers) {
                            if (remove) index.removeCacheBlock(owner, key(n), null);
                            else index.addCacheBlock(key(n), owner, null);
                        }
                        return null;
                    }));
                }
                start.countDown();
                for (var job : jobs) job.get(120, TimeUnit.SECONDS);
                double elapsedMs = (System.nanoTime() - begin) / 1_000_000.0;
                System.out.printf("BULK remove=%s totalBlocks=%d workers=%d threads=32 elapsedMs=%.3f opsPerSec=%.0f%n",
                        remove, BLOCKS, workers, elapsedMs, BLOCKS * 1000.0 / elapsedMs);
                assertEquals(remove ? 0L : BLOCKS, index.totalMappings());
                assertEquals(remove ? 0L : BLOCKS, index.totalBlocks());
            }
        }
    }

    private void run(boolean retire) throws Exception {
        assertTrue(BLOCKS >= 100_000, "Use a large real snapshot, not a toy lock test");
        GlobalCacheIndex index = new GlobalCacheIndex();
        // Warm up the actual insert/delete path before preparing the measured snapshot.
        for (int i = 0; i < 20_000; i++) {
            long key = key(i);
            index.addCacheBlock(key, "warmup", null);
            index.removeCacheBlock("warmup", key, null);
        }
        for (int i = 0; i < BLOCKS; i++) {
            index.addCacheBlock(key(i), "retired", null);
        }
        // Shared ownership must survive retirement, independently of latency assertions.
        long shared = key(0);
        index.addCacheBlock(shared, "survivor", null);
        CountDownLatch ready = new CountDownLatch(WRITERS);
        CountDownLatch start = new CountDownLatch(1);
        AtomicInteger visited = new AtomicInteger();
        var executor = Executors.newFixedThreadPool(WRITERS + 1);
        try {
            var updates = new ArrayList<java.util.concurrent.Future<long[]>>();
            for (int i = 0; i < WRITERS; i++) {
                final int writer = i;
                updates.add(executor.submit(() -> {
                    ready.countDown();
                    assertTrue(start.await(30, TimeUnit.SECONDS), "cleanup never started");
                    long begin = System.nanoTime();
                    index.addCacheBlock(key((long) BLOCKS + writer), "active-" + writer, null);
                    long elapsed = System.nanoTime() - begin;
                    return new long[]{elapsed, visited.get()};
                }));
            }
            assertTrue(ready.await(30, TimeUnit.SECONDS), "writers not ready");
            long begin = System.nanoTime();
            var cleanup = executor.submit(() -> {
                if (retire) {
                    // The iterator only signals progress. No sleep, barrier wait, mocked map,
                    // artificial hash collisions or reflection into the production lock.
                    index.removeAllCacheBlockOfEngine("retired", new AbstractSet<Long>() {
                        @Override public int size() { return BLOCKS; }
                        @Override public Iterator<Long> iterator() {
                            return new Iterator<>() {
                                int position;
                                @Override public boolean hasNext() { return position < BLOCKS; }
                                @Override public Long next() {
                                    if (!hasNext()) throw new NoSuchElementException();
                                    long value = key(position++);
                                    visited.set(position);
                                    if (position == 1) start.countDown();
                                    return value;
                                }
                            };
                        }
                    }, null);
                } else {
                    start.countDown();
                }
                return System.nanoTime() - begin;
            });
            long[] latency = new long[WRITERS];
            int progressed = 0;
            for (int i = 0; i < WRITERS; i++) {
                long[] result = updates.get(i).get(120, TimeUnit.SECONDS);
                latency[i] = result[0];
                if (result[1] < BLOCKS) progressed++;
            }
            long cleanupNs = cleanup.get(120, TimeUnit.SECONDS);
            Arrays.sort(latency);
            double maxMs = latency[WRITERS - 1] / 1_000_000.0;
            System.out.printf("RETIREMENT retire=%s blocks=%d writers=%d cleanupMs=%.3f "
                            + "updateP50Ms=%.3f updateP99Ms=%.3f progressedBeforeLastKey=%d/%d%n",
                    retire, BLOCKS, WRITERS, cleanupNs / 1_000_000.0,
                    latency[WRITERS / 2] / 1_000_000.0, maxMs, progressed, WRITERS);
            assertEquals(Map.of("survivor", 1), index.batchCalculatePrefixMatchLength(List.of("survivor"), List.of(shared)));
            if (retire) assertTrue(index.batchCalculatePrefixMatchLength(List.of("retired"), List.of(shared)).isEmpty());
            for (int i = 0; i < WRITERS; i++) {
                assertEquals(Map.of("active-" + i, 1), index.batchCalculatePrefixMatchLength(
                        List.of("active-" + i), List.of(key((long) BLOCKS + i))));
            }
            assertEquals(retire ? WRITERS + 1L : BLOCKS + WRITERS + 1L, index.totalMappings());
            int completedDuringCleanup = progressed;
            assertAll(
                    () -> assertTrue(!retire || completedDuringCleanup > 0,
                            "All 32 unrelated writers waited until the entire retirement snapshot was scanned"),
                    () -> assertTrue(maxMs < BUDGET_MS,
                            "Unrelated update p99=" + maxMs + "ms exceeded " + BUDGET_MS + "ms"));
        } finally {
            start.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(30, TimeUnit.SECONDS), "test threads leaked");
        }
    }

    // SplitMix64's bijective finalizer: distinct, random-looking keys without storing boxed snapshots.
    private static long key(long value) {
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }
}
