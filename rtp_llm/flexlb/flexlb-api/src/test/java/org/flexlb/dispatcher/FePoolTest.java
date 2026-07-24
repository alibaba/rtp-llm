package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.dispatcher.DispatcherTestSupport.fePool;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FePoolTest {

    @Test
    void roundRobinsAcrossAddresses() {
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088"));
        assertEquals("http://a:8088", pool.next());
        assertEquals("http://b:8088", pool.next());
        assertEquals("http://a:8088", pool.next());
    }

    @Test
    void skipsDeadHostsPerPredicate() {
        FePool pool = fePool(
                () -> List.of("http://a:8088", "http://b:8088", "http://c:8088"),
                url -> !url.contains("b:"));
        for (int i = 0; i < 6; i++) {
            String picked = pool.next();
            assertNotEquals("http://b:8088", picked,
                    "host marked dead by predicate must never be returned");
        }
    }

    @Test
    void distributesEvenlyAcrossAliveHostsWhenOneIsDead() {
        // Skipping a dead host must not funnel its share onto the next host in line:
        // 12 picks over 3 alive hosts must land 4/4/4, not 4/8/... on b's successor.
        FePool pool = fePool(
                () -> List.of("http://a:8088", "http://b:8088", "http://c:8088", "http://d:8088"),
                url -> !url.contains("b:"));
        java.util.Map<String, Integer> counts = new java.util.HashMap<>();
        for (int i = 0; i < 12; i++) {
            counts.merge(pool.next(), 1, Integer::sum);
        }
        assertEquals(4, counts.get("http://a:8088"), "uneven RR after dead-host skip: " + counts);
        assertEquals(4, counts.get("http://c:8088"), "dead host's successor must not inherit its share: " + counts);
        assertEquals(4, counts.get("http://d:8088"), "uneven RR after dead-host skip: " + counts);
    }

    @Test
    void fallsBackToRoundRobinWhenAllDead() {
        FePool pool = fePool(
                () -> List.of("http://a:8088", "http://b:8088"),
                url -> false);
        String picked = pool.next();
        assertTrue(picked.startsWith("http://"),
                "all-dead fallback must still return a host, not refuse service");
    }

    @Test
    void readsDynamicSupplierOnEveryNext() {
        AtomicReference<List<String>> source = new AtomicReference<>(List.of("http://a:8088"));
        FePool pool = fePool(source::get, url -> true);
        assertEquals("http://a:8088", pool.next());

        source.set(List.of("http://b:8088", "http://c:8088"));
        // Pool must observe the new snapshot — not a cached copy from construction. Cursor is
        // shared, so the exact order across the swap depends on cumulative call count; only the
        // membership matters here.
        String first = pool.next();
        String second = pool.next();
        assertTrue(first.startsWith("http://b") || first.startsWith("http://c"),
                "post-swap call returned stale address: " + first);
        assertTrue(second.startsWith("http://b") || second.startsWith("http://c"),
                "post-swap call returned stale address: " + second);
        assertNotEquals(first, second, "two consecutive next() on a 2-host snapshot must alternate");
    }

    @Test
    void emptySupplierSnapshotThrowsOnNext() {
        FePool pool = fePool(List.of());
        assertThrows(IllegalStateException.class, pool::next);
    }

    @Test
    void nextBatchReturnsExactlyCountPicksInRoundRobinOrder() {
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088"));
        assertEquals(List.of("http://a:8088", "http://b:8088", "http://a:8088"), pool.nextBatch(3),
                "nextBatch must round-robin the same way as repeated next()");
    }

    @Test
    void nextBatchSkipsDeadHostsPerPredicate() {
        FePool pool = fePool(
                () -> List.of("http://a:8088", "http://b:8088", "http://c:8088"),
                url -> !url.contains("b:"));
        List<String> picks = pool.nextBatch(6);
        assertEquals(6, picks.size(), "nextBatch must return exactly count picks");
        for (String picked : picks) {
            assertNotEquals("http://b:8088", picked, "dead host must never appear in a nextBatch pick");
        }
    }

    @Test
    void nextBatchThrowsOnEmptySnapshotBeforeAnyPick() {
        // All-or-nothing per batch: an empty snapshot throws rather than returning a short list, so
        // MasterFeAssigner stamps no target instead of a prefix.
        FePool pool = fePool(List.of());
        assertThrows(IllegalStateException.class, () -> pool.nextBatch(3));
    }

    @Test
    void nextBatchWithNonPositiveCountReturnsEmptyWithoutResolvingTheSource() {
        // count <= 0 short-circuits before touching the supplier; an empty source (which would throw
        // if resolved) proves the short-circuit. Mutation guard: drop the count check and this throws.
        FePool pool = fePool(List.of());
        assertTrue(pool.nextBatch(0).isEmpty(), "nextBatch(0) must be empty");
        assertTrue(pool.nextBatch(-5).isEmpty(), "nextBatch(negative) must be empty");
    }

    @Test
    void nextBatchFallsBackToFullSnapshotWhenAllDead() {
        FePool pool = fePool(() -> List.of("http://a:8088", "http://b:8088"), url -> false);
        List<String> picks = pool.nextBatch(4);
        assertEquals(4, picks.size(), "all-dead fallback must still return count picks, not refuse service");
        for (String picked : picks) {
            assertTrue(picked.startsWith("http://"), "all-dead fallback must return real hosts: " + picked);
        }
    }

    @Test
    void nextAndNextBatchAdvanceTheSameCursor() {
        // next() and nextBatch() share one cursor, so a mixed sequence stays a single RR walk.
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088"));
        assertEquals("http://a:8088", pool.next());                                    // cursor 0 -> 1
        assertEquals(List.of("http://b:8088", "http://a:8088"), pool.nextBatch(2));     // base 1 -> 3
        assertEquals("http://b:8088", pool.next());                                    // cursor 3 -> 4
    }

    @Test
    void nextBatchContinuesRoundRobinAcrossCallsWhenCountNotMultipleOfPoolSize() {
        // count (3) is not a multiple of pool size (2), so the second batch must resume where the
        // first left off on the shared cursor, not restart at index 0. Mutation guard: reset base
        // per call (or a wrong +i offset) and the second list stops matching.
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088"));
        assertEquals(List.of("http://a:8088", "http://b:8088", "http://a:8088"), pool.nextBatch(3),
                "first batch: cursor 0 -> 3");
        assertEquals(List.of("http://b:8088", "http://a:8088", "http://b:8088"), pool.nextBatch(3),
                "second batch must resume at cursor 3, not restart at 0");
    }

    @Test
    void nextBatchToleratesCursorIntWrapPastMaxValue() throws Exception {
        // The rotation cursor is an int advanced by getAndAdd(count); a long-lived, high-fanout pool
        // eventually wraps past Integer.MAX_VALUE into negative. The class comment claims floorMod
        // keeps every pick a valid index — pin it by seeding the cursor just below MAX_VALUE so the
        // batch crosses the wrap mid-list. Mutation guard: swap floorMod for '%' and the negative
        // operand yields a negative index -> IndexOutOfBoundsException here.
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088", "http://c:8088"));
        java.lang.reflect.Field cursorField = FePool.class.getDeclaredField("cursor");
        cursorField.setAccessible(true);
        ((AtomicInteger) cursorField.get(pool)).set(Integer.MAX_VALUE - 1);

        List<String> picks = pool.nextBatch(4); // base = MAX_VALUE-1, spans the wrap to MIN_VALUE
        assertEquals(4, picks.size(), "int wrap must not drop or duplicate picks");
        List<String> hosts = List.of("http://a:8088", "http://b:8088", "http://c:8088");
        for (String picked : picks) {
            assertTrue(hosts.contains(picked), "int wrap must not produce an out-of-range pick: " + picked);
        }
    }

    @Test
    void concurrentNextKeepsCursorAtomicSoDistributionStaysBalanced() throws InterruptedException {
        // The rotation cursor is an AtomicInteger, so under any thread interleaving the multiset
        // of slot indices handed out is exactly {0 .. total-1}; floorMod over a fixed alive-set
        // size then yields a perfectly balanced histogram. A non-atomic cursor would lose
        // increments under contention and skew the counts. Snapshot and predicate are fixed so
        // alive.size() is constant, making this an exact (non-flaky) assertion for every schedule.
        List<String> hosts = List.of("http://a:8088", "http://b:8088", "http://c:8088", "http://d:8088");
        FePool pool = fePool(() -> hosts, url -> true);

        int threads = 8;
        int picksPerThread = 1000; // total 8000, divisible by 4 hosts -> 2000 each
        Map<String, AtomicInteger> counts = new ConcurrentHashMap<>();
        for (String h : hosts) {
            counts.put(h, new AtomicInteger());
        }

        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        for (int t = 0; t < threads; t++) {
            executor.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < picksPerThread; i++) {
                        counts.get(pool.next()).incrementAndGet();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            });
        }
        start.countDown();
        assertTrue(done.await(30, TimeUnit.SECONDS), "worker threads did not finish in time");
        executor.shutdownNow();

        int expected = threads * picksPerThread / hosts.size();
        for (String h : hosts) {
            assertEquals(expected, counts.get(h).get(),
                    "atomic cursor must hand out each slot exactly once -> perfectly balanced; got " + counts);
        }
    }
}
