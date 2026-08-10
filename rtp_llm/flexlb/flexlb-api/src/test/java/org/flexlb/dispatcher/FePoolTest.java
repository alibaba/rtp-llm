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
import java.util.concurrent.atomic.AtomicLong;
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
        // Exact RR over the *filtered* subset [a,c] — the only nextBatch path that round-robins a
        // subset whose size (2) differs from the snapshot size (3), so it pins that floorMod is
        // taken over alive.size() not snapshot.size(). Fresh cursor 0 over 2 alive hosts -> a,c
        // repeated. Mutation guard: a wrong modulus or a dropped filter changes this exact sequence,
        // not just leaks a dead host (which the old membership-only assertion would have missed).
        assertEquals(
                List.of("http://a:8088", "http://c:8088", "http://a:8088",
                        "http://c:8088", "http://a:8088", "http://c:8088"),
                pool.nextBatch(6),
                "nextBatch over a filtered subset must round-robin a,c,a,c,a,c");
    }

    @Test
    void nextBatchWithDeadFirstHostRoundRobinsTheAliveTail() {
        // Boundary of livePool's lazy materialize: when index 0 is dead the filtered list is seeded
        // from an empty all-alive prefix (the `for j in [0,0)` loop adds nothing) before appending
        // the alive tail. Alive subset is [b,c]; fresh cursor 0 -> b,c,b,c. Mutation guard: an
        // off-by-one in the prefix seed (e.g. `j <= i`) would wrongly include the dead first host.
        FePool pool = fePool(
                () -> List.of("http://a:8088", "http://b:8088", "http://c:8088"),
                url -> !url.contains("a:"));
        assertEquals(
                List.of("http://b:8088", "http://c:8088", "http://b:8088", "http://c:8088"),
                pool.nextBatch(4),
                "dead-first-host: empty-prefix seed then RR over the alive tail [b,c]");
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
        // Fallback must still be plain round-robin over the full snapshot, not a fixed host: a
        // fresh cursor at 0 over the two (dead-but-only) hosts yields the RR sequence a,b,a,b.
        // Mutation guard: return snapshot.get(0) (or any constant) in the all-dead branch and the
        // exact-sequence assertion fails where a weaker "startsWith(http://)" check would pass.
        List<String> picks = pool.nextBatch(4);
        assertEquals(List.of("http://a:8088", "http://b:8088", "http://a:8088", "http://b:8088"), picks,
                "all-dead fallback must round-robin the full snapshot, not funnel onto one host");
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
    void nextBatchUsesLongCursorSoTheIntBoundaryIsPlainRoundRobin() throws Exception {
        // The cursor is a long: seeding it just below Integer.MAX_VALUE and crossing that boundary
        // must stay a clean, continuous round-robin — a,b,c,a over 3 hosts — because a long does not
        // overflow there. This doubles as the guard that the cursor is NOT an int: an int cursor
        // would overflow at base+2 (2147483648 -> MIN_VALUE) and floorMod would yield b, giving the
        // discontinuous a,b,b,c instead. Mutation guard: revert cursor to int and this exact
        // sequence goes red.
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088", "http://c:8088"));
        java.lang.reflect.Field cursorField = FePool.class.getDeclaredField("cursor");
        cursorField.setAccessible(true);
        ((AtomicLong) cursorField.get(pool)).set(Integer.MAX_VALUE - 1L);

        assertEquals(
                List.of("http://a:8088", "http://b:8088", "http://c:8088", "http://a:8088"),
                pool.nextBatch(4),
                "long cursor: crossing the int boundary must stay continuous RR, not blip");
    }

    @Test
    void nextBatchToleratesCursorWrapPastLongMaxValue() throws Exception {
        // A long cursor never wraps at any realistic QPS (2^63 picks), but the class comment still
        // claims floorMod keeps every index valid even if it did — pin that at the long boundary.
        // Seed just below Long.MAX_VALUE so the batch crosses the wrap into negative. Mutation
        // guard: swap floorMod for '%' and the negative operand yields a negative index ->
        // IndexOutOfBoundsException here.
        FePool pool = fePool(List.of("http://a:8088", "http://b:8088", "http://c:8088"));
        java.lang.reflect.Field cursorField = FePool.class.getDeclaredField("cursor");
        cursorField.setAccessible(true);
        ((AtomicLong) cursorField.get(pool)).set(Long.MAX_VALUE - 1L);

        List<String> picks = pool.nextBatch(4); // base = Long.MAX_VALUE-1, spans the wrap to MIN_VALUE
        assertEquals(4, picks.size(), "long wrap must not drop or duplicate picks");
        List<String> hosts = List.of("http://a:8088", "http://b:8088", "http://c:8088");
        for (String picked : picks) {
            assertTrue(hosts.contains(picked), "long wrap must not produce an out-of-range pick: " + picked);
        }
    }

    @Test
    void concurrentNextKeepsCursorAtomicSoDistributionStaysBalanced() throws InterruptedException {
        // The rotation cursor is an AtomicLong, so under any thread interleaving the multiset
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

    @Test
    void concurrentNextBatchKeepsCursorAtomicSoDistributionStaysBalanced() throws InterruptedException {
        // What this pins: cursor atomicity + that each nextBatch advances the cursor by exactly
        // `count` under contention. Across all threads the reserved slot indices then tile
        // {0 .. total-1} with no gap or overlap, and floorMod over a fixed alive-set yields a
        // perfectly balanced histogram (exact, non-flaky). batchSize is deliberately NOT a multiple
        // of the host count: if the cursor failed to advance per batch (e.g. getAndAdd(count)
        // regressed to a non-advancing read) every batch would return the same prefix {0,1,2} ->
        // host d starves and a/b/c skew, so this goes red; a same-size-as-hosts batch would mask
        // that (every batch covers all hosts regardless of base) — the vacuous case to avoid.
        //
        // What this does NOT pin: per-batch internal contiguity (that ONE batch's picks are a
        // single contiguous RR run). An atomic per-pick getAndIncrement() loop would also produce a
        // balanced histogram and pass here. That per-batch-contiguity contract is pinned separately
        // by the exact-sequence tests above (nextBatchReturnsExactly.../...ContinuesRoundRobin...).
        List<String> hosts = List.of("http://a:8088", "http://b:8088", "http://c:8088", "http://d:8088");
        FePool pool = fePool(() -> hosts, url -> true);

        int threads = 8;
        int batchesPerThread = 250;
        int batchSize = 3; // total 8*250*3 = 6000 picks over {0..5999}, floorMod 4 -> 1500 each
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
                    for (int i = 0; i < batchesPerThread; i++) {
                        for (String picked : pool.nextBatch(batchSize)) {
                            counts.get(picked).incrementAndGet();
                        }
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

        int expected = threads * batchesPerThread * batchSize / hosts.size();
        for (String h : hosts) {
            assertEquals(expected, counts.get(h).get(),
                    "atomic cursor advanced by count-per-batch must hand out each slot exactly once -> perfectly balanced; got " + counts);
        }
    }
}
