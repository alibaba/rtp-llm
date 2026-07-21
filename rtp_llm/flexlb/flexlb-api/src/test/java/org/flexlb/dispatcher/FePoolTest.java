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
