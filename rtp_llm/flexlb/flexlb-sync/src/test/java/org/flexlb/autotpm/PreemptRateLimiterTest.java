package org.flexlb.autotpm;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link PreemptRateLimiter} (guardrail D8).
 */
class PreemptRateLimiterTest {

    private static final String EP_A = "10.0.0.1:8080";
    private static final String EP_B = "10.0.0.2:8080";

    /** Mutable injected clock frozen inside one window unless advanced. */
    private final AtomicLong clock = new AtomicLong(1_000_000L);

    private PreemptRateLimiter limiter(int globalLimitPerMin, int endpointQpsLimit) {
        return new PreemptRateLimiter(globalLimitPerMin, endpointQpsLimit, clock::get);
    }

    // ---- global limit exactness ----

    @Test
    void globalLimit_exactlyNPass_thenN1Rejected() {
        PreemptRateLimiter limiter = limiter(5, 0); // endpoint layer unlimited

        for (int i = 0; i < 5; i++) {
            assertTrue(limiter.tryAcquire(EP_A), "acquire #" + (i + 1) + " must pass");
        }
        assertFalse(limiter.tryAcquire(EP_A), "acquire #6 must be rejected");
        assertEquals(5, limiter.globalCount());
    }

    @Test
    void globalLimitZero_neverAdmits() {
        PreemptRateLimiter limiter = limiter(0, 10);

        assertFalse(limiter.tryAcquire(EP_A));
        assertEquals(0, limiter.globalCount());
    }

    // ---- endpoint limit exactness ----

    @Test
    void endpointLimit_exact_andIndependentPerEndpoint() {
        PreemptRateLimiter limiter = limiter(100, 3);

        for (int i = 0; i < 3; i++) {
            assertTrue(limiter.tryAcquire(EP_A), "endpoint acquire #" + (i + 1) + " must pass");
        }
        assertFalse(limiter.tryAcquire(EP_A), "endpoint acquire #4 must be rejected");
        assertEquals(3, limiter.endpointCount(EP_A));
        // another endpoint still has its own budget
        assertTrue(limiter.tryAcquire(EP_B));
        assertEquals(1, limiter.endpointCount(EP_B));
    }

    // ---- window roll-over ----

    @Test
    void windowRollOver_resetsBothLayers() {
        PreemptRateLimiter limiter = limiter(2, 1);

        assertTrue(limiter.tryAcquire(EP_A));
        assertFalse(limiter.tryAcquire(EP_A)); // endpoint 1s window full

        clock.addAndGet(1_000L); // next endpoint window, same global window
        assertTrue(limiter.tryAcquire(EP_A));
        assertFalse(limiter.tryAcquire(EP_B)); // global 60s window now full

        clock.addAndGet(60_000L); // next global window
        assertTrue(limiter.tryAcquire(EP_B));
        assertEquals(1, limiter.globalCount());
    }

    // ---- rollback ----

    @Test
    void rollback_returnsPermit_acquirableAgain() {
        PreemptRateLimiter limiter = limiter(1, 1);

        assertTrue(limiter.tryAcquire(EP_A));
        assertFalse(limiter.tryAcquire(EP_A));

        limiter.rollback(EP_A);
        assertEquals(0, limiter.globalCount());
        assertEquals(0, limiter.endpointCount(EP_A));
        assertTrue(limiter.tryAcquire(EP_A), "permit must be reusable after rollback");
    }

    @Test
    void rollback_onEmptyState_neverGoesNegative() {
        PreemptRateLimiter limiter = limiter(5, 5);

        limiter.rollback(EP_A); // nothing acquired yet
        assertEquals(0, limiter.globalCount());
        assertEquals(0, limiter.endpointCount(EP_A));
        // budget unaffected: still exactly 5 permits
        for (int i = 0; i < 5; i++) {
            assertTrue(limiter.tryAcquire(EP_A + ":" + i));
        }
        assertFalse(limiter.tryAcquire(EP_B));
    }

    // ---- endpoint failure rolls back the global permit ----

    @Test
    void endpointRefused_globalPermitRolledBack() {
        PreemptRateLimiter limiter = limiter(10, 1);

        assertTrue(limiter.tryAcquire(EP_A));
        assertEquals(1, limiter.globalCount());

        // endpoint window full → global permit must be returned
        assertFalse(limiter.tryAcquire(EP_A));
        assertEquals(1, limiter.globalCount());

        // the returned budget stays usable by other endpoints: 9 more permits
        for (int i = 0; i < 9; i++) {
            assertTrue(limiter.tryAcquire("ep-" + i), "global budget acquire #" + (i + 1));
        }
        assertFalse(limiter.tryAcquire(EP_B));
    }

    // ---- concurrency stress: counts never negative ----

    @Test
    void concurrentStress_100Threads_countsNeverNegative() throws InterruptedException {
        PreemptRateLimiter limiter = limiter(50, 10);
        List<String> endpoints = List.of(EP_A, EP_B, "10.0.0.3:8080", "10.0.0.4:8080");
        int threads = 100;
        int iterations = 200;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        AtomicBoolean negativeSeen = new AtomicBoolean(false);

        for (int t = 0; t < threads; t++) {
            pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < iterations; i++) {
                        String endpoint = endpoints.get(ThreadLocalRandom.current().nextInt(endpoints.size()));
                        boolean acquired = limiter.tryAcquire(endpoint);
                        if (limiter.globalCount() < 0 || limiter.endpointCount(endpoint) < 0) {
                            negativeSeen.set(true);
                        }
                        if (acquired) {
                            limiter.rollback(endpoint);
                        }
                        if (limiter.globalCount() < 0 || limiter.endpointCount(endpoint) < 0) {
                            negativeSeen.set(true);
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
        assertTrue(done.await(60, TimeUnit.SECONDS), "stress must finish in time");
        pool.shutdownNow();

        assertFalse(negativeSeen.get(), "count must never be observed negative");
        // every successful acquire was rolled back → all counters land on 0
        assertEquals(0, limiter.globalCount());
        for (String endpoint : endpoints) {
            assertEquals(0, limiter.endpointCount(endpoint), "endpoint " + endpoint);
        }
    }
}
