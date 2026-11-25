package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillTokenWorkloadTest {
    @Test void abandonedPlansReleaseExactlyOnceWithoutConsumingHardCapacity() {
        var work = new PrefillTokenWorkload();
        var plan = work.plan(1000);
        assertEquals(1000, work.snapshot().score());
        assertEquals(0, work.snapshot().pendingTokens());
        plan.close(); plan.close();
        assertEquals(0, work.snapshot().score());
    }

    @Test void concurrentSelectionsFromOneSnapshotReserveExactlyOnePlan() throws Exception {
        var ledger = new PrefillTokenWorkload();
        var start = new CyclicBarrier(2);
        Callable<PrefillTokenWorkload.Reservation> attempt = () -> {
            var snapshot = ledger.snapshot();
            start.await(10, TimeUnit.SECONDS);
            return ledger.tryPlan(snapshot.admissionVersion(), 700L, 5L);
        };
        var executor = Executors.newFixedThreadPool(2);
        try {
            var first = executor.submit(attempt);
            var second = executor.submit(attempt);
            var a = first.get(10, TimeUnit.SECONDS);
            var b = second.get(10, TimeUnit.SECONDS);
            assertTrue((a == null) != (b == null));
            assertEquals(700L, ledger.snapshot().plannedTokens());
            assertEquals(1L, ledger.snapshot().plannedRequests());
            assertEquals(5L, ledger.snapshot().plannedDurationMs());
            (a != null ? a : b).close();
            assertEquals(0L, ledger.snapshot().score());
        } finally {
            executor.shutdownNow();
        }
    }

    @Test void validatesRequestCountEvenForFreeMigrations() {
        var ledger = new PrefillTokenWorkload();
        var snapshot = ledger.snapshot();
        ledger.updateObservedRequests(4L);
        assertNull(ledger.tryPlan(snapshot.admissionVersion(), 0L, 0L));
        snapshot = ledger.snapshot();
        try (var first = ledger.tryPlan(snapshot.admissionVersion(), 0L, 0L)) {
            assertNotNull(first);
            assertNull(ledger.tryPlan(snapshot.admissionVersion(), 0L, 0L));
        }
        assertNull(ledger.tryPlan(snapshot.admissionVersion(), 0L, 0L), "a rolled-back plan still consumed its version");
        try (var second = ledger.tryPlan(ledger.snapshot().admissionVersion(), 0L, 0L)) {
            assertNotNull(second);
        }
    }

    @Test void accumulatedPredictionsSaturateAndReturnToExactRemainingDuration() {
        var ledger = new PrefillTokenWorkload();
        var large = ledger.tryPlan(-1L, 1L, Long.MAX_VALUE);
        try (var small = ledger.tryPlan(-1L, 1L, 10L)) {
            assertEquals(Long.MAX_VALUE, ledger.snapshot().plannedDurationMs());
            large.close();
            assertEquals(10L, ledger.snapshot().plannedDurationMs());
        }
        assertEquals(0L, ledger.snapshot().plannedDurationMs());
    }

    @Test void promotionPublishesOnlyCanonicalWorkAndRetiresOnlyItsOwnPlan() {
        var ledger = new PrefillTokenWorkload();
        var plan = ledger.tryPlan(-1L, 120L, 2L);
        var other = ledger.tryPlan(-1L, 80L, 3L);
        var stale = ledger.snapshot();
        ledger.accept(120L, plan);
        plan.close();
        assertEquals(200L, ledger.snapshot().score());
        assertEquals(3L, ledger.snapshot().plannedDurationMs());
        assertEquals(1L, ledger.snapshot().plannedRequests());
        assertNull(ledger.tryPlan(stale.admissionVersion(), 5L, 1L));
        assertThrows(IllegalArgumentException.class, () -> new PrefillTokenWorkload().accept(80L, other));
        other.close();
        ledger.release(120L);
        assertEquals(0L, ledger.snapshot().score());
    }

    @Test void completionAndCancellationImmediatelyRemoveWorkFromScore() {
        var work = new PrefillTokenWorkload();
        work.accept(1000);
        work.accept(250);
        try (var plan = work.plan(80)) {
            assertEquals(1330, work.snapshot().score());
            work.release(1000);
            assertEquals(330, work.snapshot().score());
            work.release(250);
            assertEquals(80, work.snapshot().score());
        }
        assertEquals(0L, work.snapshot().score(),
                "a worker that completed work must rank like an equally idle fresh worker");
        assertThrows(IllegalArgumentException.class, () -> work.release(1));
    }

    @Test void randomizedOutstandingReservationsMatchIndependentInventory() {
        var work = new PrefillTokenWorkload();
        var active = new ArrayList<Long>();
        var random = new Random(7);
        for (int i = 0; i < 10000; i++) {
            if (active.isEmpty() || random.nextBoolean()) {
                long tokens = random.nextInt(100000);
                work.accept(tokens);
                active.add(tokens);
            } else {
                work.release(active.remove(random.nextInt(active.size())));
            }
            long expected = active.stream().mapToLong(Long::longValue).sum();
            assertEquals(expected, work.snapshot().pendingTokens());
            assertEquals((double) expected, work.snapshot().score());
        }
        active.forEach(work::release);
        assertEquals(0, work.snapshot().score());
    }
    @Test void strictThresholdAndColdRequestsIncludeZeroHitWorkers() {
        assertFalse(PrefillTokenBalance.cacheFirst(500, 1000, 50));
        assertTrue(PrefillTokenBalance.cacheFirst(501, 1000, 50));
        assertFalse(PrefillTokenBalance.cacheFirst(0, 0, 0));
        assertFalse(PrefillTokenBalance.eligible(400, 900, true));
        assertTrue(PrefillTokenBalance.eligible(900, 900, true));
        assertTrue(PrefillTokenBalance.eligible(200, 400, false));
        assertTrue(PrefillTokenBalance.eligible(0, 400, false));
        assertTrue(PrefillTokenBalance.eligible(0, 0, false));
    }
}
