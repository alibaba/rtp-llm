package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillTokenWorkloadTest {
    @Test void coldWindowBudgetIncludesCompletedAndPlannedWorkAndRecoversAfterExpiry() {
        var now = new java.util.concurrent.atomic.AtomicLong(1000);
        var work = new PrefillTokenWorkload(10_000, now::get);
        assertFalse(work.hasRecentCacheHit());
        work.complete(80, true);
        assertTrue(work.hasRecentCacheHit());
        assertTrue(PrefillTokenBalance.withinColdWindow(work.snapshot(), 20, 100L));
        try (var plan = work.plan(1)) {
            assertFalse(PrefillTokenBalance.withinColdWindow(work.snapshot(), 20, 100L));
        }
        assertTrue(PrefillTokenBalance.withinColdWindow(work.snapshot(), 20, 100L));
        assertFalse(PrefillTokenBalance.withinColdWindow(work.snapshot(), 101, 100L));
        assertTrue(PrefillTokenBalance.withinColdWindow(work.snapshot(), 1000, null));
        now.set(11_000);
        assertFalse(work.hasRecentCacheHit());
        assertTrue(PrefillTokenBalance.withinColdWindow(work.snapshot(), 101, 100L), "oversized cold work must eventually start when idle");
        work.accept(101);
        assertFalse(PrefillTokenBalance.withinColdWindow(work.snapshot(), 1, 100L));
        work.release(101);
        assertTrue(PrefillTokenBalance.withinColdWindow(work.snapshot(), 100, 100L));
    }

    @Test void disabledWindowNeverReadsClockOrRetainsCompletions() {
        var work = new PrefillTokenWorkload(0, () -> { throw new AssertionError("window disabled"); });
        work.accept(100);
        work.release(100);
        work.complete(900);
        assertEquals(0, work.snapshot().score());
    }

    @Test void enabledWindowExpiresCompletedWorkWithoutChangingAdmission() {
        var now = new java.util.concurrent.atomic.AtomicLong(1000);
        var work = new PrefillTokenWorkload(10_000, now::get);
        work.accept(100);
        work.complete(700);
        now.set(2000);
        work.complete(300);
        assertEquals(1100, work.snapshot().score());
        assertEquals(100, work.snapshot().pendingTokens());
        now.set(10_999);
        assertEquals(1000, work.snapshot().completedTokens());
        now.set(11_000);
        assertEquals(400, work.snapshot().score());
        now.set(12_000);
        assertEquals(100, work.snapshot().score());
    }

    @Test void abandonedPlansReleaseExactlyOnceWithoutConsumingHardCapacity() {
        var work = new PrefillTokenWorkload();
        var plan = work.plan(1000);
        assertEquals(1000, work.snapshot().score());
        assertEquals(0, work.snapshot().pendingTokens());
        plan.close(); plan.close();
        assertEquals(0, work.snapshot().score());
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
        assertEquals(new PrefillTokenWorkload().snapshot(), work.snapshot(),
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
