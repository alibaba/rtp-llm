package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;
import java.util.Random;
import java.util.ArrayList;
import static org.junit.jupiter.api.Assertions.*;

class PrefillTokenWorkloadTest {
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
    @Test void strictThresholdAndWeakCacheNeverSendToZeroHitWorker() {
        assertFalse(PrefillTokenBalance.cacheFirst(500, 1000, 50));
        assertTrue(PrefillTokenBalance.cacheFirst(501, 1000, 50));
        assertFalse(PrefillTokenBalance.cacheFirst(0, 0, 0));
        assertFalse(PrefillTokenBalance.eligible(400, 900, true));
        assertTrue(PrefillTokenBalance.eligible(900, 900, true));
        assertTrue(PrefillTokenBalance.eligible(200, 400, false));
        assertFalse(PrefillTokenBalance.eligible(0, 400, false));
        assertTrue(PrefillTokenBalance.eligible(0, 0, false));
    }
}
