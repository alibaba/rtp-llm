package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;
import java.util.Random;
import java.util.ArrayList;
import static org.junit.jupiter.api.Assertions.*;

class PrefillTokenWorkloadTest {
    @Test void abandonedPlansReleaseExactlyOnceWithoutConsumingHardCapacity() {
        var work = new PrefillTokenWorkload(100);
        var plan = work.plan(1000);
        assertEquals(1000, work.snapshot(0).score());
        assertEquals(0, work.snapshot(0).pendingTokens());
        plan.close(); plan.close();
        assertEquals(0, work.snapshot(0).score());
    }

    @Test void completionWindowAndCancellationConserveWork() {
        var w = new PrefillTokenWorkload(100);
        w.accept(1000); w.accept(250);
        w.release(1000); w.complete(1000, 10);
        assertEquals(250, w.snapshot(109).score());
        assertEquals(1000, w.snapshot(109).completedTokens());
        w.release(250);
        assertEquals(0, w.snapshot(109).score());
        assertEquals(0, w.snapshot(110).score());
        assertThrows(IllegalArgumentException.class, () -> w.release(1));
    }
    @Test void randomSlidingWindowMatchesIndependentReference() {
        var w = new PrefillTokenWorkload(97);
        var reference = new ArrayList<long[]>();
        var random = new Random(7);
        long now = 0;
        for (int i = 0; i < 10000; i++) {
            now += random.nextInt(5);
            long tokens = random.nextInt(100000);
            w.accept(tokens); w.release(tokens); w.complete(tokens, now);
            reference.add(new long[]{now, tokens});
            long cutoff = now - 97;
            reference.removeIf(c -> c[0] <= cutoff);
            assertEquals(reference.stream().mapToLong(c -> c[1]).sum(), w.snapshot(now).completedTokens());
            assertEquals(0, w.snapshot(now).pendingTokens());
        }
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
