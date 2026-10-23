package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class PrefillQueueSheddingTest {
    private PrefillQueueShedding.Candidate<String> candidate(String id, int priority, long sequence,
            long input, long hit, long work, int workers) {
        return new PrefillQueueShedding.Candidate<>(id, priority, sequence,
                new PrefillQueueShedding.Estimate(input, hit, work, workers));
    }

    @Test void evictsLowerHitRateEvenWhenItHasMoreHitTokens() {
        var queue = List.of(candidate("short", 50, 1, 10_000, 9_000, 1_000, 1),
                candidate("long", 50, 2, 100_000, 80_000, 1_000, 1));
        assertEquals(List.of("long"), PrefillQueueShedding.victims(queue, 1_000));
        assertEquals(2, queue.size());
    }

    @Test void priorityDominatesCacheAndTiesKeepOlderRequests() {
        var queue = List.of(candidate("high", 90, 1, 100, 0, 1_000, 1),
                candidate("low-old", 10, 2, 100, 99, 1_000, 1),
                candidate("low-new", 10, 3, 1_000, 990, 1_000, 1));
        assertEquals(List.of("low-new", "low-old"), PrefillQueueShedding.victims(queue, 1_000));
    }

    @Test void onlyShedsEnoughWorkAndUsesFleetCapacity() {
        var queue = List.of(candidate("cold", 50, 1, 100, 0, 20_000, 2),
                candidate("warm", 50, 2, 100, 90, 2_000, 2));
        assertTrue(PrefillQueueShedding.victims(queue, 11_000).isEmpty());
        assertEquals(List.of("cold"), PrefillQueueShedding.victims(queue, 10_999));
        assertTrue(PrefillQueueShedding.victims(List.of(), 1).isEmpty());
    }

    @Test void explicitPriorityThenColdThenAscendingHitOrder() {
        var queue = List.of(candidate("high-warm", 90, 1, 1000, 900, 1000, 1),
                candidate("normal-low-hit", 50, 2, 1000, 100, 1000, 1),
                candidate("normal-cold", 50, 3, 100, 0, 1000, 1),
                candidate("low-hot", 10, 4, 10000, 9990, 1000, 1));
        assertEquals(List.of("low-hot", "normal-cold", "normal-low-hit"), PrefillQueueShedding.victims(queue, 1000));
    }

    @Test void invalidCacheFactsAreRejectedAndEmptyInputsHaveZeroRate() {
        assertThrows(IllegalArgumentException.class, () -> new PrefillQueueShedding.Estimate(1, 2, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new PrefillQueueShedding.Estimate(1, 0, 1, 0));
        assertEquals(0.0, new PrefillQueueShedding.Estimate(0, 0, 0, 1).hitRate());
    }

    @Test void automaticSoftSheddingDoesNotRejectNewBackgroundOrSacrificeBetterCacheInstead() {
        var config = new org.flexlb.config.PrefillQueueSheddingConfig();
        config.setExpectedWaitThresholdMs(3_000);
        config.setMaxWaitMs(6_000);
        var queue = List.of(candidate("young-cold", 50, 2, 100, 0, 2_000, 1),
                candidate("older-warm", 50, 1, 100, 90, 2_000, 1));
        assertTrue(PrefillQueueShedding.adaptiveVictims(queue, config, null,
                id -> id.equals("young-cold") ? 100 : 4_000).isEmpty());
        assertEquals(List.of("young-cold"), PrefillQueueShedding.adaptiveVictims(queue, config, null,
                id -> 3_000));
    }

    @Test void hardOverloadDoesNotWaitForYoungRequestsOrTrafficGrace() {
        var config = new org.flexlb.config.PrefillQueueSheddingConfig();
        config.setExpectedWaitThresholdMs(3_000);
        config.setMaxWaitMs(6_000);
        var queue = List.of(candidate("cold", 50, 1, 100, 0, 7_000, 1));
        var probing = new PrefillTrafficTrend.Snapshot(PrefillTrafficTrend.State.BURST_PROBE,
                0, 0, 0, 0, 7_000, true, 3_000);
        assertEquals(List.of("cold"), PrefillQueueShedding.adaptiveVictims(queue, config, probing, id -> 0));
        assertEquals(List.of("cold"), PrefillQueueShedding.adaptiveVictims(queue, config, null, id -> 0));
    }

    @Test void explicitLegacyRecoveryRetainsItsSheddingPolicy() {
        var config = new org.flexlb.config.PrefillQueueSheddingConfig();
        config.setExpectedWaitThresholdMs(3_000);
        config.setMaxWaitMs(6_000);
        config.getRecovery().setWindowMs(1_000);
        var queue = List.of(candidate("cold", 50, 1, 100, 0, 4_000, 1));
        assertEquals(List.of("cold"), PrefillQueueShedding.adaptiveVictims(queue, config, null, id -> 0));
        config.setRecovery(null);
        assertEquals(List.of("cold"), PrefillQueueShedding.adaptiveVictims(queue, config, null, id -> 0));
    }
}
