package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class PrefillQueueSheddingTest {
    private PrefillQueueShedding.Candidate<String> candidate(String id, int priority, long sequence,
            long input, long hit, long work, int workers, long waited) {
        return new PrefillQueueShedding.Candidate<>(id, priority, sequence,
                new PrefillQueueShedding.Estimate(input, hit, work, workers), waited);
    }

    private List<PrefillQueueShedding.Candidate<String>> cohort(int count, long age) {
        var queue = new ArrayList<PrefillQueueShedding.Candidate<String>>();
        for (int i = 0; i < count; i++) { queue.add(candidate("q" + i, 50, i, 100, 0, 0, 1, age)); }
        return queue;
    }

    @Test void arbitrarilyLargePredictedBacklogCannotEvictYoungRequests() {
        var queue = List.of(candidate("new", 1, 2, 100, 0, Long.MAX_VALUE, 1, 0),
                candidate("boundary", 1, 1, 100, 0, Long.MAX_VALUE, 1, 30_000));
        assertTrue(new PrefillQueueShedding.Controller().victims(queue, 30_000, 60_000, 0).isEmpty());
    }

    @Test void fractionalPressureAccumulatesOnlyAcrossRealTimeAndIgnoresPredictedWork() {
        var queue = List.of(candidate("old", 50, 1, 100, 90, 0, 1000, 30_001),
                candidate("fresh", 1, 2, 100, 0, Long.MAX_VALUE, 1, 29_999));
        var control = new PrefillQueueShedding.Controller();
        for (int sec = 0; sec < 29; sec++) {
            assertTrue(control.victims(queue, 30_000, 60_000, sec * 1000L).isEmpty());
        }
        assertEquals(List.of("old"), control.victims(queue, 30_000, 60_000, 29_000));
    }

    @Test void olderAndDeeperQueuesIncreaseBoundedEvictionRate() {
        assertEquals(3, new PrefillQueueShedding.Controller().victims(cohort(100, 30_001), 30_000, 60_000, 0).size());
        assertEquals(6, new PrefillQueueShedding.Controller().victims(cohort(100, 45_000), 30_000, 60_000, 0).size());
        assertEquals(10, new PrefillQueueShedding.Controller().victims(cohort(100, 57_000), 30_000, 60_000, 0).size());
        assertEquals(13, new PrefillQueueShedding.Controller().victims(cohort(200, 45_000), 30_000, 60_000, 0).size());
    }

    @Test void repeatedRoundsDoNotEmptyAnOverdueCohortInFiveSeconds() {
        var control = new PrefillQueueShedding.Controller();
        var queue = cohort(100, 45_000);
        for (int sec = 0; sec <= 5; sec++) {
            var victims = control.victims(queue, 30_000, 60_000, sec * 1000L);
            queue = queue.stream().filter(c -> !victims.contains(c.value())).map(c ->
                    new PrefillQueueShedding.Candidate<>(c.value(), c.priority(), c.sequence(), c.estimate(), c.waitedMs() + 1000)).toList();
        }
        assertTrue(queue.size() >= 60, "old proportional controller evicted all 100 requests");
    }

    @Test void naturalDrainPausesEvictionButOwnEvictionsDoNotMasqueradeAsRecovery() {
        var control = new PrefillQueueShedding.Controller();
        var queue = cohort(100, 45_000);
        var first = control.victims(queue, 30_000, 60_000, 0);
        queue.removeIf(c -> first.contains(c.value()));
        assertFalse(control.victims(queue, 30_000, 60_000, 1000).isEmpty());
        assertTrue(control.victims(cohort(70, 47_000), 30_000, 60_000, 2000).isEmpty());
    }

    @Test void scanFrequencyAndLongSchedulerPausesCannotMultiplyEvictions() {
        var control = new PrefillQueueShedding.Controller();
        assertEquals(6, control.victims(cohort(100, 45_000), 30_000, 60_000, 0).size());
        for (int ms = 1; ms < 1000; ms++) {
            assertTrue(control.victims(cohort(100, 45_000), 30_000, 60_000, ms).isEmpty());
        }
        assertTrue(control.victims(cohort(100, 45_000), 30_000, 60_000, 120_000).size() <= 7);
    }

    @Test void priorityThenCacheThenNewestSelectVictimsOnlyAmongOverdueRequests() {
        var queue = new ArrayList<>(List.of(candidate("high-cold", 90, 1, 100, 0, 0, 1, 59_000),
                candidate("low-hot", 10, 2, 100, 99, 0, 1, 59_000),
                candidate("normal-cold", 50, 3, 100, 0, 0, 1, 59_000),
                candidate("normal-warm-old", 50, 4, 100, 80, 0, 1, 59_000),
                candidate("normal-warm-new", 50, 5, 1000, 800, 0, 1, 59_000),
                candidate("normal-hot", 50, 6, 100, 90, 0, 1, 59_000),
                candidate("fresh-low", 1, 7, 100, 0, Long.MAX_VALUE, 1, 1)));
        var control = new PrefillQueueShedding.Controller();
        int second = 0;
        for (String expected : List.of("low-hot", "normal-cold", "normal-warm-new", "normal-warm-old", "normal-hot", "high-cold")) {
            var victims = control.victims(queue, 30_000, 60_000, second++ * 1000L);
            assertEquals(List.of(expected), victims);
            queue.removeIf(c -> victims.contains(c.value()));
        }
    }

    @Test void sustainedOverloadConvergesWithoutIdleCapacityOrMinuteScaleErrorWaves() {
        var control = new PrefillQueueShedding.Controller();
        record Arrival(long id, long at) { }
        var queue = new ArrayList<Arrival>();
        long id = 0;
        int tailDrops = 0;
        for (int sec = 0; sec < 900; sec++) {
            // 20% sustained overload, followed by a recovery below service capacity.
            int arrivals = sec < 600 ? 120 : 80;
            for (int i = 0; i < arrivals; i++) { queue.add(new Arrival(id++, sec)); }
            int served = Math.min(100, queue.size());
            if (sec < 600) { assertEquals(100, served, "capacity must stay fully utilized"); }
            queue.subList(0, served).clear();
            assertTrue(queue.isEmpty() || sec - queue.getFirst().at() < 60, "no hard timeout waves");
            var candidates = new ArrayList<PrefillQueueShedding.Candidate<Long>>();
            for (var q : queue) { candidates.add(new PrefillQueueShedding.Candidate<>(q.id(), 50, q.id(),
                    new PrefillQueueShedding.Estimate(100, 0, 0, 1), (sec - q.at()) * 1000)); }
            var victims = control.victims(candidates, 30_000, 60_000, sec * 1000L);
            var removed = new java.util.HashSet<>(victims);
            queue.removeIf(q -> removed.contains(q.id()));
            if (sec >= 300 && sec < 600) {
                assertTrue(victims.size() >= 18 && victims.size() <= 22, "stable rejection matches excess arrivals");
                tailDrops += victims.size();
            }
            if (sec >= 600) { assertTrue(victims.isEmpty(), "stop shedding while the real queue drains"); }
        }
        assertEquals(20.0, tailDrops / 300.0, 0.1);
        assertTrue(queue.isEmpty());
    }

    @Test void invalidCacheFactsAreRejectedAndEmptyInputsHaveZeroRate() {
        assertThrows(IllegalArgumentException.class, () -> new PrefillQueueShedding.Estimate(1, 2, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> new PrefillQueueShedding.Estimate(1, 0, 1, 0));
        assertEquals(0.0, new PrefillQueueShedding.Estimate(0, 0, 0, 1).hitRate());
    }
}
