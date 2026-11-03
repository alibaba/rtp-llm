package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/** Closed-loop queue/compute model using the production controller, with a deterministic virtual clock. */
class PrefillColdRequestProtectionSimulationTest {
    private static final class Work {
        final long id;
        final long at;
        final int hit;
        int remaining;
        Work(long id, long at, boolean hot) {
            this.id = id; this.at = at; this.hit = hot ? 900 : 0; this.remaining = 1000 - hit;
        }
    }
    private record Result(long done, long hard, long lateHard, long soft, long youngSoft,
                          long busyTokens, int remaining, PrefillColdRequestProtection.State state) { }

    private Result run(long seed, boolean enabled, boolean mixed, boolean recovery) {
        int seconds = Integer.getInteger("flexlb.protectionSimulationSeconds", 1800);
        var config = new PrefillQueueSheddingConfig();
        config.setExpectedWaitThresholdMs(60_000); config.setMaxWaitMs(120_000);
        config.setSustainedOverloadProtection(enabled);
        var controller = new PrefillQueueShedding.Controller();
        var waiting = new ArrayList<Work>();
        var engine = new ArrayDeque<Work>();
        var arrivals = new Random(seed);
        var shapes = new Random(seed + 1);
        long id = 0, done = 0, hard = 0, lateHard = 0, soft = 0, youngSoft = 0, busyTokens = 0;
        double next = 0;
        for (int second = 0; second < seconds; second++) {
            long now = second * 1000L;
            for (var iterator = waiting.iterator(); iterator.hasNext();) {
                var request = iterator.next();
                if (now - request.at >= 100_000) {
                    iterator.remove(); hard++;
                    if (second >= 600) { lateHard++; }
                    if (enabled) { controller.hardTimeout(now); }
                }
            }
            double hotFraction = mixed ? .5 : 0;
            double load = recovery ? second < 300 ? 2 : second < 1200 ? 8 : .7 : mixed ? 8 : 4;
            double qps = load * 20_000 / (1000 - 900 * hotFraction);
            while (next <= now) {
                var request = new Work(++id, (long) next, shapes.nextDouble() < hotFraction);
                waiting.add(request); controller.arrive();
                if (request.hit == 0) { controller.coldArrival(); }
                next += -Math.log1p(-arrivals.nextDouble()) * 1000 / qps;
            }
            long pending = engine.stream().mapToLong(q -> 1000 - q.hit).sum();
            int take = 0;
            while (take < waiting.size() && engine.size() < 256 && pending < 400_000) {
                var request = waiting.get(take++);
                engine.addLast(request); pending += 1000 - request.hit;
            }
            if (take > 0) { waiting.subList(0, take).clear(); }
            var candidates = new ArrayList<PrefillQueueShedding.Candidate<Work>>(waiting.size());
            for (var request : waiting) {
                candidates.add(new PrefillQueueShedding.Candidate<>(request, 50, request.id,
                        new PrefillQueueShedding.Estimate(1000, request.hit, 1000 - request.hit, 2), now - request.at));
            }
            var victims = PrefillQueueShedding.adaptiveVictims(candidates, config, null, controller, now);
            var removed = new HashSet<>(victims);
            waiting.removeIf(removed::contains);
            controller.recordEvictions(now, victims.size(), victims.size());
            soft += victims.size();
            for (var victim : victims) {
                if (now - victim.at <= 60_000) { assertEquals(0, victim.hit); youngSoft++; }
            }
            int budget = 20_000;
            while (budget > 0 && !engine.isEmpty()) {
                var request = engine.getFirst();
                int used = Math.min(budget, request.remaining);
                budget -= used; request.remaining -= used;
                if (request.remaining == 0) { engine.removeFirst(); done++; }
            }
            if (!recovery && second >= 60) {
                assertEquals(0, budget, "overloaded fleet must remain busy, seed=" + seed + " second=" + second);
            }
            busyTokens += 20_000 - budget;
            assertEquals(id, done + hard + soft + waiting.size() + engine.size(), "request conservation");
        }
        var result = new Result(done, hard, lateHard, soft, youngSoft, busyTokens,
                waiting.size() + engine.size(), controller.protectionSnapshot().state());
        System.out.printf("cold-protection simulation seed=%d enabled=%s mixed=%s recovery=%s seconds=%d %s%n",
                seed, enabled, mixed, recovery, seconds, result);
        return result;
    }

    @ParameterizedTest
    @ValueSource(longs = {41, 97, 133})
    void sustainedColdOverloadKeepsCapacityBusyAndStopsRecurringHardTimeouts(long seed) {
        var baseline = run(seed, false, false, false);
        var protectedRun = run(seed, true, false, false);
        assertTrue(baseline.hard > 0);
        assertTrue(protectedRun.hard < baseline.hard / 2);
        assertEquals(0, protectedRun.lateHard);
        assertTrue(protectedRun.youngSoft > 0);
        assertEquals(baseline.busyTokens, protectedRun.busyTokens);
        assertEquals(baseline.done, protectedRun.done);
    }

    @Test void mixedOverloadProtectsSuccessfulWorkWithoutLeavingIdleCompute() {
        var baseline = run(41, false, true, false);
        var protectedRun = run(41, true, true, false);
        assertTrue(protectedRun.hard < baseline.hard / 2);
        assertEquals(0, protectedRun.lateHard);
        assertTrue(protectedRun.done >= baseline.done);
        assertEquals(baseline.busyTokens, protectedRun.busyTokens);
    }

    @Test void lowerArrivalsDrainTheQueueAndRestoreGrace() {
        var result = run(97, true, true, true);
        assertTrue(result.youngSoft > 0);
        assertEquals(0, result.remaining);
        assertEquals(PrefillColdRequestProtection.State.NORMAL, result.state);
    }
}
