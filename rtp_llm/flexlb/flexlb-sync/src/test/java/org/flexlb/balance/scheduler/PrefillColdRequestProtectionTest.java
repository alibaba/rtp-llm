package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.balance.scheduler.PrefillColdRequestProtection.State.*;

class PrefillColdRequestProtectionTest {
    private static void arrivals(PrefillColdRequestProtection policy, int count) {
        for (int i = 0; i < count; i++) { policy.arrive(); policy.coldArrival(); }
    }

    private static PrefillColdRequestProtection protecting() {
        var policy = new PrefillColdRequestProtection();
        for (long now = 0; now <= 120_000; now += 1_000) {
            arrivals(policy, 100);
            if (now % 60_000 == 0) { policy.hardTimeout(now); }
            assertEquals(0, policy.allowance(now));
        }
        assertEquals(PROTECTING, policy.snapshot().state());
        assertEquals(0, policy.snapshot().strength());
        return policy;
    }

    @Test void needsTwoCompleteWindowsAndAtLeastTwoMinutesSinceFirstHardTimeout() {
        var policy = new PrefillColdRequestProtection();
        policy.hardTimeout(59_000);
        policy.hardTimeout(61_000);
        policy.allowance(120_000);
        assertEquals(CONFIRMING, policy.snapshot().state(), "adjacent window edges cannot stand in for two minutes");
        policy.hardTimeout(121_000);
        policy.allowance(178_999);
        assertEquals(CONFIRMING, policy.snapshot().state(), "minimum duration has not elapsed");
        policy.allowance(179_000);
        assertEquals(PROTECTING, policy.snapshot().state());
        assertEquals(0, policy.snapshot().strength(), "entry cannot jump to full rejection");
    }

    @Test void aBurstOrAnEmptyMinuteCannotActivateProtection() {
        var burst = new PrefillColdRequestProtection();
        for (int i = 0; i < 1000; i++) { burst.hardTimeout(i); }
        for (long now = 1_000; now < 300_000; now += 1_000) { assertEquals(0, burst.allowance(now)); }
        assertEquals(NORMAL, burst.snapshot().state());
        var gap = new PrefillColdRequestProtection();
        gap.hardTimeout(0); gap.hardTimeout(120_000);
        gap.allowance(180_000);
        assertEquals(CONFIRMING, gap.snapshot().state());
    }

    @Test void rampsAtMostOneSecondPerRoundAndNeverBanksUnusedWholeCredits() {
        var policy = protecting();
        double previous = 0;
        for (long now = 121_000; now <= 180_000; now += 1_000) {
            arrivals(policy, 100);
            if (now % 30_000 == 0) { policy.hardTimeout(now); }
            int allowance = policy.allowance(now);
            assertTrue(allowance <= 61);
            assertTrue(policy.snapshot().strength() <= previous + 1.0 / 60 + 1e-9);
            previous = policy.snapshot().strength();
        }
        assertEquals(1, policy.snapshot().strength(), 1e-9);
        assertTrue(policy.snapshot().earlyEvictionQps() <= 60.00001);
        // Unused allowances during an empty queue are discarded, not returned next round.
        arrivals(policy, 100);
        policy.hardTimeout(181_000);
        assertTrue(policy.allowance(181_000) <= 62);
        var stalled = protecting();
        arrivals(stalled, 100);
        stalled.hardTimeout(240_000);
        assertTrue(stalled.allowance(240_000) <= 2);
        assertTrue(stalled.snapshot().strength() <= 1.0 / 60 + 1e-9);
    }

    @Test void hardSilenceAloneCannotRecoverWhileActualSoftEvictionContinues() {
        var policy = protecting();
        for (long now = 121_000; now <= 180_000; now += 1_000) {
            arrivals(policy, 100); policy.hardTimeout(now); policy.allowance(now);
        }
        for (long now = 181_000; now <= 360_000; now += 1_000) {
            arrivals(policy, 100); policy.evicted(now, 1); policy.allowance(now);
            assertEquals(PROTECTING, policy.snapshot().state());
        }
        policy.allowance(389_000);
        assertEquals(PROTECTING, policy.snapshot().state());
        policy.allowance(390_000);
        assertEquals(RECOVERING, policy.snapshot().state());
        double previous = policy.snapshot().strength();
        policy.evicted(390_001, 1);
        policy.allowance(391_000);
        assertEquals(PROTECTING, policy.snapshot().state());
        assertEquals(previous, policy.snapshot().strength(), 1e-9);
        for (long now = 392_000; now <= 550_000; now += 1_000) { policy.allowance(now); }
        assertEquals(NORMAL, policy.snapshot().state());
        assertEquals(0, policy.snapshot().strength());
    }

    @Test void aNewHardTimeoutDuringRecoveryResumesWithoutAnotherConfirmation() {
        var policy = protecting();
        for (long now = 121_000; now <= 180_000; now += 1_000) {
            arrivals(policy, 100); policy.hardTimeout(now); policy.allowance(now);
        }
        for (long now = 181_000; now <= 330_000; now += 1_000) { policy.allowance(now); }
        assertEquals(RECOVERING, policy.snapshot().state());
        double previous = policy.snapshot().strength();
        policy.hardTimeout(330_001);
        policy.allowance(331_000);
        assertEquals(PROTECTING, policy.snapshot().state());
        assertTrue(policy.snapshot().strength() > previous);
        assertTrue(policy.snapshot().strength() <= previous + 1.0 / 60 + 1e-9);
    }

    @Test void randomizedRoundsStayBoundedAcrossLongPausesAndRecovery() {
        var random = new Random(4901);
        var policy = new PrefillColdRequestProtection();
        long now = 0;
        for (int round = 0; round < 20_000; round++) {
            now += random.nextInt(20) == 0 ? 120_000 : 1_000;
            arrivals(policy, random.nextInt(200));
            if (random.nextInt(3) == 0) { policy.hardTimeout(now); }
            if (random.nextInt(3) == 0) { policy.evicted(now, 1); }
            var before = policy.snapshot();
            int allowance = policy.allowance(now);
            var after = policy.snapshot();
            assertTrue(allowance >= 0);
            assertTrue(after.strength() >= 0 && after.strength() <= 1);
            assertTrue(after.strength() <= before.strength() + 1.0 / 60 + 1e-9);
            assertTrue(after.strength() >= before.strength() - 1.0 / 120 - 1e-9);
            assertTrue(after.earlyEvictionQps() <= before.earlyEvictionQps() + Math.max(.1, .01 * after.arrivalQps()) + 1e-9);
            assertTrue(allowance <= Math.ceil(after.earlyEvictionQps()));
        }
    }

    private static PrefillQueueShedding.Controller controller(PrefillQueueSheddingConfig config) {
        var controller = new PrefillQueueShedding.Controller();
        for (long now = 0; now <= 120_000; now += 1_000) {
            for (int i = 0; i < 100; i++) { controller.arrive(); controller.coldArrival(); }
            if (now % 60_000 == 0) { controller.hardTimeout(now); }
            assertTrue(PrefillQueueShedding.adaptiveVictims(List.of(), config, null, controller, now).isEmpty());
        }
        return controller;
    }

    @Test void earlyBudgetIsColdOnlyRateLimitedAndDoesNotMultiplyWithScanFrequency() {
        var config = new PrefillQueueSheddingConfig();
        config.setExpectedWaitThresholdMs(60_000); config.setMaxWaitMs(120_000);
        var controller = controller(config);
        var candidates = new ArrayList<PrefillQueueShedding.Candidate<String>>();
        for (int i = 0; i < 1000; i++) {
            candidates.add(new PrefillQueueShedding.Candidate<>("cold" + i, 50, i,
                    new PrefillQueueShedding.Estimate(100, 0, 100, 1), 0));
            candidates.add(new PrefillQueueShedding.Candidate<>("hot" + i, 1, i,
                    new PrefillQueueShedding.Estimate(100, 90, 10, 1), 0));
        }
        for (int i = 0; i < 100; i++) { controller.arrive(); controller.coldArrival(); }
        var first = PrefillQueueShedding.adaptiveVictims(candidates, config, null, controller, 121_000);
        assertEquals(1, first.size());
        assertEquals("cold999", first.getFirst());
        controller.recordEvictions(121_000, first.size(), first.size());
        for (long now = 121_001; now < 122_000; now++) {
            assertTrue(PrefillQueueShedding.adaptiveVictims(candidates, config, null, controller, now).isEmpty());
        }
        config.setSustainedOverloadProtection(false);
        assertTrue(PrefillQueueShedding.adaptiveVictims(candidates, config, null, controller, 122_000).isEmpty());
    }

    @Test void lowEngineQueueStillSuppressesEarlySheddingAndKeepsHardEvidence() {
        var config = new PrefillQueueSheddingConfig();
        var controller = controller(config);
        var candidates = List.of(new PrefillQueueShedding.Candidate<>(1, 1, 1,
                new PrefillQueueShedding.Estimate(100, 0, 100, 1), 0));
        assertTrue(PrefillQueueShedding.adaptiveVictims(candidates, config, null, controller, 121_000, false).isEmpty());
        assertEquals(PROTECTING, controller.protectionSnapshot().state());
        assertEquals(List.of(1), PrefillQueueShedding.adaptiveVictims(candidates, config, null, controller, 122_000));
    }
}
