package org.flexlb.balance.autotpm;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link DecodeEvictionPlanner} — pure planning, no mutation.
 */
class DecodeEvictionPlannerTest {

    private static final String EP_KEY = "10.0.0.1:8080";
    private final DecodeEvictionPlanner planner = new DecodeEvictionPlanner();

    // ==================== Slot eviction ====================

    @Test
    void slotEviction_lowerPriorityEvictedFirst() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(10L, 30, 1000));
        tracker.reserve(EP_KEY, res(11L, 40, 1000));
        tracker.reserve(EP_KEY, res(12L, 50, 1000));

        DecodeEvictionPlan plan = planner.planSlotEviction(
                tracker, EP_KEY, 70, 2, 8);

        assertEquals(2, plan.victimCount());
        assertEquals(2, plan.slotsFreed());
        // P30 (lowest) first, then P40
        assertEquals(10L, plan.victims().get(0).requestId());
        assertEquals(11L, plan.victims().get(1).requestId());
    }

    @Test
    void slotEviction_samePriority_noVictims() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(10L, 50, 1000));
        tracker.reserve(EP_KEY, res(11L, 50, 1000));

        DecodeEvictionPlan plan = planner.planSlotEviction(
                tracker, EP_KEY, 50, 2, 8);

        assertTrue(plan.isEmpty());
        assertEquals(0, plan.victimCount());
    }

    @Test
    void slotEviction_higherPriorityNotEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(10L, 70, 1000));

        DecodeEvictionPlan plan = planner.planSlotEviction(
                tracker, EP_KEY, 50, 1, 8);

        assertTrue(plan.isEmpty());
    }

    @Test
    void slotEviction_runningNeverEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(10L, 30, 1000));
        tracker.markRunning(EP_KEY, 10L);
        tracker.reserve(EP_KEY, res(11L, 40, 1000));

        DecodeEvictionPlan plan = planner.planSlotEviction(
                tracker, EP_KEY, 70, 2, 8);

        assertEquals(1, plan.victimCount(), "Only P40 (not RUNNING P30) should be evicted");
        assertEquals(11L, plan.victims().get(0).requestId());
    }

    @Test
    void slotEviction_maxVictimsCapsSelection() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(10L, 30, 1000));
        tracker.reserve(EP_KEY, res(11L, 30, 1000));
        tracker.reserve(EP_KEY, res(12L, 40, 1000));

        DecodeEvictionPlan plan = planner.planSlotEviction(
                tracker, EP_KEY, 70, 3, 2);

        assertEquals(2, plan.victimCount(), "maxVictims=2 caps selection");
    }

    @Test
    void slotEviction_emptyTracker() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();

        DecodeEvictionPlan plan = planner.planSlotEviction(
                tracker, EP_KEY, 70, 1, 8);

        assertTrue(plan.isEmpty());
        assertEquals(0, plan.slotsFreed());
    }

    // ==================== KV eviction ====================

    @Test
    void kvEviction_moreKvReleasedPreferred() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(20L, 30, 500));
        tracker.reserve(EP_KEY, res(21L, 30, 2000));

        DecodeEvictionPlan plan = planner.planKvEviction(
                tracker, EP_KEY, 70, 1000, 8);

        assertEquals(1, plan.victimCount(), "Only 2000-KV victim needed to satisfy 1000");
        assertEquals(2000, plan.kvFreed());
        assertEquals(21L, plan.victims().get(0).requestId());
    }

    @Test
    void kvEviction_greedyAccumulation() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(30L, 30, 300));
        tracker.reserve(EP_KEY, res(31L, 30, 500));
        tracker.reserve(EP_KEY, res(32L, 30, 800));

        DecodeEvictionPlan plan = planner.planKvEviction(
                tracker, EP_KEY, 70, 1000, 8);

        assertEquals(2, plan.victimCount(), "800+500=1300 >= 1000");
        assertEquals(1300, plan.kvFreed());
    }

    @Test
    void kvEviction_samePriority_noVictims() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(40L, 50, 1000));

        DecodeEvictionPlan plan = planner.planKvEviction(
                tracker, EP_KEY, 50, 100, 8);

        assertTrue(plan.isEmpty());
    }

    @Test
    void kvEviction_runningNeverEvicted() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(50L, 30, 10000));
        tracker.markRunning(EP_KEY, 50L);
        tracker.reserve(EP_KEY, res(51L, 40, 100));

        DecodeEvictionPlan plan = planner.planKvEviction(
                tracker, EP_KEY, 70, 1, 8);

        assertEquals(1, plan.victimCount());
        assertEquals(51L, plan.victims().get(0).requestId());
    }

    @Test
    void kvEviction_lowPriorityFirst() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // P30 with 500 KV, P40 with 500 KV (same KV)
        tracker.reserve(EP_KEY, res(60L, 30, 500));
        tracker.reserve(EP_KEY, res(61L, 40, 500));

        DecodeEvictionPlan plan = planner.planKvEviction(
                tracker, EP_KEY, 70, 600, 8);

        assertEquals(2, plan.victimCount());
        // P30 first (lower priority)
        assertEquals(60L, plan.victims().get(0).requestId());
        assertEquals(61L, plan.victims().get(1).requestId());
    }

    // ==================== Combined eviction ====================

    @Test
    void combined_slotOnlyPlanSatisfiesBoth() {
        // Slot plan frees enough slots AND enough KV → use slot-only plan
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // One P30 victim with 2000 KV (enough for neededKv=1000) → frees 1 slot + 2000 KV
        tracker.reserve(EP_KEY, res(70L, 30, 2000));

        DecodeEvictionPlan plan = planner.planCombinedEviction(
                tracker, EP_KEY, 70, 1, 1000, 8);

        assertEquals(1, plan.victimCount(), "Slot-only plan satisfies both");
        assertEquals(70L, plan.victims().get(0).requestId());
        assertTrue(plan.satisfiesSlots(1));
        assertTrue(plan.satisfiesKv(1000));
    }

    @Test
    void combined_kvOnlyPlanSatisfiesBoth() {
        // KV plan frees enough KV AND enough slots → use kv-only plan
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // Two P30 victims: 800 KV + 500 KV = 1300 KV (enough for 1000)
        tracker.reserve(EP_KEY, res(80L, 30, 800));
        tracker.reserve(EP_KEY, res(81L, 30, 500));

        DecodeEvictionPlan plan = planner.planCombinedEviction(
                tracker, EP_KEY, 70, 2, 1000, 8);

        // KV plan: 800 + 500 = 1300 KV, 2 slots → satisfies both
        assertEquals(2, plan.victimCount(), "KV-only plan satisfies both");
        assertTrue(plan.satisfiesSlots(2));
        assertTrue(plan.satisfiesKv(1000));
    }

    @Test
    void combined_dedupVictims() {
        // Both slot and KV needed, neither plan alone satisfies both
        // Slot plan: picks victim A (1 slot, 300 KV) — not enough KV for 1000
        // KV plan: picks victims A+B (300+800=1100 KV, 2 slots) — satisfies both
        // Actually KV plan satisfies both (2 slots, 1100 KV)
        // So let's make it so KV plan doesn't satisfy slots:
        // Need 2 slots, 1000 KV
        // Slot plan: picks A+B (2 slots, 300+800=1100 KV) — satisfies both!
        // So slot plan would satisfy both → use slot plan

        // Let me construct a case where neither alone works:
        // Need 2 slots, 1000 KV
        // Victims: A(P30, 300KV), B(P30, 800KV), C(P40, 100KV)
        // Slot plan (need 2, sorted by pri asc): A, B → 2 slots, 1100 KV → satisfies both!
        // Still slot plan works...

        // To force combine: need 3 slots, 1000 KV, maxVictims=2
        // Slot plan (need 3, max 2): A, B → 2 slots (not enough for 3)
        // KV plan (need 1000, max 2): B, A → 1100 KV, 2 slots (not enough for 3)
        // Combined: A, B (deduped) → 2 slots, 1100 KV → still not 3 slots
        // But at least the combine should run

        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(90L, 30, 300));
        tracker.reserve(EP_KEY, res(91L, 30, 800));

        DecodeEvictionPlan plan = planner.planCombinedEviction(
                tracker, EP_KEY, 70, 3, 1000, 2);

        // Neither plan alone satisfies 3 slots, but combine should dedup A+B
        assertEquals(2, plan.victimCount(), "Combined plan dedup: A+B");
        // Verify no duplicate request IDs
        List<Long> ids = plan.victimRequestIds();
        assertEquals(2, ids.stream().distinct().count(), "No duplicate victims");
    }

    @Test
    void combined_noEligibleVictims() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(100L, 70, 1000)); // same priority

        DecodeEvictionPlan plan = planner.planCombinedEviction(
                tracker, EP_KEY, 70, 1, 100, 8);

        assertTrue(plan.isEmpty());
    }

    // ==================== Cost computation ====================

    @Test
    void costComputation_priorityAndStage() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // P30 RESERVED + P40 ACCEPTED
        tracker.reserve(EP_KEY, new DecodeReservation(
                110L, 30, 10_000_000L, 1000, EP_KEY, 1L,
                DecodeAdmissionState.RESERVED_NOT_ACCEPTED));
        tracker.reserve(EP_KEY, new DecodeReservation(
                111L, 40, 10_000_000L, 1000, EP_KEY, 2L,
                DecodeAdmissionState.ACCEPTED_NOT_RUNNING));

        DecodeEvictionPlan plan = planner.planSlotEviction(
                tracker, EP_KEY, 70, 2, 8);

        assertEquals(2, plan.victimCount());
        // f(30)=1, f(40)=512 → priorityCost = 513
        assertEquals(1L + 512L, plan.cost().priorityCost());
        // g(NOT_ACCEPTED)=1, g(ACCEPTED_NOT_RUNNING)=4 → stageCost = 5
        assertEquals(1 + 4, plan.cost().stageCost());
    }

    @Test
    void costComputation_resourceCost() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        tracker.reserve(EP_KEY, res(120L, 30, 500));
        tracker.reserve(EP_KEY, res(121L, 30, 300));

        DecodeEvictionPlan plan = planner.planKvEviction(
                tracker, EP_KEY, 70, 1000, 8);

        // Both victims selected (500+300=800 < 1000, but that's all we have)
        assertEquals(2, plan.victimCount());
        assertEquals(800, plan.cost().resourceCost(), "resourceCost = sum of KV tokens");
        assertEquals(800, plan.kvFreed());
    }

    // ==================== Helpers ====================

    private static DecodeReservation res(long requestId, int priority, long kvTokens) {
        return new DecodeReservation(requestId, priority, 10_000_000L,
                kvTokens, EP_KEY, requestId,
                DecodeAdmissionState.RESERVED_NOT_ACCEPTED);
    }
}
