package org.flexlb.balance.autotpm;

import org.flexlb.balance.scheduler.QueueSnapshot;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the pure {@link EvictionPlanner} — no queue mutation, no CAS.
 */
class EvictionPlannerTest {

    private final EvictionPlanner planner = new EvictionPlanner();

    // ==================== Hard rule: same priority NEVER evict ====================

    @Test
    void hardRule_samePriority_noVictims() {
        QueueSnapshot snap = snapshot(1L, 2,
                item(10L, 50, 10_000_000L, 100),
                item(11L, 50, 11_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 50, 100, 8);
        assertTrue(plan.isEmpty(), "Same-priority items must never be evicted");
        assertEquals(0, plan.victimCount());
    }

    @Test
    void hardRule_higherPriorityVictimsExcluded() {
        // Incoming P50: P70 must NOT be evicted (70 > 50), P30 CAN be evicted
        QueueSnapshot snap = snapshot(1L, 2,
                item(10L, 70, 9_000_000L, 100),
                item(11L, 30, 11_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 50, 100, 8);
        assertEquals(1, plan.victimCount());
        assertEquals(11L, plan.victimRequestIds().get(0), "only P30 victim eligible");
    }

    // ==================== Priority ordering ====================

    @Test
    void priorityOrdering_lowPriorityFirst() {
        // Incoming P70, victims P30 and P50 → P30 (lowest) evicted first
        QueueSnapshot snap = snapshot(1L, 2,
                item(20L, 50, 10_000_000L, 100),
                item(21L, 30, 10_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 8);
        assertEquals(List.of(21L, 20L), plan.victimRequestIds(),
                "P30 (req 21) before P50 (req 20)");
    }

    // ==================== Deadline sorting ====================

    @Test
    void deadlineSorting_laterDeadlineFirstAmongSamePriority() {
        // Two P30 victims, earlier vs later deadline → later deadline evicted first
        long early = 5_000_000L;
        long late = 8_000_000L;
        QueueSnapshot snap = snapshot(1L, 2,
                item(30L, 30, early, 100),
                item(31L, 30, late, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 8);
        assertEquals(List.of(31L, 30L), plan.victimRequestIds(),
                "Later-deadline P30 (req 31) evicted before earlier-deadline (req 30)");
    }

    // ==================== Cost computation ====================

    @Test
    void costComputation_sumOfFPriority() {
        // Incoming P70, victims P30 (f=1) + P40 (f=512) → priorityCost = 513
        QueueSnapshot snap = snapshot(1L, 2,
                item(40L, 30, 10_000_000L, 100),
                item(41L, 40, 10_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 8);
        assertEquals(2, plan.victimCount());
        // f(30)=1, f(40)=512
        assertEquals(1L + 512L, plan.cost().priorityCost(),
                "priorityCost = Σ f(victim.priority) = 1 + 512");
        // stageCost = 2 * g(NOT_ACCEPTED)=2*1=2
        assertEquals(2, plan.cost().stageCost());
        assertEquals(2, plan.cost().victimCount());
    }

    @Test
    void costComputation_singleVictimExponential() {
        // Single P60 victim, incoming P70 → f(60)=512^3
        QueueSnapshot snap = snapshot(1L, 1,
                item(50L, 60, 10_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 8);
        long expected = 512L * 512L * 512L;
        assertEquals(expected, plan.cost().priorityCost(),
                "f(P60) = B^3 = 512^3");
    }

    // ==================== Max victims limit ====================

    @Test
    void maxVictimsLimit_capsSelection() {
        // 5 candidates P30..P60, maxVictims=2 → only 2 selected (lowest priority first)
        QueueSnapshot snap = snapshot(1L, 5,
                item(60L, 30, 10_000_000L, 100),
                item(61L, 40, 10_000_000L, 100),
                item(62L, 50, 10_000_000L, 100),
                item(63L, 60, 10_000_000L, 100),
                item(64L, 30, 10_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 2);
        assertEquals(2, plan.victimCount(), "maxVictims caps selection to 2");
        // P30 items come first (priority asc); tie-break by requestId desc → 64 before 60
        assertEquals(64L, plan.victimRequestIds().get(0));
        assertEquals(60L, plan.victimRequestIds().get(1));
    }

    // ==================== Edge cases ====================

    @Test
    void emptyQueue_emptyPlan() {
        QueueSnapshot snap = snapshot(1L, 0);
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 8);
        assertTrue(plan.isEmpty());
        assertEquals(1L, plan.snapshotVersion());
    }

    @Test
    void nullSnapshot_emptyPlan() {
        PrefillEvictionPlan plan = planner.plan(null, 70, 100, 8);
        assertTrue(plan.isEmpty());
    }

    @Test
    void versionPropagated_intoPlan() {
        QueueSnapshot snap = snapshot(42L, 1,
                item(70L, 30, 10_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 8);
        assertEquals(42L, plan.snapshotVersion(),
                "plan must carry the snapshot version for the committer CAS");
    }

    @Test
    void zeroMaxVictims_emptyPlan() {
        QueueSnapshot snap = snapshot(1L, 1,
                item(80L, 30, 10_000_000L, 100));
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 0);
        assertTrue(plan.isEmpty(), "maxVictims=0 yields empty plan");
    }

    @Test
    void planIsEmpty_flagConsistent() {
        QueueSnapshot snap = snapshot(1L, 1,
                item(90L, 70, 10_000_000L, 100)); // same priority as incoming P70
        PrefillEvictionPlan plan = planner.plan(snap, 70, 100, 8);
        assertTrue(plan.isEmpty());
        assertFalse(!plan.isEmpty());
        assertEquals(0, plan.victimCount());
    }

    // ==================== Helpers ====================

    private static QueueSnapshot snapshot(long version, int queueSize,
                                          QueueSnapshot.ItemSummary... items) {
        List<QueueSnapshot.ItemSummary> list = new ArrayList<>(Arrays.asList(items));
        return new QueueSnapshot(version, queueSize, list);
    }

    private static QueueSnapshot.ItemSummary item(long requestId, int priority,
                                                   long deadlineMs, long seqLen) {
        return new QueueSnapshot.ItemSummary(requestId, priority, deadlineMs, seqLen);
    }
}
