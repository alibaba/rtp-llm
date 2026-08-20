package org.flexlb.balance.scheduler.priority;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Phase 3 tests for the pure {@link EvictionPlanner} (design doc 9.1-9.4):
 * strictly-lower-priority candidates only, candidate preference order,
 * deficit/feasibility rules and deterministic plan ordering.
 */
class EvictionPlannerTest {

    private Map<String, String> failures;

    @BeforeEach
    void setUp() {
        failures = new HashMap<>();
    }

    // ==================== candidate selection ====================

    @Test
    void full_queue_evicts_the_lowest_priority_candidate_first() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 50, now),
                snap(2, 30, now),
                snap(3, 50, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), failures);

        assertNotNull(proposal);
        assertEquals(1, proposal.victims().size());
        assertEquals(2L, proposal.victims().get(0).requestId());
        assertEquals(PriorityCostFunction.f(30), proposal.rawCost());
    }

    @Test
    void equal_priority_never_yields() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 50, now),
                snap(2, 60, now),
                snap(3, 70, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 50), List.of(queue), failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("ep1"));
    }

    @Test
    void same_priority_candidates_prefer_newer_arrival() {
        long now = System.currentTimeMillis();
        // All victims P30; newest arrival is evicted first.
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 30, now),
                snap(2, 30, now + 100),
                snap(3, 30, now + 500));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), failures);

        assertNotNull(proposal);
        // Newest arrival wins once priority is equal.
        assertEquals(List.of(3L),
                proposal.victims().stream().map(QueuedRequestSnapshot::requestId).toList());
    }

    @Test
    void deficit_greater_than_one_selects_multiple_victims_in_candidate_order() {
        long now = System.currentTimeMillis();
        // 4 queued + 1 incoming vs capacity 3 -> deficit 2
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 30, now),
                snap(2, 40, now),
                snap(3, 50, now),
                snap(4, 30, now + 500));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), failures);

        assertNotNull(proposal);
        // Both P30s go before the P40: newest arrival first among the P30s.
        assertEquals(List.of(4L, 1L),
                proposal.victims().stream().map(QueuedRequestSnapshot::requestId).toList());
        assertEquals(2 * PriorityCostFunction.f(30), proposal.rawCost());
    }

    // ==================== task40: no-priority items are never victims ====================

    @Test
    void no_priority_items_are_never_selected_as_victims() {
        long now = System.currentTimeMillis();
        // Queue full of legacy (no-priority) items: none may be evicted even
        // though 0 < envelope.priority numerically.
        PrefillQueueSnapshot queue = queue("ep1", 2,
                snap(1, 0, now),
                snap(2, 0, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("ep1"));
    }

    @Test
    void mixed_queue_only_evicts_priority_carrying_candidates() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot queue = queue("ep1", 2,
                snap(1, 0, now),
                snap(2, 30, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), failures);

        assertNotNull(proposal);
        assertEquals(List.of(2L),
                proposal.victims().stream().map(QueuedRequestSnapshot::requestId).toList());
    }

    // ==================== feasibility guards ====================

    @Test
    void not_full_or_unbounded_queue_is_infeasible() {
        long now = System.currentTimeMillis();

        PrefillQueueSnapshot notFull = queue("ep1", 5, snap(1, 30, now));
        assertNull(EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(notFull), failures));
        assertEquals("queue_not_full", failures.get("ep1"));

        PrefillQueueSnapshot unbounded = queue("ep2", 0, snap(2, 30, now));
        assertNull(EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(unbounded), failures));
        assertEquals("queue_unbounded", failures.get("ep2"));
    }

    @Test
    void deficit_above_eight_selects_every_required_victim() {
        long now = System.currentTimeMillis();
        // deficit = 10 + 1 - 1 = 10. There is deliberately no per-plan
        // victim cap: feasibility is determined only by eligible capacity.
        QueuedRequestSnapshot[] items = new QueuedRequestSnapshot[10];
        for (int i = 0; i < items.length; i++) {
            items[i] = snap(i + 1, 30, now + i);
        }
        PrefillQueueSnapshot queue = queue("ep1", 1, items);

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(99, 70), List.of(queue), failures);

        assertNotNull(proposal);
        assertEquals(10, proposal.victims().size());
        assertNull(failures.get("ep1"));
    }

    @Test
    void arbitrary_number_of_lower_priority_victims_beats_one_higher_priority_victim() {
        long now = System.currentTimeMillis();
        // This is the first point at which the old fixed-radix scalar inverted:
        // 1025 * f(P30) > 1 * f(P40). Exact priority buckets must still choose
        // P30 victims because no amount of lower-priority harm may spill into P40.
        QueuedRequestSnapshot[] lowerPriorityItems = new QueuedRequestSnapshot[1_025];
        for (int i = 0; i < lowerPriorityItems.length; i++) {
            lowerPriorityItems[i] = snap(i + 1, 30, now + i);
        }
        PrefillQueueSnapshot manyP30 = queue("many-p30", 1, lowerPriorityItems);
        PrefillQueueSnapshot oneP40 = queue("one-p40", 1,
                snap(2_000, 40, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(3_000, 70), List.of(oneP40, manyP30), failures);

        assertNotNull(proposal);
        assertEquals("many-p30", proposal.endpointId());
        assertEquals(1_025, proposal.victims().size());
        assertTrue(proposal.rawCost() > PriorityCostFunction.f(40));
    }

    // ==================== helpers ====================

    private static PrefillQueueSnapshot queue(String endpointId, int capacity,
                                              QueuedRequestSnapshot... items) {
        return new PrefillQueueSnapshot(endpointId, 1L, capacity, List.of(items));
    }

    private static QueuedRequestSnapshot snap(long requestId, int priority,
                                              long arrivalMs) {
        return new QueuedRequestSnapshot(requestId, priority, arrivalMs,
                128, 0, QueuedRequestSnapshot.PREFILL_QUEUED);
    }

    private static PriorityRequestEnvelope envelope(long requestId, int priority) {
        long now = System.currentTimeMillis();
        return new PriorityRequestEnvelope(requestId, priority, 128, 8,
                now, 128, 136);
    }
}
