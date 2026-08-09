package org.flexlb.balance.scheduler.priority;

import org.flexlb.config.FlexlbConfig;
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
 * deficit/feasibility rules and the anti-inversion cache benefit bound.
 */
class EvictionPlannerTest {

    private FlexlbConfig config;
    private Map<String, String> failures;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        failures = new HashMap<>();
    }

    // ==================== candidate selection ====================

    @Test
    void full_queue_evicts_the_lowest_priority_candidate_first() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 50, now + 1_000, now),
                snap(2, 30, now + 2_000, now),
                snap(3, 50, now + 3_000, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), Map.of(), config, failures);

        assertNotNull(proposal);
        assertEquals(1, proposal.victims().size());
        assertEquals(2L, proposal.victims().get(0).requestId());
        assertEquals(PriorityCostFunction.f(30), proposal.rawCost());
        assertEquals(proposal.rawCost(), proposal.netCost());
    }

    @Test
    void equal_priority_never_yields() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 50, now + 1_000, now),
                snap(2, 60, now + 2_000, now),
                snap(3, 70, now + 3_000, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 50), List.of(queue), Map.of(), config, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("ep1"));
    }

    @Test
    void same_priority_candidates_prefer_later_deadline_then_newer_arrival() {
        long now = System.currentTimeMillis();
        // All victims P30; deadline desc first, then arrival desc
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 30, now + 1_000, now),
                snap(2, 30, now + 9_000, now),
                snap(3, 30, now + 9_000, now + 500));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), Map.of(), config, failures);

        assertNotNull(proposal);
        // Latest deadline (more slack) wins; among equals the newest arrival
        assertEquals(List.of(3L),
                proposal.victims().stream().map(QueuedRequestSnapshot::requestId).toList());
    }

    @Test
    void deficit_greater_than_one_selects_multiple_victims_in_candidate_order() {
        long now = System.currentTimeMillis();
        // 4 queued + 1 incoming vs capacity 3 -> deficit 2
        PrefillQueueSnapshot queue = queue("ep1", 3,
                snap(1, 30, now + 1_000, now),
                snap(2, 40, now + 1_000, now),
                snap(3, 50, now + 1_000, now),
                snap(4, 30, now + 5_000, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), Map.of(), config, failures);

        assertNotNull(proposal);
        // Both P30s go before the P40: later deadline first among the P30s
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
                snap(1, 0, 0, now),
                snap(2, 0, 0, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), Map.of(), config, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("ep1"));
    }

    @Test
    void mixed_queue_only_evicts_priority_carrying_candidates() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot queue = queue("ep1", 2,
                snap(1, 0, 0, now),
                snap(2, 30, now + 2_000, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), Map.of(), config, failures);

        assertNotNull(proposal);
        assertEquals(List.of(2L),
                proposal.victims().stream().map(QueuedRequestSnapshot::requestId).toList());
    }

    // ==================== feasibility guards ====================

    @Test
    void not_full_or_unbounded_queue_is_infeasible() {
        long now = System.currentTimeMillis();

        PrefillQueueSnapshot notFull = queue("ep1", 5, snap(1, 30, now + 1_000, now));
        assertNull(EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(notFull), Map.of(), config, failures));
        assertEquals("queue_not_full", failures.get("ep1"));

        PrefillQueueSnapshot unbounded = queue("ep2", 0, snap(2, 30, now + 1_000, now));
        assertNull(EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(unbounded), Map.of(), config, failures));
        assertEquals("queue_unbounded", failures.get("ep2"));
    }

    @Test
    void deficit_exceeding_max_victims_per_plan_is_infeasible() {
        long now = System.currentTimeMillis();
        // deficit = 10 + 1 - 1 = 10 > MAX_VICTIMS_PER_PLAN (8)
        QueuedRequestSnapshot[] items = new QueuedRequestSnapshot[10];
        for (int i = 0; i < items.length; i++) {
            items[i] = snap(i + 1, 30, now + (i + 1) * 1_000L, now);
        }
        PrefillQueueSnapshot queue = queue("ep1", 1, items);

        assertNull(EvictionPlanner.planPrefillQueue(
                envelope(99, 70), List.of(queue), Map.of(), config, failures));
        assertEquals("deficit_exceeds_max_victims", failures.get("ep1"));
    }

    // ==================== cache benefit bound (anti-inversion) ====================

    @Test
    void bounded_cache_benefit_cannot_reverse_the_priority_boundary() {
        config.setAutoTpmPlanCacheHitBenefitCap(1_000_000);
        long now = System.currentTimeMillis();
        // epA would evict a P40 (rawCost 1024) but has a huge cache benefit;
        // epB evicts a P30 (rawCost 1) with no benefit.
        PrefillQueueSnapshot cached = queue("epA", 1, snap(1, 40, now + 1_000, now));
        PrefillQueueSnapshot uncached = queue("epB", 1, snap(2, 30, now + 1_000, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(cached, uncached),
                Map.of("epA", 1_000_000L), config, failures);

        assertNotNull(proposal);
        // Benefit is clamped to MIN_ADJACENT_GAP / 2 = 511, so
        // netCost(epA) = 1024 - 511 = 513 > netCost(epB) = 1
        assertEquals("epB", proposal.endpointId());
        assertEquals(1, proposal.netCost());

        PrefillEvictionProposal cachedOnly = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(cached), Map.of("epA", 1_000_000L), config, failures);
        assertNotNull(cachedOnly);
        assertEquals(PriorityCostFunction.MIN_ADJACENT_GAP / 2, cachedOnly.boundedCacheBenefit());
        assertEquals(PriorityCostFunction.f(40) - PriorityCostFunction.MIN_ADJACENT_GAP / 2,
                cachedOnly.netCost());
        assertTrue(cachedOnly.netCost() > PriorityCostFunction.f(30));
    }

    @Test
    void cache_benefit_defaults_to_zero_when_cap_is_unset() {
        long now = System.currentTimeMillis();
        PrefillQueueSnapshot queue = queue("ep1", 1, snap(1, 30, now + 1_000, now));

        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope(9, 70), List.of(queue), Map.of("ep1", 1_000_000L), config, failures);

        assertNotNull(proposal);
        assertEquals(0, proposal.boundedCacheBenefit());
        assertEquals(proposal.rawCost(), proposal.netCost());
    }

    // ==================== helpers ====================

    private static PrefillQueueSnapshot queue(String endpointId, int capacity,
                                              QueuedRequestSnapshot... items) {
        return new PrefillQueueSnapshot(endpointId, 1L, capacity, List.of(items));
    }

    private static QueuedRequestSnapshot snap(long requestId, int priority,
                                              long deadlineMs, long arrivalMs) {
        return new QueuedRequestSnapshot(requestId, priority, deadlineMs, arrivalMs,
                128, 0, QueuedRequestSnapshot.PREFILL_QUEUED);
    }

    private static PriorityRequestEnvelope envelope(long requestId, int priority) {
        long now = System.currentTimeMillis();
        return new PriorityRequestEnvelope(requestId, priority, 128, 8,
                now, 60_000, now + 60_000, 128, 136);
    }
}
