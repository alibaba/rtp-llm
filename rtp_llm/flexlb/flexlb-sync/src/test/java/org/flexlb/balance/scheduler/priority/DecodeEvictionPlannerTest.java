package org.flexlb.balance.scheduler.priority;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.QueueSchedulerConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Phase 4 tests for {@link EvictionPlanner#planDecode}: slot-full victim
 * ordering, the strict lower-priority boundary (design doc 3.3), greedy KV
 * selection, combined-plan dedup and cost addition (never re-multiplied by
 * h), and the reserved-only candidate rule (design doc 10.1, 11-13).
 */
class DecodeEvictionPlannerTest {

    private final FlexlbConfig config = new FlexlbConfig();
    private final EngineCancelChannel channel = mock(EngineCancelChannel.class);

    @BeforeEach
    void enableEngineCancelPlanning() {
        QueueSchedulerConfig scheduler = new QueueSchedulerConfig();
        PriorityOrderingConfig ordering = new PriorityOrderingConfig();
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_ENGINE_OWNED));
        ordering.setPreemption(preemption);
        scheduler.setOrdering(ordering);
        config.setScheduler(scheduler);
        when(channel.isSupported(any())).thenReturn(true);
    }

    // ==================== slot full: priority ordering ====================

    @Test
    void slotFull_picksLowestPriorityThenRequestId() {
        // limit=2, totalLoad=2 -> slotDeficit = 1; KV is plentiful.
        DecodeEndpointSnapshot ep = endpoint("d1", 7, 100_000, 200_000, 2, 2, List.of(
                reserved(1, 30, 128),
                reserved(2, 30, 128),
                reserved(3, 40, 128)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals("d1", proposal.endpointId());
        assertEquals(7, proposal.admissionVersion());
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        // p30 before p40; request id is the deterministic same-priority tie-break.
        assertEquals(List.of(1L), proposal.victims().stream()
                .map(DecodeRequestSnapshot::requestId).toList());
        // h(DECODE_SLOT_FULL)=4 x f(30)=1 x g(RESERVED_NOT_ACCEPTED)=1
        assertEquals(16, proposal.totalCost());
        assertEquals(128, proposal.freedKvTokens());
    }

    // ==================== 3.3: equal priority never yields ====================

    @Test
    void equalPriority_isInfeasible() {
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 2, 2, List.of(
                reserved(1, 50, 128),
                reserved(2, 50, 128)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    // ==================== task40: no-priority entries are never victims ====================

    @Test
    void noPriorityEntries_areNeverSelectedAsVictims() {
        // Slot-full endpoint whose reserved entries are all legacy (priority 0):
        // numerically 0 < incoming 50, but they must stay untouchable.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 2, 2, List.of(
                reserved(1, 0, 128),
                reserved(2, 0, 128)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    @Test
    void mixedEntries_onlyPriorityCarryingOnesAreEvicted() {
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 2, 2, List.of(
                reserved(1, 0, 128),
                reserved(2, 30, 128)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(List.of(2L), proposal.victims().stream()
                .map(DecodeRequestSnapshot::requestId).toList());
    }

    // ==================== KV full: greedy biggest-release-first ====================

    @Test
    void kvFull_greedyPicksBiggestReleaseFirst() {
        // No slot limit; hardKv=2000 > available 100 -> kvDeficit = 1900.
        DecodeEndpointSnapshot ep = endpoint("d1", 3, 100, 10_000, 3, 0, List.of(
                reserved(1, 30, 1_024),
                reserved(2, 30, 2_048),
                reserved(3, 30, 512)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 2_000), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(DecodeEvictionProposal.CASE_KV, proposal.evictionCase());
        // kvBucket desc: request 2 alone already covers the deficit.
        assertEquals(List.of(2L), proposal.victims().stream()
                .map(DecodeRequestSnapshot::requestId).toList());
        // h(DECODE_KV_FULL)=8 x f(30)=1 x g=1 x lengthWasteCost(2048)=1
        assertEquals(32, proposal.totalCost());
        assertEquals(2_048, proposal.freedKvTokens());
    }

    @Test
    void kvFull_insufficientReleasableKv_isInfeasible() {
        DecodeEndpointSnapshot ep = endpoint("d1", 3, 100, 10_000, 1, 0, List.of(
                reserved(3, 30, 512)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 2_000), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_releasable_kv", failures.get("d1"));
    }

    @Test
    void slotDeficit_aboveEight_selectsEveryRequiredVictim() {
        // limit=2, totalLoad=10 -> slotDeficit = 9. The planner must not turn
        // a feasible high-priority admission into an artificial cap failure.
        List<DecodeRequestSnapshot> candidates = new ArrayList<>();
        for (int i = 1; i <= 9; i++) {
            candidates.add(reserved(i, 30, 128));
        }
        DecodeEndpointSnapshot ep = endpoint(
                "d1", 1, 100_000, 200_000, 10, 2, candidates);

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(9, proposal.victims().size());
        assertEquals(9 * 16, proposal.totalCost());
    }

    // ==================== 13: combined slot+KV plan ====================

    @Test
    void combined_dedupsVictimsAndAddsCostsWithoutReMultiplying() {
        // slotDeficit = 3+1-2 = 2; kvDeficit = 1000-0 = 1000.
        // slotPressure 1.0 > kvPressure 0.01 -> slot part plans first.
        DecodeEndpointSnapshot ep = endpoint("d1", 9, 0, 100_000, 3, 2, List.of(
                reserved(1, 30, 0),
                reserved(2, 30, 600),
                reserved(3, 30, 1_200)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 1_000), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(DecodeEvictionProposal.CASE_SLOT_AND_KV, proposal.evictionCase());
        // Slot part [1, 2] frees 600 KV; KV part covers the remaining 400
        // with request 3 only — already-picked victims are excluded (dedup).
        assertEquals(List.of(1L, 2L, 3L), proposal.victims().stream()
                .map(DecodeRequestSnapshot::requestId).toList());
        // slotPart = 4 x (1 + 1) = 8; kvPart = 8 x 1 x 1 x 1 = 8;
        // totalCost = 8 + 8 = 16 — parts are added, never re-multiplied by h.
        assertEquals(64, proposal.totalCost());
        assertEquals(1_800, proposal.freedKvTokens());
    }

    @Test
    void combined_slotOnlyPlanIncidentallyCoveringKv_keepsSlotCase() {
        // slotDeficit = 1; kvDeficit = 500; the single slot victim frees 2048
        // KV, so no combined plan is generated (13.2).
        DecodeEndpointSnapshot ep = endpoint("d1", 2, 0, 10_000, 1, 1, List.of(
                reserved(1, 30, 2_048)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 500), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        assertEquals(List.of(1L), proposal.victims().stream()
                .map(DecodeRequestSnapshot::requestId).toList());
        assertEquals(16, proposal.totalCost());
    }

    // ==================== 10.1: confirmed requests are never candidates ====================

    @Test
    void confirmedRequests_areNeverCandidates() {
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 2, 1, List.of(
                new DecodeRequestSnapshot(1, 30, DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                        128, 136, true, false),
                new DecodeRequestSnapshot(2, 30, DecodeTaskPhase.RUNNING,
                        128, 136, true, false)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    @Test
    void sufficientCapacity_reportsNoDeficit() {
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 0, 2, List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("decode_capacity_sufficient", failures.get("d1"));
    }

    // ==================== helpers ====================

    private static DecodeEndpointSnapshot endpoint(String id, long version,
                                                   long realKvAvailable, long realKvTotal,
                                                   int totalLoad, long concurrencyLimit,
                                                   List<DecodeRequestSnapshot> reserved) {
        long hardKv = reserved.stream().mapToLong(DecodeRequestSnapshot::kvTokens).sum();
        long expectedKv = reserved.stream().mapToLong(DecodeRequestSnapshot::expectedKvTokens).sum();
        return new DecodeEndpointSnapshot(null, id, version, realKvAvailable, realKvTotal,
                totalLoad, totalLoad, concurrencyLimit, hardKv, expectedKv,
                reserved, List.of(), List.of());
    }

    private static DecodeRequestSnapshot reserved(long requestId, int priority,
                                                  long kvTokens) {
        return new DecodeRequestSnapshot(requestId, priority,
                DecodeTaskPhase.ENGINE_MAY_HAVE_SEEN,
                kvTokens, kvTokens + 8, true, false);
    }

    private static PriorityRequestEnvelope incoming(int priority, long seqLen) {
        return new PriorityRequestEnvelope(999, priority, seqLen, 8,
                System.currentTimeMillis(), seqLen, seqLen + 8);
    }
}
