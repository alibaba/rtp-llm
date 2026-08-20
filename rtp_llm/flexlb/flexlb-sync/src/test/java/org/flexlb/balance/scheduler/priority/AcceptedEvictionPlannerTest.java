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
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Phase 5 tests for the accepted-layer candidate rules of
 * {@link EvictionPlanner#planDecode}: the double gate (accepted-evict switch
 * AND per-endpoint Cancel RPC support), the strict lower-priority boundary
 * on the accepted layer, stage-ascending preference (reserved before
 * accepted, before the kvBucket tie-break), the higher accepted
 * victim cost (g=4), and running-layer selection with its larger cost (g=32).
 */
class AcceptedEvictionPlannerTest {

    private final FlexlbConfig config = new FlexlbConfig();
    private final EngineCancelChannel channel = mock(EngineCancelChannel.class);

    @BeforeEach
    void enableMasterLocalPlanningForExistingMixedDomainCases() {
        allowStages(VictimStage.DECODE_RESERVED);
    }

    // ==================== gate matrix ====================

    @Test
    void gateOff_acceptedNeverConsidered_andChannelNeverQueried() {
        // Default config: accepted-evict switch off.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
        // With the switch off the support probe must not even run.
        verify(channel, never()).isSupported(any());
    }

    @Test
    void gateOn_unsupportedEndpoint_isInfeasible() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(false);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    @Test
    void gateOn_supportedEndpoint_acceptedBecomesVictimWithHigherCost() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 5, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        assertEquals(List.of(1L), ids(proposal.victims()));
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, proposal.victims().get(0).phase());
        // h(DECODE_SLOT_FULL)=4 x f(30)=1 x g(ACCEPTED_NOT_RUNNING)=16.
        assertEquals(64, proposal.totalCost());
        assertEquals(256, proposal.freedKvTokens());
    }

    @Test
    void acceptedOnlyGateNeverSelectsCheaperMasterLocalVictim() {
        allowStages(VictimStage.DECODE_ENGINE_OWNED);
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 0, 10_000, 2, 0,
                List.of(reserved(1, 10, 2_048)),
                List.of(),
                List.of(new DecodeRequestSnapshot(2, 30, DecodeTaskPhase.RUNNING,
                        2_048, 2_048, true, false)));

        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 2_000), List.of(ep), config, channel, new HashMap<>());

        assertNotNull(proposal);
        assertEquals(List.of(2L), ids(proposal.victims()));
        assertEquals(DecodeTaskPhase.RUNNING, proposal.victims().getFirst().phase());
    }

    // ==================== 3.3: strict boundary holds on the accepted layer ====================

    @Test
    void equalPriorityAccepted_neverYields() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 50, 256)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    // ==================== stage asc: reserved preferred over accepted ====================

    @Test
    void slotDeficitSelectsEngineVictimBecauseMasterQueuedDoesNotHoldASlot() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 2, 2,
                List.of(reserved(1, 30, 128)),
                List.of(accepted(2, 30, 256)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(List.of(2L), ids(proposal.victims()));
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                proposal.victims().get(0).phase());
        assertEquals(64, proposal.totalCost());
    }

    @Test
    void kvFull_stagePrecedesKvBucket() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(true);
        // kvDeficit = 400; the accepted entry would win a pure kvBucket-desc
        // order (2048 > 512) but stage asc ranks the reserved entry first,
        // and 512 already covers the deficit.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100, 10_000, 1, 0,
                List.of(reserved(1, 30, 512)),
                List.of(accepted(2, 30, 2_048)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 500), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(DecodeEvictionProposal.CASE_KV, proposal.evictionCase());
        assertEquals(List.of(1L), ids(proposal.victims()));
        // h(DECODE_KV_FULL)=8 x f(30)=1 x g(RESERVED)=1 x lengthWasteCost(512)=1.
        assertEquals(8, proposal.totalCost());
    }

    @Test
    void kvFull_acceptedCoversDeficitWhenReservedCannot() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(true);
        // kvDeficit = 1900: reserved 512 alone is short, the greedy pass
        // extends into the accepted layer.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100, 10_000, 2, 0,
                List.of(reserved(1, 30, 512)),
                List.of(accepted(2, 30, 2_048)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 2_000), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        // A plan is ownership-homogeneous: it never combines a Master-local
        // removal with an Engine Cancel transaction.
        assertEquals(List.of(2L), ids(proposal.victims()));
        assertEquals(2_048, proposal.freedKvTokens());
        assertEquals(128, proposal.totalCost());
    }

    // ==================== running layer is cancellable behind the same gate ====================

    @Test
    void runningEntry_becomesVictimBehindTheGate() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(),
                List.of(new DecodeRequestSnapshot(1, 30, DecodeTaskPhase.RUNNING,
                        256, 256, true, false)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(List.of(1L), ids(proposal.victims()));
        assertEquals(DecodeTaskPhase.RUNNING, proposal.victims().get(0).phase());
        // h(DECODE_SLOT_FULL)=4 x f(30)=1 x g(RUNNING)=64.
        assertEquals(256, proposal.totalCost());
    }

    @Test
    void samePriority_acceptedIsPreferredBeforeRunning() {
        enableEngineOwnedPlanning();
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 2, 2,
                List.of(), List.of(accepted(1, 30, 256)),
                List.of(new DecodeRequestSnapshot(2, 30, DecodeTaskPhase.RUNNING,
                        256, 256, true, false)));

        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, new HashMap<>());

        assertNotNull(proposal);
        assertEquals(List.of(1L), ids(proposal.victims()));
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, proposal.victims().get(0).phase());
    }

    // ==================== Phase 4 overload: accepted layer disabled ====================

    @Test
    void legacyOverload_withoutChannel_neverUsesAcceptedLayer() {
        enableEngineOwnedPlanning();
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    // ==================== helpers ====================

    private static List<Long> ids(List<DecodeRequestSnapshot> victims) {
        return victims.stream().map(DecodeRequestSnapshot::requestId).toList();
    }

    private void enableEngineOwnedPlanning() {
        allowStages(VictimStage.DECODE_RESERVED, VictimStage.DECODE_ENGINE_OWNED);
    }

    private void allowStages(VictimStage... stages) {
        QueueSchedulerConfig scheduler = new QueueSchedulerConfig();
        PriorityOrderingConfig ordering = new PriorityOrderingConfig();
        PreemptionConfig preemption = new PreemptionConfig();
        EnumSet<VictimStage> allowed = EnumSet.noneOf(VictimStage.class);
        java.util.Collections.addAll(allowed, stages);
        preemption.setAllowedVictimStages(allowed);
        ordering.setPreemption(preemption);
        scheduler.setOrdering(ordering);
        config.setScheduler(scheduler);
    }

    private static DecodeEndpointSnapshot endpoint(String id, long version,
                                                   long realKvAvailable, long realKvTotal,
                                                   int totalLoad, long concurrencyLimit,
                                                   List<DecodeRequestSnapshot> reserved,
                                                   List<DecodeRequestSnapshot> accepted,
                                                   List<DecodeRequestSnapshot> running) {
        long hardKv = reserved.stream().mapToLong(DecodeRequestSnapshot::kvTokens).sum();
        long expectedKv = reserved.stream().mapToLong(DecodeRequestSnapshot::expectedKvTokens).sum();
        return new DecodeEndpointSnapshot(null, id, version, realKvAvailable, realKvTotal,
                totalLoad, totalLoad, concurrencyLimit, hardKv, expectedKv,
                reserved, accepted, running);
    }

    private static DecodeRequestSnapshot reserved(long requestId, int priority,
                                                  long kvTokens) {
        return new DecodeRequestSnapshot(requestId, priority,
                DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                kvTokens, kvTokens + 8, true, true);
    }

    private static DecodeRequestSnapshot accepted(long requestId, int priority,
                                                  long kvTokens) {
        return new DecodeRequestSnapshot(requestId, priority,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING, kvTokens, kvTokens, true, false);
    }

    private static PriorityRequestEnvelope incoming(int priority, long seqLen) {
        return new PriorityRequestEnvelope(999, priority, seqLen, 8,
                System.currentTimeMillis(), seqLen, seqLen + 8);
    }
}
