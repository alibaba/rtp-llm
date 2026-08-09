package org.flexlb.balance.scheduler.priority;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.enums.DecodeTaskPhase;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
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
 * accepted, before any kvBucket/deadline tie-break), the higher accepted
 * victim cost (g=4), and the running layer never joining the pool.
 */
class AcceptedEvictionPlannerTest {

    private final FlexlbConfig config = new FlexlbConfig();
    private final EngineCancelChannel channel = mock(EngineCancelChannel.class);

    // ==================== gate matrix ====================

    @Test
    void gateOff_acceptedNeverConsidered_andChannelNeverQueried() {
        // Default config: accepted-evict switch off.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256, 1_000)), List.of());

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
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        when(channel.isSupported(any())).thenReturn(false);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256, 1_000)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    @Test
    void gateOn_supportedEndpoint_acceptedBecomesVictimWithHigherCost() {
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 5, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256, 1_000)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        assertEquals(List.of(1L), ids(proposal.victims()));
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, proposal.victims().get(0).phase());
        // h(DECODE_SLOT_FULL)=4 x f(30)=1 x g(ACCEPTED_NOT_RUNNING)=4 — an
        // accepted victim costs 4x its reserved counterpart.
        assertEquals(16, proposal.totalCost());
        assertEquals(256, proposal.freedKvTokens());
    }

    // ==================== 3.3: strict boundary holds on the accepted layer ====================

    @Test
    void equalPriorityAccepted_neverYields() {
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 50, 256, 1_000)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    // ==================== stage asc: reserved preferred over accepted ====================

    @Test
    void samePriority_reservedIsEvictedBeforeAccepted() {
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        when(channel.isSupported(any())).thenReturn(true);
        // The accepted entry has more deadline slack — without the stage
        // tie-break it would be picked first; stage asc must win.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 2, 2,
                List.of(reserved(1, 30, 128, 1_000)),
                List.of(accepted(2, 30, 256, 9_000)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(List.of(1L), ids(proposal.victims()));
        assertEquals(DecodeTaskPhase.RESERVED_NOT_ACCEPTED, proposal.victims().get(0).phase());
        // Reserved victim keeps the cheap g=1 cost: 4 x 1 x 1.
        assertEquals(4, proposal.totalCost());
    }

    @Test
    void kvFull_stagePrecedesKvBucket() {
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        when(channel.isSupported(any())).thenReturn(true);
        // kvDeficit = 400; the accepted entry would win a pure kvBucket-desc
        // order (2048 > 512) but stage asc ranks the reserved entry first,
        // and 512 already covers the deficit.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100, 10_000, 1, 0,
                List.of(reserved(1, 30, 512, 1_000)),
                List.of(accepted(2, 30, 2_048, 2_000)), List.of());

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
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        when(channel.isSupported(any())).thenReturn(true);
        // kvDeficit = 1900: reserved 512 alone is short, the greedy pass
        // extends into the accepted layer.
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100, 10_000, 2, 0,
                List.of(reserved(1, 30, 512, 1_000)),
                List.of(accepted(2, 30, 2_048, 2_000)), List.of());

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(50, 2_000), List.of(ep), config, channel, failures);

        assertNotNull(proposal);
        assertEquals(List.of(1L, 2L), ids(proposal.victims()));
        assertEquals(2_560, proposal.freedKvTokens());
        // 8 x (f(30)=1 x g(RESERVED)=1 x waste(512)=1
        //      + f(30)=1 x g(ACCEPTED)=4 x waste(2048)=1) = 8 x 5 = 40.
        assertEquals(40, proposal.totalCost());
    }

    // ==================== running layer is never a candidate source ====================

    @Test
    void runningEntries_areNeverCandidatesEvenBehindTheGate() {
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        when(channel.isSupported(any())).thenReturn(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(),
                List.of(new DecodeRequestSnapshot(1, 30, DecodeTaskPhase.RUNNING,
                        256, 256, 1_000)));

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                incoming(70, 128), List.of(ep), config, channel, failures);

        assertNull(proposal);
        assertEquals("insufficient_lower_priority_candidates", failures.get("d1"));
    }

    // ==================== Phase 4 overload: accepted layer disabled ====================

    @Test
    void legacyOverload_withoutChannel_neverUsesAcceptedLayer() {
        config.setAutoTpmDecodeAcceptedEvictEnabled(true);
        DecodeEndpointSnapshot ep = endpoint("d1", 1, 100_000, 200_000, 1, 1,
                List.of(), List.of(accepted(1, 30, 256, 1_000)), List.of());

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

    private static DecodeEndpointSnapshot endpoint(String id, long version,
                                                   long realKvAvailable, long realKvTotal,
                                                   int totalLoad, long concurrencyLimit,
                                                   List<DecodeRequestSnapshot> reserved,
                                                   List<DecodeRequestSnapshot> accepted,
                                                   List<DecodeRequestSnapshot> running) {
        long hardKv = reserved.stream().mapToLong(DecodeRequestSnapshot::kvTokens).sum();
        long expectedKv = reserved.stream().mapToLong(DecodeRequestSnapshot::expectedKvTokens).sum();
        return new DecodeEndpointSnapshot(null, id, version, realKvAvailable, realKvTotal,
                totalLoad, concurrencyLimit, hardKv, expectedKv, reserved, accepted, running);
    }

    private static DecodeRequestSnapshot reserved(long requestId, int priority,
                                                  long kvTokens, long deadlineMs) {
        return new DecodeRequestSnapshot(requestId, priority,
                DecodeTaskPhase.RESERVED_NOT_ACCEPTED, kvTokens, kvTokens + 8, deadlineMs);
    }

    private static DecodeRequestSnapshot accepted(long requestId, int priority,
                                                  long kvTokens, long deadlineMs) {
        return new DecodeRequestSnapshot(requestId, priority,
                DecodeTaskPhase.ACCEPTED_NOT_RUNNING, kvTokens, kvTokens, deadlineMs);
    }

    private static PriorityRequestEnvelope incoming(int priority, long seqLen) {
        return new PriorityRequestEnvelope(999, priority, seqLen, 8,
                System.currentTimeMillis(), 10_000, System.currentTimeMillis() + 10_000,
                seqLen, seqLen + 8);
    }
}
