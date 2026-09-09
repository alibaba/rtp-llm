package org.flexlb.balance.endpoint;

import org.flexlb.balance.eviction.DecodeEndpointSnapshot;
import org.flexlb.balance.eviction.DecodeEvictionProposal;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.eviction.EvictionPlanner;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Exercise one demand and budget through selection, planning and authoritative commit. */
class DecodeCapacityPreemptionTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void outputBudgetRejectionProducesAPlanAndCommitRechecksCurrentUsage(boolean usageChanged) {
        DecodeEndpoint endpoint = endpoint(400L);
        DecodeEndpoint.AdmissionCapacity policy = new DecodeEndpoint.AdmissionCapacity(0L, 90L);
        long hardKvTokens = 100L;
        long expectedKvTokens = 250L;
        DecodeEndpoint.ReservationHandle victim;
        try (var pin = endpoint.tryPinGeneration()) {
            victim = endpoint.reservePinned(pin, 1L, 100L, 200L, 30);
            var reservation = endpoint.tryReservePlacementPinned(pin, 9L,
                    hardKvTokens, expectedKvTokens, 70);
            assertNotNull(reservation);
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                    endpoint.acquireEngineDispatchPermit(reservation, policy).status());
            endpoint.releaseReservationExact(reservation);
        }
        var view = endpoint.routingView();
        assertTrue(view.realKvAvailable() >= hardKvTokens);
        assertFalse(policy.evaluate(view.dispatchUsage(), hardKvTokens, expectedKvTokens).fits());
        DecodeEvictionProposal proposal = plan(endpoint, hardKvTokens, expectedKvTokens, policy, VictimStage.DECODE_ENGINE_OWNED);
        assertEquals(DecodeEvictionProposal.CASE_KV, proposal.evictionCase());
        assertEquals(List.of(1L), proposal.victims().stream().map(DecodeEndpoint.DecodeRequestView::requestId).toList());
        if (usageChanged) { updateCapacity(endpoint, 250L); }
        assertEquals(usageChanged ? DecodeEndpoint.PreemptionBeginResult.INFEASIBLE
                        : DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                endpoint.beginPriorityPreemption(1L, List.of(victim), 9L,
                        hardKvTokens, expectedKvTokens, 70, policy));
        assertNotNull(endpoint.reservationHandle(1L));
        if (usageChanged) { assertNull(endpoint.reservationHandle(9L)); }
    }

    @Test
    void queuedReservationReleasesThePlacementRequestCharge() {
        DecodeEndpoint endpoint = endpoint(1000L);
        DecodeEndpoint.AdmissionCapacity policy = new DecodeEndpoint.AdmissionCapacity(1L, 90L);
        long hardKvTokens = 100L;
        long expectedKvTokens = 100L;
        DecodeEndpoint.ReservationHandle victim;
        try (var pin = endpoint.tryPinGeneration()) {
            victim = endpoint.tryReservePlacementPinned(pin, 1L, 100L, 200L, 30);
            assertNull(endpoint.tryReservePlacementPinned(pin, 9L, hardKvTokens,
                    expectedKvTokens, 70, policy));
        }
        // The same queued reservation remains soft for ordinary Engine dispatch.
        assertTrue(policy.evaluate(endpoint.routingView().dispatchUsage(), hardKvTokens, expectedKvTokens).fits());
        DecodeEvictionProposal proposal = plan(endpoint, hardKvTokens, expectedKvTokens, policy, VictimStage.DECODE_RESERVED);
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        assertTrue(endpoint.tryEvictLocalReservationsAndReserveIncoming(List.of(victim), 9L,
                hardKvTokens, expectedKvTokens, 70, policy));
        assertNull(endpoint.reservationHandle(1L));
        assertNotNull(endpoint.reservationHandle(9L));
    }

    @Test
    void zeroPromptVictimCanReleaseItsFullOutputReservation() {
        DecodeEndpoint endpoint = endpoint(1000L);
        DecodeEndpoint.AdmissionCapacity policy = new DecodeEndpoint.AdmissionCapacity(0L, 90L);
        long hardKvTokens = 100L;
        long expectedKvTokens = 200L;
        DecodeEndpoint.ReservationHandle victim;
        try (var pin = endpoint.tryPinGeneration()) {
            victim = endpoint.tryReservePlacementPinned(pin, 1L, 0L, 900L, 30);
        }
        DecodeEvictionProposal proposal = plan(endpoint, hardKvTokens, expectedKvTokens, policy, VictimStage.DECODE_RESERVED);
        assertEquals(List.of(1L), proposal.victims().stream().map(DecodeEndpoint.DecodeRequestView::requestId).toList());
        assertTrue(endpoint.tryEvictLocalReservationsAndReserveIncoming(List.of(victim), 9L,
                hardKvTokens, expectedKvTokens, 70, policy));
        assertEquals(200L, endpoint.routingView().inflightExpectedKv());
    }

    private static DecodeEvictionProposal plan(DecodeEndpoint endpoint, long hardKvTokens, long expectedKvTokens,
                                                DecodeEndpoint.AdmissionCapacity policy, VictimStage stage) {
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(stage));
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        when(channel.isSupported(endpoint)).thenReturn(true);
        var failures = new HashMap<String, String>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(70, hardKvTokens, expectedKvTokens,
                List.of(DecodeEndpointSnapshot.capture(endpoint, policy)), preemption, channel, failures);
        assertNotNull(proposal, failures.toString());
        return proposal;
    }

    private static DecodeEndpoint endpoint(long availableKv) {
        DecodeEndpoint endpoint = new DecodeEndpoint(
                EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.1", 8080, 8081),
                EndpointTestSupport.noopEventSink());
        updateCapacity(endpoint, availableKv);
        return endpoint;
    }

    private static void updateCapacity(DecodeEndpoint endpoint, long availableKv) {
        WorkerStatusResponse status = new WorkerStatusResponse();
        status.setTotalKvCacheTokens(1000L);
        status.setAvailableKvCacheTokens(availableKv);
        EndpointTestSupport.applyStatus(endpoint, status).run();
    }
}
