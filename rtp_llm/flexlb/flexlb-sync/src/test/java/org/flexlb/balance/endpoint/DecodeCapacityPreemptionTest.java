package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.CapacityRelease;

import org.flexlb.config.FlexlbConfig;
import static org.flexlb.balance.scheduler.SchedulingTestConfig.decodeRequirements;
import org.flexlb.balance.eviction.DecodeEvictionProposal;
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
        DecodeResources.AdmissionCapacity policy = new DecodeResources.AdmissionCapacity(0L, 90L);
        long hardKvTokens = 100L;
        long expectedKvTokens = 250L;
        DecodeResources.ReservationHandle victim;
        try (var pin = endpoint.tryPinGeneration()) {
            victim = endpoint.tryReserveQueuedRequest(pin, 1L, 100L, 200L, 30, null);
            assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    endpoint.acquireDispatchPermit(victim, new DecodeResources.AdmissionCapacity(0, 100L)).permit().dispatch());
            var reservation = endpoint.tryReserveQueuedRequest(pin, 9L, hardKvTokens, expectedKvTokens, 70, null);
            assertNotNull(reservation);
            assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                    endpoint.acquireDispatchPermit(reservation, policy).status());
            endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }
        var view = endpoint.routingView();
        assertTrue(view.realKvAvailable() >= hardKvTokens);
        assertFalse(policy.evaluate(view.dispatchUsage(), hardKvTokens, expectedKvTokens, CapacityRelease.NONE).fits());
        DecodeEvictionProposal proposal = plan(endpoint, hardKvTokens, expectedKvTokens, policy, VictimStage.DECODE_ENGINE_OWNED);
        assertEquals(DecodeEvictionProposal.CASE_KV, proposal.evictionCase());
        assertEquals(List.of(1L), proposal.victims().stream().map(DecodeResources.DecodeRequestView::requestId).toList());
        if (usageChanged) { updateCapacity(endpoint, 250L); }
        assertEquals(usageChanged ? DecodeResources.PreemptionBeginResult.INFEASIBLE
                        : DecodeResources.PreemptionBeginResult.SUCCESS,
                endpoint.beginPreemption(1L, List.of(victim), 9L, hardKvTokens, expectedKvTokens, 70, policy));
        assertNotNull(EndpointTestSupport.decodeReservation(endpoint, 1L));
        if (usageChanged) { assertNull(EndpointTestSupport.decodeReservation(endpoint, 9L)); }
    }

    @Test
    void queuedReservationReleasesThePlacementRequestCharge() {
        DecodeEndpoint endpoint = endpoint(1000L);
        DecodeResources.AdmissionCapacity policy = new DecodeResources.AdmissionCapacity(1L, 90L);
        long hardKvTokens = 100L;
        long expectedKvTokens = 100L;
        DecodeResources.ReservationHandle victim;
        try (var pin = endpoint.tryPinGeneration()) {
            victim = endpoint.tryReserveQueuedRequest(pin, 1L, 100L, 200L, 30, null);
            assertNull(endpoint.tryReserveQueuedRequest(pin, 9L, hardKvTokens, expectedKvTokens, 70, policy));
        }
        // The same queued reservation remains soft for ordinary Engine dispatch.
        assertTrue(policy.evaluate(endpoint.routingView().dispatchUsage(), hardKvTokens, expectedKvTokens, CapacityRelease.NONE).fits());
        DecodeEvictionProposal proposal = plan(endpoint, hardKvTokens, expectedKvTokens, policy, VictimStage.DECODE_RESERVED);
        assertEquals(DecodeEvictionProposal.CASE_SLOT, proposal.evictionCase());
        assertNotNull(endpoint.replaceQueuedRequests(List.of(victim), 9L, hardKvTokens, expectedKvTokens, 70, policy));
        assertNull(EndpointTestSupport.decodeReservation(endpoint, 1L));
        assertNotNull(EndpointTestSupport.decodeReservation(endpoint, 9L));
    }

    @Test
    void zeroPromptVictimCanReleaseItsFullOutputReservation() {
        DecodeEndpoint endpoint = endpoint(1000L);
        DecodeResources.AdmissionCapacity policy = new DecodeResources.AdmissionCapacity(0L, 90L);
        long hardKvTokens = 100L;
        long expectedKvTokens = 200L;
        DecodeResources.ReservationHandle victim;
        try (var pin = endpoint.tryPinGeneration()) {
            victim = endpoint.tryReserveQueuedRequest(pin, 1L, 0L, 900L, 30, null);
        }
        DecodeEvictionProposal proposal = plan(endpoint, hardKvTokens, expectedKvTokens, policy, VictimStage.DECODE_RESERVED);
        assertEquals(List.of(1L), proposal.victims().stream().map(DecodeResources.DecodeRequestView::requestId).toList());
        assertNotNull(endpoint.replaceQueuedRequests(List.of(victim), 9L, hardKvTokens, expectedKvTokens, 70, policy));
        assertEquals(200L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
    }

    private static DecodeEvictionProposal plan(DecodeEndpoint endpoint, long hardKvTokens, long expectedKvTokens,
                                                DecodeResources.AdmissionCapacity policy, VictimStage stage) {
        PreemptionConfig preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(EnumSet.of(stage));
        var failures = new HashMap<String, String>();
        var request = decodeRequirements(70, hardKvTokens, expectedKvTokens, policy);
        var snapshot = endpoint.resourceSnapshot();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(request, snapshot, preemption, failures).proposal();
        assertNotNull(proposal, failures.toString());
        return proposal;
    }

    private static DecodeEndpoint endpoint(long availableKv) {
        DecodeEndpoint endpoint = EndpointTestSupport.decode(EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.1", 8080, 8081), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
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
