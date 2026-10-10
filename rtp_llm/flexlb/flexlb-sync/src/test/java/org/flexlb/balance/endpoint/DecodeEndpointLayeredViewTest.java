package org.flexlb.balance.endpoint;

import org.flexlb.balance.endpoint.DecodeResources.ReservationReleaseResult;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.eviction.EvictionPlanner;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexMetricTags;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.flexlb.balance.scheduler.SchedulingTestConfig.decodeRequirements;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_RESERVED_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_SHADOW_KV_RESERVED;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_RUNNING_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_ACCEPTED_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_ENGINE_LOAD;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Phase 5 tests for the decode layered view: calibrate splits confirmed
 * requests into accepted / running layers with priority inheritance,
 * the registry follows the WorkerStatus reports (retain / finished / TTL),
 * the shadow accounting invariants stay byte-for-byte Phase 4, and
 * the token-fenced weak-ACK transaction is all-or-nothing, retains victim
 * accounting until typed CANCELED, and provisionally reserves the incoming.
 */
class DecodeEndpointLayeredViewTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeResources.ReservationHandle> reservations =
            new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(EndpointTestSupport.noopEventSink()));
        updateStatus(Map.of(), Map.of(), 20_000);
    }

    // ==================== calibrate: layer split + inheritance ====================

    @Test
    void heartbeatCountsOnlyConfirmedOwnersRetainedAfterReconciliation() {
        for (long id = 1; id <= 4; id++) {
            reserve(id, 128, 128, 30);
        }
        updateStatus(Map.of(
                "1", runningTask(1L, TaskPhase.RUNNING, 128),
                "2", runningTask(2L, TaskPhase.RUNNING, 128),
                "3", runningTask(3L, TaskPhase.RUNNING, 128),
                "4", runningTask(4L, TaskPhase.RUNNING, 128)), null, 19_488);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(800L, List.of(1L), 9L, 128, 128, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 800L));

        // One synthetic claim, one regressed report, one current allocation,
        // and one ordinary owner absent from the FULL snapshot.
        for (int round = 0; round < 2; round++) {
            updateStatus(Map.of(
                    "2", runningTask(2L, TaskPhase.RECEIVED, 0),
                    "3", runningTask(3L, TaskPhase.RUNNING, 128)), null, 19_872);
            assertEquals(3, endpoint.resourceSnapshot().confirmedCount());
            assertEquals(4, endpoint.routingView().totalLoad());
            assertFalse(isConfirmed(4L));
        }
    }

    @Test
    void calibrate_splitsConfirmedIntoAcceptedAndRunningLayers() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 500, 508, 40);

        TaskInfo accepted = runningTask(1L, TaskPhase.KV_ALLOCATED, 256);
        TaskInfo running = runningTask(2L, TaskPhase.RUNNING, 512);
        updateStatus(Map.of("1", accepted, "2", running), null, 10_000);

        assertEquals(1, endpoint.resourceSnapshot().acceptedCount());
        assertEquals(1, endpoint.resourceSnapshot().runningCount());
        assertEquals(2, endpoint.resourceSnapshot().confirmedCount());
        assertEquals(0, endpoint.resourceSnapshot().reservedCount());
        assertTrue(isConfirmed(1L));
        assertTrue(isConfirmed(2L));

        // Layered view inherits priority from the shadow entry
        // removed this round; KV is the reported inputLength estimate.
        DecodeResources.DecodeRequestView acceptedView = confirmedView(1L);
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, acceptedView.phase());
        assertEquals(30, acceptedView.priority());
        assertTrue(acceptedView.priorityKnown());
        assertEquals(256, acceptedView.kvTokens());
        assertFalse(acceptedView.claimedForPreemption());

        DecodeResources.DecodeRequestView runningView = confirmedView(2L);
        assertEquals(DecodeTaskPhase.RUNNING, runningView.phase());
        assertEquals(40, runningView.priority());
        assertTrue(runningView.priorityKnown());
    }

    @Test
    void calibrate_unknownConfirmedFallsBackToNoPriority() {
        // Report precedes any reserve: no shadow entry to inherit from.
        // Task40: the fallback is the no-priority sentinel (0), which keeps
        // untracked engine tasks out of every eviction candidate set.
        updateStatus(Map.of("9", runningTask(9L, TaskPhase.KV_ALLOCATED, 64)), null, 10_000);

        DecodeResources.DecodeRequestView view = confirmedView(9L);
        assertEquals(0, view.priority());
        assertFalse(view.priorityKnown());
        assertEquals(64, view.kvTokens());
    }

    @Test
    void calibrate_promotesAcceptedToRunningOnRefresh() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertEquals(1, endpoint.resourceSnapshot().acceptedCount());

        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);

        assertEquals(0, endpoint.resourceSnapshot().acceptedCount());
        assertEquals(1, endpoint.resourceSnapshot().runningCount());
        // Identity fields stay from first sight.
        assertEquals(30, confirmedView(1L).priority());
    }

    @Test
    void periodicAdmissionMetricsExposePhaseSplitByIpAndPort() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 400, 408, 40);
        updateStatus(Map.of(
                "1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256),
                "2", runningTask(2L, TaskPhase.RUNNING, 256)), null, 10_000);
        reserve(3L, 300, 308, 50);
        FlexMonitor monitor = mock(FlexMonitor.class);

        endpoint.reportAdmissionMetrics(new RequestSchedulerReporter(monitor));

        FlexMetricTags tags = FlexMetricTags.of("endpoint", "10.0.0.1:8080");
        verify(monitor).report(AUTO_TPM_DECODE_RESERVED_COUNT, tags, 1.0);
        verify(monitor).report(AUTO_TPM_DECODE_SHADOW_KV_RESERVED, tags, 300.0);
        verify(monitor).report(AUTO_TPM_DECODE_RUNNING_COUNT, tags, 1.0);
        verify(monitor).report(AUTO_TPM_DECODE_ACCEPTED_COUNT, tags, 1.0);
        verify(monitor).report(AUTO_TPM_DECODE_ENGINE_LOAD, tags, 3.0);
        verifyNoMoreInteractions(monitor);
    }

    // ==================== calibrate: registry follows the reports ====================

    @Test
    void calibrate_dropsEntriesNoLongerReported() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertTrue(isConfirmed(1L));

        // Next report no longer lists the request as confirmed — this is the
        // release-confirmation signal the accepted-eviction wait polls for.
        updateStatus(Map.of(), null, 10_000);

        assertFalse(isConfirmed(1L));
        assertEquals(0, endpoint.resourceSnapshot().acceptedCount());
        assertEquals(0, endpoint.resourceSnapshot().confirmedCount());
    }

    @Test
    void calibrate_finishedRemovesTrackedEntry() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertTrue(isConfirmed(1L));

        // Same round lists it both running and finished: finished wins.
        TaskInfo finished = runningTask(1L, TaskPhase.RUNNING, 256);
        finished.setErrorCode(0);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)),
                Map.of("1", finished), 10_000);

        assertFalse(isConfirmed(1L));
    }

    @Test
    void evictExpiredRequests_purgesStaleTrackedEntries() throws InterruptedException {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertTrue(isConfirmed(1L));

        Thread.sleep(5);
        long versionBefore = endpoint.routingView().admissionVersion();
        endpoint.evictExpiredRequests(1, requestId -> false);

        assertFalse(isConfirmed(1L));
        assertEquals(0, endpoint.resourceSnapshot().confirmedCount(),
                "tracked TTL removal must release the published confirmed slot");
        assertEquals(0, endpoint.routingView().totalLoad());
        assertTrue(endpoint.routingView().admissionVersion() > versionBefore);
    }

    @Test
    void evictExpiredRequests_boundsPriorityCanceledTerminalRecords() throws InterruptedException {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 128, 136, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));
        assertTrue(endpoint.updatePreemption(101L, DecodeResources.PreemptionUpdate.canceled(reservations.get(1L))));
        assertNotNull(endpoint.commitPreemption(101L));

        // A delayed Decode report cannot resurrect a recently canceled victim.
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertFalse(isConfirmed(1L));

        Thread.sleep(5);
        endpoint.evictExpiredRequests(1, requestId -> false);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertTrue(isConfirmed(1L),
                "the cancel fence follows the configured terminal retention TTL");
    }

    @Test
    void priorityTerminalRecordIsAuthoritativeWithoutAcceptedOrWorkerCanceled() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(102L, List.of(1L),
                        9L, 128, 136, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 102L));

        assertTrue(endpoint.updatePreemption(102L, DecodeResources.PreemptionUpdate.fenced(reservations.get(1L))));
        assertNotNull(endpoint.commitPreemption(102L));

        assertFalse(isConfirmed(1L));
        assertTrue(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 9L));
        assertEquals(1, endpoint.routingView().totalLoad());
        // The same late Decode sample rejected by typed-CANCELED fencing must
        // also be rejected after the stronger absent+terminal record proof.
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertFalse(isConfirmed(1L));
    }

    // ==================== accounting invariants unchanged (iron rule 5) ====================

    @Test
    void accounting_invariantsStayPhase4Equivalent() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 300, 308, 40);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        // realKvAvailable = reportedKvAvailable - remaining shadow hard KV.
        assertEquals(10_000 - 300, endpoint.routingView().realKvAvailable());
        assertEquals(300, endpoint.routingView().inflightHardKv());
        // totalLoad = confirmed Engine-owned count + reserved inflight count.
        assertEquals(2, endpoint.routingView().totalLoad());
        assertEquals(1, endpoint.resourceSnapshot().reservedCount());
    }

    // ==================== token-fenced weak-ACK preemption ====================

    @Test
    void beginPriorityPreemption_claimsVictimAndProvisionallyReservesIncoming() {
        reserve(2L, 400, 408, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();

        DecodeResources.PreemptionBeginResult result = beginPreemption(
                101L, List.of(2L), 9L, 700, 708, 70);

        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS, result);
        // Weak ACK boundary: victim accounting is untouched and the incoming
        // reservation is provisional until typed Prefill CANCELED settles it.
        assertTrue(isConfirmed(2L));
        assertTrue(confirmedView(2L).claimedForPreemption());
        assertTrue(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 9L));
        assertEquals(700, endpoint.routingView().inflightHardKv());
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));
        assertTrue(isConfirmed(2L));
        assertTrue(endpoint.routingView().admissionVersion() > version);
    }

    @Test
    void cancelAckAndUnknownKeepVictimCapacityUntilItsOwnInactivityExpiry() {
        var victim = reserve(2L, 400L, 408L, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(2L), 9L, 700L, 708L, 70));
        var incoming = EndpointTestSupport.decodeReservation(endpoint, 9L);
        assertTrue(incoming != null);
        var before = endpoint.resourceSnapshot();
        assertEquals(1, before.runningCount());
        assertEquals(700L, before.routing().inflightHardKv());
        assertEquals(708L, EndpointTestSupport.expectedReservedKv(before));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));
        assertEquals(before.engineCapacityUsed(), endpoint.resourceSnapshot().engineCapacityUsed());
        assertEquals(EndpointTestSupport.expectedReservedKv(before), EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
        assertEquals(1, endpoint.resourceSnapshot().runningCount());

        // Expiring the incoming request must not release a victim whose Cancel outcome is unknown.
        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(incoming, DecodeResources.ReleaseReason.EXPIRED));
        assertEquals(0L, endpoint.routingView().inflightHardKv());
        assertEquals(0L, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
        assertEquals(1, endpoint.resourceSnapshot().runningCount());
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(victim, DecodeResources.ReleaseReason.EXPIRED));
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(victim, DecodeResources.ReleaseReason.EXPIRED));
        var after = endpoint.resourceSnapshot();
        assertEquals(0, after.runningCount());
        assertEquals(0, after.acceptedCount());
        assertEquals(0, after.activeDispatchPermits());
        assertEquals(0, after.engineCapacityUsed());
        assertTrue(after.reservedCount() == 0);
        assertTrue(after.confirmedCount() == 0);
    }

    @Test
    void beginPriorityPreemption_exactIdentityMismatch_appliesNothing() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        DecodeResources.ReservationHandle exact = reservations.get(1L);
        DecodeResources.ReservationHandle stale =
                new DecodeResources.ReservationHandle(
                        exact.endpointGenerationId(),
                        exact.requestId(),
                        exact.reservationToken() + 1L);
        DecodeResources.PreemptionBeginResult result =
                endpoint.beginPreemption(101L, List.of(stale), 9L, 700, 708, 70, new DecodeResources.AdmissionCapacity(1, 100));

        assertEquals(DecodeResources.PreemptionBeginResult.VICTIM_GONE, result);
        assertFalse(confirmedView(1L).claimedForPreemption());
        assertFalse(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 9L));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void beginPreemptionRejectsRepeatedVictimWithoutMutatingOwnership(boolean staleSecondToken) {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        var exact = reservations.get(1L);
        var second = staleSecondToken ? new DecodeResources.ReservationHandle(
                exact.endpointGenerationId(), exact.requestId(), exact.reservationToken() + 1) : exact;
        long version = endpoint.placementVersion();

        assertEquals(DecodeResources.PreemptionBeginResult.VICTIM_GONE,
                endpoint.beginPreemption(101L, List.of(exact, second), 9L, 700, 708, 70,
                        new DecodeResources.AdmissionCapacity(1, 100)));

        assertEquals(version, endpoint.placementVersion());
        assertFalse(confirmedView(1L).claimedForPreemption());
        assertFalse(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 9L));
    }

    @Test
    void retirementReportsActivePreemptionOwnersExactlyOnce() {
        var scheduler = mock(org.flexlb.balance.scheduler.AbstractRequestScheduler.class);
        endpoint = EndpointTestSupport.decode(status, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler));
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L), 9L, 700, 708, 70));
        var incoming = EndpointTestSupport.decodeReservation(endpoint, 9L);
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));

        var requests = org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler);
        var victimContext = requests.findActive(1L);
        var incomingContext = requests.findActive(9L);
        when(requests.findActive(1L)).thenReturn(victimContext);
        when(requests.findActive(9L)).thenReturn(incomingContext);
        endpoint.close();
        endpoint.close();

        verify(scheduler).onDecodeGenerationRetired(victimContext, endpoint, reservations.get(1L));
        verify(scheduler).onDecodeGenerationRetired(incomingContext, endpoint, incoming);
        assertTrue(endpoint.resourceSnapshot().reservedCount() == 0);
        assertTrue(endpoint.resourceSnapshot().confirmedCount() == 0);
        assertNull(endpoint.commitPreemption(101L));
        assertFalse(endpoint.abortPreemption(101L));
    }

    @Test
    void beginPriorityPreemption_victimGone_isAllOrNothing() {
        reserve(2L, 400, 408, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();

        assertEquals(DecodeResources.PreemptionBeginResult.VICTIM_GONE,
                beginPreemption(101L, List.of(2L, 999L),
                        9L, 700, 708, 70));
        assertFalse(confirmedView(2L).claimedForPreemption());
        assertFalse(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 9L));
        assertEquals(version, endpoint.routingView().admissionVersion());
    }

    @Test
    void beginPriorityPreemption_acceptsRunningAndRejectsAlreadyClaimedVictims() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 400, 408, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256),
                "2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        // RUNNING is engine-owned too and follows the same cancel path.
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));

        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(102L, List.of(2L),
                        10L, 700, 708, 70));
        assertEquals(DecodeResources.PreemptionBeginResult.VICTIM_ALREADY_CLAIMED,
                beginPreemption(103L, List.of(2L),
                        11L, 700, 708, 70));
    }

    @Test
    void ttlEvictionCannotReleaseClaimedEngineVisibleShadow() throws Exception {
        reserve(1L, 500, 508, 30);
        // Keep a wide age gap so the provisional incoming reservation cannot
        // become TTL-eligible merely because this test runs on a loaded JVM.
        Thread.sleep(150);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));

        assertEquals(0, endpoint.evictExpiredRequests(
                100, requestId -> false));
        assertTrue(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 1L),
                "generic TTL cleanup must not deduct a claimed victim");
        assertEquals(1_200, endpoint.routingView().inflightHardKv(),
                "victim and provisional incoming remain fully charged");

        endpoint.abortPreemption(101L);
        assertEquals(0, endpoint.evictExpiredRequests(100, requestId -> false));
        assertTrue(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 1L), "uncertain Cancel survives attempt rollback");
        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservations.get(1L), DecodeResources.ReleaseReason.EXPIRED));
        assertFalse(EndpointTestSupport.isReserved(endpoint.resourceSnapshot(), 1L));
        assertEquals(0, endpoint.routingView().inflightHardKv());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void claimedShadowConfirmedLaterRetainsCapacityUntilTerminal(boolean receivedAgain) {
        DecodeResources.ReservationHandle victim = reserve(1L, 500, 508, 30);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(105L, List.of(1L), 9L, 700, 708, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 105L));

        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertTrue(endpoint.isAcceptedByEngine(victim));
        assertTrue(confirmedView(1L).claimedForPreemption());
        assertEquals(700, endpoint.routingView().inflightHardKv());

        Map<String, TaskInfo> active = receivedAgain
                ? Map.of("1", runningTask(1L, TaskPhase.RECEIVED, 0)) : Map.of();
        updateStatus(active, null, 10_000);
        updateStatus(active, null, 10_000);
        assertEquals(DecodeTaskPhase.RUNNING, confirmedView(1L).phase());
        assertEquals(2, endpoint.routingView().totalLoad());
        assertEquals(8_800, endpoint.routingView().realKvAvailable(),
                "missing allocation evidence must retain the exact victim's synthetic KV once");
        assertEquals(DecodeResources.ReservationReleaseResult.ENGINE_ACCEPTED,
                endpoint.release(victim, DecodeResources.ReleaseReason.NOT_SENT));
        assertNull(endpoint.commitPreemption(105L));

        assertTrue(endpoint.updatePreemption(105L, DecodeResources.PreemptionUpdate.canceled(victim)));
        assertFalse(endpoint.updatePreemption(105L, DecodeResources.PreemptionUpdate.canceled(victim)));
        assertEquals(1, endpoint.routingView().totalLoad());
        assertEquals(9_300, endpoint.routingView().realKvAvailable());
        var incoming = EndpointTestSupport.decodeReservation(endpoint, 9L);
        assertNotNull(incoming);
        assertEquals(incoming, endpoint.commitPreemption(105L));
        assertNull(endpoint.commitPreemption(105L));
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertFalse(isConfirmed(1L), "late status must not resurrect the settled victim");
        assertEquals(1, endpoint.routingView().totalLoad());
    }

    @Test
    void activeAfterNotFoundReleasesSyntheticHeldKvWithClaim() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 101L));
        endpoint.abortPreemption(101L);

        // Decode disappears while NOT_FOUND is being reconciled. Its KV is
        // conservatively held until the original Prefill reports it active.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(9_500, endpoint.routingView().realKvAvailable());

        assertTrue(endpoint.updatePreemption(101L, DecodeResources.PreemptionUpdate.active(reservations.get(1L))));
        assertEquals(10_000, endpoint.routingView().realKvAvailable(),
                "active reconciliation must release held KV before dropping the claim");
        assertFalse(confirmedView(1L).claimedForPreemption());
    }

    @Test
    void notFoundRetainsSyntheticKvUntilExactLeaseExpiration() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(104L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 104L));

        // Decode disappearance moves the victim's 500-token charge into a
        // synthetic hold. The 700-token provisional incoming reservation is
        // still independently owned by the live preemption attempt.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(700, endpoint.routingView().inflightHardKv());
        assertEquals(2, endpoint.routingView().totalLoad(),
                "victim and provisional incoming must both remain charged before abort");
        assertEquals(8_800, endpoint.routingView().realKvAvailable());
        var beforeAbort = endpoint.routingViewSnapshot(endpoint.ipPort());
        assertTrue(endpoint.abortPreemption(104L));
        var afterAbort = endpoint.routingViewSnapshot(endpoint.ipPort());
        assertTrue(afterAbort.admissionVersion() > beforeAbort.admissionVersion());
        assertEquals(0, afterAbort.inflightHardKv(), "abort must invalidate the cached capacity view");
        assertEquals(700, beforeAbort.inflightHardKv(), "previous snapshots remain immutable");

        assertEquals(0, endpoint.routingView().inflightHardKv(),
                "aborting the attempt releases only its provisional incoming reservation");
        assertEquals(9_500, endpoint.routingView().realKvAvailable(),
                "aborting incoming work must not release the victim KV hold");
        assertEquals(1, endpoint.routingView().totalLoad(),
                "the disappeared confirmed victim remains a synthetic slot");

        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservations.get(1L), DecodeResources.ReleaseReason.EXPIRED));
        assertEquals(10_000, endpoint.routingView().realKvAvailable());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertNotEquals(ReservationReleaseResult.RELEASED, endpoint.release(reservations.get(1L), DecodeResources.ReleaseReason.EXPIRED),
                "the exact local lease expires at most once");
        assertEquals(10_000, endpoint.routingView().realKvAvailable());
    }

    // ==================== snapshot capture: request phases ====================

    @Test
    void snapshotCapture_preservesPhasesAndPlannerExcludesClaimedVictims() {
        reserve(1L, 500, 508, 70);
        reserve(2L, 400, 408, 30);
        reserve(3L, 300, 308, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256),
                "3", runningTask(3L, TaskPhase.RUNNING, 512)), null, 10_000);

        var snapshot = endpoint.resourceSnapshot();
        assertEquals(List.of(1L, 2L, 3L), snapshot.requests().keySet().stream().sorted().toList());
        assertEquals(DecodeTaskPhase.LOCAL_RESERVED,
                snapshot.requests().values().stream().filter(item -> item.requestId() == 1L).findFirst().orElseThrow().phase());
        assertEquals(DecodeTaskPhase.RUNNING,
                snapshot.requests().values().stream().filter(item -> item.requestId() == 3L).findFirst().orElseThrow().phase());
        DecodeRequestView accepted = snapshot.requests().values().stream()
                .filter(item -> item.requestId() == 2L).findFirst().orElseThrow();
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, accepted.phase());
        assertEquals(256, accepted.kvTokens());

        // A cancel-requested entry is claimed by an in-flight eviction and
        // must not be offered to planning again.
        beginPreemption(101L, List.of(2L),
                20L, 64, 72, 70);
        beginPreemption(102L, List.of(3L),
                30L, 64, 72, 70);
        var after = endpoint.resourceSnapshot();
        assertTrue(after.requests().get(2L).claimedForPreemption());
        assertTrue(after.requests().get(3L).claimedForPreemption());
        assertFalse(snapshot.requests().get(2L).claimedForPreemption());
        assertFalse(snapshot.requests().get(3L).claimedForPreemption());
        assertThrows(UnsupportedOperationException.class, () -> snapshot.requests().clear());
        var request = decodeRequirements(70, 0L, 0L, new DecodeResources.AdmissionCapacity(3L, 90L));
        var preemption = new PreemptionConfig();
        preemption.setAllowedVictimStages(java.util.EnumSet.of(VictimStage.DECODE_ENGINE_OWNED));
        var earlierPlan = EvictionPlanner.planDecode(request, snapshot, preemption, new HashMap<>()).proposal();
        assertNotNull(earlierPlan);
        assertEquals(List.of(2L), earlierPlan.victims().stream().map(DecodeRequestView::requestId).toList());
        // Both attempts reserve an incoming slot. Keep a one-slot deficit so a missing
        // claimed filter would incorrectly choose the accepted victim again.
        var laterRequest = decodeRequirements(70, 0L, 0L,
                new DecodeResources.AdmissionCapacity(after.routing().totalLoad(), 90L));
        assertNull(EvictionPlanner.planDecode(laterRequest, after, preemption, new HashMap<>()).proposal(),
                "all lower-priority Engine victims belong to existing attempts");
    }

    // ==================== helpers ====================

    @Test
    void exactOwnershipQueryRetainsEngineCapacityUntilWorkerCompletion() {
        var reservation = reserve(701L, 400, 408, 30);
        DeliverySettlementTestSupport.dispatchDecode(endpoint, reservation);
        assertTrue(endpoint.hasOwnedResources(reservation));
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
        assertEquals(400, endpoint.routingView().inflightHardKv());

        updateStatus(Map.of("701", runningTask(701L, TaskPhase.RUNNING, 400)), null, 19_600);
        assertTrue(endpoint.hasOwnedResources(reservation));
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
        assertEquals(19_600, endpoint.routingView().realKvAvailable());

        updateStatus(Map.of(), Map.of("701", runningTask(701L, TaskPhase.RUNNING, 400)), 20_000);
        assertFalse(endpoint.hasOwnedResources(reservation));
        assertFalse(endpoint.hasOwnedResources(reservation));
        assertEquals(0, endpoint.routingView().engineCapacityUsed());
    }

    @Test
    void absentOldReservationIsCompleteWithoutTouchingTheReplacement() {
        var replacement = reserve(702L, 400, 408, 30);
        var old = new DecodeResources.ReservationHandle(replacement.endpointGenerationId(),
                replacement.requestId(), replacement.reservationToken() + 1000);
        assertEquals(ReservationReleaseResult.STALE, endpoint.release(old, DecodeResources.ReleaseReason.NOT_SENT));
        assertFalse(endpoint.hasOwnedResources(old));
        assertEquals(400, endpoint.routingView().inflightHardKv());
        assertEquals(ReservationReleaseResult.RELEASED, endpoint.release(replacement, DecodeResources.ReleaseReason.NOT_SENT));
        assertEquals(ReservationReleaseResult.STALE, endpoint.release(replacement, DecodeResources.ReleaseReason.NOT_SENT));
        assertFalse(endpoint.hasOwnedResources(replacement));
        assertEquals(0, endpoint.routingView().inflightHardKv());
    }

    @Test
    void rejectedVictimCannotCompletePreemptionOrReleaseItsCapacity() {
        var victim = reserve(703L, 400, 408, 30);
        updateStatus(Map.of("703", runningTask(703L, TaskPhase.RUNNING, 400)), null, 19_600);
        assertEquals(DecodeResources.PreemptionBeginResult.SUCCESS,
                beginPreemption(704L, List.of(703L), 705L, 700, 708, 70));
        assertTrue(EndpointTestSupport.handoffPreemption(endpoint, 704L));
        var before = endpoint.routingView();

        assertTrue(endpoint.hasOwnedResources(victim));

        assertTrue(confirmedView(703L).claimedForPreemption());
        assertEquals(before.engineCapacityUsed(), endpoint.routingView().engineCapacityUsed());
        assertEquals(before.inflightHardKv(), endpoint.routingView().inflightHardKv());
        assertNull(endpoint.commitPreemption(704L));
    }

    private DecodeResources.DecodeRequestView confirmedView(long requestId) {
        return endpoint.resourceSnapshot().requests().values().stream()
                .filter(request -> request.phase().isEngineConfirmed())
                .filter(view -> view.requestId() == requestId)
                .findFirst()
                .orElseThrow(() -> new AssertionError("request " + requestId + " not tracked"));
    }

    private void updateStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(20_000L);
        EndpointTestSupport.applyStatus(endpoint, response);
    }

    private DecodeResources.ReservationHandle reserve(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            DecodeResources.ReservationHandle reservation =
                    EndpointTestSupport.reserveUnqueuedDecode(endpoint, pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private boolean isConfirmed(long requestId) {
        return endpoint.resourceSnapshot().requests().values().stream()
                .filter(request -> request.phase().isEngineConfirmed())
                .anyMatch(view -> view.requestId() == requestId);
    }

    private DecodeResources.PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<Long> victimIds,
            long incomingRequestId,
            long hardKv,
            long expectedKv,
            int priority) {
        List<DecodeResources.ReservationHandle> victims = victimIds.stream()
                .map(reservations::get)
                .toList();
        return endpoint.beginPreemption(attemptToken, victims, incomingRequestId, hardKv, expectedKv, priority, new DecodeResources.AdmissionCapacity(
                        Math.max(1, endpoint.routingView().totalLoad()), 100));
    }

    private static TaskInfo runningTask(long requestId, TaskPhase phase, long inputLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        task.setInputLength(inputLength);
        return task;
    }
}
