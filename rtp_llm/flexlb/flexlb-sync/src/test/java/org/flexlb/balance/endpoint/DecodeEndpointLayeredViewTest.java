package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.eviction.DecodeEndpointSnapshot;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
    private final Map<Long, DecodeEndpoint.ReservationHandle> reservations =
            new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint = new DecodeEndpoint(
                status, EndpointTestSupport.noopEventSink());
        updateStatus(Map.of(), Map.of(), 20_000);
    }

    // ==================== calibrate: layer split + inheritance ====================

    @Test
    void calibrate_splitsConfirmedIntoAcceptedAndRunningLayers() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 500, 508, 40);

        TaskInfo accepted = runningTask(1L, TaskPhase.KV_ALLOCATED, 256);
        TaskInfo running = runningTask(2L, TaskPhase.RUNNING, 512);
        updateStatus(Map.of("1", accepted, "2", running), null, 10_000);

        assertEquals(1, endpoint.resourceSnapshot().acceptedCount());
        assertEquals(1, endpoint.resourceSnapshot().runningCount());
        assertEquals(2, endpoint.resourceSnapshot().confirmed().size());
        assertEquals(0, endpoint.getInflightCount());
        assertTrue(isConfirmed(1L));
        assertTrue(isConfirmed(2L));

        // Layered view inherits priority from the shadow entry
        // removed this round; KV is the reported inputLength estimate.
        DecodeEndpoint.DecodeRequestView acceptedView = confirmedView(1L);
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, acceptedView.phase());
        assertEquals(30, acceptedView.priority());
        assertEquals(256, acceptedView.kvTokens());
        assertFalse(acceptedView.claimedForPreemption());

        DecodeEndpoint.DecodeRequestView runningView = confirmedView(2L);
        assertEquals(DecodeTaskPhase.RUNNING, runningView.phase());
        assertEquals(40, runningView.priority());
    }

    @Test
    void calibrate_unknownConfirmedFallsBackToNoPriority() {
        // Report precedes any reserve: no shadow entry to inherit from.
        // Task40: the fallback is the no-priority sentinel (0), which keeps
        // untracked engine tasks out of every eviction candidate set.
        updateStatus(Map.of("9", runningTask(9L, TaskPhase.KV_ALLOCATED, 64)), null, 10_000);

        DecodeEndpoint.DecodeRequestView view = confirmedView(9L);
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
        RequestSchedulerReporter reporter =
                org.mockito.Mockito.mock(RequestSchedulerReporter.class);

        endpoint.reportAdmissionMetrics(reporter);

        String endpointKey = "10.0.0.1:8080";
        org.mockito.Mockito.verify(reporter)
                .reportDecodeReservedCount(endpointKey, 1);
        org.mockito.Mockito.verify(reporter)
                .reportDecodeShadowKvReserved(endpointKey, 300L);
        org.mockito.Mockito.verify(reporter)
                .reportDecodeAcceptedCount(endpointKey, 1);
        org.mockito.Mockito.verify(reporter)
                .reportDecodeRunningCount(endpointKey, 1);
        org.mockito.Mockito.verify(reporter)
                .reportDecodeEngineLoad(endpointKey, 3);
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
        assertEquals(0, endpoint.resourceSnapshot().confirmed().size());
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
        assertEquals(0, endpoint.resourceSnapshot().confirmed().size(),
                "tracked TTL removal must release the published confirmed slot");
        assertEquals(0, endpoint.routingView().totalLoad());
        assertTrue(endpoint.routingView().admissionVersion() > versionBefore);
    }

    @Test
    void evictExpiredRequests_boundsPriorityCanceledTerminalRecords() throws InterruptedException {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 128, 136, 70));
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(endpoint.updatePreemption(
                101L,
                DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.CANCEL_REQUESTED)));
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.canceled(reservations.get(1L))));
        assertTrue(endpoint.finishPreemption(101L, DecodeEndpoint.PreemptionDecision.COMMIT));

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
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(102L, List.of(1L),
                        9L, 128, 136, 70));
        assertTrue(endpoint.updatePreemption(102L, DecodeEndpoint.PreemptionUpdate.cancelSending()));

        assertTrue(endpoint.updatePreemption(102L, DecodeEndpoint.PreemptionUpdate.fenced(reservations.get(1L))));
        assertTrue(endpoint.finishPreemption(102L, DecodeEndpoint.PreemptionDecision.COMMIT));

        assertFalse(isConfirmed(1L));
        assertTrue(endpoint.resourceSnapshot().reserved().containsKey(9L));
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
        assertEquals(10_000 - 300, endpoint.realKvAvailable());
        assertEquals(300, endpoint.routingView().inflightHardKv());
        // totalLoad = confirmed Engine-owned count + reserved inflight count.
        assertEquals(2, endpoint.routingView().totalLoad());
        assertEquals(1, endpoint.getInflightCount());
    }

    // ==================== token-fenced weak-ACK preemption ====================

    @Test
    void priorityCancelResponseTransitionPreservesProtocolOrder() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(
                1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));

        assertThrows(IllegalArgumentException.class,
                () -> endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.CLAIMED)));
        assertThrows(IllegalArgumentException.class,
                () -> endpoint.updatePreemption(
                        101L,
                        DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.CANCEL_IN_FLIGHT)));

        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(endpoint.updatePreemption(
                101L,
                DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.CANCEL_REQUESTED)));
        assertFalse(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.NOT_FOUND_STALE)),
                "NOT_FOUND only applies to the in-flight RPC boundary");
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.CANCEL_UNKNOWN)),
                "a lost terminal after ACCEPTED remains transport-unknown");
    }

    @Test
    void beginPriorityPreemption_claimsVictimAndProvisionallyReservesIncoming() {
        reserve(2L, 400, 408, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();

        DecodeEndpoint.PreemptionBeginResult result = beginPreemption(
                101L, List.of(2L), 9L, 700, 708, 70);

        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS, result);
        // Weak ACK boundary: victim accounting is untouched and the incoming
        // reservation is provisional until typed Prefill CANCELED settles it.
        assertTrue(isConfirmed(2L));
        assertTrue(confirmedView(2L).claimedForPreemption());
        assertTrue(endpoint.resourceSnapshot().reserved().containsKey(9L));
        assertEquals(700, endpoint.routingView().inflightHardKv());
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(endpoint.updatePreemption(
                101L,
                DecodeEndpoint.PreemptionUpdate.cancelReply(2L, PreemptionCancelPhase.CANCEL_REQUESTED)));
        assertTrue(isConfirmed(2L));
        assertTrue(endpoint.routingView().admissionVersion() > version);
    }

    @Test
    void cancelAckAndUnknownKeepVictimCapacityUntilItsOwnInactivityExpiry() {
        var victim = reserve(2L, 400L, 408L, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(2L), 9L, 700L, 708L, 70));
        var incoming = endpoint.reservationHandle(9L);
        assertTrue(incoming != null);
        var before = endpoint.resourceSnapshot();
        assertEquals(1, before.runningCount());
        assertEquals(700L, before.routing().inflightHardKv());
        assertEquals(708L, before.routing().inflightExpectedKv());
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(endpoint.updatePreemption(
                101L,
                DecodeEndpoint.PreemptionUpdate.cancelReply(2L, PreemptionCancelPhase.CANCEL_REQUESTED)));
        assertEquals(before.engineCapacityUsed(), endpoint.resourceSnapshot().engineCapacityUsed());
        assertEquals(before.routing().inflightExpectedKv(), endpoint.routingView().inflightExpectedKv());
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelReply(2L, PreemptionCancelPhase.CANCEL_UNKNOWN)));
        assertEquals(1, endpoint.resourceSnapshot().runningCount());

        // Expiring the incoming request must not release a victim whose Cancel outcome is unknown.
        assertTrue(endpoint.release(incoming, DecodeEndpoint.ReleaseReason.EXPIRED).released());
        assertEquals(0L, endpoint.routingView().inflightHardKv());
        assertEquals(0L, endpoint.routingView().inflightExpectedKv());
        assertEquals(1, endpoint.resourceSnapshot().runningCount());
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
        assertTrue(endpoint.release(victim, DecodeEndpoint.ReleaseReason.EXPIRED).released());
        assertFalse(endpoint.release(victim, DecodeEndpoint.ReleaseReason.EXPIRED).released());
        var after = endpoint.resourceSnapshot();
        assertEquals(0, after.runningCount());
        assertEquals(0, after.acceptedCount());
        assertEquals(0, after.activeDispatchPermits());
        assertEquals(0, after.engineCapacityUsed());
        assertTrue(after.reserved().isEmpty());
        assertTrue(after.confirmed().isEmpty());
    }

    @Test
    void beginPriorityPreemption_exactIdentityMismatch_appliesNothing() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        DecodeEndpoint.ReservationHandle exact = reservations.get(1L);
        DecodeEndpoint.ReservationHandle stale =
                new DecodeEndpoint.ReservationHandle(
                        exact.endpointGenerationId(),
                        exact.requestId(),
                        exact.reservationToken() + 1L);
        DecodeEndpoint.PreemptionBeginResult result =
                endpoint.beginPreemption(101L, List.of(stale), 9L, 700, 708, 70, new DecodeEndpoint.AdmissionCapacity(1, 100));

        assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE, result);
        assertFalse(confirmedView(1L).claimedForPreemption());
        assertFalse(endpoint.resourceSnapshot().reserved().containsKey(9L));
    }

    @Test
    void beginPriorityPreemption_victimGone_isAllOrNothing() {
        reserve(2L, 400, 408, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();

        assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE,
                beginPreemption(101L, List.of(2L, 999L),
                        9L, 700, 708, 70));
        assertFalse(confirmedView(2L).claimedForPreemption());
        assertFalse(endpoint.resourceSnapshot().reserved().containsKey(9L));
        assertEquals(version, endpoint.routingView().admissionVersion());
    }

    @Test
    void beginPriorityPreemption_acceptsRunningAndRejectsAlreadyClaimedVictims() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 400, 408, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256),
                "2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);

        // RUNNING is engine-owned too and follows the same cancel path.
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));

        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(102L, List.of(2L),
                        10L, 700, 708, 70));
        assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_ALREADY_CLAIMED,
                beginPreemption(103L, List.of(2L),
                        11L, 700, 708, 70));
    }

    @Test
    void ttlEvictionCannotReleaseClaimedEngineVisibleShadow() throws Exception {
        reserve(1L, 500, 508, 30);
        // Keep a wide age gap so the provisional incoming reservation cannot
        // become TTL-eligible merely because this test runs on a loaded JVM.
        Thread.sleep(150);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelSending()));

        assertEquals(0, endpoint.evictExpiredRequests(
                100, requestId -> false));
        assertTrue(endpoint.resourceSnapshot().reserved().containsKey(1L),
                "generic TTL cleanup must not deduct a claimed victim");
        assertEquals(1_200, endpoint.routingView().inflightHardKv(),
                "victim and provisional incoming remain fully charged");

        endpoint.finishPreemption(101L, DecodeEndpoint.PreemptionDecision.ABORT);
        assertEquals(1, endpoint.evictExpiredRequests(
                100, requestId -> false));
        assertFalse(endpoint.resourceSnapshot().reserved().containsKey(1L));
        assertEquals(0, endpoint.routingView().inflightHardKv());
    }

    @Test
    void activeAfterNotFoundReleasesSyntheticHeldKvWithClaim() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.NOT_FOUND_STALE)));
        endpoint.finishPreemption(101L, DecodeEndpoint.PreemptionDecision.ABORT);

        // Decode disappears while NOT_FOUND is being reconciled. Its KV is
        // conservatively held until the original Prefill reports it active.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(endpoint.updatePreemption(101L, DecodeEndpoint.PreemptionUpdate.active(reservations.get(1L))));
        assertEquals(10_000, endpoint.realKvAvailable(),
                "active reconciliation must release held KV before dropping the claim");
        assertFalse(confirmedView(1L).claimedForPreemption());
    }

    @Test
    void notFoundRetainsSyntheticKvUntilExactLeaseExpiration() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(104L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(endpoint.updatePreemption(104L, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(endpoint.updatePreemption(104L, DecodeEndpoint.PreemptionUpdate.cancelReply(1L, PreemptionCancelPhase.NOT_FOUND_STALE)));

        // Decode disappearance moves the victim's 500-token charge into a
        // synthetic hold. The 700-token provisional incoming reservation is
        // still independently owned by the live preemption attempt.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(700, endpoint.routingView().inflightHardKv());
        assertEquals(2, endpoint.routingView().totalLoad(),
                "victim and provisional incoming must both remain charged before abort");
        assertEquals(8_800, endpoint.realKvAvailable());
        endpoint.finishPreemption(104L, DecodeEndpoint.PreemptionDecision.ABORT);

        assertEquals(0, endpoint.routingView().inflightHardKv(),
                "aborting the attempt releases only its provisional incoming reservation");
        assertEquals(9_500, endpoint.realKvAvailable(),
                "aborting incoming work must not release the victim KV hold");
        assertEquals(1, endpoint.routingView().totalLoad(),
                "the disappeared confirmed victim remains a synthetic slot");

        assertTrue(endpoint.release(reservations.get(1L), DecodeEndpoint.ReleaseReason.EXPIRED).released());
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertFalse(endpoint.release(reservations.get(1L), DecodeEndpoint.ReleaseReason.EXPIRED).released(),
                "the exact local lease expires at most once");
        assertEquals(10_000, endpoint.realKvAvailable());
    }

    // ==================== snapshot capture: layered lists ====================

    @Test
    void snapshotCapture_splitsLayersAndExcludesCancelRequested() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 400, 408, 30);
        reserve(3L, 300, 308, 30);
        updateStatus(Map.of("2", runningTask(2L, TaskPhase.KV_ALLOCATED, 256),
                "3", runningTask(3L, TaskPhase.RUNNING, 512)), null, 10_000);

        DecodeEndpointSnapshot snapshot = DecodeEndpointSnapshot.capture(endpoint, new DecodeEndpoint.AdmissionCapacity(4L, 90L));
        assertEquals(List.of(1L), ids(snapshot.reserved()));
        assertEquals(List.of(2L), ids(snapshot.accepted()));
        assertEquals(List.of(3L), ids(snapshot.running()));
        DecodeRequestView accepted = snapshot.accepted().get(0);
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, accepted.phase());
        assertEquals(256, accepted.kvTokens());

        // A cancel-requested entry is claimed by an in-flight eviction and
        // must not be offered to planning again.
        beginPreemption(101L, List.of(2L),
                20L, 64, 72, 70);
        beginPreemption(102L, List.of(3L),
                30L, 64, 72, 70);
        DecodeEndpointSnapshot after = DecodeEndpointSnapshot.capture(endpoint, new DecodeEndpoint.AdmissionCapacity(4L, 90L));
        assertTrue(after.accepted().isEmpty());
        assertTrue(after.running().isEmpty());
    }

    // ==================== helpers ====================

    @Test
    void failureSettlementAnswersExactObligationWithoutReleasingEngineCapacity() {
        var reservation = reserve(701L, 400, 408, 30);
        DeliverySettlementTestSupport.dispatchDecode(endpoint, reservation);
        assertFalse(endpoint.settleFailedRequest(reservation, DeliveryResult.Status.PREFILL_REJECTED));
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
        assertEquals(400, endpoint.routingView().inflightHardKv());

        updateStatus(Map.of("701", runningTask(701L, TaskPhase.RUNNING, 400)), null, 19_600);
        assertFalse(endpoint.settleFailedRequest(reservation, DeliveryResult.Status.PREFILL_REJECTED));
        assertEquals(1, endpoint.routingView().engineCapacityUsed());
        assertEquals(19_600, endpoint.routingView().realKvAvailable());

        updateStatus(Map.of(), Map.of("701", runningTask(701L, TaskPhase.RUNNING, 400)), 20_000);
        assertTrue(endpoint.settleFailedRequest(reservation, DeliveryResult.Status.PREFILL_REJECTED));
        assertTrue(endpoint.settleFailedRequest(reservation, DeliveryResult.Status.PREFILL_REJECTED));
        assertEquals(0, endpoint.routingView().engineCapacityUsed());
    }

    @Test
    void absentOldReservationIsCompleteWithoutTouchingTheReplacement() {
        var replacement = reserve(702L, 400, 408, 30);
        var old = new DecodeEndpoint.ReservationHandle(replacement.endpointGenerationId(),
                replacement.requestId(), replacement.reservationToken() + 1000);
        assertTrue(endpoint.settleFailedRequest(old, DeliveryResult.Status.NOT_SENT));
        assertTrue(endpoint.settleFailedRequest(old, DeliveryResult.Status.PREFILL_REJECTED));
        assertEquals(400, endpoint.routingView().inflightHardKv());
        assertTrue(endpoint.settleFailedRequest(replacement, DeliveryResult.Status.NOT_SENT));
        assertTrue(endpoint.settleFailedRequest(replacement, DeliveryResult.Status.NOT_SENT));
        assertEquals(0, endpoint.routingView().inflightHardKv());
    }

    @Test
    void rejectedVictimCannotCompletePreemptionOrReleaseItsCapacity() {
        var victim = reserve(703L, 400, 408, 30);
        updateStatus(Map.of("703", runningTask(703L, TaskPhase.RUNNING, 400)), null, 19_600);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(704L, List.of(703L), 705L, 700, 708, 70));
        assertTrue(endpoint.updatePreemption(704L, DecodeEndpoint.PreemptionUpdate.cancelSending()));
        assertTrue(endpoint.updatePreemption(
                704L,
                DecodeEndpoint.PreemptionUpdate.cancelReply(703L, PreemptionCancelPhase.CANCEL_UNKNOWN)));
        var before = endpoint.routingView();

        assertFalse(endpoint.settleFailedRequest(victim, DeliveryResult.Status.PREFILL_REJECTED));

        assertTrue(confirmedView(703L).claimedForPreemption());
        assertEquals(before.engineCapacityUsed(), endpoint.routingView().engineCapacityUsed());
        assertEquals(before.inflightHardKv(), endpoint.routingView().inflightHardKv());
        assertFalse(endpoint.finishPreemption(704L, DecodeEndpoint.PreemptionDecision.COMMIT));
    }

    private static List<Long> ids(List<DecodeRequestView> entries) {
        return entries.stream().map(DecodeRequestView::requestId).toList();
    }

    private DecodeEndpoint.DecodeRequestView confirmedView(long requestId) {
        return endpoint.resourceSnapshot().confirmed().stream()
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

    private DecodeEndpoint.ReservationHandle reserve(
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertTrue(pin != null);
            DecodeEndpoint.ReservationHandle reservation =
                    endpoint.reserveUnqueued(pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private boolean isConfirmed(long requestId) {
        return endpoint.resourceSnapshot().confirmed().stream()
                .anyMatch(view -> view.requestId() == requestId);
    }

    private DecodeEndpoint.PreemptionBeginResult beginPreemption(
            long attemptToken,
            List<Long> victimIds,
            long incomingRequestId,
            long hardKv,
            long expectedKv,
            int priority) {
        List<DecodeEndpoint.ReservationHandle> victims = victimIds.stream()
                .map(reservations::get)
                .toList();
        return endpoint.beginPreemption(attemptToken, victims, incomingRequestId, hardKv, expectedKv, priority, new DecodeEndpoint.AdmissionCapacity(
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
