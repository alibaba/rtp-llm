package org.flexlb.balance.endpoint;

import org.flexlb.balance.eviction.DecodeEndpointSnapshot;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
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

        assertEquals(1, endpoint.layeredAdmissionView().acceptedCount());
        assertEquals(1, endpoint.layeredAdmissionView().runningCount());
        assertEquals(2, endpoint.layeredAdmissionView().confirmed().size());
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
        assertEquals(1, endpoint.layeredAdmissionView().acceptedCount());

        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);

        assertEquals(0, endpoint.layeredAdmissionView().acceptedCount());
        assertEquals(1, endpoint.layeredAdmissionView().runningCount());
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
        assertEquals(0, endpoint.layeredAdmissionView().acceptedCount());
        assertEquals(0, endpoint.layeredAdmissionView().confirmed().size());
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
        assertEquals(0, endpoint.layeredAdmissionView().confirmed().size(),
                "tracked TTL removal must release the published confirmed slot");
        assertEquals(0, endpoint.routingView().totalLoad());
        assertTrue(endpoint.routingView().admissionVersion() > versionBefore);
    }

    @Test
    void evictExpiredRequests_boundsPriorityCanceledTombstones() throws InterruptedException {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 128, 136, 70));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.recordPriorityCancelPhase(
                101L, 1L, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertTrue(endpoint.settlePriorityCanceled(
                101L, reservations.get(1L)));
        assertTrue(endpoint.commitPriorityPreemption(101L));

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
    void priorityTombstoneIsAuthoritativeWithoutAcceptedOrWorkerCanceled() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        long version = endpoint.routingView().admissionVersion();
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(102L, List.of(1L),
                        9L, 128, 136, 70));
        assertTrue(endpoint.markPriorityCancelInFlight(102L));

        assertTrue(endpoint.settlePriorityTombstoned(
                102L, reservations.get(1L)));
        assertTrue(endpoint.commitPriorityPreemption(102L));

        assertFalse(isConfirmed(1L));
        assertTrue(endpoint.layeredAdmissionView().reserved().containsKey(9L));
        assertEquals(1, endpoint.routingView().totalLoad());
        // The same late Decode sample rejected by typed-CANCELED fencing must
        // also be rejected after the stronger absent+tombstone proof.
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
                () -> endpoint.recordPriorityCancelPhase(
                        101L, 1L, PreemptionCancelPhase.CLAIMED));
        assertThrows(IllegalArgumentException.class,
                () -> endpoint.recordPriorityCancelPhase(
                        101L, 1L, PreemptionCancelPhase.CANCEL_IN_FLIGHT));

        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.recordPriorityCancelPhase(
                101L, 1L, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(endpoint.recordPriorityCancelPhase(
                101L, 1L, PreemptionCancelPhase.NOT_FOUND_STALE),
                "NOT_FOUND only applies to the in-flight RPC boundary");
        assertTrue(endpoint.recordPriorityCancelPhase(
                101L, 1L, PreemptionCancelPhase.CANCEL_UNKNOWN),
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
        assertTrue(endpoint.layeredAdmissionView().reserved().containsKey(9L));
        assertEquals(700, endpoint.routingView().inflightHardKv());
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.recordPriorityCancelPhase(
                101L, 2L, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertTrue(isConfirmed(2L));
        assertTrue(endpoint.routingView().admissionVersion() > version);
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
                endpoint.beginPriorityPreemption(
                        101L,
                        List.of(stale),
                        9L,
                        700,
                        708,
                        70,
                        new DecodeEndpoint.AdmissionCapacity(1, 100));

        assertEquals(DecodeEndpoint.PreemptionBeginResult.VICTIM_GONE, result);
        assertFalse(confirmedView(1L).claimedForPreemption());
        assertFalse(endpoint.layeredAdmissionView().reserved().containsKey(9L));
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
        assertFalse(endpoint.layeredAdmissionView().reserved().containsKey(9L));
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
        assertTrue(endpoint.markPriorityCancelInFlight(101L));

        assertEquals(0, endpoint.evictExpiredRequests(
                100, requestId -> false));
        assertTrue(endpoint.layeredAdmissionView().reserved().containsKey(1L),
                "generic TTL cleanup must not deduct a claimed victim");
        assertEquals(1_200, endpoint.routingView().inflightHardKv(),
                "victim and provisional incoming remain fully charged");

        endpoint.abortPriorityPreemption(101L);
        assertEquals(1, endpoint.evictExpiredRequests(
                100, requestId -> false));
        assertFalse(endpoint.layeredAdmissionView().reserved().containsKey(1L));
        assertEquals(0, endpoint.routingView().inflightHardKv());
    }

    @Test
    void activeAfterNotFoundReleasesSyntheticHeldKvWithClaim() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(101L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(endpoint.markPriorityCancelInFlight(101L));
        assertTrue(endpoint.recordPriorityCancelPhase(
                101L, 1L, PreemptionCancelPhase.NOT_FOUND_STALE));
        endpoint.abortPriorityPreemption(101L);

        // Decode disappears while NOT_FOUND is being reconciled. Its KV is
        // conservatively held until the original Prefill reports it active.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(9_500, endpoint.realKvAvailable());

        assertTrue(endpoint.reconcilePriorityVictimActive(
                101L, reservations.get(1L)));
        assertEquals(10_000, endpoint.realKvAvailable(),
                "active reconciliation must release held KV before dropping the claim");
        assertFalse(confirmedView(1L).claimedForPreemption());
    }

    @Test
    void notFoundTransferRetainsSyntheticKvUntilExactEngineFenceSettlement() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 500)), null, 10_000);
        assertEquals(DecodeEndpoint.PreemptionBeginResult.SUCCESS,
                beginPreemption(104L, List.of(1L),
                        9L, 700, 708, 70));
        assertTrue(endpoint.markPriorityCancelInFlight(104L));
        assertTrue(endpoint.recordPriorityCancelPhase(
                104L, 1L, PreemptionCancelPhase.NOT_FOUND_STALE));

        // Decode disappearance moves the victim's 500-token charge into a
        // synthetic hold. The 700-token provisional incoming reservation is
        // still independently owned by the live preemption attempt.
        updateStatus(Map.of(), null, 10_000);
        assertEquals(700, endpoint.routingView().inflightHardKv());
        assertEquals(2, endpoint.routingView().totalLoad(),
                "victim and provisional incoming must both remain charged before abort");
        assertEquals(8_800, endpoint.realKvAvailable());
        assertTrue(endpoint.transferPriorityNotFoundClaimToEngineFence(104L, 1L));
        assertFalse(endpoint.reconcilePriorityVictimActive(
                104L, reservations.get(1L)),
                "a transferred fence cannot return to ordinary active reconciliation");
        assertFalse(endpoint.reconcilePriorityVictimFinished(
                104L, reservations.get(1L)),
                "a transferred fence requires its exact fence settlement");
        assertFalse(endpoint.settlePriorityTombstoned(
                104L, reservations.get(1L)),
                "the original attempt cannot settle a transferred fence");
        endpoint.abortPriorityPreemption(104L);

        assertEquals(0, endpoint.routingView().inflightHardKv(),
                "aborting the attempt releases only its provisional incoming reservation");
        assertEquals(9_500, endpoint.realKvAvailable(),
                "control-owner transfer must not release the synthetic KV hold");
        assertEquals(1, endpoint.routingView().totalLoad(),
                "the disappeared confirmed victim remains a synthetic slot");

        assertTrue(endpoint.settleEngineFenceClaim(
                104L, reservations.get(1L)));
        assertEquals(10_000, endpoint.realKvAvailable());
        assertEquals(0, endpoint.routingView().totalLoad());
        assertFalse(endpoint.settleEngineFenceClaim(
                104L, reservations.get(1L)),
                "the exact fence generation settles accounting at most once");
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

        DecodeEndpointSnapshot snapshot = DecodeEndpointSnapshot.capture(endpoint, 4);
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
        DecodeEndpointSnapshot after = DecodeEndpointSnapshot.capture(endpoint, 4);
        assertTrue(after.accepted().isEmpty());
        assertTrue(after.running().isEmpty());
    }

    // ==================== helpers ====================

    private static List<Long> ids(List<DecodeRequestView> entries) {
        return entries.stream().map(DecodeRequestView::requestId).toList();
    }

    private DecodeEndpoint.DecodeRequestView confirmedView(long requestId) {
        return endpoint.layeredAdmissionView().confirmed().stream()
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
                    endpoint.reservePinned(
                            pin, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private boolean isConfirmed(long requestId) {
        return endpoint.layeredAdmissionView().confirmed().stream()
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
        return endpoint.beginPriorityPreemption(
                attemptToken,
                victims,
                incomingRequestId,
                hardKv,
                expectedKv,
                priority,
                new DecodeEndpoint.AdmissionCapacity(
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
