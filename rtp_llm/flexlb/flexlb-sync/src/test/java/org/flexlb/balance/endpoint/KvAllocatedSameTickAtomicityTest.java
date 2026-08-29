package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * First-batch reconciliation acceptance (plan section 6, stage 1): the
 * KV_ALLOCATED handover inside {@code doCalibrate} must retire the A-road
 * reservation (layer 1 inflight + queued phase + dispatch permit + counters)
 * and install the B-road engine projection (layer 3) <b>within the same
 * admission-lock tick</b>.
 *
 * <p>An auditing thread samples the eight-layer read-only projection
 * concurrently with the calibration tick and asserts that no intermediate
 * snapshot ever shows the split states the plan forbids: the reservation
 * and the engine projection <b>coexisting</b>, or <b>both missing</b>, for
 * any reserved request that has not finished.  Exactly one of the two
 * ownership layers must hold every reserved request at all times.</p>
 */
class KvAllocatedSameTickAtomicityTest {

    private static final long REQUEST_A = 900L;
    private static final long REQUEST_B = 901L;
    private static final long REQUEST_C = 902L;

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final AtomicBoolean stopAuditor = new AtomicBoolean(false);
    private final AtomicReference<String> violation = new AtomicReference<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(
                RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint = new DecodeEndpoint(status, event -> { });
    }

    @AfterEach
    void tearDown() {
        stopAuditor.set(true);
    }

    @Test
    void kvAllocatedHandoverIsAtomicWithinOneTick() throws Exception {
        reserveQueued(REQUEST_A, 500, 700);
        reserveQueued(REQUEST_B, 300, 400);
        reserveQueued(REQUEST_C, 200, 250);

        Thread auditor = startOwnershipAuditor();
        try {
            TaskInfo a = task(REQUEST_A);
            a.setPhase(TaskPhase.KV_ALLOCATED);
            TaskInfo b = task(REQUEST_B);
            b.setPhase(TaskPhase.RUNNING);
            updateStatus(Map.of("900", a, "901", b), null, 10_000);

            // A small soak window: the auditor keeps sampling while the
            // tick settles, proving the invariant holds across many
            // interleavings of the calibration and the audit reads.
            long deadline = System.nanoTime() + 50_000_000L;
            while (System.nanoTime() < deadline) {
                assertNull(violation.get(), () -> "split state observed: "
                        + violation.get());
            }
        } finally {
            stopAuditor.set(true);
            auditor.join(5_000L);
        }

        assertNull(violation.get(), () -> "split state observed: "
                + violation.get());

        DecodeEndpoint.DecodeLedgerAuditView settled = endpoint.ledgerAuditView();
        // Requests A and B crossed the critical point: exactly the engine
        // projection layer holds them.
        for (long id : List.of(REQUEST_A, REQUEST_B)) {
            assertFalse(settled.inflight().containsKey(id),
                    "layer 1 must retire request " + id + " in the same tick");
            assertTrue(settled.confirmedReservationTokens().containsKey(id),
                    "layer 3 must confirm request " + id + " in the same tick");
            assertFalse(settled.queuedPhaseRequestIds().contains(id),
                    "queued phase must leave with the reservation for " + id);
            assertFalse(settled.engineDispatchPermitRequestIds().contains(id),
                    "dispatch permit must be retired for " + id);
        }
        // Request C never reported: its reservation stays in layer 1 only.
        assertTrue(settled.inflight().containsKey(REQUEST_C));
        assertFalse(settled.confirmedReservationTokens()
                .containsKey(REQUEST_C));
        // Stage-1 fix E5: the kv counters (the aggregate mirror of layer 1)
        // decrement by exactly the retired reservations in the same tick —
        // A(500/700) and B(300/400) leave, C(200/250) stays.
        assertEquals(200L, settled.inflightKvReservedTotal(),
                "kv counter must drop with the same-tick retirement");
        assertEquals(250L, settled.inflightExpectedKvReservedTotal(),
                "expected-kv counter must drop with the same-tick");
    }

    @Test
    void queuedPhaseAndPermitRetireWithTheSameTick() {
        reserveQueued(REQUEST_A, 500, 700);
        acquirePermit(REQUEST_A);

        DecodeEndpoint.DecodeLedgerAuditView before = endpoint.ledgerAuditView();
        assertTrue(before.queuedPhaseRequestIds().contains(REQUEST_A));
        assertTrue(before.engineDispatchPermitRequestIds()
                .contains(REQUEST_A));
        // Stage-1 fix E5: the counters reflect the live layer-1 reservation
        // before the tick.
        assertEquals(500L, before.inflightKvReservedTotal());
        assertEquals(700L, before.inflightExpectedKvReservedTotal());

        TaskInfo running = task(REQUEST_A);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("900", running), null, 10_000);

        DecodeEndpoint.DecodeLedgerAuditView after = endpoint.ledgerAuditView();
        assertFalse(after.queuedPhaseRequestIds().contains(REQUEST_A));
        assertFalse(after.engineDispatchPermitRequestIds()
                .contains(REQUEST_A));
        assertTrue(after.confirmedReservationTokens()
                .containsKey(REQUEST_A));
        assertFalse(after.inflight().containsKey(REQUEST_A));
        // And the counters fall to zero together with the retirement —
        // no leaked aggregate after the layer-1 exit.
        assertEquals(0L, after.inflightKvReservedTotal());
        assertEquals(0L, after.inflightExpectedKvReservedTotal());
    }

    /**
     * Stage-2 L7 soak fix (round 2): the aggregate mirror rule reads the
     * capture-frozen queued projection, so queued retirements that land
     * after the capture must not change what an already-captured view
     * reports.  A live re-read of the mutable entry sub-state flag would
     * flip the projection against the frozen Phase-1 counters and
     * fabricate a "certified" aggregate tear on every post-capture
     * admission batch.
     */
    @Test
    void ledgerCaptureFreezesQueuedProjectionAgainstPostCaptureFlips() {
        reserveQueued(REQUEST_A, 500, 700);
        reserveQueued(REQUEST_B, 300, 400);

        DecodeEndpoint.DecodeLedgerAuditView captured =
                endpoint.ledgerAuditView();
        assertTrue(captured.certified(),
                "a quiet-window capture must certify");
        assertEquals(2, captured.queuedPhaseRequestIds().size());
        assertEquals(2, captured.queuedPhaseCount());
        assertEquals(800L, captured.queuedKvReservedTotal());
        assertEquals(1100L, captured.queuedExpectedKvReservedTotal());

        // Post-capture flip: both queued reservations retire in one tick.
        TaskInfo a = task(REQUEST_A);
        a.setPhase(TaskPhase.KV_ALLOCATED);
        TaskInfo b = task(REQUEST_B);
        b.setPhase(TaskPhase.RUNNING);
        updateStatus(Map.of("900", a, "901", b), null, 10_000);

        // The already-captured view keeps reporting the frozen projection —
        // the harness aggregate rule relies on exactly this freeze.
        assertEquals(2, captured.queuedPhaseRequestIds().size(),
                "the captured queued projection is frozen at capture time");
        assertEquals(2, captured.queuedPhaseCount());
        assertEquals(800L, captured.queuedKvReservedTotal());
        assertEquals(1100L, captured.queuedExpectedKvReservedTotal());
        assertTrue(captured.inflight().containsKey(REQUEST_A),
                "the captured inflight layer is frozen too");
    }

    /**
     * Continuously samples the eight-layer projection and records the first
     * split state: reservation and projection coexisting, or both missing,
     * for any reserved-but-not-finished request.
     */
    private Thread startOwnershipAuditor() {
        List<Long> reserved = List.of(REQUEST_A, REQUEST_B, REQUEST_C);
        Thread auditor = new Thread(() -> {
            while (!stopAuditor.get()) {
                DecodeEndpoint.DecodeLedgerAuditView view =
                        endpoint.ledgerAuditView();
                for (long id : reserved) {
                    boolean inflight = view.inflight().containsKey(id);
                    boolean confirmed = view.confirmedReservationTokens()
                            .containsKey(id);
                    if (inflight && confirmed) {
                        violation.compareAndSet(null,
                                "reservation and engine projection coexist"
                                        + " for request " + id);
                        return;
                    }
                    if (!inflight && !confirmed) {
                        violation.compareAndSet(null,
                                "reservation and engine projection both"
                                        + " missing for request " + id);
                        return;
                    }
                }
            }
        }, "same-tick-atomicity-auditor");
        auditor.setDaemon(true);
        auditor.start();
        return auditor;
    }

    private void reserveQueued(long requestId, long hardKv, long expectedKv) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            DecodeEndpoint.ReservationHandle reservation =
                    endpoint.reservePinned(
                            pin, requestId, hardKv, expectedKv, 0);
            assertNotNull(reservation);
            assertTrue(endpoint.markQueuedExact(
                            pin, reservation)
                            != DecodeEndpoint.MarkQueuedResult.NOT_OWNED,
                    "reservation must accept the queued phase mark");
        }
    }

    private DecodeEndpoint.EngineDispatchPermit acquirePermit(long requestId) {
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                endpoint.acquireEngineDispatchPermit(requestId, 0);
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED,
                acquisition.status());
        assertNotNull(acquisition.permit());
        return acquisition.permit();
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }

    private void updateStatus(Map<String, TaskInfo> running,
                              Map<String, TaskInfo> finished,
                              long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(availableKvCacheTokens);
        EndpointTestSupport.applyStatus(endpoint, response);
    }
}
