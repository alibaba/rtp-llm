package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeLedgerAuditView;
import org.flexlb.balance.endpoint.DecodeEndpoint.ReservationHandle;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Rule-level tests for the three-way ledger reconciliation harness
 * (plan section 6, stage 1 / M1).  DecodeEndpoint views are stubbed so each
 * structural split can be produced deterministically; the PrefillWorkRegistry
 * side is real (its constructor is package-private and cheap).
 */
class LedgerReconciliationHarnessTest {

    private static final long REQUEST_ID = 501L;

    private FlexlbConfig config;
    private RequestLifecycleCoordinator lifecycle;
    private ReentrantLock registryLock;
    private PrefillWorkRegistry registry;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        lifecycle = new RequestLifecycleCoordinator(
                configService,
                mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(EngineCancelChannel.class));
        registryLock = new ReentrantLock();
        registry = new PrefillWorkRegistry(
                registryLock,
                new PriorityBlockingQueue<>(64,
                        java.util.Comparator
                                .comparingLong(BatchItem::requestId)),
                () -> { });
    }

    @AfterEach
    void tearDown() {
        if (lifecycle.closeAdmissionAndAwaitMutations()) {
            lifecycle.closeOutstandingAndTerminalize();
            lifecycle.closeExpiration();
            lifecycle.closePublisher();
        }
    }

    @Test
    void cleanPublicationProducesNoDiffs() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));
        enqueueActive(registered.item());

        stubDecodeView(decode, auditView(
                cleanARoadway(reservation), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        LedgerReconciliationHarness.ReconciliationReport report =
                harness.reconcileOnce();

        assertTrue(report.realDiffs().isEmpty(),
                () -> "real diffs: " + report.realDiffs());
        assertTrue(report.transientDiffs().isEmpty(),
                () -> "transient diffs: " + report.transientDiffs());
        assertEquals(1, report.slotCount());
        harness.close();
    }

    @Test
    void duplicateDecodeOwnershipIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        Map<Long, RequestInflight> inflight = new HashMap<>();
        inflight.put(REQUEST_ID,
                inflightEntry(16L, 24L, 3, 77L));
        Map<Long, Long> confirmed = new HashMap<>();
        confirmed.put(REQUEST_ID, 77L);
        stubDecodeView(decode, auditView(inflight, confirmed, Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule.DUPLICATE_OWNERSHIP_ON_DECODE,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void unbackedSlotReservationIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        stubDecodeView(decode,
                auditView(Map.of(), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule.SLOT_RESERVATION_UNBACKED,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void reservationTokenMismatchIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        Map<Long, RequestInflight> inflight = new HashMap<>();
        inflight.put(REQUEST_ID, inflightEntry(16L, 24L, 3, 999L));
        stubDecodeView(decode,
                auditView(inflight, Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule.RESERVATION_TOKEN_MISMATCH,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void kvMirrorMismatchIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        Map<Long, RequestInflight> inflight = new HashMap<>();
        inflight.put(REQUEST_ID, inflightEntry(555L, 24L, 3, 77L));
        stubDecodeView(decode,
                auditView(inflight, Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule.KV_MIRROR_MISMATCH,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void reverseProjectionIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));
        RequestSlot slot = lifecycle.requestSlot(REQUEST_ID);
        synchronized (slot) {
            assertTrue(slot.markDecodeAccepted().acceptedBeforeCancel());
        }

        stubDecodeView(decode,
                auditView(Map.of(), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule.SLOT_PROJECTION_UNCONFIRMED,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void engineConfirmedAheadOfSlotIsTransient() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        Map<Long, Long> confirmed = new HashMap<>();
        confirmed.put(REQUEST_ID, 77L);
        stubDecodeView(decode,
                auditView(Map.of(), confirmed, Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        LedgerReconciliationHarness.ReconciliationReport report =
                harness.reconcileOnce();

        assertTrue(report.realDiffs().isEmpty(),
                () -> "real diffs: " + report.realDiffs());
        assertEquals(1, report.transientDiffs().size());
        assertEquals(
                LedgerReconciliationHarness.Rule.DECODE_CONFIRMED_AHEAD_OF_SLOT,
                report.transientDiffs().get(0).rule());
        harness.close();
    }

    @Test
    void engineSettledAheadOfSlotIsTransient() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        stubDecodeView(decode,
                auditView(Map.of(), Map.of(), Set.of(REQUEST_ID)));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        LedgerReconciliationHarness.ReconciliationReport report =
                harness.reconcileOnce();

        assertTrue(report.realDiffs().isEmpty(),
                () -> "real diffs: " + report.realDiffs());
        assertEquals(1, report.transientDiffs().size());
        assertEquals(
                LedgerReconciliationHarness.Rule.DECODE_SETTLED_AHEAD_OF_SLOT,
                report.transientDiffs().get(0).rule());
        harness.close();
    }

    @Test
    void prefillOrphanActiveItemIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        // An item that never registered with the lifecycle: the queue
        // holds it while the request-slot ledger has no entry at all.
        BatchItem orphan = buildItem(
                context(601L, 8L, 0), 0L, null, decode);
        enqueueActive(orphan);
        // Empty decode ledger: the orphan lives only in the prefill queue.
        stubDecodeView(decode, auditView(Map.of(), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule.PREFILL_ORPHAN_ACTIVE_ITEM,
                real.get(0).rule());
        assertEquals(601L, real.get(0).requestId());
        harness.close();
    }

    @Test
    void realDiffNeedsConsecutivePassesUnderAConfirmWindow() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        stubDecodeView(decode, auditView(Map.of(), Map.of(), Set.of()));
        BatchItem orphan = buildItem(context(602L, 8L, 0), 0L, null, decode);
        enqueueActive(orphan);

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null, 2);

        // First pass: the split is only a tear candidate, not confirmed.
        LedgerReconciliationHarness.ReconciliationReport first =
                harness.reconcileOnce();
        assertTrue(first.realDiffs().isEmpty(),
                () -> "first pass must stay pending: " + first.realDiffs());
        assertEquals(1, first.pendingRealDiffs().size());

        // The structural split persists: the second consecutive pass
        // confirms it as a REAL diff.
        LedgerReconciliationHarness.ReconciliationReport second =
                harness.reconcileOnce();
        assertEquals(1, second.realDiffs().size());
        assertEquals(
                LedgerReconciliationHarness.Rule.PREFILL_ORPHAN_ACTIVE_ITEM,
                second.realDiffs().get(0).rule());
        assertTrue(second.pendingRealDiffs().isEmpty());
        harness.close();
    }

    @Test
    void confirmWindowResetsWhenTheCandidateDisappears() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        stubDecodeView(decode, auditView(Map.of(), Map.of(), Set.of()));
        BatchItem orphan = buildItem(context(603L, 8L, 0), 0L, null, decode);
        enqueueActive(orphan);

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null, 2);
        assertTrue(harness.reconcileOnce().realDiffs().isEmpty());

        // The tear converges: the queue withdraws the item and the next
        // pass must be clean with neither confirmed nor pending diffs.
        registryLock.lock();
        try {
            assertTrue(registry.terminalizeActiveUnderLock(orphan));
        } finally {
            registryLock.unlock();
        }
        LedgerReconciliationHarness.ReconciliationReport converged =
                harness.reconcileOnce();
        assertTrue(converged.realDiffs().isEmpty(),
                () -> "converged pass: " + converged.realDiffs());
        assertTrue(converged.pendingRealDiffs().isEmpty(),
                () -> "converged pass pending: "
                        + converged.pendingRealDiffs());

        // Re-appearance still starts from zero: one pass is pending again.
        enqueueActive(orphan);
        LedgerReconciliationHarness.ReconciliationReport reappeared =
                harness.reconcileOnce();
        assertTrue(reappeared.realDiffs().isEmpty());
        assertEquals(1, reappeared.pendingRealDiffs().size());
        harness.close();
    }

    @Test
    void prefillItemIdentityMismatchIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        // A different frozen item carrying the same request id, built
        // outside the lifecycle so the existing slot stays untouched.
        BatchItem impostor = buildItem(
                context(REQUEST_ID, 16L, 3), 24L, reservation, decode);
        enqueueActive(impostor);

        Map<Long, RequestInflight> inflight = new HashMap<>();
        inflight.put(REQUEST_ID, inflightEntry(16L, 24L, 3, 77L));
        stubDecodeView(decode, auditView(inflight, Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .PREFILL_ITEM_IDENTITY_MISMATCH,
                real.get(0).rule());
        harness.close();
    }

    // ==================== stage-1 fix A: endpoint-left surface ====================

    @Test
    void endpointLeftSurfaceOnActiveSlotIsItsOwnRealDiff() {
        // The displaced (retired/replaced) endpoint object is still
        // referenced by the slot placement, but the reconciliation surface
        // only sees the successor-generation object — a legal failover
        // window that must surface under its own rule, not as an unbacked
        // reservation.
        DecodeEndpoint retired = mock(DecodeEndpoint.class);
        DecodeEndpoint successor = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, retired);
        assertTrue(bind(registered));

        stubDecodeView(successor,
                auditView(Map.of(), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(successor), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .ENDPOINT_LEFT_RECONCILIATION_SURFACE,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void endpointLeftSurfaceOnTerminalTrackSlotIsExempt() {
        DecodeEndpoint retired = mock(DecodeEndpoint.class);
        DecodeEndpoint successor = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, retired);
        assertTrue(bind(registered));
        // A terminal-track slot lives in the legal two-stage-death /
        // failover window: the departure of its endpoint must be skipped
        // entirely, and the coarse-phase projection that drives the
        // exemption is the one routed through the adjudication layer.
        // The claimed publication lease is released immediately: this
        // test only needs the TERMINALIZING phase flip, and a dangling
        // lease would block the publisher close in tearDown.
        RequestSlot slot = lifecycle.requestSlot(REQUEST_ID);
        TerminalAction terminalized;
        synchronized (slot) {
            terminalized = slot.beginExternalTerminalizing(
                    s -> s.snapshot());
        }
        assertNotNull(terminalized);
        if (terminalized.publicationLease() != null) {
            terminalized.publicationLease().close();
        }

        stubDecodeView(successor,
                auditView(Map.of(), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(successor), List.of(registry), null);
        LedgerReconciliationHarness.ReconciliationReport report =
                harness.reconcileOnce();

        assertTrue(report.realDiffs().isEmpty(),
                () -> "real diffs: " + report.realDiffs());
        assertTrue(report.pendingRealDiffs().isEmpty(),
                () -> "pending diffs: " + report.pendingRealDiffs());
        assertTrue(report.transientDiffs().isEmpty(),
                () -> "transient diffs: " + report.transientDiffs());
        assertEquals(1, report.slotCount());
        harness.close();
    }

    // ==================== stage-1 fix C: L4/L5 rules ====================

    @Test
    void preemptionRegistrationWithoutClaimIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));
        // The coordinator side of the L4 install window: the slot holds
        // the registration, the engine claim ledger does not back it.
        assertTrue(lifecycle.tryClaim(
                REQUEST_ID, 77L, 9001L, "rule-test claim").isPresent());

        stubDecodeView(decode, auditView(
                cleanARoadway(reservation), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .PREEMPTION_REGISTRATION_UNBACKED,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void engineFenceWithoutProtectionIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));
        // Install a real cancellation fence on the slot; the mocked
        // endpoint contributes no layer-5 protection leaf, so the slot
        // holds a registration the protection ledger does not back.
        RequestSlot slot = lifecycle.requestSlot(REQUEST_ID);
        synchronized (slot) {
            assertTrue(slot.requestCancellationFence("rule-test fence")
                    instanceof RequestSlot.FenceReduction.Start);
        }

        stubDecodeView(decode, auditView(
                cleanARoadway(reservation), Map.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule.ENGINE_FENCE_UNBACKED,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void decodeClaimOrphanIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        // Layer-4 holds a priority claim on a slot that carries no
        // preemption registration — no single-interleave legal window.
        stubDecodeView(decode, auditView(
                cleanARoadway(reservation), Map.of(), Set.of(),
                Set.of(REQUEST_ID), Set.of(), Set.of(), Set.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .DECODE_PREEMPTION_CLAIM_ORPHAN,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void decodeFenceProtectionOrphanIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        ReservationHandle reservation =
                new ReservationHandle(5L, REQUEST_ID, 77L);
        Registered registered = registerItem(
                REQUEST_ID, 16L, 24L, 3, reservation, decode);
        assertTrue(bind(registered));

        // Layer-5 holds a fence protection on a slot with neither fence
        // nor preemption registration.
        stubDecodeView(decode, auditView(
                cleanARoadway(reservation), Map.of(), Set.of(),
                Set.of(), Set.of(), Set.of(REQUEST_ID), Set.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .DECODE_FENCE_PROTECTION_ORPHAN,
                real.get(0).rule());
        harness.close();
    }

    @Test
    void preemptionAttemptIncomingOutsideInflightIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        // Engine-internal invariant, view-only construction: a layer-4b
        // attempt-incoming reservation with no layer-1 shadow backing.
        stubDecodeView(decode, auditView(
                Map.of(), Map.of(), Set.of(),
                Set.of(), Set.of(701L), Set.of(), Set.of(), Set.of()));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .PREEMPTION_ATTEMPT_INCOMING_UNBACKED,
                real.get(0).rule());
        assertEquals(701L, real.get(0).requestId());
        harness.close();
    }

    @Test
    void queuedAggregateMirrorDriftIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        // Stage-2 L7 retirement: the queued projection derives from the
        // layer-1 entry sub-state flags, so a queued member outside the
        // inflight set is structurally impossible. The rewritten rule
        // cross-checks the three O(1) queued counters against the
        // entry-derived projection — one master-queued entry here, but the
        // counters drifted (count=2 vs derived 1).
        Map<Long, RequestInflight> inflight = new HashMap<>();
        RequestInflight queuedEntry = inflightEntry(16L, 24L, 3, 7L);
        assertTrue(queuedEntry.enterMasterQueued());
        inflight.put(702L, queuedEntry);
        stubDecodeView(decode, auditView(
                inflight, Map.of(), Set.of(),
                Set.of(), Set.of(), Set.of(), Set.of(702L), Set.of(),
                2, 16L, 24L));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .QUEUED_PHASE_OUTSIDE_INFLIGHT,
                real.get(0).rule());
        assertEquals(0L, real.get(0).requestId(),
                "the rewritten rule reports the endpoint-level aggregate");
        harness.close();
    }

    @Test
    void tornCaptureAggregateMirrorDriftIsExemptFromTheRule() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        // Stage-2 L7 soak fix: an uncertified (torn fallback) capture read
        // its Phase-1 counters and its Phase-2 entry projection in
        // different admission quiet windows, so the aggregate drift it
        // shows is a capture tear, not a ledger split — and because the
        // aggregate DiffKey is fixed (request id 0), the confirm window
        // cannot absorb it by request-id rotation. The identical drift as
        // queuedAggregateMirrorDriftIsARealDiff, but certified=false,
        // must therefore produce no diff at all.
        Map<Long, RequestInflight> inflight = new HashMap<>();
        RequestInflight queuedEntry = inflightEntry(16L, 24L, 3, 7L);
        assertTrue(queuedEntry.enterMasterQueued());
        inflight.put(704L, queuedEntry);
        stubDecodeView(decode, auditView(
                inflight, Map.of(), Set.of(),
                Set.of(), Set.of(), Set.of(), Set.of(704L), Set.of(),
                2, 16L, 24L, false));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        LedgerReconciliationHarness.ReconciliationReport report =
                harness.reconcileOnce();

        assertTrue(report.realDiffs().isEmpty(),
                "torn captures carry no aggregate-mirror signal: "
                        + report.realDiffs());
        assertTrue(report.pendingRealDiffs().isEmpty(),
                "torn captures must not even surface as tear candidates: "
                        + report.pendingRealDiffs());
        harness.close();
    }

    @Test
    void dispatchPermitOutsideQueuedPhaseIsARealDiff() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        stubDecodeView(decode, auditView(
                Map.of(), Map.of(), Set.of(),
                Set.of(), Set.of(), Set.of(), Set.of(), Set.of(703L)));

        LedgerReconciliationHarness harness = new LedgerReconciliationHarness(
                lifecycle, List.of(decode), List.of(registry), null);
        List<LedgerReconciliationHarness.LedgerDiff> real =
                harness.reconcileOnce().realDiffs();

        assertEquals(1, real.size());
        assertEquals(
                LedgerReconciliationHarness.Rule
                        .DISPATCH_PERMIT_OUTSIDE_QUEUED_PHASE,
                real.get(0).rule());
        assertEquals(703L, real.get(0).requestId());
        harness.close();
    }

    // ==================== fixtures ====================

    private Map<Long, RequestInflight> cleanARoadway(
            ReservationHandle reservation) {
        Map<Long, RequestInflight> inflight = new HashMap<>();
        inflight.put(REQUEST_ID,
                inflightEntry(16L, 24L, 3, reservation.reservationToken()));
        return inflight;
    }

    private static RequestInflight inflightEntry(
            long kvTokens, long expectedKvTokens, int priority, long token) {
        return new RequestInflight(
                kvTokens,
                expectedKvTokens,
                System.currentTimeMillis(),
                priority,
                DecodeTaskPhase.MASTER_QUEUED_NOT_DISPATCHED,
                token);
    }

    private static DecodeLedgerAuditView auditView(
            Map<Long, RequestInflight> inflight,
            Map<Long, Long> confirmed,
            Set<Long> settledTombstones) {
        return auditView(inflight, confirmed, settledTombstones,
                Set.of(), Set.of(), Set.of(), Set.of(), Set.of());
    }

    /** Full-layer view constructor (stage-1 fix C rule surface). */
    private static DecodeLedgerAuditView auditView(
            Map<Long, RequestInflight> inflight,
            Map<Long, Long> confirmed,
            Set<Long> settledTombstones,
            Set<Long> preemptionClaims,
            Set<Long> preemptionAttemptIncoming,
            Set<Long> fenceProtected,
            Set<Long> queuedPhase,
            Set<Long> dispatchPermits) {
        // Stage-2 L7: aggregate counters default to the empty projection;
        // queued-state tests pass the drift values explicitly.
        return auditView(inflight, confirmed, settledTombstones,
                preemptionClaims, preemptionAttemptIncoming, fenceProtected,
                queuedPhase, dispatchPermits, 0, 0L, 0L);
    }

    /** Stage-2 L7 full-layer view constructor with queued aggregates. */
    private static DecodeLedgerAuditView auditView(
            Map<Long, RequestInflight> inflight,
            Map<Long, Long> confirmed,
            Set<Long> settledTombstones,
            Set<Long> preemptionClaims,
            Set<Long> preemptionAttemptIncoming,
            Set<Long> fenceProtected,
            Set<Long> queuedPhase,
            Set<Long> dispatchPermits,
            int queuedPhaseCount,
            long queuedKvReservedTotal,
            long queuedExpectedKvReservedTotal) {
        return auditView(inflight, confirmed, settledTombstones,
                preemptionClaims, preemptionAttemptIncoming, fenceProtected,
                queuedPhase, dispatchPermits, queuedPhaseCount,
                queuedKvReservedTotal, queuedExpectedKvReservedTotal, true);
    }

    /**
     * Stage-2 L7 soak-fix form: the trailing {@code certified} flag mirrors
     * the seqlock capture contract (torn fallback captures carry false and
     * are exempt from the cross-phase aggregate mirror rule).
     */
    private static DecodeLedgerAuditView auditView(
            Map<Long, RequestInflight> inflight,
            Map<Long, Long> confirmed,
            Set<Long> settledTombstones,
            Set<Long> preemptionClaims,
            Set<Long> preemptionAttemptIncoming,
            Set<Long> fenceProtected,
            Set<Long> queuedPhase,
            Set<Long> dispatchPermits,
            int queuedPhaseCount,
            long queuedKvReservedTotal,
            long queuedExpectedKvReservedTotal,
            boolean certified) {
        return new DecodeLedgerAuditView(
                1L,
                Map.copyOf(inflight),
                0,
                Map.copyOf(confirmed),
                Set.copyOf(preemptionClaims),
                Set.copyOf(preemptionAttemptIncoming),
                Set.copyOf(fenceProtected),
                Set.copyOf(settledTombstones),
                Set.copyOf(queuedPhase),
                Set.copyOf(dispatchPermits),
                0L,
                0L,
                queuedPhaseCount,
                queuedKvReservedTotal,
                queuedExpectedKvReservedTotal,
                certified);
    }

    private static void stubDecodeView(
            DecodeEndpoint decode, DecodeLedgerAuditView view) {
        when(decode.ledgerAuditView()).thenReturn(view);
    }

    private void enqueueActive(BatchItem item) {
        registryLock.lock();
        try {
            assertTrue(registry.enqueueActiveUnderLock(item));
        } finally {
            registryLock.unlock();
        }
    }

    private boolean bind(Registered registered) {
        try (RequestLifecycleCoordinator.AdmissionScope admission =
                     lifecycle.beginAdmission(
                             registered.item().requestId(),
                             registered.future())) {
            assertNotNull(admission);
            return lifecycle.commitInflight(
                    registered.item(), false, () -> true);
        }
    }

    private Registered registerItem(
            long requestId,
            long seqLen,
            long decodeExpectedKvTokens,
            int priority,
            ReservationHandle reservation,
            DecodeEndpoint decodeEndpoint) {
        BalanceContext context = context(requestId, seqLen, priority);
        CompletableFuture<Response> future =
                lifecycle.register(context, 8);
        BatchItem item = new BatchItem(
                context,
                future,
                new Response(),
                null,
                null,
                mock(PrefillEndpoint.class),
                decodeEndpoint,
                reservation,
                decodeExpectedKvTokens,
                System.currentTimeMillis());
        return new Registered(item, future);
    }

    /** Builds a frozen item without touching the lifecycle ledger. */
    private BatchItem buildItem(
            BalanceContext context,
            long decodeExpectedKvTokens,
            ReservationHandle reservation,
            DecodeEndpoint decodeEndpoint) {
        return new BatchItem(
                context,
                new CompletableFuture<>(),
                new Response(),
                null,
                null,
                mock(PrefillEndpoint.class),
                decodeEndpoint,
                reservation,
                decodeExpectedKvTokens,
                System.currentTimeMillis());
    }

    private BalanceContext context(
            long requestId,
            long seqLen,
            int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority,
                System.currentTimeMillis() + 60_000L));
        return context;
    }

    private record Registered(
            BatchItem item,
            CompletableFuture<Response> future) {
    }
}
