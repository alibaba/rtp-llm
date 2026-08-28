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
        return new DecodeLedgerAuditView(
                1L,
                Map.copyOf(inflight),
                0,
                Map.copyOf(confirmed),
                Set.of(),
                Set.of(),
                Set.of(),
                Set.copyOf(settledTombstones),
                Set.of(),
                Set.of());
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
