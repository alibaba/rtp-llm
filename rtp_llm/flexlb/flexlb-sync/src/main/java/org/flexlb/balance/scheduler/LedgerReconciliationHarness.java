package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeLedgerAuditView;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.scheduler.RequestSlot.SlotPlacement;
import org.flexlb.balance.scheduler.RequestSlot.SlotResourceRow;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * Three-way ledger reconciliation harness (plan section 6, stage 1 / M1):
 * periodic comparison of the {@code requestSlots} ledger against the
 * DecodeEndpoint eight-layer ledger and the PrefillWorkRegistry queue
 * ledger, with structured diff output.
 *
 * <h2>Shadow-mode discipline</h2>
 *
 * <p>The harness lives strictly off the hot path: every capture is a short
 * single-domain critical section (each DecodeEndpoint under its own
 * admission lock, each registry under its own lock, each slot under its own
 * monitor) and no two locks are ever held nested — the delivery lock
 * contract and slot-lock discipline are respected by construction.  The
 * periodic loop runs on one daemon thread; a failure inside the loop is
 * logged and swallowed, so the reconciler can never perturb production
 * scheduling.  On-demand reconciliation ({@link #reconcileOnce()}) is
 * synchronous and is what tests assert against.</p>
 *
 * <h2>Directional tolerance (v2 A/B roads)</h2>
 *
 * <p>The KV_ALLOCATED critical point is crossed inside one DecodeEndpoint
 * admission-lock tick (reservation retired, projection confirmed) while the
 * slot flips its pRow/dRow authority inside one slot-monitor mutation driven
 * by the accepted fact.  Between those two linearization points the two
 * ledgers legitimately disagree in the <em>engine-ahead</em> direction.  The
 * rules therefore distinguish:</p>
 *
 * <ul>
 *   <li><b>REAL diffs</b> — structural splits that no legal interleaving
 *       can produce: duplicate decode ownership (reservation and projection
 *       coexisting), a slot reservation with no endpoint backing at all
 *       (double-miss), token mismatch, A-road numeric mirror mismatch, a
 *       slot projection the engine never confirmed (reverse projection),
 *       prefill item identity mismatch, prefill-queue orphans.</li>
 *   <li><b>TRANSIENT diffs</b> — the engine-ahead window: the endpoint
 *       confirmed (or settled) before the slot applied the fact, and the
 *       admission window where an endpoint reservation exists before the
 *       slot publication bind.  These are reported but never fail a
 *       quiesced reconciliation.</li>
 * </ul>
 *
 * <h2>Confirm-window semantics (single-snapshot tear tolerance)</h2>
 *
 * <p>The three captures are each single-lock atomic but mutually
 * unaligned, and the slot side applies endpoint facts asynchronously —
 * so one pass can catch a legal tear: a fresh admission crossing the
 * capture order (prefill ledger seen before the slot table), or an
 * engine that already retired a projection while the slot-side
 * terminal fact is still in flight (the endpoint settles some finishes
 * without a layer-6 tombstone, so "layer 3 gone and layer 6 gone" is a
 * legal finish window, not a reverse projection).  Tears converge
 * within one or two periods; structural splits persist.  A REAL diff is
 * therefore <em>confirmed</em> only after it recurs in
 * {@code realDiffConfirmCycles} consecutive passes (default 1 = report
 * immediately, the single-shot test semantics; shadow loops should
 * configure 2-3).  A candidate observed fewer times surfaces as a
 * pending diff and its counter resets on the first pass it
 * disappears.</p>
 */
public final class LedgerReconciliationHarness implements AutoCloseable {

    private final RequestLifecycleCoordinator lifecycle;
    private final Supplier<List<DecodeEndpoint>> decodeEndpointSource;
    private final Supplier<List<PrefillCapture>> prefillCaptureSource;
    private final Listener listener;
    private final int realDiffConfirmCycles;

    /** Consecutive-pass counters backing the REAL confirm window. */
    private final ConcurrentHashMap<DiffKey, Integer> pendingRealCounts =
            new ConcurrentHashMap<>();

    private ScheduledExecutorService shadowLoop;

    /**
     * Fixed-surface form used by same-package tests: the endpoint and
     * registry lists are captured once and every reconciliation compares
     * exactly these ledgers with full item-identity comparison.
     */
    public LedgerReconciliationHarness(
            RequestLifecycleCoordinator lifecycle,
            List<DecodeEndpoint> decodeEndpoints,
            List<PrefillWorkRegistry> prefillRegistries,
            Listener listener) {
        this(lifecycle, decodeEndpoints, prefillRegistries, listener, 1);
    }

    /**
     * Fixed-surface form with an explicit REAL-diff confirm window: a
     * REAL rule fires only after it recurs this many consecutive passes
     * (tear tolerance for mutually unaligned single-domain snapshots).
     */
    public LedgerReconciliationHarness(
            RequestLifecycleCoordinator lifecycle,
            List<DecodeEndpoint> decodeEndpoints,
            List<PrefillWorkRegistry> prefillRegistries,
            Listener listener,
            int realDiffConfirmCycles) {
        this(lifecycle,
                () -> decodeEndpoints,
                () -> {
                    List<PrefillCapture> captures =
                            new ArrayList<>(prefillRegistries.size());
                    for (PrefillWorkRegistry registry : prefillRegistries) {
                        captures.add(() -> PrefillAudit.fromDetailed(
                                registry.ledgerAuditSnapshotDetailed()));
                    }
                    return captures;
                },
                listener,
                realDiffConfirmCycles);
    }

    /**
     * Registry-attached form used from the public facade
     * ({@link RequestScheduler#attachLedgerReconciliation}): every pass
     * re-snapshots the live endpoint generations, so endpoint retirement
     * and re-registration are picked up naturally.  Prefill ledgers whose
     * implementation is the scheduler-package {@link PrefillWorkRegistry}
     * upgrade to the detailed (item-identity) capture; foreign
     * implementations degrade to the request-id level interface snapshot.
     */
    static LedgerReconciliationHarness attach(
            RequestLifecycleCoordinator lifecycle,
            EndpointRegistry registry,
            Listener listener) {
        return attach(lifecycle, registry, listener, 1);
    }

    /** Attached form with an explicit REAL-diff confirm window. */
    static LedgerReconciliationHarness attach(
            RequestLifecycleCoordinator lifecycle,
            EndpointRegistry registry,
            Listener listener,
            int realDiffConfirmCycles) {
        return new LedgerReconciliationHarness(
                lifecycle,
                () -> List.copyOf(
                        registry.snapshotDecodeEndpoints().values()),
                () -> {
                    List<PrefillCapture> captures = new ArrayList<>();
                    for (PrefillEndpoint endpoint
                            : registry.snapshotPrefillEndpoints().values()) {
                        captures.add(capturePrefill(endpoint));
                    }
                    return captures;
                },
                listener,
                realDiffConfirmCycles);
    }

    private static PrefillCapture capturePrefill(PrefillEndpoint endpoint) {
        PrefillWorkLedger ledger = endpoint.workLedger();
        if (ledger instanceof PrefillWorkRegistry registry) {
            return () -> PrefillAudit.fromDetailed(
                    registry.ledgerAuditSnapshotDetailed());
        }
        return () -> PrefillAudit.fromInterface(
                ledger.ledgerAuditSnapshot());
    }

    private LedgerReconciliationHarness(
            RequestLifecycleCoordinator lifecycle,
            Supplier<List<DecodeEndpoint>> decodeEndpointSource,
            Supplier<List<PrefillCapture>> prefillCaptureSource,
            Listener listener,
            int realDiffConfirmCycles) {
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.decodeEndpointSource = Objects.requireNonNull(
                decodeEndpointSource, "decodeEndpointSource");
        this.prefillCaptureSource = Objects.requireNonNull(
                prefillCaptureSource, "prefillCaptureSource");
        this.listener = listener;
        if (realDiffConfirmCycles < 1) {
            throw new IllegalArgumentException(
                    "realDiffConfirmCycles must be >= 1: "
                            + realDiffConfirmCycles);
        }
        this.realDiffConfirmCycles = realDiffConfirmCycles;
    }

    /**
     * Runs one three-way comparison and returns the structured report.
     * REAL rules recur-check across passes through the confirm window
     * (see the class-level note); the report carries confirmed REAL
     * diffs, unconfirmed tear candidates, and TRANSIENT window diffs.
     * Never throws for ledger disagreements; only for null capture
     * inputs.
     */
    public ReconciliationReport reconcileOnce() {
        List<SlotAudit> slots = captureSlots();
        List<DecodeEndpoint> decodeEndpoints = decodeEndpointSource.get();
        IdentityHashMap<DecodeEndpoint, DecodeLedgerAuditView> decode =
                new IdentityHashMap<>();
        for (DecodeEndpoint endpoint : decodeEndpoints) {
            decode.put(endpoint, endpoint.ledgerAuditView());
        }
        List<PrefillAudit> prefill = new ArrayList<>();
        for (PrefillCapture capture : prefillCaptureSource.get()) {
            prefill.add(capture.capture());
        }

        List<LedgerDiff> raw = new ArrayList<>();
        compareSlotsAgainstDecode(slots, decode, raw);
        compareDecodeAgainstSlots(slots, decode, raw);
        comparePrefillAgainstSlots(slots, prefill, raw);

        List<LedgerDiff> confirmedReal = new ArrayList<>();
        List<LedgerDiff> pendingReal = new ArrayList<>();
        List<LedgerDiff> transientDiffs = new ArrayList<>();
        Set<DiffKey> seenReal = new HashSet<>();
        for (LedgerDiff diff : raw) {
            if (!diff.rule().isReal()) {
                transientDiffs.add(diff);
                continue;
            }
            DiffKey key = new DiffKey(diff.rule(), diff.requestId());
            seenReal.add(key);
            int consecutive =
                    pendingRealCounts.merge(key, 1, Integer::sum);
            if (consecutive >= realDiffConfirmCycles) {
                confirmedReal.add(diff);
            } else {
                pendingReal.add(diff);
            }
        }
        // A candidate that skipped this pass was a tear: reset it so a
        // later recurrence must climb the full window again.
        pendingRealCounts.keySet().retainAll(seenReal);

        ReconciliationReport report = new ReconciliationReport(
                System.currentTimeMillis(),
                slots.size(),
                decode.size(),
                prefill.size(),
                confirmedReal,
                pendingReal,
                transientDiffs);
        if (listener != null) {
            listener.onReport(report);
        }
        return report;
    }

    /**
     * Starts the shadow reconciliation loop (idempotent).  The period is
     * configurable; the loop is a single daemon thread and every iteration
     * failure is logged, never propagated.
     */
    public synchronized void startShadowLoop(long periodMs) {
        if (periodMs <= 0L) {
            throw new IllegalArgumentException(
                    "shadow loop period must be positive: " + periodMs);
        }
        if (shadowLoop != null) {
            return;
        }
        shadowLoop = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread worker = new Thread(
                    runnable, "flexlb-ledger-reconciler");
            worker.setDaemon(true);
            return worker;
        });
        shadowLoop.scheduleWithFixedDelay(
                this::reconcileSafely,
                periodMs,
                periodMs,
                TimeUnit.MILLISECONDS);
    }

    private void reconcileSafely() {
        try {
            reconcileOnce();
        } catch (RuntimeException | Error failure) {
            Logger.warn(
                    "Ledger reconciliation shadow pass failed: {}",
                    failure.toString(), failure);
        }
    }

    @Override
    public synchronized void close() {
        if (shadowLoop != null) {
            shadowLoop.shutdownNow();
            shadowLoop = null;
        }
    }

    // ==================== capture ====================

    private List<SlotAudit> captureSlots() {
        List<SlotAudit> audits = new ArrayList<>();
        for (RequestSlot slot : lifecycle.snapshotSlots()) {
            synchronized (slot) {
                if (!lifecycle.isCurrent(slot)) {
                    continue;
                }
                audits.add(new SlotAudit(
                        slot.requestId(),
                        slot.isTombstone(),
                        slot.isTerminalizingOrLater(),
                        slot.activeItem(),
                        slot.placement(),
                        slot.prefillRow(),
                        slot.decodeRow()));
            }
        }
        return audits;
    }

    // ==================== rules ====================

    private void compareSlotsAgainstDecode(
            List<SlotAudit> slots,
            IdentityHashMap<DecodeEndpoint, DecodeLedgerAuditView> decode,
            List<LedgerDiff> diffs) {
        for (SlotAudit audit : slots) {
            if (audit.item == null || audit.placement == null) {
                continue;
            }
            DecodeEndpoint endpoint = audit.placement.decodeEndpoint();
            if (endpoint == null) {
                continue;
            }
            DecodeLedgerAuditView view = decode.get(endpoint);
            if (view == null) {
                diffs.add(new LedgerDiff(
                        Rule.SLOT_RESERVATION_UNBACKED,
                        audit.requestId,
                        "slot decode endpoint is not part of the"
                                + " reconciliation surface"));
                continue;
            }
            boolean inflightPresent =
                    view.inflight().containsKey(audit.requestId);
            boolean confirmedPresent = view.confirmedReservationTokens()
                    .containsKey(audit.requestId);
            boolean settledPresent = view.settledTombstoneRequestIds()
                    .contains(audit.requestId);

            if (audit.decodeRow != null) {
                // B-road: the slot claims the engine confirmed.  Reverse
                // projection (slot ahead of the engine ledger) is REAL,
                // unless the engine already settled the projection and
                // only the slot-side terminal projection lags behind.
                if (!confirmedPresent) {
                    if (settledPresent || audit.terminalTrack) {
                        diffs.add(new LedgerDiff(
                                Rule.DECODE_SETTLED_AHEAD_OF_SLOT,
                                audit.requestId,
                                "endpoint settled the projection; slot terminal"
                                        + " projection still in flight"));
                    } else {
                        diffs.add(new LedgerDiff(
                                Rule.SLOT_PROJECTION_UNCONFIRMED,
                                audit.requestId,
                                "slot dRow is installed but the decode layer 3"
                                        + " holds no confirmed projection"));
                    }
                    continue;
                }
                compareConfirmedToken(audit, view, diffs);
                continue;
            }

            // A-road master reservation mirror.
            if (inflightPresent && confirmedPresent) {
                diffs.add(new LedgerDiff(
                        Rule.DUPLICATE_OWNERSHIP_ON_DECODE,
                        audit.requestId,
                        "decode layers 1 and 3 both hold the request at the"
                                + " captured admission version"));
                continue;
            }
            if (confirmedPresent && !inflightPresent) {
                if (!audit.terminalTrack) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_CONFIRMED_AHEAD_OF_SLOT,
                            audit.requestId,
                            "engine projection confirmed; slot authority"
                                    + " handover still in flight"));
                }
                continue;
            }
            if (!inflightPresent) {
                if (audit.terminalTrack) {
                    // Two-stage death window or tombstone: the endpoint
                    // retiring the reservation is the expected steady
                    // state, not an unbacked reservation.
                    continue;
                }
                if (settledPresent) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_SETTLED_AHEAD_OF_SLOT,
                            audit.requestId,
                            "endpoint settled the request; slot terminal"
                                    + " projection still in flight"));
                } else {
                    diffs.add(new LedgerDiff(
                            Rule.SLOT_RESERVATION_UNBACKED,
                            audit.requestId,
                            "slot pRow holds a reservation the decode"
                                    + " ledger does not back (double miss)"));
                }
                continue;
            }
            compareInflightMirror(audit, view, diffs);
        }
    }

    private void compareInflightMirror(
            SlotAudit audit,
            DecodeLedgerAuditView view,
            List<LedgerDiff> diffs) {
        long slotToken = audit.placement.decodeReservationToken();
        RequestInflight reservation =
                view.inflight().get(audit.requestId);
        if (reservation == null) {
            return;
        }
        if (reservation.reservationToken() != slotToken) {
            diffs.add(new LedgerDiff(
                    Rule.RESERVATION_TOKEN_MISMATCH,
                    audit.requestId,
                    "slot placement token " + slotToken
                            + " != decode layer-1 token "
                            + reservation.reservationToken()));
            return;
        }
        SlotResourceRow row = audit.prefillRow;
        if (row == null) {
            return;
        }
        if (row.hardKvTokens() != reservation.kvTokens()
                || row.priority() != reservation.priority()
                || (row.expectedKvTokens() != 0L
                        && row.expectedKvTokens()
                                != reservation.expectedKvTokens())) {
            diffs.add(new LedgerDiff(
                    Rule.KV_MIRROR_MISMATCH,
                    audit.requestId,
                    "slot pRow (hard=" + row.hardKvTokens()
                            + ", expected=" + row.expectedKvTokens()
                            + ", priority=" + row.priority()
                            + ") != decode layer-1 (hard="
                            + reservation.kvTokens()
                            + ", expected=" + reservation.expectedKvTokens()
                            + ", priority=" + reservation.priority() + ")"));
        }
    }

    private void compareConfirmedToken(
            SlotAudit audit,
            DecodeLedgerAuditView view,
            List<LedgerDiff> diffs) {
        Long confirmedToken = view.confirmedReservationTokens()
                .get(audit.requestId);
        if (confirmedToken == null) {
            return;
        }
        long slotToken = audit.decodeRow.reservationToken();
        if (confirmedToken != slotToken) {
            diffs.add(new LedgerDiff(
                    Rule.RESERVATION_TOKEN_MISMATCH,
                    audit.requestId,
                    "slot projection token " + slotToken
                            + " != decode layer-3 token " + confirmedToken));
        }
    }

    private void compareDecodeAgainstSlots(
            List<SlotAudit> slots,
            IdentityHashMap<DecodeEndpoint, DecodeLedgerAuditView> decode,
            List<LedgerDiff> diffs) {
        Map<Long, SlotAudit> byRequest = new java.util.HashMap<>();
        for (SlotAudit audit : slots) {
            byRequest.put(audit.requestId, audit);
        }
        for (Map.Entry<DecodeEndpoint, DecodeLedgerAuditView> entry
                : decode.entrySet()) {
            DecodeLedgerAuditView view = entry.getValue();
            for (Long requestId : view.inflight().keySet()) {
                SlotAudit audit = byRequest.get(requestId);
                if (audit == null || audit.terminalTrack) {
                    continue;
                }
                if (audit.item == null || audit.placement == null
                        || audit.placement.decodeEndpoint() != entry.getKey()) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_INFLIGHT_AHEAD_OF_SLOT,
                            requestId,
                            "decode layer-1 reservation has no bound slot on"
                                    + " this endpoint (admission window)"));
                }
            }
        }
    }

    private void comparePrefillAgainstSlots(
            List<SlotAudit> slots,
            List<PrefillAudit> prefill,
            List<LedgerDiff> diffs) {
        Map<Long, SlotAudit> byRequest = new java.util.HashMap<>();
        for (SlotAudit audit : slots) {
            byRequest.put(audit.requestId, audit);
        }
        for (PrefillAudit registry : prefill) {
            List<BatchItem> items = registry.activeItems();
            if (items != null) {
                // Detailed (scheduler-package) capture: full item-identity
                // comparison against the slot's frozen publication item.
                for (BatchItem active : items) {
                    SlotAudit audit = byRequest.get(active.requestId());
                    if (audit == null) {
                        diffs.add(new LedgerDiff(
                                Rule.PREFILL_ORPHAN_ACTIVE_ITEM,
                                active.requestId(),
                                "prefill queue holds an active item with no"
                                        + " request slot"));
                        continue;
                    }
                    if (audit.terminalTrack) {
                        continue;
                    }
                    if (audit.item != active) {
                        diffs.add(new LedgerDiff(
                                Rule.PREFILL_ITEM_IDENTITY_MISMATCH,
                                active.requestId(),
                                "prefill queue item identity differs from the"
                                        + " slot's frozen publication item"));
                    }
                }
                continue;
            }
            // Interface-level capture (public ledger port): the frozen item
            // type is scheduler-internal, so identity comparison degrades
            // to the request-id orphan check.
            for (Long requestId : registry.activeRequestIds()) {
                SlotAudit audit = byRequest.get(requestId);
                if (audit == null) {
                    diffs.add(new LedgerDiff(
                            Rule.PREFILL_ORPHAN_ACTIVE_ITEM,
                            requestId,
                            "prefill queue holds an active item with no"
                                    + " request slot"));
                }
            }
        }
    }

    // ==================== reporting ====================

    /**
     * One structured reconciliation result.  {@code realDiffs} carries
     * only REAL rules whose confirm window is satisfied (structural
     * splits that recurred across passes); a REAL candidate seen in
     * fewer consecutive passes surfaces as a {@code pendingRealDiff}
     * (single-snapshot tear candidate).  {@code transientDiffs} are the
     * engine-ahead / admission-window diffs.
     */
    public record ReconciliationReport(
            long capturedAtMs,
            int slotCount,
            int decodeEndpointCount,
            int prefillRegistryCount,
            List<LedgerDiff> realDiffs,
            List<LedgerDiff> pendingRealDiffs,
            List<LedgerDiff> transientDiffs) {

        public ReconciliationReport {
            realDiffs = List.copyOf(realDiffs);
            pendingRealDiffs = List.copyOf(pendingRealDiffs);
            transientDiffs = List.copyOf(transientDiffs);
        }
    }

    /** One ledger disagreement found by a rule. */
    public record LedgerDiff(Rule rule, long requestId, String detail) {
    }

    /**
     * Reconciliation rules; REAL rules flag structural ledger splits and
     * are confirmed only after {@code realDiffConfirmCycles} consecutive
     * passes (class-level confirm-window note).
     */
    public enum Rule {
        /** Decode layers 1 and 3 hold the same request at once. */
        DUPLICATE_OWNERSHIP_ON_DECODE(true),
        /** Slot pRow reservation with neither decode backing layer. */
        SLOT_RESERVATION_UNBACKED(true),
        /** Slot and decode disagree on the reservation token. */
        RESERVATION_TOKEN_MISMATCH(true),
        /** A-road numeric mirror (hard/expected/priority) mismatch. */
        KV_MIRROR_MISMATCH(true),
        /** Slot dRow installed while decode layer 3 never confirmed. */
        SLOT_PROJECTION_UNCONFIRMED(true),
        /** Prefill queue item differs from the slot's frozen item. */
        PREFILL_ITEM_IDENTITY_MISMATCH(true),
        /** Prefill queue item with no request slot at all. */
        PREFILL_ORPHAN_ACTIVE_ITEM(true),
        /** Engine confirmed before the slot applied the handover. */
        DECODE_CONFIRMED_AHEAD_OF_SLOT(false),
        /** Engine settled before the slot reached its terminal track. */
        DECODE_SETTLED_AHEAD_OF_SLOT(false),
        /** Decode reservation inside the admission bind window. */
        DECODE_INFLIGHT_AHEAD_OF_SLOT(false);

        private final boolean real;

        Rule(boolean real) {
            this.real = real;
        }

        /** Whether the rule flags a structural split (no legal window). */
        public boolean isReal() {
            return real;
        }
    }

    /** Shadow-mode observer; may be null. */
    public interface Listener {
        void onReport(ReconciliationReport report);
    }

    /** Confirm-window dedup key: rule + request identity. */
    private record DiffKey(Rule rule, long requestId) {
    }

    /** Slot-side audit tuple captured under the slot monitor. */
    private record SlotAudit(
            long requestId,
            boolean tombstone,
            boolean terminalTrack,
            BatchItem item,
            SlotPlacement placement,
            SlotResourceRow prefillRow,
            SlotResourceRow decodeRow) {
    }

    /**
     * Unified prefill-side audit tuple: the request-id level is always
     * captured; the frozen {@link BatchItem} identities are present only
     * for the scheduler-package detailed capture (the public ledger port
     * cannot carry them).  A null {@code activeItems} marks the degraded
     * interface-level form.
     */
    private record PrefillAudit(
            long capturedAtMs,
            List<Long> activeRequestIds,
            List<BatchItem> activeItems,
            long committedRequestCount) {

        static PrefillAudit fromDetailed(
                PrefillWorkRegistry.DetailedPrefillLedgerAudit detailed) {
            List<Long> ids = new ArrayList<>(detailed.activeItems().size());
            for (BatchItem item : detailed.activeItems()) {
                ids.add(item.requestId());
            }
            return new PrefillAudit(
                    detailed.capturedAtMs(),
                    List.copyOf(ids),
                    detailed.activeItems(),
                    detailed.committedRequestCount());
        }

        static PrefillAudit fromInterface(
                PrefillWorkLedger.PrefillLedgerAuditSnapshot snapshot) {
            return new PrefillAudit(
                    snapshot.capturedAtMs(),
                    snapshot.activeItemRequestIds(),
                    null,
                    snapshot.committedRequestCount());
        }
    }

    /** One prefill ledger capture action; each capture runs in its own
     *  short registry-lock critical section. */
    @FunctionalInterface
    private interface PrefillCapture {
        PrefillAudit capture();
    }
}
