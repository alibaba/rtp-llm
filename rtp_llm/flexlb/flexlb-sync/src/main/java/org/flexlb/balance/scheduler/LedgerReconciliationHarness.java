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
 * DecodeEndpoint eight-layer ledger projection and the PrefillWorkRegistry
 * queue ledger, with structured diff output.
 *
 * <h2>Rule coverage (stage-1 fix C — actual surface, not "eight layers")</h2>
 *
 * <p>The rules consume L1 (inflight), L3 (confirmed), L4 (preemption
 * claims) + L4b (attempt incoming), L5 (fence protections), L7 (queued
 * phase) and L8 (dispatch permits) — the last four joined this fix round —
 * plus the prefill active items.  Deliberately outside the rule surface:
 * L2 is design-level count-only (identities stay engine-side), L6 serves
 * only as a terminal-evidence exemption input, and the slot-side L4/L5
 * mirrors are the PreemptionRegistration / engine-fence domains compared
 * by the stage-1 rules below (KvAllocatedSameTickAtomicityTest additionally
 * covers the L7/L8 same-tick retirement behavior at the engine tick
 * level).  The authoritative eight-layer consolidation table lives on
 * {@link DecodeLedgerAuditView}.</p>
 *
 * <h2>Shadow-mode discipline</h2>
 *
 * <p>The harness lives strictly off the hot path: every capture is a short
 * single-domain critical section (each DecodeEndpoint under its own
 * admission lock — only the five HashMap-backed layers, never O(running);
 * each registry under its own lock; each slot under its own monitor, with
 * retention-window tombstones skipped lock-free) and no two locks are ever
 * held nested — the delivery lock contract and slot-lock discipline are
 * respected by construction.  The periodic loop runs on one daemon thread;
 * a failure inside the loop is logged and swallowed, so the reconciler can
 * never perturb production scheduling.  On-demand reconciliation
 * ({@link #reconcileOnce()}) is synchronous and is what tests assert
 * against.</p>
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
 *       prefill item identity mismatch, prefill-queue orphans, and the
 *       L4/L4b/L5/L7/L8 ownership splits (unbacked preemption / fence
 *       registrations, orphaned engine-side claims / protections, sub-state
 *       members outside their backing layer).</li>
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
     *
     * <p><b>Confirm-window default warning (stage-1 fix E3):</b> this form
     * defaults to {@code realDiffConfirmCycles = 1}, i.e. a REAL diff is
     * reported on the very first (single-snapshot) pass.  That is the
     * deterministic single-shot semantics the rule-level tests rely on,
     * but a shadow loop attached this way will surface single-snapshot
     * tears as immediately-confirmed REAL diffs.  Production shadow loops
     * must use the explicit-window overload and pass 2-3.</p>
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
        compareDecodeDomainsAgainstSlots(slots, decode, raw);
        compareDecodeInternalConsistency(decode, raw);
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
            // Best-effort bounded drain (stage-1 fix E4): an in-flight pass
            // holds no harness monitor, so a short await cannot deadlock;
            // a report may still be delivered to the listener after close()
            // returns when the drain window elapses — the listener contract
            // tolerates late delivery.
            try {
                shadowLoop.awaitTermination(500, TimeUnit.MILLISECONDS);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
            shadowLoop = null;
        }
    }

    // ==================== capture ====================

    /**
     * Performance contract (stage-1 fix D2): O(active + terminalizing)
     * slot-monitor acquisitions plus one lock-free pass over the whole slot
     * table.  Retention-window TOMBSTONE slots — the only population that
     * grows with throughput — take the lock-free fast path (their audit
     * content is fully derived: the tombstone tick already cleared item /
     * rows / registrations), so the per-pass synchronized cost scales with
     * live work, not with tombstone retention.
     */
    private List<SlotAudit> captureSlots() {
        List<SlotAudit> audits = new ArrayList<>();
        for (RequestSlot slot : lifecycle.snapshotSlots()) {
            // Fast path: skip the slot monitor for retention-window
            // tombstones.  isCurrent is a ConcurrentHashMap read, safe
            // outside the monitor.
            if (slot.isTombstonedFast()) {
                if (lifecycle.isCurrent(slot)) {
                    audits.add(SlotAudit.tombstone(slot.requestId()));
                }
                continue;
            }
            synchronized (slot) {
                if (!lifecycle.isCurrent(slot)) {
                    continue;
                }
                // The slot phase is projected through the adjudication
                // layer (plan 3.1 item 3): coarsePhaseOf is the single
                // ruling entrance the reconciliation surface consumes.
                audits.add(new SlotAudit(
                        slot.requestId(),
                        SlotPhaseAdjudicator.coarsePhaseOf(
                                slot.slotPhase()),
                        slot.activeItem(),
                        slot.placement(),
                        slot.prefillRow(),
                        slot.decodeRow(),
                        slot.preemptionOwnerView(),
                        slot.hasEngineFenceRegistration()));
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
                // Stage-1 fix A: an endpoint that left the reconciliation
                // surface (generation replaced / retired, registry handed
                // out a new object under the same address) while slots
                // still reference the old object is a legal failover
                // window, not an unbacked reservation.  Slots hold no
                // GenerationPin and retirement does not wait for slot-side
                // settlement, so the window can span tens of seconds —
                // longer than any sane confirm window.  Terminal-track
                // slots are simply skipped (symmetric with the two-stage
                // death exemption below); an ACTIVE slot referencing a
                // departed endpoint gets its own rule so operators can
                // configure a confirm window above the expected failover
                // time instead of losing trust in the unbacked-reservation
                // alarm.
                if (audit.terminalTrack()) {
                    continue;
                }
                diffs.add(new LedgerDiff(
                        Rule.ENDPOINT_LEFT_RECONCILIATION_SURFACE,
                        audit.requestId,
                        "slot decode endpoint left the reconciliation surface"
                                + " (replacement/retirement window) while the"
                                + " slot is still ACTIVE — legal failover"
                                + " window; confirm above the expected"
                                + " failover duration"));
                continue;
            }
            // Stage-1 fix C: L4/L5 domain consistency, slot→engine
            // direction.  Install windows are slot-ahead (coordinator
            // claims the slot registration before the endpoint installs
            // the L4 claim) and removal windows are engine-ahead (the
            // endpoint settles the claim / protection before the
            // slot-side fact lands); the NOT_FOUND fence transfer spans
            // both monitors.  All are single-interleave tears absorbed by
            // the confirm window — a persistent split is structural.
            if (!audit.terminalTrack()) {
                if (audit.preemptionOwner != null
                        && !view.preemptionClaimRequestIds()
                                .contains(audit.requestId)) {
                    diffs.add(new LedgerDiff(
                            Rule.PREEMPTION_REGISTRATION_UNBACKED,
                            audit.requestId,
                            "slot holds a preemption registration the decode"
                                    + " layer-4 claim ledger does not back"));
                }
                if (audit.engineFenceInstalled
                        && !view.engineFenceProtectedRequestIds()
                                .contains(audit.requestId)) {
                    diffs.add(new LedgerDiff(
                            Rule.ENGINE_FENCE_UNBACKED,
                            audit.requestId,
                            "slot holds an engine fence registration the"
                                    + " decode layer-5 protection ledger"
                                    + " does not back"));
                }
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
                    if (settledPresent || audit.terminalTrack()) {
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
                if (!audit.terminalTrack()) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_CONFIRMED_AHEAD_OF_SLOT,
                            audit.requestId,
                            "engine projection confirmed; slot authority"
                                    + " handover still in flight"));
                }
                continue;
            }
            if (!inflightPresent) {
                if (audit.terminalTrack()) {
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
                if (audit == null || audit.terminalTrack()) {
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

    /**
     * Stage-1 fix C: L4/L5 domain consistency, engine→slot direction.
     * A live engine-side claim / protection with no slot-side mirror on
     * an ACTIVE slot is structural: the endpoint installs an L4 claim
     * only after the coordinator already claimed the slot registration,
     * and it settles the claim before the slot-side fact can land — so
     * "engine holds, slot lacks" has no single-interleave legal window
     * (residual cross-capture tears are absorbed by the confirm window,
     * and terminal-track slots are exempt like everywhere else).  Claims
     * whose request has no slot at all are skipped: generation-replacement
     * residue belongs to the stage-2 retirement basis, not this rule.
     */
    private void compareDecodeDomainsAgainstSlots(
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
            for (Long victimId : view.preemptionClaimRequestIds()) {
                SlotAudit audit = byRequest.get(victimId);
                if (audit == null || audit.terminalTrack()) {
                    continue;
                }
                if (audit.preemptionOwner == null) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_PREEMPTION_CLAIM_ORPHAN,
                            victimId,
                            "decode layer-4 holds a priority claim whose slot"
                                    + " carries no preemption registration"));
                }
            }
            for (Long requestId : view.engineFenceProtectedRequestIds()) {
                SlotAudit audit = byRequest.get(requestId);
                if (audit == null || audit.terminalTrack()) {
                    continue;
                }
                if (!audit.engineFenceInstalled
                        && audit.preemptionOwner == null) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_FENCE_PROTECTION_ORPHAN,
                            requestId,
                            "decode layer-5 holds a fence protection whose slot"
                                    + " carries neither fence nor preemption"
                                    + " registration"));
                }
            }
        }
    }

    /**
     * Stage-1 fix C: endpoint-internal sub-state consistency.  These are
     * engine-internal invariants installed inside one admission tick —
     * L4b attempt incoming must hold an L1 shadow reservation
     * (reservedLocked at begin), L7 queued-phase members must sit in L1
     * (marked from an exact reservation), and L8 permit holders must hold
     * both L1 and L7 (acquire checks all three).  The layered audit
     * capture (fix D1) reads the concurrent-container layers lock-free, so
     * adjacent-layer tears of up to one admission tick surface here as
     * candidates; the confirm window absorbs them.
     */
    private void compareDecodeInternalConsistency(
            IdentityHashMap<DecodeEndpoint, DecodeLedgerAuditView> decode,
            List<LedgerDiff> diffs) {
        for (Map.Entry<DecodeEndpoint, DecodeLedgerAuditView> entry
                : decode.entrySet()) {
            DecodeLedgerAuditView view = entry.getValue();
            for (Long incomingId
                    : view.preemptionAttemptIncomingRequestIds()) {
                if (!view.inflight().containsKey(incomingId)) {
                    diffs.add(new LedgerDiff(
                            Rule.PREEMPTION_ATTEMPT_INCOMING_UNBACKED,
                            incomingId,
                            "decode layer-4b attempt incoming reservation is"
                                    + " not backed by a layer-1 shadow"));
                }
            }
            // Stage-2 L7 retarget: the queued projection derives from the
            // layer-1 entry sub-state flags, so a queued member outside the
            // inflight set is structurally impossible. The rule is
            // rewritten (not merely retargeted — same treatment as the
            // KV_MIRROR_MISMATCH stage-2 note) into an aggregate-mirror
            // check: the three O(1) queued counters must match the
            // entry-derived projection. Request id 0 marks the
            // endpoint-level aggregate identity of this diff.
            int queuedEntries = 0;
            long queuedHardKv = 0L;
            long queuedExpectedKv = 0L;
            for (Map.Entry<Long, RequestInflight> inflight
                    : view.inflight().entrySet()) {
                RequestInflight reservation = inflight.getValue();
                if (reservation.masterQueued()) {
                    queuedEntries++;
                    queuedHardKv += reservation.kvTokens();
                    queuedExpectedKv += reservation.expectedKvTokens();
                }
            }
            if (view.queuedPhaseCount() != queuedEntries
                    || view.queuedKvReservedTotal() != queuedHardKv
                    || view.queuedExpectedKvReservedTotal() != queuedExpectedKv) {
                diffs.add(new LedgerDiff(
                        Rule.QUEUED_PHASE_OUTSIDE_INFLIGHT,
                        0L,
                        "decode queued aggregate mirror drift: counters"
                                + " (count=" + view.queuedPhaseCount()
                                + ", hard=" + view.queuedKvReservedTotal()
                                + ", expected="
                                + view.queuedExpectedKvReservedTotal()
                                + ") != entry-derived projection (count="
                                + queuedEntries + ", hard=" + queuedHardKv
                                + ", expected=" + queuedExpectedKv + ")"));
            }
            for (Long requestId : view.engineDispatchPermitRequestIds()) {
                if (!view.queuedPhaseRequestIds().contains(requestId)
                        || !view.inflight().containsKey(requestId)) {
                    diffs.add(new LedgerDiff(
                            Rule.DISPATCH_PERMIT_OUTSIDE_QUEUED_PHASE,
                            requestId,
                            "decode layer-8 dispatch-permit holder is not a"
                                    + " queued layer-1 inflight reservation"));
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
                    if (audit.terminalTrack()) {
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
        /**
         * Slot and decode disagree on the reservation token.
         */
        RESERVATION_TOKEN_MISMATCH(true),
        /**
         * A-road numeric mirror (hard/expected/priority) mismatch.
         * Stage-2 note: this rule reads decode layer 1; when stage 2
         * retires L1 the numeric authority moves and this rule loses its
         * backing source — it must be rewritten (not merely retargeted)
         * against the stage-2 authority before L1 retirement lands.
         */
        KV_MIRROR_MISMATCH(true),
        /** Slot dRow installed while decode layer 3 never confirmed. */
        SLOT_PROJECTION_UNCONFIRMED(true),
        /** Prefill queue item differs from the slot's frozen item. */
        PREFILL_ITEM_IDENTITY_MISMATCH(true),
        /** Prefill queue item with no request slot at all. */
        PREFILL_ORPHAN_ACTIVE_ITEM(true),
        /**
         * Stage-1 fix A: an ACTIVE slot references a decode endpoint that
         * left the reconciliation surface (generation replacement /
         * retirement) — a legal failover window for terminal-track slots,
         * reported separately so its confirm window can be configured
         * above the expected failover duration instead of eroding the
         * SLOT_RESERVATION_UNBACKED alarm.
         */
        ENDPOINT_LEFT_RECONCILIATION_SURFACE(true),
        /**
         * Stage-1 fix C: slot PreemptionRegistration with no decode
         * layer-4 claim backing (install window is slot-ahead, removal
         * window engine-ahead — confirm-window territory).
         */
        PREEMPTION_REGISTRATION_UNBACKED(true),
        /**
         * Stage-1 fix C: decode layer-4 priority claim on an ACTIVE slot
         * with no slot-side registration (no single-interleave legal
         * window).
         */
        DECODE_PREEMPTION_CLAIM_ORPHAN(true),
        /**
         * Stage-1 fix C: decode layer-4b attempt incoming reservation
         * missing its layer-1 shadow backing (engine-internal invariant).
         */
        PREEMPTION_ATTEMPT_INCOMING_UNBACKED(true),
        /**
         * Stage-1 fix C: slot engine-fence registration with no decode
         * layer-5 protection backing (removal window engine-ahead —
         * confirm-window territory).
         */
        ENGINE_FENCE_UNBACKED(true),
        /**
         * Stage-1 fix C: decode layer-5 fence protection on an ACTIVE
         * slot with neither fence nor preemption registration (no
         * single-interleave legal window).
         */
        DECODE_FENCE_PROTECTION_ORPHAN(true),
        /**
         * Stage-1 fix C originally guarded the layer-7 set against members
         * outside the layer-1 inflight set.  Stage-2 L7 retirement rewrote
         * it (not merely retargeted): the queued projection now derives
         * from the layer-1 entry sub-state flags, so membership outside
         * inflight is structurally impossible — the rule instead
         * cross-checks the three O(1) queued aggregate counters against
         * the entry-derived projection (engine-internal invariant;
         * request id 0 marks the endpoint-level aggregate identity).
         */
        QUEUED_PHASE_OUTSIDE_INFLIGHT(true),
        /**
         * Stage-1 fix C: decode layer-8 dispatch-permit holder outside the
         * queued layer-1 inflight set (engine-internal invariant).
         */
        DISPATCH_PERMIT_OUTSIDE_QUEUED_PHASE(true),
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

    /**
     * Slot-side audit tuple captured under the slot monitor (or fully
     * derived on the lock-free tombstone fast path).  The storage/cleanup
     * phase is captured as the {@link SlotPhaseAdjudicator.CoarsePhase}
     * projection of the live slot track (plan 3.1 item 3) and every
     * phase-dependent exemption below derives from it — the adjudication
     * layer is the single projection entrance this harness consumes.
     */
    private record SlotAudit(
            long requestId,
            SlotPhaseAdjudicator.CoarsePhase coarsePhase,
            BatchItem item,
            SlotPlacement placement,
            SlotResourceRow prefillRow,
            SlotResourceRow decodeRow,
            PreemptionRegistration preemptionOwner,
            boolean engineFenceInstalled) {

        /** Fully derived tombstone audit for the lock-free fast path. */
        static SlotAudit tombstone(long requestId) {
            return new SlotAudit(
                    requestId,
                    SlotPhaseAdjudicator.CoarsePhase.TOMBSTONE,
                    null, null, null, null, null, false);
        }

        /** Terminal-track exemption: TERMINALIZING or TOMBSTONE. */
        boolean terminalTrack() {
            return coarsePhase != SlotPhaseAdjudicator.CoarsePhase.ACTIVE;
        }
    }

    /**
     * Unified prefill-side audit tuple: the request-id level is always
     * captured; the frozen {@link BatchItem} identities are present only
     * for the scheduler-package detailed capture (the public ledger port
     * cannot carry them).  A null {@code activeItems} marks the degraded
     * interface-level form.  The committed-request counter is deliberately
     * not captured (stage-1 fix C4): no reconciliation rule consumes it —
     * dead capture was removed rather than kept "for later".
     */
    private record PrefillAudit(
            long capturedAtMs,
            List<Long> activeRequestIds,
            List<BatchItem> activeItems) {

        static PrefillAudit fromDetailed(
                PrefillWorkRegistry.DetailedPrefillLedgerAudit detailed) {
            List<Long> ids = new ArrayList<>(detailed.activeItems().size());
            for (BatchItem item : detailed.activeItems()) {
                ids.add(item.requestId());
            }
            return new PrefillAudit(
                    detailed.capturedAtMs(),
                    List.copyOf(ids),
                    detailed.activeItems());
        }

        static PrefillAudit fromInterface(
                PrefillWorkLedger.PrefillLedgerAuditSnapshot snapshot) {
            return new PrefillAudit(
                    snapshot.capturedAtMs(),
                    snapshot.activeItemRequestIds(),
                    null);
        }
    }

    /** One prefill ledger capture action; each capture runs in its own
     *  short registry-lock critical section. */
    @FunctionalInterface
    private interface PrefillCapture {
        PrefillAudit capture();
    }
}
