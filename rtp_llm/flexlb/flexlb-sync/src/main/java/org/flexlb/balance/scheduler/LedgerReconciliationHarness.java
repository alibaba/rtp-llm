package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeLedgerAuditView;
import org.flexlb.balance.endpoint.DecodePlacementAuthorityPort;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.scheduler.RequestSlot.SlotDecodeAdmission;
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
        compareDecodeAdmissionMirror(slots, decode, raw);
        compareDecodeDomainsAgainstSlots(slots, decode, raw);
        compareDecodeInternalConsistency(decode, raw);
        comparePlacementProjectionRow(decode, raw);
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
                        slot.hasEngineFenceRegistration(),
                        slot.decodeAdmissionAuthorityView()));
            }
        }
        return audits;
    }

    // ==================== rules ====================

    /**
     * Stage-2 T7 S2c placement mirror rule: the aggregate placement row
     * must equal the capture-frozen entry facts (the nine placement-domain
     * components) on every certified capture. The row is maintained
     * in-transaction under the endpoint admissionLock and read inside the
     * same locked window as the version; the inflight snapshot is the
     * Phase-2 weakly-consistent copy the seqlock revalidation certifies
     * against that same version — so a certified capture proves both sides
     * share one quiet window, and any split is a row-maintenance bug (a
     * lost or duplicated in-transaction delta), never a delivery window
     * (the S2b post-commit protocol retired with the native counters).
     *
     * <p>Stage-2 T7 S2c soak round-3 fix: the queued / permit domains must
     * derive from the capture-frozen projection sets (the same treatment
     * as the L7 / L8 aggregate mirrors — soak round-2 lesson), never from
     * a live re-read of the mutable entry flags. The entry objects inside
     * {@code view.inflight()} are shared references, so a flag flip that
     * lands after the Phase-3 certification but before this rule runs
     * would fabricate a "certified" split (row high on a queued-off flip,
     * row low on a permit install): the certified window covers the
     * capture only. kvTokens / expectedKvTokens are immutable entry
     * fields, so the per-domain KV totals join through the frozen map
     * lookups. Request id 0 marks the endpoint-level aggregate identity.
     */
    private void comparePlacementProjectionRow(
            IdentityHashMap<DecodeEndpoint, DecodeLedgerAuditView> decode,
            List<LedgerDiff> diffs) {
        for (Map.Entry<DecodeEndpoint, DecodeLedgerAuditView> entry
                : decode.entrySet()) {
            DecodeLedgerAuditView view = entry.getValue();
            if (!view.certified()) {
                continue;
            }
            DecodeEndpoint.DecodePlacementProjectionRow row =
                    view.placementProjectionRow();
            int inflightCount = view.inflight().size();
            long inflightHardKv = 0L;
            long inflightExpectedKv = 0L;
            for (Map.Entry<Long, RequestInflight> inflight
                    : view.inflight().entrySet()) {
                inflightHardKv += inflight.getValue().kvTokens();
                inflightExpectedKv += inflight.getValue().expectedKvTokens();
            }
            int queuedCount = 0;
            long queuedHardKv = 0L;
            long queuedExpectedKv = 0L;
            for (Long requestId : view.queuedPhaseRequestIds()) {
                RequestInflight reservation = view.inflight().get(requestId);
                if (reservation != null) {
                    queuedCount++;
                    queuedHardKv += reservation.kvTokens();
                    queuedExpectedKv += reservation.expectedKvTokens();
                }
            }
            int permitCount = 0;
            long permitHardKv = 0L;
            long permitExpectedKv = 0L;
            for (Long requestId : view.engineDispatchPermitRequestIds()) {
                RequestInflight reservation = view.inflight().get(requestId);
                if (reservation != null) {
                    permitCount++;
                    permitHardKv += reservation.kvTokens();
                    permitExpectedKv += reservation.expectedKvTokens();
                }
            }
            if (row.inflightCount() != inflightCount
                    || row.inflightHardKv() != inflightHardKv
                    || row.inflightExpectedKv() != inflightExpectedKv
                    || row.queuedCount() != queuedCount
                    || row.queuedHardKv() != queuedHardKv
                    || row.queuedExpectedKv() != queuedExpectedKv
                    || row.permitCount() != permitCount
                    || row.permitHardKv() != permitHardKv
                    || row.permitExpectedKv() != permitExpectedKv) {
                diffs.add(new LedgerDiff(
                        Rule.PLACEMENT_PROJECTION_MISMATCH,
                        0L,
                        "decode placement row (inflight="
                                + row.inflightCount() + "/"
                                + row.inflightHardKv() + "/"
                                + row.inflightExpectedKv()
                                + ", queued=" + row.queuedCount() + "/"
                                + row.queuedHardKv() + "/"
                                + row.queuedExpectedKv()
                                + ", permit=" + row.permitCount() + "/"
                                + row.permitHardKv() + "/"
                                + row.permitExpectedKv()
                                + ", row_version="
                                + row.placementProjectionVersion()
                                + ") != entry-derived facts (inflight="
                                + inflightCount + "/"
                                + inflightHardKv + "/"
                                + inflightExpectedKv
                                + ", queued=" + queuedCount
                                + "/" + queuedHardKv + "/"
                                + queuedExpectedKv
                                + ", permit=" + permitCount + "/"
                                + permitHardKv + "/"
                                + permitExpectedKv
                                + ") — row-maintenance bug if persistent"));
            }
        }
    }

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
                                    + " decode fence registration table"
                                    + " does not back"));
                }
            }
            boolean inflightPresent =
                    view.inflight().containsKey(audit.requestId);
            boolean confirmedPresent = view.confirmedReservationTokens()
                    .containsKey(audit.requestId);

            if (audit.decodeRow != null) {
                // B-road: the slot claims the engine confirmed.  Reverse
                // projection (slot ahead of the engine ledger) is REAL.
                // Stage-2 L6 source switch: the settled-tombstone layer
                // signal is retired as a rule input — an ACTIVE slot whose
                // engine layers are all empty is structurally unconfirmed,
                // and the engine's settle→terminalize handoff runs
                // synchronously inside one status pump (onEndpointEvent,
                // µs-scale), so single-snapshot tears of that window never
                // climb the confirm window.  A terminal-track slot never
                // reaches this road at all: its audit carries no active
                // item (activeItem() is empty outside the ACTIVE phase),
                // which is the structural two-stage-death skip, so the
                // former settled-ahead transient rule retired together
                // with the settled-layer signal it consumed.
                if (!confirmedPresent) {
                    diffs.add(new LedgerDiff(
                            Rule.SLOT_PROJECTION_UNCONFIRMED,
                            audit.requestId,
                            "slot dRow is installed but the decode layer 3"
                                    + " holds no confirmed projection"));
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
                // Stage-2 L6 source switch: the settled-layer exemption is
                // retired — an ACTIVE slot with neither inflight nor
                // confirmed backing is a structural double miss.  The real
                // pipeline's settle→terminalize handoff is synchronous
                // inside one status pump, so its single-snapshot tears stay
                // below the confirm window.
                diffs.add(new LedgerDiff(
                        Rule.SLOT_RESERVATION_UNBACKED,
                        audit.requestId,
                        "slot pRow holds a reservation the decode"
                                + " ledger does not back (double miss)"));
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
                // Stage-2 T7 (S1): the incoming shadow of a priority /
                // local-eviction protocol owns its layer-1 entry before any
                // slot publication bind.  Classify the transient window so
                // soak / E2E runs can tell the protocol-round-trip shadow
                // from the ordinary reserve-to-bind admission window apart —
                // the classification is the empirical signal for the
                // incoming-shadow hosting ruling (T7 plan, S1 survey).
                boolean incomingShadow =
                        view.preemptionAttemptIncomingRequestIds()
                                .contains(requestId);
                if (audit.item == null || audit.placement == null
                        || audit.placement.decodeEndpoint()
                                != entry.getKey()) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_INFLIGHT_AHEAD_OF_SLOT,
                            requestId,
                            incomingShadow
                                    ? "decode layer-1 incoming shadow"
                                            + " (priority protocol in flight)"
                                            + " has no bound slot placement"
                                            + " yet"
                                    : "decode layer-1 reservation has no"
                                            + " bound slot on this endpoint"
                                            + " (publication bind window)"));
                    continue;
                }
                // Stage-2 T7 (S1 cross-check pre-embed): the publication
                // bind installs the placement and the pRow inside one
                // slot-monitor tick, so a layer-1 shadow whose ACTIVE slot
                // is already bound to this endpoint must carry its numeric
                // mirror.  A missing pRow here has no legal interleaving
                // — the rule is the writer-protocol regression guard
                // backing the S2 read-source switch (the L1 numeric
                // queries move onto the pRow) and the S4 layer-1
                // retirement.  Numeric-value drift between an installed
                // pRow and its layer-1 shadow stays with the
                // slot-direction KV_MIRROR_MISMATCH rule, and a
                // protocol-committed incoming shadow keeps its short
                // attempt-removal exemption.
                //
                // Stage-2 T7 S4 ruling (ruling 4 scope-out): layer-1
                // stays as the admission-domain resource row, so this
                // rule continues as the writer-protocol regression guard
                // for the pRow numeric-mirror installation — it retires
                // together with the L1 numeric read source in the M3/M4
                // layer-retirement agenda, not in M2.
                if (!incomingShadow && audit.prefillRow == null) {
                    diffs.add(new LedgerDiff(
                            Rule.INFLIGHT_PROW_CROSSCHECK,
                            requestId,
                            "decode layer-1 shadow with a bound placement"
                                    + " but no slot pRow numeric mirror"
                                    + " (the publication bind installs both"
                                    + " atomically)"));
                }
            }
        }
    }

    /**
     * Stage-2 T7 S3 (placement-domain migration): bidirectional mirror
     * audit between the slot-side decode-admission authority and the
     * layer-1 entry mirrors (see the
     * {@link Rule#DECODE_ADMISSION_MIRROR_MISMATCH} note).  The capture
     * order — slots first, decode second — makes the wrapper tear
     * direction deterministic (the install path stages the slot
     * authority before the admission body commits the entry, so a
     * mid-wrapper capture sees the staged authority and the pre-commit
     * mirror), and both directions run on the same pass, so a
     * single-interleave tear is a one-pass diff the confirm window
     * absorbs.
     *
     * <p>Stage-2 T7 S4 (ruling 4 scope-out): the layer-1 comparison
     * baseline is the capture-frozen {@code inflightEntryFacts} snapshot
     * — the exact fact shape the wrapper entryReader reads at refresh
     * time (readAdmissionEntry → entryFacts), frozen in the same Phase-2
     * pass as the sub-state projections.  The rules never live-re-read
     * the mutable entry bits (the soak round-3 capture-frozen lesson
     * applied to the mirror direction); the rule semantics stay the
     * wrapper-protocol regression audit — the slot projection must match
     * what the entryReader observed at capture time.</p>
     */
    private void compareDecodeAdmissionMirror(
            List<SlotAudit> slots,
            IdentityHashMap<DecodeEndpoint, DecodeLedgerAuditView> decode,
            List<LedgerDiff> diffs) {
        // slot→layer-1 direction: the authority must find its mirror.
        for (SlotAudit audit : slots) {
            SlotDecodeAdmission authority = audit.decodeAdmission;
            if (authority == null || audit.terminalTrack()) {
                continue;
            }
            DecodeLedgerAuditView view = decode.get(authority.endpoint());
            if (view == null) {
                // The authority's endpoint left the reconciliation
                // surface (generation replacement / retirement): the
                // fence died with the generation — the same legal
                // failover window the placement-direction
                // ENDPOINT_LEFT_RECONCILIATION_SURFACE rule classifies.
                continue;
            }
            RequestInflight mirrorEntry = view.inflight().get(audit.requestId);
            if (mirrorEntry == null) {
                if (view.confirmedReservationTokens()
                        .containsKey(audit.requestId)) {
                    // KV_ALLOCATED handover in flight: the calibrate
                    // in-pass removal already retired the mirror into
                    // the confirmed projection while the slot death
                    // path has not cleared the authority yet — the
                    // DECODE_CONFIRMED_AHEAD_OF_SLOT transient already
                    // classifies that same-tick window.
                    continue;
                }
                diffs.add(new LedgerDiff(
                        Rule.DECODE_ADMISSION_MIRROR_MISMATCH,
                        audit.requestId,
                        "slot decode-admission authority has no layer-1"
                                + " mirror entry under its own endpoint"
                                + " (projection-lag clear tear, or a lost"
                                + " wrapper delivery if it recurs)"));
                continue;
            }
            // Stage-2 T7 S4: the comparison baseline is the frozen
            // entryReader facts, not a live re-read of the entry bits.
            DecodePlacementAuthorityPort.DecodeAdmissionEntry mirror =
                    view.inflightEntryFacts().get(audit.requestId);
            if (mirror.reservationToken()
                    != authority.reservationToken()) {
                diffs.add(new LedgerDiff(
                        Rule.DECODE_ADMISSION_MIRROR_MISMATCH,
                        audit.requestId,
                        "slot decode-admission authority token "
                                + authority.reservationToken()
                                + " != layer-1 mirror token "
                                + mirror.reservationToken()
                                + " (request-id reuse tear across the"
                                + " two-phase flip)"));
                continue;
            }
            if (mirror.masterQueued() != authority.masterQueued()
                    || mirror.dispatchPermitToken()
                            != authority.dispatchPermitToken()
                    || mirror.engineLifecycleOwned()
                            != authority.engineLifecycleOwned()) {
                diffs.add(new LedgerDiff(
                        Rule.DECODE_ADMISSION_MIRROR_MISMATCH,
                        audit.requestId,
                        "slot decode-admission sub-state (queued="
                                + authority.masterQueued()
                                + ", permit="
                                + authority.dispatchPermitToken()
                                + ", lifecycle="
                                + authority.engineLifecycleOwned()
                                + ") != layer-1 mirror (queued="
                                + mirror.masterQueued()
                                + ", permit="
                                + mirror.dispatchPermitToken()
                                + ", lifecycle="
                                + mirror.engineLifecycleOwned()
                                + ") — two-phase flip / post-commit"
                                + " delivery tear if single-pass"));
            }
        }
        // layer-1→slot direction: a bound, fence-matched mirror entry
        // must carry the authority — the wrapper stages the install
        // inside the slot monitor before the admission body commits the
        // entry, so "entry without authority" has no legal interleaving
        // on the wrapper paths.
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
                // Attempt-incoming shadows install their authority by a
                // transaction-after-commit delivery — the short
                // attempt-removal window keeps the S1 exemption.
                if (view.preemptionAttemptIncomingRequestIds()
                        .contains(requestId)) {
                    continue;
                }
                if (audit.placement == null
                        || audit.placement.decodeEndpoint()
                                != entry.getKey()) {
                    // Publication-bind window or a foreign endpoint:
                    // the placement direction rules own those splits.
                    continue;
                }
                if (audit.decodeRow != null) {
                    // B-road handover already happened: the admission
                    // domain ended (the slot death path cleared the
                    // authority with the pRow), so a surviving layer-1
                    // shadow is the INFLIGHT_PROW_CROSSCHECK writer
                    // regression, not a mirror split.
                    continue;
                }
                RequestInflight mirrorEntry = view.inflight().get(requestId);
                if (mirrorEntry == null
                        || mirrorEntry.reservationToken()
                                != audit.placement.decodeReservationToken()) {
                    // Missing entry or request-id reuse window: the
                    // placement still hosts the previous fence — the
                    // placement-direction rules already report those
                    // splits.
                    continue;
                }
                // Stage-2 T7 S4: fence-matched mirror presence check
                // against the frozen entryReader facts (same baseline as
                // the slot→layer-1 direction).
                if (view.inflightEntryFacts().get(requestId) != null
                        && audit.decodeAdmission == null) {
                    diffs.add(new LedgerDiff(
                            Rule.DECODE_ADMISSION_MIRROR_MISMATCH,
                            requestId,
                            "layer-1 mirror entry with a bound,"
                                    + " fence-matched placement carries no"
                                    + " slot decode-admission authority"
                                    + " (the wrapper stages the install"
                                    + " before the admission body"
                                    + " commits)"));
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
                            "decode fence registration table holds a"
                                    + " registration whose slot carries"
                                    + " neither fence nor preemption"
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
            //
            // Stage-2 L7 fix (soak round): the aggregate identity is
            // endpoint-level, so a torn capture (seqlock retries
            // exhausted under admission contention — Phase-1 counters
            // read in a different quiet window than the Phase-2 entry
            // projection) would recur under one fixed DiffKey and climb
            // the confirm window into a false confirmed REAL diff, the
            // exact failure mode request-level rules never see (their
            // request-id rotation resets the window every pass). The
            // cross-phase aggregate rule therefore runs on certified
            // captures only; an uncertified (torn-fallback) capture
            // carries no valid aggregate signal — a persistent split
            // surfaces on the next certified capture instead.
            if (view.certified()) {
                // Stage-2 L7 fix (soak round 2): the projection derives from
                // the capture-frozen queued id set, never from a live re-read
                // of entry.masterQueued(). view.inflight() is a shallow copy
                // holding live RequestInflight references, so re-reading the
                // mutable sub-state flag here would observe post-capture
                // flips: a queued batch accepted between the endpoint capture
                // and this reconciliation pass would read masterQueued=false
                // against the frozen Phase-1 counters and fabricate a
                // "certified" aggregate tear. The capture freezes
                // queuedPhaseRequestIds inside the seqlock window, so the
                // certified flag genuinely covers that projection;
                // kvTokens/expectedKvTokens are immutable entry fields.
                int queuedEntries = 0;
                long queuedHardKv = 0L;
                long queuedExpectedKv = 0L;
                for (Long requestId : view.queuedPhaseRequestIds()) {
                    RequestInflight reservation =
                            view.inflight().get(requestId);
                    if (reservation != null) {
                        queuedEntries++;
                        queuedHardKv += reservation.kvTokens();
                        queuedExpectedKv += reservation.expectedKvTokens();
                    }
                }
                if (view.queuedPhaseCount() != queuedEntries
                        || view.queuedKvReservedTotal() != queuedHardKv
                        || view.queuedExpectedKvReservedTotal()
                                != queuedExpectedKv) {
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
                                    + ", expected=" + queuedExpectedKv
                                    + ")"));
                }
            }
            // Stage-2 L8 retarget: the permit projection derives from the
            // layer-1 entry tokens, so a permit holder outside the inflight
            // set is structurally impossible. The rule is rewritten (same
            // treatment as the L7 aggregate mirror) into two checks:
            // the per-request queued-membership invariant (a permit holder
            // must still sit in the queued sub-state — the writer protocol
            // clears the permit no later than the queued flag under one
            // admission tick) and the certified-gated aggregate mirror
            // (the three O(1) permit counters vs the entry-derived
            // projection). Both sides of the membership check are
            // capture-frozen sets, so a post-capture admission flip can
            // never fabricate a diff here (soak round-2 lesson).
            for (Long requestId : view.engineDispatchPermitRequestIds()) {
                if (!view.queuedPhaseRequestIds().contains(requestId)) {
                    diffs.add(new LedgerDiff(
                            Rule.DISPATCH_PERMIT_OUTSIDE_QUEUED_PHASE,
                            requestId,
                            "decode layer-8 dispatch-permit holder left the"
                                    + " queued layer-1 sub-state"));
                }
            }
            // Stage-2 L8 fix (soak lessons, same as the L7 aggregate): the
            // aggregate identity is endpoint-level (request id 0), so the
            // confirm window cannot absorb a recurring torn-capture drift
            // by request-id rotation — the cross-phase aggregate runs on
            // certified captures only, and the projection side reads the
            // capture-frozen permit id set, never a live re-read of the
            // mutable entry token.
            if (view.certified()) {
                int permitEntries = 0;
                long permitHardKv = 0L;
                long permitExpectedKv = 0L;
                for (Long requestId : view.engineDispatchPermitRequestIds()) {
                    RequestInflight reservation =
                            view.inflight().get(requestId);
                    if (reservation != null) {
                        permitEntries++;
                        permitHardKv += reservation.kvTokens();
                        permitExpectedKv += reservation.expectedKvTokens();
                    }
                }
                if (view.engineDispatchPermitCount() != permitEntries
                        || view.engineDispatchPermitHardKvReservedTotal()
                                != permitHardKv
                        || view.engineDispatchPermitExpectedKvReservedTotal()
                                != permitExpectedKv) {
                    diffs.add(new LedgerDiff(
                            Rule.DISPATCH_PERMIT_OUTSIDE_QUEUED_PHASE,
                            0L,
                            "decode dispatch-permit aggregate mirror drift:"
                                    + " counters (count="
                                    + view.engineDispatchPermitCount()
                                    + ", hard="
                                    + view.engineDispatchPermitHardKvReservedTotal()
                                    + ", expected="
                                    + view.engineDispatchPermitExpectedKvReservedTotal()
                                    + ") != entry-derived projection (count="
                                    + permitEntries + ", hard=" + permitHardKv
                                    + ", expected=" + permitExpectedKv
                                    + ")"));
                }
            }
            // Stage-2 L5 source switch: the fence-held aggregate is a
            // projection recomputed from registry facts on every calibration
            // pass (the synthetic-hold state machine is deleted). The rule
            // mirrors the L7/L8 aggregate treatment with one extra gate: the
            // projection identity is endpoint-level (request id 0), the
            // derivation reads only capture-frozen maps (registration tokens
            // and per-entry KV, claim tokens, the observed-confirmed engine
            // fact — never a live re-read), and — specific to the L5
            // projection semantics — a capture whose admission version still
            // differs from the fence projection stamp was taken inside an
            // out-of-band registry mutation window (lease close,
            // authoritative or claim settlement, full release): those bump
            // the version without recomputing the aggregate, so the capture
            // carries no valid aggregate signal until the next calibration
            // pass re-derives it.
            if (view.certified()
                    && view.fenceProjectionVersion()
                            == view.admissionVersion()) {
                long derivedHeldKv = 0L;
                long derivedHeldExpectedKv = 0L;
                for (java.util.Map.Entry<Long, Long> confirmedToken
                        : view.confirmedReservationTokens().entrySet()) {
                    Long requestId = confirmedToken.getKey();
                    if (view.observedConfirmedRequestIds()
                            .contains(requestId)) {
                        continue;
                    }
                    Long fenceToken = view.engineFenceProtectedReservationTokens()
                            .get(requestId);
                    if (fenceToken == null
                            || !fenceToken.equals(confirmedToken.getValue())) {
                        continue;
                    }
                    Long claimToken = view.preemptionClaimReservationTokens()
                            .get(requestId);
                    if (claimToken != null
                            && claimToken.equals(confirmedToken.getValue())) {
                        continue;
                    }
                    derivedHeldKv += view.engineFenceProtectedHardKvTokens()
                            .getOrDefault(requestId, 0L);
                    derivedHeldExpectedKv +=
                            view.engineFenceProtectedExpectedKvTokens()
                                    .getOrDefault(requestId, 0L);
                }
                if (view.engineFenceHeldKv() != derivedHeldKv
                        || view.engineFenceHeldExpectedKv()
                                != derivedHeldExpectedKv) {
                    diffs.add(new LedgerDiff(
                            Rule.ENGINE_FENCE_AGGREGATE_MISMATCH,
                            0L,
                            "decode fence-held aggregate projection drift:"
                                    + " projection (hard="
                                    + view.engineFenceHeldKv()
                                    + ", expected="
                                    + view.engineFenceHeldExpectedKv()
                                    + ") != registration-derived hold (hard="
                                    + derivedHeldKv + ", expected="
                                    + derivedHeldExpectedKv + ")"));
                }
            }
            // Stage-2 L4 source switch: the priority-held aggregate is a
            // projection recomputed from claim-registry facts on every
            // calibration pass (an ENGINE_CONFIRMED claim whose victim is
            // absent from the fresh observation) — the phase state machine
            // and the incremental hold latch are deleted. The rule mirrors
            // the L5/L7/L8 aggregate treatment with the same four gates:
            // seqlock-certified captures only, the priority projection stamp
            // must equal the captured admission version (an out-of-band
            // registry mutation — settlement, reconciliation, abort — bumps
            // the version without recomputing the aggregate, and that window
            // carries no aggregate signal until the next calibration pass),
            // the derivation reads the capture-frozen claim maps (per-victim
            // KV, the ENGINE_CONFIRMED ownership flag, the observed-confirmed
            // engine fact) rather than live re-reads, and the endpoint-level
            // identity is request id 0 (the confirm window cannot absorb it
            // by request-id rotation).
            if (view.certified()
                    && view.priorityProjectionVersion()
                            == view.admissionVersion()) {
                long derivedPriorityHeldKv = 0L;
                long derivedPriorityHeldExpectedKv = 0L;
                for (Long requestId : view.engineConfirmedClaimRequestIds()) {
                    if (view.observedConfirmedRequestIds()
                            .contains(requestId)) {
                        continue;
                    }
                    derivedPriorityHeldKv += view.preemptionClaimHardKvTokens()
                            .getOrDefault(requestId, 0L);
                    derivedPriorityHeldExpectedKv +=
                            view.preemptionClaimExpectedKvTokens()
                                    .getOrDefault(requestId, 0L);
                }
                if (view.priorityPreemptionHeldKv() != derivedPriorityHeldKv
                        || view.priorityPreemptionHeldExpectedKv()
                                != derivedPriorityHeldExpectedKv) {
                    diffs.add(new LedgerDiff(
                            Rule.PRIORITY_PREEMPTION_AGGREGATE_MISMATCH,
                            0L,
                            "decode priority-held aggregate projection drift:"
                                    + " projection (hard="
                                    + view.priorityPreemptionHeldKv()
                                    + ", expected="
                                    + view.priorityPreemptionHeldExpectedKv()
                                    + ") != claim-derived hold (hard="
                                    + derivedPriorityHeldKv + ", expected="
                                    + derivedPriorityHeldExpectedKv + ")"));
                }
            }
            // Stage-2 L3 source switch: the confirmed-slot aggregate is a
            // projection recomputed from registry facts on every calibration
            // pass (fresh observed confirmed + absent ENGINE_CONFIRMED claim
            // survivors + exact-token fence-held absent survivors, claim
            // precedence) — the four out-of-band decrement sites are
            // deleted. The rule mirrors the L4/L5 aggregate treatment with
            // the same four gates: seqlock-certified captures only (torn
            // fallback captures carry no cross-phase aggregate signal), the
            // confirmed projection stamp must equal the captured admission
            // version (an out-of-band registry mutation — settlement,
            // eviction, full release, retirement — bumps the version without
            // recomputing the aggregate, and that window carries no
            // aggregate signal until the next calibration pass), the
            // derivation reads the capture-frozen confirmed/claim/fence/
            // observed-confirmed maps rather than live re-reads, and the
            // endpoint-level identity is request id 0 (the confirm window
            // cannot absorb it by request-id rotation).
            if (view.certified()
                    && view.confirmedProjectionVersion()
                            == view.admissionVersion()) {
                int derivedConfirmedSlots =
                        view.observedConfirmedRequestIds().size();
                for (Long requestId : view.engineConfirmedClaimRequestIds()) {
                    if (!view.observedConfirmedRequestIds()
                            .contains(requestId)) {
                        derivedConfirmedSlots++;
                    }
                }
                for (java.util.Map.Entry<Long, Long> confirmedToken
                        : view.confirmedReservationTokens().entrySet()) {
                    Long requestId = confirmedToken.getKey();
                    if (view.observedConfirmedRequestIds()
                            .contains(requestId)) {
                        continue;
                    }
                    Long fenceToken = view.engineFenceProtectedReservationTokens()
                            .get(requestId);
                    if (fenceToken == null
                            || !fenceToken.equals(confirmedToken.getValue())) {
                        continue;
                    }
                    Long claimToken = view.preemptionClaimReservationTokens()
                            .get(requestId);
                    if (claimToken != null
                            && claimToken.equals(confirmedToken.getValue())) {
                        continue;
                    }
                    derivedConfirmedSlots++;
                }
                if (view.confirmedRunningCount() != derivedConfirmedSlots) {
                    diffs.add(new LedgerDiff(
                            Rule.CONFIRMED_SLOT_AGGREGATE_MISMATCH,
                            0L,
                            "decode confirmed-slot aggregate projection"
                                    + " drift: projection (count="
                                    + view.confirmedRunningCount()
                                    + ") != registry-derived hold (count="
                                    + derivedConfirmedSlots + ")"));
                }
            }
            // Stage-2 L2 source switch: the engine-lifecycle ownership lives
            // as the entry sub-state flag on each layer-1 inflight entry (the
            // identity-set storage is deleted — count projection, identities
            // stay engine-side). The rule mirrors the L8 treatment: the
            // per-request membership invariant (an engine-lifecycle owner
            // must sit in the inflight layer — every leave path removes the
            // entry itself, so a frozen lifecycle id outside the frozen
            // inflight map is a writer split) checks two capture-frozen
            // sides and needs no certified gate, while the count-vs-id-set
            // aggregate half carries the endpoint-level identity (request
            // id 0) and therefore runs on seqlock-certified captures only.
            for (Long requestId : view.engineLifecycleRequestIds()) {
                if (!view.inflight().containsKey(requestId)) {
                    diffs.add(new LedgerDiff(
                            Rule.ENGINE_LIFECYCLE_SUBSTATE_MISMATCH,
                            requestId,
                            "decode layer-2 engine-lifecycle owner left the"
                                    + " layer-1 inflight layer"));
                }
            }
            if (view.certified()
                    && view.engineLifecycleReservationCount()
                            != view.engineLifecycleRequestIds().size()) {
                diffs.add(new LedgerDiff(
                        Rule.ENGINE_LIFECYCLE_SUBSTATE_MISMATCH,
                        0L,
                        "decode engine-lifecycle aggregate projection drift:"
                                + " count projection ("
                                + view.engineLifecycleReservationCount()
                                + ") != entry-derived ids ("
                                + view.engineLifecycleRequestIds().size()
                                + ")"));
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
         * confirm-window territory).  Stage-2 L5 source switch: the
         * endpoint-side table is a pure registration ledger; the slot-side
         * registration is the stage-3 exemption authority (design §6), and
         * this rule is the slot→engine direction of their bidirectional
         * audit.
         */
        ENGINE_FENCE_UNBACKED(true),
        /**
         * Stage-1 fix C: decode layer-5 fence protection on an ACTIVE
         * slot with neither fence nor preemption registration (no
         * single-interleave legal window).  Stage-2 L5 source switch: the
         * endpoint side is a pure registration table; this rule is the
         * engine→slot direction of the bidirectional audit against the
         * slot-side registration authority.
         */
        DECODE_FENCE_PROTECTION_ORPHAN(true),
        /**
         * Stage-2 L5 source switch: the fence-held accounting is a
         * projection recomputed from registry facts (absent confirmed
         * entry + exact-match registration + no exact priority owner) on
         * every calibration pass — the synthetic-hold state machine is
         * deleted.  The rule cross-checks the two O(1) fence-held
         * aggregate totals against the capture-frozen registration-derived
         * hold (engine-internal invariant; request id 0 marks the
         * endpoint-level aggregate identity).  Four gates make the signal
         * trustworthy: seqlock-certified captures only (torn fallback
         * captures carry no cross-phase aggregate signal), the fence
         * projection stamp must equal the captured admission version (an
         * out-of-band registry mutation — lease close, authoritative or
         * claim settlement, full release — bumps the version without
         * recomputing the aggregate, and that window carries no aggregate
         * signal until the next calibration pass), the derivation reads
         * the capture-frozen registration/claim/observed-confirmed maps
         * rather than live re-reads, and the endpoint-level identity is
         * request id 0 (the confirm window cannot absorb it by request-id
         * rotation).
         */
        ENGINE_FENCE_AGGREGATE_MISMATCH(true),
        /**
         * Stage-2 L4 source switch: the priority-held accounting is a
         * projection recomputed from claim-registry facts (an
         * ENGINE_CONFIRMED claim absent from the fresh observation) on
         * every calibration pass — the phase state machine and the
         * incremental hold latch are deleted.  The rule cross-checks the
         * two O(1) priority-held aggregate totals against the
         * capture-frozen claim-derived hold (engine-internal invariant;
         * request id 0 marks the endpoint-level aggregate identity).
         * Four gates make the signal trustworthy, mirroring the L5
         * aggregate: seqlock-certified captures only (torn fallback
         * captures carry no cross-phase aggregate signal), the priority
         * projection stamp must equal the captured admission version (an
         * out-of-band registry mutation — settlement, reconciliation,
         * abort — bumps the version without recomputing the aggregate,
         * and that window carries no aggregate signal until the next
         * calibration pass), the derivation reads the capture-frozen
         * claim/observed-confirmed maps rather than live re-reads, and the
         * endpoint-level identity is request id 0 (the confirm window
         * cannot absorb it by request-id rotation).
         */
        PRIORITY_PREEMPTION_AGGREGATE_MISMATCH(true),
        /**
         * Stage-1 fix C originally guarded the layer-7 set against members
         * outside the layer-1 inflight set.  Stage-2 L7 retirement rewrote
         * it (not merely retargeted): the queued projection now derives
         * from the layer-1 entry sub-state flags, so membership outside
         * inflight is structurally impossible — the rule instead
         * cross-checks the three O(1) queued aggregate counters against
         * the entry-derived projection (engine-internal invariant;
         * request id 0 marks the endpoint-level aggregate identity).
         * Stage-2 L7 soak fix: because the aggregate identity is fixed
         * (request id 0), the rule runs on seqlock-certified captures
         * only — torn fallback captures would otherwise climb the
         * confirm window into false confirmed REAL diffs.  Soak round 2:
         * the projection side reads the capture-frozen queued id set, not a
         * live re-read of the mutable entry sub-state — the harness runs
         * after the capture, so a live read would report post-capture
         * admission flips as "certified" aggregate tears.
         */
        QUEUED_PHASE_OUTSIDE_INFLIGHT(true),
        /**
         * Stage-1 fix C originally guarded layer-8 permit holders against
         * sitting outside the queued layer-1 inflight set.  Stage-2 L8
         * retirement rewrote it (same treatment as the L7 aggregate
         * mirror): the permit projection derives from the layer-1 entry
         * tokens, so membership outside inflight is structurally
         * impossible — the rule now cross-checks the per-request queued
         * membership (capture-frozen sets on both sides) plus the three
         * O(1) permit aggregate counters against the entry-derived
         * projection (engine-internal invariant; the aggregate half runs
         * on seqlock-certified captures only and reports under the
         * endpoint-level aggregate identity, request id 0).
         */
        DISPATCH_PERMIT_OUTSIDE_QUEUED_PHASE(true),
        /**
         * Stage-2 L3 source switch: the confirmed-slot counter is a
         * projection recomputed from registry facts (fresh observed
         * confirmed + absent ENGINE_CONFIRMED claim survivors +
         * exact-token fence-held absent survivors, claim precedence) on
         * every calibration pass — the four out-of-band decrement sites
         * are deleted.  The rule cross-checks the O(1) confirmed-slot
         * projection against the capture-frozen registry-derived hold
         * (engine-internal invariant; request id 0 marks the
         * endpoint-level aggregate identity).  Four gates make the signal
         * trustworthy, mirroring the L5/L4 aggregates: seqlock-certified
         * captures only (torn fallback captures carry no cross-phase
         * aggregate signal), the confirmed projection stamp must equal
         * the captured admission version (an out-of-band registry
         * mutation — settlement, eviction, full release, retirement —
         * bumps the version without recomputing the projection, and that
         * window carries no aggregate signal until the next calibration
         * pass), the derivation reads the capture-frozen
         * confirmed/claim/fence/observed-confirmed maps rather than live
         * re-reads, and the endpoint-level identity is request id 0 (the
         * confirm window cannot absorb it by request-id rotation).
         */
        CONFIRMED_SLOT_AGGREGATE_MISMATCH(true),
        /**
         * Stage-2 L2 source switch: the engine-lifecycle ownership lives
         * as the entry-level sub-state flag on each layer-1 inflight
         * entry — the identity-set storage is deleted (count projection,
         * identities stay engine-side).  The rule mirrors the L8
         * treatment: the per-request membership invariant (an
         * engine-lifecycle owner must sit in the layer-1 inflight layer —
         * every leave path removes the entry itself, so the flag
         * disappears with it) checks two capture-frozen sides and needs
         * no certified gate, while the count-vs-id-set aggregate half
         * carries the endpoint-level identity (request id 0) and
         * therefore runs on seqlock-certified captures only (torn
         * fallback captures carry no cross-phase aggregate signal).
         */
        ENGINE_LIFECYCLE_SUBSTATE_MISMATCH(true),
        /**
         * Stage-2 T7 (S1 pre-embed): a layer-1 inflight shadow whose ACTIVE
         * slot is already bound to the same Decode endpoint carries no slot
         * pRow numeric mirror.  The publication bind installs the placement
         * and the pRow inside one slot-monitor tick, so this state has no
         * legal interleaving — the rule is the writer-protocol regression
         * guard backing the S2 read-source switch (the L1 numeric queries
         * move onto the pRow) and the S4 layer-1 retirement (where the
         * rule flips to asserting the layer is empty).  Numeric-value drift
         * between an installed pRow and its layer-1 shadow stays with the
         * slot-direction {@link #KV_MIRROR_MISMATCH} rule; the
         * protocol-committed incoming shadow keeps its short
         * attempt-removal exemption (attempt id set), and terminal-track /
         * no-current-slot entries keep the shared skip semantics of the
         * engine→slot direction rules.
         */
        INFLIGHT_PROW_CROSSCHECK(true),
        /**
         * Stage-2 T7 S3 (placement-domain migration): the decode
         * admission sub-state authority lives on the slot (fence triple
         * + preloaded numeric row + the three layer-1 sub-state bits),
         * and the layer-1 entry flags stay mirrors until the S2
         * read-source switch.  The rule is the bidirectional mirror
         * audit backing that migration:
         *
         * <ul>
         * <li>slot→layer-1: an ACTIVE slot's authority must find its
         * layer-1 mirror entry under the authority's own endpoint, with
         * the same fence token and the same three sub-state bits.  The
         * two-phase flip (slot monitor first, admissionLock second) and
         * the projection-lag clears (post-commit deliveries run outside
         * both locks) each open single-interleave tears the confirm
         * window absorbs; a persistent split is a lost wrapper
         * delivery.</li>
         * <li>layer-1→slot: a mirror entry whose ACTIVE slot already
         * carries a bound placement on this endpoint with the matching
         * reservation token must carry the slot authority — the wrapper
         * stages the install inside the slot monitor before the
         * admission body commits the entry, so "entry without
         * authority" has no legal interleaving on the wrapper paths.
         * The attempt-incoming shadows keep their post-commit delivery
         * exemption (transaction-after-commit install), and bind-window
         * / terminal-track / no-slot entries keep the shared skip
         * semantics of the engine→slot direction rules.</li>
         * </ul>
         */
        DECODE_ADMISSION_MIRROR_MISMATCH(true),
        /**
         * Stage-2 T7 S2b (channel A) dual-write rule: the delivered
         * placement-projection row must equal the nine native O(1)
         * aggregate mirrors (inflight / queued / permit × count / hard
         * KV / expected KV) on seqlock-certified captures. Endpoint-level
         * aggregate identity (request id 0); the µs delivery window
         * surfaces as a single-pass tear the confirm window absorbs,
         * and a persistent split is a lost or duplicated fact delivery —
         * the exact regression class the S2c read-source switch (native
         * maintenance retirement) depends on this rule having excluded.
         * Rows with delivery stamp zero (port-less test endpoints, never
         * delivered) carry no signal and are skipped.
         */
        PLACEMENT_PROJECTION_MISMATCH(true),
        /** Engine confirmed before the slot applied the handover. */
        DECODE_CONFIRMED_AHEAD_OF_SLOT(false),
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
            boolean engineFenceInstalled,
            SlotDecodeAdmission decodeAdmission) {

        /** Fully derived tombstone audit for the lock-free fast path. */
        static SlotAudit tombstone(long requestId) {
            return new SlotAudit(
                    requestId,
                    SlotPhaseAdjudicator.CoarsePhase.TOMBSTONE,
                    null, null, null, null, null, false, null);
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
