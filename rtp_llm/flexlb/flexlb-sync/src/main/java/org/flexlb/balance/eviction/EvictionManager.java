package org.flexlb.balance.eviction;

import org.flexlb.balance.admission.AdmissionFallback;
import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.admission.AdmissionLifecyclePort;
import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.eviction.EvictionPlacementPort.DecodePlacement;
import org.flexlb.balance.eviction.EvictionPlacementPort.PreparedDecodePlacement;
import org.flexlb.balance.eviction.EvictionPlacementPort.PrefillEvictionAdmission;
import org.flexlb.balance.eviction.EvictionPlacementPort.PrefillEvictionCommit;
import org.flexlb.balance.eviction.EvictionPlacementPort.PrefillEvictionStatus;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime.QueueSnapshot;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;

import javax.annotation.PreDestroy;

/**
 * Eviction-only admission component. It neither selects the scheduler mode nor
 * commits an ordinary placement. A successful takeover owns one exact
 * admission permit, one immutable eviction plan, and the request future until
 * ACTIVE publication or a typed terminal outcome.
 */
@Component
public class EvictionManager implements AdmissionFallback {

    /** Sentinel stored in {@link #activeAdmissionCount} after shutdown. */
    private static final int ADMISSION_CLOSED = -1;

    private final EndpointRegistry endpointRegistry;
    private final RequestSchedulerReporter reporter;
    private final EngineCancelChannel cancelChannel;
    private final DecodePreemptionCoordinator preemptionCoordinator;
    private final AdmissionLifecyclePort admissionLifecycle;
    private final VictimLifecyclePort victimLifecycle;
    private final EvictionPlacementPort placementPort;

    private volatile boolean shutdown;

    /**
     * Number of admission permits currently held. A permit is reserved before
     * placement starts and remains charged until admission fails or its
     * the canonical request slot releases it. The reservation CAS is the hard-limit
     * linearization point for the delivered-not-accepted lifecycle limit.
     */
    private final AtomicInteger activeAdmissionCount = new AtomicInteger(0);

    int activeAdmissionCount() {
        return Math.max(0, activeAdmissionCount.get());
    }

    private AdmissionPermit tryReserveAdmissionPermit(int limit) {
        AdmissionPermit permit = new AdmissionPermit(activeAdmissionCount);
        while (true) {
            int activePermits = activeAdmissionCount.get();
            if (activePermits == ADMISSION_CLOSED
                    || (limit > 0 && activePermits >= limit)
                    || activePermits == Integer.MAX_VALUE) {
                return null;
            }
            if (activeAdmissionCount.compareAndSet(activePermits, activePermits + 1)) {
                return permit;
            }
        }
    }

    /**
     * One allocation per admission attempt and no per-release allocation.
     * The field updater makes failure completion and slot termination share
     * one exact-once release without adding a lock or a nested AtomicBoolean.
     */
    private static final class AdmissionPermit implements Runnable {

        private static final AtomicIntegerFieldUpdater<AdmissionPermit> RELEASED =
                AtomicIntegerFieldUpdater.newUpdater(AdmissionPermit.class, "released");

        private final AtomicInteger activeCount;
        @SuppressWarnings("unused") // accessed by RELEASED
        private volatile int released;

        private AdmissionPermit(AtomicInteger activeCount) {
            this.activeCount = activeCount;
        }

        @Override
        public void run() {
            release();
        }

        private void release() {
            if (RELEASED.compareAndSet(this, 0, 1)) {
                while (true) {
                    int count = activeCount.get();
                    if (count == ADMISSION_CLOSED) {
                        return;
                    }
                    if (activeCount.compareAndSet(count, count - 1)) {
                        return;
                    }
                }
            }
        }
    }

    @Autowired
    public EvictionManager(EndpointRegistry endpointRegistry,
                           RequestSchedulerReporter reporter,
                           EngineCancelChannel cancelChannel,
                           DecodePreemptionCoordinator preemptionCoordinator,
                           AdmissionLifecyclePort admissionLifecycle,
                           VictimLifecyclePort victimLifecycle,
                           EvictionPlacementPort placementPort) {
        this.endpointRegistry = endpointRegistry;
        this.reporter = reporter;
        this.cancelChannel = cancelChannel;
        this.preemptionCoordinator = preemptionCoordinator;
        this.admissionLifecycle = Objects.requireNonNull(
                admissionLifecycle, "admissionLifecycle");
        this.victimLifecycle = Objects.requireNonNull(
                victimLifecycle, "victimLifecycle");
        this.placementPort = Objects.requireNonNull(
                placementPort, "placementPort");
    }

    @PreDestroy
    public void shutdown() {
        shutdown = true;
        // Canonical request slots own all live permits. Closing this CAS gate
        // prevents new reservations; exact slot cleanup retires existing ones.
        activeAdmissionCount.set(ADMISSION_CLOSED);
    }

    boolean isShutdown() {
        return shutdown;
    }

    /**
     * Attempt one eviction-backed admission after ordinary placement has
     * failed. A true result transfers the future to this manager; false leaves
     * the caller free to publish its original rejection.
     */
    @Override
    public boolean tryAdmit(BalanceContext ctx,
                            CompletableFuture<Response> future) {
        if (shutdown || future.isDone()
                || ctx.requestExpired(System.currentTimeMillis())
                || !PriorityNormalizer.hasPriority(ctx.getPriority())) {
            return false;
        }

        FlexlbConfig config = ctx.getConfig();
        if (!config.isQueue()) {
            return false;
        }
        PreemptionConfig preemption = config.queueScheduler().getOrdering()
                .preemptionPolicy().orElse(null);
        if (preemption == null) {
            return false;
        }
        reportCapacityGauges();

        if (preemption.allows(VictimStage.PREFILL_QUEUED)) {
            PrefillAdmissionOutcome prefill =
                    tryAdmitByPrefillEviction(ctx, future, config);
            if (prefill == PrefillAdmissionOutcome.TAKEN_OVER) {
                return true;
            }
            // A fresh route proved ordinary Decode capacity exists. If the
            // selected Prefill queue has no legal victims, Decode eviction is
            // unrelated destructive work.
            if (prefill == PrefillAdmissionOutcome.DECLINED_NO_EVICTION) {
                return false;
            }
        }

        if (!preemption.allows(VictimStage.DECODE_RESERVED)
                && !preemption.allows(VictimStage.DECODE_ENGINE_OWNED)) {
            return false;
        }
        PlannedDecodeEviction planned =
                planDecodeEviction(ctx, config, preemption);
        if (planned == null) {
            return false;
        }
        return commitDecodeEviction(
                ctx, future, planned, config, preemption);
    }

    private enum PrefillAdmissionOutcome {
        NO_CANONICAL_ROUTE,
        DECLINED_NO_EVICTION,
        TAKEN_OVER
    }

    /**
     * Route only far enough to capture an exact Prefill replacement
     * capability. A route whose queue does not require eviction is closed and
     * returned to the ordinary scheduler as a non-takeover.
     */
    private PrefillAdmissionOutcome tryAdmitByPrefillEviction(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            FlexlbConfig config) {
        AdmissionMutation mutation =
                admissionLifecycle.claimAdmissionMutation(
                        ctx.getRequestId(), future);
        if (mutation == null) {
            return PrefillAdmissionOutcome.DECLINED_NO_EVICTION;
        }
        try (mutation;
             PrefillEvictionAdmission admission =
                     placementPort.preparePrefillEviction(ctx, future)) {
            if (admission == null) {
                return PrefillAdmissionOutcome.NO_CANONICAL_ROUTE;
            }
            PriorityRequestEnvelope envelope = admission.envelope();
            QueueSnapshot snapshot = admission.queueSnapshot();
            PrefillEvictionProposal proposal =
                    planPrefillEviction(envelope, snapshot);
            if (proposal == null) {
                return PrefillAdmissionOutcome.DECLINED_NO_EVICTION;
            }
            AdmissionPermit permit = reserveAdmissionPermit(ctx, config);
            if (permit == null) {
                return PrefillAdmissionOutcome.DECLINED_NO_EVICTION;
            }
            PrefillEvictionCommit commit;
            try {
                if (!mutation.seal()) {
                    return PrefillAdmissionOutcome.DECLINED_NO_EVICTION;
                }
                commit = admission.commit(proposal.victims(), mutation);
                if (commit.status() != PrefillEvictionStatus.COMMITTED) {
                    reportEvictionCommit(
                            envelope.priority(),
                            envelope.requestId(),
                            "prefill_queue_full",
                            commit.status().name().toLowerCase());
                    return PrefillAdmissionOutcome.DECLINED_NO_EVICTION;
                }
                if (!bindAdmissionPermit(
                        ctx, future, mutation, permit, config)) {
                    finishCommittedPrefillEviction(
                            ctx, envelope, proposal, commit);
                    mutation.terminate(admissionError(
                            StrategyErrorType.RESOURCE_EXHAUSTED,
                            AdmissionRejectReason.RESOURCE_EXHAUSTED,
                            "Prefill admission ownership disappeared after commit"));
                    return PrefillAdmissionOutcome.TAKEN_OVER;
                }
                permit = null;
                finishCommittedPrefillEviction(
                        ctx, envelope, proposal, commit);
                return PrefillAdmissionOutcome.TAKEN_OVER;
            } finally {
                if (permit != null) {
                    permit.release();
                }
            }
        }
    }

    private AdmissionPermit reserveAdmissionPermit(
            BalanceContext ctx,
            FlexlbConfig config) {
        int limit = config.queueScheduler().getLifecycle()
                .getMaxDeliveredNotAcceptedRequestsGlobal();
        AdmissionPermit permit = tryReserveAdmissionPermit(limit);
        if (permit == null) {
            Logger.debug(
                    "[eviction-manager] admission permit unavailable:"
                            + " request_id={} active={} limit={}",
                    ctx.getRequestId(), activeAdmissionCount(), limit);
            return null;
        }
        return permit;
    }

    private boolean bindAdmissionPermit(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            AdmissionMutation mutation,
            AdmissionPermit permit,
            FlexlbConfig config) {
        long timeoutMs = config.queueScheduler().getLifecycle()
                .getDeliveredNotAcceptedTimeoutMs();
        if (!admissionLifecycle.bindAdmissionResources(
                ctx.getRequestId(), future, mutation, permit, timeoutMs)) {
            permit.release();
            return false;
        }
        return true;
    }

    // ==================== Phase 3: prefill queue eviction ====================

    private PrefillEvictionProposal planPrefillEviction(
            PriorityRequestEnvelope envelope,
            QueueSnapshot queueSnapshot) {
        Map<String, String> failures = new HashMap<>();
        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope, List.of(queueSnapshot), failures);
        if (proposal == null) {
            reportEvictionPlan(envelope.priority(), envelope.requestId(),
                    "prefill_queue_full", "infeasible");
            Logger.debug("[eviction-manager] eviction plan infeasible, request_id={} priority={} "
                            + "phase=prefill_queue candidates_seen={} reasons={}",
                    envelope.requestId(), envelope.priority(),
                    queueSnapshot.items().size(), failures);
            return null;
        }
        reportEvictionPlan(envelope.priority(), envelope.requestId(),
                "prefill_queue_full", "feasible");
        return proposal;
    }

    /** Settle victims only after replacement PNR and exact resource binding. */
    private void finishCommittedPrefillEviction(
            BalanceContext ctx,
            PriorityRequestEnvelope envelope,
            PrefillEvictionProposal proposal,
            PrefillEvictionCommit committed) {
        for (DeliveryItem victim : committed.removed()) {
            settlePrefillVictim(envelope, victim, proposal.endpointId());
        }

        reportEvictionCommit(envelope.priority(), envelope.requestId(),
                "prefill_queue_full", "success");
        if (!"decode_evict".equals(ctx.getPlanType())) {
            ctx.setPlanType("prefill_evict");
        }
        ctx.setPlanCost(ctx.getPlanCost() + proposal.rawCost());
        ctx.setVictimCount(ctx.getVictimCount() + proposal.victims().size());
        Logger.debug(
                "[eviction-manager] eviction committed: request_id={} priority={} victims={} "
                        + "raw_cost={} worker={}",
                envelope.requestId(),
                envelope.priority(),
                proposal.victims().size(),
                proposal.rawCost(),
                proposal.endpointId());
    }

    /**
     * Settle one victim removed by a committed queue replacement. The exact
     * reducer is total for legal local ownership; an exception is an invariant
     * violation and aborts the incoming handoff instead of starting a retry
     * chain or publishing past incomplete cleanup.
     */
    private void settlePrefillVictim(PriorityRequestEnvelope incoming,
                                     DeliveryItem victim,
                                     String endpointId) {
        String detail = "yielded to higher-priority request " + incoming.requestId();
        victimLifecycle.finishYielded(victim, detail);
        try {
            reporter.reportVictim(victim.priority(), incoming.priority(),
                    "prefill_queued", "prefill_queue_full");
            reporter.reportPriorityPreempt("prefill_queued");
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report prefill victim settlement: "
                            + "victim_id={} incoming_id={} worker={}",
                    victim.requestId(), incoming.requestId(), endpointId,
                    telemetryFailure);
        }
        Logger.debug("[eviction-manager] victim preempted: victim_id={} victim_priority={} "
                        + "terminal=yielded_8400 incoming_id={} incoming_priority={} worker={}",
                victim.requestId(), victim.priority(), incoming.requestId(),
                incoming.priority(), endpointId);
    }

    /** Eviction metrics are observers; they never own a committed transaction. */
    private void reportEvictionCommit(int priority,
                                      long requestId,
                                      String evictionCase,
                                      String outcome) {
        try {
            reporter.reportEvictionCommit(
                    priority, evictionCase, outcome);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report eviction commit: "
                            + "request_id={} case={} outcome={}",
                    requestId, evictionCase, outcome,
                    telemetryFailure);
        }
    }

    private void reportEvictionPlan(int priority,
                                    long requestId,
                                    String evictionCase,
                                    String outcome) {
        try {
            reporter.reportEvictionPlan(priority, evictionCase, outcome);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report eviction plan: "
                            + "request_id={} case={} outcome={}",
                    requestId, evictionCase, outcome, telemetryFailure);
        }
    }

    /** Capacity telemetry is an internal observation, never a second facade. */
    private void reportCapacityGauges() {
        try {
            endpointRegistry.snapshotPrefillEndpoints().forEach((key, ep) ->
                    reporter.reportPrefillQueueDepth(
                            key, ep.queuedRequestCount()));
            endpointRegistry.snapshotDecodeEndpoints().forEach((key, ep) -> {
                reporter.reportDecodeReservedCount(key, ep.getInflightCount());
                reporter.reportDecodeShadowKvReserved(
                        key, ep.inflightHardKvReserved());
                reporter.reportDecodeRunningCount(
                        key, ep.getRunningLayerCount());
                reporter.reportDecodeAcceptedCount(
                        key, ep.getAcceptedLayerCount());
                reporter.reportDecodeEngineLoad(key, ep.getEngineLoad());
            });
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report capacity gauges",
                    telemetryFailure);
        }
    }

    // ==================== Decode eviction ====================

    private record PlannedDecodeEviction(
            DecodeClusterSnapshot snapshot,
            DecodeEvictionProposal proposal) {
    }

    /** Build one side-effect-free plan from one exact cluster snapshot. */
    private PlannedDecodeEviction planDecodeEviction(
            BalanceContext ctx,
            FlexlbConfig config,
            PreemptionConfig preemption) {
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();
        PriorityRequestEnvelope envelope = new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), seqLen,
                config.decodeKvReservationTokens(seqLen, maxNewTokens, 0L));
        DecodeClusterSnapshot snapshot =
                DecodeClusterSnapshot.capture(endpointRegistry, config);
        List<DecodeEndpointSnapshot> decodes =
                new ArrayList<>(snapshot.decodes().values());

        // Eviction is never a substitute for an ordinary available Decode
        // endpoint. This also prevents a Prefill-only route failure from
        // destructively entering the Decode control plane.
        for (DecodeEndpointSnapshot endpoint : decodes) {
            if (!endpoint.endpoint().isRetired()
                    && EvictionPlanner.decodeEvictionCase(envelope, endpoint) == null) {
                return null;
            }
        }

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                envelope, decodes, preemption, cancelChannel, failures);
        if (proposal == null) {
            reportEvictionPlan(ctx.getPriority(), ctx.getRequestId(),
                    infeasibleDecodeCase(envelope, decodes), "infeasible");
            Logger.debug(
                    "[eviction-manager] Decode eviction infeasible:"
                            + " request_id={} priority={} candidates={} reasons={}",
                    ctx.getRequestId(), ctx.getPriority(), decodes.size(), failures);
            return null;
        }
        reportEvictionPlan(ctx.getPriority(), ctx.getRequestId(),
                proposal.evictionCase(), "feasible");
        return new PlannedDecodeEviction(snapshot, proposal);
    }

    /** Commit exactly the immutable plan selected before takeover. */
    private boolean commitDecodeEviction(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            PlannedDecodeEviction planned,
            FlexlbConfig config,
            PreemptionConfig preemption) {
        DecodeClusterSnapshot snapshot = planned.snapshot();
        DecodeEvictionProposal proposal = planned.proposal();
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();

        DecodeEndpointSnapshot target = snapshot.decodes().get(proposal.endpointId());
        DecodeEndpoint decodeEp = target.endpoint();
        long expectedKvTokens = config.decodeKvReservationTokens(
                seqLen, maxNewTokens, target.realKvTotal());
        DecodeEndpoint.AdmissionCapacity capacity =
                decodeAdmissionCapacity(config, target.concurrencyLimit());

        AdmissionMutation mutation =
                admissionLifecycle.claimAdmissionMutation(
                        ctx.getRequestId(), future);
        if (mutation == null) {
            return false;
        }
        AdmissionPermit permit = reserveAdmissionPermit(ctx, config);
        if (permit == null) {
            mutation.close();
            return false;
        }
        PreparedDecodePlacement prepared = null;
        DecodeEndpoint.ReservationHandle untransferredIncoming = null;
        try {
            prepared = placementPort.prepareDecodePlacement(
                    ctx, future, decodeEp);
            if (prepared == null) {
                return false;
            }

            // Ownership is homogeneous by planner invariant: Master-queued
            // victims use one local CAS; Engine-owned victims use Cancel.
            if (proposal.requiresEngineCancel()) {
                startEngineCancelPreemption(
                        ctx, future, preemption, proposal,
                        decodeEp, seqLen, expectedKvTokens, capacity,
                        mutation, permit, prepared, config);
                mutation = null;
                permit = null;
                prepared = null;
                return true;
            }
            if (!mutation.seal() || !prepared.seal()) {
                return false;
            }

            List<DecodeEndpoint.ReservationHandle> reservedVictims =
                    new ArrayList<>(proposal.victims().size());
            for (DecodeRequestSnapshot victim : proposal.victims()) {
                reservedVictims.add(new DecodeEndpoint.ReservationHandle(
                        decodeEp.getStatus().getGenerationId(),
                        victim.requestId(),
                        victim.reservationToken()));
            }
            DecodeEndpoint.LocalEvictionCommit eviction =
                    decodeEp.tryEvictLocalReservationsAndReserveIncoming(
                            reservedVictims,
                            ctx.getRequestId(),
                            seqLen,
                            expectedKvTokens,
                            ctx.getPriority(),
                            capacity);
            if (eviction.status()
                    != DecodeEndpoint.LocalEvictionResult.COMMITTED) {
                reportEvictionCommit(ctx.getPriority(), ctx.getRequestId(),
                        proposal.evictionCase(), "conflict");
                Logger.debug(
                        "[eviction-manager] Decode eviction conflict: request_id={} "
                                + "planned={} worker={} result={}",
                        ctx.getRequestId(),
                        reservedVictims.size(),
                        proposal.endpointId(),
                        eviction.status());
                return false;
            }
            untransferredIncoming = eviction.incoming();

            if (!bindAdmissionPermit(
                    ctx, future, mutation, permit, config)) {
                finishCommittedLocalDecodeEviction(ctx, proposal);
                mutation.terminate(admissionError(
                        StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED,
                        "Decode admission ownership disappeared after commit"));
                return true;
            }
            permit = null;

            // Transfer exact incoming ownership to canonical publication before
            // any victim callback can fail or re-enter scheduler code.
            DecodeEndpoint.ReservationHandle incoming = untransferredIncoming;
            DecodePlacement placement;
            try {
                placement = prepared.commit(incoming, mutation);
            } catch (RuntimeException placementFailure) {
                Logger.error(
                        "[eviction-manager] local Decode placement failed:"
                                + " request_id={} error={}",
                        ctx.getRequestId(), placementFailure.getMessage(),
                        placementFailure);
                placement = null;
            }
            if (placement instanceof DecodePlacement.Committed) {
                untransferredIncoming = null;
            }

            // Shadow accounting already reversed atomically. Reserved-only
            // victims were never seen by the engine — retryable 8400.
            finishCommittedLocalDecodeEviction(ctx, proposal);
            if (placement instanceof DecodePlacement.Failed failed) {
                AdmissionFailure failure = failed.failure();
                mutation.terminate(admissionError(
                        failure.errorType(), failure.reason(), failure.message()));
                return true;
            }
            if (placement == null) {
                mutation.terminate(admissionError(
                        StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED,
                        "canonical Decode placement returned no ownership result"));
                return true;
            }
            return true;
        } finally {
            if (untransferredIncoming != null) {
                decodeEp.rollbackExact(untransferredIncoming);
            }
            if (prepared != null) {
                prepared.close();
            }
            if (permit != null) {
                permit.release();
            }
            if (mutation != null) {
                mutation.close();
            }
        }
    }

    /**
     * Drive one decode eviction victim to its terminal state and emit the
     * per-victim metrics ({@code stage} distinguishes reserved vs accepted
     * victims). Terminal split per contract 5.3: a reserved-only victim was
     * never seen by the engine — retryable NO_AVAILABLE_WORKER (yielded);
     * an engine-accepted victim keeps PRIORITY_PREEMPTED.
     */
    private void finishDecodeVictim(BalanceContext ctx,
                                    DecodeRequestSnapshot victim, String stage,
                                    DecodeEvictionProposal proposal) {
        boolean accepted = victim.phase().isEngineConfirmed();
        if (accepted || victim.reservationToken() <= 0L) {
            throw new IllegalStateException(
                    "local Decode eviction requires an exact reserved victim: request_id="
                            + victim.requestId());
        }
        String terminal = accepted ? "preempted_8429" : "yielded_8400";
        String detail = accepted
                ? "preempted by higher-priority request " + ctx.getRequestId()
                : "yielded to higher-priority request " + ctx.getRequestId();
        victimLifecycle.finishYieldedReservation(
                victim.requestId(), victim.reservationToken(), detail);
        try {
            reporter.reportVictim(victim.priority(), ctx.getPriority(),
                    stage, proposal.evictionCase());
            reporter.reportPriorityPreempt(stage);
            reporter.reportVictimKvTokens(
                    victim.priority(), stage, victim.kvTokens());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report decode victim settlement: "
                            + "victim_id={} incoming_id={}",
                    victim.requestId(), ctx.getRequestId(), telemetryFailure);
        }
        Logger.debug(
                "[eviction-manager] decode victim preempted: victim_id={} victim_priority={}"
                    + " stage={} terminal={} kv_tokens={} incoming_id={} incoming_priority={}"
                    + " worker={}",
                victim.requestId(),
                victim.priority(),
                stage,
                terminal,
                victim.kvTokens(),
                ctx.getRequestId(),
                ctx.getPriority(),
                proposal.endpointId());
    }

    private void finishCommittedLocalDecodeEviction(
            BalanceContext ctx,
            DecodeEvictionProposal proposal) {
        for (DecodeRequestSnapshot victim : proposal.victims()) {
            finishDecodeVictim(
                    ctx, victim, "decode_reserved", proposal);
        }
        reportCommittedLocalDecodeEviction(ctx, proposal);
        recordDecodePlanObservability(ctx, proposal);
    }

    /** §19.1 plan observability for the decode eviction path. */
    private static void recordDecodePlanObservability(BalanceContext ctx,
                                                      DecodeEvictionProposal proposal) {
        ctx.setPlanType("decode_evict");
        ctx.setPlanCost(proposal.totalCost());
        ctx.setVictimCount(proposal.victims().size());
        Logger.debug(
                "[eviction-manager] decode eviction committed: request_id={} priority={} case={} "
                        + "victims={} total_cost={} freed_kv={} worker={}",
                ctx.getRequestId(),
                ctx.getPriority(),
                proposal.evictionCase(),
                proposal.victims().size(),
                proposal.totalCost(),
                proposal.freedKvTokens(),
                proposal.endpointId());
    }

    private void startEngineCancelPreemption(BalanceContext ctx,
                                             CompletableFuture<Response> future,
                                             PreemptionConfig preemption,
                                             DecodeEvictionProposal proposal,
                                             DecodeEndpoint decodeEp,
                                             long seqLen,
                                             long expectedKvTokens,
                                             DecodeEndpoint.AdmissionCapacity capacity,
                                             AdmissionMutation mutation,
                                             AdmissionPermit permit,
                                             PreparedDecodePlacement prepared,
                                             FlexlbConfig config) {
        String detail = "preempted by higher-priority request " + ctx.getRequestId();
        EngineCancellationConfig cancellation = requiredEngineCancellation(preemption);
        DecodePreemptionCoordinator.Request request =
                new DecodePreemptionCoordinator.Request(
                        decodeEp,
                        ctx.getRequestId(), seqLen, expectedKvTokens,
                        ctx.getPriority(),
                        capacity,
                        proposal.victims(), cancellation.getAckTimeoutMs(),
                        cancellation.getCompletionTimeoutMs(),
                        mutation, prepared::seal, detail);

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> execution;
        reportCancelRequests(ctx, proposal);
        // execute() claims every victim and starts every Cancel before return.
        execution = preemptionCoordinator.execute(request);
        if (execution == null) {
            execution = CompletableFuture.completedFuture(null);
        }

        execution.whenComplete(
                (result, error) -> finishEngineCancelPreemption(
                        ctx, future, proposal, decodeEp,
                        result, error, mutation, permit, prepared, config));
    }

    /** Complete one asynchronous Cancel attempt without terminalizing control pressure. */
    private void finishEngineCancelPreemption(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            DecodeEvictionProposal proposal,
            DecodeEndpoint decodeEp,
            DecodePreemptionCoordinator.ExecutionResult result,
            Throwable error,
            AdmissionMutation mutation,
            AdmissionPermit permit,
            PreparedDecodePlacement prepared,
            FlexlbConfig config) {
        boolean decodeCommitted = result != null
                && result.code()
                        == DecodePreemptionCoordinator.ResultCode.COMMITTED;
        boolean permitBound = false;
        DecodeEndpoint.ReservationHandle untransferredIncoming =
                decodeCommitted ? result.incoming() : null;
        try {
            if (!decodeCommitted) {
                if (error != null || result == null
                        || result.code()
                                == DecodePreemptionCoordinator.ResultCode.CONTROL_FAILED) {
                    reportCancelTimeout(ctx, proposal.endpointId());
                }
                Logger.debug(
                        "[eviction-manager] Decode control attempt deferred:"
                                + " request_id={} worker={} result={} error={}",
                        ctx.getRequestId(),
                        proposal.endpointId(),
                        result == null ? "missing" : result.code(),
                        error == null ? "none" : error.getMessage());
                return;
            }

            permitBound = bindAdmissionPermit(
                    ctx, future, mutation, permit, config);
            if (!permitBound) {
                mutation.terminate(admissionError(
                        StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED,
                        "Decode admission ownership disappeared after commit"));
                return;
            }
            DecodeEndpoint.ReservationHandle incoming = untransferredIncoming;
            DecodePlacement placement = prepared.commit(incoming, mutation);
            if (placement instanceof DecodePlacement.Committed) {
                untransferredIncoming = null;
            }
            if (placement instanceof DecodePlacement.Failed failed) {
                AdmissionFailure failure = failed.failure();
                mutation.terminate(admissionError(
                        failure.errorType(), failure.reason(), failure.message()));
                return;
            }
            if (placement == null) {
                mutation.terminate(admissionError(
                        StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED,
                        "canonical Decode placement returned no ownership result"));
                return;
            }
            reportCommittedEnginePreemption(ctx, proposal);
            recordDecodePlanObservability(ctx, proposal);
        } catch (RuntimeException | Error callbackError) {
            Logger.error(
                    "[eviction-manager] cancel completion failed:"
                            + " request_id={} error={}",
                    ctx.getRequestId(), callbackError.getMessage(),
                    callbackError);
            if (decodeCommitted) {
                mutation.terminate(admissionError(
                        StrategyErrorType.ADMISSION_UNAVAILABLE,
                        AdmissionRejectReason.UNSPECIFIED,
                        "Decode eviction placement failed: "
                                + callbackError.getMessage()));
            }
        } finally {
            if (untransferredIncoming != null) {
                decodeEp.rollbackExact(untransferredIncoming);
            }
            prepared.close();
            if (!permitBound) {
                permit.release();
            }
            mutation.close();
        }
    }

    /** Metrics never participate in the committed reservation handoff. */
    private void reportCommittedEnginePreemption(
            BalanceContext ctx, DecodeEvictionProposal proposal) {
        try {
            for (DecodeRequestSnapshot victim : proposal.victims()) {
                String stage = victim.phase() == DecodeTaskPhase.RUNNING
                        ? "decode_running" : "decode_cancel";
                reporter.reportVictim(victim.priority(), ctx.getPriority(),
                        stage, proposal.evictionCase());
                reporter.reportPriorityPreempt(stage);
                reporter.reportVictimKvTokens(
                        victim.priority(), stage, victim.kvTokens());
                reporter.reportCancelConfirm(
                        proposal.endpointId(), victim.priority());
            }
            reporter.reportEvictionCommit(ctx.getPriority(),
                    proposal.evictionCase(), "success");
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report committed decode preemption: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), proposal.endpointId(), telemetryFailure);
        }
    }

    private void reportCancelRequests(BalanceContext ctx,
                                      DecodeEvictionProposal proposal) {
        try {
            for (DecodeRequestSnapshot victim : proposal.victims()) {
                reporter.reportCancelRequest(
                        proposal.endpointId(), victim.priority());
                reporter.reportCancel(
                        victim.priority(), "PRIORITY_PREEMPTED");
            }
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report priority cancel requests: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), proposal.endpointId(), telemetryFailure);
        }
    }

    private void reportCancelTimeout(BalanceContext ctx, String endpointId) {
        try {
            reporter.reportCancelTimeout(endpointId, ctx.getPriority());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report priority cancel timeout: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), endpointId, telemetryFailure);
        }
    }

    /** Local decode-eviction metrics are outside the reservation transaction. */
    private void reportCommittedLocalDecodeEviction(
            BalanceContext ctx, DecodeEvictionProposal proposal) {
        try {
            reporter.reportEvictionCommit(
                    ctx.getPriority(), proposal.evictionCase(), "success");
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[eviction-manager] failed to report committed local decode eviction: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), proposal.endpointId(), telemetryFailure);
        }
    }

    /**
     * Case label for an infeasible decode plan: the first endpoint with a
     * deficit determines the tag (deterministic for single-endpoint setups);
     * defaults to slot-full when no snapshot shows a deficit (raced away).
     */
    private static String infeasibleDecodeCase(PriorityRequestEnvelope envelope,
                                               List<DecodeEndpointSnapshot> decodes) {
        for (DecodeEndpointSnapshot ep : decodes) {
            String evictionCase = EvictionPlanner.decodeEvictionCase(envelope, ep);
            if (evictionCase != null) {
                return evictionCase;
            }
        }
        return DecodeEvictionProposal.CASE_SLOT;
    }

    private static EngineCancellationConfig requiredEngineCancellation(
            PreemptionConfig preemption) {
        EngineCancellationConfig cancellation = preemption.getEngineCancellation();
        if (cancellation == null) {
            throw new IllegalStateException(
                    "engineCancellation is required for DECODE_ENGINE_OWNED preemption");
        }
        return cancellation;
    }

    private static DecodeEndpoint.AdmissionCapacity decodeAdmissionCapacity(
            FlexlbConfig config, long concurrencyLimit) {
        long maxKvUsagePercent = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxKvUsagePercent();
        return new DecodeEndpoint.AdmissionCapacity(
                Math.max(0L, concurrencyLimit), maxKvUsagePercent);
    }

    private static Response admissionError(StrategyErrorType errorType,
                                           AdmissionRejectReason reason,
                                           String message) {
        Response errorResp = Response.error(errorType, reason);
        errorResp.setErrorMessage(errorType.buildErrorMessage(message));
        return errorResp;
    }

}
