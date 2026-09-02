package org.flexlb.balance.eviction;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.scheduler.AdmissionMutation;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.WorkerBatcher.QueueSnapshot;
import org.flexlb.balance.scheduler.EvictionPlacement;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.EvictionPlacement.DecodePlacement;
import org.flexlb.balance.scheduler.EvictionPlacement.PrefillEvictionAdmission;
import org.flexlb.balance.scheduler.EvictionPlacement.PrefillEvictionCommit;
import org.flexlb.balance.scheduler.EvictionPlacement.PrefillEvictionStatus;
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

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Eviction-only admission component. It neither selects the scheduler mode nor
 * commits an ordinary placement. A successful takeover reuses the request
 * slot's lifecycle guard while it consumes one immutable eviction plan and
 * retains future ownership until ACTIVE publication or a typed terminal
 * outcome.
 */
@Component
public class EvictionManager {

    private final EndpointRegistry endpointRegistry;
    private final RequestSchedulerReporter reporter;
    private final EngineCancelChannel cancelChannel;
    private final DecodePreemptionCoordinator preemptionCoordinator;
    private final RequestRegistry requests;
    private final EvictionPlacement placement;

    private volatile boolean shutdown;

    @Autowired
    public EvictionManager(EndpointRegistry endpointRegistry,
                           RequestSchedulerReporter reporter,
                           EngineCancelChannel cancelChannel,
                           DecodePreemptionCoordinator preemptionCoordinator,
                           RequestRegistry requests,
                           EvictionPlacement placement) {
        this.endpointRegistry = endpointRegistry;
        this.reporter = reporter;
        this.cancelChannel = cancelChannel;
        this.preemptionCoordinator = preemptionCoordinator;
        this.requests = Objects.requireNonNull(requests, "requests");
        this.placement = Objects.requireNonNull(placement, "placement");
    }

    @PreDestroy
    public void shutdown() {
        shutdown = true;
    }

    boolean isShutdown() {
        return shutdown;
    }

    /**
     * Attempt one eviction-backed admission after ordinary placement has
     * failed. A true result transfers the future to this manager; false leaves
     * the caller free to publish its original rejection.
     */
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

        commitDecodeEviction(ctx, future, planned, config, preemption);
        return true;
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
                requests.claimAdmissionMutation(
                        ctx.getRequestId(), future);
        if (mutation == null) {
            return PrefillAdmissionOutcome.DECLINED_NO_EVICTION;
        }
        try (mutation;
             PrefillEvictionAdmission admission =
                     placement.preparePrefillEviction(ctx, future)) {
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
            PrefillEvictionCommit commit = commitPrefillEviction(
                    ctx, admission, proposal);
            if (commit.status() != PrefillEvictionStatus.COMMITTED) {
                mutation.terminate(admissionError(
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "exact Prefill eviction plan was not committed: "
                                + commit.status().name().toLowerCase()));
            }
            return PrefillAdmissionOutcome.TAKEN_OVER;
        }
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

    /** Apply the exact proposal captured from the route-owned queue. */
    private PrefillEvictionCommit commitPrefillEviction(
            BalanceContext ctx,
            PrefillEvictionAdmission admission,
            PrefillEvictionProposal proposal) {
        PriorityRequestEnvelope envelope = admission.envelope();
        PrefillEvictionCommit replacement = admission.commit(
                proposal.victims());
        if (replacement.status() != PrefillEvictionStatus.COMMITTED) {
            reportEvictionCommit(envelope.priority(), envelope.requestId(),
                    "prefill_queue_full", replacement.status().name().toLowerCase());
            return replacement;
        }

        for (ScheduledRequest victim : replacement.removed()) {
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
        return replacement;
    }

    /**
     * Settle one victim removed by a committed queue replacement. The exact
     * reducer is total for legal local ownership; an exception is an invariant
     * violation and aborts the incoming handoff instead of starting a retry
     * chain or publishing past incomplete cleanup.
     */
    private void settlePrefillVictim(PriorityRequestEnvelope incoming,
                                     ScheduledRequest victim,
                                     String endpointId) {
        String detail = "yielded to higher-priority request " + incoming.requestId();
        requests.finishYielded(victim, detail);
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

    // ==================== Decode eviction ====================

    private record PlannedDecodeEviction(
            DecodeEndpointSnapshot target,
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
        Long configuredDecodeLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        long decodeConcurrencyLimit = configuredDecodeLimit == null
                ? 0 : configuredDecodeLimit;
        List<DecodeEndpointSnapshot> decodes = new ArrayList<>();
        endpointRegistry.snapshotDecodeEndpoints().values().forEach(endpoint ->
                decodes.add(DecodeEndpointSnapshot.capture(
                        endpoint, decodeConcurrencyLimit)));

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
        DecodeEndpointSnapshot target = decodes.stream()
                .filter(endpoint -> endpoint.endpointId().equals(
                        proposal.endpointId()))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException(
                        "planned Decode endpoint is absent from its snapshot"));
        return new PlannedDecodeEviction(target, proposal);
    }

    /** Commit exactly the immutable plan selected before takeover. */
    private void commitDecodeEviction(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            PlannedDecodeEviction planned,
            FlexlbConfig config,
            PreemptionConfig preemption) {
        DecodeEvictionProposal proposal = planned.proposal();
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();

        DecodeEndpointSnapshot target = planned.target();
        DecodeEndpoint decodeEp = target.endpoint();
        long expectedKvTokens = config.decodeKvReservationTokens(
                seqLen, maxNewTokens, target.realKvTotal());
        DecodeEndpoint.AdmissionCapacity capacity =
                decodeAdmissionCapacity(config, target.concurrencyLimit());

        // Ownership is homogeneous by planner invariant: Master-queued victims
        // use a local transaction; Engine-may-have-seen/accepted/running
        // victims use the tokenized Cancel coordinator.
        if (proposal.requiresEngineCancel()) {
            startEngineCancelPreemption(ctx, future, preemption, proposal,
                    decodeEp, seqLen, expectedKvTokens, capacity);
            return;
        }

        List<DecodeEndpoint.ReservationHandle> reservedVictims =
                new ArrayList<>(proposal.victims().size());
        for (DecodeRequestSnapshot victim : proposal.victims()) {
            reservedVictims.add(new DecodeEndpoint.ReservationHandle(
                    decodeEp.getStatus().getGenerationId(),
                    victim.requestId(),
                    victim.reservationToken()));
        }

        // The victim mutation and incoming placement form one generation
        // commit. Cancel/deadline either close before any victim is touched,
        // or observe the incoming request after the complete handoff.
        AdmissionMutation mutation =
                requests.claimAdmissionMutation(
                        ctx.getRequestId(), future);
        if (mutation == null) {
            return;
        }
        try (mutation) {
          DecodeEndpoint.LocalEvictionResult eviction =
                  decodeEp.tryEvictLocalReservationsAndReserveIncoming(
                          reservedVictims, ctx.getRequestId(), seqLen, expectedKvTokens,
                          ctx.getPriority(), capacity);
          if (eviction != DecodeEndpoint.LocalEvictionResult.COMMITTED) {
              reportEvictionCommit(ctx.getPriority(), ctx.getRequestId(),
                      proposal.evictionCase(), "conflict");
                Logger.debug(
                        "[eviction-manager] Decode eviction conflict: request_id={} "
                                + "planned={} worker={} result={}",
                        ctx.getRequestId(),
                        reservedVictims.size(),
                        proposal.endpointId(),
                        eviction);
              mutation.terminate(admissionError(
                      StrategyErrorType.RESOURCE_EXHAUSTED,
                      AdmissionRejectReason.RESOURCE_EXHAUSTED,
                      "exact Decode eviction plan changed before commit"));
              return;
          }

            // Shadow accounting already reversed atomically; drive each victim
            // terminal before publishing the incoming item. Reserved-only
            // victims were never seen by the engine — retryable 8400.
            for (DecodeRequestSnapshot victim : proposal.victims()) {
                finishDecodeVictim(ctx, victim,
                        "decode_reserved", proposal);
            }
            reportCommittedLocalDecodeEviction(ctx, proposal);
            recordDecodePlanObservability(ctx, proposal);
            DecodeEndpoint.ReservationHandle incoming =
                    decodeEp.reservationHandle(ctx.getRequestId());
            if (incoming == null) {
                mutation.terminate(admissionError(
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "Decode reservation disappeared before canonical placement"));
                return;
            }
            DecodePlacement placementResult = placement.placeReservedDecode(
                    ctx, future, decodeEp, incoming);
            if (placementResult instanceof DecodePlacement.Failed failed) {
                AdmissionFailure failure = failed.failure();
                mutation.terminate(admissionError(
                        failure.errorType(), failure.reason(), failure.message()));
                return;
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
        requests.finishYieldedReservation(
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
                                             DecodeEndpoint.AdmissionCapacity capacity) {
        String detail = "preempted by higher-priority request " + ctx.getRequestId();
        EngineCancellationConfig cancellation = requiredEngineCancellation(preemption);
        DecodePreemptionCoordinator.PreemptionCommand command =
                new DecodePreemptionCoordinator.PreemptionCommand(
                        decodeEp,
                        ctx.getRequestId(), seqLen, expectedKvTokens,
                        ctx.getPriority(),
                        capacity,
                        proposal.victims(), cancellation.getAckTimeoutMs(),
                        cancellation.getCompletionTimeoutMs(),
                        () -> requests.isAdmissionOpen(
                                ctx.getRequestId(), future), detail);

        CompletableFuture<DecodePreemptionCoordinator.PreemptionResult>
                execution;
        AdmissionMutation mutation =
                requests.claimAdmissionMutation(
                        ctx.getRequestId(), future);
        if (mutation == null) {
            return;
        }
        try {
            reportCancelRequests(ctx, proposal);
            // execute() performs the victim-claim and sends every Cancel
            // before returning. The mutation claim keeps an incoming
            // Cancel pending until this asynchronous attempt settles.
            execution = preemptionCoordinator.preempt(command);
        } catch (RuntimeException | Error startFailure) {
            mutation.close();
            throw startFailure;
        }

        execution.whenComplete(
                (result, error) -> {
                    try (mutation) {
                        Response terminal;
                        try {
                            terminal = enginePreemptionTerminal(
                                    ctx, future, proposal,
                                    decodeEp, result, error);
                        } catch (RuntimeException | Error callbackError) {
                            Logger.error(
                                    "[eviction-manager] cancel completion failed:"
                                            + " request_id={} error={}",
                                    ctx.getRequestId(), callbackError.getMessage(),
                                    callbackError);
                            terminal = admissionError(
                                    StrategyErrorType.RESOURCE_EXHAUSTED,
                                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                                    "Decode eviction placement failed: "
                                            + callbackError.getMessage());
                        }
                        if (terminal != null) {
                            mutation.terminate(terminal);
                        }
                    }
                });
    }

    /** Convert one typed coordinator result into commit or one terminal response. */
    private Response enginePreemptionTerminal(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            DecodeEvictionProposal proposal,
            DecodeEndpoint decodeEp,
            DecodePreemptionCoordinator.PreemptionResult result,
            Throwable error) {
        if (error != null || result == null) {
            reportCancelTimeout(ctx, proposal.endpointId());
            Logger.error(
                    "[eviction-manager] cancel coordinator returned no typed result:"
                            + " request_id={} worker={}",
                    ctx.getRequestId(), proposal.endpointId(), error);
            return admissionError(
                    StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "Decode eviction control failed before commit");
        }
        if (result.outcome()
                == DecodePreemptionCoordinator.Outcome.COMMITTED) {
            DecodeEndpoint.ReservationHandle reservation =
                    decodeEp.reservationHandle(ctx.getRequestId());
            if (reservation == null) {
                return admissionError(
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "Decode reservation disappeared before placement");
            }
            reportCommittedEnginePreemption(ctx, proposal);
            recordDecodePlanObservability(ctx, proposal);
            DecodePlacement placementResult = placement.placeReservedDecode(
                    ctx, future, decodeEp, reservation);
            if (placementResult instanceof DecodePlacement.Failed failed) {
                AdmissionFailure failure = failed.failure();
                return admissionError(
                        failure.errorType(), failure.reason(), failure.message());
            }
            return null;
        }
        if (result.outcome()
                == DecodePreemptionCoordinator.Outcome.CONTROL_FAILED) {
            reportCancelTimeout(ctx, proposal.endpointId());
        }
        return admissionError(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                result.detail());
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
