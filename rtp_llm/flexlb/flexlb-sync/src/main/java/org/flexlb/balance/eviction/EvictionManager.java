package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.AdmissionMutation;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RouteAdmission;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeMode;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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

    private static final String YIELDED_TERMINAL =
            "yielded_" + StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode();

    private final RequestSchedulerReporter reporter;
    private final EngineCancelChannel cancelChannel;
    private final DecodePreemptionCoordinator preemptionCoordinator;
    private final RequestRegistry requests;
    private final BatchSchedulerReporter deliveryReporter;

    private volatile boolean shutdown;

    @Autowired
    public EvictionManager(RequestSchedulerReporter reporter,
                           EngineCancelChannel cancelChannel,
                           DecodePreemptionCoordinator preemptionCoordinator,
                           RequestRegistry requests,
                           BatchSchedulerReporter deliveryReporter) {
        this.reporter = reporter;
        this.cancelChannel = cancelChannel;
        this.preemptionCoordinator = preemptionCoordinator;
        this.requests = Objects.requireNonNull(requests, "requests");
        this.deliveryReporter = Objects.requireNonNull(
                deliveryReporter, "deliveryReporter");
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
     * reached exact capacity. A true result transfers the future to this
     * manager; false leaves the caller responsible for parking the original
     * exact route.
     */
    public boolean tryAdmit(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            RouteAdmission admission,
            WorkerEndpoint blockedEndpoint) {
        Objects.requireNonNull(admission, "admission");
        DecodeBinding request = admission.decodeBinding();
        if (shutdown || future.isDone()
                || ctx.requestExpired(System.currentTimeMillis())
                || !PriorityNormalizer.hasPriority(request.priority())
                || request.mode() != DecodeMode.PREEMPT_AT_PLACEMENT) {
            return false;
        }

        PreemptionConfig preemption = ctx.getConfig().queueScheduler().getOrdering()
                .preemptionPolicy().orElse(null);
        if (preemption == null) {
            return false;
        }
        if (!(blockedEndpoint instanceof DecodeEndpoint decodeEndpoint)
                || (!preemption.allows(VictimStage.DECODE_RESERVED)
                && !preemption.allows(VictimStage.DECODE_ENGINE_OWNED))) {
            return false;
        }
        PlannedDecodeEviction planned =
                planDecodeEviction(
                        request, preemption, decodeEndpoint);
        if (planned == null) {
            return false;
        }

        try {
            commitDecodeEviction(
                    ctx, future, planned, preemption, admission);
            return true;
        } catch (RuntimeException | Error failure) {
            admission.close();
            throw failure;
        }
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
            DecodeEvictionProposal proposal,
            DecodeBinding request) {
    }

    /** Build one side-effect-free plan from one exact cluster snapshot. */
    private PlannedDecodeEviction planDecodeEviction(
            DecodeBinding request,
            PreemptionConfig preemption,
            DecodeEndpoint selectedEndpoint) {
        DecodeEndpointSnapshot selected = DecodeEndpointSnapshot.capture(
                selectedEndpoint, request.capacity());
        if (selected.endpoint().isRetired()) {
            return null;
        }
        String evictionCase = EvictionPlanner.decodeEvictionCase(
                request.hardKvTokens(), request.expectedKvTokens(), selected);
        List<DecodeEndpointSnapshot> decodes = List.of(selected);
        if (evictionCase == null) {
            return null;
        }

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                request.priority(), request.hardKvTokens(), request.expectedKvTokens(),
                decodes, preemption, cancelChannel, failures);
        if (proposal == null) {
            reportEvictionPlan(request.priority(), request.requestId(),
                    evictionCase, "infeasible");
            Logger.debug(
                    "[eviction-manager] Decode eviction infeasible:"
                            + " request_id={} priority={} candidates={} reasons={}",
                    request.requestId(), request.priority(), decodes.size(), failures);
            return null;
        }
        reportEvictionPlan(request.priority(), request.requestId(),
                proposal.evictionCase(), "feasible");
        if (!selected.endpointId().equals(proposal.endpointId())) {
            throw new IllegalStateException(
                    "eviction planner changed the selected Decode endpoint");
        }
        return new PlannedDecodeEviction(selected, proposal, request);
    }

    /** Commit exactly the immutable plan selected before takeover. */
    private void commitDecodeEviction(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            PlannedDecodeEviction planned,
            PreemptionConfig preemption,
            RouteAdmission admission) {
        DecodeEvictionProposal proposal = planned.proposal();
        DecodeBinding request = planned.request();

        DecodeEndpointSnapshot target = planned.target();
        DecodeEndpoint decodeEp = target.endpoint();

        // Ownership is homogeneous by planner invariant: Master-queued victims
        // use a local transaction; Engine-may-have-seen/accepted/running
        // victims use the tokenized Cancel coordinator.
        if (proposal.requiresEngineCancel()) {
            startEngineCancelPreemption(ctx, future, preemption, proposal,
                    decodeEp, request, admission);
            return;
        }

        List<DecodeEndpoint.ReservationHandle> reservedVictims =
                new ArrayList<>(proposal.victims().size());
        for (DecodeRequestView victim : proposal.victims()) {
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
                        request.requestId(), future);
        if (mutation == null) {
            admission.close();
            return;
        }
        try (mutation; admission) {
            boolean evictionCommitted =
                    decodeEp.tryEvictLocalReservationsAndReserveIncoming(
                            reservedVictims,
                            request.requestId(),
                            request.hardKvTokens(),
                            request.expectedKvTokens(),
                            request.priority(),
                            request.capacity());
            if (!evictionCommitted) {
                reportEvictionCommit(ctx.getPriority(), ctx.getRequestId(),
                        proposal.evictionCase(), "conflict");
                Logger.debug(
                        "[eviction-manager] Decode eviction conflict: request_id={} "
                                + "planned={} worker={}",
                        ctx.getRequestId(),
                        reservedVictims.size(),
                        proposal.endpointId());
                mutation.terminate(admissionError(
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "exact Decode eviction plan changed before commit"));
                return;
            }

            // Shadow accounting already reversed atomically; drive each victim
            // terminal before publishing the incoming item. Reserved-only
            // victims were never seen by the engine, so they terminate with
            // the retryable NO_AVAILABLE_WORKER contract.
            for (DecodeRequestView victim : proposal.victims()) {
                finishDecodeVictim(ctx, victim,
                        "decode_reserved", proposal);
            }
            reportCommittedLocalDecodeEviction(ctx, proposal);
            recordDecodePlanObservability(ctx, proposal);
            DecodeEndpoint.ReservationHandle incoming =
                    decodeEp.reservationHandle(request.requestId());
            if (incoming == null) {
                mutation.terminate(admissionError(
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "Decode reservation disappeared before canonical placement"));
                return;
            }
            Response placementFailure = placeReservedDecode(
                    ctx, future, decodeEp, incoming, admission);
            if (placementFailure != null) {
                mutation.terminate(placementFailure);
                return;
            }
        }
    }

    /** Publish preempted Decode capacity through the already selected route. */
    private Response placeReservedDecode(
            BalanceContext context,
            CompletableFuture<Response> future,
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation,
            RouteAdmission admission) {
        if (!admission.adoptDecodeReservation(endpoint, reservation)) {
            return admissionError(
                    StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "Decode generation retired before canonical placement");
        }
        ScheduledRequest item = admission.createScheduledRequest(
                context, future, System.currentTimeMillis());
        context.setRouteSubmittedNanos(System.nanoTime());
        if (!admission.commitQueuedRequest(requests, item)) {
            return admissionError(
                    StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "selected Prefill capacity changed before canonical placement");
        }
        reportPlacement(context, item, "Decode eviction");
        return null;
    }

    private void reportPlacement(
            BalanceContext context,
            ScheduledRequest item,
            String kind) {
        try {
            deliveryReporter.reportRouteSubmitTimeMs(
                    org.flexlb.dao.route.RoleType.PREFILL.name(),
                    item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Failed to report {} placement: request_id={}",
                    kind,
                    context.getRequestId(),
                    telemetryFailure);
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
                                    DecodeRequestView victim, String stage,
                                    DecodeEvictionProposal proposal) {
        if (victim.phase().isEngineConfirmed()
                || victim.reservationToken() <= 0L) {
            throw new IllegalStateException(
                    "local Decode eviction requires an exact reserved victim: request_id="
                            + victim.requestId());
        }
        String detail = "yielded to higher-priority request "
                + ctx.getRequestId();
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
                YIELDED_TERMINAL,
                victim.kvTokens(),
                ctx.getRequestId(),
                ctx.getPriority(),
                proposal.endpointId());
    }

    /** Record the single committed Decode-eviction plan. */
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
                                             DecodeBinding request,
                                             RouteAdmission admission) {
        String detail = "preempted by higher-priority request " + ctx.getRequestId();
        DecodePreemptionCoordinator.PreemptionCommand command =
                new DecodePreemptionCoordinator.PreemptionCommand(
                        decodeEp,
                        request.requestId(), request.hardKvTokens(),
                        request.expectedKvTokens(), request.priority(),
                        request.capacity(),
                        proposal.victims(), 50L,
                        preemption.getTimeoutMs(),
                        () -> requests.isAdmissionOpen(
                                request.requestId(), future), detail);

        CompletableFuture<DecodePreemptionCoordinator.PreemptionResult>
                execution;
        AdmissionMutation mutation =
                requests.claimAdmissionMutation(
                        request.requestId(), future);
        if (mutation == null) {
            admission.close();
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
                    try (mutation; admission) {
                        Response terminal;
                        try {
                            terminal = enginePreemptionTerminal(
                                    ctx, future, proposal,
                                    decodeEp, result, error, admission);
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
            Throwable error,
            RouteAdmission admission) {
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
        if (result.committed()) {
            DecodeEndpoint.ReservationHandle reservation =
                    decodeEp.reservationHandle(admission.decodeBinding().requestId());
            if (reservation == null) {
                return admissionError(
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "Decode reservation disappeared before placement");
            }
            reportCommittedEnginePreemption(ctx, proposal);
            recordDecodePlanObservability(ctx, proposal);
            return placeReservedDecode(
                    ctx, future, decodeEp, reservation, admission);
        }
        if (result.controlFailure()) {
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
            for (DecodeRequestView victim : proposal.victims()) {
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
            for (DecodeRequestView victim : proposal.victims()) {
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

    private static Response admissionError(StrategyErrorType errorType,
                                           AdmissionRejectReason reason,
                                           String message) {
        Response errorResp = Response.error(errorType, reason);
        errorResp.setErrorMessage(errorType.buildErrorMessage(message));
        return errorResp;
    }

}
