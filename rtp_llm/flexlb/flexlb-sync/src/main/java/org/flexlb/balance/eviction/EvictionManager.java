package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRequestView;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.flexlb.balance.scheduler.AdmissionMutation;
import org.flexlb.balance.scheduler.QueueRouteAdmission;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.scheduler.WorkerBatcher.QueueReplacementStatus;
import org.flexlb.balance.scheduler.WorkerBatcher.QueueSnapshot;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.FlexlbConfig;
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
            QueueRouteAdmission admission,
            WorkerEndpoint blockedEndpoint) {
        Objects.requireNonNull(admission, "admission");
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
        if (blockedEndpoint instanceof PrefillEndpoint
                && preemption.allows(VictimStage.PREFILL_QUEUED)) {
            return tryAdmitByPrefillEviction(
                    ctx, future, config, admission);
        }

        if (!(blockedEndpoint instanceof DecodeEndpoint decodeEndpoint)
                || (!preemption.allows(VictimStage.DECODE_RESERVED)
                && !preemption.allows(VictimStage.DECODE_ENGINE_OWNED))) {
            return false;
        }
        PlannedDecodeEviction planned =
                planDecodeEviction(
                        ctx, config, preemption, decodeEndpoint);
        if (planned == null) {
            return false;
        }

        try {
            commitDecodeEviction(
                    ctx, future, planned, config, preemption, admission);
            return true;
        } catch (RuntimeException | Error failure) {
            admission.close();
            throw failure;
        }
    }

    /** Plan and commit replacement only on the route-owned exact Prefill. */
    private boolean tryAdmitByPrefillEviction(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            FlexlbConfig config,
            QueueRouteAdmission admission) {
        AdmissionMutation mutation =
                requests.claimAdmissionMutation(
                        ctx.getRequestId(), future);
        if (mutation == null) {
            return false;
        }
        PriorityRequestEnvelope envelope = priorityEnvelope(
                ctx, config, admission.selectedDecodeTotalKv());
        QueueSnapshot snapshot = admission.capturePrefillQueueSnapshot();
        PrefillEvictionProposal proposal =
                planPrefillEviction(envelope, snapshot);
        if (proposal == null) {
            mutation.close();
            return false;
        }
        try (mutation; admission) {
            QueueReplacementStatus status = commitPrefillEviction(
                    ctx, future, admission, envelope, proposal);
            if (status != QueueReplacementStatus.SUCCESS) {
                mutation.terminate(admissionError(
                        StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "exact Prefill eviction plan was not committed: "
                                + status.name().toLowerCase()));
            }
            return true;
        }
    }

    // ==================== Prefill queue eviction ====================

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
    private QueueReplacementStatus commitPrefillEviction(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            QueueRouteAdmission admission,
            PriorityRequestEnvelope envelope,
            PrefillEvictionProposal proposal) {
        QueueRouteAdmission.QueueReplacementCommit committed =
                admission.commitReplacingQueuedVictims(
                        ctx,
                        future,
                        requests,
                        true,
                        proposal.victims());
        QueueReplacementStatus status = committed.status();
        if (status != QueueReplacementStatus.SUCCESS) {
            reportEvictionCommit(envelope.priority(), envelope.requestId(),
                    "prefill_queue_full", status.name().toLowerCase());
            return status;
        }

        reportPlacement(ctx, committed.item(), "Prefill eviction");

        for (ScheduledRequest victim : proposal.victims()) {
            settlePrefillVictim(envelope, victim, proposal.endpointId());
        }

        reportEvictionCommit(envelope.priority(), envelope.requestId(),
                "prefill_queue_full", "success");
        ctx.setPlanType("prefill_evict");
        ctx.setPlanCost(proposal.rawCost());
        ctx.setVictimCount(proposal.victims().size());
        Logger.debug(
                "[eviction-manager] eviction committed: request_id={} priority={} victims={} "
                        + "raw_cost={} worker={}",
                envelope.requestId(),
                envelope.priority(),
                proposal.victims().size(),
                proposal.rawCost(),
                proposal.endpointId());
        return status;
    }

    private static PriorityRequestEnvelope priorityEnvelope(
            BalanceContext context,
            FlexlbConfig config,
            long selectedDecodeTotalKv) {
        long seqLen = context.getRequest().getSeqLen();
        long maxNewTokens = context.getRequest().getMaxNewTokens();
        return new PriorityRequestEnvelope(
                context.getRequestId(),
                context.getPriority(),
                seqLen,
                maxNewTokens,
                context.getStartTime(),
                seqLen,
                config.decodeKvReservationTokens(
                        seqLen, maxNewTokens, selectedDecodeTotalKv));
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
            PreemptionConfig preemption,
            DecodeEndpoint selectedEndpoint) {
        Long configuredDecodeLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        long decodeConcurrencyLimit = configuredDecodeLimit == null
                ? 0 : configuredDecodeLimit;
        DecodeEndpointSnapshot selected = DecodeEndpointSnapshot.capture(
                selectedEndpoint, decodeConcurrencyLimit);
        if (selected.endpoint().isRetired()) {
            return null;
        }
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();
        PriorityRequestEnvelope envelope = new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), seqLen,
                config.decodeKvReservationTokens(
                        seqLen, maxNewTokens, selected.realKvTotal()));
        String evictionCase = EvictionPlanner.decodeEvictionCase(envelope, selected);
        List<DecodeEndpointSnapshot> decodes = List.of(selected);
        if (evictionCase == null) {
            return null;
        }

        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal = EvictionPlanner.planDecode(
                envelope, decodes, preemption, cancelChannel, failures);
        if (proposal == null) {
            reportEvictionPlan(ctx.getPriority(), ctx.getRequestId(),
                    evictionCase, "infeasible");
            Logger.debug(
                    "[eviction-manager] Decode eviction infeasible:"
                            + " request_id={} priority={} candidates={} reasons={}",
                    ctx.getRequestId(), ctx.getPriority(), decodes.size(), failures);
            return null;
        }
        reportEvictionPlan(ctx.getPriority(), ctx.getRequestId(),
                proposal.evictionCase(), "feasible");
        if (!selected.endpointId().equals(proposal.endpointId())) {
            throw new IllegalStateException(
                    "eviction planner changed the selected Decode endpoint");
        }
        return new PlannedDecodeEviction(selected, proposal);
    }

    /** Commit exactly the immutable plan selected before takeover. */
    private void commitDecodeEviction(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            PlannedDecodeEviction planned,
            FlexlbConfig config,
            PreemptionConfig preemption,
            QueueRouteAdmission admission) {
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
                    decodeEp, seqLen, expectedKvTokens, capacity, admission);
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
                        ctx.getRequestId(), future);
        if (mutation == null) {
            admission.close();
            return;
        }
        try (mutation; admission) {
            boolean evictionCommitted =
                    decodeEp.tryEvictLocalReservationsAndReserveIncoming(
                            reservedVictims,
                            ctx.getRequestId(),
                            seqLen,
                            expectedKvTokens,
                            ctx.getPriority(),
                            capacity);
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
            // victims were never seen by the engine — retryable 8400.
            for (DecodeRequestView victim : proposal.victims()) {
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
            QueueRouteAdmission admission) {
        if (!admission.adoptDecodeReservation(endpoint, reservation)) {
            return admissionError(
                    StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "Decode generation retired before canonical placement");
        }
        ScheduledRequest item = admission.buildItem(
                context, future, System.currentTimeMillis());
        context.setRouteSubmittedNanos(System.nanoTime());
        if (!admission.commitTo(requests, item, true)) {
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
        String terminal = "yielded_8400";
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
                terminal,
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
                                             long seqLen,
                                             long expectedKvTokens,
                                             DecodeEndpoint.AdmissionCapacity capacity,
                                             QueueRouteAdmission admission) {
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
            QueueRouteAdmission admission) {
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
                    decodeEp.reservationHandle(ctx.getRequestId());
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
