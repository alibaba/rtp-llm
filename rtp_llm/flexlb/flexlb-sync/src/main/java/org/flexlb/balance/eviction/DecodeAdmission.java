package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimResolution;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.AdmissionHandle;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.QueuedRequestScheduler;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.balance.scheduler.RequestTaskScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter.CancelEvent;
import org.flexlb.service.monitor.RequestSchedulerReporter.EvictionEvent;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Obtains capacity at one selected Decode generation through local withdrawal or Engine cancellation.
 *
 * <p>The scheduler supplies the blocked endpoint and a Plan. This class owns the
 * two-phase protocol, token fencing and reservation transfer or release.
 * Engine acknowledgement is only control evidence; the canonical victim
 * resolution transaction may complete before or after that acknowledgement.</p>
 */
@Component
public class DecodeAdmission {
    private static final AbstractRequestScheduler.PreemptionEvent START_CANCEL =
            new AbstractRequestScheduler.PreemptionEvent(
                    PreemptionCancelPhase.CANCEL_IN_FLIGHT, false, null, null, null);

    public record PreemptionResult(
            DecodeResources.ReservationHandle reservation,
            boolean controlFailure,
            String detail,
            List<String> preemptRequestIds) {
        public PreemptionResult(
                DecodeResources.ReservationHandle reservation,
                boolean controlFailure,
                String detail) {
            this(reservation, controlFailure, detail, List.of());
        }

        public boolean committed() {
            return reservation != null;
        }

        public PreemptionResult {
            checkArgument(
                    reservation == null || !controlFailure,
                    "committed preemption cannot be a control failure");
            Objects.requireNonNull(detail, "detail");
            preemptRequestIds = List.copyOf(preemptRequestIds);
        }
    }

    record PreemptionCommand(
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata,
            List<DecodeRequestView> victims,
            long cancelAckTimeoutMs,
            long preemptionTimeoutMs,
            BooleanSupplier admissionOpen,
            String detail) {
        public PreemptionCommand {
            checkArgument(
                    endpoint != null && victims != null && !victims.isEmpty(),
                    "endpoint and victims are required");
            victims = List.copyOf(victims);
            Objects.requireNonNull(request, "request");
            Objects.requireNonNull(metadata, "metadata");
            checkArgument(
                    request.getRequestId() != null && !request.getRequestId().isBlank(),
                    "incoming request id must be positive");
            Set<String> victimIds = new LinkedHashSet<>();
            for (DecodeRequestView victim : victims) {
                checkArgument(
                        victim.requestId() != null
                                && !victim.requestId().isBlank()
                                && victim.reservationToken() > 0L,
                        "victim requestId and reservation token must be positive");
                checkArgument(
                        victim.phase() != null && victim.phase().requiresEngineCancel(),
                        "capacity acquisition accepts only Engine-Cancel victims");
                checkArgument(
                        victimIds.add(victim.requestId()),
                        "duplicate victim %s",
                        victim.requestId());
            }
            checkArgument(admissionOpen != null, "admission gate is required");
        }
    }

    private final RequestSchedulerReporter reporter;
    private volatile boolean shutdown;
    private final DeliveryDispatcher dispatcher;
    private final RequestRepository requests;
    private final RequestTaskScheduler runtime;
    private final AtomicLong tokenSequence = new AtomicLong(1);

    @Autowired
    public DecodeAdmission(
            DeliveryDispatcher dispatcher,
            RequestRepository requests,
            RequestTaskScheduler runtime,
            RequestSchedulerReporter reporter,
            ConfigService configService,
            AdmissionCapacity capacity) {
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        this.requests = Objects.requireNonNull(requests, "requests");
        this.runtime = Objects.requireNonNull(runtime, "runtime");
        Objects.requireNonNull(configService, "configService");
        Objects.requireNonNull(capacity, "capacity");
    }

    CompletableFuture<PreemptionResult> preempt(RequestContext context, PreemptionCommand command) {
        Objects.requireNonNull(context, "context");
        long token = nextToken();
        List<DecodeResources.ReservationHandle> victimReservations =
                new ArrayList<>(command.victims().size());
        long endpointGenerationId = command.endpoint().getStatus().getGenerationId();
        AttemptCapability capability = new AttemptCapability(command, token);
        try {
            var owners = new java.util.LinkedHashMap<String, AbstractRequestScheduler>();
            for (DecodeRequestView victim : command.victims()) {
                var owner = requests.ownerOf(victim.requestId());
                if (owner == null) {
                    reportTargetValidationFailure("rpc", "cancel_target_unavailable");
                    return CompletableFuture.completedFuture(
                            capability.abort(true, "cancel_owner_missing:" + victim.requestId()));
                }
                owners.put(victim.requestId(), owner);
            }
            for (DecodeRequestView victim : command.victims()) {
                var owner = owners.get(victim.requestId());
                var reservation =
                        new DecodeResources.ReservationHandle(
                                endpointGenerationId,
                                victim.requestId(),
                                victim.reservationToken());
                Optional<PreemptionRegistration> claimAttempt = owner.tryClaim(reservation, token);
                if (claimAttempt.isEmpty()) {
                    reportTargetValidationFailure("rpc", "request_claim_rejected");
                    return CompletableFuture.completedFuture(
                            capability.abort(false, "victim_inflight_gone"));
                }
                PreemptionRegistration claim = claimAttempt.get();
                ClaimedVictim owned = new ClaimedVictim(reservation, claim);
                capability.claims.add(owned);
                victimReservations.add(reservation);
                if (!java.util.Objects.equals(
                                claim.owner().data().getRequestId(), victim.requestId())
                        || claim.attemptToken() != token) {
                    return CompletableFuture.completedFuture(
                            capability.abort(
                                    true,
                                    "lifecycle_returned_mismatched_claim:" + owned.requestId()));
                }
            }

            DecodeResources.PreemptionBeginResult begin =
                    command.endpoint()
                            .beginPreemption(
                                    token,
                                    victimReservations,
                                    command.request().getRequestId(),
                                    command.request().getHardKvTokens(),
                                    command.request().getExpectedKvTokens(),
                                    command.metadata().priority(),
                                    context.data().getDecodeAdmissionCapacity());
            if (begin != DecodeResources.PreemptionBeginResult.SUCCESS) {
                reportPreemptionTargetValidation("rpc", begin);
                return CompletableFuture.completedFuture(
                        capability.abort(
                                begin == DecodeResources.PreemptionBeginResult.ENDPOINT_RETIRED,
                                "begin_" + begin.name().toLowerCase()));
            }
            capability.endpointBegun = true;
            capability.observe();
            for (ClaimedVictim owned : capability.claims) {
                var scheduler = owned.claim.owner().data().getScheduler();
                if (!scheduler.onPreemption(owned.claim, START_CANCEL)) {
                    return CompletableFuture.completedFuture(
                            capability.abort(
                                    true,
                                    "inflight_cancel_linearization_failed:" + owned.requestId()));
                }
            }
            for (ClaimedVictim owned : capability.claims) {
                owned.acknowledgement =
                        capability.outboundStarted(owned)
                                ? cancel(owned, command, capability.expiresAt)
                                : CompletableFuture.completedFuture(
                                        PreemptionRegistration.Reply.UNKNOWN);
            }

            CompletableFuture<PreemptionResult> protocol =
                    CompletableFuture.allOf(
                                    capability.claims.stream()
                                            .map(owned -> owned.acknowledgement)
                                            .toArray(CompletableFuture[]::new))
                            .thenComposeAsync(
                                    ignored -> handleAcknowledgements(context, capability),
                                    work -> runtime.executeRequestWork(context, work))
                            .whenComplete(
                                    (ignored, failure) -> capability.settlement.cancel(false));
            return protocol.handle(
                    (result, failure) -> {
                        if (failure != null) {
                            return capability.abort(
                                    true,
                                    "coordinator_continuation_failed:" + failureDetail(failure));
                        }
                        return result;
                    });
        } catch (RuntimeException | Error failure) {
            return CompletableFuture.completedFuture(
                    capability.abort(true, "coordinator_setup_failed:" + failureDetail(failure)));
        }
    }

    private CompletableFuture<PreemptionResult> handleAcknowledgements(
            RequestContext context, AttemptCapability capability) {
        // A transport-unknown ACK is not a negative acknowledgement: the
        // Prefill may have installed the intent before the reply was lost.
        // Such a child therefore waits for the canonical victim resolution
        // transaction exactly like an ACCEPTED child.
        int waiting = 0;
        boolean hasNotFound = false;

        for (ClaimedVictim owned : capability.claims) {
            if (owned.requestResolved()) {
                continue;
            }
            var progress = owned.acknowledgement.join();
            if (progress == PreemptionRegistration.Reply.UNKNOWN) {
                capability.transferUnknown(owned);
            } else {
                capability.transferred(owned);
            }
            if (progress == PreemptionRegistration.Reply.NOT_FOUND) {
                hasNotFound |= !owned.requestResolved();
            } else {
                waiting++;
            }
        }

        if (waiting == 0) {
            capability.settlement.complete(capability.allVictimsResolved());
            return CompletableFuture.completedFuture(
                    capability.finish(hasNotFound, capability.settlement.getNow(false)));
        }
        // Keep early rejection when the waited victims resolve but a NOT_FOUND victim has not.
        if (hasNotFound) {
            var pending =
                    CompletableFuture.allOf(
                            capability.claims.stream()
                                    .filter(
                                            owned ->
                                                    owned.acknowledgement.join()
                                                            != PreemptionRegistration.Reply
                                                                    .NOT_FOUND)
                                    .map(
                                            owned ->
                                                    runtime.signals()
                                                            .preemption(owned.claim)
                                                            .toCompletableFuture())
                                    .toArray(CompletableFuture<?>[]::new));
            pending.whenComplete(
                    (ignored, failure) ->
                            capability.settlement.complete(capability.allVictimsResolved()));
            capability.settlement.whenComplete((ignored, failure) -> pending.cancel(false));
        }
        final boolean ackNotFound = hasNotFound;
        // Late facts still settle victims; they cannot change the frozen admission result.
        return capability.settlement.thenApplyAsync(
                resolved -> capability.finish(ackNotFound, resolved),
                work -> runtime.executeRequestWork(context, work));
    }

    private CompletableFuture<PreemptionRegistration.Reply> cancel(
            ClaimedVictim victim, PreemptionCommand command, long expiresAt) {
        try {
            long remaining =
                    java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(
                            expiresAt - System.nanoTime());
            if (remaining <= 0) {
                return CompletableFuture.completedFuture(PreemptionRegistration.Reply.UNKNOWN);
            }
            var stage =
                    dispatcher.preempt(
                            victim.claim,
                            command.endpoint(),
                            command.detail(),
                            Math.min(command.cancelAckTimeoutMs(), remaining));
            if (stage == null) {
                return CompletableFuture.completedFuture(PreemptionRegistration.Reply.UNKNOWN);
            }
            return stage.handle(
                    (outcome, failure) ->
                            failure == null && outcome != null
                                    ? outcome
                                    : PreemptionRegistration.Reply.UNKNOWN);
        } catch (RuntimeException | Error failure) {
            // The capability marked this victim OUTBOUND before invocation,
            // so close conservatively transfers its exact claims to UNKNOWN.
            return CompletableFuture.completedFuture(PreemptionRegistration.Reply.UNKNOWN);
        }
    }

    private static String failureDetail(Throwable failure) {
        Throwable current = failure;
        while (current.getCause() != null && current.getCause() != current) {
            current = current.getCause();
        }
        String message = current.getMessage();
        return current.getClass().getSimpleName()
                + (message == null || message.isBlank() ? "" : ":" + message);
    }

    private enum ClaimDisposition {
        RELEASABLE,
        OUTBOUND,
        TRANSFERRED
    }

    /**
     * Exact opaque request claim paired with its immutable endpoint victim.
     */
    private final class ClaimedVictim {
        private final DecodeResources.ReservationHandle reservation;
        private final PreemptionRegistration claim;
        private CompletableFuture<PreemptionRegistration.Reply> acknowledgement;
        private volatile ClaimDisposition disposition = ClaimDisposition.RELEASABLE;

        private ClaimedVictim(
                DecodeResources.ReservationHandle reservation, PreemptionRegistration claim) {
            this.reservation = reservation;
            this.claim = claim;
        }

        private String requestId() {
            return reservation.requestId();
        }

        private boolean requestResolved() {
            VictimResolution resolution = runtime.signals().resolved(claim);
            return resolution != null
                    && java.util.Objects.equals(resolution.requestId(), requestId());
        }
    }

    /**
     * The one owner of endpoint admission plus every exact RequestContext claim.
     * A non-committed close is total: uncertain outbound claims transfer to
     * reconciliation, the incoming endpoint reservation aborts, and only
     * claims which never crossed an outbound boundary are released.
     */
    private final class AttemptCapability implements AutoCloseable {
        private final PreemptionCommand command;
        private final long token;
        private final List<ClaimedVictim> claims = new ArrayList<>();
        private final long expiresAt;
        private final CompletableFuture<Boolean> settlement = new CompletableFuture<>();
        private boolean endpointBegun;
        private boolean closed;
        private String cleanupFailure;

        private AttemptCapability(PreemptionCommand command, long token) {
            this.command = command;
            this.token = token;
            this.expiresAt =
                    System.nanoTime()
                            + java.util.concurrent.TimeUnit.MILLISECONDS.toNanos(
                                    command.preemptionTimeoutMs());
        }

        private void observe() {
            var signals =
                    claims.stream()
                            .map(v -> runtime.signals().preemption(v.claim).toCompletableFuture())
                            .toArray(CompletableFuture[]::new);
            var all = CompletableFuture.allOf(signals);
            all.whenComplete((ignored, failure) -> settlement.complete(allVictimsResolved()));
            settlement.whenComplete((ignored, failure) -> all.cancel(false));
            if (!settlement.isDone()) {
                long remaining = expiresAt - System.nanoTime();
                if (remaining <= 0) {
                    settlement.complete(false);
                } else {
                    var deadline =
                            runtime.scheduleTimeout(
                                    () -> settlement.complete(allVictimsResolved()),
                                    Math.max(
                                            1L,
                                            java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(
                                                    remaining)));
                    settlement.whenComplete((ignored, failure) -> deadline.cancel());
                }
            }
        }

        private synchronized boolean outboundStarted(ClaimedVictim owned) {
            if (owned.requestResolved() || settlement.isDone() && !settlement.getNow(false)) {
                return false;
            }
            if (owned.disposition != ClaimDisposition.RELEASABLE) {
                throw new IllegalStateException(
                        "Cancel outbound ownership changed request_id=" + owned.requestId());
            }
            // Install the conservative resource hold before RPC invocation. No ACK changes this
            // fact.
            if (!command.endpoint()
                    .updatePreemption(
                            token, DecodeResources.PreemptionUpdate.handedOff(owned.reservation))
                    .status()
                    .applied()) {
                return false;
            }
            owned.disposition = ClaimDisposition.OUTBOUND;
            return true;
        }

        private synchronized boolean allVictimsResolved() {
            if (claims.isEmpty()) {
                return false;
            }
            return claims.stream().allMatch(ClaimedVictim::requestResolved);
        }

        private synchronized void transferred(ClaimedVictim owned) {
            owned.disposition = ClaimDisposition.TRANSFERRED;
        }

        private void transferUnknown(ClaimedVictim owned) {
            if (owned.disposition != ClaimDisposition.OUTBOUND || owned.requestResolved()) {
                return;
            }
            owned.claim
                    .owner()
                    .data()
                    .getScheduler()
                    .onPreemption(
                            owned.claim,
                            new AbstractRequestScheduler.PreemptionEvent(
                                    PreemptionCancelPhase.CANCEL_UNKNOWN, false, null, null, null));
            transferred(owned);
        }

        private PreemptionResult finish(boolean hasNotFound, boolean victimsResolved) {
            DecodeResources.ReservationHandle incoming =
                    victimsResolved && !closed && command.admissionOpen().getAsBoolean()
                            ? command.endpoint().commitPreemption(token)
                            : null;
            if (incoming != null) {
                beginClose();
                return new PreemptionResult(incoming, false, "committed");
            }
            boolean cleanSingleNotFound =
                    hasNotFound && claims.size() == 1 && !claims.get(0).requestResolved();
            return abort(
                    !cleanSingleNotFound,
                    cleanSingleNotFound ? "cancel_not_found" : "cancel_terminal_unknown");
        }

        private PreemptionResult abort(boolean controlFailure, String detail) {
            close();
            String resultDetail =
                    cleanupFailure == null ? detail : detail + ";cleanup_failed=" + cleanupFailure;
            return new PreemptionResult(null, controlFailure, resultDetail);
        }

        @Override
        public void close() {
            if (!beginClose()) {
                return;
            }
            settlement.complete(false);

            // OUTBOUND means the call may have reached Prefill even if its
            // Java invocation or continuation failed. Transfer before endpoint
            // abort so neither owner can be mistaken for locally releasable.
            for (ClaimedVictim owned : claims) {
                if (owned.disposition == ClaimDisposition.OUTBOUND) {
                    cleanup("transfer_unknown:" + owned.requestId(), () -> transferUnknown(owned));
                }
            }
            if (endpointBegun) {
                cleanup("endpoint_abort", () -> command.endpoint().abortPreemption(token));
            }
            for (ClaimedVictim owned : claims) {
                if (owned.disposition == ClaimDisposition.RELEASABLE && !owned.requestResolved()) {
                    cleanup(
                            "release_claim:" + owned.requestId(),
                            () ->
                                    owned.claim
                                            .owner()
                                            .data()
                                            .getScheduler()
                                            .onPreemption(
                                                    owned.claim,
                                                    new AbstractRequestScheduler.PreemptionEvent(
                                                            null, true, null, null, null)));
                }
            }
        }

        private synchronized boolean beginClose() {
            if (closed) {
                return false;
            }
            closed = true;
            return true;
        }

        private void cleanup(String operation, Runnable action) {
            try {
                action.run();
            } catch (RuntimeException | Error failure) {
                if (cleanupFailure == null) {
                    cleanupFailure = operation + ":" + failureDetail(failure);
                }
            }
        }
    }

    private long nextToken() {
        long token = tokenSequence.getAndIncrement();
        checkState(token > 0, "preemption attempt token exhausted");
        return token;
    }

    @PreDestroy
    public void shutdown() {
        shutdown = true;
    }

    /**
     * Completes only after the Plan accepts acquired capacity or that capacity is released.
     * Null means no takeover occurred, including a local victim conflict. The returned
     * future observes the operation; cancelling it cannot interrupt ownership transfer.
     */
    public CompletableFuture<PreemptionResult> admit(
            QueuedRequestScheduler.Plan plan, WorkerEndpoint endpoint) {
        var attempt = tryReclaim(plan.context(), endpoint);
        if (attempt == null) {
            return null;
        }
        return attempt.thenApply(
                        result -> {
                            if (result == null || !result.committed()) {
                                return result;
                            }
                            var decode = (DecodeEndpoint) endpoint;
                            boolean accepted = false;
                            Throwable failure = null;
                            try {
                                accepted =
                                        plan.accept(
                                                decode,
                                                result.reservation(),
                                                result.preemptRequestIds());
                                return accepted
                                        ? result
                                        : new PreemptionResult(
                                                null, false, "plan_no_longer_current");
                            } catch (RuntimeException | Error failed) {
                                failure = failed;
                                throw failed;
                            } finally {
                                if (!accepted) {
                                    try {
                                        var released =
                                                decode.release(
                                                                result.reservation(),
                                                                DecodeResources.ReleaseReason
                                                                        .LOCAL_ROLLBACK)
                                                        .status();
                                        checkState(
                                                released
                                                                == DecodeResources
                                                                        .ReservationReleaseResult
                                                                        .RELEASED
                                                        || released
                                                                == DecodeResources
                                                                        .ReservationReleaseResult
                                                                        .STALE,
                                                "unaccepted admission retained Decode resources");
                                    } catch (RuntimeException | Error failed) {
                                        if (failure == null) {
                                            throw failed;
                                        }
                                        failure.addSuppressed(failed);
                                    }
                                }
                            }
                        })
                .copy();
    }

    CompletableFuture<PreemptionResult> tryReclaim(
            RequestContext ctx, WorkerEndpoint blockedEndpoint) {
        Request request = ctx.data().getRequest();
        var config = ctx.data().getConfig();
        PreemptionConfig policy =
                config.isPriorityOrdering() ? config.priorityOrdering().getPreemption() : null;
        SchedulingMetadata metadata = ctx.data().getSchedulingMetadata();
        if (shutdown
                || ctx.data().getFuture().isDone()
                || System.currentTimeMillis() >= ctx.data().getRequestExpiresAtMs()
                || !PriorityNormalizer.hasPriority(metadata.priority())
                || policy == null) {
            return null;
        }
        if (!(blockedEndpoint instanceof DecodeEndpoint decodeEndpoint)
                || (!policy.allows(VictimStage.DECODE_RESERVED)
                        && !policy.allows(VictimStage.DECODE_ENGINE_OWNED))) {
            return null;
        }
        DecodeEvictionProposal proposal =
                planDecodeEviction(
                        request,
                        metadata,
                        decodeEndpoint,
                        policy,
                        ctx.data().getDecodeAdmissionCapacity());
        if (proposal == null
                || !ctx.data()
                        .getScheduler()
                        .isAdmissionOpen(request.getRequestId(), ctx.data().getFuture())) {
            return null;
        }
        if (proposal.requiresEngineCancel()) {
            return policy.getEngineCancellation().getMode() == EngineCancellationConfig.Mode.RETURN
                    ? prepareReturnedPreemption(ctx, proposal, decodeEndpoint, request, metadata)
                    : startEngineCancelPreemption(
                            ctx, proposal, decodeEndpoint, request, metadata, policy);
        }
        List<DecodeResources.ReservationHandle> victims =
                new ArrayList<>(proposal.victims().size());
        for (DecodeRequestView victim : proposal.victims()) {
            victims.add(
                    new DecodeResources.ReservationHandle(
                            decodeEndpoint.getStatus().getGenerationId(),
                            victim.requestId(),
                            victim.reservationToken()));
        }
        DecodeResources.ReservationHandle incoming =
                replaceQueuedDecodeReservations(
                        decodeEndpoint,
                        victims,
                        request,
                        metadata.priority(),
                        ctx.data().getDecodeAdmissionCapacity());
        if (incoming == null) {
            reportEviction(
                    EvictionEvent.COMMIT,
                    ctx.data().getPriority(),
                    ctx.data().getRequestId(),
                    proposal.evictionCase(),
                    "conflict");
            return null;
        }
        report(
                ctx.data().getRequestId(),
                "local preemption",
                () -> {
                    for (DecodeRequestView victim : proposal.victims()) {
                        reportRequeuedVictim(ctx, victim, proposal);
                    }
                    reportEviction(
                            EvictionEvent.COMMIT,
                            ctx.data().getPriority(),
                            ctx.data().getRequestId(),
                            proposal.evictionCase(),
                            "success");
                    recordDecodePlanObservability(ctx, proposal);
                });
        return CompletableFuture.completedFuture(
                new PreemptionResult(incoming, false, "committed"));
    }

    /**
     * Metrics observe plans and commits without owning the reservation transaction.
     */
    private void reportEviction(
            EvictionEvent event,
            int priority,
            String requestId,
            String evictionCase,
            String outcome) {
        report(
                requestId,
                event == EvictionEvent.PLAN ? "eviction plan" : "eviction commit",
                () -> reporter.reportEviction(event, priority, evictionCase, outcome));
    }

    // ==================== Decode eviction ====================
    /**
     * Build one side-effect-free plan from one exact cluster snapshot.
     */
    private DecodeEvictionProposal planDecodeEviction(
            Request request,
            SchedulingMetadata metadata,
            DecodeEndpoint selectedEndpoint,
            PreemptionConfig policy,
            AdmissionCapacity capacity) {
        DecodeResources.ResourceSnapshot selected = selectedEndpoint.resourceSnapshot();
        if (selectedEndpoint.isRetired()) {
            return null;
        }
        Map<String, String> failures = new HashMap<>();
        var decision =
                EvictionPlanner.planDecode(
                        request, metadata.priority(), capacity, selected, policy, failures);
        if (decision.deficit().fits()) {
            return null;
        }
        DecodeEvictionProposal proposal = decision.proposal();
        if (proposal == null) {
            reportEviction(
                    EvictionEvent.PLAN,
                    metadata.priority(),
                    request.getRequestId(),
                    decision.evictionCase(),
                    "infeasible");
            Logger.debug(
                    "[decode-capacity] Decode eviction infeasible:"
                            + " request_id={} priority={} worker={} reasons={}",
                    request.getRequestId(),
                    metadata.priority(),
                    selected.routing().address(),
                    failures);
            return null;
        }
        reportEviction(
                EvictionEvent.PLAN,
                metadata.priority(),
                request.getRequestId(),
                proposal.evictionCase(),
                "feasible");
        return proposal;
    }

    /**
     * Observability only: the request's scheduler completes withdrawal and requeue.
     */
    private void reportRequeuedVictim(
            RequestContext ctx, DecodeRequestView victim, DecodeEvictionProposal proposal) {
        String stage = "decode_reserved";
        report(
                ctx.data().getRequestId(),
                "requeued victim",
                () -> {
                    reporter.reportVictim(
                            victim.priority(),
                            ctx.data().getPriority(),
                            stage,
                            proposal.evictionCase());
                    reporter.reportVictimKvTokens(victim.priority(), stage, victim.kvTokens());
                });
        Logger.debug(
                "[decode-capacity] decode victim preempted: victim_id={} victim_priority={}"
                        + " stage={} outcome={} kv_tokens={} incoming_id={} incoming_priority={}"
                        + " worker={}",
                victim.requestId(),
                victim.priority(),
                stage,
                "requeued",
                victim.kvTokens(),
                ctx.data().getRequestId(),
                ctx.data().getPriority(),
                proposal.endpointId());
    }

    /**
     * Record the single committed Decode-eviction plan.
     */
    private static void recordDecodePlanObservability(
            RequestContext ctx, DecodeEvictionProposal proposal) {
        long totalCost = proposal.priorityHarmProfile().totalCost();
        ctx.data().setPlanType("decode_evict");
        ctx.data().setPlanCost(totalCost);
        ctx.data().setVictimCount(proposal.victims().size());
        Logger.debug(
                "[decode-capacity] decode eviction committed: request_id={} priority={} case={} "
                        + "victims={} total_cost={} freed_kv={} worker={}",
                ctx.data().getRequestId(),
                ctx.data().getPriority(),
                proposal.evictionCase(),
                proposal.victims().size(),
                totalCost,
                proposal.freedKvTokens(),
                proposal.endpointId());
    }

    private CompletableFuture<PreemptionResult> prepareReturnedPreemption(
            RequestContext context,
            DecodeEvictionProposal proposal,
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata) {
        var victims =
                proposal.victims().stream()
                        .map(
                                victim ->
                                        new DecodeResources.ReservationHandle(
                                                endpoint.getStatus().getGenerationId(),
                                                victim.requestId(),
                                                victim.reservationToken()))
                        .toList();
        var result =
                endpoint.beginReturnedPreemption(
                        nextToken(),
                        victims,
                        request.getRequestId(),
                        request.getHardKvTokens(),
                        request.getExpectedKvTokens(),
                        metadata.priority(),
                        context.data().getDecodeAdmissionCapacity());
        reportPreemptionTargetValidation("return", result.status());
        if (result.status() != DecodeResources.PreemptionBeginResult.SUCCESS) {
            return CompletableFuture.completedFuture(
                    new PreemptionResult(
                            null,
                            result.status()
                                    == DecodeResources.PreemptionBeginResult.ENDPOINT_RETIRED,
                            "return_" + result.status().name().toLowerCase(java.util.Locale.ROOT)));
        }
        reportCommittedEnginePreemption(context, proposal, false);
        recordDecodePlanObservability(context, proposal);
        return CompletableFuture.completedFuture(
                new PreemptionResult(
                        result.reservation(),
                        false,
                        "return_committed",
                        proposal.victims().stream().map(DecodeRequestView::requestId).toList()));
    }

    private void reportPreemptionTargetValidation(
            String mode, DecodeResources.PreemptionBeginResult result) {
        String reason =
                switch (result) {
                    case VICTIM_GONE -> "victim_state_changed";
                    case VICTIM_ALREADY_CLAIMED -> "victim_already_claimed";
                    case INVALID_PRIORITY -> "priority_not_preemptible";
                    default -> null;
                };
        if (reason != null) {
            reportTargetValidationFailure(mode, reason);
        }
    }

    private void reportTargetValidationFailure(String mode, String reason) {
        try {
            reporter.reportPreemptionTargetInvalid(mode, reason);
        } catch (RuntimeException failure) {
            Logger.warn("Preemption validation metrics failed", failure);
        }
    }

    private CompletableFuture<PreemptionResult> startEngineCancelPreemption(
            RequestContext ctx,
            DecodeEvictionProposal proposal,
            DecodeEndpoint endpoint,
            Request request,
            SchedulingMetadata metadata,
            PreemptionConfig policy) {
        var command =
                new PreemptionCommand(
                        endpoint,
                        request,
                        metadata,
                        proposal.victims(),
                        50L,
                        policy.getTimeoutMs(),
                        () ->
                                ctx.data()
                                        .getScheduler()
                                        .isAdmissionOpen(
                                                request.getRequestId(), ctx.data().getFuture()),
                        "preempted by higher-priority request " + ctx.data().getRequestId());
        reportCancelRequests(ctx, proposal);
        return preempt(ctx, command)
                .whenComplete(
                        (result, error) ->
                                report(
                                        ctx.data().getRequestId(),
                                        "engine preemption",
                                        () -> {
                                            if (error != null
                                                    || result == null
                                                    || result.controlFailure()) {
                                                reportCancelTimeout(ctx, proposal.endpointId());
                                            } else if (result.committed()) {
                                                reportCommittedEnginePreemption(
                                                        ctx, proposal, true);
                                                recordDecodePlanObservability(ctx, proposal);
                                            }
                                        }));
    }

    /**
     * Metrics never participate in the committed reservation handoff.
     */
    private void reportCommittedEnginePreemption(
            RequestContext ctx, DecodeEvictionProposal proposal, boolean cancelConfirmed) {
        report(
                ctx.data().getRequestId(),
                "committed preemption",
                () -> {
                    for (DecodeRequestView victim : proposal.victims()) {
                        String stage =
                                victim.phase() == DecodeTaskPhase.RUNNING
                                        ? "decode_running"
                                        : "decode_cancel";
                        reporter.reportVictim(
                                victim.priority(),
                                ctx.data().getPriority(),
                                stage,
                                proposal.evictionCase());
                        reporter.reportVictimKvTokens(victim.priority(), stage, victim.kvTokens());
                        if (cancelConfirmed) {
                            reporter.reportEngineCancel(
                                    CancelEvent.CONFIRM, proposal.endpointId(), victim.priority());
                        }
                    }
                    reporter.reportEviction(
                            EvictionEvent.COMMIT,
                            ctx.data().getPriority(),
                            proposal.evictionCase(),
                            "success");
                });
    }

    private void reportCancelRequests(RequestContext ctx, DecodeEvictionProposal proposal) {
        report(
                ctx.data().getRequestId(),
                "cancel requests",
                () -> {
                    for (DecodeRequestView victim : proposal.victims()) {
                        reporter.reportEngineCancel(
                                CancelEvent.REQUEST, proposal.endpointId(), victim.priority());
                        reporter.reportCancel(victim.priority(), "PRIORITY_PREEMPTED");
                    }
                });
    }

    private void reportCancelTimeout(RequestContext ctx, String endpointId) {
        report(
                ctx.data().getRequestId(),
                "cancel timeout",
                () -> {
                    reporter.reportEngineCancel(
                            CancelEvent.TIMEOUT, endpointId, ctx.data().getPriority());
                });
    }

    private static void report(String requestId, String operation, Runnable metrics) {
        try {
            metrics.run();
        } catch (Throwable failure) {
            Logger.warn(
                    "[decode-capacity] failed to report {}: request_id={}",
                    operation,
                    requestId,
                    failure);
        }
    }

    private DecodeResources.ReservationHandle replaceQueuedDecodeReservations(
            DecodeEndpoint endpoint,
            List<DecodeResources.ReservationHandle> victims,
            Request request,
            int priority,
            AdmissionCapacity capacity) {
        List<AdmissionHandle> claimed = new ArrayList<>(victims.size());
        DecodeResources.ReservationHandle incoming = null;
        Runnable releaseOnPlacementFailure = null;
        try {
            for (DecodeResources.ReservationHandle victim : victims) {
                var owner = requests.ownerOf(victim.requestId());
                AdmissionHandle withdrawal =
                        owner instanceof QueuedRequestScheduler queue
                                ? queue.claimQueuedRoute(endpoint, victim, priority)
                                : null;
                if (withdrawal == null) {
                    return null;
                }
                claimed.add(withdrawal);
            }
            final var reserved =
                    endpoint.replaceQueuedRequests(
                            victims,
                            request.getRequestId(),
                            request.getHardKvTokens(),
                            request.getExpectedKvTokens(),
                            priority,
                            capacity);
            incoming = reserved;
            if (reserved != null) {
                AtomicBoolean released = new AtomicBoolean();
                releaseOnPlacementFailure =
                        () -> {
                            if (!released.compareAndSet(false, true)) {
                                return;
                            }
                            var status =
                                    endpoint.release(
                                                    reserved,
                                                    DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                                            .status();
                            if (status != DecodeResources.ReservationReleaseResult.RELEASED
                                    && status != DecodeResources.ReservationReleaseResult.STALE) {
                                throw new IllegalStateException(
                                        "failed placement retained Decode resources");
                            }
                        };
                for (AdmissionHandle withdrawal : claimed) {
                    withdrawal.onPlacementFailure(releaseOnPlacementFailure);
                }
            }
            return incoming;
        } finally {
            Throwable failure = null;
            for (AdmissionHandle withdrawal : claimed) {
                boolean committed = incoming != null;
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        withdrawal.complete(
                                                committed
                                                        ? org.flexlb.balance.scheduler.ContextResult
                                                                .PublicationResult.WITHDRAWN
                                                        : org.flexlb.balance.scheduler.ContextResult
                                                                .PublicationResult.RESTORED));
            }
            if (failure != null) {
                if (releaseOnPlacementFailure != null) {
                    failure = Failures.run(failure, releaseOnPlacementFailure);
                }
                Failures.rethrow(failure, "request cleanup failed");
            }
        }
    }
}
