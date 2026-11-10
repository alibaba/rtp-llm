package org.flexlb.balance.preemption;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.scheduler.ContextResult;
import org.flexlb.balance.scheduler.DeliveryCoordinator;
import org.flexlb.balance.scheduler.PreemptionRegistration;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.TaskTimer;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.Objects;

import static org.flexlb.balance.scheduler.ContextResult.CleanupAcknowledgement;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.flexlb.balance.scheduler.ContextResult.Snapshot;
import static org.flexlb.util.Failures.recover;

/**
 * Applies cancel results to the exact preemption attempt and reconciles its Decode reservation.
 * Request updates and follow-up effects run within the scheduler's ordinary request work scope.
 */
public final class PreemptionCompletion {
    public PreemptionCompletion() {}

    /**
     * Waits for cancellation acknowledgement while retaining the observer for a late reply.
     */
    public java.util.concurrent.CompletableFuture<PreemptionRegistration.Reply> cancelRequest(
            DeliveryCoordinator dispatcher,
            TaskTimer timer,
            PreemptionRegistration registration,
            DecodeEndpoint endpoint,
            String detail,
            long timeoutMs) {
        var target = registration.cancelTarget();
        var answer = new java.util.concurrent.CompletableFuture<PreemptionRegistration.Reply>();
        var timeout =
                timer.schedule(
                        () -> answer.complete(PreemptionRegistration.Reply.UNKNOWN), timeoutMs);
        answer.whenComplete((ignored, failure) -> timeout.cancel(false));
        dispatcher
                .cancel(
                        registration.owner(),
                        registration,
                        target.prefillIp(),
                        target.prefillGrpcPort(),
                        org.flexlb.balance.scheduler.CancelReason.PRIORITY_PREEMPTED,
                        timeoutMs)
                .whenComplete(
                        (result, failure) -> {
                            if (failure != null) {
                                answer.complete(PreemptionRegistration.Reply.UNKNOWN);
                                return;
                            }
                            var reply = result.reply();
                            try {
                                var phase =
                                        failure != null || reply == null
                                                ? PreemptionCancelPhase.CANCEL_UNKNOWN
                                                : switch (reply) {
                                                    case ACCEPTED, REQUEST_FENCED ->
                                                            PreemptionCancelPhase.CANCEL_REQUESTED;
                                                    case NOT_FOUND ->
                                                            PreemptionCancelPhase.NOT_FOUND_STALE;
                                                    case FAILED ->
                                                            PreemptionCancelPhase.CANCEL_UNKNOWN;
                                                    case REQUEST_CLEANED -> null;
                                                };
                                boolean cleaned =
                                        failure == null
                                                && reply
                                                        == org.flexlb.engine.grpc.EngineGrpcClient
                                                                .CancelReply.REQUEST_CLEANED;
                                var event =
                                        new PreemptionCompletion.PreemptionEvent(
                                                phase,
                                                false,
                                                cleaned ? endpoint : null,
                                                cleaned ? registration.reservation() : null,
                                                cleaned ? detail : null,
                                                result.cleanup());
                                var progress = event.progress(applyEvent(registration, event));
                                answer.complete(progress);
                            } catch (RuntimeException | Error failed) {
                                registration
                                        .owner()
                                        .data()
                                        .getRequestScheduler()
                                        .recordRequestFailure(failed);
                                answer.complete(PreemptionRegistration.Reply.UNKNOWN);
                            }
                        });
        return answer;
    }

    public boolean applyEvent(PreemptionRegistration attempt, PreemptionEvent event) {
        var ctx = attempt.owner();
        var scheduler = ctx.data().getRequestScheduler();
        if (!scheduler.isCurrentRequest(ctx)) return false;
        boolean applied = apply(attempt, event, scheduler::recordRequestFailure);
        scheduler.onRequestStateChanged(ctx);
        return applied;
    }

    /**
     * The exact reconciliation owns ledger retries and its post-commit capacity/resumption effects.
     */
    public static void reconcile(
            RequestContext ctx, java.util.function.Consumer<Throwable> failureReporter) {
        ContextResult.Snapshot current;
        PreemptionRegistration attempt;
        DecodeEndpoint decode;
        boolean capacityChanged;
        EndEvidence terminal;
        boolean resumeDelivery;
        synchronized (ctx) {
            current = ctx.snapshot();
            if (current.preemption() == null) {
                return;
            }
            var claim = ctx.canReconcilePreemption(current.preemption().identity());
            if (!claim) {
                return;
            }
            attempt = current.preemption().identity();
            decode = current.route().decodeEndpoint();
            terminal = current.preemption().pendingTerminal();
            resumeDelivery =
                    terminal == null
                            && current.canConfirmDelivery()
                            && !current.delivery().acknowledged();
            capacityChanged =
                    decode != null && !(terminal != null && current.close().decodeSettled());
        }
        // Reconciliation is a two-phase handoff: the ledger mutation may block on Decode state,
        // while the Context lock is reserved only for the exact commit.
        var mutation =
                !capacityChanged
                        ? null
                        : recover(
                                failureReporter,
                                () ->
                                        decode.updatePreemption(
                                                attempt.attemptToken(),
                                                terminal == null
                                                        ? DecodeResources.PreemptionUpdate.active(
                                                                attempt.reservation())
                                                        : terminal.kind() == EndKind.PRIORITY
                                                                ? DecodeResources.PreemptionUpdate
                                                                        .canceled(
                                                                                attempt
                                                                                        .reservation())
                                                                : DecodeResources.PreemptionUpdate
                                                                        .finished(
                                                                                attempt
                                                                                        .reservation())));
        boolean committed = !capacityChanged || mutation != null && mutation.status().applied();
        boolean recorded;
        synchronized (ctx) {
            recorded =
                    ctx.completePreemption(
                            attempt,
                            new PreemptionReport(
                                    null,
                                    committed,
                                    committed && terminal != null,
                                    false,
                                    terminal),
                            System.currentTimeMillis());
            if (!committed) {
                return;
            }
        }
        if (capacityChanged) {
            recover(
                    failureReporter,
                    () -> decode.publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
        }
        if (recorded && resumeDelivery) {
            recover(
                    failureReporter,
                    () ->
                            attempt.complete(
                                    new PreemptionRequestResult(
                                            ctx.data().getRequestId(),
                                            PreemptionRequestResult.Outcome.DELIVERY_RESUMED)));
        }
    }

    private static boolean apply(
            PreemptionRegistration attempt,
            PreemptionEvent event,
            java.util.function.Consumer<Throwable> failureReporter) {
        var ctx = attempt.owner();
        boolean applied = false, recorded = false, committed = false;
        Snapshot before;
        DecodeEndpoint endpoint = null;
        synchronized (ctx) {
            before = ctx.snapshot();
            if (before.preemption() != null
                    && before.preemption().identity() == attempt
                    && (event.endpoint() == null
                            || before.route().decodeEndpoint() == event.endpoint()
                                    && Objects.equals(
                                            before.route().decodeReservation(),
                                            event.reservation()))) {
                endpoint = event.endpoint();
            }
        }
        // Decode owns its ledger lock. Keep the resource mutation outside the request lock;
        // the second Context section below commits it only if this exact attempt survived.
        if (endpoint != null) {
            var target = endpoint;
            var mutation =
                    recover(
                            failureReporter,
                            () ->
                                    target.updatePreemption(
                                            attempt.attemptToken(),
                                            DecodeResources.PreemptionUpdate.fenced(
                                                    event.reservation())));
            committed = mutation != null && mutation.status().applied();
        }
        synchronized (ctx) {
            var current = ctx.snapshot();
            if (current.preemption() != null
                    && current.preemption().identity() == attempt
                    && (endpoint == null
                            || current.route().decodeEndpoint() == endpoint
                                    && Objects.equals(
                                            current.route().decodeReservation(),
                                            event.reservation()))) {
                var terminal =
                        !committed
                                ? null
                                : new EndEvidence(
                                        before.route(),
                                        EndKind.PRIORITY,
                                        null,
                                        false,
                                        0L,
                                        StrategyErrorType.PRIORITY_PREEMPTED,
                                        event.detail(),
                                        false,
                                        true);
                recorded =
                        ctx.completePreemption(
                                attempt,
                                new PreemptionReport(
                                        event.endpoint() != null && !committed
                                                ? PreemptionCancelPhase.CANCEL_UNKNOWN
                                                : event.phase(),
                                        committed,
                                        committed,
                                        event.release(),
                                        terminal),
                                System.currentTimeMillis());
                applied = recorded && (endpoint == null || committed);
            }
            // A stale attempt must not discard a current delivery's cleanup proof.
            if (event.cleanup() != null) {
                var previous = ctx.snapshot();
                var next = ctx.recordRemoteCleanup(event.cleanup());
                recorded |= next != null && next != previous;
            }
        }
        if (committed) {
            var committedEndpoint = endpoint;
            recover(
                    failureReporter,
                    () -> committedEndpoint.publishCapacity(DecodeEndpoint.CapacityEdge.BOTH));
        }
        return applied;
    }

    /**
     * Carries progress and exact cleanup evidence for one preemption attempt.
     */
    public record PreemptionEvent(
            PreemptionCancelPhase phase,
            boolean release,
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation reservation,
            String detail,
            CleanupAcknowledgement cleanup) {
        public PreemptionEvent(
                PreemptionCancelPhase phase,
                boolean release,
                DecodeEndpoint endpoint,
                DecodeResources.RequestReservation reservation,
                String detail) {
            this(phase, release, endpoint, reservation, detail, null);
        }

        public PreemptionRegistration.Reply progress(boolean applied) {
            return !applied
                    ? PreemptionRegistration.Reply.UNKNOWN
                    : phase == PreemptionCancelPhase.NOT_FOUND_STALE
                            ? PreemptionRegistration.Reply.NOT_FOUND
                            : PreemptionRegistration.Reply.WAITING;
        }
    }
}
