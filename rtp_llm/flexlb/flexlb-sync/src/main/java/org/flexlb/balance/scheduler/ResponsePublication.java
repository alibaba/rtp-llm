package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.scheduler.ContextResult.End;
import static org.flexlb.balance.scheduler.ContextResult.Granted;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.ResponseCandidate;
import static org.flexlb.util.Failures.recover;

/** One response execution owner, registered before Context freezes the selected outcome. */
final class ResponsePublication implements AutoCloseable {
    private enum Phase {
        RESERVED,
        SUBMITTED,
        RUNNING,
        COMPLETED,
        CLOSED
    }

    private final AtomicReference<Phase> phase = new AtomicReference<>(Phase.RESERVED);
    private final ResponseCompletionExecutor executor;
    final RequestContext owner;
    final boolean delivery;

    ResponsePublication(
            ResponseCompletionExecutor executor, RequestContext owner, boolean delivery) {
        this.owner = Objects.requireNonNull(owner);
        this.delivery = delivery;
        this.executor = executor;
        checkState(executor.register(this), "frontend publication is closed");
    }

    boolean publish(
            ResponseCandidate candidate,
            RequestFuture.Completion completion,
            Response response,
            Throwable failure,
            boolean synchronous,
            DeliveryMetricsReporter deliveryReporter) {
        var ctx = this.owner;
        var task =
                (java.util.function.BooleanSupplier)
                        () -> {
                            if (this.delivery && ctx.data().getAckAtMs() > 0L) {
                                var delivery = ctx.snapshot().delivery();
                                try {
                                    deliveryReporter.reportLatency(
                                            DeliveryMetricsReporter.Latency.DISPATCH_ACK,
                                            RoleType.PREFILL.name(),
                                            delivery.identity().item().prefillEp().getIp(),
                                            Math.max(
                                                    0L,
                                                    ctx.data().getAckAtMs()
                                                            - delivery.startedAtMs()));
                                } catch (Throwable ignored) {
                                }
                            }
                            var selected = ctx.selectResponse(candidate);
                            return selected instanceof Granted<ResponseCandidate>
                                    && this.complete(completion, response, failure);
                        };
        if (synchronous) {
            return this.completeNow(task);
        }
        this.submit(task);
        return true;
    }

    static ResponsePublication reserve(
            ResponseCompletionExecutor executor,
            RequestContext ctx,
            boolean delivery,
            java.util.function.Consumer<Throwable> failures) {
        var publication = recover(failures, () -> new ResponsePublication(executor, ctx, delivery));
        if (publication == null) throw new IllegalStateException("frontend publication is closed");
        return publication;
    }

    static Response response(End outcome, RequestRoute route, boolean batched) {
        if (outcome.outcome().phase() == RequestState.Phase.COMPLETED && route != null) {
            return route.successResponse(batched);
        }
        return outcome.errorType() == StrategyErrorType.RESOURCE_EXHAUSTED
                ? Response.error(
                        outcome.errorType(),
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        outcome.detail())
                : Response.buildErrorResponse(outcome.errorType(), outcome.detail());
    }

    void publishAfterFailure(
            Throwable failed,
            ResponseCandidate candidate,
            RequestFuture.Completion completion,
            Response response,
            Throwable failure,
            boolean synchronous,
            DeliveryMetricsReporter reporter) {
        if (hasCompletionOwner() || candidate == null) return;
        Failures.run(
                failed,
                () -> {
                    try (var fallback = new ResponsePublication(executor, owner, delivery)) {
                        fallback.publish(
                                candidate, completion, response, failure, synchronous, reporter);
                    }
                });
    }

    boolean hasCompletionOwner() {
        return switch (phase.get()) {
            case SUBMITTED, RUNNING, COMPLETED -> true;
            default -> false;
        };
    }

    boolean isSubmitted() {
        return phase.get() == Phase.SUBMITTED;
    }

    void submit(BooleanSupplier completion) {
        if (Thread.holdsLock(owner)) {
            close();
            throw new IllegalStateException("response submission under Context lock");
        }
        checkState(
                phase.compareAndSet(Phase.RESERVED, Phase.SUBMITTED),
                "response task already consumed");
        executor.submit(this, () -> run(Phase.SUBMITTED, completion));
    }

    boolean completeNow(BooleanSupplier completion) {
        if (Thread.holdsLock(owner)) {
            close();
            throw new IllegalStateException("response completion under Context lock");
        }
        return run(Phase.RESERVED, completion);
    }

    private boolean run(Phase expected, BooleanSupplier completion) {
        if (!phase.compareAndSet(expected, Phase.RUNNING)) {
            return false;
        }
        try {
            return executor.invoke(completion);
        } finally {
            phase.set(Phase.COMPLETED);
            executor.release(this);
        }
    }

    void reject(Throwable failure) {
        if (phase.compareAndSet(Phase.SUBMITTED, Phase.CLOSED)) {
            try {
                var data = owner.data();
                if (data != null && data.getFuture() instanceof RequestFuture future) {
                    future.completeExceptionallyOwned(failure);
                }
            } finally {
                executor.release(this);
            }
        }
    }

    boolean complete(RequestFuture.Completion kind, Response response, Throwable failure) {
        checkState(
                !Thread.holdsLock(owner),
                "response callbacks require the Context lock to be released");
        RequestFuture future = (RequestFuture) owner.data().getFuture();
        return switch (kind) {
            case RESPONSE -> future.completeOwned(response);
            case FAILURE -> future.completeExceptionallyOwned(failure);
            case CANCELLATION -> future.cancelOwned();
        };
    }

    @Override
    public void close() {
        if (phase.compareAndSet(Phase.RESERVED, Phase.CLOSED)) {
            executor.release(this);
        }
    }
}
