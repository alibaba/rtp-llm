package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

import static com.google.common.base.Preconditions.checkState;

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

    ResponsePublication(SchedulerRuntime runtime, RequestContext owner, boolean delivery) {
        this(runtime.responseCompletions(), owner, delivery);
    }

    ResponsePublication(
            ResponseCompletionExecutor executor, RequestContext owner, boolean delivery) {
        this.owner = Objects.requireNonNull(owner);
        this.delivery = delivery;
        this.executor = executor;
        checkState(executor.register(this), "frontend publication is closed");
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

    void reject() {
        if (phase.compareAndSet(Phase.SUBMITTED, Phase.CLOSED)) {
            executor.release(this);
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
