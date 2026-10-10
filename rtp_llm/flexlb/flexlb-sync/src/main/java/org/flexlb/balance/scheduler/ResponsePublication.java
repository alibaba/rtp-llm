package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkState;

/** One accepted response task. State selection belongs to Context; execution belongs to this registration. */
final class ResponsePublication implements AutoCloseable {
    private final ResponseCompletionExecutor.CompletionRegistration registration;
    final RequestContext owner;
    final boolean delivery;
    private final AtomicBoolean used = new AtomicBoolean();

    ResponsePublication(SchedulerRuntime runtime, RequestContext owner, boolean delivery) {
        this.owner = Objects.requireNonNull(owner);
        this.delivery = delivery;
        registration = runtime.registerResponseCompletion();
        checkState(registration != null, "frontend publication is closed");
    }

    void use() {
        checkState(used.compareAndSet(false, true), "response task already consumed");
    }

    boolean hasCompletionOwner() { return registration.hasCompletionOwner(); }

    void submit(Supplier<Boolean> completion) {
        if (Thread.holdsLock(owner)) { close(); throw new IllegalStateException("response submission under Context lock"); }
        use();
        registration.submit(completion::get);
    }

    boolean completeNow(Supplier<Boolean> completion) {
        if (Thread.holdsLock(owner)) { close(); throw new IllegalStateException("response completion under Context lock"); }
        use();
        return registration.completeNow(completion::get);
    }

    boolean complete(RequestFuture.Completion kind, Response response, Throwable failure) {
        checkState(!Thread.holdsLock(owner), "response callbacks require the Context lock to be released");
        RequestFuture future = (RequestFuture) owner.data().getFuture();
        return switch (kind) {
            case RESPONSE -> future.completeOwned(response);
            case FAILURE -> future.completeExceptionallyOwned(failure);
            case CANCELLATION -> future.cancelOwned();
        };
    }

    @Override public void close() {
        if (used.compareAndSet(false, true)) { registration.close(); }
    }
}
