package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/** Exact one-shot ownership of one asynchronous request admission mutation. */
public final class AdmissionMutation implements AutoCloseable {

    private final AtomicBoolean resolved = new AtomicBoolean();
    private final BiConsumer<AdmissionMutation, Response> termination;
    private final Consumer<AdmissionMutation> completion;

    AdmissionMutation(
            BiConsumer<AdmissionMutation, Response> termination,
            Consumer<AdmissionMutation> completion) {
        this.termination = Objects.requireNonNull(termination, "termination");
        this.completion = Objects.requireNonNull(completion, "completion");
    }

    /** Transfer this exact mutation to canonical terminal ownership. */
    public void terminate(Response failure) {
        if (resolved.compareAndSet(false, true)) {
            termination.accept(this, failure);
        }
    }

    /** Complete a successful or side-effect-free mutation. */
    @Override
    public void close() {
        if (resolved.compareAndSet(false, true)) {
            completion.accept(this);
        }
    }
}
