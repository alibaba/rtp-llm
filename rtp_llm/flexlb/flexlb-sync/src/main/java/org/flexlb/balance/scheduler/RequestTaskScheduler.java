package org.flexlb.balance.scheduler;

/** Request work capabilities without access to shared executor lifecycle. */
public interface RequestTaskScheduler {
    @FunctionalInterface
    interface Cancellation {
        void cancel();
    }

    RequestSignals signals();

    /** Register a one-shot timeout; the handle can only cancel this task. */
    Cancellation scheduleTimeout(Runnable work, long delayMs);

    /** Offload work; it may overlap other work for the exact request. Owners enforce dependencies. */
    void executeRequestWork(RequestContext context, Runnable work);
}
