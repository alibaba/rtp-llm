package org.flexlb.balance.scheduler;

/** Runs request work asynchronously. Request and endpoint owners enforce ordering. */
@FunctionalInterface
public interface RequestTaskExecutor {
    void execute(RequestContext request, Runnable task);
}
