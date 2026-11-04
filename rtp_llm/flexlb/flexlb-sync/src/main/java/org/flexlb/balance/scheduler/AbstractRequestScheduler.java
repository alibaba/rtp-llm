package org.flexlb.balance.scheduler;

/** Shared request lifecycle wiring; subclasses own when and how scheduling runs. */
public abstract class AbstractRequestScheduler implements RequestScheduler {
    protected final RequestLifecycle lifecycle;
    protected final RequestRepository requests;

    protected AbstractRequestScheduler(RequestLifecycle lifecycle) {
        this.lifecycle = lifecycle;
        lifecycle.bindScheduler(this);
        this.requests = lifecycle.requests;
    }

    protected final boolean checkRequestMode(RequestContext context, boolean queued) {
        if (context.data().getConfig().isQueue() == queued) return true;
        lifecycle.publish(
                context,
                org.flexlb.dao.loadbalance.Response.buildErrorResponse(
                        org.flexlb.dao.loadbalance.StrategyErrorType.DISPATCH_FAILED,
                        "request configuration does not match scheduler mode"),
                true);
        return false;
    }
}
