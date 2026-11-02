package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;

/** Shared request lifecycle wiring; subclasses own when and how scheduling runs. */
public abstract class AbstractRequestScheduler
        implements RequestScheduler, RequestLifecycle.Scheduling {
    protected final RequestLifecycle lifecycle;
    protected final RequestRepository requests;
    protected final DecodeResources.AdmissionCapacity decodeAdmissionCapacity;

    protected AbstractRequestScheduler(RequestLifecycle lifecycle) {
        this.lifecycle = lifecycle;
        lifecycle.bindScheduler(this, this);
        this.requests = lifecycle.requests;
        this.decodeAdmissionCapacity = lifecycle.decodeAdmissionCapacity;
    }

}
