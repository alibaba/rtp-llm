package org.flexlb.balance.endpoint;

/** Lifecycle of one master-local endpoint generation. */
public enum EndpointLifecycleState {
    /** Published and eligible for new scheduling operations. */
    READY,
    /** Unpublished; draining operations and releasing master-owned state. */
    RETIRING,
    /** Retirement completed. This generation can never become READY again. */
    CLOSED
}
