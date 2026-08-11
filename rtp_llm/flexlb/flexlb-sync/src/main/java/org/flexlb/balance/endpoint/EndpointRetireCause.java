package org.flexlb.balance.endpoint;

/** Why an endpoint generation left the READY registry. */
public enum EndpointRetireCause {
    HEALTH_CHECK_FAILED,
    DISCOVERY_REMOVED,
    STATUS_EXPIRED,
    GENERATION_REPLACED,
    REGISTRY_CLOSED,
    MANUAL
}
