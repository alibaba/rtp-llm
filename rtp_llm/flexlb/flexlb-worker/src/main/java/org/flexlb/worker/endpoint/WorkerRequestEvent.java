package org.flexlb.worker.endpoint;

/**
 * Reports an event for an exact route, endpoint generation and resource registration.
 */
public record WorkerRequestEvent(
        WorkerRoute route,
        org.flexlb.worker.endpoint.DecodeEndpoint.RequestRegistration reservation,
        org.flexlb.config.service.RoleType role,
        org.flexlb.worker.endpoint.WorkerEndpoint endpoint,
        RequestEventType type,
        long errorCode,
        String detail,
        long nowMs) {}
