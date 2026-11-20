package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;

/**
 * Reports one role's event for an exact endpoint and resource registration.
 */
public record WorkerRequestEvent(
        WorkerEndpoint.RequestRegistration registration,
        RoleType role,
        WorkerEndpoint endpoint,
        RequestEventType type,
        long errorCode,
        String detail,
        long nowMs) {}
