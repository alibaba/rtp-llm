package org.flexlb.balance.schedule;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.RequestEventType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.endpoint.WorkerRequestEvent;

/**
 * Builds role observations and remote-cleanup replies for request lifecycle scenarios.
 */
public final class WorkerEventTestSupport {
    static void publish(
            AbstractRequestScheduler scheduler, RequestContext context, RequestEvent event) {
        if (event.type() == RequestEventType.REMOTE_CLEANED
                || event.type() == RequestEventType.REMOTE_FENCED) {
            if (record(context, event)) scheduler.processPendingRequestActions(context);
        } else scheduler.onWorkerRequestEvent(event.route(), event.workerEvent());
    }

    static boolean record(RequestContext context, RequestEvent event) {
        if (event.type() == RequestEventType.REMOTE_CLEANED
                || event.type() == RequestEventType.REMOTE_FENCED)
            return context.recordRemoteCleanup(
                    event.route(), event.type() == RequestEventType.REMOTE_CLEANED, event.nowMs());
        return context.recordWorkerEvent(event.route(), event.workerEvent());
    }

    /**
     * Builds role feedback from a test route, or a route-wide remote cleanup reply.
     */
    public record RequestEvent(
            RequestRoute route,
            DecodeEndpoint.RequestRegistration reservation,
            RoleType role,
            WorkerEndpoint endpoint,
            RequestEventType type,
            long errorCode,
            String detail,
            long nowMs) {
        WorkerRequestEvent workerEvent() {
            var registration =
                    role == RoleType.DECODE
                            ? reservation
                            : route == null
                                    ? null
                                    : role == RoleType.ENCODER
                                            ? route.encoderRegistration()
                                            : route.prefillRegistration();
            var source = endpoint;
            if (source == null
                    && route != null
                    && type != RequestEventType.ENDPOINT_CLOSED
                    && role != null)
                source =
                        switch (role) {
                            case DECODE -> route.decodeEndpoint();
                            case ENCODER -> route.encoderEndpoint();
                            case PREFILL, PDFUSION -> route.prefillEndpoint();
                            default -> null;
                        };
            return new WorkerRequestEvent(
                    registration, role, source, type, errorCode, detail, nowMs);
        }
    }
}
