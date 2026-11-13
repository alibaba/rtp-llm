package org.flexlb.balance.schedule;

import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.WorkerEndpoint;

/**
 * Builds and inspects exact Decode reservations for mock-engine tests.
 */
public final class SchedulerTestSupport {
    /**
     * Build a real local reservation without publishing it to the master queue.
     */
    public static DecodeEndpoint.RequestRegistration registerDecode(
            DecodeEndpoint endpoint, long requestId, long hardKv, long expectedKv, int priority) {
        var acquired =
                endpoint.tryRegisterSelection(
                        endpoint.selectionSnapshot(),
                        String.valueOf(requestId),
                        hardKv,
                        expectedKv,
                        priority,
                        new DecodeResources.AdmissionCapacity(0L, 100L));
        org.junit.jupiter.api.Assertions.assertEquals(
                WorkerEndpoint.RegistrationStatus.REGISTERED, acquired.status());
        return acquired.value();
    }

    /**
     * Read shadow ownership for assertions, without granting a production lookup capability.
     */
    public static DecodeEndpoint.RequestRegistration decodeRegistration(
            DecodeEndpoint endpoint, long requestId) {
        return endpoint.hasStoppedAcceptingRequests()
                ? null
                : decodeRegistration(endpoint.resourceSnapshot(), requestId);
    }

    public static SelectionAttempt claim(
            AbstractRequestScheduler scheduler,
            String id,
            java.util.concurrent.CompletableFuture<?> future) {
        var context = scheduler.repository.findActive(id);
        return context != null && context.getFuture() == future
                ? scheduler.tryBeginRouteSelection(context)
                : null;
    }

    private static DecodeEndpoint.RequestRegistration decodeRegistration(
            DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(String.valueOf(requestId));
        if (request == null) return null;
        return switch (request.phase()) {
            case LOCAL_RESERVED, MASTER_QUEUED_NOT_DISPATCHED, ENGINE_MAY_HAVE_SEEN ->
                    request.reservation();
            default -> null;
        };
    }
}
