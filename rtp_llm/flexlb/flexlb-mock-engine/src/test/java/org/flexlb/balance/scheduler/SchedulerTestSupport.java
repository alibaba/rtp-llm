package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * Builds and inspects exact Decode reservations for mock-engine tests.
 */
public final class SchedulerTestSupport {
    /**
     * Build a real local reservation without publishing it to the master queue.
     */
    public static DecodeResources.RequestReservation reserveUnqueuedDecode(
            DecodeEndpoint endpoint,
            WorkerEndpoint selectedEndpoint,
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        var state = endpoint;
        var lock = org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(state);
        lock.lock();
        try {
            if (!Boolean.TRUE.equals(
                    ReflectionTestUtils.invokeMethod(
                            state, "requestIdAvailable", String.valueOf(requestId)))) {
                throw new IllegalStateException("Decode request id is already owned: " + requestId);
            }
            DecodeResources.Acquisition<
                            DecodeResources.ReserveStatus, DecodeResources.RequestReservation>
                    acquired =
                            ReflectionTestUtils.invokeMethod(
                                    state,
                                    "createReservation",
                                    String.valueOf(requestId),
                                    hardKv,
                                    expectedKv,
                                    priority,
                                    org.flexlb.enums.DecodeTaskPhase.LOCAL_RESERVED);
            return acquired.value();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Read shadow ownership for assertions, without granting a production lookup capability.
     */
    public static DecodeResources.RequestReservation decodeReservation(
            DecodeEndpoint endpoint, long requestId) {
        return endpoint.isRetired()
                ? null
                : decodeReservation(endpoint.resourceSnapshot(), requestId);
    }

    private static DecodeResources.RequestReservation decodeReservation(
            DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(String.valueOf(requestId));
        if (request == null || request.reservationToken() <= 0L) {
            return null;
        }
        return switch (request.phase()) {
            case LOCAL_RESERVED, MASTER_QUEUED_NOT_DISPATCHED, ENGINE_MAY_HAVE_SEEN ->
                    new DecodeResources.RequestReservation(
                            snapshot.routing().generationId(),
                            String.valueOf(requestId),
                            request.reservationToken());
            default -> null;
        };
    }

    public static PlacementOperation claim(
            AbstractRequestScheduler scheduler,
            String id,
            java.util.concurrent.CompletableFuture<?> future) {
        var context = scheduler.requests.findActive(id);
        return context != null && context.data().getFuture() == future
                ? scheduler.claimPlacement(context)
                : null;
    }
}
