package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;

/**
 * Builds and inspects exact Decode reservations for mock-engine tests.
 */
public final class SchedulerTestSupport {
    /**
     * Build a real local reservation without publishing it to the master queue.
     */
    public static DecodeResources.RequestReservation registerDecode(
            DecodeEndpoint endpoint, long requestId, long hardKv, long expectedKv, int priority) {
        var acquired =
                endpoint.tryRegisterSelection(
                        endpoint.routingView(),
                        String.valueOf(requestId),
                        hardKv,
                        expectedKv,
                        priority,
                        new DecodeResources.AdmissionCapacity(0L, 100L));
        org.junit.jupiter.api.Assertions.assertEquals(
                DecodeResources.ReserveStatus.ACQUIRED, acquired.status());
        return acquired.value();
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

    public static PlacementOperation claim(
            AbstractRequestScheduler scheduler,
            String id,
            java.util.concurrent.CompletableFuture<?> future) {
        var context = scheduler.repository.findActive(id);
        return context != null && context.data().getFuture() == future
                ? scheduler.claimPlacement(context)
                : null;
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
}
