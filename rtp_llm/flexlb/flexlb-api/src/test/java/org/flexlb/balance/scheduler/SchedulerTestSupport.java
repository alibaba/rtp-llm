package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.WeakHashMap;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Builds and inspects request ownership for API tests.
 */
public final class SchedulerTestSupport {
    private static final Map<AbstractRequestScheduler, RequestRepository> mockRepositories =
            new WeakHashMap<>();

    /**
     * Read shadow ownership for assertions, without granting a production lookup capability.
     */
    public static DecodeResources.RequestReservation decodeReservation(
            DecodeEndpoint endpoint, long requestId) {
        return endpoint.isRetired()
                ? null
                : decodeReservation(endpoint.resourceSnapshot(), requestId);
    }

    public static synchronized RequestRepository repository(AbstractRequestScheduler owner) {
        if (owner == null) {
            return null;
        }
        return owner.repository != null
                ? owner.repository
                : mockRepositories.computeIfAbsent(owner, SchedulerTestSupport::mockRepository);
    }

    public static void bindOwner(RequestContext context, AbstractRequestScheduler owner) {
        if (context.data().getRequestScheduler() == null) {
            // Minimal fixtures may bind an owner without registering a runnable request.
            ReflectionTestUtils.setField(context.data(), "requestScheduler", owner);
        }
    }

    public static RoutingContext claim(
            AbstractRequestScheduler scheduler,
            String id,
            java.util.concurrent.CompletableFuture<?> future) {
        var context = scheduler.repository.findActive(id);
        return context != null && context.data().getFuture() == future
                ? scheduler.tryBeginRouteSelection(context)
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

    private static RequestRepository mockRepository(AbstractRequestScheduler owner) {
        var repository = mock(RequestRepository.class);
        when(repository.ownerOf(org.mockito.ArgumentMatchers.anyString())).thenReturn(owner);
        when(repository.findActive(org.mockito.ArgumentMatchers.anyString()))
                .thenAnswer(
                        call -> {
                            var context = mock(RequestContext.class);
                            var data = mock(RequestData.class);
                            when(context.data()).thenReturn(data);
                            when(data.getRequestScheduler()).thenReturn(owner);
                            return context;
                        });
        return repository;
    }
}
