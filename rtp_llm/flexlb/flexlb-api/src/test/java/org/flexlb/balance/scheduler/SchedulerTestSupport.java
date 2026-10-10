package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.WeakHashMap;

import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Builds and inspects request ownership for API tests. */
public final class SchedulerTestSupport {
    /** Read shadow ownership for assertions, without granting a production lookup capability. */
    public static DecodeResources.ReservationHandle decodeReservation(DecodeEndpoint endpoint, long requestId) {
        return endpoint.isRetired() ? null : decodeReservation(endpoint.resourceSnapshot(), requestId);
    }

    private static DecodeResources.ReservationHandle decodeReservation(
            DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(requestId);
        if (request == null || request.reservationToken() <= 0L) { return null; }
        return switch (request.phase()) {
            case LOCAL_RESERVED, MASTER_QUEUED_NOT_DISPATCHED, ENGINE_MAY_HAVE_SEEN ->
                    new DecodeResources.ReservationHandle(
                            snapshot.routing().generationId(), requestId, request.reservationToken());
            default -> null;
        };
    }

    private static final Map<AbstractRequestScheduler, RequestRepository> mockRepositories = new WeakHashMap<>();
    public static synchronized RequestRepository repository(AbstractRequestScheduler owner) {
        if (owner == null) { return null; }
        return owner.requests != null ? owner.requests : mockRepositories.computeIfAbsent(owner, SchedulerTestSupport::mockRepository);
    }
    private static RequestRepository mockRepository(AbstractRequestScheduler owner) {
        var repository = mock(RequestRepository.class);
        when(repository.ownerOf(anyLong())).thenReturn(owner);
        when(repository.findActive(anyLong())).thenAnswer(call -> {
            var context = mock(RequestContext.class);
            var data = mock(RequestData.class);
            when(context.data()).thenReturn(data);
            when(data.getScheduler()).thenReturn(owner);
            return context;
        });
        return repository;
    }
    public static void bindOwner(RequestContext context, AbstractRequestScheduler owner) {
        if (context.data().getScheduler() == null) {
            // Minimal fixtures may bind an owner without registering a runnable request.
            ReflectionTestUtils.setField(context.data(), "scheduler", owner);
        }
    }
}
