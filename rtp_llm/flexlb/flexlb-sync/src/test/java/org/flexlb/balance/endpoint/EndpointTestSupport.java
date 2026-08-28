package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Supplier;

/** Shared fixtures which exercise only the frozen endpoint-facing ports. */
public final class EndpointTestSupport {

    private EndpointTestSupport() {
    }

    /** Test-only access to the real generation handoff used by PrefillState. */
    public static PrefillState.ReservationResult<PrefillState.RouteReservation>
            reserveRoute(
            PrefillState state,
            ScheduledRequest item,
            long predictedMs,
            int maximumRequests) {
        EndpointGenerationLifecycle lifecycle =
                new EndpointGenerationLifecycle(() -> { });
        EndpointGenerationLifecycle.HandoffPermit permit =
                lifecycle.tryAcquireHandoff();
        PrefillState.ReservationResult<PrefillState.RouteReservation> result =
                state.reserveRoute(
                        item, predictedMs, maximumRequests, permit);
        if (result.reservation() == null) {
            permit.close();
        }
        return result;
    }

    static EndpointEventProjector noopEventSink() {
        return org.mockito.Mockito.mock(EndpointEventProjector.class);
    }

    static WorkerStatus workerStatus(
            RoleType role,
            String ip,
            int port,
            int grpcPort) {
        return WorkerStatus.createDiscovered(
                role, null, ip, port, grpcPort, null);
    }

    static WorkerStatus workerStatus(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site) {
        return WorkerStatus.createDiscovered(
                role, group, ip, port, grpcPort, site);
    }

    static Runnable applyStatus(
            WorkerEndpoint endpoint,
            WorkerStatusResponse response) {
        WorkerStatus status = endpoint.getStatus();
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            Runnable projection =
                    endpoint.applyPreparedStatus(status, prepared);
            status.recordSuccessfulPoll(response.isAlive());
            return projection;
        } finally {
            status.lock.unlock();
        }
    }

    static void publishStatus(
            WorkerStatus status,
            WorkerStatusResponse response) {
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry,
            RoleType role,
            String address,
            WorkerStatus status) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        return publishEndpoint(registry, role, address, status, response);
    }

    static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry,
            RoleType role,
            String address,
            WorkerStatus status,
            WorkerStatusResponse response) {
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            return registry.publishPreparedEndpoint(
                    address, status, prepared).endpoint();
        } finally {
            status.lock.unlock();
        }
    }

    private static void prepareResponse(
            WorkerStatus status,
            WorkerStatusResponse response) {
        // This helper models a successful status poll for an active generation.
        // Retirement tests must exercise the explicit dead-status path instead.
        response.setAlive(true);
        if (response.getRole() == null) {
            response.setRole(status.getRole());
        }
        long committedVersion =
                status.appliedStatusCursor().statusVersion();
        Long responseVersion = response.getStatusVersion();
        if (responseVersion == null
                || responseVersion <= 0L
                || responseVersion <= committedVersion) {
            response.setStatusVersion(Math.max(
                    1L, committedVersion + 1L));
        }
        if (response.getLatestFinishedVersion() == null) {
            response.setLatestFinishedVersion(
                    status.appliedStatusCursor()
                            .latestFinishedTaskVersion());
        }
    }

    static TestRequestRuntime requestRuntime() {
        return new TestRequestRuntime();
    }

    static DeliveryStrategy routeStrategy(TestRequestRuntime runtime) {
        RouteDeliveryStrategy route = new RouteDeliveryStrategy(
                runtime.requests(), NOOP_TELEMETRY);
        DeliveryStrategy delivery = org.mockito.Mockito.mock(
                DeliveryStrategy.class);
        org.mockito.Mockito.when(delivery.projectionPolicy())
                .thenReturn(route.projectionPolicy());
        org.mockito.Mockito.when(delivery.projectGroupDurationMs(
                        org.mockito.Mockito.anyList(),
                        org.mockito.Mockito.any()))
                .thenAnswer(invocation -> route.projectGroupDurationMs(
                        invocation.getArgument(0), invocation.getArgument(1)));
        org.mockito.Mockito.when(delivery.prepare(
                        org.mockito.Mockito.anyList(),
                        org.mockito.Mockito.any(),
                        org.mockito.Mockito.any()))
                .thenAnswer(invocation -> {
                    List<ScheduledRequest> candidates = invocation.getArgument(0);
                    DeliveryStrategy.Transaction transaction =
                            org.mockito.Mockito.mock(
                                    DeliveryStrategy.Transaction.class);
                    org.mockito.Mockito.when(transaction.items())
                            .thenReturn(List.of());
                    org.mockito.Mockito.when(transaction.blockedItem())
                            .thenReturn(candidates.getFirst());
                    org.mockito.Mockito.when(transaction.blockedResult())
                            .thenReturn(PARKED_BOUNDARY);
                    return transaction;
                });
        return delivery;
    }

    static DeliveryStrategy liveRouteStrategy(TestRequestRuntime runtime) {
        return new RouteDeliveryStrategy(
                runtime.requests(), NOOP_TELEMETRY);
    }

    private static final CapacityBoundary PARKED_BOUNDARY =
            CapacityBoundary.unavailable(
                    new CapacityBoundary.Availability() {
                        @Override
                        public boolean isAvailable() {
                            return false;
                        }

                        @Override
                        public void addListener(Runnable listener) {
                        }

                        @Override
                        public void removeListener(Runnable listener) {
                        }
                    },
                    new RouteProjection.AdmissionBlockSemantics(
                            "test delivery parked",
                            RouteProjection.AfterProbeAdmission.BLOCKED,
                            "test delivery parked",
                            RoleType.PREFILL));

    static boolean offer(PrefillEndpoint endpoint, ScheduledRequest item) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                return false;
            }
            return endpoint.offerPinned(pin, item);
        }
    }

    static PrefillState.DirectRegistration registerDirect(
            PrefillEndpoint endpoint,
            long requestId,
            long predictedMs) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new IllegalStateException("endpoint is retired");
            }
            return endpoint.registerDirectRequest(pin, requestId, predictedMs);
        }
    }

    static PrefillState.CommittedHandoff commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends ScheduledRequest> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("batch requires at least one item");
        }
        List<ScheduledRequest> items = List.copyOf(exactItems);
        PrefillState.ReservationResult<PrefillState.BatchReservation> result = endpoint.reserveBatch(
                items.get(0), batchId, Integer.MAX_VALUE);
        if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
            throw new IllegalStateException(
                    "batch reservation rejected: " + result.status());
        }
        try (PrefillState.BatchReservation reservation =
                     result.reservation()) {
            return reservation.commit(items, predictedMs);
        }
    }

    static List<PrefillState.CommittedHandoff> commitRoutes(
            PrefillEndpoint endpoint,
            long predictedMs,
            List<? extends ScheduledRequest> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("route group requires an item");
        }
        List<ScheduledRequest> items = List.copyOf(exactItems);
        List<PrefillState.RouteReservation> reservations =
                new ArrayList<>(items.size());
        boolean committed = false;
        try {
            for (ScheduledRequest item : items) {
                PrefillState.ReservationResult<PrefillState.RouteReservation> result =
                        endpoint.reserveRoute(
                                item, predictedMs, Integer.MAX_VALUE);
                if (result.status()
                        != PrefillState.CapacityStatus.ACQUIRED) {
                    throw new IllegalStateException(
                            "route reservation rejected: " + result.status());
                }
                reservations.add(result.reservation());
            }
            List<PrefillState.CommittedHandoff> handoffs;
            handoffs = reservations.get(0).commitGroup(items, reservations);
            committed = true;
            return handoffs;
        } finally {
            if (!committed) {
                for (int index = reservations.size() - 1;
                     index >= 0; index--) {
                    reservations.get(index).close();
                }
            }
        }
    }

    static final DeliveryMetrics NOOP_TELEMETRY =
            org.mockito.Mockito.mock(DeliveryMetrics.class);

    static class TestRequestRuntime {
        private final RequestRegistry requests =
                org.mockito.Mockito.mock(RequestRegistry.class);
        private final EndpointEventProjector events =
                org.mockito.Mockito.mock(EndpointEventProjector.class);
        private final List<PrefillRetirement> prefillRetirements =
                new CopyOnWriteArrayList<>();
        private final List<ScheduledRequest> offerFailures =
                new CopyOnWriteArrayList<>();
        TestRequestRuntime() {
            org.mockito.Mockito.doAnswer(invocation -> Optional.ofNullable(
                    ((Supplier<?>) invocation.getArgument(1)).get()))
                    .when(requests).prepareIfOwned(
                            org.mockito.Mockito.any(),
                            org.mockito.Mockito.any());
            org.mockito.Mockito.doAnswer(invocation -> {
                if (!((java.util.function.BooleanSupplier)
                        invocation.getArgument(1)).getAsBoolean()) {
                    return null;
                }
                RequestRegistry.DeliveryClaim claim = org.mockito.Mockito.mock(
                        RequestRegistry.DeliveryClaim.class);
                org.mockito.Mockito.when(claim.item()).thenReturn(
                        invocation.getArgument(0));
                return claim;
            }).when(requests).tryClaimRouteDelivery(
                    org.mockito.Mockito.any(), org.mockito.Mockito.any());
            org.mockito.Mockito.doAnswer(invocation -> {
                onCompleted(invocation.getArgument(0),
                        invocation.getArgument(1));
                return null;
            }).when(requests).complete(
                    org.mockito.Mockito.any(), org.mockito.Mockito.any());
            org.mockito.Mockito.doAnswer(invocation -> {
                onQueueOfferFailure(
                        invocation.getArgument(0), invocation.getArgument(1));
                return null;
            }).when(events).onQueueOfferFailure(
                    org.mockito.Mockito.any(), org.mockito.Mockito.any());
            org.mockito.Mockito.doAnswer(invocation -> {
                prefillRetirements.add(new PrefillRetirement(
                        invocation.getArgument(0), invocation.getArgument(1)));
                return null;
            }).when(events).onPrefillGenerationRetired(
                    org.mockito.Mockito.any(), org.mockito.Mockito.anyList());
        }

        RequestRegistry requests() {
            return requests;
        }

        EndpointEventProjector events() {
            return events;
        }

        void onCompleted(
                RequestRegistry.DeliveryClaim claim,
                DeliveryResult result) {
        }

        void onQueueOfferFailure(
                ScheduledRequest item,
                Throwable error) {
            offerFailures.add(item);
        }

        List<PrefillRetirement> prefillRetirements() {
            return List.copyOf(prefillRetirements);
        }

        List<ScheduledRequest> offerFailures() {
            return List.copyOf(offerFailures);
        }

    }

    record PrefillRetirement(
            PrefillEndpoint endpoint,
            List<ScheduledRequest> ownedItems) {
        PrefillRetirement {
            ownedItems = List.copyOf(ownedItems);
        }
    }

}
