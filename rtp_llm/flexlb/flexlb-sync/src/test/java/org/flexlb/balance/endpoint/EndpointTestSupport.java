package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.delivery.RouteDeliveryStrategy;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/** Shared fixtures which exercise only the frozen endpoint-facing ports. */
final class EndpointTestSupport {

    private EndpointTestSupport() {
    }

    static EndpointEventSink noopEventSink() {
        return org.mockito.Mockito.mock(EndpointEventSink.class);
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

    static EndpointStatusReduction applyStatus(
            WorkerEndpoint endpoint,
            WorkerStatusResponse response) {
        WorkerStatus status = endpoint.getStatus();
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            EndpointStatusReduction reduction =
                    endpoint.applyPreparedStatus(status, prepared);
            status.recordSuccessfulPoll(response.isAlive());
            return reduction;
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
        return routeStrategy(runtime, new ControllableAdmission(true));
    }

    static DeliveryStrategy routeStrategy(
            TestRequestRuntime runtime,
            ControllableAdmission admission) {
        return new RouteDeliveryStrategy(admission, runtime, NOOP_TELEMETRY);
    }

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
        PrefillState.BatchReservationResult result = endpoint.reserveBatch(
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
                PrefillState.RouteReservationResult result =
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

    static class TestRequestRuntime implements EndpointEventSink,
            SlotDeliveryPort {
        private final List<PrefillRetirement> prefillRetirements =
                new CopyOnWriteArrayList<>();
        private final List<ScheduledRequest> offerFailures =
                new CopyOnWriteArrayList<>();
        private final List<ScheduledRequest> deliveryFailures =
                new CopyOnWriteArrayList<>();
        private final AtomicInteger completedClaims = new AtomicInteger();

        List<PrefillRetirement> prefillRetirements() {
            return List.copyOf(prefillRetirements);
        }

        List<ScheduledRequest> offerFailures() {
            return List.copyOf(offerFailures);
        }

        List<ScheduledRequest> deliveryFailures() {
            return List.copyOf(deliveryFailures);
        }

        int completedClaimCount() {
            return completedClaims.get();
        }

        @Override
        public <T> Optional<T> prepareIfOwned(
                ScheduledRequest exactItem,
                Supplier<T> preparation) {
            return Optional.ofNullable(preparation.get());
        }

        @Override
        public Claim tryClaimForDelivery(
                ScheduledRequest exactItem,
                Identity identity,
                BooleanSupplier endpointHandoff) {
            TestClaim claim = new TestClaim(exactItem);
            return endpointHandoff.getAsBoolean() ? claim : null;
        }

        @Override
        public void complete(Claim exactClaim, Completion completion) {
            completedClaims.incrementAndGet();
        }

        @Override
        public void failPrepared(ScheduledRequest exactItem, Throwable cause) {
            deliveryFailures.add(exactItem);
        }

        @Override
        public void onQueuedItemExpired(ScheduledRequest exactItem) {
        }

        @Override
        public void onQueueOfferFailure(
                ScheduledRequest exactItem,
                Throwable cause) {
            offerFailures.add(exactItem);
        }

        @Override
        public void onPreparedDeliveryFailure(
                ScheduledRequest exactItem,
                Throwable cause) {
            deliveryFailures.add(exactItem);
        }

        @Override
        public void onStatusReduced(EndpointStatusReduction reduction) {
        }

        @Override
        public void onPrefillGenerationRetired(
                PrefillEndpoint endpoint,
                List<ScheduledRequest> ownedItems) {
            prefillRetirements.add(
                    new PrefillRetirement(endpoint, ownedItems));
        }

        @Override
        public void onDecodeGenerationRetired(
                DecodeEndpoint endpoint,
                List<DecodeEndpoint.ReservationHandle> ownedReservations) {
        }
    }

    record PrefillRetirement(
            PrefillEndpoint endpoint,
            List<ScheduledRequest> ownedItems) {
        PrefillRetirement {
            ownedItems = List.copyOf(ownedItems);
        }
    }

    private record TestClaim(ScheduledRequest item)
            implements SlotDeliveryPort.Claim {
    }

    /** Test-controlled admission which exposes the exact wake capability. */
    static final class ControllableAdmission implements PrefillAdmissionPort {
        private final TestAvailability availability = new TestAvailability();

        ControllableAdmission(boolean initiallyAvailable) {
            availability.setAvailable(initiallyAvailable);
        }

        void setAvailable(boolean available) {
            availability.setAvailable(available);
        }

        @Override
        public CapacityBoundary.Attempt<PreparedAdmission> tryBegin(
                ScheduledRequest firstCandidate) {
            return CapacityBoundary.Attempt.accepted(
                    new Prepared());
        }

        private <T> CapacityBoundary.Attempt<T> unavailable() {
            return CapacityBoundary.Attempt.rejected(
                    CapacityBoundary.unavailable(
                            availability,
                            new RouteProjection.AdmissionBlockSemantics(
                                    "test capacity unavailable",
                                    RouteProjection.AfterProbeAdmission.BLOCKED,
                                    "test capacity unavailable",
                                    RoleType.PREFILL)));
        }

        private final class Prepared implements PreparedAdmission {
            private final List<ScheduledRequest> items = new ArrayList<>();
            private boolean moved;
            private boolean closed;

            @Override
            public OptionalLong correlationId() {
                return OptionalLong.empty();
            }

            @Override
            public CapacityBoundary.Attempt<ScheduledRequest> tryAppend(
                    ScheduledRequest exactNextItem,
                    long predictedMs) {
                requireOpen();
                if (!availability.isAvailable()) {
                    return unavailable();
                }
                items.add(exactNextItem);
                return CapacityBoundary.Attempt.accepted(exactNextItem);
            }

            @Override
            public CommittedAdmission commitPreparedUnderLock(
                    List<ScheduledRequest> exactItems,
                    long predictedMs) {
                requireOpen();
                requireExactOrder(items, exactItems);
                moved = true;
                return new Committed(exactItems);
            }

            @Override
            public void close() {
                closed = true;
            }

            private void requireOpen() {
                if (moved || closed) {
                    throw new IllegalStateException(
                            "prepared admission is no longer open");
                }
            }
        }

        private static final class Committed implements CommittedAdmission {
            private final Map<ScheduledRequest, Boolean> untransferred =
                    new IdentityHashMap<>();
            private boolean closed;

            Committed(List<ScheduledRequest> exactItems) {
                for (ScheduledRequest item : exactItems) {
                    if (untransferred.put(item, Boolean.TRUE) != null) {
                        throw new IllegalArgumentException(
                                "duplicate delivery identity");
                    }
                }
            }

            @Override
            public boolean transferToEndpoint(
                    ScheduledRequest exactItem) {
                if (closed || untransferred.remove(exactItem) == null) {
                    throw new IllegalStateException(
                            "unknown or already-transferred item");
                }
                return true;
            }

            @Override
            public void close() {
                closed = true;
                untransferred.clear();
            }
        }
    }

    private static final class TestAvailability
            implements CapacityBoundary.Availability {
        private final Set<Runnable> listeners = new CopyOnWriteArraySet<>();
        private volatile boolean available;

        @Override
        public boolean isAvailable() {
            return available;
        }

        @Override
        public void addListener(Runnable listener) {
            listeners.add(listener);
        }

        @Override
        public void removeListener(Runnable listener) {
            listeners.remove(listener);
        }

        void setAvailable(boolean available) {
            boolean becameAvailable = available && !this.available;
            this.available = available;
            if (becameAvailable) {
                for (Runnable listener : listeners) {
                    listener.run();
                }
            }
        }
    }

    private static void requireExactOrder(
            List<ScheduledRequest> expected,
            List<ScheduledRequest> actual) {
        if (expected.size() != actual.size()) {
            throw new IllegalArgumentException("delivery item count changed");
        }
        for (int index = 0; index < expected.size(); index++) {
            if (expected.get(index) != actual.get(index)) {
                throw new IllegalArgumentException(
                        "delivery item identity/order changed");
            }
        }
    }

}
