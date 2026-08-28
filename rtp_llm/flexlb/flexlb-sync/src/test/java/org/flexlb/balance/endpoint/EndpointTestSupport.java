package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.CommittedDelivery;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryTelemetry;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.delivery.RouteDeliveryStrategy;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;

import java.lang.reflect.Constructor;
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
import java.util.function.Supplier;

/** Shared fixtures which exercise only the frozen endpoint-facing ports. */
final class EndpointTestSupport {

    private EndpointTestSupport() {
    }

    static PrefillGenerationRuntime.Factory realRuntimeFactory() {
        try {
            Class<?> type = Class.forName(
                    "org.flexlb.balance.scheduler.WorkerBatcherFactory");
            Constructor<?> constructor = type.getDeclaredConstructor();
            constructor.setAccessible(true);
            return (PrefillGenerationRuntime.Factory) constructor.newInstance();
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(
                    "Unable to construct the production Prefill runtime", failure);
        }
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

    static boolean offer(PrefillEndpoint endpoint, DeliveryItem item) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                return false;
            }
            return endpoint.offerPinned(pin, item);
        }
    }

    static PrefillWorkLedger.DirectRegistration registerDirect(
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

    static PrefillWorkLedger.CommittedHandoff commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends DeliveryItem> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("batch requires at least one item");
        }
        List<DeliveryItem> items = List.copyOf(exactItems);
        PrefillWorkLedger.BatchReservationResult result = endpoint.reserveBatch(
                items.get(0), batchId, Integer.MAX_VALUE);
        if (result.status() != PrefillWorkLedger.CapacityStatus.ACQUIRED) {
            throw new IllegalStateException(
                    "batch reservation rejected: " + result.status());
        }
        java.util.concurrent.locks.ReentrantLock queueLock = extractQueueLock(endpoint);
        try (PrefillWorkLedger.BatchReservation reservation =
                     result.reservation()) {
            queueLock.lock();
            try {
                return reservation.commitUnderLock(items, predictedMs);
            } finally {
                queueLock.unlock();
            }
        }
    }

    static List<PrefillWorkLedger.CommittedHandoff> commitRoutes(
            PrefillEndpoint endpoint,
            long predictedMs,
            List<? extends DeliveryItem> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("route group requires an item");
        }
        List<DeliveryItem> items = List.copyOf(exactItems);
        List<PrefillWorkLedger.RouteReservation> reservations =
                new ArrayList<>(items.size());
        boolean committed = false;
        try {
            for (DeliveryItem item : items) {
                PrefillWorkLedger.RouteReservationResult result =
                        endpoint.reserveRoute(
                                item, predictedMs, Integer.MAX_VALUE);
                if (result.status()
                        != PrefillWorkLedger.CapacityStatus.ACQUIRED) {
                    throw new IllegalStateException(
                            "route reservation rejected: " + result.status());
                }
                reservations.add(result.reservation());
            }
            List<PrefillWorkLedger.CommittedHandoff> handoffs;
            java.util.concurrent.locks.ReentrantLock queueLock = extractQueueLock(endpoint);
            queueLock.lock();
            try {
                handoffs = reservations.get(0).commitGroupUnderLock(
                        items, reservations);
            } finally {
                queueLock.unlock();
            }
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

    static final DeliveryTelemetry NOOP_TELEMETRY = new DeliveryTelemetry() {
        @Override
        public void routesDelivered(
                DeliveryMetadata metadata,
                List<DeliveryItem> exactItems) {
        }

        @Override
        public void batchDispatched(
                long batchId,
                DeliveryMetadata metadata,
                List<DeliveryItem> dispatched,
                long predictedMs) {
        }
    };

    static class TestRequestRuntime implements EndpointRequestRuntime {
        private final List<EndpointEvent> events = new CopyOnWriteArrayList<>();
        private final List<DeliveryItem> offerFailures =
                new CopyOnWriteArrayList<>();
        private final List<DeliveryItem> deliveryFailures =
                new CopyOnWriteArrayList<>();
        private final AtomicInteger committedDeliveries = new AtomicInteger();
        private final AtomicInteger completedClaims = new AtomicInteger();

        List<EndpointEvent> events() {
            return List.copyOf(events);
        }

        List<DeliveryItem> offerFailures() {
            return List.copyOf(offerFailures);
        }

        List<DeliveryItem> deliveryFailures() {
            return List.copyOf(deliveryFailures);
        }

        int committedDeliveryCount() {
            return committedDeliveries.get();
        }

        int completedClaimCount() {
            return completedClaims.get();
        }

        @Override
        public <T> Optional<T> prepareIfOwned(
                DeliveryItem exactItem,
                Supplier<T> preparation) {
            return Optional.ofNullable(preparation.get());
        }

        @Override
        public Claim tryClaimForDelivery(
                DeliveryItem exactItem,
                Identity identity,
                EndpointHandoff endpointHandoff) {
            TestClaim claim = new TestClaim(exactItem);
            return endpointHandoff.transferToEndpoint() ? claim : null;
        }

        @Override
        public void complete(Claim exactClaim, Completion completion) {
            completedClaims.incrementAndGet();
        }

        @Override
        public void failPrepared(DeliveryItem exactItem, Throwable cause) {
            deliveryFailures.add(exactItem);
        }

        @Override
        public void onQueuedItemExpired(DeliveryItem exactItem) {
        }

        @Override
        public void resolveCommittedDelivery(
                CommittedDelivery delivery,
                DeliveryMetadata metadata) {
            committedDeliveries.incrementAndGet();
            delivery.deliver(metadata);
        }

        @Override
        public void onQueueOfferFailure(
                DeliveryItem exactItem,
                Throwable cause) {
            offerFailures.add(exactItem);
        }

        @Override
        public void onPreparedDeliveryFailure(
                DeliveryItem exactItem,
                Throwable cause) {
            deliveryFailures.add(exactItem);
        }

        @Override
        public void onEndpointEvent(EndpointEvent event) {
            events.add(event);
        }
    }

    private record TestClaim(DeliveryItem item)
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
                DeliveryItem firstCandidate) {
            return new CapacityBoundary.Attempt.Accepted<>(
                    new Prepared());
        }

        private <T> CapacityBoundary.Attempt<T> unavailable() {
            return new CapacityBoundary.Attempt.Rejected<>(
                    new CapacityBoundary.Unavailable(
                            availability,
                            new RouteProjection.AdmissionBlockSemantics(
                                    "test capacity unavailable",
                                    RouteProjection.AfterProbeAdmission.BLOCKED,
                                    "test capacity unavailable")));
        }

        private final class Prepared implements PreparedAdmission {
            private final List<DeliveryItem> items = new ArrayList<>();
            private boolean moved;
            private boolean closed;

            @Override
            public OptionalLong correlationId() {
                return OptionalLong.empty();
            }

            @Override
            public CapacityBoundary.Attempt<DeliveryItem> tryAppend(
                    DeliveryItem exactNextItem,
                    long predictedMs) {
                requireOpen();
                if (!availability.isAvailable()) {
                    return unavailable();
                }
                items.add(exactNextItem);
                return new CapacityBoundary.Attempt.Accepted<>(exactNextItem);
            }

            @Override
            public CommittedAdmission commitPreparedUnderLock(
                    List<DeliveryItem> exactItems,
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
            private final Map<DeliveryItem, Boolean> untransferred =
                    new IdentityHashMap<>();
            private boolean closed;

            Committed(List<DeliveryItem> exactItems) {
                for (DeliveryItem item : exactItems) {
                    if (untransferred.put(item, Boolean.TRUE) != null) {
                        throw new IllegalArgumentException(
                                "duplicate delivery identity");
                    }
                }
            }

            @Override
            public boolean transferToEndpoint(
                    DeliveryItem exactItem) {
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
            List<DeliveryItem> expected,
            List<DeliveryItem> actual) {
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

    /**
     * Extract the package-private queueLock from the WorkerBatcher runtime
     * backing this endpoint. Tests use this to satisfy the "commitUnderLock"
     * contract without starting the full batcher thread.
     */
    static java.util.concurrent.locks.ReentrantLock extractQueueLock(
            PrefillEndpoint endpoint) {
        try {
            java.lang.reflect.Field runtimeField =
                    PrefillEndpoint.class.getDeclaredField("runtime");
            runtimeField.setAccessible(true);
            Object batcher = runtimeField.get(endpoint);
            java.lang.reflect.Field lockField =
                    batcher.getClass().getDeclaredField("queueLock");
            lockField.setAccessible(true);
            return (java.util.concurrent.locks.ReentrantLock) lockField.get(batcher);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(
                    "unable to extract queueLock from PrefillEndpoint runtime",
                    failure);
        }
    }
}
