package org.flexlb.balance.strategy;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.delivery.RouteDeliveryStrategy;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointStatusReduction;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.mockito.Mockito;

import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/** Package-local fixtures built only from the frozen endpoint-facing ports. */
final class StrategyTestSupport {

    private StrategyTestSupport() {
    }

    static EndpointRegistry endpointRegistry(ConfigService configService) {
        TestRequestRuntime runtime = new TestRequestRuntime();
        DeliveryStrategy delivery = new RouteDeliveryStrategy(
                new ParkedAdmission(), runtime, NOOP_METRICS);
        return new EndpointRegistry(
                configService,
                runtime,
                Mockito.mock(BatchSchedulerReporter.class),
                delivery);
    }

    static WorkerStatus workerStatus(
            RoleType role,
            String group,
            String ip,
            int httpPort,
            int grpcPort,
            boolean alive,
            long availableKv,
            long totalKv) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                role, group, ip, httpPort, grpcPort, "test-site");
        WorkerStatusResponse response = response(
                role, alive, availableKv, totalKv,
                Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        publish(status, response);
        return status;
    }

    static WorkerStatusResponse response(
            RoleType role,
            boolean alive,
            long availableKv,
            long totalKv,
            long statusVersion) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(alive);
        response.setAvailableKvCacheTokens(availableKv);
        response.setTotalKvCacheTokens(totalKv);
        response.setStatusVersion(statusVersion);
        response.setLatestFinishedVersion(0L);
        return response;
    }

    static void publish(
            WorkerStatus status, WorkerStatusResponse response) {
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

    static EndpointStatusReduction apply(
            WorkerEndpoint endpoint, WorkerStatusResponse response) {
        WorkerStatus status = endpoint.getStatus();
        if (response.getRole() == null) {
            response.setRole(status.getRole());
        }
        if (response.getStatusVersion() == null
                || response.getStatusVersion()
                <= status.appliedStatusCursor().statusVersion()) {
            response.setStatusVersion(
                    Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        }
        if (response.getLatestFinishedVersion() == null) {
            response.setLatestFinishedVersion(
                    status.appliedStatusCursor().latestFinishedTaskVersion());
        }
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            return endpoint.applyPreparedStatus(status, prepared);
        } finally {
            status.lock.unlock();
        }
    }

    static boolean offer(
            PrefillEndpoint endpoint, ScheduledRequest exactItem) {
        try (WorkerEndpoint.GenerationPin pin =
                     endpoint.tryPinGeneration()) {
            return pin != null && endpoint.offerPinned(pin, exactItem);
        }
    }

    static PrefillState.CommittedHandoff commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends ScheduledRequest> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException(
                    "committed batch requires at least one item");
        }
        List<ScheduledRequest> items = List.copyOf(exactItems);
        PrefillState.BatchReservationResult result =
                endpoint.reserveBatch(
                        items.getFirst(), batchId, Integer.MAX_VALUE);
        if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
            throw new IllegalStateException(
                    "batch reservation rejected: " + result.status());
        }
        try (PrefillState.BatchReservation reservation =
                     result.reservation()) {
            return reservation.commit(items, predictedMs);
        }
    }

    static void setCacheStatus(
            WorkerStatus status,
            long blockSize,
            long availableKv) {
        CacheStatus cache = new CacheStatus();
        cache.setBlockSize(blockSize);
        cache.setAvailableKvCache(availableKv);
        WorkerStatusResponse response = response(
                status.getRole(), status.pollHealth().reportedAlive(),
                status.getAvailableKvCacheTokens(),
                status.getTotalKvCacheTokens(),
                Math.max(1L, status.appliedStatusCursor().statusVersion() + 1L));
        response.setCacheStatus(cache);
        publish(status, response);
    }

    private static final class TestRequestRuntime
            implements EndpointEventSink, SlotDeliveryPort {

        @Override
        public <T> Optional<T> prepareIfOwned(
                ScheduledRequest exactItem, Supplier<T> preparation) {
            return Optional.ofNullable(preparation.get());
        }

        @Override
        public Claim tryClaimForDelivery(
                ScheduledRequest exactItem,
                Identity identity,
                BooleanSupplier endpointHandoff) {
            return null;
        }

        @Override
        public void complete(Claim exactClaim, Completion completion) {
        }

        @Override
        public void failPrepared(ScheduledRequest exactItem, Throwable cause) {
        }

        @Override
        public void onQueuedItemExpired(ScheduledRequest exactItem) {
        }

        @Override
        public void onQueueOfferFailure(
                ScheduledRequest exactItem,
                Throwable cause) {
        }

        @Override
        public void onPreparedDeliveryFailure(
                ScheduledRequest exactItem,
                Throwable cause) {
        }

        @Override
        public void onStatusReduced(EndpointStatusReduction reduction) {
        }

        @Override
        public void onPrefillGenerationRetired(
                PrefillEndpoint endpoint,
                List<ScheduledRequest> ownedItems) {
        }

        @Override
        public void onDecodeGenerationRetired(
                DecodeEndpoint endpoint,
                List<DecodeEndpoint.ReservationHandle> ownedReservations) {
        }
    }

    /** Permanently unavailable delivery keeps strategy-owned queues observable. */
    private static final class ParkedAdmission
            implements PrefillAdmissionPort {
        private static final CapacityBoundary.Availability NEVER_AVAILABLE =
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
                };

        @Override
        public CapacityBoundary.Attempt<PreparedAdmission> tryBegin(
                ScheduledRequest firstCandidate) {
            return CapacityBoundary.Attempt.accepted(
                    new PreparedAdmission() {
                        @Override
                        public OptionalLong correlationId() {
                            return OptionalLong.empty();
                        }

                        @Override
                        public CapacityBoundary.Attempt<ScheduledRequest> tryAppend(
                                ScheduledRequest exactNextItem,
                                long predictedMs) {
                            return CapacityBoundary.Attempt.rejected(
                                    CapacityBoundary.unavailable(
                                            NEVER_AVAILABLE,
                                            new RouteProjection.AdmissionBlockSemantics(
                                                    "test delivery parked",
                                                    RouteProjection.AfterProbeAdmission.BLOCKED,
                                                    "test delivery parked",
                                                    RoleType.PREFILL)));
                        }

                        @Override
                        public CommittedAdmission commitPreparedUnderLock(
                                List<ScheduledRequest> exactItems,
                                long predictedMs) {
                            throw new IllegalStateException(
                                    "parked admission cannot commit");
                        }

                        @Override
                        public void close() {
                        }
                    });
        }
    }

    private static final DeliveryMetrics NOOP_METRICS =
            Mockito.mock(DeliveryMetrics.class);
}
