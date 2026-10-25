package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.BatchDeliveryStrategy;
import org.flexlb.balance.delivery.BlockedRequest;
import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.delivery.DeliveryGroup;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryTransaction;
import org.flexlb.balance.delivery.NonBatchDeliveryStrategy;
import org.flexlb.balance.endpoint.prefill.CommitResult;
import org.flexlb.balance.endpoint.prefill.ReservationResult;
import org.flexlb.balance.endpoint.prefill.batch.BatchReservation;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.estimate.BatchTtftEstimator;
import org.flexlb.balance.estimate.NonBatchTtftEstimator;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.PrefillTtftEstimator;
import org.flexlb.balance.scheduler.EndpointEventHandler;
import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PrefillQueueEnqueueResult;
import org.flexlb.service.monitor.BatchSchedulerReporter;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Supplier;

/** Shared fixtures which exercise only the frozen endpoint-facing ports. */
public final class EndpointTestSupport {

    private EndpointTestSupport() {
    }

    static PrefillEndpoint standalonePrefillEndpoint(
            WorkerStatus status, FlexlbConfig config, DeliveryStrategy deliveryStrategy,
            EndpointEventHandler endpointEvents, BatchSchedulerReporter reporter) {
        return standalonePrefillEndpoint(status, () -> config, deliveryStrategy,
                endpointEvents, reporter, new PlacementAvailability());
    }

    static PrefillEndpoint standalonePrefillEndpoint(
            WorkerStatus status, Supplier<FlexlbConfig> configSupplier, DeliveryStrategy deliveryStrategy,
            EndpointEventHandler endpointEvents, BatchSchedulerReporter reporter,
            PlacementAvailability availability) {
        EndpointRegistry registry = org.mockito.Mockito.mock(EndpointRegistry.class);
        org.mockito.Mockito.when(registry.getEndpointCount(status.getRole())).thenReturn(1);
        PrefillTtftEstimator ttftEstimator = deliveryStrategy instanceof BatchDeliveryStrategy
                ? new BatchTtftEstimator() : new NonBatchTtftEstimator();
        return new PrefillEndpoint(status, configSupplier, deliveryStrategy, ttftEstimator,
                endpointEvents, reporter, availability, registry,
                new QueueCapacity());
    }

    static EndpointEventHandler noopEventSink() {
        return org.mockito.Mockito.mock(EndpointEventHandler.class);
    }

    public static WorkerStatus workerStatus(
            RoleType role,
            String ip,
            int port,
            int grpcPort) {
        return WorkerStatus.createDiscovered(
                role, null, ip, port, grpcPort, null);
    }

    public static WorkerStatus workerStatus(
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
            Runnable statusNotifier =
                    endpoint.applyPreparedStatus(status, prepared);
            status.recordSuccessfulPoll(response.isAlive());
            return statusNotifier;
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
        return parkedDelivery();
    }

    /**
     * 一个永远不交付的 Prefill 交付策略：请求一直留在等待队列里，由测试自己提交。
     */
    public static DeliveryStrategy parkedDelivery() {
        DeliveryStrategy delivery = org.mockito.Mockito.mock(
                DeliveryStrategy.class);
        org.mockito.Mockito.when(delivery.prepare(
                        org.mockito.Mockito.any()))
                .thenAnswer(invocation -> {
                    List<ScheduledRequest> candidates = ((DeliveryGroup) invocation.getArgument(0)).requests();
                    DeliveryTransaction transaction = org.mockito.Mockito.mock(DeliveryTransaction.class);
                    org.mockito.Mockito.when(transaction.requests()).thenReturn(List.of());
                    org.mockito.Mockito.when(transaction.blocked())
                            .thenReturn(new BlockedRequest(candidates.getFirst(), PARKED_BOUNDARY));
                    return transaction;
                });
        return delivery;
    }

    static DeliveryStrategy liveRouteStrategy(TestRequestRuntime runtime) {
        return new NonBatchDeliveryStrategy(
                runtime.requests(), NOOP_TELEMETRY);
    }

    private static final DeliveryBlockReason PARKED_BOUNDARY =
            DeliveryBlockReason.unavailable(
                    new DeliveryBlockReason.Availability() {
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
                    new PrefillTtftEstimate.AdmissionBlockSemantics(
                            "test delivery parked",
                            PrefillTtftEstimate.AfterProbeAdmission.BLOCKED,
                            "test delivery parked",
                            RoleType.PREFILL));

    public static boolean offer(PrefillEndpoint endpoint, ScheduledRequest item) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                return false;
            }
            return endpoint.offerPinnedResult(pin, item) == PrefillQueueEnqueueResult.ACCEPTED;
        }
    }

    /**
     * 把 NON_BATCH 请求放进 Prefill 等待队列，取出入队时预留的 Prefill 资源，并换上给定的耗时预测。
     * 调用方随后自己提交这份预留。
     */
    public static SingleReservation offerAndTakeSingleReservation(PrefillEndpoint endpoint,
                                                                  ScheduledRequest item,
                                                                  long predictedMs) {
        if (!offer(endpoint, item)) {
            throw new IllegalStateException("Prefill queue rejected request_id=" + item.requestId());
        }
        SingleReservation reservation = item.takePublishedSingleReservation();
        reservation.updatePrediction(item, predictedMs);
        return reservation;
    }

    static CommitResult commitBatch(PrefillEndpoint endpoint,
                                    long batchId,
                                    long predictedMs,
                                    List<? extends ScheduledRequest> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("batch requires at least one item");
        }
        List<ScheduledRequest> items = List.copyOf(exactItems);
        ReservationResult<BatchReservation> result = endpoint.reserveBatch(items.get(0), batchId, Integer.MAX_VALUE);
        if (result.status() != ReservationResult.Status.SUCCESS) {
            throw new IllegalStateException(
                    "batch reservation rejected: " + result.status());
        }
        try (BatchReservation reservation = result.reservation()) {
            return reservation.commit(items, predictedMs);
        }
    }

    static List<CommitResult> commitRoutes(PrefillEndpoint endpoint,
                                           long predictedMs,
                                           List<? extends ScheduledRequest> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("route group requires an item");
        }
        List<ScheduledRequest> items = List.copyOf(exactItems);
        List<SingleReservation> reservations =
                new ArrayList<>(items.size());
        boolean committed = false;
        try {
            for (ScheduledRequest item : items) {
                ReservationResult<SingleReservation> result =
                        endpoint.reserveSingle(item, predictedMs);
                if (result.status() != ReservationResult.Status.SUCCESS) {
                    throw new IllegalStateException(
                            "single reservation rejected: " + result.status());
                }
                reservations.add(result.reservation());
            }
            PrefillEndpoint.RouteCommitAdmission admission =
                    endpoint.tryBeginRouteCommitAdmission();
            if (admission == null) {
                throw new IllegalStateException("route endpoint retired");
            }
            CommitResult handoff;
            try (admission) {
                handoff = admission.commit(items, reservations);
            }
            committed = true;
            return List.of(handoff);
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
        private final EndpointEventHandler events =
                org.mockito.Mockito.mock(EndpointEventHandler.class);
        private final List<PrefillRetirement> prefillRetirements =
                new CopyOnWriteArrayList<>();
        private final List<ScheduledRequest> offerFailures =
                new CopyOnWriteArrayList<>();
        TestRequestRuntime() {
            org.flexlb.balance.scheduler.DeliveryStrategyTestSupport.stubRouteDelivery(requests, this::onCompleted);
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

        EndpointEventHandler events() {
            return events;
        }

        void onCompleted(
                DeliveryClaim claim,
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
