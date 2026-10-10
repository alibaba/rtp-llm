package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.scheduler.DeliveryStrategy;
import org.flexlb.balance.scheduler.DeliveryTransaction;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.RouteDeliveryStrategy;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/** Shared fixtures for exact endpoint and resource ownership contracts. */
public final class EndpointTestSupport {

    private EndpointTestSupport() {
    }

    public static boolean isQueued(DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && request.phase().isMasterQueued();
    }

    public static boolean isReserved(DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && !request.phase().isEngineConfirmed();
    }

    /** Check shadow estimates against production capacity; these fixtures have no released-victim KV hold. */
    public static long expectedReservedKv(DecodeResources.ResourceSnapshot snapshot) {
        long expected = 0L;
        for (var request : snapshot.requests().values()) {
            if (!request.phase().isEngineConfirmed()) { expected += request.expectedKvTokens(); }
        }
        var fields = snapshot.routing().workerStatus().fields();
        long reported = fields.totalKvCacheTokens() > 0L
                ? Math.max(0L, fields.totalKvCacheTokens() - fields.availableKvCacheTokens()) : 0L;
        org.junit.jupiter.api.Assertions.assertEquals(
                DecodeResources.CapacityRelease.saturatedAddNonNegative(reported, expected),
                snapshot.routing().realKvUsed(), "reported + shadow KV must match the incrementally maintained capacity");
        return expected;
    }

    /** Build a real local reservation without publishing it to the master queue. */
    public static DecodeResources.ReservationHandle reserveUnqueuedDecode(
            DecodeEndpoint endpoint, WorkerEndpoint.GenerationPin pin, long requestId,
            long hardKv, long expectedKv, int priority) {
        endpoint.requirePinnedGeneration(pin);
        var state = (DecodeState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "state");
        var lock = (java.util.concurrent.locks.ReentrantLock)
                org.springframework.test.util.ReflectionTestUtils.getField(state, "admissionLock");
        lock.lock();
        try {
            if (!Boolean.TRUE.equals(org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                    state, "requestIdAvailableForReservationLocked", requestId))) {
                throw new IllegalStateException("Decode request id is already owned: " + requestId);
            }
            return org.springframework.test.util.ReflectionTestUtils.invokeMethod(state, "createReservationLocked",
                    requestId, hardKv, expectedKv, priority, org.flexlb.enums.DecodeTaskPhase.LOCAL_RESERVED);
        } finally {
            lock.unlock();
        }
    }

    /** Read shadow ownership for assertions, without granting a production lookup capability. */
    public static DecodeResources.ReservationHandle decodeReservation(DecodeEndpoint endpoint, long requestId) {
        return endpoint.isRetired() ? null : decodeReservation(endpoint.resourceSnapshot(), requestId);
    }

    static DecodeResources.ReservationHandle decodeReservation(DecodeState state, long requestId) {
        return decodeReservation(state.resourceSnapshot(), requestId);
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

    static int evictPrefill(PrefillState state, long ttlMs, java.util.function.LongPredicate retained) {
        var candidates = new java.util.HashSet<RequestRoute>();
        for (RequestRoute route : state.cleanupCandidates()) {
            if (!retained.test(route.requestId())) { candidates.add(route); }
        }
        return state.evictExpiredInflight(ttlMs, candidates);
    }

    static DecodeState.CleanupResult evictDecode(DecodeState state, long ttlMs,
                                                java.util.function.LongPredicate retained) {
        var candidates = state.cleanupCandidates();
        candidates.keySet().removeIf(retained::test);
        return state.evictExpiredRequests(ttlMs, candidates);
    }

    static boolean handoffPreemption(DecodeEndpoint endpoint, long token) {
        var victims = endpoint.resourceSnapshot().requests().values().stream()
                .filter(DecodeResources.DecodeRequestView::claimedForPreemption).toList();
        return !victims.isEmpty() && victims.stream().allMatch(victim -> endpoint.updatePreemption(token,
                DecodeResources.PreemptionUpdate.handedOff(new DecodeResources.ReservationHandle(
                        endpoint.getStatus().getGenerationId(), victim.requestId(), victim.reservationToken()))));
    }

    static boolean handoffPreemption(DecodeState state, long token) {
        var victims = state.resourceSnapshot().requests().values().stream()
                .filter(DecodeResources.DecodeRequestView::claimedForPreemption).toList();
        return !victims.isEmpty() && victims.stream().allMatch(victim -> state.updatePreemption(token,
                DecodeResources.PreemptionUpdate.handedOff(new DecodeResources.ReservationHandle(
                        state.routingView().generationId(), victim.requestId(), victim.reservationToken()))));
    }

    static PrefillState.StatusReconciliation reconcile(PrefillState state,
            WorkerStatus.StatusObservation observation,
            java.util.function.ToLongFunction<List<RequestRoute>> predictor) {
        var lock = state.ownershipLock();
        lock.lock();
        PrefillState.StatusReduction reduction;
        try { reduction = state.prepareStatusLocked(observation); }
        finally { lock.unlock(); }
        var predictions = new java.util.HashMap<Long, Long>();
        reduction.predictionInputs().forEach((batchId, members) -> predictions.put(batchId, predictor.applyAsLong(members)));
        lock.lock();
        try { return java.util.Objects.requireNonNull(state.commitStatusLocked(reduction, predictions)); }
        finally { lock.unlock(); }
    }

    static PrefillState.CommittedHandoff commitBatch(
            PrefillState state, PrefillState.BatchReservation reservation,
            List<RequestRoute> items, long predictedMs) {
        state.ownershipLock().lock();
        try {
            return reservation.commitLocked(items, predictedMs, null, System.currentTimeMillis());
        } finally {
            state.ownershipLock().unlock();
        }
    }

    static PrefillState.CommittedHandoff commitRoutes(
            PrefillState state, List<RequestRoute> items,
            List<PrefillState.RouteReservation> reservations,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        state.ownershipLock().lock();
        try {
            return state.commitRouteGroupLocked(items, reservations, generationHandoff);
        } finally {
            state.ownershipLock().unlock();
        }
    }

    public static PrefillEndpoint unstartedPrefill(
            org.flexlb.config.FlexlbConfig config, WorkerStatus status,
            DeliveryStrategy delivery, AbstractRequestScheduler scheduler) {
        var endpoint = EndpointTestSupport.prefill(status, config, delivery, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler), org.mockito.Mockito.mock(DeliveryMetricsReporter.class));
        var state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "prefillState");
        state.ownershipLock().lock();
        try { state.enableQueueLocked(config.isPriorityOrdering()
                ? org.flexlb.balance.scheduler.WorkerBatcher.PRIORITY_QUEUE_ORDER : org.flexlb.balance.scheduler.WorkerBatcher.FIFO_QUEUE_ORDER); }
        finally { state.ownershipLock().unlock(); }
        var worker = org.mockito.Mockito.mock(org.flexlb.balance.scheduler.WorkerBatcher.class, org.mockito.Mockito.withSettings()
                .useConstructor(status.getIpPort(), endpoint, org.flexlb.balance.scheduler.QueueExecutionSettings.capture(config), delivery, state)
                .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
        org.mockito.Mockito.doAnswer(call -> {
            RequestRoute route = call.getArgument(0);
            if (route.ctx().scheduler() == null) { org.flexlb.balance.scheduler.SchedulerTestSupport.bindOwner(route.ctx(), scheduler); }
            return call.callRealMethod();
        }).when(worker).offer(org.mockito.ArgumentMatchers.any());
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "batcher", worker);
        return endpoint;
    }

    public static org.flexlb.balance.scheduler.WorkerBatcher batcher(PrefillEndpoint endpoint) {
        return (org.flexlb.balance.scheduler.WorkerBatcher)
                org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "batcher");
    }

    public static DecodeEndpoint decode(WorkerStatus status,
                                        org.flexlb.balance.scheduler.RequestRepository repository) {
        return new DecodeEndpoint(status, repository, new org.flexlb.balance.scheduler.PlacementAvailability());
    }

    public static PrefillEndpoint prefill(WorkerStatus status, org.flexlb.config.FlexlbConfig config, DeliveryStrategy delivery,
            org.flexlb.balance.scheduler.RequestRepository repository, DeliveryMetricsReporter reporter) {
        var endpoint = new PrefillEndpoint(status, config, delivery, reporter,
                new org.flexlb.balance.scheduler.PlacementAvailability());
        org.flexlb.balance.scheduler.SchedulerTestSupport.associateEndpoint(endpoint, repository);
        return endpoint;
    }

    static AbstractRequestScheduler noopEventSink() {
        return org.mockito.Mockito.mock(AbstractRequestScheduler.class);
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
                    address, status, prepared);
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
        RouteDeliveryStrategy route = new RouteDeliveryStrategy(NOOP_TELEMETRY);
        DeliveryStrategy delivery = org.mockito.Mockito.mock(
                DeliveryStrategy.class);
        org.mockito.Mockito.when(delivery.projectionPolicy())
                .thenReturn(route.projectionPolicy());
        org.mockito.Mockito.when(delivery.newGroupPredictor(org.mockito.Mockito.any()))
                .thenAnswer(invocation -> route.newGroupPredictor(invocation.getArgument(0)));
        org.mockito.Mockito.when(delivery.prepare(
                        org.mockito.Mockito.anyList(),
                        org.mockito.Mockito.any(),
                        org.mockito.Mockito.any()))
                .thenAnswer(invocation -> {
                    List<RequestRoute> candidates = invocation.getArgument(0);
                    DeliveryTransaction transaction =
                            org.mockito.Mockito.mock(
                                    DeliveryTransaction.class);
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
        return new RouteDeliveryStrategy(NOOP_TELEMETRY);
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

    static boolean offer(PrefillEndpoint endpoint, RequestRoute item) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                return false;
            }
            return endpoint.offerPinned(pin, item, org.flexlb.balance.scheduler.QueueExecutionSettings.capture(item.ctx().getConfig()));
        }
    }

    static PrefillState.RouteReservation reserveUnqueued(
            PrefillEndpoint endpoint, RequestRoute item, long predictedMs) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new IllegalStateException("endpoint is retired");
            }
            return endpoint.reserveUnqueuedRoute(pin, item, predictedMs).reservation();
        }
    }

    static void commitUnqueued(PrefillEndpoint endpoint, long requestId, long predictedMs) {
        RequestRoute item = org.mockito.Mockito.mock(RequestRoute.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(requestId);
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        {
            var reservation = reserveUnqueued(endpoint, item, predictedMs);
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                 var commit = endpoint.tryBeginRouteCommitAdmission();
                 var handoff = commit.commit(List.of(item), List.of(reservation))) {
                // The endpoint ledger owns this request until authoritative termination.
            }
        }
    }

    static PrefillState.CommittedHandoff commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends RequestRoute> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("batch requires at least one item");
        }
        List<RequestRoute> items = List.copyOf(exactItems);
        items.forEach(item -> org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item));
        PrefillState.ReservationResult<PrefillState.BatchReservation> result = endpoint.reserveBatch(
                items.get(0), batchId, Integer.MAX_VALUE);
        if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
            throw new IllegalStateException(
                    "batch reservation rejected: " + result.status());
        }
        {
            PrefillState.BatchReservation reservation =
                     result.reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                PrefillState state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(endpoint, "prefillState");
                return commitBatch(state, reservation, items, predictedMs);
            }
        }
    }

    public static boolean releaseRequest(PrefillState state, RequestRoute exact) {
        return state.releaseRequest(exact) != PrefillState.RequestRelease.NONE;
    }

    public static PreparationScope preparation(PrefillState.Reservation reservation) {
        return new PreparationScope(reservation);
    }

    static void rollback(PrefillState.Reservation reservation) {
        if (reservation == null) { return; }
        var released = reservation.owner.rollbackPreparation(reservation);
        if (released.generationHandoff() != null) { released.generationHandoff().close(); }
    }

    public static final class PreparationScope implements AutoCloseable {
        private final PrefillState.Reservation reservation;
        private PreparationScope(PrefillState.Reservation reservation) { this.reservation = reservation; }
        @Override public void close() { rollback(reservation); }
    }

    static PrefillState.CommittedHandoff commitQueuedRoutes(
            PrefillState state, List<RequestRoute> items, long[] predictions,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        state.ownershipLock().lock();
        try { return state.commitQueuedRoutesLocked(items, predictions, generationHandoff, null, System.currentTimeMillis()); }
        finally { state.ownershipLock().unlock(); }
    }

    static List<PrefillState.CommittedHandoff> commitRoutes(
            PrefillEndpoint endpoint, long predictedMs, List<? extends RequestRoute> exactItems) {
        List<RequestRoute> items = List.copyOf(exactItems);
        items.forEach(item -> org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item));
        long[] predictions = new long[items.size()];
        java.util.Arrays.fill(predictions, predictedMs);
        try (var admission = endpoint.tryBeginRouteCommitAdmission()) {
            if (admission == null) { throw new IllegalStateException("route endpoint retired"); }
            var lock = org.flexlb.balance.scheduler.WorkerBatcherTestSupport.state(batcher(endpoint)).ownershipLock();
            lock.lock();
            try { return List.of(endpoint.commitQueuedRoutesLocked(items, predictions, null, System.currentTimeMillis())); }
            finally { lock.unlock(); }
        }
    }

    static final DeliveryMetricsReporter NOOP_TELEMETRY =
            org.mockito.Mockito.mock(DeliveryMetricsReporter.class);

    static class TestRequestRuntime {
        private final AbstractRequestScheduler requests =
                org.mockito.Mockito.mock(AbstractRequestScheduler.class);
        private final AbstractRequestScheduler events = requests;
        private final List<PrefillRetirement> prefillRetirements =
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
                        invocation.getArgument(0), List.of(invocation.getArgument(1, RequestRoute.class))));
                return null;
            }).when(events).onPrefillGenerationRetired(
                    org.mockito.Mockito.any(), org.mockito.Mockito.any());
        }

        AbstractRequestScheduler requests() {
            return requests;
        }

        AbstractRequestScheduler events() {
            return events;
        }

        void onCompleted(
                DeliveryClaim claim,
                DeliveryResult result) {
        }

        void onQueueOfferFailure(
                RequestRoute item,
                Throwable error) {
        }

        List<PrefillRetirement> prefillRetirements() {
            return List.copyOf(prefillRetirements);
        }

    }

    record PrefillRetirement(
            PrefillEndpoint endpoint,
            List<RequestRoute> ownedItems) {
        PrefillRetirement {
            ownedItems = List.copyOf(ownedItems);
        }
    }

}
