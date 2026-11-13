package org.flexlb.worker.endpoint;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.EngineCancellation;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.flexlb.worker.status.WorkerStatusTestSupport;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

/**
 * Shared fixtures for exact endpoint and resource ownership contracts.
 */
public final class EndpointTestSupport {
    private static final java.util.Map<PrefillEndpoint, PrefillEndpoint> RESOURCE_FIXTURES =
            java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>());
    static final DeliveryMetricsReporter NOOP_TELEMETRY =
            org.mockito.Mockito.mock(DeliveryMetricsReporter.class);

    private EndpointTestSupport() {}

    public static long occupiedRequestCount(PrefillEndpoint endpoint) {
        var state = resources(endpoint);
        return withStateLock(
                state.stateLock(),
                () -> {
                    var requests =
                            (java.util.Map<?, ?>) ReflectionTestUtils.getField(state, "requests");
                    return (long) requests.size();
                });
    }

    public static PrefillEndpoint resources(PrefillEndpoint endpoint) {
        return RESOURCE_FIXTURES.getOrDefault(endpoint, endpoint);
    }

    public static <T> T withStateLock(
            java.util.concurrent.locks.ReentrantLock lock,
            java.util.function.Supplier<T> operation) {
        lock.lock();
        try {
            return operation.get();
        } finally {
            lock.unlock();
        }
    }

    /**
     * A scheduling test may mock transport while retaining a real resource ledger.
     */
    public static void stubResources(PrefillEndpoint endpoint, PrefillEndpoint resources) {
        if (endpoint == resources) return;
        useStateLock(endpoint, stateLock(resources));
        RESOURCE_FIXTURES.put(endpoint, resources);
        org.junit.jupiter.api.Assertions.assertTrue(
                org.mockito.Mockito.mockingDetails(endpoint).isMock(),
                "resource delegation is only for a mocked transport endpoint");
        var names =
                java.util.Set.of(
                        "tryRegisterSelection",
                        "captureRouteProjectionInputs",
                        "rollbackSelection",
                        "queueDepth",
                        "releaseQueuedRequest",
                        "queueCounters",
                        "captureQueue",
                        "closeQueue",
                        "queueStopped",
                        "selectionCurrentLocked",
                        "explainAdmissionRejection",
                        "requestCapacityAvailable",
                        "batchCapacityAvailable",
                        "removeRequestsOnClose",
                        "evictExpiredRequests",
                        "stats");
        for (var method : PrefillEndpoint.class.getDeclaredMethods()) {
            if (!names.contains(method.getName())) continue;

            method.setAccessible(true);
            Object[] arguments = new Object[method.getParameterCount()];
            var delegate =
                    org.mockito.Mockito.doAnswer(
                                    call -> {
                                        try {
                                            return method.invoke(resources, call.getArguments());
                                        } catch (
                                                java.lang.reflect.InvocationTargetException
                                                        failure) {
                                            throw failure.getCause();
                                        }
                                    })
                            .when(endpoint);
            for (int i = 0; i < arguments.length; i++) {
                var type = method.getParameterTypes()[i];
                arguments[i] =
                        type == long.class
                                ? org.mockito.ArgumentMatchers.anyLong()
                                : type == int.class
                                        ? org.mockito.ArgumentMatchers.anyInt()
                                        : type == boolean.class
                                                ? org.mockito.ArgumentMatchers.anyBoolean()
                                                : org.mockito.ArgumentMatchers.any();
            }
            try {
                method.invoke(delegate, arguments);
            } catch (ReflectiveOperationException failure) {
                throw new AssertionError(failure);
            }
        }
    }

    public static void useStateLock(
            WorkerEndpoint endpoint, java.util.concurrent.locks.ReentrantLock lock) {
        ReflectionTestUtils.setField(endpoint, "stateLock", lock);
        var details = org.mockito.Mockito.mockingDetails(endpoint);
        if (details.isMock() && !details.isSpy()) {
            org.mockito.Mockito.doReturn(lock).when(endpoint).stateLock();
            if (details.getStubbings().stream()
                    .noneMatch(
                            stub ->
                                    stub.getInvocation()
                                            .getMethod()
                                            .getName()
                                            .equals("acceptsNewRequests"))) {
                org.mockito.Mockito.lenient()
                        .when(endpoint.acceptsNewRequests())
                        .thenAnswer(
                                call ->
                                        !endpoint.hasStoppedAcceptingRequests()
                                                && (endpoint.getStatus() == null
                                                        || endpoint.getStatus()
                                                                .isAvailableForScheduling()));
            }
        }
    }

    public static java.util.concurrent.locks.ReentrantLock stateLock(WorkerEndpoint endpoint) {
        var lock =
                (java.util.concurrent.locks.ReentrantLock)
                        ReflectionTestUtils.getField(endpoint, "stateLock");
        if (lock == null) {
            lock = new java.util.concurrent.locks.ReentrantLock();
            useStateLock(endpoint, lock);
        }
        return lock;
    }

    public static PrefillEndpoint.RemovedRequests removeRequestsOnClose(PrefillEndpoint endpoint) {
        return resources(endpoint).removeRequestsOnClose();
    }

    public static void awaitWorkerCleanup(java.util.concurrent.CompletionStage<Void> cleanup) {
        org.junit.jupiter.api.Assertions.assertTimeoutPreemptively(
                java.time.Duration.ofSeconds(5), () -> cleanup.toCompletableFuture().join());
    }

    public static java.util.concurrent.CompletionStage<Void> closeAsync(WorkerEndpoint endpoint) {
        return endpoint.closeAsync();
    }

    public static void closeAndWait(WorkerEndpoint endpoint) {
        endpoint.closeAsync();
        if (!endpoint.stateLock().isHeldByCurrentThread()) awaitClosed(endpoint);
    }

    public static void awaitClosed(WorkerEndpoint endpoint, long timeoutMs) {
        try {
            endpoint.closeAsync()
                    .toCompletableFuture()
                    .orTimeout(timeoutMs, java.util.concurrent.TimeUnit.MILLISECONDS)
                    .join();
        } catch (java.util.concurrent.CompletionException failure) {
            throw org.flexlb.util.Failures.propagate(
                    failure.getCause(), "DiscoveryEndpoint close timed out or failed");
        }
    }

    public static void awaitClosed(WorkerEndpoint endpoint) {
        awaitClosed(endpoint, 30_000L);
    }

    public static List<DecodeEndpoint.SelectionSnapshot> decodeRoutingSnapshot(
            EndpointRegistry registry, String group) {
        return registry
                .<DecodeEndpoint>endpoints(org.flexlb.config.service.RoleType.DECODE, null)
                .stream()
                .map(DecodeEndpoint::selectionSnapshot)
                .filter(view -> group == null || group.equals(view.topology().group()))
                .toList();
    }

    public static List<DecodeEndpoint> decodeCandidates(
            List<DecodeEndpoint.SelectionSnapshot> views) {
        return views.stream()
                .map(
                        view -> {
                            var endpoint = org.mockito.Mockito.mock(DecodeEndpoint.class);
                            org.mockito.Mockito.when(endpoint.selectionSnapshot()).thenReturn(view);
                            return endpoint;
                        })
                .toList();
    }

    public static boolean isReserved(DecodeResources.ResourceSnapshot snapshot, String requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && !request.phase().isEngineConfirmed();
    }

    /**
     * Check shadow estimates against production capacity; these fixtures have no released-request
     * selected for preemption KV
     * hold.
     */
    public static long expectedReservedKv(DecodeResources.ResourceSnapshot snapshot) {
        long expected = 0L;
        for (var request : snapshot.requests().values()) {
            if (!request.phase().isEngineConfirmed()) {
                expected += request.expectedKvTokens();
            }
        }
        var fields = snapshot.selection().workerStatus().fields();
        long reported =
                fields.totalKvCacheTokens() > 0L
                        ? Math.max(
                                0L, fields.totalKvCacheTokens() - fields.availableKvCacheTokens())
                        : 0L;
        org.junit.jupiter.api.Assertions.assertEquals(
                DecodeResources.Capacity.saturatedAddNonNegative(reported, expected),
                snapshot.selection().realKvUsed(),
                "reported + shadow KV must match the incrementally maintained capacity");
        return expected;
    }

    /**
     * Register a selected Decode request before delivery through the production admission boundary.
     */
    public static DecodeEndpoint.RequestRegistration registerDecode(
            DecodeEndpoint endpoint, long requestId, long hardKv, long expectedKv, int priority) {
        var acquired =
                endpoint.tryRegisterSelection(
                        endpoint.selectionSnapshot(),
                        String.valueOf(requestId),
                        hardKv,
                        expectedKv,
                        priority,
                        new DecodeResources.AdmissionCapacity(0L, 100L));
        org.junit.jupiter.api.Assertions.assertEquals(
                WorkerEndpoint.RegistrationStatus.REGISTERED, acquired.status());
        return acquired.value();
    }

    /**
     * Read shadow ownership for assertions, without granting a production lookup capability.
     */
    public static DecodeEndpoint.RequestRegistration decodeRegistration(
            DecodeEndpoint endpoint, long requestId) {
        return decodeRegistration(endpoint, Long.toString(requestId));
    }

    public static DecodeEndpoint.RequestRegistration decodeRegistration(
            DecodeEndpoint endpoint, String requestId) {
        return endpoint.hasStoppedAcceptingRequests()
                ? null
                : decodeRegistration(endpoint.resourceSnapshot(), requestId);
    }

    public static void runWithStateLock(
            java.util.concurrent.locks.ReentrantLock lock, Runnable operation) {
        lock.lock();
        try {
            operation.run();
        } finally {
            lock.unlock();
        }
    }

    public static DecodeEndpoint decode(
            WorkerStatus status, org.flexlb.balance.schedule.RequestRepository repository) {
        var endpoint =
                new DecodeEndpoint(
                        status, repository, new org.flexlb.worker.endpoint.CapacityNotifier());
        if (org.mockito.Mockito.mockingDetails(repository).isMock()) {
            org.mockito.Mockito.when(
                            repository.findActive(org.mockito.ArgumentMatchers.anyString()))
                    .thenAnswer(
                            call -> {
                                String id = call.getArgument(0);
                                var owner = repository.ownerOf(String.valueOf(id));
                                if (owner == null) {
                                    return null;
                                }
                                var context =
                                        org.mockito.Mockito.mock(
                                                org.flexlb.balance.schedule.RequestContext.class);

                                var snapshot =
                                        org.mockito.Mockito.mock(
                                                org.flexlb.balance.schedule.RequestContext.Snapshot
                                                        .class);
                                var route = org.mockito.Mockito.mock(RequestResult.class);

                                org.mockito.Mockito.when(context.getRequestScheduler())
                                        .thenReturn(owner);
                                org.mockito.Mockito.when(context.getRequestId())
                                        .thenReturn(String.valueOf(id));
                                org.mockito.Mockito.when(context.snapshot()).thenReturn(snapshot);
                                org.mockito.Mockito.when(snapshot.route()).thenReturn(route);
                                org.mockito.Mockito.when(route.ctx()).thenReturn(context);
                                org.mockito.Mockito.when(route.decodeEndpoint())
                                        .thenReturn(endpoint);
                                return context;
                            });
        }
        return endpoint;
    }

    public static PrefillEndpoint prefill(
            WorkerStatus status,
            org.flexlb.config.FlexlbConfig config,
            EngineCancellation cancellation,
            org.flexlb.balance.schedule.RequestRepository repository,
            DeliveryMetricsReporter reporter) {
        return realPrefill(status, config, cancellation, repository, reporter);
    }

    public static PrefillEndpoint realPrefill(
            WorkerStatus status,
            org.flexlb.config.FlexlbConfig config,
            EngineCancellation delivery,
            org.flexlb.balance.schedule.RequestRepository repository,
            DeliveryMetricsReporter reporter) {
        var endpoint =
                new PrefillEndpoint(
                        status, config, new org.flexlb.worker.endpoint.CapacityNotifier());
        org.flexlb.balance.schedule.SchedulerTestSupport.associateEndpoint(endpoint, repository);
        return endpoint;
    }

    public static boolean releaseRequest(PrefillEndpoint state, RequestResult exact) {
        return state.releaseRequest(exact);
    }

    public static PreparationScope preparation(PrefillEndpoint.RequestRegistration reservation) {
        return new PreparationScope(() -> rollback(reservation));
    }

    /**
     * Supplies scheduler callbacks for endpoint tests and records the exact routes reported during
     * Prefill cleanup.
     */
    public static boolean isQueued(DecodeResources.ResourceSnapshot snapshot, String requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && request.phase().isMasterQueued();
    }

    static DecodeEndpoint decodeEndpoint(WorkerStatus status) {
        return new DecodeEndpoint(
                status,
                org.mockito.Mockito.mock(org.flexlb.balance.schedule.RequestRepository.class),
                new org.flexlb.worker.endpoint.CapacityNotifier());
    }

    static WorkerEndpoint.Registration<DecodeEndpoint.RequestRegistration> reserveQueuedDecode(
            DecodeEndpoint endpoint,
            long id,
            long hard,
            long expected,
            int priority,
            DecodeResources.AdmissionCapacity capacity) {
        {
            var selectedEndpoint = endpoint;

            return capacity == null
                    ? org.flexlb.balance.schedule.SelectionTestSupport.registerQueuedDecode(
                            endpoint, String.valueOf(id), hard, expected, priority)
                    : endpoint.tryRegisterSelection(
                            endpoint.selectionSnapshot(),
                            String.valueOf(id),
                            hard,
                            expected,
                            priority,
                            capacity);
        }
    }

    static int evictPrefill(
            PrefillEndpoint state, long ttlMs, java.util.function.Predicate<String> retained) {
        return state.evictExpiredRequests(ttlMs, retained);
    }

    static AbstractRequestScheduler noopEventSink() {
        return org.mockito.Mockito.mock(AbstractRequestScheduler.class);
    }

    static WorkerStatus workerStatus(RoleType role, String ip, int port, int grpcPort) {
        return WorkerStatus.createDiscovered(role, null, ip, port, grpcPort, null);
    }

    static WorkerStatus workerStatus(
            RoleType role, String group, String ip, int port, int grpcPort, String site) {
        return WorkerStatus.createDiscovered(role, group, ip, port, grpcPort, site);
    }

    static Runnable applyStatus(WorkerEndpoint endpoint, WorkerStatusResponse response) {
        WorkerStatus status = endpoint.getStatus();
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            WorkerEndpoint.StatusChange projection = endpoint.applyWorkerStatus(status, prepared);
            status.recordSuccessfulPoll(response.isAlive());
            return () -> endpoint.publishStatus(projection);
        } finally {
            status.lock.unlock();
        }
    }

    static void publishStatus(WorkerStatus status, WorkerStatusResponse response) {
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(response.isAlive());
        } finally {
            status.lock.unlock();
        }
    }

    static WorkerEndpoint publishEndpoint(
            EndpointRegistry registry, RoleType role, String address, WorkerStatus status) {
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
        registry.currentOrDiscover(role, address, () -> status);
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            return registry.updateEndpoint(address, status, prepared).endpoint();
        } finally {
            status.lock.unlock();
        }
    }

    static boolean commitEncoder(
            EncoderEndpoint endpoint, RequestResult route, int limit, long tokens) {
        var registered = endpoint.tryRegisterSelection(route.requestId(), limit, tokens);
        if (registered.status() != WorkerEndpoint.RegistrationStatus.REGISTERED) return false;
        org.mockito.Mockito.doReturn(registered.value()).when(route).encoderRegistration();
        return endpoint.commitSelection(route);
    }

    static void commitUnqueued(PrefillEndpoint endpoint, long requestId, long predictedMs) {
        RequestResult item = org.mockito.Mockito.mock(RequestResult.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(String.valueOf(requestId));
        org.flexlb.balance.schedule.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        {
            var reservation = reserveUnqueued(endpoint, item, predictedMs);
            {
                var handoff = endpoint.commitSelection(item);
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    // The endpoint ledger owns this request until authoritative termination.
                }
            }
        }
    }

    static PrefillEndpoint.RequestRegistration reserveUnqueued(
            PrefillEndpoint endpoint, RequestResult item, long predictedMs) {
        org.flexlb.balance.schedule.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        {
            WorkerEndpoint selectedEndpoint = endpoint;

            if (selectedEndpoint == null) {
                throw new IllegalStateException("endpoint is retired");
            }
            return org.flexlb.balance.schedule.SelectionTestSupport.registerPrefill(
                            endpoint, item, predictedMs)
                    .value();
        }
    }

    static void rollback(PrefillEndpoint.RequestRegistration reservation) {
        if (reservation != null) reservation.owner.rollbackSelection(reservation);
    }

    private static void advanceVersion(PrefillEndpoint endpoint, String field) {
        long value = (long) ReflectionTestUtils.getField(endpoint, field);
        ReflectionTestUtils.setField(endpoint, field, value + 1);
    }

    private static DecodeEndpoint.RequestRegistration decodeRegistration(
            DecodeResources.ResourceSnapshot snapshot, String requestId) {
        var request = snapshot.requests().get(requestId);
        return request == null ? null : request.reservation();
    }

    private static void prepareResponse(WorkerStatus status, WorkerStatusResponse response) {
        // This helper models a successful status poll for an active generation.
        // WorkerCleanup tests must exercise the explicit dead-status path instead.
        response.setAlive(true);
        if (response.getRole() == null) {
            response.setRole(status.getRole());
        }
        long committedVersion = status.appliedStatusCursor().statusVersion();
        Long responseVersion = response.getStatusVersion();
        if (responseVersion == null
                || responseVersion <= 0L
                || responseVersion <= committedVersion) {
            response.setStatusVersion(Math.max(1L, committedVersion + 1L));
        }
        if (response.getLatestFinishedVersion() == null) {
            response.setLatestFinishedVersion(
                    status.appliedStatusCursor().latestFinishedTaskVersion());
        }
    }

    /**
     * Rolls back test-owned resource preparation when its scope closes.
     */
    public static final class PreparationScope implements AutoCloseable {
        private final Runnable rollback;

        private PreparationScope(Runnable rollback) {
            this.rollback = rollback;
        }

        @Override
        public void close() {
            rollback.run();
        }
    }
}
