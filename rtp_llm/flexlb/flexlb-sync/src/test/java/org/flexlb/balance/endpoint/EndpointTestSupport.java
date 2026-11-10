package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.DeliveryCoordinator;
import org.flexlb.balance.scheduler.DeliveryExecution;
import org.flexlb.balance.scheduler.RequestExecutionTestSupport;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.slf4j.Logger;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Shared fixtures for exact endpoint and resource ownership contracts.
 */
public final class EndpointTestSupport {
    private static final java.util.Map<PrefillEndpoint, PrefillEndpoint> RESOURCE_FIXTURES =
            java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>());
    private static final java.util.Map<
                    PrefillEndpoint, org.flexlb.balance.scheduler.PrefillQueueScheduler>
            WORK = java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>());
    private static final CapacityBoundary PARKED_BOUNDARY =
            CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH);
    static final DeliveryMetricsReporter NOOP_TELEMETRY =
            org.mockito.Mockito.mock(DeliveryMetricsReporter.class);

    private EndpointTestSupport() {}

    public static boolean isQueued(PrefillEndpoint endpoint, RouteResult route) {
        return resources(endpoint).captureQueue(Integer.MAX_VALUE).items().stream()
                .anyMatch(item -> item == route);
    }

    public static PrefillEndpoint resources(PrefillEndpoint endpoint) {
        return RESOURCE_FIXTURES.getOrDefault(endpoint, endpoint);
    }

    public static org.flexlb.balance.projection.RequestTimeEstimation.SelectionVersion versions(
            PrefillEndpoint endpoint) {
        var state = resources(endpoint);
        return withStateLock(
                state.stateLock(),
                () ->
                        new org.flexlb.balance.projection.RequestTimeEstimation.SelectionVersion(
                                state.getStatus().getGenerationId(),
                                (long) ReflectionTestUtils.getField(state, "requestStateVersion"),
                                (long)
                                        ReflectionTestUtils.getField(
                                                state, "schedulingInputVersion")));
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

    public static WorkSnapshot committedWork(PrefillEndpoint endpoint) {
        var ledger = resources(endpoint);
        var clock = (java.util.function.LongSupplier) ReflectionTestUtils.getField(ledger, "clock");
        WorkSnapshot work =
                withStateLock(
                        ledger.stateLock(),
                        () ->
                                ReflectionTestUtils.invokeMethod(
                                        ledger,
                                        "captureWorkLocked",
                                        clock.getAsLong(),
                                        false,
                                        null));
        return work.calculate();
    }

    public static WorkSnapshot commitDirect(PrefillEndpoint endpoint, RouteResult route) {
        var work = endpoint.commitDirect(route);
        if (work == null) return null;
        return work;
    }

    public static org.flexlb.balance.scheduler.PrefillQueueScheduler worker(
            String key,
            PrefillEndpoint endpoint,
            org.flexlb.balance.scheduler.QueueExecutionSettings settings,
            DeliveryCoordinator dispatcher,
            PrefillEndpoint resources,
            DeliveryExecution execution) {
        ReflectionTestUtils.setField(resources, "queueSettings", settings);
        stubResources(endpoint, resources);
        ReflectionTestUtils.setField(dispatcher, "execution", execution);
        return new org.flexlb.balance.scheduler.PrefillQueueScheduler(
                key, endpoint, settings, dispatcher);
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
                        "rollback",
                        "queueDepth",
                        "releaseQueuedRequest",
                        "queueCounters",
                        "captureQueue",
                        "addQueuedLocked",
                        "closeQueue",
                        "queueStopped",
                        "selectionCurrentLocked",
                        "admissionSummary",
                        "requestCapacityAvailable",
                        "batchCapacityAvailable",
                        "commitStatusLocked",
                        "retireResources",
                        "cleanupCandidates",
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
                                        !endpoint.isRetired()
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

    public static PrefillEndpoint.Retirement retireResources(PrefillEndpoint endpoint) {
        return resources(endpoint).retireResources();
    }

    public static RouteResult enqueue(PrefillEndpoint endpoint, RouteResult route) {
        var ledger = resources(endpoint);
        var lock = ledger.stateLock();
        return withStateLock(
                lock,
                () -> {
                    var registered =
                            org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                    ledger, route, route.prefillWorkMs(), true);
                    if (!registered.accepted()) return null;
                    boolean published = false;
                    try {
                        var queued = ledger.addQueuedLocked(route);
                        published = queued != null;
                        return queued;
                    } finally {
                        if (!published) ledger.rollback(registered.value());
                    }
                });
    }

    public static RouteResult removeQueued(PrefillEndpoint endpoint, RouteResult route) {
        var ledger = resources(endpoint);
        var lock = ledger.stateLock();
        return withStateLock(
                lock,
                () -> {
                    return Boolean.TRUE.equals(
                                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                                            ledger, "undoEnqueueLocked", route))
                            ? route
                            : null;
                });
    }

    public static PrefillEndpoint prefillEndpoint(
            java.util.concurrent.locks.ReentrantLock lock,
            PrefillRequestQueue queue,
            java.util.function.LongSupplier clock) {
        var config = new org.flexlb.config.FlexlbConfig();
        var status =
                WorkerStatus.createDiscovered(
                        org.flexlb.dao.route.RoleType.PREFILL,
                        "g",
                        "127.0.0.1",
                        8080,
                        8090,
                        "test");
        var endpoint =
                new PrefillEndpoint(
                        status,
                        config,
                        org.mockito.Mockito.mock(DeliveryCoordinator.class).mode(),
                        org.mockito.Mockito.mock(DeliveryMetricsReporter.class),
                        new org.flexlb.balance.scheduler.CapacityNotifier(),
                        config.isQueue()
                                ? org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                        config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config))
                                : null,
                        () -> OrderedPrefillRequestQueue.fifo(16));
        ReflectionTestUtils.setField(endpoint, "stateLock", lock);
        ReflectionTestUtils.setField(endpoint, "queuedRequests", queue);
        ReflectionTestUtils.setField(endpoint, "clock", clock);
        return endpoint;
    }

    public static WorkSnapshot commitQueued(
            PrefillEndpoint endpoint, PrefillEndpoint.DeliverySelection plan) {
        var lock = endpoint.stateLock();
        return withStateLock(
                lock,
                () -> {
                    return endpoint.dequeueForDelivery(plan).value();
                });
    }

    public static void completeRetirement(
            EndpointRegistry.Retirement retirement,
            CacheAwareService cacheAwareService,
            Logger logger) {
        ReflectionTestUtils.invokeMethod(retirement, "complete", cacheAwareService, logger, true);
    }

    public static void awaitRetirement(WorkerEndpoint endpoint) {
        Object lifecycle = endpoint;
        ReflectionTestUtils.invokeMethod(lifecycle, "awaitRetirement", 30_000L);
    }

    public static List<DecodeResources.DecodeRoutingView> decodeRoutingSnapshot(
            EndpointRegistry registry, String group) {
        var views = new java.util.ArrayList<DecodeResources.DecodeRoutingView>();
        registry.decodeRoutingSnapshot(group, views);
        return List.copyOf(views);
    }

    public static boolean isQueued(DecodeResources.ResourceSnapshot snapshot, String requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && request.phase().isMasterQueued();
    }

    public static boolean isReserved(DecodeResources.ResourceSnapshot snapshot, String requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && !request.phase().isEngineConfirmed();
    }

    /**
     * Check shadow estimates against production capacity; these fixtures have no released-request selected for preemption KV
     * hold.
     */
    public static long expectedReservedKv(DecodeResources.ResourceSnapshot snapshot) {
        long expected = 0L;
        for (var request : snapshot.requests().values()) {
            if (!request.phase().isEngineConfirmed()) {
                expected += request.expectedKvTokens();
            }
        }
        var fields = snapshot.routing().workerStatus().fields();
        long reported =
                fields.totalKvCacheTokens() > 0L
                        ? Math.max(
                                0L, fields.totalKvCacheTokens() - fields.availableKvCacheTokens())
                        : 0L;
        org.junit.jupiter.api.Assertions.assertEquals(
                DecodeResources.CapacityRelease.saturatedAddNonNegative(reported, expected),
                snapshot.routing().realKvUsed(),
                "reported + shadow KV must match the incrementally maintained capacity");
        return expected;
    }

    /**
     * Register a selected Decode request before delivery through the production admission boundary.
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
        return decodeReservation(endpoint, Long.toString(requestId));
    }

    public static DecodeResources.RequestReservation decodeReservation(
            DecodeEndpoint endpoint, String requestId) {
        return endpoint.isRetired()
                ? null
                : decodeReservation(endpoint.resourceSnapshot(), requestId);
    }

    public static PrefillEndpoint unstartedPrefill(
            org.flexlb.config.FlexlbConfig config,
            WorkerStatus status,
            DeliveryCoordinator delivery,
            AbstractRequestScheduler scheduler) {
        var endpoint =
                EndpointTestSupport.prefill(
                        status,
                        config,
                        delivery,
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler),
                        org.mockito.Mockito.mock(DeliveryMetricsReporter.class));
        endpoint.enableQueue(
                org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                        config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
        ReflectionTestUtils.setField(
                delivery,
                "execution",
                org.flexlb.balance.scheduler.SchedulerTestSupport.queueDependencies(
                        scheduler, config));
        var worker =
                org.mockito.Mockito.mock(
                        org.flexlb.balance.scheduler.PrefillQueueScheduler.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(
                                        status.getLogicalIpPort(),
                                        endpoint,
                                        org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                                config,
                                                org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                                        config)),
                                        delivery)
                                .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));

        RequestExecutionTestSupport.attachDriver(
                worker,
                scheduler,
                org.flexlb.balance.scheduler.RequestExecutionTestSupport::schedule);
        org.flexlb.balance.scheduler.DeliveryTestSupport.installQueueFixture(worker, delivery);
        org.flexlb.balance.endpoint.EndpointTestSupport.attachWork(endpoint, worker);
        org.springframework.test.util.ReflectionTestUtils.setField(
                endpoint,
                "queueSettings",
                org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                        config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));

        return endpoint;
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

    public static org.flexlb.balance.scheduler.PrefillQueueScheduler attach(
            org.flexlb.balance.scheduler.PrefillQueueScheduler work) {
        attachWork(work.prefillEndpoint(), work);
        return work;
    }

    public static void attachWork(
            PrefillEndpoint endpoint, org.flexlb.balance.scheduler.PrefillQueueScheduler work) {
        ReflectionTestUtils.setField(resources(endpoint), "queueSettings", work.settings());
        WORK.put(endpoint, work);
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "queueEnabled", true);
        if (org.mockito.Mockito.mockingDetails(endpoint).isMock()
                && !org.mockito.Mockito.mockingDetails(endpoint).isSpy()) {
            var availability =
                    org.springframework.test.util.ReflectionTestUtils.getField(
                            endpoint, "capacityNotifier");
            if (availability == null)
                org.springframework.test.util.ReflectionTestUtils.setField(
                        endpoint,
                        "capacityNotifier",
                        new org.flexlb.balance.scheduler.CapacityNotifier());
            org.mockito.Mockito.doAnswer(
                            call -> {
                                org.flexlb.balance.scheduler.RequestExecutionTestSupport
                                        .endpointChanged(endpoint, call.getArgument(0));
                                return null;
                            })
                    .when(endpoint)
                    .signal(org.mockito.Mockito.any());
            org.mockito.Mockito.doCallRealMethod().when(endpoint).queueEnabled();
            org.mockito.Mockito.doCallRealMethod()
                    .when(endpoint)
                    .tryEnqueueRoute(org.mockito.Mockito.any());
        }
    }

    public static DecodeEndpoint decode(
            WorkerStatus status, org.flexlb.balance.scheduler.RequestRepository repository) {
        var endpoint =
                new DecodeEndpoint(
                        status, repository, new org.flexlb.balance.scheduler.CapacityNotifier());
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
                                                org.flexlb.balance.scheduler.RequestContext.class);
                                var data =
                                        org.mockito.Mockito.mock(
                                                org.flexlb.balance.scheduler.RequestData.class);
                                var snapshot =
                                        org.mockito.Mockito.mock(
                                                org.flexlb.balance.scheduler.ContextResult.Snapshot
                                                        .class);
                                var route = org.mockito.Mockito.mock(RouteResult.class);
                                org.mockito.Mockito.when(context.data()).thenReturn(data);
                                org.mockito.Mockito.when(data.getRequestScheduler())
                                        .thenReturn(owner);
                                org.mockito.Mockito.when(data.getRequestId())
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

    public static PrefillEndpoint realPrefill(
            WorkerStatus status,
            org.flexlb.config.FlexlbConfig config,
            DeliveryCoordinator delivery,
            org.flexlb.balance.scheduler.RequestRepository repository,
            DeliveryMetricsReporter reporter) {
        var endpoint =
                new PrefillEndpoint(
                        status,
                        config,
                        delivery.mode(),
                        reporter,
                        new org.flexlb.balance.scheduler.CapacityNotifier(),
                        config.isQueue()
                                ? org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                        config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config))
                                : null,
                        () -> OrderedPrefillRequestQueue.fifo(16));
        org.flexlb.balance.scheduler.SchedulerTestSupport.associateEndpoint(endpoint, repository);
        return endpoint;
    }

    public static PrefillEndpoint prefill(
            WorkerStatus status,
            org.flexlb.config.FlexlbConfig config,
            DeliveryCoordinator delivery,
            org.flexlb.balance.scheduler.RequestRepository repository,
            DeliveryMetricsReporter reporter) {
        var endpoint =
                new PrefillEndpoint(
                        status,
                        config,
                        delivery.mode(),
                        reporter,
                        new org.flexlb.balance.scheduler.CapacityNotifier(),
                        config.isQueue()
                                ? org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                        config,
                                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config))
                                : null,
                        () -> OrderedPrefillRequestQueue.fifo(16)) {
                    @Override
                    public boolean enqueueRoute(
                            RouteResult route,
                            org.flexlb.balance.scheduler.QueueExecutionSettings settings) {
                        enableQueueRuntime(
                                this,
                                settings,
                                org.flexlb.balance.scheduler.SchedulerTestSupport.queueDependencies(
                                        route.ctx().data().getRequestScheduler(), config),
                                delivery);
                        return super.enqueueRoute(route, settings);
                    }
                };
        org.flexlb.balance.scheduler.SchedulerTestSupport.associateEndpoint(endpoint, repository);
        return endpoint;
    }

    public static void enableQueueRuntime(
            PrefillEndpoint endpoint,
            org.flexlb.balance.scheduler.QueueExecutionSettings settings,
            DeliveryExecution dependencies,
            DeliveryCoordinator delivery) {
        var state = endpoint;
        var created =
                withStateLock(
                        state.stateLock(),
                        () -> {
                            if (endpoint.isRetired() || requestExecution(endpoint) != null) {
                                return null;
                            }
                            ReflectionTestUtils.setField(state, "queueSettings", settings);
                            state.enableQueue(settings);
                            ReflectionTestUtils.setField(delivery, "execution", dependencies);
                            var worker =
                                    org.mockito.Mockito.mock(
                                            org.flexlb.balance.scheduler.PrefillQueueScheduler
                                                    .class,
                                            org.mockito.Mockito.withSettings()
                                                    .useConstructor(
                                                            endpoint.ipPort(),
                                                            endpoint,
                                                            settings,
                                                            delivery)
                                                    .defaultAnswer(
                                                            org.mockito.Mockito
                                                                    .CALLS_REAL_METHODS));
                            org.flexlb.balance.scheduler.DeliveryTestSupport.installQueueFixture(
                                    worker, delivery);
                            org.flexlb.balance.endpoint.EndpointTestSupport.attachWork(
                                    endpoint, worker);
                            RequestExecutionTestSupport.attachDriver(
                                    worker,
                                    null,
                                    org.flexlb.balance.scheduler.RequestExecutionTestSupport
                                            ::schedule);
                            return worker;
                        });
        if (created != null) RequestExecutionTestSupport.start(created);
    }

    public static org.flexlb.balance.scheduler.PrefillQueueScheduler requestExecution(
            PrefillEndpoint endpoint) {
        return WORK.get(endpoint);
    }

    public static boolean releaseRequest(PrefillEndpoint state, RouteResult exact) {
        return state.releaseRequest(exact);
    }

    public static PreparationScope preparation(PrefillEndpoint.RequestRecord reservation) {
        return new PreparationScope(() -> rollback(reservation));
    }

    static DecodeEndpoint decodeEndpoint(WorkerStatus status) {
        return new DecodeEndpoint(
                status,
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.RequestRepository.class),
                new org.flexlb.balance.scheduler.CapacityNotifier());
    }

    static DecodeResources.Acquisition<
                    DecodeResources.ReserveStatus, DecodeResources.RequestReservation>
            reserveQueuedDecode(
                    DecodeEndpoint endpoint,
                    long id,
                    long hard,
                    long expected,
                    int priority,
                    DecodeResources.AdmissionCapacity capacity) {
        {
            var selectedEndpoint = endpoint;

            return capacity == null
                    ? org.flexlb.balance.scheduler.SelectionTestSupport.registerQueuedDecode(
                            endpoint, String.valueOf(id), hard, expected, priority)
                    : endpoint.tryRegisterSelection(
                            endpoint.routingView(),
                            String.valueOf(id),
                            hard,
                            expected,
                            priority,
                            capacity);
        }
    }

    static int evictPrefill(
            PrefillEndpoint state, long ttlMs, java.util.function.Predicate<String> retained) {
        var candidates = new java.util.HashSet<RouteResult>();
        for (RouteResult route : state.cleanupCandidates()) {
            if (!retained.test(route.requestId())) {
                candidates.add(route);
            }
        }
        return state.evictExpiredInflight(ttlMs, candidates);
    }

    static DecodeEndpoint.CleanupResult evictDecode(
            DecodeEndpoint state, long ttlMs, java.util.function.Predicate<String> retained) {
        var candidates = state.cleanupCandidates();
        candidates.keySet().removeIf(retained::test);
        return state.expireReservations(ttlMs, candidates);
    }

    static boolean handoffPreemption(DecodeEndpoint endpoint, long token) {
        var requestsToPreempt =
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(DecodeResources.DecodeRequestView::claimedForPreemption)
                        .toList();
        return !requestsToPreempt.isEmpty()
                && requestsToPreempt.stream()
                        .allMatch(
                                requestToPreempt ->
                                        endpoint.updatePreemption(
                                                        token,
                                                        DecodeResources.PreemptionUpdate.handedOff(
                                                                new DecodeResources
                                                                        .RequestReservation(
                                                                        endpoint.getStatus()
                                                                                .getGenerationId(),
                                                                        requestToPreempt
                                                                                .requestId(),
                                                                        requestToPreempt
                                                                                .reservationToken())))
                                                .status()
                                                .applied());
    }

    static WorkerEndpoint.StatusChange.Prefill reconcile(
            PrefillEndpoint state,
            WorkerStatus.StatusObservation observation,
            java.util.function.ToLongFunction<List<RouteResult>> predictor) {
        var lock = state.stateLock();
        var reduction = withStateLock(lock, () -> state.prepareStatusLocked(observation, true));
        var predictions = new java.util.HashMap<Long, Long>();
        reduction
                .predictionInputs()
                .forEach(
                        (batchId, members) ->
                                predictions.put(batchId, predictor.applyAsLong(members)));
        return withStateLock(
                lock,
                () -> {
                    return java.util.Objects.requireNonNull(
                            state.commitStatusLocked(reduction, predictions));
                });
    }

    static WorkSnapshot commitBatch(
            PrefillEndpoint state,
            org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput reservation,
            List<RouteResult> items,
            long predictedMs) {
        return withStateLock(
                state.stateLock(),
                () -> {
                    return state.dequeueForDelivery(
                                    org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                            items, reservation, new long[] {predictedMs}))
                            .value();
                });
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
                    status.prepareNewStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange projection =
                    endpoint.applyPreparedStatus(status, prepared);
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
                    status.prepareNewStatus(
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
        prepareResponse(status, response);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    status.prepareNewStatus(
                            WorkerStatusTestSupport.freezeStatusResponse(status, response));
            return registry.publishPreparedEndpoint(address, status, prepared);
        } finally {
            status.lock.unlock();
        }
    }

    static TestRequestRuntime requestRuntime() {
        return new TestRequestRuntime();
    }

    static DeliveryCoordinator routeStrategy(TestRequestRuntime runtime) {
        DeliveryCoordinator route =
                org.flexlb.balance.scheduler.DeliveryTestSupport.route(NOOP_TELEMETRY);
        return org.flexlb.balance.scheduler.DeliveryTestSupport.parked(PARKED_BOUNDARY);
    }

    static DeliveryCoordinator liveRouteStrategy(TestRequestRuntime runtime) {
        var dispatcher =
                org.mockito.Mockito.spy(
                        org.flexlb.balance.scheduler.DeliveryTestSupport.route(NOOP_TELEMETRY));
        org.flexlb.balance.scheduler.DeliveryTestSupport.stubRouteDelivery(
                dispatcher, runtime.requests, runtime::onCompleted);
        return dispatcher;
    }

    static boolean offer(PrefillEndpoint endpoint, RouteResult item) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        {
            WorkerEndpoint selectedEndpoint = endpoint;

            if (selectedEndpoint == null) {
                return false;
            }
            if (!org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                            endpoint, item, item.prefillWorkMs(), true)
                    .accepted()) return false;
            boolean accepted =
                    endpoint.enqueueRoute(
                            item,
                            org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                                    item.ctx().data().getConfig(),
                                    org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                            item.ctx().data().getConfig())));
            if (accepted) endpoint.signal(PrefillEndpoint.Signal.INPUT);
            return accepted;
        }
    }

    static void commitUnqueued(PrefillEndpoint endpoint, long requestId, long predictedMs) {
        RouteResult item = org.mockito.Mockito.mock(RouteResult.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(String.valueOf(requestId));
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        {
            var reservation = reserveUnqueued(endpoint, item, predictedMs);
            {
                var handoff =
                        org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                endpoint, item);
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    // The endpoint ledger owns this request until authoritative termination.
                }
            }
        }
    }

    static PrefillEndpoint.RequestRecord reserveUnqueued(
            PrefillEndpoint endpoint, RouteResult item, long predictedMs) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        {
            WorkerEndpoint selectedEndpoint = endpoint;

            if (selectedEndpoint == null) {
                throw new IllegalStateException("endpoint is retired");
            }
            return org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                            endpoint, item, predictedMs)
                    .value();
        }
    }

    static WorkSnapshot commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends RouteResult> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("batch requires at least one item");
        }
        List<RouteResult> items = List.copyOf(exactItems);
        items.forEach(
                item ->
                        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(
                                endpoint, item));
        return commitBatch(
                endpoint,
                new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(
                        batchId, Integer.MAX_VALUE),
                items,
                predictedMs);
    }

    static void rollback(PrefillEndpoint.RequestRecord reservation) {
        if (reservation != null) reservation.owner.rollback(reservation);
    }

    static WorkSnapshot commitQueuedRoutes(
            PrefillEndpoint state, List<RouteResult> items, long[] predictions) {
        return withStateLock(
                state.stateLock(),
                () -> {
                    return state.dequeueForDelivery(
                                    org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                            items, null, predictions))
                            .value();
                });
    }

    static List<WorkSnapshot> commitRoutes(
            PrefillEndpoint endpoint, long predictedMs, List<? extends RouteResult> exactItems) {
        List<RouteResult> items = List.copyOf(exactItems);
        items.forEach(
                item ->
                        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(
                                endpoint, item));
        long[] predictions = new long[items.size()];
        java.util.Arrays.fill(predictions, predictedMs);
        var lock = endpoint.stateLock();
        return withStateLock(
                lock,
                () -> {
                    return List.of(
                            endpoint.dequeueForDelivery(
                                            org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                                    items, null, predictions))
                                    .value());
                });
    }

    private static DecodeResources.RequestReservation decodeReservation(
            DecodeResources.ResourceSnapshot snapshot, String requestId) {
        var request = snapshot.requests().get(requestId);
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

    private static void prepareResponse(WorkerStatus status, WorkerStatusResponse response) {
        // This helper models a successful status poll for an active generation.
        // Retirement tests must exercise the explicit dead-status path instead.
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

    /**
     * Supplies scheduler callbacks for endpoint tests and records the exact routes reported during Prefill retirement.
     */
    static class TestRequestRuntime {
        private final AbstractRequestScheduler requests =
                org.mockito.Mockito.mock(AbstractRequestScheduler.class);
        private final AbstractRequestScheduler events = requests;
        private final List<PrefillRetirement> prefillRetirements = new CopyOnWriteArrayList<>();

        TestRequestRuntime() {
            org.mockito.Mockito.doAnswer(
                            invocation -> {
                                var evidence =
                                        invocation.getArgument(
                                                1,
                                                org.flexlb.balance.scheduler.ContextResult
                                                        .EndpointRetirement.class);
                                prefillRetirements.add(
                                        new PrefillRetirement(
                                                evidence.route().prefillEndpoint(),
                                                List.of(evidence.route())));
                                onRetired(evidence.route());
                                return null;
                            })
                    .when(events)
                    .onEndpointRetired(org.mockito.Mockito.any(), org.mockito.Mockito.any());
        }

        AbstractRequestScheduler requests() {
            return requests;
        }

        AbstractRequestScheduler events() {
            return events;
        }

        void onCompleted(Delivery.Member claim, Delivery.Result result) {}

        void onRetired(RouteResult item) {}

        List<PrefillRetirement> prefillRetirements() {
            return List.copyOf(prefillRetirements);
        }
    }

    /**
     * Captures a retiring Prefill endpoint and a stable copy of the routes reported for its cleanup.
     */
    record PrefillRetirement(PrefillEndpoint endpoint, List<RouteResult> ownedItems) {
        PrefillRetirement {
            ownedItems = List.copyOf(ownedItems);
        }
    }
}
