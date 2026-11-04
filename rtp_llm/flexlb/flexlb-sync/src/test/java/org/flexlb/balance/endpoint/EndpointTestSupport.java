package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.Delivery;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.RequestExecutionTestSupport;
import org.flexlb.balance.scheduler.RequestLifecycle;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.slf4j.Logger;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/** Shared fixtures for exact endpoint and resource ownership contracts. */
public final class EndpointTestSupport {

    private EndpointTestSupport() {}

    /** Tests retain the caller's generation pin explicitly until their assertions finish. */
    public static Delivery commitDirect(
            PrefillEndpoint endpoint,
            RequestRoute route,
            PrefillEndpoint.RouteReservation reservation) {
        var pin = endpoint.tryPinGeneration();
        if (pin == null) return null;
        Delivery committed = null;
        try {
            var work = endpoint.commitDirect(pin, route, reservation);
            if (work == null) return null;
            committed = new Delivery(endpoint, false, 0, null, "test");
            committed.committed(pin, work, 0, false);
            return committed;
        } finally {
            if (committed == null) pin.close();
        }
    }

    public static org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork worker(
            String key,
            PrefillEndpoint endpoint,
            org.flexlb.balance.scheduler.QueueExecutionSettings settings,
            DeliveryDispatcher dispatcher,
            PrefillEndpoint resources,
            org.flexlb.balance.scheduler.DeliveryExecution execution) {
        stubResources(endpoint, resources);
        ReflectionTestUtils.setField(dispatcher, "execution", execution);
        return new org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork(
                key, endpoint, settings, dispatcher);
    }

    private static final java.util.Map<PrefillEndpoint, PrefillEndpoint> RESOURCE_FIXTURES =
            java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>());

    public static PrefillEndpoint.Retirement retireResources(PrefillEndpoint endpoint) {
        return resources(endpoint).retireResources();
    }

    public static PrefillEndpoint resources(PrefillEndpoint endpoint) {
        return RESOURCE_FIXTURES.getOrDefault(endpoint, endpoint);
    }

    public static RequestRoute enqueue(PrefillEndpoint endpoint, RequestRoute route) {
        var ledger = resources(endpoint);
        var lock = ledger.resourceLock();
        lock.lock();
        try {
            return ledger.addQueuedLocked(route);
        } finally {
            lock.unlock();
        }
    }

    public static RequestRoute removeQueued(PrefillEndpoint endpoint, RequestRoute route) {
        var ledger = resources(endpoint);
        var lock = ledger.resourceLock();
        lock.lock();
        try {
            return ledger.removeQueuedLocked(route) ? route : null;
        } finally {
            lock.unlock();
        }
    }

    /** A scheduling test may mock transport while retaining a real resource ledger. */
    public static void stubResources(PrefillEndpoint endpoint, PrefillEndpoint resources) {
        if (endpoint == resources) return;
        RESOURCE_FIXTURES.put(endpoint, resources);
        org.junit.jupiter.api.Assertions.assertTrue(
                org.mockito.Mockito.mockingDetails(endpoint).isMock(),
                "resource delegation is only for a mocked transport endpoint");
        var names =
                java.util.Set.of(
                        "resourceLock",
                        "progress",
                        "queueCounters",
                        "captureQueue",
                        "schedulingInputsChangedLocked",
                        "addQueuedLocked",
                        "removeQueuedLocked",
                        "closeQueue",
                        "queueStopped",
                        "ownsSelectionLocked",
                        "selectionCurrentLocked",
                        "queueHeadLocked",
                        "admissionSummary",
                        "requestCapacityAvailable",
                        "batchCapacityAvailable",
                        "commitStatusLocked",
                        "retireResources",
                        "cleanupCandidates",
                        "committedSnapshot",
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
                        org.mockito.Mockito.mock(DeliveryDispatcher.class).mode(),
                        org.mockito.Mockito.mock(DeliveryMetricsReporter.class),
                        new org.flexlb.balance.scheduler.PlacementAvailability(),
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config),
                        order -> PrefillActiveIndex.ordered(16, order));
        ReflectionTestUtils.setField(endpoint, "lock", lock);
        ReflectionTestUtils.setField(endpoint, "activeIndex", queue);
        ReflectionTestUtils.setField(endpoint, "clock", clock);
        return endpoint;
    }

    static DecodeEndpoint decodeEndpoint(WorkerStatus status) {
        return new DecodeEndpoint(
                status,
                org.mockito.Mockito.mock(org.flexlb.balance.scheduler.RequestRepository.class),
                new org.flexlb.balance.scheduler.PlacementAvailability());
    }

    static java.util.concurrent.locks.ReentrantLock decodeLock(DecodeEndpoint endpoint) {
        return (java.util.concurrent.locks.ReentrantLock)
                ReflectionTestUtils.getField(endpoint, "admissionLock");
    }

    static DecodeResources.Acquisition<
                    DecodeResources.ReserveStatus, DecodeResources.ReservationHandle>
            reserveQueuedDecode(
                    DecodeEndpoint endpoint,
                    long id,
                    long hard,
                    long expected,
                    int priority,
                    DecodeResources.AdmissionCapacity capacity) {
        try (var pin = endpoint.tryPinGeneration()) {
            return endpoint.tryReserveQueuedRequest(pin, id, hard, expected, priority, capacity);
        }
    }

    public static org.flexlb.balance.scheduler.Delivery commitQueued(
            PrefillEndpoint endpoint, org.flexlb.balance.scheduler.Delivery plan) {
        var lock = endpoint.resourceLock();
        lock.lock();
        try {
            return endpoint.commitLocked(plan);
        } finally {
            lock.unlock();
        }
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

    public static boolean isQueued(DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && request.phase().isMasterQueued();
    }

    public static boolean isReserved(DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(requestId);
        return request != null && !request.phase().isEngineConfirmed();
    }

    /**
     * Check shadow estimates against production capacity; these fixtures have no released-victim KV
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

    /** Build a real local reservation without publishing it to the master queue. */
    public static DecodeResources.ReservationHandle reserveUnqueuedDecode(
            DecodeEndpoint endpoint,
            WorkerEndpoint.GenerationPin pin,
            long requestId,
            long hardKv,
            long expectedKv,
            int priority) {
        endpoint.requirePinnedGeneration(pin);
        var state = endpoint;
        var lock =
                (java.util.concurrent.locks.ReentrantLock)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                state, "admissionLock");
        lock.lock();
        try {
            if (!Boolean.TRUE.equals(
                    org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                            state, "requestIdAvailable", requestId))) {
                throw new IllegalStateException("Decode request id is already owned: " + requestId);
            }
            DecodeResources.Acquisition<
                            DecodeResources.ReserveStatus, DecodeResources.ReservationHandle>
                    acquired =
                            org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                                    state,
                                    "createReservation",
                                    requestId,
                                    hardKv,
                                    expectedKv,
                                    priority,
                                    org.flexlb.enums.DecodeTaskPhase.LOCAL_RESERVED);
            return acquired.handle();
        } finally {
            lock.unlock();
        }
    }

    /** Read shadow ownership for assertions, without granting a production lookup capability. */
    public static DecodeResources.ReservationHandle decodeReservation(
            DecodeEndpoint endpoint, long requestId) {
        return endpoint.isRetired()
                ? null
                : decodeReservation(endpoint.resourceSnapshot(), requestId);
    }

    private static DecodeResources.ReservationHandle decodeReservation(
            DecodeResources.ResourceSnapshot snapshot, long requestId) {
        var request = snapshot.requests().get(requestId);
        if (request == null || request.reservationToken() <= 0L) {
            return null;
        }
        return switch (request.phase()) {
            case LOCAL_RESERVED, MASTER_QUEUED_NOT_DISPATCHED, ENGINE_MAY_HAVE_SEEN ->
                    new DecodeResources.ReservationHandle(
                            snapshot.routing().generationId(),
                            requestId,
                            request.reservationToken());
            default -> null;
        };
    }

    static int evictPrefill(
            PrefillEndpoint state, long ttlMs, java.util.function.LongPredicate retained) {
        var candidates = new java.util.HashSet<RequestRoute>();
        for (RequestRoute route : state.cleanupCandidates()) {
            if (!retained.test(route.requestId())) {
                candidates.add(route);
            }
        }
        return state.evictExpiredInflight(ttlMs, candidates);
    }

    static DecodeEndpoint.CleanupResult evictDecode(
            DecodeEndpoint state, long ttlMs, java.util.function.LongPredicate retained) {
        var candidates = state.cleanupCandidates();
        candidates.keySet().removeIf(retained::test);
        return state.expireReservations(ttlMs, candidates);
    }

    static boolean handoffPreemption(DecodeEndpoint endpoint, long token) {
        var victims =
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(DecodeResources.DecodeRequestView::claimedForPreemption)
                        .toList();
        return !victims.isEmpty()
                && victims.stream()
                        .allMatch(
                                victim ->
                                        endpoint.updatePreemption(
                                                        token,
                                                        DecodeResources.PreemptionUpdate.handedOff(
                                                                new DecodeResources
                                                                        .ReservationHandle(
                                                                        endpoint.getStatus()
                                                                                .getGenerationId(),
                                                                        victim.requestId(),
                                                                        victim.reservationToken())))
                                                .status()
                                                .applied());
    }

    static WorkerEndpoint.StatusChange.Prefill reconcile(
            PrefillEndpoint state,
            WorkerStatus.StatusObservation observation,
            java.util.function.ToLongFunction<List<RequestRoute>> predictor) {
        var lock = state.resourceLock();
        lock.lock();
        PrefillEndpoint.StatusReduction reduction;
        try {
            reduction = state.prepareStatusLocked(observation, true);
        } finally {
            lock.unlock();
        }
        var predictions = new java.util.HashMap<Long, Long>();
        reduction
                .predictionInputs()
                .forEach(
                        (batchId, members) ->
                                predictions.put(batchId, predictor.applyAsLong(members)));
        lock.lock();
        try {
            return java.util.Objects.requireNonNull(
                    state.commitStatusLocked(reduction, predictions));
        } finally {
            lock.unlock();
        }
    }

    static org.flexlb.balance.scheduler.Delivery commitBatch(
            PrefillEndpoint state,
            PrefillEndpoint.BatchReservation reservation,
            List<RequestRoute> items,
            long predictedMs) {
        state.resourceLock().lock();
        try {
            return state.commitLocked(
                    org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                            items,
                            reservation,
                            new long[] {predictedMs},
                            null,
                            System.currentTimeMillis()));
        } finally {
            state.resourceLock().unlock();
        }
    }

    public static PrefillEndpoint unstartedPrefill(
            org.flexlb.config.FlexlbConfig config,
            WorkerStatus status,
            DeliveryDispatcher delivery,
            RequestLifecycle scheduler) {
        var endpoint =
                EndpointTestSupport.prefill(
                        status,
                        config,
                        delivery,
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler),
                        org.mockito.Mockito.mock(DeliveryMetricsReporter.class));
        var state = (PrefillEndpoint) endpoint;
        state.resourceLock().lock();
        try {
            state.enableQueueLocked(
                    PrefillActiveIndex.ordered(
                            16,
                            org.flexlb.balance.scheduler.QueueExecutionSettings.queueOrder(
                                    config.isPriorityOrdering())));
        } finally {
            state.resourceLock().unlock();
        }
        ReflectionTestUtils.setField(
                delivery,
                "execution",
                org.flexlb.balance.scheduler.SchedulerTestSupport.queueDependencies(
                        scheduler, config));
        var worker =
                org.mockito.Mockito.mock(
                        org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(
                                        status.getIpPort(),
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

    private static final java.util.Map<
                    PrefillEndpoint,
                    org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork>
            WORK = java.util.Collections.synchronizedMap(new java.util.IdentityHashMap<>());

    public static org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork requestExecution(
            PrefillEndpoint endpoint) {
        return WORK.get(endpoint);
    }

    public static org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork attach(
            org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork work) {
        attachWork(work.prefillEndpoint(), work);
        return work;
    }

    public static void attachWork(
            PrefillEndpoint endpoint,
            org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork work) {
        WORK.put(endpoint, work);
        endpoint.observeQueueWait(work::admissionBlockLocked, work::getLatestQueueWaitSnapshot);
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "queueEnabled", true);
        if (org.mockito.Mockito.mockingDetails(endpoint).isMock()
                && !org.mockito.Mockito.mockingDetails(endpoint).isSpy()) {
            var availability =
                    org.springframework.test.util.ReflectionTestUtils.getField(
                            endpoint, "placementAvailability");
            if (availability == null)
                org.springframework.test.util.ReflectionTestUtils.setField(
                        endpoint,
                        "placementAvailability",
                        new org.flexlb.balance.scheduler.PlacementAvailability());
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
                    .offerQueued(org.mockito.Mockito.any());
            org.mockito.Mockito.doCallRealMethod()
                    .when(endpoint)
                    .removeQueued(org.mockito.Mockito.any());
        }
    }

    public static DecodeEndpoint decode(
            WorkerStatus status, org.flexlb.balance.scheduler.RequestRepository repository) {
        var endpoint =
                new DecodeEndpoint(
                        status,
                        repository,
                        new org.flexlb.balance.scheduler.PlacementAvailability());
        if (org.mockito.Mockito.mockingDetails(repository).isMock()) {
            org.mockito.Mockito.when(repository.findActive(org.mockito.ArgumentMatchers.anyLong()))
                    .thenAnswer(
                            call -> {
                                long id = call.getArgument(0);
                                var owner = repository.ownerOf(id);
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
                                var route = org.mockito.Mockito.mock(RequestRoute.class);
                                org.mockito.Mockito.when(context.data()).thenReturn(data);
                                org.mockito.Mockito.when(data.getLifecycle()).thenReturn(owner);
                                org.mockito.Mockito.when(data.getRequestId()).thenReturn(id);
                                org.mockito.Mockito.when(context.snapshot()).thenReturn(snapshot);
                                org.mockito.Mockito.when(snapshot.route()).thenReturn(route);
                                org.mockito.Mockito.when(route.ctx()).thenReturn(context);
                                org.mockito.Mockito.when(route.decodeEp()).thenReturn(endpoint);
                                return context;
                            });
        }
        return endpoint;
    }

    public static PrefillEndpoint realPrefill(
            WorkerStatus status,
            org.flexlb.config.FlexlbConfig config,
            DeliveryDispatcher delivery,
            org.flexlb.balance.scheduler.RequestRepository repository,
            DeliveryMetricsReporter reporter) {
        var endpoint =
                new PrefillEndpoint(
                        status,
                        config,
                        delivery.mode(),
                        reporter,
                        new org.flexlb.balance.scheduler.PlacementAvailability(),
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config),
                        order -> PrefillActiveIndex.ordered(16, order));
        org.flexlb.balance.scheduler.SchedulerTestSupport.associateEndpoint(endpoint, repository);
        return endpoint;
    }

    public static PrefillEndpoint prefill(
            WorkerStatus status,
            org.flexlb.config.FlexlbConfig config,
            DeliveryDispatcher delivery,
            org.flexlb.balance.scheduler.RequestRepository repository,
            DeliveryMetricsReporter reporter) {
        var endpoint =
                new PrefillEndpoint(
                        status,
                        config,
                        delivery.mode(),
                        reporter,
                        new org.flexlb.balance.scheduler.PlacementAvailability(),
                        org.flexlb.balance.planner.BuiltinGroupPlanner.from(config),
                        order -> PrefillActiveIndex.ordered(16, order)) {
                    @Override
                    public boolean offerPinned(
                            GenerationPin pin,
                            RequestRoute route,
                            org.flexlb.balance.scheduler.QueueExecutionSettings settings) {
                        enableQueueRuntime(
                                this,
                                settings,
                                org.flexlb.balance.scheduler.SchedulerTestSupport.queueDependencies(
                                        route.ctx().data().getLifecycle(), config),
                                delivery);
                        return super.offerPinned(pin, route, settings);
                    }
                };
        org.flexlb.balance.scheduler.SchedulerTestSupport.associateEndpoint(endpoint, repository);
        return endpoint;
    }

    public static void enableQueueRuntime(
            PrefillEndpoint endpoint,
            org.flexlb.balance.scheduler.QueueExecutionSettings settings,
            org.flexlb.balance.scheduler.DeliveryExecution dependencies,
            DeliveryDispatcher delivery) {
        var state = endpoint;
        state.resourceLock().lock();
        try {
            if (requestExecution(endpoint) != null) {
                return;
            }
            state.enableQueueLocked(
                    PrefillActiveIndex.ordered(
                            16,
                            org.flexlb.balance.scheduler.QueueExecutionSettings.queueOrder(
                                    settings.priorityOrdering())));
            ReflectionTestUtils.setField(delivery, "execution", dependencies);
            var worker =
                    org.mockito.Mockito.mock(
                            org.flexlb.balance.scheduler.QueuedRequestScheduler.EndpointWork.class,
                            org.mockito.Mockito.withSettings()
                                    .useConstructor(endpoint.ipPort(), endpoint, settings, delivery)
                                    .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
            org.flexlb.balance.scheduler.DeliveryTestSupport.installQueueFixture(worker, delivery);
            org.flexlb.balance.endpoint.EndpointTestSupport.attachWork(endpoint, worker);
            org.springframework.test.util.ReflectionTestUtils.setField(
                    endpoint, "queueSettings", settings);

            org.springframework.test.util.ReflectionTestUtils.setField(
                    endpoint, "projectionSource", null);
            RequestExecutionTestSupport.attachDriver(
                    worker,
                    null,
                    org.flexlb.balance.scheduler.RequestExecutionTestSupport::schedule);
            RequestExecutionTestSupport.start(worker);
        } finally {
            state.resourceLock().unlock();
        }
    }

    static RequestLifecycle noopEventSink() {
        return org.mockito.Mockito.mock(RequestLifecycle.class);
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

    static TestRequestRuntime requestRuntime() {
        return new TestRequestRuntime();
    }

    static DeliveryDispatcher routeStrategy(TestRequestRuntime runtime) {
        DeliveryDispatcher route =
                org.flexlb.balance.scheduler.DeliveryTestSupport.route(NOOP_TELEMETRY);
        return org.flexlb.balance.scheduler.DeliveryTestSupport.parked(PARKED_BOUNDARY);
    }

    static DeliveryDispatcher liveRouteStrategy(TestRequestRuntime runtime) {
        return org.flexlb.balance.scheduler.DeliveryTestSupport.route(NOOP_TELEMETRY);
    }

    private static final CapacityBoundary PARKED_BOUNDARY =
            CapacityBoundary.unavailable(CapacityBoundary.Resource.PREFILL_BATCH);

    static boolean offer(PrefillEndpoint endpoint, RequestRoute item) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                return false;
            }
            return endpoint.offerPinned(
                    pin,
                    item,
                    org.flexlb.balance.scheduler.QueueExecutionSettings.capture(
                            item.ctx().data().getConfig(),
                            org.flexlb.balance.planner.BuiltinGroupPlanner.from(
                                    item.ctx().data().getConfig())));
        }
    }

    static PrefillEndpoint.RouteReservation reserveUnqueued(
            PrefillEndpoint endpoint, RequestRoute item, long predictedMs) {
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new IllegalStateException("endpoint is retired");
            }
            return endpoint.reserveUnqueuedRoute(pin, item, predictedMs).value();
        }
    }

    static void commitUnqueued(PrefillEndpoint endpoint, long requestId, long predictedMs) {
        RequestRoute item = org.mockito.Mockito.mock(RequestRoute.class);
        org.mockito.Mockito.when(item.requestId()).thenReturn(requestId);
        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(endpoint, item);
        {
            var reservation = reserveUnqueued(endpoint, item, predictedMs);
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                    var handoff =
                            org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                    endpoint, item, reservation)) {
                // The endpoint ledger owns this request until authoritative termination.
            }
        }
    }

    static org.flexlb.balance.scheduler.Delivery commitBatch(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<? extends RequestRoute> exactItems) {
        if (exactItems.isEmpty()) {
            throw new IllegalArgumentException("batch requires at least one item");
        }
        List<RequestRoute> items = List.copyOf(exactItems);
        items.forEach(
                item ->
                        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(
                                endpoint, item));
        CapacityBoundary.Attempt<PrefillEndpoint.BatchReservation> result =
                endpoint.reserveBatch(batchId, Integer.MAX_VALUE);
        if (!result.accepted()) {
            throw new IllegalStateException("batch reservation rejected: " + result.boundary());
        }
        {
            PrefillEndpoint.BatchReservation reservation = result.value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                PrefillEndpoint state = (PrefillEndpoint) endpoint;
                return commitBatch(state, reservation, items, predictedMs);
            }
        }
    }

    public static boolean releaseRequest(PrefillEndpoint state, RequestRoute exact) {
        return state.releaseRequest(exact);
    }

    public static PreparationScope preparation(PrefillEndpoint.Reservation reservation) {
        return new PreparationScope(reservation);
    }

    static void rollback(PrefillEndpoint.Reservation reservation) {
        if (reservation == null) {
            return;
        }
        reservation.owner.rollback(reservation);
    }

    public static final class PreparationScope implements AutoCloseable {
        private final PrefillEndpoint.Reservation reservation;

        private PreparationScope(PrefillEndpoint.Reservation reservation) {
            this.reservation = reservation;
        }

        @Override
        public void close() {
            rollback(reservation);
        }
    }

    static org.flexlb.balance.scheduler.Delivery commitQueuedRoutes(
            PrefillEndpoint state, List<RequestRoute> items, long[] predictions) {
        state.resourceLock().lock();
        try {
            return state.commitLocked(
                    org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                            items, null, predictions, null, System.currentTimeMillis()));
        } finally {
            state.resourceLock().unlock();
        }
    }

    static List<org.flexlb.balance.scheduler.Delivery> commitRoutes(
            PrefillEndpoint endpoint, long predictedMs, List<? extends RequestRoute> exactItems) {
        List<RequestRoute> items = List.copyOf(exactItems);
        items.forEach(
                item ->
                        org.flexlb.balance.scheduler.SchedulerTestSupport.bindEndpointOwner(
                                endpoint, item));
        long[] predictions = new long[items.size()];
        java.util.Arrays.fill(predictions, predictedMs);
        var lock = endpoint.resourceLock();
        lock.lock();
        try {
            return List.of(
                    endpoint.commitLocked(
                            org.flexlb.balance.scheduler.DeliveryTestSupport.queued(
                                    items, null, predictions, null, System.currentTimeMillis())));
        } finally {
            lock.unlock();
        }
    }

    static final DeliveryMetricsReporter NOOP_TELEMETRY =
            org.mockito.Mockito.mock(DeliveryMetricsReporter.class);

    static class TestRequestRuntime {
        private final RequestLifecycle requests = org.mockito.Mockito.mock(RequestLifecycle.class);
        private final RequestLifecycle events = requests;
        private final List<PrefillRetirement> prefillRetirements = new CopyOnWriteArrayList<>();

        TestRequestRuntime() {
            org.flexlb.balance.scheduler.DeliveryTestSupport.stubRouteDelivery(
                    requests, this::onCompleted);
            org.mockito.Mockito.doAnswer(
                            invocation -> {
                                var evidence =
                                        invocation.getArgument(
                                                1,
                                                org.flexlb.balance.scheduler.ContextResult
                                                        .EngineEvidence.class);
                                if (evidence.kind()
                                        == org.flexlb.balance.scheduler.ContextResult.EngineKind
                                                .RETIRED) {
                                    prefillRetirements.add(
                                            new PrefillRetirement(
                                                    evidence.route().prefillEp(),
                                                    List.of(evidence.route())));
                                    onRetired(evidence.route());
                                }
                                return null;
                            })
                    .when(events)
                    .onEngineEvidence(org.mockito.Mockito.any(), org.mockito.Mockito.any());
        }

        RequestLifecycle requests() {
            return requests;
        }

        RequestLifecycle events() {
            return events;
        }

        void onCompleted(Delivery.Member claim, Delivery.Result result) {}

        void onRetired(RequestRoute item) {}

        List<PrefillRetirement> prefillRetirements() {
            return List.copyOf(prefillRetirements);
        }
    }

    record PrefillRetirement(PrefillEndpoint endpoint, List<RequestRoute> ownedItems) {
        PrefillRetirement {
            ownedItems = List.copyOf(ownedItems);
        }
    }
}
