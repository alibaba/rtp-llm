package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkState;

/**
 * Shared test setup and teardown for scheduler protocol tests.
 */
public final class SchedulerTestServices {
    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;

    private static final java.util.Map<AbstractRequestScheduler, SchedulerTestServices> owners =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    private static final long TERMINAL_RETENTION_MS = 10_000L;

    private final RequestRepository requests;
    private final EndpointRegistry endpoints;
    final DeliveryMetricsReporter deliveryReporter;
    final RequestSchedulerReporter requestReporter;
    public final DeliveryCoordinator dispatcher;
    private final LongSupplier clock;
    private final DeliveryExecution deliveries;
    final EngineGrpcClient engineClient;
    private final ScheduledTaskTimer taskTimer = new ScheduledTaskTimer();
    private final ScheduledTaskTimer deadlineTaskTimer =
            new ScheduledTaskTimer(1, "request-scheduler-expiration-");
    final RequestTaskPool tasks = new RequestTaskPool();
    private final ResponseCompletionExecutor responseCompletions;
    private final ExpirationTimer timer;
    private final Object schedulerLock = new Object();

    private RequestScheduler scheduler;
    private Throwable failure;

    SchedulerTestServices(
            RequestRepository requests,
            EndpointRegistry endpoints,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            DeliveryCoordinator dispatcher,
            ConfigService config,
            EngineGrpcClient engineClient) {
        this(
                requests,
                endpoints,
                deliveryReporter,
                requestReporter,
                dispatcher,
                config,
                engineClient,
                System::currentTimeMillis);
    }

    SchedulerTestServices(
            RequestRepository requests,
            EndpointRegistry endpoints,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            DeliveryCoordinator dispatcher,
            ConfigService config,
            EngineGrpcClient engineClient,
            LongSupplier clock) {
        this.engineClient = Objects.requireNonNull(engineClient);
        this.deliveries = new DeliveryExecution(config, this::recordFailure);
        this.requests = Objects.requireNonNull(requests, "requests");
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter, "deliveryReporter");
        deliveryReporter.registerExecutor(this.deliveries);
        this.requestReporter = Objects.requireNonNull(requestReporter, "requestReporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        org.flexlb.balance.scheduler.DeliveryTestSupport.setField(dispatcher, "requests", requests);
        org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                dispatcher, "mode", config.loadBalanceConfig().getDispatcher().getType());
        org.springframework.test.util.ReflectionTestUtils.setField(
                dispatcher, "execution", this.deliveries);
        if (org.flexlb.balance.scheduler.DeliveryTestSupport.getField(dispatcher, "grpcClient")
                == null) {
            org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                    dispatcher, "grpcClient", engineClient);
        }
        this.clock = Objects.requireNonNull(clock, "clock");
        this.responseCompletions =
                new ResponseCompletionExecutor(
                        config.loadBalanceConfig()
                                .getInternalRuntime()
                                .getBatchDispatchCompletionThreads());
        this.timer = new ExpirationTimer(requests, deadlineTaskTimer);
    }

    /**
     * Publish deterministic fixture choices through the production endpoint registration boundary.
     */
    public static RoutingResult<RouteResult, CapacityKey> registerSelection(
            RequestContext context, List<RoleSelection> selections) {
        context.data().initializeExecutionOrder(System.currentTimeMillis());
        var data = context.data();
        var config = data.getConfig();
        var limits = config.getRouter().getRoles().getDecode().getAvailability();
        var capacity =
                new DecodeResources.AdmissionCapacity(
                        limits.getMaxEngineRequests() == null ? 0 : limits.getMaxEngineRequests(),
                        limits.getMaxKvUsagePercent());
        var registered = new ArrayList<RoleSelection>();
        boolean transferred = false;
        try {
            for (var role : selections) {
                var metadata = role.toServerStatus(data.getRequestId());
                if (role instanceof RoleSelection.Prefill prefill) {
                    var endpoint = (PrefillEndpoint) prefill.endpoint();
                    var registration =
                            endpoint.tryRegisterSelection(
                                    new GroupPlanner.Item(
                                            data.getRequestId(),
                                            data.getPriority(),
                                            data.getWorkerEnqueueSequence(),
                                            data.getFirstWorkerEnqueueTime(),
                                            data.getRequestExpiresAtMs(),
                                            data.getRequest().getHardKvTokens(),
                                            prefill.hitCacheTokens()),
                                    endpoint.captureRouteProjectionInputs(),
                                    prefill.workMs(),
                                    config.isQueue());
                    if (registration.status() == RoutingResult.Registration.CONFLICT
                            || registration.status() == RoutingResult.Registration.UNAVAILABLE)
                        return RoutingResult.retry();
                    if (registration.status() != RoutingResult.Registration.REGISTERED)
                        return RoutingResult.blocked(prefill.capacityKey());
                    registered.add(
                            new RoleSelection.Prefill(
                                    endpoint, metadata, prefill.workMs(), registration.record()));
                } else if (role instanceof RoleSelection.Decode decode) {
                    var endpoint = (DecodeEndpoint) decode.endpoint();
                    var request = data.getRequest();
                    var acquired =
                            endpoint.tryRegisterSelection(
                                    endpoint.routingView(),
                                    data.getRequestId(),
                                    request.getHardKvTokens(),
                                    request.getExpectedKvTokens(),
                                    data.getPriority(),
                                    capacity);
                    if (acquired.status() == DecodeResources.ReserveStatus.VERSION_CONFLICT
                            || acquired.status() == DecodeResources.ReserveStatus.ENDPOINT_RETIRED)
                        return RoutingResult.retry();
                    if (acquired.status() != DecodeResources.ReserveStatus.ACQUIRED)
                        return RoutingResult.blocked(decode.capacityKey());
                    registered.add(new RoleSelection.Decode(endpoint, metadata, acquired.value()));
                } else registered.add(role);
            }
            var route = new RouteResult(context, registered);
            transferred = true;
            return RoutingResult.success(route);
        } finally {
            if (!transferred) {
                Throwable failure = null;
                for (var role : registered) {
                    if (role instanceof RoleSelection.Prefill prefill)
                        failure =
                                Failures.run(
                                        failure,
                                        () ->
                                                ((PrefillEndpoint) prefill.endpoint())
                                                        .rollback(prefill.registration));
                    else if (role instanceof RoleSelection.Decode decode)
                        failure =
                                Failures.run(
                                        failure,
                                        () ->
                                                RoleSelection.Decode.release(
                                                        (DecodeEndpoint) decode.endpoint(),
                                                        decode.registration,
                                                        DecodeResources.ReleaseReason
                                                                .LOCAL_ROLLBACK));
                }
                Failures.rethrow(failure, "fixture registration rollback");
            }
        }
    }

    public static AbstractRequestScheduler entry(AbstractRequestScheduler scheduler) {
        return SchedulerTestServices.lifecycle(scheduler);
    }

    public TaskTimer taskTimer() {
        return taskTimer;
    }

    public static RequestScheduler scheduler(AbstractRequestScheduler owner) {
        return owner;
    }

    public RequestTaskExecutor tasks() {
        return tasks;
    }

    public static SchedulerTestServices forOwner(Object value) {
        var owner =
                value instanceof AbstractRequestScheduler r
                        ? r
                        : lifecycle((RequestScheduler) value);
        synchronized (owners) {
            var found = owners.get(owner);
            if (found != null) return found;
            for (var entry : owners.entrySet()) {
                if (owner.taskPool != null && owner.taskPool == entry.getKey().taskPool) {
                    found = entry.getValue();
                    break;
                }
            }
            if (found != null) owners.put(owner, found);
            return found;
        }
    }

    public static AbstractRequestScheduler lifecycle(RequestScheduler scheduler) {
        return (AbstractRequestScheduler) scheduler;
    }

    /**
     * Count accepted requests before delivery handoff, including direct requests.
     */
    public static int pendingDeliveryRequestCount(RequestRepository repository) {
        int pending = 0;
        for (RequestContext context : repository.snapshotActive()) {
            var state = context.snapshot();
            if (state.phase() == ContextResult.Phase.OPEN && state.delivery() == null) {
                pending++;
            }
        }
        return pending;
    }

    /** Replaces a test dependency in both setup and the already assembled request owners. */
    public static void setDependency(SchedulerTestServices fixture, String name, Object value) {
        org.springframework.test.util.ReflectionTestUtils.setField(fixture, name, value);
        if (name.equals("deliveries")) {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    fixture.dispatcher, "execution", value);
            return;
        }
        if (name.equals("dispatcher")) {
            org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                    value, "requests", fixture.requests());
            org.springframework.test.util.ReflectionTestUtils.setField(
                    value, "execution", fixture.deliveries());
        }
        String field =
                switch (name) {
                    case "timer" -> "expirationTimer";
                    case "tasks" -> "taskPool";
                    default -> name;
                };
        if (!java.util.Set.of(
                        "tasks", "dispatcher", "responseCompletions", "deliveryReporter", "timer")
                .contains(name)) return;
        synchronized (owners) {
            for (var entry : owners.entrySet()) {
                if (entry.getValue() == fixture)
                    org.springframework.test.util.ReflectionTestUtils.setField(
                            entry.getKey(), field, value);
            }
        }
    }

    public static void throwIfFailed(RequestTaskPool pool) {
        synchronized (owners) {
            for (var owner : owners.keySet()) {
                if (org.springframework.test.util.ReflectionTestUtils.getField(owner, "taskPool")
                        == pool) owner.throwIfRequestFailed();
            }
        }
    }

    public static void bind(AbstractRequestScheduler owner, SchedulerTestServices runtime) {
        if (org.mockito.Mockito.mockingDetails(owner).isMock()
                && !org.mockito.Mockito.mockingDetails(owner).isSpy()) {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    owner, "closeQueue", new RequestCloseQueue());
            org.mockito.Mockito.doCallRealMethod()
                    .when(owner)
                    .recordRequestFailure(org.mockito.ArgumentMatchers.any());
            org.mockito.Mockito.doCallRealMethod().when(owner).throwIfRequestFailed();
        }
        owners.put(owner, runtime);
        for (String name :
                new String[] {"tasks", "dispatcher", "responseCompletions", "deliveryReporter"}) {
            try {
                var source = SchedulerTestServices.class.getDeclaredField(name);
                source.setAccessible(true);
                var target =
                        AbstractRequestScheduler.class.getDeclaredField(
                                name.equals("tasks") ? "taskPool" : name);
                target.setAccessible(true);
                target.set(owner, source.get(runtime));
            } catch (ReflectiveOperationException e) {
                throw new AssertionError(e);
            }
        }
        org.springframework.test.util.ReflectionTestUtils.setField(
                owner, "expirationTimer", runtime.deadlines());
    }

    public AbstractRequestScheduler newLifecycle(
            org.flexlb.config.FlexlbConfig config,
            DecodeResources.AdmissionCapacity capacity,
            org.flexlb.service.TheoryCacheHitReporter trace) {
        org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                dispatcher, "decodeAdmissionCapacity", capacity);
        var owner =
                new PassiveScheduler(
                        requests(),
                        tasks,
                        deadlines(),
                        dispatcher,
                        responseCompletions(),
                        deliveryReporter,
                        trace,
                        endpoints) {
                    @Override
                    public boolean allowsRoutePreemption() {
                        return config.isQueue();
                    }

                    @Override
                    protected org.flexlb.dao.loadbalance.StrategyErrorType
                            schedulingTimeoutError() {
                        return config.isQueue()
                                ? org.flexlb.dao.loadbalance.StrategyErrorType.QUEUE_TIMEOUT
                                : org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED;
                    }
                };
        owners.put(owner, this);
        return owner;
    }

    /** Startup wiring: register the placement owner once for ordered shutdown. */
    public void registerScheduler(RequestScheduler scheduler) {
        synchronized (schedulerLock) {
            checkState(isAccepting(), "scheduler is closed");
            checkState(this.scheduler == null, "scheduler already registered");
            this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
            if (scheduler instanceof AbstractRequestScheduler requestScheduler) {
                owners.put(requestScheduler, this);
            }
        }
    }

    /**
     * Uses the production shutdown sequence, with a caller-selected response grace period.
     */
    public ShutdownResult shutdown(ShutdownMode requestedMode, Duration timeout) {
        Objects.requireNonNull(requestedMode, "requestedMode");
        Objects.requireNonNull(timeout, "timeout");
        if (timeout.isNegative())
            throw new IllegalArgumentException("shutdown timeout must not be negative");
        // Reject callbacks before taking the fixture monitor, which shutdown may already hold.
        checkState(
                !tasks.isExecutingOnCurrentThread()
                        && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
        synchronized (this) {
            if (requests.isClosed()) {
                boolean drained =
                        requestedMode == ShutdownMode.DRAIN && requests.liveRequestCount() == 0;
                return new ShutdownResult(
                        requestedMode,
                        drained ? ShutdownMode.DRAIN : ShutdownMode.IMMEDIATE,
                        drained);
            }
            var scheduling = scheduling();
            boolean drained = false;
            Throwable shutdownFailure = null;
            try {
                drained =
                        scheduling.shutdown(
                                requestedMode == ShutdownMode.DRAIN
                                        ? com.google.common.math.LongMath.saturatedAdd(
                                                System.nanoTime(), timeout.toNanos())
                                        : System.nanoTime());
            } catch (Throwable cause) {
                shutdownFailure = cause;
            }
            shutdownFailure = Failures.run(shutdownFailure, this::closeTimers);
            synchronized (schedulerLock) {
                shutdownFailure = Failures.append(failure(this), shutdownFailure);
            }
            Failures.rethrow(shutdownFailure, "Scheduler shutdown failed");
            return new ShutdownResult(
                    requestedMode,
                    requestedMode == ShutdownMode.DRAIN && drained
                            ? ShutdownMode.DRAIN
                            : ShutdownMode.IMMEDIATE,
                    requestedMode == ShutdownMode.DRAIN && drained);
        }
    }

    public static Throwable failure(SchedulerTestServices fixture) {
        synchronized (owners) {
            for (var entry : owners.entrySet()) {
                if (entry.getValue() != fixture) continue;
                var failure =
                        (Throwable)
                                org.springframework.test.util.ReflectionTestUtils.getField(
                                        entry.getKey(), "requestProcessingFailure");
                if (failure != null) return failure;
            }
        }
        return fixture.failure;
    }

    public static AbstractRequestScheduler create(
            AbstractRequestScheduler dependencies,
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.preemption.DecodePreemption preemption,
            CapacityNotifier availability,
            GroupPlanner grouping,
            SchedulingQueue queue) {
        return config.isQueue()
                ? queued(config, router, preemption, dependencies, availability, queue, grouping)
                : direct(router, dependencies);
    }

    public static QueuedRequestScheduler queued(
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.preemption.DecodePreemption preemption,
            AbstractRequestScheduler dependencies,
            CapacityNotifier availability,
            SchedulingQueue queue,
            GroupPlanner grouping) {
        return new QueuedRequestScheduler(
                config,
                router,
                preemption,
                dependencies.repository,
                dependencies.taskPool,
                dependencies.expirationTimer,
                dependencies.dispatcher,
                dependencies.responseCompletions,
                dependencies.deliveryReporter,
                dependencies.cacheTrace,
                availability,
                queue,
                grouping,
                dependencies.endpoints);
    }

    public static DirectRequestScheduler direct(
            RequestRouteSelector router, AbstractRequestScheduler dependencies) {
        return new DirectRequestScheduler(
                router,
                dependencies.repository,
                dependencies.taskPool,
                dependencies.expirationTimer,
                dependencies.dispatcher,
                dependencies.responseCompletions,
                dependencies.deliveryReporter,
                dependencies.cacheTrace,
                dependencies.endpoints);
    }

    public static Object[] constructorArguments(
            RequestRouteSelector router, AbstractRequestScheduler d) {
        return new Object[] {
            router,
            d.repository,
            d.taskPool,
            d.expirationTimer,
            d.dispatcher,
            d.responseCompletions,
            d.deliveryReporter,
            d.cacheTrace,
            d.endpoints
        };
    }

    public static Object[] constructorArguments(
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.preemption.DecodePreemption preemption,
            AbstractRequestScheduler d,
            CapacityNotifier availability,
            SchedulingQueue queue,
            GroupPlanner grouping) {
        return new Object[] {
            config,
            router,
            preemption,
            d.repository,
            d.taskPool,
            d.expirationTimer,
            d.dispatcher,
            d.responseCompletions,
            d.deliveryReporter,
            d.cacheTrace,
            availability,
            queue,
            grouping,
            d.endpoints
        };
    }

    DeliveryExecution deliveries() {
        return deliveries;
    }

    RequestRepository requests() {
        return requests;
    }

    ResponseCompletionExecutor responseCompletions() {
        return responseCompletions;
    }

    ExpirationTimer deadlines() {
        return timer;
    }

    boolean isAccepting() {
        return !requests.isStopping() && !requests.isClosed();
    }

    /** 后台清理、准入或续接失败在此汇总；停服时统一报告，不执行 Future 监听或关闭流程。 */
    void recordFailure(Throwable cause) {
        synchronized (schedulerLock) {
            failure = Failures.append(failure, cause);
        }
    }

    @Scheduled(fixedRate = EXPIRATION_MAINTENANCE_INTERVAL_MS)
    void maintainExpiration() {
        if (requests.isClosed()) {
            return;
        }
        long ttlMs = TERMINAL_RETENTION_MS;
        long nowMs = clock.getAsLong();
        Throwable failure =
                Failures.run(
                        null,
                        () ->
                                requests.expireTerminalRecords(
                                        com.google.common.math.LongMath.saturatedSubtract(
                                                nowMs, ttlMs)));
        failure =
                Failures.run(
                        failure,
                        () -> endpoints.evictExpiredOrphans(ttlMs, requests::retainsIdentity));
        Failures.rethrow(failure, "expiration maintenance failed");
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void report() {
        if (requests.isClosed()) {
            return;
        }
        try {
            deliveryReporter.reportSchedulerInflight(
                    requests.liveRequestCount(), requests.oldestLiveRequestAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated("Failed to report scheduler inflight metrics", failure);
        }
        reportEndpoints(
                "Prefill",
                endpoints::prefillStats,
                deliveryReporter::reportPrefill,
                (address, endpoint) ->
                        requestReporter.reportPrefillQueueDepth(address, endpoint.queueDepth()));
        reportEndpoints(
                "Decode",
                endpoints::decodeStats,
                deliveryReporter::reportDecode,
                (address, endpoint) ->
                        requestReporter.reportDecodePreemption(
                                address, endpoint.resourceSnapshot()));
    }

    AbstractRequestScheduler scheduling() {
        AbstractRequestScheduler owner;
        synchronized (schedulerLock) {
            owner =
                    scheduler == null
                            ? new PassiveScheduler(
                                    requests,
                                    tasks,
                                    timer,
                                    dispatcher,
                                    responseCompletions,
                                    deliveryReporter,
                                    org.mockito.Mockito.mock(
                                            org.flexlb.service.TheoryCacheHitReporter.class),
                                    endpoints)
                            : lifecycle(scheduler);
        }
        return owner;
    }

    void closeTimers() {
        Throwable failure = Failures.run(null, deadlineTaskTimer::close);
        failure = Failures.run(failure, taskTimer::close);
        Failures.rethrow(failure, "fixture timers failed to close");
    }

    private static <T> void reportEndpoints(
            String role,
            Supplier<Map<String, T>> snapshot,
            Consumer<T> batchMetrics,
            BiConsumer<String, T> admissionMetrics) {
        Map<String, T> endpoints;
        try {
            endpoints = snapshot.get();
        } catch (RuntimeException failure) {
            warnIsolated("Failed to snapshot " + role + " endpoints for metrics", failure);
            return;
        }
        for (Map.Entry<String, T> entry : endpoints.entrySet()) {
            try {
                batchMetrics.accept(entry.getValue());
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " endpoint metrics: endpoint="
                                + entry.getKey(),
                        failure);
            }
            try {
                admissionMetrics.accept(entry.getKey(), entry.getValue());
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " admission metrics: endpoint="
                                + entry.getKey(),
                        failure);
            }
        }
    }

    private static void warnIsolated(String message, RuntimeException failure) {
        try {
            Logger.warn(message, failure);
        } catch (Throwable ignored) {
            // Telemetry must never couple otherwise independent metric leaves.
        }
    }

    /**
     * Real request management without automatic placement, for protocol tests that drive events explicitly.
     */
    public static class PassiveScheduler extends AbstractRequestScheduler {
        public PassiveScheduler(
                RequestRepository requests,
                RequestTaskPool tasks,
                ExpirationTimer deadlines,
                DeliveryCoordinator dispatcher,
                ResponseCompletionExecutor responses,
                DeliveryMetricsReporter reporter,
                org.flexlb.service.TheoryCacheHitReporter trace,
                EndpointRegistry endpoints) {
            super(requests, tasks, deadlines, dispatcher, responses, reporter, trace, endpoints);
        }

        public void start() {}

        public void scheduleRequest(RequestContext request) {}

        public void removeRequest(RequestContext request) {}

        public void onEndpointChanged(org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {}

        public void stop() {}

        public boolean awaitStopped(long deadlineNanos) {
            return true;
        }

        public void close() {}

        @Override
        protected org.flexlb.dao.loadbalance.StrategyErrorType schedulingTimeoutError() {
            return org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED;
        }
    }
}
