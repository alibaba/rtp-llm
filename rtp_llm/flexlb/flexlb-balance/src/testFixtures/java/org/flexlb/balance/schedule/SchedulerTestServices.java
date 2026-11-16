package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.springframework.scheduling.annotation.Scheduled;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;
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
    public final EngineCancellation dispatcher;
    private final LongSupplier clock;
    final EngineGrpcClient engineClient;
    private final java.util.concurrent.ScheduledThreadPoolExecutor cleanupTimerExecutor =
            SchedulerConfiguration.newTimerExecutor(2, "flexlb-request-cleanup-");
    private final TaskTimer taskTimer =
            new SchedulerConfiguration().taskTimer(cleanupTimerExecutor);
    private final java.util.concurrent.ScheduledThreadPoolExecutor deadlineTimerExecutor =
            SchedulerConfiguration.newTimerExecutor(1, "request-scheduler-expiration-");
    private final TaskTimer deadlineTaskTimer =
            new SchedulerConfiguration().deadlineTaskTimer(deadlineTimerExecutor);
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
            EngineCancellation dispatcher,
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
            EngineCancellation dispatcher,
            ConfigService config,
            EngineGrpcClient engineClient,
            LongSupplier clock) {
        this.engineClient = Objects.requireNonNull(engineClient);
        this.requests = Objects.requireNonNull(requests, "requests");
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter, "deliveryReporter");
        this.requestReporter = Objects.requireNonNull(requestReporter, "requestReporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        org.springframework.test.util.ReflectionTestUtils.setField(
                dispatcher, "grpcClient", engineClient);
        this.clock = Objects.requireNonNull(clock, "clock");
        this.responseCompletions =
                new ResponseCompletionExecutor(
                        config.loadBalanceConfig()
                                .getInternalRuntime()
                                .getResponseCompletionThreads());
        this.timer = new ExpirationTimer(requests, deadlineTaskTimer);
    }

    /**
     * Publish deterministic fixture choices through the production endpoint registration boundary.
     */
    public static SelectionResult<RequestRoute, org.flexlb.worker.endpoint.CapacityKey>
            registerSelection(RequestContext context, List<RoleSelection> selections) {
        var data = context;
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
                if (role.role().supportsPrefill()) {
                    var endpoint = (PrefillEndpoint) role.endpoint();
                    var registration =
                            endpoint.tryRegisterSelection(
                                    data.getRequestId(),
                                    endpoint.selectionSnapshot(),
                                    role.computeTokens());
                    if (registration.status() == WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT
                            || registration.status()
                                    == WorkerEndpoint.RegistrationStatus.UNAVAILABLE)
                        return SelectionResult.retry();
                    if (registration.status() != WorkerEndpoint.RegistrationStatus.REGISTERED)
                        return SelectionResult.blocked(role.capacityKey());
                    registered.add(
                            org.flexlb.balance.schedule.SelectionTestSupport.prefill(
                                    endpoint,
                                    metadata,
                                    role.computeTokens(),
                                    registration.value()));
                } else if (role.role() == RoleType.DECODE) {
                    var endpoint = (DecodeEndpoint) role.endpoint();
                    var request = data.getRequest();
                    var acquired =
                            endpoint.tryRegisterSelection(
                                    data.getRequestId(),
                                    endpoint.selectionSnapshot(),
                                    request.getHardKvTokens(),
                                    request.getExpectedKvTokens(),
                                    data.getPriority(),
                                    capacity);
                    if (acquired.status() == WorkerEndpoint.RegistrationStatus.VERSION_CONFLICT
                            || acquired.status() == WorkerEndpoint.RegistrationStatus.UNAVAILABLE)
                        return SelectionResult.retry();
                    if (acquired.status() != WorkerEndpoint.RegistrationStatus.REGISTERED)
                        return SelectionResult.blocked(role.capacityKey());
                    registered.add(
                            org.flexlb.balance.schedule.SelectionTestSupport.decode(
                                    endpoint, metadata, acquired.value()));
                } else registered.add(role);
            }
            var route = new RequestRoute(context, registered);
            transferred = true;
            return SelectionResult.success(route);
        } finally {
            if (!transferred) {
                Throwable failure = null;
                for (var role : registered) {
                    if (role.registration() != null)
                        failure =
                                Failures.run(
                                        failure,
                                        () ->
                                                ((org.flexlb.worker.endpoint.SelectionRollback)
                                                                role.endpoint())
                                                        .rollbackSelection(role.registration()));
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

    public Executor tasks() {
        return tasks;
    }

    /**
     * Count accepted requests before delivery handoff, including direct requests.
     */
    public static int pendingDeliveryRequestCount(RequestRepository repository) {
        return repository
                .snapshotActive(
                        request -> {
                            var state = request.snapshot();
                            return state.phase() == RequestContext.Phase.OPEN
                                    && state.committedRoute() == null;
                        })
                .size();
    }

    /**
     * Replaces a test dependency in both setup and the already assembled request owners.
     */
    public static void setDependency(SchedulerTestServices fixture, String name, Object value) {
        org.springframework.test.util.ReflectionTestUtils.setField(fixture, name, value);
        String field =
                switch (name) {
                    case "timer" -> "expirationTimer";
                    case "tasks" -> "taskPool";
                    case "dispatcher" -> "engineCancellation";
                    default -> name;
                };
        if (!java.util.Set.of("tasks", "dispatcher", "responseCompletions", "timer").contains(name))
            return;
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
        for (String name : new String[] {"tasks", "dispatcher", "responseCompletions"}) {
            try {
                var source = SchedulerTestServices.class.getDeclaredField(name);
                source.setAccessible(true);
                var target =
                        AbstractRequestScheduler.class.getDeclaredField(
                                name.equals("tasks")
                                        ? "taskPool"
                                        : name.equals("dispatcher") ? "engineCancellation" : name);
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
            org.flexlb.balance.requestlog.TheoryCacheHitReporter trace) {
        var owner =
                new PassiveScheduler(
                        requests(),
                        tasks,
                        deadlines(),
                        dispatcher,
                        responseCompletions(),
                        deliveryReporter,
                        trace) {

                    @Override
                    protected org.flexlb.worker.endpoint.StrategyErrorType
                            schedulingTimeoutError() {
                        return config.isQueue()
                                ? org.flexlb.worker.endpoint.StrategyErrorType.QUEUE_TIMEOUT
                                : org.flexlb.worker.endpoint.StrategyErrorType
                                        .SCHEDULING_DEADLINE_EXCEEDED;
                    }
                };
        owners.put(owner, this);
        return owner;
    }

    /**
     * Startup wiring: register the placement owner once for ordered shutdown.
     */
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
     * Closes this test environment with a caller-selected response grace period.
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
                        shutdownScheduling(
                                scheduling,
                                requests,
                                tasks,
                                timer,
                                dispatcher,
                                responseCompletions,
                                endpoints,
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

    public static boolean shutdownScheduling(
            AbstractRequestScheduler scheduling,
            RequestRepository requests,
            RequestTaskPool tasks,
            ExpirationTimer timer,
            EngineCancellation dispatcher,
            ResponseCompletionExecutor responses,
            EndpointRegistry endpoints,
            long deadlineNanos) {
        scheduling.checkShutdownThread();
        requests.stopAccepting();
        boolean drained = false;
        Throwable failure = null;
        try {
            drained = requests.awaitResponses(deadlineNanos);
        } catch (Throwable cause) {
            failure = cause;
        }
        requests.closeRegistration();
        var schedulers = new java.util.LinkedHashSet<AbstractRequestScheduler>();
        schedulers.add(scheduling);
        for (var request : requests.snapshotActive()) schedulers.add(request.getRequestScheduler());
        for (var owner : schedulers) failure = Failures.run(failure, owner::stopAndCancelRequests);
        failure = Failures.run(failure, dispatcher::awaitStops);
        failure = Failures.run(failure, tasks::awaitIdle);
        for (var owner : schedulers) failure = Failures.run(failure, owner::awaitRequestCleanup);
        failure = Failures.run(failure, endpoints::close);
        failure = Failures.run(failure, dispatcher::awaitStops);
        failure = Failures.run(failure, timer::close);
        failure = Failures.run(failure, tasks::awaitIdle);
        for (var owner : schedulers) failure = Failures.run(failure, owner::finishRequestCleanup);
        for (var owner : schedulers) failure = Failures.run(failure, owner::awaitRequestCleanup);
        failure = Failures.run(failure, tasks::awaitIdle);
        failure = Failures.run(failure, dispatcher::awaitStops);
        failure = Failures.run(failure, tasks::close);
        failure = Failures.run(failure, responses::close);
        for (var owner : schedulers) failure = Failures.run(failure, owner::throwIfRequestFailed);
        if (requests.liveRequestCount() != 0)
            failure =
                    Failures.append(
                            failure,
                            new IllegalStateException(
                                    "unsettled requests at shutdown: "
                                            + requests.liveRequestCount()));
        Failures.rethrow(failure, "fixture shutdown failed");
        return drained;
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
            org.flexlb.balance.schedule.RequestPreemption preemption,
            CapacityNotifier availability,
            SchedulingQueue queue) {
        return config.isQueue()
                ? queued(config, router, preemption, dependencies, availability, queue)
                : direct(router, dependencies);
    }

    public static QueuedRequestScheduler queued(
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.schedule.RequestPreemption preemption,
            AbstractRequestScheduler dependencies,
            CapacityNotifier availability,
            SchedulingQueue queue) {
        return new QueuedRequestScheduler(
                config,
                router,
                preemption,
                dependencies.repository,
                dependencies.taskPool,
                dependency(dependencies, "expirationTimer"),
                dependency(dependencies, "engineCancellation"),
                dependency(dependencies, "responseCompletions"),
                forOwner(dependencies).deliveryReporter,
                dependency(dependencies, "cacheTrace"),
                availability,
                queue);
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

    public static DirectRequestScheduler direct(
            RequestRouteSelector router, AbstractRequestScheduler dependencies) {
        return new DirectRequestScheduler(
                router,
                dependencies.repository,
                dependencies.taskPool,
                dependency(dependencies, "expirationTimer"),
                dependency(dependencies, "engineCancellation"),
                dependency(dependencies, "responseCompletions"),
                forOwner(dependencies).deliveryReporter,
                dependency(dependencies, "cacheTrace"));
    }

    public static Object[] constructorArguments(
            RequestRouteSelector router, AbstractRequestScheduler d) {
        return new Object[] {
            router,
            d.repository,
            d.taskPool,
            dependency(d, "expirationTimer"),
            dependency(d, "engineCancellation"),
            dependency(d, "responseCompletions"),
            forOwner(d).deliveryReporter,
            dependency(d, "cacheTrace")
        };
    }

    public static Object[] constructorArguments(
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.schedule.RequestPreemption preemption,
            AbstractRequestScheduler d,
            CapacityNotifier availability,
            SchedulingQueue queue) {
        return new Object[] {
            config,
            router,
            preemption,
            d.repository,
            d.taskPool,
            dependency(d, "expirationTimer"),
            dependency(d, "engineCancellation"),
            dependency(d, "responseCompletions"),
            forOwner(d).deliveryReporter,
            dependency(d, "cacheTrace"),
            availability,
            queue
        };
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

    /**
     * 后台清理、准入或续接失败在此汇总；停服时统一报告，不执行 Future 监听或关闭流程。
     */
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
                () ->
                        endpoints.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                                org.flexlb.config.service.RoleType.PREFILL, null),
                deliveryReporter::reportPrefill,
                (address, endpoint) -> requestReporter.reportPrefillQueueDepth(address, 0));
        reportEndpoints(
                "Decode",
                () ->
                        endpoints.<org.flexlb.worker.endpoint.DecodeEndpoint>endpoints(
                                org.flexlb.config.service.RoleType.DECODE, null),
                deliveryReporter::reportDecode,
                (address, endpoint) ->
                        requestReporter.reportDecodePreemption(address, endpoint.stats()));
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
                                            org.flexlb.balance.requestlog.TheoryCacheHitReporter
                                                    .class))
                            : lifecycle(scheduler);
        }
        return owner;
    }

    void closeTimers() {
        Throwable failure = Failures.run(null, deadlineTimerExecutor::close);
        failure = Failures.run(failure, cleanupTimerExecutor::close);
        Failures.rethrow(failure, "fixture timers failed to close");
    }

    @SuppressWarnings("unchecked")
    private static <T> T dependency(AbstractRequestScheduler owner, String name) {
        return (T) org.springframework.test.util.ReflectionTestUtils.getField(owner, name);
    }

    private static <T extends org.flexlb.worker.endpoint.WorkerEndpoint> void reportEndpoints(
            String role,
            Supplier<java.util.List<T>> snapshot,
            Consumer<T> batchMetrics,
            BiConsumer<String, T> admissionMetrics) {
        java.util.List<T> endpoints;
        try {
            endpoints = snapshot.get();
        } catch (RuntimeException failure) {
            warnIsolated("Failed to snapshot " + role + " endpoints for metrics", failure);
            return;
        }
        for (T entry : endpoints) {
            try {
                batchMetrics.accept(entry);
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " endpoint metrics: endpoint="
                                + entry.ipPort(),
                        failure);
            }
            try {
                admissionMetrics.accept(entry.ipPort(), entry);
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " admission metrics: endpoint="
                                + entry.ipPort(),
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
                EngineCancellation dispatcher,
                ResponseCompletionExecutor responses,
                DeliveryMetricsReporter reporter,
                org.flexlb.balance.requestlog.TheoryCacheHitReporter trace) {
            super(requests, tasks, deadlines, dispatcher, responses, reporter, trace);
        }

        public void start() {}

        public void scheduleRequest(RequestContext request) {}

        public void removeRequest(RequestContext request) {}

        public void onEndpointChanged(WorkerEndpoint endpoint) {}

        public void stop() {}

        public boolean awaitStopped(long deadlineNanos) {
            return true;
        }

        public void close() {}

        @Override
        protected org.flexlb.worker.endpoint.StrategyErrorType schedulingTimeoutError() {
            return org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED;
        }
    }
}
