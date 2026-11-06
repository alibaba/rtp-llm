package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;

import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkState;

/** Shared test setup and teardown for scheduler protocol tests. */
public final class SchedulerTestServices {

    public static AbstractRequestScheduler entry(AbstractRequestScheduler scheduler) {
        return SchedulerTestServices.lifecycle(scheduler);
    }

    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;
    private static final Duration DEFAULT_SHUTDOWN_TIMEOUT = Duration.ofSeconds(30L);

    private final RequestRepository requests;
    private final EndpointRegistry endpoints;
    final DeliveryMetricsReporter deliveryReporter;
    final RequestSchedulerReporter requestReporter;
    public final DeliveryDispatcher dispatcher;
    private final LongSupplier clock;

    private final DeliveryExecution deliveries;
    final EngineGrpcClient engineClient;

    DeliveryExecution deliveries() {
        return deliveries;
    }

    private final ScheduledTaskTimer taskTimer = new ScheduledTaskTimer();
    private final ScheduledTaskTimer deadlineTaskTimer =
            new ScheduledTaskTimer(1, "request-scheduler-expiration-");

    public TaskTimer taskTimer() {
        return taskTimer;
    }

    final RequestTaskPool tasks = new RequestTaskPool();

    public static RequestScheduler scheduler(AbstractRequestScheduler owner) {
        return owner;
    }

    public RequestTaskExecutor tasks() {
        return tasks;
    }

    private final ResponseCompletionExecutor responseCompletions;
    private final ExpirationTimer timer;

    RequestRepository requests() {
        return requests;
    }

    ResponseCompletionExecutor responseCompletions() {
        return responseCompletions;
    }

    ExpirationTimer deadlines() {
        return timer;
    }

    private static final java.util.Map<AbstractRequestScheduler, SchedulerTestServices> owners =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    public static AbstractRequestScheduler lifecycle(RequestScheduler scheduler) {
        return (AbstractRequestScheduler) scheduler;
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
                if (owner.tasks != null && owner.tasks == entry.getKey().tasks) {
                    found = entry.getValue();
                    break;
                }
            }
            if (found != null) owners.put(owner, found);
            return found;
        }
    }

    /** Replaces a test dependency in both setup and the already assembled request owners. */
    public static void setDependency(SchedulerTestServices fixture, String name, Object value) {
        org.springframework.test.util.ReflectionTestUtils.setField(fixture, name, value);
        if (name.equals("deliveries")) {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    fixture.dispatcher, "execution", value);
            return;
        }
        if (name.equals("taskTimer")) {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    fixture.dispatcher, "timer", value);
            return;
        }
        if (name.equals("dispatcher")) {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    value, "execution", fixture.deliveries());
            org.springframework.test.util.ReflectionTestUtils.setField(
                    value, "timer", fixture.taskTimer());
        }
        String field = name.equals("timer") ? "expirationTimer" : name;
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

    public static Throwable failure(SchedulerTestServices fixture) {
        var workFailure =
                (Throwable)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                fixture.tasks, "failure");
        return workFailure != null ? workFailure : fixture.failure;
    }

    public static void bind(AbstractRequestScheduler owner, SchedulerTestServices runtime) {
        owners.put(owner, runtime);
        for (String name :
                new String[] {"tasks", "dispatcher", "responseCompletions", "deliveryReporter"}) {
            try {
                var source = SchedulerTestServices.class.getDeclaredField(name);
                source.setAccessible(true);
                var target = AbstractRequestScheduler.class.getDeclaredField(name);
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
            org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity,
            org.flexlb.service.TheoryCacheHitReporter trace) {
        org.springframework.test.util.ReflectionTestUtils.setField(
                dispatcher, "decodeAdmissionCapacity", capacity);
        var owner =
                new PassiveScheduler(
                        requests(),
                        tasks,
                        deadlines(),
                        dispatcher,
                        responseCompletions(),
                        deliveryReporter,
                        trace);
        owners.put(owner, this);
        return owner;
    }

    private final Object schedulerLock = new Object();
    private RequestScheduler scheduler;
    private Throwable failure;

    boolean isAccepting() {
        return !requests.isStopping() && !requests.isClosed();
    }

    /** 后台清理、准入或续接失败在此汇总；停服时统一报告，不执行 Future 监听或关闭流程。 */
    void recordFailure(Throwable cause) {
        synchronized (schedulerLock) {
            failure = Failures.append(failure, cause);
        }
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

    SchedulerTestServices(
            RequestRepository requests,
            EndpointRegistry endpoints,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            DeliveryDispatcher dispatcher,
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
            DeliveryDispatcher dispatcher,
            ConfigService config,
            EngineGrpcClient engineClient,
            LongSupplier clock) {
        this.engineClient = Objects.requireNonNull(engineClient);
        deliveries = new DeliveryExecution(config, this::recordFailure);
        this.requests = Objects.requireNonNull(requests, "requests");
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter, "deliveryReporter");
        deliveryReporter.registerExecutor(deliveries);
        this.requestReporter = Objects.requireNonNull(requestReporter, "requestReporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        org.springframework.test.util.ReflectionTestUtils.setField(
                dispatcher, "execution", deliveries);
        org.springframework.test.util.ReflectionTestUtils.setField(dispatcher, "timer", taskTimer);
        if (org.springframework.test.util.ReflectionTestUtils.getField(dispatcher, "grpcClient")
                == null) {
            org.springframework.test.util.ReflectionTestUtils.setField(
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

    private static final long TERMINAL_RETENTION_MS = 10_000L;

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

    private enum SettlementScope {
        BATCH,
        ALL
    }

    private void awaitDeliveryCleanup(SettlementScope scope) {
        var active = requests.snapshotActive();
        var pending =
                active.stream()
                        .map(context -> context.snapshot().delivery())
                        .filter(Objects::nonNull)
                        .map(ContextResult.DeliveryFacts::identity)
                        .filter(Objects::nonNull)
                        .filter(
                                delivery ->
                                        scope == SettlementScope.ALL
                                                || delivery.kind()
                                                        == Delivery.SendMode.BATCH_ENQUEUE)
                        .map(delivery -> delivery.completion().toCompletableFuture())
                        .toArray(CompletableFuture[]::new);
        try {
            CompletableFuture.allOf(pending)
                    .get(DeliveryDispatcher.CLEANUP_TIMEOUT_MS + 1_000L, TimeUnit.MILLISECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(
                    "Interrupted waiting for delivery cleanup", interrupted);
        } catch (ExecutionException | TimeoutException failure) {
            throw new IllegalStateException(
                    "Unsettled deliveries at shutdown: "
                            + active.stream()
                                    .map(context -> context.data().getRequestId())
                                    .toList(),
                    failure);
        }
    }

    private void closeRequestExecutors() {
        Throwable failure = null;
        tasks.awaitIdle();
        try {
            dispatcher.awaitStops();
        } catch (Throwable cause) {
            failure = cause;
        }
        try {
            tasks.close();
        } catch (Throwable cause) {
            failure = Failures.append(failure, cause);
        }
        try {
            responseCompletions.close();
        } catch (Throwable cause) {
            failure = Failures.append(failure, cause);
        }
        failure = Failures.run(failure, taskTimer::close);
        Failures.rethrow(failure, "request executors failed to close");
    }

    /**
     * Stops accepting requests and settles the scheduler within one bounded lifecycle operation.
     * DRAIN keeps requests already registered before the cutoff eligible for placement. If the
     * accepted responses do not complete before the deadline, the operation escalates to IMMEDIATE.
     */
    public ShutdownResult shutdown(ShutdownMode requestedMode, Duration timeout) {
        Objects.requireNonNull(requestedMode, "requestedMode");
        Objects.requireNonNull(timeout, "timeout");
        if (timeout.isNegative()) {
            throw new IllegalArgumentException("shutdown timeout must not be negative");
        }
        checkState(
                !tasks.isExecutingOnCurrentThread()
                        && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
        synchronized (this) {
            requests.stopAccepting();
            RequestScheduler scheduler;
            synchronized (schedulerLock) {
                scheduler = this.scheduler;
            }
            long deadlineNanos = shutdownDeadline(timeout);
            boolean drained = false;
            ShutdownMode completedMode = requestedMode;
            Throwable firstFailure = null;
            if (requestedMode == ShutdownMode.DRAIN) {
                try {
                    drained = requests.awaitResponses(deadlineNanos);
                } catch (Throwable failure) {
                    firstFailure = failure;
                }
                if (!drained) {
                    completedMode = ShutdownMode.IMMEDIATE;
                }
            }
            if (!requests.closeRegistration()) {
                return new ShutdownResult(requestedMode, completedMode, drained);
            }
            if (scheduler != null) {
                firstFailure = Failures.run(firstFailure, scheduler::stop);
                firstFailure = Failures.run(firstFailure, scheduler::close);
            }
            var owners = new java.util.LinkedHashSet<AbstractRequestScheduler>();
            if (scheduler != null) {
                owners.add(lifecycle(scheduler));
            }
            for (var context : requests.snapshotActive()) {
                owners.add(context.data().getRequestScheduler());
            }
            for (var owner : owners) {
                firstFailure = Failures.run(firstFailure, owner::stopAdmissionsAndAwaitCompletion);
            }
            for (var context : requests.snapshotActive()) {
                firstFailure =
                        Failures.run(
                                firstFailure,
                                () ->
                                        context.data()
                                                .getRequestScheduler()
                                                .cancel(context, 0L, CancelReason.SHUTDOWN));
            }
            firstFailure = Failures.run(firstFailure, deliveries::awaitDeliveries);
            // Preserve BATCH cleanup failures before retiring their ledgers.
            firstFailure =
                    Failures.run(firstFailure, () -> awaitDeliveryCleanup(SettlementScope.BATCH));
            // Retirement supplies exact local settlement facts for route decisions.
            firstFailure = Failures.run(firstFailure, endpoints::close);
            firstFailure =
                    Failures.run(firstFailure, () -> awaitDeliveryCleanup(SettlementScope.ALL));
            firstFailure = Failures.run(firstFailure, timer::close);
            firstFailure = Failures.run(firstFailure, deadlineTaskTimer::close);
            firstFailure = Failures.run(firstFailure, tasks::awaitIdle);
            for (var owner : owners) {
                firstFailure = Failures.run(firstFailure, owner::closeOutstandingAndTerminalize);
            }
            firstFailure = Failures.run(firstFailure, this::closeRequestExecutors);
            if (requests.liveRequestCount() != 0) {
                firstFailure =
                        Failures.append(
                                firstFailure,
                                new IllegalStateException(
                                        "Unsettled requests at shutdown: "
                                                + requests.liveRequestCount()));
            }
            firstFailure = Failures.run(firstFailure, tasks::throwIfFailed);
            synchronized (schedulerLock) {
                firstFailure = Failures.append(failure, firstFailure);
            }
            Failures.rethrow(firstFailure, "Scheduler shutdown failed");
            return new ShutdownResult(requestedMode, completedMode, drained);
        }
    }

    /** Spring invokes the bounded graceful lifecycle operation. */
    private void destroy() {
        shutdown(ShutdownMode.DRAIN, DEFAULT_SHUTDOWN_TIMEOUT);
    }

    private static long shutdownDeadline(Duration timeout) {
        long now = System.nanoTime();
        long nanos;
        try {
            nanos = timeout.toNanos();
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
        if (nanos == 0L) {
            return now;
        }
        try {
            return Math.addExact(now, nanos);
        } catch (ArithmeticException overflow) {
            return Long.MAX_VALUE;
        }
    }

    public static DirectRequestScheduler direct(
            RequestRouteSelector router, AbstractRequestScheduler dependencies) {
        return new DirectRequestScheduler(
                router,
                dependencies.requests,
                dependencies.tasks,
                dependencies.expirationTimer,
                dependencies.dispatcher,
                dependencies.responseCompletions,
                dependencies.deliveryReporter,
                dependencies.cacheTrace);
    }

    public static QueuedRequestScheduler queued(
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.eviction.DecodePreemption preemption,
            AbstractRequestScheduler dependencies,
            PlacementAvailability availability,
            PlacementQueue queue,
            org.flexlb.balance.planner.GroupPlanner grouping) {
        return new QueuedRequestScheduler(
                config,
                router,
                preemption,
                dependencies.requests,
                dependencies.tasks,
                dependencies.expirationTimer,
                dependencies.dispatcher,
                dependencies.responseCompletions,
                dependencies.deliveryReporter,
                dependencies.cacheTrace,
                availability,
                queue,
                grouping);
    }

    public static AbstractRequestScheduler create(
            AbstractRequestScheduler dependencies,
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.eviction.DecodePreemption preemption,
            PlacementAvailability availability,
            org.flexlb.balance.planner.GroupPlanner grouping,
            PlacementQueue queue) {
        return config.isQueue()
                ? queued(config, router, preemption, dependencies, availability, queue, grouping)
                : direct(router, dependencies);
    }

    public static Object[] constructorArguments(
            RequestRouteSelector router, AbstractRequestScheduler d) {
        return new Object[] {
            router,
            d.requests,
            d.tasks,
            d.expirationTimer,
            d.dispatcher,
            d.responseCompletions,
            d.deliveryReporter,
            d.cacheTrace
        };
    }

    public static Object[] constructorArguments(
            org.flexlb.config.FlexlbConfig config,
            RequestRouteSelector router,
            org.flexlb.balance.eviction.DecodePreemption preemption,
            AbstractRequestScheduler d,
            PlacementAvailability availability,
            PlacementQueue queue,
            org.flexlb.balance.planner.GroupPlanner grouping) {
        return new Object[] {
            config,
            router,
            preemption,
            d.requests,
            d.tasks,
            d.expirationTimer,
            d.dispatcher,
            d.responseCompletions,
            d.deliveryReporter,
            d.cacheTrace,
            availability,
            queue,
            grouping
        };
    }

    /**
     * Real request management without automatic placement, for protocol tests that drive events explicitly.
     */
    public static class PassiveScheduler extends AbstractRequestScheduler {
        public PassiveScheduler(
                RequestRepository requests,
                RequestTaskPool tasks,
                ExpirationTimer deadlines,
                DeliveryDispatcher dispatcher,
                ResponseCompletionExecutor responses,
                DeliveryMetricsReporter reporter,
                org.flexlb.service.TheoryCacheHitReporter trace) {
            super(requests, tasks, deadlines, dispatcher, responses, reporter, trace);
        }

        public void start() {}

        public void schedule() {}

        public void onRequest(RequestContext request) {}

        public void removeRequest(RequestContext request) {}

        public void onEndpointChanged(org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {}

        public void stop() {}

        public boolean awaitStopped(long deadlineNanos) {
            return true;
        }

        public void close() {}
    }
}
