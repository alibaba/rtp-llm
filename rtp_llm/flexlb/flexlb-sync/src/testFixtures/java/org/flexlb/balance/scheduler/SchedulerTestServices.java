package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.route.RoleType;
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

    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;
    private static final Duration DEFAULT_SHUTDOWN_TIMEOUT = Duration.ofSeconds(30L);

    private final RequestRepository requests;
    private final EndpointRegistry endpoints;
    final DeliveryMetricsReporter deliveryReporter;
    final RequestSchedulerReporter requestReporter;
    public final DeliveryDispatcher dispatcher;
    private final LongSupplier clock;

    private final DeliveryExecution deliveries;
    final PrefillQueueRunner.Dependencies queueDependencies;
    final EngineGrpcClient engineClient;

    private final RequestSignals signals = new RequestSignals();

    DeliveryExecution deliveries() {
        return deliveries;
    }

    public RequestSignals signals() {
        return signals;
    }

    private final ScheduledTaskTimer taskTimer = new ScheduledTaskTimer();

    public TaskTimer taskTimer() {
        return taskTimer;
    }

    final RequestTaskPool tasks = new RequestTaskPool(this::recordFailure);

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

    private static final java.util.Map<RequestLifecycle, SchedulerTestServices> owners =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    public static SchedulerTestServices forOwner(Object value) {
        var owner =
                value instanceof RequestLifecycle r ? r : ((RequestScheduler) value).lifecycle();
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
        String field = name.equals("timer") ? "expirationTimer" : name;
        if (!java.util.Set.of(
                        "tasks",
                        "signals",
                        "dispatcher",
                        "responseCompletions",
                        "deliveryReporter",
                        "taskTimer",
                        "timer")
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

    public static void bind(RequestLifecycle owner, SchedulerTestServices runtime) {
        owners.put(owner, runtime);
        for (String name :
                new String[] {
                    "tasks",
                    "signals",
                    "dispatcher",
                    "responseCompletions",
                    "deliveryReporter",
                    "taskTimer"
                }) {
            try {
                var source = SchedulerTestServices.class.getDeclaredField(name);
                source.setAccessible(true);
                var target = RequestLifecycle.class.getDeclaredField(name);
                target.setAccessible(true);
                target.set(owner, source.get(runtime));
            } catch (ReflectiveOperationException e) {
                throw new AssertionError(e);
            }
        }
        org.springframework.test.util.ReflectionTestUtils.setField(
                owner, "expirationTimer", runtime.deadlines());
    }

    public RequestLifecycle newLifecycle(
            org.flexlb.config.FlexlbConfig config,
            org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity,
            org.flexlb.service.RecentCacheKeyTraceReporter trace) {
        var owner =
                new RequestLifecycle(
                        requests(),
                        tasks,
                        deadlines(),
                        signals(),
                        dispatcher,
                        responseCompletions(),
                        deliveryReporter,
                        taskTimer(),
                        capacity,
                        trace);
        owners.put(owner, this);
        return owner;
    }

    public RequestLifecycle boundLifecycle(
            RequestScheduler scheduler,
            RequestLifecycle.Scheduling timing,
            org.flexlb.config.FlexlbConfig config,
            org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity,
            org.flexlb.service.RecentCacheKeyTraceReporter trace) {
        var lifecycle = newLifecycle(config, capacity, trace);
        lifecycle.bindScheduler(scheduler, timing);
        return lifecycle;
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
        this.queueDependencies =
                new PrefillQueueRunner.Dependencies(
                        deliveries, deliveryReporter, this::recordFailure);
        this.requestReporter = Objects.requireNonNull(requestReporter, "requestReporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
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
        this.timer = new ExpirationTimer(requests);
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
                () -> endpoints.endpointSnapshot(RoleType.PREFILL),
                endpoint -> ((PrefillEndpoint) endpoint).reportBatchMetrics(deliveryReporter),
                (address, endpoint) ->
                        requestReporter.reportPrefillQueueDepth(
                                address,
                                ((PrefillEndpoint) endpoint).ownership.progress().queueDepth()));
        reportEndpoints(
                "Decode",
                () -> endpoints.endpointSnapshot(RoleType.DECODE),
                endpoint -> ((DecodeEndpoint) endpoint).reportBatchMetrics(deliveryReporter),
                (address, endpoint) ->
                        ((DecodeEndpoint) endpoint).reportAdmissionMetrics(requestReporter));
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
                                                        == DeliveryClaimKind.BATCH_ENQUEUE)
                        .map(delivery -> signals.delivery(delivery).toCompletableFuture())
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
     * placement queue does not drain before the deadline, the operation escalates to IMMEDIATE.
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
                    drained =
                            scheduler == null || scheduler.awaitSchedulingCompletion(deadlineNanos);
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
                firstFailure = Failures.run(firstFailure, scheduler::close);
            }
            var owners = new java.util.LinkedHashSet<RequestLifecycle>();
            if (scheduler != null) {
                owners.add(scheduler.lifecycle());
            }
            for (var context : requests.snapshotActive()) {
                owners.add(context.data().getScheduler());
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
                                                .getScheduler()
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
}
