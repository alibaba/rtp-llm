package org.flexlb.balance.scheduler;

import io.micrometer.core.instrument.MeterRegistry;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkState;

/** Owns scheduler executors, delivery capacity, metrics, maintenance, and ordered shutdown. */
@Component
public final class SchedulerRuntime implements RequestTaskScheduler {

    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;
    private static final Duration DEFAULT_SHUTDOWN_TIMEOUT = Duration.ofSeconds(30L);

    private final RequestRepository requests;
    private final EndpointRegistry endpoints;
    private final DeliveryMetricsReporter deliveryReporter;
    private final RequestSchedulerReporter requestReporter;
    final DeliveryDispatcher dispatcher;
    private final ConfigService config;
    private final LongSupplier clock;

    private final DeliveryExecution deliveries;
    private final RequestSignals signals = new RequestSignals();

    DeliveryExecution deliveries() {
        return deliveries;
    }

    public RequestSignals signals() {
        return signals;
    }

    @Autowired(required = false)
    void configureDeliveryMetrics(MeterRegistry registry, Environment environment) {
        deliveries.configureDeliveryMetrics(registry);
        deliveries.configureBatchIds(environment);
    }

    final EngineGrpcClient engineClient;
    private final ScheduledThreadPoolExecutor cleanupExecutor =
            new ScheduledThreadPoolExecutor(
                    2, Thread.ofPlatform().daemon().name("flexlb-request-cleanup-", 1).factory());

    @Override
    public Cancellation scheduleTimeout(Runnable work, long delayMs) {
        var scheduled =
                cleanupExecutor.schedule(
                        Objects.requireNonNull(work, "work"), delayMs, TimeUnit.MILLISECONDS);
        return () -> scheduled.cancel(false);
    }

    private final ExecutorService requestWorkers =
            Executors.newFixedThreadPool(
                    Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8),
                    Thread.ofPlatform().daemon().name("request-work-", 1).factory());
    private final java.util.concurrent.Executor stopWorkers =
            work -> Thread.ofVirtual().name("delivery-stop-work").start(work);
    // Never run request code while holding this monitor. Context -> workLifecycle is the only edge.
    private final RequestWorkQueue requestWorkQueue = new RequestWorkQueue();
    private final ThreadLocal<RequestContext> executingRequest = new ThreadLocal<>();
    private final Object workLifecycle = new Object();
    private boolean acceptingWork = true;
    private long pendingWork, pendingStops;
    private final Object admissionLifecycle = new Object();
    private boolean admitting = true;
    private long admissions;

    public boolean registerAdmission() {
        synchronized (admissionLifecycle) {
            if (!admitting || requests.isClosed()) {
                return false;
            }
            admissions++;
            return true;
        }
    }

    public void finishAdmission() {
        synchronized (admissionLifecycle) {
            checkState(admissions > 0L, "admission ownership underflow");
            if (--admissions == 0L) {
                admissionLifecycle.notifyAll();
            }
        }
    }

    public void awaitAdmissions() {
        boolean interrupted = false;
        synchronized (admissionLifecycle) {
            admitting = false;
            while (admissions != 0L) {
                try {
                    admissionLifecycle.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private final ResponseCompletionExecutor responseCompletions;
    private final ExpirationTimer timer;

    RequestRepository requests() {
        return requests;
    }

    /** Offloads effects; Context and endpoint owners arbitrate concurrent work, not executor order. */
    @Override
    public void executeRequestWork(RequestContext context, Runnable work) {
        Objects.requireNonNull(work, "work");
        try (var registration = registerRequestWork(context)) {
            registration.submit(work);
        }
    }

    /** Coalesce the exact stop workflow; local close work has a separate Context key. */
    boolean executeStopWork(RequestContext context, Runnable work) {
        Objects.requireNonNull(work, "work");
        try (var registration = registerRequestWork(context)) {
            synchronized (workLifecycle) {
                pendingWork--;
                pendingStops++;
                registration.stop = true;
                if (pendingWork == 0) {
                    workLifecycle.notifyAll();
                }
            }
            return registration.submit(work, work, false);
        }
    }

    /** Register before publishing state that obliges a later effect or local cleanup. */
    WorkRegistration registerRequestWork(RequestContext context) {
        WorkRegistration registration = tryRegisterRequestWork(context);
        checkState(registration != null, "request work is closed");
        return registration;
    }

    WorkRegistration tryRegisterRequestWork(RequestContext context) {
        Objects.requireNonNull(context, "context");
        synchronized (workLifecycle) {
            if (!acceptingWork && executingRequest.get() != context) {
                return null;
            }
            pendingWork++;
        }
        return new WorkRegistration(context);
    }

    /** Counts both the submitting source and its optional task, including the handoff between them. */
    final class WorkRegistration implements AutoCloseable {
        private final RequestContext context;
        private boolean stop;
        private int owners = 1;
        private boolean submitted;
        private boolean sourceClosed;
        private boolean started;
        private boolean executed;

        private WorkRegistration(RequestContext context) {
            this.context = context;
        }

        void submit(Runnable work) {
            submit(work, null, false);
        }

        void submitClose(Runnable work, boolean inline) {
            submit(work, context, inline);
        }

        private boolean submit(Runnable work, Object owner, boolean inline) {
            Objects.requireNonNull(work, "work");
            checkState(!Thread.holdsLock(context), "request work submission under Context lock");
            synchronized (this) {
                checkState(!sourceClosed && !submitted, "request work already handed off");
                submitted = true;
                owners++;
            }
            try {
                if (owner == null) {
                    dispatch(() -> run(work, inline));
                    return true;
                }
                if (requestWorkQueue.execute(
                        owner,
                        () -> run(work, inline),
                        this::reject,
                        inline ? Runnable::run : this::dispatch)) {
                    return true;
                }
                release();
                return false;
            } catch (RuntimeException | Error failure) {
                reject(failure);
                throw failure;
            }
        }

        private void dispatch(Runnable task) {
            (stop ? stopWorkers : requestWorkers).execute(task);
        }

        private void reject(Throwable cause) {
            synchronized (this) {
                if (started) {
                    return;
                }
                started = true;
            }
            recordFailure(cause);
            release();
        }

        private void run(Runnable work, boolean propagate) {
            synchronized (this) {
                if (started) {
                    return;
                }
                started = true;
                executed = true;
            }
            var previous = executingRequest.get();
            executingRequest.set(context);
            try {
                work.run();
            } catch (Throwable failure) {
                recordFailure(failure);
                try {
                    Logger.error(
                            "Request work failed: request_id={}",
                            context.data().getRequestId(),
                            failure);
                } catch (Throwable ignored) {
                }
                if (propagate) {
                    throw Failures.propagate(failure, "request control failed");
                }
            } finally {
                executingRequest.set(previous);
                release();
            }
        }

        private void release() {
            synchronized (this) {
                if (--owners != 0) {
                    return;
                }
            }
            synchronized (workLifecycle) {
                if (stop) {
                    pendingStops--;
                } else {
                    pendingWork--;
                }
                if (pendingWork == 0) {
                    workLifecycle.notifyAll();
                }
            }
        }

        synchronized boolean hasTaskOwner() {
            return owners > (sourceClosed ? 0 : 1);
        }

        synchronized boolean canRetrySubmission() {
            return !executed && owners == (sourceClosed ? 0 : 1);
        }

        @Override
        public void close() {
            synchronized (this) {
                if (sourceClosed) {
                    return;
                }
                sourceClosed = true;
            }
            release();
        }
    }

    private void awaitRequestWork() {
        boolean interrupted = false;
        synchronized (workLifecycle) {
            // Local drains do not wait for remote cleanup; shutdown also drains stop owners.
            while (pendingWork != 0 || !acceptingWork && pendingStops != 0) {
                try {
                    workLifecycle.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void closeRequestWork() {
        synchronized (workLifecycle) {
            acceptingWork = false;
        }
        // A running owner can still hand off its exact request through executingRequest.
        awaitRequestWork();
        var executors = java.util.List.of(requestWorkers);
        executors.forEach(ExecutorService::shutdown);
        boolean interrupted = false;
        for (var executor : executors) {
            while (!executor.isTerminated()) {
                try {
                    executor.awaitTermination(1, TimeUnit.DAYS);
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    ResponseCompletionExecutor responseCompletions() {
        return responseCompletions;
    }

    ExpirationTimer deadlines() {
        return timer;
    }

    DeliveryMetricsReporter deliveryReporter() {
        return deliveryReporter;
    }

    RequestSchedulerReporter requestReporter() {
        return requestReporter;
    }

    private final Object schedulerLock = new Object();
    private AbstractRequestScheduler scheduler;
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
    void registerScheduler(AbstractRequestScheduler scheduler) {
        synchronized (schedulerLock) {
            checkState(isAccepting(), "scheduler is closed");
            checkState(this.scheduler == null, "scheduler already registered");
            this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
        }
    }

    @Autowired
    SchedulerRuntime(
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

    SchedulerRuntime(
            RequestRepository requests,
            EndpointRegistry endpoints,
            DeliveryMetricsReporter deliveryReporter,
            RequestSchedulerReporter requestReporter,
            DeliveryDispatcher dispatcher,
            ConfigService config,
            EngineGrpcClient engineClient,
            LongSupplier clock) {
        deliveries = new DeliveryExecution(config, this::recordFailure);
        this.engineClient = Objects.requireNonNull(engineClient, "engineClient");
        cleanupExecutor.setRemoveOnCancelPolicy(true);
        cleanupExecutor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        cleanupExecutor.setContinueExistingPeriodicTasksAfterShutdownPolicy(false);
        this.requests = Objects.requireNonNull(requests, "requests");
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.deliveryReporter = Objects.requireNonNull(deliveryReporter, "deliveryReporter");
        this.requestReporter = Objects.requireNonNull(requestReporter, "requestReporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        this.config = Objects.requireNonNull(config, "config");
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
        awaitRequestWork();
        try {
            dispatcher.awaitStops(this);
        } catch (Throwable cause) {
            failure = cause;
        }
        try {
            closeRequestWork();
        } catch (Throwable cause) {
            failure = Failures.append(failure, cause);
        }
        try {
            responseCompletions.close();
        } catch (Throwable cause) {
            failure = Failures.append(failure, cause);
        }
        cleanupExecutor.shutdown();
        boolean interrupted = false;
        while (!cleanupExecutor.isTerminated()) {
            try {
                cleanupExecutor.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        Failures.rethrow(failure, "request executors failed to close");
    }

    /**
     * Stops accepting requests and settles the scheduler within one bounded lifecycle operation.
     * DRAIN keeps requests already registered before the cutoff eligible for placement. If the
     * placement queue does not drain before the deadline, the operation escalates to IMMEDIATE.
     */
    public synchronized ShutdownResult shutdown(ShutdownMode requestedMode, Duration timeout) {
        Objects.requireNonNull(requestedMode, "requestedMode");
        Objects.requireNonNull(timeout, "timeout");
        if (timeout.isNegative()) {
            throw new IllegalArgumentException("shutdown timeout must not be negative");
        }
        checkState(
                executingRequest.get() == null
                        && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
        requests.stopAccepting();
        AbstractRequestScheduler scheduler;
        synchronized (schedulerLock) {
            scheduler = this.scheduler;
        }
        long deadlineNanos = shutdownDeadline(timeout);
        boolean drained = false;
        ShutdownMode completedMode = requestedMode;
        Throwable firstFailure = null;
        if (requestedMode == ShutdownMode.DRAIN) {
            try {
                drained = scheduler == null || scheduler.awaitPlacementDrain(deadlineNanos);
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
            firstFailure = Failures.run(firstFailure, scheduler::stopAdmission);
            firstFailure = Failures.run(firstFailure, scheduler::closePlacement);
            firstFailure = Failures.run(firstFailure, this::awaitAdmissions);
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
        firstFailure = Failures.run(firstFailure, () -> awaitDeliveryCleanup(SettlementScope.ALL));
        firstFailure = Failures.run(firstFailure, timer::close);
        firstFailure = Failures.run(firstFailure, this::awaitRequestWork);
        var owners = new java.util.LinkedHashSet<AbstractRequestScheduler>();
        if (scheduler != null) {
            owners.add(scheduler);
        }
        for (var context : requests.snapshotActive()) {
            owners.add(context.data().getScheduler());
        }
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
        synchronized (schedulerLock) {
            firstFailure = Failures.append(failure, firstFailure);
        }
        Failures.rethrow(firstFailure, "Scheduler shutdown failed");
        return new ShutdownResult(requestedMode, completedMode, drained);
    }

    /** Compatibility entry point for existing package clients; immediate shutdown is explicit. */
    public ShutdownResult shutdown() {
        return shutdown(ShutdownMode.IMMEDIATE, Duration.ZERO);
    }

    /** Spring invokes the bounded graceful lifecycle operation. */
    @PreDestroy
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
