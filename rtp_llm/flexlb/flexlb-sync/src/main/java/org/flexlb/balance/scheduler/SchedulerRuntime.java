package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.balance.delivery.BatchIdGenerator;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.metric.SafeFlexMonitor;
import org.flexlb.constant.MetricConstant;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.core.env.Environment;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;
import javax.annotation.PreDestroy;

import static com.google.common.base.Preconditions.checkState;

/** Owns scheduler executors, delivery capacity, metrics, maintenance, and ordered shutdown. */
@Component
public final class SchedulerRuntime implements RequestTaskScheduler {

    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;

    private final RequestRepository requests;
    private final EndpointRegistry endpoints;
    private final DeliveryMetricsReporter deliveryReporter;
    private final RequestSchedulerReporter requestReporter;
    private final DeliveryDispatcher dispatcher;
    private final ConfigService config;
    private final LongSupplier clock;

    private final DeliveryExecution deliveries;
    private final RequestSignals signals = new RequestSignals();
    DeliveryExecution deliveries() { return deliveries; }
    public RequestSignals signals() { return signals; }

    @Autowired(required = false)
    void configureDeliveryMetrics(MeterRegistry registry, Environment environment) {
        deliveries.configureDeliveryMetrics(registry);
        deliveries.configureBatchIds(environment);
    }

    private final EngineCancelChannel cancelChannel;
    final ExecutorService deliveryCleanupExecutor = Executors.newThreadPerTaskExecutor(
            Thread.ofVirtual().name("flexlb-delivery-cleanup-", 1).factory());
    private final ScheduledThreadPoolExecutor cleanupExecutor =
            new ScheduledThreadPoolExecutor(2,
                    Thread.ofPlatform().daemon().name("flexlb-request-cleanup-", 1).factory());

    @Override
    public Cancellation scheduleTimeout(Runnable work, long delayMs) {
        var scheduled = cleanupExecutor.schedule(Objects.requireNonNull(work, "work"),
                delayMs, TimeUnit.MILLISECONDS);
        return () -> scheduled.cancel(false);
    }

    void startDeliveryCleanup(DeliveryClaim claim) {
        dispatcher.cleanupDelivery(claim, cancelChannel, this);
    }

    private final ExecutorService requestWorkers = Executors.newFixedThreadPool(
            Math.clamp(Runtime.getRuntime().availableProcessors(), 2, 8),
            Thread.ofPlatform().daemon().name("request-work-", 1).factory());
    private final ExecutorService requestRecovery = Executors.newSingleThreadExecutor(
            Thread.ofPlatform().daemon().name("request-work-recovery-", 1).factory());
    // Never run request code while holding this monitor. Context -> workLifecycle is the only edge.
    private final RequestWorkQueue requestWorkQueue = new RequestWorkQueue();
    private final ThreadLocal<RequestContext> executingRequest = new ThreadLocal<>();
    private final Object workLifecycle = new Object();
    private boolean acceptingWork = true;
    private long pendingWork;
    private final Object admissionLifecycle = new Object();
    private boolean admitting = true;
    private long admissions;

    public boolean registerAdmission() {
        synchronized (admissionLifecycle) {
            if (!admitting || requests.isClosed()) { return false; }
            admissions++;
            return true;
        }
    }
    public void finishAdmission() {
        synchronized (admissionLifecycle) {
            checkState(admissions > 0L, "admission ownership underflow");
            if (--admissions == 0L) { admissionLifecycle.notifyAll(); }
        }
    }
    public void awaitAdmissions() {
        boolean interrupted = false;
        synchronized (admissionLifecycle) {
            admitting = false;
            while (admissions != 0L) {
                try { admissionLifecycle.wait(); }
                catch (InterruptedException ignored) { interrupted = true; }
            }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
    }

    private final ResponseCompletionExecutor responseCompletions;
    private final ExpirationTimer timer;

    RequestRepository requests() { return requests; }

    /** Offloads effects; Context and endpoint owners arbitrate concurrent work, not executor order. */
    @Override
    public void executeRequestWork(RequestContext context, Runnable work) {
        Objects.requireNonNull(work, "work");
        try (var registration = registerRequestWork(context)) { registration.submit(work); }
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
            if (!acceptingWork && executingRequest.get() != context) { return null; }
            pendingWork++;
        }
        return new WorkRegistration(context);
    }

    /** Counts both the submitting source and its optional task, including the handoff between them. */
    final class WorkRegistration implements AutoCloseable {
        private final RequestContext context;
        private int owners = 1;
        private boolean submitted;
        private boolean sourceClosed;
        private boolean started;

        private WorkRegistration(RequestContext context) { this.context = context; }

        void submit(Runnable work) {
            Objects.requireNonNull(work, "work");
            checkState(!Thread.holdsLock(context), "request work submission under Context lock");
            synchronized (this) {
                checkState(!sourceClosed && !submitted, "request work already handed off");
                submitted = true;
                owners++;
            }
            try { dispatch(() -> run(work, false)); }
            catch (RuntimeException | Error failure) { reject(failure); throw failure; }
        }

        private void dispatch(Runnable task) {
            try { requestWorkers.execute(task); }
            catch (RuntimeException | Error rejected) {
                if (rejected instanceof Error) { recordFailure(rejected); }
                try { requestRecovery.execute(task); }
                catch (RuntimeException | Error recoveryRejected) {
                    // Accepted terminal/publication owners cannot be dropped or run on an RPC callback.
                    recordFailure(Failures.append(rejected, recoveryRejected));
                    try { Thread.ofPlatform().daemon().name("request-work-emergency").start(task); }
                    catch (RuntimeException | Error failedToStart) {
                        throw failedToStart;
                    }
                }
            }
        }

        void submitSettlement(Runnable work, boolean inline) {
            checkState(!Thread.holdsLock(context), "resource settlement submission under Context lock");
            synchronized (this) {
                checkState(!sourceClosed && !submitted, "request work already handed off");
                submitted = true;
                owners++;
            }
            try {
                if (!requestWorkQueue.execute(context, () -> run(work, inline), this::reject,
                        inline ? Runnable::run : this::dispatch)) { release(); }
            } catch (RuntimeException | Error failure) { reject(failure); throw failure; }
        }

        private void reject(Throwable cause) {
            synchronized (this) {
                if (started) { return; }
                started = true;
            }
            recordFailure(cause);
            release();
        }

        private void run(Runnable work, boolean propagate) {
            synchronized (this) {
                if (started) { return; }
                started = true;
            }
            var previous = executingRequest.get();
            executingRequest.set(context);
            try { work.run(); }
            catch (Throwable failure) {
                recordFailure(failure);
                try { Logger.error("Request work failed: request_id={}", context.data().getRequestId(), failure); }
                catch (Throwable ignored) { }
                if (propagate) { throw Failures.propagate(failure, "request control failed"); }
            } finally {
                executingRequest.set(previous);
                release();
            }
        }

        private void release() {
            synchronized (this) { if (--owners != 0) { return; } }
            synchronized (workLifecycle) {
                pendingWork--;
                if (pendingWork == 0) { workLifecycle.notifyAll(); }
            }
        }

        synchronized boolean hasTaskOwner() { return owners > (sourceClosed ? 0 : 1); }

        @Override
        public void close() {
            synchronized (this) {
                if (sourceClosed) { return; }
                sourceClosed = true;
            }
            release();
        }
    }

    private void awaitRequestWork() {
        boolean interrupted = false;
        synchronized (workLifecycle) {
            while (pendingWork != 0) {
                try { workLifecycle.wait(); }
                catch (InterruptedException ignored) { interrupted = true; }
            }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
    }

    private void closeRequestWork() {
        synchronized (workLifecycle) { acceptingWork = false; }
        // A running owner can still hand off its exact request through executingRequest.
        awaitRequestWork();
        requestWorkers.shutdown();
        requestRecovery.shutdown();
        boolean interrupted = false;
        while (!requestWorkers.isTerminated() || !requestRecovery.isTerminated()) {
            try {
                requestWorkers.awaitTermination(1, TimeUnit.DAYS);
                requestRecovery.awaitTermination(1, TimeUnit.DAYS);
            } catch (InterruptedException ignored) { interrupted = true; }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
    }
    ResponseCompletionExecutor.CompletionRegistration registerResponseCompletion() {
        return responseCompletions.tryRegister();
    }
    ExpirationTimer deadlines() { return timer; }
    DeliveryMetricsReporter deliveryReporter() { return deliveryReporter; }
    RequestSchedulerReporter requestReporter() { return requestReporter; }

    private final Object schedulerLock = new Object();
    private AbstractRequestScheduler scheduler;
    private Throwable failure;

    boolean isAccepting() { return !requests.isStopping() && !requests.isClosed(); }

    /** 后台清理、准入或续接失败在此汇总；停服时统一报告，不执行 Future 监听或关闭流程。 */
    void recordFailure(Throwable cause) {
        synchronized (schedulerLock) { failure = Failures.append(failure, cause); }
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
            DeliveryDispatcher dispatcher, ConfigService config,
            EngineCancelChannel cancelChannel) {
        this(requests, endpoints, deliveryReporter, requestReporter, dispatcher, config, cancelChannel, System::currentTimeMillis);
    }

    SchedulerRuntime(RequestRepository requests, EndpointRegistry endpoints, DeliveryMetricsReporter deliveryReporter,
                     RequestSchedulerReporter requestReporter, DeliveryDispatcher dispatcher,
                     ConfigService config,
                     EngineCancelChannel cancelChannel, LongSupplier clock) {
        deliveries = new DeliveryExecution(config, this::recordFailure);
        this.cancelChannel = Objects.requireNonNull(cancelChannel, "cancelChannel");
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
        this.responseCompletions = new ResponseCompletionExecutor(config.loadBalanceConfig().getInternalRuntime().getBatchDispatchCompletionThreads());
        this.timer = new ExpirationTimer(requests);
    }

    @Scheduled(fixedRate = EXPIRATION_MAINTENANCE_INTERVAL_MS)
    void maintainExpiration() {
        if (requests.isClosed()) { return; }
        long ttlMs = config.loadBalanceConfig().getWorkerRegistry().getHealth().getStatusStaleAfterMs();
        long nowMs = clock.getAsLong();
        Throwable failure = Failures.run(null,
                () -> requests.expireTerminalRecords(com.google.common.math.LongMath.saturatedSubtract(nowMs, ttlMs)));
        failure = Failures.run(failure,
                () -> endpoints.evictExpiredOrphans(ttlMs, requests::retainsIdentity));
        Failures.rethrow(failure, "expiration maintenance failed");
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void report() {
        if (requests.isClosed()) {
            return;
        }
        try {
            deliveryReporter.reportSchedulerInflight(requests.liveRequestCount(), requests.oldestLiveRequestAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated("Failed to report scheduler inflight metrics", failure);
        }
        reportEndpoints("Prefill", () -> endpoints.endpointSnapshot(RoleType.PREFILL),
                endpoint -> ((PrefillEndpoint) endpoint).reportBatchMetrics(deliveryReporter),
                (address, endpoint) -> requestReporter.reportPrefillQueueDepth(address,
                        ((PrefillEndpoint) endpoint).ownership.progress().queueDepth()));
        reportEndpoints("Decode", () -> endpoints.endpointSnapshot(RoleType.DECODE),
                endpoint -> ((DecodeEndpoint) endpoint).reportBatchMetrics(deliveryReporter),
                (address, endpoint) -> ((DecodeEndpoint) endpoint).reportAdmissionMetrics(requestReporter));
    }

    private static <T> void reportEndpoints(
            String role, Supplier<Map<String, T>> snapshot,
            Consumer<T> batchMetrics, BiConsumer<String, T> admissionMetrics) {
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
                warnIsolated("Failed to report " + role + " endpoint metrics: endpoint=" + entry.getKey(), failure);
            }
            try {
                admissionMetrics.accept(entry.getKey(), entry.getValue());
            } catch (RuntimeException failure) {
                warnIsolated("Failed to report " + role + " admission metrics: endpoint=" + entry.getKey(), failure);
            }
        }
    }

    private static void warnIsolated(
            String message, RuntimeException failure) {
        try {
            Logger.warn(message, failure);
        } catch (Throwable ignored) {
            // Telemetry must never couple otherwise independent metric leaves.
        }
    }

    private enum SettlementScope { BATCH, ALL }

    private void awaitDeliveryCleanup(SettlementScope scope) {
        var active = requests.snapshotActive();
        var pending = active.stream().map(context -> context.snapshot().delivery())
                .filter(Objects::nonNull).map(ContextResult.DeliveryFacts::identity)
                .filter(Objects::nonNull)
                .filter(delivery -> scope == SettlementScope.ALL || delivery.kind() == DeliveryClaimKind.BATCH_ENQUEUE)
                .map(delivery -> signals.delivery(delivery).toCompletableFuture())
                .toArray(CompletableFuture[]::new);
        try {
            CompletableFuture.allOf(pending).get(DeliveryDispatcher.CLEANUP_TIMEOUT_MS + 1_000L,
                    TimeUnit.MILLISECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted waiting for delivery cleanup", interrupted);
        } catch (ExecutionException | TimeoutException failure) {
            throw new IllegalStateException("Unsettled deliveries at shutdown: "
                    + active.stream().map(context -> context.data().getRequestId()).toList(), failure);
        }
    }

    private void closeRequestExecutors() {
        Throwable failure = null;
        awaitRequestWork();
        deliveryCleanupExecutor.close();
        try { closeRequestWork(); } catch (Throwable cause) { failure = cause; }
        try { responseCompletions.close(); } catch (Throwable cause) { failure = Failures.append(failure, cause); }
        cleanupExecutor.shutdown();
        boolean interrupted = false;
        while (!cleanupExecutor.isTerminated()) {
            try { cleanupExecutor.awaitTermination(1, TimeUnit.DAYS); }
            catch (InterruptedException ignored) { interrupted = true; }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
        Failures.rethrow(failure, "request executors failed to close");
    }

    /** Spring 停服入口：停止接收、等待请求资源结算，再关闭共享执行设施。 */
    @PreDestroy
    void shutdown() {
        checkState(executingRequest.get() == null && !responseCompletions.isCompletingOnCurrentThread(),
                "scheduler shutdown cannot wait for its own request work or response callback");
        requests.stopAccepting();
        if (!requests.closeRegistration()) { return; }
        AbstractRequestScheduler scheduler;
        synchronized (schedulerLock) { scheduler = this.scheduler; }
        Throwable firstFailure = null;
        if (scheduler != null) {
            firstFailure = Failures.run(firstFailure, scheduler::closePlacement);
            firstFailure = Failures.run(firstFailure, this::awaitAdmissions);
        }
        for (var context : requests.snapshotActive()) {
            firstFailure = Failures.run(firstFailure,
                    () -> context.data().getScheduler().cancel(context, 0L, CancelReason.SHUTDOWN));
        }
        firstFailure = Failures.run(firstFailure, deliveries::awaitDeliveries);
        // Preserve BATCH cleanup failures before retiring their ledgers.
        firstFailure = Failures.run(firstFailure, () -> awaitDeliveryCleanup(SettlementScope.BATCH));
        // Retirement supplies exact local settlement facts for route decisions.
        firstFailure = Failures.run(firstFailure, endpoints::close);
        firstFailure = Failures.run(firstFailure, () -> awaitDeliveryCleanup(SettlementScope.ALL));
        firstFailure = Failures.run(firstFailure, timer::close);
        firstFailure = Failures.run(firstFailure, this::awaitRequestWork);
        var owners = new java.util.LinkedHashSet<AbstractRequestScheduler>();
        if (scheduler != null) { owners.add(scheduler); }
        for (var context : requests.snapshotActive()) { owners.add(context.data().getScheduler()); }
        for (var owner : owners) { firstFailure = Failures.run(firstFailure, owner::closeOutstandingAndTerminalize); }
        firstFailure = Failures.run(firstFailure, this::closeRequestExecutors);
        if (requests.liveRequestCount() != 0) {
            firstFailure = Failures.append(firstFailure,
                    new IllegalStateException("Unsettled requests at shutdown: " + requests.liveRequestCount()));
        }
        synchronized (schedulerLock) { firstFailure = Failures.append(failure, firstFailure); }
        Failures.rethrow(firstFailure, "Scheduler shutdown failed");
    }

}
