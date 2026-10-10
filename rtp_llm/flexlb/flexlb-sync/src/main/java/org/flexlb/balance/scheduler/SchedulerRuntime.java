package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkState;

/** Owns scheduler maintenance, metrics traversal, and ordered shutdown. */
@Component
public final class SchedulerRuntime {

    private static final long EXPIRATION_MAINTENANCE_INTERVAL_MS = 60_000L;

    private final RequestRepository requests;
    private final EndpointRegistry endpoints;
    private final DeliveryMetricsReporter reporter;
    private final RequestSchedulerReporter admissionReporter;
    private final DefaultBatchDispatcher dispatcher;
    private final ConfigService config;
    private final LongSupplier clock;

    private final org.flexlb.balance.eviction.EngineCancelChannel cancelChannel;
    private final java.util.concurrent.ScheduledThreadPoolExecutor cleanupExecutor =
            new java.util.concurrent.ScheduledThreadPoolExecutor(2,
                    Thread.ofPlatform().daemon().name("flexlb-request-cleanup-", 1).factory());

    public java.util.concurrent.ScheduledExecutorService cleanupExecutor() { return cleanupExecutor; }

    void startDeliveryCleanup(RequestContext.DeliveryClaim claim) {
        new DeliveryCleanupTask(claim, cancelChannel, cleanupExecutor,
                work -> continuations.submit(claim.item.ctx(), work)).start();
    }

    private final RequestContinuationExecutor continuations = new RequestContinuationExecutor();
    private final ResponseCompletionExecutor responseCompletions;
    private final ExpirationTimer timer;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    RequestRepository requests() { return requests; }
    RequestContinuationExecutor continuations() { return continuations; }

    /** Serializes accepted request work; admission owners drain before this executor closes. */
    public void executeContinuation(RequestContext context, Runnable work) {
        continuations.submit(context, work);
    }
    ResponseCompletionExecutor responseCompletions() { return responseCompletions; }
    ExpirationTimer timer() { return timer; }
    DeliveryMetricsReporter deliveryReporter() { return reporter; }
    RequestSchedulerReporter requestReporter() { return admissionReporter; }
    RecentCacheKeyTraceReporter recentCacheKeyTraceReporter() { return recentCacheKeyTraceReporter; }

    private final Object schedulerLock = new Object();
    private AbstractRequestScheduler scheduler;
    private volatile boolean stopping;
    private Throwable failure;

    boolean isAccepting() { return !stopping && !requests.isClosed(); }

    /** 只停收。已注册请求继续运行；shutdown 才执行整套关闭流程。 */
    public void stopAccepting() {
        synchronized (schedulerLock) { stopping = true; }
    }

    /** 后台清理、准入或续接失败在此汇总；停服时统一报告，不执行 Future 监听或关闭流程。 */
    void recordFailure(Throwable cause) {
        synchronized (schedulerLock) { failure = Failures.append(failure, cause); }
    }

    void initializeScheduler(AbstractRequestScheduler scheduler) {
        synchronized (schedulerLock) {
            checkState(!stopping, "scheduler is closed");
            checkState(this.scheduler == null, "scheduler already initialized");
            this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
        }
    }

    private void closeSchedulers() {
        AbstractRequestScheduler scheduler;
        synchronized (schedulerLock) {
            if (this.scheduler == null) { return; }
            scheduler = this.scheduler;
        }
        scheduler.closePlacement();
    }

    @Autowired
    SchedulerRuntime(
            RequestRepository requests,
            EndpointRegistry endpoints,
            DeliveryMetricsReporter reporter,
            RequestSchedulerReporter admissionReporter,
            DefaultBatchDispatcher dispatcher, ConfigService config, RecentCacheKeyTraceReporter recentCacheKeyTraceReporter,
            org.flexlb.balance.eviction.EngineCancelChannel cancelChannel) {
        this(requests, endpoints, reporter, admissionReporter, dispatcher, config, recentCacheKeyTraceReporter, cancelChannel, System::currentTimeMillis);
    }

    SchedulerRuntime(RequestRepository requests, EndpointRegistry endpoints, DeliveryMetricsReporter reporter,
                     RequestSchedulerReporter admissionReporter, DefaultBatchDispatcher dispatcher,
                     ConfigService config, RecentCacheKeyTraceReporter recentCacheKeyTraceReporter,
                     org.flexlb.balance.eviction.EngineCancelChannel cancelChannel, LongSupplier clock) {
        this.cancelChannel = Objects.requireNonNull(cancelChannel, "cancelChannel");
        cleanupExecutor.setRemoveOnCancelPolicy(true);
        cleanupExecutor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        cleanupExecutor.setContinueExistingPeriodicTasksAfterShutdownPolicy(false);
        this.requests = Objects.requireNonNull(requests, "requests");
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
        this.admissionReporter = Objects.requireNonNull(admissionReporter, "admissionReporter");
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
        this.config = Objects.requireNonNull(config, "config");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.recentCacheKeyTraceReporter = Objects.requireNonNull(recentCacheKeyTraceReporter);
        this.responseCompletions = new ResponseCompletionExecutor(config.loadBalanceConfig().getInternalRuntime().getBatchDispatchCompletionThreads());
        this.timer = new ExpirationTimer(requests);
    }

    @Scheduled(fixedRate = EXPIRATION_MAINTENANCE_INTERVAL_MS)
    void maintainExpiration() {
        if (requests.isClosed()) { return; }
        long ttlMs = config.loadBalanceConfig().getWorkerRegistry().getHealth().getStatusStaleAfterMs();
        long nowMs = clock.getAsLong();
        Throwable failure = Failures.run(null,
                () -> requests.expireTerminalRecords(subtractSaturated(nowMs, ttlMs)));
        failure = Failures.run(failure,
                () -> endpoints.evictExpiredOrphans(ttlMs, requests::retainsIdentity));
        Failures.rethrow(failure, "expiration maintenance failed");
    }

    private static long subtractSaturated(long value, long decrement) {
        try { return Math.subtractExact(value, decrement); }
        catch (ArithmeticException underflow) { return Long.MIN_VALUE; }
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    void report() {
        if (requests.isClosed()) {
            return;
        }
        reportSchedulerInflight();
        reportEndpoints("Prefill", endpoints::snapshotPrefillEndpoints,
                endpoint -> endpoint.reportBatchMetrics(reporter),
                (address, endpoint) -> admissionReporter.reportPrefillQueueDepth(address, endpoint.queuedRequestCount()));
        reportEndpoints("Decode", endpoints::snapshotDecodeEndpoints,
                endpoint -> endpoint.reportBatchMetrics(reporter),
                (address, endpoint) -> endpoint.reportAdmissionMetrics(admissionReporter));
    }

    private void reportSchedulerInflight() {
        try {
            reporter.reportSchedulerInflight(requests.liveRequestCount(), requests.oldestLiveRequestAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated(
                    "Failed to report scheduler inflight metrics", failure);
        }
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

    private AbstractRequestScheduler initializedScheduler() {
        synchronized (schedulerLock) {
            return scheduler;
        }
    }

    private void awaitAdmissionMutations() {
        var scheduler = initializedScheduler();
        if (scheduler != null) { scheduler.awaitAdmissionMutations(); }
    }

    private void closeOutstandingRequests() {
        var scheduler = initializedScheduler();
        if (scheduler != null) { scheduler.closeOutstandingAndTerminalize(); }
    }

    void closeRequestExecutors() {
        Throwable failure = null;
        try { continuations.close(); } catch (Throwable cause) { failure = cause; }
        try { responseCompletions.close(); } catch (Throwable cause) { failure = Failures.append(failure, cause); }
        cleanupExecutor.shutdown();
        boolean interrupted = false;
        while (!cleanupExecutor.isTerminated()) {
            try { cleanupExecutor.awaitTermination(1, java.util.concurrent.TimeUnit.DAYS); }
            catch (InterruptedException ignored) { interrupted = true; }
        }
        if (interrupted) { Thread.currentThread().interrupt(); }
        Failures.rethrow(failure, "request executors failed to close");
    }

    private void abandonOutstandingDeliveries() {
        for (RequestContext context : requests.snapshotActive()) {
            context.scheduler().cancelRequest(context, 0L, CancelReason.SHUTDOWN);
        }
    }

    private void awaitDeliveryCleanup() {
        var pending = requests.snapshotActive().stream().map(RequestContext::delivery)
                .filter(Objects::nonNull).map(delivery -> delivery.settlement().toCompletableFuture())
                .toArray(java.util.concurrent.CompletableFuture[]::new);
        try { java.util.concurrent.CompletableFuture.allOf(pending).get(DeliveryCleanupTask.CLEANUP_TIMEOUT_MS + 1_000L, java.util.concurrent.TimeUnit.MILLISECONDS); }
        catch (Exception failure) {
            throw new IllegalStateException("Unsettled deliveries at shutdown: "
                    + requests.snapshotActive().stream().map(RequestContext::getRequestId).toList(), failure);
        }
    }

    /** Spring 停服入口：停止接收、等待请求资源结算，再关闭共享执行设施。 */
    @PreDestroy
    void shutdown() {
        stopAccepting();
        if (!requests.closeRegistration()) { return; }
        Runnable[] steps = {
                this::closeSchedulers,
                this::awaitAdmissionMutations,
                this::abandonOutstandingDeliveries,
                dispatcher::shutdownAndAwait,
                this::awaitDeliveryCleanup,
                endpoints::close,
                timer::close,
                continuations::awaitIdle,
                this::closeOutstandingRequests,
                this::closeRequestExecutors
        };
        Throwable firstFailure = null;
        for (Runnable step : steps) {
            try { step.run(); }
            catch (Throwable failure) {
                if (firstFailure == null) { firstFailure = failure; }
                else if (failure != firstFailure) { firstFailure.addSuppressed(failure); }
            }
        }
        if (requests.liveRequestCount() != 0) {
            firstFailure = Failures.append(firstFailure,
                    new IllegalStateException("Unsettled requests at shutdown: " + requests.liveRequestCount()));
        }
        synchronized (schedulerLock) { firstFailure = Failures.append(failure, firstFailure); }
        Failures.rethrow(firstFailure, "Scheduler shutdown failed");
    }

}
