package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.priority.AdmissionFailure;
import org.flexlb.balance.scheduler.priority.AdmissionFailureClassifier;
import org.flexlb.balance.scheduler.priority.AdmissionLease;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.InflightRegistrar;
import org.flexlb.balance.scheduler.priority.InflightRegistrar.PriorityCanceledObservation;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Coordinates priority-aware request scheduling for FlexLB disaggregated inference.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Request admission and routing</li>
 *   <li>Inflight lifecycle management (inflight map, TTL cleanup)</li>
 *   <li>Priority decision-group coordination through {@link WorkerBatcher}</li>
 *   <li>Delivery-independent lifecycle and resource ownership</li>
 *   <li>Batch enqueue or caller-owned route-decision delivery</li>
 *   <li>Resource rollback on failure or completion</li>
 * </ul>
 *
 * <p>External exposure is delegated to mode-specific {@link DecisionDelivery}
 * implementations. This class
 * commits endpoint ledgers and lifecycle ownership before a delivery can make
 * a request visible to either the engine or caller.
 */
@Component
public class PriorityScheduler implements DecisionGroupHandler, DecisionDelivery.Callback, InflightRegistrar {

    private static final int DEFAULT_COMPLETION_WORKERS = Math.max(
            2, Math.min(8, Runtime.getRuntime().availableProcessors()));
    private static final int DEFAULT_COMPLETION_QUEUE_CAPACITY = 1_024;
    private static final long DEFAULT_CANCEL_ACK_TIMEOUT_MS = 50L;
    private static final int OUTSTANDING_ADMISSION_CLOSED = -1;
    private static final long DELIVERY_LIFECYCLE_CLOSED = Long.MIN_VALUE;
    private static final long DELIVERY_LIFECYCLE_COUNT_MASK = Long.MAX_VALUE;

    /**
     * Isolates caller continuations from WorkerBatcher/RPC/status threads with
     * a strict thread and retained-task bound. Saturation falls back to the
     * submitting thread only at the lock-free completion boundary, providing
     * natural backpressure without dropping a terminal response.
     */
    private final ThreadPoolExecutor responseCompletionExecutor;

    /**
     * Timer threads only enqueue deadline reducers; they never run scheduler
     * state transitions or caller continuations themselves.
     */
    private final ScheduledThreadPoolExecutor requestExpirationTimer;
    /** Owns every delayed Engine-fence reconciliation for this scheduler instance. */
    private final ScheduledThreadPoolExecutor engineFenceRetryTimer;
    /** One-way lifecycle gate; terminal completions remain allowed after it closes. */
    private final AtomicBoolean shuttingDown = new AtomicBoolean();
    /**
     * Sign bit closes new delivery work; the remaining bits count groups which
     * already crossed the shutdown gate. Normal entry/exit is monitor-free.
     */
    private final AtomicLong deliveryLifecycle = new AtomicLong();
    /**
     * Exact cluster-wide QUEUE ownership bound. Unlike {@code inflight.size()},
     * this counter includes admissions which have not reached registration yet.
     * The CAS increment is the capacity linearization point for every submit.
     */
    private final AtomicInteger outstandingRequestCount = new AtomicInteger();
    /** Used only by shutdown and the final active delivery leaving after close. */
    private final Object deliveryDrainMonitor = new Object();

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchEnqueueDelivery batchEnqueueDelivery;
    private final DecisionDelivery<List<BatchItem>> routeDecisionDelivery;
    private final BatchSchedulerReporter reporter;
    private final PriorityAdmissionScheduler admissionScheduler;
    private final EngineCancelChannel engineCancelChannel;
    private final EngineFencePolicy engineFencePolicy;
    private final Map<String, InflightEntry> inflight = new ConcurrentHashMap<>();
    private final Map<String, RequestLifecycleSnapshot> terminalStates = new ConcurrentHashMap<>();
    /**
     * One request-generation gate shared by routing, admission, registration,
     * queue commit, deadline and external cancellation. It is never a global
     * hot lock, and the gate itself is the public future so no wrapper is
     * allocated on the request path.
     */
    private final Map<String, RequestGenerationGate> generationGates =
            new ConcurrentHashMap<>();
    /**
     * Cold-path index for fences which exhausted their bounded fast retries.
     * Values do not retain an {@link InflightEntry}; the authoritative request
     * graph remains in {@link #inflight} until an Engine terminal proof arrives.
     */
    private final Map<String, EngineFenceRegistration> quarantinedEngineFences =
            new ConcurrentHashMap<>();
    /** Fair round-robin probe order; stale generation refs are discarded lazily. */
    private final ConcurrentLinkedQueue<EngineFenceProbeRef> quarantinedProbeQueue =
            new ConcurrentLinkedQueue<>();
    private final BatchIdGenerator batchIdGenerator;
    /** Linearizes the final endpoint-ledger commit and external delivery with fencing. */
    private final Object deliveryFence = new Object();

    @Autowired
    public PriorityScheduler(ConfigService configService,
                                Router router,
                                EndpointRegistry endpointRegistry,
                                BatchDispatcher batchDispatcher,
                                BatchSchedulerReporter reporter,
                                PriorityAdmissionScheduler admissionScheduler,
                                Environment environment,
                                EngineCancelChannel engineCancelChannel) {
        this(configService, router, endpointRegistry, batchDispatcher, reporter,
                admissionScheduler, environment, engineCancelChannel,
                EngineFencePolicy.productionDefaults());
    }

    /** Package-visible policy injection keeps bounded-retry tests deterministic. */
    PriorityScheduler(ConfigService configService,
                      Router router,
                      EndpointRegistry endpointRegistry,
                      BatchDispatcher batchDispatcher,
                      BatchSchedulerReporter reporter,
                      PriorityAdmissionScheduler admissionScheduler,
                      Environment environment,
                      EngineCancelChannel engineCancelChannel,
                      EngineFencePolicy engineFencePolicy) {
        this(configService, router, endpointRegistry, batchDispatcher, reporter,
                admissionScheduler, environment, engineCancelChannel,
                engineFencePolicy, RouteDecisionDelivery.INSTANCE);
    }

    /** Package-visible delivery injection makes publication races deterministic in tests. */
    PriorityScheduler(ConfigService configService,
                      Router router,
                      EndpointRegistry endpointRegistry,
                      BatchDispatcher batchDispatcher,
                      BatchSchedulerReporter reporter,
                      PriorityAdmissionScheduler admissionScheduler,
                      Environment environment,
                      EngineCancelChannel engineCancelChannel,
                      EngineFencePolicy engineFencePolicy,
                      DecisionDelivery<List<BatchItem>> routeDecisionDelivery) {
        this(configService, router, endpointRegistry, batchDispatcher, reporter,
                admissionScheduler, environment, engineCancelChannel,
                engineFencePolicy, routeDecisionDelivery,
                CompletionExecutorPolicy.productionDefaults());
    }

    /** Package-visible executor sizing keeps saturation tests small and deterministic. */
    PriorityScheduler(ConfigService configService,
                      Router router,
                      EndpointRegistry endpointRegistry,
                      BatchDispatcher batchDispatcher,
                      BatchSchedulerReporter reporter,
                      PriorityAdmissionScheduler admissionScheduler,
                      Environment environment,
                      EngineCancelChannel engineCancelChannel,
                      EngineFencePolicy engineFencePolicy,
                      DecisionDelivery<List<BatchItem>> routeDecisionDelivery,
                      CompletionExecutorPolicy completionExecutorPolicy) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.batchEnqueueDelivery = new BatchEnqueueDelivery(batchDispatcher);
        this.routeDecisionDelivery = Objects.requireNonNull(routeDecisionDelivery);
        this.reporter = reporter;
        this.admissionScheduler = admissionScheduler;
        this.engineCancelChannel = Objects.requireNonNull(engineCancelChannel);
        this.engineFencePolicy = Objects.requireNonNull(engineFencePolicy);
        this.responseCompletionExecutor = newResponseCompletionExecutor(
                Objects.requireNonNull(completionExecutorPolicy));
        this.requestExpirationTimer = newTimer("priority-scheduler-request-expiration");
        this.engineFenceRetryTimer = newTimer("priority-scheduler-engine-fence-timer");
        // Initialize Snowflake batch ID generator with master identity
        this.batchIdGenerator = new BatchIdGenerator(detectLocalIp(), detectPort(environment));
    }

    private static ThreadPoolExecutor newResponseCompletionExecutor(
            CompletionExecutorPolicy policy) {
        AtomicInteger workerSequence = new AtomicInteger();
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
                policy.workerCount(), policy.workerCount(),
                0L, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(policy.queueCapacity()),
                runnable -> {
                    Thread thread = new Thread(runnable,
                            "priority-scheduler-completion-"
                                    + workerSequence.getAndIncrement());
                    thread.setDaemon(true);
                    return thread;
                },
                new ThreadPoolExecutor.AbortPolicy());
        executor.prestartAllCoreThreads();
        return executor;
    }

    private static ScheduledThreadPoolExecutor newTimer(String threadName) {
        ScheduledThreadPoolExecutor timer = new ScheduledThreadPoolExecutor(1, runnable -> {
            Thread thread = new Thread(runnable, threadName);
            thread.setDaemon(true);
            return thread;
        });
        timer.setRemoveOnCancelPolicy(true);
        timer.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        return timer;
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            Logger.warn("Failed to detect local IP, using 127.0.0.1 as fallback", e);
            return "127.0.0.1";
        }
    }

    private static int detectPort(Environment environment) {
        String portStr = environment == null ? null : environment.getProperty("server.port");
        if (portStr == null) {
            portStr = System.getProperty("server.port", "7001");
        }
        try {
            return Integer.parseInt(portStr);
        } catch (NumberFormatException e) {
            return 7001;
        }
    }

    /** Reserve one global request slot without a check-then-act window. */
    private boolean tryAcquireOutstandingPermit(int limit) {
        while (true) {
            int current = outstandingRequestCount.get();
            if (current == OUTSTANDING_ADMISSION_CLOSED
                    || current == Integer.MAX_VALUE
                    || (limit > 0 && current >= limit)) {
                return false;
            }
            if (outstandingRequestCount.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }

    /** Package-visible exact capacity diagnostic used by concurrency tests. */
    int outstandingRequestCount() {
        return Math.max(0, outstandingRequestCount.get());
    }

    // ==================== Request submission ====================

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        RequestGenerationGate generation = new RequestGenerationGate();
        CompletableFuture<Response> future = generation;
        try {
            if (shuttingDown.get()) {
                completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                        "priority scheduler is shutting down");
                return future;
            }
            if (ctx == null || ctx.getRequest() == null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST, null);
                return future;
            }
            ctx.setEnqueueTime(System.currentTimeMillis());
            RequestGenerationGate prior = generationGates.putIfAbsent(
                    ctx.getRequestId(), generation);
            if (prior != null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }
            future.whenComplete((ignored, error) -> {
                // A public terminal is no longer outstanding admission work.
                // Later lifecycle cleanup calls the same exact-once release.
                generation.releaseOutstandingPermit();
                InflightEntry entry = inflight.get(ctx.getRequestId());
                boolean mutationInProgress;
                synchronized (generation) {
                    mutationInProgress = generation.admissionMutationInProgress;
                }
                if (!mutationInProgress
                        && (entry == null || entry.item.future() != generation)) {
                    generationGates.remove(ctx.getRequestId(), generation);
                }
            });

            if (inflight.containsKey(ctx.getRequestId())
                    || terminalStates.containsKey(ctx.getRequestId())) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }

            FlexlbConfig activeConfig = configService.loadBalanceConfig();
            int maxOutstanding = activeConfig.queueScheduler()
                    .getCapacity().getMaxOutstandingRequestsGlobal();
            if (!tryAcquireOutstandingPermit(maxOutstanding)) {
                if (shuttingDown.get()
                        || outstandingRequestCount.get() == OUTSTANDING_ADMISSION_CLOSED) {
                    completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                            "priority scheduler is shutting down");
                } else if (activeConfig.isPriorityOrdering()) {
                    Response response = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                            AdmissionRejectReason.RESOURCE_EXHAUSTED);
                    response.setErrorMessage(StrategyErrorType.RESOURCE_EXHAUSTED
                            .buildErrorMessage("master outstanding capacity exhausted"));
                    future.complete(response);
                } else {
                    completeError(future, StrategyErrorType.QUEUE_FULL, null);
                }
                return future;
            }
            if (!generation.bindOutstandingPermit(outstandingRequestCount)) {
                return future;
            }
            if (shuttingDown.get()) {
                completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                        "priority scheduler is shutting down");
                return future;
            }

            long nowMs = System.currentTimeMillis();
            if (ctx.requestExpired(nowMs)) {
                if (activeConfig.isPriorityOrdering()) {
                    AdmissionFailure failure = AdmissionFailure.resourceExhausted();
                    Response response = Response.error(
                            failure.errorType(), failure.reason());
                    response.setErrorMessage(failure.errorType().buildErrorMessage(
                            "request expired: expires_at_ms="
                                    + ctx.getRequestExpiresAtMs() + " now_ms=" + nowMs));
                    future.complete(response);
                } else {
                    completeError(future, generation.deadlineErrorType,
                            "request scheduling deadline has expired");
                }
                return future;
            }
            // Arm the one absolute-expiration reducer before scheduling can
            // publish a delivery. Priority admission rechecks expiration
            // inside its mutation boundary to close this observation race.
            attachRequestExpiration(ctx, future);

            // PRIORITY ordering delegates plan/commit to the priority
            // admission scheduler; FIFO continues through the ordinary path.
            if (activeConfig.isPriorityOrdering() && admissionScheduler != null) {
                if (shuttingDown.get()) {
                    completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                            "priority scheduler is shutting down");
                    return future;
                }
                admissionScheduler.schedule(ctx, future, this);
                return future;
            }

            // route() may reserve Decode capacity before it returns. Keep that
            // resource mutation in the same generation ownership window as
            // register/offer so Cancel cannot publish a terminal tombstone
            // before the reservation is either committed or rolled back.
            if (!claimAdmissionMutation(ctx.getRequestId(), future)) {
                return future;
            }
            Response routeResponse = null;
            BatchItem submittedItem = null;
            InflightEntry submittedEntry = null;
            try {
                routeResponse = router.route(ctx);
                if (routeResponse == null || !routeResponse.isSuccess()) {
                    if (routeResponse != null) {
                        future.complete(routeResponse);
                    } else {
                        completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER, null);
                    }
                    return future;
                }
                if (shuttingDown.get()) {
                    rollback(routeResponse);
                    completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                            "priority scheduler is shutting down");
                    return future;
                }

                ServerStatus prefill = findPrefillServer(routeResponse);
                ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
                if (prefill == null) {
                    rollback(routeResponse);
                    completeError(future, StrategyErrorType.NO_PREFILL_WORKER, null);
                    return future;
                }

                String prefillIpPort = prefill.getLogicalIpPort();
                WorkerEndpoint selectedEndpoint = prefill.getRole() == RoleType.PREFILL
                        ? endpointRegistry.getPrefill(prefillIpPort)
                        : endpointRegistry.get(prefill.getRole(), prefillIpPort);
                if (!(selectedEndpoint instanceof PrefillEndpoint prefillEp)) {
                    rollback(routeResponse);
                    StrategyErrorType errorType = prefill.getRole() == RoleType.PDFUSION
                            ? StrategyErrorType.NO_PDFUSION_WORKER
                            : StrategyErrorType.NO_PREFILL_WORKER;
                    completeError(future, errorType, null);
                    return future;
                }

                DecodeEndpoint decodeEp = null;
                if (decode != null) {
                    String decodeIpPort = decode.getLogicalIpPort();
                    decodeEp = endpointRegistry.getDecode(decodeIpPort);
                }

                BatchItem item = new BatchItem(
                        ctx, future, routeResponse, copyOf(prefill), copyOf(decode),
                        prefillEp, decodeEp, System.currentTimeMillis());
                InflightEntry entry = new InflightEntry(item, false);
                submittedItem = item;
                submittedEntry = entry;
                StrategyErrorType commitError = null;
                ResponseCompletion commitFailure = null;
                ResponseCompletion offerExceptionPublication = null;
                Throwable offerException = null;
                boolean generationClosed = false;
                synchronized (generation) {
                    if (!generation.isOpen()) {
                        generationClosed = true;
                    } else {
                        InflightEntry existing = inflight.putIfAbsent(ctx.getRequestId(), entry);
                        if (existing != null || terminalStates.containsKey(ctx.getRequestId())) {
                            if (existing == null) {
                                inflight.remove(ctx.getRequestId(), entry);
                            }
                            commitError = StrategyErrorType.INVALID_REQUEST;
                        } else if (shuttingDown.get()) {
                            inflight.remove(ctx.getRequestId(), entry);
                            commitError = StrategyErrorType.BATCH_DISPATCH_FAILED;
                        } else {
                            observeExternalFutureTerminal(entry);
                            ctx.setRouteSubmittedNanos(System.nanoTime());
                            try {
                                if (!prefillEp.getBatcher().tryOffer(item)) {
                                    synchronized (entry) {
                                        commitFailure = reduceOrdinaryTerminalLocked(
                                                entry,
                                                DeferredTerminal.failure(
                                                        StrategyErrorType.BATCH_DISPATCH_FAILED,
                                                        "Worker scheduling queue rejected request",
                                                        false));
                                    }
                                }
                            } catch (Throwable failure) {
                                // tryOffer is permitted to fail after mutating
                                // its queue. Claim cleanup and publication
                                // before releasing the generation monitor, so
                                // Cancel/worker reducers can only observe an
                                // existing owner during the unlocked unwind.
                                synchronized (entry) {
                                    entry.engineOwnershipState =
                                            EngineOwnershipState.TERMINAL;
                                    entry.cleanupOwned = true;
                                    offerExceptionPublication = errorPublicationLocked(
                                            entry,
                                            StrategyErrorType.BATCH_DISPATCH_FAILED,
                                            "Worker scheduling queue offer failed: "
                                                    + failure.getMessage());
                                }
                                offerException = failure;
                            }
                        }
                    }
                }
                if (offerException != null) {
                    String detail = "Worker scheduling queue offer failed: "
                            + offerException.getMessage();
                    releaseLocallyOwnedResources(entry, detail);
                    synchronized (entry) {
                        if (inflight.get(ctx.getRequestId()) == entry) {
                            RequestLifecycleSnapshot failed =
                                    entry.lifecycle.fail(detail);
                            finishEntry(entry, failed);
                        }
                    }
                    Logger.error("PriorityScheduler queue offer failed: request_id={}",
                            ctx.getRequestId(), offerException);
                    submitResponseCompletion(offerExceptionPublication);
                    return future;
                }
                if (commitFailure != null) {
                    submitResponseCompletion(commitFailure);
                    return future;
                }
                if (generationClosed || commitError != null) {
                    rollback(item);
                    if (commitError != null) {
                        String detail = commitError == StrategyErrorType.INVALID_REQUEST
                                ? "duplicate request_id: " + ctx.getRequestId()
                                : "priority scheduler is shutting down";
                        completeError(future, commitError, detail);
                    }
                    return future;
                }

                // Report route+submit time: from schedule() entry (ctx.startTime)
                // to batcher offer completion.
                try {
                    reporter.reportRouteSubmitTimeMs(
                            RoleType.PREFILL.name(),
                            prefillEp.getIp(),
                            System.currentTimeMillis() - ctx.getStartTime());
                } catch (RuntimeException telemetryFailure) {
                    Logger.warn("Failed to record route-submit telemetry: request_id={}",
                            ctx.getRequestId(), telemetryFailure);
                }
            } catch (Throwable submitFailure) {
                cleanupFailedFifoSubmission(
                        submittedEntry, submittedItem, routeResponse, submitFailure);
                throw submitFailure;
            } finally {
                completeAdmissionMutation(ctx.getRequestId(), future);
            }
        } catch (Throwable t) {
            Logger.error("PriorityScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            String detail = "Submit failed: " + t.getMessage();
            if (ctx != null) {
                InflightEntry entry = inflight.get(ctx.getRequestId());
                if (entry != null && entry.item.future() == future) {
                    // A post-commit exception must obey the same delivery and
                    // Engine-ownership fence as an externally failed future;
                    // it is never safe to detach or blindly roll back here.
                    reduceExternalFutureTerminal(entry.item, detail);
                }
            }
            completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED, detail);
        }
        return future;
    }

    /**
     * Unwind the exact FIFO route transaction before its mutation claim is
     * released. A pending Cancel therefore cannot become terminal while a
     * route reservation or a partially offered queue item is still live.
     */
    private void cleanupFailedFifoSubmission(
            InflightEntry entry,
            BatchItem item,
            Response routeResponse,
            Throwable submitFailure) {
        if (entry == null) {
            rollback(routeResponse);
            return;
        }
        boolean cleanup = false;
        synchronized (deliveryFence) {
            synchronized (entry) {
                if (inflight.get(entry.item.requestId()) == entry
                        && !entry.cleanupOwned) {
                    entry.engineOwnershipState = EngineOwnershipState.TERMINAL;
                    entry.cleanupOwned = true;
                    cleanup = true;
                }
            }
        }
        if (cleanup) {
            releaseLocalAdmissionCleanup(
                    entry, "FIFO submit failed: " + submitFailure.getMessage());
        } else if (item != null && inflight.get(item.requestId()) == null) {
            // Registration did not commit, or another exact reducer already
            // detached it. Endpoint release is idempotent and cannot affect a
            // reused id while the generation mutation remains held.
            rollbackOnce(entry);
        }
    }

    /**
     * Schedule request expiration as a reducer event. Directly attaching
     * {@link CompletableFuture#orTimeout(long, TimeUnit)} would let the timer
     * permanently complete the frontend future while a priority Cancel owns
     * the request; a later authoritative CANCELED observation could then no
     * longer publish PRIORITY_PREEMPTED. FIFO and PRIORITY both arm this same
     * absolute-expiration timer.
     */
    void attachRequestExpiration(BalanceContext ctx,
                                 CompletableFuture<Response> future) {
        if (shuttingDown.get()) {
            return;
        }
        long remainingMs = ctx.getRequestExpiresAtMs() - System.currentTimeMillis();
        long delayMs = Math.max(1, remainingMs);
        String requestId = ctx.getRequestId();
        ScheduledFuture<?> timeout;
        try {
            timeout = requestExpirationTimer.schedule(
                    () -> executeResponseTask(
                            () -> onRequestExpired(requestId, future)),
                    delayMs, TimeUnit.MILLISECONDS);
        } catch (RejectedExecutionException timerStopped) {
            if (shuttingDown.get()) {
                return;
            }
            throw timerStopped;
        }
        // Eager cancellation plus remove-on-cancel prevents the timer queue
        // from retaining future -> request context until expiration.
        future.whenComplete((ignored, error) -> timeout.cancel(false));
    }

    int requestExpirationQueueSize() {
        return requestExpirationTimer.getQueue().size();
    }

    boolean removesCanceledRequestExpirations() {
        return requestExpirationTimer.getRemoveOnCancelPolicy();
    }

    int engineFenceRetryQueueSize() {
        return engineFenceRetryTimer.getQueue().size();
    }

    CompletionExecutorSnapshot completionExecutorSnapshot() {
        int queueSize = responseCompletionExecutor.getQueue().size();
        return new CompletionExecutorSnapshot(
                responseCompletionExecutor.getMaximumPoolSize(),
                queueSize + responseCompletionExecutor.getQueue().remainingCapacity(),
                queueSize,
                responseCompletionExecutor.getLargestPoolSize(),
                responseCompletionExecutor.getCompletedTaskCount(),
                responseCompletionExecutor.isShutdown());
    }

    boolean awaitCompletionExecutorTermination(long timeout, TimeUnit unit)
            throws InterruptedException {
        return responseCompletionExecutor.awaitTermination(timeout, unit);
    }

    /** Deliver request expiration through the ordinary-terminal reducer. */
    // Package-visible for dispatch/expiration linearization tests.
    void onRequestExpired(String requestId,
                          CompletableFuture<Response> expectedFuture) {
        if (shuttingDown.get()) {
            return;
        }
        RequestGenerationGate gate = generationGates.get(requestId);
        ResponseCompletion publication = null;
        boolean reduceInflight = false;
        if (gate == expectedFuture) {
            synchronized (gate) {
                InflightEntry entry = inflight.get(requestId);
                if (entry != null && entry.item.future() == expectedFuture) {
                    if (expectedFuture.isDone() || !gate.closeCommits()) {
                        return;
                    }
                    reduceInflight = true;
                } else {
                    publication = reduceRequestExpiration(
                            requestId, expectedFuture, gate);
                }
            }
        } else {
            // Direct/legacy test seams already have an inflight entry and do
            // not participate in the pre-registration admission race.
            InflightEntry entry = inflight.get(requestId);
            if (entry != null && entry.item.future() == expectedFuture
                    && !expectedFuture.isDone()) {
                reduceInflight = true;
            } else {
                publication = reduceRequestExpiration(requestId, expectedFuture, null);
            }
        }
        if (reduceInflight) {
            // All live-request expiration semantics share the public Cancel
            // reducer: priority handoff, NOT_FOUND transfer, delivery fencing
            // and first-cause retention therefore cannot drift by caller.
            cancelRequest(requestId, 0, CancelReason.DEADLINE_EXCEEDED);
            return;
        }
        submitResponseCompletion(publication);
    }

    /** Reduce expiration while the optional request-scoped admission gate is held. */
    private ResponseCompletion reduceRequestExpiration(
            String requestId,
            CompletableFuture<Response> expectedFuture,
            RequestGenerationGate gate) {
        if (expectedFuture.isDone()) {
            return null;
        }
        if (gate != null && !gate.isOpen()) {
            // Every pre-registration terminal reducer claims this same gate.
            // Its response may still be queued on the completion executor, so
            // future.isDone() alone is not a sufficient first-cause check.
            return null;
        }
        if (gate == null || generationGates.get(requestId) != gate
                || terminalStates.containsKey(requestId)) {
            // finishEntry publishes its tombstone before retiring inflight.
            // Its response may still be queued, so the future itself can be
            // incomplete even though this generation already has a winner.
            return null;
        }
        if (inflight.containsKey(requestId)
                || !gate.closeCommits()) {
            return null;
        }
        String detail = "request scheduling deadline exceeded before inflight registration";
        RequestLifecycle lifecycle = new RequestLifecycle(requestId);
        lifecycle.requestCancel(detail);
        if (gate.admissionMutationInProgress) {
            gate.pendingAdmissionCancellation = lifecycle;
            gate.pendingAdmissionCancelReason = CancelReason.DEADLINE_EXCEEDED;
            return null;
        }
        RequestLifecycleSnapshot terminal = lifecycle.timeout(detail);
        terminalStates.put(requestId, terminal);
        return ResponseCompletion.terminal(gate,
                buildErrorResponse(gate == null
                        ? StrategyErrorType.BATCH_SLO_EXPIRED
                        : gate.deadlineErrorType, detail));
    }

    // ==================== InflightRegistrar (priority commit protocol) ====================

    /**
     * Register a priority-admitted item into the shared inflight tracking so
     * dispatch, completion, expiration, and rollback behave identically.
     * Mirrors the duplicate-request check in {@link #submit}.
     */
    @Override
    public boolean registerInflight(BatchItem item) {
        if (shuttingDown.get()) {
            return false;
        }
        // This registrar is the priority-admission commit boundary. FIFO
        // submit() constructs its entry directly with priorityAdmission=false.
        RequestGenerationGate gate = generationGates.get(item.requestId());
        if (gate == item.future()) {
            synchronized (gate) {
                if (!gate.isOpen()) {
                    return false;
                }
                return registerInflightOpen(item, true);
            }
        }
        if (gate != null || item.future().isDone()
                || terminalStates.containsKey(item.requestId())) {
            return false;
        }
        return registerInflightOpen(item, true);
    }

    @Override
    public boolean isInflightGeneration(
            String requestId, CompletableFuture<?> future) {
        InflightEntry entry = inflight.get(requestId);
        return entry != null && entry.item.future() == future;
    }

    @Override
    public boolean isAdmissionOpen(String requestId, CompletableFuture<?> future) {
        if (shuttingDown.get()) {
            return false;
        }
        RequestGenerationGate gate = generationGates.get(requestId);
        if (gate != future) {
            return gate == null && !future.isDone()
                    && !terminalStates.containsKey(requestId);
        }
        synchronized (gate) {
            return gate.isOpen();
        }
    }

    @Override
    public boolean claimAdmissionMutation(
            String requestId, CompletableFuture<?> future) {
        if (shuttingDown.get()) {
            return false;
        }
        RequestGenerationGate generation = generationGates.get(requestId);
        if (generation != future) {
            return false;
        }
        synchronized (generation) {
            if (!generation.isOpen()
                    || generation.admissionMutationInProgress) {
                return false;
            }
            generation.admissionMutationInProgress = true;
            return true;
        }
    }

    @Override
    public void completeAdmissionMutation(
            String requestId, CompletableFuture<?> future) {
        RequestGenerationGate generation = generationGates.get(requestId);
        if (generation != future) {
            return;
        }
        ResponseCompletion publication = null;
        synchronized (generation) {
            if (!generation.admissionMutationInProgress) {
                return;
            }
            generation.admissionMutationInProgress = false;
            if (generation.pendingAdmissionCancellation != null) {
                RequestLifecycleSnapshot terminal = settleCancellationLifecycle(
                        generation.pendingAdmissionCancellation,
                        generation.pendingAdmissionCancelReason,
                        cancelDetail(generation.pendingAdmissionCancelReason));
                terminalStates.put(requestId, terminal);
                publication = ResponseCompletion.terminal(
                        generation,
                        buildErrorResponse(
                                cancelErrorType(generation.pendingAdmissionCancelReason,
                                        generation.deadlineErrorType),
                                terminal.detail()));
            } else if (future.isDone() && inflight.get(requestId) == null) {
                generationGates.remove(requestId, generation);
            }
        }
        submitResponseCompletion(publication);
    }

    private boolean registerInflightOpen(BatchItem item, boolean priorityAdmission) {
        if (shuttingDown.get() || item.future().isDone()
                || terminalStates.containsKey(item.requestId())) {
            return false;
        }
        InflightEntry entry = new InflightEntry(item, priorityAdmission);
        InflightEntry existing = inflight.putIfAbsent(item.requestId(), entry);
        if (existing == null && !terminalStates.containsKey(item.requestId())
                && !shuttingDown.get()) {
            observeExternalFutureTerminal(entry);
            return true;
        }
        if (existing == null) {
            inflight.remove(item.requestId(), entry);
        }
        return false;
    }

    /**
     * Bind external completion to the exact registered generation. This is a
     * scheduler concern for both FIFO and PRIORITY requests; an optional
     * AdmissionLease may independently close its permit/timer, but is not the
     * only path which can drive request resource reconciliation.
     */
    private void observeExternalFutureTerminal(InflightEntry entry) {
        entry.item.future().whenComplete((response, error) -> {
            if (error == null && response != null && response.isSuccess()) {
                return;
            }
            synchronized (entry) {
                if (inflight.get(entry.item.requestId()) != entry
                        || entry.admissionLease != null
                        || entry.externalFutureTerminalClaimed) {
                    return;
                }
                // FIFO requests have no AdmissionLease. Claim their future
                // terminal here; priority requests attach a lease under this
                // same entry monitor and therefore have exactly one observer.
                entry.externalFutureTerminalClaimed = true;
            }
            reduceExternalFutureTerminal(
                    entry.item, "external schedule future terminal");
        });
    }

    @Override
    public boolean attachAdmissionLease(BatchItem item, AdmissionLease lease) {
        Objects.requireNonNull(lease, "lease");
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (inflight.get(item.requestId()) != entry || entry.cleanupOwned) {
                return false;
            }
            if (entry.externalFutureTerminalClaimed) {
                return false;
            }
            if (entry.admissionLease != null && entry.admissionLease != lease) {
                throw new IllegalStateException(
                        "admission lease already attached for request_id=" + item.requestId());
            }
            entry.admissionLease = lease;
            if (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED) {
                lease.markDecodeAccepted();
            }
            return true;
        }
    }

    @Override
    public void unregisterInflight(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        if (entry != null && entry.item == item) {
            synchronized (entry) {
                if (inflight.get(item.requestId()) == entry) {
                    if (entry.lifecycle.hasDeliveryClaim()
                            || entry.preemption != null || entry.engineFence != null) {
                        Logger.warn("Ignoring unregister after engine ownership boundary: "
                                        + "request_id={} lifecycle={} preemption={} engine_fence={}",
                                item.requestId(), entry.lifecycle.snapshot().state(),
                                entry.preemption != null, entry.engineFence != null);
                        return;
                    }
                    releasePrefillAccounting(entry);
                    inflight.remove(item.requestId(), entry);
                }
            }
        }
    }

    @Override
    public PostDeliveryFenceResult fenceAfterDeliveryTimeout(BatchItem item, String detail) {
        if (!tryAcquireDeliveryPermit()) {
            return PostDeliveryFenceResult.ALREADY_TERMINAL;
        }
        try {
            InflightEntry entry = entryFor(item);
            if (entry == null) {
                return PostDeliveryFenceResult.ALREADY_TERMINAL;
            }
            EngineFenceRegistration started;
            synchronized (entry) {
                if (inflight.get(item.requestId()) != entry || entry.cleanupOwned
                        || entry.lifecycle.isTerminal()) {
                    return PostDeliveryFenceResult.ALREADY_TERMINAL;
                }
                if (entry.engineFence != null) {
                    return PostDeliveryFenceResult.ALREADY_FENCED;
                }
                if (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED) {
                    return PostDeliveryFenceResult.ENGINE_OWNED;
                }
                if (entry.preemption != null) {
                    entry.preemption.postDeliveryFenceDetail = detail;
                    if (entry.preemption.state
                            != PreemptionRegistrationState.NOT_FOUND_STALE) {
                        return PostDeliveryFenceResult.JOINED_PREEMPTION;
                    }
                    started = transferNotFoundFenceLocked(entry, detail, false);
                    if (started == null) {
                        return entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                                ? PostDeliveryFenceResult.ENGINE_OWNED
                                : PostDeliveryFenceResult.JOINED_PREEMPTION;
                    }
                } else {
                    started = installEngineFenceLocked(
                            entry, EngineFenceCause.POST_DELIVERY_RECONCILIATION, detail);
                    if (started == null) {
                        return entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                                ? PostDeliveryFenceResult.ENGINE_OWNED
                                : PostDeliveryFenceResult.ALREADY_TERMINAL;
                    }
                }
            }
            reconcileEngineFence(entry, started, 0);
            return PostDeliveryFenceResult.STARTED;
        } finally {
            releaseDeliveryPermit();
        }
    }

    @Override
    public boolean reduceExternalFutureTerminal(BatchItem item, String detail) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            // The exact generation is already gone. The lease's fallback
            // release is idempotent and cannot conflict with a live claim.
            return false;
        }

        EngineFenceRegistration startedFence = null;
        InflightEntry localCleanup = null;
        synchronized (deliveryFence) {
            synchronized (entry) {
                if (inflight.get(item.requestId()) != entry) {
                    return false;
                }
                if (entry.cleanupOwned || entry.engineFence != null
                        || entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                        || entry.lifecycle.isTerminal()) {
                    return true;
                }

                if (entry.preemption != null) {
                    deferOrdinaryTerminalLocked(
                            entry, DeferredTerminal.admissionCleanup(detail));
                    return true;
                }

                RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
                boolean batchClaimed = snapshot.deliveryClaimKind()
                        == DeliveryClaimKind.BATCH_ENQUEUE;
                boolean batchSendStarted = batchClaimed
                        && entry.lifecycle.getBatchEnqueueStartedAtMs() > 0;
                boolean routePublicationClaimed = snapshot.deliveryClaimKind()
                        == DeliveryClaimKind.ROUTE_DECISION
                        && entry.responseCompletionClaimed;
                if (batchSendStarted || routePublicationClaimed) {
                    startedFence = installEngineFenceLocked(
                            entry,
                            batchSendStarted ? EngineFenceCause.BATCH_ACK_UNCERTAIN
                                    : EngineFenceCause.POST_DELIVERY_RECONCILIATION,
                            detail == null ? "external future terminal" : detail);
                    // A null registration means another authoritative owner
                    // won while entering this reducer. It still owns cleanup.
                    if (startedFence == null) {
                        return true;
                    }
                } else {
                    // Keep the generation registered until all request-id keyed
                    // endpoint releases finish. cleanupOwned blocks delivery,
                    // WorkerStatus and preemption without opening a request-id
                    // reuse window onto the old accounting.
                    entry.engineOwnershipState = EngineOwnershipState.TERMINAL;
                    entry.cleanupOwned = true;
                    localCleanup = entry;
                }
            }
        }

        if (localCleanup != null) {
            releaseLocalAdmissionCleanup(localCleanup, detail);
        }
        if (startedFence != null) {
            reconcileEngineFence(entry, startedFence, 0);
        }
        return true;
    }

    /** Release a locally reversible admission, then retire its exact generation. */
    private void releaseLocalAdmissionCleanup(InflightEntry entry, String detail) {
        try {
            releaseLocallyOwnedResources(entry, detail);
        } finally {
            removeInflightGeneration(entry);
        }
    }

    /** Release request-local queue and endpoint ledgers; every step is idempotent. */
    private void releaseLocallyOwnedResources(InflightEntry entry, String detail) {
        removeQueuedItem(entry, detail);
        try {
            rollbackOnce(entry);
        } catch (RuntimeException decodeFailure) {
            Logger.warn("Local cleanup could not release Decode reservation: "
                            + "request_id={} detail={}",
                    entry.item.requestId(), detail, decodeFailure);
        }
        try {
            releasePrefillAccounting(entry);
        } catch (RuntimeException prefillFailure) {
            Logger.warn("Local cleanup could not release Prefill accounting: "
                            + "request_id={} detail={}",
                    entry.item.requestId(), detail, prefillFailure);
        }
    }

    /** Remove the exact queued request; dispatch-won races are an idempotent no-op. */
    private void removeQueuedItem(InflightEntry entry, String detail) {
        PrefillEndpoint prefill = entry.item.prefillEp();
        try {
            if (prefill != null) {
                prefill.getBatcher().queueManager().tryRemove(
                        entry.item.requestId(), "TERMINAL_RELEASE");
            }
        } catch (RuntimeException queueFailure) {
            Logger.warn("Local cleanup could not remove queued request: "
                            + "request_id={} detail={}",
                    entry.item.requestId(), detail, queueFailure);
        }
    }

    /**
     * Terminate an evicted victim with {@link StrategyErrorType#PRIORITY_PREEMPTED}
     * (engine-accepted victims, contract 5.3): release its decode reservation,
     * complete its future and tombstone the request id. Mirrors
     * {@link #onOfferFailure} and is idempotent — the lifecycle transition,
     * rollback CAS and future completion each apply at most once (design doc 17.3).
     */
    @Override
    public void finishPreempted(BatchItem victim, String detail) {
        finishVictim(victim, StrategyErrorType.PRIORITY_PREEMPTED, detail);
    }

    /**
     * {@link #finishPreempted} by request id (victims whose BatchItem is not
     * at hand, design doc 11.5). A missing inflight entry means the request
     * already reached a terminal state — no-op, idempotent.
     */
    @Override
    public void finishPreemptedById(String requestId, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry != null) {
            finishPreempted(entry.item, detail);
        } else {
            // N1: a settle on an unknown id is harmless (already terminal or
            // never registered) but worth surfacing — a burst points at a
            // registration/cleanup race.
            Logger.debug("finishPreemptedById miss: request_id={} not inflight, detail={}",
                    requestId, detail);
            // P2-2: surface the miss as a metric too — warn logs alone are
            // not alertable.
            if (admissionScheduler != null) {
                admissionScheduler.onInflightSettleMiss("preempted");
            }
        }
    }

    /**
     * Terminate a yielded victim — one the engine never saw (prefill queue
     * eviction / decode reserved-only eviction, contract 5.3) — with the
     * retryable {@link StrategyErrorType#NO_AVAILABLE_WORKER}. Shares the
     * idempotent release/tombstone chain of {@link #finishPreempted}.
     */
    @Override
    public void finishYielded(BatchItem victim, String detail) {
        finishVictim(victim, StrategyErrorType.NO_AVAILABLE_WORKER, detail);
    }

    /**
     * {@link #finishYielded} by request id (decode reserved-only victims).
     * A missing inflight entry means the request already reached a terminal
     * state — no-op, idempotent.
     */
    @Override
    public void finishYieldedById(String requestId, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry != null) {
            finishYielded(entry.item, detail);
        } else {
            // N1: same rationale as finishPreemptedById — no-op, but observable.
            Logger.debug("finishYieldedById miss: request_id={} not inflight, detail={}",
                    requestId, detail);
            // P2-2: metric alongside the warn log.
            if (admissionScheduler != null) {
                admissionScheduler.onInflightSettleMiss("yielded");
            }
        }
    }

    /**
     * Shared victim terminal chain: rollback CAS, lifecycle fail, future
     * completion with the caller's terminal error type, tombstone. Each step
     * applies at most once regardless of repeats or terminal-path races.
     */
    private void finishVictim(BatchItem victim, StrategyErrorType errorType, String detail) {
        InflightEntry entry = entryFor(victim);
        if (entry != null) {
            ResponseCompletion publication;
            synchronized (entry) {
                publication = reduceOrdinaryTerminalLocked(entry,
                        DeferredTerminal.failure(errorType, detail, false));
            }
            submitResponseCompletion(publication);
        } else if (!victim.future().isDone() && !terminalStates.containsKey(victim.requestId())) {
            rollback(victim);
            completeError(victim.future(), errorType, detail);
        }
    }

    @Override
    public boolean claimForPreemption(String requestId, long attemptToken, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (inflight.get(requestId) != entry) {
                return false;
            }
            RequestLifecycleSnapshot lifecycle = entry.lifecycle.snapshot();
            if (entry.cleanupOwned || entry.preemption != null
                    || entry.engineFence != null || entry.lifecycle.isTerminal()
                    || entry.cancellationReason != null
                    || (lifecycle.deliveryClaimKind() == DeliveryClaimKind.ROUTE_DECISION
                        && lifecycle.state() == RequestLifecycleState.DISPATCHING)) {
                // A route decision is not engine-visible until the frontend
                // receives it. Canceling during publication can tombstone an
                // id which the same publication is about to enqueue, leaving
                // neither side able to make progress. ACKNOWLEDGED route
                // requests remain ordinary preemption candidates.
                return false;
            }
            entry.preemption = new PreemptionRegistration(attemptToken, detail);
            return true;
        }
    }

    @Override
    public boolean releasePreemptionClaim(String requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        EngineFenceRegistration started = null;
        ResponseCompletion publication = null;
        synchronized (entry) {
            if (inflight.get(requestId) != entry || !entry.hasPreemption(attemptToken)
                    || (entry.preemption.state != PreemptionRegistrationState.CLAIMED
                        && entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_IN_FLIGHT)) {
                return false;
            }
            PreemptionRegistration registration = entry.preemption;
            if (registration.postDeliveryFenceDetail != null) {
                if (entry.cancellationReason == null
                        && entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED) {
                    // The aborted priority attempt no longer owns Decode, and
                    // fresh Decode ownership makes an ordinary cleanup fence
                    // unnecessary. Replay only work cached behind the claim.
                    entry.preemption = null;
                    publication = replayAfterReleasedClaimLocked(entry, registration);
                } else {
                    started = entry.cancellationReason != null
                            ? installCancellationFenceLocked(
                                    entry, registration.postDeliveryFenceDetail,
                                    registration, 0)
                            : installEngineFenceLocked(
                                    entry,
                                    EngineFenceCause.POST_DELIVERY_RECONCILIATION,
                                    registration.postDeliveryFenceDetail,
                                    0, false, registration);
                    if (started == null) {
                        return false;
                    }
                }
            } else {
                entry.preemption = null;
                publication = replayAfterReleasedClaimLocked(entry, registration);
            }
        }
        submitResponseCompletion(publication);
        if (started != null) {
            reconcileEngineFence(entry, started, 0);
        }
        return true;
    }

    @Override
    public boolean markPreemptionCancelInFlight(String requestId, long attemptToken) {
        return transitionPreemption(requestId, attemptToken,
                PreemptionRegistrationState.CLAIMED,
                PreemptionRegistrationState.CANCEL_IN_FLIGHT, false);
    }

    @Override
    public boolean markPreemptionCancelAccepted(String requestId, long attemptToken) {
        return transitionPreemption(requestId, attemptToken,
                PreemptionRegistrationState.CANCEL_IN_FLIGHT,
                PreemptionRegistrationState.CANCEL_REQUESTED, true);
    }

    @Override
    public boolean markPreemptionNotFound(String requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        EngineFenceRegistration started = null;
        ResponseCompletion publication = null;
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)
                    || entry.preemption.state
                        != PreemptionRegistrationState.CANCEL_IN_FLIGHT) {
                return false;
            }
            entry.preemption.state = PreemptionRegistrationState.NOT_FOUND_STALE;
            if (entry.preemption.postDeliveryFenceDetail != null) {
                // NOT_FOUND is not a release proof, but the priority attempt
                // no longer owns an installed intent. Transfer only the
                // control claim; all victim accounting remains charged under
                // the new request-scoped Engine fence.
                started = transferNotFoundFenceLocked(
                        entry,
                        entry.preemption.postDeliveryFenceDetail,
                        entry.cancellationReason != null);
            }
            // A terminal delta is incremental and may never be sent again.
            // NOT_FOUND proves the priority intent was not installed, so the
            // first cached ordinary outcome resumes its original path.
            if (entry.hasPreemption(attemptToken)) {
                publication = replayAfterNegativeCancelLocked(
                        entry, attemptToken, false);
            }
        }
        submitResponseCompletion(publication);
        if (started != null) {
            reconcileEngineFence(entry, started, 0);
        }
        return true;
    }

    @Override
    public boolean markPreemptionUnknown(String requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        ResponseCompletion publication;
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)
                    || (entry.preemption.state != PreemptionRegistrationState.CANCEL_IN_FLIGHT
                        && entry.preemption.state != PreemptionRegistrationState.CANCEL_REQUESTED)) {
                return false;
            }
            entry.preemption.state = PreemptionRegistrationState.CANCEL_UNKNOWN;
            // UNKNOWN does not prove that the Cancel intent was rejected.
            // Only an authoritative worker terminal may disambiguate it;
            // local timeout/failure/lease cleanup must retain accounting for
            // a later typed CANCELED observation.
            publication = replayAfterNegativeCancelLocked(entry, attemptToken, true);
        }
        submitResponseCompletion(publication);
        return true;
    }

    @Override
    public CompletableFuture<PriorityCanceledObservation> priorityCanceledSignal(
            String requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return CompletableFuture.failedFuture(
                    new IllegalStateException("victim is not inflight: " + requestId));
        }
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)) {
                return CompletableFuture.failedFuture(
                        new IllegalStateException("stale preemption token for " + requestId));
            }
            return entry.preemption.priorityCanceled;
        }
    }

    @Override
    public boolean finishPreemptedById(String requestId, long attemptToken, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        ResponseCompletion publication;
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)
                    || (entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_REQUESTED
                        && entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_UNKNOWN)
                    || !entry.preemption.priorityCanceled.isDone()) {
                return false;
            }
            publication = settlePriorityEntryLocked(entry, detail);
        }
        submitResponseCompletion(publication);
        return true;
    }

    @Override
    public boolean finishTombstonedById(String requestId,
                                        long attemptToken,
                                        String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        ResponseCompletion publication;
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)
                    || (entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_IN_FLIGHT
                        && entry.preemption.state
                            != PreemptionRegistrationState.NOT_FOUND_STALE
                        && entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_UNKNOWN)) {
                return false;
            }
            publication = settlePriorityEntryLocked(entry, detail);
        }
        submitResponseCompletion(publication);
        return true;
    }

    /** Token ownership is validated by the caller; called with entry locked. */
    private ResponseCompletion settlePriorityEntryLocked(InflightEntry entry, String detail) {
        entry.preemption.state = PreemptionRegistrationState.SETTLED;
        entry.cleanupOwned = true;
        rollbackOnce(entry);
        // Priority terminal proofs may omit the original Prefill batch id.
        // The lifecycle still owns the exact dispatch generation, so retire
        // its delivery ledger explicitly before publishing the tombstone.
        releasePrefillAccounting(entry);
        RequestLifecycleSnapshot terminal = entry.lifecycle.cancel(detail);
        ResponseCompletion publication = errorPublicationLocked(
                entry, StrategyErrorType.PRIORITY_PREEMPTED, detail);
        finishEntry(entry, terminal);
        return publication;
    }

    @Override
    public boolean reconcilePreemptionActive(String requestId) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        EngineFenceRegistration started = null;
        synchronized (entry) {
            if (entry.preemption == null
                    || entry.preemption.state != PreemptionRegistrationState.NOT_FOUND_STALE) {
                return false;
            }
            PreemptionRegistration registration = entry.preemption;
            DecodeEndpoint endpoint = entry.item.decodeEp();
            if (registration.postDeliveryFenceDetail != null) {
                started = transferNotFoundFenceLocked(
                        entry,
                        registration.postDeliveryFenceDetail,
                        entry.cancellationReason != null);
                if (started == null && entry.preemption == registration) {
                    return false;
                }
            } else {
                if (endpoint != null
                        && !endpoint.reconcilePriorityVictimActive(requestId)) {
                    // A racing typed CANCELED or ordinary terminal may already
                    // own the endpoint CAS. Keep the scheduler token fence intact.
                    return false;
                }
                entry.preemption = null;
            }
        }
        if (started != null) {
            reconcileEngineFence(entry, started, 0);
        }
        return true;
    }

    private boolean transitionPreemption(String requestId, long attemptToken,
                                         PreemptionRegistrationState expected,
                                         PreemptionRegistrationState next,
                                         boolean updateLifecycle) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken) || entry.preemption.state != expected) {
                return false;
            }
            if (expected == PreemptionRegistrationState.CLAIMED
                    && entry.cancellationReason != null) {
                return false;
            }
            entry.preemption.state = next;
            if (updateLifecycle) {
                entry.lifecycle.requestCancel(entry.preemption.detail);
            }
            return true;
        }
    }

    @Override
    public EngineCancelChannel.CancelTarget resolveCancelTarget(String requestId) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return null;
        }
        ServerStatus prefill = entry.item.prefill();
        if (prefill == null) {
            return null;
        }
        return new EngineCancelChannel.CancelTarget(
                prefill.getServerIp(), prefill.getGrpcPort());
    }

    // ==================== External cancellation ====================

    /**
     * Cancel one request generation owned by this scheduler.
     *
     * <p>This method is the only reducer for the frontend-facing Cancel RPC.
     * Requests which have not crossed an external delivery boundary are
     * released locally.  Once an EnqueueBatch send has started, or a route
     * decision may have been published, the existing request-scoped Engine
     * fence owns reconciliation and resources remain charged until an
     * authoritative terminal is observed.</p>
     *
     * @return the current lifecycle for the matching request generation, or
     *         {@code null} when the request is unknown or {@code batchId}
     *         addresses a different generation
     */
    public RequestLifecycleSnapshot cancelRequest(String requestId,
                                                   long expectedBatchId,
                                                   CancelReason reason) {
        Objects.requireNonNull(reason, "reason");
        RequestGenerationGate generation = generationGates.get(requestId);
        if (generation != null) {
            ResponseCompletion generationCompletion = null;
            RequestLifecycleSnapshot generationResult = null;
            synchronized (generation) {
                // registerInflight uses this same gate. Exactly one side sees
                // an absent entry while the gate remains open.
                InflightEntry generationEntry = inflight.get(requestId);
                if (generationEntry == null) {
                    RequestLifecycleSnapshot existing =
                            matchingTerminalState(requestId, expectedBatchId);
                    if (existing != null) {
                        return existing;
                    }
                    if (expectedBatchId != 0) {
                        return null;
                    }
                    if (generation.pendingAdmissionCancellation != null) {
                        return generation.pendingAdmissionCancellation.snapshot();
                    }
                    if (!generation.closeCommits()) {
                        return null;
                    }
                    RequestLifecycle lifecycle = new RequestLifecycle(requestId);
                    generationResult = lifecycle.requestCancel(cancelDetail(reason));
                    if (generation.admissionMutationInProgress) {
                        generation.pendingAdmissionCancellation = lifecycle;
                        generation.pendingAdmissionCancelReason = reason;
                    } else {
                        generationResult = settleCancellationLifecycle(
                                lifecycle, reason, cancelDetail(reason));
                        terminalStates.put(requestId, generationResult);
                        generationCompletion = ResponseCompletion.terminal(
                                generation,
                                buildErrorResponse(
                                        cancelErrorType(reason,
                                                generation.deadlineErrorType),
                                        generationResult.detail()));
                    }
                } else if (generationEntry.item.future() == generation) {
                    // A stale generation-specific Cancel must not close the
                    // current generation. Once the generation matches,
                    // closing this gate prevents any later commit/offer edge.
                    synchronized (generationEntry) {
                        if (inflight.get(requestId) != generationEntry
                                || !batchMatches(
                                        generationEntry.lifecycle.snapshot(),
                                        expectedBatchId)) {
                            return null;
                        }
                        generation.closeCommits();
                    }
                }
            }
            if (generationCompletion != null) {
                submitResponseCompletion(generationCompletion);
            }
            if (generationResult != null) {
                return generationResult;
            }
        }

        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return matchingTerminalState(requestId, expectedBatchId);
        }

        EngineFenceRegistration startedFence = null;
        ResponseCompletion confirmedDelivery = null;
        boolean finishLocally = false;
        RequestLifecycleSnapshot result;
        synchronized (deliveryFence) {
            synchronized (entry) {
                if (inflight.get(requestId) != entry) {
                    return matchingTerminalState(requestId, expectedBatchId);
                }
                RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
                if (!batchMatches(current, expectedBatchId)) {
                    return null;
                }
                if (current.state().isTerminal() || entry.cancellationReason != null) {
                    return current;
                }
                if (entry.cleanupOwned) {
                    // Another terminal reducer already owns cleanup.  Do not
                    // relabel its outcome or claim that this cancel won.
                    return current;
                }
                if (reason == CancelReason.DEADLINE_EXCEEDED
                        && entry.responseCompletionClaimed) {
                    // Delivery already owns the public response even when its
                    // asynchronous publication has not run yet. The response
                    // claim, rather than future.isDone(), is the deadline
                    // linearization point. A later client Cancel remains
                    // allowed to cancel the acknowledged running request.
                    return current;
                }

                if (entry.preemption != null
                        && entry.preemption.state != PreemptionRegistrationState.CLAIMED
                        && entry.preemption.state
                            != PreemptionRegistrationState.NOT_FOUND_STALE) {
                    // Priority cancellation already crossed its engine-facing
                    // linearization point.  It remains the first-cause owner;
                    // returning its current lifecycle avoids misattributing a
                    // later cancellation as REQUEST_CANCELLED.
                    return current;
                }

                if (reason == CancelReason.DEADLINE_EXCEEDED
                        && entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                        && entry.engineFence == null) {
                    // Decode acceptance is authoritative evidence that the
                    // scheduling decision crossed its delivery boundary before
                    // the admission timer. Publish that already-won success;
                    // a client-initiated Cancel remains allowed to cancel the
                    // running request through the ordinary fence path.
                    if (entry.preemption == null) {
                        confirmedDelivery = confirmDeliveryLocked(
                                entry, current.batchId());
                    } else {
                        PreemptionRegistration registration = entry.preemption;
                        retainDeliveryConfirmation(registration, current.batchId());
                        if (registration.state
                                == PreemptionRegistrationState.NOT_FOUND_STALE) {
                            confirmedDelivery = replayAfterNegativeCancelLocked(
                                    entry, registration.attemptToken, false);
                        }
                    }
                    result = entry.lifecycle.snapshot();
                } else if (entry.preemption != null
                        && entry.preemption.state
                            == PreemptionRegistrationState.NOT_FOUND_STALE) {
                    startedFence = transferNotFoundFenceLocked(
                            entry, cancelDetail(reason), true);
                    if (startedFence == null) {
                        return current;
                    }
                    entry.cancellationReason = reason;
                    result = entry.lifecycle.requestCancel(cancelDetail(reason));
                } else if (entry.preemption != null) {
                    entry.cancellationReason = reason;
                    result = entry.lifecycle.requestCancel(cancelDetail(reason));
                    // The client won before the priority attempt sent Cancel.
                    // Force that attempt to abort; releasePreemptionClaim then
                    // transfers the exact entry to the common Engine fence.
                    entry.preemption.postDeliveryFenceDetail = cancelDetail(reason);
                } else if (entry.engineFence != null) {
                    entry.cancellationReason = reason;
                    result = entry.lifecycle.requestCancel(cancelDetail(reason));
                    // A delivery-timeout fence already owns this generation.
                    // The cancellation cause is retained on the entry and will
                    // determine the terminal state when that fence settles.
                } else if (isLocallyReversible(entry, current)) {
                    entry.cancellationReason = reason;
                    result = entry.lifecycle.requestCancel(cancelDetail(reason));
                    entry.engineOwnershipState = EngineOwnershipState.TERMINAL;
                    entry.cleanupOwned = true;
                    finishLocally = true;
                } else {
                    EngineFenceCause cause = cancelFenceCause(entry, current);
                    if (cause == EngineFenceCause.BATCH_ACK_UNCERTAIN) {
                        // Protect the exact Prefill batch member before the API
                        // claims acceptance. A null result proves settlement
                        // won the ledger race, so its current lifecycle is the
                        // only honest response and no engine Cancel is sent.
                        startedFence = installCancellationFenceLocked(
                                entry, cancelDetail(reason));
                        if (startedFence == null) {
                            return current;
                        }
                    }
                    entry.cancellationReason = reason;
                    result = entry.lifecycle.requestCancel(cancelDetail(reason));
                    if (startedFence == null) {
                        startedFence = installCancellationFenceLocked(
                                entry, cancelDetail(reason));
                    }
                    if (startedFence == null) {
                        throw new IllegalStateException(
                                "cancel owner could not be installed for request " + requestId);
                    }
                }
            }
        }

        if (finishLocally) {
            return finishLocalCancellation(entry);
        }
        if (confirmedDelivery != null) {
            submitResponseCompletion(confirmedDelivery);
        }
        if (startedFence != null) {
            reconcileEngineFence(entry, startedFence, 0);
        }
        return result;
    }

    private RequestLifecycleSnapshot matchingTerminalState(String requestId,
                                                            long expectedBatchId) {
        RequestLifecycleSnapshot terminal = terminalStates.get(requestId);
        return batchMatches(terminal, expectedBatchId) ? terminal : null;
    }

    /** Called with both delivery and entry ownership held. */
    private static boolean isLocallyReversible(InflightEntry entry,
                                               RequestLifecycleSnapshot snapshot) {
        return switch (snapshot.deliveryClaimKind()) {
            case NONE -> true;
            case BATCH_ENQUEUE -> entry.lifecycle.getBatchEnqueueStartedAtMs() == 0;
            case ROUTE_DECISION -> !entry.responseCompletionClaimed;
        };
    }

    /** Select the ownership fence at the exact delivery-confirmation boundary. */
    private static EngineFenceCause cancelFenceCause(
            InflightEntry entry,
            RequestLifecycleSnapshot snapshot) {
        if (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED) {
            return EngineFenceCause.POST_DELIVERY_RECONCILIATION;
        }
        boolean batchAckUncertain = snapshot.deliveryClaimKind()
                == DeliveryClaimKind.BATCH_ENQUEUE
                && snapshot.state() == RequestLifecycleState.DISPATCHING
                && entry.lifecycle.getBatchEnqueueStartedAtMs() > 0;
        return batchAckUncertain
                ? EngineFenceCause.BATCH_ACK_UNCERTAIN
                : EngineFenceCause.POST_DELIVERY_RECONCILIATION;
    }

    /** Install the one fence policy shared by direct Cancel and claim handoff. */
    private EngineFenceRegistration installCancellationFenceLocked(
            InflightEntry entry,
            String detail) {
        return installCancellationFenceLocked(entry, detail, null, 0);
    }

    /**
     * Called with the request entry locked. When {@code transferredPreemption}
     * is non-null, ownership moves only after every fence resource has been
     * acquired; a failed installation leaves the old claim intact.
     */
    private EngineFenceRegistration installCancellationFenceLocked(
            InflightEntry entry,
            String detail,
            PreemptionRegistration transferredPreemption,
            long transferredPreemptionToken) {
        EngineFenceRegistration prepared = prepareEngineFenceLocked(
                entry,
                cancelFenceCause(entry, entry.lifecycle.snapshot()),
                detail,
                transferredPreemptionToken,
                true,
                transferredPreemption);
        return commitPreparedEngineFenceLocked(
                entry, prepared, transferredPreemption);
    }

    /** Prepare cancellation resources without changing either live owner. */
    private EngineFenceRegistration prepareCancellationFenceLocked(
            InflightEntry entry,
            String detail,
            PreemptionRegistration transferredPreemption,
            long transferredPreemptionToken) {
        return prepareEngineFenceLocked(
                entry,
                cancelFenceCause(entry, entry.lifecycle.snapshot()),
                detail,
                transferredPreemptionToken,
                true,
                transferredPreemption);
    }

    /**
     * Transfer a priority NOT_FOUND claim to the request-scoped fence. Fence
     * resources are prepared before the endpoint CAS; the scheduler owner is
     * swapped only after that CAS succeeds.
     */
    private EngineFenceRegistration transferNotFoundFenceLocked(
            InflightEntry entry,
            String detail,
            boolean externalCancellation) {
        PreemptionRegistration registration = entry.preemption;
        if (registration == null
                || registration.state != PreemptionRegistrationState.NOT_FOUND_STALE) {
            return null;
        }
        DecodeEndpoint endpoint = entry.item.decodeEp();
        if (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                && !externalCancellation) {
            if (endpoint == null
                    || endpoint.reconcilePriorityVictimActive(entry.item.requestId())) {
                entry.preemption = null;
            }
            return null;
        }

        long transferredToken = entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                || endpoint == null ? 0 : registration.attemptToken;
        EngineFenceRegistration prepared = externalCancellation
                ? prepareCancellationFenceLocked(
                        entry, detail, registration, transferredToken)
                : prepareEngineFenceLocked(
                        entry, EngineFenceCause.POST_DELIVERY_RECONCILIATION,
                        detail, transferredToken, false, registration);
        if (prepared == null) {
            return null;
        }

        boolean endpointTransferred = endpoint == null
                || (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                    ? endpoint.reconcilePriorityVictimActive(entry.item.requestId())
                    : endpoint.transferPriorityNotFoundClaimToEngineFence(
                            registration.attemptToken, entry.item.requestId()));
        if (!endpointTransferred) {
            discardPreparedEngineFence(prepared);
            return null;
        }
        return commitPreparedEngineFenceLocked(entry, prepared, registration);
    }

    /**
     * Release a locally reversible cancellation without retaining the global
     * delivery fence across endpoint/queue cleanup.
     */
    private RequestLifecycleSnapshot finishLocalCancellation(InflightEntry entry) {
        releaseLocallyOwnedResources(entry, cancelDetail(entry.cancellationReason));

        ResponseCompletion publication = null;
        RequestLifecycleSnapshot terminal;
        synchronized (entry) {
            if (inflight.get(entry.item.requestId()) != entry) {
                return matchingTerminalState(entry.item.requestId(),
                        entry.lifecycle.snapshot().batchId());
            }
            terminal = settleCancellationLifecycle(
                    entry.lifecycle,
                    entry.cancellationReason,
                    cancelDetail(entry.cancellationReason));
            publication = errorPublicationLocked(entry,
                    cancelErrorType(entry.cancellationReason,
                            entry.deadlineErrorType), terminal.detail());
            finishEntry(entry, terminal);
        }
        submitResponseCompletion(publication);
        return terminal;
    }

    /** Called with the request entry locked after an authoritative engine terminal. */
    private ResponseCompletion settleCancellationLocked(InflightEntry entry,
                                                        String proof) {
        releaseLocallyOwnedResources(entry, proof);
        RequestLifecycleSnapshot terminal = settleCancellationLifecycle(
                entry.lifecycle,
                entry.cancellationReason,
                cancelDetail(entry.cancellationReason) + "; " + proof);
        ResponseCompletion publication = errorPublicationLocked(entry,
                cancelErrorType(entry.cancellationReason,
                        entry.deadlineErrorType), terminal.detail());
        finishEntry(entry, terminal);
        return publication;
    }

    /** Called only after local rollback or authoritative engine settlement. */
    private static RequestLifecycleSnapshot settleCancellationLifecycle(
            RequestLifecycle lifecycle,
            CancelReason reason,
            String detail) {
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? lifecycle.timeout(detail)
                : lifecycle.cancel(detail);
    }

    private static String cancelDetail(CancelReason reason) {
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? "request deadline exceeded"
                : "request cancelled by client";
    }

    private static StrategyErrorType cancelErrorType(
            CancelReason reason,
            StrategyErrorType deadlineErrorType) {
        return reason == CancelReason.DEADLINE_EXCEEDED
                ? deadlineErrorType
                : StrategyErrorType.REQUEST_CANCELLED;
    }

    // ==================== Completion from worker status ====================

    public void onWorkerStatusUpdate(WorkerStatusResponse response) {
        if (response == null) {
            return;
        }
        boolean isPrefill = response.getRole() == RoleType.PREFILL;
        boolean isDecode = response.getRole() == RoleType.DECODE;

        // Decode KV_ALLOCATED/RUNNING is the authoritative acceptance signal
        // for the post-success lease.  Close the lease immediately instead of
        // retaining one active slot until its 30s fallback timer.  RECEIVED is
        // deliberately excluded: the engine has seen the request but has not
        // yet accepted Decode ownership/KV.
        if (isDecode && response.getRunningTaskInfo() != null) {
            for (TaskInfo task : response.getRunningTaskInfo().values()) {
                TaskPhase phase = task.getPhase();
                if (phase == TaskPhase.KV_ALLOCATED || phase == TaskPhase.RUNNING) {
                    markDecodeAccepted(task.getRequestId());
                }
            }
        }

        // NOT_FOUND_STALE is reopened only by a fresh active observation from
        // the original Prefill control owner.
        if (isPrefill && response.getRunningTaskInfo() != null) {
            for (TaskInfo task : response.getRunningTaskInfo().values()) {
                reconcilePreemptionActive(task.getRequestId());
            }
        }

        Map<String, TaskInfo> finishedTaskInfo = response.getFinishedTaskInfo();
        if (finishedTaskInfo == null || finishedTaskInfo.isEmpty()) {
            return;
        }

        for (TaskInfo task : finishedTaskInfo.values()) {
            String requestId = task.getRequestId();
            WorkerTerminalObservation observation = new WorkerTerminalObservation(
                    isPrefill, task.getBatchId(), task.getErrorCode());

            InflightEntry entry = inflight.get(requestId);
            if (entry == null) {
                continue;
            }
            WorkerStatusPublication publication;
            synchronized (deliveryFence) {
              synchronized (entry) {
                publication = reduceWorkerStatusLocked(
                        entry, task, observation, isPrefill, isDecode);
              }
            }
            submitResponseCompletion(publication.completion());
            publishPriorityCanceled(publication);
        }
    }

    /** Reduce one worker delta without invoking any CompletableFuture continuation. */
    private WorkerStatusPublication reduceWorkerStatusLocked(
            InflightEntry entry,
            TaskInfo task,
            WorkerTerminalObservation observation,
            boolean isPrefill,
            boolean isDecode) {
        String requestId = task.getRequestId();
        if (entry.engineFence != null && entry.cancellationReason != null) {
            RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
            boolean generationMatches = isDecode
                    || snapshot.deliveryClaimKind() == DeliveryClaimKind.ROUTE_DECISION
                    || task.getBatchId() == snapshot.batchId();
            if (generationMatches && observation.isTerminal()) {
                return WorkerStatusPublication.completion(
                        settleCancellationLocked(entry,
                                "engine terminal observed after cancellation"));
            }
            Logger.debug("Ignoring non-terminal or stale worker update during cancellation: "
                            + "request_id={} role={} task_batch_id={} entry_batch_id={} "
                            + "error_code={}",
                    requestId, isDecode ? RoleType.DECODE : RoleType.PREFILL,
                    task.getBatchId(), snapshot.batchId(), task.getErrorCode());
            return WorkerStatusPublication.NONE;
        }
        if (isPrefill && entry.engineFence != null) {
            RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
            boolean generationMatches = snapshot.deliveryClaimKind() == DeliveryClaimKind.ROUTE_DECISION
                    || task.getBatchId() == snapshot.batchId();
            boolean authoritativeCanceled = generationMatches
                    && task.getPriorityPreemptionProgress()
                        == PriorityPreemptionProgress.CANCELED
                    && task.getErrorCode() == ENGINE_ERROR_PRIORITY_PREEMPTED;
            if (authoritativeCanceled) {
                // Only the typed cancel terminal for the exact dispatch
                // generation proves that ambiguous Engine ownership settled.
                return WorkerStatusPublication.completion(settleEngineFenceLocked(
                        entry, "Engine fence reconciled by typed Prefill CANCELED"));
            }
            Logger.debug("Ignoring non-authoritative Prefill terminal during "
                            + "Engine fence reconciliation: request_id={} "
                            + "task_batch_id={} entry_batch_id={} error_code={} "
                            + "preemption_progress={}",
                    requestId, task.getBatchId(), snapshot.batchId(),
                    task.getErrorCode(), task.getPriorityPreemptionProgress());
            return WorkerStatusPublication.NONE;
        }
        if (entry.preemption != null) {
            boolean authoritativeCanceled = isPrefill
                    && task.getPriorityPreemptionProgress()
                        == PriorityPreemptionProgress.CANCELED
                    && task.getErrorCode() == ENGINE_ERROR_PRIORITY_PREEMPTED;
            if (authoritativeCanceled) {
                // Capture the exact signal generation under the entry fence;
                // its coordinator continuation is published after all locks.
                return WorkerStatusPublication.priorityCanceled(
                        entry.preemption.priorityCanceled,
                        new PriorityCanceledObservation(requestId, task.getErrorCode()));
            }
        }
        if (isDecode) {
            // A Decode terminal proves acceptance even when no active phase was
            // reported. Ownership and terminal state remain one locked reduce.
            markDecodeTerminalLocked(entry);
        }
        // Successful Prefill completion is not the end of a P/D request.
        if (!observation.isTerminal()) {
            return WorkerStatusPublication.NONE;
        }
        return WorkerStatusPublication.completion(reduceOrdinaryTerminalLocked(
                entry, DeferredTerminal.worker(observation)));
    }

    private void publishPriorityCanceled(WorkerStatusPublication publication) {
        if (publication.priorityCanceledSignal() == null) {
            return;
        }
        Runnable completion = () -> publication.priorityCanceledSignal()
                .complete(publication.priorityCanceledObservation());
        executeResponseTask(completion);
    }

    private void markDecodeAccepted(String requestId) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return;
        }
        ResponseCompletion publication = null;
        synchronized (deliveryFence) {
            synchronized (entry) {
                if (inflight.get(requestId) != entry) {
                    return;
                }
                EngineFenceRegistration fence = entry.engineFence;
                boolean acceptedBeforeCancel = markDecodeAcceptedLocked(entry);
                if (fence != null && acceptedBeforeCancel) {
                    // Decode KV ownership is stronger than a missing Prefill
                    // Enqueue ACK. Stop the Prefill cancel-fence retry chain
                    // and publish the logical ACK while both ownership paths
                    // are linearized by the same delivery fence.
                    publication = confirmDeliveryLocked(
                            entry, entry.lifecycle.snapshot().batchId());
                }
            }
        }
        submitResponseCompletion(publication);
    }

    /** Record authoritative Decode ownership. Called with {@code entry} locked. */
    private boolean markDecodeAcceptedLocked(InflightEntry entry) {
        if (entry.cleanupOwned
                || entry.engineOwnershipState == EngineOwnershipState.TERMINAL) {
            return false;
        }
        entry.engineOwnershipState = EngineOwnershipState.DECODE_OWNED;
        EngineFenceRegistration fence = entry.engineFence;
        if (fence != null && entry.cancellationReason != null) {
            // An accepted cancellation is authoritative regardless of a
            // racing active sample.  Keep its fence until a terminal proof.
            return false;
        }
        if (fence != null && fence.cancelMayHaveBeenInstalled()) {
            // Once Cancel has crossed its invocation boundary, a later active
            // sample cannot prove that the intent was not installed. Keep the
            // fence until typed CANCELED, TOMBSTONED, or Decode terminal.
            return false;
        }
        if (fence != null) {
            releaseTransferredDecodeFenceActive(entry, fence);
            clearEngineFenceLocked(entry, fence);
        }
        if (entry.admissionLease != null) {
            entry.admissionLease.markDecodeAccepted();
        }
        return true;
    }

    /** A Decode terminal is authoritative even after Cancel was invoked. */
    private void markDecodeTerminalLocked(InflightEntry entry) {
        markDecodeAcceptedLocked(entry);
        settleTransferredDecodeFence(entry, entry.engineFence);
        if (entry.admissionLease != null) {
            entry.admissionLease.markDecodeAccepted();
        }
    }

    /**
     * Single reducer for every non-priority terminal. A live preemption claim
     * owns Decode accounting, so the first real ordinary outcome is retained
     * instead of rolling back/unregistering underneath the Cancel protocol.
     */
    private ResponseCompletion reduceOrdinaryTerminalLocked(InflightEntry entry,
                                                             DeferredTerminal terminal) {
        if (inflight.get(entry.item.requestId()) != entry) {
            return null;
        }
        if (entry.cleanupOwned) {
            return null;
        }
        if (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                && terminal.deliveryFailure()) {
            // KV_ALLOCATED/RUNNING is a stronger ownership observation than
            // an absent/failed Enqueue ACK. Preserve the live inflight entry,
            // Decode accounting, and public schedule success.
            if (entry.preemption != null) {
                PreemptionRegistration registration = entry.preemption;
                long batchId = entry.lifecycle.snapshot().batchId();
                retainDeliveryConfirmation(registration, batchId);
            } else {
                return confirmDeliveryLocked(
                        entry, entry.lifecycle.snapshot().batchId());
            }
            return null;
        }
        if (entry.preemption != null) {
            return deferOrdinaryTerminalLocked(entry, terminal);
        }
        entry.engineOwnershipState = EngineOwnershipState.TERMINAL;
        entry.cleanupOwned = true;
        return applyOrdinaryTerminalLocked(entry, terminal);
    }

    /** Called with {@code entry} locked. */
    private ResponseCompletion deferOrdinaryTerminalLocked(InflightEntry entry,
                                                            DeferredTerminal terminal) {
        PreemptionRegistration registration = entry.preemption;
        if (registration == null) {
            entry.cleanupOwned = true;
            return applyOrdinaryTerminalLocked(entry, terminal);
        }
        if (registration.pendingTerminal == null
                || (!registration.pendingTerminal.authoritativeWorker()
                    && terminal.authoritativeWorker())) {
            registration.pendingTerminal = terminal;
        }
        if (registration.state == PreemptionRegistrationState.NOT_FOUND_STALE) {
            return replayAfterNegativeCancelLocked(
                    entry, registration.attemptToken, false);
        } else if (registration.state == PreemptionRegistrationState.CANCEL_UNKNOWN) {
            return replayAfterNegativeCancelLocked(
                    entry, registration.attemptToken, true);
        }
        return null;
    }

    /** Retain one idempotent delivery acknowledgement while Cancel owns the entry. */
    private static void retainDeliveryConfirmation(PreemptionRegistration registration,
                                                  long batchId) {
        if (!registration.pendingDeliveryConfirmation) {
            registration.pendingDeliveryConfirmation = true;
            registration.pendingConfirmationBatchId = batchId;
        }
    }

    /**
     * Replay work cached while a claim existed when that claim is rolled back
     * before any Cancel RPC. Endpoint ownership has already been aborted (or
     * was never installed), so no endpoint reconciliation is needed here.
     */
    private ResponseCompletion replayAfterReleasedClaimLocked(
            InflightEntry entry,
            PreemptionRegistration registration) {
        if (registration.pendingTerminal != null) {
            entry.cleanupOwned = true;
            return applyOrdinaryTerminalLocked(entry, registration.pendingTerminal);
        } else if (registration.pendingDeliveryConfirmation) {
            return confirmDeliveryLocked(
                    entry, registration.pendingConfirmationBatchId);
        }
        return null;
    }

    /**
     * Resolve a cached outcome after NOT_FOUND or transport UNKNOWN. The
     * Decode endpoint CAS is the winner selection against a racing typed
     * CANCELED settlement. If typed CANCELED won first, keep the scheduler
     * registration intact for its token-fenced continuation.
     */
    private ResponseCompletion replayAfterNegativeCancelLocked(
            InflightEntry entry,
            long attemptToken,
            boolean transportUnknown) {
        if (!entry.hasPreemption(attemptToken)) {
            return null;
        }
        PreemptionRegistration registration = entry.preemption;
        DeferredTerminal pending = registration.pendingTerminal;
        if (pending != null) {
            // NOT_FOUND proves no Cancel intent was installed, so every
            // cached terminal can resume. Transport UNKNOWN does not; replay
            // only a terminal observed authoritatively from worker status.
            if (transportUnknown && !pending.authoritativeWorker()) {
                return null;
            }
            DecodeEndpoint endpoint = entry.item.decodeEp();
            boolean ordinaryWon = endpoint == null
                    || endpoint.reconcilePriorityVictimFinished(entry.item.requestId());
            if (!ordinaryWon) {
                return null;
            }
            entry.preemption = null;
            entry.cleanupOwned = true;
            return applyOrdinaryTerminalLocked(entry, pending);
        }

        // A concrete NOT_FOUND plus the delayed EnqueueBatch success proves
        // the request is active and the Cancel intent was not installed.
        // Transport UNKNOWN cannot make that assertion, so it keeps waiting
        // for typed CANCELED or an actual ordinary terminal.
        if (!transportUnknown && registration.pendingDeliveryConfirmation) {
            DecodeEndpoint endpoint = entry.item.decodeEp();
            boolean activeWon = endpoint == null
                    || endpoint.reconcilePriorityVictimActive(entry.item.requestId());
            if (!activeWon) {
                return null;
            }
            long batchId = registration.pendingConfirmationBatchId;
            entry.preemption = null;
            return confirmDeliveryLocked(entry, batchId);
        }
        return null;
    }

    /** Apply an already-owned ordinary outcome. Called with {@code entry} locked. */
    private ResponseCompletion applyOrdinaryTerminalLocked(InflightEntry entry,
                                                            DeferredTerminal terminal) {
        switch (terminal.kind()) {
            case ADMISSION_CLEANUP -> {
                PrefillEndpoint prefill = entry.item.prefillEp();
                if (prefill != null) {
                    prefill.getBatcher().queueManager().tryRemove(
                            entry.item.requestId(), "LEASE_RELEASE");
                }
                rollbackOnce(entry);
                releasePrefillAccounting(entry);
                removeInflightGeneration(entry);
                return null;
            }
            case FAILURE -> {
                // A failure can escape after priority scheduling registration/offer
                // (for example from telemetry or timer setup). Remove any
                // still-queued item before retiring the generation.
                removeQueuedItem(entry, terminal.detail());
                rollbackOnce(entry);
                if (terminal.releasePrefillAccounting()) {
                    releasePrefillAccounting(entry);
                }
                RequestLifecycleSnapshot failed = entry.lifecycle.fail(terminal.detail());
                ResponseCompletion publication = errorPublicationLocked(
                        entry, terminal.errorType(), terminal.detail());
                finishEntry(entry, failed);
                return publication;
            }
            case TIMEOUT -> {
                return timeoutEntry(entry, terminal.detail());
            }
            case WORKER -> {
                return applyWorkerTerminalLocked(entry, terminal.workerObservation());
            }
        }
        throw new IllegalStateException("Unhandled terminal kind " + terminal.kind());
    }

    /** Existing ordinary worker-terminal semantics, called with entry locked. */
    private ResponseCompletion applyWorkerTerminalLocked(
            InflightEntry entry,
            WorkerTerminalObservation observation) {
        String requestId = entry.item.requestId();
        RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
        // Decode workers do not carry a reliable Prefill batch id.
        if (current.deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE
                && observation.batchId() >= 0
                && observation.batchId() != current.batchId()) {
            Logger.warn("Worker completion batchId mismatch: "
                            + "request_id={} task_batch_id={} entry_batch_id={} is_prefill={}",
                    requestId, observation.batchId(), current.batchId(), observation.prefill());
            if (observation.prefill()) {
                return null;
            }
        }
        RequestLifecycleSnapshot terminal;
        ResponseCompletion publication;
        if (observation.errorCode() == 0) {
            terminal = entry.lifecycle.complete("decode completed");
            publication = responsePublicationLocked(
                    entry, buildSuccessResponse(entry.item));
        } else {
            terminal = entry.lifecycle.fail("worker error code " + observation.errorCode());
            publication = errorPublicationLocked(
                    entry, StrategyErrorType.WORKER_EXECUTION_FAILED,
                    "worker error code " + observation.errorCode());
        }
        if (observation.prefill()
                || current.deliveryClaimKind() == DeliveryClaimKind.ROUTE_DECISION) {
            rollbackOnce(entry);
            releasePrefillAccounting(entry);
        }
        finishEntry(entry, terminal);
        return publication;
    }

    public int getInflightSize() {
        return inflight.size();
    }

    /**
     * Current number of requests waiting in the canonical per-Prefill
     * scheduler queues. This is the operational queue-depth view for every
     * QUEUE ordering/dispatcher combination.
     */
    public int getQueuedRequestCount() {
        long queued = 0;
        for (PrefillEndpoint endpoint : endpointRegistry.getPrefillEndpoints().values()) {
            queued += endpoint.getBatcher().queueSize();
            if (queued >= Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
        }
        return (int) queued;
    }

    /**
     * Weakly-consistent immutable view of all scheduler-owned live request
     * lifecycles. The inflight map is authoritative; no diagnostic-only
     * shadow queue is maintained.
     */
    public List<RequestLifecycleSnapshot> snapshotActiveRequests() {
        List<RequestLifecycleSnapshot> snapshots = new ArrayList<>(inflight.size());
        for (Map.Entry<String, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            synchronized (entry) {
                if (inflight.get(candidate.getKey()) == entry) {
                    snapshots.add(entry.lifecycle.snapshot());
                }
            }
        }
        snapshots.sort((left, right) -> {
            int createdOrder = Long.compare(left.createdAtMs(), right.createdAtMs());
            return createdOrder != 0
                    ? createdOrder : left.requestId().compareTo(right.requestId());
        });
        return List.copyOf(snapshots);
    }

    /** Package-visible exact-generation retention diagnostic for leak tests. */
    int generationGateCount() {
        return generationGates.size();
    }

    /** Package-visible retained-ref diagnostic for deterministic leak tests. */
    int quarantinedProbeQueueSize() {
        return quarantinedProbeQueue.size();
    }

    /** Production RTP-LLM raw {@code ErrorCode::PRIORITY_PREEMPTED}. */
    private static final long ENGINE_ERROR_PRIORITY_PREEMPTED = 8429;

    /** Victim's decode endpoint key for the settle metric; "unknown" when absent. */
    private static String decodeEndpointKey(BatchItem item) {
        return item.decodeEp() != null ? item.decodeEp().ipPort() : "unknown";
    }

    public RequestLifecycleSnapshot getRequestState(String requestId,
                                                    long expectedBatchId) {
        RequestGenerationGate generation = generationGates.get(requestId);
        if (generation != null) {
            synchronized (generation) {
                InflightEntry entry = inflight.get(requestId);
                RequestLifecycleSnapshot live = entry != null
                        && entry.item.future() == generation
                        ? entry.lifecycle.snapshot()
                        : generation.pendingAdmissionCancellation != null
                            ? generation.pendingAdmissionCancellation.snapshot()
                            : null;
                if (batchMatches(live, expectedBatchId)) {
                    return live;
                }
            }
        }
        InflightEntry entry = inflight.get(requestId);
        RequestLifecycleSnapshot snapshot = entry != null
                ? entry.lifecycle.snapshot()
                : terminalStates.get(requestId);
        return batchMatches(snapshot, expectedBatchId) ? snapshot : null;
    }

    // ==================== Inflight TTL cleanup ====================

    @Scheduled(fixedRate = 60000L)
    public void cleanupInflight() {
        if (shuttingDown.get()) {
            return;
        }
        FlexlbConfig config = configService.loadBalanceConfig();
        if (!config.isQueue()) {
            return;
        }
        long ttlMs = config.queueScheduler().getLifecycle().getStaleInflightTimeoutMs();
        long now = System.currentTimeMillis();
        int expiredCount = 0;
        long oldestExpiredAgeMs = 0;
        List<String> expiredRequestSamples = new ArrayList<>(3);
        for (Map.Entry<String, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            long ageMs = now - entry.createdAtMs();
            if (ageMs <= ttlMs) {
                continue;
            }
            ResponseCompletion publication = null;
            EngineFenceRegistration startedFence = null;
            boolean expired = false;
            synchronized (deliveryFence) {
                synchronized (entry) {
                    if (inflight.get(candidate.getKey()) != entry) {
                        continue;
                    }
                    if (entry.preemption != null || entry.engineFence != null
                            || entry.cleanupOwned || entry.cancellationReason != null
                            || entry.lifecycle.isTerminal()) {
                        // Cancel ambiguity is reconciled by token/WorkerStatus;
                        // a concurrent cleanup owner is likewise already settling
                        // the entry and must not be raced by TTL.
                        continue;
                    }
                    RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
                    if (entry.engineOwnershipState != EngineOwnershipState.DECODE_OWNED
                            && isLocallyReversible(entry, snapshot)) {
                        publication = timeoutEntry(entry, "inflight TTL expired");
                    } else {
                        // The request may already be visible to Prefill/Decode.
                        // Install the same request-scoped cancellation fence as
                        // public Cancel and retain both ledgers until an
                        // authoritative Engine terminal settles the timeout.
                        startedFence = installCancellationFenceLocked(
                                entry, "inflight TTL expired");
                        if (startedFence == null) {
                            continue;
                        }
                        entry.cancellationReason = CancelReason.DEADLINE_EXCEEDED;
                        entry.lifecycle.requestCancel("inflight TTL expired");
                    }
                    oldestExpiredAgeMs = Math.max(oldestExpiredAgeMs, ageMs);
                    if (expiredRequestSamples.size() < 3) {
                        expiredRequestSamples.add(candidate.getKey());
                    }
                    expired = true;
                }
            }
            submitResponseCompletion(publication);
            if (startedFence != null) {
                reconcileEngineFence(entry, startedFence, 0);
            }
            if (expired) {
                expiredCount++;
            }
        }
        if (expiredCount > 0) {
            reporter.reportInflightTtlExpired(expiredCount);
            Logger.info("event=scheduler_inflight_ttl_eviction evicted={} "
                            + "oldest_age_ms={} ttl_ms={} request_samples={}",
                    expiredCount, oldestExpiredAgeMs, ttlMs, expiredRequestSamples);
        }
        probeQuarantinedEngineFences(now);
        long cutoff = System.currentTimeMillis() - ttlMs;
        terminalStates.entrySet().removeIf(entry -> entry.getValue().updatedAtMs() < cutoff);

        // N1 (P1-4): reclaim orphan decode reservations — shadow entries past
        // the TTL with no matching scheduler inflight entry (e.g. interrupted
        // between route() and registerInflight, or a victim settle that raced
        // this cleanup). Without this pass they distort the admission view
        // until the registry's 300s eviction catches them.
        for (Map.Entry<String, DecodeEndpoint> decodeEntry
                : endpointRegistry.getDecodeEndpoints().entrySet()) {
            DecodeEndpoint decodeEp = decodeEntry.getValue();
            for (Map.Entry<String, RequestInflight> reserved : decodeEp.reservedView().entrySet()) {
                String requestId = reserved.getKey();
                if (now - reserved.getValue().createdAtMs() > ttlMs
                        && releaseOrphanDecodeReservation(
                                decodeEp, requestId, reserved.getValue())) {
                    Logger.warn("orphan decode reservation reclaimed: request_id={} worker={} age_ms={}",
                            requestId, decodeEntry.getKey(),
                            now - reserved.getValue().createdAtMs());
                }
            }
        }
        // This is the single scheduled TTL owner. Endpoint ledgers run only
        // after live scheduler entries were settled locally or fenced, and
        // remove only ids with no scheduler generation left.
        endpointRegistry.evictExpiredOrphans(ttlMs, this::ownsRequestGeneration);
    }

    /** Whether scheduler lifecycle still owns endpoint accounting for this id. */
    public boolean ownsRequestGeneration(String requestId) {
        return generationGates.containsKey(requestId)
                || inflight.containsKey(requestId);
    }

    /**
     * Release a snapshot reservation only while no scheduler generation owns it.
     *
     * <p>Registration and admission-mutation handoff use the same generation
     * monitor, so the ownership check and endpoint conditional release form
     * one critical section. If no generation exists, endpoint object identity
     * prevents a concurrently reused request ID from losing its newer
     * reservation.</p>
     */
    private boolean releaseOrphanDecodeReservation(
            DecodeEndpoint decodeEndpoint,
            String requestId,
            RequestInflight expectedReservation) {
        RequestGenerationGate generation = generationGates.get(requestId);
        if (generation == null) {
            return !inflight.containsKey(requestId)
                    && decodeEndpoint.releaseReservationIfCurrent(
                            requestId, expectedReservation);
        }
        synchronized (generation) {
            if (generationGates.get(requestId) != generation
                    || generation.admissionMutationInProgress
                    || inflight.containsKey(requestId)) {
                return false;
            }
            return decodeEndpoint.releaseReservationIfCurrent(
                    requestId, expectedReservation);
        }
    }

    /**
     * Low-frequency recovery for request graphs retained after fast Cancel
     * reconciliation was exhausted.
     *
     * <p>Endpoint-registry removal is deliberately not a terminal proof: a
     * route decision already published to a frontend can still arrive at that
     * address. The ledgers therefore stay charged. This sweep makes the
     * retention observable and starts only a bounded number of one-shot probes;
     * non-terminal probe outcomes return directly to quarantine and allocate no
     * delayed retry task.</p>
     */
    private void probeQuarantinedEngineFences(long nowMs) {
        if (shuttingDown.get()) {
            return;
        }
        if (quarantinedEngineFences.isEmpty()) {
            // Do not call clear(): a concurrently quarantined generation may
            // publish map ownership immediately before its FIFO ref. Drain only
            // refs still proven stale; preserve and requeue the first exact live
            // generation observed during the race.
            EngineFenceProbeRef stale;
            while ((stale = quarantinedProbeQueue.poll()) != null) {
                if (quarantinedEngineFences.get(stale.requestId())
                        == stale.registration()) {
                    requeueEngineFenceProbe(stale);
                    break;
                }
            }
            return;
        }
        int retained = 0;
        long oldestAgeMs = 0;
        for (Map.Entry<String, EngineFenceRegistration> candidate
                : quarantinedEngineFences.entrySet()) {
            String requestId = candidate.getKey();
            EngineFenceRegistration registration = candidate.getValue();
            InflightEntry entry = inflight.get(requestId);
            if (entry == null) {
                quarantinedEngineFences.remove(requestId, registration);
                continue;
            }
            synchronized (entry) {
                if (inflight.get(requestId) != entry
                        || entry.engineFence != registration
                        || entry.cleanupOwned || entry.lifecycle.isTerminal()
                        || !registration.wasQuarantined()) {
                    quarantinedEngineFences.remove(requestId, registration);
                    continue;
                }
                retained++;
                oldestAgeMs = Math.max(oldestAgeMs, nowMs - entry.createdAtMs());
            }
        }

        // A CHM traversal has no rotating cursor; repeatedly probing its first
        // N entries can starve every later fence forever. The generation-tagged
        // FIFO below provides deterministic round-robin fairness. Each live ref
        // is appended once after it is visited; stale refs are dropped lazily.
        int probesStarted = 0;
        int rotationBudget = quarantinedProbeQueue.size();
        for (int visited = 0;
             visited < rotationBudget
                     && probesStarted < engineFencePolicy.maxProbesPerCleanup();
             visited++) {
            EngineFenceProbeRef ref = quarantinedProbeQueue.poll();
            if (ref == null) {
                break;
            }
            EngineFenceRegistration registration = ref.registration();
            if (quarantinedEngineFences.get(ref.requestId()) != registration) {
                continue;
            }
            InflightEntry entry = inflight.get(ref.requestId());
            boolean live = false;
            boolean probe = false;
            if (entry != null) {
                synchronized (entry) {
                    live = inflight.get(ref.requestId()) == entry
                            && entry.engineFence == registration
                            && !entry.cleanupOwned && !entry.lifecycle.isTerminal()
                            && registration.wasQuarantined();
                    probe = live && registration.isQuarantined();
                }
            }
            if (!live) {
                quarantinedEngineFences.remove(ref.requestId(), registration);
                continue;
            }
            requeueEngineFenceProbe(ref);
            if (probe) {
                probesStarted++;
                reconcileEngineFence(entry, registration,
                        engineFencePolicy.maxFastAttempts(), true);
            }
        }
        if (retained > 0) {
            Logger.error("event=engine_fence_quarantine_summary retained={} "
                            + "oldest_age_ms={} probes_started={} probe_cap={}",
                    retained, oldestAgeMs, probesStarted,
                    engineFencePolicy.maxProbesPerCleanup());
        }
    }

    // ==================== DecisionGroupHandler callbacks (from WorkerBatcher) ====================

    @Override
    public void onExpired(BatchItem head) {
        if (entryFor(head) != null) {
            // The batcher and the request timer may observe the same absolute
            // expiration concurrently. Both must enter the cancellation
            // reducer so first-cause ownership and the existing external
            // timeout classification cannot depend on which thread wins.
            onRequestExpired(head.requestId(), head.future());
        } else if (!head.future().isDone() && !terminalStates.containsKey(head.requestId())) {
            rollback(head);
        }
    }

    @Override
    public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata metadata) {
        if (items == null || items.isEmpty()) {
            return;
        }
        if (!tryAcquireDeliveryPermit()) {
            failDecisionGroupAfterShutdown(items);
            return;
        }
        try {
            DeliveryMode mode = items.get(0).deliveryMode();
            int mixedAt = firstDifferentDeliveryMode(items, mode);
            if (mixedAt < 0) {
                deliverDecisionGroup(mode, items, metadata);
                return;
            }

            // A live config update may leave old and new delivery modes in the
            // same worker queue. The common case above allocates nothing; only the
            // transition window pays for partitioning, and ownership protocols are
            // never mixed in one externally visible operation.
            List<BatchItem> batchItems = new ArrayList<>(items.size());
            List<BatchItem> routeItems = new ArrayList<>(items.size());
            for (BatchItem item : items) {
                (item.deliveryMode() == DeliveryMode.BATCH_ENQUEUE
                        ? batchItems : routeItems).add(item);
            }
            if (!batchItems.isEmpty()) {
                deliverDecisionGroup(DeliveryMode.BATCH_ENQUEUE, batchItems, metadata);
            }
            if (!routeItems.isEmpty()) {
                deliverDecisionGroup(DeliveryMode.ROUTE_DECISION, routeItems, metadata);
            }
        } finally {
            releaseDeliveryPermit();
        }
    }

    /** Reject staged work after the lifecycle gate closes without acquiring new ledgers. */
    private void failDecisionGroupAfterShutdown(List<BatchItem> items) {
        IllegalStateException shutdownFailure = new IllegalStateException(
                "priority scheduler stopped before decision delivery");
        for (BatchItem item : items) {
            onDeliveryFailure(item, shutdownFailure);
        }
    }

    private static int firstDifferentDeliveryMode(List<BatchItem> items,
                                                  DeliveryMode expected) {
        for (int i = 1; i < items.size(); i++) {
            if (items.get(i).deliveryMode() != expected) {
                return i;
            }
        }
        return -1;
    }

    private void deliverDecisionGroup(DeliveryMode mode,
                                      List<BatchItem> items,
                                      DecisionGroupMetadata metadata) {
        if (mode == DeliveryMode.BATCH_ENQUEUE) {
            enqueueBatch(items, metadata);
        } else {
            deliverRouteDecisions(items, metadata);
        }
    }

    @Override
    public void onOfferFailure(BatchItem item, Throwable error) {
        // priority scheduling: over-capacity requests carry a dedicated non-retryable error code
        // instead of the generic (retryable) delivery failure (design doc 8.3).
        StrategyErrorType errorType = error instanceof BatchTokenCapacityExceededException
                ? StrategyErrorType.BATCH_TOKEN_CAPACITY_EXCEEDED
                : StrategyErrorType.BATCH_DISPATCH_FAILED;
        String failureDetail = error == null ? "queue full" : error.getMessage();
        InflightEntry entry = entryFor(item);
        if (entry != null) {
            ResponseCompletion publication;
            synchronized (entry) {
                publication = reduceOrdinaryTerminalLocked(entry, DeferredTerminal.failure(
                        errorType,
                        "Worker scheduling queue rejected request: " + failureDetail,
                        false));
            }
            submitResponseCompletion(publication);
        } else if (!item.future().isDone() && !terminalStates.containsKey(item.requestId())) {
            rollback(item);
            completeError(item.future(), errorType,
                    "Worker scheduling queue rejected request: " + failureDetail);
        }
    }

    @Override
    public void onDeliveryFailure(BatchItem item, Throwable error) {
        String detail = error == null || error.getMessage() == null
                ? "unknown delivery failure"
                : error.getMessage();
        reduceDeliveryFailure(item, detail,
                "Decision delivery failed: " + detail);
    }

    // ==================== Delivery pipeline ====================

    /**
     * Commit batch to PrefillEndpoint, then delegate to {@link BatchDispatcher}
     * for asynchronous gRPC dispatch.
     * <p>
     * The heavy gRPC I/O is handled asynchronously by the batch dispatcher's thread pool.
     */
    private void enqueueBatch(List<BatchItem> items, DecisionGroupMetadata metadata) {
        String reason = metadata.reason();
        PrefillEndpoint prefillEp = items.get(0).prefillEp();
        WorkerBatcher batcher = prefillEp != null ? prefillEp.getBatcher() : null;

        // [SYNC] Compute prediction and commit only active items to endpoint
        long predMs = 0;
        long batchId = batchIdGenerator.nextBatchId();
        Long configuredDecodeLimit = configService.loadBalanceConfig().getRouter()
                .getRoles().getDecode().getAvailability().getMaxEngineRequests();
        long decodeConcurrencyLimit = configuredDecodeLimit == null
                ? 0 : configuredDecodeLimit;
        List<BatchItem> readyForEnqueue = new ArrayList<>(items.size());
        for (BatchItem item : items) {
            boolean callbackOwnsPending = false;
            // stageForDelivery removed this item from the live queue while
            // retaining its capacity slot. Claim callback ownership before
            // any Decode or lifecycle mutation; shutdown drains only items
            // that have not crossed this fence.
            if (batcher != null) {
                BatcherContext.PendingClaimResult pendingClaim =
                        batcher.claimPendingDelivery(item);
                if (pendingClaim == BatcherContext.PendingClaimResult.STOPPED) {
                    // stopAndDrainTo owns and drains this still-STAGED item.
                    continue;
                }
                callbackOwnsPending = pendingClaim
                        == BatcherContext.PendingClaimResult.CLAIMED;
                // NOT_PENDING preserves direct callback/test and legacy use:
                // those items were never removed from this batcher's queue.
            }
            try {
                InflightEntry entry = entryFor(item);
                if (entry == null) {
                    continue;
                }
                ResponseCompletion publication = null;
                synchronized (entry) {
                    if (!item.future().isDone()
                            && !entry.lifecycle.isTerminal() && !entry.cleanupOwned) {
                        DecodeEndpoint.DispatchClaimResult claim = item.decodeEp() == null
                                ? DecodeEndpoint.DispatchClaimResult.CLAIMED
                                : item.decodeEp().tryClaimEngineDispatch(
                                        item.requestId(), decodeConcurrencyLimit);
                        if (claim == DecodeEndpoint.DispatchClaimResult.CAPACITY_FULL) {
                            restorePendingDelivery(batcher, item);
                        } else if (claim == DecodeEndpoint.DispatchClaimResult.CLAIMED) {
                            entry.lifecycle.startBatchEnqueue(batchId);
                            readyForEnqueue.add(item);
                        } else {
                            // A scheduler preemption claim intentionally owns the
                            // later terminal. Missing endpoint ownership without
                            // such a claim is an invariant violation; fail it now
                            // instead of leaving an item outside both queue and
                            // engine indefinitely.
                            if (entry.preemption == null) {
                                publication = reduceOrdinaryTerminalLocked(entry,
                                        DeferredTerminal.failure(
                                                StrategyErrorType.BATCH_DISPATCH_FAILED,
                                                "Decode dispatch ownership lost before send",
                                                false));
                            }
                        }
                    }
                }
                submitResponseCompletion(publication);
            } catch (Throwable claimFailure) {
                failClaimedDelivery(item, claimFailure);
            } finally {
                // CAPACITY_FULL has already restored and removed its pending
                // record. Every other callback-owned outcome is terminal or
                // dispatching and must release the charged queue slot here.
                if (callbackOwnsPending) {
                    completePendingDelivery(batcher, item);
                }
            }
        }

        if (readyForEnqueue.isEmpty()) {
            return;
        }
        boolean enqueueCallStarted = false;
        BatchEnqueueDelivery.Submission deliverySubmission = null;
        Throwable preSendFailure = null;
        try {
            if (prefillEp != null) {
                PrefillTimePredictor predictor = prefillEp.getPredictor();
                predMs = (long) predictor.predictBatchMs(readyForEnqueue);
            }
            synchronized (deliveryFence) {
                // Prediction may yield to a request-expiration/cancel fence.
                // Revalidate dispatch ownership immediately before the first
                // externally visible commit/send step.
                readyForEnqueue = readyForEnqueue.stream().filter(item -> {
                    InflightEntry entry = entryFor(item);
                    if (entry == null) {
                        return false;
                    }
                    synchronized (entry) {
                        return inflight.get(item.requestId()) == entry
                                && !entry.cleanupOwned
                                && entry.engineFence == null
                                && !entry.lifecycle.isTerminal()
                                && entry.lifecycle.snapshot().batchId() == batchId;
                    }
                }).toList();
                if (readyForEnqueue.isEmpty()) {
                    return;
                }
                if (prefillEp != null) {
                    prefillEp.commitBatch(batchId, predMs, readyForEnqueue);
                }

                // Record dispatch timestamp for dispatch-to-ACK latency metric
                for (BatchItem item : readyForEnqueue) {
                    InflightEntry entry = entryFor(item);
                    if (entry != null) {
                        entry.lifecycle.markBatchEnqueueStarted();
                        item.ctx().setBatchDispatchedNanos(System.nanoTime());
                    }
                }

                // Submitting the RPC while holding the fence closes the final
                // validate->commit->send gap. Callback reconciliation reacquires
                // the same fence after this method returns. A dispatcher is
                // allowed to reject synchronously; its callback gate is opened
                // only after this synchronized block has released the fence.
                deliverySubmission = batchEnqueueDelivery.prepare(
                        new BatchEnqueueDelivery.Plan(readyForEnqueue, prefillEp,
                                batchId, predMs, reason),
                        this);
                enqueueCallStarted = true;
                try {
                    deliverySubmission.submit();
                } catch (Throwable deliveryStartFailure) {
                    preSendFailure = deliveryStartFailure;
                }
            }
        } catch (Throwable preparationFailure) {
            preSendFailure = preparationFailure;
        }

        if (enqueueCallStarted) {
            reportBatchDispatch(
                    readyForEnqueue, items, prefillEp, batchId, predMs, reason, metadata);
        }

        // The transport may synchronously report rejection from submit(). Drain
        // those outcomes only after deliveryFence is released. The gate itself
        // isolates per-item callback failures, so every deferred outcome runs.
        if (deliverySubmission != null) {
            deliverySubmission.releaseCallbacks();
        }

        if (preSendFailure != null) {
            if (enqueueCallStarted) {
                // A batch dispatcher may throw after starting its
                // network invocation. Preserve both ledgers and use the same
                // request-id cancel fence as an asynchronous lost ACK.
                for (BatchItem item : readyForEnqueue) {
                    onUncertain(item, preSendFailure);
                }
            } else {
                if (prefillEp != null) {
                    prefillEp.releaseBatch(batchId);
                }
                for (BatchItem item : readyForEnqueue) {
                    failClaimedDelivery(item, preSendFailure);
                }
            }
        }
    }

    /** Report dispatch telemetry outside the scheduler-wide delivery fence. */
    private void reportBatchDispatch(List<BatchItem> dispatched,
                                     List<BatchItem> originalBatch,
                                     PrefillEndpoint prefillEp,
                                     long batchId,
                                     long predictedMs,
                                     String reason,
                                     DecisionGroupMetadata metadata) {
        try {
            long nowMs = System.currentTimeMillis();
            long waitMs = nowMs - originalBatch.get(0).enqueuedAtMs();
            Map<Integer, Long> oldestEnqueueByPriority = new HashMap<>();
            for (BatchItem item : dispatched) {
                oldestEnqueueByPriority.merge(
                        item.priority(), item.enqueuedAtMs(), Math::min);
            }
            String engineIp = prefillEp != null ? prefillEp.getIp() : "";
            for (Map.Entry<Integer, Long> waitEntry
                    : oldestEnqueueByPriority.entrySet()) {
                reporter.reportBatchWaitTimeMs(
                        RoleType.PREFILL.name(),
                        engineIp,
                        nowMs - waitEntry.getValue(),
                        waitEntry.getKey());
            }
            BatchDispatcherConfig batch = configService.loadBalanceConfig().batchDispatcher();
            Logger.debug("flexlb_batch_dispatch batch_id={} reason={} batch_size={} wait_ms={} "
                            + "predicted_ms={} threshold_ms={} fixed_wait_ms={} batch_size_max={} "
                            + "queue_after={} worker={}",
                    batchId, reason, dispatched.size(), waitMs, predictedMs,
                    batch.getEarlyDispatchPredictedExecutionMs(),
                    batch.getMaxCollectionWaitMs(),
                    batch.getMaxRequests(), metadata.queueDepth(),
                    prefillEp != null ? prefillEp.ipPort() : "");
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("Failed to report batch dispatch: batch_id={}",
                    batchId, telemetryFailure);
        }
    }

    /**
     * Commit request-scoped Prefill accounting and publish route decisions.
     *
     * <p>Unlike batch RPC delivery there is no ambiguous network send from the
     * master. Each request acquires its Prefill and Decode ownership while its
     * inflight entry is locked, then crosses a request-scoped lifecycle fence.
     * No scheduler-wide dispatch lock is taken on this hot path.</p>
     */
    private void deliverRouteDecisions(List<BatchItem> items, DecisionGroupMetadata metadata) {
        PrefillEndpoint prefillEp = items.get(0).prefillEp();
        if (prefillEp == null) {
            for (BatchItem item : items) {
                failClaimedDelivery(item,
                        new IllegalStateException("route decision has no Prefill endpoint"));
            }
            return;
        }

        WorkerBatcher batcher = prefillEp.getBatcher();
        PrefillTimePredictor predictor = prefillEp.getPredictor();
        FlexlbConfig config = configService.loadBalanceConfig();
        Integer configuredPrefillLimit = config.getDispatcher()
                instanceof NonBatchDispatcherConfig nonBatch
                ? nonBatch.getMaxInflightRequestsPerPrefillWorker() : null;
        int prefillRequestLimit = configuredPrefillLimit == null
                ? 0 : configuredPrefillLimit;
        Long configuredDecodeLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        long decodeConcurrencyLimit = configuredDecodeLimit == null
                ? 0 : configuredDecodeLimit;
        List<BatchItem> deliverable = new ArrayList<>(items.size());

        for (BatchItem item : items) {
            boolean callbackOwnsPending = false;
            boolean requestLedgerCommitted = false;
            ResponseCompletion publication = null;
            boolean decodeOwnershipLost = false;
            if (batcher != null) {
                BatcherContext.PendingClaimResult pendingClaim =
                        batcher.claimPendingDelivery(item);
                if (pendingClaim == BatcherContext.PendingClaimResult.STOPPED) {
                    continue;
                }
                callbackOwnsPending = pendingClaim
                        == BatcherContext.PendingClaimResult.CLAIMED;
            }

            try {
                InflightEntry entry = entryFor(item);
                if (entry == null) {
                    continue;
                }
                long predictedMs = Math.max(0L,
                        predictor.estimateMs(item.seqLen(), item.hitCache()));
                synchronized (entry) {
                    if (item.future().isDone() || entry.lifecycle.isTerminal()
                            || entry.cleanupOwned || inflight.get(item.requestId()) != entry) {
                        continue;
                    }

                    if (!prefillEp.tryCommitRequest(
                            item.requestId(), predictedMs, prefillRequestLimit)) {
                        restorePendingDelivery(batcher, item);
                        continue;
                    }
                    requestLedgerCommitted = true;

                    DecodeEndpoint.DispatchClaimResult claim = item.decodeEp() == null
                            ? DecodeEndpoint.DispatchClaimResult.CLAIMED
                            : item.decodeEp().tryClaimEngineDispatch(
                                    item.requestId(), decodeConcurrencyLimit);
                    if (claim == DecodeEndpoint.DispatchClaimResult.CAPACITY_FULL) {
                        prefillEp.releaseRequest(item.requestId());
                        requestLedgerCommitted = false;
                        restorePendingDelivery(batcher, item);
                        continue;
                    }
                    if (claim != DecodeEndpoint.DispatchClaimResult.CLAIMED) {
                        prefillEp.releaseRequest(item.requestId());
                        requestLedgerCommitted = false;
                        if (entry.preemption == null) {
                            publication = reduceOrdinaryTerminalLocked(entry,
                                    DeferredTerminal.failure(
                                            StrategyErrorType.BATCH_DISPATCH_FAILED,
                                            "Decode dispatch ownership lost before route delivery",
                                            false));
                        }
                        decodeOwnershipLost = true;
                    } else {
                        entry.lifecycle.startRouteDecisionDelivery();
                        deliverable.add(item);
                    }
                }
                submitResponseCompletion(publication);
                if (decodeOwnershipLost) {
                    continue;
                }
            } catch (Throwable claimFailure) {
                if (requestLedgerCommitted) {
                    prefillEp.releaseRequest(item.requestId());
                }
                failClaimedDelivery(item, claimFailure);
            } finally {
                if (callbackOwnsPending) {
                    completePendingDelivery(batcher, item);
                }
            }
        }

        if (deliverable.isEmpty()) {
            return;
        }

        routeDecisionDelivery.deliver(deliverable, this);
    }

    private void failClaimedDelivery(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return;
        }
        ResponseCompletion publication;
        synchronized (entry) {
            publication = reduceOrdinaryTerminalLocked(entry, DeferredTerminal.failure(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Delivery preparation failed: " + error.getMessage(),
                    true));
        }
        submitResponseCompletion(publication);
    }

    private static void completePendingDelivery(WorkerBatcher batcher,
                                                BatchItem item) {
        if (batcher != null) {
            batcher.completePendingDelivery(item);
        }
    }

    private void restorePendingDelivery(WorkerBatcher batcher,
                                        BatchItem item) {
        if (batcher == null) {
            return;
        }
        BatcherContext.PendingRestoreResult result = batcher.restorePendingDelivery(item);
        if (result == BatcherContext.PendingRestoreResult.STOPPED) {
            onDeliveryFailure(item,
                    new CancellationException(
                            "FlexLB worker scheduling queue stopped while Decode capacity was full"));
        }
    }

    // ==================== DecisionDelivery.Callback implementation ====================

    /** Called with {@code entry} locked. */
    private ResponseCompletion confirmRouteDecisionLocked(InflightEntry entry,
                                                          BatchItem item) {
        RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
        if (inflight.get(item.requestId()) != entry || entry.cleanupOwned
                || current.deliveryClaimKind() != DeliveryClaimKind.ROUTE_DECISION
                || current.state().isTerminal()) {
            return null;
        }
        if (entry.preemption != null) {
            retainDeliveryConfirmation(entry.preemption, 0);
            if (entry.preemption.state == PreemptionRegistrationState.NOT_FOUND_STALE) {
                return replayAfterNegativeCancelLocked(
                        entry, entry.preemption.attemptToken, false);
            }
            return null;
        }
        return confirmDeliveryLocked(entry, 0);
    }

    @Override
    public void onDelivered(BatchItem item) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            // entry 已被 worker-status/cancel/timeout/onFailure/onOfferFailure 等终态路径移除，
            // 所有终态路径均在 finishEntry 前完成 future，故此处无需补发。
            return;
        }

        ResponseCompletion publication;
        if (item.deliveryMode() == DeliveryMode.ROUTE_DECISION) {
            synchronized (entry) {
                publication = confirmRouteDecisionLocked(entry, item);
            }
        } else {
            synchronized (deliveryFence) {
                synchronized (entry) {
                    publication = confirmBatchEnqueueLocked(entry, item);
                }
            }
        }
        submitResponseCompletion(publication);
    }

    /** Called with the delivery fence and {@code entry} locked. */
    private ResponseCompletion confirmBatchEnqueueLocked(InflightEntry entry,
                                                         BatchItem item) {
        if (entry.cleanupOwned) {
            return null;
        }
        RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
        if (current.deliveryClaimKind() != DeliveryClaimKind.BATCH_ENQUEUE) {
            Logger.debug("Ignoring EnqueueBatch ACK without a batch claim request_id={}",
                    item.requestId());
            return null;
        }
        long batchId = current.batchId();
        if (entry.engineFence != null) {
            Logger.debug("Retaining late EnqueueBatch ACK during reconciliation: "
                            + "request_id={} batch_id={}",
                    item.requestId(), batchId);
            return null;
        }
        if (entry.preemption != null) {
            PreemptionRegistration registration = entry.preemption;
            retainDeliveryConfirmation(registration, batchId);
            if (registration.state == PreemptionRegistrationState.NOT_FOUND_STALE) {
                return replayAfterNegativeCancelLocked(
                        entry, registration.attemptToken, false);
            }
            // CANCEL_IN_FLIGHT/CANCEL_REQUESTED/UNKNOWN retain the ACK.
            // It must not turn a priority victim into a successful frontend
            // response while the Cancel outcome is unresolved.
            return null;
        }
        return confirmDeliveryLocked(entry, batchId);
    }

    /**
     * Confirm a delivery after its ownership decision is final.
     * The returned publication must be executed only after every scheduler lock
     * has been released: {@code CompletableFuture.complete} runs arbitrary user
     * continuations synchronously on the completing thread.
     */
    private ResponseCompletion confirmDeliveryLocked(InflightEntry entry, long batchId) {
        // Delivery confirmation is an edge, not a level. A duplicate callback
        // can observe ACKNOWLEDGED while the first asynchronous completion is
        // still queued. Only DISPATCHING may create the response completion.
        if (entry.lifecycle.snapshot().state() != RequestLifecycleState.DISPATCHING) {
            return null;
        }
        BatchItem item = entry.item;
        // Build before claiming the lifecycle edge. If defensive copying ever
        // fails, the caller can still reduce the original delivery failure;
        // no completion ownership has been stranded.
        Response response = buildSuccessResponse(item);
        long batchEnqueueStartedAtMs = entry.lifecycle.getBatchEnqueueStartedAtMs();
        PrefillEndpoint prefill = item.prefillEp();
        String prefillIp = prefill != null ? prefill.getIp() : "";
        RequestLifecycleSnapshot snapshot = entry.lifecycle.markDeliveryConfirmed();
        if (snapshot.state() != RequestLifecycleState.ACKNOWLEDGED) {
            return null;
        }
        // Reserve response completion under the same entry lock as the
        // lifecycle transition. A racing worker terminal may advance the
        // lifecycle, but cannot overtake the delivery that owns the response.
        if (!claimResponseCompletionLocked(entry)) {
            return null;
        }
        // Internal timestamp fields are committed with the delivery edge. Reporter
        // callbacks are deliberately deferred to the unlocked publisher.
        long nowMs = System.currentTimeMillis();
        item.ctx().setAckAtMs(nowMs);
        item.ctx().setAckAtNanos(System.nanoTime());
        // A Prefill-only delivery has no later Decode-acceptance edge. Capture
        // its exact lease so admission backpressure can be retired before the
        // successful public future becomes observable. Completing the future
        // first and relying on whenComplete lets a waiting caller submit its
        // next request before the lease callback has run.
        AdmissionLease prefillOnlyLease = item.decodeEp() == null
                ? entry.admissionLease : null;
        return ResponseCompletion.success(item, response, prefillOnlyLease,
                snapshot.deliveryClaimKind(), batchId, reporter, prefillIp,
                snapshot.deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE
                        && batchEnqueueStartedAtMs > 0
                        ? nowMs - batchEnqueueStartedAtMs
                        : -1);
    }

    /** Called only while holding the request's {@link InflightEntry} monitor. */
    private static boolean claimResponseCompletionLocked(InflightEntry entry) {
        if (entry.responseCompletionClaimed || entry.item.future().isDone()) {
            return false;
        }
        entry.responseCompletionClaimed = true;
        return true;
    }

    /** Build the immutable public completion while holding only request state. */
    private static ResponseCompletion responsePublicationLocked(
            InflightEntry entry,
            Response response) {
        if (!claimResponseCompletionLocked(entry)) {
            return null;
        }
        return ResponseCompletion.terminal(entry.item.future(), response);
    }

    private static ResponseCompletion errorPublicationLocked(
            InflightEntry entry,
            StrategyErrorType errorType,
            String message) {
        Response response = buildErrorResponse(errorType, message);
        if (!claimResponseCompletionLocked(entry)) {
            return null;
        }
        return ResponseCompletion.terminal(
                entry.item.future(), response);
    }

    private static ResponseCompletion admissionErrorPublicationLocked(
            InflightEntry entry,
            AdmissionFailure failure,
            String trigger) {
        Response response = buildAdmissionErrorResponse(failure, trigger);
        if (!claimResponseCompletionLocked(entry)) {
            return null;
        }
        return ResponseCompletion.terminal(
                entry.item.future(), response);
    }

    private void submitResponseCompletion(ResponseCompletion publication) {
        if (publication == null) {
            return;
        }
        executeResponseTask(() -> completeResponse(publication));
    }

    /** Execute only after scheduler locks have been released. */
    private void executeResponseTask(Runnable task) {
        try {
            responseCompletionExecutor.execute(task);
        } catch (RejectedExecutionException executorUnavailable) {
            // Queue saturation is intentional backpressure, and shutdown can
            // race a final terminal observation. Every caller crosses this
            // boundary only after releasing scheduler locks, so caller-runs
            // cannot widen a scheduler critical section or lose completion.
            task.run();
        }
    }

    private static void completeResponse(ResponseCompletion publication) {
        if (publication.batchEnqueueAckLatencyMs() >= 0 && publication.reporter() != null) {
            try {
                publication.reporter().reportDispatchAckTimeMs(
                        RoleType.PREFILL.name(), publication.prefillIp(),
                        publication.batchEnqueueAckLatencyMs());
            } catch (RuntimeException telemetryFailure) {
                Logger.warn("Failed to record delivery ACK telemetry: request_id={} kind={}",
                        publication.item().requestId(), publication.deliveryClaimKind(),
                        telemetryFailure);
            }
        }
        boolean completed = publication.future() instanceof RequestGenerationGate generation
                ? generation.completeOwned(
                        publication.response(), publication.prefillOnlyLease())
                : publication.future().complete(publication.response());
        if (!completed) {
            return;
        }
        if (publication.deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE) {
            Logger.debug("FlexLB batch enqueued request {} in batch_id={}",
                    publication.item().requestId(), publication.batchId());
        } else if (publication.deliveryClaimKind() == DeliveryClaimKind.ROUTE_DECISION) {
            Logger.debug("FlexLB route decision delivered request {}",
                    publication.item().requestId());
        }
    }

    private Response buildSuccessResponse(BatchItem item) {
        Response success = copyResponse(item.routeResponse());
        success.setSuccess(true);
        success.setCode(200);
        success.setEnqueuedByMaster(item.deliveryMode() == DeliveryMode.BATCH_ENQUEUE);
        success.setQueueLength(inflight.size());
        return success;
    }

    @Override
    public void onFailure(BatchItem item, Throwable error) {
        String detail = error == null || error.getMessage() == null
                ? "unknown delivery failure"
                : error.getMessage();
        String failureMessage = item.deliveryMode() == DeliveryMode.ROUTE_DECISION
                ? "Route decision delivery failed: " + detail
                : detail;
        reduceDeliveryFailure(item, detail, failureMessage);
    }

    private void reduceDeliveryFailure(BatchItem item,
                                       String detail,
                                       String failureMessage) {
        InflightEntry entry = entryFor(item);
        if (entry != null) {
            ResponseCompletion publication;
            synchronized (entry) {
                if (item.deliveryMode() == DeliveryMode.BATCH_ENQUEUE
                        && entry.engineFence != null) {
                    Logger.debug("Retaining EnqueueBatch failure during reconciliation: "
                                    + "request_id={} batch_id={} cause={}",
                            item.requestId(), entry.lifecycle.snapshot().batchId(),
                            detail);
                    return;
                }
                // Even a raw EnqueueBatch 8429 is not the priority resource
                // fence. It follows the same deferred ordinary-terminal path;
                // only typed Prefill CANCELED may perform priority settlement.
                publication = reduceOrdinaryTerminalLocked(entry, DeferredTerminal.failure(
                        StrategyErrorType.BATCH_DISPATCH_FAILED,
                        failureMessage, true));
            }
            submitResponseCompletion(publication);
            return;
        }
        if (!item.future().isDone() && !terminalStates.containsKey(item.requestId())) {
            rollback(item);
            completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED,
                    failureMessage);
        }
    }

    @Override
    public void onTimeout(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return;
        }
        ResponseCompletion publication;
        synchronized (entry) {
            if (entry.engineFence != null) {
                Logger.debug("Retaining EnqueueBatch timeout during reconciliation: "
                                + "request_id={} batch_id={} cause={}",
                        item.requestId(), entry.lifecycle.snapshot().batchId(),
                        error == null ? "unknown" : error.getMessage());
                return;
            }
            publication = reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                    "EnqueueBatch deadline exceeded: " + error.getMessage()));
        }
        submitResponseCompletion(publication);
    }

    @Override
    public void onUncertain(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return;
        }
        EngineFenceRegistration started = null;
        ResponseCompletion publication = null;
        long batchId;
        synchronized (deliveryFence) {
            synchronized (entry) {
                RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
                batchId = snapshot.batchId();
                if (inflight.get(item.requestId()) != entry || entry.cleanupOwned
                        || entry.item.future().isDone()
                        || entry.lifecycle.isTerminal()
                        || snapshot.state() == RequestLifecycleState.ACKNOWLEDGED
                        || snapshot.deliveryClaimKind() != DeliveryClaimKind.BATCH_ENQUEUE
                        || batchId <= 0) {
                    return;
                }
                if (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED) {
                    publication = confirmDeliveryLocked(entry, batchId);
                } else {
                    started = installEngineFenceLocked(
                            entry, EngineFenceCause.BATCH_ACK_UNCERTAIN,
                            error == null ? "EnqueueBatch acknowledgement uncertain"
                                    : error.getMessage());
                }
            }
        }
        submitResponseCompletion(publication);
        if (started != null) {
            Logger.debug("EnqueueBatch ACK uncertain; fencing before settlement: "
                            + "request_id={} batch_id={} engine={} cause={}",
                    item.requestId(), batchId,
                    item.prefillEp() != null ? item.prefillEp().ipPort() : "unknown",
                    error == null ? "deadline exceeded" : error.getMessage());
            reconcileEngineFence(entry, started, 0);
        }
    }

    /** Called with {@code entry} locked. */
    private EngineFenceRegistration installEngineFenceLocked(
            InflightEntry entry,
            EngineFenceCause cause,
            String detail) {
        return installEngineFenceLocked(entry, cause, detail, 0, false, null);
    }

    /** Called with {@code entry} locked; transferred token is endpoint-exact. */
    private EngineFenceRegistration installEngineFenceLocked(
            InflightEntry entry,
            EngineFenceCause cause,
            String detail,
            long transferredPreemptionToken) {
        return installEngineFenceLocked(
                entry, cause, detail, transferredPreemptionToken, false, null);
    }

    /** Called with {@code entry} locked. */
    private EngineFenceRegistration installEngineFenceLocked(
            InflightEntry entry,
            EngineFenceCause cause,
            String detail,
            long transferredPreemptionToken,
            boolean allowDecodeOwnedCancellation,
            PreemptionRegistration transferredPreemption) {
        EngineFenceRegistration prepared = prepareEngineFenceLocked(
                entry, cause, detail, transferredPreemptionToken,
                allowDecodeOwnedCancellation, transferredPreemption);
        return commitPreparedEngineFenceLocked(
                entry, prepared, transferredPreemption);
    }

    /** Prepare every resource before changing scheduler ownership. */
    private EngineFenceRegistration prepareEngineFenceLocked(
            InflightEntry entry,
            EngineFenceCause cause,
            String detail,
            long transferredPreemptionToken,
            boolean allowDecodeOwnedCancellation,
            PreemptionRegistration transferredPreemption) {
        if (entry.engineFence != null
                || entry.preemption != transferredPreemption
                || entry.cleanupOwned || entry.lifecycle.isTerminal()) {
            return null;
        }
        if (entry.engineOwnershipState == EngineOwnershipState.DECODE_OWNED
                && !allowDecodeOwnedCancellation) {
            // Decode ownership is authoritative and cannot be fenced by a
            // Prefill Cancel. Callers which are still awaiting a batch ACK
            // publish that ACK explicitly after releasing their locks.  An
            // accepted cancellation is different: it intentionally targets
            // an already-running Decode request through its original Prefill.
            return null;
        }
        RequestLifecycleSnapshot lifecycle = entry.lifecycle.snapshot();
        long batchId = lifecycle.batchId();
        PrefillEndpoint prefill = entry.item.prefillEp();
        boolean batchMemberProtected = false;
        boolean batchDelivery = lifecycle.deliveryClaimKind()
                == DeliveryClaimKind.BATCH_ENQUEUE;
        if (batchDelivery && prefill != null && batchId > 0) {
            batchMemberProtected = prefill.tryProtectBatchMember(
                    batchId, entry.item.requestId());
            if (!batchMemberProtected
                    && cause == EngineFenceCause.BATCH_ACK_UNCERTAIN) {
                Logger.debug("Engine fence lost Prefill batch-member protection race: "
                                + "request_id={} batch_id={}",
                        entry.item.requestId(), batchId);
                return null;
            }
        } else if (cause == EngineFenceCause.BATCH_ACK_UNCERTAIN) {
            // A batch ACK fence without the exact Prefill ledger key cannot
            // retain its admission gate and must never be published.
            return null;
        }

        EngineFenceResources resources = EngineFenceResources.acquire(
                entry.item, batchId, batchMemberProtected);
        try {
            return new EngineFenceRegistration(
                    cause, detail == null ? cause.name() : detail,
                    transferredPreemptionToken, resources);
        } catch (RuntimeException | Error installationFailure) {
            try {
                resources.release();
            } catch (RuntimeException | Error cleanupFailure) {
                installationFailure.addSuppressed(cleanupFailure);
            }
            throw installationFailure;
        }
    }

    /** Install a prepared fence and only then retire the exact previous owner. */
    private static EngineFenceRegistration commitPreparedEngineFenceLocked(
            InflightEntry entry,
            EngineFenceRegistration prepared,
            PreemptionRegistration transferredPreemption) {
        if (prepared == null) {
            return null;
        }
        if (entry.engineFence != null
                || entry.preemption != transferredPreemption
                || entry.cleanupOwned || entry.lifecycle.isTerminal()) {
            prepared.resources.release();
            return null;
        }
        entry.engineFence = prepared;
        if (transferredPreemption != null) {
            entry.preemption = null;
        }
        return prepared;
    }

    private static void discardPreparedEngineFence(
            EngineFenceRegistration prepared) {
        if (prepared != null) {
            prepared.resources.release();
        }
    }

    private void reconcileEngineFence(InflightEntry entry,
                                      EngineFenceRegistration registration,
                                      int attempt) {
        reconcileEngineFence(entry, registration, attempt, false);
    }

    /**
     * Schedule one cancellable reconciliation on this scheduler's lifecycle-
     * owned timer. The task never invokes Cancel until it has revalidated the
     * exact request/fence generation under the request entry lock.
     */
    private void scheduleEngineFenceReconciliation(
            InflightEntry entry,
            EngineFenceRegistration registration,
            int attempt,
            long delayMs) {
        synchronized (entry) {
            if (shuttingDown.get()
                    || inflight.get(entry.item.requestId()) != entry
                    || entry.engineFence != registration || entry.cleanupOwned
                    || entry.lifecycle.isTerminal()
                    || !registration.canScheduleReconciliation()) {
                return;
            }
            try {
                ScheduledFuture<?> scheduled = engineFenceRetryTimer.schedule(
                        () -> runScheduledEngineFenceReconciliation(
                                entry, registration, attempt),
                        Math.max(0, delayMs), TimeUnit.MILLISECONDS);
                registration.installScheduledReconciliation(scheduled);
            } catch (RejectedExecutionException timerStopped) {
                if (!shuttingDown.get()) {
                    Logger.error("Engine fence timer rejected reconciliation: "
                                    + "request_id={} cause={} attempt={}",
                            entry.item.requestId(), registration.cause, attempt,
                            timerStopped);
                }
            }
        }
    }

    private void runScheduledEngineFenceReconciliation(
            InflightEntry entry,
            EngineFenceRegistration registration,
            int attempt) {
        synchronized (entry) {
            if (!registration.consumeScheduledReconciliation()) {
                return;
            }
        }
        reconcileEngineFence(entry, registration, attempt);
    }

    /**
     * Run one request-scoped Cancel probe.
     *
     * <p>Normal calls form a bounded fast-retry chain. A quarantine probe is
     * started only by the 60-second cleanup sweep and always returns to
     * {@link EngineFenceState#QUARANTINED} after a non-terminal acknowledgement;
     * it never creates another delayed task.</p>
     */
    private void reconcileEngineFence(InflightEntry entry,
                                      EngineFenceRegistration registration,
                                      int attempt,
                                      boolean quarantineProbe) {
        if (shuttingDown.get()) {
            return;
        }
        CompletableFuture<EngineCancelChannel.CancelOutcome> cancelFuture = null;
        Throwable synchronousFailure = null;
        long cancelAckTimeoutMs = DEFAULT_CANCEL_ACK_TIMEOUT_MS;
        synchronized (entry) {
            if (shuttingDown.get()
                    || inflight.get(entry.item.requestId()) != entry
                    || entry.engineFence != registration || entry.cleanupOwned
                    || entry.lifecycle.isTerminal()
                    || !registration.beginCancelAttempt(quarantineProbe)) {
                return;
            }
            try {
                // Invocation happens while the request-entry fence is held.
                // A WorkerStatus terminal therefore cannot win after this
                // liveness check and before a stale absent tombstone is sent.
                ServerStatus prefill = entry.item.prefill();
                EngineCancelChannel.CancelTarget target = prefill == null ? null
                        : new EngineCancelChannel.CancelTarget(
                                prefill.getServerIp(), prefill.getGrpcPort());
                cancelFuture = engineCancelChannel.cancel(
                        target, entry.item.requestId(), cancelAckTimeoutMs);
            } catch (Throwable error) {
                synchronousFailure = error;
            }
        }
        if (synchronousFailure != null) {
            Logger.warn("Engine fence Cancel threw synchronously: "
                            + "request_id={} cause={} attempt={} quarantine_probe={}",
                    entry.item.requestId(), registration.cause, attempt,
                    quarantineProbe, synchronousFailure);
            handleEngineFenceOutcome(entry, registration,
                    EngineCancelChannel.CancelOutcome.failed(), attempt, quarantineProbe);
            return;
        }
        if (cancelFuture == null) {
            Logger.warn("Engine fence Cancel returned null future: "
                            + "request_id={} cause={} attempt={} quarantine_probe={}",
                    entry.item.requestId(), registration.cause, attempt, quarantineProbe);
            handleEngineFenceOutcome(entry, registration,
                    EngineCancelChannel.CancelOutcome.failed(), attempt, quarantineProbe);
            return;
        }
        // The channel contract is bounded, but this final guard prevents a
        // buggy implementation from pinning the fence in CANCEL_IN_FLIGHT and
        // disabling every later quarantine probe.
        cancelFuture.completeOnTimeout(
                        EngineCancelChannel.CancelOutcome.failed(),
                        cancelAckTimeoutMs, TimeUnit.MILLISECONDS)
                .exceptionally(ignored -> EngineCancelChannel.CancelOutcome.failed())
                .thenAccept(outcome -> handleEngineFenceOutcome(
                        entry, registration,
                        outcome == null ? EngineCancelChannel.CancelOutcome.failed() : outcome,
                        attempt, quarantineProbe));
    }

    private void handleEngineFenceOutcome(
            InflightEntry entry,
            EngineFenceRegistration registration,
            EngineCancelChannel.CancelOutcome outcome,
            int attempt,
            boolean quarantineProbe) {
        boolean retry = false;
        boolean enteredQuarantine = false;
        ResponseCompletion publication = null;
        synchronized (entry) {
            if (inflight.get(entry.item.requestId()) != entry
                    || entry.engineFence != registration || entry.cleanupOwned
                    || entry.lifecycle.isTerminal()
                    || registration.state != EngineFenceState.CANCEL_IN_FLIGHT) {
                return;
            }
            if (shuttingDown.get()
                    && outcome.ack() != EngineCancelChannel.CancelAck.TOMBSTONED) {
                return;
            }
            switch (outcome.ack()) {
                case TOMBSTONED -> publication = settleEngineFenceLocked(entry,
                        registration.detail + "; engine fenced late enqueue");
                case ACCEPTED -> {
                    // ACCEPTED proves intent installation, not execution. A
                    // retry may observe TOMBSTONED after execution; otherwise
                    // typed CANCELED or Decode terminal settles the entry.
                    if (quarantineProbe) {
                        registration.returnToQuarantine();
                    } else if (attempt + 1 >= engineFencePolicy.maxFastAttempts()) {
                        enteredQuarantine = registration.enterQuarantine(
                                System.currentTimeMillis());
                    } else {
                        registration.awaitRetry();
                        retry = true;
                    }
                    Logger.debug("Engine fence cancel accepted; retaining ledgers: "
                                    + "request_id={} cause={} attempt={}",
                            entry.item.requestId(), registration.cause, attempt);
                }
                case NOT_FOUND, FAILED, UNSUPPORTED -> {
                    // None of these is a safe local release fact. In
                    // particular, NOT_FOUND leaves a frontend late-send race.
                    if (quarantineProbe) {
                        registration.returnToQuarantine();
                    } else if (attempt + 1 >= engineFencePolicy.maxFastAttempts()) {
                        enteredQuarantine = registration.enterQuarantine(
                                System.currentTimeMillis());
                    } else {
                        registration.awaitRetry();
                        retry = true;
                    }
                }
            }
            if (enteredQuarantine && !publishQuarantinedEngineFence(
                    entry.item.requestId(), registration)) {
                enteredQuarantine = false;
            }
        }
        submitResponseCompletion(publication);
        if (enteredQuarantine) {
            reportEngineFenceQuarantine(entry, registration, attempt + 1, outcome);
        }
        if (retry) {
            long delayMs = engineFencePolicy.retryDelayMs(attempt);
            scheduleEngineFenceReconciliation(
                    entry, registration, attempt + 1, delayMs);
        }
    }

    /** Cold-path publication with a post-offer gate check for shutdown races. */
    private boolean publishQuarantinedEngineFence(
            String requestId,
            EngineFenceRegistration registration) {
        if (shuttingDown.get()) {
            return false;
        }
        quarantinedEngineFences.put(requestId, registration);
        if (shuttingDown.get()) {
            quarantinedEngineFences.remove(requestId, registration);
            return false;
        }
        EngineFenceProbeRef ref = new EngineFenceProbeRef(requestId, registration);
        if (!requeueEngineFenceProbe(ref)) {
            quarantinedEngineFences.remove(requestId, registration);
            return false;
        }
        return true;
    }

    /** Cold-path requeue that cannot survive a concurrent shutdown clear. */
    private boolean requeueEngineFenceProbe(EngineFenceProbeRef ref) {
        if (shuttingDown.get()) {
            return false;
        }
        quarantinedProbeQueue.offer(ref);
        if (shuttingDown.get()) {
            quarantinedProbeQueue.remove(ref);
            return false;
        }
        return true;
    }

    private void reportEngineFenceQuarantine(
            InflightEntry entry,
            EngineFenceRegistration registration,
            int attempts,
            EngineCancelChannel.CancelOutcome lastOutcome) {
        Logger.error("event=engine_fence_quarantined request_id={} cause={} attempts={} "
                        + "last_ack={} lifecycle={} endpoint={} detail={}",
                entry.item.requestId(), registration.cause, attempts,
                lastOutcome.ack(), entry.lifecycle.snapshot().state(),
                entry.item.prefillEp() == null ? "unknown" : entry.item.prefillEp().ipPort(),
                registration.detail);
        if (admissionScheduler != null) {
            try {
                admissionScheduler.onInflightSettleMiss("engine_fence_quarantined");
            } catch (RuntimeException metricFailure) {
                Logger.warn("Failed to report engine-fence quarantine metric: request_id={}",
                        entry.item.requestId(), metricFailure);
            }
        }
    }

    /** Called with the request entry locked. */
    private ResponseCompletion settleEngineFenceLocked(
            InflightEntry entry,
            String detail) {
        if (entry.engineFence == null) {
            return null;
        }
        settleTransferredDecodeFence(entry, entry.engineFence);
        DecodeEndpoint decode = entry.item.decodeEp();
        if (decode != null) {
            decode.settleTombstonedRequest(entry.item.requestId());
        }
        entry.engineOwnershipState = EngineOwnershipState.TERMINAL;
        entry.cleanupOwned = true;
        if (entry.cancellationReason != null) {
            return settleCancellationLocked(entry, detail);
        }
        return timeoutEntry(entry, detail);
    }

    /** Called with the request entry locked. */
    private void clearEngineFenceLocked(InflightEntry entry,
                                        EngineFenceRegistration registration) {
        if (registration == null || entry.engineFence != registration) {
            return;
        }
        registration.cancelScheduledReconciliation();
        quarantinedEngineFences.remove(entry.item.requestId(), registration);
        entry.engineFence = null;
        registration.resources.release();
    }

    /** Release a transferred priority hold only on a fresh pre-Cancel active proof. */
    private static void releaseTransferredDecodeFenceActive(
            InflightEntry entry,
            EngineFenceRegistration registration) {
        if (registration == null || registration.transferredPreemptionToken == 0) {
            return;
        }
        DecodeEndpoint endpoint = entry.item.decodeEp();
        if (endpoint != null) {
            endpoint.releaseEngineFenceClaimActive(
                    registration.transferredPreemptionToken, entry.item.requestId());
        }
    }

    /** Release a transferred priority hold on an authoritative terminal proof. */
    private static void settleTransferredDecodeFence(
            InflightEntry entry,
            EngineFenceRegistration registration) {
        if (registration == null || registration.transferredPreemptionToken == 0) {
            return;
        }
        DecodeEndpoint endpoint = entry.item.decodeEp();
        if (endpoint != null) {
            endpoint.settleEngineFenceClaim(
                    registration.transferredPreemptionToken, entry.item.requestId());
        }
    }

    // ==================== Internal: resource rollback ====================

    private void rollbackOnce(InflightEntry entry) {
        if (entry.rolledBack.compareAndSet(false, true)) {
            rollback(entry.item);
        }
    }

    /** Rollback using endpoint references already held by the item (no registry lookup). */
    private void rollback(BatchItem item) {
        DecodeEndpoint decodeEp = item.decodeEp();
        if (decodeEp != null && item.decode() != null) {
            decodeEp.release(item.decode().getRequestId());
        }
    }

    /**
     * Rollback using route response — used only in submit() early-return paths
     * where BatchItem has not been created yet.
     */
    private void rollback(Response routeResponse) {
        if (routeResponse == null || routeResponse.getServerStatus() == null) {
            return;
        }
        for (ServerStatus serverStatus : routeResponse.getServerStatus()) {
            rollback(serverStatus);
        }
    }

    private void rollback(ServerStatus serverStatus) {
        if (serverStatus == null) {
            return;
        }
        if (serverStatus.getRole() == RoleType.DECODE) {
            String ipPort = serverStatus.getLogicalIpPort();
            DecodeEndpoint ep = endpointRegistry.getDecode(ipPort);
            if (ep != null) {
                ep.release(serverStatus.getRequestId());
            }
        }
    }

    // ==================== Internal: inflight queries ====================

    private InflightEntry entryFor(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        return entry != null && entry.item == item ? entry : null;
    }

    /** Release the delivery-specific Prefill ledger entry idempotently. */
    private void releasePrefillAccounting(InflightEntry entry) {
        PrefillEndpoint prefillEp = entry.item.prefillEp();
        if (prefillEp == null) {
            return;
        }
        RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
        if (snapshot.deliveryClaimKind() == DeliveryClaimKind.ROUTE_DECISION) {
            if (prefillEp.releaseRequest(entry.item.requestId())) {
                Logger.debug("FlexLB release Prefill request accounting: request_id={} engine={}",
                        entry.item.requestId(), prefillEp.getIp());
            }
            return;
        }
        if (snapshot.deliveryClaimKind() == DeliveryClaimKind.BATCH_ENQUEUE && snapshot.batchId() > 0) {
            prefillEp.repackBatch(snapshot.batchId(), Set.of(entry.item.requestId()));
            Logger.debug("FlexLB remove from Prefill batch: request_id={} batch_id={} engine={}",
                    entry.item.requestId(), snapshot.batchId(), prefillEp.getIp());
        }
    }

    private ResponseCompletion timeoutEntry(InflightEntry entry, String detail) {
        AdmissionFailure admissionFailure = null;
        PrefillEndpoint prefill = entry.item.prefillEp();
        if (entry.priorityAdmission) {
            admissionFailure = classifyAdmissionTimeout(entry.item, prefill);
            if (prefill != null) {
                prefill.getBatcher().queueManager().tryRemove(
                        entry.item.requestId(), "ADMISSION_TIMEOUT");
            }
        }
        RequestLifecycleSnapshot terminal = entry.lifecycle.timeout(detail);
        rollbackOnce(entry);
        releasePrefillAccounting(entry);
        ResponseCompletion publication;
        if (admissionFailure != null) {
            Logger.debug("[priority] admission timeout classified: request_id={} "
                            + "priority={} lifecycle={} error_code={} reason={} trigger={}",
                    entry.item.requestId(), entry.item.priority(), terminal.state(),
                    admissionFailure.errorType().getErrorCode(), admissionFailure.reason(), detail);
            publication = admissionErrorPublicationLocked(entry, admissionFailure, detail);
        } else {
            publication = errorPublicationLocked(
                    entry, entry.deadlineErrorType, detail);
        }
        finishEntry(entry, terminal);
        return publication;
    }

    private static AdmissionFailure classifyAdmissionTimeout(BatchItem item,
                                                              PrefillEndpoint prefill) {
        if (prefill == null) {
            return AdmissionFailure.resourceExhausted();
        }
        List<QueuedRequestSnapshot> ahead = new ArrayList<>();
        for (QueuedRequestSnapshot queued
                : prefill.getBatcher().queueManager().snapshot().items()) {
            if (Objects.equals(queued.requestId(), item.requestId())) {
                return AdmissionFailureClassifier.classifyQueuedTimeout(
                        item.priority(), ahead);
            }
            ahead.add(queued);
        }
        // The item already left the queue (dispatch/expiry won); there is no
        // queue-order evidence for HIGHER or SAME.
        return AdmissionFailure.resourceExhausted();
    }

    private static void completeError(CompletableFuture<Response> future,
                                      StrategyErrorType errorType,
                                      String message) {
        if (future.isDone()) {
            return;
        }
        future.complete(buildErrorResponse(errorType, message));
    }

    private static Response buildErrorResponse(StrategyErrorType errorType,
                                               String message) {
        Response errorResp = Response.error(errorType);
        errorResp.setErrorMessage(errorType.buildErrorMessage(message));
        return errorResp;
    }

    private static Response buildAdmissionErrorResponse(AdmissionFailure failure,
                                                        String trigger) {
        Response errorResp = Response.error(failure.errorType(), failure.reason());
        String detail = failure.message() + "; trigger=" + trigger;
        errorResp.setErrorMessage(failure.errorType().buildErrorMessage(detail));
        return errorResp;
    }

    private void finishEntry(InflightEntry entry,
                             RequestLifecycleSnapshot terminal) {
        clearEngineFenceLocked(entry, entry.engineFence);
        if (entry.admissionLease != null) {
            try {
                entry.admissionLease.markRequestSettled();
            } catch (RuntimeException | Error terminationFailure) {
                Logger.error("Failed to terminate admission lease: request_id={}",
                        entry.item.requestId(), terminationFailure);
            }
        }
        // Publish the tombstone before removing inflight. submit() then observes
        // at least one side of the ownership transition and cannot revive the request ID.
        terminalStates.put(terminal.requestId(), terminal);
        removeInflightGeneration(entry);
    }

    /** Detach exactly one inflight/future generation; stale cleanup cannot remove a reuse. */
    private void removeInflightGeneration(InflightEntry entry) {
        inflight.remove(entry.item.requestId(), entry);
        if (entry.item.future() instanceof RequestGenerationGate generation) {
            generation.releaseOutstandingPermit();
            // An incomplete future is still the generation's publication
            // owner. Retain its gate until whenComplete (or the mutation
            // handoff) observes the completion; this prevents a late planner
            // from claiming the small finishEntry -> async-publication window.
            // If the future is already terminal, no new mutation can pass
            // RequestGenerationGate.isOpen(), so this exact removal is race-free
            // without taking the gate monitor while the entry lock is held.
            if (generation.isDone()
                    && !generation.admissionMutationInProgress) {
                generationGates.remove(entry.item.requestId(), generation);
            }
        }
    }

    private static boolean batchMatches(RequestLifecycleSnapshot snapshot,
                                        long expectedBatchId) {
        if (snapshot == null) {
            return false;
        }
        return expectedBatchId == 0 || snapshot.batchId() == expectedBatchId;
    }

    // ==================== Internal: static utilities ====================

    /** Locate the first server of a role in a route response (shared with the priority scheduling path). */
    public static ServerStatus findServer(Response response, RoleType roleType) {
        if (response.getServerStatus() == null) {
            return null;
        }
        for (ServerStatus serverStatus : response.getServerStatus()) {
            if (serverStatus != null && roleType == serverStatus.getRole()) {
                return serverStatus;
            }
        }
        return null;
    }

    /** Locate the selected Prefill endpoint, falling back to the fused P/D role. */
    public static ServerStatus findPrefillServer(Response response) {
        ServerStatus prefill = findServer(response, RoleType.PREFILL);
        return prefill != null ? prefill : findServer(response, RoleType.PDFUSION);
    }

    private static Response copyResponse(Response src) {
        Response response = new Response();
        response.setServerStatus(copyServerList(src.getServerStatus()));
        response.setSuccess(src.isSuccess());
        response.setCode(src.getCode());
        response.setErrorMessage(src.getErrorMessage());
        response.setRealMasterHost(src.getRealMasterHost());
        response.setQueueLength(src.getQueueLength());
        response.setEnqueuedByMaster(src.isEnqueuedByMaster());
        response.setAdmissionRejectReason(src.getAdmissionRejectReason());
        return response;
    }

    private static List<ServerStatus> copyServerList(List<ServerStatus> src) {
        if (src == null) {
            return null;
        }
        List<ServerStatus> result = new ArrayList<>(src.size());
        for (ServerStatus serverStatus : src) {
            result.add(copyOf(serverStatus));
        }
        return result;
    }

    /** Defensive copy of a route server status (shared with the priority scheduling path). */
    public static ServerStatus copyOf(ServerStatus src) {
        if (src == null) {
            return null;
        }
        ServerStatus status = new ServerStatus();
        status.setRole(src.getRole());
        status.setServerIp(src.getServerIp());
        status.setHttpPort(src.getHttpPort());
        status.setGrpcPort(src.getGrpcPort());
        status.setDpRank(src.getDpRank());
        status.setSelectedEngineIndex(src.getRoutingEngineIndex(),
                src.getEngineIndex() == null ? 1 : Math.max(2, src.getRoutingEngineIndex() + 1));
        status.setPrefillTime(src.getPrefillTime());
        status.setGroup(src.getGroup());
        status.setDebugInfo(copyOf(src.getDebugInfo()));
        status.setRequestId(src.getRequestId());
        status.setSuccess(src.isSuccess());
        status.setCode(src.getCode());
        status.setMessage(src.getMessage());
        return status;
    }

    private static DebugInfo copyOf(DebugInfo src) {
        if (src == null) {
            return null;
        }
        DebugInfo info = new DebugInfo();
        info.setRunningBatchSize(src.getRunningBatchSize());
        info.setQueueSize(src.getQueueSize());
        info.setWaitingTimeMs(src.getWaitingTimeMs());
        info.setAvailableKvCacheLen(src.getAvailableKvCacheLen());
        info.setEstimateTtftMs(src.getEstimateTtftMs());
        info.setEstimateTpotMs(src.getEstimateTpotMs());
        info.setHitCacheLen(src.getHitCacheLen());
        return info;
    }

    // ==================== Lifecycle ====================

    /**
     * Enter the delivery lifecycle without a monitor or per-group allocation.
     * The successful CAS is the before/after-shutdown linearization point.
     */
    private boolean tryAcquireDeliveryPermit() {
        while (true) {
            long state = deliveryLifecycle.get();
            if ((state & DELIVERY_LIFECYCLE_CLOSED) != 0) {
                return false;
            }
            long active = state & DELIVERY_LIFECYCLE_COUNT_MASK;
            if (active == DELIVERY_LIFECYCLE_COUNT_MASK) {
                throw new IllegalStateException("delivery lifecycle permit count overflow");
            }
            if (deliveryLifecycle.compareAndSet(state, state + 1)) {
                return true;
            }
        }
    }

    private void releaseDeliveryPermit() {
        while (true) {
            long state = deliveryLifecycle.get();
            long active = state & DELIVERY_LIFECYCLE_COUNT_MASK;
            if (active == 0) {
                throw new IllegalStateException("delivery lifecycle permit underflow");
            }
            long next = state - 1;
            if (!deliveryLifecycle.compareAndSet(state, next)) {
                continue;
            }
            if ((next & DELIVERY_LIFECYCLE_CLOSED) != 0
                    && (next & DELIVERY_LIFECYCLE_COUNT_MASK) == 0) {
                synchronized (deliveryDrainMonitor) {
                    deliveryDrainMonitor.notifyAll();
                }
            }
            return;
        }
    }

    /** Close new delivery work, then wait for operations which crossed the gate. */
    private void closeDeliveryLifecycleAndAwait() {
        while (true) {
            long state = deliveryLifecycle.get();
            if ((state & DELIVERY_LIFECYCLE_CLOSED) != 0
                    || deliveryLifecycle.compareAndSet(
                    state, state | DELIVERY_LIFECYCLE_CLOSED)) {
                break;
            }
        }

        boolean interrupted = false;
        synchronized (deliveryDrainMonitor) {
            while ((deliveryLifecycle.get() & DELIVERY_LIFECYCLE_COUNT_MASK) != 0) {
                try {
                    deliveryDrainMonitor.wait();
                } catch (InterruptedException shutdownInterrupted) {
                    // Endpoint close cannot race an already-committed delivery.
                    // Finish the deterministic drain, then restore the signal.
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    int activeDeliveryPermitCount() {
        return Math.toIntExact(
                deliveryLifecycle.get() & DELIVERY_LIFECYCLE_COUNT_MASK);
    }

    boolean isDeliveryLifecycleClosed() {
        return (deliveryLifecycle.get() & DELIVERY_LIFECYCLE_CLOSED) != 0;
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    public void reportBatchMetrics() {
        if (shuttingDown.get()) {
            return;
        }
        reporter.reportSchedulerInflightSize(inflight.size());

        // Per-worker metrics: prefill endpoints
        for (Map.Entry<String, PrefillEndpoint> entry : endpointRegistry.getPrefillEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }

        // Per-worker metrics: decode endpoints
        for (Map.Entry<String, DecodeEndpoint> entry : endpointRegistry.getDecodeEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }

        // PRIORITY: per-endpoint admission queue/resource gauges.
        if (admissionScheduler != null
                && configService.loadBalanceConfig().isPriorityOrdering()) {
            admissionScheduler.reportPrefillQueueDepths();
            admissionScheduler.reportDecodeAdmissionGauges();
        }
    }

    @PreDestroy
    public void shutdown() {
        if (!shuttingDown.compareAndSet(false, true)) {
            return;
        }
        // Close the permit CAS before draining generations. A submit which
        // crossed the boolean check cannot reserve after this linearization
        // point; generation release remains exact-idempotent below.
        outstandingRequestCount.getAndSet(OUTSTANDING_ADMISSION_CLOSED);
        for (RequestGenerationGate generation : generationGates.values()) {
            generation.releaseOutstandingPermit();
        }
        closeDeliveryLifecycleAndAwait();
        completeOutstandingRequestsForShutdown();
        requestExpirationTimer.shutdownNow();
        engineFenceRetryTimer.shutdownNow();
        for (InflightEntry entry : inflight.values()) {
            synchronized (entry) {
                if (entry.engineFence != null) {
                    entry.engineFence.cancelScheduledReconciliation();
                }
            }
        }
        try {
            endpointRegistry.close();
        } finally {
            responseCompletionExecutor.shutdown();
            quarantinedEngineFences.clear();
            quarantinedProbeQueue.clear();
        }
    }

    /**
     * Complete requests which have no earlier response-publication owner.
     *
     * <p>A Cancel/deadline closes {@link RequestGenerationGate} before its
     * Engine proof arrives, so ordinary {@code future.complete(...)} is
     * intentionally rejected. Shutdown is different: it cannot wait for a
     * retry timer which it is about to stop. Claim live-entry publication
     * under the entry monitor, then use the reducer-owned completion path
     * outside all scheduler locks. Existing delivery/terminal publications
     * retain precedence.</p>
     */
    private void completeOutstandingRequestsForShutdown() {
        String detail = "priority scheduler is shutting down";
        List<ResponseCompletion> publications = new ArrayList<>();
        // Registered requests are authoritative even when their caller did
        // not originate from submit() (for example an InflightRegistrar
        // integration). Do not make shutdown publication depend on the
        // presence or concrete type of the generation gate.
        for (InflightEntry entry : inflight.values()) {
            String requestId = entry.item.requestId();
            synchronized (entry) {
                if (inflight.get(requestId) == entry
                        && !entry.responseCompletionClaimed) {
                    ResponseCompletion publication = errorPublicationLocked(
                            entry,
                            StrategyErrorType.BATCH_DISPATCH_FAILED,
                            detail);
                    if (publication != null) {
                        publications.add(publication);
                    }
                }
            }
        }
        for (Map.Entry<String, RequestGenerationGate> generationEntry
                : generationGates.entrySet()) {
            String requestId = generationEntry.getKey();
            RequestGenerationGate generation = generationEntry.getValue();
            InflightEntry entry = inflight.get(requestId);
            if (entry != null && entry.item.future() == generation) {
                synchronized (entry) {
                    if (inflight.get(requestId) == entry
                            && !entry.responseCompletionClaimed) {
                        ResponseCompletion publication = errorPublicationLocked(
                                entry,
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
                                detail);
                        if (publication != null) {
                            publications.add(publication);
                        }
                    }
                }
            } else if (!terminalStates.containsKey(requestId)
                    && !generation.isDone()) {
                publications.add(ResponseCompletion.terminal(
                        generation,
                        buildErrorResponse(
                                StrategyErrorType.BATCH_DISPATCH_FAILED,
                                detail)));
            }
            if (!generation.admissionMutationInProgress) {
                generationGates.remove(requestId, generation);
            }
        }
        for (ResponseCompletion publication : publications) {
            submitResponseCompletion(publication);
        }
    }

    // ==================== Inflight entry ====================

    record CompletionExecutorPolicy(int workerCount, int queueCapacity) {
        CompletionExecutorPolicy {
            if (workerCount < 1 || queueCapacity < 1) {
                throw new IllegalArgumentException("completion executor bounds must be positive");
            }
        }

        static CompletionExecutorPolicy productionDefaults() {
            return new CompletionExecutorPolicy(
                    DEFAULT_COMPLETION_WORKERS, DEFAULT_COMPLETION_QUEUE_CAPACITY);
        }
    }

    record CompletionExecutorSnapshot(int workerLimit,
                                      int queueCapacity,
                                      int queueSize,
                                      int largestPoolSize,
                                      long completedTaskCount,
                                      boolean shutdown) {
    }

    /**
     * Internal reconciliation policy, intentionally not another live config
     * surface. Production performs eight quick probes: exponential delays from
     * 100 ms capped at 5 s (11.3 s total delay), plus the internal Cancel ACK
     * bound for each invocation. Afterwards the request moves to the
     * minute-level quarantine sweep.
     */
    record EngineFencePolicy(int maxFastAttempts,
                             long initialRetryDelayMs,
                             long maxRetryDelayMs,
                             int maxProbesPerCleanup) {
        private static final int PRODUCTION_FAST_ATTEMPTS = 8;
        private static final long PRODUCTION_INITIAL_RETRY_DELAY_MS = 100L;
        private static final long PRODUCTION_MAX_RETRY_DELAY_MS = 5_000L;
        private static final int PRODUCTION_MAX_PROBES_PER_CLEANUP = 64;

        EngineFencePolicy {
            if (maxFastAttempts < 1 || initialRetryDelayMs < 1
                    || maxRetryDelayMs < initialRetryDelayMs
                    || maxProbesPerCleanup < 1) {
                throw new IllegalArgumentException("invalid Engine fence policy");
            }
        }

        static EngineFencePolicy productionDefaults() {
            return new EngineFencePolicy(
                    PRODUCTION_FAST_ATTEMPTS,
                    PRODUCTION_INITIAL_RETRY_DELAY_MS,
                    PRODUCTION_MAX_RETRY_DELAY_MS,
                    PRODUCTION_MAX_PROBES_PER_CLEANUP);
        }

        long retryDelayMs(int completedAttempt) {
            int shift = Math.min(Math.max(0, completedAttempt), 30);
            long scaled;
            try {
                scaled = Math.multiplyExact(initialRetryDelayMs, 1L << shift);
            } catch (ArithmeticException overflow) {
                scaled = Long.MAX_VALUE;
            }
            return Math.min(maxRetryDelayMs, scaled);
        }
    }

    /** Request id plus exact fence generation for lazy stale-FIFO rejection. */
    private record EngineFenceProbeRef(String requestId,
                                       EngineFenceRegistration registration) {
    }

    private static final class InflightEntry {
        final BatchItem item;
        final RequestLifecycle lifecycle;
        final boolean priorityAdmission;
        final StrategyErrorType deadlineErrorType;
        final AtomicBoolean rolledBack = new AtomicBoolean(false);
        AdmissionLease admissionLease;
        EngineOwnershipState engineOwnershipState = EngineOwnershipState.DECODE_PENDING;
        boolean cleanupOwned;
        /** First state-machine transition which owns completion of {@code item.future()}. */
        boolean responseCompletionClaimed;
        /** Exact owner of a non-successful public-future terminal when no lease is attached. */
        boolean externalFutureTerminalClaimed;
        PreemptionRegistration preemption;
        EngineFenceRegistration engineFence;
        /** Non-null only when the frontend-facing Cancel reducer won first cause. */
        CancelReason cancellationReason;

        InflightEntry(BatchItem item, boolean priorityAdmission) {
            this.item = Objects.requireNonNull(item);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
            this.lifecycle = new RequestLifecycle(item.requestId());
            this.priorityAdmission = priorityAdmission;
            this.deadlineErrorType = item.future() instanceof RequestGenerationGate generation
                    ? generation.deadlineErrorType
                    : StrategyErrorType.BATCH_SLO_EXPIRED;
        }

        public long createdAtMs() {
            return lifecycle.snapshot().createdAtMs();
        }

        boolean hasPreemption(long attemptToken) {
            return preemption != null && preemption.attemptToken == attemptToken;
        }
    }

    /**
     * Request-local generation future. Direct completions claim the gate;
     * reducer-owned publications use {@link #completeOwned(Response)} after
     * they have already claimed it under this monitor.
     */
    private static final class RequestGenerationGate extends CompletableFuture<Response> {
        private boolean closed;
        private volatile boolean admissionMutationInProgress;
        private volatile StrategyErrorType deadlineErrorType =
                StrategyErrorType.BATCH_SLO_EXPIRED;
        private RequestLifecycle pendingAdmissionCancellation;
        private CancelReason pendingAdmissionCancelReason;
        private AtomicInteger outstandingCounter;
        private boolean outstandingPermitReleaseRequested;
        private boolean outstandingPermitReleased;
        /** Exact-once guard for a response whose reducer already owns publication. */
        private boolean ownedCompletionPublished;

        /**
         * Bind the slot already reserved by the scheduler. Completion or
         * shutdown may win before this method; in that case binding performs
         * the deferred release and tells submit to stop immediately.
         */
        private synchronized boolean bindOutstandingPermit(AtomicInteger counter) {
            if (outstandingCounter != null) {
                throw new IllegalStateException("outstanding permit already bound");
            }
            outstandingCounter = Objects.requireNonNull(counter);
            if (outstandingPermitReleaseRequested || isDone()) {
                releaseOutstandingPermitLocked();
                return false;
            }
            return true;
        }

        /** One terminal/reconciliation entry point for every permit owner. */
        private synchronized void releaseOutstandingPermit() {
            outstandingPermitReleaseRequested = true;
            releaseOutstandingPermitLocked();
        }

        private void releaseOutstandingPermitLocked() {
            if (outstandingCounter == null || outstandingPermitReleased) {
                return;
            }
            outstandingPermitReleased = true;
            while (true) {
                int current = outstandingCounter.get();
                if (current == OUTSTANDING_ADMISSION_CLOSED) {
                    return;
                }
                if (current <= 0) {
                    throw new IllegalStateException(
                            "outstanding request permit counter underflow");
                }
                if (outstandingCounter.compareAndSet(current, current - 1)) {
                    return;
                }
            }
        }

        private boolean isOpen() {
            return !closed && !isDone();
        }

        private boolean closeCommits() {
            if (!isOpen()) {
                return false;
            }
            closed = true;
            return true;
        }

        @Override
        public boolean complete(Response response) {
            synchronized (this) {
                if (!closeCommits()) {
                    return false;
                }
            }
            return super.complete(response);
        }

        @Override
        public boolean completeExceptionally(Throwable error) {
            synchronized (this) {
                if (!closeCommits()) {
                    return false;
                }
            }
            return super.completeExceptionally(error);
        }

        @Override
        public boolean cancel(boolean mayInterruptIfRunning) {
            synchronized (this) {
                if (!closeCommits()) {
                    return false;
                }
            }
            return super.cancel(mayInterruptIfRunning);
        }

        private boolean completeOwned(Response response) {
            return completeOwned(response, null);
        }

        /**
         * Publish a reducer-owned response without trying to claim the gate a
         * second time. A {@link ResponseCompletion} exists only after either
         * the request entry claimed response publication or the gate was
         * closed by the winning pre-registration reducer. Requiring
         * {@link #closeCommits()} again would reject that winner and leave the
         * public future incomplete.
         *
         * <p>The exact-once bit also preserves the response-claim ordering
         * when a later Cancel closes the gate before the asynchronous
         * publication runs. Prefill-only lease retirement remains ordered
         * before the public success becomes observable.</p>
         */
        private boolean completeOwned(Response response,
                                      AdmissionLease prefillOnlyLease) {
            synchronized (this) {
                if (ownedCompletionPublished || isDone()) {
                    return false;
                }
                ownedCompletionPublished = true;
                closed = true;
            }
            if (prefillOnlyLease != null) {
                prefillOnlyLease.markDeliverySucceeded();
            }
            return super.complete(response);
        }
    }

    /** Linearized ownership decision between delivery and Decode WorkerStatus. */
    private enum EngineOwnershipState {
        DECODE_PENDING,
        DECODE_OWNED,
        TERMINAL
    }

    private enum PreemptionRegistrationState {
        CLAIMED,
        CANCEL_IN_FLIGHT,
        CANCEL_REQUESTED,
        NOT_FOUND_STALE,
        CANCEL_UNKNOWN,
        SETTLED
    }

    private enum EngineFenceCause {
        BATCH_ACK_UNCERTAIN,
        POST_DELIVERY_RECONCILIATION
    }

    private enum EngineFenceState {
        /** Installed, but no Cancel invocation has crossed the entry fence. */
        ARMED,
        /** One asynchronous Cancel invocation may have installed intent. */
        CANCEL_IN_FLIGHT,
        /** The last acknowledgement was non-terminal; one retry is scheduled. */
        RETRY_WAIT,
        /** Fast retries exhausted; only the minute-level bounded sweep probes it. */
        QUARANTINED
    }

    /** Request-scoped Cancel owner, always mutated under its InflightEntry. */
    private static final class EngineFenceRegistration {
        private final EngineFenceCause cause;
        private final String detail;
        /** Original priority attempt token, or zero when no claim was transferred. */
        private final long transferredPreemptionToken;
        private final EngineFenceResources resources;
        private EngineFenceState state = EngineFenceState.ARMED;
        private long quarantinedAtMs;
        private ScheduledFuture<?> scheduledReconciliation;

        private EngineFenceRegistration(EngineFenceCause cause,
                                        String detail,
                                        long transferredPreemptionToken,
                                        EngineFenceResources resources) {
            this.cause = Objects.requireNonNull(cause);
            this.detail = Objects.requireNonNull(detail);
            this.transferredPreemptionToken = transferredPreemptionToken;
            this.resources = Objects.requireNonNull(resources);
        }

        private boolean beginCancelAttempt(boolean quarantineProbe) {
            if (state != EngineFenceState.ARMED
                    && state != EngineFenceState.RETRY_WAIT
                    && !(quarantineProbe && state == EngineFenceState.QUARANTINED)) {
                return false;
            }
            state = EngineFenceState.CANCEL_IN_FLIGHT;
            return true;
        }

        private boolean canScheduleReconciliation() {
            return scheduledReconciliation == null
                    && (state == EngineFenceState.ARMED
                    || state == EngineFenceState.RETRY_WAIT);
        }

        private void installScheduledReconciliation(ScheduledFuture<?> scheduled) {
            if (!canScheduleReconciliation()) {
                scheduled.cancel(false);
                throw new IllegalStateException(
                        "Engine fence reconciliation scheduled from invalid state " + state);
            }
            scheduledReconciliation = Objects.requireNonNull(scheduled);
        }

        private boolean consumeScheduledReconciliation() {
            if (scheduledReconciliation == null) {
                return false;
            }
            scheduledReconciliation = null;
            return true;
        }

        private void cancelScheduledReconciliation() {
            ScheduledFuture<?> scheduled = scheduledReconciliation;
            scheduledReconciliation = null;
            if (scheduled != null) {
                scheduled.cancel(false);
            }
        }

        private void awaitRetry() {
            if (state != EngineFenceState.CANCEL_IN_FLIGHT) {
                throw new IllegalStateException(
                        "Engine fence retry from invalid state " + state);
            }
            state = EngineFenceState.RETRY_WAIT;
        }

        private boolean enterQuarantine(long nowMs) {
            if (state != EngineFenceState.CANCEL_IN_FLIGHT) {
                return false;
            }
            if (quarantinedAtMs == 0) {
                quarantinedAtMs = nowMs;
            }
            state = EngineFenceState.QUARANTINED;
            return true;
        }

        private void returnToQuarantine() {
            if (state != EngineFenceState.CANCEL_IN_FLIGHT || quarantinedAtMs == 0) {
                throw new IllegalStateException(
                        "Engine fence probe returned from invalid state " + state);
            }
            state = EngineFenceState.QUARANTINED;
        }

        private boolean wasQuarantined() {
            return quarantinedAtMs != 0;
        }

        private boolean isQuarantined() {
            return state == EngineFenceState.QUARANTINED;
        }

        private boolean cancelMayHaveBeenInstalled() {
            return state != EngineFenceState.ARMED;
        }
    }

    /**
     * One request-scoped ownership handle for every endpoint ledger protected
     * by an Engine fence. Endpoint begin/end operations are idempotent and
     * execute under their existing request stripe/admission lock; this handle
     * prevents the scheduler from scattering protection booleans across its
     * state machine.
     */
    private static final class EngineFenceResources {
        private final String requestId;
        private final PrefillEndpoint prefill;
        private final DecodeEndpoint decode;
        private final long batchId;
        private final boolean batchMemberProtected;
        private final boolean prefillProtected;
        private final boolean decodeProtected;
        private boolean released;

        private EngineFenceResources(String requestId,
                                     PrefillEndpoint prefill,
                                     DecodeEndpoint decode,
                                     long batchId,
                                     boolean batchMemberProtected,
                                     boolean prefillProtected,
                                     boolean decodeProtected) {
            this.requestId = requestId;
            this.prefill = prefill;
            this.decode = decode;
            this.batchId = batchId;
            this.batchMemberProtected = batchMemberProtected;
            this.prefillProtected = prefillProtected;
            this.decodeProtected = decodeProtected;
        }

        private static EngineFenceResources acquire(BatchItem item,
                                                    long batchId,
                                                    boolean batchMemberProtected) {
            PrefillEndpoint prefill = item.prefillEp();
            DecodeEndpoint decode = item.decodeEp();
            boolean prefillProtected = false;
            boolean decodeProtected = false;
            try {
                prefillProtected = prefill != null
                        && prefill.beginEngineFenceProtection(item.requestId());
                decodeProtected = decode != null
                        && decode.beginEngineFenceProtection(item.requestId());
                return new EngineFenceResources(item.requestId(), prefill, decode,
                        batchId, batchMemberProtected,
                        prefillProtected, decodeProtected);
            } catch (RuntimeException | Error protectionFailure) {
                try {
                    if (decodeProtected && decode != null) {
                        decode.endEngineFenceProtection(item.requestId());
                    }
                } catch (RuntimeException | Error cleanupFailure) {
                    protectionFailure.addSuppressed(cleanupFailure);
                }
                try {
                    if (prefillProtected && prefill != null) {
                        prefill.endEngineFenceProtection(item.requestId());
                    }
                } catch (RuntimeException | Error cleanupFailure) {
                    protectionFailure.addSuppressed(cleanupFailure);
                }
                try {
                    if (batchMemberProtected && prefill != null) {
                        prefill.releaseBatchMemberProtection(
                                batchId, item.requestId());
                    }
                } catch (RuntimeException | Error cleanupFailure) {
                    protectionFailure.addSuppressed(cleanupFailure);
                }
                throw protectionFailure;
            }
        }

        /** Called only under the owning InflightEntry; idempotent defensively. */
        private void release() {
            if (released) {
                return;
            }
            released = true;
            try {
                if (decodeProtected && decode != null) {
                    decode.endEngineFenceProtection(requestId);
                }
            } finally {
                try {
                    if (prefillProtected && prefill != null) {
                        prefill.endEngineFenceProtection(requestId);
                    }
                } finally {
                    if (batchMemberProtected && prefill != null) {
                        prefill.releaseBatchMemberProtection(batchId, requestId);
                    }
                }
            }
        }
    }

    private static final class PreemptionRegistration {
        private final long attemptToken;
        private final String detail;
        private final CompletableFuture<PriorityCanceledObservation> priorityCanceled =
                new CompletableFuture<>();
        private PreemptionRegistrationState state = PreemptionRegistrationState.CLAIMED;
        private DeferredTerminal pendingTerminal;
        private boolean pendingDeliveryConfirmation;
        private long pendingConfirmationBatchId;
        private String postDeliveryFenceDetail;

        private PreemptionRegistration(long attemptToken, String detail) {
            this.attemptToken = attemptToken;
            this.detail = detail;
        }
    }

    /** Immutable action created under the entry lock and completed after the lock is released. */
    private record ResponseCompletion(CompletableFuture<Response> future,
                                       Response response,
                                       BatchItem item,
                                       AdmissionLease prefillOnlyLease,
                                       DeliveryClaimKind deliveryClaimKind,
                                       long batchId,
                                       BatchSchedulerReporter reporter,
                                       String prefillIp,
                                       long batchEnqueueAckLatencyMs) {
        private static ResponseCompletion success(BatchItem item,
                                                   Response response,
                                                   AdmissionLease prefillOnlyLease,
                                                   DeliveryClaimKind deliveryClaimKind,
                                                   long batchId,
                                                   BatchSchedulerReporter reporter,
                                                   String prefillIp,
                                                   long batchEnqueueAckLatencyMs) {
            return new ResponseCompletion(
                    item.future(), response, item, prefillOnlyLease,
                    deliveryClaimKind, batchId,
                    reporter, prefillIp, batchEnqueueAckLatencyMs);
        }

        private static ResponseCompletion terminal(CompletableFuture<Response> future,
                                                    Response response) {
            return new ResponseCompletion(future, response, null, null, null,
                    0, null, null, -1);
        }
    }

    private record WorkerTerminalObservation(boolean prefill,
                                             long batchId,
                                             long errorCode) {
        boolean isTerminal() {
            return !prefill || errorCode != 0;
        }
    }

    /** Immutable result of worker-status reduction, published after locks are released. */
    private record WorkerStatusPublication(
            ResponseCompletion completion,
            CompletableFuture<PriorityCanceledObservation> priorityCanceledSignal,
            PriorityCanceledObservation priorityCanceledObservation) {
        private static final WorkerStatusPublication NONE =
                new WorkerStatusPublication(null, null, null);

        private static WorkerStatusPublication completion(ResponseCompletion completion) {
            return completion == null ? NONE
                    : new WorkerStatusPublication(completion, null, null);
        }

        private static WorkerStatusPublication priorityCanceled(
                CompletableFuture<PriorityCanceledObservation> signal,
                PriorityCanceledObservation observation) {
            return new WorkerStatusPublication(null, signal, observation);
        }
    }

    private enum DeferredTerminalKind {
        ADMISSION_CLEANUP,
        FAILURE,
        TIMEOUT,
        WORKER
    }

    /** First ordinary terminal observed while priority Cancel owns the entry. */
    private record DeferredTerminal(DeferredTerminalKind kind,
                                    StrategyErrorType errorType,
                                    String detail,
                                    boolean releasePrefillAccounting,
                                    WorkerTerminalObservation workerObservation) {
        static DeferredTerminal admissionCleanup(String detail) {
            return new DeferredTerminal(DeferredTerminalKind.ADMISSION_CLEANUP,
                    null, detail, false, null);
        }

        static DeferredTerminal failure(StrategyErrorType errorType,
                                        String detail,
                                        boolean releasePrefillAccounting) {
            return new DeferredTerminal(DeferredTerminalKind.FAILURE,
                    Objects.requireNonNull(errorType), detail,
                    releasePrefillAccounting, null);
        }

        static DeferredTerminal timeout(String detail) {
            return new DeferredTerminal(DeferredTerminalKind.TIMEOUT,
                    StrategyErrorType.BATCH_SLO_EXPIRED, detail, true, null);
        }

        static DeferredTerminal worker(WorkerTerminalObservation observation) {
            return new DeferredTerminal(DeferredTerminalKind.WORKER,
                    null, null, false, Objects.requireNonNull(observation));
        }

        boolean authoritativeWorker() {
            return kind == DeferredTerminalKind.WORKER;
        }

        boolean deliveryFailure() {
            return (kind == DeferredTerminalKind.FAILURE && releasePrefillAccounting)
                    || kind == DeferredTerminalKind.TIMEOUT;
        }
    }
}
