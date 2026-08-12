package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.scheduler.priority.AdmissionFailure;
import org.flexlb.balance.scheduler.priority.AdmissionFailureClassifier;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.InflightRegistrar;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
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
import org.flexlb.balance.scheduler.priority.InflightRegistrar.PriorityCanceledObservation;
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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Coordinates batch scheduling for FlexLB disaggregated inference.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Request admission and routing</li>
 *   <li>Inflight lifecycle management (inflight map, TTL cleanup)</li>
 *   <li>Batch assembly coordination — commits to PrefillEndpoint,
 *       delegates gRPC dispatch to {@link BatchDispatcher}</li>
 *   <li>Resource rollback on failure or completion</li>
 * </ul>
 *
 * <p>The actual gRPC dispatch (build protobuf, send, parse response) is
 * delegated to {@link BatchDispatcher}. Per-item results come back through
 * {@link DispatchCallback} which this class implements.
 */
@Component
public class FlexlbBatchScheduler implements BatchDecisionHandler, DispatchCallback, InflightRegistrar {

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchDispatcher dispatcher;
    private final BatchSchedulerReporter reporter;
    private final PriorityAdmissionScheduler priorityScheduler;
    private final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    private final Map<Long, RequestLifecycleSnapshot> terminalStates = new ConcurrentHashMap<>();
    private final BatchIdGenerator batchIdGenerator;

    @Autowired
    public FlexlbBatchScheduler(ConfigService configService,
                                Router router,
                                EndpointRegistry endpointRegistry,
                                BatchDispatcher dispatcher,
                                BatchSchedulerReporter reporter,
                                PriorityAdmissionScheduler priorityScheduler,
                                Environment environment) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.dispatcher = dispatcher;
        this.reporter = reporter;
        this.priorityScheduler = priorityScheduler;
        // Initialize Snowflake batch ID generator with master identity
        this.batchIdGenerator = new BatchIdGenerator(detectLocalIp(), detectPort(environment));
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

    // ==================== Request submission ====================

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            if (ctx == null || ctx.getRequest() == null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST, null);
                return future;
            }

            if (inflight.containsKey(ctx.getRequestId()) || terminalStates.containsKey(ctx.getRequestId())) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }

            int maxInflight = configService.loadBalanceConfig().getFlexlbBatchMaxInflight();
            if (maxInflight > 0 && inflight.size() >= maxInflight) {
                if (configService.loadBalanceConfig().isAutoTpmEnabled()) {
                    Response response = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                            AdmissionRejectReason.RESOURCE_EXHAUSTED);
                    response.setErrorMessage(StrategyErrorType.RESOURCE_EXHAUSTED
                            .buildErrorMessage("master inflight capacity exhausted"));
                    future.complete(response);
                } else {
                    completeError(future, StrategyErrorType.QUEUE_FULL, null);
                }
                return future;
            }

            // Auto-TPM priority path: delegate plan/commit to the priority
            // scheduler. Disabled by default — the legacy path below is
            // byte-for-byte unchanged when the switch is off.
            // normalize() always assigns 1-100, so every request participates
            // when Auto-TPM is enabled; no separate hasPriority gate needed.
            if (configService.loadBalanceConfig().isAutoTpmEnabled() && priorityScheduler != null) {
                priorityScheduler.schedule(ctx, future, this);
                // The deadline is an ordinary terminal event in the same
                // reducer as dispatch/worker failures. It must compete with a
                // priority-preemption claim before completing the public future.
                attachAdmissionTimeout(ctx, future);
                return future;
            }

            Response routeResponse = router.route(ctx);
            if (routeResponse == null || !routeResponse.isSuccess()) {
                if (routeResponse != null) {
                    future.complete(routeResponse);
                } else {
                    completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER, null);
                }
                return future;
            }

            ServerStatus prefill = findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
            if (prefill == null) {
                rollback(routeResponse);
                completeError(future, StrategyErrorType.NO_PREFILL_WORKER, null);
                return future;
            }

            String prefillIpPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
            PrefillEndpoint prefillEp = endpointRegistry.getPrefill(prefillIpPort);
            if (prefillEp == null) {
                rollback(routeResponse);
                completeError(future, StrategyErrorType.NO_PREFILL_WORKER, null);
                return future;
            }

            DecodeEndpoint decodeEp = null;
            if (decode != null) {
                String decodeIpPort = decode.getServerIp() + ":" + decode.getHttpPort();
                decodeEp = endpointRegistry.getDecode(decodeIpPort);
            }

            BatchItem item = new BatchItem(ctx, future, routeResponse, copyOf(prefill), copyOf(decode),
                    prefillEp, decodeEp, System.currentTimeMillis());
            InflightEntry entry = new InflightEntry(item, false);
            InflightEntry existing = inflight.putIfAbsent(ctx.getRequestId(), entry);
            if (existing != null || terminalStates.containsKey(ctx.getRequestId())) {
                if (existing == null) {
                    inflight.remove(ctx.getRequestId(), entry);
                }
                rollback(item);
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }
            WorkerBatcher batcher = prefillEp.getBatcher();
            ctx.setRouteSubmittedNanos(System.nanoTime());
            batcher.offer(item);

            // Report route+submit time: from schedule() entry (ctx.startTime) to batcher offer completion
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    prefillEp.getIp(),
                    System.currentTimeMillis() - ctx.getStartTime());
        } catch (Throwable t) {
            if (ctx != null) {
                inflight.remove(ctx.getRequestId());
            }
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Submit failed: " + t.getMessage());
        }
        return future;
    }

    /**
     * Schedule the admission deadline as a reducer event. Directly attaching
     * {@link CompletableFuture#orTimeout(long, TimeUnit)} would let the timer
     * permanently complete the frontend future while a priority Cancel owns
     * the request; a later authoritative CANCELED observation could then no
     * longer publish PRIORITY_PREEMPTED. The legacy submit path does not
     * attach this admission timer and keeps its existing TTL cleanup.
     */
    private void attachAdmissionTimeout(BalanceContext ctx,
                                        CompletableFuture<Response> future) {
        if (ctx.budget() == null) {
            return;
        }
        long remainingMs = ctx.budget().remainingMs(System.currentTimeMillis());
        long delayMs = Math.max(1, remainingMs);
        long requestId = ctx.getRequestId();
        CompletableFuture.delayedExecutor(delayMs, TimeUnit.MILLISECONDS)
                .execute(() -> onAdmissionDeadline(requestId, future));
    }

    /** Deliver one admission deadline through the ordinary-terminal reducer. */
    private void onAdmissionDeadline(long requestId,
                                     CompletableFuture<Response> expectedFuture) {
        // The future monitor is the admission handoff fence shared with the
        // asynchronous Decode-preemption COMMITTED callback. It makes the
        // pre-inflight deadline decision atomic with register+offer without
        // introducing another lifecycle state machine.
        synchronized (expectedFuture) {
            if (expectedFuture.isDone()) {
                return;
            }
            InflightEntry entry = inflight.get(requestId);
            if (entry == null) {
                // Decode preemption is asynchronous: until it commits, the
                // incoming owns a provisional Decode reservation but has no
                // inflight entry yet. Closing the public admission gate here
                // lets the coordinator abort that reservation instead of
                // admitting a request after its deadline.
                completeAdmissionError(expectedFuture,
                        AdmissionFailure.resourceExhausted(),
                        "admission deadline exceeded before inflight registration");
                return;
            }
            if (entry.item.future() != expectedFuture) {
                return;
            }
            synchronized (entry) {
                if (inflight.get(requestId) != entry || expectedFuture.isDone()) {
                    return;
                }
                reduceOrdinaryTerminalLocked(entry,
                        DeferredTerminal.timeout("admission deadline exceeded"));
            }
        }
    }

    // ==================== InflightRegistrar (Auto-TPM commit protocol) ====================

    /**
     * Register an Auto-TPM admitted item into the same inflight tracking as
     * the legacy path, so dispatch/completion/TTL/rollback behave identically.
     * Mirrors the duplicate handoff check in {@link #submit}.
     */
    @Override
    public boolean registerInflight(BatchItem item) {
        // This registrar is the Auto-TPM commit boundary. Legacy submit()
        // constructs its entry directly with autoTpmAdmission=false.
        InflightEntry entry = new InflightEntry(item, true);
        InflightEntry existing = inflight.putIfAbsent(item.requestId(), entry);
        if (existing != null || terminalStates.containsKey(item.requestId())) {
            if (existing == null) {
                inflight.remove(item.requestId(), entry);
            }
            return false;
        }
        return true;
    }

    @Override
    public void unregisterInflight(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        if (entry != null && entry.item == item) {
            inflight.remove(item.requestId(), entry);
        }
    }

    @Override
    public boolean registrarOwnsAdmissionCleanup(BatchItem item, String detail) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            // The release sequence is idempotent. An already-removed entry has
            // no live claim that this lease could violate.
            return false;
        }
        synchronized (entry) {
            if (inflight.get(item.requestId()) != entry) {
                return false;
            }
            if (entry.cleanupOwned) {
                return true;
            }
            DeferredTerminal cleanup = DeferredTerminal.admissionCleanup(detail);
            if (entry.preemption != null) {
                deferOrdinaryTerminalLocked(entry, cleanup);
                return true;
            }
            // This flag closes the small window between this decision and the
            // lease's idempotent Decode release/unregister calls.
            entry.cleanupOwned = true;
            return false;
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
    public void finishPreemptedById(long requestId, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry != null) {
            finishPreempted(entry.item, detail);
        } else {
            // N1: a settle on an unknown id is harmless (already terminal or
            // never registered) but worth surfacing — a burst points at a
            // registration/cleanup race.
            Logger.warn("finishPreemptedById miss: request_id={} not inflight, detail={}",
                    requestId, detail);
            // P2-2: surface the miss as a metric too — warn logs alone are
            // not alertable.
            if (priorityScheduler != null) {
                priorityScheduler.onInflightSettleMiss("preempted");
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
    public void finishYieldedById(long requestId, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry != null) {
            finishYielded(entry.item, detail);
        } else {
            // N1: same rationale as finishPreemptedById — no-op, but observable.
            Logger.warn("finishYieldedById miss: request_id={} not inflight, detail={}",
                    requestId, detail);
            // P2-2: metric alongside the warn log.
            if (priorityScheduler != null) {
                priorityScheduler.onInflightSettleMiss("yielded");
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
            synchronized (entry) {
                reduceOrdinaryTerminalLocked(entry,
                        DeferredTerminal.failure(errorType, detail, false));
            }
        } else if (!victim.future().isDone() && !terminalStates.containsKey(victim.requestId())) {
            rollback(victim);
            completeError(victim.future(), errorType, detail);
        }
    }

    @Override
    public boolean claimForPreemption(long requestId, long attemptToken, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (inflight.get(requestId) != entry) {
                return false;
            }
            if (entry.cleanupOwned || entry.preemption != null || entry.lifecycle.isTerminal()) {
                return false;
            }
            entry.preemption = new PreemptionRegistration(attemptToken, detail);
            return true;
        }
    }

    @Override
    public boolean releasePreemptionClaim(long requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (inflight.get(requestId) != entry || !entry.hasPreemption(attemptToken)
                    || (entry.preemption.state != PreemptionRegistrationState.CLAIMED
                        && entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_IN_FLIGHT)) {
                return false;
            }
            PreemptionRegistration registration = entry.preemption;
            entry.preemption = null;
            replayAfterReleasedClaimLocked(entry, registration);
            return true;
        }
    }

    @Override
    public boolean markPreemptionCancelInFlight(long requestId, long attemptToken) {
        return transitionPreemption(requestId, attemptToken,
                PreemptionRegistrationState.CLAIMED,
                PreemptionRegistrationState.CANCEL_IN_FLIGHT, false);
    }

    @Override
    public boolean markPreemptionCancelAccepted(long requestId, long attemptToken) {
        return transitionPreemption(requestId, attemptToken,
                PreemptionRegistrationState.CANCEL_IN_FLIGHT,
                PreemptionRegistrationState.CANCEL_REQUESTED, true);
    }

    @Override
    public boolean markPreemptionNotFound(long requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)
                    || entry.preemption.state
                        != PreemptionRegistrationState.CANCEL_IN_FLIGHT) {
                return false;
            }
            entry.preemption.state = PreemptionRegistrationState.NOT_FOUND_STALE;
            // A terminal delta is incremental and may never be sent again.
            // NOT_FOUND proves the priority intent was not installed, so the
            // first cached ordinary outcome resumes its original path.
            replayAfterNegativeCancelLocked(entry, attemptToken, false);
        }
        return true;
    }

    @Override
    public boolean markPreemptionUnknown(long requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
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
            replayAfterNegativeCancelLocked(entry, attemptToken, true);
        }
        return true;
    }

    @Override
    public CompletableFuture<PriorityCanceledObservation> priorityCanceledSignal(
            long requestId, long attemptToken) {
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
    public boolean finishPreemptedById(long requestId, long attemptToken, String detail) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)
                    || (entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_REQUESTED
                        && entry.preemption.state
                            != PreemptionRegistrationState.CANCEL_UNKNOWN)
                    || !entry.preemption.priorityCanceled.isDone()) {
                return false;
            }
            entry.preemption.state = PreemptionRegistrationState.SETTLED;
            entry.cleanupOwned = true;
            rollbackOnce(entry);
            RequestLifecycleSnapshot terminal = entry.lifecycle.cancel(detail);
            completeError(entry.item.future(), StrategyErrorType.PRIORITY_PREEMPTED, detail);
            finishEntry(entry, terminal);
            return true;
        }
    }

    @Override
    public boolean reconcilePreemptionActive(long requestId) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (entry.preemption == null
                    || entry.preemption.state != PreemptionRegistrationState.NOT_FOUND_STALE) {
                return false;
            }
            DecodeEndpoint endpoint = entry.item.decodeEp();
            if (endpoint != null && !endpoint.reconcilePriorityVictimActive(requestId)) {
                // A racing typed CANCELED or ordinary terminal may already
                // own the endpoint CAS. Keep the scheduler token fence intact
                // so that winner can finish its corresponding transition.
                return false;
            }
            entry.preemption = null;
            return true;
        }
    }

    private boolean transitionPreemption(long requestId, long attemptToken,
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
            entry.preemption.state = next;
            if (updateLifecycle) {
                entry.lifecycle.requestCancel(entry.preemption.detail);
            }
            return true;
        }
    }

    @Override
    public EngineCancelChannel.CancelTarget resolveCancelTarget(long requestId) {
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

    // ==================== Completion from worker status ====================

    public void onWorkerStatusUpdate(WorkerStatusResponse response) {
        if (response == null) {
            return;
        }
        boolean isPrefill = response.getRole() == RoleType.PREFILL;

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
            long requestId = task.getRequestId();
            WorkerTerminalObservation observation = new WorkerTerminalObservation(
                    isPrefill, task.getBatchId(), task.getErrorCode());

            InflightEntry entry = inflight.get(requestId);
            if (entry == null) {
                continue;
            }
            synchronized (entry) {
                if (entry.preemption != null) {
                    boolean authoritativeCanceled = isPrefill
                            && task.getPriorityPreemptionProgress()
                                == PriorityPreemptionProgress.CANCELED
                            && task.getErrorCode() == ENGINE_ERROR_PRIORITY_PREEMPTED;
                    if (authoritativeCanceled) {
                        // This only wakes the token owner. Decode accounting
                        // and victim terminal settlement happen in the
                        // coordinator's token-fenced continuation.
                        entry.preemption.priorityCanceled.complete(
                                new PriorityCanceledObservation(requestId,
                                        task.getErrorCode()));
                        continue;
                    }
                }
                // Successful Prefill completion is not the end of a P/D
                // request; Decode still owns it.
                if (!observation.isTerminal()) {
                    continue;
                }
                reduceOrdinaryTerminalLocked(entry,
                        DeferredTerminal.worker(observation));
            }
        }
    }

    /**
     * Single reducer for every non-priority terminal. A live preemption claim
     * owns Decode accounting, so the first real ordinary outcome is retained
     * instead of rolling back/unregistering underneath the Cancel protocol.
     */
    private void reduceOrdinaryTerminalLocked(InflightEntry entry,
                                              DeferredTerminal terminal) {
        if (inflight.get(entry.item.requestId()) != entry) {
            return;
        }
        if (entry.cleanupOwned) {
            return;
        }
        if (entry.preemption != null) {
            deferOrdinaryTerminalLocked(entry, terminal);
            return;
        }
        entry.cleanupOwned = true;
        applyOrdinaryTerminalLocked(entry, terminal);
    }

    /** Called with {@code entry} locked. */
    private void deferOrdinaryTerminalLocked(InflightEntry entry,
                                             DeferredTerminal terminal) {
        PreemptionRegistration registration = entry.preemption;
        if (registration == null) {
            entry.cleanupOwned = true;
            applyOrdinaryTerminalLocked(entry, terminal);
            return;
        }
        if (registration.pendingTerminal == null
                || (!registration.pendingTerminal.authoritativeWorker()
                    && terminal.authoritativeWorker())) {
            registration.pendingTerminal = terminal;
        }
        if (registration.state == PreemptionRegistrationState.NOT_FOUND_STALE) {
            replayAfterNegativeCancelLocked(entry, registration.attemptToken, false);
        } else if (registration.state == PreemptionRegistrationState.CANCEL_UNKNOWN) {
            replayAfterNegativeCancelLocked(entry, registration.attemptToken, true);
        }
    }

    /**
     * Replay work cached while a claim existed when that claim is rolled back
     * before any Cancel RPC. Endpoint ownership has already been aborted (or
     * was never installed), so no endpoint reconciliation is needed here.
     */
    private void replayAfterReleasedClaimLocked(InflightEntry entry,
                                                PreemptionRegistration registration) {
        if (registration.pendingTerminal != null) {
            entry.cleanupOwned = true;
            applyOrdinaryTerminalLocked(entry, registration.pendingTerminal);
        } else if (registration.pendingAcknowledgeBatchId != 0) {
            applyAcknowledgeLocked(entry, registration.pendingAcknowledgeBatchId);
        }
    }

    /**
     * Resolve a cached outcome after NOT_FOUND or transport UNKNOWN. The
     * Decode endpoint CAS is the winner selection against a racing typed
     * CANCELED settlement. If typed CANCELED won first, keep the scheduler
     * registration intact for its token-fenced continuation.
     */
    private void replayAfterNegativeCancelLocked(InflightEntry entry,
                                                 long attemptToken,
                                                 boolean transportUnknown) {
        if (!entry.hasPreemption(attemptToken)) {
            return;
        }
        PreemptionRegistration registration = entry.preemption;
        DeferredTerminal pending = registration.pendingTerminal;
        if (pending != null) {
            // NOT_FOUND proves no Cancel intent was installed, so every
            // cached terminal can resume. Transport UNKNOWN does not; replay
            // only a terminal observed authoritatively from worker status.
            if (transportUnknown && !pending.authoritativeWorker()) {
                return;
            }
            DecodeEndpoint endpoint = entry.item.decodeEp();
            boolean ordinaryWon = endpoint == null
                    || endpoint.reconcilePriorityVictimFinished(entry.item.requestId());
            if (!ordinaryWon) {
                return;
            }
            entry.preemption = null;
            entry.cleanupOwned = true;
            applyOrdinaryTerminalLocked(entry, pending);
            return;
        }

        // A concrete NOT_FOUND plus the delayed EnqueueBatch success proves
        // the request is active and the Cancel intent was not installed.
        // Transport UNKNOWN cannot make that assertion, so it keeps waiting
        // for typed CANCELED or an actual ordinary terminal.
        if (!transportUnknown && registration.pendingAcknowledgeBatchId != 0) {
            DecodeEndpoint endpoint = entry.item.decodeEp();
            boolean activeWon = endpoint == null
                    || endpoint.reconcilePriorityVictimActive(entry.item.requestId());
            if (!activeWon) {
                return;
            }
            long batchId = registration.pendingAcknowledgeBatchId;
            entry.preemption = null;
            applyAcknowledgeLocked(entry, batchId);
        }
    }

    /** Apply an already-owned ordinary outcome. Called with {@code entry} locked. */
    private void applyOrdinaryTerminalLocked(InflightEntry entry,
                                             DeferredTerminal terminal) {
        switch (terminal.kind()) {
            case ADMISSION_CLEANUP -> {
                PrefillEndpoint prefill = entry.item.prefillEp();
                if (prefill != null) {
                    prefill.getBatcher().queueManager().tryRemove(
                            entry.item.requestId(), "LEASE_RELEASE");
                }
                rollbackOnce(entry);
                inflight.remove(entry.item.requestId(), entry);
            }
            case FAILURE -> {
                rollbackOnce(entry);
                if (terminal.removeFromPrefillBatch()) {
                    removeFromPrefillBatch(entry);
                }
                RequestLifecycleSnapshot failed = entry.lifecycle.fail(terminal.detail());
                completeError(entry.item.future(), terminal.errorType(), terminal.detail());
                finishEntry(entry, failed);
            }
            case TIMEOUT -> timeoutEntry(entry, terminal.detail());
            case WORKER -> applyWorkerTerminalLocked(entry, terminal.workerObservation());
        }
    }

    /** Existing ordinary worker-terminal semantics, called with entry locked. */
    private void applyWorkerTerminalLocked(InflightEntry entry,
                                           WorkerTerminalObservation observation) {
        long requestId = entry.item.requestId();
        RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
        // Decode workers do not carry a reliable Prefill batch id.
        if (observation.batchId() >= 0 && observation.batchId() != current.batchId()) {
            Logger.warn("Worker completion batchId mismatch: "
                            + "request_id={} task_batch_id={} entry_batch_id={} is_prefill={}",
                    requestId, observation.batchId(), current.batchId(), observation.prefill());
            if (observation.prefill()) {
                return;
            }
        }
        RequestLifecycleSnapshot terminal;
        if (observation.errorCode() == 0) {
            terminal = entry.lifecycle.complete("decode completed");
            completeSuccess(entry.item);
        } else {
            terminal = entry.lifecycle.fail("worker error code " + observation.errorCode());
            completeError(entry.item.future(), StrategyErrorType.WORKER_EXECUTION_FAILED,
                    "worker error code " + observation.errorCode());
        }
        if (observation.prefill()) {
            rollbackOnce(entry);
            removeFromPrefillBatch(entry);
        }
        finishEntry(entry, terminal);
    }

    public int getInflightSize() {
        return inflight.size();
    }

    /** Production RTP-LLM raw {@code ErrorCode::PRIORITY_PREEMPTED}. */
    private static final long ENGINE_ERROR_PRIORITY_PREEMPTED = 8429;

    /** Victim's decode endpoint key for the settle metric; "unknown" when absent. */
    private static String decodeEndpointKey(BatchItem item) {
        return item.decodeEp() != null ? item.decodeEp().ipPort() : "unknown";
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        InflightEntry entry = inflight.get(requestId);
        RequestLifecycleSnapshot snapshot = entry != null
                ? entry.lifecycle.snapshot()
                : terminalStates.get(requestId);
        return batchMatches(snapshot, expectedBatchId) ? snapshot : null;
    }

    // ==================== Inflight TTL cleanup ====================

    @Scheduled(fixedRate = 60000L)
    public void cleanupInflight() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        long now = System.currentTimeMillis();
        int expiredCount = 0;
        for (Map.Entry<Long, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            if (now - entry.createdAtMs() <= ttlMs) {
                continue;
            }
            synchronized (entry) {
                if (inflight.get(candidate.getKey()) != entry) {
                    continue;
                }
                if (entry.preemption != null || entry.cleanupOwned) {
                    // Cancel ambiguity is reconciled by token/WorkerStatus;
                    // a concurrent cleanup owner is likewise already settling
                    // the entry and must not be raced by TTL.
                    continue;
                }
                timeoutEntry(entry, "inflight TTL expired");
                expiredCount++;
            }
        }
        if (expiredCount > 0) {
            reporter.reportInflightTtlExpired(expiredCount);
        }
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
            for (Map.Entry<Long, RequestInflight> reserved : decodeEp.reservedView().entrySet()) {
                long requestId = reserved.getKey();
                if (now - reserved.getValue().createdAtMs() > ttlMs
                        && !inflight.containsKey(requestId)) {
                    decodeEp.release(requestId);
                    Logger.warn("orphan decode reservation reclaimed: request_id={} worker={} age_ms={}",
                            requestId, decodeEntry.getKey(),
                            now - reserved.getValue().createdAtMs());
                }
            }
        }
    }

    // ==================== BatchDecisionHandler callbacks (from WorkerBatcher) ====================

    @Override
    public void onExpired(BatchItem head) {
        InflightEntry entry = entryFor(head);
        if (entry != null) {
            synchronized (entry) {
                reduceOrdinaryTerminalLocked(entry,
                        DeferredTerminal.timeout("batch SLO expired before dispatch"));
            }
        } else if (!head.future().isDone() && !terminalStates.containsKey(head.requestId())) {
            rollback(head);
        }
    }

    @Override
    public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
        flushItems(items, meta);
    }

    @Override
    public void onOfferFailure(BatchItem item, Throwable error) {
        // Auto-TPM: over-capacity requests carry a dedicated non-retryable error code
        // instead of the generic (retryable) dispatch failure (design doc 8.3).
        StrategyErrorType errorType = error instanceof BatchTokenCapacityExceededException
                ? StrategyErrorType.BATCH_TOKEN_CAPACITY_EXCEEDED
                : StrategyErrorType.BATCH_DISPATCH_FAILED;
        InflightEntry entry = entryFor(item);
        if (entry != null) {
            synchronized (entry) {
                reduceOrdinaryTerminalLocked(entry, DeferredTerminal.failure(
                        errorType, "Batcher offer failed: " + error.getMessage(), false));
            }
        } else if (!item.future().isDone() && !terminalStates.containsKey(item.requestId())) {
            rollback(item);
            completeError(item.future(), errorType,
                    "Batcher offer failed: " + error.getMessage());
        }
    }

    // ==================== Dispatch pipeline ====================

    /**
     * Commit batch to PrefillEndpoint, then delegate to {@link BatchDispatcher}
     * for asynchronous gRPC dispatch.
     * <p>
     * The heavy gRPC I/O is handled asynchronously by the dispatcher's own thread pool.
     */
    private void flushItems(List<BatchItem> items, DispatchMeta meta) {
        String reason = meta.reason();
        PrefillEndpoint prefillEp = items.get(0).prefillEp();

        // A timeout or prior failure may finish an item while it is still queued.
        List<BatchItem> active = items.stream()
                .filter(item -> !item.future().isDone())
                .toList();

        if (active.isEmpty()) {
            return;
        }

        // [SYNC] Compute prediction and commit only active items to endpoint
        long predMs = 0;
        long batchId = batchIdGenerator.nextBatchId();
        List<BatchItem> dispatchable = new ArrayList<>(active.size());
        for (BatchItem item : active) {
            InflightEntry entry = entryFor(item);
            if (entry == null) {
                continue;
            }
            synchronized (entry) {
                if (entry.lifecycle.isTerminal() || entry.cleanupOwned) {
                    continue;
                }
                // Linearize dispatch ownership against Master-local eviction.
                // If eviction/timeout released the reservation first, this
                // item is no longer allowed to reach the engine. Legacy
                // reservations are non-queued but still pass this claim.
                if (item.decodeEp() != null
                        && !item.decodeEp().tryMarkEngineMayHaveSeen(item.requestId())) {
                    continue;
                }
                entry.lifecycle.startDispatch(batchId);
                dispatchable.add(item);
            }
        }

        if (dispatchable.isEmpty()) {
            return;
        }
        if (prefillEp != null) {
            PrefillTimePredictor predictor = prefillEp.getPredictor();
            predMs = (long) predictor.predictBatchMs(dispatchable);
            prefillEp.commitBatch(batchId, predMs, dispatchable);
        }

        // [ASYNC] Delegate gRPC dispatch — dispatcher owns its own thread pool
        long nowMs = System.currentTimeMillis();
        long waitMs = nowMs - items.get(0).enqueuedAtMs();
        // Batch wait tagged by normalized priority: one report per priority
        // present in the batch, using that priority's oldest enqueue time.
        Map<Integer, Long> oldestEnqueueByPriority = new HashMap<>();
        for (BatchItem item : dispatchable) {
            oldestEnqueueByPriority.merge(item.priority(), item.enqueuedAtMs(), Math::min);
        }
        String engineIp = prefillEp != null ? prefillEp.getIp() : "";
        for (Map.Entry<Integer, Long> waitEntry : oldestEnqueueByPriority.entrySet()) {
            reporter.reportBatchWaitTimeMs(
                    RoleType.PREFILL.name(), engineIp, nowMs - waitEntry.getValue(), waitEntry.getKey());
        }
        FlexlbConfig config = configService.loadBalanceConfig();
        Logger.info("flexlb_batch_dispatch batch_id={} reason={} batch_size={} wait_ms={} "
                        + "predicted_ms={} threshold_ms={} fixed_wait_ms={} batch_size_max={} "
                        + "queue_after={} worker={}",
                batchId, reason, dispatchable.size(), waitMs, predMs,
                config.getFlexlbBatchPredictThresholdMs(), config.getFlexlbBatchFixedWaitMs(),
                config.getFlexlbBatchSizeMax(), meta.queueDepth(),
                prefillEp != null ? prefillEp.ipPort() : "");

        // Record dispatch timestamp for dispatch-to-ACK latency metric
        for (BatchItem item : dispatchable) {
            InflightEntry entry = entryFor(item);
            if (entry != null) {
                entry.lifecycle.markDispatched();
                item.ctx().setBatchDispatchedNanos(System.nanoTime());
            }
        }

        dispatcher.dispatch(dispatchable, prefillEp, batchId, predMs, reason, this);
    }

    // ==================== DispatchCallback implementation ====================

    @Override
    public void onSuccess(BatchItem item, long batchId) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            // entry 已被 worker-status/cancel/timeout/onFailure/onOfferFailure 等终态路径移除，
            // 所有终态路径均在 finishEntry 前完成 future，故此处无需补发。
            return;
        }

        synchronized (entry) {
            if (entry.cleanupOwned) {
                return;
            }
            long assignedBatchId = entry.lifecycle.snapshot().batchId();
            if (batchId != assignedBatchId) {
                Logger.warn("Ignoring stale EnqueueBatch ACK request_id={} batch_id={}",
                        item.requestId(), batchId);
                return;
            }
            if (entry.preemption != null) {
                PreemptionRegistration registration = entry.preemption;
                if (registration.pendingAcknowledgeBatchId == 0) {
                    registration.pendingAcknowledgeBatchId = batchId;
                }
                if (registration.state == PreemptionRegistrationState.NOT_FOUND_STALE) {
                    replayAfterNegativeCancelLocked(
                            entry, registration.attemptToken, false);
                }
                // CANCEL_IN_FLIGHT/CANCEL_REQUESTED/UNKNOWN retain the ACK.
                // It must not turn a priority victim into a successful frontend
                // response while the Cancel outcome is unresolved.
                return;
            }
            applyAcknowledgeLocked(entry, batchId);
        }
    }

    /** Apply an EnqueueBatch success after its ownership decision is final. */
    private void applyAcknowledgeLocked(InflightEntry entry, long batchId) {
        RequestLifecycleSnapshot snapshot = entry.lifecycle.acknowledge();
        if (snapshot.state() != RequestLifecycleState.ACKNOWLEDGED) {
            return;
        }
        BatchItem item = entry.item;
        // Record ACK timestamp for ack_to_response_time_ms metric (reported in FlexlbServiceImpl.completeSchedule)
        item.ctx().setAckAtMs(System.currentTimeMillis());
        item.ctx().setAckAtNanos(System.nanoTime());

        long dispatchedAtMs = entry.lifecycle.getDispatchedAtMs();
        if (dispatchedAtMs > 0) {
            PrefillEndpoint ep = item.prefillEp();
            reporter.reportDispatchAckTimeMs(
                    RoleType.PREFILL.name(),
                    ep != null ? ep.getIp() : "",
                    System.currentTimeMillis() - dispatchedAtMs);
        }
        if (!item.future().isDone()) {
            completeSuccess(item);
            Logger.debug("FlexLB batch enqueued request {} in batch_id={}",
                    item.requestId(), batchId);
        }
    }

    private void completeSuccess(BatchItem item) {
        if (item.future().isDone()) {
            return;
        }
        Response success = copyResponse(item.routeResponse());
        success.setSuccess(true);
        success.setCode(200);
        success.setEnqueuedByMaster(true);
        success.setQueueLength(inflight.size());
        item.future().complete(success);
    }

    @Override
    public void onFailure(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry != null) {
            synchronized (entry) {
                // Even a raw EnqueueBatch 8429 is not the priority resource
                // fence. It follows the same deferred ordinary-terminal path;
                // only typed Prefill CANCELED may perform priority settlement.
                reduceOrdinaryTerminalLocked(entry, DeferredTerminal.failure(
                        StrategyErrorType.BATCH_DISPATCH_FAILED,
                        error.getMessage(), true));
            }
            return;
        }
        if (!item.future().isDone() && !terminalStates.containsKey(item.requestId())) {
            rollback(item);
            completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED,
                    error.getMessage());
        }
    }

    @Override
    public void onTimeout(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return;
        }
        synchronized (entry) {
            reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                    "EnqueueBatch deadline exceeded: " + error.getMessage()));
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
            String ipPort = serverStatus.getServerIp() + ":" + serverStatus.getHttpPort();
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

    /**
     * Remove a failed or timed-out request from its prefill batch entry.
     * Uses {@link PrefillEndpoint#repackBatch} which:
     * <ul>
     *   <li>Single-request batch → removes the entire entry (batch becomes empty)</li>
     *   <li>Multi-request batch → keeps survivors, removes only this request</li>
     *   <li>Batch already removed (calibrate or releaseBatch ran first) → no-op</li>
     * </ul>
     * Safe to call multiple times (idempotent via ConcurrentHashMap.computeIfPresent).
     */
    private void removeFromPrefillBatch(InflightEntry entry) {
        long batchId = entry.lifecycle.snapshot().batchId();
        if (batchId <= 0) {
            return;
        }
        PrefillEndpoint prefillEp = entry.item.prefillEp();
        if (prefillEp != null) {
            prefillEp.repackBatch(batchId, Set.of(entry.item.requestId()));
            Logger.info("FlexLB remove from prefill batch: request_id={} batch_id={} engine={}",
                    entry.item.requestId(), batchId, prefillEp.getIp());
        }
    }

    private void timeoutEntry(InflightEntry entry, String detail) {
        AdmissionFailure admissionFailure = null;
        PrefillEndpoint prefill = entry.item.prefillEp();
        if (entry.autoTpmAdmission) {
            admissionFailure = classifyAdmissionTimeout(entry.item, prefill);
            if (prefill != null) {
                prefill.getBatcher().queueManager().tryRemove(
                        entry.item.requestId(), "ADMISSION_TIMEOUT");
            }
        }
        RequestLifecycleSnapshot terminal = entry.lifecycle.timeout(detail);
        rollbackOnce(entry);
        removeFromPrefillBatch(entry);
        if (admissionFailure != null) {
            Logger.info("[auto-tpm] admission timeout classified: request_id={} "
                            + "priority={} lifecycle={} error_code={} reason={} trigger={}",
                    entry.item.requestId(), entry.item.priority(), terminal.state(),
                    admissionFailure.errorType().getErrorCode(), admissionFailure.reason(), detail);
            completeAdmissionError(entry.item.future(), admissionFailure, detail);
        } else {
            completeError(entry.item.future(), StrategyErrorType.BATCH_SLO_EXPIRED, detail);
        }
        finishEntry(entry, terminal);
    }

    private static AdmissionFailure classifyAdmissionTimeout(BatchItem item,
                                                              PrefillEndpoint prefill) {
        if (prefill == null) {
            return AdmissionFailure.resourceExhausted();
        }
        List<QueuedRequestSnapshot> ahead = new ArrayList<>();
        for (QueuedRequestSnapshot queued
                : prefill.getBatcher().queueManager().snapshot().items()) {
            if (queued.requestId() == item.requestId()) {
                return AdmissionFailureClassifier.classifyQueuedDeadline(
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
        Response errorResp = Response.error(errorType);
        errorResp.setErrorMessage(errorType.buildErrorMessage(message));
        future.complete(errorResp);
    }

    private static void completeAdmissionError(CompletableFuture<Response> future,
                                               AdmissionFailure failure,
                                               String trigger) {
        if (future.isDone()) {
            return;
        }
        Response errorResp = Response.error(failure.errorType(), failure.reason());
        String detail = failure.message() + "; trigger=" + trigger;
        errorResp.setErrorMessage(failure.errorType().buildErrorMessage(detail));
        future.complete(errorResp);
    }

    private void finishEntry(InflightEntry entry,
                             RequestLifecycleSnapshot terminal) {
        // Publish the tombstone before removing inflight. submit() then observes
        // at least one side of the handoff and cannot revive the request ID.
        terminalStates.put(terminal.requestId(), terminal);
        inflight.remove(terminal.requestId(), entry);
    }

    private static boolean batchMatches(RequestLifecycleSnapshot snapshot,
                                        long expectedBatchId) {
        if (snapshot == null) {
            return false;
        }
        return expectedBatchId == 0 || snapshot.batchId() == expectedBatchId;
    }

    // ==================== Internal: static utilities ====================

    /** Locate the first server of a role in a route response (shared with the Auto-TPM path). */
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

    /** Defensive copy of a route server status (shared with the Auto-TPM path). */
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

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    public void reportBatchMetrics() {
        reporter.reportSchedulerInflightSize(inflight.size());

        // Per-worker metrics: prefill endpoints
        for (Map.Entry<String, PrefillEndpoint> entry : endpointRegistry.getPrefillEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }

        // Per-worker metrics: decode endpoints
        for (Map.Entry<String, DecodeEndpoint> entry : endpointRegistry.getDecodeEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }

        // Auto-TPM: per-endpoint prefill queue depth gauge (design doc 19.2)
        if (priorityScheduler != null && configService.loadBalanceConfig().isAutoTpmEnabled()) {
            priorityScheduler.reportPrefillQueueDepths();
            priorityScheduler.reportDecodeAdmissionGauges();
        }
    }

    @PreDestroy
    public void shutdown() {
        endpointRegistry.close();
    }

    // ==================== Inflight entry ====================

    private static final class InflightEntry {
        final BatchItem item;
        final RequestLifecycle lifecycle;
        final boolean autoTpmAdmission;
        final AtomicBoolean rolledBack = new AtomicBoolean(false);
        boolean cleanupOwned;
        PreemptionRegistration preemption;

        InflightEntry(BatchItem item, boolean autoTpmAdmission) {
            this.item = Objects.requireNonNull(item);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
            this.lifecycle = new RequestLifecycle(item.requestId());
            this.autoTpmAdmission = autoTpmAdmission;
        }

        public long createdAtMs() {
            return lifecycle.snapshot().createdAtMs();
        }

        boolean hasPreemption(long attemptToken) {
            return preemption != null && preemption.attemptToken == attemptToken;
        }
    }

    private enum PreemptionRegistrationState {
        CLAIMED,
        CANCEL_IN_FLIGHT,
        CANCEL_REQUESTED,
        NOT_FOUND_STALE,
        CANCEL_UNKNOWN,
        SETTLED
    }

    private static final class PreemptionRegistration {
        private final long attemptToken;
        private final String detail;
        private final CompletableFuture<PriorityCanceledObservation> priorityCanceled =
                new CompletableFuture<>();
        private PreemptionRegistrationState state = PreemptionRegistrationState.CLAIMED;
        private DeferredTerminal pendingTerminal;
        private long pendingAcknowledgeBatchId;

        private PreemptionRegistration(long attemptToken, String detail) {
            this.attemptToken = attemptToken;
            this.detail = detail;
        }
    }

    private record WorkerTerminalObservation(boolean prefill,
                                             long batchId,
                                             long errorCode) {
        boolean isTerminal() {
            return !prefill || errorCode != 0;
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
                                    boolean removeFromPrefillBatch,
                                    WorkerTerminalObservation workerObservation) {
        static DeferredTerminal admissionCleanup(String detail) {
            return new DeferredTerminal(DeferredTerminalKind.ADMISSION_CLEANUP,
                    null, detail, false, null);
        }

        static DeferredTerminal failure(StrategyErrorType errorType,
                                        String detail,
                                        boolean removeFromPrefillBatch) {
            return new DeferredTerminal(DeferredTerminalKind.FAILURE,
                    Objects.requireNonNull(errorType), detail,
                    removeFromPrefillBatch, null);
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
    }
}
