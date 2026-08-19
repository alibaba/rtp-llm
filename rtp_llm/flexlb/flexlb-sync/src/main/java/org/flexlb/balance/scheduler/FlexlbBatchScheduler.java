package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.endpoint.WorkerEndpoint.EndpointRetiredException;
import org.flexlb.balance.scheduler.priority.AdmissionFailure;
import org.flexlb.balance.scheduler.priority.AdmissionFailureClassifier;
import org.flexlb.balance.scheduler.priority.AdmissionLease;
import org.flexlb.balance.scheduler.priority.EngineCancelChannel;
import org.flexlb.balance.scheduler.priority.InflightRegistrar;
import org.flexlb.balance.scheduler.priority.PriorityAdmissionScheduler;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.balance.scheduler.priority.UnsupportedEngineCancelChannel;
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
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
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
    private final EngineCancelChannel engineCancelChannel;
    private final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    private final Map<Long, RequestLifecycleSnapshot> terminalStates = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, AdmissionClaim> admissionClaims =
            new ConcurrentHashMap<>();
    private final BatchIdGenerator batchIdGenerator;
    /** Linearizes the final endpoint-ledger commit/RPC handoff with fencing. */
    private final Object dispatchFence = new Object();

    public FlexlbBatchScheduler(ConfigService configService,
                                Router router,
                                EndpointRegistry endpointRegistry,
                                BatchDispatcher dispatcher,
                                BatchSchedulerReporter reporter,
                                PriorityAdmissionScheduler priorityScheduler,
                                Environment environment) {
        this(configService, router, endpointRegistry, dispatcher, reporter,
                priorityScheduler, environment, new UnsupportedEngineCancelChannel());
    }

    @Autowired
    public FlexlbBatchScheduler(ConfigService configService,
                                Router router,
                                EndpointRegistry endpointRegistry,
                                BatchDispatcher dispatcher,
                                BatchSchedulerReporter reporter,
                                PriorityAdmissionScheduler priorityScheduler,
                                Environment environment,
                                EngineCancelChannel engineCancelChannel) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.dispatcher = dispatcher;
        this.reporter = reporter;
        this.priorityScheduler = priorityScheduler;
        this.engineCancelChannel = Objects.requireNonNull(engineCancelChannel);
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
        boolean ownsSubmissionClaim = false;
        boolean autoTpmPath = false;
        Response routeResponse = null;
        InflightEntry registeredEntry = null;
        boolean queueOwnsItem = false;
        try {
            if (ctx == null || ctx.getRequest() == null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST, null);
                return future;
            }

            ownsSubmissionClaim = tryAcquireAdmissionClaim(ctx.getRequestId(), future);
            if (!ownsSubmissionClaim) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
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
                autoTpmPath = true;
                priorityScheduler.schedule(ctx, future, this);
                // The deadline is an ordinary terminal event in the same
                // reducer as dispatch/worker failures. It must compete with a
                // priority-preemption claim before completing the public future.
                attachAdmissionTimeout(ctx, future);
                return future;
            }

            routeResponse = router.route(ctx);
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
                if (decodeEp == null) {
                    rollback(routeResponse);
                    completeError(future, StrategyErrorType.NO_DECODE_WORKER, null);
                    return future;
                }
            }

            BatchItem item = new BatchItem(ctx, future, routeResponse, copyOf(prefill), copyOf(decode),
                    prefillEp, decodeEp, System.currentTimeMillis());
            InflightEntry entry = new InflightEntry(item, false);
            registeredEntry = entry;
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
            queueOwnsItem = true;

            // Report route+submit time: from schedule() entry (ctx.startTime) to batcher offer completion
            try {
                reporter.reportRouteSubmitTimeMs(
                        RoleType.PREFILL.name(),
                        prefillEp.getIp(),
                        System.currentTimeMillis() - ctx.getStartTime());
            } catch (RuntimeException metricFailure) {
                Logger.warn("Failed to report route submit metric: request_id={}",
                        ctx.getRequestId(), metricFailure);
            }
        } catch (Throwable t) {
            if (!autoTpmPath && registeredEntry != null && !queueOwnsItem
                    && inflight.remove(ctx.getRequestId(), registeredEntry)) {
                rollbackOnce(registeredEntry);
            } else if (!autoTpmPath && registeredEntry == null && routeResponse != null) {
                rollback(routeResponse);
            }
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            if (!queueOwnsItem) {
                completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                        "Submit failed: " + t.getMessage());
            }
        } finally {
            if (ownsSubmissionClaim) {
                releaseAdmissionHold(ctx.getRequestId(), future);
            }
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
    // Package-visible for the dispatch/deadline linearization test.
    void onAdmissionDeadline(long requestId,
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
            synchronized (dispatchFence) {
              synchronized (entry) {
                if (inflight.get(requestId) != entry || expectedFuture.isDone()) {
                    return;
                }
                if (entry.lifecycle.hasDispatchClaim()) {
                    if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED) {
                        // Decode WorkerStatus is stronger than the missing
                        // Enqueue ACK. Publish the logical ACK now; otherwise
                        // the admission future would remain pending forever.
                        applyAcknowledgeLocked(entry,
                                entry.lifecycle.snapshot().batchId());
                        return;
                    }
                    // startDispatch(batchId) is the point of no return. The
                    // batcher already owns a local snapshot and may publish it
                    // after this lock is released. Completing 8431 and deleting
                    // the ledgers here would let the frontend retry while the
                    // original request is still accepted by the engine.
                    if (startDispatchReconciliationLocked(entry)) {
                        CompletableFuture.runAsync(() -> reconcileUncertainDispatch(entry, 0));
                    }
                    return;
                }
                reduceOrdinaryTerminalLocked(entry,
                        DeferredTerminal.timeout("admission deadline exceeded"));
              }
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
            if (entry.admissionLease != null && entry.admissionLease != lease) {
                throw new IllegalStateException(
                        "admission lease already attached for request_id=" + item.requestId());
            }
            entry.admissionLease = lease;
            if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED) {
                lease.markDecodeAccepted();
            }
            return true;
        }
    }

    @Override
    public void unregisterInflight(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        if (entry != null && entry.item == item) {
            inflight.remove(item.requestId(), entry);
        }
    }

    @Override
    public boolean retainPendingAdmission(
            long requestId, CompletableFuture<Response> expectedFuture) {
        AdmissionClaim claim = admissionClaims.get(requestId);
        if (claim == null || claim.future != expectedFuture) {
            return false;
        }
        synchronized (claim) {
            if (admissionClaims.get(requestId) != claim
                    || claim.publicCompleted || expectedFuture.isDone()) {
                return false;
            }
            claim.holds++;
            return true;
        }
    }

    @Override
    public void releasePendingAdmission(
            long requestId, CompletableFuture<Response> expectedFuture) {
        releaseAdmissionHold(requestId, expectedFuture);
    }

    @Override
    public boolean requestPostHandoverReconciliation(BatchItem item, String detail) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return false;
        }
        boolean start = false;
        synchronized (dispatchFence) {
            synchronized (entry) {
                if (inflight.get(item.requestId()) != entry
                        || entry.cleanupOwned || entry.lifecycle.isTerminal()) {
                    return false;
                }
                if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED) {
                    if (entry.admissionLease != null) {
                        entry.admissionLease.markDecodeAccepted();
                    }
                    return true;
                }
                if (entry.preemption != null) {
                    // The tokenized priority Cancel already owns all Engine
                    // side effects. Resume handover reconciliation only if
                    // that attempt later proves NOT_FOUND and replays the ACK.
                    entry.postHandoverReconciliationRequested = true;
                    return true;
                }
                if (entry.dispatchReconciliation) {
                    return true;
                }
                start = startDispatchReconciliationLocked(entry);
            }
        }
        if (start) {
            reconcileUncertainDispatch(entry, 0);
        }
        return start;
    }

    private boolean tryAcquireAdmissionClaim(
            long requestId, CompletableFuture<Response> future) {
        AdmissionClaim claim = new AdmissionClaim(future);
        if (admissionClaims.putIfAbsent(requestId, claim) != null) {
            return false;
        }
        future.whenComplete((ignored, failure) -> {
            synchronized (claim) {
                if (admissionClaims.get(requestId) != claim) {
                    return;
                }
                claim.publicCompleted = true;
                removeAdmissionClaimIfReleased(requestId, claim);
            }
        });
        return true;
    }

    private void releaseAdmissionHold(
            long requestId, CompletableFuture<Response> expectedFuture) {
        AdmissionClaim claim = admissionClaims.get(requestId);
        if (claim == null || claim.future != expectedFuture) {
            return;
        }
        synchronized (claim) {
            if (admissionClaims.get(requestId) != claim || claim.holds == 0) {
                return;
            }
            claim.holds--;
            removeAdmissionClaimIfReleased(requestId, claim);
        }
    }

    private void removeAdmissionClaimIfReleased(long requestId, AdmissionClaim claim) {
        if (claim.publicCompleted && claim.holds == 0) {
            admissionClaims.remove(requestId, claim);
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
            if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED) {
                // Decode WorkerStatus is authoritative. A failed/timed-out
                // Enqueue ACK cannot reclaim engine-owned accounting.
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
            // A settle on an unknown id is harmless (already terminal or
            // never registered); surface it through an alertable metric.
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
            // Same rationale as finishPreemptedById.
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
            if (entry.cleanupOwned || entry.preemption != null
                    || entry.dispatchReconciliation || entry.lifecycle.isTerminal()) {
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
    public boolean markPreemptionCompletionTimedOut(long requestId, long attemptToken) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            if (!entry.hasPreemption(attemptToken)
                    || entry.preemption.state
                        != PreemptionRegistrationState.CANCEL_REQUESTED) {
                return false;
            }
            entry.preemption.acceptedCompletionTimedOut = true;
            settleAcceptedCancelFromDecodeTerminalLocked(entry);
            return true;
        }
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
            // The typed terminal was already fenced by the exact positive
            // Prefill batch generation before this signal was completed.
            finishPriorityPreemptedLocked(entry, entry.preemption, detail);
            return true;
        }
    }

    private boolean reconcilePreemptionActiveLocked(InflightEntry entry) {
        if (entry.preemption == null
                || entry.preemption.state != PreemptionRegistrationState.NOT_FOUND_STALE) {
            return false;
        }
        DecodeEndpoint endpoint = entry.item.decodeEp();
        if (endpoint != null
                && !endpoint.reconcilePriorityVictimActive(entry.item.requestId())) {
            return false;
        }
        entry.preemption = null;
        resumePostHandoverReconciliationLocked(entry);
        return true;
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
        PrefillEndpoint prefillGeneration = entry.item.prefillEp();
        if (prefillGeneration == null) {
            return null;
        }
        return new EngineCancelChannel.CancelTarget(prefillGeneration);
    }

    // ==================== Completion from worker status ====================

    public void recordRequestActivity(WorkerStatus source,
                                      WorkerStatusResponse response) {
        if (!validWorkerSource(source, response)) {
            return;
        }
        touchObservedTasks(source, response.getRunningTaskInfo(), true);
        touchObservedTasks(source, response.getFinishedTaskInfo(), false);
    }

    /**
     * Apply worker observations to request lifecycles. Endpoint calibration is
     * a separate EndpointRegistry responsibility.
     */
    public void updateRequestLifecycleFromWorkerStatus(
            WorkerStatus source, WorkerStatusResponse response) {
        if (!validWorkerSource(source, response)) {
            return;
        }
        boolean isPrefill = response.getRole() == RoleType.PREFILL;
        boolean isDecode = response.getRole() == RoleType.DECODE;

        Map<String, TaskInfo> running = response.getRunningTaskInfo();
        if (running != null) {
            for (TaskInfo task : running.values()) {
                if (task == null || task.priorityCancelOverlayOnly()) {
                    continue;
                }
                InflightEntry entry = inflight.get(task.getRequestId());
                if (entry == null) {
                    continue;
                }
                synchronized (dispatchFence) {
                  synchronized (entry) {
                    if (inflight.get(task.getRequestId()) != entry
                            || !matchesAuthoritativeWorkerObservation(
                                    entry, source, task)) {
                        continue;
                    }
                    if (isDecode) {
                        TaskPhase phase = task.getPhase();
                        if (phase == TaskPhase.KV_ALLOCATED
                                || phase == TaskPhase.RUNNING) {
                            markDecodeAcceptedLocked(entry);
                            if (entry.dispatchReconciliation) {
                                clearDispatchReconciliation(entry);
                                applyAcknowledgeLocked(entry,
                                        entry.lifecycle.snapshot().batchId());
                            }
                        }
                    } else if (isPrefill) {
                        reconcilePreemptionActiveLocked(entry);
                    }
                  }
                }
            }
        }

        Map<String, TaskInfo> finished = response.getFinishedTaskInfo();
        if (finished == null || finished.isEmpty()) {
            return;
        }
        for (TaskInfo task : finished.values()) {
            if (task == null) {
                continue;
            }
            long requestId = task.getRequestId();
            InflightEntry entry = inflight.get(requestId);
            if (entry == null) {
                continue;
            }
            PriorityCanceledEffect priorityCanceledEffect = null;
            synchronized (dispatchFence) {
              synchronized (entry) {
                if (inflight.get(requestId) != entry
                        || !matchesAuthoritativeWorkerObservation(
                                    entry, source, task)) {
                    continue;
                }
                boolean authoritativeCanceled = isPrefill
                        && task.getPriorityPreemptionProgress()
                            == PriorityPreemptionProgress.CANCELED
                        && task.getErrorCode() == ENGINE_ERROR_PRIORITY_PREEMPTED;
                if (entry.dispatchReconciliation) {
                    if (authoritativeCanceled) {
                        reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                                "EnqueueBatch reconciled by typed Prefill CANCELED"));
                    } else if (isDecode) {
                        // The original Prefill generation may no longer be
                        // reachable for Cancel. An exact observation from the
                        // Decode generation held by this BatchItem is still an
                        // authoritative ownership/terminal proof.
                        markDecodeAcceptedLocked(entry);
                        WorkerTerminalObservation observation =
                                new WorkerTerminalObservation(false, task.getErrorCode());
                        reduceOrdinaryTerminalLocked(entry,
                                DeferredTerminal.worker(observation));
                    }
                } else if (entry.preemption != null && authoritativeCanceled) {
                    PreemptionRegistration registration = entry.preemption;
                    if (!registration.priorityCanceledObserved) {
                        registration.priorityCanceledObserved = true;
                        priorityCanceledEffect = new PriorityCanceledEffect(
                                registration.priorityCanceled,
                                new PriorityCanceledObservation(
                                        requestId, task.getErrorCode()));
                    }
                } else {
                    if (isDecode) {
                        markDecodeAcceptedLocked(entry);
                    }
                    WorkerTerminalObservation observation =
                            new WorkerTerminalObservation(isPrefill, task.getErrorCode());
                    if (observation.isTerminal()) {
                        reduceOrdinaryTerminalLocked(entry,
                                DeferredTerminal.worker(observation));
                    }
                }
              }
            }
            if (priorityCanceledEffect != null) {
                priorityCanceledEffect.publish();
            }
        }
    }

    private void touchObservedTasks(WorkerStatus source,
                                    Map<String, TaskInfo> tasks,
                                    boolean activeSnapshot) {
        if (tasks == null) {
            return;
        }
        for (TaskInfo task : tasks.values()) {
            if (task == null || (activeSnapshot && task.priorityCancelOverlayOnly())) {
                continue;
            }
            InflightEntry entry = inflight.get(task.getRequestId());
            if (entry == null) {
                continue;
            }
            synchronized (entry) {
                if (inflight.get(task.getRequestId()) == entry
                        && matchesAuthoritativeWorkerObservation(entry, source, task)) {
                    entry.lifecycle.touch();
                }
            }
        }
    }

    private static boolean validWorkerSource(WorkerStatus source,
                                             WorkerStatusResponse response) {
        return source != null && response != null
                && source.getRole() == response.getRole()
                && (response.getRole() == RoleType.PREFILL
                    || response.getRole() == RoleType.DECODE);
    }

    private static boolean matchesAuthoritativeWorkerObservation(
            InflightEntry entry, WorkerStatus source, TaskInfo task) {
        RoleType role = source.getRole();
        long expectedBatchId = entry.lifecycle.snapshot().batchId();
        long taskBatchId = task.getBatchId();
        if (role == RoleType.PREFILL) {
            PrefillEndpoint endpoint = entry.item.prefillEp();
            return endpoint != null
                    && endpoint.getStatus() == source
                    && expectedBatchId > 0
                    && taskBatchId > 0
                    && taskBatchId == expectedBatchId;
        }
        if (role == RoleType.DECODE) {
            DecodeEndpoint endpoint = entry.item.decodeEp();
            return endpoint != null
                    && endpoint.getStatus() == source
                    && (taskBatchId <= 0 || taskBatchId == expectedBatchId);
        }
        return false;
    }

    /** Record authoritative Decode ownership. Called with {@code entry} locked. */
    private void markDecodeAcceptedLocked(InflightEntry entry) {
        if (entry.cleanupOwned
                || entry.dispatchOwnership == DispatchOwnership.TERMINAL) {
            return;
        }
        entry.dispatchOwnership = DispatchOwnership.DECODE_OWNED;
        if (entry.admissionLease != null) {
            entry.admissionLease.markDecodeAccepted();
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
        if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED
                && terminal.dispatchAckFailure()) {
            // KV_ALLOCATED/RUNNING is a stronger ownership observation than
            // an absent/failed Enqueue ACK. Preserve the live inflight entry,
            // Decode accounting, and public schedule success.
            if (entry.preemption != null) {
                PreemptionRegistration registration = entry.preemption;
                long batchId = entry.lifecycle.snapshot().batchId();
                if (registration.pendingAcknowledgeBatchId == 0) {
                    registration.pendingAcknowledgeBatchId = batchId;
                }
            } else {
                applyAcknowledgeLocked(entry, entry.lifecycle.snapshot().batchId());
            }
            return;
        }
        if (entry.preemption != null) {
            deferOrdinaryTerminalLocked(entry, terminal);
            return;
        }
        entry.dispatchOwnership = DispatchOwnership.TERMINAL;
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
        if (terminal.decodeWorkerTerminal()) {
            registration.decodeTerminalObserved = true;
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
        } else if (registration.state == PreemptionRegistrationState.CANCEL_REQUESTED
                && registration.acceptedCompletionTimedOut) {
            settleAcceptedCancelFromDecodeTerminalLocked(entry);
        }
    }

    /**
     * ACCEPTED fixes the priority first-cause; an exact Decode terminal fixes
     * resource ownership. Once the typed-completion budget has elapsed, the
     * combination is sufficient to settle as PRIORITY_PREEMPTED without ever
     * replaying the Decode terminal as an ordinary client outcome.
     */
    private boolean settleAcceptedCancelFromDecodeTerminalLocked(InflightEntry entry) {
        PreemptionRegistration registration = entry.preemption;
        if (registration == null
                || registration.state != PreemptionRegistrationState.CANCEL_REQUESTED
                || !registration.acceptedCompletionTimedOut
                || !registration.decodeTerminalObserved) {
            return false;
        }
        DecodeEndpoint endpoint = entry.item.decodeEp();
        if (endpoint == null || !endpoint.settlePriorityCanceled(
                registration.attemptToken, entry.item.requestId())) {
            return false;
        }
        finishPriorityPreemptedLocked(entry, registration, registration.detail);
        return true;
    }

    /** Complete the already token-owned priority outcome exactly once. */
    private void finishPriorityPreemptedLocked(
            InflightEntry entry, PreemptionRegistration registration, String detail) {
        registration.state = PreemptionRegistrationState.SETTLED;
        entry.cleanupOwned = true;
        rollbackOnce(entry);
        removeFromPrefillBatch(entry);
        RequestLifecycleSnapshot terminal = entry.lifecycle.cancel(detail);
        completeError(entry.item.future(), StrategyErrorType.PRIORITY_PREEMPTED, detail);
        finishEntry(entry, terminal);
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
        // A post-handover timeout may have arrived after the ACK was already
        // applied, so there is not necessarily a pending ACK or terminal to
        // replay. Releasing the preemption claim is itself the ownership
        // transition that makes the deferred reconciliation runnable.
        resumePostHandoverReconciliationLocked(entry);
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
            resumePostHandoverReconciliationLocked(entry);
        }
    }

    /** Called with {@code entry} locked after a priority claim releases. */
    private void resumePostHandoverReconciliationLocked(InflightEntry entry) {
        if (!entry.postHandoverReconciliationRequested
                || entry.preemption != null || entry.dispatchReconciliation
                || entry.cleanupOwned || entry.lifecycle.isTerminal()) {
            return;
        }
        if (startDispatchReconciliationLocked(entry)) {
            CompletableFuture.runAsync(() -> reconcileUncertainDispatch(entry, 0));
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
        long oldestExpiredAgeMs = 0;
        List<Long> expiredRequestSamples = new ArrayList<>(3);
        List<InflightEntry> reconciliationStarts = new ArrayList<>();
        for (Map.Entry<Long, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            long idleMs = now - entry.updatedAtMs();
            if (idleMs <= ttlMs) {
                continue;
            }
            synchronized (dispatchFence) {
              synchronized (entry) {
                idleMs = now - entry.updatedAtMs();
                if (inflight.get(candidate.getKey()) != entry || idleMs <= ttlMs
                        || entry.preemption != null || entry.dispatchReconciliation
                        || entry.cleanupOwned) {
                    // Cancel ambiguity is reconciled by token/WorkerStatus;
                    // a concurrent cleanup owner is likewise already settling
                    // the entry and must not be raced by TTL.
                    continue;
                }
                if (entry.lifecycle.hasDispatchClaim()) {
                    // Once startDispatch assigned a positive batch id, TTL no
                    // longer proves that the Engine did not observe the
                    // request. Keep both ledgers and enter the same Cancel
                    // reconciliation used by an uncertain dispatch callback.
                    if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED) {
                        applyAcknowledgeLocked(
                                entry, entry.lifecycle.snapshot().batchId());
                    } else if (startDispatchReconciliationLocked(entry)) {
                        reconciliationStarts.add(entry);
                    }
                    continue;
                }
                oldestExpiredAgeMs = Math.max(oldestExpiredAgeMs, idleMs);
                if (expiredRequestSamples.size() < 3) {
                    expiredRequestSamples.add(candidate.getKey());
                }
                timeoutEntry(entry, "inflight TTL expired");
                expiredCount++;
              }
            }
        }
        for (InflightEntry entry : reconciliationStarts) {
            reconcileUncertainDispatch(entry, 0);
        }
        if (expiredCount > 0) {
            reporter.reportInflightTtlExpired(expiredCount);
            Logger.info("event=scheduler_inflight_ttl_eviction evicted={} "
                            + "oldest_age_ms={} ttl_ms={} request_samples={}",
                    expiredCount, oldestExpiredAgeMs, ttlMs, expiredRequestSamples);
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
        WorkerBatcher batcher = prefillEp != null ? prefillEp.getBatcher() : null;

        // [SYNC] Compute prediction and commit only active items to endpoint
        long predMs = 0;
        long batchId = batchIdGenerator.nextBatchId();
        long decodeConcurrencyLimit = configService.loadBalanceConfig()
                .getDecodeConcurrencyLimit();
        List<BatchItem> dispatchable = new ArrayList<>(items.size());
        for (BatchItem item : items) {
            boolean callbackOwnsPending = false;
            // stageForDispatch removed this item from the live queue while
            // retaining its capacity slot. Claim callback ownership before
            // any Decode or lifecycle mutation; shutdown drains only items
            // that have not crossed this fence.
            if (batcher != null) {
                BatcherContext.PendingClaimResult pendingClaim =
                        batcher.claimPendingDispatch(item);
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
                synchronized (entry) {
                    if (!item.future().isDone()
                            && !entry.lifecycle.isTerminal() && !entry.cleanupOwned) {
                        DecodeEndpoint.DispatchClaimResult claim = item.decodeEp() == null
                                ? DecodeEndpoint.DispatchClaimResult.CLAIMED
                                : item.decodeEp().tryClaimEngineDispatch(
                                        item.requestId(), decodeConcurrencyLimit);
                        if (claim == DecodeEndpoint.DispatchClaimResult.CAPACITY_FULL) {
                            restorePendingDispatch(batcher, item);
                        } else if (claim == DecodeEndpoint.DispatchClaimResult.CLAIMED) {
                            entry.lifecycle.startDispatch(batchId);
                            dispatchable.add(item);
                        } else {
                            // A scheduler preemption claim intentionally owns the
                            // later terminal. Missing endpoint ownership without
                            // such a claim is an invariant violation; fail it now
                            // instead of leaving an item outside both queue and
                            // engine indefinitely.
                            if (entry.preemption == null) {
                                reduceOrdinaryTerminalLocked(entry,
                                        DeferredTerminal.failure(
                                                StrategyErrorType.BATCH_DISPATCH_FAILED,
                                                "Decode dispatch ownership lost before send",
                                                false));
                            }
                        }
                    }
                }
            } catch (Throwable claimFailure) {
                failClaimedBeforeSend(item, claimFailure);
            } finally {
                // CAPACITY_FULL has already restored and removed its pending
                // record. Every other callback-owned outcome is terminal or
                // dispatching and must release the charged queue slot here.
                if (callbackOwnsPending) {
                    completePendingDispatch(batcher, item);
                }
            }
        }

        if (dispatchable.isEmpty()) {
            return;
        }
        boolean dispatcherEntered = false;
        try {
            if (prefillEp != null) {
                PrefillTimePredictor predictor = prefillEp.getPredictor();
                predMs = (long) predictor.predictBatchMs(dispatchable);
            }
            synchronized (dispatchFence) {
                // Prediction may yield to an admission deadline/cancel fence.
                // Revalidate dispatch ownership immediately before the first
                // externally visible commit/send step.
                dispatchable = dispatchable.stream().filter(item -> {
                    InflightEntry entry = entryFor(item);
                    if (entry == null) {
                        return false;
                    }
                    synchronized (entry) {
                        return inflight.get(item.requestId()) == entry
                                && !entry.cleanupOwned
                                && !entry.lifecycle.isTerminal()
                                && entry.lifecycle.snapshot().batchId() == batchId;
                    }
                }).toList();
                if (dispatchable.isEmpty()) {
                    return;
                }
                if (prefillEp != null) {
                    prefillEp.commitBatch(batchId, predMs, dispatchable);
                }

                // [ASYNC] Delegate gRPC dispatch — dispatcher owns its own thread pool
                long nowMs = System.currentTimeMillis();
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

                // Record dispatch timestamp for dispatch-to-ACK latency metric
                for (BatchItem item : dispatchable) {
                    InflightEntry entry = entryFor(item);
                    if (entry != null) {
                        entry.lifecycle.markDispatched();
                        item.ctx().setBatchDispatchedNanos(System.nanoTime());
                    }
                }

                // Submitting the RPC while holding the fence closes the final
                // validate->commit->send gap. Callback reconciliation reacquires
                // the same fence after this method returns.
                dispatcherEntered = true;
                dispatcher.dispatch(dispatchable, prefillEp, batchId, predMs, reason, this);
            }
        } catch (Throwable preSendFailure) {
            if (dispatcherEntered) {
                // A dispatcher implementation may throw after starting its
                // network invocation. Preserve both ledgers and use the same
                // request-id cancel fence as an asynchronous lost ACK.
                for (BatchItem item : dispatchable) {
                    onDispatchUncertain(item, batchId, preSendFailure);
                }
            } else {
                if (prefillEp != null) {
                    prefillEp.releaseBatch(batchId);
                }
                for (BatchItem item : dispatchable) {
                    failClaimedBeforeSend(item, preSendFailure);
                }
            }
        }
    }

    private void failClaimedBeforeSend(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return;
        }
        synchronized (entry) {
            reduceOrdinaryTerminalLocked(entry, DeferredTerminal.failure(
                    StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Batch dispatch preparation failed: " + error.getMessage(),
                    true));
        }
    }

    private static void completePendingDispatch(WorkerBatcher batcher,
                                                BatchItem item) {
        if (batcher != null) {
            batcher.completePendingDispatch(item);
        }
    }

    private void restorePendingDispatch(WorkerBatcher batcher,
                                        BatchItem item) {
        if (batcher == null) {
            return;
        }
        BatcherContext.PendingRestoreResult result = batcher.restorePendingDispatch(item);
        if (result == BatcherContext.PendingRestoreResult.STOPPED) {
            onOfferFailure(item,
                    new java.util.concurrent.CancellationException(
                            "FlexLB batcher stopped while Decode capacity was full"));
        }
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

        synchronized (dispatchFence) {
          synchronized (entry) {
            if (entry.cleanupOwned) {
                return;
            }
            long assignedBatchId = entry.lifecycle.snapshot().batchId();
            if (batchId != assignedBatchId) {
                return;
            }
            if (entry.dispatchReconciliation) {
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
                if (entry.dispatchReconciliation) {
                    return;
                }
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
            if (entry.dispatchReconciliation) {
                return;
            }
            reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                    "EnqueueBatch deadline exceeded: " + error.getMessage()));
        }
    }

    @Override
    public void onDispatchUncertain(BatchItem item, long batchId, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return;
        }
        boolean start = false;
        synchronized (dispatchFence) {
            synchronized (entry) {
                RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
                if (inflight.get(item.requestId()) != entry || entry.cleanupOwned
                        || entry.item.future().isDone()
                        || entry.lifecycle.isTerminal()
                        || snapshot.state() == RequestLifecycleState.ACKNOWLEDGED
                        || snapshot.batchId() != batchId) {
                    return;
                }
                if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED) {
                    applyAcknowledgeLocked(entry, batchId);
                    return;
                }
                start = startDispatchReconciliationLocked(entry);
            }
        }
        if (start) {
            reconcileUncertainDispatch(entry, 0);
        }
    }

    /** Called with {@code entry} locked. */
    private boolean startDispatchReconciliationLocked(InflightEntry entry) {
        if (entry.dispatchReconciliation || entry.preemption != null) {
            return false;
        }
        if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED) {
            applyAcknowledgeLocked(entry, entry.lifecycle.snapshot().batchId());
            return false;
        }
        entry.postHandoverReconciliationRequested = false;
        entry.dispatchReconciliation = true;
        long batchId = entry.lifecycle.snapshot().batchId();
        PrefillEndpoint prefill = entry.item.prefillEp();
        if (prefill != null && batchId > 0) {
            prefill.beginDispatchReconciliation(batchId, entry.item.requestId());
        }
        return true;
    }

    private void reconcileUncertainDispatch(InflightEntry entry, int attempt) {
        CompletableFuture<EngineCancelChannel.CancelOutcome> cancelFuture = null;
        Throwable synchronousFailure = null;
        boolean generationRetired = false;
        synchronized (entry) {
            if (inflight.get(entry.item.requestId()) != entry
                    || !entry.dispatchReconciliation || entry.cleanupOwned
                    || entry.lifecycle.isTerminal()) {
                return;
            }
            PrefillEndpoint prefillGeneration = entry.item.prefillEp();
            if (prefillGeneration == null) {
                generationRetired = true;
            } else {
                try {
                    long timeoutMs = Math.max(1,
                            configService.loadBalanceConfig().getAutoTpmCancelAckTimeoutMs());
                    EngineCancelChannel.CancelTarget target =
                            new EngineCancelChannel.CancelTarget(prefillGeneration);
                    // Hold only the original endpoint generation's synchronous
                    // invocation boundary. The lease is released as soon as
                    // cancel() returns its future.
                    cancelFuture = prefillGeneration.initiateGenerationDispatch(
                            List.of(), () -> engineCancelChannel.cancel(
                                    target, entry.item.requestId(), timeoutMs));
                } catch (EndpointRetiredException retired) {
                    generationRetired = true;
                } catch (Throwable error) {
                    synchronousFailure = error;
                }
            }
        }
        if (generationRetired) {
            // Retirement prevents this generation from receiving any new
            // control RPC, but it is not proof that an earlier EnqueueBatch
            // failed to reach Prefill or Decode. Keep reconciliation and both
            // ledgers fenced until the exact Decode generation reports
            // ownership/terminal state (or typed Prefill CANCELED was already
            // observed). Never redirect the old request-id Cancel to a
            // replacement process at the same address.
            return;
        }
        if (synchronousFailure != null) {
            Logger.warn("EnqueueBatch reconciliation Cancel threw synchronously: "
                            + "request_id={} batch_id={} attempt={}",
                    entry.item.requestId(), entry.lifecycle.snapshot().batchId(), attempt,
                    synchronousFailure);
            handleDispatchReconciliation(
                    entry, EngineCancelChannel.CancelOutcome.failed(), attempt);
            return;
        }
        if (cancelFuture == null) {
            Logger.warn("EnqueueBatch reconciliation Cancel returned null future: "
                            + "request_id={} batch_id={} attempt={}",
                    entry.item.requestId(), entry.lifecycle.snapshot().batchId(), attempt);
            handleDispatchReconciliation(
                    entry, EngineCancelChannel.CancelOutcome.failed(), attempt);
            return;
        }
        cancelFuture.exceptionally(ignored -> EngineCancelChannel.CancelOutcome.failed())
                .thenAccept(outcome -> handleDispatchReconciliation(
                        entry,
                        outcome == null ? EngineCancelChannel.CancelOutcome.failed() : outcome,
                        attempt));
    }

    private void handleDispatchReconciliation(
            InflightEntry entry,
            EngineCancelChannel.CancelOutcome outcome,
            int attempt) {
        boolean retry = false;
        synchronized (dispatchFence) {
            synchronized (entry) {
                if (inflight.get(entry.item.requestId()) != entry
                        || !entry.dispatchReconciliation || entry.cleanupOwned
                        || entry.lifecycle.isTerminal()) {
                    return;
                }
                switch (outcome.ack()) {
                    case TOMBSTONED -> {
                        reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                                "EnqueueBatch deadline exceeded; engine fenced late enqueue"));
                    }
                    case ACCEPTED -> {
                        // ACCEPTED is deliberately a weak acknowledgement: it
                        // proves intent installation, not terminal execution.
                        // Continue the same serialized retry chain until the
                        // Engine returns TOMBSTONED or typed WorkerStatus
                        // CANCELED settles the exact batch member.
                        retry = true;
                    }
                    case NOT_FOUND, FAILED, UNSUPPORTED -> retry = true;
                }
            }
        }
        if (retry) {
            long delayMs = Math.min(5_000L, 100L << Math.min(attempt, 5));
            CompletableFuture.delayedExecutor(delayMs, TimeUnit.MILLISECONDS)
                    .execute(() -> reconcileUncertainDispatch(entry, attempt + 1));
        }
    }

    private void clearDispatchReconciliation(InflightEntry entry) {
        if (!entry.dispatchReconciliation) {
            return;
        }
        entry.dispatchReconciliation = false;
        long batchId = entry.lifecycle.snapshot().batchId();
        PrefillEndpoint prefill = entry.item.prefillEp();
        if (prefill != null && batchId > 0) {
            prefill.endDispatchReconciliation(batchId, entry.item.requestId());
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
        clearDispatchReconciliation(entry);
        entry.postHandoverReconciliationRequested = false;
        if (entry.admissionLease != null) {
            entry.admissionLease.completeSchedulerSettlement();
        }
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

    private static final class AdmissionClaim {
        private final CompletableFuture<Response> future;
        private int holds = 1;
        private boolean publicCompleted;

        private AdmissionClaim(CompletableFuture<Response> future) {
            this.future = future;
        }
    }

    private static final class InflightEntry {
        final BatchItem item;
        final RequestLifecycle lifecycle;
        final boolean autoTpmAdmission;
        final AtomicBoolean rolledBack = new AtomicBoolean(false);
        AdmissionLease admissionLease;
        DispatchOwnership dispatchOwnership = DispatchOwnership.ACK_PENDING;
        boolean cleanupOwned;
        PreemptionRegistration preemption;
        boolean dispatchReconciliation;
        boolean postHandoverReconciliationRequested;

        InflightEntry(BatchItem item, boolean autoTpmAdmission) {
            this.item = Objects.requireNonNull(item);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
            this.lifecycle = new RequestLifecycle(item.requestId());
            this.autoTpmAdmission = autoTpmAdmission;
        }

        long updatedAtMs() {
            return lifecycle.snapshot().updatedAtMs();
        }

        boolean hasPreemption(long attemptToken) {
            return preemption != null && preemption.attemptToken == attemptToken;
        }
    }

    /** Linearized ownership decision between Enqueue and Decode WorkerStatus. */
    private enum DispatchOwnership {
        ACK_PENDING,
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

    private static final class PreemptionRegistration {
        private final long attemptToken;
        private final String detail;
        private final CompletableFuture<PriorityCanceledObservation> priorityCanceled =
                new CompletableFuture<>();
        /** First exact typed CANCELED observation captured under the entry fence. */
        private boolean priorityCanceledObserved;
        /** ACCEPTED waited its full typed-completion budget without a typed terminal. */
        private boolean acceptedCompletionTimedOut;
        /** Exact Decode terminal observed while the priority claim owned settlement. */
        private boolean decodeTerminalObserved;
        private PreemptionRegistrationState state = PreemptionRegistrationState.CLAIMED;
        private DeferredTerminal pendingTerminal;
        private long pendingAcknowledgeBatchId;

        private PreemptionRegistration(long attemptToken, String detail) {
            this.attemptToken = attemptToken;
            this.detail = detail;
        }
    }

    /** Publishes the already-fenced typed observation after scheduler locks are released. */
    private record PriorityCanceledEffect(
            CompletableFuture<PriorityCanceledObservation> signal,
            PriorityCanceledObservation observation) {
        void publish() {
            signal.complete(observation);
        }
    }

    private record WorkerTerminalObservation(boolean prefill,
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

        boolean decodeWorkerTerminal() {
            return authoritativeWorker() && !workerObservation.prefill();
        }

        boolean dispatchAckFailure() {
            return (kind == DeferredTerminalKind.FAILURE && removeFromPrefillBatch)
                    || kind == DeferredTerminalKind.TIMEOUT;
        }
    }
}
