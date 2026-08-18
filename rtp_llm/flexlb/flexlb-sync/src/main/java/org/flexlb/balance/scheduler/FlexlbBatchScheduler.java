package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
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
                    if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED
                            && !configService.loadBalanceConfig()
                                .isFlexlbAckOnlyReleaseEnabled()) {
                        // Legacy shortcut: Decode WorkerStatus is stronger than
                        // the missing Enqueue ACK. Publish the logical ACK now;
                        // otherwise the admission future would remain pending
                        // forever. Under the ack-only gate the reconciliation
                        // below settles this entry through the S1 semantic
                        // (late ACK / Prefill observation) instead.
                        applyAcknowledgeLocked(entry,
                                entry.lifecycle.snapshot().batchId());
                        return;
                    }
                    // startDispatch(batchId) is the point of no return. The
                    // batcher already owns a local snapshot and may publish it
                    // after this lock is released. Completing 8431 and deleting
                    // the ledgers here would let the frontend retry while the
                    // original request is still accepted by the engine.
                    Logger.debug("Admission deadline observed after dispatch claim; "
                                    + "starting Engine fence reconciliation: "
                                    + "request_id={} batch_id={} lifecycle={}",
                            requestId, entry.lifecycle.snapshot().batchId(),
                            entry.lifecycle.snapshot().state());
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
            // N1: a settle on an unknown id is harmless (already terminal or
            // never registered) but worth surfacing — a burst points at a
            // registration/cleanup race.
            Logger.debug("finishPreemptedById miss: request_id={} not inflight, detail={}",
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
            Logger.debug("finishYieldedById miss: request_id={} not inflight, detail={}",
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
            // A typed priority-cancel terminal is allowed to omit the original
            // Prefill batch id (production Engine currently reports -1).  The
            // lifecycle still owns the authoritative dispatch generation, so
            // retire this member from the Master-side batch ledger here instead
            // of waiting for the generic inflight TTL to unblock the worker.
            // repackBatch is member-scoped and idempotent: endpoint calibration
            // may already have removed the same member before this callback.
            removeFromPrefillBatch(entry);
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
                // Ack-only release: an active Prefill observation of the same
                // dispatch generation proves the engine stored the fetch slot
                // even though the Enqueue ACK never fired.
                releaseOnPrefillObserved(task.getRequestId(), task.getBatchId());
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
            synchronized (dispatchFence) {
              synchronized (entry) {
                if (isPrefill && entry.dispatchReconciliation) {
                    RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
                    boolean authoritativeCanceled = task.getBatchId() == snapshot.batchId()
                            && task.getPriorityPreemptionProgress()
                                == PriorityPreemptionProgress.CANCELED
                            && task.getErrorCode() == ENGINE_ERROR_PRIORITY_PREEMPTED;
                    if (authoritativeCanceled) {
                        // Only the typed cancel terminal for the exact dispatch
                        // generation proves that the ambiguous Engine ownership
                        // has been settled. Ordinary Prefill success/error may
                        // belong to normal execution (or stale status) and must
                        // not roll back the Master ledgers underneath it.
                        reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                                "EnqueueBatch reconciled by typed Prefill CANCELED"));
                    } else if (task.getErrorCode() == 0
                            && task.getBatchId() == snapshot.batchId()
                            && configService.loadBalanceConfig()
                                .isFlexlbAckOnlyReleaseEnabled()) {
                        // Ack-only release: a successful Prefill terminal of the
                        // exact dispatch generation proves the engine accepted
                        // and stored this batch member even though the Enqueue
                        // ACK never fired (covers members whose running window
                        // was skipped by polling). Release through the single
                        // S1 semantic instead of retaining on the cancel fence.
                        Logger.info("event=ack_only_release source=prefill_finished "
                                        + "request_id={} batch_id={}",
                                requestId, snapshot.batchId());
                        clearDispatchReconciliation(entry);
                        applyAcknowledgeLocked(entry, snapshot.batchId());
                    } else {
                        Logger.debug("Ignoring non-authoritative Prefill terminal during "
                                        + "dispatch reconciliation: request_id={} "
                                        + "task_batch_id={} entry_batch_id={} error_code={} "
                                        + "preemption_progress={}",
                                requestId, task.getBatchId(), snapshot.batchId(),
                                task.getErrorCode(), task.getPriorityPreemptionProgress());
                    }
                    continue;
                }
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
                if (isDecode) {
                    // A Decode terminal proves acceptance even when no active
                    // phase was reported. Linearize that ownership transfer
                    // with the terminal reducer under the same entry lock, so
                    // a racing Enqueue failure cannot publish false success in
                    // between the two observations.
                    markDecodeAcceptedLocked(entry);
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
    }

    private void markDecodeAccepted(long requestId) {
        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            return;
        }
        synchronized (dispatchFence) {
            synchronized (entry) {
                if (inflight.get(requestId) != entry) {
                    return;
                }
                markDecodeAcceptedLocked(entry);
                if (entry.dispatchReconciliation
                        && !configService.loadBalanceConfig()
                            .isFlexlbAckOnlyReleaseEnabled()) {
                    // Legacy shortcut: Decode KV ownership is stronger than a
                    // missing Prefill Enqueue ACK. Stop the Prefill cancel-fence
                    // retry chain and publish the logical ACK while both
                    // ownership paths are linearized by the same dispatch fence.
                    // Under the ack-only gate Decode observations only record
                    // ownership; release waits for the S1 semantic (late ACK /
                    // Prefill observation).
                    clearDispatchReconciliation(entry);
                    applyAcknowledgeLocked(entry, entry.lifecycle.snapshot().batchId());
                }
            }
        }
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
     * Ack-only release: a Prefill WorkerStatus observation (running task or
     * successful terminal) of the exact dispatch generation happens strictly
     * after the engine stored the deferred fetch slot, so it carries the same
     * "prefill accepted" semantic as the EnqueueBatch ACK. It releases entries
     * whose ACK never fired (uncertain dispatch) instead of the legacy Decode
     * shortcut, keeping the release gate single-sourced on Prefill evidence.
     */
    private void releaseOnPrefillObserved(long requestId, long batchId) {
        if (!configService.loadBalanceConfig().isFlexlbAckOnlyReleaseEnabled()) {
            return;
        }
        InflightEntry entry = inflight.get(requestId);
        if (entry == null || !entry.dispatchReconciliation) {
            return;
        }
        synchronized (dispatchFence) {
            synchronized (entry) {
                if (inflight.get(requestId) != entry || entry.cleanupOwned
                        || !entry.dispatchReconciliation
                        || batchId != entry.lifecycle.snapshot().batchId()) {
                    return;
                }
                Logger.info("event=ack_only_release source=prefill_observed "
                                + "request_id={} batch_id={}",
                        requestId, batchId);
                clearDispatchReconciliation(entry);
                applyAcknowledgeLocked(entry, batchId);
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
        if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED
                && terminal.dispatchAckFailure()
                && !configService.loadBalanceConfig().isFlexlbAckOnlyReleaseEnabled()) {
            // Legacy shortcut: KV_ALLOCATED/RUNNING is a stronger ownership
            // observation than an absent/failed Enqueue ACK. Preserve the live
            // inflight entry, Decode accounting, and public schedule success.
            // Under the ack-only gate an explicit Enqueue failure/timeout takes
            // its ordinary terminal below: a rejected member must not be
            // published as schedule success (its fetch slot never exists).
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

    /**
     * Whether the scheduler still owns an inflight entry for {@code requestId}.
     * Used by endpoint hard-age eviction as a race guard: an endpoint ledger
     * entry must not be force-evicted while the scheduler lifecycle still
     * references the same request id (the scheduler's own cleanup — TTL or
     * hard cap — will cascade the endpoint release instead).
     */
    public boolean hasInflightRequest(long requestId) {
        return inflight.containsKey(requestId);
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
        long hardMaxAgeMs = configService.loadBalanceConfig().getFlexlbInflightHardMaxAgeMs();
        long now = System.currentTimeMillis();
        int expiredCount = 0;
        // Hard-age-cap subset of expiredCount, split out for the reason tag.
        int hardCapCount = 0;
        // F3 observability: entries past the TTL retained only by a fence
        // (preemption / dispatch reconciliation / cleanup ownership). A
        // persistently non-zero rate is the inflight-leak signature.
        int skippedFenced = 0;
        long oldestExpiredAgeMs = 0;
        List<Long> expiredRequestSamples = new ArrayList<>(3);
        for (Map.Entry<Long, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            long ageMs = now - entry.createdAtMs();
            if (ageMs <= ttlMs) {
                continue;
            }
            synchronized (entry) {
                if (inflight.get(candidate.getKey()) != entry) {
                    continue;
                }
                if (entry.preemption != null || entry.dispatchReconciliation || entry.cleanupOwned) {
                    // Cancel ambiguity is reconciled by token/WorkerStatus;
                    // a concurrent cleanup owner is likewise already settling
                    // the entry and must not be raced by TTL. The hard age cap
                    // still applies: a reconciliation that never settles (e.g.
                    // zombie cancel overlay in the engine report) must not pin
                    // the entry — and its endpoint ledgers — forever.
                    if (hardMaxAgeMs <= 0 || ageMs <= hardMaxAgeMs) {
                        skippedFenced++;
                        continue;
                    }
                    RequestLifecycleSnapshot snapshot = entry.lifecycle.snapshot();
                    Logger.warn("event=scheduler_inflight_hard_age_eviction request_id={} "
                                    + "age_ms={} hard_max_age_ms={} created_at_ms={} preemption={} "
                                    + "dispatch_reconciliation={} cleanup_owned={} lifecycle_state={} "
                                    + "lifecycle_detail={}",
                            candidate.getKey(), ageMs, hardMaxAgeMs, entry.createdAtMs(),
                            entry.preemption != null, entry.dispatchReconciliation,
                            entry.cleanupOwned, snapshot.state(), snapshot.detail());
                    timeoutEntry(entry, "inflight hard age cap exceeded");
                    oldestExpiredAgeMs = Math.max(oldestExpiredAgeMs, ageMs);
                    if (expiredRequestSamples.size() < 3) {
                        expiredRequestSamples.add(candidate.getKey());
                    }
                    expiredCount++;
                    hardCapCount++;
                    continue;
                }
                oldestExpiredAgeMs = Math.max(oldestExpiredAgeMs, ageMs);
                if (expiredRequestSamples.size() < 3) {
                    expiredRequestSamples.add(candidate.getKey());
                }
                timeoutEntry(entry, "inflight TTL expired");
                expiredCount++;
            }
        }
        if (expiredCount - hardCapCount > 0) {
            reporter.reportSchedulerInflightTtlExpired("ttl", expiredCount - hardCapCount);
        }
        if (hardCapCount > 0) {
            reporter.reportSchedulerInflightTtlExpired("hard_age_cap", hardCapCount);
        }
        if (skippedFenced > 0) {
            reporter.reportInflightCleanupSkippedFenced(skippedFenced);
        }
        if (expiredCount > 0 || skippedFenced > 0) {
            Logger.info("event=scheduler_inflight_ttl_eviction evicted={} "
                            + "oldest_age_ms={} ttl_ms={} skipped_fenced={} request_samples={}",
                    expiredCount, oldestExpiredAgeMs, ttlMs, skippedFenced, expiredRequestSamples);
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
            int orphanReclaimed = 0;
            for (Map.Entry<Long, RequestInflight> reserved : decodeEp.reservedView().entrySet()) {
                long requestId = reserved.getKey();
                if (now - reserved.getValue().createdAtMs() > ttlMs
                        && !inflight.containsKey(requestId)) {
                    decodeEp.release(requestId);
                    orphanReclaimed++;
                    Logger.warn("orphan decode reservation reclaimed: request_id={} worker={} age_ms={}",
                            requestId, decodeEntry.getKey(),
                            now - reserved.getValue().createdAtMs());
                }
            }
            if (orphanReclaimed > 0) {
                reporter.reportEndpointInflightTtlExpired(RoleType.DECODE.name(),
                        decodeEp.getIp(), "orphan_reservation", orphanReclaimed);
            }
        }
    }

    // ==================== Post-ACK inflight audit (F1) ====================

    /**
     * F1 backstop: force-settle ledger entries the ordinary paths can no
     * longer reach. An entry qualifies only when ALL of the following hold:
     * <ol>
     *   <li>age exceeds {@code flexlbInflightAuditAfterMs} (the ACK round is
     *       long over),</li>
     *   <li>its public future is already completed — the client saw a
     *       terminal response, so no one is waiting,</li>
     *   <li>no fence retains it (preemption claim / dispatch reconciliation /
     *       cleanup ownership — those have their own reconciliation),</li>
     *   <li>neither side is visible: the prefill batch ledger no longer
     *       tracks the request, and on the decode side neither the engine-
     *       confirmed registry (KV allocated / running) nor the shadow
     *       reservation layer still holds it (or either endpoint is
     *       absent).</li>
     * </ol>
     * Such an entry is a post-ACK leak: nothing observable can settle it, yet
     * it keeps charging the inflight capacity and the endpoint ledgers until
     * the TTL/hard-cap eviction notices it minutes later. The audit clears it
     * in seconds.
     *
     * <p>Decode visibility spans BOTH admission layers (R1): a request queued
     * inside a saturated decode engine is not yet engine-confirmed (no KV
     * allocated), but its shadow reservation is still live — force-settling
     * the entry would roll that reservation back and oversell admission KV.
     * Only when neither layer holds the request is decode truly invisible.
     *
     * <p>Lock order mirrors {@link #cleanupInflight()} (R5): the visibility
     * probes ({@link PrefillEndpoint#tracksRequest} /
     * {@link DecodeEndpoint#isEngineConfirmed} /
     * {@link DecodeEndpoint#isReserved}) are lock-free CHM reads and run
     * OUTSIDE the entry monitor so the O(batches × members) prefill scan
     * never extends a monitor critical section; the monitor is then taken
     * only for the re-verify ({@code inflight.get(key) == entry}), the fence
     * checks, and the settlement via {@link #timeoutEntry} (which never
     * takes the dispatch fence — so nesting it here cannot invert the global
     * order). A visibility verdict sampled before the monitor can only be
     * stale in the conservative direction: a request that became visible
     * after sampling is still protected by the entry's fence/future checks
     * on the next audit tick.
     */
    @Scheduled(fixedRateString = "${flexlb.inflight.audit.rate.ms:10000}")
    public void auditInflight() {
        long auditAfterMs = configService.loadBalanceConfig().getFlexlbInflightAuditAfterMs();
        if (auditAfterMs <= 0) {
            return;
        }
        long now = System.currentTimeMillis();
        for (Map.Entry<Long, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            long ageMs = now - entry.createdAtMs();
            if (ageMs <= auditAfterMs) {
                continue;
            }
            // Visibility probes outside the entry monitor (R5) — lock-free
            // CHM reads. Decode visible = engine-confirmed OR shadow
            // reservation still held (R1); a null endpoint stays invisible.
            PrefillEndpoint prefillEp = entry.item.prefillEp();
            DecodeEndpoint decodeEp = entry.item.decodeEp();
            boolean prefillVisible = prefillEp != null
                    && prefillEp.tracksRequest(candidate.getKey());
            boolean decodeVisible = decodeEp != null
                    && (decodeEp.isEngineConfirmed(candidate.getKey())
                            || decodeEp.isReserved(candidate.getKey()));
            synchronized (entry) {
                if (inflight.get(candidate.getKey()) != entry) {
                    continue;
                }
                if (entry.preemption != null || entry.dispatchReconciliation || entry.cleanupOwned) {
                    continue;
                }
                if (!entry.item.future().isDone()) {
                    continue;
                }
                if (prefillVisible || decodeVisible) {
                    continue;
                }
                Logger.warn("event=scheduler_inflight_audit_release request_id={} "
                                + "age_ms={} threshold_ms={} reason=post_ack_both_sides_invisible",
                        candidate.getKey(), ageMs, auditAfterMs);
                timeoutEntry(entry, "post-ACK inflight audit: both endpoints invisible");
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
                Logger.debug("flexlb_batch_dispatch batch_id={} reason={} batch_size={} wait_ms={} "
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
        if (result == BatcherContext.PendingRestoreResult.RESTORED) {
            // F3 observability: how often a dispatched item bounces back to
            // the batcher queue (decode-capacity full / pre-send failure).
            reporter.reportSchedulerRestorePendingDispatch();
        }
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
                Logger.debug("Ignoring stale EnqueueBatch ACK request_id={} batch_id={}",
                        item.requestId(), batchId);
                return;
            }
            if (entry.dispatchReconciliation) {
                if (!configService.loadBalanceConfig().isFlexlbAckOnlyReleaseEnabled()) {
                    Logger.debug("Retaining late EnqueueBatch ACK during reconciliation: "
                                    + "request_id={} batch_id={}",
                            item.requestId(), batchId);
                    return;
                }
                // Ack-only release: the ACK is the S1 release semantic itself.
                // It proves the engine stored the fetch slot for this exact
                // dispatch generation (batch id already validated above), so a
                // late arrival dissolves the reconciliation uncertainty instead
                // of being dropped.
                Logger.info("event=ack_only_release source=late_enqueue_ack "
                                + "request_id={} batch_id={}",
                        item.requestId(), batchId);
                clearDispatchReconciliation(entry);
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
                if (entry.dispatchReconciliation) {
                    Logger.debug("Retaining EnqueueBatch failure during reconciliation: "
                                    + "request_id={} batch_id={} cause={}",
                            item.requestId(), entry.lifecycle.snapshot().batchId(),
                            error == null ? "unknown" : error.getMessage());
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
                Logger.debug("Retaining EnqueueBatch timeout during reconciliation: "
                                + "request_id={} batch_id={} cause={}",
                        item.requestId(), entry.lifecycle.snapshot().batchId(),
                        error == null ? "unknown" : error.getMessage());
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
                if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED
                        && !configService.loadBalanceConfig()
                            .isFlexlbAckOnlyReleaseEnabled()) {
                    // Legacy shortcut: Decode ownership settles an uncertain
                    // dispatch without the fence. The ack-only gate instead
                    // starts reconciliation and waits for the S1 semantic.
                    applyAcknowledgeLocked(entry, batchId);
                    return;
                }
                start = startDispatchReconciliationLocked(entry);
            }
        }
        if (start) {
            Logger.debug("EnqueueBatch ACK uncertain; fencing before settlement: "
                            + "request_id={} batch_id={} engine={} cause={}",
                    item.requestId(), batchId,
                    item.prefillEp() != null ? item.prefillEp().ipPort() : "unknown",
                    error == null ? "deadline exceeded" : error.getMessage());
            reconcileUncertainDispatch(entry, 0);
        }
    }

    /** Called with {@code entry} locked. */
    private boolean startDispatchReconciliationLocked(InflightEntry entry) {
        if (entry.dispatchReconciliation || entry.preemption != null) {
            return false;
        }
        if (entry.dispatchOwnership == DispatchOwnership.DECODE_OWNED
                && !configService.loadBalanceConfig().isFlexlbAckOnlyReleaseEnabled()) {
            // Legacy shortcut mirror of onDispatchUncertain's DECODE_OWNED
            // branch for callers entering reconciliation directly.
            applyAcknowledgeLocked(entry, entry.lifecycle.snapshot().batchId());
            return false;
        }
        entry.dispatchReconciliation = true;
        long batchId = entry.lifecycle.snapshot().batchId();
        PrefillEndpoint prefill = entry.item.prefillEp();
        if (prefill != null && batchId > 0) {
            prefill.beginDispatchReconciliation(batchId, entry.item.requestId());
        }
        Logger.info("event=dispatch_reconciliation_start request_id={} batch_id={} target={}",
                entry.item.requestId(), batchId,
                prefill != null ? prefill.ipPort() : "unknown");
        reporter.reportDispatchReconciliationEvent("start", "uncertain_ack");
        return true;
    }

    private void reconcileUncertainDispatch(InflightEntry entry, int attempt) {
        CompletableFuture<EngineCancelChannel.CancelOutcome> cancelFuture = null;
        Throwable synchronousFailure = null;
        synchronized (dispatchFence) {
            synchronized (entry) {
                if (inflight.get(entry.item.requestId()) != entry
                        || !entry.dispatchReconciliation || entry.cleanupOwned
                        || entry.lifecycle.isTerminal()) {
                    return;
                }
                ServerStatus prefill = entry.item.prefill();
                if (settleIfReconcileTargetDeregisteredLocked(entry, prefill, attempt)) {
                    return;
                }
                EngineCancelChannel.CancelTarget target = prefill == null ? null
                        : new EngineCancelChannel.CancelTarget(
                                prefill.getServerIp(), prefill.getGrpcPort());
                long timeoutMs = Math.max(1,
                        configService.loadBalanceConfig().getAutoTpmCancelAckTimeoutMs());
                try {
                    // Invoke while holding the entry fence so a delayed retry
                    // cannot pass its liveness check, lose to a terminal
                    // WorkerStatus, and then install a stale absent tombstone.
                    cancelFuture = engineCancelChannel.cancel(
                            target, entry.item.requestId(), timeoutMs);
                } catch (Throwable error) {
                    synchronousFailure = error;
                }
            }
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
                        Logger.info("event=dispatch_reconciliation_settled "
                                        + "request_id={} batch_id={} attempt={}",
                                entry.item.requestId(),
                                entry.lifecycle.snapshot().batchId(), attempt);
                        reporter.reportDispatchReconciliationEvent("settled", "engine_tombstoned");
                        reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                                "EnqueueBatch deadline exceeded; engine fenced late enqueue"));
                    }
                    case ACCEPTED -> {
                        // ACCEPTED is deliberately a weak acknowledgement: it
                        // proves intent installation, not terminal execution.
                        // Continue the same serialized retry chain until the
                        // Engine returns TOMBSTONED or typed WorkerStatus
                        // CANCELED settles the exact batch member.
                        Logger.debug("EnqueueBatch reconciliation cancel accepted; "
                                        + "retrying until terminal fence: "
                                        + "request_id={} batch_id={} attempt={}",
                                entry.item.requestId(),
                                entry.lifecycle.snapshot().batchId(), attempt);
                        entry.reconcileConsecutiveFailures = 0;
                        retry = true;
                    }
                    case NOT_FOUND, FAILED, UNSUPPORTED -> retry =
                            !settleIfReconcileFailureCapReachedLocked(entry, outcome, attempt);
                }
            }
        }
        if (retry) {
            long delayMs = Math.min(5_000L, 100L << Math.min(attempt, 5));
            CompletableFuture.delayedExecutor(delayMs, TimeUnit.MILLISECONDS)
                    .execute(() -> reconcileUncertainDispatch(entry, attempt + 1));
        }
    }

    /**
     * Fix A (source-level stop): terminal-settles a fenced uncertain dispatch
     * whose Cancel target has left the EndpointRegistry. The Cancel address is
     * frozen at dispatch time ({@code entry.item.prefill()}), so after a
     * rolling deploy the retry chain keeps calling a dead pod forever (FAILED
     * loop) while the TTL sweep skips fenced entries. Registry removal is
     * driven by ExpirationCleaner only after workerTimeoutMs of status
     * silence, and a second grace window is applied here, so a deregistered
     * target means the engine process is gone and the request cannot still be
     * running — the ordinary terminal path (rollback + fence release) is safe.
     *
     * <p>Called with {@code dispatchFence} and {@code entry} locked.
     *
     * @return true when the entry was settled and the retry chain must stop
     */
    private boolean settleIfReconcileTargetDeregisteredLocked(InflightEntry entry,
                                                              ServerStatus prefill,
                                                              int attempt) {
        long graceMs = configService.loadBalanceConfig()
                .getFlexlbReconcileTargetMissingTerminalMs();
        if (graceMs <= 0 || prefill == null) {
            return false;
        }
        String ipPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
        RoleType role = prefill.getRole() != null ? prefill.getRole() : RoleType.PREFILL;
        if (endpointRegistry.get(role, ipPort) != null) {
            entry.reconcileTargetMissingSinceMs = 0;
            return false;
        }
        long now = System.currentTimeMillis();
        if (entry.reconcileTargetMissingSinceMs == 0) {
            entry.reconcileTargetMissingSinceMs = now;
            return false;
        }
        if (now - entry.reconcileTargetMissingSinceMs < graceMs) {
            return false;
        }
        Logger.warn("event=dispatch_reconciliation_forced_terminal "
                        + "reason=target_deregistered request_id={} batch_id={} "
                        + "target={} attempt={} missing_ms={}",
                entry.item.requestId(), entry.lifecycle.snapshot().batchId(),
                ipPort, attempt, now - entry.reconcileTargetMissingSinceMs);
        reporter.reportDispatchReconciliationEvent("forced_terminal", "target_deregistered");
        reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                "dispatch reconciliation target deregistered: " + ipPort));
        return true;
    }

    /**
     * Fix B (D3 backstop): caps the otherwise unbounded reconciliation retry
     * chain. FAILED, UNSUPPORTED and NOT_FOUND all count — NOT_FOUND must not
     * settle immediately because only TOMBSTONED installs the absent fence on
     * the engine; a buffered late EnqueueBatch could still land after a bare
     * NOT_FOUND. At the default cap (36 tries ≈ 3 minutes at the 5s backoff
     * ceiling) the EnqueueBatch deadline has long expired, so the
     * late-landing window is closed and forcing the ordinary terminal is safe.
     *
     * <p>Called with {@code dispatchFence} and {@code entry} locked.
     *
     * @return true when the entry was settled and the retry chain must stop
     */
    private boolean settleIfReconcileFailureCapReachedLocked(
            InflightEntry entry,
            EngineCancelChannel.CancelOutcome outcome,
            int attempt) {
        int maxFailures = configService.loadBalanceConfig()
                .getFlexlbReconcileMaxConsecutiveFailures();
        if (maxFailures <= 0) {
            return false;
        }
        entry.reconcileConsecutiveFailures++;
        if (entry.reconcileConsecutiveFailures < maxFailures) {
            return false;
        }
        ServerStatus prefill = entry.item.prefill();
        Logger.warn("event=dispatch_reconciliation_forced_terminal "
                        + "reason=failure_cap request_id={} batch_id={} target={} "
                        + "attempt={} consecutive_failures={} last_outcome={}",
                entry.item.requestId(), entry.lifecycle.snapshot().batchId(),
                prefill == null ? "unknown"
                        : prefill.getServerIp() + ":" + prefill.getGrpcPort(),
                attempt, entry.reconcileConsecutiveFailures, outcome.ack());
        reporter.reportDispatchReconciliationEvent("forced_terminal", "failure_cap");
        reduceOrdinaryTerminalLocked(entry, DeferredTerminal.timeout(
                "dispatch reconciliation exhausted after "
                        + entry.reconcileConsecutiveFailures + " failed cancels"));
        return true;
    }

    private void clearDispatchReconciliation(InflightEntry entry) {
        if (!entry.dispatchReconciliation) {
            return;
        }
        entry.dispatchReconciliation = false;
        entry.reconcileConsecutiveFailures = 0;
        entry.reconcileTargetMissingSinceMs = 0;
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
            Logger.debug("FlexLB remove from prefill batch: request_id={} batch_id={} engine={}",
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
            Logger.debug("[auto-tpm] admission timeout classified: request_id={} "
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
        clearDispatchReconciliation(entry);
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
        // F3 observability: age of the oldest inflight entry. With a healthy
        // TTL the size gauge alone cannot distinguish "busy" from "leaking";
        // a max age creeping toward the TTL window is the leak signature.
        reporter.reportSchedulerInflightMaxAgeMs(
                InflightEvictor.maxAgeMs(inflight, System.currentTimeMillis()));
        // Live dispatch-reconciliation fence population. Stateless rescan of
        // the ledger (racy unlocked read of the fence flag is fine for a
        // gauge); an incremental counter would drift on a missed decrement.
        int reconciliationFenced = 0;
        for (InflightEntry fenceProbe : inflight.values()) {
            if (fenceProbe.dispatchReconciliation) {
                reconciliationFenced++;
            }
        }
        reporter.reportDispatchReconciliationFenceSize(reconciliationFenced);

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

    /**
     * Implements {@link InflightEvictor.TtlTracked} so the scheduler-level
     * max-age gauge can reuse {@link InflightEvictor#maxAgeMs} (same package,
     * no import needed).
     */
    private static final class InflightEntry implements InflightEvictor.TtlTracked {
        final BatchItem item;
        final RequestLifecycle lifecycle;
        final boolean autoTpmAdmission;
        /**
         * Cached lifecycle creation timestamp. {@link #createdAtMs()} sits on
         * the 2s gauge hot path ({@code InflightEvictor.maxAgeMs} over every
         * entry) and on the TTL/audit age probes; reading this final field
         * avoids the per-call {@link RequestLifecycle#snapshot()} allocation
         * (~8.5k snapshots/s at 17k entries) and its monitor contention with
         * the dispatch/ACK state transitions. The lifecycle timestamp is
         * itself final, so the cached copy is an invariant.
         */
        final long createdAtMs;
        final AtomicBoolean rolledBack = new AtomicBoolean(false);
        AdmissionLease admissionLease;
        DispatchOwnership dispatchOwnership = DispatchOwnership.ACK_PENDING;
        boolean cleanupOwned;
        PreemptionRegistration preemption;
        boolean dispatchReconciliation;
        /** Consecutive failed reconciliation Cancels; reset on ACCEPTED or fence clear. */
        int reconcileConsecutiveFailures;
        /** First millis the Cancel target was missing from the registry; 0 = present. */
        long reconcileTargetMissingSinceMs;

        InflightEntry(BatchItem item, boolean autoTpmAdmission) {
            this.item = Objects.requireNonNull(item);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
            this.lifecycle = new RequestLifecycle(item.requestId());
            this.createdAtMs = this.lifecycle.snapshot().createdAtMs();
            this.autoTpmAdmission = autoTpmAdmission;
        }

        @Override
        public long createdAtMs() {
            return createdAtMs;
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

        boolean dispatchAckFailure() {
            return (kind == DeferredTerminalKind.FAILURE && removeFromPrefillBatch)
                    || kind == DeferredTerminalKind.TIMEOUT;
        }
    }
}
