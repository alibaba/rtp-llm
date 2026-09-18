package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.strategy.LoadBalanceStrategy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

/**
 * Priority admission scheduler for {@code QUEUE + PRIORITY}.
 *
 * <p>Each placement attempt:
 * <ol>
 *   <li>Capture a read-only {@link ClusterSnapshot}</li>
 *   <li>Build a {@link NormalPlacementPlan} by reusing the existing
 *       {@link Router#route} (which also performs the decode reservation),
 *       guaranteeing consistent placement behavior</li>
 *   <li>Commit via {@link PlanCommitter}; release the decode reservation on
 *       rejection and retry only for capacity or victim-presence conflicts</li>
 *   <li>When prefill-queue preemption is allowed and the
 *       offer fails because the prefill queue is full, plan the cheapest
 *       strictly-lower-priority eviction ({@link EvictionPlanner}) and commit
 *       it atomically via {@link PrefillQueueManager#tryReplaceVictimsPresent};
 *       all evicted victims terminate with {@code PRIORITY_PREEMPTED}</li>
 * </ol>
 * A queue-capacity rejection permits one fallback placement. Victim-presence
 * conflicts have a separate budget of {@link #MAX_EVICTION_REPLANS} replans.
 * Rejections retain the decision's error code and admission reason. Prefill and decode
 * victims are considered only when their exact lifecycle stages are listed in
 * {@link org.flexlb.config.PreemptionConfig#getAllowedVictimStages()}.
 */
@Component
public class PriorityAdmissionScheduler {

    /**
     * Maximum replans after victim-presence conflicts, independent of offer retries.
     */
    private static final int MAX_EVICTION_REPLANS = 3;
    /** Sentinel stored in {@link #activeAdmissionCount} after shutdown. */
    private static final int ADMISSION_CLOSED = -1;

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final PlanCommitter planCommitter;
    private final PrioritySchedulerReporter priorityReporter;
    private final BatchSchedulerReporter batchReporter;
    private final EngineCancelChannel cancelChannel;
    private final DecodePreemptionCoordinator preemptionCoordinator;
    private final Map<Long, AtomicInteger> cancelNotFoundReplans = new ConcurrentHashMap<>();

    /** Bean-owned timer; no timeout task outlives this scheduler instance. */
    private final ScheduledThreadPoolExecutor softTimeoutExecutor;
    private final Object softTimeoutLifecycle = new Object();
    /** Leases with an armed soft timeout, bounded by active admissions. */
    private final Set<AdmissionLease> pendingSoftTimeoutLeases =
            ConcurrentHashMap.newKeySet();
    /** Guarded by softTimeoutLifecycle; callbacks themselves always run unlocked. */
    private int activeSoftTimeoutCallbacks;
    private final AdmissionLease.SoftTimeoutScheduler softTimeoutScheduler =
            new AdmissionLease.SoftTimeoutScheduler() {
                @Override
                public ScheduledFuture<?> schedule(AdmissionLease lease,
                                                   Runnable task,
                                                   long delay,
                                                   TimeUnit unit) {
                    return scheduleSoftTimeout(lease, task, delay, unit);
                }

                @Override
                public void onLeaseTerminated(AdmissionLease lease) {
                    pendingSoftTimeoutLeases.remove(lease);
                }
            };
    private volatile boolean shutdown;

    /**
     * Number of admission permits currently held. A permit is reserved before
     * placement starts and remains charged until admission fails or its
     * {@link AdmissionLease} closes. The reservation CAS is the hard-limit
     * linearization point for the delivered-not-accepted lifecycle limit.
     */
    private final AtomicInteger activeAdmissionCount = new AtomicInteger(0);

    int activeAdmissionCount() {
        return Math.max(0, activeAdmissionCount.get());
    }

    private AdmissionPermit tryReserveAdmissionPermit(int limit) {
        AdmissionPermit permit = new AdmissionPermit(activeAdmissionCount);
        while (true) {
            int activePermits = activeAdmissionCount.get();
            if (activePermits == ADMISSION_CLOSED
                    || (limit > 0 && activePermits >= limit)
                    || activePermits == Integer.MAX_VALUE) {
                return null;
            }
            if (activeAdmissionCount.compareAndSet(activePermits, activePermits + 1)) {
                return permit;
            }
        }
    }

    /**
     * One allocation per admission attempt and no per-release allocation.
     * The field updater makes failure completion and lease termination share
     * one exact-once release without adding a lock or a nested AtomicBoolean.
     */
    private static final class AdmissionPermit
            implements Runnable, BiConsumer<Response, Throwable> {

        private static final AtomicIntegerFieldUpdater<AdmissionPermit> RELEASED =
                AtomicIntegerFieldUpdater.newUpdater(AdmissionPermit.class, "released");

        private final AtomicInteger activeCount;
        @SuppressWarnings("unused") // accessed by RELEASED
        private volatile int released;

        private AdmissionPermit(AtomicInteger activeCount) {
            this.activeCount = activeCount;
        }

        @Override
        public void accept(Response response, Throwable error) {
            if (error != null || response == null || !response.isSuccess()) {
                release();
            }
        }

        @Override
        public void run() {
            release();
        }

        private void release() {
            if (RELEASED.compareAndSet(this, 0, 1)) {
                while (true) {
                    int count = activeCount.get();
                    if (count == ADMISSION_CLOSED) {
                        return;
                    }
                    if (activeCount.compareAndSet(count, count - 1)) {
                        return;
                    }
                }
            }
        }
    }

    @Autowired
    public PriorityAdmissionScheduler(ConfigService configService,
                                      Router router,
                                      EndpointRegistry endpointRegistry,
                                      PlanCommitter planCommitter,
                                      PrioritySchedulerReporter priorityReporter,
                                      BatchSchedulerReporter batchReporter,
                                      EngineCancelChannel cancelChannel,
                                      DecodePreemptionCoordinator preemptionCoordinator) {
        this(configService, router, endpointRegistry, planCommitter,
                priorityReporter, batchReporter, cancelChannel,
                preemptionCoordinator, createSoftTimeoutExecutor());
    }

    PriorityAdmissionScheduler(ConfigService configService,
                               Router router,
                               EndpointRegistry endpointRegistry,
                               PlanCommitter planCommitter,
                               PrioritySchedulerReporter priorityReporter,
                               BatchSchedulerReporter batchReporter,
                               EngineCancelChannel cancelChannel,
                               DecodePreemptionCoordinator preemptionCoordinator,
                               ScheduledThreadPoolExecutor softTimeoutExecutor) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.planCommitter = planCommitter;
        this.priorityReporter = priorityReporter;
        this.batchReporter = batchReporter;
        this.cancelChannel = cancelChannel;
        this.preemptionCoordinator = preemptionCoordinator;
        this.softTimeoutExecutor = Objects.requireNonNull(softTimeoutExecutor);
        this.softTimeoutExecutor.setRemoveOnCancelPolicy(true);
        this.softTimeoutExecutor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
    }

    /** Test/local-construction convenience: use the same channel for orchestration. */
    PriorityAdmissionScheduler(ConfigService configService,
                               Router router,
                               EndpointRegistry endpointRegistry,
                               PlanCommitter planCommitter,
                               PrioritySchedulerReporter priorityReporter,
                               BatchSchedulerReporter batchReporter,
                               EngineCancelChannel cancelChannel) {
        this(configService, router, endpointRegistry, planCommitter,
                priorityReporter, batchReporter, cancelChannel,
                new DecodePreemptionCoordinator(cancelChannel));
    }

    private static ScheduledThreadPoolExecutor createSoftTimeoutExecutor() {
        ScheduledThreadPoolExecutor executor = new ScheduledThreadPoolExecutor(1, task -> {
            Thread thread = new Thread(task, "priority-admission-soft-timeout");
            thread.setDaemon(true);
            return thread;
        });
        executor.setRemoveOnCancelPolicy(true);
        executor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        return executor;
    }

    /**
     * Publishes a lease timer in the same lifecycle critical section as bean
     * shutdown. A callback either completes before shutdown linearizes or
     * observes {@link #shutdown} and never enters registrar/endpoint code.
     */
    private ScheduledFuture<?> scheduleSoftTimeout(AdmissionLease lease,
                                                    Runnable task,
                                                    long delay,
                                                    TimeUnit unit) {
        synchronized (softTimeoutLifecycle) {
            if (shutdown) {
                throw new RejectedExecutionException(
                        "priority admission scheduler is shut down");
            }
            pendingSoftTimeoutLeases.add(lease);
            try {
                return softTimeoutExecutor.schedule(
                        () -> runSoftTimeout(lease, task), delay, unit);
            } catch (RuntimeException | Error schedulingFailure) {
                pendingSoftTimeoutLeases.remove(lease);
                throw schedulingFailure;
            }
        }
    }

    /** Claim under the lifecycle monitor, execute the endpoint/registrar callback outside it. */
    private void runSoftTimeout(AdmissionLease lease, Runnable task) {
        synchronized (softTimeoutLifecycle) {
            if (shutdown || !pendingSoftTimeoutLeases.remove(lease)) {
                return;
            }
            activeSoftTimeoutCallbacks++;
        }
        try {
            task.run();
        } finally {
            synchronized (softTimeoutLifecycle) {
                if (--activeSoftTimeoutCallbacks == 0 && shutdown) {
                    softTimeoutLifecycle.notifyAll();
                }
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        AdmissionLease[] pending;
        synchronized (softTimeoutLifecycle) {
            if (shutdown) {
                return;
            }
            shutdown = true;
            // The negative sentinel closes the same CAS gate used by permit
            // reservation. A concurrent release either wins before this write
            // or observes the sentinel and cannot underflow the public count.
            activeAdmissionCount.set(ADMISSION_CLOSED);

            // A callback which already claimed its lease is absent from this
            // set and is tracked by activeSoftTimeoutCallbacks. Everything else
            // is lifecycle-only termination work performed outside the monitor.
            pending = pendingSoftTimeoutLeases.toArray(AdmissionLease[]::new);
            pendingSoftTimeoutLeases.clear();
        }

        softTimeoutExecutor.shutdownNow();
        for (AdmissionLease lease : pending) {
            lease.terminateForSchedulerShutdown();
        }

        boolean interrupted = false;
        synchronized (softTimeoutLifecycle) {
            while (activeSoftTimeoutCallbacks != 0) {
                try {
                    softTimeoutLifecycle.wait();
                } catch (InterruptedException shutdownInterrupted) {
                    // Do not let bean destruction overtake a callback already
                    // inside registrar/endpoint reconciliation.
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    int softTimeoutQueueSize() {
        return softTimeoutExecutor.getQueue().size();
    }

    boolean removesCanceledSoftTimeouts() {
        return softTimeoutExecutor.getRemoveOnCancelPolicy();
    }

    boolean isShutdown() {
        return shutdown;
    }

    int pendingSoftTimeoutLeaseCount() {
        return pendingSoftTimeoutLeases.size();
    }

    int activeSoftTimeoutCallbackCount() {
        synchronized (softTimeoutLifecycle) {
            return activeSoftTimeoutCallbacks;
        }
    }

    /**
     * Schedule one request. Called by {@code PriorityScheduler.submit()}
     * after its duplicate / max-inflight guards for PRIORITY ordering.
     *
     * <p>On success the item is inflight-registered and queued on the prefill
     * batcher; the future is completed later by the common dispatch pipeline,
     * just as it is for FIFO. On failure the future is completed with an error.
     */
    public void schedule(BalanceContext ctx,
                         CompletableFuture<Response> future,
                         InflightRegistrar registrar) {
        if (shutdown) {
            completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "priority admission scheduler is shut down");
            return;
        }
        FlexlbConfig config = configService.loadBalanceConfig();
        int backpressureLimit = config.queueScheduler().getLifecycle()
                .getMaxDeliveredNotAcceptedRequestsGlobal();
        AdmissionPermit permit = tryReserveAdmissionPermit(backpressureLimit);
        if (permit == null) {
            if (shutdown) {
                completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "priority admission scheduler is shut down");
                return;
            }
            int activePermits = activeAdmissionCount();
            Logger.debug("[priority-scheduler] backpressure limit exceeded, reject request_id={} "
                            + "active_admissions={} limit={}",
                    ctx.getRequestId(), activePermits, backpressureLimit);
            completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "post-success backpressure: active_admissions=" + activePermits
                            + " limit=" + backpressureLimit);
            return;
        }

        // Failed/cancelled futures and lease termination can race. Both use
        // the same exact-once permit, so neither a leak nor an underflow is
        // possible. Successful delivery deliberately keeps the permit until
        // Decode acceptance or post-delivery reconciliation closes the lease.
        future.whenComplete(permit);
        try {
            scheduleWithPermit(ctx, future, registrar, config, permit);
        } catch (RuntimeException | Error error) {
            permit.release();
            throw error;
        }
    }

    private void scheduleWithPermit(BalanceContext ctx,
                                    CompletableFuture<Response> future,
                                    InflightRegistrar registrar,
                                    FlexlbConfig config,
                                    AdmissionPermit permit) {
        if (ctx.requestExpired(System.currentTimeMillis())) {
            future.complete(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED, "Master placement did not complete within budget"));
            return;
        }
        // Capacity offers and victim-presence replans have separate bounded budgets.
        int offerFailures = 0;
        int evictionReplans = 0;
        while (registrar.isAdmissionOpen(ctx.getRequestId(), future)) {
            ctx.setSchedulingDiagnostics(null);
            ctx.setScheduleAttempt(ctx.getScheduleAttempt() + 1);
            if (!registrar.claimAdmissionMutation(ctx.getRequestId(), future)) {
                return;
            }
            ClusterSnapshot snapshot;
            PlacementOutcome outcome;
            try {
                snapshot = ClusterSnapshot.capture(endpointRegistry, config);
                outcome = tryNormalPlacement(ctx, future);
            } catch (RuntimeException | Error routeFailure) {
                registrar.completeAdmissionMutation(ctx.getRequestId(), future);
                throw routeFailure;
            }

            EvictionOutcome eviction;
            if (outcome.plan == null) {
                // Failed routing owns no reservation. Engine-Cancel coordination
                // claims its own mutation, independently of this route attempt.
                registrar.completeAdmissionMutation(ctx.getRequestId(), future);
                if (!registrar.isAdmissionOpen(ctx.getRequestId(), future)) {
                    return;
                }
                if (!isDecodeCapacityFailure(outcome.failureResponse) || snapshot.decodes().isEmpty()) {
                    recordPlacementDiagnostics(ctx, snapshot, outcome.failureResponse == null
                            ? "placement returned no route or failure decision"
                            : outcome.failureResponse.getErrorMessage());
                    onInfeasible(ctx, future, outcome.failureResponse);
                    return;
                }
                if (!preemptionAllows(config, VictimStage.DECODE_RESERVED)
                        && !preemptionAllows(config, VictimStage.DECODE_ENGINE_OWNED)) {
                    PriorityRequestEnvelope envelope = new PriorityRequestEnvelope(
                            ctx.getRequestId(), ctx.getPriority(),
                            ctx.getRequest().getSeqLen(), ctx.getRequest().getMaxNewTokens(),
                            ctx.getStartTime(), ctx.getRequest().getSeqLen(),
                            config.decodeKvReservationTokens(
                                    ctx.getRequest().getSeqLen(), ctx.getRequest().getMaxNewTokens(), 0L));
                    recordPlacementDiagnostics(ctx, snapshot, "decode admission capacity exhausted; preemption disabled");
                    future.complete(AdmissionFailureClassifier.classifyDecode(
                            envelope, new ArrayList<>(snapshot.decodes().values())));
                    return;
                }
                eviction = tryDecodeEviction(ctx, future, snapshot, config, registrar, permit);
            } else {
                try {
                    // Publish the queued phase before a worker can claim delivery.
                    if (outcome.plan.decodeEp() != null) {
                        outcome.plan.decodeEp().markQueuedPhase(ctx.getRequestId());
                    }
                    PlanCommitter.CommitResult commit = planCommitter.commit(outcome.plan, registrar);
                    if (commit.committed()) {
                        onCommitted(ctx, outcome.plan);
                        bindAdmissionLease(outcome.plan, registrar, permit);
                        return;
                    }
                    if (commit.failure() == null || !registrar.isAdmissionOpen(ctx.getRequestId(), future)) {
                        releaseDecodeReservation(outcome.plan);
                        return;
                    }
                    Response failure = commit.failure();
                    if (!StrategyErrorType.fromErrorCode(failure.getCode()).isCapacityRejection()) {
                        releaseDecodeReservation(outcome.plan);
                        future.complete(failure);
                        return;
                    }
                    ctx.setExcludedPrefillIpPort(outcome.plan.prefillEp().ipPort());
                    if (!preemptionAllows(config, VictimStage.PREFILL_QUEUED)) {
                        releaseDecodeReservation(outcome.plan);
                        if (++offerFailures < 2) {
                            continue;
                        }
                        future.complete(failure);
                        return;
                    }
                    eviction = tryPrefillQueueEviction(outcome.plan, config, registrar);
                    if (eviction.committed()) {
                        onCommitted(ctx, outcome.plan);
                        bindAdmissionLease(outcome.plan, registrar, permit);
                        return;
                    }
                    releaseDecodeReservation(outcome.plan);
                } finally {
                    // Rollback or handoff precedes pending cancellation publication.
                    registrar.completeAdmissionMutation(ctx.getRequestId(), future);
                }
            }

            // Both eviction paths have released their provisional resources before retrying.
            if (eviction.retryFailure() == null) {
                return;
            }
            if (++evictionReplans > MAX_EVICTION_REPLANS) {
                future.complete(eviction.retryFailure());
                return;
            }
            backoffBeforeEvictionReplan();
        }
    }

    // ==================== Phase 3: prefill queue eviction ====================

    /** One eviction attempt: committed, handled by an owner, or retry with its actual cause. */
    private record EvictionOutcome(boolean committed, Response retryFailure) {
        static final EvictionOutcome COMMITTED = new EvictionOutcome(true, null);
        static final EvictionOutcome FINISHED = new EvictionOutcome(false, null);

        static EvictionOutcome retry(Response failure) {
            return new EvictionOutcome(false, java.util.Objects.requireNonNull(failure));
        }
    }

    /**
     * Plan and commit a prefill-queue eviction on the router-selected endpoint
     * (design doc 9.1-9.5, 17.2). The incoming decode reservation is already
     * held; on any non-COMMITTED outcome the caller releases it.
     */
    private EvictionOutcome tryPrefillQueueEviction(NormalPlacementPlan plan,
                                                    FlexlbConfig config,
                                                    InflightRegistrar registrar) {
        PriorityRequestEnvelope envelope = plan.envelope();
        BatchItem item = plan.item();
        PrefillQueueManager queueManager = plan.prefillEp().getBatcher().queueManager();

        PrefillQueueSnapshot queueSnapshot = queueManager.snapshot();
        Map<String, String> failures = new HashMap<>();
        PrefillEvictionProposal proposal = EvictionPlanner.planPrefillQueue(
                envelope, List.of(queueSnapshot), failures);
        if (proposal == null) {
            reportEvictionPlan(envelope.priority(), envelope.requestId(),
                    "prefill_queue_full", "infeasible");
            Logger.debug("[priority-scheduler] eviction plan infeasible, request_id={} priority={} "
                            + "phase=prefill_queue candidates_seen={} reasons={}",
                    envelope.requestId(), envelope.priority(),
                    queueSnapshot.items().size(), failures);
            // The victim view excludes charged pending deliveries. A real offer
            // determines admission, including slots freed since the first offer.
            PlanCommitter.CommitResult direct = planCommitter.commit(plan, registrar);
            if (direct.committed()) {
                return EvictionOutcome.COMMITTED;
            }
            Response failure = direct.failure();
            if (failure != null) {
                item.future().complete(failure);
            }
            return EvictionOutcome.FINISHED;
        }
        reportEvictionPlan(envelope.priority(), envelope.requestId(),
                "prefill_queue_full", "feasible");

        PrefillEvictionPlan evictionPlan = new PrefillEvictionPlan(
                envelope, item, plan.routeResponse(), proposal);
        List<Long> victimIds = new ArrayList<>(proposal.victims().size());
        for (QueuedRequestSnapshot victim : proposal.victims()) {
            victimIds.add(victim.requestId());
        }
        PrefillQueueManager.ReplaceOutcome replace;
        // This is the eviction variant of PlanCommitter's generation commit:
        // registration and the queue replacement/direct-offer edge are one
        // request-local unit with respect to Cancel and deadline.
        synchronized (item.future()) {
            PlanCommitter.CommitResult registration = PlanCommitter.register(item, registrar);
            if (!registration.committed()) {
                Response failure = registration.failure();
                if (failure != null) {
                    item.future().complete(failure);
                }
                return EvictionOutcome.FINISHED;
            }
            // Only the selected victims are guarded; unrelated queue mutations
            // do not abort the commit.
            replace = queueManager.tryReplaceVictimsPresent(victimIds, item);

            if (replace.isVictimGone()) {
                // Zero-side-effect abort: a victim left the queue — usually
                // the queue freed a slot, so try one direct offer before
                // replanning. The item is already inflight-registered.
                reportEvictionCommit(envelope.priority(), envelope.requestId(),
                        "prefill_queue_full", "victim_gone");
                Response offerFailure = plan.prefillEp().getBatcher().tryOffer(item);
                if (offerFailure == null) {
                    Logger.debug("[priority-scheduler] eviction victims gone, direct offer succeeded: "
                                    + "request_id={} missing_victims={} worker={}",
                            envelope.requestId(), replace.missingVictimIds(),
                            proposal.endpointId());
                    return EvictionOutcome.COMMITTED;
                }
                registrar.unregisterInflight(item);
                if (!StrategyErrorType.fromErrorCode(offerFailure.getCode()).isCapacityRejection()) {
                    item.future().complete(offerFailure);
                    return EvictionOutcome.FINISHED;
                }
                Logger.debug("[priority-scheduler] eviction victims gone, replan: request_id={} "
                                + "missing_victims={} worker={}",
                        envelope.requestId(), replace.missingVictimIds(), proposal.endpointId());
                return EvictionOutcome.retry(offerFailure);
            }
        }

        // Victims removed from the queue are never re-inserted (design doc
        // 9.5): drive each to its terminal state, releasing its decode
        // reservation and publishing PRIORITY_PREEMPTED. Settlement is
        // idempotent via the inflight lifecycle.
        for (BatchItem victim : replace.removed()) {
            settlePrefillVictim(envelope, registrar, victim, proposal.endpointId());
        }

        if (replace.isPartialFailure()) {
            // Defensive path — unreachable under the atomic replace: victims
            // (if any) went terminal above, the incoming fails explicitly.
            registrar.unregisterInflight(item);
            reportEvictionCommit(envelope.priority(), envelope.requestId(),
                    "prefill_queue_full", "partial_failure");
            Logger.error("[priority-scheduler] eviction commit partial failure, request_id={} victims_removed={}",
                    envelope.requestId(), replace.removed().size());
            item.future().complete(replace.failure());
            return EvictionOutcome.FINISHED;
        }

        reportEvictionCommit(envelope.priority(), envelope.requestId(),
                "prefill_queue_full", "success");
        // §19.1 plan observability: on the combined decode+prefill eviction
        // path keep the primary decode_evict label; accumulate cost/victims.
        BalanceContext itemCtx = item.ctx();
        if (!"decode_evict".equals(itemCtx.getPlanType())) {
            itemCtx.setPlanType("prefill_evict");
        }
        itemCtx.setPlanCost(itemCtx.getPlanCost() + proposal.rawCost());
        itemCtx.setVictimCount(itemCtx.getVictimCount() + proposal.victims().size());
        Logger.debug("[priority-scheduler] eviction committed: request_id={} priority={} victims={} "
                        + "raw_cost={} worker={}",
                envelope.requestId(), envelope.priority(), evictionPlan.proposal().victims().size(),
                proposal.rawCost(), proposal.endpointId());
        return EvictionOutcome.COMMITTED;
    }

    /**
     * Settle one victim removed by a committed queue replacement. Core
     * settlement and observability are independent: a faulty observer cannot
     * interrupt the drain of the remaining victims or reverse the incoming
     * request's already-committed queue transaction.
     */
    private void settlePrefillVictim(PriorityRequestEnvelope incoming,
                                     InflightRegistrar registrar,
                                     BatchItem victim,
                                     String endpointId) {
        String detail = "preempted by higher-priority request " + incoming.requestId();
        try {
            registrar.finishPreempted(victim, detail);
        } catch (RuntimeException firstFailure) {
            // Continue draining even when one reducer invocation fails. The
            // reducer is idempotent, so one immediate retry closes transient
            // failures without allocating a second recovery registry.
            Logger.warn("[priority-scheduler] retrying removed prefill victim settlement: "
                            + "victim_id={} incoming_id={} worker={}",
                    victim.requestId(), incoming.requestId(), endpointId,
                    firstFailure);
            try {
                registrar.finishPreempted(victim, detail);
            } catch (RuntimeException retryFailure) {
                Logger.error("[priority-scheduler] failed to settle removed prefill victim after retry: "
                                + "victim_id={} incoming_id={} worker={}",
                        victim.requestId(), incoming.requestId(), endpointId,
                        retryFailure);
            }
        }
        try {
            priorityReporter.reportVictim(victim.priority(), incoming.priority(),
                    "prefill_queued", "prefill_queue_full");
            priorityReporter.reportPriorityPreempt("prefill_queued");
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report prefill victim settlement: "
                            + "victim_id={} incoming_id={} worker={}",
                    victim.requestId(), incoming.requestId(), endpointId,
                    telemetryFailure);
        }
        Logger.debug("[priority-scheduler] victim preempted: victim_id={} victim_priority={} "
                        + "terminal=preempted_8429 incoming_id={} incoming_priority={} worker={}",
                victim.requestId(), victim.priority(), incoming.requestId(),
                incoming.priority(), endpointId);
    }

    /** Eviction metrics are observers; they never own a committed transaction. */
    private void reportEvictionCommit(int priority,
                                      long requestId,
                                      String evictionCase,
                                      String outcome) {
        try {
            priorityReporter.reportEvictionCommit(
                    priority, evictionCase, outcome);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report eviction commit: "
                            + "request_id={} case={} outcome={}",
                    requestId, evictionCase, outcome,
                    telemetryFailure);
        }
    }

    private void reportEvictionPlan(int priority,
                                    long requestId,
                                    String evictionCase,
                                    String outcome) {
        try {
            priorityReporter.reportEvictionPlan(priority, evictionCase, outcome);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report eviction plan: "
                            + "request_id={} case={} outcome={}",
                    requestId, evictionCase, outcome, telemetryFailure);
        }
    }

    /** Per-endpoint prefill queue depth gauge (design doc 19.2). */
    public void reportPrefillQueueDepths() {
        endpointRegistry.getPrefillEndpoints().forEach((key, ep) ->
                priorityReporter.reportPrefillQueueDepth(key, ep.getBatcher().queueSize()));
    }

    /** Per-endpoint decode shadow reservation gauges (design doc 19.2). */
    public void reportDecodeAdmissionGauges() {
        endpointRegistry.getDecodeEndpoints().forEach((key, ep) -> {
            priorityReporter.reportDecodeReservedCount(key, ep.getInflightCount());
            priorityReporter.reportDecodeShadowKvReserved(key, ep.inflightHardKvReserved());
            // §19.2 Phase 5 layered split: true running layer + accepted layer
            // (their sum equals the former merged confirmedRunningCount).
            priorityReporter.reportDecodeRunningCount(key, ep.getRunningLayerCount());
            priorityReporter.reportDecodeAcceptedCount(key, ep.getAcceptedLayerCount());
            // N2/§3.8: engine-facing load vs the shadow reserved count above
            // directly monitors the root-cause-C gap (queued reservations).
            priorityReporter.reportDecodeEngineLoad(key, ep.getEngineLoad());
        });
    }

    // ==================== Decode eviction ====================

    /**
     * Plan and atomically commit a Decode eviction (design doc 11-13, 17.2),
     * then place the incoming request on the freed endpoint. Master-local and
     * Engine-owned victim domains are enabled independently by configuration.
     * Uses the pre-route {@link ClusterSnapshot}; commit checks the selected
     * victims' reservation presence, not unrelated endpoint mutations.
     */
    private EvictionOutcome tryDecodeEviction(BalanceContext ctx,
                                               CompletableFuture<Response> future,
                                               ClusterSnapshot snapshot,
                                               FlexlbConfig config,
                                               InflightRegistrar registrar,
                                               AdmissionPermit permit) {
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();
        // Planning envelope contains only placement demand and priority.
        PriorityRequestEnvelope planEnvelope = new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), seqLen,
                config.decodeKvReservationTokens(seqLen, maxNewTokens, 0L));

        List<DecodeEndpointSnapshot> decodes = new ArrayList<>(snapshot.decodes().values());
        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal =
                EvictionPlanner.planDecode(planEnvelope, decodes, config, cancelChannel, failures);
        if (proposal == null) {
            reportEvictionPlan(ctx.getPriority(), ctx.getRequestId(),
                    infeasibleDecodeCase(planEnvelope, decodes), "infeasible");
            // Redesign D-1: carry the plan phase and candidate counters so an
            // infeasible burst is attributable from logs alone.
            int candidatesSeen = 0;
            int candidatesEligible = 0;
            boolean localEnabled = preemptionAllows(config, VictimStage.DECODE_RESERVED);
            boolean engineEnabled = preemptionAllows(config, VictimStage.DECODE_ENGINE_OWNED);
            for (DecodeEndpointSnapshot ep : decodes) {
                for (DecodeRequestSnapshot candidate : ep.reserved()) {
                    boolean enabledDomain = (localEnabled && candidate.phase().isMasterQueued())
                            || (engineEnabled && !candidate.phase().isMasterQueued());
                    if (enabledDomain) {
                        candidatesSeen++;
                        if (candidate.priorityKnown()
                                && PriorityNormalizer.hasPriority(candidate.priority())
                                && candidate.priority() < planEnvelope.priority()) {
                            candidatesEligible++;
                        }
                    }
                }
                if (engineEnabled) {
                    for (DecodeRequestSnapshot candidate : ep.accepted()) {
                        candidatesSeen++;
                        if (candidate.priorityKnown()
                                && PriorityNormalizer.hasPriority(candidate.priority())
                                && candidate.priority() < planEnvelope.priority()) {
                            candidatesEligible++;
                        }
                    }
                    for (DecodeRequestSnapshot candidate : ep.running()) {
                        candidatesSeen++;
                        if (candidate.priorityKnown()
                                && PriorityNormalizer.hasPriority(candidate.priority())
                                && candidate.priority() < planEnvelope.priority()) {
                            candidatesEligible++;
                        }
                    }
                }
            }
            String phase = localEnabled && engineEnabled
                    ? "decode_reserved_or_engine"
                    : engineEnabled ? "decode_engine" : "decode_reserved";
            Logger.debug("[priority-scheduler] decode eviction plan infeasible, request_id={} priority={} "
                            + "phase={} candidates_seen={} candidates_eligible={} reasons={}",
                    ctx.getRequestId(), ctx.getPriority(), phase,
                    candidatesSeen, candidatesEligible, failures);
            recordPlacementDiagnostics(ctx, snapshot, "decode eviction infeasible; reasons=" + failures);
            Response failure = AdmissionFailureClassifier.classifyDecode(
                    planEnvelope, decodes);
            future.complete(failure);
            return EvictionOutcome.FINISHED;
        }
        reportEvictionPlan(ctx.getPriority(), ctx.getRequestId(),
                proposal.evictionCase(), "feasible");

        DecodeEndpointSnapshot target = snapshot.decodes().get(proposal.endpointId());
        DecodeEndpoint decodeEp = target.endpoint();
        long expectedKvTokens = config.decodeKvReservationTokens(
                seqLen, maxNewTokens, target.realKvTotal());

        // Ownership is homogeneous by planner invariant: Master-queued victims
        // use a local transaction; Engine-may-have-seen/accepted/running
        // victims use the tokenized Cancel coordinator.
        if (proposal.requiresEngineCancel()) {
            startEngineCancelPreemption(ctx, future, config, registrar, proposal,
                    target, decodeEp, seqLen, expectedKvTokens, permit);
            return EvictionOutcome.FINISHED;
        }

        List<Long> reservedVictimIds = new ArrayList<>(proposal.victims().size());
        for (DecodeRequestSnapshot victim : proposal.victims()) {
            reservedVictimIds.add(victim.requestId());
        }

        // The victim mutation and incoming placement form one generation
        // commit. Cancel/deadline either close before any victim is touched,
        // or observe the incoming request after the complete handoff.
        synchronized (future) {
          if (!registrar.isAdmissionOpen(ctx.getRequestId(), future)) {
            return EvictionOutcome.FINISHED;
          }
          // Presence-guarded commit conditionally releases each victim still
          // holding its reservation; unrelated endpoint churn cannot abort it.
          DecodeEndpoint.PresenceEvictionOutcome presence =
                  decodeEp.tryReleaseVictimsIfHeldAndReserveIncoming(
                          reservedVictimIds, ctx.getRequestId(), seqLen, expectedKvTokens,
                          ctx.getPriority());
          if (!presence.success()) {
              // Victims already freed are not rolled back; their host
              // requests are driven terminal before the incoming request replans.
              for (DecodeRequestSnapshot victim : proposal.victims()) {
                  if (presence.freedVictimIds().contains(victim.requestId())) {
                      finishDecodeVictim(ctx, registrar, victim, "decode_reserved", proposal);
                  }
              }
              reportEvictionCommit(ctx.getPriority(), ctx.getRequestId(),
                      proposal.evictionCase(), "victim_gone");
              Logger.debug("[priority-scheduler] decode eviction victims gone, replan: request_id={} "
                              + "freed={} planned={} worker={}",
                      ctx.getRequestId(), presence.freedVictimIds().size(),
                      reservedVictimIds.size(), proposal.endpointId());
              // Retry the changed reservation set. If the budget is exhausted,
              // admission capacity was not acquired; the conflict stays diagnostic.
              ctx.setSchedulingDiagnostics(Map.of("cause", "decode victim reservations changed",
                      "endpoint", proposal.endpointId(), "plannedVictims", reservedVictimIds.size(),
                      "freedVictims", presence.freedVictimIds().size()));
              return EvictionOutcome.retry(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                      AdmissionRejectReason.RESOURCE_EXHAUSTED));
          }

          return completeDecodeReservationHandoff(
                  ctx, future, registrar, decodeEp, () -> {
                      // Shadow accounting already reversed atomically; drive
                      // each victim terminal before publishing the incoming
                      // item. All evicted requests receive the same preemption outcome.
                      for (DecodeRequestSnapshot victim : proposal.victims()) {
                          finishDecodeVictim(ctx, registrar, victim,
                                  "decode_reserved", proposal);
                      }
                      reportCommittedLocalDecodeEviction(ctx, proposal);
                      recordDecodePlanObservability(ctx, proposal);
                      return placeAfterDecodeEviction(
                              ctx, future, config, registrar, decodeEp, permit);
                  });
        }
    }

    /**
     * Drive one decode eviction victim to its terminal state and emit the
     * per-victim metrics ({@code stage} distinguishes reserved vs accepted
     * victims). Delivery stage changes cleanup, not the public preemption outcome.
     */
    private void finishDecodeVictim(BalanceContext ctx, InflightRegistrar registrar,
                                    DecodeRequestSnapshot victim, String stage,
                                    DecodeEvictionProposal proposal) {
        String detail = "preempted by higher-priority request " + ctx.getRequestId();
        try {
            registrar.finishPreemptedById(victim.requestId(), detail);
        } catch (RuntimeException firstFailure) {
            // The Decode shadow swap already removed every victim. Keep one
            // faulty reducer invocation from stranding the rest, and use the
            // idempotent reducer once more for a transient failure.
            Logger.warn("[priority-scheduler] retrying removed decode victim settlement: "
                            + "victim_id={} incoming_id={} worker={}",
                    victim.requestId(), ctx.getRequestId(), proposal.endpointId(),
                    firstFailure);
            try {
                registrar.finishPreemptedById(victim.requestId(), detail);
            } catch (RuntimeException retryFailure) {
                Logger.error("[priority-scheduler] failed to settle removed decode victim after retry: "
                                + "victim_id={} incoming_id={} worker={}",
                        victim.requestId(), ctx.getRequestId(), proposal.endpointId(),
                        retryFailure);
            }
        }
        try {
            priorityReporter.reportVictim(victim.priority(), ctx.getPriority(),
                    stage, proposal.evictionCase());
            priorityReporter.reportPriorityPreempt(stage);
            priorityReporter.reportVictimKvTokens(
                    victim.priority(), stage, victim.kvTokens());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report decode victim settlement: "
                            + "victim_id={} incoming_id={}",
                    victim.requestId(), ctx.getRequestId(), telemetryFailure);
        }
        Logger.debug("[priority-scheduler] decode victim preempted: victim_id={} victim_priority={} "
                        + "stage={} terminal=preempted_8429 kv_tokens={} incoming_id={} incoming_priority={} worker={}",
                victim.requestId(), victim.priority(), stage, victim.kvTokens(),
                ctx.getRequestId(), ctx.getPriority(), proposal.endpointId());
    }

    /** §19.1 plan observability for the decode eviction path. */
    private static void recordDecodePlanObservability(BalanceContext ctx,
                                                      DecodeEvictionProposal proposal) {
        ctx.setPlanType("decode_evict");
        ctx.setPlanCost(proposal.totalCost());
        ctx.setVictimCount(proposal.victims().size());
        Logger.debug("[priority-scheduler] decode eviction committed: request_id={} priority={} case={} "
                        + "victims={} total_cost={} freed_kv={} worker={}",
                ctx.getRequestId(), ctx.getPriority(), proposal.evictionCase(),
                proposal.victims().size(), proposal.totalCost(), proposal.freedKvTokens(),
                proposal.endpointId());
    }

    private void startEngineCancelPreemption(BalanceContext ctx,
                                             CompletableFuture<Response> future,
                                             FlexlbConfig config,
                                             InflightRegistrar registrar,
                                             DecodeEvictionProposal proposal,
                                             DecodeEndpointSnapshot target,
                                             DecodeEndpoint decodeEp,
                                             long seqLen,
                                             long expectedKvTokens,
                                             AdmissionPermit permit) {
        String detail = "preempted by higher-priority request " + ctx.getRequestId();
        EngineCancellationConfig cancellation = requiredEngineCancellation(config);
        DecodePreemptionCoordinator.Request request =
                new DecodePreemptionCoordinator.Request(
                        decodeEp, proposal.admissionVersion(),
                        false,
                        ctx.getRequestId(), seqLen, expectedKvTokens,
                        ctx.getPriority(),
                        proposal.victims(), cancellation.getAckTimeoutMs(),
                        cancellation.getCompletionTimeoutMs(),
                        () -> registrar.isAdmissionOpen(ctx.getRequestId(), future), detail);

        CompletableFuture<DecodePreemptionCoordinator.ExecutionResult> execution;
        synchronized (future) {
            if (!registrar.claimAdmissionMutation(ctx.getRequestId(), future)) {
                return;
            }
            try {
                reportCancelRequests(ctx, proposal);
                // execute() performs the victim-claim and sends every Cancel
                // before returning. The mutation claim keeps an incoming
                // Cancel pending until this asynchronous attempt settles.
                execution = preemptionCoordinator.execute(request, registrar);
            } catch (RuntimeException | Error startFailure) {
                registrar.completeAdmissionMutation(ctx.getRequestId(), future);
                throw startFailure;
            }
        }

        execution.whenComplete((result, error) -> {
            boolean mutationCompleted = false;
            try {
                if (error != null || result == null) {
                    cancelNotFoundReplans.remove(ctx.getRequestId());
                    // execute() owns the endpoint attempt, but an unexpected
                    // exceptional completion has no typed result to hand off.
                    // Release the provisional incoming reservation before
                    // publishing failure; release is exact-idempotent.
                    decodeEp.release(ctx.getRequestId());
                    reportCancelTimeout(ctx, proposal.endpointId());
                    completeAdmissionError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                            AdmissionRejectReason.UNSPECIFIED,
                            "priority cancel coordinator failed");
                    return;
                }
                switch (result.code()) {
                    case COMMITTED -> {
                        cancelNotFoundReplans.remove(ctx.getRequestId());
                        if (!registrar.isAdmissionOpen(ctx.getRequestId(), future)) {
                            decodeEp.release(ctx.getRequestId());
                            Logger.debug("[priority-scheduler] drop committed preemption after admission close: "
                                            + "request_id={} worker={}",
                                    ctx.getRequestId(), proposal.endpointId());
                            return;
                        }
                        reportCommittedEnginePreemption(ctx, proposal);
                        EvictionOutcome handoff = completeDecodeReservationHandoff(
                                ctx, future, registrar, decodeEp, () -> {
                                    recordDecodePlanObservability(ctx, proposal);
                                    // PriorityScheduler.registerInflight owns the
                                    // request-local deadline fence. If the deadline
                                    // closes after the check above, registration
                                    // rejects atomically and normal failure cleanup
                                    // releases the provisional Decode reservation.
                                    return placeAfterDecodeEviction(
                                            ctx, future, config, registrar,
                                            decodeEp, permit);
                                });
                        if (handoff.retryFailure() != null) {
                            future.complete(handoff.retryFailure());
                        }
                    }
                    case REPLAN_NOT_FOUND, CONFLICT -> {
                        int replans = cancelNotFoundReplans
                                .computeIfAbsent(ctx.getRequestId(), ignored -> new AtomicInteger())
                                .incrementAndGet();
                        if (replans <= 1 && registrar.isAdmissionOpen(
                                ctx.getRequestId(), future)) {
                            // This attempt has no remaining request-id keyed
                            // cleanup. Release its mutation before a replan can
                            // claim the next asynchronous attempt.
                            registrar.completeAdmissionMutation(
                                    ctx.getRequestId(), future);
                            mutationCompleted = true;
                            scheduleWithPermit(ctx, future, registrar,
                                    configService.loadBalanceConfig(), permit);
                        } else {
                            cancelNotFoundReplans.remove(ctx.getRequestId());
                            completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                                    result.detail());
                        }
                    }
                    case CONTROL_FAILED -> {
                        cancelNotFoundReplans.remove(ctx.getRequestId());
                        reportCancelTimeout(ctx, proposal.endpointId());
                        completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                                result.detail());
                    }
                }
            } catch (RuntimeException | Error callbackError) {
                permit.release();
                cancelNotFoundReplans.remove(ctx.getRequestId());
                Logger.error("[priority-scheduler] cancel completion failed: request_id={} error={}",
                        ctx.getRequestId(), callbackError.getMessage(), callbackError);
                completeAdmissionError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                        AdmissionRejectReason.UNSPECIFIED,
                        "priority cancel completion failed");
            } finally {
                if (!mutationCompleted) {
                    // Retire the generation only after COMMITTED cleanup or
                    // commit handoff is complete; otherwise a reused id could
                    // be released by this old callback.
                    registrar.completeAdmissionMutation(
                            ctx.getRequestId(), future);
                }
            }
        });
    }

    /** Metrics never participate in the committed reservation handoff. */
    private void reportCommittedEnginePreemption(
            BalanceContext ctx, DecodeEvictionProposal proposal) {
        try {
            for (DecodeRequestSnapshot victim : proposal.victims()) {
                String stage = victim.phase() == DecodeTaskPhase.RUNNING
                        ? "decode_running" : "decode_cancel";
                priorityReporter.reportVictim(victim.priority(), ctx.getPriority(),
                        stage, proposal.evictionCase());
                priorityReporter.reportPriorityPreempt(stage);
                priorityReporter.reportVictimKvTokens(
                        victim.priority(), stage, victim.kvTokens());
                priorityReporter.reportCancelConfirm(
                        proposal.endpointId(), victim.priority());
            }
            priorityReporter.reportEvictionCommit(ctx.getPriority(),
                    proposal.evictionCase(), "success");
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report committed decode preemption: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), proposal.endpointId(), telemetryFailure);
        }
    }

    private void reportCancelRequests(BalanceContext ctx,
                                      DecodeEvictionProposal proposal) {
        try {
            for (DecodeRequestSnapshot victim : proposal.victims()) {
                priorityReporter.reportCancelRequest(
                        proposal.endpointId(), victim.priority());
                priorityReporter.reportCancel(
                        victim.priority(), "PRIORITY_PREEMPTED");
            }
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report priority cancel requests: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), proposal.endpointId(), telemetryFailure);
        }
    }

    private void reportCancelTimeout(BalanceContext ctx, String endpointId) {
        try {
            priorityReporter.reportCancelTimeout(endpointId, ctx.getPriority());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report priority cancel timeout: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), endpointId, telemetryFailure);
        }
    }

    /** Local decode-eviction metrics are outside the reservation transaction. */
    private void reportCommittedLocalDecodeEviction(
            BalanceContext ctx, DecodeEvictionProposal proposal) {
        try {
            priorityReporter.reportEvictionCommit(
                    ctx.getPriority(), proposal.evictionCase(), "success");
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report committed local decode eviction: "
                            + "request_id={} worker={}",
                    ctx.getRequestId(), proposal.endpointId(), telemetryFailure);
        }
    }

    /**
     * Transfer one provisional Decode reservation to the exact scheduler
     * generation. Before register/offer succeeds this method is the sole
     * owner; afterwards the scheduler terminal reducer owns every cleanup.
     */
    private EvictionOutcome completeDecodeReservationHandoff(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            InflightRegistrar registrar,
            DecodeEndpoint decodeEp,
            Supplier<EvictionOutcome> handoff) {
        RequestInflight expectedReservation =
                decodeEp.reservationFor(ctx.getRequestId());
        try {
            return handoff.get();
        } catch (RuntimeException | Error handoffFailure) {
            if (!registrar.isInflightGeneration(ctx.getRequestId(), future)
                    && expectedReservation != null) {
                decodeEp.releaseReservationIfCurrent(
                        ctx.getRequestId(), expectedReservation);
            }
            throw handoffFailure;
        }
    }

    /**
     * Place the incoming request after a successful decode eviction. The
     * normal route cannot be replayed — the incoming reservation just taken
     * would make the decode strategy filter out its own endpoint — so the
     * decode {@link ServerStatus} is built manually and only prefill goes
     * through its selection strategy.
     */
    private EvictionOutcome placeAfterDecodeEviction(BalanceContext ctx,
                                                      CompletableFuture<Response> future,
                                                      FlexlbConfig config,
                                                      InflightRegistrar registrar,
                                                      DecodeEndpoint decodeEp,
                                                      AdmissionPermit permit) {
        ServerStatus prefill = selectPrefillForDecodeEviction(
                ctx, config, decodeEp.getStatus().getGroup());
        if (prefill == null || !prefill.isSuccess()) {
            decodeEp.release(ctx.getRequestId());
            completeAdmissionError(future, StrategyErrorType.NO_PREFILL_WORKER,
                    AdmissionRejectReason.UNSPECIFIED,
                    "no prefill worker after decode eviction");
            return EvictionOutcome.FINISHED;
        }
        PrefillEndpoint prefillEp = endpointRegistry.getPrefill(
                prefill.getServerIp() + ":" + prefill.getHttpPort());
        if (prefillEp == null) {
            decodeEp.release(ctx.getRequestId());
            completeAdmissionError(future, StrategyErrorType.NO_PREFILL_WORKER,
                    AdmissionRejectReason.UNSPECIFIED,
                    "prefill endpoint not registered after decode eviction");
            return EvictionOutcome.FINISHED;
        }

        ServerStatus decode = buildDecodeServerStatus(ctx, decodeEp);
        Response routeResponse = new Response();
        routeResponse.setSuccess(true);
        routeResponse.setServerStatus(List.of(prefill, decode));

        PriorityRequestEnvelope envelope = buildEnvelope(ctx, decodeEp);

        BatchItem item = new BatchItem(ctx, future, routeResponse,
                PriorityScheduler.copyOf(prefill), PriorityScheduler.copyOf(decode),
                prefillEp, decodeEp, System.currentTimeMillis());

        NormalPlacementPlan plan = new NormalPlacementPlan(envelope, item, routeResponse);

        // P1-1: queued-phase mark precedes the commit (same rationale as the
        // normal path); every failure path below releases the reservation,
        // which clears the mark.
        decodeEp.markQueuedPhase(ctx.getRequestId());
        PlanCommitter.CommitResult commit = planCommitter.commit(plan, registrar);
        if (commit.committed()) {
            onCommitted(ctx, plan);
            bindAdmissionLease(plan, registrar, permit);
            return EvictionOutcome.COMMITTED;
        }
        if (commit.failure() == null || !registrar.isAdmissionOpen(ctx.getRequestId(), future)) {
            releaseDecodeReservation(plan);
            return EvictionOutcome.FINISHED;
        }
        Response result = commit.failure();
        if (!StrategyErrorType.fromErrorCode(result.getCode()).isCapacityRejection()) {
            releaseDecodeReservation(plan);
            future.complete(result);
            return EvictionOutcome.FINISHED;
        }
        // A capacity rejection may still be resolved by prefill-queue eviction.
        if (preemptionAllows(config, VictimStage.PREFILL_QUEUED)) {
            EvictionOutcome eviction = tryPrefillQueueEviction(plan, config, registrar);
            if (eviction.committed()) {
                onCommitted(ctx, plan);
                bindAdmissionLease(plan, registrar, permit);
            } else {
                releaseDecodeReservation(plan);
            }
            return eviction;
        }
        releaseDecodeReservation(plan);
        future.complete(result);
        return EvictionOutcome.FINISHED;
    }

    /**
     * Select a prefill endpoint for a decode-eviction placement, following the
     * freed decode endpoint's group for affinity. Protected as a test seam —
     * production resolves the configured strategy from the static factory.
     */
    protected ServerStatus selectPrefillForDecodeEviction(BalanceContext ctx,
                                                          FlexlbConfig config,
                                                          String group) {
        LoadBalanceStrategy strategy = LoadBalanceStrategyFactory.getLoadBalanceStrategy(
                config.strategyFor(RoleType.PREFILL));
        if (strategy == null) {
            return null;
        }
        return strategy.select(ctx, RoleType.PREFILL, group);
    }

    /** Mirror of {@code CostBasedDecodeStrategy.buildServerStatus} field-for-field. */
    private static ServerStatus buildDecodeServerStatus(BalanceContext ctx, DecodeEndpoint decodeEp) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(RoleType.DECODE);
        status.setServerIp(decodeEp.getIp());
        status.setHttpPort(decodeEp.getHttpPort());
        status.setGrpcPort(CommonUtils.toGrpcPort(decodeEp.getHttpPort()));
        status.setDpRank(decodeEp.getStatus().getDpRank());
        status.setGroup(decodeEp.getStatus().getGroup());
        status.setRequestId(ctx.getRequestId());
        return status;
    }

    /** Route failed specifically because no decode worker had capacity. */
    private static boolean isDecodeCapacityFailure(Response response) {
        return response != null && !response.isSuccess()
                && response.getCode() == StrategyErrorType.NO_DECODE_WORKER.getErrorCode();
    }

    /**
     * Case label for an infeasible decode plan: the first endpoint with a
     * deficit determines the tag (deterministic for single-endpoint setups);
     * defaults to slot-full when no snapshot shows a deficit (raced away).
     */
    private static String infeasibleDecodeCase(PriorityRequestEnvelope envelope,
                                               List<DecodeEndpointSnapshot> decodes) {
        for (DecodeEndpointSnapshot ep : decodes) {
            String evictionCase = EvictionPlanner.decodeEvictionCase(envelope, ep);
            if (evictionCase != null) {
                return evictionCase;
            }
        }
        return DecodeEvictionProposal.CASE_SLOT;
    }

    // ==================== Plan building ====================

    private PlacementOutcome tryNormalPlacement(BalanceContext ctx,
                                                CompletableFuture<Response> future) {
        Response routeResponse = router.route(ctx);
        // P1-4: the exclusion steers exactly one re-route — clear it so later
        // attempts (or a rescue re-entry) see the full candidate set again.
        ctx.setExcludedPrefillIpPort(null);
        if (routeResponse == null || !routeResponse.isSuccess()) {
            // A failed route holds no reservation, matching FIFO routing.
            return PlacementOutcome.infeasible(routeResponse);
        }

        try {

            ServerStatus prefill = PriorityScheduler.findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = PriorityScheduler.findServer(routeResponse, RoleType.DECODE);
            if (prefill == null) {
                rollbackRoute(routeResponse);
                return PlacementOutcome.infeasible(null);
            }

            String prefillIpPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
            PrefillEndpoint prefillEp = endpointRegistry.getPrefill(prefillIpPort);
            if (prefillEp == null) {
                rollbackRoute(routeResponse);
                return PlacementOutcome.infeasible(null);
            }

            DecodeEndpoint decodeEp = null;
            if (decode != null) {
                decodeEp = endpointRegistry.getDecode(decode.getServerIp() + ":" + decode.getHttpPort());
            }

            PriorityRequestEnvelope envelope = buildEnvelope(ctx, decodeEp);

            BatchItem item = new BatchItem(ctx, future, routeResponse,
                    PriorityScheduler.copyOf(prefill), PriorityScheduler.copyOf(decode),
                    prefillEp, decodeEp, System.currentTimeMillis());

            return PlacementOutcome.of(new NormalPlacementPlan(
                    envelope, item, routeResponse));
        } catch (RuntimeException | Error planFailure) {
            // route() has already reserved Decode capacity. Every failure in
            // plan construction must unwind that reservation before the
            // generation mutation is released and a pending Cancel publishes.
            rollbackRoute(routeResponse);
            throw planFailure;
        }
    }

    private PriorityRequestEnvelope buildEnvelope(BalanceContext ctx,
                                                  DecodeEndpoint decodeEp) {
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();
        long kvTotal = decodeEp != null ? decodeEp.realKvTotal() : 0;
        long expectedKvTokens = configService.loadBalanceConfig()
                .decodeKvReservationTokens(seqLen, maxNewTokens, kvTotal);
        return new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), seqLen, expectedKvTokens);
    }

    // ==================== Outcome handling ====================

    /**
     * Create an {@link AdmissionLease} and bind it to the request future
     * (PR-D §2.4, fix for triple-lock OOM). Called at every plan-commit
     * success point after {@link #onCommitted}. The lease is the single
     * ownership boundary: success → {@code markDeliverySucceeded} (seal +
     * schedule soft timeout); failure/timeout → {@code close} (tryRemove +
     * release + unregister). The soft timeout fires when decode hasn't
     * accepted within the configured delivered-not-accepted timeout →
     * {@code reconcileAfterDeliveryTimeout} (scheduler-owned Engine fence).
     * <p>The admission permit was already reserved before planning and is
     * released when the lease closes (via the onClose callback).
     */
    private void bindAdmissionLease(NormalPlacementPlan plan,
                                    InflightRegistrar registrar,
                                    AdmissionPermit permit) {
        BalanceContext ctx = plan.item().ctx();
        FlexlbConfig config = configService.loadBalanceConfig();
        long softTimeoutMs = config.queueScheduler().getLifecycle()
                .getDeliveredNotAcceptedTimeoutMs();
        PrefillQueueManager prefillQueue = plan.prefillEp().getBatcher().queueManager();
        AdmissionLease lease = new AdmissionLease(
                plan.item(), plan.decodeEp(), prefillQueue, registrar,
                softTimeoutMs, permit, softTimeoutScheduler);
        try {
            // WorkerStatus acceptance belongs to the registrar's exact
            // inflight generation.  Attach before binding the future so a
            // racing ACK/status observation cannot strand the counter.
            if (!registrar.attachAdmissionLease(plan.item(), lease)) {
                // The exact scheduler terminal observer or another terminal
                // reducer already owns this generation. Retire only the lease
                // permit/timer; endpoint cleanup must not run a second time.
                lease.markRequestSettled();
                return;
            }
            lease.bindTo(plan.item().future());
        } catch (RuntimeException | Error error) {
            // Attachment and future binding are both outside scheduler locks.
            // close() owns queued/inflight rollback and invokes the same
            // exact-once permit used by public-future failure completion.
            lease.close();
            throw error;
        }
    }

    private void onCommitted(BalanceContext ctx, NormalPlacementPlan plan) {
        ctx.setRouteSubmittedNanos(System.nanoTime());
        // N3 §3.8: plan age quantifies how stale the committed plan view was
        // (snapshot/build → successful commit).
        try {
            priorityReporter.reportPlanAge(plan.envelope().priority(),
                    Math.max(0, System.currentTimeMillis() - plan.createdAtMs()));
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report plan age: request_id={}",
                    ctx.getRequestId(), telemetryFailure);
        }
        // N2/P1-1: the queued-phase mark is set BEFORE the commit (schedule /
        // placeAfterDecodeEviction) — marking here raced the dispatch side's
        // tryMarkEngineMayHaveSeen for items that dispatched immediately.
        // §19.1 plan_type: eviction paths set their label before this point.
        if (ctx.getPlanType() == null || ctx.getPlanType().isEmpty()) {
            ctx.setPlanType("normal");
        }
        ServerStatus prefill = plan.item().prefill();
        ctx.setScheduledPrefillEndpoint(prefill.getServerIp() + ":" + prefill.getHttpPort());
        try {
            priorityReporter.reportNormalPlacement(plan.envelope().priority());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report normal placement: request_id={}",
                    ctx.getRequestId(), telemetryFailure);
        }
        // Keep the same route+submit latency metric used by FIFO.
        try {
            batchReporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    plan.prefillEp().getIp(),
                    System.currentTimeMillis() - ctx.getStartTime());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report route-submit latency: request_id={}",
                    ctx.getRequestId(), telemetryFailure);
        }
    }

    private static void recordPlacementDiagnostics(BalanceContext ctx, ClusterSnapshot snapshot, String cause) {
        ctx.setSchedulingDiagnostics(Map.of("cause", cause == null ? "placement failed" : cause,
                "prefillEndpointCount", snapshot.prefills().size(),
                "decodeEndpointCount", snapshot.decodes().size()));
    }

    /** Preserve the router's rejection; a missing decision is a scheduling failure. */
    private void onInfeasible(BalanceContext ctx,
                              CompletableFuture<Response> future,
                              Response failureResponse) {
        if (failureResponse != null) {
            future.complete(failureResponse);
            return;
        }
        Logger.debug("[priority-scheduler] no feasible placement, request_id={} priority={}",
                ctx.getRequestId(), ctx.getPriority());
        future.complete(Response.error(StrategyErrorType.BATCH_DISPATCH_FAILED,
                AdmissionRejectReason.UNSPECIFIED, "placement returned no route or failure decision"));
    }

    // ==================== Rollback helpers ====================

    private static boolean preemptionAllows(FlexlbConfig config, VictimStage stage) {
        if (!config.isPriorityOrdering()) {
            return false;
        }
        PreemptionConfig preemption = config.priorityOrdering().getPreemption();
        return preemption != null && preemption.allows(stage);
    }

    private static PreemptionConfig requiredPreemption(FlexlbConfig config) {
        PreemptionConfig preemption = config.priorityOrdering().getPreemption();
        if (preemption == null) {
            throw new IllegalStateException("active preemption policy is required");
        }
        return preemption;
    }

    private static EngineCancellationConfig requiredEngineCancellation(FlexlbConfig config) {
        EngineCancellationConfig cancellation = requiredPreemption(config)
                .getEngineCancellation();
        if (cancellation == null) {
            throw new IllegalStateException(
                    "engineCancellation is required for DECODE_ENGINE_OWNED preemption");
        }
        return cancellation;
    }

    /**
     * N1/P2-2: a victim settle (finishPreemptedById) that found no
     * inflight entry — harmless in isolation, but a burst points at a
     * registration/cleanup race, so it is surfaced as a metric, not only a
     * warn log.
     */
    public void onInflightSettleMiss(String kind) {
        try {
            priorityReporter.reportInflightSettleMiss(kind);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("[priority-scheduler] failed to report inflight settle miss: kind={}",
                    kind, telemetryFailure);
        }
    }

    /**
     * Eviction replan backoff: full jitter in [10, 30] ms (N3 §3.6), damping
     * planning storms over the same shifting victim set. The replan count is
     * bounded by {@link #MAX_EVICTION_REPLANS}, independently of offer retries.
     */
    private static void backoffBeforeEvictionReplan() {
        try {
            Thread.sleep(ThreadLocalRandom.current().nextLong(10, 31));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void releaseDecodeReservation(NormalPlacementPlan plan) {
        DecodeEndpoint decodeEp = plan.decodeEp();
        ServerStatus decode = plan.item().decode();
        if (decodeEp != null && decode != null) {
            decodeEp.release(decode.getRequestId());
        }
    }

    /** Release reservations held by a route response (pre-BatchItem failure paths). */
    private void rollbackRoute(Response routeResponse) {
        if (routeResponse == null || routeResponse.getServerStatus() == null) {
            return;
        }
        for (ServerStatus serverStatus : routeResponse.getServerStatus()) {
            if (serverStatus != null && serverStatus.getRole() == RoleType.DECODE) {
                DecodeEndpoint ep = endpointRegistry.getDecode(
                        serverStatus.getServerIp() + ":" + serverStatus.getHttpPort());
                if (ep != null) {
                    ep.release(serverStatus.getRequestId());
                }
            }
        }
    }

    private static void completeAdmissionError(CompletableFuture<Response> future,
                                               StrategyErrorType errorType,
                                               AdmissionRejectReason reason,
                                               String message) {
        future.complete(Response.error(errorType, reason, message));
    }

    // ==================== Internal ====================

    private record PlacementOutcome(NormalPlacementPlan plan, Response failureResponse) {

        static PlacementOutcome of(NormalPlacementPlan plan) {
            return new PlacementOutcome(plan, null);
        }

        static PlacementOutcome infeasible(Response failureResponse) {
            return new PlacementOutcome(null, failureResponse);
        }
    }
}
