package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.strategy.LoadBalanceStrategy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Auto-TPM priority admission scheduler.
 *
 * <p>Per attempt (up to {@link #MAX_PLAN_RETRIES}):
 * <ol>
 *   <li>Capture a read-only {@link ClusterSnapshot} (prefill queue versions)</li>
 *   <li>Build a {@link NormalPlacementPlan} by reusing the existing
 *       {@link Router#route} (which also performs the decode reservation),
 *       guaranteeing placement parity with the legacy path</li>
 *   <li>Commit via {@link PlanCommitter}: version check then prefill offer;
 *       on {@code VERSION_MISMATCH} / {@code OFFER_FAILED} the decode
 *       reservation is released and the attempt is retried</li>
 *   <li>Phase 3 (gated by {@code autoTpmPrefillQueueEvictEnabled}): when the
 *       offer fails because the prefill queue is full, plan the cheapest
 *       strictly-lower-priority eviction ({@link EvictionPlanner}) and commit
 *       it atomically via {@link PrefillQueueManager#tryReplaceVictimsWithIncoming};
 *       queued victims yield with the retryable {@code NO_AVAILABLE_WORKER}
 *       (contract 5.3 — only engine-accepted victims terminate with
 *       {@code PRIORITY_PREEMPTED})</li>
 * </ol>
 * When no placement is feasible or retries are exhausted, Auto-TPM returns a
 * typed admission failure: proven priority blocker (8430), or admission
 * capacity unavailable within the request budget (8431).
 *
 * <p><b>Auto-TPM switch matrix</b> (all default off; each row gates one
 * behavior entry point, rows below the first also require the first):
 * <pre>
 * | Switch (FlexlbConfig)                       | Gated behavior entry                                        |
 * |---------------------------------------------|-------------------------------------------------------------|
 * | autoTpmEnabled                              | FlexlbBatchScheduler.submit() routes into schedule() at all |
 * |                                             | (off = legacy path; also a precondition for every row below)|
 * | autoTpmPrefillQueueEvictEnabled             | Phase 3 tryPrefillQueueEviction() on OFFER_FAILED           |
 * | autoTpmDecodeReservedEvictEnabled           | Master-local Decode victims join the decode plan pool       |
 * | autoTpmDecodeAcceptedEvictEnabled           | Engine-owned accepted/running victims join the plan pool    |
 * |   + cancelChannel.isSupported(endpoint)     | (both required, per endpoint — EvictionPlanner gate) and    |
 * |                                             | DecodePreemptionCoordinator two-phase Cancel path           |
 * | (removed) DeadlineRescueService            | PR-D replaced by AdmissionLease + reducer-owned deadline     |
 * </pre>
 *
 * <p><b>Plan-commit redesign switches</b> (N3, both default to the new
 * strategy; the legacy value is a gray-release fallback):
 * <pre>
 * | autoTpmCommitStrategy  = lockfree (default) | normal-path commit skips all version checks (§3.3)          |
 * |                          versioned          | legacy optimistic-concurrency commit protocol               |
 * | autoTpmVictimGuardMode = victim_presence    | eviction commits guard each victim's presence (§3.4)        |
 * |                          queue_version      | legacy whole-queue / endpoint version guard                 |
 * </pre>
 */
@Component
public class PriorityAdmissionScheduler {

    /**
     * Safety valve: max placement-plan attempts per request (task43 收编：
     * 原 autoTpmMaxPlanRetries 配置，无需对外暴露，取原默认值）.
     */
    private static final int MAX_PLAN_RETRIES = 3;

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final PlanCommitter planCommitter;
    private final PrioritySloPolicy sloPolicy;
    private final PrioritySchedulerReporter priorityReporter;
    private final BatchSchedulerReporter batchReporter;
    private final EngineCancelChannel cancelChannel;
    private final DecodePreemptionCoordinator preemptionCoordinator;
    private final Map<Long, AtomicInteger> cancelNotFoundReplans = new ConcurrentHashMap<>();

    /**
     * Backpressure counter (Fix C): number of AdmissionLease objects currently
     * in the HANDED_OVER state (prefill succeeded, decode not yet accepted).
     * Incremented at lease creation, decremented when the lease transitions to
     * CLOSED (close or forceCloseAfterHandover). When this exceeds
     * {@code autoTpmPostSuccessBackpressureLimit}, new requests are rejected
     * with 8502 to prevent KV cache exhaustion.
     */
    private final AtomicInteger activeLeaseCount = new AtomicInteger(0);

    int activeLeaseCount() {
        return activeLeaseCount.get();
    }

    @Autowired
    public PriorityAdmissionScheduler(ConfigService configService,
                                      Router router,
                                      EndpointRegistry endpointRegistry,
                                      PlanCommitter planCommitter,
                                      PrioritySloPolicy sloPolicy,
                                      PrioritySchedulerReporter priorityReporter,
                                      BatchSchedulerReporter batchReporter,
                                      EngineCancelChannel cancelChannel,
                                      DecodePreemptionCoordinator preemptionCoordinator) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.planCommitter = planCommitter;
        this.sloPolicy = sloPolicy;
        this.priorityReporter = priorityReporter;
        this.batchReporter = batchReporter;
        this.cancelChannel = cancelChannel;
        this.preemptionCoordinator = preemptionCoordinator;
    }

    /** Test/local-construction convenience: use the same channel for orchestration. */
    PriorityAdmissionScheduler(ConfigService configService,
                               Router router,
                               EndpointRegistry endpointRegistry,
                               PlanCommitter planCommitter,
                               PrioritySloPolicy sloPolicy,
                               PrioritySchedulerReporter priorityReporter,
                               BatchSchedulerReporter batchReporter,
                               EngineCancelChannel cancelChannel) {
        this(configService, router, endpointRegistry, planCommitter, sloPolicy,
                priorityReporter, batchReporter, cancelChannel,
                new DecodePreemptionCoordinator(cancelChannel));
    }

    /**
     * Schedule one request. Called by {@code FlexlbBatchScheduler.submit()}
     * after its duplicate / max-inflight guards, when {@code autoTpmEnabled}.
     *
     * <p>On success the item is inflight-registered and queued on the prefill
     * batcher; the future is completed later by the dispatch pipeline exactly
     * like the legacy path. On failure the future is completed with an error.
     */
    public void schedule(BalanceContext ctx,
                         CompletableFuture<Response> future,
                         InflightRegistrar registrar) {
        FlexlbConfig config = configService.loadBalanceConfig();
        int maxRetries = MAX_PLAN_RETRIES;
        boolean lockfree = isLockfreeCommit(config);
        boolean victimPresence = isVictimPresenceGuard(config);

        // Fix C: backpressure check — reject new requests when too many leases
        // are stuck in the HANDED_OVER state (prefill succeeded but decode hasn't
        // accepted). This prevents KV cache exhaustion when decode is slow/stuck.
        int backpressureLimit = config.getAutoTpmPostSuccessBackpressureLimit();
        if (backpressureLimit > 0 && activeLeaseCount.get() >= backpressureLimit) {
            Logger.debug("[auto-tpm] backpressure limit exceeded, reject request_id={} "
                            + "active_leases={} limit={}",
                    ctx.getRequestId(), activeLeaseCount.get(), backpressureLimit);
            AdmissionFailure failure = AdmissionFailure.resourceExhausted();
            completeAdmissionError(future, failure.errorType(), failure.reason(),
                    "post-success backpressure: active_leases=" + activeLeaseCount.get()
                            + " limit=" + backpressureLimit);
            return;
        }

        // Task40 change 2: reject requests whose SLO deadline has already
        // expired (remaining <= 0) instead of leaving them in the queue.
        // Only the auto-tpm path rejects; deadline rescue is unaffected —
        // it migrates danger-zone requests that still have remaining time.
        long deadlineMs = ctx.getDeadlineMs();
        if (deadlineMs > 0) {
            long nowMs = System.currentTimeMillis();
            if (deadlineMs <= nowMs) {
                Logger.debug("[auto-tpm] slo deadline exceeded, reject request_id={} deadline_ms={} now_ms={}",
                        ctx.getRequestId(), deadlineMs, nowMs);
                AdmissionFailure failure = AdmissionFailure.resourceExhausted();
                completeAdmissionError(future, failure.errorType(), failure.reason(),
                        "admission budget already expired: deadline_ms=" + deadlineMs
                                + " now_ms=" + nowMs);
                return;
            }
        }

        // Tracks whether every failed attempt was an optimistic-concurrency
        // conflict (VERSION_MISMATCH / eviction CONFLICT). Capacity failures
        // retain the typed causal classification from their own snapshot.
        boolean allConflicts = true;
        // Diagnostic tag for retry exhaustion; causal attribution is carried
        // independently by lastCapacityFailure, never inferred from this text.
        String lastFailureReason = null;
        AdmissionFailure lastCapacityFailure = null;
        // Lockfree normal-path retry shrink (N3 §3.3): one primary offer plus
        // one fallback re-route — a second capacity failure rejects fast
        // instead of burning the whole retry budget on full re-routes.
        int offerFailures = 0;
        // P2-1: victims-gone eviction replans spend their own budget (same
        // size as the capacity retry budget) — capacity churn must neither
        // consume the capacity retries nor feed the fast-reject counter.
        int evictionReplans = 0;

        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            // §19.1 schedule_attempt: final value = attempts consumed.
            ctx.setScheduleAttempt(attempt);
            ClusterSnapshot snapshot = ClusterSnapshot.capture(endpointRegistry, config);
            PlacementOutcome outcome = tryNormalPlacement(ctx, future, snapshot);

            if (outcome.plan == null) {
                // Decode eviction (gated, default off): the route failed
                // specifically for Decode capacity. Either victim domain may
                // independently open this planning entry point; the planner
                // applies each switch to its own ownership domain.
                // This method is only reached via the Auto-TPM priority path,
                // so every request here already carries a normalized priority.
                if ((config.isAutoTpmDecodeReservedEvictEnabled()
                        || config.isAutoTpmDecodeAcceptedEvictEnabled())
                        && isDecodeCapacityFailure(outcome.failureResponse)) {
                    DecodeEvictionOutcome eviction =
                            tryDecodeEviction(ctx, future, snapshot, config, registrar);
                    if (eviction == DecodeEvictionOutcome.CONFLICT) {
                        Logger.debug("[auto-tpm] decode eviction conflict (attempt {}/{}), request_id={}",
                                attempt, maxRetries, ctx.getRequestId());
                        if (victimPresence) {
                            // N3 §3.4/3.6: presence-guard conflicts mean the
                            // victims left (capacity churn, not an OCC race)
                            // — replan with jittered backoff, attribute the
                            // exhaustion as a capacity failure.
                            allConflicts = false;
                            lastFailureReason = "victims_gone";
                            backoffBeforeEvictionReplan();
                            // P2-1: spend the dedicated replan budget, not a
                            // capacity retry.
                            if (++evictionReplans <= maxRetries) {
                                attempt--;
                                continue;
                            }
                            AdmissionFailure failure = AdmissionFailure.resourceExhausted();
                            completeAdmissionError(future, failure.errorType(), failure.reason(),
                                    "auto-tpm eviction replans exhausted, reason=victims_gone");
                            return;
                        }
                        continue;
                    }
                    if (eviction == DecodeEvictionOutcome.INFEASIBLE) {
                        // Redesign C-2: no evictable candidates is an ordinary
                        // capacity failure, not a first-attempt terminal state
                        // — the reservations racing us may drain, so retry
                        // with a fresh plan and only fail on exhaustion.
                        allConflicts = false;
                        lastFailureReason = "capacity_no_evict_candidates";
                        Logger.debug("[auto-tpm] no feasible eviction plan (attempt {}/{}), request_id={} priority={}",
                                attempt, maxRetries, ctx.getRequestId(), ctx.getPriority());
                        continue;
                    }
                    // COMMITTED, or FAILED with the future already completed.
                    return;
                }
                if (isDecodeCapacityFailure(outcome.failureResponse)) {
                    PriorityRequestEnvelope envelope = new PriorityRequestEnvelope(
                            ctx.getRequestId(), ctx.getPriority(),
                            ctx.getRequest().getSeqLen(), ctx.getRequest().getMaxNewTokens(),
                            ctx.getStartTime(), ctx.getRequestSloMs(), ctx.getDeadlineMs(),
                            ctx.getRequest().getSeqLen(),
                            ctx.getRequest().getSeqLen()
                                    + config.effectiveMaxNewTokensForReservation(ctx.getRequest().getMaxNewTokens()));
                    AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                            envelope, new ArrayList<>(snapshot.decodes().values()));
                    completeAdmissionError(future, failure.errorType(),
                            failure.reason(), failure.message());
                    return;
                }
                onInfeasible(ctx, future, outcome.failureResponse);
                return;
            }

            // P1-1: flip the reservation into the queued phase BEFORE the
            // commit can publish the item to the batcher — marking after the
            // commit (in onCommitted) races the dispatch side's
            // the dispatch ownership claim and can leave a stale queued mark that hides
            // the request from the engine concurrency gate. Every failure path
            // below runs releaseDecodeReservation() (and a retry re-reserves),
            // both of which clear the mark.
            if (outcome.plan.decodeEp() != null) {
                outcome.plan.decodeEp().markQueuedPhase(ctx.getRequestId());
            }
            PlanCommitter.CommitResult result = planCommitter.commit(outcome.plan, registrar, lockfree);
            if (result == PlanCommitter.CommitResult.SUCCESS) {
                onCommitted(ctx, outcome.plan);
                bindAdmissionLease(outcome.plan, registrar);
                return;
            }

            if (result == PlanCommitter.CommitResult.VERSION_MISMATCH) {
                priorityReporter.reportPlanConflict("normal_placement_version");
            }
            if (result == PlanCommitter.CommitResult.OFFER_FAILED) {
                // Capacity-rooted failure (typically a full prefill queue).
                allConflicts = false;
                // P1-4 (design §B.3 deviation): the fallback re-route would
                // deterministically re-pick the worker whose queue just
                // rejected the offer (same cost view) — steer the next route
                // away from it instead of reordering candidates (N4).
                ctx.setExcludedPrefillIpPort(outcome.plan.prefillEp().ipPort());
                lastCapacityFailure = AdmissionFailureClassifier.classifyPrefill(
                        outcome.plan.envelope(),
                        outcome.plan.prefillEp().getBatcher().queueManager().snapshot());
            }

            // Phase 3: the offer failed — typically a full prefill queue.
            // Try to free queue slots by evicting strictly lower-priority
            // queued requests (gated, default off). This method is only
            // reached via the Auto-TPM priority path, so every request here
            // already carries a normalized priority.
            if (result == PlanCommitter.CommitResult.OFFER_FAILED
                    && config.isAutoTpmPrefillQueueEvictEnabled()) {
                EvictionOutcome eviction = tryPrefillQueueEviction(outcome.plan, config, registrar);
                switch (eviction) {
                    case COMMITTED -> {
                        onCommitted(ctx, outcome.plan);
                        bindAdmissionLease(outcome.plan, registrar);
                        return;
                    }
                    case INFEASIBLE -> {
                        // Redesign C-2: same fall-back as the decode-eviction
                        // INFEASIBLE — capacity failure, retry with a fresh plan.
                        releaseDecodeReservation(outcome.plan);
                        allConflicts = false;
                        // P2-1: a genuine queue-full failure (nothing evictable)
                        // — counts toward the lockfree fast-reject.
                        offerFailures++;
                        lastFailureReason = "capacity_no_evict_candidates";
                        Logger.debug("[auto-tpm] no feasible eviction plan (attempt {}/{}), request_id={} priority={}",
                                attempt, maxRetries, ctx.getRequestId(), ctx.getPriority());
                        continue;
                    }
                    case PARTIAL_FAILURE -> {
                        releaseDecodeReservation(outcome.plan);
                        AdmissionFailure failure = AdmissionFailure.resourceExhausted();
                        completeAdmissionError(future, failure.errorType(), failure.reason(),
                                "eviction commit partial failure");
                        return;
                    }
                    case REJECTED -> {
                        releaseDecodeReservation(outcome.plan);
                        return;
                    }
                    case CONFLICT -> {
                        // fall through: release the decode reservation and retry
                        if (victimPresence) {
                            // N3 §3.4/3.6: queued victims left before commit —
                            // capacity churn; replan with jittered backoff.
                            // P2-1: spends the dedicated replan budget and
                            // never feeds the fast-reject counter (the churn
                            // that took the victims may also drain the queue).
                            lastFailureReason = "victims_gone";
                            releaseDecodeReservation(outcome.plan);
                            backoffBeforeEvictionReplan();
                            if (++evictionReplans <= maxRetries) {
                                attempt--;
                                continue;
                            }
                            AdmissionFailure failure = AdmissionFailure.resourceExhausted();
                            completeAdmissionError(future, failure.errorType(), failure.reason(),
                                    "auto-tpm eviction replans exhausted, reason=victims_gone");
                            return;
                        }
                    }
                }
            }

            // VERSION_MISMATCH / OFFER_FAILED / eviction conflict: nothing
            // queued — release the decode reservation taken by route() and
            // retry with a fresh plan.
            releaseDecodeReservation(outcome.plan);
            Logger.debug("[auto-tpm] plan commit {} (attempt {}/{}), request_id={}",
                    result, attempt, maxRetries, ctx.getRequestId());

            // P2-1: only genuine capacity-rooted offer failures reach this
            // point (victims_gone replans continue above) — count them here.
            if (result == PlanCommitter.CommitResult.OFFER_FAILED) {
                offerFailures++;
            }
            if (lockfree && result == PlanCommitter.CommitResult.OFFER_FAILED && offerFailures >= 2) {
                // N3 §3.3: primary + one fallback offer both hit a capacity
                // failure — fast reject with queue-full semantics instead of
                // exhausting the full re-route budget (no version conflicts
                // exist on the lockfree path, so waiting cannot help within
                // the commit window).
                // P2-1: the reason is the current attempt's own failure cause,
                // never a stale lastFailureReason from an earlier attempt.
                AdmissionFailure failure = lastCapacityFailure != null
                        ? lastCapacityFailure : AdmissionFailure.resourceExhausted();
                completeAdmissionError(future, failure.errorType(), failure.reason(),
                        failure.message());
                return;
            }
        }

        AdmissionFailure finalFailure = !allConflicts && lastCapacityFailure != null
                ? lastCapacityFailure : AdmissionFailure.resourceExhausted();
        completeAdmissionError(future, finalFailure.errorType(), finalFailure.reason(),
                lastFailureReason != null
                        ? "auto-tpm plan retries exhausted, reason=" + lastFailureReason
                        : finalFailure.message());
    }

    // ==================== Phase 3: prefill queue eviction ====================

    private enum EvictionOutcome {
        /** Incoming committed in place of the victims. */
        COMMITTED,
        /** No feasible plan (no/insufficient strictly-lower-priority candidates). */
        INFEASIBLE,
        /** Optimistic-concurrency conflict — retry with a fresh plan. */
        CONFLICT,
        /** Defensive: victims removed but incoming not enqueued (should be unreachable). */
        PARTIAL_FAILURE,
        /** Typed incoming rejection already completed on the item future. */
        REJECTED
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
                envelope, List.of(queueSnapshot),
                Map.of(queueSnapshot.endpointId(), item.hitCache()), config, failures);
        if (proposal == null) {
            priorityReporter.reportEvictionPlan(envelope.priority(), "prefill_queue_full", "infeasible");
            Logger.debug("[auto-tpm] eviction plan infeasible, request_id={} priority={} "
                            + "phase=prefill_queue candidates_seen={} reasons={}",
                    envelope.requestId(), envelope.priority(),
                    queueSnapshot.items().size(), failures);
            AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                    envelope, queueSnapshot);
            completeAdmissionError(item.future(), failure.errorType(),
                    failure.reason(), failure.message());
            return EvictionOutcome.REJECTED;
        }
        priorityReporter.reportEvictionPlan(envelope.priority(), "prefill_queue_full", "feasible");

        PrefillEvictionPlan evictionPlan = new PrefillEvictionPlan(
                envelope, item, plan.routeResponse(), proposal);
        if (!registrar.registerInflight(item)) {
            Logger.warn("[auto-tpm] eviction commit failed: duplicate request_id={}",
                    envelope.requestId());
            return EvictionOutcome.CONFLICT;
        }

        List<Long> victimIds = new ArrayList<>(proposal.victims().size());
        for (QueuedRequestSnapshot victim : proposal.victims()) {
            victimIds.add(victim.requestId());
        }
        // N3 §3.4: victim-presence guard replaces the whole-queue version
        // guard — unrelated queue mutations no longer abort the commit.
        PrefillQueueManager.ReplaceOutcome replace = isVictimPresenceGuard(config)
                ? queueManager.tryReplaceVictimsPresent(victimIds, item)
                : queueManager.tryReplaceVictimsWithIncoming(victimIds, item, proposal.queueVersion());

        if (replace.isVersionMismatch()) {
            registrar.unregisterInflight(item);
            priorityReporter.reportPlanConflict("prefill_queue_version");
            priorityReporter.reportEvictionCommit(envelope.priority(), "prefill_queue_full",
                    "version_mismatch");
            return EvictionOutcome.CONFLICT;
        }

        if (replace.isVictimGone()) {
            // Zero-side-effect abort: a victim left the queue — usually the
            // queue freed a slot, so try one direct offer before replanning
            // (N3 §3.4). The item is already inflight-registered.
            priorityReporter.reportEvictionCommit(envelope.priority(), "prefill_queue_full",
                    "victim_gone");
            if (plan.prefillEp().getBatcher().tryOffer(item)) {
                Logger.debug("[auto-tpm] eviction victims gone, direct offer succeeded: "
                                + "request_id={} missing_victims={} worker={}",
                        envelope.requestId(), replace.missingVictimIds(), proposal.endpointId());
                return EvictionOutcome.COMMITTED;
            }
            registrar.unregisterInflight(item);
            Logger.debug("[auto-tpm] eviction victims gone, replan: request_id={} "
                            + "missing_victims={} worker={}",
                    envelope.requestId(), replace.missingVictimIds(), proposal.endpointId());
            return EvictionOutcome.CONFLICT;
        }

        // Victims removed from the queue are never re-inserted (design doc
        // 9.5): drive each to its terminal state, releasing its decode
        // reservation. The engine never saw a queued victim, so the
        // client-visible terminal is the retryable NO_AVAILABLE_WORKER
        // (yielded, contract 5.3); metrics still count it as preempted.
        // Idempotent via the inflight lifecycle.
        for (BatchItem victim : replace.removed()) {
            registrar.finishYielded(victim,
                    "yielded to higher-priority request " + envelope.requestId());
            priorityReporter.reportVictim(victim.priority(), envelope.priority(),
                    "prefill_queued", "prefill_queue_full");
            priorityReporter.reportPriorityPreempt("prefill_queued");
            Logger.debug("[auto-tpm] victim preempted: victim_id={} victim_priority={} "
                            + "terminal=yielded_8400 incoming_id={} incoming_priority={} worker={}",
                    victim.requestId(), victim.priority(),
                    envelope.requestId(), envelope.priority(), proposal.endpointId());
        }

        if (replace.isPartialFailure()) {
            // Defensive path — unreachable under the atomic replace: victims
            // (if any) went terminal above, the incoming fails explicitly.
            registrar.unregisterInflight(item);
            priorityReporter.reportEvictionCommit(envelope.priority(), "prefill_queue_full",
                    "partial_failure");
            Logger.error("[auto-tpm] eviction commit partial failure, request_id={} victims_removed={}",
                    envelope.requestId(), replace.removed().size());
            return EvictionOutcome.PARTIAL_FAILURE;
        }

        priorityReporter.reportEvictionCommit(envelope.priority(), "prefill_queue_full", "success");
        // §19.1 plan observability: on the combined decode+prefill eviction
        // path keep the primary decode_evict label; accumulate cost/victims.
        BalanceContext itemCtx = item.ctx();
        if (!"decode_evict".equals(itemCtx.getPlanType())) {
            itemCtx.setPlanType("prefill_evict");
        }
        itemCtx.setPlanCost(itemCtx.getPlanCost() + proposal.netCost());
        itemCtx.setVictimCount(itemCtx.getVictimCount() + proposal.victims().size());
        Logger.debug("[auto-tpm] eviction committed: request_id={} priority={} victims={} "
                        + "net_cost={} worker={}",
                envelope.requestId(), envelope.priority(), evictionPlan.proposal().victims().size(),
                proposal.netCost(), proposal.endpointId());
        return EvictionOutcome.COMMITTED;
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

    private enum DecodeEvictionOutcome {
        /** Eviction applied and the incoming request committed. */
        COMMITTED,
        /** No feasible plan (no/insufficient strictly-lower-priority enabled victims). */
        INFEASIBLE,
        /** Optimistic-concurrency conflict — retry with a fresh plan. */
        CONFLICT,
        /** Eviction applied but placement failed; future already completed. */
        FAILED,
        /** Engine Cancel transaction continues asynchronously. */
        PENDING
    }

    /**
     * Plan and atomically commit a Decode eviction (design doc 11-13, 17.2),
     * then place the incoming request on the freed endpoint. Master-local and
     * Engine-owned victim domains are enabled independently by configuration.
     * Uses the pre-route {@link ClusterSnapshot} decode views so the
     * admission-version check detects any interference since plan build.
     */
    private DecodeEvictionOutcome tryDecodeEviction(BalanceContext ctx,
                                                    CompletableFuture<Response> future,
                                                    ClusterSnapshot snapshot,
                                                    FlexlbConfig config,
                                                    InflightRegistrar registrar) {
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();
        long effectiveMaxNewTokens = config.effectiveMaxNewTokensForReservation(maxNewTokens);
        // Planning envelope: the decode planners consume only priority and
        // hardKvTokens; the endpoint-aware deadline is rebuilt after placement.
        PriorityRequestEnvelope planEnvelope = new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), ctx.getRequestSloMs(), ctx.getDeadlineMs(),
                seqLen, seqLen + effectiveMaxNewTokens);

        List<DecodeEndpointSnapshot> decodes = new ArrayList<>(snapshot.decodes().values());
        Map<String, String> failures = new HashMap<>();
        DecodeEvictionProposal proposal =
                EvictionPlanner.planDecode(planEnvelope, decodes, config, cancelChannel, failures);
        if (proposal == null) {
            priorityReporter.reportEvictionPlan(ctx.getPriority(),
                    infeasibleDecodeCase(planEnvelope, decodes), "infeasible");
            // Redesign D-1: carry the plan phase and candidate counters so an
            // infeasible burst is attributable from logs alone.
            int candidatesSeen = 0;
            int candidatesEligible = 0;
            boolean localEnabled = config.isAutoTpmDecodeReservedEvictEnabled();
            boolean engineEnabled = config.isAutoTpmDecodeAcceptedEvictEnabled();
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
            Logger.debug("[auto-tpm] decode eviction plan infeasible, request_id={} priority={} "
                            + "phase={} candidates_seen={} candidates_eligible={} reasons={}",
                    ctx.getRequestId(), ctx.getPriority(), phase,
                    candidatesSeen, candidatesEligible, failures);
            AdmissionFailure failure = AdmissionFailureClassifier.classifyDecode(
                    planEnvelope, decodes);
            completeAdmissionError(future, failure.errorType(), failure.reason(), failure.message());
            return DecodeEvictionOutcome.FAILED;
        }
        priorityReporter.reportEvictionPlan(ctx.getPriority(), proposal.evictionCase(), "feasible");

        DecodeEndpointSnapshot target = snapshot.decodes().get(proposal.endpointId());
        DecodeEndpoint decodeEp = target.endpoint();
        long expectedKvTokens = target.realKvTotal() > 0
                ? Math.min(seqLen + effectiveMaxNewTokens, target.realKvTotal())
                : seqLen + effectiveMaxNewTokens;

        // Ownership is homogeneous by planner invariant: Master-queued victims
        // use a local transaction; Engine-may-have-seen/accepted/running
        // victims use the tokenized Cancel coordinator.
        if (proposal.requiresEngineCancel()) {
            startEngineCancelPreemption(ctx, future, config, registrar, proposal,
                    target, decodeEp, seqLen, expectedKvTokens);
            return DecodeEvictionOutcome.PENDING;
        }

        List<Long> reservedVictimIds = new ArrayList<>(proposal.victims().size());
        for (DecodeRequestSnapshot victim : proposal.victims()) {
            reservedVictimIds.add(victim.requestId());
        }

        if (isVictimPresenceGuard(config)) {
            // N3 §3.4: presence-guarded commit — conditionally release each
            // victim still holding its reservation; no admission-version check.
            DecodeEndpoint.PresenceEvictionOutcome presence =
                    decodeEp.tryReleaseVictimsIfHeldAndReserveIncoming(
                            reservedVictimIds, ctx.getRequestId(), seqLen, expectedKvTokens,
                            ctx.getPriority(), ctx.getDeadlineMs());
            if (!presence.success()) {
                // Victims already freed are NOT rolled back (design doc 9.5)
                // — drive them terminal; the gone victims were dispatched or
                // settled and must not be touched (§3.4 common rule).
                for (DecodeRequestSnapshot victim : proposal.victims()) {
                    if (presence.freedVictimIds().contains(victim.requestId())) {
                        finishDecodeVictim(ctx, registrar, victim, "decode_reserved", proposal);
                    }
                }
                priorityReporter.reportEvictionCommit(ctx.getPriority(), proposal.evictionCase(),
                        "victim_gone");
                Logger.debug("[auto-tpm] decode eviction victims gone, replan: request_id={} "
                                + "freed={} planned={} worker={}",
                        ctx.getRequestId(), presence.freedVictimIds().size(),
                        reservedVictimIds.size(), proposal.endpointId());
                return DecodeEvictionOutcome.CONFLICT;
            }
        } else {
            DecodeEndpoint.ReleaseReserveResult release = decodeEp.tryReleaseVictimsAndReserveIncoming(
                    reservedVictimIds, ctx.getRequestId(), seqLen, expectedKvTokens,
                    ctx.getPriority(), ctx.getDeadlineMs(), proposal.admissionVersion());
            if (release != DecodeEndpoint.ReleaseReserveResult.SUCCESS) {
                priorityReporter.reportPlanConflict("decode_admission_version");
                priorityReporter.reportEvictionCommit(ctx.getPriority(), proposal.evictionCase(),
                        release == DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE
                                ? "victim_gone" : "version_mismatch");
                return DecodeEvictionOutcome.CONFLICT;
            }
        }

        // Shadow accounting already reversed atomically; drive each victim to
        // its terminal state (idempotent, design doc 17.3). Reserved-only
        // victims were never seen by the engine — retryable 8400 (contract 5.3).
        for (DecodeRequestSnapshot victim : proposal.victims()) {
            finishDecodeVictim(ctx, registrar, victim, "decode_reserved", proposal);
        }
        priorityReporter.reportEvictionCommit(ctx.getPriority(), proposal.evictionCase(), "success");
        recordDecodePlanObservability(ctx, proposal);

        return placeAfterDecodeEviction(ctx, future, config, registrar, decodeEp);
    }

    /**
     * Drive one decode eviction victim to its terminal state and emit the
     * per-victim metrics ({@code stage} distinguishes reserved vs accepted
     * victims). Terminal split per contract 5.3: a reserved-only victim was
     * never seen by the engine — retryable NO_AVAILABLE_WORKER (yielded);
     * an engine-accepted victim keeps PRIORITY_PREEMPTED.
     */
    private void finishDecodeVictim(BalanceContext ctx, InflightRegistrar registrar,
                                    DecodeRequestSnapshot victim, String stage,
                                    DecodeEvictionProposal proposal) {
        boolean accepted = victim.phase().isEngineConfirmed();
        String terminal;
        if (accepted) {
            terminal = "preempted_8429";
            registrar.finishPreemptedById(victim.requestId(),
                    "preempted by higher-priority request " + ctx.getRequestId());
        } else {
            terminal = "yielded_8400";
            registrar.finishYieldedById(victim.requestId(),
                    "yielded to higher-priority request " + ctx.getRequestId());
        }
        priorityReporter.reportVictim(victim.priority(), ctx.getPriority(),
                stage, proposal.evictionCase());
        priorityReporter.reportPriorityPreempt(stage);
        priorityReporter.reportVictimKvTokens(victim.priority(), stage, victim.kvTokens());
        Logger.debug("[auto-tpm] decode victim preempted: victim_id={} victim_priority={} "
                        + "stage={} terminal={} kv_tokens={} incoming_id={} incoming_priority={} worker={}",
                victim.requestId(), victim.priority(), stage, terminal, victim.kvTokens(),
                ctx.getRequestId(), ctx.getPriority(), proposal.endpointId());
    }

    /** §19.1 plan observability for the decode eviction path. */
    private static void recordDecodePlanObservability(BalanceContext ctx,
                                                      DecodeEvictionProposal proposal) {
        ctx.setPlanType("decode_evict");
        ctx.setPlanCost(proposal.totalCost());
        ctx.setVictimCount(proposal.victims().size());
        Logger.debug("[auto-tpm] decode eviction committed: request_id={} priority={} case={} "
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
                                             long expectedKvTokens) {
        String detail = "preempted by higher-priority request " + ctx.getRequestId();
        DecodePreemptionCoordinator.Request request =
                new DecodePreemptionCoordinator.Request(
                        decodeEp, proposal.admissionVersion(),
                        !isVictimPresenceGuard(config),
                        ctx.getRequestId(), seqLen, expectedKvTokens,
                        ctx.getPriority(), ctx.getDeadlineMs(),
                        proposal.victims(), config.getAutoTpmCancelAckTimeoutMs(),
                        config.getAutoTpmCancelCompletionTimeoutMs(),
                        () -> !future.isDone(), detail);

        for (DecodeRequestSnapshot victim : proposal.victims()) {
            priorityReporter.reportCancelRequest(proposal.endpointId(), victim.priority());
            priorityReporter.reportCancel(victim.priority(), "PRIORITY_PREEMPTED");
        }

        preemptionCoordinator.execute(request, registrar).whenComplete((result, error) -> {
            if (error != null || result == null) {
                cancelNotFoundReplans.remove(ctx.getRequestId());
                priorityReporter.reportCancelTimeout(proposal.endpointId(), ctx.getPriority());
                completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED,
                        "priority cancel coordinator failed");
                return;
            }
            switch (result.code()) {
                case COMMITTED -> {
                    cancelNotFoundReplans.remove(ctx.getRequestId());
                    // The client admission deadline can race the async Cancel
                    // completion callback. The same future monitor is held by
                    // the pre-inflight deadline reducer, making the terminal
                    // check and register+offer handoff one atomic decision.
                    synchronized (future) {
                        if (future.isDone()) {
                            decodeEp.release(ctx.getRequestId());
                            Logger.debug("[auto-tpm] drop committed preemption after admission close: "
                                            + "request_id={} worker={}",
                                    ctx.getRequestId(), proposal.endpointId());
                            return;
                        }
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
                        recordDecodePlanObservability(ctx, proposal);
                        placeAfterDecodeEviction(ctx, future, config, registrar, decodeEp);
                    }
                }
                case REPLAN_NOT_FOUND, CONFLICT -> {
                    int replans = cancelNotFoundReplans
                            .computeIfAbsent(ctx.getRequestId(), ignored -> new AtomicInteger())
                            .incrementAndGet();
                    if (replans <= 1 && !future.isDone()) {
                        schedule(ctx, future, registrar);
                    } else {
                        cancelNotFoundReplans.remove(ctx.getRequestId());
                        completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                                result.detail());
                    }
                }
                case CONTROL_FAILED -> {
                    cancelNotFoundReplans.remove(ctx.getRequestId());
                    priorityReporter.reportCancelTimeout(
                            proposal.endpointId(), ctx.getPriority());
                    completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                            AdmissionRejectReason.RESOURCE_EXHAUSTED,
                            result.detail());
                }
            }
        });
    }

    /**
     * Place the incoming request after a successful decode eviction. The
     * normal route cannot be replayed — the incoming reservation just taken
     * would make the decode strategy filter out its own endpoint — so the
     * decode {@link ServerStatus} is built manually and only prefill goes
     * through its selection strategy.
     */
    private DecodeEvictionOutcome placeAfterDecodeEviction(BalanceContext ctx,
                                                           CompletableFuture<Response> future,
                                                           FlexlbConfig config,
                                                           InflightRegistrar registrar,
                                                           DecodeEndpoint decodeEp) {
        ServerStatus prefill = selectPrefillForDecodeEviction(
                ctx, config, decodeEp.getStatus().getGroup());
        if (prefill == null || !prefill.isSuccess()) {
            decodeEp.release(ctx.getRequestId());
            completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "no prefill worker after decode eviction");
            return DecodeEvictionOutcome.FAILED;
        }
        PrefillEndpoint prefillEp = endpointRegistry.getPrefill(
                prefill.getServerIp() + ":" + prefill.getHttpPort());
        if (prefillEp == null) {
            decodeEp.release(ctx.getRequestId());
            completeAdmissionError(future, StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    "prefill endpoint not registered after decode eviction");
            return DecodeEvictionOutcome.FAILED;
        }

        ServerStatus decode = buildDecodeServerStatus(ctx, decodeEp);
        Response routeResponse = new Response();
        routeResponse.setSuccess(true);
        routeResponse.setServerStatus(List.of(prefill, decode));

        PriorityRequestEnvelope envelope = buildEnvelope(ctx, prefill, prefillEp, decodeEp);

        BatchItem item = new BatchItem(ctx, future, routeResponse,
                FlexlbBatchScheduler.copyOf(prefill), FlexlbBatchScheduler.copyOf(decode),
                prefillEp, decodeEp, System.currentTimeMillis());

        NormalPlacementPlan plan = new NormalPlacementPlan(envelope, item, routeResponse,
                prefillEp.getBatcher().queueVersion(), decodeEp.admissionVersion());

        // P1-1: queued-phase mark precedes the commit (same rationale as the
        // normal path); every failure path below releases the reservation,
        // which clears the mark.
        decodeEp.markQueuedPhase(ctx.getRequestId());
        PlanCommitter.CommitResult result = planCommitter.commit(plan, registrar,
                isLockfreeCommit(config));
        if (result == PlanCommitter.CommitResult.SUCCESS) {
            onCommitted(ctx, plan);
            bindAdmissionLease(plan, registrar);
            return DecodeEvictionOutcome.COMMITTED;
        }
        if (result == PlanCommitter.CommitResult.VERSION_MISMATCH) {
            releaseDecodeReservation(plan);
            priorityReporter.reportPlanConflict("normal_placement_version");
            return DecodeEvictionOutcome.CONFLICT;
        }

        // OFFER_FAILED — combine with Phase 3: try a prefill-queue eviction
        // before giving up (design doc 13.5).
        if (config.isAutoTpmPrefillQueueEvictEnabled()) {
            EvictionOutcome eviction = tryPrefillQueueEviction(plan, config, registrar);
            switch (eviction) {
                case COMMITTED -> {
                    onCommitted(ctx, plan);
                    bindAdmissionLease(plan, registrar);
                    return DecodeEvictionOutcome.COMMITTED;
                }
                case CONFLICT -> {
                    releaseDecodeReservation(plan);
                    return DecodeEvictionOutcome.CONFLICT;
                }
                case PARTIAL_FAILURE -> {
                    releaseDecodeReservation(plan);
                    AdmissionFailure failure = AdmissionFailure.resourceExhausted();
                    completeAdmissionError(future, failure.errorType(), failure.reason(),
                            "eviction commit partial failure");
                    return DecodeEvictionOutcome.FAILED;
                }
                case REJECTED -> {
                    // tryPrefillQueueEviction already completed the typed error.
                    releaseDecodeReservation(plan);
                    return DecodeEvictionOutcome.FAILED;
                }
                case INFEASIBLE -> {
                    // Fall through to classify the unchanged current queue.
                }
            }
        }
        releaseDecodeReservation(plan);
        AdmissionFailure failure = AdmissionFailureClassifier.classifyPrefill(
                envelope, prefillEp.getBatcher().queueManager().snapshot());
        completeAdmissionError(future, failure.errorType(), failure.reason(), failure.message());
        return DecodeEvictionOutcome.FAILED;
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
                config.getStrategyForRoleType(RoleType.PREFILL));
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
                                                CompletableFuture<Response> future,
                                                ClusterSnapshot snapshot) {
        Response routeResponse = router.route(ctx);
        // P1-4: the exclusion steers exactly one re-route — clear it so later
        // attempts (or a rescue re-entry) see the full candidate set again.
        ctx.setExcludedPrefillIpPort(null);
        if (routeResponse == null || !routeResponse.isSuccess()) {
            // Parity with the legacy path: a failed route holds no reservation.
            return PlacementOutcome.infeasible(routeResponse);
        }

        ServerStatus prefill = FlexlbBatchScheduler.findServer(routeResponse, RoleType.PREFILL);
        ServerStatus decode = FlexlbBatchScheduler.findServer(routeResponse, RoleType.DECODE);
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

        PriorityRequestEnvelope envelope = buildEnvelope(ctx, prefill, prefillEp, decodeEp);

        BatchItem item = new BatchItem(ctx, future, routeResponse,
                FlexlbBatchScheduler.copyOf(prefill), FlexlbBatchScheduler.copyOf(decode),
                prefillEp, decodeEp, System.currentTimeMillis());

        // Prefill queue version comes from the snapshot (pre-route) when
        // available; decode admission version is captured post-reserve so only
        // plan-to-commit interference is detected.
        PrefillEndpointSnapshot prefillSnapshot = snapshot.prefills().get(prefillIpPort);
        long prefillQueueVersion = prefillSnapshot != null
                ? prefillSnapshot.queueVersion()
                : prefillEp.getBatcher().queueVersion();
        long decodeAdmissionVersion = decodeEp != null ? decodeEp.admissionVersion() : 0;

        return PlacementOutcome.of(new NormalPlacementPlan(
                envelope, item, routeResponse, prefillQueueVersion, decodeAdmissionVersion));
    }

    private PriorityRequestEnvelope buildEnvelope(BalanceContext ctx,
                                                  ServerStatus prefill,
                                                  PrefillEndpoint prefillEp,
                                                  DecodeEndpoint decodeEp) {
        long seqLen = ctx.getRequest().getSeqLen();
        long maxNewTokens = ctx.getRequest().getMaxNewTokens();
        long effectiveMaxNewTokens = configService.loadBalanceConfig().effectiveMaxNewTokensForReservation(maxNewTokens);
        long hitCache = prefill.getDebugInfo() != null ? prefill.getDebugInfo().getHitCacheLen() : 0;
        long predictedPrefillMs = prefillEp.getPredictor().estimateMs(seqLen, hitCache);
        // Priority and SLO are read from the immutable budget (single source
        // of truth); getPriority() / getRequestSloMs() delegate to ctx.budget().
        int priority = ctx.getPriority();
        long requestSloMs = ctx.getRequestSloMs() > 0
                ? ctx.getRequestSloMs()
                : sloPolicy.requestSloMs(seqLen, priority);
        // The admission deadline is always the coarse budget deadline
        // (admittedAtMs + requestSloMs). getDeadlineMs() delegates to
        // ctx.budget() (coarse deadline).
        long deadlineMs = ctx.getDeadlineMs() > 0
                ? ctx.getDeadlineMs()
                : PrioritySloPolicy.deadlineMs(ctx.getStartTime(), requestSloMs, predictedPrefillMs);
        long kvTotal = decodeEp != null ? decodeEp.realKvTotal() : 0;
        long expectedKvTokens = kvTotal > 0
                ? Math.min(seqLen + effectiveMaxNewTokens, kvTotal)
                : seqLen + effectiveMaxNewTokens;
        return new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), requestSloMs, deadlineMs, seqLen, expectedKvTokens);
    }

    // ==================== Outcome handling ====================

    /**
     * Create an {@link AdmissionLease} and bind it to the request future
     * (PR-D §2.4, fix for triple-lock OOM). Called at every plan-commit
     * success point after {@link #onCommitted}. The lease is the single
     * ownership boundary: success → {@code handoverToEngine} (seal +
     * schedule soft timeout); failure/timeout → {@code close} (tryRemove +
     * release + unregister). The soft timeout fires when decode hasn't
     * accepted within {@code autoTpmPostSuccessSoftTimeoutMs} →
     * {@code forceCloseAfterHandover} (release + cancel signal).
     * <p>Fix C: increments {@link #activeLeaseCount} at creation and
     * decrements when the lease closes (via the onClose callback).
     * <p>Legacy path ({@code budget == null}) never creates a lease.
     */
    private void bindAdmissionLease(NormalPlacementPlan plan,
                                    InflightRegistrar registrar) {
        BalanceContext ctx = plan.item().ctx();
        if (ctx.budget() == null) {
            return;
        }
        FlexlbConfig config = configService.loadBalanceConfig();
        long softTimeoutMs = config.getAutoTpmPostSuccessSoftTimeoutMs();
        PrefillQueueManager prefillQueue = plan.prefillEp().getBatcher().queueManager();
        AdmissionLease lease = new AdmissionLease(
                plan.item(), plan.decodeEp(), prefillQueue, registrar,
                softTimeoutMs, activeLeaseCount::decrementAndGet);
        // Fix C: increment after construction, before bindTo. If bindTo
        // synchronously triggers close() (future already completed), the
        // callback's decrementAndGet sees the counter already incremented.
        activeLeaseCount.incrementAndGet();
        try {
            // WorkerStatus acceptance belongs to the registrar's exact
            // inflight generation.  Attach before binding the future so a
            // racing ACK/status observation cannot strand the counter.
            if (!registrar.attachAdmissionLease(plan.item(), lease)) {
                // Attachment failure says only that this inflight generation
                // is no longer live; it is not evidence of Decode acceptance.
                lease.close();
                return;
            }
            lease.bindTo(plan.item().future());
        } catch (RuntimeException e) {
            activeLeaseCount.decrementAndGet();
            throw e;
        }
    }

    private void onCommitted(BalanceContext ctx, NormalPlacementPlan plan) {
        ctx.setRouteSubmittedNanos(System.nanoTime());
        // N3 §3.8: plan age quantifies how stale the committed plan view was
        // (snapshot/build → successful commit).
        priorityReporter.reportPlanAge(plan.envelope().priority(),
                Math.max(0, System.currentTimeMillis() - plan.createdAtMs()));
        // N2/P1-1: the queued-phase mark is set BEFORE the commit (schedule /
        // placeAfterDecodeEviction) — marking here raced the dispatch side's
        // tryMarkEngineMayHaveSeen for items that dispatched immediately.
        // §19.1 plan_type: eviction paths set their label before this point.
        if (ctx.getPlanType() == null || ctx.getPlanType().isEmpty()) {
            ctx.setPlanType("normal");
        }
        ServerStatus prefill = plan.item().prefill();
        ctx.setScheduledPrefillEndpoint(prefill.getServerIp() + ":" + prefill.getHttpPort());
        priorityReporter.reportNormalPlacement(plan.envelope().priority());
        // Parity with the legacy path's route+submit latency metric.
        batchReporter.reportRouteSubmitTimeMs(
                RoleType.PREFILL.name(),
                plan.prefillEp().getIp(),
                System.currentTimeMillis() - ctx.getStartTime());
    }

    /**
     * No feasible normal placement. Eviction-based rescue is Phase 2+; the
     * MVP fails the request (never a silent drop — the caller gets an
     * explicit retryable error, or the router's own failure response).
     */
    private void onInfeasible(BalanceContext ctx,
                              CompletableFuture<Response> future,
                              Response failureResponse) {
        if (failureResponse != null) {
            if (isCapacityFailure(failureResponse)) {
                AdmissionFailure failure = AdmissionFailure.resourceExhausted();
                completeAdmissionError(future, failure.errorType(), failure.reason(),
                        failure.message());
            } else {
                future.complete(failureResponse);
            }
            return;
        }
        Logger.debug("[auto-tpm] no feasible placement, request_id={} priority={}",
                ctx.getRequestId(), ctx.getPriority());
        AdmissionFailure failure = AdmissionFailure.resourceExhausted();
        completeAdmissionError(future, failure.errorType(), failure.reason(), failure.message());
    }

    private static boolean isCapacityFailure(Response response) {
        int code = response.getCode();
        return code == StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode()
                || code == StrategyErrorType.NO_PREFILL_WORKER.getErrorCode()
                || code == StrategyErrorType.NO_DECODE_WORKER.getErrorCode()
                || code == StrategyErrorType.QUEUE_FULL.getErrorCode();
    }

    // ==================== Rollback helpers ====================

    /** {@code autoTpmCommitStrategy}: any value but the legacy "versioned" is lockfree. */
    private static boolean isLockfreeCommit(FlexlbConfig config) {
        return !"versioned".equalsIgnoreCase(config.getAutoTpmCommitStrategy());
    }

    /** {@code autoTpmVictimGuardMode}: any value but the legacy "queue_version" is presence. */
    private static boolean isVictimPresenceGuard(FlexlbConfig config) {
        return !"queue_version".equalsIgnoreCase(config.getAutoTpmVictimGuardMode());
    }

    /**
     * N1/P2-2: a victim settle (finishYielded/PreemptedById) that found no
     * inflight entry — harmless in isolation, but a burst points at a
     * registration/cleanup race, so it is surfaced as a metric, not only a
     * warn log.
     */
    public void onInflightSettleMiss(String kind) {
        priorityReporter.reportInflightSettleMiss(kind);
    }

    /**
     * Eviction replan backoff: full jitter in [10, 30] ms (N3 §3.6), damping
     * planning storms over the same shifting victim set. The replan count is
     * bounded by the dedicated eviction-replan budget (P2-1: sized like
     * {@link #MAX_PLAN_RETRIES}, spent independently of capacity retries).
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
        if (future.isDone()) {
            return;
        }
        Response errorResp = Response.error(errorType, reason);
        errorResp.setErrorMessage(errorType.buildErrorMessage(message));
        future.complete(errorResp);
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
