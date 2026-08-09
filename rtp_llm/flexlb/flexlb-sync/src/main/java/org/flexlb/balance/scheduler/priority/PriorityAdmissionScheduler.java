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
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

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
 * When no placement is feasible or retries are exhausted, the request fails
 * with {@link StrategyErrorType#NO_AVAILABLE_WORKER}. Decode reserved-only
 * eviction is a later phase.
 *
 * <p><b>Auto-TPM switch matrix</b> (all default off; each row gates one
 * behavior entry point, rows below the first also require the first):
 * <pre>
 * | Switch (FlexlbConfig)                       | Gated behavior entry                                        |
 * |---------------------------------------------|-------------------------------------------------------------|
 * | autoTpmEnabled                              | FlexlbBatchScheduler.submit() routes into schedule() at all |
 * |                                             | (off = legacy path; also a precondition for every row below)|
 * | autoTpmPrefillQueueEvictEnabled             | Phase 3 tryPrefillQueueEviction() on OFFER_FAILED           |
 * | autoTpmDecodeReservedEvictEnabled           | Phase 4 tryDecodeEviction() on decode-capacity route failure|
 * | autoTpmDecodeAcceptedEvictEnabled           | Phase 5 accepted-layer victims join the decode plan pool    |
 * |   + cancelChannel.isSupported(endpoint)     | (both required, per endpoint — EvictionPlanner gate) and    |
 * |                                             | commitAcceptedEviction() cancel-wait-confirm path           |
 * | (removed) DeadlineRescueService            | PR-D replaced by AdmissionLease + orTimeout fail-fast        |
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

    public PriorityAdmissionScheduler(ConfigService configService,
                                      Router router,
                                      EndpointRegistry endpointRegistry,
                                      PlanCommitter planCommitter,
                                      PrioritySloPolicy sloPolicy,
                                      PrioritySchedulerReporter priorityReporter,
                                      BatchSchedulerReporter batchReporter,
                                      EngineCancelChannel cancelChannel) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.planCommitter = planCommitter;
        this.sloPolicy = sloPolicy;
        this.priorityReporter = priorityReporter;
        this.batchReporter = batchReporter;
        this.cancelChannel = cancelChannel;
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

        // Task40 change 2: reject requests whose SLO deadline has already
        // expired (remaining <= 0) instead of leaving them in the queue.
        // Only the auto-tpm path rejects; deadline rescue is unaffected —
        // it migrates danger-zone requests that still have remaining time.
        long deadlineMs = ctx.getDeadlineMs();
        if (deadlineMs > 0) {
            long nowMs = System.currentTimeMillis();
            if (deadlineMs <= nowMs) {
                Logger.info("[auto-tpm] slo deadline exceeded, reject request_id={} deadline_ms={} now_ms={}",
                        ctx.getRequestId(), deadlineMs, nowMs);
                completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
                        "slo deadline exceeded: deadline_ms=" + deadlineMs + " now_ms=" + nowMs
                                + " reason=slo_deadline_exceeded");
                return;
            }
        }

        // Tracks whether every failed attempt was an optimistic-concurrency
        // conflict (VERSION_MISMATCH / eviction CONFLICT). Any capacity-rooted
        // failure (OFFER_FAILED) keeps exhaustion on NO_AVAILABLE_WORKER;
        // pure-conflict exhaustion maps to SCHEDULER_PLAN_CONFLICT (§16.3).
        boolean allConflicts = true;
        // Reason tag of the most recent capacity-rooted failure, appended to
        // the exhaustion message for 8400 attribution (redesign B-1).
        String lastFailureReason = null;
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
                // Phase 4 (gated, default off): the route failed specifically
                // for decode capacity — try a decode reserved-only eviction.
                // This method is only reached via the Auto-TPM priority path,
                // so every request here already carries a normalized priority.
                if (config.isAutoTpmDecodeReservedEvictEnabled()
                        && isDecodeCapacityFailure(outcome.failureResponse)) {
                    DecodeEvictionOutcome eviction =
                            tryDecodeEviction(ctx, future, snapshot, config, registrar);
                    if (eviction == DecodeEvictionOutcome.CONFLICT) {
                        Logger.info("[auto-tpm] decode eviction conflict (attempt {}/{}), request_id={}",
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
                            completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
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
                        Logger.info("[auto-tpm] no feasible eviction plan (attempt {}/{}), request_id={} priority={}",
                                attempt, maxRetries, ctx.getRequestId(), ctx.getPriority());
                        continue;
                    }
                    // COMMITTED, or FAILED with the future already completed.
                    return;
                }
                onInfeasible(ctx, future, outcome.failureResponse);
                return;
            }

            // P1-1: flip the reservation into the queued phase BEFORE the
            // commit can publish the item to the batcher — marking after the
            // commit (in onCommitted) races the dispatch side's
            // markDispatchedPhase and can leave a stale queued mark that hides
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
                        Logger.info("[auto-tpm] no feasible eviction plan (attempt {}/{}), request_id={} priority={}",
                                attempt, maxRetries, ctx.getRequestId(), ctx.getPriority());
                        continue;
                    }
                    case PARTIAL_FAILURE -> {
                        releaseDecodeReservation(outcome.plan);
                        completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
                                "eviction commit partial failure");
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
                            completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
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
            Logger.info("[auto-tpm] plan commit {} (attempt {}/{}), request_id={}",
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
                completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
                        "prefill offer failed after fallback, reason=prefill_queue_full");
                return;
            }
        }

        completeError(future,
                allConflicts ? StrategyErrorType.SCHEDULER_PLAN_CONFLICT
                        : StrategyErrorType.NO_AVAILABLE_WORKER,
                lastFailureReason != null
                        ? "auto-tpm plan retries exhausted, reason=" + lastFailureReason
                        : "auto-tpm plan retries exhausted");
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
        PARTIAL_FAILURE
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
            Logger.info("[auto-tpm] eviction plan infeasible, request_id={} priority={} "
                            + "phase=prefill_queue candidates_seen={} reasons={}",
                    envelope.requestId(), envelope.priority(),
                    queueSnapshot.items().size(), failures);
            return EvictionOutcome.INFEASIBLE;
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
                Logger.info("[auto-tpm] eviction victims gone, direct offer succeeded: "
                                + "request_id={} missing_victims={} worker={}",
                        envelope.requestId(), replace.missingVictimIds(), proposal.endpointId());
                return EvictionOutcome.COMMITTED;
            }
            registrar.unregisterInflight(item);
            Logger.info("[auto-tpm] eviction victims gone, replan: request_id={} "
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
            Logger.info("[auto-tpm] victim preempted: victim_id={} victim_priority={} "
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
        Logger.info("[auto-tpm] eviction committed: request_id={} priority={} victims={} "
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

    /**
     * Metrics hook for an accepted-eviction victim settled by the later
     * WorkerStatus CANCELLED report (outside the commit wait window): counts
     * the late release confirmation. Called by the batch scheduler's
     * attribution path (Phase 5).
     */
    public void onAcceptedPreemptSettled(String endpoint) {
        priorityReporter.reportCancelConfirm(endpoint);
        priorityReporter.reportPriorityPreempt("decode_accepted");
    }

    // ==================== Phase 4: decode reserved-only eviction ====================

    private enum DecodeEvictionOutcome {
        /** Eviction applied and the incoming request committed. */
        COMMITTED,
        /** No feasible plan (no/insufficient strictly-lower-priority reserved victims). */
        INFEASIBLE,
        /** Optimistic-concurrency conflict — retry with a fresh plan. */
        CONFLICT,
        /** Eviction applied but placement failed; future already completed. */
        FAILED
    }

    /**
     * Plan and atomically commit a decode reserved-only eviction (design doc
     * 11-13, 17.2), then place the incoming request on the freed endpoint.
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
        // Planning envelope: the decode planners consume only priority and
        // hardKvTokens; the endpoint-aware deadline is rebuilt after placement.
        PriorityRequestEnvelope planEnvelope = new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), ctx.getRequestSloMs(), ctx.getDeadlineMs(),
                seqLen, seqLen + maxNewTokens);

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
            for (DecodeEndpointSnapshot ep : decodes) {
                for (DecodeRequestSnapshot candidate : ep.reserved()) {
                    candidatesSeen++;
                    if (PriorityNormalizer.hasPriority(candidate.priority())
                            && candidate.priority() < planEnvelope.priority()) {
                        candidatesEligible++;
                    }
                }
            }
            Logger.info("[auto-tpm] decode eviction plan infeasible, request_id={} priority={} "
                            + "phase=decode_reserved candidates_seen={} candidates_eligible={} reasons={}",
                    ctx.getRequestId(), ctx.getPriority(), candidatesSeen, candidatesEligible, failures);
            return DecodeEvictionOutcome.INFEASIBLE;
        }
        priorityReporter.reportEvictionPlan(ctx.getPriority(), proposal.evictionCase(), "feasible");

        DecodeEndpointSnapshot target = snapshot.decodes().get(proposal.endpointId());
        DecodeEndpoint decodeEp = target.endpoint();
        long expectedKvTokens = target.realKvTotal() > 0
                ? Math.min(seqLen + maxNewTokens, target.realKvTotal())
                : seqLen + maxNewTokens;

        // Phase 5: accepted-layer victims need the cancel-wait-confirm commit;
        // a reserved-only plan keeps the Phase 4 atomic path byte-for-byte.
        List<DecodeRequestSnapshot> acceptedVictims = new ArrayList<>();
        List<Long> reservedVictimIds = new ArrayList<>(proposal.victims().size());
        for (DecodeRequestSnapshot victim : proposal.victims()) {
            if (victim.phase() == DecodeTaskPhase.ACCEPTED_NOT_RUNNING) {
                acceptedVictims.add(victim);
            } else {
                reservedVictimIds.add(victim.requestId());
            }
        }
        if (!acceptedVictims.isEmpty()) {
            return commitAcceptedEviction(ctx, future, config, registrar, proposal,
                    decodeEp, reservedVictimIds, acceptedVictims, seqLen, expectedKvTokens);
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
                Logger.info("[auto-tpm] decode eviction victims gone, replan: request_id={} "
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
        boolean accepted = victim.phase() == DecodeTaskPhase.ACCEPTED_NOT_RUNNING;
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
        Logger.info("[auto-tpm] decode victim preempted: victim_id={} victim_priority={} "
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
        Logger.info("[auto-tpm] decode eviction committed: request_id={} priority={} case={} "
                        + "victims={} total_cost={} freed_kv={} worker={}",
                ctx.getRequestId(), ctx.getPriority(), proposal.evictionCase(),
                proposal.victims().size(), proposal.totalCost(), proposal.freedKvTokens(),
                proposal.endpointId());
    }

    /**
     * Phase 5 accepted-eviction commit (cancel-wait-confirm):
     * <ol>
     *   <li>Atomically release the reserved victims and mark the accepted
     *       victims CANCEL_REQUESTED ({@code tryBeginAcceptedEviction}); the
     *       incoming request is NOT reserved yet</li>
     *   <li>Issue the engine cancel (PRIORITY_PREEMPTED) per accepted victim</li>
     *   <li>Wait up to {@code autoTpmCommitWaitReleaseTimeoutMs} for each
     *       release to be confirmed — i.e. the next WorkerStatus calibration
     *       drops the request from the confirmed layer (not-found /
     *       already-finished cancel outcomes count as released immediately)</li>
     *   <li>All confirmed → reserve the incoming, drive the victims terminal
     *       and place the request; any timeout / unsupported outcome → the
     *       plan fails without touching the incoming (iron rule 4: never
     *       assume a cancel released anything). Unconfirmed victims keep their
     *       CANCEL_REQUESTED mark and are settled by the later WorkerStatus
     *       CANCELLED report</li>
     * </ol>
     * The bounded wait (default 50ms) runs on the scheduling thread — an
     * accepted eviction is a rare, gated slow path and the spec explicitly
     * allows a short timed wait here (no unbounded spinning).
     */
    private DecodeEvictionOutcome commitAcceptedEviction(BalanceContext ctx,
                                                         CompletableFuture<Response> future,
                                                         FlexlbConfig config,
                                                         InflightRegistrar registrar,
                                                         DecodeEvictionProposal proposal,
                                                         DecodeEndpoint decodeEp,
                                                         List<Long> reservedVictimIds,
                                                         List<DecodeRequestSnapshot> acceptedVictims,
                                                         long seqLen,
                                                         long expectedKvTokens) {
        String endpointKey = proposal.endpointId();
        List<Long> acceptedVictimIds = new ArrayList<>(acceptedVictims.size());
        for (DecodeRequestSnapshot victim : acceptedVictims) {
            acceptedVictimIds.add(victim.requestId());
        }

        DecodeEndpoint.ReleaseReserveResult begin = isVictimPresenceGuard(config)
                // N3 §3.4: same all-or-nothing victim validation, no version check.
                ? decodeEp.tryBeginAcceptedEvictionPresent(reservedVictimIds, acceptedVictimIds)
                : decodeEp.tryBeginAcceptedEviction(
                        reservedVictimIds, acceptedVictimIds, proposal.admissionVersion());
        if (begin != DecodeEndpoint.ReleaseReserveResult.SUCCESS) {
            if (!isVictimPresenceGuard(config)) {
                // Version-conflict metric only exists on the legacy guard;
                // the presence guard aborts solely on gone victims (§3.8
                // plan_conflict → victim_gone migration).
                priorityReporter.reportPlanConflict("decode_admission_version");
            }
            priorityReporter.reportEvictionCommit(ctx.getPriority(), proposal.evictionCase(),
                    begin == DecodeEndpoint.ReleaseReserveResult.VICTIM_GONE
                            ? "victim_gone" : "version_mismatch");
            return DecodeEvictionOutcome.CONFLICT;
        }

        // Reserved victims: shadow accounting already reversed — terminal now.
        for (DecodeRequestSnapshot victim : proposal.victims()) {
            if (victim.phase() != DecodeTaskPhase.ACCEPTED_NOT_RUNNING) {
                finishDecodeVictim(ctx, registrar, victim, "decode_reserved", proposal);
            }
        }

        // Accepted victims: mark for CANCELLED attribution, then inject the
        // cancel intent. The mark rides the victim's inflight entry so a
        // post-timeout WorkerStatus CANCELLED still maps to 8429.
        String detail = "preempted by higher-priority request " + ctx.getRequestId();
        List<CompletableFuture<EngineCancelChannel.CancelOutcome>> cancels =
                new ArrayList<>(acceptedVictims.size());
        for (DecodeRequestSnapshot victim : acceptedVictims) {
            registrar.markCancelRequested(victim.requestId(), detail);
            priorityReporter.reportCancelRequest(endpointKey);
            cancels.add(cancelChannel.cancel(decodeEp, victim.requestId(),
                    EngineCancelChannel.CancelReason.PRIORITY_PREEMPTED));
        }

        String failReason = awaitCancelReleases(config, decodeEp, acceptedVictims, cancels);
        if (failReason != null) {
            priorityReporter.reportCancelTimeout(endpointKey);
            priorityReporter.reportEvictionCommit(ctx.getPriority(), proposal.evictionCase(),
                    failReason);
            Logger.warn("[auto-tpm] accepted eviction {}: request_id={} victims={} worker={}",
                    failReason, ctx.getRequestId(), acceptedVictimIds, endpointKey);
            completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
                    "accepted eviction " + failReason);
            return DecodeEvictionOutcome.FAILED;
        }

        // Every release confirmed — only now may the incoming take the freed
        // capacity (iron rule 4).
        decodeEp.reserve(ctx.getRequestId(), seqLen, expectedKvTokens,
                ctx.getPriority(), ctx.getDeadlineMs());
        for (DecodeRequestSnapshot victim : acceptedVictims) {
            registrar.finishPreemptedById(victim.requestId(), detail);
            priorityReporter.reportVictim(victim.priority(), ctx.getPriority(),
                    "decode_accepted", proposal.evictionCase());
            priorityReporter.reportPriorityPreempt("decode_accepted");
            priorityReporter.reportVictimKvTokens(victim.priority(), "decode_accepted",
                    victim.kvTokens());
            priorityReporter.reportCancelConfirm(endpointKey);
            Logger.info("[auto-tpm] decode victim preempted: victim_id={} victim_priority={} "
                            + "stage=decode_accepted terminal=preempted_8429 kv_tokens={} "
                            + "incoming_id={} incoming_priority={} worker={}",
                    victim.requestId(), victim.priority(), victim.kvTokens(),
                    ctx.getRequestId(), ctx.getPriority(), endpointKey);
        }
        priorityReporter.reportEvictionCommit(ctx.getPriority(), proposal.evictionCase(), "success");
        recordDecodePlanObservability(ctx, proposal);

        return placeAfterDecodeEviction(ctx, future, config, registrar, decodeEp);
    }

    /**
     * Bounded wait for every accepted victim's release confirmation.
     * A victim counts as released when its cancel outcome is not-found /
     * already-finished, or when calibrate drops it from the confirmed layer
     * ({@code isConfirmedTracked} turns false) within the window.
     *
     * <p>Design semantics of a non-null return (iron rule 4: a cancel timeout
     * never assumes the resource was released): the plan fails, but some
     * accepted victims may already carry the injected cancel intent and their
     * CANCEL_REQUESTED mark. Those victims are NOT rolled back here — whether
     * each one actually releases is settled later by the WorkerStatus
     * calibration path (CANCELLED report → late confirm + 8429 attribution).
     *
     * @return {@code null} when all victims are confirmed released, otherwise
     *         the eviction-commit failure label
     */
    private static String awaitCancelReleases(FlexlbConfig config,
                                              DecodeEndpoint decodeEp,
                                              List<DecodeRequestSnapshot> acceptedVictims,
                                              List<CompletableFuture<EngineCancelChannel.CancelOutcome>> cancels) {
        long timeoutMs = Math.max(1, config.getAutoTpmCommitWaitReleaseTimeoutMs());
        long deadline = System.currentTimeMillis() + timeoutMs;
        for (int i = 0; i < acceptedVictims.size(); i++) {
            long victimId = acceptedVictims.get(i).requestId();
            EngineCancelChannel.CancelOutcome outcome;
            try {
                long remaining = deadline - System.currentTimeMillis();
                outcome = cancels.get(i).get(Math.max(1, remaining), TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                // Plan fails; already-cancelled victims stay CANCEL_REQUESTED
                // and are settled by the later WorkerStatus report.
                return "cancel_timeout";
            } catch (Exception e) {
                Logger.warn("[auto-tpm] cancel rpc failed for victim_id={}: {}",
                        victimId, e.getMessage());
                // Same semantics: no release is assumed for this victim or
                // any earlier one whose cancel intent was already injected.
                return "cancel_timeout";
            }
            if (outcome.unsupported()) {
                // Planning-gate violation — must never pick accepted victims
                // on an endpoint without the Cancel RPC. The plan fails, yet
                // victims already sent a cancel keep their intent; settle is
                // still owned by WorkerStatus (no release assumed).
                return "cancel_unsupported";
            }
            if (!outcome.found() || outcome.alreadyFinished()) {
                // The engine no longer owns the request — released already.
                continue;
            }
            if (!pollReleased(decodeEp, victimId, deadline)) {
                // Confirmed layer still tracks the victim at the deadline:
                // fail the plan without assuming the release ever happens;
                // the victim stays CANCEL_REQUESTED until WorkerStatus
                // settles it (design doc iron rule 4).
                return "cancel_timeout";
            }
        }
        return null;
    }

    /** Poll the confirmed layer until the victim disappears or the deadline hits. */
    private static boolean pollReleased(DecodeEndpoint decodeEp, long victimId, long deadline) {
        while (decodeEp.isConfirmedTracked(victimId)) {
            if (System.currentTimeMillis() >= deadline) {
                return false;
            }
            try {
                Thread.sleep(2);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return true;
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
            completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
                    "no prefill worker after decode eviction");
            return DecodeEvictionOutcome.FAILED;
        }
        PrefillEndpoint prefillEp = endpointRegistry.getPrefill(
                prefill.getServerIp() + ":" + prefill.getHttpPort());
        if (prefillEp == null) {
            decodeEp.release(ctx.getRequestId());
            completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
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
                case INFEASIBLE, PARTIAL_FAILURE -> {
                    // fall through to the explicit failure below
                }
            }
        }
        releaseDecodeReservation(plan);
        completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
                "prefill offer failed after decode eviction");
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
                ? Math.min(seqLen + maxNewTokens, kvTotal)
                : seqLen + maxNewTokens;
        return new PriorityRequestEnvelope(
                ctx.getRequestId(), ctx.getPriority(), seqLen, maxNewTokens,
                ctx.getStartTime(), requestSloMs, deadlineMs, seqLen, expectedKvTokens);
    }

    // ==================== Outcome handling ====================

    /**
     * Create an {@link AdmissionLease} and bind it to the request future
     * (PR-D §2.4). Called at every plan-commit success point after
     * {@link #onCommitted}. The lease is the single ownership boundary:
     * success → {@code handoverToEngine} (seal, no resource release);
     * failure/timeout → {@code close} (tryRemove + release + unregister).
     * <p>Legacy path ({@code budget == null}) never creates a lease.
     */
    private static void bindAdmissionLease(NormalPlacementPlan plan,
                                            InflightRegistrar registrar) {
        BalanceContext ctx = plan.item().ctx();
        if (ctx.budget() == null) {
            return;
        }
        PrefillQueueManager prefillQueue = plan.prefillEp().getBatcher().queueManager();
        AdmissionLease lease = new AdmissionLease(
                plan.item(), plan.decodeEp(), prefillQueue, registrar);
        lease.bindTo(plan.item().future());
    }

    private void onCommitted(BalanceContext ctx, NormalPlacementPlan plan) {
        ctx.setRouteSubmittedNanos(System.nanoTime());
        // N3 §3.8: plan age quantifies how stale the committed plan view was
        // (snapshot/build → successful commit).
        priorityReporter.reportPlanAge(plan.envelope().priority(),
                Math.max(0, System.currentTimeMillis() - plan.createdAtMs()));
        // N2/P1-1: the queued-phase mark is set BEFORE the commit (schedule /
        // placeAfterDecodeEviction) — marking here raced the dispatch side's
        // markDispatchedPhase for items that dispatched immediately.
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
            future.complete(failureResponse);
            return;
        }
        Logger.info("[auto-tpm] no feasible placement, request_id={} priority={}",
                ctx.getRequestId(), ctx.getPriority());
        completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER, null);
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

    private static void completeError(CompletableFuture<Response> future,
                                      StrategyErrorType errorType,
                                      String message) {
        if (future.isDone()) {
            return;
        }
        Response errorResp = Response.error(errorType);
        errorResp.setErrorMessage(message == null ? errorType.getErrorMsg() : message);
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
