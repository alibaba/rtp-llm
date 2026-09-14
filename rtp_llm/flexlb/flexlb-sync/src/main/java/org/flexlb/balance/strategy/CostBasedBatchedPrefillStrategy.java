package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.batch.BatchCandidate;
import org.flexlb.balance.strategy.batch.BatchPlanner;
import org.flexlb.balance.strategy.batch.BatchPlanningRequest;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.GlobalDecisionConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Jointly places the Prefill requests collected in one global decision window.
 *
 * <p>This strategy is bound once at startup when
 * {@code scheduler.globalDecision.type=FIXED_WINDOW}. The global coordinator
 * owns the collection window and its maximum size; this class receives one
 * already eligible priority tier and decides its placements together.
 * Configuration requires the local decision type to be {@code SINGLE}, so a
 * worker-local batching window cannot change the meaning of this group.</p>
 *
 * <h2>Planning model</h2>
 *
 * <p>Each request contributes a snapshot of its live, ordinary-cost-policy
 * candidates. A candidate records its projected standalone TTFT, estimated
 * Prefill work, cache-hit tokens, delivery credits, and available KV capacity.
 * A plan is an input-order vector of candidate indexes, with {@code -1}
 * meaning unplaced. It is feasible only when every selected worker stays
 * within both its aggregate KV-token capacity and request-credit limit.</p>
 *
 * <p>Projected TTFT is virtual: selected requests assigned to the same worker
 * accrue the preceding selected requests' Prefill work in original submission
 * order. The optimizer first seeks a feasible placement for as much of the
 * group as possible, then minimizes aggregate virtual TTFT. When cache
 * affinity is enabled, a final pass may improve aggregate cache hits only if
 * every affected request remains within its TTFT allowance from the TTFT
 * baseline. The pure {@link BatchPlanner} performs the bounded planning
 * phases when greedy placement leaves requests unplaced.</p>
 *
 * <h2>Boundary and result semantics</h2>
 *
 * <p>Candidate discovery and planning only read snapshots; they do not reserve
 * engine capacity. The final materialization creates the ordinary selections
 * in input order. An infeasible request becomes a blocked result without
 * failing unrelated requests. If materialization fails, this invocation closes
 * every selection it has already created before rethrowing the failure.</p>
 *
 * <p>Global batch planning conventionally uses {@code BEST_ONLY}. This class
 * relies on configuration validation and does not repeat that check for every
 * request on the routing hot path.</p>
 */
public final class CostBasedBatchedPrefillStrategy extends PrefillStrategy {

    public CostBasedBatchedPrefillStrategy(WorkerDirectory workerDirectory,
                                           CacheAwareService cacheAwareService,
                                           EngineHealthReporter engineHealthReporter) {
        super(workerDirectory, cacheAwareService, engineHealthReporter);
    }

    /**
     * Retains the routing snapshot needed to materialize one planner result.
     *
     * <p>This is deliberately separate from the pure batch-planning model:
     * it contains endpoint objects, cache-match evidence, and blocker details
     * that only the strategy adapter needs after the plan is chosen.</p>
     */
    private record BatchCandidates(BatchRequest request,
                                   EndpointDiscovery discovery,
                                   PrefillCandidateSet candidates,
                                   CacheMatchResult cacheMatch,
                                   Map<String, Integer> rejections,
                                   RoleType blocker) {
    }

    /**
     * Plans and materializes placements for one global coordinator priority tier.
     *
     * <p>For a single request this preserves the ordinary {@link #selectOne}
     * path. Otherwise, the method snapshots candidates for every request,
     * delegates pure joint planning to {@link BatchPlanner}, then materializes
     * results in the same order as {@code requests}. Planning is side-effect
     * free; only materialization consumes normal delivery reservations.</p>
     *
     * @param requests requests collected by the global coordinator for one
     *                 AutoTPM-eligible priority tier
     * @return one result per input request in the same order; an infeasible
     * request is represented by a blocked result
     */
    @Override
    public List<PlacementResult<SelectedRole, RoleType>> selectBatch(List<BatchRequest> requests) {
        Objects.requireNonNull(requests, "requests");
        if (requests.size() == 1) {
            BatchRequest request = requests.getFirst();
            return List.of(selectOne(
                    request.context(), request.roleType(), request.group()));
        }
        List<BatchCandidates> snapshots = new ArrayList<>();
        List<BatchPlanningRequest> demands = new ArrayList<>();
        for (BatchRequest request : requests) {
            BalanceContext context = request.context();
            context.beginRoutingAttempt(request.roleType());
            EndpointDiscovery discovery = discoverAliveEndpoints(
                    request.roleType(), request.group());
            CacheMatchResult cacheMatch = getCacheMatchResult(
                    context, request.roleType(), request.group());
            Map<String, Integer> rejections = new HashMap<>();
            Map<RoleType, Integer> blockers = new EnumMap<>(RoleType.class);
            PrefillCandidateSet candidates = evaluateCandidates(
                    discovery,
                    context,
                    context.getConfig(),
                    cacheMatch,
                    rejections,
                    blockers,
                    new PrefillCandidateSet.Scratch());
            RoleType blocker = provenPoolWideBlocker(
                    blockers, discovery.registeredCount());
            snapshots.add(new BatchCandidates(
                    request,
                    discovery,
                    candidates,
                    cacheMatch,
                    rejections,
                    blocker == null ? request.roleType() : blocker));
            List<BatchCandidate> choices = new ArrayList<>();
            for (int i = 0; i < candidates.size() && candidates.selectable(i); i++) {
                choices.add(new BatchCandidate(
                        candidates.endpointAddress(i),
                        candidates.projectedTtftMs(i),
                        candidates.prefillMs(i),
                        candidates.cacheHit(i),
                        Math.max(0L, context.getRequest().getSeqLen()),
                        availableKvTokens(candidates.endpoint(i).getStatus()),
                        Math.max(1, candidates.endpoint(i).availableDeliveryCredits())));
            }
            RoutingConfig.CacheAffinityConfig affinity = context
                    .getConfig()
                    .getRouter()
                    .getRoles()
                    .getPrefill()
                    .getCacheAffinity();
            demands.add(new BatchPlanningRequest(
                    choices,
                    affinity == null
                            ? 0L : Math.max(0L, affinity.getMaxExtraTtftMs()),
                    affinity == null
                            ? 0.0 : normalizedHitRate(affinity.getMinPrefixHitPercent()),
                    context.getRequest().getSeqLen(),
                    affinity != null));
        }
        List<Integer> plan = BatchPlanner.plan(
                List.copyOf(demands), maxPlanEvaluations(requests));
        List<PlacementResult<SelectedRole, RoleType>> results = new ArrayList<>();
        Map<String, Long> precedingWork = new HashMap<>();
        try {
            for (int i = 0; i < snapshots.size(); i++) {
                BatchCandidates snapshot = snapshots.get(i);
                BatchRequest request = snapshot.request();
                PrefillCandidateSet candidates = snapshot.candidates();
                int selected = plan.get(i);
                boolean unmodeled = candidates.size() > 0
                        && !candidates.selectable(0);
                if (unmodeled) {
                    selected = selectUnmodeledCandidate(candidates);
                }
                if (selected < 0) {
                    request.context().recordSelectionReason(
                            request.roleType(), "BATCH_NO_AVAILABLE_CANDIDATE");
                    recordDecision(
                            request.context(),
                            request.roleType(),
                            request.group(),
                            snapshot.discovery().registeredCount(),
                            candidates,
                            -1,
                            "BATCH_NO_AVAILABLE_CANDIDATE",
                            snapshot.rejections());
                    results.add(PlacementResult.blocked(snapshot.blocker()));
                    continue;
                }
                OptionalLong ttft = OptionalLong.empty();
                if (!unmodeled) {
                    String worker = candidates.endpointAddress(selected);
                    long earlier = precedingWork.getOrDefault(worker, 0L);
                    ttft = OptionalLong.of(saturatingAdd(
                            candidates.projectedTtftMs(selected), earlier));
                    precedingWork.put(worker, saturatingAdd(
                            earlier, candidates.prefillMs(selected)));
                }
                String reason = unmodeled ? "UNMODELED_PENDING_LRU"
                        : demands.get(i).affinity()
                        ? "BATCH_BEST_ONLY/CACHE_AFFINITY"
                        : "BATCH_BEST_ONLY";
                request.context().recordSelectionReason(request.roleType(), reason);
                results.add(materialize(
                        request.context(),
                        request.roleType(),
                        request.group(),
                        snapshot.discovery(),
                        candidates,
                        snapshot.cacheMatch(),
                        snapshot.rejections(),
                        selected,
                        ttft));
            }
            return List.copyOf(results);
        } catch (RuntimeException | Error failure) {
            results.stream()
                    .filter(result -> result.value() != null)
                    .forEach(result -> result.value().close());
            throw failure;
        }
    }

    /**
     * Converts engine KV telemetry into the capacity constraint used by the planner.
     *
     * <p>This matches {@code WorkerBatcher}: absent or non-positive total KV
     * telemetry is treated as unbounded, while an observed total is capped by
     * its non-negative available value.</p>
     */
    private static long availableKvTokens(WorkerStatus workerStatus) {
        WorkerStatus.EngineObservation engine =
                workerStatus.committedEngineObservation();
        long total = engine.totalKvCacheTokens();
        if (total <= 0L) {
            return Long.MAX_VALUE;
        }
        return Math.min(total, Math.max(0L, engine.availableKvCacheTokens()));
    }

    /**
     * Reads the global planner budget from this frontier's configuration snapshot.
     *
     * <p>A global frontier never mixes request configuration snapshots. The
     * empty-list default makes this helper total; normal production frontiers
     * contain at least one request.</p>
     */
    private static int maxPlanEvaluations(List<BatchRequest> requests) {
        if (requests.isEmpty()) {
            return GlobalDecisionConfig.DEFAULT_MAX_PLAN_EVALUATIONS;
        }
        return requests.getFirst()
                .context()
                .getConfig()
                .queueScheduler()
                .getGlobalDecision()
                .getMaxPlanEvaluations();
    }
}
