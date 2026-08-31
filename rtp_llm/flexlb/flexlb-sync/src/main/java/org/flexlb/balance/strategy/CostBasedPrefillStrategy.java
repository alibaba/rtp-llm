package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Prefill worker choice from feat/dsv4_on_dev, adapted only at the ownership
 * boundary required by the current router.
 */
@Component
public class CostBasedPrefillStrategy implements LoadBalanceStrategy {

    private final WorkerDirectory workerDirectory;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;

    public CostBasedPrefillStrategy(
            WorkerDirectory workerDirectory,
            CacheAwareService cacheAwareService,
            ResourceMeasureFactory resourceMeasureFactory,
            EngineHealthReporter engineHealthReporter) {
        this.workerDirectory = workerDirectory;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
    }

    @Override
    public boolean supports(
            RoleType role,
            RoutingConfig.EndpointSelectorConfig configured) {
        return (role == RoleType.PREFILL || role == RoleType.PDFUSION)
                && configured
                instanceof RoutingConfig.EstimatedTtftSelectorConfig estimated
                && !(estimated.getCandidateChoice()
                        instanceof RoutingConfig.LeastRecentlyUsedInPoolConfig);
    }

    @Override
    public SelectedRole select(
            BalanceContext context,
            RoleType role,
            String group) {
        try {
            return doSelect(context, role, group);
        } catch (RuntimeException failure) {
            Logger.warn(
                    "Cost-based Prefill selection failed: request_id={}",
                    context == null ? null : context.getRequestId(),
                    failure);
            return null;
        }
    }

    private SelectedRole doSelect(
            BalanceContext context,
            RoleType role,
            String group) {
        FlexlbConfig config = context.getConfig();
        PrefillResourceMeasure measure = (PrefillResourceMeasure)
                resourceMeasureFactory.getMeasure(
                        config.resourceMeasureFor(role));
        if (measure == null) {
            return null;
        }

        Map<String, Integer> cacheMatches = cacheAwareService
                .findMatchingEngines(
                        context.getRequest().getBlockCacheKeys(),
                        role,
                        group);
        List<Candidate> candidates = captureAndScore(
                context, role, group, measure, cacheMatches);
        try {
            if (candidates.isEmpty()) {
                Logger.debug(
                        "Prefill select failed: no available endpoint,"
                                + " request_id={}",
                        context.getRequestId());
                return null;
            }

            List<Candidate> survivors =
                    rejectOutliers(candidates, config);
            long minimumScore = Long.MAX_VALUE;
            for (Candidate candidate : survivors) {
                minimumScore = Math.min(
                        minimumScore, candidate.scoreMs);
            }
            int selectedIndex = selectBestCandidate(
                    new CandidateSet(survivors),
                    minimumScore,
                    context,
                    role,
                    group,
                    context.getRequest().getSeqLen(),
                    config);
            if (selectedIndex < 0) {
                return null;
            }
            Candidate selected = survivors.get(selectedIndex);
            reportSelection(
                    role,
                    survivors,
                    selected,
                    context.getRequest().getSeqLen(),
                    config);
            return selected.takeSelection(role, context);
        } finally {
            closeCandidates(candidates);
        }
    }

    /**
     * The current endpoint runtime supplies a coherent snapshot; the policy
     * still makes the old decision: score only available workers and select
     * from that fixed candidate set once.
     */
    private List<Candidate> captureAndScore(
            BalanceContext context,
            RoleType role,
            String group,
            PrefillResourceMeasure measure,
            Map<String, Integer> cacheMatches) {
        List<WorkerEndpoint.GenerationPin> pins =
                workerDirectory.captureEndpoints(role, group);
        List<Candidate> candidates = new ArrayList<>(pins.size());
        WorkerEndpoint.GenerationPin excluded = null;
        String excludedAddress = context.getExcludedPrefillIpPort();
        try {
            for (int index = 0; index < pins.size(); index++) {
                WorkerEndpoint.GenerationPin pin = pins.get(index);
                if (!(pin.endpoint() instanceof PrefillEndpoint)) {
                    pin.close();
                    pins.set(index, null);
                    continue;
                }
                if (excludedAddress != null
                        && excludedAddress.equals(pin.endpoint().ipPort())) {
                    excluded = pin;
                    pins.set(index, null);
                    continue;
                }
                Candidate candidate = scoreCandidate(
                        pin, context, measure, cacheMatches);
                pins.set(index, null);
                if (candidate != null) {
                    candidates.add(candidate);
                }
            }
            if (candidates.isEmpty() && excluded != null) {
                Candidate fallback = scoreCandidate(
                        excluded, context, measure, cacheMatches);
                excluded = null;
                if (fallback != null) {
                    candidates.add(fallback);
                }
            }
            return candidates;
        } catch (Throwable failure) {
            closeCandidates(candidates);
            throw failure;
        } finally {
            if (excluded != null) {
                excluded.close();
            }
            for (WorkerEndpoint.GenerationPin pin : pins) {
                if (pin != null) {
                    pin.close();
                }
            }
        }
    }

    /**
     * One candidate is evaluated exactly once. Projection is only the current
     * runtime's coherent wait-time reader; it does not add a second selection,
     * fallback, blocker, or retry policy.
     */
    private Candidate scoreCandidate(
            WorkerEndpoint.GenerationPin pin,
            BalanceContext context,
            PrefillResourceMeasure measure,
            Map<String, Integer> cacheMatches) {
        try {
            PrefillEndpoint endpoint =
                    (PrefillEndpoint) pin.endpoint();
            RouteProjection.Inputs inputs =
                    endpoint.captureRouteProjectionInputs();
            long pending = inputs.pendingRequestCount();
            if (!measure.isResourceAvailable(pending)) {
                return null;
            }

            CacheTokenMatch cacheMatch = calculateCacheMatch(
                    endpoint, cacheMatches, context.getRequest());
            long cacheHit = cacheMatch.effectiveHitTokens();
            PrefillTimePredictor predictor = endpoint.getPredictor();
            if (predictor == null) {
                return null;
            }
            RouteProjection.Probe probe =
                    new RouteProjection.Probe(
                            context.getRequestId(),
                            context.getPriority(),
                            inputs.queue().capturedAtMs(),
                            context.getRequestExpiresAtMs(),
                            context.getRequest().getSeqLen(),
                            cacheHit,
                            cacheHit,
                            RouteProjection.Demand.TTFT_ONLY);
            RouteProjection.Candidate projection =
                    RouteProjection.project(
                            withoutAdmissionBlock(inputs),
                            probe,
                            predictor.evaluator(),
                            endpoint.deliveryProjection());
            if (!projection.projection().selectable()) {
                return null;
            }
            OptionalLong committedWait =
                    inputs.work().totalRemainingWorkMs();
            if (committedWait.isEmpty()) {
                return null;
            }

            Candidate candidate = new Candidate(
                    pin,
                    endpoint,
                    projection.projectedTtftMs(),
                    projection.incomingPrefillMs(),
                    cacheHit,
                    cacheMatch.routingHitTokens(),
                    committedWait.getAsLong(),
                    pending,
                    endpoint.getLastSelectedTime().get());
            pin = null;
            return candidate;
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }

    /**
     * The legacy worker chooser scores Prefill load only. A delivery-time
     * Decode blocker belongs to the admission transaction after selection;
     * letting it remove this Prefill candidate would park the request on the
     * wrong role and miss the later Decode capacity signal.
     */
    static RouteProjection.Inputs withoutAdmissionBlock(
            RouteProjection.Inputs inputs) {
        QueueSnapshot queue = inputs.queue();
        if (queue.admissionBlock() == null) {
            return inputs;
        }
        return new RouteProjection.Inputs(
                new QueueSnapshot(
                        queue.capturedAtMs(),
                        queue.queueScheduling(),
                        queue.ordering(),
                        queue.constraints(),
                        queue.activeItems(),
                        null),
                inputs.work(),
                inputs.pendingRequestCount());
    }

    private static List<Candidate> rejectOutliers(
            List<Candidate> candidates,
            FlexlbConfig config) {
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig)
                        config.getRouter().getRoles()
                                .getPrefill().getSelector();
        RoutingConfig.OutlierRejectionConfig outlier =
                outlierRejection(selector.getCandidateChoice());
        if (outlier == null || candidates.size() <= 1) {
            return candidates;
        }

        double pendingMultiplier =
                outlier.getMaxPendingVsAverageMultiplier();
        double waitMultiplier =
                outlier.getMaxProjectedDrainVsAverageMultiplier();
        if (pendingMultiplier <= 0.0 && waitMultiplier <= 0.0) {
            return candidates;
        }

        long totalPending = 0L;
        long totalWait = 0L;
        Candidate leastLoaded = null;
        for (Candidate candidate : candidates) {
            totalPending = saturatingAdd(
                    totalPending, candidate.pendingCount);
            totalWait = saturatingAdd(
                    totalWait, candidate.committedWaitMs);
            if (leastLoaded == null
                    || candidate.committedWaitMs
                            < leastLoaded.committedWaitMs) {
                leastLoaded = candidate;
            }
        }
        long averagePending = totalPending / candidates.size();
        long averageWait = totalWait / candidates.size();

        List<Candidate> survivors =
                new ArrayList<>(candidates.size());
        for (Candidate candidate : candidates) {
            if (pendingMultiplier > 0.0
                    && averagePending > 0L
                    && candidate.pendingCount
                            > averagePending * pendingMultiplier) {
                continue;
            }
            if (waitMultiplier > 0.0
                    && averageWait > 0L
                    && candidate.committedWaitMs
                            > averageWait * waitMultiplier) {
                continue;
            }
            survivors.add(candidate);
        }
        if (survivors.isEmpty() && leastLoaded != null) {
            survivors.add(leastLoaded);
        }
        return survivors;
    }

    protected int selectBestCandidate(
            CandidateSet survivors,
            long minimumScore,
            BalanceContext context,
            RoleType role,
            String group,
            long sequenceLength,
            FlexlbConfig config) {
        RoutingConfig.CacheAffinityConfig affinityConfig =
                config.getRouter().getRoles()
                        .getPrefill().getCacheAffinity();
        if (affinityConfig == null) {
            return selectBaselineCandidate(
                    survivors, minimumScore, config);
        }

        long referenceHit = 0L;
        for (int index = 0; index < survivors.size(); index++) {
            if (survivors.scoreMs(index) == minimumScore) {
                referenceHit = Math.max(
                        referenceHit,
                        survivors.cacheHit(index));
            }
        }
        CacheAffinityPolicy.Decision affinity =
                CacheAffinityPolicy.evaluate(
                        survivors.size(),
                        survivors::scoreMs,
                        survivors::cacheHit,
                        minimumScore,
                        referenceHit,
                        sequenceLength,
                        affinityConfig.getMaxExtraTtftMs(),
                        affinityConfig.getMinPrefixHitPercent());

        int selected = affinity.hasPreference()
                ? selectCacheLeader(survivors, affinity)
                : selectBaselineCandidate(
                        survivors, minimumScore, config);
        if (selected >= 0) {
            reportCacheAffinityDecision(
                    role,
                    survivors.endpoint(selected).getIp(),
                    affinity.reason().name());
        }
        return selected;
    }

    private static int selectBaselineCandidate(
            CandidateSet candidates,
            long minimumScore,
            FlexlbConfig config) {
        long tolerance = 0L;
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig)
                        config.getRouter().getRoles()
                                .getPrefill().getSelector();
        if (selector.getCandidateChoice()
                instanceof RoutingConfig.RandomWithinToleranceConfig random) {
            long relative = (long)
                    (minimumScore * random.getRelativeTolerance());
            tolerance = Math.max(
                    Math.max(0L, relative),
                    Math.max(0L, random.getMinimumToleranceMs()));
        }
        long cutoff = saturatingAdd(minimumScore, tolerance);
        int selected = -1;
        int seen = 0;
        for (int index = 0; index < candidates.size(); index++) {
            if (candidates.scoreMs(index) <= cutoff
                    && ThreadLocalRandom.current()
                            .nextInt(++seen) == 0) {
                selected = index;
            }
        }
        return selected;
    }

    private static int selectCacheLeader(
            CandidateSet candidates,
            CacheAffinityPolicy.Decision affinity) {
        int selected = affinity.preferredIndex(0);
        long bestHit = candidates.cacheHit(selected);
        long bestScore = candidates.scoreMs(selected);
        int seen = 1;
        for (int index = 1;
                index < affinity.preferredCount();
                index++) {
            int candidate = affinity.preferredIndex(index);
            if (candidates.cacheHit(candidate) != bestHit
                    || candidates.scoreMs(candidate) != bestScore) {
                break;
            }
            if (ThreadLocalRandom.current()
                    .nextInt(++seen) == 0) {
                selected = candidate;
            }
        }
        return selected;
    }

    private static RoutingConfig.OutlierRejectionConfig
            outlierRejection(
            RoutingConfig.CandidateChoiceConfig choice) {
        if (choice
                instanceof RoutingConfig.RandomWithinToleranceConfig random) {
            return random.getOutlierRejection();
        }
        if (choice instanceof RoutingConfig.BestOnlyConfig best) {
            return best.getOutlierRejection();
        }
        return null;
    }

    private static CacheTokenMatch calculateCacheMatch(
            PrefillEndpoint endpoint,
            Map<String, Integer> cacheMatches,
            Request request) {
        if (cacheMatches == null
                || cacheMatches.isEmpty()
                || request == null
                || request.getSeqLen() <= 0L) {
            return CacheTokenMatch.NONE;
        }
        Integer matchedBlocks =
                cacheMatches.get(endpoint.ipPort());
        if (matchedBlocks == null || matchedBlocks <= 0) {
            return CacheTokenMatch.NONE;
        }
        long blockSize = request.getCacheKeyBlockSize();
        CacheStatus cacheStatus =
                endpoint.getStatus().getCacheStatus();
        if (blockSize <= 0L && cacheStatus != null) {
            blockSize = cacheStatus.getBlockSize();
        }
        if (blockSize <= 0L) {
            return CacheTokenMatch.NONE;
        }
        long hit;
        try {
            hit = Math.multiplyExact(
                    blockSize, matchedBlocks.longValue());
        } catch (ArithmeticException overflow) {
            hit = request.getSeqLen();
        }
        long routingHit = Math.min(
                request.getSeqLen(), Math.max(0L, hit));
        long effectiveHit = hit >= request.getSeqLen()
                ? Math.max(0L, request.getSeqLen() - blockSize)
                : routingHit;
        return new CacheTokenMatch(effectiveHit, routingHit);
    }

    private void reportSelection(
            RoleType role,
            List<Candidate> candidates,
            Candidate selected,
            long sequenceLength,
            FlexlbConfig config) {
        double hitRate = sequenceLength > 0L
                ? selected.cacheHit / (double) sequenceLength
                : 0.0;
        try {
            engineHealthReporter.reportCacheHitMetrics(
                    role, selected.cacheHit, hitRate);
            long candidateMaxRoutingHit = 0L;
            for (Candidate candidate : candidates) {
                candidateMaxRoutingHit = Math.max(
                        candidateMaxRoutingHit,
                        candidate.routingCacheMatchTokens);
            }
            engineHealthReporter.reportRoutingSelectedCacheMatchMetrics(
                    role,
                    selected.routingCacheMatchTokens,
                    sequenceLength);
            engineHealthReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                    role, candidateMaxRoutingHit);
            engineHealthReporter.reportPrefillSelectedEstimates(
                    role,
                    selected.endpoint.getIp(),
                    config.getDispatcher().typeName(),
                    selected.scoreMs,
                    selected.prefillMs);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Prefill selection metric failed: engine={}",
                    selected.endpoint.ipPort(),
                    telemetryFailure);
        }
    }

    protected void reportCacheAffinityDecision(
            RoleType role,
            String engineIp,
            String decision) {
        engineHealthReporter.reportCacheAffinityDecision(
                role, engineIp, decision);
    }

    protected static long saturatingAdd(
            long left,
            long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        return left + right;
    }

    private record CacheTokenMatch(
            long effectiveHitTokens,
            long routingHitTokens) {
        private static final CacheTokenMatch NONE =
                new CacheTokenMatch(0L, 0L);
    }

    private static void closeCandidates(
            List<Candidate> candidates) {
        for (Candidate candidate : candidates) {
            candidate.close();
        }
    }

    protected static final class CandidateSet {
        private final List<Candidate> candidates;

        private CandidateSet(List<Candidate> candidates) {
            this.candidates = candidates;
        }

        protected int size() {
            return candidates.size();
        }

        protected PrefillEndpoint endpoint(int index) {
            return candidates.get(index).endpoint;
        }

        protected String endpointAddress(int index) {
            return endpoint(index).ipPort();
        }

        protected long scoreMs(int index) {
            return candidates.get(index).scoreMs;
        }

        protected long cacheHit(int index) {
            return candidates.get(index).cacheHit;
        }

        protected long prefillMs(int index) {
            return candidates.get(index).prefillMs;
        }

        protected long lastSelectedTime(int index) {
            return candidates.get(index).lastSelectedTime;
        }
    }

    private static final class Candidate
            implements AutoCloseable {
        private WorkerEndpoint.GenerationPin pin;
        private final PrefillEndpoint endpoint;
        private final long scoreMs;
        private final long prefillMs;
        private final long cacheHit;
        private final long routingCacheMatchTokens;
        private final long committedWaitMs;
        private final long pendingCount;
        private final long lastSelectedTime;

        private Candidate(
                WorkerEndpoint.GenerationPin pin,
                PrefillEndpoint endpoint,
                long scoreMs,
                long prefillMs,
                long cacheHit,
                long routingCacheMatchTokens,
                long committedWaitMs,
                long pendingCount,
                long lastSelectedTime) {
            this.pin = pin;
            this.endpoint = endpoint;
            this.scoreMs = scoreMs;
            this.prefillMs = prefillMs;
            this.cacheHit = cacheHit;
            this.routingCacheMatchTokens =
                    routingCacheMatchTokens;
            this.committedWaitMs = committedWaitMs;
            this.pendingCount = pendingCount;
            this.lastSelectedTime = lastSelectedTime;
        }

        private SelectedRole takeSelection(
                RoleType role,
                BalanceContext context) {
            WorkerEndpoint.GenerationPin owned = pin;
            if (owned == null) {
                throw new IllegalStateException(
                        "Prefill candidate pin already consumed");
            }
            WorkerStatus.TopologySnapshot topology =
                    endpoint.getStatus().topologySnapshot();
            WorkerStatus.EngineObservation status =
                    endpoint.getStatus()
                            .committedEngineObservation();
            DebugInfo debug = new DebugInfo();
            debug.setHitCacheLen(cacheHit);
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(role);
            result.setRequestId(context.getRequestId());
            result.setPrefillTime(scoreMs);
            result.setGroup(topology.group());
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(
                    CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setDebugInfo(debug);
            pin = null;
            return SelectedRole.prefill(
                    owned, result, prefillMs);
        }

        @Override
        public void close() {
            WorkerEndpoint.GenerationPin owned = pin;
            pin = null;
            if (owned != null) {
                owned.close();
            }
        }
    }
}
