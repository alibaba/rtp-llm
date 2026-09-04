package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
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

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class CostBasedPrefillStrategy {

    private static final ThreadLocal<CandidateScratch> CANDIDATE_SCRATCH =
            ThreadLocal.withInitial(CandidateScratch::new);

    private final WorkerDirectory workerDirectory;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final AtomicLong equalTtftCursor = new AtomicLong();
    private final AtomicLong equalCacheLeaderCursor = new AtomicLong();

    public CostBasedPrefillStrategy(WorkerDirectory workerDirectory,
                                    CacheAwareService cacheAwareService,
                                    EngineHealthReporter engineHealthReporter) {
        this.workerDirectory = workerDirectory;
        this.cacheAwareService = cacheAwareService;
        this.engineHealthReporter = engineHealthReporter;
    }

    public SelectedRole select(BalanceContext balanceContext, RoleType roleType, String group) {
        PlacementResult<SelectedRole, RoleType> selection =
                selectForQueue(balanceContext, roleType, group);
        return selection.value();
    }

    public PlacementResult<SelectedRole, RoleType> selectForQueue(
            BalanceContext balanceContext,
            RoleType roleType,
            String group) {
        return doSelect(balanceContext, roleType, group);
    }

    private PlacementResult<SelectedRole, RoleType> doSelect(
            BalanceContext balanceContext,
            RoleType roleType,
            String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        EndpointDiscovery discovery = discoverAliveEndpoints(roleType, group);
        if (discovery.registeredCount() == 0) {
            Logger.debug("Prefill select failed: no registered endpoints, request_id={}",
                    requestId);
            return PlacementResult.blocked(roleType);
        }
        PlacementResult<SelectedRole, RoleType> onlyEndpoint = selectOnlyEndpoint(
                discovery,
                balanceContext,
                roleType,
                group,
                config);
        if (onlyEndpoint != null) {
            return onlyEndpoint;
        }
        Map<String, Integer> cacheMatchResults =
                getCacheMatchResults(balanceContext, roleType, discovery);
        Map<String, Integer> rejections = new java.util.HashMap<>();
        Map<RoleType, Integer> poolWideBlockers =
                new EnumMap<>(RoleType.class);
        CandidateSet survivors = evaluateCandidates(
                discovery,
                balanceContext,
                config,
                cacheMatchResults,
                rejections,
                poolWideBlockers);
        CandidateSet selectedCandidates = survivors;
        if (selectedCandidates.size() == 0) {
            Logger.debug(
                    "Prefill select failed: no available endpoints, request_id={},"
                        + " rejections={}",
                    requestId,
                    rejections);
            RoleType poolWideBlocker = provenPoolWideBlocker(
                    poolWideBlockers,
                    discovery.registeredCount());
            if (poolWideBlocker != null) {
                return PlacementResult.blocked(poolWideBlocker);
            }
            return PlacementResult.blocked(roleType);
        }

        boolean modeledSelection = selectedCandidates.selectable(0);
        final int selectedIndex;
        if (modeledSelection) {
            selectedIndex = selectBestCandidate(
                    survivors,
                    selectedCandidates.minimumProjectedTtftMs,
                    roleType,
                    group,
                    seqLen,
                    config);
        } else {
            // Existing Engine work has no honest duration. This fallback never
            // invents a TTFT or passes the candidates through TTFT/cache policy.
            selectedIndex = selectUnmodeledCandidate(selectedCandidates);
        }

        if (selectedIndex < 0) {
            Logger.debug(
                    "Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId,
                    rejections);
            return PlacementResult.blocked(roleType);
        }

        PrefillEndpoint best = selectedCandidates.endpoint(selectedIndex);
        long bestCacheHit = selectedCandidates.cacheHit(selectedIndex);
        long selectedPrefillMs = selectedCandidates.prefillMs(selectedIndex);
        WorkerEndpoint.GenerationPin selectedPin =
                workerDirectory.captureEndpoint(
                        roleType,
                        selectedCandidates.endpointAddress(selectedIndex));
        if (selectedPin == null || selectedPin.endpoint() != best) {
            if (selectedPin != null) {
                selectedPin.close();
            }
            // The planning endpoint retired or was replaced after the
            // full-fleet snapshot. Re-enter the queue rather than
            // transferring ownership for a stale generation.
            return PlacementResult.blocked(roleType);
        }
        OptionalLong selectedTtft = OptionalLong.of(
                selectedCandidates.projectedTtftMs(selectedIndex));
        SelectedRole selectedRole = buildSelectedRole(
                best,
                roleType,
                requestId,
                selectedTtft,
                selectedPrefillMs,
                bestCacheHit,
                selectedCandidates.ownershipVersion(selectedIndex),
                selectedPin);
        reportSelectedEstimates(
                roleType,
                best,
                config,
                selectedTtft,
                selectedPrefillMs);
        reportCacheHitMetrics(roleType, bestCacheHit, seqLen);
        reportRoutingCacheMatchMetrics(
                roleType,
                selectedCandidates.routingCacheMatchTokens(selectedIndex),
                selectedCandidates.maximumRoutingCacheMatchTokens,
                seqLen);
        return PlacementResult.success(selectedRole);
    }

    /**
     * Avoid replaying an ever-growing queue when there is no placement choice.
     * The eventual queue publication remains the authoritative concurrent
     * admission point; this path performs only stable physical checks and the
     * configured pending-owner bound before transferring the exact pin.
     */
    private PlacementResult<SelectedRole, RoleType> selectOnlyEndpoint(
            EndpointDiscovery discovery,
            BalanceContext context,
            RoleType roleType,
            String group,
            FlexlbConfig config) {
        EndpointRegistry.PrefillRoutingEntry only =
                discovery.onlyPreferredEndpoint();
        if (only == null) {
            return null;
        }

        PrefillEndpoint endpoint = only.endpoint();
        long pending = endpoint.admissionPendingRequestCount();
        if (pending < 0L) {
            return PlacementResult.blocked(roleType);
        }
        if (pending == 0L) {
            return null;
        }

        Request request = context.getRequest();
        long seqLen = Math.max(0L, request.getSeqLen());
        WorkerStatus.EngineObservation observation =
                endpoint.getStatus().committedEngineObservation();
        long engineTokenLimit = observation.maxBatchTokensSize() > 0L
                ? observation.maxBatchTokensSize()
                : observation.maxSeqLen();
        long fallbackTokenLimit = config.getInternalRuntime()
                .getFallbackBatchTokenCapacity();
        if (exceedsPositiveLimit(seqLen, engineTokenLimit)
                || exceedsPositiveLimit(seqLen, fallbackTokenLimit)
                || exceedsPositiveLimit(
                        seqLen, observation.totalKvCacheTokens())) {
            return PlacementResult.blocked(roleType);
        }

        PrefillTimePredictor predictor = endpoint.getPredictor();
        if (predictor == null) {
            return null;
        }
        Map<String, Integer> cacheMatches =
                getCacheMatchResults(context, roleType, discovery);
        CacheTokenMatch cacheMatch = calculateCacheMatch(
                endpoint,
                only.address(),
                cacheMatches,
                request);
        final long prefillMs;
        try {
            prefillMs = PrefillPredictionBoundary.predictSingleRequestMs(
                    predictor.evaluator(), seqLen,
                    cacheMatch.effectiveHitTokens());
        } catch (RuntimeException predictionFailure) {
            return PlacementResult.blocked(roleType);
        }
        publishMonotonically(endpoint.getLastSelectedTime());
        reportCacheHitMetrics(
                roleType, cacheMatch.effectiveHitTokens(), seqLen);
        reportRoutingCacheMatchMetrics(
                roleType,
                cacheMatch.routingHitTokens(),
                cacheMatch.routingHitTokens(),
                seqLen);
        WorkerEndpoint.GenerationPin selectedPin = workerDirectory.captureEndpoint(
                roleType, only.address());
        if (selectedPin == null || selectedPin.endpoint() != endpoint) {
            if (selectedPin != null) {
                selectedPin.close();
            }
            return PlacementResult.blocked(roleType);
        }
        return PlacementResult.success(buildSelectedRole(
                endpoint,
                roleType,
                context.getRequestId(),
                OptionalLong.empty(),
                prefillMs,
                cacheMatch.effectiveHitTokens(),
                endpoint.placementVersion(),
                selectedPin));
    }

    private static boolean exceedsPositiveLimit(long required, long limit) {
        return limit > 0L && required > limit;
    }

    /**
     * Select an availability fallback by coherent pending ownership, then live LRU. The clock is a
     * fairness hint only; races may change ordering but can never turn a non-empty candidate set
     * into an availability failure.
     */
    private static int selectUnmodeledCandidate(CandidateSet candidates) {
        int selectedIndex = -1;
        long selectedPending = Long.MAX_VALUE;
        long selectedTime = Long.MAX_VALUE;
        for (int i = 0; i < candidates.size(); i++) {
            if (!candidates.engineWorkUnmodeled(i)) {
                throw new IllegalStateException(
                        "unmodeled fallback received a modeled candidate");
            }
            long pending = candidates.pendingCount(i);
            long lastSelected = candidates.endpoint(i).getLastSelectedTime().get();
            if (selectedIndex < 0
                    || pending < selectedPending
                    || pending == selectedPending && lastSelected < selectedTime) {
                selectedIndex = i;
                selectedPending = pending;
                selectedTime = lastSelected;
            }
        }
        if (selectedIndex >= 0) {
            long nowMicros = System.nanoTime() / 1_000L;
            candidates.endpoint(selectedIndex).getLastSelectedTime().updateAndGet(
                    current -> current == Long.MAX_VALUE
                            ? Long.MAX_VALUE
                            : Math.max(nowMicros, current + 1L));
        }
        return selectedIndex;
    }

    /** Select from candidates that already passed the common hard filters. */
    private int selectBestCandidate(CandidateSet survivors,
                                      long minProjectedTtftMs,
                                      RoleType roleType,
                                      String group,
                                      long seqLen,
                                      FlexlbConfig config) {
        if (survivors.size() == 0) {
            return -1;
        }

        RoutingConfig.CandidateChoiceConfig choice = config.getRouter()
                .getRoles().getPrefill().getCandidateChoice();
        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter()
                .getRoles().getPrefill().getCacheAffinity();
        BitSet preferredCandidates = new BitSet(survivors.size());
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        if (cacheAffinity != null) {
            long referenceHitTokens = 0L;
            long minHitTokens = survivors.minimumCacheHit;
            long maxHitTokens = survivors.maximumCacheHit;
            affinityCutoffMs = saturatingAdd(
                    minProjectedTtftMs,
                    Math.max(0L, cacheAffinity.getMaxExtraTtftMs()));
            affinityReason = "NO_CACHE_LEAD";
            if (maxHitTokens > minHitTokens) {
                for (int i = 0; i < survivors.size(); i++) {
                    if (survivors.projectedTtftMs(i)
                            == minProjectedTtftMs) {
                        referenceHitTokens = Math.max(
                                referenceHitTokens,
                                survivors.cacheHit(i));
                    }
                }
                boolean minimumHitRateMet = false;
                double minimumHitRate = normalizedHitRate(
                        cacheAffinity.getMinPrefixHitPercent());
                for (int i = 0; i < survivors.size(); i++) {
                    long hitTokens = survivors.cacheHit(i);
                    if (hitTokens <= minHitTokens
                            || hitTokens < referenceHitTokens
                            || minimumHitRate > 0.0
                                    && (seqLen <= 0L
                                        || hitTokens * 100.0 / seqLen
                                                < minimumHitRate)) {
                        continue;
                    }
                    minimumHitRateMet = true;
                    if (survivors.projectedTtftMs(i) <= affinityCutoffMs) {
                        preferredCandidates.set(i);
                    }
                }
                affinityReason = !preferredCandidates.isEmpty()
                        ? "CACHE_LEADER"
                        : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
            }
        }

        if (choice.getType()
                == RoutingConfig.CandidateChoiceType.LEAST_RECENTLY_USED_IN_POOL) {
            return selectLeastRecentlyUsed(
                    survivors, preferredCandidates, cacheAffinity != null,
                    affinityReason, affinityCutoffMs, minProjectedTtftMs,
                    roleType, group, config);
        }

        int selectedIndex;
        if (!preferredCandidates.isEmpty()) {
            selectedIndex = selectCacheLeader(survivors, preferredCandidates);
        } else {
            selectedIndex = selectBaselineCandidate(
                    survivors, minProjectedTtftMs, config);
        }

        if (selectedIndex >= 0 && cacheAffinity != null) {
            reportCacheAffinityDecision(
                    roleType, survivors.endpoint(selectedIndex).getIp(),
                    affinityReason);
            if (Logger.isDebugEnabled()) {
                Logger.debug(
                        "CostBasedPrefill cache-affinity decision - role: {}, group: {}, "
                                + "selected: {}, minProjectedTtftMs: {}, "
                                + "selectedProjectedTtftMs: {}, ttftCutoffMs: {}, "
                                + "hitTokens: {}, reason: {}",
                        roleType,
                        group,
                        survivors.endpointAddress(selectedIndex),
                        minProjectedTtftMs,
                        survivors.projectedTtftMs(selectedIndex),
                        affinityCutoffMs,
                        survivors.cacheHit(selectedIndex),
                        affinityReason);
            }
        }
        return selectedIndex;
    }

    private static double normalizedHitRate(double configuredRate) {
        if (Double.isNaN(configuredRate)
                || configuredRate == Double.POSITIVE_INFINITY) {
            return 100.0;
        }
        if (configuredRate == Double.NEGATIVE_INFINITY) {
            return 0.0;
        }
        return Math.min(100.0, Math.max(0.0, configuredRate));
    }

    private int selectLeastRecentlyUsed(
            CandidateSet candidates,
            BitSet preferredCandidates,
            boolean affinityEnabled,
            String affinityReason,
            long affinityCutoffMs,
            long minProjectedTtftMs,
            RoleType roleType,
            String group,
            FlexlbConfig config) {
        int selectedIndex = claimLeastRecentlyUsed(
                candidates,
                preferredCandidates,
                baselinePoolMask(candidates,
                        config.shortestTtftCandidateCount(candidates.size())));
        if (selectedIndex < 0) {
            return -1;
        }
        if (affinityEnabled) {
            String reason = contains(preferredCandidates, selectedIndex)
                    ? "CACHE_LEADER"
                    : !preferredCandidates.isEmpty()
                            ? "CACHE_AFFINITY_FALLBACK"
                            : affinityReason;
            reportCacheAffinityDecision(
                    roleType, candidates.endpoint(selectedIndex).getIp(), reason);
            if (Logger.isDebugEnabled()) {
                Logger.debug(
                        "Prefill LRU cache-affinity decision - role: {}, group: {}, "
                                + "selected: {}, minTtftMs: {}, selectedTtftMs: {}, "
                                + "ttftCutoffMs: {}, hitTokens: {}, reason: {}",
                        roleType,
                        group,
                        candidates.endpointAddress(selectedIndex),
                        minProjectedTtftMs,
                        candidates.projectedTtftMs(selectedIndex),
                        affinityCutoffMs,
                        candidates.cacheHit(selectedIndex),
                        reason);
            }
        }
        return selectedIndex;
    }

    /** Preserve CostBasedPrefill's original tie-window selection when affinity is disabled or gated off. */
    private int selectBaselineCandidate(
            CandidateSet survivors,
            long minProjectedTtftMs,
            FlexlbConfig config) {

        long tieThreshold = 0L;
        RoutingConfig.CandidateChoiceConfig candidateChoice = config.getRouter()
                .getRoles().getPrefill().getCandidateChoice();
        if (candidateChoice.getType()
                == RoutingConfig.CandidateChoiceType.RANDOM_WITHIN_TOLERANCE) {
            long percentageThreshold = (long) (
                    minProjectedTtftMs
                            * candidateChoice.getRelativeTolerance());
            tieThreshold = Math.max(
                    Math.max(0L, percentageThreshold),
                    Math.max(0L, candidateChoice.getMinimumToleranceMs()));
        }
        long ttftCutoffMs = saturatingAdd(minProjectedTtftMs, tieThreshold);
        int eligibleCount = 0;
        boolean allExactlyEqual = true;
        for (int i = 0; i < survivors.size(); i++) {
            long projectedTtftMs = survivors.projectedTtftMs(i);
            if (projectedTtftMs <= ttftCutoffMs) {
                eligibleCount++;
                allExactlyEqual &= projectedTtftMs == minProjectedTtftMs;
            }
        }
        if (eligibleCount == 0) {
            return -1;
        }
        int selectedOffset = allExactlyEqual
                ? (int) Math.floorMod(
                        equalTtftCursor.getAndIncrement(),
                        (long) eligibleCount)
                : ThreadLocalRandom.current().nextInt(eligibleCount);
        for (int i = 0; i < survivors.size(); i++) {
            if (survivors.projectedTtftMs(i) <= ttftCutoffMs
                    && selectedOffset-- == 0) {
                return i;
            }
        }
        throw new IllegalStateException(
                "TTFT candidate count changed during immutable selection");
    }

    /** Fairly rotate only endpoints with the same best cache hit and projected TTFT. */
    private int selectCacheLeader(
            CandidateSet survivors, BitSet preferredCandidates) {
        long bestHit = Long.MIN_VALUE;
        long bestProjectedTtftMs = Long.MAX_VALUE;
        int tiedCount = 0;
        for (int candidateIndex = 0;
                candidateIndex < survivors.size(); candidateIndex++) {
            if (!contains(preferredCandidates, candidateIndex)) {
                continue;
            }
            long hit = survivors.cacheHit(candidateIndex);
            long projectedTtftMs = survivors.projectedTtftMs(candidateIndex);
            if (hit > bestHit
                    || hit == bestHit && projectedTtftMs < bestProjectedTtftMs) {
                bestHit = hit;
                bestProjectedTtftMs = projectedTtftMs;
                tiedCount = 1;
            } else if (hit == bestHit
                    && projectedTtftMs == bestProjectedTtftMs) {
                tiedCount++;
            }
        }
        if (tiedCount == 0) {
            return -1;
        }
        int selectedOffset = (int) Math.floorMod(
                equalCacheLeaderCursor.getAndIncrement(),
                (long) tiedCount);
        for (int candidateIndex = 0;
                candidateIndex < survivors.size(); candidateIndex++) {
            if (!contains(preferredCandidates, candidateIndex)
                    || survivors.cacheHit(candidateIndex) != bestHit
                    || survivors.projectedTtftMs(candidateIndex)
                            != bestProjectedTtftMs) {
                continue;
            }
            if (selectedOffset-- == 0) {
                return candidateIndex;
            }
        }
        throw new IllegalStateException(
                "cache-leader count changed during immutable selection");
    }

    private static BitSet baselinePoolMask(
            CandidateSet candidates, int configuredCount) {
        int count = Math.min(Math.max(1, configuredCount), candidates.size());
        BitSet baseline = new BitSet(candidates.size());
        baseline.set(0, count);
        if (count == candidates.size()) {
            return baseline;
        }
        int selectedIndex = 0;
        for (int i = 1; i < candidates.size(); i++) {
            if (candidates.projectedTtftMs(i)
                    < candidates.projectedTtftMs(selectedIndex)) {
                selectedIndex = i;
            }
        }
        baseline.clear();
        baseline.set(selectedIndex);
        return baseline;
    }

    /** Claim the least-recently-used live clock, retrying one exact CAS race. */
    private static int claimLeastRecentlyUsed(
            CandidateSet candidates,
            BitSet preferredCandidates,
            BitSet baselineCandidates) {
        boolean hasPreferredCandidates = !preferredCandidates.isEmpty();
        for (int attempt = 0; attempt < 2; attempt++) {
            int selectedIndex = -1;
            AtomicLong selectedClock = null;
            long selectedValue = Long.MAX_VALUE;
            for (int poolPass = 0;
                    poolPass < (hasPreferredCandidates ? 2 : 1)
                            && selectedIndex < 0;
                    poolPass++) {
                BitSet pool = hasPreferredCandidates && poolPass == 0
                        ? preferredCandidates : baselineCandidates;
                for (int i = pool.nextSetBit(0);
                        i >= 0;
                        i = pool.nextSetBit(i + 1)) {
                    AtomicLong clock = candidates.endpoint(i)
                            .getLastSelectedTime();
                    long value = clock.get();
                    if (value == Long.MAX_VALUE) {
                        continue;
                    }
                    if (selectedIndex < 0
                            || value < selectedValue
                            || value == selectedValue
                                    && (candidates.projectedTtftMs(i)
                                            < candidates.projectedTtftMs(
                                                    selectedIndex)
                                        || candidates.projectedTtftMs(i)
                                            == candidates.projectedTtftMs(
                                                    selectedIndex)
                                            && i < selectedIndex)) {
                        selectedIndex = i;
                        selectedClock = clock;
                        selectedValue = value;
                    }
                }
            }
            if (selectedIndex < 0) {
                return -1;
            }
            if (attempt == 0) {
                long nowMicros = System.nanoTime() / 1_000L;
                if (selectedClock.compareAndSet(
                        selectedValue,
                        Math.max(nowMicros, selectedValue + 1L))) {
                    return selectedIndex;
                }
            } else {
                publishMonotonically(selectedClock);
                return selectedIndex;
            }
        }
        throw new IllegalStateException("unreachable LRU claim state");
    }

    private static void publishMonotonically(AtomicLong clock) {
        long nowMicros = System.nanoTime() / 1_000L;
        clock.updateAndGet(current -> current == Long.MAX_VALUE
                ? Long.MAX_VALUE
                : Math.max(nowMicros, current + 1L));
    }

    private static boolean contains(BitSet candidates, int index) {
        return candidates.get(index);
    }

    private record EndpointDiscovery(
            List<EndpointRegistry.PrefillRoutingEntry> candidates) {

        private EndpointDiscovery {
            candidates = java.util.Objects.requireNonNull(
                    candidates, "candidates");
        }

        private EndpointRegistry.PrefillRoutingEntry onlyPreferredEndpoint() {
            return candidates.size() == 1 ? candidates.getFirst() : null;
        }

        private int registeredCount() {
            return candidates.size();
        }

        /** Zero-copy address view over the exact fleet used by this decision. */
        private List<String> addresses() {
            return new AbstractList<>() {
                @Override
                public String get(int index) {
                    return candidates.get(index).address();
                }

                @Override
                public int size() {
                    return candidates.size();
                }
            };
        }
    }

    private static final class CandidateSet {
        private String[] endpointAddresses = new String[0];
        private PrefillEndpoint[] endpoints = new PrefillEndpoint[0];
        private RouteProjection.Candidate.State[] states =
                new RouteProjection.Candidate.State[0];
        private long[] projectedTtftMs = new long[0];
        private long[] projectedDrainMs = new long[0];
        private long[] incomingPrefillMs = new long[0];
        private long[] cacheHitTokens = new long[0];
        private long[] routingCacheMatchTokens = new long[0];
        private long[] pendingCounts = new long[0];
        private long[] ownershipVersions = new long[0];
        private int size;
        private long projectedDrainTotalMs;
        private int knownDrainCount;
        private long pendingRequestTotal;
        private long maximumPendingCount;
        private long maximumProjectedDrainMs;
        private long minimumProjectedTtftMs;
        private long minimumCacheHit;
        private long maximumCacheHit;
        private long maximumRoutingCacheMatchTokens;

        private void reset(int expectedCapacity) {
            ensureCapacity(expectedCapacity);
            size = 0;
            projectedDrainTotalMs = 0L;
            knownDrainCount = 0;
            pendingRequestTotal = 0L;
            maximumPendingCount = 0L;
            maximumProjectedDrainMs = 0L;
            minimumProjectedTtftMs = Long.MAX_VALUE;
            minimumCacheHit = Long.MAX_VALUE;
            maximumCacheHit = 0L;
            maximumRoutingCacheMatchTokens = 0L;
        }

        private void addCandidate(
                String endpointAddress,
                PrefillEndpoint endpoint,
                RouteProjection.CandidateView candidate,
                long ownershipVersion) {
            ensureCapacity(size + 1);
            endpointAddresses[size] = endpointAddress;
            endpoints[size] = endpoint;
            states[size] = candidate.state();
            projectedTtftMs[size] = candidate.projectedTtftMsValue();
            projectedDrainMs[size] = candidate.projectedDrainMsValue();
            incomingPrefillMs[size] = candidate.incomingPrefillMs();
            cacheHitTokens[size] = candidate.cacheHitTokens();
            routingCacheMatchTokens[size] =
                    candidate.routingCacheMatchTokens();
            pendingCounts[size] = candidate.pendingCountValue();
            ownershipVersions[size] = ownershipVersion;
            size++;
            if (candidate.hasProjectedDrain()) {
                projectedDrainTotalMs = saturatingAdd(
                        projectedDrainTotalMs,
                        candidate.requiredProjectedDrainMs());
                maximumProjectedDrainMs = Math.max(
                        maximumProjectedDrainMs,
                        candidate.requiredProjectedDrainMs());
                knownDrainCount++;
            }
            pendingRequestTotal = saturatingAdd(
                    pendingRequestTotal, candidate.requiredPendingCount());
            maximumPendingCount = Math.max(
                    maximumPendingCount, candidate.requiredPendingCount());
            if (candidate.selectable()) {
                minimumProjectedTtftMs = Math.min(
                        minimumProjectedTtftMs,
                        candidate.requiredProjectedTtftMs());
            }
            minimumCacheHit = Math.min(
                    minimumCacheHit, candidate.cacheHitTokens());
            maximumCacheHit = Math.max(
                    maximumCacheHit, candidate.cacheHitTokens());
            maximumRoutingCacheMatchTokens = Math.max(
                    maximumRoutingCacheMatchTokens,
                    candidate.routingCacheMatchTokens());
        }

        private boolean selectable(int index) {
            return states[index] == RouteProjection.Candidate.State.MODELED
                    && projectedTtftMs[index] != RouteProjection.Candidate.UNKNOWN;
        }

        private boolean engineWorkUnmodeled(int index) {
            return states[index]
                    == RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK;
        }

        private PrefillEndpoint endpoint(int index) {
            return endpoints[index];
        }

        private String endpointAddress(int index) {
            return endpointAddresses[index];
        }

        private long cacheHit(int index) {
            return cacheHitTokens[index];
        }

        private long projectedTtftMs(int index) {
            return projectedTtftMs[index];
        }

        private long prefillMs(int index) {
            return incomingPrefillMs[index];
        }

        private boolean hasProjectedDrain(int index) {
            return projectedDrainMs[index] != RouteProjection.Candidate.UNKNOWN;
        }

        private long projectedDrainMs(int index) {
            return projectedDrainMs[index];
        }

        private long pendingCount(int index) {
            return pendingCounts[index];
        }

        private long routingCacheMatchTokens(int index) {
            return routingCacheMatchTokens[index];
        }

        private long ownershipVersion(int index) {
            return ownershipVersions[index];
        }

        private int size() {
            return size;
        }

        private void retainAt(int source, int target) {
            if (source == target) {
                return;
            }
            endpointAddresses[target] = endpointAddresses[source];
            endpoints[target] = endpoints[source];
            states[target] = states[source];
            projectedTtftMs[target] = projectedTtftMs[source];
            projectedDrainMs[target] = projectedDrainMs[source];
            incomingPrefillMs[target] = incomingPrefillMs[source];
            cacheHitTokens[target] = cacheHitTokens[source];
            routingCacheMatchTokens[target] =
                    routingCacheMatchTokens[source];
            pendingCounts[target] = pendingCounts[source];
            ownershipVersions[target] = ownershipVersions[source];
        }

        private void beginRetainedSummary() {
            projectedDrainTotalMs = 0L;
            knownDrainCount = 0;
            pendingRequestTotal = 0L;
            maximumPendingCount = 0L;
            maximumProjectedDrainMs = 0L;
            minimumProjectedTtftMs = Long.MAX_VALUE;
            minimumCacheHit = Long.MAX_VALUE;
            maximumCacheHit = 0L;
            maximumRoutingCacheMatchTokens = 0L;
        }

        private void retainAndSummarize(int source, int target) {
            retainAt(source, target);
            long drainMs = projectedDrainMs[target];
            if (drainMs != RouteProjection.Candidate.UNKNOWN) {
                projectedDrainTotalMs = saturatingAdd(
                        projectedDrainTotalMs, drainMs);
                maximumProjectedDrainMs = Math.max(
                        maximumProjectedDrainMs, drainMs);
                knownDrainCount++;
            }
            long pending = pendingCounts[target];
            pendingRequestTotal = saturatingAdd(
                    pendingRequestTotal, pending);
            maximumPendingCount = Math.max(
                    maximumPendingCount, pending);
            if (selectable(target)) {
                minimumProjectedTtftMs = Math.min(
                        minimumProjectedTtftMs,
                        projectedTtftMs[target]);
            }
            minimumCacheHit = Math.min(
                    minimumCacheHit, cacheHitTokens[target]);
            maximumCacheHit = Math.max(
                    maximumCacheHit, cacheHitTokens[target]);
            maximumRoutingCacheMatchTokens = Math.max(
                    maximumRoutingCacheMatchTokens,
                    routingCacheMatchTokens[target]);
        }

        private void finishRetainedSummary(int retainedSize) {
            size = retainedSize;
        }

        private void ensureCapacity(int expectedCapacity) {
            if (states.length >= expectedCapacity) {
                return;
            }
            int capacity = Math.max(expectedCapacity,
                    Math.max(16, states.length << 1));
            endpointAddresses = Arrays.copyOf(endpointAddresses, capacity);
            endpoints = Arrays.copyOf(endpoints, capacity);
            states = Arrays.copyOf(states, capacity);
            projectedTtftMs = Arrays.copyOf(projectedTtftMs, capacity);
            projectedDrainMs = Arrays.copyOf(projectedDrainMs, capacity);
            incomingPrefillMs = Arrays.copyOf(incomingPrefillMs, capacity);
            cacheHitTokens = Arrays.copyOf(cacheHitTokens, capacity);
            routingCacheMatchTokens = Arrays.copyOf(
                    routingCacheMatchTokens, capacity);
            pendingCounts = Arrays.copyOf(pendingCounts, capacity);
            ownershipVersions = Arrays.copyOf(ownershipVersions, capacity);
        }

    }

    /** Per-planner reusable candidate columns; a planner never re-enters selection. */
    private static final class CandidateScratch {
        private final CandidateSet modeled = new CandidateSet();
        private final CandidateSet unmodeled = new CandidateSet();

        private void reset(int expectedCapacity) {
            modeled.reset(expectedCapacity);
            unmodeled.reset(0);
        }
    }

    private CandidateSet evaluateCandidates(
            EndpointDiscovery discovery,
            BalanceContext balanceContext,
            FlexlbConfig config,
            Map<String, Integer> cacheMatchResults,
            Map<String, Integer> rejections,
            Map<RoleType, Integer> poolWideBlockers) {
        Request request = balanceContext.getRequest();
        int eligibleSize = discovery.candidates().size();
        CandidateScratch scratch = CANDIDATE_SCRATCH.get();
        scratch.reset(eligibleSize);
        CandidateSet modeled = scratch.modeled;
        CandidateSet unmodeled = scratch.unmodeled;
        long planningAtMs = System.currentTimeMillis();
        RouteProjection.Session projectionSession = RouteProjection.session();

        // Build one coherent projection per live endpoint. Cache
        // hit is part of both the incoming service prediction and batch-group
        // boundary planning; availability consumes this projection's pending
        // count instead of taking a second, potentially contradictory snapshot.
        RouteProjection.Demand projectionDemand = projectionDemand(config);
        for (int i = 0; i < discovery.candidates().size(); i++) {
                EndpointRegistry.PrefillRoutingEntry routingEntry =
                        discovery.candidates().get(i);
                PrefillEndpoint ep = routingEntry.endpoint();
                String endpointAddress = routingEntry.address();
                CacheTokenMatch cacheMatch =
                        calculateCacheMatch(ep, endpointAddress, cacheMatchResults, request);
                long cacheHit = cacheMatch.effectiveHitTokens();
                long routingCacheMatchTokens = cacheMatch.routingHitTokens();
                RouteProjection.Inputs projectionInputs =
                        ep.captureRouteProjectionInputs();
                PrefillTimePredictor predictor = ep.getPredictor();
                RouteProjection.CandidateView projection =
                        projectionSession.projectView(
                                projectionInputs,
                                request.getRequestId(),
                                balanceContext.getPriority(),
                                planningAtMs,
                                // RequestRegistry owns terminal deadlines.
                                // Selection scores endpoint work only; an expiry race
                                // must not be reported as a capacity blocker.
                                Long.MAX_VALUE,
                                request.getSeqLen(),
                                cacheHit,
                                routingCacheMatchTokens,
                                projectionDemand,
                                predictor == null
                                        ? null : predictor.evaluator(),
                                ep.deliveryProjection(),
                                planningAtMs);
                boolean modeledProjection = projection.selectable();
                boolean unmodeledProjection = projection.engineWorkUnmodeled();
                if (!modeledProjection && !unmodeledProjection) {
                    RoleType blockerRole = projection.blockerRole();
                    if (blockerRole != null) {
                        poolWideBlockers.merge(
                                blockerRole, 1, Integer::sum);
                    }
                    rejections.merge(
                            "PROJECTION_"
                                    + projection.state().name()
                                    + "_"
                                    + projection.detail(),
                            1,
                            Integer::sum);
                    continue;
                }

                if (modeledProjection) {
                    modeled.addCandidate(
                            endpointAddress, ep, projection,
                            projectionInputs.ownershipVersion());
                } else {
                    unmodeled.addCandidate(
                            endpointAddress, ep, projection,
                            projectionInputs.ownershipVersion());
                }
                // One routing request may inspect hundreds of endpoints. Keep the
                // per-candidate diagnostic at TRACE so enabling ordinary DEBUG
                // cannot turn the selection hot path into a log flood.
                if (Logger.isTraceEnabled()) {
                    if (modeledProjection) {
                        Logger.trace(
                                "Prefill projection - ip: {}, order: {}, hitCache: {}, "
                                        + "ttftMs: {}, drainMs: {}",
                                endpointAddress,
                                config.isPriorityOrdering() ? "PRIORITY" : "FIFO",
                                cacheHit,
                                projection.requiredProjectedTtftMs(),
                                projection.hasProjectedDrain()
                                        ? Long.toString(
                                                projection.requiredProjectedDrainMs())
                                        : projectionDemand.drainRequired()
                                                ? "unknown"
                                                : "not-requested");
                    } else {
                        Logger.trace(
                                "Prefill projection unmodeled - ip: {}, pendingRequests: {},"
                                    + " reason: engine_work",
                                endpointAddress,
                                projection.requiredPendingCount());
                    }
                }
        }

        if (modeled.size() != 0) {
            return rejectOutliers(
                    modeled, config, rejections);
        }
        return unmodeled;
    }

    private CandidateSet rejectOutliers(
            CandidateSet feasible,
            FlexlbConfig config,
            Map<String, Integer> rejections) {
        RoutingConfig.OutlierRejectionConfig outlier =
                outlierRejection(config.getRouter().getRoles().getPrefill()
                        .getCandidateChoice());
        double hotspotMultiplier = outlier == null
                ? 0.0 : outlier.getMaxPendingVsAverageMultiplier();
        double imbalanceMultiplier = outlier == null
                ? 0.0 : outlier.getMaxProjectedDrainVsAverageMultiplier();
        if (hotspotMultiplier <= 0.0 && imbalanceMultiplier <= 0.0) {
            return feasible;
        }
        int feasibleSize = feasible.size();
        long avgDrainMs = feasible.knownDrainCount == 0
                ? 0L
                : feasible.projectedDrainTotalMs / feasible.knownDrainCount;
        long avgPendingCount = feasible.pendingRequestTotal / feasibleSize;

        // Balanced fleets are the dominant path. The maxima were reduced
        // while candidates were projected, so prove that no outlier can be
        // rejected without walking all endpoints a second time.
        boolean mayRejectHotspot = hotspotMultiplier > 0.0
                && avgPendingCount > 0L
                && feasible.maximumPendingCount
                        > avgPendingCount * hotspotMultiplier;
        boolean mayRejectDrain = imbalanceMultiplier > 0.0
                && avgDrainMs > 0L
                && feasible.maximumProjectedDrainMs
                        > avgDrainMs * imbalanceMultiplier;
        if (!mayRejectHotspot && !mayRejectDrain) {
            return feasible;
        }

        // Round 2: hotspot / drain-imbalance filter using the same projections.
        int retainedCount = 0;
        int leastLoadedIndex = -1;
        long leastDrainMs = Long.MAX_VALUE;
        int leastPendingIndex = -1;
        long leastPendingCount = Long.MAX_VALUE;
        int hotspotRejected = 0;
        int imbalanceRejected = 0;
        feasible.beginRetainedSummary();
        for (int i = 0; i < feasibleSize; i++) {
            long pendingCount = feasible.pendingCount(i);

            if (feasible.hasProjectedDrain(i)
                    && feasible.projectedDrainMs(i) < leastDrainMs) {
                leastDrainMs = feasible.projectedDrainMs(i);
                leastLoadedIndex = i;
            }
            if (pendingCount < leastPendingCount) {
                leastPendingCount = pendingCount;
                leastPendingIndex = i;
            }

            boolean rejected = false;
            if (mayRejectHotspot
                    && pendingCount > avgPendingCount * hotspotMultiplier) {
                hotspotRejected++;
                rejected = true;
            }
            if (!rejected && mayRejectDrain
                    && feasible.hasProjectedDrain(i)
                    && feasible.projectedDrainMs(i)
                    > avgDrainMs * imbalanceMultiplier) {
                imbalanceRejected++;
                rejected = true;
            }
            if (!rejected) {
                feasible.retainAndSummarize(i, retainedCount++);
            }
        }

        if (hotspotRejected > 0) {
            rejections.merge(
                    "HOTSPOT_FILTERED", hotspotRejected, Integer::sum);
        }
        if (imbalanceRejected > 0) {
            rejections.merge(
                    "IMBALANCE_FILTERED", imbalanceRejected, Integer::sum);
        }
        if (retainedCount == feasibleSize) {
            return feasible;
        }

        if (retainedCount == 0) {
            int fallbackIndex = leastLoadedIndex >= 0
                    ? leastLoadedIndex : leastPendingIndex;
            if (fallbackIndex >= 0) {
                feasible.retainAndSummarize(
                        fallbackIndex, retainedCount++);
            }
        }

        feasible.finishRetainedSummary(retainedCount);
        return feasible;
    }

    static RoleType provenPoolWideBlocker(
            Map<RoleType, Integer> blockers,
            int registeredEndpoints) {
        if (registeredEndpoints <= 0) {
            return null;
        }
        for (Map.Entry<RoleType, Integer> blocker : blockers.entrySet()) {
            if (blocker.getValue() >= registeredEndpoints) {
                return blocker.getKey();
            }
        }
        return null;
    }

    /**
     * Drain is a cost-policy input only when drain-imbalance rejection is live.
     * Pending-count filtering and final TTFT choice do not consume it.
     */
    private RouteProjection.Demand projectionDemand(FlexlbConfig config) {
        RoutingConfig.OutlierRejectionConfig outlier =
                outlierRejection(config.getRouter().getRoles().getPrefill()
                        .getCandidateChoice());
        return outlier != null
                && outlier.getMaxProjectedDrainVsAverageMultiplier() > 0.0
                ? RouteProjection.Demand.TTFT_AND_DRAIN
                : RouteProjection.Demand.TTFT_ONLY;
    }

    private static RoutingConfig.OutlierRejectionConfig outlierRejection(
            RoutingConfig.CandidateChoiceConfig candidateChoice) {
        return switch (candidateChoice.getType()) {
            case RANDOM_WITHIN_TOLERANCE, BEST_ONLY ->
                    candidateChoice.getOutlierRejection();
            case LEAST_RECENTLY_USED_IN_POOL -> null;
        };
    }

    private EndpointDiscovery discoverAliveEndpoints(
            RoleType roleType,
            String group) {
        List<EndpointRegistry.PrefillRoutingEntry> directory =
                workerDirectory.prefillRoutingSnapshot(roleType);
        if (group == null) {
            return new EndpointDiscovery(directory);
        }
        List<EndpointRegistry.PrefillRoutingEntry> matching = new ArrayList<>();
        for (EndpointRegistry.PrefillRoutingEntry entry : directory) {
            WorkerStatus.TopologySnapshot topology = entry.endpoint()
                    .getStatus().topologySnapshot();
            if (group.equals(topology.group())) {
                matching.add(entry);
            }
        }
        return new EndpointDiscovery(List.copyOf(matching));
    }

    private Map<String, Integer> getCacheMatchResults(
            BalanceContext balanceContext,
            RoleType roleType,
            EndpointDiscovery discovery) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(
                blockCacheKeys, roleType, discovery.addresses());
    }

    private record CacheTokenMatch(
            long effectiveHitTokens,
            long routingHitTokens) {
        private static final CacheTokenMatch NONE =
                new CacheTokenMatch(0L, 0L);
    }

    private CacheTokenMatch calculateCacheMatch(
            PrefillEndpoint ep,
            String endpointAddress,
            Map<String, Integer> cacheMatchResults,
            Request request) {
        if (cacheMatchResults == null || cacheMatchResults.isEmpty() || request == null) {
            return CacheTokenMatch.NONE;
        }
        long seqLen = request.getSeqLen();
        if (seqLen <= 0L) {
            return CacheTokenMatch.NONE;
        }
        Integer prefixMatchLength = cacheMatchResults.get(endpointAddress);
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return CacheTokenMatch.NONE;
        }
        long blockSize = request.getCacheKeyBlockSize();
        WorkerStatus status = ep.getStatus();
        CacheStatus cacheStatus = status == null ? null : status.getCacheStatus();
        if (blockSize <= 0L && cacheStatus != null) {
            blockSize = cacheStatus.getBlockSize();
        }
        if (blockSize <= 0L) {
            return CacheTokenMatch.NONE;
        }
        long rawHit;
        try {
            rawHit = Math.multiplyExact(
                    blockSize, prefixMatchLength.longValue());
        } catch (ArithmeticException overflow) {
            rawHit = seqLen;
        }
        long routingHit = Math.min(seqLen, Math.max(0L, rawHit));
        long effectiveHit = rawHit >= seqLen
                ? Math.max(0L, seqLen - blockSize)
                : routingHit;
        return new CacheTokenMatch(effectiveHit, routingHit);
    }

    private void reportCacheHitMetrics(RoleType roleType, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, hitCacheTokens, hitRate);
    }

    /**
     * Publish only estimates that were part of the actual selection decision.
     * Unmodeled fallbacks intentionally have no projected TTFT, so emitting a
     * fabricated value for them would corrupt the metric distribution.
     */
    private void reportSelectedEstimates(
            RoleType roleType,
            PrefillEndpoint endpoint,
            FlexlbConfig config,
            OptionalLong projectedTtftMs,
            long executionTimeMs) {
        if (projectedTtftMs.isEmpty()) {
            return;
        }
        String deliveryMode = config.getDispatcher().typeName();
        try {
            engineHealthReporter.reportPrefillSelectedEstimates(
                    roleType,
                    endpoint.getIp(),
                    deliveryMode,
                    projectedTtftMs.getAsLong(),
                    executionTimeMs);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Prefill selected-estimate metric failed: engine={}, delivery_mode={}",
                    endpoint.ipPort(), deliveryMode, telemetryFailure);
        }
    }

    private void reportRoutingCacheMatchMetrics(
            RoleType roleType,
            long selectedHitTokens,
            long candidateMaxHitTokens,
            long totalTokens) {
        engineHealthReporter.reportRoutingSelectedCacheMatchMetrics(
                roleType, selectedHitTokens, totalTokens);
        engineHealthReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                roleType, candidateMaxHitTokens);
    }

    private SelectedRole buildSelectedRole(
            PrefillEndpoint ep,
            RoleType roleType,
            long requestId,
            OptionalLong projectedTtftMs,
            long selectedPrefillMs,
            long bestCacheHit,
            long placementVersion,
            WorkerEndpoint.GenerationPin selectedPin) {
        try {
            // Populate DebugInfo so ScheduledRequest.hitCache() can read
            // hitCacheLen for batch metrics.
            DebugInfo debugInfo = new DebugInfo();
            debugInfo.setHitCacheLen(bestCacheHit);

            if (selectedPin == null || selectedPin.endpoint() != ep) {
                throw new IllegalStateException(
                        "selected Prefill endpoint generation changed before handoff");
            }
            WorkerStatus workerStatus = ep.getStatus();
            WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
            WorkerStatus.EngineObservation status = workerStatus.committedEngineObservation();
            ServerStatus result = new ServerStatus();
            result.setRole(roleType);
            result.setRequestId(requestId);
            if (projectedTtftMs.isPresent()) {
                result.setPrefillTime(projectedTtftMs.getAsLong());
            }
            // For the unmodeled fallback, leave the public primitive field unset.
            // Its default zero is wire-compatible metadata only: production code
            // never reads it as a TTFT (RequestScheduler only copies the DTO).
            result.setGroup(topology.group());
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setDebugInfo(debugInfo);
            result.setSuccess(true);
            WorkerEndpoint.GenerationPin ownedPin = selectedPin;
            selectedPin = null;
            return SelectedRole.prefill(
                    ownedPin, result, selectedPrefillMs, placementVersion);
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

    private void reportCacheAffinityDecision(
            RoleType roleType, String engineIp, String decision) {
        engineHealthReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    private static long saturatingAdd(long left, long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        if (right < 0L && left < Long.MIN_VALUE - right) {
            return Long.MIN_VALUE;
        }
        return left + right;
    }

}
