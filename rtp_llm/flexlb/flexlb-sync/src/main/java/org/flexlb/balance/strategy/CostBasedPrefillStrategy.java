package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
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

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class CostBasedPrefillStrategy {

    private static final int MAX_PROJECTED_CANDIDATES = 2;

    private final WorkerDirectory workerDirectory;
    private final CacheAwareService cacheAwareService;
    private final EngineHealthReporter engineHealthReporter;
    private final AtomicLong candidateWindowCursor = new AtomicLong();

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

        EndpointDiscovery discovery = discoverAliveEndpoints(
                roleType, group, balanceContext.getExcludedPrefillIpPort());
        if (discovery.registeredCount() == 0) {
            Logger.debug("Prefill select failed: no registered endpoints, request_id={}",
                    requestId);
            discovery.close();
            return PlacementResult.blocked(roleType);
        }
        try (discovery) {
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
                    getCacheMatchResults(balanceContext, roleType, group);
            Map<String, Integer> rejections = new java.util.HashMap<>(discovery.rejections());
            Map<RoleType, Integer> poolWideBlockers =
                    new EnumMap<>(RoleType.class);
            CandidateSet survivors = evaluateCandidates(
                    discovery.takePreferredEndpoints(),
                    balanceContext,
                    config,
                    cacheMatchResults,
                    rejections,
                    poolWideBlockers);
            if (survivors.size() == 0 && discovery.hasExcludedEndpoint()) {
                // The endpoint rejected by the previous queue offer is a true
                // fallback: project it only after every other live endpoint has
                // proved unusable. It therefore cannot skew normal candidate
                // averages or cache-affinity decisions. A resource-available
                // unmodeled endpoint is usable, so it also keeps this rejected
                // endpoint out of the candidate domain.
                survivors.close();
                CandidateSet excluded = new CandidateSet(1);
                discovery.moveExcludedTo(excluded);
                survivors = evaluateCandidates(
                        excluded,
                        balanceContext,
                        config,
                        cacheMatchResults,
                        rejections,
                        poolWideBlockers);
            }
            CandidateSet selectedCandidates = survivors;
            try (selectedCandidates) {
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
                        return PlacementResult.blocked(
                                poolWideBlocker);
                    }
                    return PlacementResult.blocked(roleType);
                }

                boolean modeledSelection =
                        selectedCandidates.candidate(0).selectable();
                final int selectedIndex;
                if (modeledSelection) {
                    // Numeric TTFT policies see only candidates with a complete model.
                    long minProjectedTtftMs = Long.MAX_VALUE;
                    for (int i = 0; i < selectedCandidates.size(); i++) {
                        long projectedTtftMs = selectedCandidates.projectedTtftMs(i);
                        if (projectedTtftMs < minProjectedTtftMs) {
                            minProjectedTtftMs = projectedTtftMs;
                        }
                    }
                    selectedIndex =
                            selectBestCandidate(
                                    survivors,
                                    minProjectedTtftMs,
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
                RouteProjection.Candidate selectedCandidate = selectedCandidates.candidate(selectedIndex);
                long bestCacheHit = selectedCandidates.cacheHit(selectedIndex);
                long selectedPrefillMs = selectedCandidates.prefillMs(selectedIndex);
                reportCacheHitMetrics(roleType, bestCacheHit, seqLen);
                long candidateMaxRoutingHit = 0L;
                for (int i = 0; i < selectedCandidates.size(); i++) {
                    candidateMaxRoutingHit =
                            Math.max(
                                    candidateMaxRoutingHit,
                                    selectedCandidates.candidate(i).routingCacheMatchTokens());
                }
                reportRoutingCacheMatchMetrics(
                        roleType,
                        selectedCandidates.candidate(selectedIndex).routingCacheMatchTokens(),
                        candidateMaxRoutingHit,
                        seqLen);

                SelectedRole selectedRole =
                        buildSelectedRole(
                                selectedCandidates,
                                selectedIndex,
                                roleType,
                                requestId,
                                selectedCandidate.projectedTtftMs(),
                                selectedPrefillMs,
                                bestCacheHit);
                reportSelectedEstimates(
                        roleType,
                        best,
                        config,
                        selectedCandidate.projectedTtftMs(),
                        selectedPrefillMs);
                return PlacementResult.success(selectedRole);
            }
        }
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
        CandidateSet candidates = discovery.preferredEndpoints();
        if (candidates.size() != 1 || discovery.hasExcludedEndpoint()) {
            return null;
        }

        PrefillEndpoint endpoint = candidates.endpoint(0);
        long pending = endpoint.admissionPendingRequestCount();
        long maxPending = config.getRouter().getRoles().getPrefill()
                .getAvailability().getMaxPendingRequests();
        if (pending < 0L || pending >= maxPending) {
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
                getCacheMatchResults(context, roleType, group);
        CacheTokenMatch cacheMatch = calculateCacheMatch(
                endpoint,
                candidates.endpointAddress(0),
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
        return PlacementResult.success(buildSelectedRole(
                candidates,
                0,
                roleType,
                context.getRequestId(),
                OptionalLong.empty(),
                prefillMs,
                cacheMatch.effectiveHitTokens()));
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
            RouteProjection.Candidate candidate = candidates.candidate(i);
            if (!candidate.engineWorkUnmodeled()) {
                throw new IllegalStateException(
                        "unmodeled fallback received a modeled candidate");
            }
            long pending = candidate.requiredPendingCount();
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
        long preferredMask = 0L;
        long affinityCutoffMs = 0L;
        String affinityReason = null;
        if (cacheAffinity != null) {
            long referenceHitTokens = 0L;
            long minHitTokens = Long.MAX_VALUE;
            long maxHitTokens = 0L;
            for (int i = 0; i < survivors.size(); i++) {
                long hitTokens = survivors.cacheHit(i);
                minHitTokens = Math.min(minHitTokens, hitTokens);
                maxHitTokens = Math.max(maxHitTokens, hitTokens);
                if (survivors.projectedTtftMs(i) == minProjectedTtftMs) {
                    referenceHitTokens = Math.max(
                            referenceHitTokens, hitTokens);
                }
            }
            affinityCutoffMs = saturatingAdd(
                    minProjectedTtftMs,
                    Math.max(0L, cacheAffinity.getMaxExtraTtftMs()));
            affinityReason = "NO_CACHE_LEAD";
            if (maxHitTokens > minHitTokens) {
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
                        preferredMask |= 1L << i;
                    }
                }
                affinityReason = preferredMask != 0L
                        ? "CACHE_LEADER"
                        : minimumHitRateMet ? "OVER_CAP" : "LOW_CACHE_HIT";
            }
        }

        if (choice.getType()
                == RoutingConfig.CandidateChoiceType.LEAST_RECENTLY_USED_IN_POOL) {
            return selectLeastRecentlyUsed(
                    survivors, preferredMask, cacheAffinity != null,
                    affinityReason, affinityCutoffMs, minProjectedTtftMs,
                    roleType, group, config);
        }

        int selectedIndex;
        if (preferredMask != 0L) {
            selectedIndex = selectCacheLeader(survivors, preferredMask);
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
            long preferredMask,
            boolean affinityEnabled,
            String affinityReason,
            long affinityCutoffMs,
            long minProjectedTtftMs,
            RoleType roleType,
            String group,
            FlexlbConfig config) {
        int selectedIndex = claimLeastRecentlyUsed(
                candidates,
                preferredMask,
                baselinePoolMask(candidates,
                        config.shortestTtftCandidateCount(candidates.size())));
        if (selectedIndex < 0) {
            return -1;
        }
        if (affinityEnabled) {
            String reason = contains(preferredMask, selectedIndex)
                    ? "CACHE_LEADER"
                    : preferredMask != 0L
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
        int selectedIndex = -1;
        int tiedCount = 0;
        for (int i = 0; i < survivors.size(); i++) {
            if (survivors.projectedTtftMs(i) <= ttftCutoffMs
                    && ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                selectedIndex = i;
            }
        }
        return selectedIndex;
    }

    /** Randomize only endpoints with the same best cache hit and projected TTFT. */
    private int selectCacheLeader(
            CandidateSet survivors, long preferredMask) {
        long bestHit = Long.MIN_VALUE;
        long bestProjectedTtftMs = Long.MAX_VALUE;
        int selectedIndex = -1;
        int tiedCount = 0;
        for (int candidateIndex = 0;
                candidateIndex < survivors.size(); candidateIndex++) {
            if (!contains(preferredMask, candidateIndex)) {
                continue;
            }
            long hit = survivors.cacheHit(candidateIndex);
            long projectedTtftMs = survivors.projectedTtftMs(candidateIndex);
            if (hit > bestHit
                    || hit == bestHit && projectedTtftMs < bestProjectedTtftMs) {
                bestHit = hit;
                bestProjectedTtftMs = projectedTtftMs;
                selectedIndex = candidateIndex;
                tiedCount = 1;
            } else if (hit == bestHit
                    && projectedTtftMs == bestProjectedTtftMs
                    && ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                selectedIndex = candidateIndex;
            }
        }
        return selectedIndex;
    }

    private static long baselinePoolMask(
            CandidateSet candidates, int configuredCount) {
        int count = Math.min(Math.max(1, configuredCount), candidates.size());
        if (count == candidates.size()) {
            return (1L << count) - 1L;
        }
        int selectedIndex = 0;
        for (int i = 1; i < candidates.size(); i++) {
            if (candidates.projectedTtftMs(i)
                    < candidates.projectedTtftMs(selectedIndex)) {
                selectedIndex = i;
            }
        }
        return 1L << selectedIndex;
    }

    /** Claim the least-recently-used live clock, retrying one exact CAS race. */
    private static int claimLeastRecentlyUsed(
            CandidateSet candidates,
            long preferredMask,
            long baselineMask) {
        for (int attempt = 0; attempt < 2; attempt++) {
            int selectedIndex = -1;
            AtomicLong selectedClock = null;
            long selectedValue = Long.MAX_VALUE;
            long pool = preferredMask != 0L ? preferredMask : baselineMask;
            for (int pass = 0; pass < 2 && selectedIndex < 0; pass++) {
                for (int i = 0; i < candidates.size(); i++) {
                    if (!contains(pool, i)) {
                        continue;
                    }
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
                if (pool == baselineMask) {
                    break;
                }
                pool = baselineMask;
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

    private static boolean contains(long mask, int index) {
        return (mask & 1L << index) != 0L;
    }

    private static final class EndpointDiscovery implements AutoCloseable {
        private CandidateSet preferredEndpoints;
        private WorkerEndpoint.GenerationPin excludedPin;
        private final String excludedIpPort;
        private final int registeredCount;
        private final Map<String, Integer> rejections;

        private EndpointDiscovery(
                CandidateSet preferredEndpoints,
                WorkerEndpoint.GenerationPin excludedPin,
                String excludedIpPort,
                int registeredCount,
                Map<String, Integer> rejections) {
            this.preferredEndpoints = preferredEndpoints;
            this.excludedPin = excludedPin;
            this.excludedIpPort = excludedIpPort;
            this.registeredCount = registeredCount;
            this.rejections = Map.copyOf(rejections);
        }

        private CandidateSet preferredEndpoints() {
            if (preferredEndpoints == null) {
                throw new IllegalStateException(
                        "preferred candidate ownership was already moved");
            }
            return preferredEndpoints;
        }

        private CandidateSet takePreferredEndpoints() {
            CandidateSet owned = preferredEndpoints();
            preferredEndpoints = null;
            return owned;
        }

        private boolean hasExcludedEndpoint() {
            return excludedPin != null;
        }

        private void moveExcludedTo(CandidateSet target) {
            WorkerEndpoint.GenerationPin owned = excludedPin;
            if (owned == null) {
                throw new IllegalStateException(
                        "excluded candidate ownership is absent");
            }
            target.addEndpoint(excludedIpPort, owned);
            excludedPin = null;
        }

        private String excludedIpPort() {
            return excludedIpPort;
        }

        private int registeredCount() {
            return registeredCount;
        }

        private Map<String, Integer> rejections() {
            return rejections;
        }

        @Override
        public void close() {
            CandidateSet preferred = preferredEndpoints;
            preferredEndpoints = null;
            if (preferred != null) {
                preferred.close();
            }
            WorkerEndpoint.GenerationPin excluded = excludedPin;
            excludedPin = null;
            if (excluded != null) {
                excluded.close();
            }
        }
    }

    private static final class CandidateSet implements AutoCloseable {
        private static final class Entry {
            private final String endpointAddress;
            private WorkerEndpoint.GenerationPin pin;
            private final RouteProjection.Candidate candidate;

            private Entry(
                    String endpointAddress,
                    WorkerEndpoint.GenerationPin pin,
                    RouteProjection.Candidate candidate) {
                this.endpointAddress = endpointAddress;
                this.pin = pin;
                this.candidate = candidate;
            }

            private String endpointAddress() {
                return endpointAddress;
            }

            private WorkerEndpoint.GenerationPin pin() {
                return pin;
            }

            private RouteProjection.Candidate candidate() {
                return candidate;
            }
        }

        private final ArrayList<Entry> entries;

        CandidateSet() {
            this(0);
        }

        private CandidateSet(int expectedCapacity) {
            entries = new ArrayList<>(Math.max(0, expectedCapacity));
        }

        private void addEndpoint(
                String endpointAddress,
                WorkerEndpoint.GenerationPin pin) {
            if (!(pin.endpoint() instanceof PrefillEndpoint)) {
                throw new IllegalArgumentException(
                        "Prefill candidate requires Prefill endpoint pin");
            }
            entries.add(new Entry(endpointAddress, pin, null));
        }

        private void addCandidate(
                String endpointAddress,
                WorkerEndpoint.GenerationPin pin,
                RouteProjection.Candidate candidate) {
            entries.add(new Entry(
                    endpointAddress,
                    pin,
                    candidate));
        }

        private RouteProjection.Candidate candidate(int index) {
            RouteProjection.Candidate candidate = entries.get(index).candidate();
            if (candidate == null) {
                throw new IllegalStateException("endpoint has not been projected");
            }
            return candidate;
        }

        private PrefillEndpoint endpoint(int index) {
            return (PrefillEndpoint) requirePin(index).endpoint();
        }

        private String endpointAddress(int index) {
            return entries.get(index).endpointAddress();
        }

        private long cacheHit(int index) {
            return candidate(index).cacheHitTokens();
        }

        private long projectedTtftMs(int index) {
            return candidate(index).projectedTtftMs().orElseThrow();
        }

        private long prefillMs(int index) {
            return candidate(index).incomingPrefillMs();
        }

        private int size() {
            return entries.size();
        }

        private void moveCandidateTo(
                int index,
                CandidateSet target,
                RouteProjection.Candidate projection) {
            Entry source = entries.get(index);
            WorkerEndpoint.GenerationPin pin = requirePin(index);
            target.addCandidate(
                    source.endpointAddress(), pin, projection);
            source.pin = null;
        }

        private WorkerEndpoint.GenerationPin requirePin(int index) {
            WorkerEndpoint.GenerationPin pin = entries.get(index).pin();
            if (pin == null) {
                throw new IllegalStateException(
                        "candidate pin was already consumed index=" + index);
            }
            return pin;
        }

        private WorkerEndpoint.GenerationPin takePin(int index) {
            Entry entry = entries.get(index);
            WorkerEndpoint.GenerationPin owned = requirePin(index);
            entry.pin = null;
            return owned;
        }

        @Override
        public void close() {
            for (int index = 0; index < entries.size(); index++) {
                Entry entry = entries.get(index);
                if (entry.pin() != null) {
                    entry.pin().close();
                    entry.pin = null;
                }
            }
        }
    }
    private CandidateSet evaluateCandidates(
            CandidateSet eligible,
            BalanceContext balanceContext,
            FlexlbConfig config,
            Map<String, Integer> cacheMatchResults,
            Map<String, Integer> rejections,
            Map<RoleType, Integer> poolWideBlockers) {
        Request request = balanceContext.getRequest();
        int eligibleSize = eligible.size();
        CandidateSet modeled = new CandidateSet(eligibleSize);
        CandidateSet unmodeled = new CandidateSet(eligibleSize);

        // Build one coherent projection per live endpoint. Cache
        // hit is part of both the incoming service prediction and batch-group
        // boundary planning; availability consumes this projection's pending
        // count instead of taking a second, potentially contradictory snapshot.
        RouteProjection.Demand projectionDemand = projectionDemand(config);
        try (eligible) {
            for (int i = 0; i < eligibleSize; i++) {
                PrefillEndpoint ep = eligible.endpoint(i);
                String endpointAddress = eligible.endpointAddress(i);
                CacheTokenMatch cacheMatch =
                        calculateCacheMatch(ep, endpointAddress, cacheMatchResults, request);
                long cacheHit = cacheMatch.effectiveHitTokens();
                long routingCacheMatchTokens = cacheMatch.routingHitTokens();
                RouteProjection.Inputs projectionInputs =
                        ep.captureRouteProjectionInputs();
                RouteProjection.Probe probe = new RouteProjection.Probe(
                        request.getRequestId(),
                        balanceContext.getPriority(),
                        projectionInputs.queue().capturedAtMs(),
                        // RequestRegistry owns terminal deadlines.
                        // Selection scores endpoint work only; an expiry race
                        // must not be reported as a capacity blocker.
                        Long.MAX_VALUE,
                        request.getSeqLen(),
                        cacheHit,
                        routingCacheMatchTokens,
                        projectionDemand);
                PrefillTimePredictor predictor = ep.getPredictor();
                RouteProjection.Candidate projection =
                        RouteProjection.project(
                                projectionInputs,
                                probe,
                                predictor == null
                                        ? null : predictor.evaluator(),
                                ep.deliveryProjection());
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
                if (projection.requiredPendingCount()
                        >= config.getRouter().getRoles().getPrefill()
                                .getAvailability().getMaxPendingRequests()) {
                    rejections.merge("RESOURCE_UNAVAILABLE", 1, Integer::sum);
                    continue;
                }
                if (modeledProjection) {
                    eligible.moveCandidateTo(i, modeled, projection);
                } else {
                    eligible.moveCandidateTo(i, unmodeled, projection);
                }
                // One routing request may inspect hundreds of endpoints. Keep the
                // per-candidate diagnostic at TRACE so enabling ordinary DEBUG
                // cannot turn the selection hot path into a log flood.
                if (Logger.isTraceEnabled()) {
                    if (modeledProjection) {
                        OptionalLong projectedDrainMs = projection.projectedDrainMs();
                        Logger.trace(
                                "Prefill projection - ip: {}, order: {}, hitCache: {}, "
                                        + "ttftMs: {}, drainMs: {}",
                                endpointAddress,
                                config.isPriorityOrdering() ? "PRIORITY" : "FIFO",
                                cacheHit,
                                projection.projectedTtftMs(),
                                projectedDrainMs.isPresent()
                                        ? Long.toString(projectedDrainMs.getAsLong())
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
        } catch (RuntimeException | Error failure) {
            modeled.close();
            unmodeled.close();
            throw failure;
        }

        if (modeled.size() != 0) {
            unmodeled.close();
            return rejectOutliers(modeled, config, rejections);
        }
        modeled.close();
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
        long sumDrainMs = 0L;
        int knownDrainCount = 0;
        long sumPendingCount = 0L;
        int feasibleSize = feasible.size();
        for (int i = 0; i < feasibleSize; i++) {
            RouteProjection.Candidate projection = feasible.candidate(i);
            OptionalLong projectedDrainMs = projection.projectedDrainMs();
            if (projectedDrainMs.isPresent()) {
                sumDrainMs = saturatingAdd(
                        sumDrainMs, projectedDrainMs.getAsLong());
                knownDrainCount++;
            }
            sumPendingCount = saturatingAdd(
                    sumPendingCount, projection.requiredPendingCount());
        }

        long avgDrainMs = knownDrainCount == 0 ? 0L : sumDrainMs / knownDrainCount;
        long avgPendingCount = sumPendingCount / feasible.size();

        // Round 2: hotspot / drain-imbalance filter using the same projections.
        CandidateSet survivors = new CandidateSet(feasibleSize);
        int leastLoadedIndex = -1;
        long leastDrainMs = Long.MAX_VALUE;
        int leastPendingIndex = -1;
        long leastPendingCount = Long.MAX_VALUE;
        try (feasible) {
            for (int i = 0; i < feasibleSize; i++) {
                RouteProjection.Candidate projection = feasible.candidate(i);
                OptionalLong projectedDrainMs = projection.projectedDrainMs();
                long pendingCount = projection.requiredPendingCount();

                if (projectedDrainMs.isPresent()
                        && projectedDrainMs.getAsLong() < leastDrainMs) {
                    leastDrainMs = projectedDrainMs.getAsLong();
                    leastLoadedIndex = i;
                }
                if (pendingCount < leastPendingCount) {
                    leastPendingCount = pendingCount;
                    leastPendingIndex = i;
                }

                if (hotspotMultiplier > 0 && avgPendingCount > 0
                        && pendingCount > avgPendingCount * hotspotMultiplier) {
                    rejections.merge("HOTSPOT_FILTERED", 1, Integer::sum);
                    continue;
                }
                if (imbalanceMultiplier > 0 && avgDrainMs > 0
                        && projectedDrainMs.isPresent()
                        && projectedDrainMs.getAsLong()
                        > avgDrainMs * imbalanceMultiplier) {
                    rejections.merge("IMBALANCE_FILTERED", 1, Integer::sum);
                    continue;
                }

                feasible.moveCandidateTo(i, survivors, projection);
            }

            if (survivors.size() == 0) {
                int fallbackIndex = leastLoadedIndex >= 0
                        ? leastLoadedIndex : leastPendingIndex;
                if (fallbackIndex >= 0) {
                    feasible.moveCandidateTo(
                            fallbackIndex,
                            survivors,
                            feasible.candidate(fallbackIndex));
                }
            }
        } catch (RuntimeException | Error failure) {
            survivors.close();
            throw failure;
        }

        return survivors;
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
            RoleType roleType, String group, String excludedIpPort) {
        List<String> addresses =
                workerDirectory.endpointAddressSnapshot(roleType);
        int addressCount = addresses.size();
        if (addressCount == 0) {
            return new EndpointDiscovery(
                    new CandidateSet(), null, excludedIpPort, 0, Map.of());
        }
        int candidateLimit = Math.min(
                MAX_PROJECTED_CANDIDATES, addressCount);
        long cursor = candidateWindowCursor.getAndAdd(candidateLimit);
        int start = (int) Math.floorMod(cursor, addressCount);
        CandidateSet result = new CandidateSet(candidateLimit);
        Map<String, Integer> rejections = new java.util.HashMap<>();
        WorkerEndpoint.GenerationPin excludedPin = null;
        int registeredCount = 0;
        try {
            for (int offset = 0;
                    offset < addressCount && result.size() < candidateLimit;
                    offset++) {
                String address = addresses.get((start + offset) % addressCount);
                WorkerEndpoint.GenerationPin pin =
                        workerDirectory.captureEndpoint(roleType, address);
                if (pin == null) {
                    continue;
                }
                if (!(pin.endpoint() instanceof PrefillEndpoint)) {
                    pin.close();
                    continue;
                }
                WorkerStatus.TopologySnapshot topology =
                        pin.endpoint().getStatus().topologySnapshot();
                if (group != null && !group.equals(topology.group())) {
                    pin.close();
                    continue;
                }
                registeredCount++;
                String ipPort = pin.endpoint().ipPort();
                if (excludedIpPort != null && excludedIpPort.equals(ipPort)) {
                    if (excludedPin != null) {
                        pin.close();
                        throw new IllegalStateException(
                                "duplicate Prefill endpoint address " + ipPort);
                    }
                    excludedPin = pin;
                    rejections.merge("EXCLUDED_RETRY", 1, Integer::sum);
                    continue;
                }
                result.addEndpoint(ipPort, pin);
            }
            return new EndpointDiscovery(
                    result, excludedPin, excludedIpPort,
                    registeredCount, rejections);
        } catch (RuntimeException | Error failure) {
            result.close();
            if (excludedPin != null) {
                excludedPin.close();
            }
            throw failure;
        }
    }

    private Map<String, Integer> getCacheMatchResults(
            BalanceContext balanceContext,
            RoleType roleType,
            String group) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(
                blockCacheKeys, roleType, group);
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
            CandidateSet owner,
            int selectedIndex,
            RoleType roleType,
            long requestId,
            OptionalLong projectedTtftMs,
            long selectedPrefillMs,
            long bestCacheHit) {
        // Populate DebugInfo so ScheduledRequest.hitCache() can read hitCacheLen for batch metrics
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(bestCacheHit);

        PrefillEndpoint ep = owner.endpoint(selectedIndex);
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
        return SelectedRole.prefill(owner.takePin(selectedIndex), result, selectedPrefillMs);
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
