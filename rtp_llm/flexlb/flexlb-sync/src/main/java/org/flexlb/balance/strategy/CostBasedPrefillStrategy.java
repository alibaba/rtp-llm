package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.EngineGeneration;
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
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.concurrent.ThreadLocalRandom;

@Component
public class CostBasedPrefillStrategy implements LoadBalanceStrategy {

    private final EngineWorkerStatus engineWorkerStatus;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;

    public CostBasedPrefillStrategy(EngineWorkerStatus engineWorkerStatus,
                                    CacheAwareService cacheAwareService,
                                    ResourceMeasureFactory resourceMeasureFactory,
                                    EngineHealthReporter engineHealthReporter) {
        this.engineWorkerStatus = engineWorkerStatus;
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
    public SelectedRole select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } finally {
            releasePerSelectionState();
        }
    }

    /** Subclasses clear request-scoped ThreadLocal references here. */
    protected void releasePerSelectionState() {
    }

    private SelectedRole doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        EndpointDiscovery discovery = discoverAliveEndpoints(
                roleType, group, balanceContext.getExcludedPrefillIpPort());
        if (discovery.registeredCount() == 0) {
            Logger.debug("Prefill select failed: no registered endpoints, request_id={}",
                    requestId);
            discovery.close();
            return null;
        }
        PrefillResourceMeasure resourceMeasure = (PrefillResourceMeasure)
                resourceMeasureFactory.getMeasure(config.resourceMeasureFor(roleType));
        if (resourceMeasure == null) {
            Logger.debug("Prefill select failed: no resource measure, request_id={}", requestId);
            discovery.close();
            return null;
        }

        try (discovery) {
            Map<EngineGeneration, CacheMatch> cacheMatchResults =
                    getCacheMatchResults(balanceContext, roleType, discovery);
            Map<String, Integer> rejections = new java.util.HashMap<>(discovery.rejections());
            FilterResult filterResult =
                    evaluateCandidates(
                            discovery.takePreferredEndpoints(),
                            balanceContext,
                            config,
                            cacheMatchResults,
                            resourceMeasure);
            filterResult
                    .rejections()
                    .forEach((reason, count) -> rejections.merge(reason, count, Integer::sum));

            CandidateSet survivors = filterResult.candidates();
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
                filterResult =
                        evaluateCandidates(
                                excluded,
                                balanceContext,
                                config,
                                cacheMatchResults,
                                resourceMeasure);
                filterResult
                        .rejections()
                        .forEach((reason, count) -> rejections.merge(reason, count, Integer::sum));
                survivors = filterResult.candidates();
            }
            CandidateSet selectedCandidates = survivors;
            try (selectedCandidates) {
                if (selectedCandidates.size() == 0) {
                    Logger.debug(
                            "Prefill select failed: no available endpoints, request_id={},"
                                + " rejections={}",
                            requestId,
                            rejections);
                    return null;
                }

                boolean modeledSelection =
                        selectedCandidates.candidate(0).projection().selectable();
                long minProjectedTtftMs = Long.MAX_VALUE;
                long selectionNowMs = System.currentTimeMillis();
                RoutingConfig.CacheAffinityConfig cacheAffinity =
                        config.getRouter().getRoles().getPrefill().getCacheAffinity();
                long remainingAffinityBudgetMs = cacheAffinity == null
                        ? 0L
                        : remainingCacheAffinityBudgetMs(
                                balanceContext,
                                cacheAffinity.getMaxExtraTtftMs(),
                                selectionNowMs);
                final int selectedIndex;
                if (modeledSelection) {
                    // Numeric TTFT policies see only candidates with a complete model.
                    boolean hasAvailable = selectedCandidates.hasAvailable();
                    for (int i = 0; i < selectedCandidates.size(); i++) {
                        if (hasAvailable
                                && !selectedCandidates.resourceAvailable(i)) {
                            continue;
                        }
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
                                    config,
                                    remainingAffinityBudgetMs);
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
                    return null;
                }

                PrefillEndpoint best = selectedCandidates.endpoint(selectedIndex);
                RouteProjection.Candidate selectedCandidate = selectedCandidates.candidate(selectedIndex);
                long bestCacheHit = selectedCandidates.cacheHit(selectedIndex);
                long selectedPrefillMs = selectedCandidates.prefillMs(selectedIndex);
                long reselectNotAfterMs = modeledSelection
                        ? cacheReselectNotAfterMs(
                                selectedCandidates,
                                selectedIndex,
                                minProjectedTtftMs,
                                balanceContext,
                                config,
                                remainingAffinityBudgetMs,
                                selectionNowMs)
                        : Long.MAX_VALUE;
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
                                selectedCandidate.projection().projectedTtftMs(),
                                selectedPrefillMs,
                                bestCacheHit,
                                reselectNotAfterMs);
                return selectedRole;
            }
        }
    }

    /**
     * Select an availability fallback by coherent pending ownership, then live LRU. The clock is a
     * fairness hint only; races may change ordering but can never turn a non-empty candidate set
     * into an availability failure.
     */
    private static int selectUnmodeledCandidate(CandidateSet candidates) {
        boolean hasAvailable = candidates.hasAvailable();
        int selectedIndex = -1;
        long selectedPending = Long.MAX_VALUE;
        long selectedTime = Long.MAX_VALUE;
        for (int i = 0; i < candidates.size(); i++) {
            if (hasAvailable && !candidates.resourceAvailable(i)) {
                continue;
            }
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

    /**
     * Select from candidates that already passed CostBasedPrefill's hard filters.
     * Subclasses may add a bounded preference and delegate here for exact baseline fallback.
     */
    protected int selectBestCandidate(CandidateSet survivors,
                                      long minProjectedTtftMs,
                                      RoleType roleType,
                                      String group,
                                      long seqLen,
                                      FlexlbConfig config,
                                      long remainingAffinityBudgetMs) {
        if (survivors.size() == 0) {
            return -1;
        }

        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter().getRoles()
                .getPrefill().getCacheAffinity();
        if (cacheAffinity == null) {
            return selectBaselineCandidate(survivors, minProjectedTtftMs, config);
        }

        boolean hasAvailable = survivors.hasAvailable();
        long referenceHitTokens = 0L;
        for (int i = 0; i < survivors.size(); i++) {
            if ((!hasAvailable || survivors.resourceAvailable(i))
                    && survivors.projectedTtftMs(i) == minProjectedTtftMs) {
                referenceHitTokens = Math.max(referenceHitTokens, survivors.cacheHit(i));
            }
        }
        CacheAffinityPolicy.Decision affinity = CacheAffinityPolicy.evaluate(
                survivors.size(),
                index -> remainingAffinityBudgetMs > 0L
                        || !hasAvailable
                        || survivors.resourceAvailable(index),
                survivors::projectedTtftMs,
                survivors::cacheHit,
                minProjectedTtftMs,
                referenceHitTokens,
                seqLen,
                remainingAffinityBudgetMs,
                cacheAffinity.getMinPrefixHitPercent());

        int selectedIndex;
        if (affinity.hasPreference()) {
            selectedIndex = selectCacheLeader(survivors, affinity);
        } else {
            selectedIndex = selectBaselineCandidate(
                    survivors, minProjectedTtftMs, config);
        }

        if (selectedIndex >= 0) {
            String reason = affinity.reason().name();
            reportCacheAffinityDecision(
                    roleType, survivors.endpoint(selectedIndex).getIp(), reason);
            if (Logger.isDebugEnabled()) {
                Logger.debug(
                        "CostBasedPrefill cache-affinity decision - role: {}, group: {}, "
                                + "selected: {}, minProjectedTtftMs: {}, "
                                + "selectedProjectedTtftMs: {}, ttftCutoffMs: {}, "
                                + "hitTokens: {}, reason: {}",
                        roleType,
                        group,
                        survivors.endpointAddress(selectedIndex),
                        affinity.minProjectedTtftMs(),
                        survivors.projectedTtftMs(selectedIndex),
                        affinity.projectedTtftCutoffMs(),
                        survivors.cacheHit(selectedIndex),
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
        RoutingConfig.PrefillSelectorConfig selector = config.getRouter().getRoles()
                .getPrefill().getSelector();
        RoutingConfig.CandidateChoiceConfig candidateChoice =
                ((RoutingConfig.EstimatedTtftSelectorConfig) selector).getCandidateChoice();
        if (candidateChoice instanceof RoutingConfig.RandomWithinToleranceConfig random) {
            long percentageThreshold = (long) (
                    minProjectedTtftMs * random.getRelativeTolerance());
            tieThreshold = Math.max(
                    Math.max(0L, percentageThreshold),
                    Math.max(0L, random.getMinimumToleranceMs()));
        }
        long ttftCutoffMs = saturatingAdd(minProjectedTtftMs, tieThreshold);
        boolean hasAvailable = survivors.hasAvailable();
        int selectedIndex = -1;
        int tiedCount = 0;
        for (int i = 0; i < survivors.size(); i++) {
            if ((!hasAvailable || survivors.resourceAvailable(i))
                    && survivors.projectedTtftMs(i) <= ttftCutoffMs
                    && ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                selectedIndex = i;
            }
        }
        return selectedIndex;
    }

    /** Randomize only endpoints with the same best cache hit and projected TTFT. */
    private int selectCacheLeader(
            CandidateSet survivors, CacheAffinityPolicy.Decision affinity) {
        int firstIndex = affinity.preferredIndex(0);
        long bestHit = survivors.cacheHit(firstIndex);
        long bestProjectedTtftMs = survivors.projectedTtftMs(firstIndex);
        int selectedIndex = firstIndex;
        int tiedCount = 1;
        for (int i = 1; i < affinity.preferredCount(); i++) {
            int candidateIndex = affinity.preferredIndex(i);
            if (survivors.cacheHit(candidateIndex) != bestHit
                    || survivors.projectedTtftMs(candidateIndex)
                    != bestProjectedTtftMs) {
                break;
            }
            if (ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                selectedIndex = candidateIndex;
            }
        }
        return selectedIndex;
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

        private PrefillEndpoint excludedEndpoint() {
            return excludedPin == null
                    ? null : (PrefillEndpoint) excludedPin.endpoint();
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

    protected static final class CandidateSet implements AutoCloseable {
        private static final class Entry {
            private final String endpointAddress;
            private WorkerEndpoint.GenerationPin pin;
            private final RouteProjection.Candidate projection;
            private final boolean resourceAvailable;

            private Entry(
                    String endpointAddress,
                    WorkerEndpoint.GenerationPin pin,
                    RouteProjection.Candidate projection,
                    boolean resourceAvailable) {
                this.endpointAddress = endpointAddress;
                this.pin = pin;
                this.projection = projection;
                this.resourceAvailable = resourceAvailable;
            }

            private String endpointAddress() {
                return endpointAddress;
            }

            private WorkerEndpoint.GenerationPin pin() {
                return pin;
            }

            private RouteProjection.Candidate projection() {
                return projection;
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
            entries.add(new Entry(endpointAddress, pin, null, false));
        }

        private void addCandidate(
                String endpointAddress,
                WorkerEndpoint.GenerationPin pin,
                RouteProjection.Candidate projection,
                boolean resourceAvailable) {
            entries.add(new Entry(
                    endpointAddress,
                    pin,
                    projection,
                    resourceAvailable));
        }

        protected RouteProjection.Candidate candidate(int index) {
            RouteProjection.Candidate projection = entries.get(index).projection();
            if (projection == null) {
                throw new IllegalStateException("endpoint has not been projected");
            }
            return projection;
        }

        protected PrefillEndpoint endpoint(int index) {
            return (PrefillEndpoint) requirePin(index).endpoint();
        }

        protected String endpointAddress(int index) {
            return entries.get(index).endpointAddress();
        }

        protected long cacheHit(int index) {
            return candidate(index).cacheHitTokens();
        }

        protected long projectedTtftMs(int index) {
            return candidate(index).projectedTtftMs();
        }

        protected long prefillMs(int index) {
            return candidate(index).incomingPrefillMs();
        }

        protected boolean resourceAvailable(int index) {
            return entries.get(index).resourceAvailable;
        }

        protected boolean hasAvailable() {
            for (int i = 0; i < entries.size(); i++) {
                if (resourceAvailable(i)) {
                    return true;
                }
            }
            return false;
        }

        protected int size() {
            return entries.size();
        }

        private void moveCandidateTo(
                int index,
                CandidateSet target,
                RouteProjection.Candidate projection,
                boolean resourceAvailable) {
            Entry source = entries.get(index);
            WorkerEndpoint.GenerationPin pin = requirePin(index);
            target.addCandidate(
                    source.endpointAddress(), pin, projection, resourceAvailable);
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
    private record FilterResult(CandidateSet candidates, Map<String, Integer> rejections) {}

    private FilterResult evaluateCandidates(
            CandidateSet eligible,
            BalanceContext balanceContext,
            FlexlbConfig config,
            Map<EngineGeneration, CacheMatch> cacheMatchResults,
            PrefillResourceMeasure resourceMeasure) {
        Request request = balanceContext.getRequest();
        int eligibleSize = eligible.size();
        Map<String, Integer> rejections = new java.util.HashMap<>();
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
                        balanceContext.getRequestExpiresAtMs(),
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
                boolean modeledProjection = projection.projection().selectable();
                boolean unmodeledProjection = projection.engineWorkUnmodeled();
                if (!modeledProjection && !unmodeledProjection) {
                    rejections.merge(
                            "PROJECTION_"
                                    + projection.projection().state().name()
                                    + "_"
                                    + projection.projection().detail(),
                            1,
                            Integer::sum);
                    continue;
                }
                // QUEUE takes an endpoint-owned capacity hold only after
                // cache/TTFT selection. Filtering here would erase a hot
                // worker merely because its binding seats are momentarily full.
                boolean resourceAvailable = resourceMeasure.isResourceAvailable(
                        projection.requiredPendingCount());
                if (!config.isQueue() && !resourceAvailable) {
                    rejections.merge("RESOURCE_UNAVAILABLE", 1, Integer::sum);
                    continue;
                }
                if (modeledProjection) {
                    eligible.moveCandidateTo(
                            i, modeled, projection, resourceAvailable);
                } else {
                    eligible.moveCandidateTo(
                            i, unmodeled, projection, resourceAvailable);
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

        if (modeled.hasAvailable()
                || modeled.size() != 0 && !unmodeled.hasAvailable()) {
            unmodeled.close();
            return rejectOutliers(modeled, config, rejections);
        }
        modeled.close();
        return new FilterResult(unmodeled, rejections);
    }

    /** Cache affinity spends one request-scoped budget across every reselect. */
    protected static long remainingCacheAffinityBudgetMs(
            BalanceContext context,
            long configuredBudgetMs,
            long nowMs) {
        long budgetMs = Math.max(0L, configuredBudgetMs);
        long startedAtMs = context.getEnqueueTime();
        if (startedAtMs <= 0L) {
            return budgetMs;
        }
        long elapsedMs = Math.max(
                0L, nowMs - startedAtMs);
        return Math.max(0L, budgetMs - elapsedMs);
    }

    /**
     * Return one absolute, policy-neutral refresh time only when elapsed
     * affinity budget can change the selected affinity decision. The
     * scheduler sees neither cache scores nor policy reasons.
     */
    private static long cacheReselectNotAfterMs(
            CandidateSet candidates,
            int selectedIndex,
            long baselineTtftMs,
            BalanceContext context,
            FlexlbConfig config,
            long remainingAffinityBudgetMs,
            long selectionNowMs) {
        RoutingConfig.CacheAffinityConfig affinity = config.getRouter().getRoles()
                .getPrefill().getCacheAffinity();
        if (!config.isQueue()
                || affinity == null
                || candidates.projectedTtftMs(selectedIndex) <= baselineTtftMs) {
            return Long.MAX_VALUE;
        }
        long minimumHit = Long.MAX_VALUE;
        for (int index = 0; index < candidates.size(); index++) {
            minimumHit = Math.min(minimumHit, candidates.cacheHit(index));
        }
        if (candidates.cacheHit(selectedIndex) <= minimumHit) {
            return Long.MAX_VALUE;
        }
        long budgetMs = Math.max(0L, affinity.getMaxExtraTtftMs());
        if (budgetMs == 0L
                || remainingAffinityBudgetMs == 0L) {
            return Long.MAX_VALUE;
        }
        long startedAtMs = context.getEnqueueTime() > 0L
                ? context.getEnqueueTime()
                : selectionNowMs;
        return saturatingAdd(startedAtMs, budgetMs);
    }

    private FilterResult rejectOutliers(
            CandidateSet feasible,
            FlexlbConfig config,
            Map<String, Integer> rejections) {
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig) config.getRouter().getRoles()
                        .getPrefill().getSelector();
        RoutingConfig.OutlierRejectionConfig outlier =
                outlierRejection(selector.getCandidateChoice());
        double hotspotMultiplier = outlier == null
                ? 0.0 : outlier.getMaxPendingVsAverageMultiplier();
        double imbalanceMultiplier = outlier == null
                ? 0.0 : outlier.getMaxProjectedDrainVsAverageMultiplier();
        if (hotspotMultiplier <= 0.0 && imbalanceMultiplier <= 0.0) {
            return new FilterResult(feasible, rejections);
        }
        long sumDrainMs = 0L;
        int knownDrainCount = 0;
        long sumPendingCount = 0L;
        int feasibleSize = feasible.size();
        int availableCount = 0;
        for (int i = 0; i < feasibleSize; i++) {
            if (!feasible.resourceAvailable(i)) {
                continue;
            }
            availableCount++;
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
        if (availableCount == 0) {
            return new FilterResult(feasible, rejections);
        }

        long avgDrainMs = knownDrainCount == 0 ? 0L : sumDrainMs / knownDrainCount;
        long avgPendingCount = sumPendingCount / availableCount;

        // Round 2: hotspot / drain-imbalance filter using the same projections.
        CandidateSet survivors = new CandidateSet(feasibleSize);
        int leastLoadedIndex = -1;
        long leastDrainMs = Long.MAX_VALUE;
        int leastPendingIndex = -1;
        long leastPendingCount = Long.MAX_VALUE;
        int availableSurvivors = 0;
        try (feasible) {
            for (int i = 0; i < feasibleSize; i++) {
                RouteProjection.Candidate projection = feasible.candidate(i);
                if (!feasible.resourceAvailable(i)) {
                    feasible.moveCandidateTo(
                            i, survivors, projection, false);
                    continue;
                }
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

                feasible.moveCandidateTo(i, survivors, projection, true);
                availableSurvivors++;
            }

            if (availableSurvivors == 0) {
                int fallbackIndex = leastLoadedIndex >= 0
                        ? leastLoadedIndex : leastPendingIndex;
                if (fallbackIndex >= 0) {
                    feasible.moveCandidateTo(
                            fallbackIndex,
                            survivors,
                            feasible.candidate(fallbackIndex),
                            true);
                }
            }
        } catch (RuntimeException | Error failure) {
            survivors.close();
            throw failure;
        }

        return new FilterResult(survivors, rejections);
    }

    /**
     * Drain is a cost-policy input only when drain-imbalance rejection is live.
     * Pending-count filtering and final TTFT choice do not consume it.
     */
    protected RouteProjection.Demand projectionDemand(FlexlbConfig config) {
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig) config.getRouter().getRoles()
                        .getPrefill().getSelector();
        RoutingConfig.OutlierRejectionConfig outlier =
                outlierRejection(selector.getCandidateChoice());
        return outlier != null
                && outlier.getMaxProjectedDrainVsAverageMultiplier() > 0.0
                ? RouteProjection.Demand.TTFT_AND_DRAIN
                : RouteProjection.Demand.TTFT_ONLY;
    }

    private static RoutingConfig.OutlierRejectionConfig outlierRejection(
            RoutingConfig.CandidateChoiceConfig candidateChoice) {
        return candidateChoice instanceof RoutingConfig.RandomWithinToleranceConfig random
                ? random.getOutlierRejection()
                : candidateChoice instanceof RoutingConfig.BestOnlyConfig best
                        ? best.getOutlierRejection() : null;
    }

    private EndpointDiscovery discoverAliveEndpoints(
            RoleType roleType, String group, String excludedIpPort) {
        List<WorkerEndpoint.GenerationPin> captured =
                engineWorkerStatus.captureModelWorkerEndpoints(roleType, group);
        CandidateSet result = new CandidateSet(captured.size());
        Map<String, Integer> rejections = new java.util.HashMap<>();
        WorkerEndpoint.GenerationPin excludedPin = null;
        try {
            for (int index = 0; index < captured.size(); index++) {
                WorkerEndpoint.GenerationPin pin = captured.get(index);
                if (!(pin.endpoint() instanceof PrefillEndpoint)) {
                    pin.close();
                    captured.set(index, null);
                    continue;
                }
                String ipPort = pin.endpoint().ipPort();
                if (excludedIpPort != null && excludedIpPort.equals(ipPort)) {
                    if (excludedPin != null) {
                        throw new IllegalStateException(
                                "duplicate Prefill endpoint address " + ipPort);
                    }
                    excludedPin = pin;
                    captured.set(index, null);
                    rejections.merge("EXCLUDED_RETRY", 1, Integer::sum);
                    continue;
                }
                result.addEndpoint(ipPort, pin);
                captured.set(index, null);
            }
            return new EndpointDiscovery(
                    result, excludedPin, excludedIpPort,
                    captured.size(), rejections);
        } catch (RuntimeException | Error failure) {
            result.close();
            if (excludedPin != null) {
                excludedPin.close();
            }
            for (WorkerEndpoint.GenerationPin pin : captured) {
                if (pin != null) {
                    pin.close();
                }
            }
            throw failure;
        }
    }

    private Map<EngineGeneration, CacheMatch> getCacheMatchResults(
            BalanceContext balanceContext,
            RoleType roleType,
            EndpointDiscovery discovery) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Map.of();
        }
        List<EngineGeneration> candidates = new ArrayList<>(
                discovery.preferredEndpoints().size()
                        + (discovery.excludedEndpoint() == null ? 0 : 1));
        CandidateSet preferred = discovery.preferredEndpoints();
        for (int index = 0; index < preferred.size(); index++) {
            candidates.add(engineGeneration(
                    preferred.endpointAddress(index), preferred.endpoint(index)));
        }
        if (discovery.excludedEndpoint() != null) {
            candidates.add(engineGeneration(
                    discovery.excludedIpPort(), discovery.excludedEndpoint()));
        }
        return cacheAwareService.findMatchingEngines(
                blockCacheKeys, roleType, candidates);
    }

    private static EngineGeneration engineGeneration(
            String endpointAddress,
            PrefillEndpoint endpoint) {
        WorkerStatus status = endpoint.getStatus();
        if (status == null) {
            throw new IllegalStateException(
                    "Prefill endpoint has no WorkerStatus: " + endpointAddress);
        }
        return new EngineGeneration(
                endpointAddress, status.getGenerationId());
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
            Map<EngineGeneration, CacheMatch> cacheMatchResults,
            Request request) {
        if (cacheMatchResults == null || cacheMatchResults.isEmpty() || request == null) {
            return CacheTokenMatch.NONE;
        }
        long seqLen = request.getSeqLen();
        if (seqLen <= 0L) {
            return CacheTokenMatch.NONE;
        }
        EngineGeneration exactGeneration = engineGeneration(
                endpointAddress, ep);
        CacheMatch cacheMatch = cacheMatchResults.get(exactGeneration);
        if (cacheMatch == null
                || cacheMatch.prefixMatchLength() <= 0) {
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
                    blockSize, (long) cacheMatch.prefixMatchLength());
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
            long bestCacheHit,
            long reselectNotAfterMs) {
        // Populate DebugInfo so BatchItem.hitCache() can read hitCacheLen for batch metrics
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
        return SelectedRole.prefill(
                owner.takePin(selectedIndex),
                result,
                selectedPrefillMs,
                reselectNotAfterMs);
    }

    protected void reportCacheAffinityDecision(
            RoleType roleType, String engineIp, String decision) {
        engineHealthReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    protected static long saturatingAdd(long left, long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        if (right < 0L && left < Long.MIN_VALUE - right) {
            return Long.MIN_VALUE;
        }
        return left + right;
    }

}
