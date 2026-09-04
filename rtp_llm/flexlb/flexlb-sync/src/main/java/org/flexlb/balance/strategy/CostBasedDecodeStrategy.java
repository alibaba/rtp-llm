package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeEndpoint.DecodeRoutingView;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class CostBasedDecodeStrategy {

    private static final int SNAPSHOT_CAPTURE_ATTEMPTS = 2;
    private static final ThreadLocal<CandidateBuffer> CANDIDATES =
            ThreadLocal.withInitial(CandidateBuffer::new);

    private final WorkerDirectory workerDirectory;
    private final AtomicLong equalCostCursor = new AtomicLong();

    public CostBasedDecodeStrategy(WorkerDirectory workerDirectory) {
        this.workerDirectory = workerDirectory;
    }

    public PlacementResult<SelectedRole, RoleType> select(
            BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();
        RoutingConfig.DecodeConfig selector = config.getRouter()
                .getRoles().getDecode();

        // Queues without preemption retain transient Decode pressure at the
        // exact pre-delivery permit. Only preemptive placement needs a miss here
        // in order to enter victim planning.
        boolean softQueuePlacement =
                config.defersDecodeCapacityUntilDispatch();
        for (int attempt = 0; attempt < SNAPSHOT_CAPTURE_ATTEMPTS; attempt++) {
            List<DecodeRoutingView> snapshots =
                    workerDirectory.decodeRoutingSnapshot(group);
            int registered = snapshots.size();
            if (registered == 0) {
                logNoAvailableEndpoint(
                        balanceContext, 0, Map.of("NO_REGISTERED", 1));
                return PlacementResult.blocked(roleType);
            }
            CandidateBuffer candidates = CANDIDATES.get();
            captureCandidates(
                    candidates, snapshots, seqLen,
                    selector, softQueuePlacement, config);
            Response staticRejection = validateFleet(candidates, seqLen);
            if (staticRejection != null) {
                return PlacementResult.rejected(staticRejection);
            }

            if (candidates.availabilityEligible == 0) {
                logNoAvailableEndpoint(
                        balanceContext, registered,
                        rejectionMap("RESOURCE_UNAVAILABLE",
                                candidates.availabilityRejected));
                return PlacementResult.blocked(roleType);
            }

            if (candidates.isEmpty()) {
                logAllFilteredOut(
                        balanceContext,
                        candidates);
                return PlacementResult.blocked(roleType);
            }
            double kvDecay = selector.getDecayPerToken();
            double loadDecay = selector.getLoadDecayPerRequest();
            if (softQueuePlacement) {
                preferImmediatelyDispatchable(
                        candidates, balanceContext, kvDecay, loadDecay);
            }
            DecodeRoutingView selected = weightedRandomSelection(
                    candidates, kvDecay, loadDecay);
            if (selected == null) {
                logAllFilteredOut(balanceContext, candidates);
                return PlacementResult.blocked(roleType);
            }

            WorkerEndpoint.GenerationPin pin =
                    workerDirectory.captureDecodeGeneration(selected);
            if (pin != null) {
                return PlacementResult.success(buildSelectedRole(
                        selected, pin, roleType, balanceContext));
            }
        }

        Logger.debug(
                "Decode snapshot winner changed repeatedly; retry placement,"
                    + " request_id={}",
                balanceContext.getRequestId());
        return PlacementResult.blocked(roleType);
    }

    private Response validateFleet(
            CandidateBuffer snapshots,
            long requiredKv) {
        if (!snapshots.physicalKvUnknown
                && Math.max(0L, requiredKv)
                        > snapshots.maximumPhysicalKv) {
            return staticCapacityFailure(
                    requiredKv, snapshots.maximumPhysicalKv);
        }
        // Selection policies need the complete live fleet. A rotating subset
        // can hide the endpoint with the best load/KV state and makes the
        // result depend on request timing rather than the captured snapshot.
        return null;
    }

    private static Response staticCapacityFailure(
            long requiredKv,
            long maximumPhysicalKv) {
        Response response = Response.error(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        String detail = "Decode request seq_len=" + requiredKv
                + " exceeds max known physical KV=" + maximumPhysicalKv;
        response.setErrorMessage(StrategyErrorType.RESOURCE_EXHAUSTED
                .buildErrorMessage(detail));
        return response;
    }

    static boolean hasDecodeCapacity(
            FlexlbConfig config,
            DecodeEndpoint.DecodeRoutingView view,
            boolean engineFacing) {
        if (view == null) {
            return false;
        }
        var availability = config.getRouter().getRoles()
                .getDecode().getAvailability();
        Long configuredLimit = availability.getMaxEngineRequests();
        long engineLoad = engineFacing
                ? view.engineCapacityUsed() : view.engineLoad();
        if (configuredLimit != null
                && configuredLimit > 0
                && engineLoad >= configuredLimit) {
            return false;
        }
        long used = engineFacing
                ? view.engineFacingKvUsed() : view.realKvUsed();
        return view.totalKv() == 0
                || used * 100.0 / view.totalKv()
                        < availability.getMaxKvUsagePercent();
    }

    /**
     * Narrow to endpoints that can dispatch immediately when that set is
     * non-empty. If all endpoints are busy, queued placement keeps the full
     * hard-valid domain and waits on the chosen endpoint's exact capacity
     * edge. This is one candidate-domain rule, not a second selection pass.
     */
    private static void preferImmediatelyDispatchable(
            CandidateBuffer eligible,
            BalanceContext context,
            double kvDecay,
            double loadDecay) {
        var availability = context.getConfig().getRouter().getRoles()
                .getDecode().getAvailability();
        Long configuredMaximumRequests = availability.getMaxEngineRequests();
        DecodeEndpoint.AdmissionCapacity capacity =
                new DecodeEndpoint.AdmissionCapacity(
                        configuredMaximumRequests == null
                                ? 0L : configuredMaximumRequests,
                        availability.getMaxKvUsagePercent());
        long hardKv = Math.max(0L, context.getRequest().getSeqLen());
        long maxNewTokens = context.getRequest().getMaxNewTokens();
        int availableCount = 0;
        long allMinKv = Long.MAX_VALUE;
        long allMaxKv = Long.MIN_VALUE;
        int allMinLoad = Integer.MAX_VALUE;
        int allMaxLoad = Integer.MIN_VALUE;
        double allMaximumLogWeight = Double.NEGATIVE_INFINITY;
        long availableMinKv = Long.MAX_VALUE;
        long availableMaxKv = Long.MIN_VALUE;
        int availableMinLoad = Integer.MAX_VALUE;
        int availableMaxLoad = Integer.MIN_VALUE;
        double availableMaximumLogWeight = Double.NEGATIVE_INFINITY;
        long previousTotalKv = Long.MIN_VALUE;
        long expectedKv = 0L;
        for (int index = 0; index < eligible.size; index++) {
            DecodeRoutingView view = eligible.values[index];
            long used = view.realKvUsed();
            int load = view.totalLoad();
            double logWeight = rawLogWeight(view, kvDecay, loadDecay);
            allMinKv = Math.min(allMinKv, used);
            allMaxKv = Math.max(allMaxKv, used);
            allMinLoad = Math.min(allMinLoad, load);
            allMaxLoad = Math.max(allMaxLoad, load);
            allMaximumLogWeight = Math.max(
                    allMaximumLogWeight, logWeight);
            if (view.totalKv() != previousTotalKv) {
                previousTotalKv = view.totalKv();
                expectedKv = Math.max(
                        hardKv,
                        context.getConfig().decodeKvReservationTokens(
                                hardKv, maxNewTokens, previousTotalKv));
            }
            if (DecodeEndpoint.canAcquireEngineDispatchPermit(
                    view, hardKv, expectedKv, capacity)) {
                eligible.values[availableCount++] = view;
                availableMinKv = Math.min(availableMinKv, used);
                availableMaxKv = Math.max(availableMaxKv, used);
                availableMinLoad = Math.min(availableMinLoad, load);
                availableMaxLoad = Math.max(availableMaxLoad, load);
                availableMaximumLogWeight = Math.max(
                        availableMaximumLogWeight, logWeight);
            }
        }
        if (availableCount > 0) {
            eligible.size = availableCount;
            eligible.setCostRange(
                    availableMinKv, availableMaxKv,
                    availableMinLoad, availableMaxLoad,
                    availableMaximumLogWeight);
        } else {
            eligible.setCostRange(
                    allMinKv, allMaxKv,
                    allMinLoad, allMaxLoad,
                    allMaximumLogWeight);
        }
    }

    /** Capture and hard-filter the full fleet in one coherent traversal. */
    private static void captureCandidates(
            CandidateBuffer eligible,
            List<DecodeRoutingView> snapshots,
            long seqLen,
            RoutingConfig.DecodeConfig selector,
            boolean softQueuePlacement,
            FlexlbConfig config) {
        eligible.beginCapture(snapshots.size());
        RoutingConfig.DecodeOutlierRejectionConfig outlier = selector.getOutlierRejection();
        double hotspotMultiplier = outlier == null
                ? 0.0 : outlier.getMaxEngineLoadVsAverageMultiplier();
        double imbalanceMultiplier = outlier == null
                ? 0.0 : outlier.getMaxKvUsedVsAverageMultiplier();

        long sumLoad = 0;
        long sumCacheUsed = 0;
        int maximumSurvivorLoad = 0;
        long maximumSurvivorCacheUsed = 0L;
        int capacitySurvivors = 0;
        int availabilityEligible = 0;
        int availabilityRejected = 0;
        int capacityRejected = 0;
        for (int index = 0; index < snapshots.size(); index++) {
            DecodeRoutingView view = snapshots.get(index);
            eligible.observePhysicalKv(view.totalKv());
            if (!softQueuePlacement
                    && !hasDecodeCapacity(config, view, false)) {
                availabilityRejected++;
                continue;
            }
            availabilityEligible++;
            sumLoad += view.engineLoad();
            sumCacheUsed += view.realKvUsed();

            long totalKv = view.totalKv();
            long availableKv = softQueuePlacement
                    ? totalKv : view.realKvAvailable();
            if (totalKv > 0 && availableKv < seqLen) {
                capacityRejected++;
                continue;
            }
            eligible.values[capacitySurvivors++] = view;
            maximumSurvivorLoad = Math.max(
                    maximumSurvivorLoad, view.engineLoad());
            maximumSurvivorCacheUsed = Math.max(
                    maximumSurvivorCacheUsed, view.realKvUsed());
        }
        eligible.size = capacitySurvivors;
        eligible.availabilityEligible = availabilityEligible;
        eligible.availabilityRejected = availabilityRejected;
        eligible.capacityRejected = capacityRejected;

        if (availabilityEligible == 0) {
            return;
        }
        long avgLoad = sumLoad / availabilityEligible;
        long avgCacheUsed = sumCacheUsed / availabilityEligible;

        boolean filterHotspots = hotspotMultiplier > 0 && avgLoad > 0
                && maximumSurvivorLoad > avgLoad * hotspotMultiplier;
        boolean filterImbalance = imbalanceMultiplier > 0 && avgCacheUsed > 0
                && maximumSurvivorCacheUsed
                        > avgCacheUsed * imbalanceMultiplier;
        if (eligible.isEmpty()
                || (!filterHotspots && !filterImbalance)) {
            return;
        }

        int survivorCount = 0;
        int hotspotRejected = 0;
        int imbalanceRejected = 0;
        for (int index = 0; index < eligible.size; index++) {
            DecodeRoutingView view = eligible.values[index];
            if (filterHotspots
                    && view.engineLoad() > avgLoad * hotspotMultiplier) {
                hotspotRejected++;
                continue;
            }
            long cacheUsed = view.realKvUsed();
            if (filterImbalance
                    && cacheUsed > avgCacheUsed * imbalanceMultiplier) {
                imbalanceRejected++;
                continue;
            }
            eligible.values[survivorCount++] = view;
        }
        eligible.size = survivorCount;
        eligible.hotspotRejected = hotspotRejected;
        eligible.imbalanceRejected = imbalanceRejected;
    }

    private DecodeRoutingView weightedRandomSelection(
            CandidateBuffer candidates,
            double kvDecay,
            double loadDecay) {
        if (candidates.isEmpty()) {
            return null;
        }

        int n = candidates.size;
        if (!candidates.costRangeReady) {
            candidates.computeCostRange(kvDecay, loadDecay);
        }

        if ((candidates.minimumCacheUsed == candidates.maximumCacheUsed
                        || kvDecay == 0.0)
                && (candidates.minimumLoad == candidates.maximumLoad
                        || loadDecay == 0.0)) {
            int index = (int) Math.floorMod(
                    equalCostCursor.getAndIncrement(), (long) n);
            return candidates.values[index];
        }

        // Normalize once into a reusable primitive buffer. This retains the
        // exact categorical distribution while avoiding the two logarithms
        // per endpoint required by Gumbel-max.
        ThreadLocalRandom random = ThreadLocalRandom.current();
        double[] weights = candidates.weights;
        double totalWeight = 0.0;
        long loadRange = (long) candidates.maximumLoad
                - candidates.minimumLoad;
        if (candidates.minimumCacheUsed == candidates.maximumCacheUsed
                && loadRange >= 0L
                && loadRange < n) {
            // Queue traffic commonly leaves equal KV ownership and only a few
            // discrete request-count tiers. Sampling a tier by its aggregate
            // weight and then one member within that tier is the same
            // categorical distribution as expanding every equal weight.
            int tierCount = (int) loadRange + 1;
            java.util.Arrays.fill(
                    candidates.tierCounts, 0, tierCount, 0);
            for (int i = 0; i < n; i++) {
                int tier = candidates.values[i].totalLoad()
                        - candidates.minimumLoad;
                candidates.tierCounts[tier]++;
            }
            for (int tier = 0; tier < tierCount; tier++) {
                int members = candidates.tierCounts[tier];
                if (members == 0) {
                    continue;
                }
                double weight = Math.exp(rawLogWeight(
                        candidates.values[0].realKvUsed(),
                        candidates.minimumLoad + tier,
                        kvDecay, loadDecay)
                        - candidates.maximumLogWeight);
                candidates.tierWeights[tier] = weight;
                totalWeight += members * weight;
            }
            double target = random.nextDouble(totalWeight);
            for (int tier = 0; tier < tierCount; tier++) {
                int members = candidates.tierCounts[tier];
                if (members == 0) {
                    continue;
                }
                double weight = candidates.tierWeights[tier];
                double tierWeight = members * weight;
                if (target < tierWeight) {
                    int memberOffset = Math.min(
                            members - 1, (int) (target / weight));
                    int selectedLoad = candidates.minimumLoad + tier;
                    for (int index = 0; index < n; index++) {
                        if (candidates.values[index].totalLoad()
                                        == selectedLoad
                                && memberOffset-- == 0) {
                            return candidates.values[index];
                        }
                    }
                    throw new IllegalStateException(
                            "Decode tier count changed during selection");
                }
                target -= tierWeight;
            }
            return candidates.values[n - 1];
        }

        for (int i = 0; i < n; i++) {
            double weight = Math.exp(rawLogWeight(
                    candidates.values[i], kvDecay, loadDecay)
                    - candidates.maximumLogWeight);
            weights[i] = weight;
            totalWeight += weight;
        }
        double target = random.nextDouble(totalWeight);
        for (int i = 0; i < n - 1; i++) {
            target -= weights[i];
            if (target < 0.0) {
                return candidates.values[i];
            }
        }
        return candidates.values[n - 1];
    }

    private static double rawLogWeight(
            DecodeRoutingView candidate,
            double kvDecay,
            double loadDecay) {
        return rawLogWeight(candidate.realKvUsed(), candidate.totalLoad(),
                kvDecay, loadDecay);
    }

    private static double rawLogWeight(
            long realKvUsed,
            int totalLoad,
            double kvDecay,
            double loadDecay) {
        return -kvDecay * realKvUsed - loadDecay * totalLoad;
    }

    private SelectedRole buildSelectedRole(
            DecodeRoutingView selected,
            WorkerEndpoint.GenerationPin selectedPin,
            RoleType roleType,
            BalanceContext balanceContext) {
        try {
            if (selectedPin.generationId() != selected.generationId()
                    || !(selectedPin.endpoint() instanceof DecodeEndpoint)) {
                throw new IllegalStateException(
                        "Decode snapshot pin changed before selection handoff");
            }
            WorkerStatus.TopologySnapshot topology = selected.topology();
            DecodeEndpoint.DecodeRoutingView routing = selected;
            WorkerStatus.EngineObservation status =
                    routing.workerStatus().fields();
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(roleType);
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setGroup(topology.group());
            result.setRequestId(balanceContext.getRequestId());

            // SelectedRole consumes the pin even if its validation rejects.
            WorkerEndpoint.GenerationPin factoryPin = selectedPin;
            selectedPin = null;
            return SelectedRole.decode(
                    factoryPin, result, routing.totalKv(),
                    routing.admissionVersion());
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }

    private static void logNoAvailableEndpoint(
            BalanceContext balanceContext,
            int registered,
            Map<String, Integer> rejections) {
        Logger.debug(
                "Decode select failed: no available endpoints, request_id={}, registered={},"
                    + " eligible=0, rejections={}",
                balanceContext.getRequestId(),
                registered,
                rejections);
    }

    private static void logAllFilteredOut(
            BalanceContext balanceContext,
            CandidateBuffer candidates) {
        Map<String, Integer> merged = new HashMap<>();
        putRejection(merged, "RESOURCE_UNAVAILABLE",
                candidates.availabilityRejected);
        putRejection(merged, "KV_CAPACITY", candidates.capacityRejected);
        putRejection(merged, "HOTSPOT_FILTERED",
                candidates.hotspotRejected);
        putRejection(merged, "IMBALANCE_FILTERED",
                candidates.imbalanceRejected);
        Logger.debug(
                "Decode select failed: all filtered out, request_id={}, rejections={}",
                balanceContext.getRequestId(), merged);
    }

    private static Map<String, Integer> rejectionMap(
            String reason, int count) {
        return count > 0 ? Map.of(reason, count) : Map.of();
    }

    private static void putRejection(
            Map<String, Integer> rejections,
            String reason,
            int count) {
        if (count > 0) {
            rejections.put(reason, count);
        }
    }

    /** Per-planner reusable full-fleet columns and compaction workspace. */
    private static final class CandidateBuffer {
        private DecodeRoutingView[] values = new DecodeRoutingView[0];
        private double[] weights = new double[0];
        private double[] tierWeights = new double[0];
        private int[] tierCounts = new int[0];
        private int size;
        private int availabilityEligible;
        private int availabilityRejected;
        private int capacityRejected;
        private int hotspotRejected;
        private int imbalanceRejected;
        private long maximumPhysicalKv;
        private boolean physicalKvUnknown;
        private boolean costRangeReady;
        private long minimumCacheUsed;
        private long maximumCacheUsed;
        private int minimumLoad;
        private int maximumLoad;
        private double maximumLogWeight;

        private void beginCapture(int expected) {
            ensureCapacity(expected);
            size = 0;
            availabilityEligible = 0;
            availabilityRejected = 0;
            capacityRejected = 0;
            hotspotRejected = 0;
            imbalanceRejected = 0;
            maximumPhysicalKv = 0L;
            physicalKvUnknown = false;
            costRangeReady = false;
        }

        private void observePhysicalKv(long physicalKv) {
            if (physicalKv <= 0L) {
                physicalKvUnknown = true;
            } else {
                maximumPhysicalKv = Math.max(
                        maximumPhysicalKv, physicalKv);
            }
        }

        private void computeCostRange(double kvDecay, double loadDecay) {
            long minKv = values[0].realKvUsed();
            long maxKv = minKv;
            int minLoad = values[0].totalLoad();
            int maxLoad = minLoad;
            double maxLogWeight = rawLogWeight(
                    values[0], kvDecay, loadDecay);
            for (int index = 1; index < size; index++) {
                DecodeRoutingView candidate = values[index];
                long used = candidate.realKvUsed();
                int load = candidate.totalLoad();
                minKv = Math.min(minKv, used);
                maxKv = Math.max(maxKv, used);
                minLoad = Math.min(minLoad, load);
                maxLoad = Math.max(maxLoad, load);
                maxLogWeight = Math.max(
                        maxLogWeight,
                        rawLogWeight(candidate, kvDecay, loadDecay));
            }
            setCostRange(minKv, maxKv, minLoad, maxLoad, maxLogWeight);
        }

        private void setCostRange(
                long minKv,
                long maxKv,
                int minLoad,
                int maxLoad,
                double maxLogWeight) {
            minimumCacheUsed = minKv;
            maximumCacheUsed = maxKv;
            minimumLoad = minLoad;
            maximumLoad = maxLoad;
            maximumLogWeight = maxLogWeight;
            costRangeReady = true;
        }

        private boolean isEmpty() {
            return size == 0;
        }

        private void ensureCapacity(int expected) {
            if (values.length >= expected) {
                return;
            }
            int capacity = Math.max(expected,
                    Math.max(16, values.length << 1));
            values = java.util.Arrays.copyOf(values, capacity);
            weights = java.util.Arrays.copyOf(weights, capacity);
            tierWeights = java.util.Arrays.copyOf(
                    tierWeights, capacity);
            tierCounts = java.util.Arrays.copyOf(
                    tierCounts, capacity);
        }
    }
}
