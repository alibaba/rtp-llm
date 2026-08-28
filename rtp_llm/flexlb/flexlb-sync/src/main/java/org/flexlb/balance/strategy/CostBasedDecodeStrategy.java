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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class CostBasedDecodeStrategy {

    private static final int SNAPSHOT_CAPTURE_ATTEMPTS = 2;
    private static final int MAX_PROJECTED_CANDIDATES = 8;

    private final WorkerDirectory workerDirectory;
    private final AtomicLong candidateWindowCursor = new AtomicLong();

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
            PlacementResult<List<DecodeRoutingView>, RoleType> window =
                    candidateWindow(snapshots, seqLen);
            if (window.status() == PlacementResult.Status.REJECTED) {
                return PlacementResult.rejected(window.rejection());
            }
            List<DecodeRoutingView> candidates = window.value();

            Map<String, Integer> availabilityRejections = new HashMap<>(1);
            List<DecodeRoutingView> eligible = softQueuePlacement
                    ? candidates
                    : filterAvailableEndpoints(
                            candidates, config, availabilityRejections);
            if (eligible.isEmpty()) {
                logNoAvailableEndpoint(
                        balanceContext, registered, availabilityRejections);
                return PlacementResult.blocked(roleType);
            }

            Map<String, Integer> hardRejections = new HashMap<>(3);
            DecodeRoutingView selected = selectPreferredThenFallback(
                    eligible, balanceContext, selector,
                    softQueuePlacement, hardRejections);
            if (selected == null) {
                logAllFilteredOut(
                        balanceContext,
                        availabilityRejections,
                        hardRejections);
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

    private static List<DecodeRoutingView> filterAvailableEndpoints(
            List<DecodeRoutingView> snapshots,
            FlexlbConfig config,
            Map<String, Integer> rejections) {
        ArrayList<DecodeRoutingView> filtered = null;
        int unavailable = 0;
        for (int index = 0; index < snapshots.size(); index++) {
            DecodeRoutingView snapshot = snapshots.get(index);
            if (!hasDecodeCapacity(config, snapshot, false)) {
                unavailable++;
                if (filtered == null) {
                    filtered = new ArrayList<>(snapshots.size());
                    filtered.addAll(snapshots.subList(0, index));
                }
            } else if (filtered != null) {
                filtered.add(snapshot);
            }
        }
        if (filtered == null) {
            return snapshots;
        }
        rejections.put("RESOURCE_UNAVAILABLE", unavailable);
        return Collections.unmodifiableList(filtered);
    }

    private PlacementResult<List<DecodeRoutingView>, RoleType> candidateWindow(
            List<DecodeRoutingView> snapshots,
            long requiredKv) {
        int size = snapshots.size();
        DecodeRoutingView minimum = null;
        long minimumKv = Long.MAX_VALUE;
        long maximumKv = Long.MIN_VALUE;
        DecodeRoutingView leastLoaded = null;
        long minimumLoad = Long.MAX_VALUE;
        long maximumLoad = Long.MIN_VALUE;
        long maximumPhysicalKv = 0L;
        boolean physicalKvUnknown = false;
        for (DecodeRoutingView snapshot : snapshots) {
            long usedKv = snapshot.realKvUsed();
            if (usedKv < minimumKv) {
                minimum = snapshot;
                minimumKv = usedKv;
            }
            maximumKv = Math.max(maximumKv, usedKv);
            long load = snapshot.engineLoad();
            if (load < minimumLoad) {
                leastLoaded = snapshot;
                minimumLoad = load;
            }
            maximumLoad = Math.max(maximumLoad, load);
            long physicalKv = snapshot.totalKv();
            if (physicalKv <= 0L) {
                physicalKvUnknown = true;
            } else {
                maximumPhysicalKv = Math.max(
                        maximumPhysicalKv, physicalKv);
            }
        }
        if (!physicalKvUnknown
                && Math.max(0L, requiredKv) > maximumPhysicalKv) {
            return PlacementResult.rejected(
                    staticCapacityFailure(requiredKv, maximumPhysicalKv));
        }
        if (size <= MAX_PROJECTED_CANDIDATES) {
            return PlacementResult.success(snapshots);
        }
        long cursor = candidateWindowCursor.getAndAdd(MAX_PROJECTED_CANDIDATES);
        int start = (int) Math.floorMod(cursor, size);
        ArrayList<DecodeRoutingView> window =
                new ArrayList<>(MAX_PROJECTED_CANDIDATES);
        for (int offset = 0; offset < MAX_PROJECTED_CANDIDATES; offset++) {
            window.add(snapshots.get((start + offset) % size));
        }
        if (minimumLoad < maximumLoad && !window.contains(leastLoaded)) {
            window.add(leastLoaded);
        }
        if (minimumKv < maximumKv && !window.contains(minimum)) {
            window.add(minimum);
        }
        return PlacementResult.success(
                Collections.unmodifiableList(window));
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
     * Prefer endpoints that can dispatch immediately. If every endpoint is
     * temporarily full, retain the complete eligible set so QUEUE placement
     * can wait for the next exact dispatch-capacity signal.
     */
    private DecodeRoutingView selectPreferredThenFallback(
            List<DecodeRoutingView> eligible,
            BalanceContext context,
            RoutingConfig.DecodeConfig selector,
            boolean softQueuePlacement,
            Map<String, Integer> rejections) {
        long seqLen = context.getRequest().getSeqLen();
        List<DecodeRoutingView> dispatchable = softQueuePlacement
                ? dispatchableForRequest(eligible, context)
                : eligible;
        List<DecodeRoutingView> projected = softQueuePlacement
                ? leastProjectedOwnership(dispatchable)
                : dispatchable;
        DecodeRoutingView selected = selectWithin(
                projected, seqLen, selector, softQueuePlacement, rejections);
        if (selected != null) {
            return selected;
        }
        if (projected.size() != dispatchable.size()) {
            selected = selectWithin(
                    dispatchable, seqLen, selector, softQueuePlacement,
                    rejections);
            if (selected != null) {
                return selected;
            }
        }
        if (dispatchable.size() != eligible.size()) {
            return selectWithin(
                    eligible, seqLen, selector, softQueuePlacement,
                    rejections);
        }
        return null;
    }

    private DecodeRoutingView selectWithin(
            List<DecodeRoutingView> candidates,
            long seqLen,
            RoutingConfig.DecodeConfig selector,
            boolean softQueuePlacement,
            Map<String, Integer> rejections) {
        List<DecodeRoutingView> filtered = applyHardFilters(
                candidates, seqLen, selector, softQueuePlacement, rejections);
        return weightedRandomSelection(
                filtered, selector.getDecayPerToken());
    }

    /** Return the original list when all or no endpoints fit this request. */
    private static List<DecodeRoutingView> dispatchableForRequest(
            List<DecodeRoutingView> eligible,
            BalanceContext context) {
        ArrayList<DecodeRoutingView> available = new ArrayList<>(eligible.size());
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
        for (DecodeRoutingView candidate : eligible) {
            DecodeEndpoint.DecodeRoutingView view = candidate;
            long expectedKv = Math.max(
                    hardKv,
                    context.getConfig().decodeKvReservationTokens(
                            hardKv, maxNewTokens, view.totalKv()));
            if (DecodeEndpoint.canAcquireEngineDispatchPermit(
                    view, hardKv, expectedKv, capacity)) {
                available.add(candidate);
            }
        }
        if (available.isEmpty() || available.size() == eligible.size()) {
            return eligible;
        }
        return Collections.unmodifiableList(available);
    }

    /**
     * Queued ownership is excluded from the hard Engine gate, but it remains
     * future Decode demand. Balance that demand before the immutable route is
     * published; KV weighting then breaks ties inside the least-owned tier.
     */
    private static List<DecodeRoutingView> leastProjectedOwnership(
            List<DecodeRoutingView> candidates) {
        int minimum = Integer.MAX_VALUE;
        ArrayList<DecodeRoutingView> leastOwned = new ArrayList<>(candidates.size());
        for (DecodeRoutingView candidate : candidates) {
            int projected = candidate.totalLoad();
            if (projected < minimum) {
                minimum = projected;
                leastOwned.clear();
                leastOwned.add(candidate);
            } else if (projected == minimum) {
                leastOwned.add(candidate);
            }
        }
        if (leastOwned.size() == candidates.size()) {
            return candidates;
        }
        return Collections.unmodifiableList(leastOwned);
    }

    private List<DecodeRoutingView> applyHardFilters(
            List<DecodeRoutingView> eligible,
            long seqLen,
            RoutingConfig.DecodeConfig selector,
            boolean softQueuePlacement,
            Map<String, Integer> rejections) {
        rejections.clear();
        RoutingConfig.DecodeOutlierRejectionConfig outlier = selector.getOutlierRejection();
        double hotspotMultiplier = outlier == null
                ? 0.0 : outlier.getMaxEngineLoadVsAverageMultiplier();
        double imbalanceMultiplier = outlier == null
                ? 0.0 : outlier.getMaxKvUsedVsAverageMultiplier();

        int n = eligible.size();
        long sumLoad = 0;
        long sumCacheUsed = 0;
        ArrayList<DecodeRoutingView> capacitySurvivors = null;
        int capacityRejected = 0;
        for (int index = 0; index < n; index++) {
            DecodeRoutingView candidate = eligible.get(index);
            DecodeEndpoint.DecodeRoutingView view = candidate;
            sumLoad += view.engineLoad();
            sumCacheUsed += view.realKvUsed();

            long totalKv = view.totalKv();
            long availableKv = softQueuePlacement
                    ? totalKv : view.realKvAvailable();
            if (totalKv > 0 && availableKv < seqLen) {
                capacityRejected++;
                if (capacitySurvivors == null) {
                    capacitySurvivors = new ArrayList<>(n);
                    for (int survivorIndex = 0;
                            survivorIndex < index;
                            survivorIndex++) {
                        capacitySurvivors.add(eligible.get(survivorIndex));
                    }
                }
                continue;
            }
            if (capacitySurvivors != null) {
                capacitySurvivors.add(candidate);
            }
        }
        long avgLoad = sumLoad / n;
        long avgCacheUsed = sumCacheUsed / n;

        List<DecodeRoutingView> capacityCandidates = capacitySurvivors == null
                ? eligible
                : Collections.unmodifiableList(capacitySurvivors);
        boolean filterHotspots = hotspotMultiplier > 0 && avgLoad > 0;
        boolean filterImbalance = imbalanceMultiplier > 0 && avgCacheUsed > 0;
        if (capacityCandidates.isEmpty()
                || (!filterHotspots && !filterImbalance)) {
            recordHardFilterRejections(
                    rejections, capacityRejected, 0, 0);
            return capacityCandidates;
        }

        ArrayList<DecodeRoutingView> outlierSurvivors = null;
        int hotspotRejected = 0;
        int imbalanceRejected = 0;
        for (int index = 0; index < capacityCandidates.size(); index++) {
            DecodeRoutingView candidate = capacityCandidates.get(index);
            DecodeEndpoint.DecodeRoutingView view = candidate;
            if (filterHotspots
                    && view.engineLoad() > avgLoad * hotspotMultiplier) {
                hotspotRejected++;
                if (outlierSurvivors == null) {
                    outlierSurvivors = new ArrayList<>(capacityCandidates.size());
                    for (int survivorIndex = 0;
                            survivorIndex < index;
                            survivorIndex++) {
                        outlierSurvivors.add(
                                capacityCandidates.get(survivorIndex));
                    }
                }
                continue;
            }
            long cacheUsed = view.realKvUsed();
            if (filterImbalance
                    && cacheUsed > avgCacheUsed * imbalanceMultiplier) {
                imbalanceRejected++;
                if (outlierSurvivors == null) {
                    outlierSurvivors = new ArrayList<>(capacityCandidates.size());
                    for (int survivorIndex = 0;
                            survivorIndex < index;
                            survivorIndex++) {
                        outlierSurvivors.add(
                                capacityCandidates.get(survivorIndex));
                    }
                }
                continue;
            }
            if (outlierSurvivors != null) {
                outlierSurvivors.add(candidate);
            }
        }

        List<DecodeRoutingView> candidates = outlierSurvivors == null
                ? capacityCandidates
                : Collections.unmodifiableList(outlierSurvivors);
        recordHardFilterRejections(
                rejections,
                capacityRejected,
                hotspotRejected,
                imbalanceRejected);
        return candidates;
    }

    private static void recordHardFilterRejections(
            Map<String, Integer> rejections,
            int capacityRejected,
            int hotspotRejected,
            int imbalanceRejected) {
        if (capacityRejected > 0) {
            rejections.put("KV_CAPACITY", capacityRejected);
        }
        if (hotspotRejected > 0) {
            rejections.put("HOTSPOT_FILTERED", hotspotRejected);
        }
        if (imbalanceRejected > 0) {
            rejections.put("IMBALANCE_FILTERED", imbalanceRejected);
        }
    }

    private DecodeRoutingView weightedRandomSelection(
            List<DecodeRoutingView> candidates,
            double decayFactor) {
        if (candidates.isEmpty()) {
            return null;
        }

        int n = candidates.size();
        long minCacheUsed = candidates.getFirst().realKvUsed();
        long maxCacheUsed = minCacheUsed;
        int minCacheUsedIndex = 0;
        for (int index = 1; index < n; index++) {
            long used = candidates.get(index).realKvUsed();
            if (used < minCacheUsed) {
                minCacheUsed = used;
                minCacheUsedIndex = index;
            }
            maxCacheUsed = Math.max(maxCacheUsed, used);
        }

        if (minCacheUsed == maxCacheUsed || decayFactor == 0.0) {
            return candidates.get(ThreadLocalRandom.current().nextInt(n));
        }

        double[] weights = new double[n];
        double totalWeight = 0;
        // Subtract the value that produces the maximum log-weight before exponentiation.
        // This is mathematically equivalent to the previous average-centered weights, but
        // keeps every exponent <= 0 and avoids exp(...) overflowing for large KV gaps.
        long referenceCacheUsed = decayFactor >= 0
                ? minCacheUsed
                : maxCacheUsed;
        for (int i = 0; i < n; i++) {
            long cacheUsed = candidates.get(i).realKvUsed();
            double normalizedValue = (double) cacheUsed - referenceCacheUsed;
            weights[i] = Math.exp(-decayFactor * normalizedValue);
            totalWeight += weights[i];
        }
        if (!Double.isFinite(totalWeight) || totalWeight <= 0) {
            Logger.warn(
                    "Decode weighted selection produced invalid total weight: decayFactor={},"
                        + " totalWeight={}",
                    decayFactor,
                    totalWeight);
            return candidates.get(minCacheUsedIndex);
        }

        // 加权随机选择
        double r = ThreadLocalRandom.current().nextDouble(totalWeight);
        double cumulativeWeight = 0;
        for (int i = 0; i < n; i++) {
            cumulativeWeight += weights[i];
            if (r <= cumulativeWeight) {
                return candidates.get(i);
            }
        }

        // fallback: 返回使用率最低的
        return candidates.get(minCacheUsedIndex);
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
                    factoryPin, result, routing.totalKv());
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
            Map<String, Integer> availabilityRejections,
            Map<String, Integer> hardRejections) {
        Map<String, Integer> merged =
                new HashMap<>(availabilityRejections);
        hardRejections.forEach(
                (key, count) -> merged.merge(key, count, Integer::sum));
        Logger.debug(
                "Decode select failed: all filtered out, request_id={}, rejections={}",
                balanceContext.getRequestId(), merged);
    }
}
